# %%
"""
MNIST Pairwise Tensor Similarity Experiment

Part A (1-layer):
1. Train a 1-layer bilinear network on MNIST (784 -> hidden -> 10)
2. For each pair of output classes (i,j), train a rank-1 bilinear to approximate
   the 2-class slice of the full model using tensor similarity loss
3. Visualize what each rank-1 approximation "detects"

Part B (2-layer, no residual):
1. Train a 2-layer bilinear network on MNIST
2. For each pair, train a small 2-layer bilinear (rank-1 per layer) to match the slice
3. Compare with analytical spectral baseline
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from itertools import combinations
from tqdm import tqdm

# %%
# =============================================================================
# MODELS
# =============================================================================

class BilinearMNIST(nn.Module):
    """Single bilinear layer: logits = D @ ((L @ x) * (R @ x))"""

    def __init__(self, d_input: int, d_hidden: int, d_output: int):
        super().__init__()
        self.d_input = d_input
        self.d_hidden = d_hidden
        self.d_output = d_output
        self.bi_linear = nn.Linear(d_input, 2 * d_hidden, bias=False)
        self.projection = nn.Linear(d_hidden, d_output, bias=False)
        nn.init.normal_(self.bi_linear.weight, std=0.02)
        nn.init.normal_(self.projection.weight, std=0.02)

    @property
    def w_l(self):
        return self.bi_linear.weight[:self.d_hidden]

    @property
    def w_r(self):
        return self.bi_linear.weight[self.d_hidden:]

    @property
    def w_p(self):
        return self.projection.weight

    def forward(self, x):
        if x.dim() > 2:
            x = x.view(x.size(0), -1)
        combined = self.bi_linear(x)
        left, right = combined.chunk(2, dim=-1)
        return self.projection(left * right)


class ResidualBilinearMNIST(nn.Module):
    """
    embed(784→d_hidden) + bilinear with residual + head(d_hidden→10)
    output = head(h + D@(Lh⊙Rh)) where h = embed(x)
    """

    def __init__(self, d_input: int, d_hidden: int, d_rank: int, d_output: int):
        super().__init__()
        self.d_input = d_input
        self.d_hidden = d_hidden
        self.d_rank = d_rank
        self.d_output = d_output
        self.embed = nn.Linear(d_input, d_hidden, bias=False)
        self.bi_linear = nn.Linear(d_hidden, 2 * d_rank, bias=False)
        self.down = nn.Linear(d_rank, d_hidden, bias=False)
        self.head = nn.Linear(d_hidden, d_output, bias=False)
        nn.init.normal_(self.embed.weight, std=0.02)
        nn.init.normal_(self.bi_linear.weight, std=0.02)
        nn.init.normal_(self.down.weight, std=0.02)
        nn.init.normal_(self.head.weight, std=0.02)

    @property
    def L(self):
        return self.bi_linear.weight[:self.d_rank]  # (rank, d_hidden)

    @property
    def R(self):
        return self.bi_linear.weight[self.d_rank:]  # (rank, d_hidden)

    @property
    def D(self):
        return self.down.weight  # (d_hidden, rank)

    @property
    def W_embed(self):
        return self.embed.weight  # (d_hidden, 784)

    @property
    def W_head(self):
        return self.head.weight  # (10, d_hidden)

    def forward(self, x):
        if x.dim() > 2:
            x = x.view(x.size(0), -1)
        h = self.embed(x)
        combined = self.bi_linear(h)
        left, right = combined.chunk(2, dim=-1)
        h = h + self.down(left * right)
        return self.head(h)

    def get_augmented_weights_for_pair(self, class_i, class_j):
        """
        Get augmented weights for a 2-class slice, folding embed and head.

        Returns L_aug, R_aug, D_aug such that:
          f_{i,j}(x) = D_aug @ (L_aug @ x̂ ⊙ R_aug @ x̂)
        where x̂ = [x, 1] (x is the original 784-dim input).
        """
        with torch.no_grad():
            W_e = self.W_embed
            W_h = self.W_head[[class_i, class_j]]

            # Fold embed into bilinear
            L_eff = self.L @ W_e   # (rank, 784)
            R_eff = self.R @ W_e   # (rank, 784)
            D_eff = W_h @ self.D   # (2, rank)

            # Residual path: W_head_slice @ W_embed
            W_res = W_h @ W_e      # (2, 784)

            # Build augmented weights
            n_in = W_e.shape[1]  # 784
            rank = self.d_rank
            device = W_e.device
            dtype = W_e.dtype

            # Residual part
            L_res = torch.zeros(n_in, n_in + 1, device=device, dtype=dtype)
            L_res[:, -1] = 1.0
            R_res = torch.zeros(n_in, n_in + 1, device=device, dtype=dtype)
            R_res[:, :n_in] = torch.eye(n_in, device=device, dtype=dtype)
            D_res = W_res  # (2, 784)

            # Bilinear part (pad with zero column)
            L_bil = torch.cat([L_eff, torch.zeros(rank, 1, device=device, dtype=dtype)], dim=1)
            R_bil = torch.cat([R_eff, torch.zeros(rank, 1, device=device, dtype=dtype)], dim=1)

            L_aug = torch.cat([L_res, L_bil], dim=0)  # (784+rank, 785)
            R_aug = torch.cat([R_res, R_bil], dim=0)
            D_aug = torch.cat([D_res, D_eff], dim=1)  # (2, 784+rank)

        return L_aug, R_aug, D_aug


class TwoLayerBilinear(nn.Module):
    """
    2-layer bilinear (no residual): y = D @ ((L2 @ mid) * (R2 @ mid))
    where mid = (L1 @ x) * (R1 @ x)
    """

    def __init__(self, d_input: int, d1: int, d2: int, d_output: int):
        super().__init__()
        self.d_input = d_input
        self.d1 = d1
        self.d2 = d2
        self.d_output = d_output
        self.layer1 = nn.Linear(d_input, 2 * d1, bias=False)
        self.layer2 = nn.Linear(d1, 2 * d2, bias=False)
        self.projection = nn.Linear(d2, d_output, bias=False)
        nn.init.normal_(self.layer1.weight, std=0.02)
        nn.init.normal_(self.layer2.weight, std=0.02)
        nn.init.normal_(self.projection.weight, std=0.02)

    @property
    def L1(self):
        return self.layer1.weight[:self.d1]

    @property
    def R1(self):
        return self.layer1.weight[self.d1:]

    @property
    def L2(self):
        return self.layer2.weight[:self.d2]

    @property
    def R2(self):
        return self.layer2.weight[self.d2:]

    @property
    def D(self):
        return self.projection.weight

    def forward(self, x):
        if x.dim() > 2:
            x = x.view(x.size(0), -1)
        h = self.layer1(x)
        l1, r1 = h.chunk(2, dim=-1)
        mid = l1 * r1
        h2 = self.layer2(mid)
        l2, r2 = h2.chunk(2, dim=-1)
        return self.projection(l2 * r2)


# %%
# =============================================================================
# TN INNER PRODUCTS
# =============================================================================

def tn_inner_1layer(w_l1, w_r1, w_p1, w_l2, w_r2, w_p2):
    """
    Symmetrized TN inner product for 1-layer bilinear tensors.
    Accounts for L<->R exchange symmetry.
    """
    LL = w_l1 @ w_l2.T
    RR = w_r1 @ w_r2.T
    LR = w_l1 @ w_r2.T
    RL = w_r1 @ w_l2.T
    core = 0.5 * (LL * RR + LR * RL)
    DD = w_p1.T @ w_p2
    return (core * DD).sum()


def tn_sim_1layer(w_l1, w_r1, w_p1, w_l2, w_r2, w_p2):
    """Cosine similarity in tensor space (1-layer)."""
    inner = tn_inner_1layer(w_l1, w_r1, w_p1, w_l2, w_r2, w_p2)
    norm1 = torch.sqrt(tn_inner_1layer(w_l1, w_r1, w_p1, w_l1, w_r1, w_p1).clamp(min=1e-12))
    norm2 = torch.sqrt(tn_inner_1layer(w_l2, w_r2, w_p2, w_l2, w_r2, w_p2).clamp(min=1e-12))
    return inner / (norm1 * norm2)


def tn_inner_2layer(L1_A, R1_A, L2_A, R2_A, D_A, L1_B, R1_B, L2_B, R2_B, D_B):
    """
    Fully S_4-symmetrized TN inner product for 2-layer bilinear (no residual).

    The degree-4 tensor is:
    T[i, j, k, p, q] = sum_{b,a,c} D[i,b] L2[b,a] R2[b,c]
                        L1[a,j] R1[a,k] L1[c,p] R1[c,q]

    Enumerates all 24 permutations of the 4 input positions.
    """
    from itertools import permutations as _perms
    from collections import defaultdict

    # Gram matrices: grams[aw][bw] where 0=L1, 1=R1
    gram_lookup = [
        [L1_A @ L1_B.T, L1_A @ R1_B.T],  # A=L1, B=L1 or B=R1
        [R1_A @ L1_B.T, R1_A @ R1_B.T],   # A=R1, B=L1 or B=R1
    ]

    DD = D_A.T @ D_B  # (d2_A, d2_B)
    d1_A, d1_B = L1_A.shape[0], L1_B.shape[0]

    # A positions: 0=L1(a), 1=R1(a), 2=L1(a'), 3=R1(a')
    # A hidden: 0,1 -> group 0 (branch via L2_A), 2,3 -> group 1 (branch via R2_A)
    A_wt = [0, 1, 0, 1]   # weight type: 0=L1, 1=R1
    A_hid = [0, 0, 1, 1]  # hidden group
    B_wt = [0, 1, 0, 1]
    B_hid = [0, 0, 1, 1]

    total = torch.tensor(0.0, device=L1_A.device)

    for perm in _perms(range(4)):
        # For each of 4 A positions, find which gram and which (A_hid, B_hid) pair
        contribs = defaultdict(list)
        for i in range(4):
            ah = A_hid[i]
            bh = B_hid[perm[i]]
            g = gram_lookup[A_wt[i]][B_wt[perm[i]]]
            contribs[(ah, bh)].append(g)

        # Build combo matrices: element-wise product of all grams for each (ah, bh)
        C = {}
        for (ah, bh), glist in contribs.items():
            prod = glist[0]
            for g in glist[1:]:
                prod = prod * g
            C[(ah, bh)] = prod

        # Default to ones (no constraint on that index pair)
        ones = torch.ones(d1_A, d1_B, device=L1_A.device)
        C00 = C.get((0, 0), ones)
        C01 = C.get((0, 1), ones)
        C10 = C.get((1, 0), ones)
        C11 = C.get((1, 1), ones)

        # Full contraction:
        # Σ_{ba,bb,a,c,b,d} DD[ba,bb] L2_A[ba,a] R2_A[ba,c] L2_B[bb,b] R2_B[bb,d]
        #                    C00[a,b] C01[a,d] C10[c,b] C11[c,d]
        val = torch.einsum('xy,xa,xc,yb,yd,ab,ad,cb,cd->',
                           DD, L2_A, R2_A, L2_B, R2_B, C00, C01, C10, C11)
        total = total + val

    return total / 24.0


def tn_sim_2layer(model_a, model_b):
    """Cosine similarity for 2-layer bilinear models."""
    inner = tn_inner_2layer(
        model_a.L1, model_a.R1, model_a.L2, model_a.R2, model_a.D,
        model_b.L1, model_b.R1, model_b.L2, model_b.R2, model_b.D
    )
    norm_a = torch.sqrt(tn_inner_2layer(
        model_a.L1, model_a.R1, model_a.L2, model_a.R2, model_a.D,
        model_a.L1, model_a.R1, model_a.L2, model_a.R2, model_a.D
    ).clamp(min=1e-12))
    norm_b = torch.sqrt(tn_inner_2layer(
        model_b.L1, model_b.R1, model_b.L2, model_b.R2, model_b.D,
        model_b.L1, model_b.R1, model_b.L2, model_b.R2, model_b.D
    ).clamp(min=1e-12))
    return inner / (norm_a * norm_b)


# ---- Specialized fast versions for d1_A=d2_A=1 (rank-1 per layer) ----

_S4_PERMS = list(__import__('itertools').permutations(range(4)))
_A_WT = [0, 1, 0, 1]   # weight type: 0=L1, 1=R1
_A_HID = [0, 0, 1, 1]  # hidden group
_B_WT = [0, 1, 0, 1]
_B_HID = [0, 0, 1, 1]


def _precompute_perm_indices():
    """
    Precompute which gram products go into f_b and f_d for each of the 24 permutations.
    Returns lookup tables used to build (24, d1_B) batched f_b, f_d tensors.
    """
    A_wt = [0, 1, 0, 1]
    A_hid = [0, 0, 1, 1]
    B_wt = [0, 1, 0, 1]
    B_hid = [0, 0, 1, 1]

    # For each perm, compute which grams contribute to f_b and f_d
    # f_b = C00 * C10 (product of grams with B_hid=0, from both A branches)
    # f_d = C01 * C11 (product of grams with B_hid=1)
    # Each position i contributes gram[A_wt[i]][B_wt[perm[i]]] to (A_hid[i], B_hid[perm[i]])

    # We need, for each perm: which indices contribute to {(ah, bh=0)} and {(ah, bh=1)}
    perm_specs = []
    for perm in _S4_PERMS:
        # Group by bh
        bh0_grams = []  # (aw, bw) pairs for positions going to bh=0
        bh1_grams = []
        for i in range(4):
            aw = A_wt[i]
            bw = B_wt[perm[i]]
            bh = B_hid[perm[i]]
            if bh == 0:
                bh0_grams.append((aw, bw))
            else:
                bh1_grams.append((aw, bw))
        perm_specs.append((bh0_grams, bh1_grams))
    return perm_specs

_PERM_SPECS = _precompute_perm_indices()


def tn_inner_2layer_r1_full(approx, L1_B, R1_B, L2_B, R2_B, D_B_slice):
    """
    Fast TN inner product: approx has d1=d2=1, full model is arbitrary.
    Batches all 24 permutations into single tensor operations.
    """
    l1 = approx.L1          # (1, input_dim)
    r1 = approx.R1          # (1, input_dim)
    l2_scalar = approx.L2[0, 0]
    r2_scalar = approx.R2[0, 0]
    d_a = approx.D           # (n_out, 1)

    d1_B = L1_B.shape[0]
    device = l1.device

    # 4 gram vectors: (1, d1_B) each
    g_ll = (l1 @ L1_B.T).squeeze(0)  # (d1_B,)
    g_lr = (l1 @ R1_B.T).squeeze(0)
    g_rl = (r1 @ L1_B.T).squeeze(0)
    g_rr = (r1 @ R1_B.T).squeeze(0)

    gram_flat = [[g_ll, g_lr], [g_rl, g_rr]]  # gram_flat[aw][bw] = (d1_B,)

    DD = d_a.T @ D_B_slice  # (1, d2_B)
    dd = DD.squeeze(0)      # (d2_B,)

    # Build batched f_b, f_d: (24, d1_B)
    ones = torch.ones(d1_B, device=device)
    f_b_list = []
    f_d_list = []
    for bh0_grams, bh1_grams in _PERM_SPECS:
        # f_b = product of all grams going to bh=0
        fb = ones
        for aw, bw in bh0_grams:
            fb = fb * gram_flat[aw][bw]
        # f_d = product of all grams going to bh=1
        fd = ones
        for aw, bw in bh1_grams:
            fd = fd * gram_flat[aw][bw]
        f_b_list.append(fb)
        f_d_list.append(fd)

    F_b = torch.stack(f_b_list)  # (24, d1_B)
    F_d = torch.stack(f_d_list)  # (24, d1_B)

    # Batched matmul and reduction
    E_b = F_b @ L2_B.T  # (24, d2_B)
    E_d = F_d @ R2_B.T  # (24, d2_B)
    vals = (dd.unsqueeze(0) * E_b * E_d).sum()  # scalar

    return l2_scalar * r2_scalar * vals / 24.0


def tn_inner_2layer_r1_r1(approx_a, approx_b):
    """
    Fast TN inner product when BOTH models have d1=d2=1.
    Everything is scalar.
    """
    l1a, r1a = approx_a.L1, approx_a.R1  # (1, input_dim)
    l1b, r1b = approx_b.L1, approx_b.R1

    gram = [
        [(l1a @ l1b.T).squeeze(), (l1a @ r1b.T).squeeze()],
        [(r1a @ l1b.T).squeeze(), (r1a @ r1b.T).squeeze()],
    ]

    l2a = approx_a.L2[0, 0]
    r2a = approx_a.R2[0, 0]
    l2b = approx_b.L2[0, 0]
    r2b = approx_b.R2[0, 0]
    dd = (approx_a.D.T @ approx_b.D).squeeze()  # scalar

    total = torch.tensor(0.0, device=l1a.device)

    for perm in _S4_PERMS:
        combos = {}
        for i in range(4):
            key = (_A_HID[i], _B_HID[perm[i]])
            g = gram[_A_WT[i]][_B_WT[perm[i]]]
            combos[key] = combos[key] * g if key in combos else g

        # With d1=1 on both sides, all combos are scalars
        c00 = combos.get((0, 0), torch.tensor(1.0, device=l1a.device))
        c01 = combos.get((0, 1), torch.tensor(1.0, device=l1a.device))
        c10 = combos.get((1, 0), torch.tensor(1.0, device=l1a.device))
        c11 = combos.get((1, 1), torch.tensor(1.0, device=l1a.device))

        total = total + dd * l2a * r2a * l2b * r2b * c00 * c01 * c10 * c11

    return total / 24.0


def tn_sim_2layer_r1_slice(approx, full_model, class_i, class_j, target_norm=None):
    """
    Fast TN cosine similarity for rank-1 approx vs 2-class slice of full model.
    Pass pre-computed target_norm to avoid recomputing each step.
    """
    D_slice = full_model.D[[class_i, class_j]]

    inner = tn_inner_2layer_r1_full(
        approx, full_model.L1, full_model.R1, full_model.L2, full_model.R2, D_slice
    )
    norm_a = torch.sqrt(tn_inner_2layer_r1_r1(approx, approx).clamp(min=1e-12))

    if target_norm is None:
        target_norm = torch.sqrt(tn_inner_2layer(
            full_model.L1, full_model.R1, full_model.L2, full_model.R2, D_slice,
            full_model.L1, full_model.R1, full_model.L2, full_model.R2, D_slice
        ).clamp(min=1e-12))

    return inner / (norm_a * target_norm)


# %%
# =============================================================================
# DATA
# =============================================================================

def get_mnist_loaders(batch_size: int = 256):
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])
    train = datasets.MNIST('./data', train=True, download=True, transform=transform)
    test = datasets.MNIST('./data', train=False, transform=transform)
    train_loader = DataLoader(train, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test, batch_size=batch_size, shuffle=False)
    return train_loader, test_loader


def eval_accuracy(model, loader, device):
    model.eval()
    correct = total = 0
    with torch.no_grad():
        for data, target in loader:
            data, target = data.to(device), target.to(device)
            pred = model(data.view(data.size(0), -1)).argmax(1)
            correct += (pred == target).sum().item()
            total += target.size(0)
    return correct / total


# %%
# =============================================================================
# PART A: 1-LAYER EXPERIMENT
# =============================================================================

def train_full_model_1layer(d_hidden=64, epochs=10, lr=1e-3, wd=0.01, device='cuda'):
    """Train full 10-class 1-layer bilinear MNIST model."""
    model = BilinearMNIST(784, d_hidden, 10).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=wd)
    train_loader, test_loader = get_mnist_loaders(batch_size=256)

    for epoch in range(epochs):
        model.train()
        total_loss = 0
        for data, target in train_loader:
            data, target = data.to(device), target.to(device)
            x = data.view(data.size(0), -1)
            loss = F.cross_entropy(model(x), target)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        if (epoch + 1) % 2 == 0 or epoch == 0:
            train_acc = eval_accuracy(model, train_loader, device)
            test_acc = eval_accuracy(model, test_loader, device)
            print(f"  Epoch {epoch+1}/{epochs}: loss={total_loss/len(train_loader):.4f} "
                  f"train={train_acc:.4f} test={test_acc:.4f}")

    test_acc = eval_accuracy(model, test_loader, device)
    print(f"  Final test accuracy: {test_acc:.4f}")
    return model


def optimize_rank1_for_pair(full_model, class_i, class_j, steps=3000, lr=1e-3,
                            n_seeds=5, device='cuda'):
    """Train rank-1 bilinear to maximize TN-sim with 2-class slice. Returns (sim, model)."""
    with torch.no_grad():
        target_w_l = full_model.w_l.clone()
        target_w_r = full_model.w_r.clone()
        target_w_p = full_model.w_p[[class_i, class_j]].clone()

    best_sim = -1.0
    best_model = None

    for seed in range(n_seeds):
        torch.manual_seed(seed * 1000 + class_i * 10 + class_j)
        approx = BilinearMNIST(784, 1, 2).to(device)
        optimizer = torch.optim.Adam(approx.parameters(), lr=lr)

        for step in range(steps):
            optimizer.zero_grad()
            sim = tn_sim_1layer(
                approx.w_l, approx.w_r, approx.w_p,
                target_w_l, target_w_r, target_w_p
            )
            (-sim).backward()
            optimizer.step()

        final_sim = sim.item()
        if final_sim > best_sim:
            best_sim = final_sim
            best_model = approx

    return best_sim, best_model


def run_all_pairs_1layer(full_model, steps=3000, lr=1e-3, n_seeds=5, device='cuda'):
    """Run rank-1 TN-sim optimization for all 45 pairs. Returns (sims_dict, models_dict)."""
    pairs = list(combinations(range(10), 2))
    sims = {}
    models = {}

    print(f"\n  Optimizing rank-1 for {len(pairs)} pairs ({n_seeds} seeds x {steps} steps)...")
    for ci, cj in tqdm(pairs):
        sim, model = optimize_rank1_for_pair(
            full_model, ci, cj, steps=steps, lr=lr, n_seeds=n_seeds, device=device
        )
        sims[(ci, cj)] = sim
        models[(ci, cj)] = model

    return sims, models


def compute_slice_spectral(model):
    """Compute spectral concentration for each pair slice (analytical baseline)."""
    with torch.no_grad():
        w_l, w_r, w_p = model.w_l, model.w_r, model.w_p
        LL = w_l @ w_l.T
        RR = w_r @ w_r.T
        LR = w_l @ w_r.T
        core = 0.5 * (LL * RR + LR * LR.T)

        results = {}
        for ci, cj in combinations(range(10), 2):
            D_slice = w_p[[ci, cj]]
            DD = D_slice.T @ D_slice
            M = DD * core
            eigvals = torch.linalg.eigvalsh(M).clamp(min=0)
            total = eigvals.sum().item()
            top = eigvals[-1].item()
            results[(ci, cj)] = {'top_eig_frac': top / (total + 1e-12),
                                 'norm': np.sqrt(max(total, 0))}
    return results


# %%
# =============================================================================
# VISUALIZATION: RANK-1 LEARNED PATTERNS
# =============================================================================

def visualize_rank1_patterns(pair_models, pair_sims, n_show=10):
    """
    For each rank-1 model, show what it detects.

    A rank-1 bilinear with L=(1,784), R=(1,784), D=(2,1) computes:
      output_i(x) = D[i,0] * (l^T x)(r^T x)
    The symmetrized quadratic form is M_i = D[i,0] * 0.5*(l r^T + r l^T)
    Eigenvectors of (l r^T + r l^T) are proportional to (l+r) and (l-r)
    with eigenvalues 0.5*(||l||^2 + ||r||^2 + 2 l.r) and 0.5*(...- 2 l.r).
    The max-activating input is the top eigenvector of M_i.
    """
    sorted_pairs = sorted(pair_sims.items(), key=lambda x: x[1], reverse=True)
    show_pairs = [p for p, _ in sorted_pairs[:n_show//2]] + [p for p, _ in sorted_pairs[-n_show//2:]]

    fig, axes = plt.subplots(len(show_pairs), 5, figsize=(12, 2.2 * len(show_pairs)))
    fig.suptitle('Rank-1 Learned Patterns: L, R, (L+R), (L-R), and Max-Activating per Class', fontsize=12)

    for row, (ci, cj) in enumerate(show_pairs):
        model = pair_models[(ci, cj)]
        sim = pair_sims[(ci, cj)]
        with torch.no_grad():
            l = model.w_l[0].cpu().numpy()  # (784,)
            r = model.w_r[0].cpu().numpy()
            d = model.w_p.cpu().numpy()     # (2, 1)

        lpr = l + r
        lmr = l - r

        # Max-activating direction for each class
        # M_i = D[i,0] * 0.5 * (l r^T + r l^T)
        # The quadratic form on unit vector v is: D[i,0] * (l.v)(r.v)
        # Eigvecs of (l r^T + r l^T) are (l+r)/||l+r|| and (l-r)/||l-r||
        # with eigenvalues ||l+r||^2/2 and -||l-r||^2/2
        # (since l r^T + r l^T = 0.5*((l+r)(l+r)^T - (l-r)(l-r)^T))

        imgs = [l.reshape(28, 28), r.reshape(28, 28),
                lpr.reshape(28, 28), lmr.reshape(28, 28)]
        titles = ['L', 'R', 'L+R', 'L-R']

        for col in range(4):
            ax = axes[row, col]
            im = ax.imshow(imgs[col], cmap='RdBu_r', aspect='equal')
            vabs = max(abs(imgs[col].min()), abs(imgs[col].max()))
            im.set_clim(-vabs, vabs)
            if row == 0:
                ax.set_title(titles[col], fontsize=10)
            ax.set_xticks([])
            ax.set_yticks([])
            if col == 0:
                ax.set_ylabel(f'{ci}-{cj}\nsim={sim:.3f}', fontsize=9)

        # Show D weights (how the two classes weight the single hidden unit)
        ax = axes[row, 4]
        ax.barh([0, 1], [d[0, 0], d[1, 0]], color=['C0', 'C1'])
        ax.set_yticks([0, 1])
        ax.set_yticklabels([f'cls {ci}', f'cls {cj}'], fontsize=8)
        ax.axvline(0, color='k', linewidth=0.5)
        if row == 0:
            ax.set_title('D weights', fontsize=10)

    plt.tight_layout()
    plt.savefig('mnist_rank1_patterns.png', dpi=150, bbox_inches='tight')
    plt.show()
    print("Saved to mnist_rank1_patterns.png")


# %%
# =============================================================================
# PART B: 2-LAYER EXPERIMENT
# =============================================================================

def train_full_model_2layer(d1=32, d2=32, epochs=15, lr=1e-3, wd=0.01, device='cuda'):
    """Train full 10-class 2-layer bilinear MNIST model (no residual)."""
    model = TwoLayerBilinear(784, d1, d2, 10).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=wd)
    train_loader, test_loader = get_mnist_loaders(batch_size=256)

    for epoch in range(epochs):
        model.train()
        total_loss = 0
        for data, target in train_loader:
            data, target = data.to(device), target.to(device)
            x = data.view(data.size(0), -1)
            loss = F.cross_entropy(model(x), target)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        if (epoch + 1) % 3 == 0 or epoch == 0:
            train_acc = eval_accuracy(model, train_loader, device)
            test_acc = eval_accuracy(model, test_loader, device)
            print(f"  Epoch {epoch+1}/{epochs}: loss={total_loss/len(train_loader):.4f} "
                  f"train={train_acc:.4f} test={test_acc:.4f}")

    test_acc = eval_accuracy(model, test_loader, device)
    print(f"  Final test accuracy: {test_acc:.4f}")
    return model


def optimize_2layer_rank1_for_pair(full_model, class_i, class_j, target_norm,
                                    steps=3000, lr=1e-3, n_seeds=5, device='cuda'):
    """
    Train a rank-1-per-layer 2-layer bilinear to maximize TN-sim with
    the 2-class slice of the full 2-layer model.

    Uses specialized fast path since approx has d1=d2=1.
    target_norm is pre-computed to avoid repeated slow computation.
    """
    best_sim = -1.0
    best_model = None

    for seed in range(n_seeds):
        torch.manual_seed(seed * 1000 + class_i * 10 + class_j)
        approx = TwoLayerBilinear(784, 1, 1, 2).to(device)
        optimizer = torch.optim.Adam(approx.parameters(), lr=lr)

        for step in range(steps):
            optimizer.zero_grad()
            sim = tn_sim_2layer_r1_slice(approx, full_model, class_i, class_j,
                                          target_norm=target_norm)
            (-sim).backward()
            optimizer.step()

        final_sim = sim.item()
        if final_sim > best_sim:
            best_sim = final_sim
            best_model = approx

    return best_sim, best_model


def run_all_pairs_2layer(full_model, steps=3000, lr=1e-3, n_seeds=5, device='cuda'):
    """Run 2-layer rank-1 TN-sim optimization for all 45 pairs."""
    pairs = list(combinations(range(10), 2))
    sims = {}
    models = {}

    # Pre-compute target norms (slow, but done once per pair)
    print("\n  Pre-computing target slice norms...")
    target_norms = {}
    for ci, cj in tqdm(pairs, desc="  Target norms"):
        D_slice = full_model.D[[ci, cj]]
        with torch.no_grad():
            norm = torch.sqrt(tn_inner_2layer(
                full_model.L1, full_model.R1, full_model.L2, full_model.R2, D_slice,
                full_model.L1, full_model.R1, full_model.L2, full_model.R2, D_slice
            ).clamp(min=1e-12))
        target_norms[(ci, cj)] = norm

    print(f"\n  Optimizing 2-layer rank-1 for {len(pairs)} pairs ({n_seeds} seeds x {steps} steps)...")
    for ci, cj in tqdm(pairs):
        sim, model = optimize_2layer_rank1_for_pair(
            full_model, ci, cj, target_norms[(ci, cj)],
            steps=steps, lr=lr, n_seeds=n_seeds, device=device
        )
        sims[(ci, cj)] = sim
        models[(ci, cj)] = model

    return sims, models


# %%
# =============================================================================
# PLOTTING
# =============================================================================

def plot_pairwise_results(pair_sims, slice_info, title_prefix, save_name):
    """Plot heatmap, sorted bars, and spectral correlation."""
    sim_matrix = np.zeros((10, 10))
    for (i, j), sim in pair_sims.items():
        sim_matrix[i, j] = sim
        sim_matrix[j, i] = sim
    np.fill_diagonal(sim_matrix, 1.0)

    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    vmin = min(pair_sims.values())
    vmax = max(pair_sims.values())

    # Heatmap
    ax = axes[0]
    im = ax.imshow(sim_matrix, cmap='RdYlBu_r', vmin=vmin - 0.02, vmax=vmax + 0.02)
    ax.set_xticks(range(10)); ax.set_yticks(range(10))
    ax.set_xlabel('Class'); ax.set_ylabel('Class')
    ax.set_title(f'{title_prefix}: TN Similarity by Pair')
    plt.colorbar(im, ax=ax, shrink=0.8)
    for i in range(10):
        for j in range(10):
            if i != j:
                ax.text(j, i, f'{sim_matrix[i,j]:.2f}', ha='center', va='center', fontsize=7)

    # Sorted bars
    ax = axes[1]
    sorted_pairs = sorted(pair_sims.items(), key=lambda x: x[1], reverse=True)
    labels = [f'{i}-{j}' for (i, j), _ in sorted_pairs]
    values = [v for _, v in sorted_pairs]
    norm_vals = [(v - min(values)) / (max(values) - min(values) + 1e-8) for v in values]
    ax.barh(range(len(labels)), values, color=plt.cm.RdYlBu_r(np.array(norm_vals)))
    ax.set_yticks(range(len(labels)))
    ax.set_yticklabels(labels, fontsize=8)
    ax.set_xlabel('TN Similarity')
    ax.set_title(f'{title_prefix}: All 45 Pairs Sorted')
    ax.set_xlim(min(values) - 0.02, max(values) + 0.02)
    ax.invert_yaxis()

    # Spectral correlation
    if slice_info:
        ax = axes[2]
        xs, ys, labs = [], [], []
        for (i, j), opt_sim in pair_sims.items():
            xs.append(slice_info[(i, j)]['top_eig_frac'])
            ys.append(opt_sim)
            labs.append(f'{i}-{j}')
        ax.scatter(xs, ys, s=40, alpha=0.7, edgecolors='k', linewidths=0.3)
        for x, y, lab in zip(xs, ys, labs):
            ax.annotate(lab, (x, y), fontsize=7, alpha=0.8, xytext=(3, 3),
                        textcoords='offset points')
        ax.set_xlabel('Top eigenvalue fraction')
        ax.set_ylabel('Rank-1 TN similarity (optimized)')
        ax.set_title('Approximability vs Spectral Concentration')
        corr = np.corrcoef(xs, ys)[0, 1]
        ax.text(0.05, 0.95, f'r = {corr:.3f}', transform=ax.transAxes,
                fontsize=14, fontweight='bold', verticalalignment='top')

    plt.tight_layout()
    plt.savefig(save_name, dpi=150, bbox_inches='tight')
    plt.show()
    print(f"Saved to {save_name}")


def plot_1layer_vs_2layer(sims_1l, sims_2l):
    """Scatter plot comparing 1-layer and 2-layer pair similarities."""
    fig, ax = plt.subplots(figsize=(7, 7))
    xs, ys, labs = [], [], []
    for (i, j) in sims_1l:
        xs.append(sims_1l[(i, j)])
        ys.append(sims_2l[(i, j)])
        labs.append(f'{i}-{j}')

    ax.scatter(xs, ys, s=50, alpha=0.7, edgecolors='k', linewidths=0.3)
    for x, y, lab in zip(xs, ys, labs):
        ax.annotate(lab, (x, y), fontsize=8, alpha=0.8, xytext=(3, 3),
                    textcoords='offset points')

    corr = np.corrcoef(xs, ys)[0, 1]
    ax.set_xlabel('1-Layer Rank-1 TN Similarity')
    ax.set_ylabel('2-Layer Rank-1 TN Similarity')
    ax.set_title(f'1-Layer vs 2-Layer Pair Approximability (r = {corr:.3f})')

    # Diagonal reference
    mn = min(min(xs), min(ys)) - 0.02
    mx = max(max(xs), max(ys)) + 0.02
    ax.plot([mn, mx], [mn, mx], 'k--', alpha=0.3)
    ax.set_xlim(mn, mx); ax.set_ylim(mn, mx)

    plt.tight_layout()
    plt.savefig('mnist_1layer_vs_2layer.png', dpi=150, bbox_inches='tight')
    plt.show()
    print("Saved to mnist_1layer_vs_2layer.png")


# %%
# =============================================================================
# 2-LAYER SPECTRAL BASELINE
# =============================================================================

# %%
# =============================================================================
# PART C: 1-LAYER RESIDUAL EXPERIMENT
# =============================================================================

def train_full_model_residual(d_hidden=64, d_rank=32, epochs=10, lr=1e-3, wd=0.01, device='cuda'):
    """Train full 10-class residual bilinear MNIST model."""
    model = ResidualBilinearMNIST(784, d_hidden, d_rank, 10).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=wd)
    train_loader, test_loader = get_mnist_loaders(batch_size=256)

    for epoch in range(epochs):
        model.train()
        total_loss = 0
        for data, target in train_loader:
            data, target = data.to(device), target.to(device)
            x = data.view(data.size(0), -1)
            loss = F.cross_entropy(model(x), target)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        if (epoch + 1) % 2 == 0 or epoch == 0:
            train_acc = eval_accuracy(model, train_loader, device)
            test_acc = eval_accuracy(model, test_loader, device)
            print(f"  Epoch {epoch+1}/{epochs}: loss={total_loss/len(train_loader):.4f} "
                  f"train={train_acc:.4f} test={test_acc:.4f}")

    test_acc = eval_accuracy(model, test_loader, device)
    print(f"  Final test accuracy: {test_acc:.4f}")
    return model


def optimize_rank1_residual_for_pair(full_model, class_i, class_j, target_norm,
                                      steps=3000, lr=1e-3, n_seeds=5, device='cuda'):
    """
    Train rank-1 bilinear (in augmented 785-dim space) to maximize TN-sim with
    the augmented 2-class slice of the residual model.
    """
    with torch.no_grad():
        L_aug, R_aug, D_aug = full_model.get_augmented_weights_for_pair(class_i, class_j)

    best_sim = -1.0
    best_model = None

    for seed in range(n_seeds):
        torch.manual_seed(seed * 1000 + class_i * 10 + class_j)
        # Rank-1 in augmented space (785 input dims)
        approx = BilinearMNIST(785, 1, 2).to(device)
        optimizer = torch.optim.Adam(approx.parameters(), lr=lr)

        for step in range(steps):
            optimizer.zero_grad()
            sim = tn_sim_1layer(
                approx.w_l, approx.w_r, approx.w_p,
                L_aug, R_aug, D_aug
            )
            (-sim).backward()
            optimizer.step()

        final_sim = sim.item()
        if final_sim > best_sim:
            best_sim = final_sim
            best_model = approx

    return best_sim, best_model


def run_all_pairs_residual(full_model, steps=3000, lr=1e-3, n_seeds=5, device='cuda'):
    """Run rank-1 TN-sim optimization for all 45 pairs on the residual model."""
    pairs = list(combinations(range(10), 2))
    sims = {}
    models = {}

    # Pre-compute target norms (augmented weights + self-inner-product)
    print("\n  Pre-computing augmented target norms...")
    target_norms = {}
    for ci, cj in tqdm(pairs, desc="  Target norms"):
        with torch.no_grad():
            L_aug, R_aug, D_aug = full_model.get_augmented_weights_for_pair(ci, cj)
            norm = torch.sqrt(tn_inner_1layer(
                L_aug, R_aug, D_aug, L_aug, R_aug, D_aug
            ).clamp(min=1e-12))
        target_norms[(ci, cj)] = norm

    print(f"\n  Optimizing rank-1 for {len(pairs)} pairs ({n_seeds} seeds x {steps} steps)...")
    for ci, cj in tqdm(pairs):
        sim, model = optimize_rank1_residual_for_pair(
            full_model, ci, cj, target_norms[(ci, cj)],
            steps=steps, lr=lr, n_seeds=n_seeds, device=device
        )
        sims[(ci, cj)] = sim
        models[(ci, cj)] = model

    return sims, models


def compute_slice_spectral_residual(model):
    """Compute spectral concentration for each pair slice of the residual model."""
    with torch.no_grad():
        results = {}
        for ci, cj in combinations(range(10), 2):
            L_aug, R_aug, D_aug = model.get_augmented_weights_for_pair(ci, cj)
            LL = L_aug @ L_aug.T
            RR = R_aug @ R_aug.T
            LR = L_aug @ R_aug.T
            core = 0.5 * (LL * RR + LR * LR.T)
            DD = D_aug.T @ D_aug
            M = DD * core
            eigvals = torch.linalg.eigvalsh(M).clamp(min=0)
            total = eigvals.sum().item()
            top = eigvals[-1].item()
            results[(ci, cj)] = {'top_eig_frac': top / (total + 1e-12),
                                 'norm': np.sqrt(max(total, 0))}
    return results


def compute_slice_spectral_2layer(model):
    """
    Analytical spectral baseline for 2-layer model.

    For the self-inner product of a 2-class slice, the "Type 1" (same-pairing)
    terms dominate. These have the form: (DD * E_L * E_R).sum()
    where E_L = L2 @ C @ L2.T, E_R = R2 @ C @ R2.T, C = 0.5*(LL*RR + LR*RL).

    We compute the effective matrix DD * E_L * E_R and look at its spectral concentration.
    """
    with torch.no_grad():
        L1, R1 = model.L1, model.R1
        L2, R2 = model.L2, model.R2
        D = model.D

        # Layer 1 self-Gram
        LL = L1 @ L1.T
        RR = R1 @ R1.T
        LR = L1 @ R1.T
        C_same = LL * RR
        C_swap = LR * LR.T

        results = {}
        for ci, cj in combinations(range(10), 2):
            D_slice = D[[ci, cj]]
            DD = D_slice.T @ D_slice  # (d2, d2)

            # Compute Type-1 contribution (same pairing, dominant terms)
            total_diag = torch.zeros(DD.shape[0], device=DD.device)
            for C in [C_same, C_swap]:
                E_L = L2 @ C @ L2.T
                E_R = R2 @ C @ R2.T
                total_diag += (DD * E_L * E_R).sum(dim=1)  # per d2_A
                E_L = L2 @ C @ R2.T
                E_R = R2 @ C @ L2.T
                total_diag += (DD * E_L * E_R).sum(dim=1)

            # Use the full self-inner product for the spectral measure
            self_inner = tn_inner_2layer(
                L1, R1, L2, R2, D_slice,
                L1, R1, L2, R2, D_slice
            ).item()

            # As a spectral proxy, compute the contribution from each d2 unit
            # This is analogous to the 1-layer case where we looked at (DD * core)
            M_eff = DD * (L2 @ C_same @ L2.T) * (R2 @ C_same @ R2.T)
            eigvals = torch.linalg.eigvalsh(M_eff).clamp(min=0)
            total = eigvals.sum().item()
            top = eigvals[-1].item()
            top_frac = top / (total + 1e-12)

            results[(ci, cj)] = {
                'top_eig_frac': top_frac,
                'norm': np.sqrt(max(self_inner, 0))
            }

    return results


# %%
# =============================================================================
# MAIN
# =============================================================================

def print_summary(pair_sims, label):
    sorted_pairs = sorted(pair_sims.items(), key=lambda x: x[1], reverse=True)
    print(f"\n  Top 10 easiest pairs ({label}):")
    for (i, j), sim in sorted_pairs[:10]:
        print(f"    {i}-{j}: {sim:.4f}")
    print(f"  Bottom 10 hardest pairs:")
    for (i, j), sim in sorted_pairs[-10:]:
        print(f"    {i}-{j}: {sim:.4f}")
    print(f"\n  Per-class average:")
    for c in range(10):
        sims_c = [v for (i, j), v in pair_sims.items() if i == c or j == c]
        print(f"    Class {c}: {np.mean(sims_c):.4f} (std={np.std(sims_c):.4f})")


if __name__ == '__main__':
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Device: {device}")

    # ================================================================
    # PART A: 1-Layer
    # ================================================================
    print("\n" + "=" * 70)
    print("PART A: 1-Layer Bilinear")
    print("=" * 70)

    print("\nA1. Training full 1-layer model...")
    full_1l = train_full_model_1layer(d_hidden=64, epochs=10, device=device)

    print("\nA2. Computing spectral baseline...")
    spectral_1l = compute_slice_spectral(full_1l)

    print("\nA3. Optimizing rank-1 for all pairs...")
    sims_1l, models_1l = run_all_pairs_1layer(full_1l, steps=3000, n_seeds=5, device=device)
    print_summary(sims_1l, "1-layer rank-1")

    print("\nA4. Plotting 1-layer results...")
    plot_pairwise_results(sims_1l, spectral_1l, '1-Layer', 'mnist_1layer_tnsim.png')

    print("\nA5. Visualizing rank-1 patterns (top 5 + bottom 5)...")
    visualize_rank1_patterns(models_1l, sims_1l, n_show=10)

    # ================================================================
    # PART B: 2-Layer
    # ================================================================
    print("\n" + "=" * 70)
    print("PART B: 2-Layer Bilinear (no residual)")
    print("=" * 70)

    print("\nB1. Training full 2-layer model...")
    full_2l = train_full_model_2layer(d1=32, d2=32, epochs=15, device=device)

    print("\nB2. Computing 2-layer spectral baseline...")
    spectral_2l = compute_slice_spectral_2layer(full_2l)

    print("\nB3. Optimizing 2-layer rank-1 for all pairs...")
    sims_2l, models_2l = run_all_pairs_2layer(full_2l, steps=1500, n_seeds=3, device=device)
    print_summary(sims_2l, "2-layer rank-1")

    print("\nB4. Plotting 2-layer results...")
    plot_pairwise_results(sims_2l, spectral_2l, '2-Layer', 'mnist_2layer_tnsim.png')

    # ================================================================
    # COMPARISON
    # ================================================================
    print("\n" + "=" * 70)
    print("COMPARISON: 1-Layer vs 2-Layer")
    print("=" * 70)
    plot_1layer_vs_2layer(sims_1l, sims_2l)

    # Rank correlation
    from scipy.stats import spearmanr
    pairs = list(sims_1l.keys())
    v1 = [sims_1l[p] for p in pairs]
    v2 = [sims_2l[p] for p in pairs]
    rho, pval = spearmanr(v1, v2)
    print(f"\n  Spearman rank correlation: rho={rho:.3f}, p={pval:.2e}")

    # ================================================================
    # PART C: 1-Layer with Residual
    # ================================================================
    print("\n" + "=" * 70)
    print("PART C: 1-Layer Bilinear with Residual")
    print("=" * 70)

    print("\nC1. Training full residual model...")
    full_res = train_full_model_residual(d_hidden=64, d_rank=32, epochs=10, device=device)

    print("\nC2. Computing spectral baseline...")
    spectral_res = compute_slice_spectral_residual(full_res)

    print("\nC3. Optimizing rank-1 for all pairs...")
    sims_res, models_res = run_all_pairs_residual(
        full_res, steps=3000, n_seeds=5, device=device
    )
    print_summary(sims_res, "residual rank-1")

    print("\nC4. Plotting residual results...")
    plot_pairwise_results(sims_res, spectral_res, 'Residual', 'mnist_residual_tnsim.png')

    print("\nC5. Visualizing rank-1 patterns...")
    visualize_rank1_patterns(models_res, sims_res, n_show=10)

    # ================================================================
    # COMPARISON: 1-Layer vs Residual
    # ================================================================
    print("\n" + "=" * 70)
    print("COMPARISON: 1-Layer vs Residual")
    print("=" * 70)

    fig, axes = plt.subplots(1, 2, figsize=(14, 7))

    # 1L vs Residual
    ax = axes[0]
    xs = [sims_1l[p] for p in pairs]
    ys = [sims_res[p] for p in pairs]
    labs = [f'{i}-{j}' for i, j in pairs]
    ax.scatter(xs, ys, s=50, alpha=0.7, edgecolors='k', linewidths=0.3)
    for x, y, lab in zip(xs, ys, labs):
        ax.annotate(lab, (x, y), fontsize=7, alpha=0.8, xytext=(3, 3),
                    textcoords='offset points')
    rho_1r, pval_1r = spearmanr(xs, ys)
    ax.set_xlabel('1-Layer (no residual) TN-Sim')
    ax.set_ylabel('Residual TN-Sim')
    ax.set_title(f'1-Layer vs Residual (rho={rho_1r:.3f}, p={pval_1r:.2e})')
    mn = min(min(xs), min(ys)) - 0.02
    mx = max(max(xs), max(ys)) + 0.02
    ax.plot([mn, mx], [mn, mx], 'k--', alpha=0.3)

    # 2L vs Residual
    ax = axes[1]
    xs2 = [sims_2l[p] for p in pairs]
    ax.scatter(xs2, ys, s=50, alpha=0.7, edgecolors='k', linewidths=0.3)
    for x, y, lab in zip(xs2, ys, labs):
        ax.annotate(lab, (x, y), fontsize=7, alpha=0.8, xytext=(3, 3),
                    textcoords='offset points')
    rho_2r, pval_2r = spearmanr(xs2, ys)
    ax.set_xlabel('2-Layer (no residual) TN-Sim')
    ax.set_ylabel('Residual TN-Sim')
    ax.set_title(f'2-Layer vs Residual (rho={rho_2r:.3f}, p={pval_2r:.2e})')
    mn = min(min(xs2), min(ys)) - 0.02
    mx = max(max(xs2), max(ys)) + 0.02
    ax.plot([mn, mx], [mn, mx], 'k--', alpha=0.3)

    plt.tight_layout()
    plt.savefig('mnist_all_comparisons.png', dpi=150, bbox_inches='tight')
    plt.show()
    print("Saved to mnist_all_comparisons.png")
    print(f"\n  1L vs Residual: rho={rho_1r:.3f}, p={pval_1r:.2e}")
    print(f"  2L vs Residual: rho={rho_2r:.3f}, p={pval_2r:.2e}")
