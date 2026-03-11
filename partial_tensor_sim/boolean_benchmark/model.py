"""
Boolean benchmark models: residual bilinear networks for binary inputs.

1-layer: logits = W_head @ (x + D @ ((L@x) ⊙ (R@x)))
2-layer: logits = W_head @ (h + D2 @ ((L2@h) ⊙ (R2@h)))
         where  h = x + D1 @ ((L1@x) ⊙ (R1@x))
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from itertools import combinations


class BooleanBilinear1Layer(nn.Module):
    """1-layer residual bilinear for binary inputs."""

    def __init__(self, n_input, d_hidden, d_rank, n_classes):
        super().__init__()
        self.n_input = n_input
        self.d_hidden = d_hidden
        self.d_rank = d_rank
        self.n_classes = n_classes

        self.embed = nn.Linear(n_input, d_hidden, bias=False)
        self.bi_linear = nn.Linear(d_hidden, 2 * d_rank, bias=False)
        self.down = nn.Linear(d_rank, d_hidden, bias=False)
        self.head = nn.Linear(d_hidden, n_classes, bias=False)

        for p in self.parameters():
            nn.init.normal_(p, std=0.02)

    @property
    def L(self):
        return self.bi_linear.weight[:self.d_rank]

    @property
    def R(self):
        return self.bi_linear.weight[self.d_rank:]

    @property
    def D(self):
        return self.down.weight

    @property
    def W_embed(self):
        return self.embed.weight

    @property
    def W_head(self):
        return self.head.weight

    def forward(self, x):
        h = self.embed(x)
        left = F.linear(h, self.L)
        right = F.linear(h, self.R)
        bilinear = (left * right) @ self.D.T  # (batch, rank) @ (rank, d_hidden)
        h = h + bilinear
        return self.head(h)

    def get_effective_weights(self):
        """Get effective weights folded through embed."""
        with torch.no_grad():
            W_e = self.W_embed
            L_eff = self.L @ W_e      # (rank, n_input)
            R_eff = self.R @ W_e
            D_eff = self.D             # (d_hidden, rank)
            W_res_full = W_e           # (d_hidden, n_input) residual in hidden space
        return L_eff, R_eff, D_eff, W_res_full

    def get_augmented_weights_for_class(self, c):
        """
        Get augmented weights for class c in x̂ = [x, 1] space (dim n_input+1).
        Returns L_aug, R_aug, D_aug for the full tensor T_c.
        """
        with torch.no_grad():
            W_e = self.W_embed    # (d_hidden, n_input)
            W_h = self.W_head     # (n_classes, d_hidden)
            n = self.n_input
            d = self.d_hidden
            rank = self.d_rank

            # Effective weights for class c
            w_head_c = W_h[[c]]   # (1, d_hidden)

            L_eff = self.L @ W_e  # (rank, n_input)
            R_eff = self.R @ W_e
            D_eff = w_head_c @ self.D  # (1, rank)
            W_res = w_head_c @ W_e     # (1, n_input)

            # Augmented residual part: L_res[h,:] = e_{n+1}, R_res[h,:] = e_h
            device = W_e.device
            L_res = torch.zeros(n, n + 1, device=device)
            L_res[:, -1] = 1.0  # constant selector
            R_res = torch.zeros(n, n + 1, device=device)
            R_res[:, :n] = torch.eye(n, device=device)  # pixel selector
            D_res = W_res  # (1, n_input)

            # Augmented bilinear part: zero-pad to dim n+1
            L_bil = torch.zeros(rank, n + 1, device=device)
            L_bil[:, :n] = L_eff
            R_bil = torch.zeros(rank, n + 1, device=device)
            R_bil[:, :n] = R_eff

            # Stack
            L_aug = torch.cat([L_res, L_bil], dim=0)  # (n + rank, n+1)
            R_aug = torch.cat([R_res, R_bil], dim=0)
            D_aug = torch.cat([D_res, D_eff], dim=1)  # (1, n + rank)

        return L_aug, R_aug, D_aug

    def get_augmented_weights_for_pair(self, ci, cj):
        """Get augmented weights for binary classification between classes ci and cj."""
        with torch.no_grad():
            W_e = self.W_embed
            W_h = self.W_head
            n = self.n_input
            rank = self.d_rank
            device = W_e.device

            w_head_pair = W_h[[ci, cj]]  # (2, d_hidden)
            L_eff = self.L @ W_e
            R_eff = self.R @ W_e
            D_eff = w_head_pair @ self.D  # (2, rank)
            W_res = w_head_pair @ W_e     # (2, n_input)

            L_res = torch.zeros(n, n + 1, device=device)
            L_res[:, -1] = 1.0
            R_res = torch.zeros(n, n + 1, device=device)
            R_res[:, :n] = torch.eye(n, device=device)
            D_res = W_res

            L_bil = torch.zeros(rank, n + 1, device=device)
            L_bil[:, :n] = L_eff
            R_bil = torch.zeros(rank, n + 1, device=device)
            R_bil[:, :n] = R_eff

            L_aug = torch.cat([L_res, L_bil], dim=0)
            R_aug = torch.cat([R_res, R_bil], dim=0)
            D_aug = torch.cat([D_res, D_eff], dim=1)

        return L_aug, R_aug, D_aug

    def get_bilinear_only_weights_for_class(self, c):
        """Get just the bilinear part (no residual) in augmented space."""
        with torch.no_grad():
            W_e = self.W_embed
            W_h = self.W_head
            n = self.n_input
            rank = self.d_rank
            device = W_e.device

            w_head_c = W_h[[c]]
            L_eff = self.L @ W_e
            R_eff = self.R @ W_e
            D_eff = w_head_c @ self.D

            L_bil = torch.zeros(rank, n + 1, device=device)
            L_bil[:, :n] = L_eff
            R_bil = torch.zeros(rank, n + 1, device=device)
            R_bil[:, :n] = R_eff

        return L_bil, R_bil, D_eff

    def get_residual_only_weights_for_class(self, c):
        """Get just the residual part in augmented space."""
        with torch.no_grad():
            W_e = self.W_embed
            W_h = self.W_head
            n = self.n_input
            device = W_e.device

            W_res = W_h[[c]] @ W_e  # (1, n_input)

            L_res = torch.zeros(n, n + 1, device=device)
            L_res[:, -1] = 1.0
            R_res = torch.zeros(n, n + 1, device=device)
            R_res[:, :n] = torch.eye(n, device=device)
            D_res = W_res

        return L_res, R_res, D_res


class BooleanBilinear2Layer(nn.Module):
    """2-layer residual bilinear for binary inputs."""

    def __init__(self, n_input, d_hidden, d_rank, n_classes):
        super().__init__()
        self.n_input = n_input
        self.d_hidden = d_hidden
        self.d_rank = d_rank
        self.n_classes = n_classes

        self.embed = nn.Linear(n_input, d_hidden, bias=False)
        # Layer 1
        self.bi_linear1 = nn.Linear(d_hidden, 2 * d_rank, bias=False)
        self.down1 = nn.Linear(d_rank, d_hidden, bias=False)
        # Layer 2
        self.bi_linear2 = nn.Linear(d_hidden, 2 * d_rank, bias=False)
        self.down2 = nn.Linear(d_rank, d_hidden, bias=False)
        self.head = nn.Linear(d_hidden, n_classes, bias=False)

        for p in self.parameters():
            nn.init.normal_(p, std=0.02)

    @property
    def W_embed(self):
        return self.embed.weight

    @property
    def W_head(self):
        return self.head.weight

    def get_layer_weights(self, layer):
        """Get L, R, D for given layer (1 or 2)."""
        if layer == 1:
            bl = self.bi_linear1
            down = self.down1
        else:
            bl = self.bi_linear2
            down = self.down2
        rank = self.d_rank
        L = bl.weight[:rank]
        R = bl.weight[rank:]
        D = down.weight
        return L, R, D

    def forward(self, x):
        h = self.embed(x)
        # Layer 1
        L1, R1, D1 = self.get_layer_weights(1)
        left1 = F.linear(h, L1)
        right1 = F.linear(h, R1)
        h = h + (left1 * right1) @ D1.T
        # Layer 2
        L2, R2, D2 = self.get_layer_weights(2)
        left2 = F.linear(h, L2)
        right2 = F.linear(h, R2)
        h = h + (left2 * right2) @ D2.T
        return self.head(h)


# =============================================================================
# DATASET GENERATION
# =============================================================================

def generate_all_binary(n):
    """Generate all 2^n binary vectors as float tensor."""
    patterns = []
    for i in range(2 ** n):
        bits = [(i >> b) & 1 for b in range(n)]
        patterns.append(bits)
    return torch.tensor(patterns, dtype=torch.float32)


def make_level1_dataset(device='cpu'):
    """
    Level 1: Two AND features, 4 classes.
    A = x0 AND x1, B = x2 AND x3
    class 0: A=0, B=0 | class 1: A=1, B=0 | class 2: A=0, B=1 | class 3: A=1, B=1
    """
    X = generate_all_binary(4).to(device)
    A = (X[:, 0] * X[:, 1]).long()
    B = (X[:, 2] * X[:, 3]).long()
    labels = A * 1 + B * 2  # 0,1,2,3
    return X, labels


def make_level2_dataset(device='cpu'):
    """
    Level 2: Three AND features, 8 classes.
    A = x0 AND x1, B = x2 AND x3, C = x4 AND x5
    class = 4*C + 2*B + A  (binary encoding of feature activations)
    """
    X = generate_all_binary(6).to(device)
    A = (X[:, 0] * X[:, 1]).long()
    B = (X[:, 2] * X[:, 3]).long()
    C = (X[:, 4] * X[:, 5]).long()
    labels = A + 2 * B + 4 * C
    return X, labels


def make_level3_dataset(device='cpu'):
    """
    Level 3: Mixed degrees. 2-layer target.
    Features: A = x0∧x1, B = x2∧x3, C = x4∧x5
    P = A AND B (degree 4)

    Classes (mutually exclusive, priority order):
      class 0: P=1                    (degree 4)
      class 1: A=1, B=0              (degree 2, only A)
      class 2: B=1, A=0              (degree 2, only B)
      class 3: C=1, A=0, B=0         (degree 2, only C)
      class 4: otherwise             (default)
    """
    X = generate_all_binary(6).to(device)
    A = (X[:, 0] * X[:, 1]).long()
    B = (X[:, 2] * X[:, 3]).long()
    C = (X[:, 4] * X[:, 5]).long()
    P = (A * B).long()

    labels = torch.full((len(X),), 4, device=device, dtype=torch.long)  # default
    labels[C.bool() & ~A.bool() & ~B.bool()] = 3
    labels[B.bool() & ~A.bool()] = 2
    labels[A.bool() & ~B.bool()] = 1
    labels[P.bool()] = 0

    return X, labels


# =============================================================================
# TRAINING
# =============================================================================

def train_model(model, X, labels, epochs=2000, lr=0.01, wd=0.01,
                print_every=500, device='cpu'):
    """Train model on complete Boolean dataset."""
    model.to(device)
    X, labels = X.to(device), labels.to(device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=wd)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, epochs)

    for epoch in range(epochs):
        logits = model(X)
        loss = F.cross_entropy(logits, labels)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        scheduler.step()

        if epoch % print_every == 0 or epoch == epochs - 1:
            with torch.no_grad():
                preds = logits.argmax(dim=1)
                acc = (preds == labels).float().mean().item()
                print(f"  Epoch {epoch:5d}: loss={loss.item():.4f} acc={acc:.1%}")

    with torch.no_grad():
        preds = model(X).argmax(dim=1)
        acc = (preds == labels).float().mean().item()

    return acc


# =============================================================================
# TN-SIM UTILITIES
# =============================================================================

def tn_inner_1layer(L_a, R_a, D_a, L_b, R_b, D_b):
    """Symmetrized TN inner product for 1-layer bilinear."""
    LL = L_a @ L_b.T
    RR = R_a @ R_b.T
    LR = L_a @ R_b.T
    RL = R_a @ L_b.T
    core = 0.5 * (LL * RR + LR * RL)
    DD = D_a.T @ D_b
    return (core * DD).sum()


def tn_sim_1layer(L_a, R_a, D_a, L_b, R_b, D_b):
    """TN-sim (cosine similarity of tensors)."""
    ab = tn_inner_1layer(L_a, R_a, D_a, L_b, R_b, D_b)
    aa = tn_inner_1layer(L_a, R_a, D_a, L_a, R_a, D_a)
    bb = tn_inner_1layer(L_b, R_b, D_b, L_b, R_b, D_b)
    return ab / (torch.sqrt(aa.clamp(min=1e-12)) * torch.sqrt(bb.clamp(min=1e-12)))


def compute_relative_norms(model, c, device):
    """
    Compute ρ(residual) and ρ(bilinear) for class c.
    ρ(A) = <T|A>/<T|T>, ρ(B) = <T|B>/<T|T>, sum = 1.
    """
    L_T, R_T, D_T = model.get_augmented_weights_for_class(c)
    L_A, R_A, D_A = model.get_residual_only_weights_for_class(c)
    L_B, R_B, D_B = model.get_bilinear_only_weights_for_class(c)

    TT = tn_inner_1layer(L_T, R_T, D_T, L_T, R_T, D_T).item()
    TA = tn_inner_1layer(L_T, R_T, D_T, L_A, R_A, D_A).item()
    TB = tn_inner_1layer(L_T, R_T, D_T, L_B, R_B, D_B).item()

    rho_A = TA / TT if abs(TT) > 1e-12 else 0.0
    rho_B = TB / TT if abs(TT) > 1e-12 else 0.0

    return {'rho_res': rho_A, 'rho_bil': rho_B, 'TT': TT, 'TA': TA, 'TB': TB}


# =============================================================================
# RANK-1 OPTIMIZATION
# =============================================================================

def batched_tn_inner(l1, r1, p1, l2, r2, p2):
    """Batched TN inner product. All tensors have leading batch dim B."""
    LL = torch.bmm(l1, l2.transpose(1, 2))
    RR = torch.bmm(r1, r2.transpose(1, 2))
    LR = torch.bmm(l1, r2.transpose(1, 2))
    RL = torch.bmm(r1, l2.transpose(1, 2))
    core = 0.5 * (LL * RR + LR * RL)
    DD = torch.bmm(p1.transpose(1, 2), p2)
    return (core * DD).sum(dim=(1, 2))


def optimize_rank1(target_L, target_R, target_D, n_seeds=10, steps=2000,
                   lr=1e-2, device='cuda'):
    """
    Optimize rank-1 approximation to a target tensor.
    Returns best (l, r, d) and TN-sim.
    """
    B = n_seeds
    h_target = target_L.shape[0]
    d_in = target_L.shape[1]  # n+1 for augmented
    n_out = target_D.shape[0]

    # Repeat target for batch
    tL = target_L.unsqueeze(0).expand(B, -1, -1)
    tR = target_R.unsqueeze(0).expand(B, -1, -1)
    tD = target_D.unsqueeze(0).expand(B, -1, -1)

    # Initialize rank-1 params
    w_l = torch.zeros(B, 1, d_in, device=device)
    w_r = torch.zeros(B, 1, d_in, device=device)
    w_p = torch.zeros(B, n_out, 1, device=device)
    for i in range(B):
        torch.manual_seed(i * 137 + 42)
        w_l[i] = torch.randn(1, d_in) * 0.1
        w_r[i] = torch.randn(1, d_in) * 0.1
        w_p[i] = torch.randn(n_out, 1) * 0.1

    w_l.requires_grad_(True)
    w_r.requires_grad_(True)
    w_p.requires_grad_(True)

    optimizer = torch.optim.Adam([w_l, w_r, w_p], lr=lr)

    for step in range(steps):
        optimizer.zero_grad()
        ab = batched_tn_inner(w_l, w_r, w_p, tL, tR, tD)
        aa = batched_tn_inner(w_l, w_r, w_p, w_l, w_r, w_p)
        bb = batched_tn_inner(tL, tR, tD, tL, tR, tD)
        sims = ab / (torch.sqrt(aa.clamp(min=1e-12)) * torch.sqrt(bb.clamp(min=1e-12)))
        loss = -sims.mean()
        loss.backward()
        optimizer.step()

    with torch.no_grad():
        ab = batched_tn_inner(w_l, w_r, w_p, tL, tR, tD)
        aa = batched_tn_inner(w_l, w_r, w_p, w_l, w_r, w_p)
        bb = batched_tn_inner(tL, tR, tD, tL, tR, tD)
        final_sims = ab / (torch.sqrt(aa.clamp(min=1e-12)) * torch.sqrt(bb.clamp(min=1e-12)))
        best = final_sims.argmax().item()

    return {
        'l': w_l[best].detach(),
        'r': w_r[best].detach(),
        'd': w_p[best].detach(),
        'sim': final_sims[best].item(),
    }


def project_rank1_onto_class(rank1_weights, model, target_class, device):
    """
    Compute <T_target | rank1> / <T_target | T_target>.
    How much of target class's tensor is explained by this rank-1?
    """
    l, r, d = rank1_weights['l'], rank1_weights['r'], rank1_weights['d']
    L_T, R_T, D_T = model.get_augmented_weights_for_class(target_class)

    T_r1 = tn_inner_1layer(L_T, R_T, D_T, l, r, d).item()
    TT = tn_inner_1layer(L_T, R_T, D_T, L_T, R_T, D_T).item()

    return T_r1 / TT if abs(TT) > 1e-12 else 0.0
