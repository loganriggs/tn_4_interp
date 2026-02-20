# %% [markdown]
# # Sparse Circuit Discovery: Phases 3-5
#
# Phase 3: Weight perturbation for additional sparsity (with fidelity cost)
# Phase 4: Hidden dimension expansion to resolve superposition
# Phase 5: Circuit interpretation and feature cards
#
# Run sparse_rotation.py first (Phase 2) to get rotation-only results.

# %% Configuration
import sys, os

PROJECT_ROOT = "/workspace/tn_4_interp"
ODT_DIR = os.path.join(PROJECT_ROOT, "odt")
sys.path.insert(0, ODT_DIR)
os.chdir(ODT_DIR)

CHECKPOINT = "models/chi_net_L3_h256.pt"
ODT_CACHE = "models/odt_cache_{model_name}.pt"
DEVICE = "cuda"
RESULTS_DIR = os.path.join(PROJECT_ROOT, "sparse_circuits", "results")
os.makedirs(RESULTS_DIR, exist_ok=True)

# Phase 3 hyperparameters
FIDELITY_LAMBDA = 1.0    # weight on fidelity loss (sweep this)
PRUNE_LR = 0.005
PRUNE_STEPS = 3000
PRUNE_LOG_EVERY = 200

# %% Imports
import torch
import numpy as np
import matplotlib.pyplot as plt
import math
from copy import deepcopy

from model import ChiNet, ResidualBilinearNet
from odt import run_odt, effective_bond_dim, truncate
from analyze import load_model, extract_cores, evaluate_truncated
from train import get_svhn_loaders


# %% [markdown]
# ## Phase 3: Joint rotation + perturbation

# %%
def tensor_similarity(T1, T2):
    """Generalized cosine similarity between two tensors (Frobenius inner product / norms)."""
    inner = (T1 * T2).sum()
    n1 = T1.norm()
    n2 = T2.norm()
    return inner / (n1 * n2 + 1e-12)


def hoyer(x, eps=1e-8):
    """Hoyer sparsity: 0 = dense, 1 = one-hot."""
    n = x.numel()
    if n <= 1:
        return torch.tensor(0.0, device=x.device)
    l1 = x.abs().sum()
    l2 = x.norm()
    return (math.sqrt(n) - l1 / (l2 + eps)) / (math.sqrt(n) - 1)


def skew_to_rotation(params, n):
    """Convert n*(n-1)/2 parameters to n x n orthogonal matrix."""
    A = torch.zeros(n, n, device=params.device, dtype=params.dtype)
    idx = torch.triu_indices(n, n, offset=1)
    A[idx[0], idx[1]] = params
    A = A - A.T
    return torch.matrix_exp(A)


def rotate_core(T, Q_in, Q_out):
    """Apply rotations to core tensor."""
    T1 = torch.einsum('oij,ko->kij', T, Q_out)
    T2 = torch.einsum('kij,ai->kaj', T1, Q_in)
    T3 = torch.einsum('kaj,bj->kab', T2, Q_in)
    return T3


def joint_loss(cores_orig, cores_perturbed, rotations, ranks, lam):
    """Joint sparsity + fidelity loss.

    Args:
        cores_orig: original truncated cores (no perturbation)
        cores_perturbed: cores + delta (learnable perturbation)
        rotations: orthogonal matrices per bond
        ranks: truncation ranks
        lam: fidelity weight

    Returns:
        total_loss, sparsity_loss, fidelity_loss
    """
    n_layers = len(cores_orig)
    sparsity = torch.tensor(0.0, device=cores_orig[0].device)
    fidelity = torch.tensor(0.0, device=cores_orig[0].device)

    for ell in range(n_layers):
        Q_in = rotations[ell]
        Q_out = rotations[ell + 1]

        # Rotated perturbed core
        T_rot = rotate_core(cores_perturbed[ell], Q_in, Q_out)
        T_sym = 0.5 * (T_rot + T_rot.transpose(-1, -2))

        # Sparsity: normalized Hoyer per output feature
        r_out = T_sym.shape[0]
        for k in range(r_out):
            M_k = T_sym[k]
            norm_k = M_k.norm()
            if norm_k > 1e-8:
                sparsity = sparsity + (1.0 - hoyer(M_k / norm_k))

        # Fidelity: tensor similarity between original and perturbed
        sim = tensor_similarity(cores_orig[ell], cores_perturbed[ell])
        fidelity = fidelity + (1.0 - sim)

    sparsity = sparsity / sum(c.shape[0] for c in cores_orig)
    fidelity = fidelity / n_layers

    total = sparsity + lam * fidelity
    return total, sparsity, fidelity


def run_phase3(cores_trunc, ranks, device, lam=1.0, lr=0.005, n_steps=3000,
               log_every=200, init_rotations=None, early_stop_patience=100,
               early_stop_rtol=1e-4):
    """Run Phase 3: joint rotation + perturbation optimization.

    Args:
        cores_trunc: list of truncated core tensors
        ranks: truncation ranks
        device: torch device
        lam: fidelity weight
        lr: learning rate
        n_steps: max optimization steps
        log_every: logging frequency
        init_rotations: initial rotation params (from Phase 2), or None for identity
        early_stop_patience: stop if loss doesn't improve by rtol over this many steps
        early_stop_rtol: relative improvement threshold for early stopping

    Returns:
        dict with final cores, rotations, losses, accuracy info
    """
    cores_dev = [c.detach().to(device) for c in cores_trunc]

    # Learnable perturbation (initialized to zero)
    deltas = [torch.zeros_like(c, requires_grad=True) for c in cores_dev]

    # Rotation parameters
    rot_params = []
    for i, r in enumerate(ranks):
        n_params = r * (r - 1) // 2
        if init_rotations is not None:
            p = init_rotations[i].clone().to(device).requires_grad_(True)
        else:
            p = torch.zeros(n_params, device=device, requires_grad=True)
        rot_params.append(p)

    all_params = deltas + rot_params
    optimizer = torch.optim.Adam(all_params, lr=lr)
    history = {'total': [], 'sparsity': [], 'fidelity': []}

    best_loss = float('inf')
    steps_without_improvement = 0

    for step in range(n_steps):
        optimizer.zero_grad()

        cores_pert = [c + d for c, d in zip(cores_dev, deltas)]
        rotations = [skew_to_rotation(p, r) for p, r in zip(rot_params, ranks)]

        total, sp, fi = joint_loss(cores_dev, cores_pert, rotations, ranks, lam)
        total.backward()
        optimizer.step()

        loss_val = total.item()
        history['total'].append(loss_val)
        history['sparsity'].append(sp.item())
        history['fidelity'].append(fi.item())

        # Early stopping check
        if loss_val < best_loss * (1 - early_stop_rtol):
            best_loss = loss_val
            steps_without_improvement = 0
        else:
            steps_without_improvement += 1

        if step % log_every == 0 or step == n_steps - 1:
            print(f"  Step {step:4d}: total={loss_val:.4f} "
                  f"(sparsity={sp.item():.4f}, fidelity={fi.item():.6f})")

        if steps_without_improvement >= early_stop_patience and step >= early_stop_patience:
            print(f"  Early stopping at step {step} (no improvement for {early_stop_patience} steps)")
            break

    # Extract final results
    with torch.no_grad():
        rotations_final = [skew_to_rotation(p, r) for p, r in zip(rot_params, ranks)]
        cores_final = []
        for ell in range(len(cores_trunc)):
            c_pert = cores_dev[ell] + deltas[ell]
            T_rot = rotate_core(c_pert, rotations_final[ell], rotations_final[ell + 1])
            cores_final.append(T_rot)

    return {
        'cores_final': [c.detach().cpu() for c in cores_final],
        'deltas': [d.detach().cpu() for d in deltas],
        'rot_params': [p.detach().cpu() for p in rot_params],
        'rotations': [Q.detach().cpu() for Q in rotations_final],
        'history': history,
        'lam': lam,
    }


# %% [markdown]
# ## Phase 4: Hidden dimension expansion (stub)

# %%
def expand_cores(cores, ranks, expansion_factor=2):
    """Expand hidden dimensions by adding zero-initialized auxiliary dims.

    The first r_ℓ dimensions reproduce the original network exactly.
    Additional dimensions start at zero and can be optimized for sparsity.

    Args:
        cores: list of truncated core tensors (r_out, r_in, r_in)
        ranks: original truncation ranks
        expansion_factor: multiply each rank by this

    Returns:
        expanded_cores: list of expanded tensors
        expanded_ranks: new rank list
    """
    expanded_ranks = [r * expansion_factor for r in ranks]
    expanded_cores = []

    for ell, core in enumerate(cores):
        r_out, r_in, _ = core.shape
        r_out_new = expanded_ranks[ell + 1]
        r_in_new = expanded_ranks[ell]

        T_new = torch.zeros(r_out_new, r_in_new, r_in_new, dtype=core.dtype)
        T_new[:r_out, :r_in, :r_in] = core
        expanded_cores.append(T_new)

    return expanded_cores, expanded_ranks


def description_length(cores, threshold_frac=1e-3, bits_per_value=16):
    """Compute description length: total nonzeros * (log(dim) + bits_per_value).

    Lower = more compressible circuit.
    """
    total_dl = 0
    for core in cores:
        max_val = core.abs().max()
        threshold = threshold_frac * max_val
        nnz = (core.abs() > threshold).sum().item()
        dim = max(core.shape)
        total_dl += nnz * (math.log2(dim) + bits_per_value)
    return total_dl


# %% [markdown]
# ## Run experiments
#
# Uncomment to run after Phase 2 (sparse_rotation.py).

# %%
if __name__ == '__main__':
    print("Loading model and ODT results...")
    model, ckpt = load_model(CHECKPOINT, DEVICE)
    model_name = f"{ckpt['model_type']}_L{ckpt['n_layers']}_h{ckpt['hidden_dim']}"

    cache_path = ODT_CACHE.format(model_name=model_name) if ODT_CACHE else None
    if cache_path and os.path.isfile(cache_path):
        print(f"Loading cached ODT results from {cache_path}")
        odt_results = torch.load(cache_path, map_location='cpu', weights_only=False)
    else:
        embed, cores_raw, unembed = extract_cores(model.cpu())
        odt_results = run_odt(embed, cores_raw, unembed, verbose=True)
        if cache_path:
            torch.save(odt_results, cache_path)
            print(f"Saved ODT cache to {cache_path}")
    ranks = effective_bond_dim(odt_results['eigenvalues'])

    embed_trunc, cores_trunc, unembed_trunc = truncate(
        odt_results['embed_orth'], odt_results['cores_orth'],
        odt_results['unembed_mod'], odt_results['eigenvectors'], ranks)

    # Load Phase 2 rotation results as initialization
    phase2 = torch.load(
        os.path.join(RESULTS_DIR, f'{model_name}_sparse_rotation.pt'),
        map_location='cpu', weights_only=False)

    # Sweep fidelity weight
    print("\n--- Phase 3: Sweeping fidelity weight ---")
    for lam in [10.0, 1.0, 0.1, 0.01]:
        print(f"\nlambda = {lam}")
        result = run_phase3(
            cores_trunc, ranks, DEVICE,
            lam=lam, lr=PRUNE_LR, n_steps=PRUNE_STEPS,
            log_every=PRUNE_LOG_EVERY,
            init_rotations=phase2['rot_params'])

        # Save
        save_path = os.path.join(RESULTS_DIR, f'{model_name}_phase3_lam{lam}.pt')
        torch.save(result, save_path)
        print(f"  Saved to {save_path}")
