# %% [markdown]
# # Sparse Circuit Discovery: Phase 2 — Rotation Optimization
#
# Optimizes orthogonal rotations Q_ℓ at each ODT bond to minimize
# the Hoyer sparsity of the interaction matrices T'[k,:,:].
# The network's function is *exactly* invariant under these rotations.

# %% Imports
import sys, os
import torch
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
import math
from copy import deepcopy

PROJECT_ROOT = "/workspace/tn_4_interp"
ODT_DIR = os.path.join(PROJECT_ROOT, "odt")
sys.path.insert(0, ODT_DIR)

from odt import run_odt, effective_bond_dim, truncate

# %% [markdown]
# ## Core functions

# %%
def skew_to_rotation(params, n):
    """Convert n*(n-1)/2 parameters to n x n orthogonal matrix via matrix exponential."""
    A = torch.zeros(n, n, device=params.device, dtype=params.dtype)
    idx = torch.triu_indices(n, n, offset=1)
    A[idx[0], idx[1]] = params
    A = A - A.T  # skew-symmetric
    return torch.matrix_exp(A)


def rotate_core(T, Q_in, Q_out):
    """Apply rotations to a core tensor: T'[k,i,j] = Q_out @ T @ (Q_in x Q_in).
    Q_in rotates the input bond, Q_out rotates the output bond."""
    T1 = torch.einsum('oij,ko->kij', T, Q_out)
    T2 = torch.einsum('kij,ai->kaj', T1, Q_in)
    T3 = torch.einsum('kaj,bj->kab', T2, Q_in)
    return T3


def hoyer(x, eps=1e-8):
    """Hoyer sparsity measure: 0 = maximally dense, 1 = one-hot."""
    n = x.numel()
    if n <= 1:
        return torch.tensor(0.0, device=x.device)
    l1 = x.abs().sum()
    l2 = x.norm()
    return (math.sqrt(n) - l1 / (l2 + eps)) / (math.sqrt(n) - 1)


def sparsity_loss(cores, rotations, ranks):
    """Compute total sparsity loss: normalized Hoyer of each T'[k,:,:].

    Args:
        cores: list of truncated core tensors (r_out, r_in, r_in)
        rotations: list of orthogonal matrices Q_ℓ, one per bond
        ranks: list of ranks per bond

    Returns:
        loss: scalar, lower = sparser
        stats: dict with per-layer info
    """
    n_layers = len(cores)
    total_loss = torch.tensor(0.0, device=cores[0].device)
    stats = {}

    for ell in range(n_layers):
        Q_in = rotations[ell]
        Q_out = rotations[ell + 1]
        T_rot = rotate_core(cores[ell], Q_in, Q_out)

        # Symmetrize (bilinear with cloning)
        T_sym = 0.5 * (T_rot + T_rot.transpose(-1, -2))

        r_out = T_sym.shape[0]
        layer_loss = torch.tensor(0.0, device=cores[0].device)
        n_active = 0

        for k in range(r_out):
            M_k = T_sym[k]
            norm_k = M_k.norm()
            if norm_k > 1e-8:
                layer_loss = layer_loss + (1.0 - hoyer(M_k / norm_k))
                n_active += 1

        if n_active > 0:
            layer_loss = layer_loss / n_active

        stats[f'layer_{ell}'] = {
            'loss': layer_loss.item(),
            'n_active': n_active,
            'shape': tuple(T_sym.shape),
        }
        total_loss = total_loss + layer_loss

    total_loss = total_loss / n_layers
    return total_loss, stats


def count_nonzeros(cores, rotations, threshold_frac=1e-3):
    """Count nonzero entries in rotated cores (relative threshold)."""
    counts = {}
    for ell in range(len(cores)):
        Q_in = rotations[ell]
        Q_out = rotations[ell + 1]
        T_rot = rotate_core(cores[ell], Q_in, Q_out)
        T_sym = 0.5 * (T_rot + T_rot.transpose(-1, -2))

        max_val = T_sym.abs().max()
        threshold = threshold_frac * max_val
        nnz = (T_sym.abs() > threshold).sum().item()
        total = T_sym.numel()
        counts[f'layer_{ell}'] = {
            'nnz': nnz,
            'total': total,
            'sparsity': 1.0 - nnz / total,
            'max_val': max_val.item(),
        }
    return counts


# %% [markdown]
# ## Main experiment (only runs when executed directly)

# %%
if __name__ == '__main__':
    os.chdir(ODT_DIR)

    from analyze import load_model, extract_cores, evaluate_truncated
    from train import get_svhn_loaders

    # Configuration
    CHECKPOINT = "models/chi_net_L3_h256.pt"  # or "models/residual_L3_h256.pt"
    ODT_CACHE = "models/odt_cache_{model_name}.pt"
    DEVICE = "cuda"
    SAVE_DIR = os.path.join(PROJECT_ROOT, "sparse_circuits", "results")
    os.makedirs(SAVE_DIR, exist_ok=True)
    LR = 0.01
    N_STEPS = 2000
    LOG_EVERY = 100

    # Load model
    print("Loading model...")
    model, ckpt = load_model(CHECKPOINT, DEVICE)
    model_name = f"{ckpt['model_type']}_L{ckpt['n_layers']}_h{ckpt['hidden_dim']}"

    cache_path = ODT_CACHE.format(model_name=model_name)
    if os.path.isfile(cache_path):
        print(f"Loading cached ODT results from {cache_path}")
        odt_results = torch.load(cache_path, map_location='cpu', weights_only=False)
    else:
        print("Extracting cores and running ODT...")
        embed, cores_raw, unembed = extract_cores(model.cpu())
        odt_results = run_odt(embed, cores_raw, unembed, verbose=True)
        torch.save(odt_results, cache_path)
        print(f"Saved ODT cache to {cache_path}")

    # Get ranks and truncate
    eff_ranks = effective_bond_dim(odt_results['eigenvalues'])
    print(f"Effective ranks: {eff_ranks}")
    ranks = eff_ranks

    embed_trunc, cores_trunc, unembed_trunc = truncate(
        odt_results['embed_orth'], odt_results['cores_orth'],
        odt_results['unembed_mod'], odt_results['eigenvectors'], ranks)

    print(f"Truncated core shapes: {[c.shape for c in cores_trunc]}")
    total_params = sum(math.comb(r, 2) for r in ranks)
    print(f"Rotation parameters: {total_params} (ranks {ranks})")

    # Evaluate baseline
    print("\n--- Baseline (identity rotations) ---")
    device = torch.device(DEVICE)
    cores_trunc_dev = [c.detach().to(device) for c in cores_trunc]
    identity_rots = [torch.eye(r, device=device) for r in ranks]

    baseline_loss, baseline_stats = sparsity_loss(cores_trunc_dev, identity_rots, ranks)
    baseline_nnz = count_nonzeros(cores_trunc_dev, identity_rots)

    print(f"Sparsity loss (lower=sparser): {baseline_loss.item():.4f}")
    for ell in range(len(cores_trunc)):
        s = baseline_stats[f'layer_{ell}']
        n = baseline_nnz[f'layer_{ell}']
        print(f"  Layer {ell}: loss={s['loss']:.4f}, shape={s['shape']}, "
              f"nnz={n['nnz']}/{n['total']} ({n['sparsity']:.1%} sparse)")

    # Optimize rotations
    print(f"\n--- Optimizing rotations (lr={LR}, {N_STEPS} steps) ---")
    rot_params = []
    for r in ranks:
        n_params = r * (r - 1) // 2
        p = torch.zeros(n_params, device=device, requires_grad=True)
        rot_params.append(p)

    optimizer = torch.optim.Adam(rot_params, lr=LR)
    history = {'loss': [], 'nnz': []}

    for step in range(N_STEPS):
        optimizer.zero_grad()
        rotations = [skew_to_rotation(p, r) for p, r in zip(rot_params, ranks)]
        loss, stats = sparsity_loss(cores_trunc_dev, rotations, ranks)
        loss.backward()
        optimizer.step()
        history['loss'].append(loss.item())

        if step % LOG_EVERY == 0 or step == N_STEPS - 1:
            nnz = count_nonzeros(cores_trunc_dev, rotations)
            total_nnz = sum(v['nnz'] for v in nnz.values())
            total_entries = sum(v['total'] for v in nnz.values())
            history['nnz'].append(total_nnz)
            print(f"  Step {step:4d}: loss={loss.item():.4f}, "
                  f"nnz={total_nnz}/{total_entries} ({1-total_nnz/total_entries:.1%} sparse)")

    # Results
    print("\n--- After rotation ---")
    with torch.no_grad():
        rotations_final = [skew_to_rotation(p, r) for p, r in zip(rot_params, ranks)]
        final_loss, final_stats = sparsity_loss(cores_trunc_dev, rotations_final, ranks)
        final_nnz = count_nonzeros(cores_trunc_dev, rotations_final)

    print(f"Sparsity loss: {baseline_loss.item():.4f} -> {final_loss.item():.4f}")
    for ell in range(len(cores_trunc)):
        sb = baseline_stats[f'layer_{ell}']
        sf = final_stats[f'layer_{ell}']
        nb = baseline_nnz[f'layer_{ell}']
        nf = final_nnz[f'layer_{ell}']
        print(f"  Layer {ell}: loss {sb['loss']:.4f}->{sf['loss']:.4f}, "
              f"nnz {nb['nnz']}->{nf['nnz']}/{nf['total']} "
              f"({nb['sparsity']:.1%}->{nf['sparsity']:.1%} sparse)")

    # Accuracy check
    print("\n--- Accuracy check ---")
    _, test_loader, _ = get_svhn_loaders(
        data_root=os.path.join(PROJECT_ROOT, "modular_addition/dev_interp/notebooks/data"))

    with torch.no_grad():
        rotated_cores = []
        for ell in range(len(cores_trunc)):
            Q_in = rotations_final[ell]
            Q_out = rotations_final[ell + 1]
            T_rot = rotate_core(cores_trunc_dev[ell], Q_in, Q_out)
            rotated_cores.append(T_rot)

        Q_first = rotations_final[0]
        Q_last = rotations_final[-1]
        embed_rot = Q_first @ embed_trunc.to(device)
        unembed_rot = unembed_trunc.to(device) @ Q_last.T

    acc_before = evaluate_truncated(
        embed_trunc, cores_trunc, unembed_trunc, model, test_loader, DEVICE)
    acc_after = evaluate_truncated(
        embed_rot.cpu(), [c.cpu() for c in rotated_cores], unembed_rot.cpu(),
        model, test_loader, DEVICE)
    print(f"  Accuracy: {acc_before:.4f} (before) -> {acc_after:.4f} (after)")
    print(f"  Difference: {abs(acc_after - acc_before):.6f} (should be ~0)")

    # Visualize
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
    ax1.plot(history['loss'])
    ax1.set_xlabel('Step')
    ax1.set_ylabel('Sparsity loss')
    ax1.set_title('Rotation optimization')
    ax1.axhline(baseline_loss.item(), color='r', ls='--', label='baseline')
    ax1.legend()

    n_layers = len(cores_trunc)
    fig2, axes = plt.subplots(2, n_layers, figsize=(4 * n_layers, 8))
    if n_layers == 1:
        axes = axes.reshape(2, 1)

    with torch.no_grad():
        for ell in range(n_layers):
            T_before = 0.5 * (cores_trunc_dev[ell] + cores_trunc_dev[ell].transpose(-1, -2))
            Q_in = rotations_final[ell]
            Q_out = rotations_final[ell + 1]
            T_after = rotate_core(cores_trunc_dev[ell], Q_in, Q_out)
            T_after = 0.5 * (T_after + T_after.transpose(-1, -2))

            feat = 0
            M_before = T_before[feat].cpu().numpy()
            M_after = T_after[feat].cpu().numpy()

            vmax = max(np.abs(M_before).max(), np.abs(M_after).max())
            axes[0, ell].imshow(M_before, cmap='RdBu_r', vmin=-vmax, vmax=vmax, aspect='equal')
            axes[0, ell].set_title(f'Layer {ell} feat 0\nBEFORE', fontsize=9)
            axes[1, ell].imshow(M_after, cmap='RdBu_r', vmin=-vmax, vmax=vmax, aspect='equal')
            axes[1, ell].set_title(f'Layer {ell} feat 0\nAFTER', fontsize=9)

    fig2.suptitle(f'{model_name}: T\'[0,:,:] before/after sparse rotation', fontsize=12)
    plt.tight_layout()
    plt.savefig(os.path.join(SAVE_DIR, f'{model_name}_rotation_comparison.png'), dpi=150)
    plt.close()

    for ell in range(n_layers):
        Q_in = rotations_final[ell]
        Q_out = rotations_final[ell + 1]
        with torch.no_grad():
            T_rot = rotate_core(cores_trunc_dev[ell], Q_in, Q_out)
            T_sym = 0.5 * (T_rot + T_rot.transpose(-1, -2))

        r_out, r_in, _ = T_sym.shape
        n_cols = min(8, r_out)
        n_rows = (r_out + n_cols - 1) // n_cols
        fig3, axes3 = plt.subplots(n_rows, n_cols, figsize=(2.5 * n_cols, 2.5 * n_rows))
        if n_rows == 1:
            axes3 = axes3.reshape(1, -1)

        for k in range(r_out):
            r, c = k // n_cols, k % n_cols
            M_k = T_sym[k].cpu().numpy()
            vmax = np.abs(M_k).max()
            axes3[r, c].imshow(M_k, cmap='RdBu_r',
                               vmin=-vmax if vmax > 0 else -1,
                               vmax=vmax if vmax > 0 else 1,
                               aspect='equal')
            axes3[r, c].set_title(f'k={k}', fontsize=8)
            axes3[r, c].set_xticks(range(r_in))
            axes3[r, c].set_yticks(range(r_in))
            axes3[r, c].tick_params(labelsize=5)

        for k in range(r_out, n_rows * n_cols):
            r, c = k // n_cols, k % n_cols
            axes3[r, c].axis('off')

        fig3.suptitle(f'{model_name} Layer {ell}: all T\'[k,:,:] after rotation '
                      f'({r_out}x{r_in}x{r_in})', fontsize=11)
        plt.tight_layout()
        plt.savefig(os.path.join(SAVE_DIR, f'{model_name}_layer{ell}_all_Tprime.png'), dpi=150)
        plt.close()

    print(f"\nSaved results to {SAVE_DIR}/")

    # Save
    save_path = os.path.join(SAVE_DIR, f'{model_name}_sparse_rotation.pt')
    torch.save({
        'model_name': model_name,
        'ranks': ranks,
        'rot_params': [p.detach().cpu() for p in rot_params],
        'rotations': [Q.detach().cpu() for Q in rotations_final],
        'rotated_cores': [c.detach().cpu() for c in rotated_cores],
        'embed_rot': embed_rot.detach().cpu(),
        'unembed_rot': unembed_rot.detach().cpu(),
        'baseline_loss': baseline_loss.item(),
        'final_loss': final_loss.item(),
        'acc_before': acc_before,
        'acc_after': acc_after,
        'history': history,
    }, save_path)
    print(f"Saved rotated cores to {save_path}")
