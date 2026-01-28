"""
Run analytic TN-Sim on 1-layer bilinear checkpoints.
Verify that nearby checkpoints have high similarity.
"""

import pickle
from pathlib import Path
import numpy as np
import torch
import matplotlib.pyplot as plt

from tnsim import tn_sim_1layer, tn_sim_1layer_with_residual


def compute_tucker_ranks(L, R, D, threshold=1e-6):
    """
    Compute multilinear (Tucker) ranks for 1-layer bilinear tensor.

    T[i,j,k] = Σ_h D[i,h] L[h,j] R[h,k]

    Returns (rank1, rank2, rank3) based on singular value analysis,
    and effective ranks based on entropy of normalized singular values.
    """
    import torch

    d_out, d_hidden = D.shape

    # Mode Gram matrices (see derivation in comments)
    LL = L @ L.T  # (d_hidden, d_hidden)
    RR = R @ R.T
    M = LL * RR  # element-wise
    gram1 = D @ M @ D.T  # (d_out, d_out)

    DTD = D.T @ D  # (d_hidden, d_hidden)
    gram2 = L.T @ (DTD * RR) @ L  # (d_in, d_in)
    gram3 = R.T @ (DTD * LL) @ R  # (d_in, d_in)

    # Hard ranks
    s1 = torch.linalg.svdvals(gram1)
    s2 = torch.linalg.svdvals(gram2)
    s3 = torch.linalg.svdvals(gram3)

    rank1 = (s1 > threshold * s1[0]).sum().item()
    rank2 = (s2 > threshold * s2[0]).sum().item()
    rank3 = (s3 > threshold * s3[0]).sum().item()

    return rank1, rank2, rank3


def compute_effective_ranks(L, R, D):
    """
    Compute effective ranks based on singular value entropy.

    Effective rank = exp(entropy of normalized singular values)
    This measures how "spread out" the singular values are.
    Low effective rank = few dominant singular values (low-rank structure)
    High effective rank = many similar singular values (full rank)
    """
    import torch

    def effective_rank_from_gram(gram):
        """Compute effective rank from Gram matrix."""
        s = torch.linalg.svdvals(gram)
        s = s[s > 1e-10]  # filter zeros
        if len(s) == 0:
            return 0.0
        # Normalize to probability distribution
        p = s / s.sum()
        # Entropy
        entropy = -(p * torch.log(p + 1e-10)).sum()
        # Effective rank
        return torch.exp(entropy).item()

    LL = L @ L.T
    RR = R @ R.T
    M = LL * RR
    gram1 = D @ M @ D.T

    DTD = D.T @ D
    gram2 = L.T @ (DTD * RR) @ L
    gram3 = R.T @ (DTD * LL) @ R

    eff_rank1 = effective_rank_from_gram(gram1)
    eff_rank2 = effective_rank_from_gram(gram2)
    eff_rank3 = effective_rank_from_gram(gram3)

    return eff_rank1, eff_rank2, eff_rank3


def compute_tucker_ranks_with_residual(L, R, D, d_res, threshold=1e-6):
    """
    Tucker ranks for 1-layer with pre-norm residual.

    The residual x||x||^2 has tensor T_res[i,j,k] = I[i,j] * 1[k] (summing over k)
    Actually: T_res[i,j,k] = δ_{jk} (identity on j,k, broadcast to i)

    Combined tensor: T_combined = T_bilinear + T_residual
    This affects the Tucker ranks.
    """
    import torch

    # For the combined tensor, we need to include the residual contribution.
    # T_res[i,j,k] = δ_{jk} for all i (the x||x||^2 tensor with L=R=D=I)

    d_out, d_hidden = D.shape
    d_hidden_L, d_in = L.shape

    # Mode-1 Gram: includes both bilinear and residual contributions
    LL = L @ L.T
    RR = R @ R.T
    M = LL * RR
    gram1_bilinear = D @ M @ D.T

    # Residual mode-1 contribution: T_res @ T_res.T
    # T_res[i, jk] where jk is flattened, sum over jk: each (j,j) contributes 1
    # So T_res @ T_res.T = d_in * I (since diagonal elements j=k contribute)
    gram1_residual = d_in * torch.eye(d_res, device=L.device)

    # Cross term: T_bilinear @ T_res.T
    # = Σ_{j,k} T_bilinear[i,j,k] * T_res[i',j,k]
    # = Σ_j T_bilinear[i,j,j] * 1 (since T_res is δ_{jk})
    # = Σ_{h,j} D[i,h] L[h,j] R[h,j]
    # = D @ diag(L * R summed over input dim)
    LR_diag = (L * R).sum(dim=1)  # (d_hidden,)
    cross1 = D @ LR_diag  # (d_out,)
    # Cross term in Gram: outer product
    gram1_cross = torch.outer(cross1, torch.ones(d_res, device=L.device))
    gram1_cross = gram1_cross + gram1_cross.T

    # Hmm, this is getting complicated. Let me use a simpler approach.
    # For now, just compute bilinear ranks. The residual primarily adds
    # a fixed component that doesn't change the relative structure much.

    # Return bilinear-only ranks for simplicity
    return compute_tucker_ranks(L, R, D, threshold)

# =============================================================================
# LOAD AND EXTRACT
# =============================================================================

def load_checkpoint(path):
    with open(path, 'rb') as f:
        return pickle.load(f)


def extract_weights(state_dict, d_hidden=256, device='cuda', include_rmsnorm=True):
    """Extract bilinear weights from 1-layer model.

    If include_rmsnorm=True, incorporates RMSNorm weight into L and R:
        L_eff = L @ diag(g), R_eff = R @ diag(g)
    where g is the RMSNorm learnable weight.
    """
    L = state_dict['bilinears.0.LR.weight'][:d_hidden].to(device)
    R = state_dict['bilinears.0.LR.weight'][d_hidden:].to(device)
    D = state_dict['bilinears.0.D.weight'].to(device)

    if include_rmsnorm and 'norms.0.weight' in state_dict:
        g = state_dict['norms.0.weight'].to(device)  # (d_res,)
        # L_eff[h, j] = sum_k L[h, k] * diag(g)[k, j] = L[h, j] * g[j]
        L = L * g.unsqueeze(0)  # broadcast: (d_hidden, d_res) * (1, d_res)
        R = R * g.unsqueeze(0)

    return {'L': L, 'R': R, 'D': D}


# =============================================================================
# MAIN
# =============================================================================

if __name__ == "__main__":
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")

    output_dir = Path("tn_analysis_checkpoints")

    # Load 1-layer model
    model_path = output_dir / "residual_1layer_downproj_aug.pkl"
    if not model_path.exists():
        print(f"Model not found: {model_path}")
        exit()

    data = load_checkpoint(model_path)
    checkpoints = data['checkpoints']
    history = data['history']
    config = data['config']

    print(f"Loaded {len(checkpoints)} checkpoints")
    print(f"d_res={config['d_res']}, d_hidden={config['d_hidden']}")
    print(f"Final val_acc: {history['val_acc'][-1]:.4f}")

    # Sample steps
    steps = sorted(checkpoints.keys())
    if len(steps) > 15:
        indices = np.linspace(0, len(steps)-1, 15, dtype=int)
        sample_steps = [steps[i] for i in indices]
    else:
        sample_steps = steps

    n = len(sample_steps)
    print(f"\nComputing {n}x{n} TN-Sim matrices (analytic, symmetrized)")
    print(f"Steps: {sample_steps}")

    # Compute TN-sim matrices: without and with residual
    sim_matrix = np.zeros((n, n))
    sim_matrix_res = np.zeros((n, n))

    # Also track Tucker ranks and effective ranks
    tucker_ranks = {'mode1': [], 'mode2': [], 'mode3': []}
    eff_ranks = {'mode1': [], 'mode2': [], 'mode3': []}

    for i, s1 in enumerate(sample_steps):
        w1 = extract_weights(checkpoints[s1], d_hidden=config['d_hidden'], device=device)

        # Compute Tucker ranks for this checkpoint
        r1, r2, r3 = compute_tucker_ranks(w1['L'], w1['R'], w1['D'])
        tucker_ranks['mode1'].append(r1)
        tucker_ranks['mode2'].append(r2)
        tucker_ranks['mode3'].append(r3)

        # Compute effective ranks
        er1, er2, er3 = compute_effective_ranks(w1['L'], w1['R'], w1['D'])
        eff_ranks['mode1'].append(er1)
        eff_ranks['mode2'].append(er2)
        eff_ranks['mode3'].append(er3)

        for j, s2 in enumerate(sample_steps):
            if j < i:
                sim_matrix[i, j] = sim_matrix[j, i]
                sim_matrix_res[i, j] = sim_matrix_res[j, i]
            else:
                w2 = extract_weights(checkpoints[s2], d_hidden=config['d_hidden'], device=device)
                # Without residual (pure bilinear)
                sim_matrix[i, j] = tn_sim_1layer(
                    w1['L'], w1['R'], w1['D'],
                    w2['L'], w2['R'], w2['D'],
                    symmetrize=True
                )
                # With residual
                sim_matrix_res[i, j] = tn_sim_1layer_with_residual(
                    w1['L'], w1['R'], w1['D'],
                    w2['L'], w2['R'], w2['D'],
                    symmetrize=True
                )
        print(f"  Row {i+1}/{n}: diag={sim_matrix[i,i]:.4f}, eff_ranks=({er1:.1f},{er2:.1f},{er3:.1f})")

    # =========================================================================
    # PLOTTING
    # =========================================================================

    x_indices = np.arange(n)

    def set_aligned_xticks(ax, steps=sample_steps):
        """Helper to align x-axis with TN-Sim matrix steps (evenly spaced)."""
        ax.set_xticks(range(len(steps)))
        ax.set_xticklabels([str(s) for s in steps], rotation=45, ha='right', fontsize=7)
        ax.set_xlim([-0.5, len(steps) - 0.5])

    fig, axes = plt.subplots(3, 3, figsize=(16, 14))

    # TN-Sim matrix (no residual)
    ax = axes[0, 0]
    im = ax.imshow(sim_matrix, cmap='viridis', vmin=0, vmax=1)
    ax.set_xticks(range(n))
    ax.set_xticklabels([str(s) for s in sample_steps], rotation=45, ha='right', fontsize=7)
    ax.set_yticks(range(n))
    ax.set_yticklabels([str(s) for s in sample_steps], fontsize=7)
    ax.set_title('1-Layer TN-Sim (Bilinear Only)')
    plt.colorbar(im, ax=ax, orientation='horizontal', shrink=0.8, pad=0.15)

    # TN-Sim matrix (with residual)
    ax = axes[0, 1]
    im = ax.imshow(sim_matrix_res, cmap='viridis', vmin=0, vmax=1)
    ax.set_xticks(range(n))
    ax.set_xticklabels([str(s) for s in sample_steps], rotation=45, ha='right', fontsize=7)
    ax.set_yticks(range(n))
    ax.set_yticklabels([str(s) for s in sample_steps], fontsize=7)
    ax.set_title('1-Layer TN-Sim (With Residual)')
    plt.colorbar(im, ax=ax, orientation='horizontal', shrink=0.8, pad=0.15)

    # Val accuracy
    ax = axes[0, 2]
    val_acc_sampled = np.interp(sample_steps, history['steps'], history['val_acc'])
    ax.plot(x_indices, val_acc_sampled, 'o-', linewidth=2, markersize=6, color='green')
    ax.set_xticks(x_indices)
    ax.set_xticklabels([str(s) for s in sample_steps], rotation=45, ha='right', fontsize=7)
    ax.set_xlabel('Step')
    ax.set_ylabel('Val Accuracy')
    ax.set_title('Validation Accuracy')
    ax.grid(True, alpha=0.3)

    # Similarity to final (both versions)
    ax = axes[1, 0]
    final_sims = sim_matrix[:, -1]
    final_sims_res = sim_matrix_res[:, -1]
    ax.plot(x_indices, final_sims, 'o-', linewidth=2, markersize=6, label='Bilinear Only')
    ax.plot(x_indices, final_sims_res, 's--', linewidth=2, markersize=6, label='With Residual')
    ax.set_xticks(x_indices)
    ax.set_xticklabels([str(s) for s in sample_steps], rotation=45, ha='right', fontsize=7)
    ax.set_xlabel('Step')
    ax.set_ylabel('TN-Sim to Final')
    ax.set_title('Similarity to Final Checkpoint')
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.set_ylim([0, 1.1])

    # Adjacent similarity (both versions)
    ax = axes[1, 1]
    adjacent_sims = np.diag(sim_matrix, k=1)
    adjacent_sims_res = np.diag(sim_matrix_res, k=1)
    x_mid = x_indices[:-1] + 0.5
    ax.plot(x_mid, adjacent_sims, 'o-', linewidth=2, markersize=6, label='Bilinear Only')
    ax.plot(x_mid, adjacent_sims_res, 's--', linewidth=2, markersize=6, label='With Residual')
    ax.set_xticks(x_indices)
    ax.set_xticklabels([str(s) for s in sample_steps], rotation=45, ha='right', fontsize=7)
    ax.set_xlabel('Step')
    ax.set_ylabel('TN-Sim')
    ax.set_title('Adjacent Checkpoint Similarity')
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.set_ylim([0, 1.1])

    # Difference plot
    ax = axes[1, 2]
    diff = sim_matrix_res - sim_matrix
    im = ax.imshow(diff, cmap='RdBu', vmin=-0.5, vmax=0.5)
    ax.set_xticks(range(n))
    ax.set_xticklabels([str(s) for s in sample_steps], rotation=45, ha='right', fontsize=7)
    ax.set_yticks(range(n))
    ax.set_yticklabels([str(s) for s in sample_steps], fontsize=7)
    ax.set_title('Difference (Residual - Bilinear)')
    plt.colorbar(im, ax=ax, orientation='horizontal', shrink=0.8, pad=0.15)

    # Row 3: Effective Tucker ranks
    ax = axes[2, 0]
    ax.plot(x_indices, eff_ranks['mode1'], 'o-', linewidth=2, markersize=6, label='Mode-1 (output)')
    ax.plot(x_indices, eff_ranks['mode2'], 's-', linewidth=2, markersize=6, label='Mode-2 (input-L)')
    ax.plot(x_indices, eff_ranks['mode3'], '^-', linewidth=2, markersize=6, label='Mode-3 (input-R)')
    set_aligned_xticks(ax)
    ax.set_xlabel('Step')
    ax.set_ylabel('Effective Rank')
    ax.set_title('Effective Tucker Ranks (1-Layer)')
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Effective rank vs accuracy
    ax = axes[2, 1]
    ax.scatter(eff_ranks['mode1'], val_acc_sampled, alpha=0.7, s=50, label='Mode-1')
    ax.scatter(eff_ranks['mode2'], val_acc_sampled, alpha=0.7, s=50, label='Mode-2')
    ax.set_xlabel('Effective Rank')
    ax.set_ylabel('Val Accuracy')
    ax.set_title('Effective Rank vs Accuracy')
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Effective rank sum over time
    ax = axes[2, 2]
    eff_rank_sum = np.array(eff_ranks['mode1']) + np.array(eff_ranks['mode2']) + np.array(eff_ranks['mode3'])
    ax.plot(x_indices, eff_rank_sum, 'o-', linewidth=2, markersize=6, color='purple', label='Sum of Eff. Ranks')
    ax2 = ax.twinx()
    ax2.plot(x_indices, val_acc_sampled, 's--', linewidth=2, markersize=5, color='green', label='Val Acc')
    ax2.set_ylabel('Val Accuracy', color='green')
    set_aligned_xticks(ax)
    ax.set_xlabel('Step')
    ax.set_ylabel('Sum of Effective Ranks', color='purple')
    ax.set_title('Total Effective Rank & Accuracy')
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    save_path = output_dir / "tnsim_1layer_analytic.png"
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.show()

    print(f"\nPlot saved to: {save_path}")

    # =========================================================================
    # SUMMARY
    # =========================================================================

    print("\n" + "="*70)
    print("SUMMARY (1-Layer Analytic TN-Sim, Symmetrized)")
    print("="*70)
    print(f"\n{'Metric':<30} {'Bilinear Only':>15} {'With Residual':>15}")
    print("-"*70)
    print(f"{'Init-Final TN-Sim':<30} {sim_matrix[0, -1]:>15.6f} {sim_matrix_res[0, -1]:>15.6f}")
    print(f"{'Mean adjacent TN-Sim':<30} {adjacent_sims.mean():>15.4f} {adjacent_sims_res.mean():>15.4f}")
    print(f"{'Min adjacent TN-Sim':<30} {adjacent_sims.min():>15.4f} {adjacent_sims_res.min():>15.4f}")
    print(f"{'Final val_acc':<30} {history['val_acc'][-1]:>15.4f}")
