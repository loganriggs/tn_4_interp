"""
Analyze T2 tensor structure and cross-layer alignment across checkpoints.
Focus on phase changes in seed 1.
"""

import torch
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import pickle


def analyze_T2(D, L2, R2):
    """
    Analyze the layer-2 mixing tensor.

    T2[p, h1, h1'] = sum_h2 D[p,h2] * L2[h2,h1] * R2[h2,h1']

    This tells us how layer 2 combines layer 1 features for each output class.
    """
    # T2[p, h1, h1'] = sum_h2 D[p,h2] * L2[h2,h1] * R2[h2,h1']
    T2 = torch.einsum('ph,ha,hb->pab', D, L2, R2)

    P, d1, _ = T2.shape

    results = {}

    # 1. Diagonal vs off-diagonal energy (per output class)
    diag_fracs = []
    for p in range(P):
        diag_energy = torch.diag(T2[p]).pow(2).sum()
        total_energy = T2[p].pow(2).sum()
        diag_fracs.append(float(diag_energy / (total_energy + 1e-10)))
    results['diag_frac_per_class'] = diag_fracs
    results['diag_frac_mean'] = np.mean(diag_fracs)

    # 2. Symmetry: T2[p] vs T2[p].T
    sym_errors = []
    for p in range(P):
        sym_err = (T2[p] - T2[p].T).pow(2).sum() / (T2[p].pow(2).sum() + 1e-10)
        sym_errors.append(float(sym_err))
    results['symmetry_error_per_class'] = sym_errors
    results['symmetry_error_mean'] = np.mean(sym_errors)

    # 3. Low-rank structure of T2[p]
    ranks = []
    top_sv_fracs = []
    for p in range(P):
        svd_vals = torch.linalg.svdvals(T2[p])
        rank = int((svd_vals > 0.01 * svd_vals[0]).sum())
        ranks.append(rank)
        top_sv_fracs.append(float(svd_vals[0] / (svd_vals.sum() + 1e-10)))
    results['T2_rank_per_class'] = ranks
    results['T2_rank_mean'] = np.mean(ranks)
    results['top_sv_frac_per_class'] = top_sv_fracs

    # 4. Total Frobenius norm of T2
    results['T2_norm'] = float(T2.pow(2).sum().sqrt())

    return T2, results


def analyze_cross_layer_alignment(L1, R1, D1, L2, R2):
    """
    Analyze how layer 2's receptive fields align with layer 1's output fields.
    """
    results = {}

    # Layer 1 output covariance (in hidden space)
    # What subspace does layer 1 output to?
    G_L1_out = L1 @ L1.T + R1 @ R1.T  # (d_h1, d_h1)

    # Layer 2 input covariance
    # What subspace does layer 2 read from?
    G_L2_in = L2.T @ L2 + R2.T @ R2  # (d_h1, d_h1)

    # Normalize
    G_L1_out_norm = G_L1_out / (torch.trace(G_L1_out) + 1e-10)
    G_L2_in_norm = G_L2_in / (torch.trace(G_L2_in) + 1e-10)

    # Alignment: Frobenius inner product of normalized covariances
    alignment = torch.sum(G_L1_out_norm * G_L2_in_norm)
    results['alignment'] = float(alignment)

    # Eigenvalue analysis
    eig_L1 = torch.linalg.eigvalsh(G_L1_out)
    eig_L2 = torch.linalg.eigvalsh(G_L2_in)

    # Effective rank (entropy-based)
    def effective_rank(eigs):
        eigs = eigs[eigs > 1e-10]
        p = eigs / eigs.sum()
        entropy = -(p * torch.log(p + 1e-10)).sum()
        return float(torch.exp(entropy))

    results['L1_out_eff_rank'] = effective_rank(eig_L1)
    results['L2_in_eff_rank'] = effective_rank(eig_L2)

    # Subspace overlap via eigendecomposition
    _, V_L1 = torch.linalg.eigh(G_L1_out)
    _, V_L2 = torch.linalg.eigh(G_L2_in)

    # Take top-k eigenvectors and compute overlap
    k = min(3, V_L1.shape[1])
    V_L1_top = V_L1[:, -k:]  # top k
    V_L2_top = V_L2[:, -k:]

    # Subspace angle (using SVD of V1.T @ V2)
    overlap_matrix = V_L1_top.T @ V_L2_top
    svd_overlap = torch.linalg.svdvals(overlap_matrix)
    results['top_subspace_overlap'] = float(svd_overlap.mean())

    return results


def analyze_checkpoint(state_dict):
    """Analyze a single checkpoint."""
    L1 = state_dict['layers.0.L']
    R1 = state_dict['layers.0.R']
    D1 = state_dict['layers.0.D']
    L2 = state_dict['layers.1.L']
    R2 = state_dict['layers.1.R']
    D2 = state_dict['layers.1.D']

    # T2 analysis
    T2, t2_results = analyze_T2(D2, L2, R2)

    # Cross-layer alignment
    align_results = analyze_cross_layer_alignment(L1, R1, D1, L2, R2)

    return {**t2_results, **align_results}


def run_analysis(seed):
    """Run analysis on checkpoints for given seed."""
    # Load checkpoints
    ckpt_path = Path(f"checkpoints_r4_seeds/seed{seed}_checkpoints.pkl")
    with open(ckpt_path, 'rb') as f:
        data = pickle.load(f)

    checkpoints = data['checkpoints']
    history = data['history']
    steps = sorted(checkpoints.keys())

    print(f"Loaded {len(steps)} checkpoints for seed {seed}")
    print(f"Final accuracy: {history['eval_acc'][-1]:.3f}")

    # Analyze each checkpoint
    all_metrics = {
        'diag_frac_mean': [],
        'symmetry_error_mean': [],
        'T2_rank_mean': [],
        'T2_norm': [],
        'alignment': [],
        'L1_out_eff_rank': [],
        'L2_in_eff_rank': [],
        'top_subspace_overlap': [],
    }

    # Per-class metrics
    n_classes = 4
    for p in range(n_classes):
        all_metrics[f'diag_frac_class{p}'] = []
        all_metrics[f'T2_rank_class{p}'] = []

    for step in steps:
        results = analyze_checkpoint(checkpoints[step])

        all_metrics['diag_frac_mean'].append(results['diag_frac_mean'])
        all_metrics['symmetry_error_mean'].append(results['symmetry_error_mean'])
        all_metrics['T2_rank_mean'].append(results['T2_rank_mean'])
        all_metrics['T2_norm'].append(results['T2_norm'])
        all_metrics['alignment'].append(results['alignment'])
        all_metrics['L1_out_eff_rank'].append(results['L1_out_eff_rank'])
        all_metrics['L2_in_eff_rank'].append(results['L2_in_eff_rank'])
        all_metrics['top_subspace_overlap'].append(results['top_subspace_overlap'])

        for p in range(n_classes):
            all_metrics[f'diag_frac_class{p}'].append(results['diag_frac_per_class'][p])
            all_metrics[f'T2_rank_class{p}'].append(results['T2_rank_per_class'][p])

    # Plot
    fig, axes = plt.subplots(7, 1, figsize=(14, 24))

    # Row 1: Accuracy and Loss
    ax1 = axes[0]
    ax1.plot(history['steps'], history['eval_acc'], 'b-', linewidth=2, label='Accuracy')
    ax1.set_ylabel('Accuracy', color='b')
    ax1.set_xlabel('Step')
    ax1.set_xlim(0, max(steps))
    ax1.tick_params(axis='y', labelcolor='b')
    ax1.grid(True, alpha=0.3)
    ax1.set_title(f'Seed {seed}: Phase Change Analysis (rank=4)', fontsize=12)

    ax1b = ax1.twinx()
    ax1b.plot(history['steps'], history['loss'], 'r--', linewidth=1.5, alpha=0.7, label='Loss')
    ax1b.set_ylabel('Loss', color='r')
    ax1b.tick_params(axis='y', labelcolor='r')

    # Row 2: T2 diagonal dominance (per class)
    ax2 = axes[1]
    colors = ['b', 'r', 'g', 'm']
    for p in range(n_classes):
        ax2.plot(steps, all_metrics[f'diag_frac_class{p}'],
                 color=colors[p], linewidth=1.5, label=f'Class {p}')
    ax2.set_ylabel('Diagonal Fraction')
    ax2.set_xlabel('Step')
    ax2.set_xlim(0, max(steps))
    ax2.set_ylim(0, 1)
    ax2.legend(loc='right', fontsize=9)
    ax2.grid(True, alpha=0.3)
    ax2.set_title('T2 Diagonal Dominance (per class): Higher = features used independently', fontsize=10)

    # Row 3: T2 rank per class
    ax3 = axes[2]
    for p in range(n_classes):
        ax3.plot(steps, all_metrics[f'T2_rank_class{p}'],
                 color=colors[p], linewidth=1.5, label=f'Class {p}')
    ax3.set_ylabel('T2 Rank')
    ax3.set_xlabel('Step')
    ax3.set_xlim(0, max(steps))
    ax3.set_ylim(0, 5)
    ax3.legend(loc='right', fontsize=9)
    ax3.grid(True, alpha=0.3)
    ax3.set_title('T2 Rank per Class: Complexity of class decision boundary', fontsize=10)

    # Row 4: T2 symmetry error
    ax4 = axes[3]
    ax4.plot(steps, all_metrics['symmetry_error_mean'], 'purple', linewidth=2)
    ax4.set_ylabel('Symmetry Error')
    ax4.set_xlabel('Step')
    ax4.set_xlim(0, max(steps))
    ax4.grid(True, alpha=0.3)
    ax4.set_title('T2 Symmetry Error: 0 = T2[p] is symmetric (L2 ≈ R2)', fontsize=10)

    # Row 5: Cross-layer alignment
    ax5 = axes[4]
    ax5.plot(steps, all_metrics['alignment'], 'b-', linewidth=2, label='Alignment')
    ax5.plot(steps, all_metrics['top_subspace_overlap'], 'g--', linewidth=2, label='Top Subspace Overlap')
    ax5.set_ylabel('Alignment')
    ax5.set_xlabel('Step')
    ax5.set_xlim(0, max(steps))
    ax5.legend(loc='right', fontsize=9)
    ax5.grid(True, alpha=0.3)
    ax5.set_title('Cross-Layer Alignment: Does L2 use what L1 computes?', fontsize=10)

    # Row 6: Effective ranks
    ax6 = axes[5]
    ax6.plot(steps, all_metrics['L1_out_eff_rank'], 'b-', linewidth=2, label='L1 Output Eff Rank')
    ax6.plot(steps, all_metrics['L2_in_eff_rank'], 'r--', linewidth=2, label='L2 Input Eff Rank')
    ax6.set_ylabel('Effective Rank')
    ax6.set_xlabel('Step')
    ax6.set_xlim(0, max(steps))
    ax6.legend(loc='right', fontsize=9)
    ax6.grid(True, alpha=0.3)
    ax6.set_title('Effective Ranks of Layer Covariances', fontsize=10)

    # Row 7: T2 norm
    ax7 = axes[6]
    ax7.plot(steps, all_metrics['T2_norm'], 'k-', linewidth=2)
    ax7.set_ylabel('T2 Frobenius Norm')
    ax7.set_xlabel('Step')
    ax7.set_xlim(0, max(steps))
    ax7.grid(True, alpha=0.3)
    ax7.set_title('T2 Tensor Norm: Overall magnitude of layer-2 mixing', fontsize=10)

    plt.tight_layout()
    plt.savefig(f'T2_analysis_seed{seed}.png', dpi=150, bbox_inches='tight')
    plt.close()

    print(f"Saved: T2_analysis_seed{seed}.png")

    return all_metrics, steps


if __name__ == "__main__":
    import sys
    seed = int(sys.argv[1]) if len(sys.argv) > 1 else 1
    run_analysis(seed)
