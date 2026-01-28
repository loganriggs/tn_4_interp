"""
Run analytic TN-Sim on 2-layer bilinear checkpoints.
"""

import pickle
from pathlib import Path
import numpy as np
import torch
import matplotlib.pyplot as plt

from tnsim import tn_sim_2layer, tn_sim_2layer_with_residual, bilinear_core


def compute_effective_rank_from_gram(gram):
    """Compute effective rank from Gram matrix using entropy."""
    s = torch.linalg.svdvals(gram)
    s = s[s > 1e-10]
    if len(s) == 0:
        return 0.0
    p = s / s.sum()
    entropy = -(p * torch.log(p + 1e-10)).sum()
    return torch.exp(entropy).item()


def compute_layer_effective_ranks(L, R, D):
    """Compute effective ranks for a single bilinear layer."""
    LL = L @ L.T
    RR = R @ R.T
    M = LL * RR
    gram1 = D @ M @ D.T  # mode-1 (output)

    DTD = D.T @ D
    gram2 = L.T @ (DTD * RR) @ L  # mode-2 (input-L)
    gram3 = R.T @ (DTD * LL) @ R  # mode-3 (input-R)

    return (
        compute_effective_rank_from_gram(gram1),
        compute_effective_rank_from_gram(gram2),
        compute_effective_rank_from_gram(gram3)
    )


def compute_2layer_effective_ranks(L1, R1, D1, L2, R2, D2):
    """
    Compute effective ranks for 2-layer bilinear.

    Returns dict with:
    - layer1: (mode1, mode2, mode3) effective ranks for layer 1
    - layer2: (mode1, mode2, mode3) effective ranks for layer 2
    - composed: effective ranks for the composed view (L2@D1, R2@D1)
    """
    # Layer 1 standalone
    l1_ranks = compute_layer_effective_ranks(L1, R1, D1)

    # Layer 2 standalone
    l2_ranks = compute_layer_effective_ranks(L2, R2, D2)

    # Composed view: how layer 2 "sees" layer 1's output
    # A = L2 @ D1, B = R2 @ D1
    A = L2 @ D1  # (d_h2, d_h1)
    B = R2 @ D1

    # Effective rank of composition matrices
    # This captures how much layer 2 utilizes layer 1's hidden dimensions
    AA = A @ A.T
    BB = B @ B.T
    comp_rank_A = compute_effective_rank_from_gram(AA)
    comp_rank_B = compute_effective_rank_from_gram(BB)

    return {
        'layer1': l1_ranks,
        'layer2': l2_ranks,
        'composed': (comp_rank_A, comp_rank_B)
    }


def compute_5th_order_tucker_ranks(L1, R1, D1, L2, R2, D2):
    """
    Compute Tucker ranks for the full 5th-order tensor T[n,j,k,p,q].

    T[n,j,k,p,q] = Σ_{m,h,h'} D2[n,m] A[m,h] L1[h,j] R1[h,k] B[m,h'] L1[h',p] R1[h',q]

    where A = L2 @ D1, B = R2 @ D1.

    We compute Gram matrices for each mode unfolding without materializing
    the full tensor.

    Returns dict with effective ranks for all 5 modes:
    - mode1 (n): output
    - mode2 (j): input-L first copy
    - mode3 (k): input-R first copy
    - mode4 (p): input-L second copy
    - mode5 (q): input-R second copy

    Note: Due to symmetry, mode2≈mode4 and mode3≈mode5.
    """
    A = L2 @ D1  # (d_h2, d_h1)
    B = R2 @ D1

    LL = L1 @ L1.T  # (d_h1, d_h1)
    RR = R1 @ R1.T
    C1 = LL * RR    # Layer 1 core (element-wise)

    DD = D2.T @ D2  # (d_h2, d_h2)

    # =========================================================================
    # Mode 1 (output index n):
    # Gram_1 = D2 @ ((A @ C1 @ A.T) * (B @ C1 @ B.T)) @ D2.T
    # =========================================================================
    term_A = A @ C1 @ A.T  # (d_h2, d_h2)
    term_B = B @ C1 @ B.T
    gram1 = D2 @ (term_A * term_B) @ D2.T
    rank1 = compute_effective_rank_from_gram(gram1)

    # =========================================================================
    # Mode 2 (index j) - only L1[h,j] depends on j:
    # Gram_2 = L1.T @ (RR * (A.T @ M @ A)) @ L1
    # where M = DD * (B @ C1 @ B.T)
    # =========================================================================
    M = DD * term_B  # (d_h2, d_h2)
    gram2 = L1.T @ (RR * (A.T @ M @ A)) @ L1
    rank2 = compute_effective_rank_from_gram(gram2)

    # =========================================================================
    # Mode 3 (index k) - only R1[h,k] depends on k:
    # Gram_3 = R1.T @ (LL * (A.T @ M @ A)) @ R1
    # =========================================================================
    gram3 = R1.T @ (LL * (A.T @ M @ A)) @ R1
    rank3 = compute_effective_rank_from_gram(gram3)

    # =========================================================================
    # Mode 4 (index p) - only L1[h',p] depends on p:
    # Similar structure but with B instead of A for the "active" part
    # Gram_4 = L1.T @ (RR * (B.T @ M' @ B)) @ L1
    # where M' = DD * (A @ C1 @ A.T)
    # =========================================================================
    M_prime = DD * term_A
    gram4 = L1.T @ (RR * (B.T @ M_prime @ B)) @ L1
    rank4 = compute_effective_rank_from_gram(gram4)

    # =========================================================================
    # Mode 5 (index q) - only R1[h',q] depends on q:
    # Gram_5 = R1.T @ (LL * (B.T @ M' @ B)) @ R1
    # =========================================================================
    gram5 = R1.T @ (LL * (B.T @ M_prime @ B)) @ R1
    rank5 = compute_effective_rank_from_gram(gram5)

    return {
        'mode1_output': rank1,
        'mode2_inputL1': rank2,
        'mode3_inputR1': rank3,
        'mode4_inputL2': rank4,
        'mode5_inputR2': rank5,
    }

# =============================================================================
# LOAD AND EXTRACT
# =============================================================================

def load_checkpoint(path):
    with open(path, 'rb') as f:
        return pickle.load(f)


def extract_weights(state_dict, d_hidden=256, device='cuda', include_rmsnorm=True):
    """Extract bilinear weights from 2-layer model.

    If include_rmsnorm=True, incorporates RMSNorm weights into L and R:
        L_eff = L @ diag(g), R_eff = R @ diag(g)
    where g is the RMSNorm learnable weight for each layer.
    """
    L1 = state_dict['bilinears.0.LR.weight'][:d_hidden].to(device)
    R1 = state_dict['bilinears.0.LR.weight'][d_hidden:].to(device)
    D1 = state_dict['bilinears.0.D.weight'].to(device)
    L2 = state_dict['bilinears.1.LR.weight'][:d_hidden].to(device)
    R2 = state_dict['bilinears.1.LR.weight'][d_hidden:].to(device)
    D2 = state_dict['bilinears.1.D.weight'].to(device)

    if include_rmsnorm:
        if 'norms.0.weight' in state_dict:
            g1 = state_dict['norms.0.weight'].to(device)
            L1 = L1 * g1.unsqueeze(0)
            R1 = R1 * g1.unsqueeze(0)
        if 'norms.1.weight' in state_dict:
            g2 = state_dict['norms.1.weight'].to(device)
            L2 = L2 * g2.unsqueeze(0)
            R2 = R2 * g2.unsqueeze(0)

    return {
        'L1': L1, 'R1': R1, 'D1': D1,
        'L2': L2, 'R2': R2, 'D2': D2,
    }


# =============================================================================
# MAIN
# =============================================================================

if __name__ == "__main__":
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")

    output_dir = Path("tn_analysis_checkpoints")

    # Load 2-layer model
    model_path = output_dir / "residual_2layer_downproj_aug.pkl"
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

    # Track effective ranks - both per-layer AND proper 5th-order Tucker ranks
    eff_ranks = {
        'layer1_mode1': [], 'layer1_mode2': [], 'layer1_mode3': [],
        'layer2_mode1': [], 'layer2_mode2': [], 'layer2_mode3': [],
        'composed_A': [], 'composed_B': []
    }
    tucker_5th = {
        'mode1': [], 'mode2': [], 'mode3': [], 'mode4': [], 'mode5': []
    }

    for i, s1 in enumerate(sample_steps):
        w1 = extract_weights(checkpoints[s1], d_hidden=config['d_hidden'], device=device)

        # Compute per-layer effective ranks
        ranks = compute_2layer_effective_ranks(
            w1['L1'], w1['R1'], w1['D1'],
            w1['L2'], w1['R2'], w1['D2']
        )
        eff_ranks['layer1_mode1'].append(ranks['layer1'][0])
        eff_ranks['layer1_mode2'].append(ranks['layer1'][1])
        eff_ranks['layer1_mode3'].append(ranks['layer1'][2])
        eff_ranks['layer2_mode1'].append(ranks['layer2'][0])
        eff_ranks['layer2_mode2'].append(ranks['layer2'][1])
        eff_ranks['layer2_mode3'].append(ranks['layer2'][2])
        eff_ranks['composed_A'].append(ranks['composed'][0])
        eff_ranks['composed_B'].append(ranks['composed'][1])

        # Compute proper 5th-order Tucker ranks
        tucker = compute_5th_order_tucker_ranks(
            w1['L1'], w1['R1'], w1['D1'],
            w1['L2'], w1['R2'], w1['D2']
        )
        tucker_5th['mode1'].append(tucker['mode1_output'])
        tucker_5th['mode2'].append(tucker['mode2_inputL1'])
        tucker_5th['mode3'].append(tucker['mode3_inputR1'])
        tucker_5th['mode4'].append(tucker['mode4_inputL2'])
        tucker_5th['mode5'].append(tucker['mode5_inputR2'])

        for j, s2 in enumerate(sample_steps):
            if j < i:
                sim_matrix[i, j] = sim_matrix[j, i]
                sim_matrix_res[i, j] = sim_matrix_res[j, i]
            else:
                w2 = extract_weights(checkpoints[s2], d_hidden=config['d_hidden'], device=device)
                # Without residual (pure bilinear composition)
                sim_matrix[i, j] = tn_sim_2layer(
                    w1['L1'], w1['R1'], w1['D1'], w1['L2'], w1['R2'], w1['D2'],
                    w2['L1'], w2['R1'], w2['D1'], w2['L2'], w2['R2'], w2['D2'],
                    symmetrize=True
                )
                # With residual (full polynomial)
                sim_matrix_res[i, j] = tn_sim_2layer_with_residual(
                    w1['L1'], w1['R1'], w1['D1'], w1['L2'], w1['R2'], w1['D2'],
                    w2['L1'], w2['R1'], w2['D1'], w2['L2'], w2['R2'], w2['D2'],
                    symmetrize=True
                )

        t_ranks = [tucker['mode1_output'], tucker['mode2_inputL1'], tucker['mode3_inputR1'],
                   tucker['mode4_inputL2'], tucker['mode5_inputR2']]
        print(f"  Row {i+1}/{n}: diag={sim_matrix[i,i]:.4f}, Tucker5=({t_ranks[0]:.1f},{t_ranks[1]:.1f},{t_ranks[2]:.1f},{t_ranks[3]:.1f},{t_ranks[4]:.1f})")

    # =========================================================================
    # PLOTTING
    # =========================================================================

    x_indices = np.arange(n)

    def set_aligned_xticks(ax, steps=sample_steps):
        ax.set_xticks(range(len(steps)))
        ax.set_xticklabels([str(s) for s in steps], rotation=45, ha='right', fontsize=7)
        ax.set_xlim([-0.5, len(steps) - 0.5])

    fig, axes = plt.subplots(4, 3, figsize=(16, 18))

    # Row 1: TN-Sim matrices
    ax = axes[0, 0]
    im = ax.imshow(sim_matrix, cmap='viridis', vmin=0, vmax=1)
    ax.set_xticks(range(n))
    ax.set_xticklabels([str(s) for s in sample_steps], rotation=45, ha='right', fontsize=7)
    ax.set_yticks(range(n))
    ax.set_yticklabels([str(s) for s in sample_steps], fontsize=7)
    ax.set_title('TN-Sim (Bilinear Only)')
    plt.colorbar(im, ax=ax, orientation='horizontal', shrink=0.8, pad=0.15)

    ax = axes[0, 1]
    im = ax.imshow(sim_matrix_res, cmap='viridis', vmin=0, vmax=1)
    ax.set_xticks(range(n))
    ax.set_xticklabels([str(s) for s in sample_steps], rotation=45, ha='right', fontsize=7)
    ax.set_yticks(range(n))
    ax.set_yticklabels([str(s) for s in sample_steps], fontsize=7)
    ax.set_title('TN-Sim (With Residual)')
    plt.colorbar(im, ax=ax, orientation='horizontal', shrink=0.8, pad=0.15)

    ax = axes[0, 2]
    val_acc_sampled = np.interp(sample_steps, history['steps'], history['val_acc'])
    ax.plot(x_indices, val_acc_sampled, 'o-', linewidth=2, markersize=6, color='green')
    set_aligned_xticks(ax)
    ax.set_xlabel('Step')
    ax.set_ylabel('Val Accuracy')
    ax.set_title('Validation Accuracy')
    ax.grid(True, alpha=0.3)

    # Row 2: TN-Sim derived metrics
    ax = axes[1, 0]
    final_sims = sim_matrix[:, -1]
    final_sims_res = sim_matrix_res[:, -1]
    ax.plot(x_indices, final_sims, 'o-', linewidth=2, markersize=6, label='Bilinear Only')
    ax.plot(x_indices, final_sims_res, 's--', linewidth=2, markersize=6, label='With Residual')
    set_aligned_xticks(ax)
    ax.set_xlabel('Step')
    ax.set_ylabel('TN-Sim to Final')
    ax.set_title('Similarity to Final Checkpoint')
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.set_ylim([0, 1.1])

    ax = axes[1, 1]
    adjacent_sims = np.diag(sim_matrix, k=1)
    adjacent_sims_res = np.diag(sim_matrix_res, k=1)
    x_mid = x_indices[:-1] + 0.5
    ax.plot(x_mid, adjacent_sims, 'o-', linewidth=2, markersize=6, label='Bilinear Only')
    ax.plot(x_mid, adjacent_sims_res, 's--', linewidth=2, markersize=6, label='With Residual')
    set_aligned_xticks(ax)
    ax.set_xlabel('Step')
    ax.set_ylabel('TN-Sim')
    ax.set_title('Adjacent Checkpoint Similarity')
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.set_ylim([0, 1.1])

    ax = axes[1, 2]
    diff = sim_matrix_res - sim_matrix
    im = ax.imshow(diff, cmap='RdBu', vmin=-0.5, vmax=0.5)
    ax.set_xticks(range(n))
    ax.set_xticklabels([str(s) for s in sample_steps], rotation=45, ha='right', fontsize=7)
    ax.set_yticks(range(n))
    ax.set_yticklabels([str(s) for s in sample_steps], fontsize=7)
    ax.set_title('Difference (Residual - Bilinear)')
    plt.colorbar(im, ax=ax, orientation='horizontal', shrink=0.8, pad=0.15)

    # Row 3: Effective ranks
    ax = axes[2, 0]
    ax.plot(x_indices, eff_ranks['layer1_mode1'], 'o-', linewidth=2, markersize=5, label='Layer 1')
    ax.plot(x_indices, eff_ranks['layer2_mode1'], 's-', linewidth=2, markersize=5, label='Layer 2')
    set_aligned_xticks(ax)
    ax.set_xlabel('Step')
    ax.set_ylabel('Effective Rank (Mode-1)')
    ax.set_title('Output Mode Effective Rank')
    ax.legend()
    ax.grid(True, alpha=0.3)

    ax = axes[2, 1]
    ax.plot(x_indices, eff_ranks['composed_A'], 'o-', linewidth=2, markersize=5, label='L2@D1')
    ax.plot(x_indices, eff_ranks['composed_B'], 's-', linewidth=2, markersize=5, label='R2@D1')
    set_aligned_xticks(ax)
    ax.set_xlabel('Step')
    ax.set_ylabel('Effective Rank')
    ax.set_title('Composition Effective Rank (Layer 2 → Layer 1)')
    ax.legend()
    ax.grid(True, alpha=0.3)

    ax = axes[2, 2]
    total_eff = np.array(eff_ranks['layer1_mode1']) + np.array(eff_ranks['layer2_mode1'])
    ax.plot(x_indices, total_eff, 'o-', linewidth=2, markersize=5, color='purple', label='L1+L2 Eff Rank')
    ax2 = ax.twinx()
    ax2.plot(x_indices, val_acc_sampled, 's--', linewidth=2, markersize=4, color='green', label='Val Acc')
    ax2.set_ylabel('Val Accuracy', color='green')
    set_aligned_xticks(ax)
    ax.set_xlabel('Step')
    ax.set_ylabel('Sum of Effective Ranks', color='purple')
    ax.set_title('Per-Layer Effective Rank & Accuracy')
    ax.grid(True, alpha=0.3)

    # Row 4: Proper 5th-order Tucker ranks
    ax = axes[3, 0]
    ax.plot(x_indices, tucker_5th['mode1'], 'o-', linewidth=2, markersize=5, label='Mode-1 (output n)')
    ax.plot(x_indices, tucker_5th['mode2'], 's-', linewidth=2, markersize=5, label='Mode-2 (input j)')
    ax.plot(x_indices, tucker_5th['mode3'], '^-', linewidth=2, markersize=5, label='Mode-3 (input k)')
    set_aligned_xticks(ax)
    ax.set_xlabel('Step')
    ax.set_ylabel('Effective Tucker Rank')
    ax.set_title('5th-Order Tucker Ranks (Modes 1-3)')
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    ax = axes[3, 1]
    ax.plot(x_indices, tucker_5th['mode4'], 'd-', linewidth=2, markersize=5, label='Mode-4 (input p)')
    ax.plot(x_indices, tucker_5th['mode5'], 'v-', linewidth=2, markersize=5, label='Mode-5 (input q)')
    # Also show mode2,3 for comparison (should be similar to mode4,5)
    ax.plot(x_indices, tucker_5th['mode2'], 's--', linewidth=1, markersize=3, alpha=0.5, label='Mode-2 (ref)')
    ax.plot(x_indices, tucker_5th['mode3'], '^--', linewidth=1, markersize=3, alpha=0.5, label='Mode-3 (ref)')
    set_aligned_xticks(ax)
    ax.set_xlabel('Step')
    ax.set_ylabel('Effective Tucker Rank')
    ax.set_title('5th-Order Tucker Ranks (Modes 4-5 vs 2-3)')
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    ax = axes[3, 2]
    total_tucker = (np.array(tucker_5th['mode1']) + np.array(tucker_5th['mode2']) +
                    np.array(tucker_5th['mode3']) + np.array(tucker_5th['mode4']) +
                    np.array(tucker_5th['mode5']))
    ax.plot(x_indices, total_tucker, 'o-', linewidth=2, markersize=5, color='darkblue', label='Sum of 5 Tucker Ranks')
    ax2 = ax.twinx()
    ax2.plot(x_indices, val_acc_sampled, 's--', linewidth=2, markersize=4, color='green', label='Val Acc')
    ax2.set_ylabel('Val Accuracy', color='green')
    set_aligned_xticks(ax)
    ax.set_xlabel('Step')
    ax.set_ylabel('Sum of 5th-Order Tucker Ranks', color='darkblue')
    ax.set_title('Total 5th-Order Tucker Rank & Accuracy')
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    save_path = output_dir / "tnsim_2layer_analytic.png"
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.show()

    print(f"\nPlot saved to: {save_path}")

    # =========================================================================
    # SUMMARY
    # =========================================================================

    print("\n" + "="*70)
    print("SUMMARY (Analytic TN-Sim, Symmetrized)")
    print("="*70)
    print(f"\n{'Metric':<30} {'Bilinear Only':>15} {'With Residual':>15}")
    print("-"*70)
    print(f"{'Init-Final TN-Sim':<30} {sim_matrix[0, -1]:>15.6f} {sim_matrix_res[0, -1]:>15.6f}")
    print(f"{'Mean adjacent TN-Sim':<30} {adjacent_sims.mean():>15.4f} {adjacent_sims_res.mean():>15.4f}")
    print(f"{'Min adjacent TN-Sim':<30} {adjacent_sims.min():>15.4f} {adjacent_sims_res.min():>15.4f}")
    print(f"{'Final val_acc':<30} {history['val_acc'][-1]:>15.4f}")
