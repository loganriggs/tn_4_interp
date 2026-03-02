"""
Detailed Tucker rank analysis for rank=5.
- 5th-order: all 5 modes
- Layer 1 (3rd-order): 3 modes
- Layer 2 (3rd-order): 3 modes
"""

import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).parent.parent / "dev_interp" / "notebooks"))
from tnsim import tn_sim_2layer_with_residual

from bilinear_residual_rmsnorm import (
    BilinearResidualRMSNorm,
    task_2nd_argmax,
)


# =============================================================================
# TUCKER RANK COMPUTATION
# =============================================================================

def compute_effective_rank_from_gram(gram):
    """Compute effective rank from Gram matrix using entropy."""
    s = torch.linalg.svdvals(gram)
    s = s[s > 1e-10]
    if len(s) == 0:
        return 0.0
    p = s / s.sum()
    entropy = -(p * torch.log(p + 1e-10)).sum()
    return torch.exp(entropy).item()


def compute_layer_tucker_ranks(L, R, D):
    """
    Compute Tucker ranks for a single bilinear layer (3rd-order tensor).
    T[i,j,k] = Σ_h D[i,h] L[h,j] R[h,k]

    Returns dict with mode names.
    """
    LL = L @ L.T
    RR = R @ R.T
    M = LL * RR
    gram1 = D @ M @ D.T  # mode-1 (output)

    DTD = D.T @ D
    gram2 = L.T @ (DTD * RR) @ L  # mode-2 (input-L)
    gram3 = R.T @ (DTD * LL) @ R  # mode-3 (input-R)

    return {
        'output': compute_effective_rank_from_gram(gram1),
        'input_L': compute_effective_rank_from_gram(gram2),
        'input_R': compute_effective_rank_from_gram(gram3),
    }


def compute_5th_order_tucker_ranks(L1, R1, D1, L2, R2, D2):
    """
    Compute Tucker ranks for full 5th-order tensor T[n,j,k,p,q].
    """
    A = L2 @ D1
    B = R2 @ D1

    LL = L1 @ L1.T
    RR = R1 @ R1.T
    C1 = LL * RR

    DD = D2.T @ D2

    # Mode 1 (output n)
    term_A = A @ C1 @ A.T
    term_B = B @ C1 @ B.T
    gram1 = D2 @ (term_A * term_B) @ D2.T
    rank1 = compute_effective_rank_from_gram(gram1)

    # Mode 2 (input j)
    M = DD * term_B
    gram2 = L1.T @ (RR * (A.T @ M @ A)) @ L1
    rank2 = compute_effective_rank_from_gram(gram2)

    # Mode 3 (input k)
    gram3 = R1.T @ (LL * (A.T @ M @ A)) @ R1
    rank3 = compute_effective_rank_from_gram(gram3)

    # Mode 4 (input p)
    M_prime = DD * term_A
    gram4 = L1.T @ (RR * (B.T @ M_prime @ B)) @ L1
    rank4 = compute_effective_rank_from_gram(gram4)

    # Mode 5 (input q)
    gram5 = R1.T @ (LL * (B.T @ M_prime @ B)) @ R1
    rank5 = compute_effective_rank_from_gram(gram5)

    return {
        'mode1_n': rank1,
        'mode2_j': rank2,
        'mode3_k': rank3,
        'mode4_p': rank4,
        'mode5_q': rank5,
    }


# =============================================================================
# TRAINING
# =============================================================================

def train_with_checkpoints(n, num_layers, rank, steps=10000, checkpoint_every=200):
    """Train model and save checkpoints."""
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = BilinearResidualRMSNorm(n, num_layers, rank, use_rmsnorm=True).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.01, weight_decay=0.001)

    checkpoints = {}
    history = {'steps': [], 'loss': [], 'eval_acc': []}

    for step in range(steps + 1):
        model.train()
        x = torch.randn(128, n, device=device)
        targets = task_2nd_argmax(x)
        logits = model(x)
        loss = nn.functional.cross_entropy(logits, targets)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if step % checkpoint_every == 0:
            checkpoints[step] = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            model.eval()
            with torch.no_grad():
                x_eval = torch.randn(10000, n, device=device)
                targets_eval = task_2nd_argmax(x_eval)
                logits_eval = model(x_eval)
                eval_acc = (logits_eval.argmax(-1) == targets_eval).float().mean().item()
            history['steps'].append(step)
            history['loss'].append(loss.item())
            history['eval_acc'].append(eval_acc)

    return checkpoints, history


def compute_all_metrics(checkpoints, num_layers):
    """Compute TN-sim matrix and all Tucker ranks."""
    steps = sorted(checkpoints.keys())
    n_ckpts = len(steps)

    def get_weights(state_dict):
        L1 = state_dict['layers.0.L']
        R1 = state_dict['layers.0.R']
        D1 = state_dict['layers.0.D']
        L2 = state_dict['layers.1.L']
        R2 = state_dict['layers.1.R']
        D2 = state_dict['layers.1.D']
        return L1, R1, D1, L2, R2, D2

    all_weights = [get_weights(checkpoints[s]) for s in steps]

    # TN-sim matrix
    tnsim_matrix = np.zeros((n_ckpts, n_ckpts))
    for i in range(n_ckpts):
        for j in range(i, n_ckpts):
            L1_a, R1_a, D1_a, L2_a, R2_a, D2_a = all_weights[i]
            L1_b, R1_b, D1_b, L2_b, R2_b, D2_b = all_weights[j]
            sim = tn_sim_2layer_with_residual(
                L1_a, R1_a, D1_a, L2_a, R2_a, D2_a,
                L1_b, R1_b, D1_b, L2_b, R2_b, D2_b
            )
            tnsim_matrix[i, j] = sim
            tnsim_matrix[j, i] = sim

    # Tucker ranks
    tucker_L1 = {'output': [], 'input_L': [], 'input_R': []}
    tucker_L2 = {'output': [], 'input_L': [], 'input_R': []}
    tucker_5th = {'mode1_n': [], 'mode2_j': [], 'mode3_k': [], 'mode4_p': [], 'mode5_q': []}

    for L1, R1, D1, L2, R2, D2 in all_weights:
        # Layer 1 (3rd order)
        r1 = compute_layer_tucker_ranks(L1, R1, D1)
        for k in tucker_L1:
            tucker_L1[k].append(r1[k])

        # Layer 2 (3rd order)
        r2 = compute_layer_tucker_ranks(L2, R2, D2)
        for k in tucker_L2:
            tucker_L2[k].append(r2[k])

        # 5th order
        r5 = compute_5th_order_tucker_ranks(L1, R1, D1, L2, R2, D2)
        for k in tucker_5th:
            tucker_5th[k].append(r5[k])

    return {
        'tnsim_matrix': tnsim_matrix,
        'steps': steps,
        'tucker_L1': tucker_L1,
        'tucker_L2': tucker_L2,
        'tucker_5th': tucker_5th,
    }


def run_r5_analysis():
    """Run detailed analysis for rank=5."""
    n = 4
    num_layers = 2
    rank = 5
    steps = 10000

    print(f"Training n={n}, layers={num_layers}, rank={rank}")
    checkpoints, history = train_with_checkpoints(n, num_layers, rank, steps)
    print(f"Final acc: {history['eval_acc'][-1]:.3f}")

    print("Computing metrics...")
    metrics = compute_all_metrics(checkpoints, num_layers)
    ckpt_steps = metrics['steps']

    # Create figure: 6 rows
    # Row 1: Accuracy
    # Row 2: Loss
    # Row 3: TN-sim matrix
    # Row 4: 5th-order Tucker ranks (5 lines)
    # Row 5: Layer 1 Tucker ranks (3 lines)
    # Row 6: Layer 2 Tucker ranks (3 lines)

    fig, axes = plt.subplots(6, 1, figsize=(14, 18),
                              gridspec_kw={'height_ratios': [1, 1, 2.5, 1.2, 1.2, 1.2]})

    # Row 1: Accuracy
    ax1 = axes[0]
    ax1.plot(history['steps'], history['eval_acc'], 'b-', linewidth=1.5)
    ax1.set_ylabel('Accuracy')
    ax1.set_xlim(0, max(history['steps']))
    ax1.set_ylim(0, 1)
    ax1.grid(True, alpha=0.3)
    ax1.set_xticklabels([])
    ax1.set_title(f'Detailed Tucker Rank Analysis: n={n}, 2-layer, rank={rank}', fontsize=12)

    # Row 2: Loss
    ax2 = axes[1]
    ax2.plot(history['steps'], history['loss'], 'r-', linewidth=1.5)
    ax2.set_ylabel('Loss')
    ax2.set_xlim(0, max(history['steps']))
    ax2.grid(True, alpha=0.3)
    ax2.set_xticklabels([])

    # Row 3: TN-sim matrix
    ax3 = axes[2]
    tnsim_matrix = metrics['tnsim_matrix']
    im = ax3.imshow(tnsim_matrix, aspect='auto', cmap='RdBu_r', vmin=-1, vmax=1)
    ax3.set_ylabel('Checkpoint')
    n_ckpts = len(ckpt_steps)
    n_labels = 10
    label_indices = np.linspace(0, n_ckpts - 1, n_labels).astype(int)
    ax3.set_xticks(label_indices)
    ax3.set_xticklabels([])
    ax3.set_yticks(label_indices)
    ax3.set_yticklabels([str(ckpt_steps[i]) for i in label_indices])
    cbar = fig.colorbar(im, ax=ax3, orientation='vertical', shrink=0.8)
    cbar.set_label('TN Similarity')

    # Row 4: 5th-order Tucker ranks (5 modes)
    ax4 = axes[3]
    tucker_5th = metrics['tucker_5th']
    ax4.plot(ckpt_steps, tucker_5th['mode1_n'], 'b-', linewidth=2, label='mode1 (output n)')
    ax4.plot(ckpt_steps, tucker_5th['mode2_j'], 'r--', linewidth=2, label='mode2 (input j)')
    ax4.plot(ckpt_steps, tucker_5th['mode3_k'], 'g:', linewidth=2, label='mode3 (input k)')
    ax4.plot(ckpt_steps, tucker_5th['mode4_p'], 'm-.', linewidth=2, label="mode4 (input p)")
    ax4.plot(ckpt_steps, tucker_5th['mode5_q'], 'c-', linewidth=1.5, alpha=0.7, label="mode5 (input q)")
    ax4.set_ylabel('Tucker Rank')
    ax4.set_xlim(0, max(ckpt_steps))
    ax4.set_ylim(0, rank + 1)
    ax4.grid(True, alpha=0.3)
    ax4.legend(loc='right', fontsize=9)
    ax4.set_xticklabels([])
    ax4.set_title('5th-Order Tensor Tucker Ranks', fontsize=10)

    # Row 5: Layer 1 Tucker ranks (3 modes)
    ax5 = axes[4]
    tucker_L1 = metrics['tucker_L1']
    ax5.plot(ckpt_steps, tucker_L1['output'], 'b-', linewidth=2, label='output')
    ax5.plot(ckpt_steps, tucker_L1['input_L'], 'r--', linewidth=2, label='input_L')
    ax5.plot(ckpt_steps, tucker_L1['input_R'], 'g:', linewidth=2, label='input_R')
    ax5.set_ylabel('Tucker Rank')
    ax5.set_xlim(0, max(ckpt_steps))
    ax5.set_ylim(0, rank + 1)
    ax5.grid(True, alpha=0.3)
    ax5.legend(loc='right', fontsize=9)
    ax5.set_xticklabels([])
    ax5.set_title('Layer 1 (3rd-Order) Tucker Ranks', fontsize=10)

    # Row 6: Layer 2 Tucker ranks (3 modes)
    ax6 = axes[5]
    tucker_L2 = metrics['tucker_L2']
    ax6.plot(ckpt_steps, tucker_L2['output'], 'b-', linewidth=2, label='output')
    ax6.plot(ckpt_steps, tucker_L2['input_L'], 'r--', linewidth=2, label='input_L')
    ax6.plot(ckpt_steps, tucker_L2['input_R'], 'g:', linewidth=2, label='input_R')
    ax6.set_ylabel('Tucker Rank')
    ax6.set_xlabel('Step')
    ax6.set_xlim(0, max(ckpt_steps))
    ax6.set_ylim(0, rank + 1)
    ax6.grid(True, alpha=0.3)
    ax6.legend(loc='right', fontsize=9)
    ax6.set_title('Layer 2 (3rd-Order) Tucker Ranks', fontsize=10)

    plt.tight_layout()
    plt.savefig('tucker_detailed_r5.png', dpi=150, bbox_inches='tight')
    plt.close()

    print("Saved: tucker_detailed_r5.png")
    return metrics, history


if __name__ == "__main__":
    run_r5_analysis()
