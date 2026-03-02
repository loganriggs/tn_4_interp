"""
Run TN-sim and Tucker ranks across checkpoints for ranks 2-8.
Two approaches:
1. Per-layer: 3rd-order tensor for each layer separately
2. Combined: 5th-order tensor for full 2-layer composition
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

    Returns (mode1_output, mode2_inputL, mode3_inputR) effective ranks.
    """
    LL = L @ L.T  # (d_hidden, d_hidden)
    RR = R @ R.T
    M = LL * RR  # element-wise (core)
    gram1 = D @ M @ D.T  # mode-1 (output)

    DTD = D.T @ D
    gram2 = L.T @ (DTD * RR) @ L  # mode-2 (input-L)
    gram3 = R.T @ (DTD * LL) @ R  # mode-3 (input-R)

    return (
        compute_effective_rank_from_gram(gram1),
        compute_effective_rank_from_gram(gram2),
        compute_effective_rank_from_gram(gram3)
    )


def compute_5th_order_tucker_ranks(L1, R1, D1, L2, R2, D2):
    """
    Compute Tucker ranks for full 5th-order tensor T[n,j,k,p,q].

    T[n,j,k,p,q] = Σ_{m,h,h'} D2[n,m] A[m,h] L1[h,j] R1[h,k] B[m,h'] L1[h',p] R1[h',q]

    where A = L2 @ D1, B = R2 @ D1.

    Returns dict with effective ranks for all 5 modes.
    """
    A = L2 @ D1  # (d_h2, d_h1)
    B = R2 @ D1

    LL = L1 @ L1.T  # (d_h1, d_h1)
    RR = R1 @ R1.T
    C1 = LL * RR    # Layer 1 core

    DD = D2.T @ D2  # (d_h2, d_h2)

    # Mode 1 (output n)
    term_A = A @ C1 @ A.T
    term_B = B @ C1 @ B.T
    gram1 = D2 @ (term_A * term_B) @ D2.T
    rank1 = compute_effective_rank_from_gram(gram1)

    # Mode 2 (input j - L1 first copy)
    M = DD * term_B
    gram2 = L1.T @ (RR * (A.T @ M @ A)) @ L1
    rank2 = compute_effective_rank_from_gram(gram2)

    # Mode 3 (input k - R1 first copy)
    gram3 = R1.T @ (LL * (A.T @ M @ A)) @ R1
    rank3 = compute_effective_rank_from_gram(gram3)

    # Mode 4 (input p - L1 second copy)
    M_prime = DD * term_A
    gram4 = L1.T @ (RR * (B.T @ M_prime @ B)) @ L1
    rank4 = compute_effective_rank_from_gram(gram4)

    # Mode 5 (input q - R1 second copy)
    gram5 = R1.T @ (LL * (B.T @ M_prime @ B)) @ R1
    rank5 = compute_effective_rank_from_gram(gram5)

    return {
        'mode1_output': rank1,
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


def compute_all_metrics(checkpoints, n, num_layers, rank):
    """Compute TN-sim matrix and Tucker ranks for all checkpoints."""
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

    # Tucker ranks per checkpoint
    tucker_per_layer = {'L1': [], 'L2': []}
    tucker_5th_order = {'mode1': [], 'mode2': [], 'mode3': [], 'mode4': [], 'mode5': []}

    for L1, R1, D1, L2, R2, D2 in all_weights:
        # Per-layer (3rd order)
        r1 = compute_layer_tucker_ranks(L1, R1, D1)
        r2 = compute_layer_tucker_ranks(L2, R2, D2)
        tucker_per_layer['L1'].append(r1)
        tucker_per_layer['L2'].append(r2)

        # Combined (5th order)
        r5 = compute_5th_order_tucker_ranks(L1, R1, D1, L2, R2, D2)
        tucker_5th_order['mode1'].append(r5['mode1_output'])
        tucker_5th_order['mode2'].append(r5['mode2_j'])
        tucker_5th_order['mode3'].append(r5['mode3_k'])
        tucker_5th_order['mode4'].append(r5['mode4_p'])
        tucker_5th_order['mode5'].append(r5['mode5_q'])

    return {
        'tnsim_matrix': tnsim_matrix,
        'steps': steps,
        'tucker_per_layer': tucker_per_layer,
        'tucker_5th_order': tucker_5th_order,
    }


def run_sweep():
    """Run sweep for ranks 2-8 with Tucker ranks."""
    ranks = [2, 3, 4, 5, 6, 7, 8]
    n = 4
    num_layers = 2
    steps = 10000

    all_results = {}

    for rank in ranks:
        print(f"\n{'='*50}")
        print(f"Training rank={rank}")
        print(f"{'='*50}")

        checkpoints, history = train_with_checkpoints(n, num_layers, rank, steps)
        metrics = compute_all_metrics(checkpoints, n, num_layers, rank)
        metrics['history'] = history

        all_results[rank] = metrics
        print(f"  Final acc: {history['eval_acc'][-1]:.3f}")

    # Plot: 7 rows (ranks), 4 columns (acc, loss, TN-sim, Tucker ranks)
    fig, axes = plt.subplots(len(ranks), 4, figsize=(20, 4 * len(ranks)),
                              gridspec_kw={'width_ratios': [1, 1, 2, 1.5]})

    for idx, rank in enumerate(ranks):
        result = all_results[rank]
        history = result['history']
        tnsim_matrix = result['tnsim_matrix']
        ckpt_steps = result['steps']
        tucker_5th = result['tucker_5th_order']

        # Accuracy
        ax1 = axes[idx, 0]
        ax1.plot(history['steps'], history['eval_acc'], 'b-', linewidth=1.5)
        ax1.set_ylabel(f'r={rank}\nAccuracy')
        ax1.set_xlim(0, max(history['steps']))
        ax1.set_ylim(0, 1)
        ax1.grid(True, alpha=0.3)
        if idx == len(ranks) - 1:
            ax1.set_xlabel('Step')

        # Loss
        ax2 = axes[idx, 1]
        ax2.plot(history['steps'], history['loss'], 'r-', linewidth=1.5)
        ax2.set_ylabel('Loss')
        ax2.set_xlim(0, max(history['steps']))
        ax2.grid(True, alpha=0.3)
        if idx == len(ranks) - 1:
            ax2.set_xlabel('Step')

        # TN-sim matrix
        ax3 = axes[idx, 2]
        im = ax3.imshow(tnsim_matrix, aspect='auto', cmap='RdBu_r', vmin=-1, vmax=1)
        ax3.set_ylabel('Checkpoint')
        n_ckpts = len(ckpt_steps)
        n_labels = 6
        label_indices = np.linspace(0, n_ckpts - 1, n_labels).astype(int)
        ax3.set_xticks(label_indices)
        ax3.set_yticks(label_indices)
        ax3.set_yticklabels([str(ckpt_steps[i]) for i in label_indices])
        if idx == len(ranks) - 1:
            ax3.set_xticklabels([str(ckpt_steps[i]) for i in label_indices], rotation=45, ha='right')
            ax3.set_xlabel('Step')
        else:
            ax3.set_xticklabels([])

        # Tucker ranks (5th order)
        ax4 = axes[idx, 3]
        ax4.plot(ckpt_steps, tucker_5th['mode1'], 'b-', linewidth=1.5, label='out')
        ax4.plot(ckpt_steps, tucker_5th['mode2'], 'r--', linewidth=1.5, label='j')
        ax4.plot(ckpt_steps, tucker_5th['mode3'], 'g:', linewidth=1.5, label='k')
        ax4.set_ylabel('Tucker Rank')
        ax4.set_xlim(0, max(ckpt_steps))
        ax4.set_ylim(0, rank + 1)
        ax4.grid(True, alpha=0.3)
        if idx == 0:
            ax4.legend(loc='upper right', fontsize=8)
        if idx == len(ranks) - 1:
            ax4.set_xlabel('Step')

    # Colorbar
    cbar_ax = fig.add_axes([0.45, 0.02, 0.2, 0.012])
    cbar = fig.colorbar(im, cax=cbar_ax, orientation='horizontal')
    cbar.set_label('TN Similarity')

    fig.suptitle('TN-Sim & Tucker Ranks: n=4, 2-layer, ranks 2-8', fontsize=14, y=0.995)
    plt.tight_layout(rect=[0, 0.04, 1, 0.99])
    plt.savefig('tnsim_tucker_sweep_2to8.png', dpi=150, bbox_inches='tight')
    plt.close()

    print(f"\nSaved: tnsim_tucker_sweep_2to8.png")
    return all_results


if __name__ == "__main__":
    run_sweep()
