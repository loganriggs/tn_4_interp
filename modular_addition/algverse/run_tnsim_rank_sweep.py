"""
Run TN-sim across checkpoints for ranks 2-8, plot all on same figure.
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


def compute_tnsim_matrix(checkpoints, n, num_layers, rank):
    """Compute pairwise TN-sim matrix."""
    steps = sorted(checkpoints.keys())
    n_ckpts = len(steps)

    def get_weights(state_dict):
        weights = []
        for i in range(num_layers):
            L = state_dict[f'layers.{i}.L']
            R = state_dict[f'layers.{i}.R']
            D = state_dict[f'layers.{i}.D']
            weights.append((L, R, D))
        return weights

    all_weights = [get_weights(checkpoints[s]) for s in steps]
    tnsim_matrix = np.zeros((n_ckpts, n_ckpts))

    for i in range(n_ckpts):
        for j in range(i, n_ckpts):
            (L1_a, R1_a, D1_a), (L2_a, R2_a, D2_a) = all_weights[i]
            (L1_b, R1_b, D1_b), (L2_b, R2_b, D2_b) = all_weights[j]
            sim = tn_sim_2layer_with_residual(
                L1_a, R1_a, D1_a, L2_a, R2_a, D2_a,
                L1_b, R1_b, D1_b, L2_b, R2_b, D2_b
            )
            tnsim_matrix[i, j] = sim
            tnsim_matrix[j, i] = sim

    return tnsim_matrix, steps


def run_rank_sweep():
    """Run sweep for ranks 2-8 and plot all together."""
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
        tnsim_matrix, ckpt_steps = compute_tnsim_matrix(checkpoints, n, num_layers, rank)

        all_results[rank] = {
            'history': history,
            'tnsim_matrix': tnsim_matrix,
            'steps': ckpt_steps,
        }

        print(f"  Final acc: {history['eval_acc'][-1]:.3f}")

    # Plot all on same figure
    fig, axes = plt.subplots(len(ranks), 3, figsize=(16, 4 * len(ranks)),
                              gridspec_kw={'width_ratios': [1, 1, 2.5]})

    for idx, rank in enumerate(ranks):
        result = all_results[rank]
        history = result['history']
        tnsim_matrix = result['tnsim_matrix']
        ckpt_steps = result['steps']

        # Accuracy
        ax1 = axes[idx, 0]
        ax1.plot(history['steps'], history['eval_acc'], 'b-', linewidth=1.5)
        ax1.set_ylabel(f'rank={rank}\nAccuracy')
        ax1.set_xlim(0, max(history['steps']))
        ax1.set_ylim(0, 1)
        ax1.grid(True, alpha=0.3)
        if idx < len(ranks) - 1:
            ax1.set_xticklabels([])
        else:
            ax1.set_xlabel('Step')

        # Loss
        ax2 = axes[idx, 1]
        ax2.plot(history['steps'], history['loss'], 'r-', linewidth=1.5)
        ax2.set_ylabel('Loss')
        ax2.set_xlim(0, max(history['steps']))
        ax2.grid(True, alpha=0.3)
        if idx < len(ranks) - 1:
            ax2.set_xticklabels([])
        else:
            ax2.set_xlabel('Step')

        # TN-sim matrix
        ax3 = axes[idx, 2]
        im = ax3.imshow(tnsim_matrix, aspect='auto', cmap='RdBu_r', vmin=-1, vmax=1)
        ax3.set_ylabel('Checkpoint')

        n_ckpts = len(ckpt_steps)
        n_labels = 8
        label_indices = np.linspace(0, n_ckpts - 1, n_labels).astype(int)
        ax3.set_xticks(label_indices)
        if idx < len(ranks) - 1:
            ax3.set_xticklabels([])
        else:
            ax3.set_xticklabels([str(ckpt_steps[i]) for i in label_indices], rotation=45, ha='right')
            ax3.set_xlabel('Step')
        ax3.set_yticks(label_indices)
        ax3.set_yticklabels([str(ckpt_steps[i]) for i in label_indices])

    # Add colorbar at bottom
    cbar_ax = fig.add_axes([0.55, 0.02, 0.35, 0.015])
    cbar = fig.colorbar(im, cax=cbar_ax, orientation='horizontal')
    cbar.set_label('TN Similarity')

    fig.suptitle('TN-Sim Across Checkpoints: n=4, 2-layer, ranks 2-8', fontsize=14, y=0.995)
    plt.tight_layout(rect=[0, 0.04, 1, 0.99])
    plt.savefig('tnsim_rank_sweep_2to8.png', dpi=150, bbox_inches='tight')
    plt.close()

    print(f"\nSaved: tnsim_rank_sweep_2to8.png")

    # Print summary
    print("\n" + "="*50)
    print("SUMMARY")
    print("="*50)
    print(f"{'Rank':>6} {'Params':>8} {'Final Acc':>10}")
    print("-"*30)
    for rank in ranks:
        n_params = 2 * (3 * rank * n + n)  # 2 layers, each has L,R,D + norm weights
        final_acc = all_results[rank]['history']['eval_acc'][-1]
        print(f"{rank:>6} {n_params:>8} {final_acc:>10.3f}")

    return all_results


if __name__ == "__main__":
    run_rank_sweep()
