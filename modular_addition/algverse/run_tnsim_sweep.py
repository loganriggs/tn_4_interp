"""
Run TN-sim across checkpoints for multiple configurations.
"""

import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).parent.parent / "dev_interp" / "notebooks"))
from tnsim import tn_sim_2layer_with_residual, tn_sim_1layer_with_residual

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

    n_params = sum(p.numel() for p in model.parameters())
    print(f"  Model: n={n}, layers={num_layers}, rank={rank}, params={n_params}")

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

    print(f"  Final acc: {history['eval_acc'][-1]:.3f}")
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
            if num_layers == 1:
                L_a, R_a, D_a = all_weights[i][0]
                L_b, R_b, D_b = all_weights[j][0]
                sim = tn_sim_1layer_with_residual(L_a, R_a, D_a, L_b, R_b, D_b)
            else:
                (L1_a, R1_a, D1_a), (L2_a, R2_a, D2_a) = all_weights[i]
                (L1_b, R1_b, D1_b), (L2_b, R2_b, D2_b) = all_weights[j]
                sim = tn_sim_2layer_with_residual(
                    L1_a, R1_a, D1_a, L2_a, R2_a, D2_a,
                    L1_b, R1_b, D1_b, L2_b, R2_b, D2_b
                )
            tnsim_matrix[i, j] = sim
            tnsim_matrix[j, i] = sim

    return tnsim_matrix, steps


def plot_and_save(history, tnsim_matrix, steps, title, save_path):
    """Plot and save results."""
    fig, axes = plt.subplots(3, 1, figsize=(12, 10),
                              gridspec_kw={'height_ratios': [1, 1, 2.5]})

    ax1 = axes[0]
    ax1.plot(history['steps'], history['eval_acc'], 'b-', linewidth=1.5)
    ax1.set_ylabel('Eval Accuracy')
    ax1.set_xlim(0, max(history['steps']))
    ax1.grid(True, alpha=0.3)
    ax1.set_xticklabels([])

    ax2 = axes[1]
    ax2.plot(history['steps'], history['loss'], 'r-', linewidth=1.5)
    ax2.set_ylabel('Loss')
    ax2.set_xlim(0, max(history['steps']))
    ax2.grid(True, alpha=0.3)
    ax2.set_xticklabels([])

    ax3 = axes[2]
    im = ax3.imshow(tnsim_matrix, aspect='auto', cmap='RdBu_r', vmin=-1, vmax=1)
    ax3.set_ylabel('Checkpoint')
    ax3.set_xlabel('Step')

    n_ckpts = len(steps)
    n_labels = min(12, n_ckpts)
    label_indices = np.linspace(0, n_ckpts - 1, n_labels).astype(int)
    ax3.set_xticks(label_indices)
    ax3.set_xticklabels([str(steps[i]) for i in label_indices], rotation=45, ha='right')
    ax3.set_yticks(label_indices)
    ax3.set_yticklabels([str(steps[i]) for i in label_indices])

    cbar = fig.colorbar(im, ax=ax3, orientation='horizontal', location='bottom',
                        pad=0.15, shrink=0.8)
    cbar.set_label('TN Similarity')

    fig.suptitle(title, fontsize=12)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {save_path}")


def run_config(n, num_layers, rank, steps=10000):
    """Run a single configuration."""
    print(f"\n{'='*60}")
    print(f"Config: n={n}, layers={num_layers}, rank={rank}")
    print(f"{'='*60}")

    checkpoints, history = train_with_checkpoints(n, num_layers, rank, steps)
    tnsim_matrix, ckpt_steps = compute_tnsim_matrix(checkpoints, n, num_layers, rank)

    title = f"TN-Sim: n={n}, {num_layers}-layer, rank={rank}"
    save_path = f"tnsim_n{n}_L{num_layers}_r{rank}.png"
    plot_and_save(history, tnsim_matrix, ckpt_steps, title, save_path)

    return tnsim_matrix, history


if __name__ == "__main__":
    # n=4, 2-layer, varying rank
    for rank in [8, 16, 32]:
        run_config(n=4, num_layers=2, rank=rank)

    # n=5, 2-layer
    run_config(n=5, num_layers=2, rank=16)

    # n=3, 1-layer
    run_config(n=3, num_layers=1, rank=16)

    print("\nAll done!")
