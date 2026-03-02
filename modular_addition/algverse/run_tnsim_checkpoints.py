"""
Train BilinearResidualRMSNorm and compute TN-sim across checkpoints.
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


def train_with_checkpoints(
    n: int = 4,
    num_layers: int = 2,
    rank: int = 16,
    steps: int = 10_000,
    batch_size: int = 128,
    lr: float = 0.01,
    weight_decay: float = 0.001,
    checkpoint_every: int = 200,
    device: str = "cuda" if torch.cuda.is_available() else "cpu",
):
    """Train model and save checkpoints."""
    model = BilinearResidualRMSNorm(n, num_layers, rank, use_rmsnorm=True).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)

    checkpoints = {}
    history = {'steps': [], 'loss': [], 'eval_acc': []}

    n_params = sum(p.numel() for p in model.parameters())
    print(f"Model: n={n}, layers={num_layers}, rank={rank}")
    print(f"Parameters: {n_params}")

    for step in range(steps + 1):
        model.train()
        x = torch.randn(batch_size, n, device=device)
        targets = task_2nd_argmax(x)
        logits = model(x)
        loss = nn.functional.cross_entropy(logits, targets)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # Save checkpoint
        if step % checkpoint_every == 0:
            checkpoints[step] = {k: v.cpu().clone() for k, v in model.state_dict().items()}

            # Eval
            model.eval()
            with torch.no_grad():
                x_eval = torch.randn(10000, n, device=device)
                targets_eval = task_2nd_argmax(x_eval)
                logits_eval = model(x_eval)
                eval_acc = (logits_eval.argmax(-1) == targets_eval).float().mean().item()

            history['steps'].append(step)
            history['loss'].append(loss.item())
            history['eval_acc'].append(eval_acc)

            if step % 1000 == 0:
                print(f"Step {step:5d}: loss={loss.item():.4f}, eval_acc={eval_acc:.3f}")

    return model, checkpoints, history


def compute_tnsim_matrix(checkpoints: dict, n: int, num_layers: int, rank: int):
    """Compute pairwise TN-sim matrix between checkpoints (weight-based)."""
    steps = sorted(checkpoints.keys())
    n_ckpts = len(steps)

    print(f"Computing TN-sim matrix for {n_ckpts} checkpoints...")

    # Create model template
    model = BilinearResidualRMSNorm(n, num_layers, rank, use_rmsnorm=True)

    # Extract weights for each checkpoint
    def get_weights(state_dict):
        weights = []
        for i in range(num_layers):
            L = state_dict[f'layers.{i}.L']
            R = state_dict[f'layers.{i}.R']
            D = state_dict[f'layers.{i}.D']
            weights.append((L, R, D))
        return weights

    all_weights = [get_weights(checkpoints[s]) for s in steps]

    # Compute pairwise TN-sim
    tnsim_matrix = np.zeros((n_ckpts, n_ckpts))

    for i in range(n_ckpts):
        for j in range(i, n_ckpts):
            if num_layers == 1:
                L_a, R_a, D_a = all_weights[i][0]
                L_b, R_b, D_b = all_weights[j][0]
                sim = tn_sim_1layer_with_residual(L_a, R_a, D_a, L_b, R_b, D_b)
            elif num_layers == 2:
                (L1_a, R1_a, D1_a), (L2_a, R2_a, D2_a) = all_weights[i]
                (L1_b, R1_b, D1_b), (L2_b, R2_b, D2_b) = all_weights[j]
                sim = tn_sim_2layer_with_residual(
                    L1_a, R1_a, D1_a, L2_a, R2_a, D2_a,
                    L1_b, R1_b, D1_b, L2_b, R2_b, D2_b
                )
            else:
                raise NotImplementedError

            tnsim_matrix[i, j] = sim
            tnsim_matrix[j, i] = sim

        if (i + 1) % 10 == 0:
            print(f"  Processed {i+1}/{n_ckpts} checkpoints")

    return tnsim_matrix, steps


def plot_results(history, tnsim_matrix, steps, save_path=None):
    """Plot accuracy and TN-sim matrix."""
    fig, axes = plt.subplots(3, 1, figsize=(12, 10),
                              gridspec_kw={'height_ratios': [1, 1, 2.5]})

    # Accuracy
    ax1 = axes[0]
    ax1.plot(history['steps'], history['eval_acc'], 'b-', linewidth=1.5)
    ax1.set_ylabel('Eval Accuracy')
    ax1.set_xlim(0, max(history['steps']))
    ax1.grid(True, alpha=0.3)
    ax1.set_xticklabels([])

    # Loss
    ax2 = axes[1]
    ax2.plot(history['steps'], history['loss'], 'r-', linewidth=1.5)
    ax2.set_ylabel('Loss')
    ax2.set_xlim(0, max(history['steps']))
    ax2.grid(True, alpha=0.3)
    ax2.set_xticklabels([])

    # TN-sim matrix
    ax3 = axes[2]
    im = ax3.imshow(tnsim_matrix, aspect='auto', cmap='RdBu_r', vmin=-1, vmax=1)
    ax3.set_ylabel('Checkpoint')
    ax3.set_xlabel('Step')

    # Sparse labels
    n_ckpts = len(steps)
    n_labels = min(12, n_ckpts)
    label_indices = np.linspace(0, n_ckpts - 1, n_labels).astype(int)
    ax3.set_xticks(label_indices)
    ax3.set_xticklabels([str(steps[i]) for i in label_indices], rotation=45, ha='right')
    ax3.set_yticks(label_indices)
    ax3.set_yticklabels([str(steps[i]) for i in label_indices])

    # Colorbar
    cbar = fig.colorbar(im, ax=ax3, orientation='horizontal', location='bottom',
                        pad=0.15, shrink=0.8)
    cbar.set_label('TN Similarity')

    fig.suptitle(f"TN-Sim Across Checkpoints (n=4, 2-layer, rank=16)", fontsize=12)
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Saved to {save_path}")

    plt.show()
    return fig


if __name__ == "__main__":
    # Train with checkpoints
    model, checkpoints, history = train_with_checkpoints(
        n=4,
        num_layers=2,
        rank=16,
        steps=10_000,
        checkpoint_every=200,
    )

    # Compute TN-sim matrix
    tnsim_matrix, steps = compute_tnsim_matrix(
        checkpoints, n=4, num_layers=2, rank=16
    )

    # Plot
    plot_results(history, tnsim_matrix, steps, save_path="tnsim_checkpoints.png")

    print("\nDone!")
