# %%
"""
1-Layer Symmetric Bilinear Analysis for n=3, 4, 5

Loads pre-trained models from checkpoints folder.
- Visualize weights (L, D, D@L)
- Show example predictions
"""

import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import sys

# =============================================================================
# PROJECT ROOT - Edit this if running from a different location
# =============================================================================
PROJECT_ROOT = Path("/workspace/tn_4_interp/modular_addition/algverse")

sys.path.insert(0, str(PROJECT_ROOT))
from models import SymmetricBilinearResidual, task_2nd_argmax

# %%
# =============================================================================
# LOAD CHECKPOINTS
# =============================================================================
checkpoint_dir = PROJECT_ROOT / "checkpoints"

results = {}
for n in [3, 4, 5]:
    # Find checkpoint for this n
    checkpoints = list(checkpoint_dir.glob(f"1layer_n{n}_*.pt"))
    if not checkpoints:
        print(f"No checkpoint found for n={n}. Run training/train_1layer_n345.py first.")
        continue

    ckpt_path = checkpoints[0]  # Take first match
    print(f"Loading {ckpt_path}")

    data = torch.load(ckpt_path, map_location='cpu')
    cfg = data['config']
    state = data['state_dict']
    acc = data['accuracy']

    # Create model and load weights
    model = SymmetricBilinearResidual(cfg['n'], cfg['num_layers'], cfg['rank'])
    model.load_state_dict(state)

    # Extract weights (1-layer has layers.0.L, layers.0.D, norms.0.weight)
    L = state['layers.0.L']
    D = state['layers.0.D']
    norm_w = state['norms.0.weight']

    L_sparsity = (L == 0).sum().item() / L.numel()
    D_sparsity = (D == 0).sum().item() / D.numel()

    results[n] = {
        'L': L, 'D': D, 'norm_weight': norm_w,
        'acc': acc, 'L_sparsity': L_sparsity, 'D_sparsity': D_sparsity,
        'model': model, 'seed': cfg['seed'],
    }
    print(f"  n={n}: acc={acc:.1%}, L sparse={L_sparsity:.0%}, D sparse={D_sparsity:.0%}")

if not results:
    raise RuntimeError("No checkpoints found. Run training/train_1layer_n345.py first.")

# %%
# =============================================================================
# WEIGHT VISUALIZATION
# =============================================================================
print("\n" + "="*60)
print("WEIGHT VISUALIZATION")
print("="*60)

n_models = len(results)
fig, axes = plt.subplots(n_models, 3, figsize=(15, 4 * n_models))
if n_models == 1:
    axes = axes.reshape(1, -1)

def plot_weights(ax, mat, title):
    mat_np = mat.numpy()
    vmax = max(abs(mat_np.min()), abs(mat_np.max()))
    if vmax == 0:
        vmax = 1
    im = ax.imshow(mat_np, cmap='RdBu_r', vmin=-vmax, vmax=vmax, aspect='auto')
    ax.set_title(title, fontsize=10)
    for i in range(mat_np.shape[0]):
        for j in range(mat_np.shape[1]):
            val = mat_np[i, j]
            color = 'white' if abs(val) > vmax * 0.5 else 'black'
            ax.text(j, i, f'{val:.2f}', ha='center', va='center', fontsize=8, color=color)
    plt.colorbar(im, ax=ax, shrink=0.7)

for row, n in enumerate(sorted(results.keys())):
    res = results[n]
    L, D = res['L'], res['D']
    DL = D @ L

    plot_weights(axes[row, 0], L, f'n={n}: L ({L.shape[0]}×{L.shape[1]}, {res["L_sparsity"]:.0%} sparse)')
    plot_weights(axes[row, 1], D, f'n={n}: D ({D.shape[0]}×{D.shape[1]}, {res["D_sparsity"]:.0%} sparse)')
    plot_weights(axes[row, 2], DL, f'n={n}: D@L acc={res["acc"]:.1%}')

    axes[row, 0].set_ylabel(f'n={n}', fontsize=12, fontweight='bold')

plt.suptitle('1-Layer Symmetric Bilinear Weights (L1 pruned)', fontsize=14)
plt.tight_layout()
images_dir = PROJECT_ROOT / "images"
plt.savefig(images_dir / '1layer_weights_n345.png', dpi=150)
plt.show()

# %%
# =============================================================================
# EXAMPLE PREDICTIONS
# =============================================================================
print("\n" + "="*60)
print("EXAMPLE PREDICTIONS")
print("="*60)

fig, axes = plt.subplots(n_models, 5, figsize=(18, 3.5 * n_models))
if n_models == 1:
    axes = axes.reshape(1, -1)

for row, n in enumerate(sorted(results.keys())):
    res = results[n]
    L, D, norm_w = res['L'], res['D'], res['norm_weight']

    # Generate examples
    torch.manual_seed(42)
    x_examples = torch.randn(100, n)
    targets = task_2nd_argmax(x_examples)

    # Compute outputs
    with torch.no_grad():
        h = norm_w * (x_examples / torch.sqrt((x_examples ** 2).mean(dim=-1, keepdim=True) + 1e-6))
        Lh = h @ L.T
        bilinear = (Lh ** 2) @ D.T
        output = x_examples + bilinear
        preds = output.argmax(dim=1)
        correct = (preds == targets)

    # Get 3 correct, 2 incorrect
    correct_idx = torch.where(correct)[0][:3]
    wrong_idx = torch.where(~correct)[0][:2]
    example_indices = list(correct_idx.numpy()) + list(wrong_idx.numpy())

    for col, idx in enumerate(example_indices):
        ax = axes[row, col]
        x_i = x_examples[idx].numpy()
        out_i = output[idx].numpy()
        target_i = targets[idx].item()
        pred_i = preds[idx].item()

        x_pos = np.arange(n)
        width = 0.35

        ax.bar(x_pos - width/2, x_i, width, label='Input x', color='steelblue', alpha=0.7)
        ax.bar(x_pos + width/2, out_i, width, label='Output', color='coral', alpha=0.7)

        # Mark target and prediction
        ax.axvline(x=target_i, color='green', linestyle='--', linewidth=2, label=f'Target={target_i}')
        if pred_i != target_i:
            ax.axvline(x=pred_i, color='red', linestyle=':', linewidth=2, label=f'Pred={pred_i}')

        is_correct = pred_i == target_i
        status = "CORRECT" if is_correct else "WRONG"
        ax.set_title(f'{status}: pred={pred_i}, target={target_i}', fontsize=9,
                    color='green' if is_correct else 'red')
        ax.set_xticks(x_pos)
        ax.set_xlabel('Position')
        if col == 0:
            ax.set_ylabel(f'n={n}', fontsize=11, fontweight='bold')
        if row == 0 and col == 0:
            ax.legend(fontsize=7, loc='upper right')

plt.suptitle('Example Predictions: Input (blue) vs Output (orange)\nGreen line = 2nd-argmax target', fontsize=12)
plt.tight_layout()
plt.savefig(images_dir / '1layer_examples_n345.png', dpi=150)
plt.show()

# %%
# =============================================================================
# SUMMARY
# =============================================================================
print("\n" + "="*60)
print("SUMMARY")
print("="*60)

print(f"\n{'n':<5} {'Seed':<6} {'Accuracy':<12} {'L sparse':<12} {'D sparse':<12} {'Random':<10}")
print("-" * 65)
for n in sorted(results.keys()):
    res = results[n]
    random_baseline = 1.0 / n
    print(f"{n:<5} {res['seed']:<6} {res['acc']:<12.1%} {res['L_sparsity']:<12.0%} {res['D_sparsity']:<12.0%} {random_baseline:<10.1%}")

# %%
# =============================================================================
# INTERPRETATION
# =============================================================================
print("\n" + "="*60)
print("INTERPRETATION")
print("="*60)

for n in sorted(results.keys()):
    res = results[n]
    L, D = res['L'], res['D']
    DL = D @ L

    print(f"\nn={n}:")
    print(f"  Model: output = x + D @ (L @ norm(x))²")
    print(f"  Accuracy: {res['acc']:.1%} (random = {100/n:.1f}%)")
    print(f"  Gain over random: {(res['acc'] - 1/n)*100:.1f}%")

    # D @ L structure
    DL_np = DL.numpy()
    diag_mean = np.mean(np.diag(DL_np))
    offdiag_vals = DL_np[~np.eye(n, dtype=bool)]
    offdiag_mean = np.mean(offdiag_vals)

    print(f"  D@L diagonal mean: {diag_mean:.3f}")
    print(f"  D@L off-diagonal mean: {offdiag_mean:.3f}")

# %%
