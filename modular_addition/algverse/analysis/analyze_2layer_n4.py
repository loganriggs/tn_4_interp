# %%
"""
2-Layer Symmetric Bilinear Analysis for n=4

Computational paths:
  output = x + r1 + r2

Where r1 = layer1(norm1(x)) and r2 = layer2(norm2(x + r1))

We can decompose r2 into 3 parts based on the bilinear expansion:
  h2 = norm2(x + r1)

If we linearly approximate: h2 ≈ a*x_norm + b*r1_norm (ignoring norm interaction)
Then (L2 @ h2)² expands to:
  A: (L2 @ x_component)² term
  B: 2*(L2 @ x_component)*(L2 @ r1_component) cross term
  C: (L2 @ r1_component)² term

So 5 paths total: x, r1, A, B, C
"""

import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import sys
from itertools import combinations

# =============================================================================
# PROJECT ROOT
# =============================================================================
PROJECT_ROOT = Path("/workspace/tn_4_interp/modular_addition/algverse")
sys.path.insert(0, str(PROJECT_ROOT))
from models import task_2nd_argmax
from analysis.analysis_utils import (
    compute_quadratic_forms, bilinear_forward, rmsnorm,
    plot_weight_matrix, plot_quadratic_forms, print_quadratic_analysis,
    powerset_ablation, removal_ablation
)

checkpoint_dir = PROJECT_ROOT / "checkpoints"
images_dir = PROJECT_ROOT / "images"

# %%
# =============================================================================
# LOAD CHECKPOINT (sparse model from L1 pruning)
# =============================================================================
import pickle

prune_path = checkpoint_dir / "prune_symmetric_results.pkl"
print(f"Loading sparse model from {prune_path}")

with open(prune_path, 'rb') as f:
    prune_data = pickle.load(f)

# Get the best sparse result
best_result = max(prune_data['results'], key=lambda x: x['final_acc'])
cfg = prune_data['config']
state = best_result['state_dict']
acc = best_result['final_acc']
sparsity_info = best_result['sparsity']

print(f"Threshold: {best_result['threshold']}")
print(f"Sparsity: L1={sparsity_info['L1']:.0%}, D1={sparsity_info['D1']:.0%}, "
      f"L2={sparsity_info['L2']:.0%}, D2={sparsity_info['D2']:.0%}")

n = cfg['n']
rank = cfg['rank']
num_layers = cfg['num_layers']
seed = cfg['seed']

# Extract weights
L1 = state['layers.0.L']
D1 = state['layers.0.D']
norm1_w = state['norms.0.weight']

L2 = state['layers.1.L']
D2 = state['layers.1.D']
norm2_w = state['norms.1.weight']

print(f"Config: n={n}, layers={num_layers}, rank={rank}, seed={seed}")
print(f"Accuracy: {acc:.1%}")

# %%
# =============================================================================
# WEIGHT VISUALIZATION
# =============================================================================
print("\n" + "="*60)
print("WEIGHT VISUALIZATION")
print("="*60)

fig, axes = plt.subplots(2, 3, figsize=(15, 8))

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
            ax.text(j, i, f'{val:.2f}', ha='center', va='center', fontsize=7, color=color)
    plt.colorbar(im, ax=ax, shrink=0.8)

# Layer 1
plot_weights(axes[0, 0], L1, f'L1 ({rank}×{n})')
plot_weights(axes[0, 1], D1, f'D1 ({n}×{rank})')
plot_weights(axes[0, 2], D1 @ L1, f'D1 @ L1 [NOT effective!]')

# Layer 2
plot_weights(axes[1, 0], L2, f'L2 ({rank}×{n})')
plot_weights(axes[1, 1], D2, f'D2 ({n}×{rank})')
plot_weights(axes[1, 2], D2 @ L2, f'D2 @ L2 [NOT effective!]')

axes[0, 0].set_ylabel('Layer 1', fontsize=12, fontweight='bold')
axes[1, 0].set_ylabel('Layer 2', fontsize=12, fontweight='bold')

plt.suptitle(f'2-Layer Weights: n={n}, seed={seed}', fontsize=12)
plt.tight_layout()
plt.savefig(images_dir / f'2layer_n{n}_weights.png', dpi=150)
plt.show()

# %%
# =============================================================================
# QUADRATIC FORM ANALYSIS (THE CORRECT MATH)
# =============================================================================
print("\n" + "="*60)
print("QUADRATIC FORM ANALYSIS")
print("="*60)
print("""
Key insight: Each bilinear layer computes D @ (L @ h)² where ² is ELEMENTWISE.
This creates QUADRATIC FORMS:

    bilinear_i = h^T M^(i) h

where M^(i)_jk = Σ_r D_ir L_rj L_rk is a symmetric matrix for each output i.
""")

# Compute quadratic form matrices for both layers
M1 = compute_quadratic_forms(L1, D1)
M2 = compute_quadratic_forms(L2, D2)

print(f"M1 shape: {M1.shape} (Layer 1: {n} matrices of size {n}×{n})")
print(f"M2 shape: {M2.shape} (Layer 2: {n} matrices of size {n}×{n})")

# Print analysis for both layers
print_quadratic_analysis(M1, "Layer 1")
print_quadratic_analysis(M2, "Layer 2")

# Visualize quadratic form matrices
fig1 = plot_quadratic_forms(M1, title_prefix="M1", save_path=images_dir / f'2layer_n{n}_M1_quadforms.png')
plt.show()

fig2 = plot_quadratic_forms(M2, title_prefix="M2", save_path=images_dir / f'2layer_n{n}_M2_quadforms.png')
plt.show()

# %%
# =============================================================================
# COMPUTE 5 COMPONENT PATHS
# =============================================================================
def compute_5_paths(x):
    """
    Compute the 5 computational paths:
    - x: input (residual)
    - r1: layer 1 output
    - A: x contribution through layer 2
    - B: cross term in layer 2
    - C: r1 contribution through layer 2

    Returns dict with all components.
    """
    batch = x.shape[0]

    # Layer 1
    rms1 = torch.sqrt((x ** 2).mean(dim=-1, keepdim=True) + 1e-6)
    h1 = norm1_w * (x / rms1)
    Lh1 = h1 @ L1.T
    r1 = (Lh1 ** 2) @ D1.T

    # After layer 1
    h_mid = x + r1

    # Layer 2 - full computation
    rms2 = torch.sqrt((h_mid ** 2).mean(dim=-1, keepdim=True) + 1e-6)
    h2 = norm2_w * (h_mid / rms2)
    Lh2 = h2 @ L2.T
    r2_full = (Lh2 ** 2) @ D2.T

    # Decompose layer 2 into A, B, C
    # h2 = norm2_w * (x + r1) / rms2
    # Let's compute x and r1 contributions separately through the norm
    # x_contrib = norm2_w * x / rms2
    # r1_contrib = norm2_w * r1 / rms2

    x_contrib = norm2_w * x / rms2
    r1_contrib = norm2_w * r1 / rms2

    # L2 projections
    Lx = x_contrib @ L2.T   # (batch, rank)
    Lr1 = r1_contrib @ L2.T  # (batch, rank)

    # A: (Lx)² term
    A = (Lx ** 2) @ D2.T

    # C: (Lr1)² term
    C = (Lr1 ** 2) @ D2.T

    # B: 2 * Lx * Lr1 cross term
    B = (2 * Lx * Lr1) @ D2.T

    # Verify: A + B + C should equal r2_full (approximately, up to norm interactions)
    r2_reconstructed = A + B + C

    # Full output
    output = x + r1 + r2_full

    return {
        'x': x,
        'r1': r1,
        'A': A,
        'B': B,
        'C': C,
        'r2_full': r2_full,
        'r2_reconstructed': r2_reconstructed,
        'output': output,
    }

# %%
# =============================================================================
# POWERSET ABLATION
# =============================================================================
print("\n" + "="*60)
print("POWERSET ABLATION (all 32 combinations)")
print("="*60)

torch.manual_seed(123)
x_eval = torch.randn(10000, n)
targets = task_2nd_argmax(x_eval)
paths = compute_5_paths(x_eval)

# Component names
components = ['x', 'r1', 'A', 'B', 'C']

# Store results
ablation_results = []

# Try all 32 combinations (powerset)
for num_active in range(6):  # 0 to 5 components
    for combo in combinations(range(5), num_active):
        # Build output from selected components
        output = torch.zeros_like(x_eval)
        active_names = []
        for i in combo:
            output = output + paths[components[i]]
            active_names.append(components[i])

        if len(active_names) == 0:
            active_names = ['none']

        preds = output.argmax(dim=1)
        acc = (preds == targets).float().mean().item()

        ablation_results.append({
            'components': tuple(active_names),
            'num_components': len(combo),
            'accuracy': acc,
        })

# Sort by accuracy
ablation_results.sort(key=lambda x: -x['accuracy'])

print(f"\n{'Components':<30} {'Acc':>8}")
print("-" * 40)
for r in ablation_results:
    comp_str = '+'.join(r['components'])
    print(f"{comp_str:<30} {r['accuracy']*100:>7.1f}%")

# %%
# =============================================================================
# KEY ABLATION INSIGHTS
# =============================================================================
print("\n" + "="*60)
print("KEY ABLATION INSIGHTS")
print("="*60)

# Full model
full_acc = paths['output'].argmax(dim=1).eq(targets).float().mean().item()
print(f"\nFull model (x+r1+A+B+C): {full_acc:.1%}")

# Find most important single component
single_results = [r for r in ablation_results if r['num_components'] == 1]
print("\nSingle component accuracies:")
for r in single_results:
    print(f"  {r['components'][0]}: {r['accuracy']:.1%}")

# Remove one at a time from full
print("\nRemove one component from full:")
for comp in components:
    remaining = [c for c in components if c != comp]
    output = sum(paths[c] for c in remaining)
    acc = output.argmax(dim=1).eq(targets).float().mean().item()
    delta = acc - full_acc
    print(f"  Without {comp}: {acc:.1%} (Δ={delta*100:+.1f}%)")

# %%
# =============================================================================
# EXAMPLE VISUALIZATION FUNCTION
# =============================================================================
def plot_example_2layer(x_i, target, full_pred, title=""):
    """Plot 2-layer computation flow."""
    x_t = torch.tensor(x_i).unsqueeze(0)
    paths_i = compute_5_paths(x_t)

    # Extract all components
    x_np = x_i.reshape(1, -1)
    r1_np = paths_i['r1'].numpy()
    A_np = paths_i['A'].numpy()
    B_np = paths_i['B'].numpy()
    C_np = paths_i['C'].numpy()
    output_np = paths_i['output'].numpy()

    # Create figure
    fig, axes = plt.subplots(6, 1, figsize=(10, 10),
                              gridspec_kw={'height_ratios': [1, 1, 1, 1, 1, 1]})

    def plot_row(ax, data, title, vmax=None):
        if vmax is None:
            vmax = max(abs(data.min()), abs(data.max()))
            if vmax == 0:
                vmax = 1
        im = ax.imshow(data, cmap='RdBu_r', vmin=-vmax, vmax=vmax, aspect='auto')
        ax.set_title(title, fontsize=9, loc='left')
        ax.set_xticks(range(data.shape[1]))
        ax.set_yticks([])
        for j in range(data.shape[1]):
            val = data[0, j]
            color = 'white' if abs(val) > vmax * 0.5 else 'black'
            ax.text(j, 0, f'{val:.2f}', ha='center', va='center', fontsize=8, color=color)

        # Mark target and prediction
        rect_t = plt.Rectangle((target - 0.5, -0.5), 1, 1,
                               fill=False, edgecolor='green', linewidth=2)
        ax.add_patch(rect_t)
        if full_pred != target:
            rect_p = plt.Rectangle((full_pred - 0.5, -0.5), 1, 1,
                                   fill=False, edgecolor='red', linewidth=2, linestyle='--')
            ax.add_patch(rect_p)
        return im

    # Find global vmax
    all_data = np.concatenate([x_np, r1_np, A_np, B_np, C_np, output_np], axis=0)
    vmax = max(abs(all_data.min()), abs(all_data.max()))

    plot_row(axes[0], x_np, 'x (input)', vmax)
    plot_row(axes[1], r1_np, 'r1 = layer1(norm(x))', vmax)
    plot_row(axes[2], A_np, 'A = (L2 @ x_norm)² @ D2 (x→layer2)', vmax)
    plot_row(axes[3], B_np, 'B = 2*(L2@x)*(L2@r1) @ D2 (cross term)', vmax)
    plot_row(axes[4], C_np, 'C = (L2 @ r1_norm)² @ D2 (r1→layer2)', vmax)
    plot_row(axes[5], output_np, 'Output = x + r1 + A + B + C', vmax)
    axes[5].set_xlabel('Position')

    legend_text = f"Target: {target} (green)"
    if full_pred != target:
        legend_text += f" | Pred: {full_pred} (red)"

    fig.suptitle(f'{title}\n{legend_text}', fontsize=11)
    plt.tight_layout()
    return fig

# %%
# =============================================================================
# CATEGORIZE EXAMPLES
# =============================================================================
print("\n" + "="*60)
print("EXAMPLE CATEGORIES")
print("="*60)

torch.manual_seed(42)
x_sample = torch.randn(1000, n)
targets_s = task_2nd_argmax(x_sample)
paths_s = compute_5_paths(x_sample)

# Get predictions from different component combinations
full_preds = paths_s['output'].argmax(dim=1)
no_x_preds = (paths_s['r1'] + paths_s['A'] + paths_s['B'] + paths_s['C']).argmax(dim=1)
no_r1_preds = (paths_s['x'] + paths_s['A'] + paths_s['B'] + paths_s['C']).argmax(dim=1)
xr1_only_preds = (paths_s['x'] + paths_s['r1']).argmax(dim=1)

full_correct = (full_preds == targets_s)
full_acc = full_correct.float().mean().item()
print(f"Full model accuracy: {full_acc:.1%}")

# %%
# =============================================================================
# EXAMPLE: Full model CORRECT
# =============================================================================
print("\n" + "="*60)
print("EXAMPLE: Full Model CORRECT")
print("="*60)

correct_idx = torch.where(full_correct)[0][0].item()
target = targets_s[correct_idx].item()
pred = full_preds[correct_idx].item()

fig = plot_example_2layer(
    x_sample[correct_idx].numpy(),
    target, pred,
    f"Full Model CORRECT"
)
plt.show()

# %%
# =============================================================================
# EXAMPLE: Full model WRONG
# =============================================================================
print("\n" + "="*60)
print("EXAMPLE: Full Model WRONG")
print("="*60)

wrong_idx = torch.where(~full_correct)[0][0].item()
target = targets_s[wrong_idx].item()
pred = full_preds[wrong_idx].item()

fig = plot_example_2layer(
    x_sample[wrong_idx].numpy(),
    target, pred,
    f"Full Model WRONG"
)
plt.show()

# %%
# =============================================================================
# SUMMARY
# =============================================================================
print("\n" + "="*60)
print("SUMMARY")
print("="*60)

# Get top 5 and bottom 5 combinations
print("\nTop 5 component combinations:")
for r in ablation_results[:5]:
    print(f"  {'+'.join(r['components'])}: {r['accuracy']:.1%}")

print("\nBottom 5 component combinations:")
for r in ablation_results[-5:]:
    print(f"  {'+'.join(r['components'])}: {r['accuracy']:.1%}")

# Most important components (by removal impact)
print("\nComponent importance (accuracy drop when removed):")
importance = []
for comp in components:
    remaining = [c for c in components if c != comp]
    output = sum(paths[c] for c in remaining)
    acc = output.argmax(dim=1).eq(targets).float().mean().item()
    importance.append((comp, full_acc - acc))
importance.sort(key=lambda x: -x[1])
for comp, drop in importance:
    print(f"  {comp}: {drop*100:+.1f}%")

# %%
