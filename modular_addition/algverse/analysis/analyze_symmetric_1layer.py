# %%
"""
1-Layer Symmetric Bilinear Analysis

Loads pre-trained model from checkpoints.
Visualizes weights and example predictions with ablation.
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
from analysis.analysis_utils import (
    compute_quadratic_forms, bilinear_forward, bilinear_as_quadratic,
    rmsnorm, plot_weight_matrix, plot_quadratic_forms, analyze_quadratic_form,
    print_quadratic_analysis
)

checkpoint_dir = PROJECT_ROOT / "checkpoints"
images_dir = PROJECT_ROOT / "images"

# %%
# =============================================================================
# CONFIG - Choose which n to analyze
# =============================================================================
N = 4  # Change to 3, 4, or 5

# %%
# =============================================================================
# LOAD CHECKPOINT
# =============================================================================
checkpoints = list(checkpoint_dir.glob(f"1layer_n{N}_*.pt"))
if not checkpoints:
    raise RuntimeError(f"No checkpoint for n={N}. Run training/train_1layer_n345.py first.")

ckpt_path = checkpoints[0]
print(f"Loading {ckpt_path}")

data = torch.load(ckpt_path, map_location='cpu', weights_only=False)
cfg = data['config']
state = data['state_dict']
acc = data['accuracy']

n = cfg['n']
rank = cfg['rank']
seed = cfg['seed']

# Extract weights
L = state['layers.0.L']
D = state['layers.0.D']
norm_w = state['norms.0.weight']

L_sparsity = (L == 0).sum().item() / L.numel()
D_sparsity = (D == 0).sum().item() / D.numel()

print(f"Config: n={n}, rank={rank}, seed={seed}")
print(f"Accuracy: {acc:.1%}")
print(f"L sparsity: {L_sparsity:.0%}, D sparsity: {D_sparsity:.0%}")

# %%
# =============================================================================
# WEIGHT VISUALIZATION
# =============================================================================
print("\n" + "="*60)
print("WEIGHT VISUALIZATION")
print("="*60)

fig, axes = plt.subplots(1, 3, figsize=(14, 4))

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
    plt.colorbar(im, ax=ax, shrink=0.8)

plot_weights(axes[0], L, f'L ({rank}×{n}, {L_sparsity:.0%} sparse)')
plot_weights(axes[1], D, f'D ({n}×{rank}, {D_sparsity:.0%} sparse)')
DL = D @ L
plot_weights(axes[2], DL, f'D @ L ({n}×{n}) [NOT the effective matrix!]')

plt.suptitle(f'1-Layer Weights: n={n}, seed={seed}, acc={acc:.1%}', fontsize=12)
plt.tight_layout()
plt.savefig(images_dir / f'1layer_n{n}_weights.png', dpi=150)
plt.show()

# %%
# =============================================================================
# QUADRATIC FORM ANALYSIS (THE CORRECT MATH)
# =============================================================================
print("\n" + "="*60)
print("QUADRATIC FORM ANALYSIS")
print("="*60)
print("""
Key insight: The bilinear layer computes D @ (L @ h)² where ² is ELEMENTWISE.
This is a QUADRATIC FORM, not a linear transform:

    bilinear_i = Σ_r D_ir (Σ_j L_rj h_j)²
               = Σ_{j,k} M^(i)_jk h_j h_k
               = h^T M^(i) h

where M^(i)_jk = Σ_r D_ir L_rj L_rk is a symmetric matrix for each output i.
""")

# Compute quadratic form matrices
M = compute_quadratic_forms(L, D)
print(f"M shape: {M.shape} = ({n} matrices of size {n}×{n})")

# Print analysis
print_quadratic_analysis(M, "Layer 1")

# Visualize the quadratic form matrices
fig = plot_quadratic_forms(M, title_prefix="M", save_path=images_dir / f'1layer_n{n}_quadratic_forms.png')
plt.show()

# Verify the math: compare bilinear_forward with bilinear_as_quadratic
print("\nVerification: bilinear_forward vs bilinear_as_quadratic")
torch.manual_seed(999)
test_h = torch.randn(100, n)
output_direct, _ = bilinear_forward(test_h, L, D)
output_quadratic = bilinear_as_quadratic(test_h, M)
max_diff = (output_direct - output_quadratic).abs().max().item()
print(f"Max difference: {max_diff:.2e} (should be ~1e-6)")

# %%
# =============================================================================
# HELPER: Compute model components
# =============================================================================
def compute_components(x):
    """Compute input, bilinear output, and full output."""
    # RMSNorm
    rms = torch.sqrt((x ** 2).mean(dim=-1, keepdim=True) + 1e-6)
    h = norm_w * (x / rms)

    # Bilinear layer
    Lh = h @ L.T
    bilinear = (Lh ** 2) @ D.T

    # Full output
    full = x + bilinear

    return {
        'x': x,
        'bilinear': bilinear,
        'full': full,
    }

def get_predictions(comps, x):
    """Get predictions and correctness for each component."""
    targets = task_2nd_argmax(x)

    bilinear_preds = comps['bilinear'].argmax(dim=1)
    full_preds = comps['full'].argmax(dim=1)

    bilinear_correct = (bilinear_preds == targets)
    full_correct = (full_preds == targets)

    return {
        'targets': targets,
        'bilinear_preds': bilinear_preds,
        'full_preds': full_preds,
        'bilinear_correct': bilinear_correct,
        'full_correct': full_correct,
    }

# %%
# =============================================================================
# ABLATION ACCURACY
# =============================================================================
print("\n" + "="*60)
print("ABLATION ACCURACY")
print("="*60)

torch.manual_seed(123)
x_eval = torch.randn(10000, n)
comps = compute_components(x_eval)
preds = get_predictions(comps, x_eval)

full_acc = preds['full_correct'].float().mean().item()
bilinear_acc = preds['bilinear_correct'].float().mean().item()
x_only_acc = (x_eval.argmax(dim=1) == preds['targets']).float().mean().item()

print(f"\nFull model (x + bilinear): {full_acc:.1%}")
print(f"Bilinear only: {bilinear_acc:.1%}")
print(f"x only: {x_only_acc:.1%}")
print(f"Random baseline: {100/n:.1f}%")

# Count categories
both_correct = (preds['bilinear_correct'] & preds['full_correct']).sum().item()
bilinear_only_correct = (preds['bilinear_correct'] & ~preds['full_correct']).sum().item()
full_only_correct = (~preds['bilinear_correct'] & preds['full_correct']).sum().item()
both_wrong = (~preds['bilinear_correct'] & ~preds['full_correct']).sum().item()

print(f"\nBreakdown (out of {len(x_eval)}):")
print(f"  Both correct: {both_correct}")
print(f"  Bilinear correct, Full wrong: {bilinear_only_correct}")
print(f"  Bilinear wrong, Full correct: {full_only_correct}")
print(f"  Both wrong: {both_wrong}")

# %%
# =============================================================================
# EXAMPLE VISUALIZATION FUNCTION
# =============================================================================
def plot_example_flow(x_i, target, bilinear_pred, full_pred, title=""):
    """
    Plot computation flow with imshow:
    1. Input x (1×n)
    2. h = norm(x) (1×n)
    3. L matrix (rank×n)
    4. Lh and Lh² (rank×1) - the scaling factors
    5. D.T matrix (rank×n)
    6. Scaled D.T: each row scaled by Lh²[i]
    7. Bilinear output = sum of scaled D (1×n)
    8. Output = x + bilinear (1×n)
    """
    # Convert to tensor for computation
    x_t = torch.tensor(x_i).unsqueeze(0)

    # RMSNorm: h = γ * (x / rms) where γ is scalar
    rms = torch.sqrt((x_t ** 2).mean(dim=-1, keepdim=True) + 1e-6)
    gamma = norm_w.item()  # scalar weight
    rms_val = rms.item()
    h = (norm_w * (x_t / rms)).squeeze(0)  # (n,)

    # Compute intermediates
    Lh = (h @ L.T)  # (rank,)
    Lh_sq = Lh ** 2  # (rank,)

    # Scaled D: each row i of D.T scaled by Lh_sq[i]
    # D is (n, rank), D.T is (rank, n)
    scaled_D = D.T * Lh_sq.unsqueeze(1)  # (rank, n)

    bilinear = scaled_D.sum(dim=0)  # (n,)
    output = x_t.squeeze(0) + bilinear  # (n,)

    # Convert to numpy
    x_np = x_i.reshape(1, -1)
    h_np = h.numpy().reshape(1, -1)
    L_np = L.numpy()
    Lh_np = Lh.numpy().reshape(-1, 1)  # (rank, 1) column
    Lh_sq_np = Lh_sq.numpy().reshape(-1, 1)  # (rank, 1) column
    D_T_np = D.T.numpy()
    scaled_D_np = scaled_D.numpy()
    bilinear_np = bilinear.numpy().reshape(1, -1)
    output_np = output.numpy().reshape(1, -1)

    # Create figure with grid layout
    fig = plt.figure(figsize=(14, 12))

    # Use gridspec for complex layout
    gs = fig.add_gridspec(8, 6, height_ratios=[1, 1, rank, rank, rank, 1, 1, 0.5],
                          width_ratios=[1, 1, 1, 1, 0.3, 0.8], hspace=0.4, wspace=0.3)

    def plot_heatmap(ax, data, title, vmin=None, vmax=None, mark_col=None):
        if vmin is None or vmax is None:
            vm = max(abs(data.min()), abs(data.max()))
            if vm == 0:
                vm = 1
            vmin, vmax = -vm, vm

        im = ax.imshow(data, cmap='RdBu_r', vmin=vmin, vmax=vmax, aspect='auto')
        ax.set_title(title, fontsize=9, loc='left')
        ax.set_xticks(range(data.shape[1]))
        if data.shape[0] == 1:
            ax.set_yticks([])
        else:
            ax.set_yticks(range(data.shape[0]))

        for i in range(data.shape[0]):
            for j in range(data.shape[1]):
                val = data[i, j]
                color = 'white' if abs(val) > (vmax - vmin) * 0.3 else 'black'
                ax.text(j, i, f'{val:.2f}', ha='center', va='center', fontsize=7, color=color)

        if mark_col is not None:
            for col, (color, style) in mark_col.items():
                if col < data.shape[1]:
                    rect = plt.Rectangle((col - 0.5, -0.5), 1, data.shape[0],
                                         fill=False, edgecolor=color, linewidth=2, linestyle=style)
                    ax.add_patch(rect)
        return im

    markers = {target: ('green', '-')}
    if full_pred != target:
        markers[full_pred] = ('red', '--')

    # Row 0: Input x
    ax0 = fig.add_subplot(gs[0, :4])
    plot_heatmap(ax0, x_np, 'Input x', mark_col=markers)

    # Row 1: h = γ * x / rms
    ax1 = fig.add_subplot(gs[1, :4])
    plot_heatmap(ax1, h_np, f'h = γ·(x/rms)  [γ={gamma:.3f}, rms={rms_val:.3f}]', mark_col=markers)

    # Row 2: L matrix
    ax2 = fig.add_subplot(gs[2, :4])
    plot_heatmap(ax2, L_np, f'L ({rank}×{n})')
    ax2.set_ylabel('rank')

    # Row 2 right: Lh values
    ax2r = fig.add_subplot(gs[2, 5])
    plot_heatmap(ax2r, Lh_np, 'Lh')

    # Row 3: D.T matrix
    ax3 = fig.add_subplot(gs[3, :4])
    plot_heatmap(ax3, D_T_np, f'D.T ({rank}×{n})')
    ax3.set_ylabel('rank')

    # Row 3 right: Lh² values (scaling factors)
    ax3r = fig.add_subplot(gs[3, 5])
    plot_heatmap(ax3r, Lh_sq_np, 'Lh²')

    # Row 4: Scaled D.T
    ax4 = fig.add_subplot(gs[4, :4])
    plot_heatmap(ax4, scaled_D_np, 'Scaled D.T = D.T * Lh² (row-wise)', mark_col=markers)
    ax4.set_ylabel('rank')

    # Row 5: Bilinear output
    ax5 = fig.add_subplot(gs[5, :4])
    plot_heatmap(ax5, bilinear_np, 'Bilinear = sum(Scaled D.T, axis=0)', mark_col=markers)

    # Row 6: Output
    ax6 = fig.add_subplot(gs[6, :4])
    plot_heatmap(ax6, output_np, 'Output = x + Bilinear', mark_col=markers)
    ax6.set_xlabel('Position')

    # Legend
    legend_text = f"Target: {target} (green box)"
    if full_pred != target:
        legend_text += f" | Pred: {full_pred} (red dashed)"

    fig.suptitle(f'{title}\n{legend_text}', fontsize=11)
    return fig

# %%
# =============================================================================
# EXAMPLES: Bilinear got RIGHT (sample 5)
# =============================================================================
print("\n" + "="*60)
print("EXAMPLES: Bilinear got RIGHT")
print("="*60)

torch.manual_seed(42)
x_sample = torch.randn(500, n)
comps_sample = compute_components(x_sample)
preds_sample = get_predictions(comps_sample, x_sample)

# Find where bilinear is correct
bilinear_right_idx = torch.where(preds_sample['bilinear_correct'])[0][:5]

for i, idx in enumerate(bilinear_right_idx):
    idx = idx.item()
    target = preds_sample['targets'][idx].item()
    bilinear_pred = preds_sample['bilinear_preds'][idx].item()
    full_pred = preds_sample['full_preds'][idx].item()

    full_status = "✓" if preds_sample['full_correct'][idx] else "✗"
    title = f"Bilinear CORRECT | Target={target}, Full pred={full_pred} ({full_status})"

    fig = plot_example_flow(
        x_sample[idx].numpy(),
        target, bilinear_pred, full_pred, title
    )
    plt.show()

# %%
# =============================================================================
# EXAMPLES: Bilinear WRONG but Full got RIGHT (sample 5)
# =============================================================================
print("\n" + "="*60)
print("EXAMPLES: Bilinear WRONG, Full RIGHT")
print("="*60)

# Find where bilinear wrong but full correct
bilinear_wrong_full_right_idx = torch.where(
    ~preds_sample['bilinear_correct'] & preds_sample['full_correct']
)[0][:5]

if len(bilinear_wrong_full_right_idx) == 0:
    print("No examples found where bilinear wrong but full correct.")
else:
    for i, idx in enumerate(bilinear_wrong_full_right_idx):
        idx = idx.item()
        target = preds_sample['targets'][idx].item()
        bilinear_pred = preds_sample['bilinear_preds'][idx].item()
        full_pred = preds_sample['full_preds'][idx].item()

        title = f"Bilinear WRONG, Full CORRECT | Target={target}"

        fig = plot_example_flow(
            x_sample[idx].numpy(),
            target, bilinear_pred, full_pred, title
        )
        plt.show()

# %%
# =============================================================================
# EXAMPLES: Both WRONG (sample 5)
# =============================================================================
print("\n" + "="*60)
print("EXAMPLES: Both WRONG")
print("="*60)

both_wrong_idx = torch.where(
    ~preds_sample['bilinear_correct'] & ~preds_sample['full_correct']
)[0][:5]

if len(both_wrong_idx) == 0:
    print("No examples found where both are wrong.")
else:
    for i, idx in enumerate(both_wrong_idx):
        idx = idx.item()
        target = preds_sample['targets'][idx].item()
        bilinear_pred = preds_sample['bilinear_preds'][idx].item()
        full_pred = preds_sample['full_preds'][idx].item()

        title = f"BOTH WRONG | Target={target}"

        fig = plot_example_flow(
            x_sample[idx].numpy(),
            target, bilinear_pred, full_pred, title
        )
        plt.show()

# %%
# =============================================================================
# SUMMARY
# =============================================================================
print("\n" + "="*60)
print("SUMMARY")
print("="*60)
print(f"""
Model: 1-Layer Symmetric Bilinear
  n={n}, rank={rank}, seed={seed}
  output = x + D @ (L @ norm(x))²

Accuracy:
  Full model: {full_acc:.1%}
  Bilinear only: {bilinear_acc:.1%}
  x only (baseline): {x_only_acc:.1%}
  Random: {100/n:.1f}%

The residual connection (adding x back) provides {(full_acc - bilinear_acc)*100:.1f}% gain.
""")

# %%
