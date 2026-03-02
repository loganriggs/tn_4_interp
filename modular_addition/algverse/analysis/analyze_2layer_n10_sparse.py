# %%
"""
Analysis of Sparse 2-Layer N=10 Model

Analyze the structure of the iteratively pruned 2-layer model:
- Weight sparsity patterns
- Quadratic form matrices
- Layer contributions
"""

import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import sys
import pickle

# =============================================================================
# PROJECT ROOT
# =============================================================================
PROJECT_ROOT = Path("/workspace/tn_4_interp/modular_addition/algverse")
sys.path.insert(0, str(PROJECT_ROOT))
from models import task_2nd_argmax
from analysis.analysis_utils import (
    compute_quadratic_forms, bilinear_forward, rmsnorm,
    bilinear_as_quadratic
)

checkpoint_dir = PROJECT_ROOT / "checkpoints"
images_dir = PROJECT_ROOT / "images"
images_dir.mkdir(exist_ok=True)

# %%
# =============================================================================
# LOAD SPARSE MODEL
# =============================================================================
checkpoint_path = checkpoint_dir / "2layer_n10_r10_seed8_sparse.pkl"
print(f"Loading sparse model from {checkpoint_path}")

with open(checkpoint_path, 'rb') as f:
    checkpoint = pickle.load(f)

state = checkpoint['state_dict']
config = checkpoint['config']
n = config['n']
rank = config['rank']
num_layers = config['num_layers']

print(f"\nConfig: n={n}, rank={rank}, num_layers={num_layers}")
print(f"Accuracy: {checkpoint['accuracy']*100:.1f}%")
print(f"Sparsity: {checkpoint['sparsity']*100:.1f}%")
print(f"Baseline: {checkpoint['baseline_acc']*100:.1f}%")
print(f"Iteration: {checkpoint['iteration']}")

# Extract weights
L1 = state['layers.0.L']
D1 = state['layers.0.D']
gamma1 = state['norms.0.weight']

L2 = state['layers.1.L']
D2 = state['layers.1.D']
gamma2 = state['norms.1.weight']

print(f"\nWeight shapes:")
print(f"  L1: {L1.shape}, D1: {D1.shape}, gamma1: {gamma1}")
print(f"  L2: {L2.shape}, D2: {D2.shape}, gamma2: {gamma2}")

# %%
# =============================================================================
# SPARSITY ANALYSIS
# =============================================================================
print("\n" + "=" * 60)
print("SPARSITY ANALYSIS")
print("=" * 60)

def analyze_sparsity(tensor, name):
    zeros = (tensor == 0).sum().item()
    total = tensor.numel()
    sparsity = zeros / total * 100
    nonzero_vals = tensor[tensor != 0]
    print(f"\n{name}:")
    print(f"  Shape: {tensor.shape}")
    print(f"  Sparsity: {sparsity:.1f}% ({zeros}/{total} zeros)")
    if len(nonzero_vals) > 0:
        print(f"  Non-zero values: min={nonzero_vals.min():.4f}, max={nonzero_vals.max():.4f}, mean={nonzero_vals.mean():.4f}")
    return sparsity

sp_L1 = analyze_sparsity(L1, "L1")
sp_D1 = analyze_sparsity(D1, "D1")
sp_L2 = analyze_sparsity(L2, "L2")
sp_D2 = analyze_sparsity(D2, "D2")

# %%
# =============================================================================
# VISUALIZE WEIGHT MATRICES
# =============================================================================
print("\n" + "=" * 60)
print("WEIGHT VISUALIZATIONS")
print("=" * 60)

fig, axes = plt.subplots(2, 2, figsize=(14, 12))

def plot_matrix(ax, mat, title, show_values=True):
    mat_np = mat.numpy()
    vmax = max(abs(mat_np.min()), abs(mat_np.max()))
    if vmax == 0:
        vmax = 1

    im = ax.imshow(mat_np, cmap='RdBu_r', vmin=-vmax, vmax=vmax, aspect='auto')
    ax.set_title(title, fontsize=12)
    plt.colorbar(im, ax=ax, shrink=0.8)

    # Show zeros in gray
    zeros_mask = (mat_np == 0)
    if zeros_mask.any():
        ax.imshow(zeros_mask, cmap='Greys', alpha=0.3, aspect='auto')

    if show_values and mat_np.size <= 100:
        for i in range(mat_np.shape[0]):
            for j in range(mat_np.shape[1]):
                val = mat_np[i, j]
                if val != 0:
                    color = 'white' if abs(val) > vmax * 0.5 else 'black'
                    ax.text(j, i, f'{val:.2f}', ha='center', va='center', fontsize=6, color=color)

plot_matrix(axes[0, 0], L1, f'L1 ({sp_L1:.0f}% sparse)')
plot_matrix(axes[0, 1], D1, f'D1 ({sp_D1:.0f}% sparse)')
plot_matrix(axes[1, 0], L2, f'L2 ({sp_L2:.0f}% sparse)')
plot_matrix(axes[1, 1], D2, f'D2 ({sp_D2:.0f}% sparse)')

plt.suptitle(f'Sparse 2-Layer n={n} Weights (Total: {checkpoint["sparsity"]*100:.1f}% sparse, Acc: {checkpoint["accuracy"]*100:.1f}%)', fontsize=14)
plt.tight_layout()
plt.savefig(images_dir / 'sparse_2layer_n10_weights.png', dpi=150)
plt.show()

# %%
# =============================================================================
# QUADRATIC FORM ANALYSIS
# =============================================================================
print("\n" + "=" * 60)
print("QUADRATIC FORM ANALYSIS")
print("=" * 60)

M1 = compute_quadratic_forms(L1, D1)  # (n, n, n)
M2 = compute_quadratic_forms(L2, D2)

print(f"\nM1 (layer 1 quadratic forms): {M1.shape}")
print(f"M2 (layer 2 quadratic forms): {M2.shape}")

# Analyze structure
for i in range(n):
    M1_i = M1[i]
    diag = M1_i.diag()
    offdiag = M1_i - torch.diag(diag)
    print(f"\nM1[{i}]:")
    print(f"  Diag: mean={diag.mean():.4f}, std={diag.std():.4f}")
    print(f"  Off-diag: mean={offdiag.mean():.4f}, std={offdiag.std():.4f}")
    print(f"  Diag[{i}] (self): {diag[i]:.4f}")

# %%
# =============================================================================
# VISUALIZE QUADRATIC FORMS
# =============================================================================
# Show first 5 quadratic form matrices for each layer
n_show = min(5, n)

fig, axes = plt.subplots(2, n_show, figsize=(4*n_show, 8))

for i in range(n_show):
    M1_i = M1[i].numpy()
    M2_i = M2[i].numpy()

    vmax1 = max(abs(M1_i.min()), abs(M1_i.max()))
    vmax2 = max(abs(M2_i.min()), abs(M2_i.max()))

    im1 = axes[0, i].imshow(M1_i, cmap='RdBu_r', vmin=-vmax1, vmax=vmax1)
    axes[0, i].set_title(f'M1[{i}]', fontsize=10)
    plt.colorbar(im1, ax=axes[0, i], shrink=0.7)

    im2 = axes[1, i].imshow(M2_i, cmap='RdBu_r', vmin=-vmax2, vmax=vmax2)
    axes[1, i].set_title(f'M2[{i}]', fontsize=10)
    plt.colorbar(im2, ax=axes[1, i], shrink=0.7)

axes[0, 0].set_ylabel('Layer 1', fontsize=12, fontweight='bold')
axes[1, 0].set_ylabel('Layer 2', fontsize=12, fontweight='bold')

plt.suptitle(f'Quadratic Form Matrices M^(i) for Sparse 2-Layer n={n}', fontsize=12)
plt.tight_layout()
plt.savefig(images_dir / 'sparse_2layer_n10_quadforms.png', dpi=150)
plt.show()

# %%
# =============================================================================
# FORWARD PASS ANALYSIS
# =============================================================================
print("\n" + "=" * 60)
print("FORWARD PASS ANALYSIS")
print("=" * 60)

def forward_2layer(x, L1, D1, gamma1, L2, D2, gamma2):
    """Run 2-layer symmetric bilinear forward pass."""
    # Layer 1
    h1, rms1 = rmsnorm(x, gamma1)
    r1, Lh1 = bilinear_forward(h1, L1, D1)

    # Layer 2
    h_mid = x + r1
    h2, rms2 = rmsnorm(h_mid, gamma2)
    r2, Lh2 = bilinear_forward(h2, L2, D2)

    output = x + r1 + r2

    return output, {
        'h1': h1, 'r1': r1, 'Lh1': Lh1,
        'h2': h2, 'r2': r2, 'Lh2': Lh2,
    }

# Evaluate
torch.manual_seed(123)
x_eval = torch.randn(10000, n)
targets = task_2nd_argmax(x_eval)

output, intermediates = forward_2layer(x_eval, L1, D1, gamma1, L2, D2, gamma2)

preds = output.argmax(dim=1)
accuracy = (preds == targets).float().mean().item()

print(f"\nModel accuracy: {accuracy:.1%}")

# Analyze layer contributions
r1 = intermediates['r1']
r2 = intermediates['r2']

r1_norm = r1.norm(dim=1).mean().item()
r2_norm = r2.norm(dim=1).mean().item()
x_norm = x_eval.norm(dim=1).mean().item()

print(f"\nNorm contributions (mean):")
print(f"  ||x||:  {x_norm:.4f}")
print(f"  ||r1||: {r1_norm:.4f}")
print(f"  ||r2||: {r2_norm:.4f}")

# Layer contributions to correct predictions
print(f"\nProgressive accuracy:")
print(f"  x only:      {(x_eval.argmax(dim=1) == targets).float().mean().item():.1%}")
print(f"  x + r1:      {((x_eval + r1).argmax(dim=1) == targets).float().mean().item():.1%}")
print(f"  x + r1 + r2: {accuracy:.1%}")

# Ablation (removing one layer)
print(f"\nAblation (removing one layer):")
print(f"  Without r1 (x + r2):      {((x_eval + r2).argmax(dim=1) == targets).float().mean().item():.1%}")
print(f"  Without r2 (x + r1):      {((x_eval + r1).argmax(dim=1) == targets).float().mean().item():.1%}")

# Single layer contributions
print(f"\nSingle layer contributions:")
print(f"  x + r1 only: {((x_eval + r1).argmax(dim=1) == targets).float().mean().item():.1%}")
print(f"  x + r2 only: {((x_eval + r2).argmax(dim=1) == targets).float().mean().item():.1%}")

# %%
# =============================================================================
# ACTIVE RANK ANALYSIS
# =============================================================================
print("\n" + "=" * 60)
print("ACTIVE RANK ANALYSIS")
print("=" * 60)

# Count active rows in L matrices (rows with at least one non-zero)
L1_active_rows = (L1.abs().sum(dim=1) > 0).sum().item()
L2_active_rows = (L2.abs().sum(dim=1) > 0).sum().item()

# Count active cols in D matrices
D1_active_cols = (D1.abs().sum(dim=0) > 0).sum().item()
D2_active_cols = (D2.abs().sum(dim=0) > 0).sum().item()

print(f"\nActive dimensions (effective rank):")
print(f"  L1: {L1_active_rows}/{rank} rows active")
print(f"  D1: {D1_active_cols}/{rank} cols active")
print(f"  L2: {L2_active_rows}/{rank} rows active")
print(f"  D2: {D2_active_cols}/{rank} cols active")

# Active rank = min of L rows and D cols for each layer
eff_rank1 = min(L1_active_rows, D1_active_cols)
eff_rank2 = min(L2_active_rows, D2_active_cols)
print(f"\nEffective rank per layer:")
print(f"  Layer 1: {eff_rank1}")
print(f"  Layer 2: {eff_rank2}")

# %%
# =============================================================================
# SUMMARY
# =============================================================================
print("\n" + "=" * 60)
print("SUMMARY")
print("=" * 60)

print(f"""
Sparse 2-Layer Model (n={n}, rank={rank})
=========================================

Accuracy: {accuracy:.1%} (baseline: {checkpoint['baseline_acc']*100:.1f}%)
Total Sparsity: {checkpoint['sparsity']*100:.1f}%

Per-layer sparsity:
  L1: {sp_L1:.1f}%
  D1: {sp_D1:.1f}%
  L2: {sp_L2:.1f}%  <-- Most sparse!
  D2: {sp_D2:.1f}%  <-- Most sparse!

Effective rank:
  Layer 1: {eff_rank1} (of {rank})
  Layer 2: {eff_rank2} (of {rank})

Layer contributions:
  x only:      ~10% (random guess)
  x + L1:      {((x_eval + r1).argmax(dim=1) == targets).float().mean().item():.1%}
  x + L1 + L2: {accuracy:.1%}

Key observations:
- Layer 2 (output layer) is pruned most aggressively
- Layer 1 (input layer) stays relatively dense
- The model maintains {accuracy:.1%} accuracy with {checkpoint['sparsity']*100:.1f}% of weights removed
""")

# %%
