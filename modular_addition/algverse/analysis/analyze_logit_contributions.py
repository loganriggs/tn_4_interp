# %%
"""
Logit Contribution Analysis for Seed 4 - PRUNED MODEL (t=0.15)

Decompose the output into:
1. Direct path: x → output (residual connection)
2. Layer 1 contribution: bilinear1(norm(x)) → output
3. Layer 2 contribution: bilinear2(norm(h1)) → output

Where h1 = x + bilinear1(norm(x))

Using the L1-pruned model with threshold=0.15 (19.8% weights pruned, 87.7% accuracy)
"""

import torch
import numpy as np
import matplotlib.pyplot as plt
import pickle
from pathlib import Path

# %%
# Load pruned seed 4 model (t=0.15)
ckpt_path = Path("prune_seed4_results.pkl")
with open(ckpt_path, 'rb') as f:
    data = pickle.load(f)

config = data['config']
results = data['results']

# Find t=0.15 result
prune_result = [r for r in results if r['threshold'] == 0.15][0]
state_dict = prune_result['state_dict']

print(f"Loaded PRUNED seed {config['seed']} model (threshold=0.15)")
print(f"Config: n={config['n']}, rank={config['rank']}, layers={config['num_layers']}")
print(f"Accuracy: {prune_result['final_acc']:.1%}, Pruned: {prune_result['pruned_frac']:.1%}")

# %%
# Extract weights
L1 = state_dict['layers.0.L']
R1 = state_dict['layers.0.R']
D1 = state_dict['layers.0.D']
norm1_weight = state_dict['norms.0.weight']

L2 = state_dict['layers.1.L']
R2 = state_dict['layers.1.R']
D2 = state_dict['layers.1.D']
norm2_weight = state_dict['norms.1.weight']

n = config['n']
rank = config['rank']

print(f"L1: {L1.shape}, R1: {R1.shape}, D1: {D1.shape}")
print(f"L2: {L2.shape}, R2: {R2.shape}, D2: {D2.shape}")

# %%
# =============================================================================
# WEIGHT VISUALIZATION
# =============================================================================
print("="*60)
print("WEIGHT MATRICES")
print("="*60)

def plot_matrix(ax, mat, title, cmap='RdBu_r'):
    """Plot matrix with 0 as white."""
    vmax = max(abs(mat.min()), abs(mat.max()))
    im = ax.imshow(mat, cmap=cmap, vmin=-vmax, vmax=vmax, aspect='auto')
    ax.set_title(title, fontsize=10)
    # Add text annotations
    for i in range(mat.shape[0]):
        for j in range(mat.shape[1]):
            val = mat[i, j]
            color = 'white' if abs(val) > vmax * 0.6 else 'black'
            ax.text(j, i, f'{val:.2f}', ha='center', va='center', fontsize=8, color=color)
    return im

fig, axes = plt.subplots(2, 4, figsize=(16, 8))

# Layer 1 weights
plot_matrix(axes[0, 0], L1.numpy(), f'L1 ({L1.shape[0]}x{L1.shape[1]})\nhidden x resid')
plot_matrix(axes[0, 1], R1.numpy(), f'R1 ({R1.shape[0]}x{R1.shape[1]})\nhidden x resid')
plot_matrix(axes[0, 2], D1.numpy(), f'D1 ({D1.shape[0]}x{D1.shape[1]})\nresid x hidden')
im = plot_matrix(axes[0, 3], norm1_weight.numpy().reshape(1, -1), f'Norm1 weight ({n})')
axes[0, 3].set_yticks([])

# Layer 2 weights
plot_matrix(axes[1, 0], L2.numpy(), f'L2 ({L2.shape[0]}x{L2.shape[1]})\nhidden x resid')
plot_matrix(axes[1, 1], R2.numpy(), f'R2 ({R2.shape[0]}x{R2.shape[1]})\nhidden x resid')
plot_matrix(axes[1, 2], D2.numpy(), f'D2 ({D2.shape[0]}x{D2.shape[1]})\nresid x hidden')
plot_matrix(axes[1, 3], norm2_weight.numpy().reshape(1, -1), f'Norm2 weight ({n})')
axes[1, 3].set_yticks([])

# Add row labels
axes[0, 0].set_ylabel('Layer 1', fontsize=12, fontweight='bold')
axes[1, 0].set_ylabel('Layer 2', fontsize=12, fontweight='bold')

plt.suptitle('All Weight Matrices (white = 0)', fontsize=14)
plt.tight_layout()
plt.show()

# %%
# =============================================================================
# BILINEAR LAYER COVARIANCE MATRICES (hidden x hidden)
# =============================================================================
print("="*60)
print("BILINEAR COVARIANCE MATRICES: L @ L.T * R @ R.T")
print("(hidden x hidden)")
print("="*60)

# Covariance: L @ L.T * R @ R.T (element-wise product)
# This is the "core" of the CP decomposition viewed as Tucker
cov1 = (L1 @ L1.T) * (R1 @ R1.T)  # (rank, rank) = (hidden, hidden)
cov2 = (L2 @ L2.T) * (R2 @ R2.T)  # (rank, rank) = (hidden, hidden)

fig, axes = plt.subplots(1, 3, figsize=(14, 4))

# Layer 1 covariance
ax1 = axes[0]
vmax1 = max(abs(cov1.min()), abs(cov1.max()))
im1 = ax1.imshow(cov1.numpy(), cmap='RdBu_r', vmin=-vmax1, vmax=vmax1)
ax1.set_title(f'Layer 1: L1@L1.T * R1@R1.T\n({rank}x{rank} hidden x hidden)', fontsize=11)
ax1.set_xlabel('hidden dim')
ax1.set_ylabel('hidden dim')
for i in range(rank):
    for j in range(rank):
        val = cov1[i, j].item()
        color = 'white' if abs(val) > vmax1 * 0.6 else 'black'
        ax1.text(j, i, f'{val:.2f}', ha='center', va='center', fontsize=9, color=color)
plt.colorbar(im1, ax=ax1, shrink=0.8)

# Layer 2 covariance
ax2 = axes[1]
vmax2 = max(abs(cov2.min()), abs(cov2.max()))
im2 = ax2.imshow(cov2.numpy(), cmap='RdBu_r', vmin=-vmax2, vmax=vmax2)
ax2.set_title(f'Layer 2: L2@L2.T * R2@R2.T\n({rank}x{rank} hidden x hidden)', fontsize=11)
ax2.set_xlabel('hidden dim')
ax2.set_ylabel('hidden dim')
for i in range(rank):
    for j in range(rank):
        val = cov2[i, j].item()
        color = 'white' if abs(val) > vmax2 * 0.6 else 'black'
        ax2.text(j, i, f'{val:.2f}', ha='center', va='center', fontsize=9, color=color)
plt.colorbar(im2, ax=ax2, shrink=0.8)

# Eigenvalues comparison
ax3 = axes[2]
eig1 = torch.linalg.eigvalsh(cov1).numpy()[::-1]
eig2 = torch.linalg.eigvalsh(cov2).numpy()[::-1]
x_eig = np.arange(rank)
width = 0.35
ax3.bar(x_eig - width/2, eig1, width, label='Layer 1', color='steelblue', edgecolor='black')
ax3.bar(x_eig + width/2, eig2, width, label='Layer 2', color='coral', edgecolor='black')
ax3.set_xlabel('Eigenvalue index')
ax3.set_ylabel('Eigenvalue')
ax3.set_title('Eigenvalues of Covariance Matrices', fontsize=11)
ax3.set_xticks(x_eig)
ax3.legend()
ax3.grid(axis='y', alpha=0.3)

plt.tight_layout()
plt.show()

print(f"\nLayer 1 covariance eigenvalues: {eig1.round(3)}")
print(f"Layer 2 covariance eigenvalues: {eig2.round(3)}")
print(f"Layer 1 effective rank (entropy): {np.exp(-(eig1/eig1.sum() * np.log(eig1/eig1.sum() + 1e-10)).sum()):.2f}")
print(f"Layer 2 effective rank (entropy): {np.exp(-(eig2/eig2.sum() * np.log(eig2/eig2.sum() + 1e-10)).sum()):.2f}")

# %%
# =============================================================================
# CROSS-LAYER COMPOSITION: D1.T @ L2.T ... @ D1 -> (hid1_a, hid2, hid1_b)
# =============================================================================
print("="*60)
print("CROSS-LAYER COMPOSED TENSOR")
print("D1.T, L2.T, R2.T, D1 -> (hid1_a, hid2, hid1_b)")
print("="*60)

# Einsum: contract resid1_a and resid1_b, keep hid1_a, hid2, hid1_b
# D1.T: (hid1, resid) = (rank, n)
# L2.T: (resid, hid2) = (n, rank)
# R2.T: (resid, hid2) = (n, rank)
# D1:   (resid, hid1) = (n, rank)
#
# 'ar,rh,sh,sb->ahb' where a=hid1_a, r=resid1_a, h=hid2, s=resid1_b, b=hid1_b
composed = torch.einsum('ar,rh,sh,sb->ahb', D1.T, L2.T, R2.T, D1)
print(f"Composed tensor shape: {composed.shape} = (hid1_a, hid2, hid1_b)")

# Visualize as slices along hid2
fig, axes = plt.subplots(1, rank + 1, figsize=(4 * (rank + 1), 4))

vmax = composed.abs().max().item()

for h2 in range(rank):
    ax = axes[h2]
    slice_h2 = composed[:, h2, :].numpy()  # (hid1_a, hid1_b)
    im = ax.imshow(slice_h2, cmap='RdBu_r', vmin=-vmax, vmax=vmax)
    ax.set_title(f'hid2 = {h2}', fontsize=11)
    ax.set_xlabel('hid1_b')
    ax.set_ylabel('hid1_a')
    for i in range(rank):
        for j in range(rank):
            val = slice_h2[i, j]
            color = 'white' if abs(val) > vmax * 0.6 else 'black'
            ax.text(j, i, f'{val:.2f}', ha='center', va='center', fontsize=8, color=color)

# Summary: Frobenius norm per slice
ax_sum = axes[rank]
slice_norms = [composed[:, h2, :].norm().item() for h2 in range(rank)]
ax_sum.bar(range(rank), slice_norms, color='purple', edgecolor='black')
ax_sum.set_xlabel('hid2 index')
ax_sum.set_ylabel('Frobenius norm')
ax_sum.set_title('Norm per hid2 slice', fontsize=11)
ax_sum.set_xticks(range(rank))

plt.suptitle('Composed Tensor: D1.T @ L2.T @ R2.T @ D1\nSlices along hid2 dimension', fontsize=12)
plt.tight_layout()
plt.show()

# Also show the full tensor flattened as a 2D view: (hid1_a, hid2*hid1_b)
print(f"\nComposed tensor statistics:")
print(f"  Shape: {tuple(composed.shape)}")
print(f"  Frobenius norm: {composed.norm().item():.3f}")
print(f"  Max abs value: {composed.abs().max().item():.3f}")
print(f"  Slice norms (per hid2): {[f'{n:.2f}' for n in slice_norms]}")

# SVD analysis of the mode-1 unfolding (hid1_a vs (hid2, hid1_b))
unfolded = composed.reshape(rank, -1)  # (hid1_a, hid2*hid1_b)
svd_vals = torch.linalg.svdvals(unfolded).numpy()
print(f"  Mode-1 unfolding SVD: {svd_vals.round(3)}")
print(f"  Mode-1 effective rank: {np.exp(-(svd_vals/svd_vals.sum() * np.log(svd_vals/svd_vals.sum() + 1e-10)).sum()):.2f}")

# %%
def rmsnorm(x, weight, eps=1e-6):
    """Apply RMSNorm."""
    rms = torch.sqrt((x ** 2).mean(dim=-1, keepdim=True) + eps)
    return weight * (x / rms)

def bilinear(x, L, R, D):
    """Apply bilinear layer: D @ [(L @ x) * (R @ x)]"""
    Lx = x @ L.T  # (batch, rank)
    Rx = x @ R.T  # (batch, rank)
    return (Lx * Rx) @ D.T  # (batch, n)

def task_2nd_argmax(x):
    """Find position of 2nd largest element."""
    return x.argsort(-1)[..., -2]

def forward_with_contributions(x):
    """
    Forward pass returning all contributions.

    Returns:
        logits: final output
        contributions: dict with
            - 'direct': contribution from residual (just x)
            - 'layer1': contribution from bilinear1 only
            - 'layer2': contribution from bilinear2 only
    """
    # Layer 1
    x_norm1 = rmsnorm(x, norm1_weight)
    layer1_out = bilinear(x_norm1, L1, R1, D1)
    h1 = x + layer1_out  # residual

    # Layer 2
    h1_norm = rmsnorm(h1, norm2_weight)
    layer2_out = bilinear(h1_norm, L2, R2, D2)
    h2 = h1 + layer2_out  # residual

    # Final logits = x + layer1_out + layer2_out
    # (since h2 = h1 + layer2_out = x + layer1_out + layer2_out)

    return h2, {
        'direct': x.clone(),
        'layer1': layer1_out,
        'layer2': layer2_out,
        'h1': h1,  # intermediate hidden state
    }

# %%
def analyze_example(x, idx=0):
    """
    Analyze a single example and visualize contributions with grouped bar charts.

    Args:
        x: input tensor (batch, n) or (n,)
        idx: which example in batch to analyze
    """
    if x.dim() == 1:
        x = x.unsqueeze(0)
        idx = 0

    logits, contributions = forward_with_contributions(x)

    # Get this example
    x_i = x[idx]
    logits_i = logits[idx]
    direct_i = contributions['direct'][idx]
    layer1_i = contributions['layer1'][idx]
    layer2_i = contributions['layer2'][idx]

    # Ground truth
    target = task_2nd_argmax(x_i.unsqueeze(0)).item()
    max_pos = x_i.argmax().item()
    pred = logits_i.argmax().item()
    correct = pred == target

    # Create figure with 2 subplots
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    positions = np.arange(n)
    width = 0.2

    # ===== Left plot: Input values =====
    ax1 = axes[0]
    colors_input = []
    for i in range(n):
        if i == max_pos:
            colors_input.append('red')
        elif i == target:
            colors_input.append('orange')
        else:
            colors_input.append('steelblue')

    bars = ax1.bar(positions, x_i.numpy(), width=0.6, color=colors_input, edgecolor='black', linewidth=1.5)
    ax1.set_xlabel('Position', fontsize=12)
    ax1.set_ylabel('Input Value', fontsize=12)
    ax1.set_title(f'Input x\nMax (red) = pos {max_pos}, 2nd Max (orange) = pos {target}', fontsize=11)
    ax1.set_xticks(positions)
    ax1.set_xticklabels([f'{i}' for i in positions], fontsize=11)
    ax1.axhline(y=0, color='k', linestyle='-', linewidth=0.5)
    ax1.grid(axis='y', alpha=0.3)

    # ===== Right plot: Grouped bar chart of contributions =====
    ax2 = axes[1]

    # Grouped bars: Direct, Layer1, Layer2, Final
    x_pos = np.arange(n)

    bars_direct = ax2.bar(x_pos - 1.5*width, direct_i.numpy(), width,
                          label='Direct (x)', color='gray', edgecolor='black')
    bars_l1 = ax2.bar(x_pos - 0.5*width, layer1_i.numpy(), width,
                      label='Layer 1', color='lightgreen', edgecolor='black')
    bars_l2 = ax2.bar(x_pos + 0.5*width, layer2_i.numpy(), width,
                      label='Layer 2', color='plum', edgecolor='black')
    bars_final = ax2.bar(x_pos + 1.5*width, logits_i.numpy(), width,
                         label='Final Logit', color='gold', edgecolor='black')

    # Highlight target position
    ax2.axvspan(target - 0.4, target + 0.4, alpha=0.2, color='green', label=f'Target (pos {target})')

    ax2.set_xlabel('Position', fontsize=12)
    ax2.set_ylabel('Logit Contribution', fontsize=12)
    correct_str = '✓ CORRECT' if correct else '✗ WRONG'
    ax2.set_title(f'Logit Contributions by Layer\nPrediction: pos {pred} {correct_str}', fontsize=11)
    ax2.set_xticks(x_pos)
    ax2.set_xticklabels([f'{i}' for i in positions], fontsize=11)
    ax2.axhline(y=0, color='k', linestyle='-', linewidth=0.5)
    ax2.legend(loc='best', fontsize=9)
    ax2.grid(axis='y', alpha=0.3)

    plt.tight_layout()
    plt.show()

    # Print summary
    print(f"\n{'='*60}")
    print(f"Input x:        {x_i.numpy().round(3)}")
    print(f"Sorted indices: {x_i.argsort(descending=True).numpy()} (max to min)")
    print(f"Target (2nd max): position {target}")
    print(f"{'='*60}")
    print(f"Direct (x):     {direct_i.numpy().round(3)}")
    print(f"Layer 1:        {layer1_i.numpy().round(3)}")
    print(f"Layer 2:        {layer2_i.numpy().round(3)}")
    print(f"{'='*60}")
    print(f"Final logits:   {logits_i.numpy().round(3)}")
    print(f"Prediction:     position {pred} {'✓ CORRECT' if pred == target else '✗ WRONG'}")
    print(f"{'='*60}")

    return logits_i, contributions

# %%
# Generate random examples and analyze
print("Analyzing random examples...")
print("Re-run this cell for new examples!\n")

import time
torch.manual_seed(int(time.time() * 1000) % (2**32))  # Random seed each time
x_batch = torch.randn(5, n)

for i in range(5):
    print(f"\n{'#'*60}")
    print(f"EXAMPLE {i+1}")
    print(f"{'#'*60}")
    analyze_example(x_batch, idx=i)

# %%
# Analyze a specific interesting case: when values are close
print("\n" + "="*60)
print("EDGE CASE: Close values (harder to distinguish)")
print("="*60)

# Create input where top 2 values are close
x_close = torch.randn(n)
x_close[1] = 2.0  # max
x_close[3] = 1.9  # 2nd max (close!)
x_close[0] = 0.5
x_close[2] = -0.3

analyze_example(x_close)

# %%
# Analyze contribution magnitudes across many examples
print("\n" + "="*60)
print("CONTRIBUTION MAGNITUDE ANALYSIS (1000 examples)")
print("="*60)

x_many = torch.randn(1000, n)
logits_many, contribs_many = forward_with_contributions(x_many)

direct_norm = contribs_many['direct'].norm(dim=1).mean()
layer1_norm = contribs_many['layer1'].norm(dim=1).mean()
layer2_norm = contribs_many['layer2'].norm(dim=1).mean()

print(f"Average L2 norm of contributions:")
print(f"  Direct (x):  {direct_norm:.3f}")
print(f"  Layer 1:     {layer1_norm:.3f}")
print(f"  Layer 2:     {layer2_norm:.3f}")

# Accuracy
targets_many = task_2nd_argmax(x_many)
preds_many = logits_many.argmax(dim=1)
accuracy = (preds_many == targets_many).float().mean()
print(f"\nAccuracy: {accuracy:.1%}")

# Which layer contributes most to correct predictions?
correct_mask = (preds_many == targets_many)
print(f"\nFor CORRECT predictions ({correct_mask.sum()} examples):")
print(f"  Layer 1 norm: {contribs_many['layer1'][correct_mask].norm(dim=1).mean():.3f}")
print(f"  Layer 2 norm: {contribs_many['layer2'][correct_mask].norm(dim=1).mean():.3f}")

wrong_mask = ~correct_mask
if wrong_mask.sum() > 0:
    print(f"\nFor WRONG predictions ({wrong_mask.sum()} examples):")
    print(f"  Layer 1 norm: {contribs_many['layer1'][wrong_mask].norm(dim=1).mean():.3f}")
    print(f"  Layer 2 norm: {contribs_many['layer2'][wrong_mask].norm(dim=1).mean():.3f}")

# %%
# =============================================================================
# OPTIMAL INPUTS FOR LAYER 1 CHANNELS
# =============================================================================
print("\n" + "="*60)
print("OPTIMAL INPUTS FOR LAYER 1 CHANNELS")
print("="*60)
print("""
For bilinear layer: activation_h = (L[h] · x) * (R[h] · x)

Since L[h] and R[h] only interact with each other:
- Positive max: x* ∝ L[h] + R[h]  (both dot products positive)
- Negative max: x* ∝ L[h] - R[h]  (opposite signs)
""")

def compute_optimal_inputs(L, R):
    """
    Compute optimal inputs for each channel of a bilinear layer.

    For channel h: activation = (L[h] · x) * (R[h] · x)

    Returns:
        x_pos: (rank, n) - inputs that maximize positive activation
        x_neg: (rank, n) - inputs that maximize negative activation
        info: dict with L vectors, R vectors, angles, etc.
    """
    rank, n = L.shape

    x_pos = []
    x_neg = []
    info = {'l_vecs': [], 'r_vecs': [], 'angles': [], 'l_dot_r': []}

    for h in range(rank):
        l_h = L[h]  # (n,)
        r_h = R[h]  # (n,)

        # Optimal inputs (unnormalized)
        x_pos_h = l_h + r_h
        x_neg_h = l_h - r_h

        # Normalize
        x_pos_h = x_pos_h / (x_pos_h.norm() + 1e-8)
        x_neg_h = x_neg_h / (x_neg_h.norm() + 1e-8)

        x_pos.append(x_pos_h)
        x_neg.append(x_neg_h)

        # Info
        info['l_vecs'].append(l_h)
        info['r_vecs'].append(r_h)
        l_dot_r = (l_h @ r_h).item()
        info['l_dot_r'].append(l_dot_r)
        # Angle between l and r
        cos_angle = l_dot_r / (l_h.norm() * r_h.norm() + 1e-8)
        angle_deg = np.arccos(np.clip(cos_angle.item(), -1, 1)) * 180 / np.pi
        info['angles'].append(angle_deg)

    return torch.stack(x_pos), torch.stack(x_neg), info

# Compute for Layer 1
x_pos_L1, x_neg_L1, info_L1 = compute_optimal_inputs(L1, R1)

print("Layer 1 channel analysis:")
print(f"{'Ch':<4} {'L·R':<8} {'Angle(L,R)':<12} {'||L||':<8} {'||R||':<8}")
print("-" * 45)
for h in range(rank):
    l_norm = info_L1['l_vecs'][h].norm().item()
    r_norm = info_L1['r_vecs'][h].norm().item()
    print(f"{h:<4} {info_L1['l_dot_r'][h]:<8.3f} {info_L1['angles'][h]:<12.1f}° {l_norm:<8.3f} {r_norm:<8.3f}")

# %%
# Visualize optimal inputs for Layer 1
fig, axes = plt.subplots(2, rank, figsize=(4*rank, 8))

for h in range(rank):
    # Positive optimal input
    ax_pos = axes[0, h]
    x_opt = x_pos_L1[h].numpy()
    colors = ['green' if v > 0 else 'red' for v in x_opt]
    ax_pos.bar(range(n), x_opt, color=colors, edgecolor='black')
    ax_pos.set_title(f'Ch {h}: x* = L+R (pos)', fontsize=10)
    ax_pos.set_xlabel('Position')
    ax_pos.set_ylabel('Value')
    ax_pos.axhline(y=0, color='k', linewidth=0.5)
    ax_pos.set_ylim(-1, 1)

    # Negative optimal input
    ax_neg = axes[1, h]
    x_opt = x_neg_L1[h].numpy()
    colors = ['green' if v > 0 else 'red' for v in x_opt]
    ax_neg.bar(range(n), x_opt, color=colors, edgecolor='black')
    ax_neg.set_title(f'Ch {h}: x* = L-R (neg)', fontsize=10)
    ax_neg.set_xlabel('Position')
    ax_neg.set_ylabel('Value')
    ax_neg.axhline(y=0, color='k', linewidth=0.5)
    ax_neg.set_ylim(-1, 1)

plt.suptitle('Layer 1: Optimal Inputs per Channel\n(Top: maximize positive activation, Bottom: maximize negative)', fontsize=12)
plt.tight_layout()
plt.show()

# %%
# Test: what activations do these optimal inputs produce?
print("\n" + "="*60)
print("ACTIVATIONS FROM OPTIMAL INPUTS")
print("="*60)

def get_layer1_activations(x):
    """Get pre-D activations: (L @ x) * (R @ x)"""
    x_norm = rmsnorm(x, norm1_weight)
    Lx = x_norm @ L1.T  # (batch, rank)
    Rx = x_norm @ R1.T  # (batch, rank)
    return Lx * Rx  # (batch, rank)

# Test positive optimal inputs
print("\nPositive optimal inputs (x* = L+R, normalized):")
acts_pos = get_layer1_activations(x_pos_L1)
print(f"{'Input for Ch':<14} {'Act[0]':<10} {'Act[1]':<10} {'Act[2]':<10} {'Act[3]':<10}")
print("-" * 55)
for h in range(rank):
    acts = acts_pos[h].numpy()
    print(f"Ch {h:<11} {acts[0]:<10.3f} {acts[1]:<10.3f} {acts[2]:<10.3f} {acts[3]:<10.3f}")

print("\nNegative optimal inputs (x* = L-R, normalized):")
acts_neg = get_layer1_activations(x_neg_L1)
print(f"{'Input for Ch':<14} {'Act[0]':<10} {'Act[1]':<10} {'Act[2]':<10} {'Act[3]':<10}")
print("-" * 55)
for h in range(rank):
    acts = acts_neg[h].numpy()
    print(f"Ch {h:<11} {acts[0]:<10.3f} {acts[1]:<10.3f} {acts[2]:<10.3f} {acts[3]:<10.3f}")

# %%
# What do these optimal inputs predict?
print("\n" + "="*60)
print("PREDICTIONS FROM OPTIMAL INPUTS")
print("="*60)

print("\nPositive optimal inputs:")
for h in range(rank):
    x_opt = x_pos_L1[h:h+1]
    logits, _ = forward_with_contributions(x_opt)
    pred = logits.argmax().item()
    target = task_2nd_argmax(x_opt).item()
    logits_str = ', '.join([f'{v:.2f}' for v in logits[0].numpy()])
    print(f"  Ch {h} input: pred={pred}, target={target}, logits=[{logits_str}]")

print("\nNegative optimal inputs:")
for h in range(rank):
    x_opt = x_neg_L1[h:h+1]
    logits, _ = forward_with_contributions(x_opt)
    pred = logits.argmax().item()
    target = task_2nd_argmax(x_opt).item()
    logits_str = ', '.join([f'{v:.2f}' for v in logits[0].numpy()])
    print(f"  Ch {h} input: pred={pred}, target={target}, logits=[{logits_str}]")

# %%
# Visualize L and R vectors for Layer 1
print("\n" + "="*60)
print("L AND R VECTORS FOR LAYER 1")
print("="*60)

fig, axes = plt.subplots(2, rank, figsize=(4*rank, 6))

for h in range(rank):
    l_h = L1[h].numpy()
    r_h = R1[h].numpy()

    ax_l = axes[0, h]
    colors = ['steelblue' if v >= 0 else 'coral' for v in l_h]
    ax_l.bar(range(n), l_h, color=colors, edgecolor='black')
    ax_l.set_title(f'L1[{h}]', fontsize=11)
    ax_l.set_xlabel('Position')
    ax_l.axhline(y=0, color='k', linewidth=0.5)
    ax_l.set_ylim(-1, 1)

    ax_r = axes[1, h]
    colors = ['steelblue' if v >= 0 else 'coral' for v in r_h]
    ax_r.bar(range(n), r_h, color=colors, edgecolor='black')
    ax_r.set_title(f'R1[{h}]', fontsize=11)
    ax_r.set_xlabel('Position')
    ax_r.axhline(y=0, color='k', linewidth=0.5)
    ax_r.set_ylim(-1, 1)

plt.suptitle('Layer 1: L and R vectors per channel', fontsize=12)
plt.tight_layout()
plt.show()

# %%
# =============================================================================
# JOINT OPTIMIZATION: 8x8 GRID OF CHANNEL COMBINATIONS
# =============================================================================
print("\n" + "="*60)
print("JOINT OPTIMIZATION: ALL 8x8 CHANNEL COMBINATIONS")
print("="*60)
print("""
8 directions: 4 positive (+ch0..+ch3) and 4 opposite (-ch0..-ch3)
Joint optimal input for directions i,j: x* = normalize(d_i + d_j)
When combining +ch_k with -ch_k, they cancel to 0.
""")

# Compute 8 directions: 4 positive + 4 negative (opposite)
directions = []
dir_labels = []

# Positive directions (L + R, normalized)
for h in range(rank):
    d = L1[h] + R1[h]
    d = d / (d.norm() + 1e-8)
    directions.append(d)
    dir_labels.append(f'+ch{h}')

# Negative (opposite) directions
for h in range(rank):
    directions.append(-directions[h])
    dir_labels.append(f'-ch{h}')

directions = torch.stack(directions)  # (8, n)

print("8 directions:")
for i, label in enumerate(dir_labels):
    print(f"  {label}: {directions[i].numpy().round(3)}")

# %%
# Compute all 8x8 joint optimal inputs
joint_inputs = torch.zeros(8, 8, n)
is_zero = torch.zeros(8, 8, dtype=torch.bool)  # Track which ones cancel

for i in range(8):
    for j in range(8):
        d_sum = directions[i] + directions[j]
        norm = d_sum.norm()

        if norm < 1e-6:  # Cancellation (e.g., +ch0 + -ch0)
            is_zero[i, j] = True
            joint_inputs[i, j] = torch.zeros(n)
        else:
            joint_inputs[i, j] = d_sum / norm

# %%
# Plot 8x8 grid of bar charts
fig, axes = plt.subplots(8, 8, figsize=(20, 20))

for i in range(8):
    for j in range(8):
        ax = axes[i, j]
        x_opt = joint_inputs[i, j].numpy()

        if is_zero[i, j]:
            # Cancellation - show empty/gray
            ax.set_facecolor('#f0f0f0')
            ax.text(0.5, 0.5, '0', ha='center', va='center', fontsize=12,
                    transform=ax.transAxes, color='gray')
            ax.set_xticks([])
            ax.set_yticks([])
        else:
            # Show bar chart
            colors = ['green' if v > 0.1 else 'red' if v < -0.1 else 'gray' for v in x_opt]
            ax.bar(range(n), x_opt, color=colors, edgecolor='black', linewidth=0.5)
            ax.axhline(y=0, color='k', linewidth=0.3)
            ax.set_ylim(-1, 1)
            ax.set_xticks([])
            ax.set_yticks([])

        # Labels on edges
        if i == 0:
            ax.set_title(dir_labels[j], fontsize=9)
        if j == 0:
            ax.set_ylabel(dir_labels[i], fontsize=9, rotation=0, ha='right', va='center')

plt.suptitle('Joint Optimal Inputs: x* = normalize(d_i + d_j)\n(Rows = direction i, Cols = direction j, Gray = cancellation)',
             fontsize=14)
plt.tight_layout()
plt.show()

# %%
# Compute activations for all joint inputs
print("\n" + "="*60)
print("ACTIVATIONS FROM JOINT OPTIMAL INPUTS")
print("="*60)

# Get layer 1 activations for all 64 joint inputs
joint_inputs_flat = joint_inputs.reshape(64, n)
acts_joint = get_layer1_activations(joint_inputs_flat).reshape(8, 8, rank)

# Show as heatmaps - one per channel
fig, axes = plt.subplots(1, rank + 1, figsize=(4*(rank+1), 4))

for ch in range(rank):
    ax = axes[ch]
    act_ch = acts_joint[:, :, ch].numpy()

    # Mask out zero inputs
    act_ch_masked = np.where(is_zero.numpy(), np.nan, act_ch)

    vmax = np.nanmax(np.abs(act_ch_masked))
    im = ax.imshow(act_ch_masked, cmap='RdBu_r', vmin=-vmax, vmax=vmax)
    ax.set_title(f'Channel {ch} activation', fontsize=11)
    ax.set_xticks(range(8))
    ax.set_xticklabels(dir_labels, fontsize=8, rotation=45, ha='right')
    ax.set_yticks(range(8))
    ax.set_yticklabels(dir_labels, fontsize=8)
    plt.colorbar(im, ax=ax, shrink=0.6)

# Sum of all channel activations (total hidden activation)
ax_sum = axes[rank]
act_sum = acts_joint.sum(dim=-1).numpy()
act_sum_masked = np.where(is_zero.numpy(), np.nan, act_sum)
vmax = np.nanmax(np.abs(act_sum_masked))
im = ax_sum.imshow(act_sum_masked, cmap='RdBu_r', vmin=-vmax, vmax=vmax)
ax_sum.set_title('Sum of all channels', fontsize=11)
ax_sum.set_xticks(range(8))
ax_sum.set_xticklabels(dir_labels, fontsize=8, rotation=45, ha='right')
ax_sum.set_yticks(range(8))
ax_sum.set_yticklabels(dir_labels, fontsize=8)
plt.colorbar(im, ax=ax_sum, shrink=0.6)

plt.suptitle('Layer 1 Activations for Joint Optimal Inputs', fontsize=12)
plt.tight_layout()
plt.show()

# %%
# What does the model predict for each joint input?
print("\n" + "="*60)
print("MODEL PREDICTIONS FOR JOINT OPTIMAL INPUTS")
print("="*60)

predictions = torch.zeros(8, 8, dtype=torch.long)
targets = torch.zeros(8, 8, dtype=torch.long)
correct = torch.zeros(8, 8, dtype=torch.bool)

for i in range(8):
    for j in range(8):
        if is_zero[i, j]:
            predictions[i, j] = -1  # Invalid
            targets[i, j] = -1
            correct[i, j] = False
        else:
            x_opt = joint_inputs[i, j:j+1]
            logits, _ = forward_with_contributions(x_opt)
            pred = logits.argmax().item()
            target = task_2nd_argmax(x_opt).item()
            predictions[i, j] = pred
            targets[i, j] = target
            correct[i, j] = (pred == target)

# Plot predictions
fig, axes = plt.subplots(1, 3, figsize=(15, 5))

# Predictions
ax1 = axes[0]
pred_display = predictions.numpy().astype(float)
pred_display[is_zero.numpy()] = np.nan
im1 = ax1.imshow(pred_display, cmap='tab10', vmin=0, vmax=3)
ax1.set_title('Predicted Position', fontsize=11)
ax1.set_xticks(range(8))
ax1.set_xticklabels(dir_labels, fontsize=8, rotation=45, ha='right')
ax1.set_yticks(range(8))
ax1.set_yticklabels(dir_labels, fontsize=8)
for i in range(8):
    for j in range(8):
        if not is_zero[i, j]:
            ax1.text(j, i, str(predictions[i,j].item()), ha='center', va='center', fontsize=9, color='white')
plt.colorbar(im1, ax=ax1, shrink=0.6)

# Targets
ax2 = axes[1]
tgt_display = targets.numpy().astype(float)
tgt_display[is_zero.numpy()] = np.nan
im2 = ax2.imshow(tgt_display, cmap='tab10', vmin=0, vmax=3)
ax2.set_title('Target Position (2nd max)', fontsize=11)
ax2.set_xticks(range(8))
ax2.set_xticklabels(dir_labels, fontsize=8, rotation=45, ha='right')
ax2.set_yticks(range(8))
ax2.set_yticklabels(dir_labels, fontsize=8)
for i in range(8):
    for j in range(8):
        if not is_zero[i, j]:
            ax2.text(j, i, str(targets[i,j].item()), ha='center', va='center', fontsize=9, color='white')
plt.colorbar(im2, ax=ax2, shrink=0.6)

# Correct/Wrong
ax3 = axes[2]
correct_display = correct.numpy().astype(float)
correct_display[is_zero.numpy()] = 0.5  # Gray for invalid
im3 = ax3.imshow(correct_display, cmap='RdYlGn', vmin=0, vmax=1)
ax3.set_title('Correct (green) / Wrong (red)', fontsize=11)
ax3.set_xticks(range(8))
ax3.set_xticklabels(dir_labels, fontsize=8, rotation=45, ha='right')
ax3.set_yticks(range(8))
ax3.set_yticklabels(dir_labels, fontsize=8)

n_valid = (~is_zero).sum().item()
n_correct = correct[~is_zero].sum().item()
ax3.set_xlabel(f'{n_correct}/{n_valid} correct ({100*n_correct/n_valid:.0f}%)')

plt.suptitle('Model Predictions for Joint Optimal Inputs', fontsize=12)
plt.tight_layout()
plt.show()

# %%
# =============================================================================
# LAYER 2 ANALYSIS: CONNECTING L1 CHANNELS TO L2 CHANNELS
# =============================================================================
print("\n" + "="*60)
print("LAYER 2: OPTIMAL INPUTS IN h1 SPACE")
print("="*60)
print("""
Layer 2 sees h1 (the intermediate representation).
For L2 channel h2: activation = (L2[h2]·h1)(R2[h2]·h1)
Optimal h1 for channel h2: h1* ∝ L2[h2] + R2[h2]
""")

# Compute optimal h1 for each Layer 2 channel
h1_opt_pos = []
h1_opt_neg = []

for h2 in range(rank):
    l2_h = L2[h2]
    r2_h = R2[h2]

    h1_pos = l2_h + r2_h
    h1_pos = h1_pos / (h1_pos.norm() + 1e-8)

    h1_neg = l2_h - r2_h
    h1_neg = h1_neg / (h1_neg.norm() + 1e-8)

    h1_opt_pos.append(h1_pos)
    h1_opt_neg.append(h1_neg)

h1_opt_pos = torch.stack(h1_opt_pos)
h1_opt_neg = torch.stack(h1_opt_neg)

# Visualize
fig, axes = plt.subplots(2, rank, figsize=(4*rank, 6))

for h2 in range(rank):
    ax_pos = axes[0, h2]
    h1 = h1_opt_pos[h2].numpy()
    colors = ['green' if v > 0 else 'red' for v in h1]
    ax_pos.bar(range(n), h1, color=colors, edgecolor='black')
    ax_pos.set_title(f'L2 ch{h2}: h1* = L2+R2', fontsize=10)
    ax_pos.axhline(y=0, color='k', linewidth=0.5)
    ax_pos.set_ylim(-1, 1)

    ax_neg = axes[1, h2]
    h1 = h1_opt_neg[h2].numpy()
    colors = ['green' if v > 0 else 'red' for v in h1]
    ax_neg.bar(range(n), h1, color=colors, edgecolor='black')
    ax_neg.set_title(f'L2 ch{h2}: h1* = L2-R2', fontsize=10)
    ax_neg.axhline(y=0, color='k', linewidth=0.5)
    ax_neg.set_ylim(-1, 1)

plt.suptitle('Layer 2: Optimal h1 (intermediate representation) per channel', fontsize=12)
plt.tight_layout()
plt.show()

# %%
# =============================================================================
# COMPOSED TENSOR: How L1 channel pairs contribute to L2 channels
# =============================================================================
print("\n" + "="*60)
print("COMPOSED TENSOR: L1 CHANNEL PAIRS → L2 CHANNELS")
print("="*60)
print("""
T[a, h2, b] = how L1 channels (a, b) jointly contribute to L2 channel h2

This is computed as: einsum('ar,rh,sh,sb->ahb', D1.T, L2.T, R2.T, D1)

For L2 channel h2, the 4x4 matrix T[:, h2, :] shows:
- Diagonal: single L1 channel squared contributions
- Off-diagonal: cross-terms between different L1 channels
""")

# Recompute composed tensor
composed = torch.einsum('ar,rh,sh,sb->ahb', D1.T, L2.T, R2.T, D1)
print(f"Composed tensor shape: {composed.shape} = (L1_ch_a, L2_ch, L1_ch_b)")

# Visualize: one 4x4 matrix per L2 channel
fig, axes = plt.subplots(1, rank + 1, figsize=(4*(rank+1), 4))

for h2 in range(rank):
    ax = axes[h2]
    T_h2 = composed[:, h2, :].numpy()
    vmax = np.abs(T_h2).max()
    im = ax.imshow(T_h2, cmap='RdBu_r', vmin=-vmax, vmax=vmax)
    ax.set_title(f'T[:, {h2}, :]\nL2 channel {h2}', fontsize=11)
    ax.set_xlabel('L1 channel b')
    ax.set_ylabel('L1 channel a')
    ax.set_xticks(range(rank))
    ax.set_yticks(range(rank))

    for i in range(rank):
        for j in range(rank):
            val = T_h2[i, j]
            color = 'white' if abs(val) > vmax * 0.6 else 'black'
            ax.text(j, i, f'{val:.2f}', ha='center', va='center', fontsize=9, color=color)

    plt.colorbar(im, ax=ax, shrink=0.6)

# Summary: which L2 channel has strongest/weakest cross-terms
ax_sum = axes[rank]
diag_strength = []
offdiag_strength = []
for h2 in range(rank):
    T_h2 = composed[:, h2, :]
    diag = torch.diag(T_h2).pow(2).sum().sqrt().item()
    offdiag_mask = ~torch.eye(rank, dtype=torch.bool)
    offdiag = T_h2[offdiag_mask].pow(2).sum().sqrt().item()
    diag_strength.append(diag)
    offdiag_strength.append(offdiag)

x_pos = np.arange(rank)
width = 0.35
ax_sum.bar(x_pos - width/2, diag_strength, width, label='Diagonal', color='steelblue', edgecolor='black')
ax_sum.bar(x_pos + width/2, offdiag_strength, width, label='Off-diagonal', color='coral', edgecolor='black')
ax_sum.set_xlabel('L2 channel')
ax_sum.set_ylabel('Frobenius norm')
ax_sum.set_title('Diagonal vs Off-diagonal\nstrength per L2 channel', fontsize=11)
ax_sum.set_xticks(x_pos)
ax_sum.legend()

plt.suptitle('Composed Tensor: How L1 channel pairs contribute to L2', fontsize=12)
plt.tight_layout()
plt.show()

# %%
# =============================================================================
# WHAT DO L1 OPTIMAL INPUTS PRODUCE IN LAYER 2?
# =============================================================================
print("\n" + "="*60)
print("L1 OPTIMAL INPUTS → LAYER 2 ACTIVATIONS")
print("="*60)
print("""
For each of our 8x8 joint L1 optimal inputs, what Layer 2 activations result?
This shows which L1 input patterns activate which L2 channels.
""")

def get_layer2_activations(x):
    """Get Layer 2 pre-D activations: (L2 @ h1_norm) * (R2 @ h1_norm)"""
    # Layer 1
    x_norm1 = rmsnorm(x, norm1_weight)
    layer1_out = bilinear(x_norm1, L1, R1, D1)
    h1 = x + layer1_out

    # Layer 2 (before D2)
    h1_norm = rmsnorm(h1, norm2_weight)
    L2h1 = h1_norm @ L2.T
    R2h1 = h1_norm @ R2.T
    return L2h1 * R2h1  # (batch, rank)

# Get L2 activations for all 64 joint L1 inputs
acts_L2 = get_layer2_activations(joint_inputs_flat).reshape(8, 8, rank)

# Visualize as heatmaps
fig, axes = plt.subplots(1, rank + 1, figsize=(4*(rank+1), 4))

for ch in range(rank):
    ax = axes[ch]
    act_ch = acts_L2[:, :, ch].numpy()
    act_ch_masked = np.where(is_zero.numpy(), np.nan, act_ch)

    vmax = np.nanmax(np.abs(act_ch_masked)) if not np.all(np.isnan(act_ch_masked)) else 1
    im = ax.imshow(act_ch_masked, cmap='RdBu_r', vmin=-vmax, vmax=vmax)
    ax.set_title(f'L2 Channel {ch}', fontsize=11)
    ax.set_xticks(range(8))
    ax.set_xticklabels(dir_labels, fontsize=7, rotation=45, ha='right')
    ax.set_yticks(range(8))
    ax.set_yticklabels(dir_labels, fontsize=7)
    plt.colorbar(im, ax=ax, shrink=0.6)

# Total L2 activation
ax_sum = axes[rank]
act_sum = acts_L2.sum(dim=-1).numpy()
act_sum_masked = np.where(is_zero.numpy(), np.nan, act_sum)
vmax = np.nanmax(np.abs(act_sum_masked)) if not np.all(np.isnan(act_sum_masked)) else 1
im = ax_sum.imshow(act_sum_masked, cmap='RdBu_r', vmin=-vmax, vmax=vmax)
ax_sum.set_title('Sum L2 channels', fontsize=11)
ax_sum.set_xticks(range(8))
ax_sum.set_xticklabels(dir_labels, fontsize=7, rotation=45, ha='right')
ax_sum.set_yticks(range(8))
ax_sum.set_yticklabels(dir_labels, fontsize=7)
plt.colorbar(im, ax=ax_sum, shrink=0.6)

plt.suptitle('Layer 2 Activations from Joint L1 Optimal Inputs', fontsize=12)
plt.tight_layout()
plt.show()

# %%
# =============================================================================
# GRADIENT-BASED OPTIMIZATION: Find x that maximizes each L2 channel
# =============================================================================
print("\n" + "="*60)
print("GRADIENT OPTIMIZATION: FIND x THAT MAXIMIZES L2 CHANNELS")
print("="*60)
print("""
Since the composition is complex (quadratic in L1 which feeds into L2),
we use gradient ascent to find x that maximizes each L2 channel activation.
""")

def optimize_for_L2_channel(target_ch, n_steps=500, lr=0.1):
    """Find input x that maximizes Layer 2 channel target_ch."""
    x = torch.randn(1, n, requires_grad=True)

    for step in range(n_steps):
        # Forward to get L2 activation
        x_norm1 = rmsnorm(x, norm1_weight)
        layer1_out = bilinear(x_norm1, L1, R1, D1)
        h1 = x + layer1_out
        h1_norm = rmsnorm(h1, norm2_weight)
        L2h1 = h1_norm @ L2.T
        R2h1 = h1_norm @ R2.T
        act_L2 = (L2h1 * R2h1)[0, target_ch]

        # Gradient ascent
        act_L2.backward()
        with torch.no_grad():
            x.data += lr * x.grad
            x.data = x.data / (x.data.norm() + 1e-8)  # Keep normalized
            x.grad.zero_()

    return x.detach()

# Optimize for each L2 channel (positive activation)
print("Finding optimal x for each L2 channel...")
x_opt_L2 = []
for ch in range(rank):
    x_opt = optimize_for_L2_channel(ch)
    x_opt_L2.append(x_opt[0])
x_opt_L2 = torch.stack(x_opt_L2)

# Also find negative optima (minimize = maximize negative)
def optimize_for_L2_channel_neg(target_ch, n_steps=500, lr=0.1):
    """Find input x that minimizes (maximally negative) Layer 2 channel."""
    x = torch.randn(1, n, requires_grad=True)
    for step in range(n_steps):
        x_norm1 = rmsnorm(x, norm1_weight)
        layer1_out = bilinear(x_norm1, L1, R1, D1)
        h1 = x + layer1_out
        h1_norm = rmsnorm(h1, norm2_weight)
        L2h1 = h1_norm @ L2.T
        R2h1 = h1_norm @ R2.T
        act_L2 = -(L2h1 * R2h1)[0, target_ch]  # Negative for minimization
        act_L2.backward()
        with torch.no_grad():
            x.data += lr * x.grad
            x.data = x.data / (x.data.norm() + 1e-8)
            x.grad.zero_()
    return x.detach()

x_opt_L2_neg = []
for ch in range(rank):
    x_opt = optimize_for_L2_channel_neg(ch)
    x_opt_L2_neg.append(x_opt[0])
x_opt_L2_neg = torch.stack(x_opt_L2_neg)

# %%
# Visualize gradient-optimized inputs for L2
fig, axes = plt.subplots(2, rank, figsize=(4*rank, 6))

for ch in range(rank):
    # Positive
    ax = axes[0, ch]
    x_opt = x_opt_L2[ch].numpy()
    colors = ['green' if v > 0.1 else 'red' if v < -0.1 else 'gray' for v in x_opt]
    ax.bar(range(n), x_opt, color=colors, edgecolor='black')
    ax.set_title(f'Max L2 ch{ch}', fontsize=10)
    ax.axhline(y=0, color='k', linewidth=0.5)
    ax.set_ylim(-1, 1)

    # Negative
    ax = axes[1, ch]
    x_opt = x_opt_L2_neg[ch].numpy()
    colors = ['green' if v > 0.1 else 'red' if v < -0.1 else 'gray' for v in x_opt]
    ax.bar(range(n), x_opt, color=colors, edgecolor='black')
    ax.set_title(f'Min L2 ch{ch}', fontsize=10)
    ax.axhline(y=0, color='k', linewidth=0.5)
    ax.set_ylim(-1, 1)

plt.suptitle('Gradient-Optimized x for Layer 2 Channels\n(Top: maximize, Bottom: minimize)', fontsize=12)
plt.tight_layout()
plt.show()

# %%
# Compare: what L1 activations do these L2-optimal inputs produce?
print("\n" + "="*60)
print("L2-OPTIMAL INPUTS: WHAT L1 PATTERN DO THEY USE?")
print("="*60)

acts_L1_from_L2opt = get_layer1_activations(x_opt_L2)
acts_L1_from_L2opt_neg = get_layer1_activations(x_opt_L2_neg)

print("\nL1 activations from L2-maximizing inputs:")
print(f"{'L2 target':<12} {'L1 ch0':<10} {'L1 ch1':<10} {'L1 ch2':<10} {'L1 ch3':<10}")
print("-" * 55)
for ch in range(rank):
    acts = acts_L1_from_L2opt[ch].numpy()
    print(f"Max L2 ch{ch:<4} {acts[0]:<10.3f} {acts[1]:<10.3f} {acts[2]:<10.3f} {acts[3]:<10.3f}")

print("\nL1 activations from L2-minimizing inputs:")
for ch in range(rank):
    acts = acts_L1_from_L2opt_neg[ch].numpy()
    print(f"Min L2 ch{ch:<4} {acts[0]:<10.3f} {acts[1]:<10.3f} {acts[2]:<10.3f} {acts[3]:<10.3f}")

# %%
# Final: what does the model predict for L2-optimal inputs?
print("\n" + "="*60)
print("MODEL PREDICTIONS FOR L2-OPTIMAL INPUTS")
print("="*60)

print("\nL2-maximizing inputs:")
for ch in range(rank):
    x_opt = x_opt_L2[ch:ch+1]
    logits, _ = forward_with_contributions(x_opt)
    pred = logits.argmax().item()
    target = task_2nd_argmax(x_opt).item()
    logits_str = ', '.join([f'{v:.2f}' for v in logits[0].numpy()])
    status = '✓' if pred == target else '✗'
    print(f"  Max L2 ch{ch}: pred={pred}, target={target} {status}, logits=[{logits_str}]")

print("\nL2-minimizing inputs:")
for ch in range(rank):
    x_opt = x_opt_L2_neg[ch:ch+1]
    logits, _ = forward_with_contributions(x_opt)
    pred = logits.argmax().item()
    target = task_2nd_argmax(x_opt).item()
    logits_str = ', '.join([f'{v:.2f}' for v in logits[0].numpy()])
    status = '✓' if pred == target else '✗'
    print(f"  Min L2 ch{ch}: pred={pred}, target={target} {status}, logits=[{logits_str}]")

# %%
