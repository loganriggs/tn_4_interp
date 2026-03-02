# %%
"""
Analysis of Symmetric Bilinear Model: D @ (L @ x)²

Much cleaner interpretation since activations are always positive!
"""

import torch
import numpy as np
import matplotlib.pyplot as plt
import pickle
from pathlib import Path

# %%
# Load PRUNED model (threshold 0.25)
ckpt_path = Path("prune_symmetric_results.pkl")
with open(ckpt_path, 'rb') as f:
    data = pickle.load(f)

config = data['config']
n = config['n']
rank = config['rank']

# Get threshold 0.25 result
prune_result = [r for r in data['results'] if r['threshold'] == 0.25][0]
state_dict = prune_result['state_dict']

print(f"Loaded PRUNED symmetric bilinear model (threshold=0.25)")
print(f"Config: n={n}, rank={rank}, layers={config['num_layers']}")
print(f"Accuracy: {prune_result['final_acc']:.1%}, Pruned: {prune_result['pruned_frac']:.1%}")

# Extract weights
L1 = state_dict['layers.0.L']
D1 = state_dict['layers.0.D']
L2 = state_dict['layers.1.L']
D2 = state_dict['layers.1.D']
norm1_weight = state_dict['norms.0.weight']
norm2_weight = state_dict['norms.1.weight']

print(f"\nL1: {L1.shape}, D1: {D1.shape}")
print(f"L2: {L2.shape}, D2: {D2.shape}")

# %%
# Helper functions
def rmsnorm(x, weight, eps=1e-6):
    rms = torch.sqrt((x ** 2).mean(dim=-1, keepdim=True) + eps)
    return weight * (x / rms)

def symmetric_bilinear(x, L, D):
    """D @ (L @ x)²"""
    Lx = x @ L.T
    return (Lx ** 2) @ D.T

def task_2nd_argmax(x):
    return x.argsort(-1)[..., -2]

def forward(x):
    h = x
    h_norm = rmsnorm(h, norm1_weight)
    h = h + symmetric_bilinear(h_norm, L1, D1)
    h_norm = rmsnorm(h, norm2_weight)
    h = h + symmetric_bilinear(h_norm, L2, D2)
    return h

def get_L1_activations(x):
    """Get L1 hidden activations: (L1 @ x_norm)²"""
    x_norm = rmsnorm(x, norm1_weight)
    return (x_norm @ L1.T) ** 2

def get_L2_activations(x):
    """Get L2 hidden activations after going through L1"""
    h = x
    h_norm = rmsnorm(h, norm1_weight)
    h = h + symmetric_bilinear(h_norm, L1, D1)
    h_norm = rmsnorm(h, norm2_weight)
    return (h_norm @ L2.T) ** 2

# %%
# =============================================================================
# WEIGHT VISUALIZATION
# =============================================================================
print("\n" + "="*60)
print("WEIGHT MATRICES")
print("="*60)

def plot_matrix(ax, mat, title):
    vmax = max(abs(mat.min()), abs(mat.max()))
    im = ax.imshow(mat, cmap='RdBu_r', vmin=-vmax, vmax=vmax, aspect='auto')
    ax.set_title(title, fontsize=10)
    for i in range(mat.shape[0]):
        for j in range(mat.shape[1]):
            val = mat[i, j]
            color = 'white' if abs(val) > vmax * 0.6 else 'black'
            ax.text(j, i, f'{val:.2f}', ha='center', va='center', fontsize=8, color=color)
    return im

fig, axes = plt.subplots(2, 3, figsize=(14, 8))

plot_matrix(axes[0, 0], L1.numpy(), f'L1 ({rank}x{n})\nhidden x input')
plot_matrix(axes[0, 1], D1.numpy(), f'D1 ({n}x{rank})\noutput x hidden')
plot_matrix(axes[0, 2], norm1_weight.numpy().reshape(1, -1), 'Norm1 weight')
axes[0, 2].set_yticks([])

plot_matrix(axes[1, 0], L2.numpy(), f'L2 ({rank}x{n})\nhidden x input')
plot_matrix(axes[1, 1], D2.numpy(), f'D2 ({n}x{rank})\noutput x hidden')
plot_matrix(axes[1, 2], norm2_weight.numpy().reshape(1, -1), 'Norm2 weight')
axes[1, 2].set_yticks([])

axes[0, 0].set_ylabel('Layer 1', fontsize=12, fontweight='bold')
axes[1, 0].set_ylabel('Layer 2', fontsize=12, fontweight='bold')

plt.suptitle('Symmetric Bilinear Weights (white = 0)', fontsize=14)
plt.tight_layout()
plt.show()

# %%
# =============================================================================
# OPTIMAL INPUTS FOR LAYER 1
# =============================================================================
print("\n" + "="*60)
print("OPTIMAL INPUTS FOR LAYER 1")
print("="*60)
print("""
For symmetric bilinear: activation_h = (L[h] · x)²

Optimal input: x* = L[h] / ||L[h]||
(Note: -x* gives same activation since it's squared!)
""")

# Optimal inputs = normalized L vectors
x_opt_L1 = L1 / (L1.norm(dim=1, keepdim=True) + 1e-8)

fig, axes = plt.subplots(1, rank, figsize=(4*rank, 4))

for h in range(rank):
    ax = axes[h]
    x_opt = x_opt_L1[h].numpy()
    colors = ['green' if v > 0.1 else 'red' if v < -0.1 else 'gray' for v in x_opt]
    ax.bar(range(n), x_opt, color=colors, edgecolor='black')
    ax.set_title(f'Ch {h}: x* = L1[{h}]', fontsize=11)
    ax.axhline(y=0, color='k', linewidth=0.5)
    ax.set_ylim(-1, 1)
    ax.set_xlabel('Position')

plt.suptitle('Layer 1: Optimal Inputs (= normalized L vectors)', fontsize=12)
plt.tight_layout()
plt.show()

# Test activations
acts = get_L1_activations(x_opt_L1)
print("\nL1 activations from optimal inputs:")
print(f"{'Input':<10}", end="")
for h in range(rank):
    print(f"{'Act['+str(h)+']':<10}", end="")
print()
print("-" * 50)
for h in range(rank):
    print(f"x* ch{h:<5}", end="")
    for j in range(rank):
        print(f"{acts[h, j].item():<10.3f}", end="")
    print()

# %%
# =============================================================================
# JOINT OPTIMIZATION: 4x4 GRID (simpler - no negative directions needed!)
# =============================================================================
print("\n" + "="*60)
print("JOINT OPTIMIZATION: COMBINING L1 CHANNELS")
print("="*60)
print("""
Since activations are always positive, we only need 4 directions (not 8).
Joint optimal for channels i,j: x* = normalize(L1[i] + L1[j])
""")

# Compute 4x4 joint optimal inputs
joint_inputs = torch.zeros(rank, rank, n)

for i in range(rank):
    for j in range(rank):
        x_sum = x_opt_L1[i] + x_opt_L1[j]
        joint_inputs[i, j] = x_sum / (x_sum.norm() + 1e-8)

# Visualize 4x4 grid
fig, axes = plt.subplots(rank, rank, figsize=(12, 12))

for i in range(rank):
    for j in range(rank):
        ax = axes[i, j]
        x_opt = joint_inputs[i, j].numpy()
        colors = ['green' if v > 0.1 else 'red' if v < -0.1 else 'gray' for v in x_opt]
        ax.bar(range(n), x_opt, color=colors, edgecolor='black', linewidth=0.5)
        ax.axhline(y=0, color='k', linewidth=0.3)
        ax.set_ylim(-1, 1)
        ax.set_xticks([])
        ax.set_yticks([])

        if i == 0:
            ax.set_title(f'Ch {j}', fontsize=10)
        if j == 0:
            ax.set_ylabel(f'Ch {i}', fontsize=10)

plt.suptitle('Joint Optimal Inputs: x* = normalize(L1[i] + L1[j])', fontsize=12)
plt.tight_layout()
plt.show()

# %%
# Activations for joint inputs
print("\nL1 activations for joint optimal inputs:")

joint_flat = joint_inputs.reshape(rank * rank, n)
acts_joint = get_L1_activations(joint_flat).reshape(rank, rank, rank)

fig, axes = plt.subplots(1, rank + 1, figsize=(4*(rank+1), 4))

for ch in range(rank):
    ax = axes[ch]
    act_ch = acts_joint[:, :, ch].numpy()
    im = ax.imshow(act_ch, cmap='Greens', vmin=0)  # All positive!
    ax.set_title(f'L1 Ch {ch} activation', fontsize=11)
    ax.set_xlabel('Col channel j')
    ax.set_ylabel('Row channel i')
    ax.set_xticks(range(rank))
    ax.set_yticks(range(rank))
    for ii in range(rank):
        for jj in range(rank):
            ax.text(jj, ii, f'{act_ch[ii,jj]:.2f}', ha='center', va='center', fontsize=9)
    plt.colorbar(im, ax=ax, shrink=0.6)

# Total activation
ax_sum = axes[rank]
act_sum = acts_joint.sum(dim=-1).numpy()
im = ax_sum.imshow(act_sum, cmap='Greens', vmin=0)
ax_sum.set_title('Sum all channels', fontsize=11)
ax_sum.set_xlabel('Col channel j')
ax_sum.set_ylabel('Row channel i')
ax_sum.set_xticks(range(rank))
ax_sum.set_yticks(range(rank))
for ii in range(rank):
    for jj in range(rank):
        ax_sum.text(jj, ii, f'{act_sum[ii,jj]:.2f}', ha='center', va='center', fontsize=9)
plt.colorbar(im, ax=ax_sum, shrink=0.6)

plt.suptitle('L1 Activations from Joint Optimal Inputs (all positive!)', fontsize=12)
plt.tight_layout()
plt.show()

# %%
# =============================================================================
# MODEL PREDICTIONS FOR JOINT INPUTS
# =============================================================================
print("\n" + "="*60)
print("MODEL PREDICTIONS FOR JOINT OPTIMAL INPUTS")
print("="*60)

predictions = torch.zeros(rank, rank, dtype=torch.long)
targets = torch.zeros(rank, rank, dtype=torch.long)

for i in range(rank):
    for j in range(rank):
        x_opt = joint_inputs[i, j:j+1]
        logits = forward(x_opt)
        predictions[i, j] = logits.argmax().item()
        targets[i, j] = task_2nd_argmax(x_opt).item()

correct = (predictions == targets)

fig, axes = plt.subplots(1, 3, figsize=(12, 4))

# Predictions
ax1 = axes[0]
im1 = ax1.imshow(predictions.numpy(), cmap='tab10', vmin=0, vmax=3)
ax1.set_title('Predicted Position', fontsize=11)
ax1.set_xlabel('Col channel j')
ax1.set_ylabel('Row channel i')
for i in range(rank):
    for j in range(rank):
        ax1.text(j, i, str(predictions[i,j].item()), ha='center', va='center', fontsize=12, color='white')
plt.colorbar(im1, ax=ax1, shrink=0.6)

# Targets
ax2 = axes[1]
im2 = ax2.imshow(targets.numpy(), cmap='tab10', vmin=0, vmax=3)
ax2.set_title('Target Position', fontsize=11)
ax2.set_xlabel('Col channel j')
ax2.set_ylabel('Row channel i')
for i in range(rank):
    for j in range(rank):
        ax2.text(j, i, str(targets[i,j].item()), ha='center', va='center', fontsize=12, color='white')
plt.colorbar(im2, ax=ax2, shrink=0.6)

# Correct
ax3 = axes[2]
im3 = ax3.imshow(correct.numpy(), cmap='RdYlGn', vmin=0, vmax=1)
ax3.set_title(f'Correct: {correct.sum()}/{rank*rank}', fontsize=11)
ax3.set_xlabel('Col channel j')
ax3.set_ylabel('Row channel i')
for i in range(rank):
    for j in range(rank):
        ax3.text(j, i, '✓' if correct[i,j] else '✗', ha='center', va='center', fontsize=14)

plt.suptitle('Predictions for Joint Optimal Inputs', fontsize=12)
plt.tight_layout()
plt.show()

# %%
# =============================================================================
# LAYER 2 ANALYSIS
# =============================================================================
print("\n" + "="*60)
print("LAYER 2: OPTIMAL INPUTS IN h1 SPACE")
print("="*60)

# Optimal h1 for L2 channels
h1_opt_L2 = L2 / (L2.norm(dim=1, keepdim=True) + 1e-8)

fig, axes = plt.subplots(1, rank, figsize=(4*rank, 4))

for h2 in range(rank):
    ax = axes[h2]
    h1 = h1_opt_L2[h2].numpy()
    colors = ['green' if v > 0.1 else 'red' if v < -0.1 else 'gray' for v in h1]
    ax.bar(range(n), h1, color=colors, edgecolor='black')
    ax.set_title(f'L2 Ch {h2}: h1* = L2[{h2}]', fontsize=11)
    ax.axhline(y=0, color='k', linewidth=0.5)
    ax.set_ylim(-1, 1)
    ax.set_xlabel('Position')

plt.suptitle('Layer 2: Optimal h1 (intermediate representation)', fontsize=12)
plt.tight_layout()
plt.show()

# %%
# =============================================================================
# COMPOSED TENSOR: L1 -> L2 CONNECTION (simpler now!)
# =============================================================================
print("\n" + "="*60)
print("COMPOSED TENSOR: L1 CHANNELS -> L2 CHANNELS")
print("="*60)
print("""
For symmetric bilinear:
L2 activation = (L2 @ h1)² where h1 = x + D1 @ (L1 @ x)²

The connection from L1 to L2 is through D1.
T[a, h2] = (L2[h2] @ D1[:, a])² tells us how L1 channel a contributes to L2 channel h2.

But since L1 activations are squared, the contribution is always positive!
""")

# Connection matrix: how much does L1 channel a contribute to L2 channel h2?
# L2[h2] reads from h1, which includes D1[:, a] * (L1_activation_a)
# So the coupling is: L2[h2] @ D1[:, a]

coupling = L2 @ D1  # (rank, rank) - L2_ch x L1_ch

fig, axes = plt.subplots(1, 2, figsize=(10, 4))

# Coupling matrix
ax1 = axes[0]
vmax = coupling.abs().max().item()
im1 = ax1.imshow(coupling.numpy(), cmap='RdBu_r', vmin=-vmax, vmax=vmax)
ax1.set_title('L2 @ D1: How L1 channels affect L2', fontsize=11)
ax1.set_xlabel('L1 channel')
ax1.set_ylabel('L2 channel')
ax1.set_xticks(range(rank))
ax1.set_yticks(range(rank))
for i in range(rank):
    for j in range(rank):
        val = coupling[i, j].item()
        color = 'white' if abs(val) > vmax * 0.6 else 'black'
        ax1.text(j, i, f'{val:.2f}', ha='center', va='center', fontsize=10, color=color)
plt.colorbar(im1, ax=ax1, shrink=0.8)

# Squared coupling (since L1 activations are squared, this is the actual contribution)
coupling_sq = coupling ** 2
ax2 = axes[1]
im2 = ax2.imshow(coupling_sq.numpy(), cmap='Greens', vmin=0)
ax2.set_title('(L2 @ D1)²: Actual contribution (always positive)', fontsize=11)
ax2.set_xlabel('L1 channel')
ax2.set_ylabel('L2 channel')
ax2.set_xticks(range(rank))
ax2.set_yticks(range(rank))
for i in range(rank):
    for j in range(rank):
        val = coupling_sq[i, j].item()
        ax2.text(j, i, f'{val:.2f}', ha='center', va='center', fontsize=10)
plt.colorbar(im2, ax=ax2, shrink=0.8)

plt.tight_layout()
plt.show()

# %%
# L2 activations from joint L1 inputs
print("\n" + "="*60)
print("L2 ACTIVATIONS FROM JOINT L1 OPTIMAL INPUTS")
print("="*60)

acts_L2 = get_L2_activations(joint_flat).reshape(rank, rank, rank)

fig, axes = plt.subplots(1, rank + 1, figsize=(4*(rank+1), 4))

for ch in range(rank):
    ax = axes[ch]
    act_ch = acts_L2[:, :, ch].numpy()
    im = ax.imshow(act_ch, cmap='Greens', vmin=0)
    ax.set_title(f'L2 Ch {ch}', fontsize=11)
    ax.set_xlabel('Col j')
    ax.set_ylabel('Row i')
    ax.set_xticks(range(rank))
    ax.set_yticks(range(rank))
    for ii in range(rank):
        for jj in range(rank):
            ax.text(jj, ii, f'{act_ch[ii,jj]:.2f}', ha='center', va='center', fontsize=8)
    plt.colorbar(im, ax=ax, shrink=0.6)

ax_sum = axes[rank]
act_sum = acts_L2.sum(dim=-1).numpy()
im = ax_sum.imshow(act_sum, cmap='Greens', vmin=0)
ax_sum.set_title('Sum L2', fontsize=11)
ax_sum.set_xlabel('Col j')
ax_sum.set_ylabel('Row i')
ax_sum.set_xticks(range(rank))
ax_sum.set_yticks(range(rank))
for ii in range(rank):
    for jj in range(rank):
        ax_sum.text(jj, ii, f'{act_sum[ii,jj]:.2f}', ha='center', va='center', fontsize=8)
plt.colorbar(im, ax=ax_sum, shrink=0.6)

plt.suptitle('L2 Activations from Joint L1 Inputs (all positive!)', fontsize=12)
plt.tight_layout()
plt.show()

# %%
# =============================================================================
# LOGIT CONTRIBUTION ANALYSIS
# =============================================================================
print("\n" + "="*60)
print("LOGIT CONTRIBUTION ANALYSIS")
print("="*60)

def forward_with_contributions(x):
    """Forward pass returning contributions from each component."""
    h = x.clone()

    # Layer 1
    h_norm = rmsnorm(h, norm1_weight)
    Lh = h_norm @ L1.T
    layer1_out = (Lh ** 2) @ D1.T
    h = h + layer1_out

    # Layer 2
    h_norm = rmsnorm(h, norm2_weight)
    Lh = h_norm @ L2.T
    layer2_out = (Lh ** 2) @ D2.T
    h = h + layer2_out

    return h, {
        'direct': x.clone(),
        'layer1': layer1_out,
        'layer2': layer2_out,
    }

# Test on random examples
import time
torch.manual_seed(int(time.time() * 1000) % (2**32))
x_test = torch.randn(5, n)

print("\nRandom example analysis:")
for i in range(5):
    x_i = x_test[i:i+1]
    logits, contribs = forward_with_contributions(x_i)
    pred = logits.argmax().item()
    target = task_2nd_argmax(x_i).item()
    status = '✓' if pred == target else '✗'

    print(f"\nExample {i+1}: pred={pred}, target={target} {status}")
    print(f"  Input:   {x_i[0].numpy().round(3)}")
    print(f"  Direct:  {contribs['direct'][0].numpy().round(3)}")
    print(f"  Layer1:  {contribs['layer1'][0].numpy().round(3)}")
    print(f"  Layer2:  {contribs['layer2'][0].numpy().round(3)}")
    print(f"  Logits:  {logits[0].numpy().round(3)}")

# %%
# Contribution magnitudes
x_many = torch.randn(1000, n)
logits_many, contribs_many = forward_with_contributions(x_many)

print("\n" + "="*60)
print("CONTRIBUTION MAGNITUDES (1000 examples)")
print("="*60)

print(f"Average L2 norm:")
print(f"  Direct:  {contribs_many['direct'].norm(dim=1).mean():.3f}")
print(f"  Layer1:  {contribs_many['layer1'].norm(dim=1).mean():.3f}")
print(f"  Layer2:  {contribs_many['layer2'].norm(dim=1).mean():.3f}")

targets_many = task_2nd_argmax(x_many)
preds_many = logits_many.argmax(dim=1)
accuracy = (preds_many == targets_many).float().mean()
print(f"\nAccuracy: {accuracy:.1%}")

# %%
