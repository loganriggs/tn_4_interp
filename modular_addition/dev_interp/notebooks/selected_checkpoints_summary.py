# %% [markdown]
# # Selected Checkpoints Summary
#
# Three vertically stacked plots with shared x-axis (selected checkpoint steps):
# 1. Train/Val Loss
# 2. Train/Val Accuracy
# 3. TN Similarity (consecutive checkpoints)

# %%
import json
import pickle
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
import torch
import sys

sys.path.insert(0, str(Path("../../..").resolve()))

from modular_addition.core import init_model, symmetric_similarity, get_device

# %%
# Path to results - change this to switch between experiments
# "../results" = WD=1e-2 (original), "../results_wd6e2" = WD=6e-2
results_dir = Path("../results")

# Load config
with open(results_dir / "config.json") as f:
    config = json.load(f)
print("Configuration:")
for k, v in config.items():
    print(f"  {k}: {v}")

# %%
# Load training history
with open(results_dir / "training_history.json") as f:
    history_raw = json.load(f)

history = {int(k): v for k, v in history_raw.items()}
all_history_steps = sorted(history.keys())

print(f"Total history steps: {len(all_history_steps)}")

# %%
# Load checkpoints
with open(results_dir / "checkpoints.pkl", "rb") as f:
    checkpoints = pickle.load(f)

print(f"Loaded {len(checkpoints)} checkpoints")

# %%
# Define selected checkpoints (same logic as grokking_analysis.py)
all_steps = sorted(checkpoints.keys())
all_steps_arr = np.array(all_steps)

def find_closest_checkpoints(target_values, available_steps):
    """Find the closest available checkpoint steps to target values."""
    available = np.array(available_steps)
    closest = []
    for target in target_values:
        idx = np.argmin(np.abs(available - target))
        closest.append(available[idx])
    return closest

# Base checkpoints
selected_steps = [0, 26, 168]

# Add 5 evenly spaced between 168 and 504
between_168_504 = np.linspace(168, 504, 7)[1:-1]
selected_steps.extend(find_closest_checkpoints(between_168_504, all_steps))

# Add 504
selected_steps.append(504)

# Add 4 evenly spaced between 504 and 3416
between_504_3416 = np.linspace(504, 3416, 6)[1:-1]
selected_steps.extend(find_closest_checkpoints(between_504_3416, all_steps))

# Add 3416
selected_steps.append(3416)

# Add 3 evenly spaced between 3416 and 22904
between_3416_22904 = np.linspace(3416, 22904, 5)[1:-1]  # 3 points excluding endpoints
selected_steps.extend(find_closest_checkpoints(between_3416_22904, all_steps))

# Add 22904
selected_steps.append(22904)

# Add 10 evenly spaced between 22904 and last step
last_step = all_steps[-1]
between_22904_final = np.linspace(22904, last_step, 12)[1:-1]
selected_steps.extend(find_closest_checkpoints(between_22904_final, all_steps))

# Add final step
selected_steps.append(last_step)

# Remove duplicates and sort
selected_steps = sorted(set(selected_steps))

print(f"Selected {len(selected_steps)} checkpoint steps:")
print(f"  {selected_steps}")

# %%
# Extract metrics at selected steps
# For steps not in history, use closest available
def get_metric_at_step(step, history, metric_name):
    """Get metric value at step, or closest available."""
    if step in history:
        return history[step][metric_name]
    # Find closest step in history
    history_steps = np.array(sorted(history.keys()))
    idx = np.argmin(np.abs(history_steps - step))
    return history[history_steps[idx]][metric_name]

train_losses = [get_metric_at_step(s, history, 'train_loss') for s in selected_steps]
val_losses = [get_metric_at_step(s, history, 'val_loss') for s in selected_steps]
train_accs = [get_metric_at_step(s, history, 'train_acc') for s in selected_steps]
val_accs = [get_metric_at_step(s, history, 'val_acc') for s in selected_steps]

print(f"Extracted metrics for {len(selected_steps)} selected steps")

# %%
# Load models and compute consecutive TN similarity
device = get_device()
P = config['P']
d_hidden = config['d_hidden']

print("Loading models for selected checkpoints...")
models = {}
for step in selected_steps:
    model = init_model(p=P, d_hidden=d_hidden).to(device)
    model.load_state_dict(checkpoints[step])
    model.eval()
    models[step] = model

print(f"Loaded {len(models)} models")

# Compute full pairwise TN similarity matrix
print("Computing pairwise TN similarity matrix...")
n_selected = len(selected_steps)
tn_sim_matrix = np.zeros((n_selected, n_selected))

with torch.no_grad():
    for i in range(n_selected):
        for j in range(i, n_selected):
            step_i = selected_steps[i]
            step_j = selected_steps[j]
            sim = float(symmetric_similarity(models[step_i], models[step_j]))
            tn_sim_matrix[i, j] = sim
            tn_sim_matrix[j, i] = sim

print(f"Computed {n_selected}x{n_selected} TN similarity matrix")

# %%
# Create the three vertically stacked plots
fig, axes = plt.subplots(3, 1, figsize=(14, 14),
                          gridspec_kw={'height_ratios': [1, 1, 1.5]})

x_positions = np.arange(len(selected_steps))
x_labels = [str(s) for s in selected_steps]
n_points = len(selected_steps)

# Define target steps for vertical marker lines
target_steps = [168, 1092, 8442, 36848]
# Find x-positions for targets (closest match in selected_steps)
target_x_positions = []
for target in target_steps:
    idx = np.argmin(np.abs(np.array(selected_steps) - target))
    target_x_positions.append(idx)

# Shifted x positions for line plots (align with left edge of matrix cells)
x_positions_shifted = x_positions - 0.5

# Plot 1: Accuracy
ax = axes[0]
ax.plot(x_positions_shifted, train_accs, 'b-o', label='Train Acc', markersize=4, linewidth=1.5)
ax.plot(x_positions_shifted, val_accs, 'r-o', label='Val Acc', markersize=4, linewidth=1.5)
ax.axhline(y=0.95, color='green', linestyle='--', alpha=0.5, label='95% threshold')
ax.set_ylabel('Accuracy', fontsize=11)
ax.set_title('Train/Val Accuracy at Selected Checkpoints', fontsize=12)
ax.set_ylim([0, 1.05])
ax.set_xlim(-0.5, n_points - 0.5)
ax.set_xticks(x_positions_shifted)
ax.set_xticklabels(x_labels, rotation=45, ha='right', fontsize=8)
ax.legend(loc='lower right')
ax.grid(True, alpha=0.3)
# Add vertical marker lines
for x_pos in target_x_positions:
    ax.axvline(x=x_pos - 0.5, color='red', linestyle='--', alpha=0.7, linewidth=1)

# Plot 2: Loss
ax = axes[1]
ax.semilogy(x_positions_shifted, train_losses, 'b-o', label='Train Loss', markersize=4, linewidth=1.5)
ax.semilogy(x_positions_shifted, val_losses, 'r-o', label='Val Loss', markersize=4, linewidth=1.5)
ax.set_ylabel('Loss (log scale)', fontsize=11)
ax.set_title('Train/Val Loss at Selected Checkpoints', fontsize=12)
ax.set_xlim(-0.5, n_points - 0.5)
ax.set_xticks(x_positions_shifted)
ax.set_xticklabels(x_labels, rotation=45, ha='right', fontsize=8)
ax.legend(loc='upper right')
ax.grid(True, alpha=0.3)
# Add vertical marker lines
for x_pos in target_x_positions:
    ax.axvline(x=x_pos - 0.5, color='red', linestyle='--', alpha=0.7, linewidth=1)

# Plot 3: Full TN Similarity Matrix (2D heatmap)
ax = axes[2]
im = ax.imshow(tn_sim_matrix, cmap='viridis', vmin=0, vmax=1, aspect='auto')
ax.set_xlabel('Training Step', fontsize=11)
ax.set_ylabel('Training Step', fontsize=11)
ax.set_title('Pairwise TN Similarity Matrix', fontsize=12)
ax.set_xticks(x_positions)
ax.set_xticklabels(x_labels, rotation=45, ha='right', fontsize=8)
ax.set_yticks(x_positions)
ax.set_yticklabels(x_labels, fontsize=8)
# Add vertical marker lines (shifted to align with top plots)
for x_pos in target_x_positions:
    ax.axvline(x=x_pos - 0.5, color='red', linestyle='--', alpha=0.7, linewidth=1)

# Add colorbar at the bottom (horizontal) to keep plots aligned
cbar = plt.colorbar(im, ax=ax, orientation='horizontal', location='bottom', shrink=0.8, pad=0.15)
cbar.set_label('TN Similarity', fontsize=10)

plt.tight_layout()
plt.savefig(results_dir / "selected_checkpoints_summary.png", dpi=150, bbox_inches='tight')
plt.show()

print(f"\nSaved: {results_dir / 'selected_checkpoints_summary.png'}")

# %%
# Print summary statistics
print("\n" + "=" * 60)
print("Summary Statistics")
print("=" * 60)

# Find grokking point
grok_idx = next((i for i, acc in enumerate(val_accs) if acc >= 0.95), None)
if grok_idx is not None:
    print(f"Grokking (95% val acc) at step: {selected_steps[grok_idx]}")
else:
    print("Did not reach 95% val accuracy")

# TN similarity stats (upper triangle, excluding diagonal)
upper_tri = tn_sim_matrix[np.triu_indices(n_selected, k=1)]
print(f"\nPairwise TN Similarity (off-diagonal):")
print(f"  Min: {upper_tri.min():.4f}")
print(f"  Max: {upper_tri.max():.4f}")
print(f"  Mean: {upper_tri.mean():.4f}")

# Consecutive similarities
consecutive_sims = [tn_sim_matrix[i, i+1] for i in range(n_selected - 1)]
print(f"\nConsecutive TN Similarity:")
print(f"  Min: {min(consecutive_sims):.4f}")
print(f"  Max: {max(consecutive_sims):.4f}")
print(f"  Mean: {np.mean(consecutive_sims):.4f}")

# Find largest jumps (lowest consecutive similarity)
sorted_idx = np.argsort(consecutive_sims)
print(f"\nLargest model changes (lowest consecutive TN sim):")
for i in sorted_idx[:5]:
    print(f"  Step {selected_steps[i]} -> {selected_steps[i+1]}: {consecutive_sims[i]:.4f}")

# %%
