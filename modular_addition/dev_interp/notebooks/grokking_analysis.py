# %% [markdown]
# # Grokking Trajectory Analysis
# 
# This notebook analyzes the results of the grokking trajectory experiment.
# We visualize how TN similarity, activation similarity, and JS divergence
# change throughout training to understand the grokking phenomenon.

# %%
import json
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm

# Set up plotting style
plt.rcParams['figure.figsize'] = (12, 8)
plt.rcParams['font.size'] = 12

# %% [markdown]
# ## 1. Load Results

# %%
# Path to results - change this to switch between experiments
# "../results" = WD=1e-2 (original), "../results_wd6e2" = WD=6e-2
results_dir = Path("../results_wd6e2")  # WD=6e-2 experiment

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

# Convert string keys back to integers and sort
history = {int(k): v for k, v in history_raw.items()}
steps = sorted(history.keys())

# Extract metrics
train_losses = [history[s]['train_loss'] for s in steps]
val_losses = [history[s]['val_loss'] for s in steps]
val_accs = [history[s]['val_acc'] for s in steps]

print(f"Number of checkpoints: {len(steps)}")
print(f"Step range: {steps[0]} to {steps[-1]}")
print(f"Final val accuracy: {val_accs[-1]:.4f}")

# %%
# Load similarity matrices
tn_sim = np.load(results_dir / "tn_similarity.npy")
act_sim = np.load(results_dir / "act_similarity.npy")
js_div = np.load(results_dir / "js_divergence.npy")
checkpoint_steps = np.load(results_dir / "checkpoint_steps.npy")

print(f"Matrix shapes: {tn_sim.shape}")
print(f"TN sim range: [{tn_sim.min():.4f}, {tn_sim.max():.4f}]")
print(f"Act sim range: [{act_sim.min():.4f}, {act_sim.max():.4f}]")
print(f"JS div range: [{js_div.min():.4f}, {js_div.max():.4f}]")

# %% [markdown]
# ## 2. Training Curves

# %%
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# Loss curves
ax = axes[0]
ax.semilogy(steps, train_losses, label='Train Loss', alpha=0.8)
ax.semilogy(steps, val_losses, label='Val Loss', alpha=0.8)
ax.set_xlabel('Training Steps')
ax.set_ylabel('Loss (log scale)')
ax.set_title('Loss Curves')
ax.legend()
ax.grid(True, alpha=0.3)

# Accuracy curve
ax = axes[1]
ax.plot(steps, val_accs, 'g-', linewidth=2, label='Val Accuracy')
ax.axhline(y=0.95, color='r', linestyle='--', alpha=0.5, label='95% threshold')
ax.set_xlabel('Training Steps')
ax.set_ylabel('Validation Accuracy')
ax.set_title('Validation Accuracy')
ax.legend()
ax.grid(True, alpha=0.3)
ax.set_ylim([0, 1.05])

plt.tight_layout()
plt.show()

# Identify grokking point
grok_threshold = 0.95
grok_idx = next((i for i, acc in enumerate(val_accs) if acc >= grok_threshold), None)
if grok_idx is not None:
    grok_step = steps[grok_idx]
    print(f"Grokking occurred around step {grok_step} (checkpoint index {grok_idx})")
else:
    print(f"Did not reach {grok_threshold*100}% validation accuracy")
    grok_step = None

# %% [markdown]
# ## 3. Similarity Heatmaps

# %%
def plot_similarity_heatmap(matrix, title, ax, cmap='viridis', vmin=None, vmax=None):
    """Plot a similarity/divergence heatmap."""
    N = matrix.shape[0]
    
    # Create tick positions and labels (show every 20 checkpoints)
    tick_stride = 20
    tick_positions = list(range(0, N, tick_stride))
    tick_labels = [str(checkpoint_steps[i]) for i in tick_positions]
    
    im = ax.imshow(matrix, cmap=cmap, vmin=vmin, vmax=vmax, aspect='auto')
    ax.set_xticks(tick_positions)
    ax.set_xticklabels(tick_labels, rotation=45, ha='right')
    ax.set_yticks(tick_positions)
    ax.set_yticklabels(tick_labels)
    ax.set_xlabel('Step')
    ax.set_ylabel('Step')
    ax.set_title(title)
    
    # Add grokking line if found
    if grok_idx is not None:
        ax.axhline(y=grok_idx, color='red', linestyle='--', alpha=0.7, linewidth=1)
        ax.axvline(x=grok_idx, color='red', linestyle='--', alpha=0.7, linewidth=1)
    
    return im

# %%
fig, axes = plt.subplots(1, 3, figsize=(18, 6))

# TN Similarity
im1 = plot_similarity_heatmap(tn_sim, 'TN Similarity', axes[0], cmap='viridis', vmin=0, vmax=1)
plt.colorbar(im1, ax=axes[0], label='Similarity')

# Activation Similarity
im2 = plot_similarity_heatmap(act_sim, 'Activation Similarity', axes[1], cmap='viridis', vmin=0, vmax=1)
plt.colorbar(im2, ax=axes[1], label='Similarity')

# JS Divergence
im3 = plot_similarity_heatmap(js_div, 'JS Divergence', axes[2], cmap='magma')
plt.colorbar(im3, ax=axes[2], label='Divergence')

plt.suptitle('Similarity Matrices Throughout Training\n(Red lines indicate grokking transition)', fontsize=14)
plt.tight_layout()
plt.show()

# %% [markdown]
# ## 4. Diagonal Band Analysis
# 
# We analyze how similar consecutive checkpoints are by looking at diagonal bands.

# %%
def extract_diagonal_band(matrix, offset=1):
    """Extract diagonal band at given offset from main diagonal."""
    return np.diag(matrix, k=offset)

# Consecutive checkpoint similarities (k=1 diagonal)
tn_consec = extract_diagonal_band(tn_sim, 1)
act_consec = extract_diagonal_band(act_sim, 1)
js_consec = extract_diagonal_band(js_div, 1)

# Steps for plotting (between consecutive checkpoints)
consec_steps = checkpoint_steps[:-1]

fig, axes = plt.subplots(2, 2, figsize=(14, 10))

# TN similarity between consecutive checkpoints
ax = axes[0, 0]
ax.plot(consec_steps, tn_consec, 'b-', linewidth=1.5)
if grok_step:
    ax.axvline(x=grok_step, color='red', linestyle='--', alpha=0.7, label=f'Grok @ {grok_step}')
ax.set_xlabel('Step')
ax.set_ylabel('TN Similarity')
ax.set_title('TN Similarity (Consecutive Checkpoints)')
ax.grid(True, alpha=0.3)
ax.legend()

# Activation similarity between consecutive checkpoints
ax = axes[0, 1]
ax.plot(consec_steps, act_consec, 'g-', linewidth=1.5)
if grok_step:
    ax.axvline(x=grok_step, color='red', linestyle='--', alpha=0.7, label=f'Grok @ {grok_step}')
ax.set_xlabel('Step')
ax.set_ylabel('Activation Similarity')
ax.set_title('Activation Similarity (Consecutive Checkpoints)')
ax.grid(True, alpha=0.3)
ax.legend()

# JS divergence between consecutive checkpoints
ax = axes[1, 0]
ax.plot(consec_steps, js_consec, 'm-', linewidth=1.5)
if grok_step:
    ax.axvline(x=grok_step, color='red', linestyle='--', alpha=0.7, label=f'Grok @ {grok_step}')
ax.set_xlabel('Step')
ax.set_ylabel('JS Divergence')
ax.set_title('JS Divergence (Consecutive Checkpoints)')
ax.grid(True, alpha=0.3)
ax.legend()

# Rate of change (1 - similarity represents change)
ax = axes[1, 1]
ax.plot(consec_steps, 1 - tn_consec, 'b-', linewidth=1.5, label='TN change')
ax.plot(consec_steps, 1 - act_consec, 'g-', linewidth=1.5, label='Act change')
if grok_step:
    ax.axvline(x=grok_step, color='red', linestyle='--', alpha=0.7, label=f'Grok @ {grok_step}')
ax.set_xlabel('Step')
ax.set_ylabel('1 - Similarity (Change)')
ax.set_title('Rate of Change Between Checkpoints')
ax.grid(True, alpha=0.3)
ax.legend()

plt.tight_layout()
plt.show()

# %% [markdown]
# ## 5. Similarity to Final Model
# 
# How similar is each checkpoint to the final trained model?

# %%
# Similarity to final checkpoint
tn_to_final = tn_sim[:, -1]
act_to_final = act_sim[:, -1]
js_to_final = js_div[:, -1]

fig, axes = plt.subplots(1, 3, figsize=(18, 5))

# TN similarity to final
ax = axes[0]
ax.plot(checkpoint_steps, tn_to_final, 'b-', linewidth=2)
ax.fill_between(checkpoint_steps, tn_to_final, alpha=0.3)
if grok_step:
    ax.axvline(x=grok_step, color='red', linestyle='--', alpha=0.7, label=f'Grok @ {grok_step}')
ax.set_xlabel('Step')
ax.set_ylabel('TN Similarity')
ax.set_title('TN Similarity to Final Model')
ax.grid(True, alpha=0.3)
ax.legend()

# Activation similarity to final
ax = axes[1]
ax.plot(checkpoint_steps, act_to_final, 'g-', linewidth=2)
ax.fill_between(checkpoint_steps, act_to_final, alpha=0.3, color='green')
if grok_step:
    ax.axvline(x=grok_step, color='red', linestyle='--', alpha=0.7, label=f'Grok @ {grok_step}')
ax.set_xlabel('Step')
ax.set_ylabel('Activation Similarity')
ax.set_title('Activation Similarity to Final Model')
ax.grid(True, alpha=0.3)
ax.legend()

# JS divergence to final
ax = axes[2]
ax.plot(checkpoint_steps, js_to_final, 'm-', linewidth=2)
ax.fill_between(checkpoint_steps, js_to_final, alpha=0.3, color='magenta')
if grok_step:
    ax.axvline(x=grok_step, color='red', linestyle='--', alpha=0.7, label=f'Grok @ {grok_step}')
ax.set_xlabel('Step')
ax.set_ylabel('JS Divergence')
ax.set_title('JS Divergence from Final Model')
ax.grid(True, alpha=0.3)
ax.legend()

plt.suptitle('Similarity/Divergence to Final Trained Model', fontsize=14)
plt.tight_layout()
plt.show()

# %% [markdown]
# ## 6. Combined Analysis with Validation Accuracy

# %%
fig, axes = plt.subplots(2, 2, figsize=(14, 10))

# Validation accuracy
ax = axes[0, 0]
ax.plot(steps, val_accs, 'k-', linewidth=2, label='Val Accuracy')
ax.axhline(y=0.95, color='r', linestyle='--', alpha=0.5)
ax.set_ylabel('Validation Accuracy')
ax.set_title('Validation Accuracy')
ax.legend()
ax.grid(True, alpha=0.3)
ax.set_ylim([0, 1.05])

# TN similarity to final with val acc on secondary axis
ax = axes[0, 1]
ax2 = ax.twinx()
ln1 = ax.plot(checkpoint_steps, tn_to_final, 'b-', linewidth=2, label='TN Sim to Final')
ln2 = ax2.plot(steps, val_accs, 'k--', alpha=0.5, label='Val Acc')
ax.set_ylabel('TN Similarity', color='blue')
ax2.set_ylabel('Val Acc', color='black')
ax.set_title('TN Similarity to Final vs Val Accuracy')
lns = ln1 + ln2
labs = [l.get_label() for l in lns]
ax.legend(lns, labs)
ax.grid(True, alpha=0.3)

# Activation similarity to final with val acc
ax = axes[1, 0]
ax2 = ax.twinx()
ln1 = ax.plot(checkpoint_steps, act_to_final, 'g-', linewidth=2, label='Act Sim to Final')
ln2 = ax2.plot(steps, val_accs, 'k--', alpha=0.5, label='Val Acc')
ax.set_xlabel('Step')
ax.set_ylabel('Activation Similarity', color='green')
ax2.set_ylabel('Val Acc', color='black')
ax.set_title('Activation Similarity to Final vs Val Accuracy')
lns = ln1 + ln2
labs = [l.get_label() for l in lns]
ax.legend(lns, labs)
ax.grid(True, alpha=0.3)

# JS divergence to final with val acc
ax = axes[1, 1]
ax2 = ax.twinx()
ln1 = ax.plot(checkpoint_steps, js_to_final, 'm-', linewidth=2, label='JS Div to Final')
ln2 = ax2.plot(steps, val_accs, 'k--', alpha=0.5, label='Val Acc')
ax.set_xlabel('Step')
ax.set_ylabel('JS Divergence', color='magenta')
ax2.set_ylabel('Val Acc', color='black')
ax.set_title('JS Divergence from Final vs Val Accuracy')
lns = ln1 + ln2
labs = [l.get_label() for l in lns]
ax.legend(lns, labs)
ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.show()

# %% [markdown]
# ## 7. Summary Statistics

# %%
# Compute correlation between similarity metrics
upper_tri_idx = np.triu_indices(tn_sim.shape[0], k=1)

tn_upper = tn_sim[upper_tri_idx]
act_upper = act_sim[upper_tri_idx]
js_upper = js_div[upper_tri_idx]

print("Correlations between metrics (upper triangular values):")
print(f"  TN vs Activation similarity: {np.corrcoef(tn_upper, act_upper)[0,1]:.4f}")
print(f"  TN similarity vs JS divergence: {np.corrcoef(tn_upper, js_upper)[0,1]:.4f}")
print(f"  Activation similarity vs JS divergence: {np.corrcoef(act_upper, js_upper)[0,1]:.4f}")

print("\nMetric statistics:")
print(f"  TN similarity: mean={tn_upper.mean():.4f}, std={tn_upper.std():.4f}")
print(f"  Activation similarity: mean={act_upper.mean():.4f}, std={act_upper.std():.4f}")
print(f"  JS divergence: mean={js_upper.mean():.4f}, std={js_upper.std():.4f}")

# %%
# Scatter plots of metric pairs
fig, axes = plt.subplots(1, 3, figsize=(15, 4))

ax = axes[0]
ax.scatter(tn_upper, act_upper, alpha=0.3, s=5)
ax.set_xlabel('TN Similarity')
ax.set_ylabel('Activation Similarity')
ax.set_title(f'TN vs Activation (r={np.corrcoef(tn_upper, act_upper)[0,1]:.3f})')
ax.grid(True, alpha=0.3)

ax = axes[1]
ax.scatter(tn_upper, js_upper, alpha=0.3, s=5)
ax.set_xlabel('TN Similarity')
ax.set_ylabel('JS Divergence')
ax.set_title(f'TN Sim vs JS Div (r={np.corrcoef(tn_upper, js_upper)[0,1]:.3f})')
ax.grid(True, alpha=0.3)

ax = axes[2]
ax.scatter(act_upper, js_upper, alpha=0.3, s=5)
ax.set_xlabel('Activation Similarity')
ax.set_ylabel('JS Divergence')
ax.set_title(f'Act Sim vs JS Div (r={np.corrcoef(act_upper, js_upper)[0,1]:.3f})')
ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.show()

# %% [markdown]
# ## 8. Grokking Phase Identification
# 
# Identify different phases of training based on metric behavior.

# %%
# Identify phases based on validation accuracy
# Phase 1: Memorization (high train acc, low val acc)
# Phase 2: Grokking transition
# Phase 3: Generalization (high val acc)

val_acc_array = np.array(val_accs)

# Find transition points
low_acc_threshold = 0.2  # Random chance would be ~1/P
high_acc_threshold = 0.9

# Phase boundaries
memorization_end = None
generalization_start = None

for i, acc in enumerate(val_accs):
    if memorization_end is None and acc > low_acc_threshold:
        memorization_end = i
    if generalization_start is None and acc > high_acc_threshold:
        generalization_start = i
        break

print("Phase identification:")
if memorization_end is not None:
    print(f"  Memorization phase: steps 0 - {steps[memorization_end]}")
    print(f"  Transition begins at checkpoint {memorization_end}")
if generalization_start is not None:
    print(f"  Generalization phase starts: step {steps[generalization_start]}")
    print(f"  Generalization starts at checkpoint {generalization_start}")

# Analyze metric behavior in each phase
if generalization_start is not None:
    print("\nMetric analysis by phase:")
    
    # Pre-grokking
    pre_tn = tn_to_final[:generalization_start].mean()
    pre_act = act_to_final[:generalization_start].mean()
    pre_js = js_to_final[:generalization_start].mean()
    
    # Post-grokking
    post_tn = tn_to_final[generalization_start:].mean()
    post_act = act_to_final[generalization_start:].mean()
    post_js = js_to_final[generalization_start:].mean()
    
    print(f"  Pre-grokking (avg similarity to final):")
    print(f"    TN: {pre_tn:.4f}, Act: {pre_act:.4f}, JS: {pre_js:.4f}")
    print(f"  Post-grokking (avg similarity to final):")
    print(f"    TN: {post_tn:.4f}, Act: {post_act:.4f}, JS: {post_js:.4f}")

# %% [markdown]
# ## 9. Selected Checkpoints TN Similarity
#
# Load model weights for selected checkpoints and compute TN similarity directly.

# %%
import pickle
import torch
import sys
sys.path.insert(0, str(Path("../../..").resolve()))

from modular_addition.core import init_model, symmetric_similarity, get_device

# %%
# Load checkpoints
with open(results_dir / "checkpoints.pkl", "rb") as f:
    checkpoints = pickle.load(f)

print(f"Loaded {len(checkpoints)} checkpoints")
print(f"Checkpoint steps available: {sorted(checkpoints.keys())[:10]}... to {max(checkpoints.keys())}")

# %%
# Select specific checkpoints with denser sampling in key regions
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
between_168_504 = np.linspace(168, 504, 7)[1:-1]  # 5 points excluding endpoints
selected_steps.extend(find_closest_checkpoints(between_168_504, all_steps))

# Add 504
selected_steps.append(504)

# Add 4 evenly spaced between 504 and 3416
between_504_3416 = np.linspace(504, 3416, 6)[1:-1]  # 4 points excluding endpoints
selected_steps.extend(find_closest_checkpoints(between_504_3416, all_steps))

# Add 3416
selected_steps.append(3416)

# Add 3 evenly spaced between 3416 and 22904
between_3416_22904 = np.linspace(3416, 22904, 5)[1:-1]  # 3 points excluding endpoints
selected_steps.extend(find_closest_checkpoints(between_3416_22904, all_steps))

# Add 22904
selected_steps.append(22904)

# Add 10 evenly spaced between 22904 and last step (replacing the 2nd-to-last)
last_step = all_steps[-1]
between_22904_final = np.linspace(22904, last_step, 12)[1:-1]  # 10 points excluding endpoints
selected_steps.extend(find_closest_checkpoints(between_22904_final, all_steps))

# Add final step
selected_steps.append(last_step)

# Remove duplicates and sort
selected_steps = sorted(set(selected_steps))

print(f"Selected {len(selected_steps)} checkpoint steps:")
print(f"  {selected_steps}")

# %%
# Load models for selected checkpoints
device = get_device()
P = config['P']
d_hidden = config['d_hidden']

models = {}
for step in selected_steps:
    model = init_model(p=P, d_hidden=d_hidden).to(device)
    model.load_state_dict(checkpoints[step])
    model.eval()
    models[step] = model

print(f"Loaded {len(models)} models")

# %%
# Compute pairwise TN similarity
n = len(selected_steps)
selected_tn_sim = np.zeros((n, n))

print("Computing TN similarity matrix for selected checkpoints...")
with torch.no_grad():
    for i, step_i in enumerate(selected_steps):
        for j, step_j in enumerate(selected_steps):
            if j >= i:
                sim = float(symmetric_similarity(models[step_i], models[step_j]))
                selected_tn_sim[i, j] = sim
                selected_tn_sim[j, i] = sim

print("\nTN Similarity Matrix (selected checkpoints):")
print(selected_tn_sim.round(4))

# %%
# Plot selected checkpoints TN similarity
fig_size = max(12, n * 0.6)  # Scale figure size with number of checkpoints
fig, ax = plt.subplots(figsize=(fig_size, fig_size * 0.85))

im = ax.imshow(selected_tn_sim, cmap='viridis', vmin=0, vmax=1)
cbar = plt.colorbar(im, ax=ax)
cbar.set_label('TN Similarity', fontsize=12)

# Labels
step_labels = [str(s) for s in selected_steps]
ax.set_xticks(range(n))
ax.set_xticklabels(step_labels, rotation=45, ha='right', fontsize=9)
ax.set_yticks(range(n))
ax.set_yticklabels(step_labels, fontsize=9)

ax.set_xlabel('Step', fontsize=12)
ax.set_ylabel('Step', fontsize=12)
ax.set_title('TN Similarity: Selected Checkpoints\n(0=random init, last=final)', fontsize=14)

# Add value annotations only if not too many checkpoints
if n <= 12:
    for i in range(n):
        for j in range(n):
            val = selected_tn_sim[i, j]
            color = 'white' if val < 0.5 else 'black'
            ax.text(j, i, f'{val:.2f}', ha='center', va='center', color=color, fontsize=8)

plt.tight_layout()
plt.savefig(results_dir / "selected_checkpoints_tn_sim.png", dpi=150)
plt.show()

print(f"\nSaved: {results_dir / 'selected_checkpoints_tn_sim.png'}")

# %% [markdown]
# ## 10. Frequency Analysis of Grokked Model
#
# Visualize the Fourier frequency structure that emerges after grokking.
# The key insight from the literature (e.g., Neel Nanda's work) is that
# grokked models learn to use specific Fourier frequencies to solve
# modular addition: (a + b) mod P.

# %%
from modular_addition.core import (
    compute_interaction_matrix,
    compute_eigendecomposition,
    compute_frequency_distribution,
)

# %%
# Load the final (grokked) model
final_step = 3416
3416
final_model = models[final_step]  # Already loaded from section 9

print(f"Analyzing grokked model from step {final_step}")

# %%
# Compute interaction matrix: (P, 2P, 2P)
# B[r, i, j] = interaction weight between inputs i,j for remainder r
int_mat = compute_interaction_matrix(final_model)
print(f"Interaction matrix shape: {int_mat.shape}")

# %%
# Compute frequency heatmap: p(frequency | remainder)
n_freqs = P // 2 + 1
n_evecs = 4  # Number of top eigenvectors to use

freq_heatmap = np.zeros((P, n_freqs))

for r in range(P):
    evals, evecs = compute_eigendecomposition(int_mat[r])
    freq_heatmap[r] = compute_frequency_distribution(evals, evecs, P, n_evecs)

print(f"Frequency heatmap shape: {freq_heatmap.shape}")
print(f"  Rows: {P} remainders (0 to {P-1})")
print(f"  Cols: {n_freqs} frequencies (0 to {P//2})")

# %%
# Plot frequency heatmap
fig, axes = plt.subplots(1, 2, figsize=(16, 6))

# Full heatmap
ax = axes[0]
im = ax.imshow(freq_heatmap, aspect='auto', cmap='hot')
ax.set_xlabel('Frequency k', fontsize=12)
ax.set_ylabel('Remainder r', fontsize=12)
ax.set_title('p(frequency | remainder) - Grokked Model', fontsize=14)
plt.colorbar(im, ax=ax, label='Probability')

# Highlight dominant frequencies
ax = axes[1]
# Sum across remainders to see which frequencies are most used overall
freq_marginal = freq_heatmap.sum(axis=0)
ax.bar(range(n_freqs), freq_marginal, color='steelblue', alpha=0.7)
ax.set_xlabel('Frequency k', fontsize=12)
ax.set_ylabel('Total weight (sum over remainders)', fontsize=12)
ax.set_title('Frequency Usage (Marginal Distribution)', fontsize=14)
ax.set_xlim(-0.5, n_freqs - 0.5)

# Mark top frequencies
top_k = 5
top_freqs = np.argsort(freq_marginal)[-top_k:][::-1]
for i, f in enumerate(top_freqs):
    ax.annotate(f'k={f}', xy=(f, freq_marginal[f]),
                xytext=(f, freq_marginal[f] + 0.5),
                ha='center', fontsize=10, color='red')

plt.tight_layout()
plt.savefig(results_dir / "frequency_heatmap_grokked.png", dpi=150)
plt.show()

print(f"Top {top_k} frequencies: {top_freqs}")

# %%
# Analyze eigenvector structure for a specific remainder
remainder_to_analyze = 0

evals, evecs = compute_eigendecomposition(int_mat[remainder_to_analyze])

fig, axes = plt.subplots(2, 4, figsize=(16, 8))

for i in range(4):
    evec = evecs[:, i]
    eval_i = evals[i]

    # Top row: eigenvector components
    ax = axes[0, i]
    ax.plot(evec[:P], 'b-', alpha=0.7, label='Input a')
    ax.plot(evec[P:], 'r-', alpha=0.7, label='Input b')
    ax.set_xlabel('Position')
    ax.set_ylabel('Weight')
    ax.set_title(f'Eigenvector {i+1} (λ={eval_i:.3f})')
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    # Bottom row: FFT of eigenvector
    ax = axes[1, i]
    fft_a = np.abs(np.fft.rfft(evec[:P]))
    fft_b = np.abs(np.fft.rfft(evec[P:]))
    freqs = np.arange(len(fft_a))

    ax.bar(freqs - 0.2, fft_a, width=0.4, alpha=0.7, label='Input a', color='blue')
    ax.bar(freqs + 0.2, fft_b, width=0.4, alpha=0.7, label='Input b', color='red')
    ax.set_xlabel('Frequency k')
    ax.set_ylabel('FFT Magnitude')
    ax.set_title(f'FFT of Eigenvector {i+1}')
    ax.legend(fontsize=8)
    ax.set_xlim(-0.5, n_freqs - 0.5)  # Show all frequencies

plt.suptitle(f'Eigenvector Analysis for Remainder r={remainder_to_analyze}', fontsize=14)
plt.tight_layout()
plt.savefig(results_dir / "eigenvector_fft_analysis.png", dpi=150)
plt.show()

# %%
# Compute frequency heatmaps for ALL selected checkpoints
freq_heatmaps = {}

for step in selected_steps:
    model = models[step]
    int_mat_step = compute_interaction_matrix(model)

    heatmap = np.zeros((P, n_freqs))
    for r in range(P):
        evals, evecs = compute_eigendecomposition(int_mat_step[r])
        heatmap[r] = compute_frequency_distribution(evals, evecs, P, n_evecs)

    freq_heatmaps[step] = heatmap
    print(f"Computed frequency heatmap for step {step}")

# Save frequency heatmaps and marginals for later use
freq_heatmaps_array = np.array([freq_heatmaps[s] for s in selected_steps])
freq_marginals_array = np.array([freq_heatmaps[s].sum(axis=0) for s in selected_steps])
np.savez(results_dir / "freq_heatmaps.npz",
         heatmaps=freq_heatmaps_array,
         marginals=freq_marginals_array,
         selected_steps=np.array(selected_steps))
print(f"Saved frequency heatmaps and marginals to {results_dir / 'freq_heatmaps.npz'}")

# %%
# Plot frequency heatmaps for all selected checkpoints
n_checkpoints = len(selected_steps)
n_cols = 5
n_rows = (n_checkpoints + n_cols - 1) // n_cols  # Ceiling division

fig, axes = plt.subplots(n_rows, n_cols, figsize=(4 * n_cols, 4 * n_rows))
axes = axes.flatten()

for idx, step in enumerate(selected_steps):
    ax = axes[idx]
    im = ax.imshow(freq_heatmaps[step], aspect='auto', cmap='hot')
    ax.set_xlabel('Frequency k', fontsize=10)
    ax.set_ylabel('Remainder r', fontsize=10)
    ax.set_title(f'Step {step}', fontsize=12)
    plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

# Hide unused subplots
for idx in range(n_checkpoints, len(axes)):
    axes[idx].axis('off')

plt.suptitle('p(frequency | remainder) Across Training', fontsize=16)
plt.tight_layout()
plt.savefig(results_dir / "frequency_heatmaps_all_checkpoints.png", dpi=150)
plt.show()

# %%
# Plot frequency usage (marginal) for all selected checkpoints
fig, axes = plt.subplots(n_rows, n_cols, figsize=(4 * n_cols, 3 * n_rows))
axes = axes.flatten()

for idx, step in enumerate(selected_steps):
    ax = axes[idx]
    freq_marginal = freq_heatmaps[step].sum(axis=0)
    ax.bar(range(n_freqs), freq_marginal, color='steelblue', alpha=0.7)
    ax.set_xlabel('Frequency k', fontsize=10)
    ax.set_ylabel('Total weight', fontsize=10)
    ax.set_title(f'Step {step}', fontsize=12)
    ax.set_xlim(-0.5, n_freqs - 0.5)  # Show all frequencies

    # Mark top 3 frequencies
    top_freqs = np.argsort(freq_marginal)[-3:][::-1]
    for f in top_freqs:
        if freq_marginal[f] > 0.1:  # Only annotate significant peaks
            ax.annotate(f'k={f}', xy=(f, freq_marginal[f]),
                       xytext=(f, freq_marginal[f] * 1.1),
                       ha='center', fontsize=8, color='red')

# Hide unused subplots
for idx in range(n_checkpoints, len(axes)):
    axes[idx].axis('off')

plt.suptitle('Frequency Usage (Marginal) Across Training', fontsize=16)
plt.tight_layout()
plt.savefig(results_dir / "frequency_usage_all_checkpoints.png", dpi=150)
plt.show()

# %%
# Combined view: heatmap + marginal side by side for each checkpoint
fig, axes = plt.subplots(n_checkpoints, 2, figsize=(12, 4 * n_checkpoints))

for idx, step in enumerate(selected_steps):
    # Heatmap
    ax = axes[idx, 0]
    im = ax.imshow(freq_heatmaps[step], aspect='auto', cmap='hot')
    ax.set_xlabel('Frequency k')
    ax.set_ylabel('Remainder r')
    ax.set_title(f'Step {step}: p(freq | remainder)')
    plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    # Marginal
    ax = axes[idx, 1]
    freq_marginal = freq_heatmaps[step].sum(axis=0)
    ax.bar(range(n_freqs), freq_marginal, color='steelblue', alpha=0.7)
    ax.set_xlabel('Frequency k')
    ax.set_ylabel('Total weight')
    ax.set_title(f'Step {step}: Frequency Usage')
    ax.set_xlim(-0.5, n_freqs - 0.5)  # Show all frequencies

    # Mark top 5 frequencies
    top_freqs = np.argsort(freq_marginal)[-5:][::-1]
    for f in top_freqs:
        if freq_marginal[f] > freq_marginal.max() * 0.1:
            ax.annotate(f'{f}', xy=(f, freq_marginal[f]),
                       xytext=(f, freq_marginal[f] * 1.05),
                       ha='center', fontsize=9, color='red')

plt.suptitle('Frequency Structure Evolution Through Training', fontsize=16, y=1.01)
plt.tight_layout()
plt.savefig(results_dir / "frequency_evolution_combined.png", dpi=150)
plt.show()

# %% [markdown]
# ## 11. Group Checkpoints by TN Similarity Blocks
#
# Cluster checkpoints that are highly similar (>= threshold) into groups,
# then plot one representative frequency marginal per group.

# %%
from scipy.cluster.hierarchy import linkage, fcluster
from scipy.spatial.distance import squareform

# Convert similarity to distance
tn_distance = 1 - selected_tn_sim

# Hierarchical clustering
# Use a high similarity threshold (e.g., 0.95) to group very similar checkpoints
similarity_threshold = 0.7
distance_threshold = 1 - similarity_threshold

# Make sure diagonal is exactly 0 for distance matrix
np.fill_diagonal(tn_distance, 0)

# Convert to condensed form for linkage
condensed_dist = squareform(tn_distance)

# Perform hierarchical clustering
Z = linkage(condensed_dist, method='complete')

# Cut tree at distance threshold to get clusters
cluster_labels = fcluster(Z, t=distance_threshold, criterion='distance')

# Group checkpoints by cluster
groups = {}
for idx, (step, label) in enumerate(zip(selected_steps, cluster_labels)):
    if label not in groups:
        groups[label] = []
    groups[label].append((idx, step))

# Sort groups by the first step in each group
sorted_groups = sorted(groups.items(), key=lambda x: x[1][0][1])

print(f"Found {len(sorted_groups)} groups with similarity threshold {similarity_threshold}:")
for group_id, (label, members) in enumerate(sorted_groups):
    steps_in_group = [step for _, step in members]
    print(f"  Group {group_id + 1}: {steps_in_group}")

# %%
# Compute global y-axis max for consistent scaling
all_marginals = []
for step in selected_steps:
    marginal = freq_heatmaps[step].sum(axis=0)
    all_marginals.append(marginal)
global_y_max = max(m.max() for m in all_marginals) * 1.1

print(f"Global y-axis max: {global_y_max:.2f}")

# %%
# Plot one representative per group (first checkpoint in each group)
n_groups = len(sorted_groups)
fig, axes = plt.subplots(1, n_groups, figsize=(4 * n_groups, 4))
if n_groups == 1:
    axes = [axes]

for group_idx, (label, members) in enumerate(sorted_groups):
    ax = axes[group_idx]

    # Use first checkpoint as representative
    rep_idx, rep_step = members[0]
    freq_marginal = freq_heatmaps[rep_step].sum(axis=0)

    ax.bar(range(n_freqs), freq_marginal, color='steelblue', alpha=0.7)
    ax.set_xlabel('Frequency k', fontsize=10)
    ax.set_ylabel('Total weight', fontsize=10)
    ax.set_xlim(-0.5, n_freqs - 0.5)
    ax.set_ylim(0, global_y_max)  # Same y-axis scale

    # Title with group info
    all_steps = [step for _, step in members]
    if len(all_steps) <= 3:
        title = f"Steps: {all_steps}"
    else:
        title = f"Steps: {all_steps[0]}...{all_steps[-1]} ({len(all_steps)})"
    ax.set_title(title, fontsize=10)

    # Mark top 3 frequencies
    top_freqs = np.argsort(freq_marginal)[-3:][::-1]
    for f in top_freqs:
        if freq_marginal[f] > global_y_max * 0.05:
            ax.annotate(f'{f}', xy=(f, freq_marginal[f]),
                       xytext=(f, freq_marginal[f] + global_y_max * 0.03),
                       ha='center', fontsize=9, color='red')

plt.suptitle(f'Frequency Usage by TN Similarity Groups (threshold={similarity_threshold})', fontsize=14)
plt.tight_layout()
plt.savefig(results_dir / "frequency_by_groups_representative.png", dpi=150)
plt.show()

# %%
# Plot ALL checkpoints within each group (stacked vertically by group)
fig, axes = plt.subplots(n_groups, 1, figsize=(14, 3 * n_groups))
if n_groups == 1:
    axes = [axes]

colors = plt.cm.viridis(np.linspace(0, 1, max(len(m) for _, m in sorted_groups)))

for group_idx, (label, members) in enumerate(sorted_groups):
    ax = axes[group_idx]

    # Plot all checkpoints in this group with slight offset for visibility
    width = 0.8 / len(members)
    for i, (idx, step) in enumerate(members):
        freq_marginal = freq_heatmaps[step].sum(axis=0)
        offset = (i - len(members) / 2 + 0.5) * width
        ax.bar(np.arange(n_freqs) + offset, freq_marginal, width=width,
               alpha=0.7, label=f'{step}', color=colors[i])

    ax.set_xlabel('Frequency k', fontsize=10)
    ax.set_ylabel('Total weight', fontsize=10)
    ax.set_xlim(-0.5, n_freqs - 0.5)
    ax.set_ylim(0, global_y_max)  # Same y-axis scale

    all_steps = [step for _, step in members]
    ax.set_title(f'Group {group_idx + 1}: Steps {all_steps[0]} - {all_steps[-1]}', fontsize=12)
    ax.legend(loc='upper right', fontsize=8, ncol=min(5, len(members)))

plt.suptitle(f'Frequency Usage: All Checkpoints by Group (threshold={similarity_threshold})', fontsize=14)
plt.tight_layout()
plt.savefig(results_dir / "frequency_by_groups_all.png", dpi=150)
plt.show()

# %%
# Summary: show group boundaries on TN similarity matrix
fig, ax = plt.subplots(figsize=(fig_size, fig_size * 0.85))

im = ax.imshow(selected_tn_sim, cmap='viridis', vmin=0, vmax=1)
cbar = plt.colorbar(im, ax=ax)
cbar.set_label('TN Similarity', fontsize=12)

# Draw group boundaries
cumsum = 0
for label, members in sorted_groups:
    group_size = len(members)
    if cumsum > 0:
        ax.axhline(y=cumsum - 0.5, color='red', linestyle='-', linewidth=2)
        ax.axvline(x=cumsum - 0.5, color='red', linestyle='-', linewidth=2)
    cumsum += group_size

# Labels
step_labels = [str(s) for s in selected_steps]
ax.set_xticks(range(n))
ax.set_xticklabels(step_labels, rotation=45, ha='right', fontsize=8)
ax.set_yticks(range(n))
ax.set_yticklabels(step_labels, fontsize=8)

ax.set_xlabel('Step', fontsize=12)
ax.set_ylabel('Step', fontsize=12)
ax.set_title(f'TN Similarity with Group Boundaries (threshold={similarity_threshold})', fontsize=14)

plt.tight_layout()
plt.savefig(results_dir / "tn_sim_with_groups.png", dpi=150)
plt.show()

print(f"\nGroup summary:")
for group_idx, (label, members) in enumerate(sorted_groups):
    steps_list = [step for _, step in members]
    print(f"  Group {group_idx + 1} ({len(members)} checkpoints): {steps_list}")

# %% [markdown]
# ## 12. Pairwise Frequency Heatmap Addition/Subtraction Grid
#
# See `frequency_grid_analysis.py` for pairwise addition/subtraction grids.
# This was moved to a separate file to speed up iteration.
# Run that file after this one to generate the grid plots.

# %%
