# %%
"""
Comprehensive Quantitative Error Analysis for Seed 4 Final Checkpoint

Analyze what the model gets right vs wrong based on:
1. Position-based discrepancy (which position is the 2nd max)
2. Closeness between 2nd largest and largest (margin at top)
3. Closeness between 2nd largest and 3rd largest (margin below)
4. Correct answer's distance from mean
"""

import torch
import numpy as np
import matplotlib.pyplot as plt
import pickle
from pathlib import Path
from scipy import stats

# %%
# Load seed 4 final checkpoint
ckpt_path = Path("checkpoints_r4_seeds/seed4_checkpoints.pkl")
with open(ckpt_path, 'rb') as f:
    data = pickle.load(f)

checkpoints = data['checkpoints']
config = data['config']
final_step = max(checkpoints.keys())
state_dict = checkpoints[final_step]

n = config['n']
rank = config['rank']

print(f"Loaded seed {config['seed']}, step {final_step}")
print(f"Config: n={n}, rank={rank}, layers={config['num_layers']}")

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

def rmsnorm(x, weight, eps=1e-6):
    rms = torch.sqrt((x ** 2).mean(dim=-1, keepdim=True) + eps)
    return weight * (x / rms)

def bilinear(x, L, R, D):
    Lx = x @ L.T
    Rx = x @ R.T
    return (Lx * Rx) @ D.T

def forward(x):
    x_norm1 = rmsnorm(x, norm1_weight)
    layer1_out = bilinear(x_norm1, L1, R1, D1)
    h1 = x + layer1_out
    h1_norm = rmsnorm(h1, norm2_weight)
    layer2_out = bilinear(h1_norm, L2, R2, D2)
    return h1 + layer2_out

def task_2nd_argmax(x):
    return x.argsort(-1)[..., -2]

# %%
# Generate large dataset for analysis
print("Generating dataset for analysis...")
import time
torch.manual_seed(int(time.time() * 1000) % (2**32))  # New random data each run
N_samples = 50000
x_all = torch.randn(N_samples, n)

with torch.no_grad():
    logits_all = forward(x_all)
    preds_all = logits_all.argmax(dim=1)
    targets_all = task_2nd_argmax(x_all)
    correct_all = (preds_all == targets_all)

accuracy = correct_all.float().mean().item()
print(f"Overall accuracy: {accuracy:.2%} ({correct_all.sum().item()}/{N_samples})")

# %%
# Compute all analysis features
print("\nComputing analysis features...")

# Sort each input to get ranked values
sorted_vals, sorted_idx = x_all.sort(dim=1, descending=True)

# Features for each sample
features = {
    'target_position': targets_all.numpy(),  # Which position is 2nd max
    'pred_position': preds_all.numpy(),      # Which position was predicted
    'correct': correct_all.numpy(),

    # Gap between 1st and 2nd largest
    'gap_1st_2nd': (sorted_vals[:, 0] - sorted_vals[:, 1]).numpy(),

    # Gap between 2nd and 3rd largest
    'gap_2nd_3rd': (sorted_vals[:, 1] - sorted_vals[:, 2]).numpy(),

    # 2nd largest value's distance from mean
    'val_2nd_from_mean': (sorted_vals[:, 1] - x_all.mean(dim=1)).numpy(),

    # Absolute value of 2nd largest
    'val_2nd_abs': sorted_vals[:, 1].numpy(),

    # Input mean and std
    'input_mean': x_all.mean(dim=1).numpy(),
    'input_std': x_all.std(dim=1).numpy(),

    # Logit margin (confidence)
    'logit_margin': (logits_all.max(dim=1).values -
                     logits_all.topk(2, dim=1).values[:, 1]).numpy(),
}

# %%
# =============================================================================
# ANALYSIS 1: Position-based accuracy
# =============================================================================
print("\n" + "="*70)
print("ANALYSIS 1: POSITION-BASED ACCURACY")
print("="*70)

fig, axes = plt.subplots(1, 3, figsize=(15, 5))

# 1a. Accuracy by target position
ax1 = axes[0]
pos_acc = []
pos_counts = []
for pos in range(n):
    mask = features['target_position'] == pos
    if mask.sum() > 0:
        acc = features['correct'][mask].mean()
        pos_acc.append(acc)
        pos_counts.append(mask.sum())
    else:
        pos_acc.append(0)
        pos_counts.append(0)

bars = ax1.bar(range(n), pos_acc, color='steelblue', edgecolor='black')
ax1.axhline(y=accuracy, color='red', linestyle='--', linewidth=2, label=f'Overall: {accuracy:.1%}')
ax1.set_xlabel('Target Position (2nd max)', fontsize=12)
ax1.set_ylabel('Accuracy', fontsize=12)
ax1.set_title('Accuracy by Target Position', fontsize=12)
ax1.set_xticks(range(n))
ax1.set_ylim(0, 1)
ax1.legend()

# Add count labels
for i, (acc, cnt) in enumerate(zip(pos_acc, pos_counts)):
    ax1.text(i, acc + 0.02, f'{acc:.1%}\n(n={cnt})', ha='center', fontsize=9)

# 1b. Confusion matrix
ax2 = axes[1]
conf_matrix = np.zeros((n, n))
for true_pos in range(n):
    for pred_pos in range(n):
        mask = (features['target_position'] == true_pos) & (features['pred_position'] == pred_pos)
        conf_matrix[true_pos, pred_pos] = mask.sum()

# Normalize by row (true class)
conf_matrix_norm = conf_matrix / conf_matrix.sum(axis=1, keepdims=True)

im = ax2.imshow(conf_matrix_norm, cmap='Blues', vmin=0, vmax=1)
ax2.set_xlabel('Predicted Position', fontsize=12)
ax2.set_ylabel('True Position (Target)', fontsize=12)
ax2.set_title('Confusion Matrix (row-normalized)', fontsize=12)
ax2.set_xticks(range(n))
ax2.set_yticks(range(n))

# Add text annotations
for i in range(n):
    for j in range(n):
        text = f'{conf_matrix_norm[i, j]:.2f}'
        color = 'white' if conf_matrix_norm[i, j] > 0.5 else 'black'
        ax2.text(j, i, text, ha='center', va='center', fontsize=11, color=color)

plt.colorbar(im, ax=ax2, shrink=0.8)

# 1c. Error patterns: what does the model predict when wrong?
ax3 = axes[2]
wrong_mask = ~features['correct']
if wrong_mask.sum() > 0:
    # For each true position, what are the common wrong predictions?
    error_patterns = np.zeros((n, n))
    for true_pos in range(n):
        for pred_pos in range(n):
            if true_pos != pred_pos:
                mask = (features['target_position'] == true_pos) & \
                       (features['pred_position'] == pred_pos) & wrong_mask
                error_patterns[true_pos, pred_pos] = mask.sum()

    im3 = ax3.imshow(error_patterns, cmap='Reds')
    ax3.set_xlabel('Wrong Prediction', fontsize=12)
    ax3.set_ylabel('True Position', fontsize=12)
    ax3.set_title('Error Counts by Position', fontsize=12)
    ax3.set_xticks(range(n))
    ax3.set_yticks(range(n))

    for i in range(n):
        for j in range(n):
            if error_patterns[i, j] > 0:
                ax3.text(j, i, f'{int(error_patterns[i, j])}',
                        ha='center', va='center', fontsize=11, color='white')
    plt.colorbar(im3, ax=ax3, shrink=0.8)

plt.tight_layout()
plt.show()

# Print detailed stats
print("\nDetailed position statistics:")
print(f"{'Position':<10} {'Count':<10} {'Accuracy':<12} {'# Errors':<10}")
print("-" * 45)
for pos in range(n):
    mask = features['target_position'] == pos
    cnt = mask.sum()
    acc = features['correct'][mask].mean() if cnt > 0 else 0
    errs = (~features['correct'] & mask).sum()
    print(f"{pos:<10} {cnt:<10} {acc:<12.2%} {errs:<10}")

# %%
# =============================================================================
# ANALYSIS 2: Gap between 1st and 2nd largest
# =============================================================================
print("\n" + "="*70)
print("ANALYSIS 2: GAP BETWEEN 1ST AND 2ND LARGEST")
print("="*70)

fig, axes = plt.subplots(1, 3, figsize=(15, 5))

gap_1st_2nd = features['gap_1st_2nd']

# 2a. Distribution of gaps for correct vs wrong
ax1 = axes[0]
ax1.hist(gap_1st_2nd[features['correct']], bins=50, alpha=0.7,
         label=f'Correct (n={features["correct"].sum()})', color='green', density=True)
ax1.hist(gap_1st_2nd[~features['correct']], bins=50, alpha=0.7,
         label=f'Wrong (n={(~features["correct"]).sum()})', color='red', density=True)
ax1.set_xlabel('Gap (1st - 2nd largest)', fontsize=12)
ax1.set_ylabel('Density', fontsize=12)
ax1.set_title('Distribution of Gap by Correctness', fontsize=12)
ax1.legend()
ax1.axvline(x=gap_1st_2nd[features['correct']].mean(), color='darkgreen',
            linestyle='--', linewidth=2, label=f'Mean correct: {gap_1st_2nd[features["correct"]].mean():.3f}')
ax1.axvline(x=gap_1st_2nd[~features['correct']].mean(), color='darkred',
            linestyle='--', linewidth=2)

# 2b. Accuracy vs gap (binned)
ax2 = axes[1]
gap_bins = np.linspace(0, gap_1st_2nd.max(), 21)
gap_bin_centers = (gap_bins[:-1] + gap_bins[1:]) / 2
gap_bin_acc = []
gap_bin_counts = []

for i in range(len(gap_bins) - 1):
    mask = (gap_1st_2nd >= gap_bins[i]) & (gap_1st_2nd < gap_bins[i+1])
    if mask.sum() > 0:
        gap_bin_acc.append(features['correct'][mask].mean())
        gap_bin_counts.append(mask.sum())
    else:
        gap_bin_acc.append(np.nan)
        gap_bin_counts.append(0)

ax2.bar(gap_bin_centers, gap_bin_acc, width=gap_bins[1]-gap_bins[0],
        color='steelblue', edgecolor='black', alpha=0.7)
ax2.axhline(y=accuracy, color='red', linestyle='--', linewidth=2, label=f'Overall: {accuracy:.1%}')
ax2.set_xlabel('Gap (1st - 2nd largest)', fontsize=12)
ax2.set_ylabel('Accuracy', fontsize=12)
ax2.set_title('Accuracy vs Gap Size', fontsize=12)
ax2.set_ylim(0, 1)
ax2.legend()

# 2c. Scatter: gap vs logit margin, colored by correctness
ax3 = axes[2]
sample_idx = np.random.choice(N_samples, min(5000, N_samples), replace=False)
ax3.scatter(gap_1st_2nd[sample_idx][features['correct'][sample_idx]],
            features['logit_margin'][sample_idx][features['correct'][sample_idx]],
            alpha=0.3, s=10, c='green', label='Correct')
ax3.scatter(gap_1st_2nd[sample_idx][~features['correct'][sample_idx]],
            features['logit_margin'][sample_idx][~features['correct'][sample_idx]],
            alpha=0.5, s=15, c='red', label='Wrong')
ax3.set_xlabel('Gap (1st - 2nd largest)', fontsize=12)
ax3.set_ylabel('Logit Margin (confidence)', fontsize=12)
ax3.set_title('Input Gap vs Model Confidence', fontsize=12)
ax3.legend()

plt.tight_layout()
plt.show()

# Statistical test
correct_gaps = gap_1st_2nd[features['correct']]
wrong_gaps = gap_1st_2nd[~features['correct']]
t_stat, p_val = stats.ttest_ind(correct_gaps, wrong_gaps)

print(f"\nStatistical Summary:")
print(f"  Correct predictions - Mean gap: {correct_gaps.mean():.4f} (std: {correct_gaps.std():.4f})")
print(f"  Wrong predictions   - Mean gap: {wrong_gaps.mean():.4f} (std: {wrong_gaps.std():.4f})")
print(f"  t-test: t={t_stat:.3f}, p={p_val:.2e}")
print(f"  Effect size (Cohen's d): {(correct_gaps.mean() - wrong_gaps.mean()) / np.sqrt((correct_gaps.std()**2 + wrong_gaps.std()**2) / 2):.3f}")

# %%
# =============================================================================
# ANALYSIS 3: Gap between 2nd and 3rd largest
# =============================================================================
print("\n" + "="*70)
print("ANALYSIS 3: GAP BETWEEN 2ND AND 3RD LARGEST")
print("="*70)

fig, axes = plt.subplots(1, 3, figsize=(15, 5))

gap_2nd_3rd = features['gap_2nd_3rd']

# 3a. Distribution of gaps for correct vs wrong
ax1 = axes[0]
ax1.hist(gap_2nd_3rd[features['correct']], bins=50, alpha=0.7,
         label=f'Correct', color='green', density=True)
ax1.hist(gap_2nd_3rd[~features['correct']], bins=50, alpha=0.7,
         label=f'Wrong', color='red', density=True)
ax1.set_xlabel('Gap (2nd - 3rd largest)', fontsize=12)
ax1.set_ylabel('Density', fontsize=12)
ax1.set_title('Distribution of Gap by Correctness', fontsize=12)
ax1.legend()

# 3b. Accuracy vs gap (binned)
ax2 = axes[1]
gap_bins = np.linspace(0, gap_2nd_3rd.max(), 21)
gap_bin_centers = (gap_bins[:-1] + gap_bins[1:]) / 2
gap_bin_acc = []

for i in range(len(gap_bins) - 1):
    mask = (gap_2nd_3rd >= gap_bins[i]) & (gap_2nd_3rd < gap_bins[i+1])
    if mask.sum() > 0:
        gap_bin_acc.append(features['correct'][mask].mean())
    else:
        gap_bin_acc.append(np.nan)

ax2.bar(gap_bin_centers, gap_bin_acc, width=gap_bins[1]-gap_bins[0],
        color='steelblue', edgecolor='black', alpha=0.7)
ax2.axhline(y=accuracy, color='red', linestyle='--', linewidth=2, label=f'Overall: {accuracy:.1%}')
ax2.set_xlabel('Gap (2nd - 3rd largest)', fontsize=12)
ax2.set_ylabel('Accuracy', fontsize=12)
ax2.set_title('Accuracy vs Gap Size', fontsize=12)
ax2.set_ylim(0, 1)
ax2.legend()

# 3c. 2D heatmap: both gaps
ax3 = axes[2]
# Bin both gaps
n_bins_2d = 15
gap1_bins = np.linspace(0, np.percentile(gap_1st_2nd, 99), n_bins_2d + 1)
gap2_bins = np.linspace(0, np.percentile(gap_2nd_3rd, 99), n_bins_2d + 1)

acc_2d = np.zeros((n_bins_2d, n_bins_2d))
count_2d = np.zeros((n_bins_2d, n_bins_2d))

for i in range(n_bins_2d):
    for j in range(n_bins_2d):
        mask = (gap_1st_2nd >= gap1_bins[i]) & (gap_1st_2nd < gap1_bins[i+1]) & \
               (gap_2nd_3rd >= gap2_bins[j]) & (gap_2nd_3rd < gap2_bins[j+1])
        if mask.sum() > 10:  # Require minimum samples
            acc_2d[j, i] = features['correct'][mask].mean()
            count_2d[j, i] = mask.sum()
        else:
            acc_2d[j, i] = np.nan

im = ax3.imshow(acc_2d, origin='lower', aspect='auto', cmap='RdYlGn', vmin=0.5, vmax=1.0,
                extent=[gap1_bins[0], gap1_bins[-1], gap2_bins[0], gap2_bins[-1]])
ax3.set_xlabel('Gap (1st - 2nd)', fontsize=12)
ax3.set_ylabel('Gap (2nd - 3rd)', fontsize=12)
ax3.set_title('Accuracy Heatmap\n(both gaps)', fontsize=12)
plt.colorbar(im, ax=ax3, label='Accuracy')

plt.tight_layout()
plt.show()

# Statistical test
correct_gaps_2nd3rd = gap_2nd_3rd[features['correct']]
wrong_gaps_2nd3rd = gap_2nd_3rd[~features['correct']]
t_stat, p_val = stats.ttest_ind(correct_gaps_2nd3rd, wrong_gaps_2nd3rd)

print(f"\nStatistical Summary:")
print(f"  Correct predictions - Mean gap: {correct_gaps_2nd3rd.mean():.4f} (std: {correct_gaps_2nd3rd.std():.4f})")
print(f"  Wrong predictions   - Mean gap: {wrong_gaps_2nd3rd.mean():.4f} (std: {wrong_gaps_2nd3rd.std():.4f})")
print(f"  t-test: t={t_stat:.3f}, p={p_val:.2e}")
print(f"  Effect size (Cohen's d): {(correct_gaps_2nd3rd.mean() - wrong_gaps_2nd3rd.mean()) / np.sqrt((correct_gaps_2nd3rd.std()**2 + wrong_gaps_2nd3rd.std()**2) / 2):.3f}")

# %%
# =============================================================================
# ANALYSIS 4: 2nd largest value's distance from mean
# =============================================================================
print("\n" + "="*70)
print("ANALYSIS 4: 2ND LARGEST VALUE'S DISTANCE FROM MEAN")
print("="*70)

fig, axes = plt.subplots(1, 3, figsize=(15, 5))

val_from_mean = features['val_2nd_from_mean']

# 4a. Distribution for correct vs wrong
ax1 = axes[0]
ax1.hist(val_from_mean[features['correct']], bins=50, alpha=0.7,
         label=f'Correct', color='green', density=True)
ax1.hist(val_from_mean[~features['correct']], bins=50, alpha=0.7,
         label=f'Wrong', color='red', density=True)
ax1.set_xlabel('2nd largest - mean(x)', fontsize=12)
ax1.set_ylabel('Density', fontsize=12)
ax1.set_title('2nd Largest Distance from Mean', fontsize=12)
ax1.legend()

# 4b. Accuracy vs distance (binned)
ax2 = axes[1]
dist_bins = np.linspace(val_from_mean.min(), val_from_mean.max(), 21)
dist_bin_centers = (dist_bins[:-1] + dist_bins[1:]) / 2
dist_bin_acc = []

for i in range(len(dist_bins) - 1):
    mask = (val_from_mean >= dist_bins[i]) & (val_from_mean < dist_bins[i+1])
    if mask.sum() > 0:
        dist_bin_acc.append(features['correct'][mask].mean())
    else:
        dist_bin_acc.append(np.nan)

ax2.bar(dist_bin_centers, dist_bin_acc, width=dist_bins[1]-dist_bins[0],
        color='steelblue', edgecolor='black', alpha=0.7)
ax2.axhline(y=accuracy, color='red', linestyle='--', linewidth=2, label=f'Overall: {accuracy:.1%}')
ax2.set_xlabel('2nd largest - mean(x)', fontsize=12)
ax2.set_ylabel('Accuracy', fontsize=12)
ax2.set_title('Accuracy vs Distance from Mean', fontsize=12)
ax2.set_ylim(0, 1)
ax2.legend()

# 4c. Also look at absolute value of 2nd largest
ax3 = axes[2]
val_2nd = features['val_2nd_abs']
val_bins = np.linspace(val_2nd.min(), val_2nd.max(), 21)
val_bin_centers = (val_bins[:-1] + val_bins[1:]) / 2
val_bin_acc = []

for i in range(len(val_bins) - 1):
    mask = (val_2nd >= val_bins[i]) & (val_2nd < val_bins[i+1])
    if mask.sum() > 0:
        val_bin_acc.append(features['correct'][mask].mean())
    else:
        val_bin_acc.append(np.nan)

ax3.bar(val_bin_centers, val_bin_acc, width=val_bins[1]-val_bins[0],
        color='purple', edgecolor='black', alpha=0.7)
ax3.axhline(y=accuracy, color='red', linestyle='--', linewidth=2, label=f'Overall: {accuracy:.1%}')
ax3.set_xlabel('2nd largest value (absolute)', fontsize=12)
ax3.set_ylabel('Accuracy', fontsize=12)
ax3.set_title('Accuracy vs 2nd Largest Value', fontsize=12)
ax3.set_ylim(0, 1)
ax3.legend()

plt.tight_layout()
plt.show()

# Statistical test
correct_dist = val_from_mean[features['correct']]
wrong_dist = val_from_mean[~features['correct']]
t_stat, p_val = stats.ttest_ind(correct_dist, wrong_dist)

print(f"\nStatistical Summary (distance from mean):")
print(f"  Correct predictions - Mean: {correct_dist.mean():.4f} (std: {correct_dist.std():.4f})")
print(f"  Wrong predictions   - Mean: {wrong_dist.mean():.4f} (std: {wrong_dist.std():.4f})")
print(f"  t-test: t={t_stat:.3f}, p={p_val:.2e}")

# %%
# =============================================================================
# COMBINED ANALYSIS: Feature correlations with accuracy
# =============================================================================
print("\n" + "="*70)
print("COMBINED ANALYSIS: FEATURE CORRELATIONS")
print("="*70)

# Prepare features
feature_data = {
    'Gap 1st-2nd': features['gap_1st_2nd'],
    'Gap 2nd-3rd': features['gap_2nd_3rd'],
    '2nd from mean': features['val_2nd_from_mean'],
    '2nd abs value': features['val_2nd_abs'],
    'Input std': features['input_std'],
}

# Compute point-biserial correlations with correctness
correlations = {}
for fname, fdata in feature_data.items():
    corr, p_val = stats.pointbiserialr(features['correct'], fdata)
    correlations[fname] = (corr, p_val)

fig, axes = plt.subplots(1, 2, figsize=(12, 5))

# Correlation bar chart
ax1 = axes[0]
fnames = list(correlations.keys())
corrs = [correlations[f][0] for f in fnames]
sorted_idx = np.argsort(np.abs(corrs))[::-1]
colors = ['green' if c > 0 else 'red' for c in [corrs[i] for i in sorted_idx]]
ax1.barh(range(len(fnames)), [corrs[i] for i in sorted_idx], color=colors, edgecolor='black')
ax1.set_yticks(range(len(fnames)))
ax1.set_yticklabels([fnames[i] for i in sorted_idx])
ax1.set_xlabel('Point-biserial correlation with correctness', fontsize=12)
ax1.set_title('Feature Correlations with Accuracy\n(Green = helps, Red = hurts)', fontsize=12)
ax1.axvline(x=0, color='black', linestyle='-', linewidth=0.5)

# Accuracy curves for each feature by percentile
ax2 = axes[1]
for fname, color in zip(['Gap 1st-2nd', 'Gap 2nd-3rd', '2nd from mean'], ['blue', 'orange', 'green']):
    feat = feature_data[fname]
    bins = np.percentile(feat, np.linspace(0, 100, 21))
    bin_acc = []
    for j in range(len(bins) - 1):
        mask = (feat >= bins[j]) & (feat < bins[j+1])
        if mask.sum() > 0:
            bin_acc.append(features['correct'][mask].mean())
        else:
            bin_acc.append(np.nan)
    ax2.plot(np.linspace(0, 100, 20), bin_acc, '-o', markersize=4, label=fname, color=color, linewidth=2)

ax2.axhline(y=accuracy, color='red', linestyle='--', linewidth=2, alpha=0.5, label='Overall accuracy')
ax2.set_xlabel('Feature Percentile', fontsize=12)
ax2.set_ylabel('Accuracy', fontsize=12)
ax2.set_title('Accuracy vs Feature Percentile', fontsize=12)
ax2.legend(loc='lower right')
ax2.set_ylim(0.5, 1.0)
ax2.grid(True, alpha=0.3)

plt.tight_layout()
plt.show()

print(f"\nPoint-biserial correlations with correctness:")
for i in sorted_idx:
    fname = fnames[i]
    corr, p_val = correlations[fname]
    sign = '+' if corr > 0 else ''
    print(f"  {fname:<20}: r = {sign}{corr:.4f}  (p = {p_val:.2e})")

# %%
# =============================================================================
# SUMMARY: Error characteristics
# =============================================================================
print("\n" + "="*70)
print("SUMMARY: CHARACTERISTICS OF ERRORS")
print("="*70)

wrong_mask = ~features['correct']
n_wrong = wrong_mask.sum()

print(f"\nTotal errors: {n_wrong} / {N_samples} ({n_wrong/N_samples:.2%})")

print(f"\n1. POSITION BIAS:")
for pos in range(n):
    mask = features['target_position'] == pos
    err_rate = (~features['correct'] & mask).sum() / mask.sum() if mask.sum() > 0 else 0
    print(f"   Position {pos}: {err_rate:.2%} error rate")

print(f"\n2. GAP (1st - 2nd) ANALYSIS:")
print(f"   Errors occur more when gap is SMALL (close competition at top)")
print(f"   Mean gap for correct: {gap_1st_2nd[features['correct']].mean():.3f}")
print(f"   Mean gap for wrong:   {gap_1st_2nd[~features['correct']].mean():.3f}")

print(f"\n3. GAP (2nd - 3rd) ANALYSIS:")
print(f"   Errors occur more when gap is SMALL (2nd place not well separated)")
print(f"   Mean gap for correct: {gap_2nd_3rd[features['correct']].mean():.3f}")
print(f"   Mean gap for wrong:   {gap_2nd_3rd[~features['correct']].mean():.3f}")

print(f"\n4. DISTANCE FROM MEAN:")
print(f"   2nd largest tends to be closer to mean in error cases")
print(f"   Mean distance for correct: {val_from_mean[features['correct']].mean():.3f}")
print(f"   Mean distance for wrong:   {val_from_mean[~features['correct']].mean():.3f}")

# Most common error pattern
print(f"\n5. MOST COMMON ERROR PATTERNS:")
error_pairs = list(zip(features['target_position'][wrong_mask],
                       features['pred_position'][wrong_mask]))
from collections import Counter
error_counts = Counter(error_pairs)
for (true, pred), count in error_counts.most_common(5):
    print(f"   True={true} -> Pred={pred}: {count} errors ({count/n_wrong:.1%} of all errors)")

# %%
# Visualize the hardest examples
print("\n" + "="*70)
print("HARDEST EXAMPLES (smallest gaps)")
print("="*70)

# Find examples with smallest gap_1st_2nd that were wrong
wrong_idx = np.where(wrong_mask)[0]
gap_at_wrong = gap_1st_2nd[wrong_idx]
hardest_idx = wrong_idx[np.argsort(gap_at_wrong)[:5]]

fig, axes = plt.subplots(1, 5, figsize=(20, 4))
for i, idx in enumerate(hardest_idx):
    ax = axes[i]
    x_i = x_all[idx].numpy()
    sorted_order = np.argsort(x_i)[::-1]

    colors = ['gray'] * n
    colors[sorted_order[0]] = 'red'  # max
    colors[sorted_order[1]] = 'orange'  # 2nd max (target)

    ax.bar(range(n), x_i, color=colors, edgecolor='black')
    ax.set_title(f'Target={features["target_position"][idx]}, Pred={features["pred_position"][idx]}\n'
                 f'Gap 1-2: {gap_1st_2nd[idx]:.3f}', fontsize=10)
    ax.set_xlabel('Position')
    ax.set_ylabel('Value')
    ax.axhline(y=0, color='k', linewidth=0.5)

plt.suptitle('Hardest Wrong Examples (smallest gap between 1st and 2nd)', fontsize=12)
plt.tight_layout()
plt.show()

# %%
