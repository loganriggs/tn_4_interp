# %%
"""
Sweep over 5 seeds x 5 pruning thresholds for L1 pruning experiment.
"""

import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).parent))
from bilinear_residual_rmsnorm import BilinearResidualRMSNorm, task_2nd_argmax

# %%
# Configuration
n = 4
num_layers = 2
rank = 4
device = "cuda" if torch.cuda.is_available() else "cpu"

seeds = [0, 1, 2, 3, 4]
thresholds = [0.05, 0.10, 0.15, 0.20, 0.25]

print(f"Config: n={n}, layers={num_layers}, rank={rank}, device={device}")
print(f"Seeds: {seeds}")
print(f"Thresholds: {thresholds}")

# %%
def train_and_prune(seed, prune_threshold, verbose=False):
    """
    Train with L1, prune, and fine-tune. Returns dict of metrics.
    """
    torch.manual_seed(seed)
    model = BilinearResidualRMSNorm(n, num_layers, rank, use_rmsnorm=True).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.01, weight_decay=0.001)

    # Phase 1a: Warm-up (no L1)
    for step in range(5001):
        model.train()
        x = torch.randn(128, n, device=device)
        targets = task_2nd_argmax(x)
        logits = model(x)
        loss = nn.functional.cross_entropy(logits, targets)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    # Eval after warmup
    model.eval()
    with torch.no_grad():
        x_eval = torch.randn(10000, n, device=device)
        targets_eval = task_2nd_argmax(x_eval)
        logits_eval = model(x_eval)
        warmup_acc = (logits_eval.argmax(-1) == targets_eval).float().mean().item()

    # Phase 1b: L1 training
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.005, weight_decay=0.001)
    l1_lambda = 0.001

    for step in range(10001):
        model.train()
        x = torch.randn(128, n, device=device)
        targets = task_2nd_argmax(x)
        logits = model(x)
        ce_loss = nn.functional.cross_entropy(logits, targets)

        l1_loss = 0
        for layer in model.layers:
            l1_loss += layer.L.abs().sum()
            l1_loss += layer.R.abs().sum()
            l1_loss += layer.D.abs().sum()

        loss = ce_loss + l1_lambda * l1_loss
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    # Eval after L1
    model.eval()
    with torch.no_grad():
        x_eval = torch.randn(10000, n, device=device)
        targets_eval = task_2nd_argmax(x_eval)
        logits_eval = model(x_eval)
        l1_acc = (logits_eval.argmax(-1) == targets_eval).float().mean().item()

    # Phase 2: Prune
    masks = {}
    with torch.no_grad():
        for i, layer in enumerate(model.layers):
            masks[f'L{i}'] = (layer.L.abs() >= prune_threshold).float()
            masks[f'R{i}'] = (layer.R.abs() >= prune_threshold).float()
            masks[f'D{i}'] = (layer.D.abs() >= prune_threshold).float()
            layer.L.data *= masks[f'L{i}']
            layer.R.data *= masks[f'R{i}']
            layer.D.data *= masks[f'D{i}']

    total_params = sum(m.numel() for m in masks.values())
    remaining_params = sum(m.sum().item() for m in masks.values())
    pruned_frac = 1 - remaining_params / total_params

    # Eval after prune
    model.eval()
    with torch.no_grad():
        x_eval = torch.randn(10000, n, device=device)
        targets_eval = task_2nd_argmax(x_eval)
        logits_eval = model(x_eval)
        prune_acc = (logits_eval.argmax(-1) == targets_eval).float().mean().item()

    # Phase 3: Fine-tune
    optimizer_ft = torch.optim.AdamW(model.parameters(), lr=0.005, weight_decay=0.001)

    for step in range(5001):
        model.train()
        x = torch.randn(128, n, device=device)
        targets = task_2nd_argmax(x)
        logits = model(x)
        loss = nn.functional.cross_entropy(logits, targets)

        optimizer_ft.zero_grad()
        loss.backward()

        # Mask gradients
        with torch.no_grad():
            for i, layer in enumerate(model.layers):
                if layer.L.grad is not None:
                    layer.L.grad *= masks[f'L{i}']
                if layer.R.grad is not None:
                    layer.R.grad *= masks[f'R{i}']
                if layer.D.grad is not None:
                    layer.D.grad *= masks[f'D{i}']

        optimizer_ft.step()

        # Re-apply mask
        with torch.no_grad():
            for i, layer in enumerate(model.layers):
                layer.L.data *= masks[f'L{i}']
                layer.R.data *= masks[f'R{i}']
                layer.D.data *= masks[f'D{i}']

    # Final eval
    model.eval()
    with torch.no_grad():
        x_eval = torch.randn(10000, n, device=device)
        targets_eval = task_2nd_argmax(x_eval)
        logits_eval = model(x_eval)
        final_acc = (logits_eval.argmax(-1) == targets_eval).float().mean().item()

    return {
        'seed': seed,
        'threshold': prune_threshold,
        'warmup_acc': warmup_acc,
        'l1_acc': l1_acc,
        'prune_acc': prune_acc,
        'final_acc': final_acc,
        'pruned_frac': pruned_frac,
    }

# %%
# Run sweep
print("\n" + "="*70)
print("RUNNING SWEEP: 5 seeds x 5 thresholds = 25 experiments")
print("="*70)

all_results = []

for seed in seeds:
    for thresh in thresholds:
        print(f"\nSeed {seed}, Threshold {thresh}...", end=" ", flush=True)
        result = train_and_prune(seed, thresh)
        all_results.append(result)
        print(f"Final: {result['final_acc']:.1%}, Pruned: {result['pruned_frac']:.1%}")

# %%
# Create results table
print("\n" + "="*70)
print("RESULTS TABLE")
print("="*70)

# Convert to numpy arrays for easier manipulation
results_matrix = np.zeros((len(seeds), len(thresholds)))
pruned_matrix = np.zeros((len(seeds), len(thresholds)))

for r in all_results:
    i = seeds.index(r['seed'])
    j = thresholds.index(r['threshold'])
    results_matrix[i, j] = r['final_acc']
    pruned_matrix[i, j] = r['pruned_frac']

# Print table
print(f"\n{'Final Accuracy (%)':^60}")
print(f"{'Seed':<8}", end="")
for t in thresholds:
    print(f"t={t:<6}", end="  ")
print("Mean")
print("-" * 60)

for i, seed in enumerate(seeds):
    print(f"{seed:<8}", end="")
    for j in range(len(thresholds)):
        print(f"{100*results_matrix[i,j]:<8.1f}", end="")
    print(f"{100*results_matrix[i,:].mean():.1f}")

print("-" * 60)
print(f"{'Mean':<8}", end="")
for j in range(len(thresholds)):
    print(f"{100*results_matrix[:,j].mean():<8.1f}", end="")
print(f"{100*results_matrix.mean():.1f}")

# Pruning fractions
print(f"\n{'Pruned Fraction (%)':^60}")
print(f"{'Seed':<8}", end="")
for t in thresholds:
    print(f"t={t:<6}", end="  ")
print("Mean")
print("-" * 60)

for i, seed in enumerate(seeds):
    print(f"{seed:<8}", end="")
    for j in range(len(thresholds)):
        print(f"{100*pruned_matrix[i,j]:<8.1f}", end="")
    print(f"{100*pruned_matrix[i,:].mean():.1f}")

print("-" * 60)
print(f"{'Mean':<8}", end="")
for j in range(len(thresholds)):
    print(f"{100*pruned_matrix[:,j].mean():<8.1f}", end="")
print(f"{100*pruned_matrix.mean():.1f}")

# %%
# Visualization
fig, axes = plt.subplots(1, 3, figsize=(15, 5))

# Heatmap of final accuracy
ax1 = axes[0]
im1 = ax1.imshow(results_matrix * 100, cmap='RdYlGn', vmin=50, vmax=90, aspect='auto')
ax1.set_xticks(range(len(thresholds)))
ax1.set_xticklabels([f'{t}' for t in thresholds])
ax1.set_yticks(range(len(seeds)))
ax1.set_yticklabels([f'Seed {s}' for s in seeds])
ax1.set_xlabel('Pruning Threshold')
ax1.set_title('Final Accuracy (%)')
for i in range(len(seeds)):
    for j in range(len(thresholds)):
        ax1.text(j, i, f'{100*results_matrix[i,j]:.1f}', ha='center', va='center', fontsize=10)
plt.colorbar(im1, ax=ax1, shrink=0.8)

# Heatmap of pruned fraction
ax2 = axes[1]
im2 = ax2.imshow(pruned_matrix * 100, cmap='Blues', vmin=0, vmax=50, aspect='auto')
ax2.set_xticks(range(len(thresholds)))
ax2.set_xticklabels([f'{t}' for t in thresholds])
ax2.set_yticks(range(len(seeds)))
ax2.set_yticklabels([f'Seed {s}' for s in seeds])
ax2.set_xlabel('Pruning Threshold')
ax2.set_title('Pruned Fraction (%)')
for i in range(len(seeds)):
    for j in range(len(thresholds)):
        ax2.text(j, i, f'{100*pruned_matrix[i,j]:.1f}', ha='center', va='center', fontsize=10)
plt.colorbar(im2, ax=ax2, shrink=0.8)

# Accuracy vs pruning trade-off
ax3 = axes[2]
colors = plt.cm.tab10(np.linspace(0, 1, len(seeds)))
for i, seed in enumerate(seeds):
    ax3.plot(pruned_matrix[i, :] * 100, results_matrix[i, :] * 100, 'o-',
             color=colors[i], label=f'Seed {seed}', markersize=8, linewidth=2)

# Add mean line
ax3.plot(pruned_matrix.mean(axis=0) * 100, results_matrix.mean(axis=0) * 100, 'k--',
         linewidth=3, label='Mean', marker='s', markersize=10)

ax3.set_xlabel('Pruned Fraction (%)')
ax3.set_ylabel('Final Accuracy (%)')
ax3.set_title('Accuracy vs Pruning Trade-off')
ax3.legend(loc='lower left')
ax3.grid(True, alpha=0.3)
ax3.set_xlim(0, 50)
ax3.set_ylim(50, 90)

plt.tight_layout()
plt.show()

# %%
# Summary statistics
print("\n" + "="*70)
print("SUMMARY STATISTICS")
print("="*70)

print(f"\nAcross all 25 experiments:")
print(f"  Accuracy: {100*results_matrix.mean():.1f}% +/- {100*results_matrix.std():.1f}%")
print(f"  Range: {100*results_matrix.min():.1f}% - {100*results_matrix.max():.1f}%")

print(f"\nBest configurations (>= 80% accuracy):")
for r in sorted(all_results, key=lambda x: x['final_acc'], reverse=True):
    if r['final_acc'] >= 0.80:
        print(f"  Seed {r['seed']}, t={r['threshold']}: {r['final_acc']:.1%} acc, {r['pruned_frac']:.1%} pruned")

print(f"\nMost aggressive pruning with >= 75% accuracy:")
aggressive = [r for r in all_results if r['final_acc'] >= 0.75]
if aggressive:
    best = max(aggressive, key=lambda x: x['pruned_frac'])
    print(f"  Seed {best['seed']}, t={best['threshold']}: {best['final_acc']:.1%} acc, {best['pruned_frac']:.1%} pruned")

# %%
# Save results
import pickle
save_path = Path("sweep_l1_pruning_results.pkl")
with open(save_path, 'wb') as f:
    pickle.dump({
        'all_results': all_results,
        'results_matrix': results_matrix,
        'pruned_matrix': pruned_matrix,
        'seeds': seeds,
        'thresholds': thresholds,
        'config': {'n': n, 'num_layers': num_layers, 'rank': rank},
    }, f)
print(f"\nSaved results to {save_path}")

# %%
