# %%
"""
L1 Pruning + Ablation for 1-Layer Symmetric Bilinear Model

1. Train with L1 penalty
2. Prune smallest weights
3. Fine-tune
4. Visualize weights
5. Ablation: x vs bilinear contribution
"""

import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

# %%
# =============================================================================
# MODEL DEFINITION
# =============================================================================

class RMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        rms = torch.sqrt((x ** 2).mean(dim=-1, keepdim=True) + self.eps)
        return self.weight * (x / rms)


class SymmetricBilinear1Layer(nn.Module):
    def __init__(self, n: int, rank: int = 4):
        super().__init__()
        self.n = n
        self.rank = rank
        self.norm = RMSNorm(n)
        self.L = nn.Parameter(torch.randn(rank, n) * 0.1)
        self.D = nn.Parameter(torch.randn(n, rank) * 0.1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.norm(x)
        Lh = h @ self.L.T
        return x + (Lh ** 2) @ self.D.T


def task_2nd_argmax(x):
    return x.argsort(-1)[..., -2]


# %%
# Configuration
n = 4
rank = 4
seed = 4
device = "cuda" if torch.cuda.is_available() else "cpu"

thresholds = [0.05, 0.10, 0.15, 0.20, 0.25]

print(f"Config: n={n}, rank={rank}, seed={seed}")
print(f"Thresholds: {thresholds}")

# %%
# =============================================================================
# PHASE 1a: Warm-up training (no L1)
# =============================================================================
print("\n" + "="*60)
print("PHASE 1a: Warm-up training (no L1)")
print("="*60)

torch.manual_seed(seed)
model = SymmetricBilinear1Layer(n, rank).to(device)
optimizer = torch.optim.AdamW(model.parameters(), lr=0.01, weight_decay=0.001)

for step in range(5001):
    model.train()
    x = torch.randn(128, n, device=device)
    targets = task_2nd_argmax(x)
    logits = model(x)
    loss = nn.functional.cross_entropy(logits, targets)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    if step % 1000 == 0:
        model.eval()
        with torch.no_grad():
            x_eval = torch.randn(10000, n, device=device)
            targets_eval = task_2nd_argmax(x_eval)
            logits_eval = model(x_eval)
            eval_acc = (logits_eval.argmax(-1) == targets_eval).float().mean().item()
        print(f"Step {step:5d}: Loss={loss.item():.4f}, Acc={eval_acc:.3f}")

warmup_acc = eval_acc
print(f"\nWarm-up accuracy: {warmup_acc:.1%}")

# %%
# =============================================================================
# PHASE 1b: L1 penalty training
# =============================================================================
print("\n" + "="*60)
print("PHASE 1b: L1 penalty training")
print("="*60)

optimizer = torch.optim.AdamW(model.parameters(), lr=0.005, weight_decay=0.001)
l1_lambda = 0.001

for step in range(10001):
    model.train()
    x = torch.randn(128, n, device=device)
    targets = task_2nd_argmax(x)
    logits = model(x)
    ce_loss = nn.functional.cross_entropy(logits, targets)

    # L1 on L and D matrices
    l1_loss = model.L.abs().sum() + model.D.abs().sum()
    loss = ce_loss + l1_lambda * l1_loss

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    if step % 2000 == 0:
        model.eval()
        with torch.no_grad():
            x_eval = torch.randn(10000, n, device=device)
            targets_eval = task_2nd_argmax(x_eval)
            logits_eval = model(x_eval)
            eval_acc = (logits_eval.argmax(-1) == targets_eval).float().mean().item()
        print(f"Step {step:5d}: CE={ce_loss.item():.4f}, L1={l1_loss.item():.2f}, Acc={eval_acc:.3f}")

l1_acc = eval_acc
print(f"\nAfter L1 training: {l1_acc:.1%}")

# Save L1-trained state
l1_state = {k: v.clone() for k, v in model.state_dict().items()}

# %%
# =============================================================================
# PHASE 2 & 3: Prune + Fine-tune at different thresholds
# =============================================================================
print("\n" + "="*60)
print("PHASE 2 & 3: Pruning + Fine-tuning")
print("="*60)

results = []

for thresh in thresholds:
    print(f"\n--- Threshold {thresh} ---")

    # Reload L1-trained model
    model_pruned = SymmetricBilinear1Layer(n, rank).to(device)
    model_pruned.load_state_dict(l1_state)

    # Create masks and prune
    masks = {}
    with torch.no_grad():
        masks['L'] = (model_pruned.L.abs() >= thresh).float()
        masks['D'] = (model_pruned.D.abs() >= thresh).float()
        model_pruned.L.data *= masks['L']
        model_pruned.D.data *= masks['D']

    total_params = sum(m.numel() for m in masks.values())
    remaining_params = sum(m.sum().item() for m in masks.values())
    pruned_frac = 1 - remaining_params / total_params

    # Eval after prune
    model_pruned.eval()
    with torch.no_grad():
        x_eval = torch.randn(10000, n, device=device)
        targets_eval = task_2nd_argmax(x_eval)
        logits_eval = model_pruned(x_eval)
        prune_acc = (logits_eval.argmax(-1) == targets_eval).float().mean().item()

    print(f"  After prune: {prune_acc:.1%} acc, {pruned_frac:.1%} pruned")

    # Fine-tune
    optimizer_ft = torch.optim.AdamW(model_pruned.parameters(), lr=0.005, weight_decay=0.001)

    for step in range(5001):
        model_pruned.train()
        x = torch.randn(128, n, device=device)
        targets = task_2nd_argmax(x)
        logits = model_pruned(x)
        loss = nn.functional.cross_entropy(logits, targets)

        optimizer_ft.zero_grad()
        loss.backward()

        # Mask gradients
        with torch.no_grad():
            if model_pruned.L.grad is not None:
                model_pruned.L.grad *= masks['L']
            if model_pruned.D.grad is not None:
                model_pruned.D.grad *= masks['D']

        optimizer_ft.step()

        # Re-apply mask
        with torch.no_grad():
            model_pruned.L.data *= masks['L']
            model_pruned.D.data *= masks['D']

    # Final eval
    model_pruned.eval()
    with torch.no_grad():
        x_eval = torch.randn(10000, n, device=device)
        targets_eval = task_2nd_argmax(x_eval)
        logits_eval = model_pruned(x_eval)
        final_acc = (logits_eval.argmax(-1) == targets_eval).float().mean().item()

    print(f"  After fine-tune: {final_acc:.1%} acc")

    # Per-matrix sparsity
    sparsity = {
        'L': (model_pruned.L == 0).sum().item() / model_pruned.L.numel(),
        'D': (model_pruned.D == 0).sum().item() / model_pruned.D.numel(),
    }

    results.append({
        'threshold': thresh,
        'prune_acc': prune_acc,
        'final_acc': final_acc,
        'pruned_frac': pruned_frac,
        'sparsity': sparsity,
        'state_dict': {k: v.cpu() for k, v in model_pruned.state_dict().items()},
    })

# %%
# =============================================================================
# RESULTS SUMMARY
# =============================================================================
print("\n" + "="*60)
print("RESULTS SUMMARY")
print("="*60)

print(f"\n{'Threshold':<12} {'Pruned %':<12} {'After Prune':<14} {'After Fine-tune':<16}")
print("-" * 55)
for r in results:
    print(f"{r['threshold']:<12} {100*r['pruned_frac']:<12.1f} {100*r['prune_acc']:<14.1f} {100*r['final_acc']:<16.1f}")

print(f"\nBaseline (after L1): {l1_acc:.1%}")

# %%
# Per-matrix sparsity
print("\n" + "="*60)
print("PER-MATRIX SPARSITY")
print("="*60)

print(f"\n{'Threshold':<10} {'L':<8} {'D':<8}")
print("-" * 30)
for r in results:
    sp = r['sparsity']
    print(f"{r['threshold']:<10} {100*sp['L']:<8.0f} {100*sp['D']:<8.0f}")

# %%
# =============================================================================
# VISUALIZE BEST PRUNED MODEL WEIGHTS (t=0.15)
# =============================================================================
print("\n" + "="*60)
print("BEST PRUNED MODEL WEIGHTS")
print("="*60)

# Choose t=0.15 or best available
best_thresh = 0.15
best_result = [r for r in results if r['threshold'] == best_thresh]
if not best_result:
    best_result = [results[len(results)//2]]
best_result = best_result[0]
best_state = best_result['state_dict']

L = best_state['L']
D = best_state['D']
norm_weight = best_state['norm.weight']

print(f"\nUsing threshold t={best_result['threshold']}")
print(f"Accuracy: {best_result['final_acc']:.1%}")
print(f"Pruned: {best_result['pruned_frac']:.1%}")

print(f"\nRMSNorm weights: {norm_weight.numpy().round(3)}")

fig, axes = plt.subplots(1, 3, figsize=(15, 4))

def plot_sparse_mat(ax, mat, title):
    vmax = max(abs(mat.min()), abs(mat.max())) if mat.max() != mat.min() else 1
    im = ax.imshow(mat, cmap='RdBu_r', vmin=-vmax, vmax=vmax, aspect='auto')
    ax.set_title(title)
    for i in range(mat.shape[0]):
        for j in range(mat.shape[1]):
            val = mat[i, j]
            if val == 0:
                ax.text(j, i, '0', ha='center', va='center', fontsize=9, color='gray')
            else:
                color = 'white' if abs(val) > vmax * 0.6 else 'black'
                ax.text(j, i, f'{val:.2f}', ha='center', va='center', fontsize=8, color=color)
    plt.colorbar(im, ax=ax, shrink=0.8)

plot_sparse_mat(axes[0], L.numpy(), f"L (pruned, {100*best_result['sparsity']['L']:.0f}% sparse)")
plot_sparse_mat(axes[1], D.numpy(), f"D (pruned, {100*best_result['sparsity']['D']:.0f}% sparse)")

# D @ L composition
DL = D @ L
plot_sparse_mat(axes[2], DL.numpy(), "D @ L (effective)")

plt.suptitle(f'1-Layer Pruned Weights (t={best_result["threshold"]}, {best_result["final_acc"]:.1%} acc)', fontsize=12)
plt.tight_layout()
plt.show()

# %%
# =============================================================================
# ABLATION ANALYSIS (for 1-layer)
# =============================================================================
print("\n" + "="*60)
print("ABLATION ANALYSIS")
print("="*60)
print("Decomposition: output = x + bilinear(norm(x))")
print("  where bilinear(h) = D @ (L @ h)²")

def rmsnorm(x, weight, eps=1e-6):
    rms = torch.sqrt((x ** 2).mean(dim=-1, keepdim=True) + eps)
    return weight * (x / rms)

def compute_components(x, L, D, norm_weight):
    """Compute x and bilinear contributions separately."""
    h = rmsnorm(x, norm_weight)
    Lh = h @ L.T
    bilinear = (Lh ** 2) @ D.T
    full = x + bilinear
    return {
        'x': x,
        'bilinear': bilinear,
        'full': full,
    }

def compute_accuracy(logits, x):
    targets = task_2nd_argmax(x)
    preds = logits.argmax(dim=1)
    return (preds == targets).float().mean().item()

# Generate test data
torch.manual_seed(123)
x_eval = torch.randn(10000, n)
comps = compute_components(x_eval, L, D, norm_weight)

print(f"\nFull model accuracy: {compute_accuracy(comps['full'], x_eval):.1%}")

print(f"\nComponent magnitudes (L2 norm, avg):")
for name in ['x', 'bilinear']:
    norm = comps[name].norm(dim=1).mean().item()
    print(f"  {name}: {norm:.3f}")

print(f"\nAblation results:")
print(f"{'Configuration':<25} {'Accuracy':<12}")
print("-" * 40)

# Full
full_acc = compute_accuracy(comps['full'], x_eval)
print(f"{'Full (x + bilinear)':<25} {full_acc:.1%}")

# Remove bilinear
no_bilinear_acc = compute_accuracy(comps['x'], x_eval)
print(f"{'Only x (no bilinear)':<25} {no_bilinear_acc:.1%}")

# Remove x
no_x_acc = compute_accuracy(comps['bilinear'], x_eval)
print(f"{'Only bilinear (no x)':<25} {no_x_acc:.1%}")

# %%
# Visualize ablation
fig, axes = plt.subplots(1, 2, figsize=(12, 5))

# Bar chart of ablation results
ax1 = axes[0]
configs = ['Full\n(x + bilinear)', 'Only x\n(no bilinear)', 'Only bilinear\n(no x)']
accs = [full_acc, no_bilinear_acc, no_x_acc]
colors = ['green', 'orange', 'steelblue']
bars = ax1.bar(configs, [a*100 for a in accs], color=colors, edgecolor='black')
ax1.axhline(y=25, color='gray', linestyle=':', linewidth=1, label='Random (25%)')
ax1.set_ylabel('Accuracy (%)')
ax1.set_title('1-Layer Ablation: Which component matters?')
ax1.set_ylim(0, 100)
ax1.legend()
for bar, acc in zip(bars, accs):
    ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 2, f'{acc:.1%}',
             ha='center', fontsize=11, fontweight='bold')

# Component contribution at target position
ax2 = axes[1]
targets = task_2nd_argmax(x_eval)
x_at_target = comps['x'][torch.arange(len(targets)), targets].numpy()
bilinear_at_target = comps['bilinear'][torch.arange(len(targets)), targets].numpy()

ax2.scatter(x_at_target[:500], bilinear_at_target[:500], alpha=0.5, s=10)
ax2.axhline(y=0, color='gray', linestyle='-', linewidth=0.5)
ax2.axvline(x=0, color='gray', linestyle='-', linewidth=0.5)
ax2.set_xlabel('x component at target')
ax2.set_ylabel('Bilinear component at target')
ax2.set_title('Component values at 2nd-argmax position')

plt.tight_layout()
plt.show()

# %%
# =============================================================================
# INTERPRETATION
# =============================================================================
print("\n" + "="*60)
print("INTERPRETATION")
print("="*60)

print(f"""
1-Layer Model Summary:
- Final accuracy: {best_result['final_acc']:.1%}
- Pruned fraction: {best_result['pruned_frac']:.1%}

Ablation Findings:
- Full model: {full_acc:.1%}
- x only: {no_bilinear_acc:.1%} (the residual connection alone is {'helpful' if no_bilinear_acc > 25 else 'useless'})
- bilinear only: {no_x_acc:.1%}

Key Insight:
The 1-layer model achieves {best_result['final_acc']:.1%} accuracy on 2nd-argmax,
which is significantly better than random (25%).

Comparing with 2-layer model ablation:
- 2-layer "A only" (L2 seeing x only): ~50%
- 1-layer model (trained for x only): {best_result['final_acc']:.1%}

This shows that when trained specifically for the 1-layer setting,
the model can learn better weights than what layer 2 uses in the
2-layer setting (where L2 specializes for working with r1).
""")

# %%
# Save results
import pickle
save_path = Path("prune_symmetric_1layer_results.pkl")
with open(save_path, 'wb') as f:
    pickle.dump({
        'results': results,
        'l1_state': l1_state,
        'l1_acc': l1_acc,
        'warmup_acc': warmup_acc,
        'config': {'n': n, 'rank': rank, 'seed': seed},
    }, f)
print(f"\nSaved to {save_path}")

# %%
