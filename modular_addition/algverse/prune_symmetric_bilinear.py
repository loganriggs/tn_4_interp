# %%
"""
L1 Pruning on Symmetric Bilinear Model: D @ (L @ x)²

1. Train with L1 penalty
2. Prune smallest weights
3. Fine-tune
"""

import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

# %%
# =============================================================================
# MODEL DEFINITION (same as train_symmetric_bilinear.py)
# =============================================================================

class RMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        rms = torch.sqrt((x ** 2).mean(dim=-1, keepdim=True) + self.eps)
        return self.weight * (x / rms)


class SymmetricBilinearLayer(nn.Module):
    def __init__(self, dim: int, rank: int):
        super().__init__()
        self.L = nn.Parameter(torch.randn(rank, dim) * 0.1)
        self.D = nn.Parameter(torch.randn(dim, rank) * 0.1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        Lx = x @ self.L.T
        return (Lx ** 2) @ self.D.T


class SymmetricBilinearResidual(nn.Module):
    def __init__(self, n: int, num_layers: int = 2, rank: int = 4):
        super().__init__()
        self.n = n
        self.num_layers = num_layers
        self.rank = rank
        self.norms = nn.ModuleList([RMSNorm(n) for _ in range(num_layers)])
        self.layers = nn.ModuleList([
            SymmetricBilinearLayer(n, rank) for _ in range(num_layers)
        ])

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = x
        for i in range(self.num_layers):
            h_norm = self.norms[i](h)
            h = h + self.layers[i](h_norm)
        return h


def task_2nd_argmax(x):
    return x.argsort(-1)[..., -2]


# %%
# Configuration
n = 4
num_layers = 2
rank = 4
seed = 4
device = "cuda" if torch.cuda.is_available() else "cpu"

thresholds = [0.05, 0.10, 0.15, 0.20, 0.25]

print(f"Config: n={n}, layers={num_layers}, rank={rank}, seed={seed}")
print(f"Thresholds: {thresholds}")

# %%
# =============================================================================
# PHASE 1a: Warm-up training (no L1)
# =============================================================================
print("\n" + "="*60)
print("PHASE 1a: Warm-up training (no L1)")
print("="*60)

torch.manual_seed(seed)
model = SymmetricBilinearResidual(n, num_layers, rank).to(device)
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
    l1_loss = 0
    for layer in model.layers:
        l1_loss += layer.L.abs().sum()
        l1_loss += layer.D.abs().sum()

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
    model_pruned = SymmetricBilinearResidual(n, num_layers, rank).to(device)
    model_pruned.load_state_dict(l1_state)

    # Create masks and prune
    masks = {}
    with torch.no_grad():
        for i, layer in enumerate(model_pruned.layers):
            masks[f'L{i}'] = (layer.L.abs() >= thresh).float()
            masks[f'D{i}'] = (layer.D.abs() >= thresh).float()
            layer.L.data *= masks[f'L{i}']
            layer.D.data *= masks[f'D{i}']

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
            for i, layer in enumerate(model_pruned.layers):
                if layer.L.grad is not None:
                    layer.L.grad *= masks[f'L{i}']
                if layer.D.grad is not None:
                    layer.D.grad *= masks[f'D{i}']

        optimizer_ft.step()

        # Re-apply mask
        with torch.no_grad():
            for i, layer in enumerate(model_pruned.layers):
                layer.L.data *= masks[f'L{i}']
                layer.D.data *= masks[f'D{i}']

    # Final eval
    model_pruned.eval()
    with torch.no_grad():
        x_eval = torch.randn(10000, n, device=device)
        targets_eval = task_2nd_argmax(x_eval)
        logits_eval = model_pruned(x_eval)
        final_acc = (logits_eval.argmax(-1) == targets_eval).float().mean().item()

    print(f"  After fine-tune: {final_acc:.1%} acc")

    # Per-matrix sparsity
    sparsity = {}
    for i, layer in enumerate(model_pruned.layers):
        sparsity[f'L{i+1}'] = (layer.L == 0).sum().item() / layer.L.numel()
        sparsity[f'D{i+1}'] = (layer.D == 0).sum().item() / layer.D.numel()

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
# Visualization
fig, axes = plt.subplots(1, 3, figsize=(15, 5))

thresh_vals = [r['threshold'] for r in results]
prune_accs = [r['prune_acc'] for r in results]
final_accs = [r['final_acc'] for r in results]
pruned_fracs = [r['pruned_frac'] for r in results]

# Accuracy vs threshold
ax1 = axes[0]
ax1.plot(thresh_vals, [100*a for a in prune_accs], 'ro--', markersize=10, linewidth=2, label='After Prune')
ax1.plot(thresh_vals, [100*a for a in final_accs], 'go-', markersize=10, linewidth=2, label='After Fine-tune')
ax1.axhline(y=100*l1_acc, color='blue', linestyle='--', linewidth=2, label=f'L1 baseline: {100*l1_acc:.1f}%')
ax1.axhline(y=80, color='gray', linestyle=':', linewidth=1, label='80% target')
ax1.set_xlabel('Pruning Threshold')
ax1.set_ylabel('Accuracy (%)')
ax1.set_title('Accuracy vs Threshold')
ax1.legend()
ax1.grid(True, alpha=0.3)
ax1.set_ylim(50, 100)

# Pruned fraction
ax2 = axes[1]
ax2.bar(thresh_vals, [100*p for p in pruned_fracs], width=0.03, color='steelblue', edgecolor='black')
ax2.set_xlabel('Pruning Threshold')
ax2.set_ylabel('Pruned %')
ax2.set_title('Weights Pruned')
ax2.grid(True, alpha=0.3)

# Trade-off
ax3 = axes[2]
ax3.plot([100*p for p in pruned_fracs], [100*a for a in final_accs], 'go-', markersize=12, linewidth=2)
for p, a, t in zip(pruned_fracs, final_accs, thresh_vals):
    ax3.annotate(f't={t}', (100*p, 100*a), textcoords='offset points', xytext=(5, 5), fontsize=9)
ax3.axhline(y=80, color='gray', linestyle=':', linewidth=1)
ax3.set_xlabel('Pruned %')
ax3.set_ylabel('Final Accuracy %')
ax3.set_title('Accuracy vs Sparsity')
ax3.grid(True, alpha=0.3)
ax3.set_ylim(50, 100)

plt.tight_layout()
plt.show()

# %%
# Per-matrix sparsity
print("\n" + "="*60)
print("PER-MATRIX SPARSITY")
print("="*60)

print(f"\n{'Threshold':<10} {'L1':<8} {'D1':<8} {'L2':<8} {'D2':<8}")
print("-" * 45)
for r in results:
    sp = r['sparsity']
    print(f"{r['threshold']:<10} {100*sp['L1']:<8.0f} {100*sp['D1']:<8.0f} {100*sp['L2']:<8.0f} {100*sp['D2']:<8.0f}")

# %%
# Save results
import pickle
save_path = Path("prune_symmetric_results.pkl")
with open(save_path, 'wb') as f:
    pickle.dump({
        'results': results,
        'l1_state': l1_state,
        'l1_acc': l1_acc,
        'warmup_acc': warmup_acc,
        'config': {'n': n, 'num_layers': num_layers, 'rank': rank, 'seed': seed},
    }, f)
print(f"\nSaved to {save_path}")

# %%
# Visualize best pruned model weights
print("\n" + "="*60)
print("BEST PRUNED MODEL WEIGHTS (t=0.15)")
print("="*60)

best_result = [r for r in results if r['threshold'] == 0.15][0]
best_state = best_result['state_dict']

fig, axes = plt.subplots(2, 2, figsize=(10, 8))

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

plot_sparse_mat(axes[0, 0], best_state['layers.0.L'].numpy(), f"L1 (pruned)")
plot_sparse_mat(axes[0, 1], best_state['layers.0.D'].numpy(), f"D1 (pruned)")
plot_sparse_mat(axes[1, 0], best_state['layers.1.L'].numpy(), f"L2 (pruned)")
plot_sparse_mat(axes[1, 1], best_state['layers.1.D'].numpy(), f"D2 (pruned)")

plt.suptitle(f'Pruned Weights (t=0.15, {best_result["pruned_frac"]:.1%} pruned, {best_result["final_acc"]:.1%} acc)', fontsize=12)
plt.tight_layout()
plt.show()

# %%
