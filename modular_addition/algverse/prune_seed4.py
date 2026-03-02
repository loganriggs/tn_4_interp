# %%
"""
L1 pruning experiment on seed 4 (the best performer at 86% accuracy).
Sweep over 5 pruning thresholds.
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
seed = 4  # Best seed from earlier experiments
device = "cuda" if torch.cuda.is_available() else "cpu"

thresholds = [0.05, 0.10, 0.15, 0.20, 0.25]

print(f"Config: n={n}, layers={num_layers}, rank={rank}, seed={seed}, device={device}")

# %%
# =============================================================================
# PHASE 1a: Train seed 4 without L1 (warm-up)
# =============================================================================
print("\n" + "="*60)
print("PHASE 1a: Training seed 4 without L1 (warm-up)")
print("="*60)

torch.manual_seed(seed)
model = BilinearResidualRMSNorm(n, num_layers, rank, use_rmsnorm=True).to(device)
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
# PHASE 1b: Continue with L1 penalty
# =============================================================================
print("\n" + "="*60)
print("PHASE 1b: Training with L1 penalty")
print("="*60)

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

    if step % 2000 == 0:
        model.eval()
        with torch.no_grad():
            x_eval = torch.randn(10000, n, device=device)
            targets_eval = task_2nd_argmax(x_eval)
            logits_eval = model(x_eval)
            eval_acc = (logits_eval.argmax(-1) == targets_eval).float().mean().item()
        print(f"Step {step:5d}: CE={ce_loss.item():.4f}, L1={l1_loss.item():.2f}, Acc={eval_acc:.3f}")

l1_acc = eval_acc
print(f"\nAfter L1 training accuracy: {l1_acc:.1%}")

# Save the L1-trained model state for reuse
l1_state = {k: v.clone() for k, v in model.state_dict().items()}

# %%
# =============================================================================
# PHASE 2 & 3: Try different pruning thresholds
# =============================================================================
print("\n" + "="*60)
print("PHASE 2 & 3: Pruning + Fine-tuning at different thresholds")
print("="*60)

results = []

for thresh in thresholds:
    print(f"\n--- Threshold {thresh} ---")

    # Reload L1-trained model
    model_pruned = BilinearResidualRMSNorm(n, num_layers, rank, use_rmsnorm=True).to(device)
    model_pruned.load_state_dict(l1_state)

    # Create masks and prune
    masks = {}
    with torch.no_grad():
        for i, layer in enumerate(model_pruned.layers):
            masks[f'L{i}'] = (layer.L.abs() >= thresh).float()
            masks[f'R{i}'] = (layer.R.abs() >= thresh).float()
            masks[f'D{i}'] = (layer.D.abs() >= thresh).float()
            layer.L.data *= masks[f'L{i}']
            layer.R.data *= masks[f'R{i}']
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

        with torch.no_grad():
            for i, layer in enumerate(model_pruned.layers):
                if layer.L.grad is not None:
                    layer.L.grad *= masks[f'L{i}']
                if layer.R.grad is not None:
                    layer.R.grad *= masks[f'R{i}']
                if layer.D.grad is not None:
                    layer.D.grad *= masks[f'D{i}']

        optimizer_ft.step()

        with torch.no_grad():
            for i, layer in enumerate(model_pruned.layers):
                layer.L.data *= masks[f'L{i}']
                layer.R.data *= masks[f'R{i}']
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
        for name, param in [('L', layer.L), ('R', layer.R), ('D', layer.D)]:
            zeros = (param == 0).sum().item()
            total = param.numel()
            sparsity[f'{name}{i+1}'] = zeros / total

    results.append({
        'threshold': thresh,
        'prune_acc': prune_acc,
        'final_acc': final_acc,
        'pruned_frac': pruned_frac,
        'sparsity': sparsity,
        'masks': {k: v.cpu() for k, v in masks.items()},
        'state_dict': {k: v.cpu() for k, v in model_pruned.state_dict().items()},
    })

# %%
# =============================================================================
# RESULTS SUMMARY
# =============================================================================
print("\n" + "="*60)
print("RESULTS SUMMARY (Seed 4)")
print("="*60)

print(f"\n{'Threshold':<12} {'Pruned %':<12} {'After Prune':<14} {'After Fine-tune':<16}")
print("-" * 55)
for r in results:
    print(f"{r['threshold']:<12} {100*r['pruned_frac']:<12.1f} {100*r['prune_acc']:<14.1f} {100*r['final_acc']:<16.1f}")

print(f"\nBaseline (after L1 training): {l1_acc:.1%}")

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
ax1.set_xlabel('Pruning Threshold', fontsize=12)
ax1.set_ylabel('Accuracy (%)', fontsize=12)
ax1.set_title('Accuracy vs Pruning Threshold', fontsize=12)
ax1.legend()
ax1.grid(True, alpha=0.3)
ax1.set_ylim(50, 90)

# Pruned fraction vs threshold
ax2 = axes[1]
ax2.bar(thresh_vals, [100*p for p in pruned_fracs], width=0.03, color='steelblue', edgecolor='black')
ax2.set_xlabel('Pruning Threshold', fontsize=12)
ax2.set_ylabel('Pruned Fraction (%)', fontsize=12)
ax2.set_title('Pruned Weights vs Threshold', fontsize=12)
ax2.grid(True, alpha=0.3)

# Accuracy vs pruned fraction trade-off
ax3 = axes[2]
ax3.plot([100*p for p in pruned_fracs], [100*a for a in final_accs], 'go-', markersize=12, linewidth=2)
for i, (p, a, t) in enumerate(zip(pruned_fracs, final_accs, thresh_vals)):
    ax3.annotate(f't={t}', (100*p, 100*a), textcoords='offset points', xytext=(5, 5), fontsize=9)
ax3.axhline(y=80, color='gray', linestyle=':', linewidth=1, label='80% target')
ax3.set_xlabel('Pruned Fraction (%)', fontsize=12)
ax3.set_ylabel('Final Accuracy (%)', fontsize=12)
ax3.set_title('Accuracy vs Sparsity Trade-off', fontsize=12)
ax3.legend()
ax3.grid(True, alpha=0.3)
ax3.set_ylim(50, 90)

plt.tight_layout()
plt.show()

# %%
# Per-matrix sparsity breakdown for best threshold
print("\n" + "="*60)
print("PER-MATRIX SPARSITY BREAKDOWN")
print("="*60)

print(f"\n{'Threshold':<10} {'L1':<8} {'R1':<8} {'D1':<8} {'L2':<8} {'R2':<8} {'D2':<8}")
print("-" * 60)
for r in results:
    sp = r['sparsity']
    print(f"{r['threshold']:<10} {100*sp['L1']:<8.0f} {100*sp['R1']:<8.0f} {100*sp['D1']:<8.0f} "
          f"{100*sp['L2']:<8.0f} {100*sp['R2']:<8.0f} {100*sp['D2']:<8.0f}")

# %%
# Save results
import pickle
save_path = Path("prune_seed4_results.pkl")
with open(save_path, 'wb') as f:
    pickle.dump({
        'results': results,
        'l1_state': l1_state,
        'l1_acc': l1_acc,
        'warmup_acc': warmup_acc,
        'config': {'n': n, 'num_layers': num_layers, 'rank': rank, 'seed': seed},
    }, f)
print(f"\nSaved results to {save_path}")

# %%
