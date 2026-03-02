# %%
"""
Train with L1 penalty, prune smallest weights, and fine-tune.

1. Train with L1 regularization to encourage sparsity
2. Ablate (zero out) the smallest weights
3. Fine-tune the remaining weights with a mask
4. Check accuracy remains ~80%+
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

print(f"Config: n={n}, layers={num_layers}, rank={rank}, device={device}")

# %%
# =============================================================================
# PHASE 1a: Train normally first (no L1)
# =============================================================================
print("\n" + "="*60)
print("PHASE 1a: Training without L1 (warm-up)")
print("="*60)

torch.manual_seed(42)
model = BilinearResidualRMSNorm(n, num_layers, rank, use_rmsnorm=True).to(device)
optimizer = torch.optim.AdamW(model.parameters(), lr=0.01, weight_decay=0.001)

steps_warmup = 5000

for step in range(steps_warmup + 1):
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
# PHASE 1b: Continue training with L1 penalty
# =============================================================================
print("\n" + "="*60)
print("PHASE 1b: Training with L1 penalty")
print("="*60)

# Reset optimizer with lower LR for L1 phase
optimizer = torch.optim.AdamW(model.parameters(), lr=0.005, weight_decay=0.001)

l1_lambda = 0.001  # Small L1 penalty
steps_phase1 = 10000

history_phase1 = {'steps': [], 'loss': [], 'l1_loss': [], 'ce_loss': [], 'eval_acc': []}

for step in range(steps_phase1 + 1):
    model.train()
    x = torch.randn(128, n, device=device)
    targets = task_2nd_argmax(x)
    logits = model(x)

    # Cross-entropy loss
    ce_loss = nn.functional.cross_entropy(logits, targets)

    # L1 penalty on all weights (L, R, D matrices)
    l1_loss = 0
    for layer in model.layers:
        l1_loss += layer.L.abs().sum()
        l1_loss += layer.R.abs().sum()
        l1_loss += layer.D.abs().sum()

    loss = ce_loss + l1_lambda * l1_loss

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    if step % 500 == 0:
        model.eval()
        with torch.no_grad():
            x_eval = torch.randn(10000, n, device=device)
            targets_eval = task_2nd_argmax(x_eval)
            logits_eval = model(x_eval)
            eval_acc = (logits_eval.argmax(-1) == targets_eval).float().mean().item()

        history_phase1['steps'].append(step)
        history_phase1['loss'].append(loss.item())
        history_phase1['l1_loss'].append(l1_loss.item())
        history_phase1['ce_loss'].append(ce_loss.item())
        history_phase1['eval_acc'].append(eval_acc)

        if step % 2000 == 0:
            print(f"Step {step:5d}: CE={ce_loss.item():.4f}, L1={l1_loss.item():.2f}, Acc={eval_acc:.3f}")

print(f"\nPhase 1 final accuracy: {history_phase1['eval_acc'][-1]:.1%}")

# %%
# Analyze weight distribution after L1 training
print("\n" + "="*60)
print("WEIGHT DISTRIBUTION AFTER L1 TRAINING")
print("="*60)

all_weights = []
weight_names = []
for i, layer in enumerate(model.layers):
    all_weights.extend([layer.L.detach().cpu().flatten(),
                        layer.R.detach().cpu().flatten(),
                        layer.D.detach().cpu().flatten()])
    weight_names.extend([f'L{i+1}', f'R{i+1}', f'D{i+1}'])

all_weights_flat = torch.cat(all_weights).numpy()
total_weights = len(all_weights_flat)

print(f"Total weights: {total_weights}")
print(f"Weight stats: mean={all_weights_flat.mean():.4f}, std={all_weights_flat.std():.4f}")
print(f"Min abs: {np.abs(all_weights_flat).min():.6f}, Max abs: {np.abs(all_weights_flat).max():.4f}")

# Plot weight distribution
fig, axes = plt.subplots(1, 2, figsize=(12, 4))

ax1 = axes[0]
ax1.hist(all_weights_flat, bins=50, color='steelblue', edgecolor='black', alpha=0.7)
ax1.axvline(x=0, color='red', linestyle='--', linewidth=2)
ax1.set_xlabel('Weight value')
ax1.set_ylabel('Count')
ax1.set_title('Weight Distribution After L1 Training')

ax2 = axes[1]
abs_weights_sorted = np.sort(np.abs(all_weights_flat))
ax2.plot(abs_weights_sorted, 'b-', linewidth=1.5)
ax2.set_xlabel('Weight index (sorted by |weight|)')
ax2.set_ylabel('|Weight|')
ax2.set_title('Sorted Absolute Weight Magnitudes')
ax2.axhline(y=0.1, color='red', linestyle='--', label='threshold=0.1')
ax2.legend()

plt.tight_layout()
plt.show()

# %%
# =============================================================================
# PHASE 2: Prune smallest weights
# =============================================================================
print("\n" + "="*60)
print("PHASE 2: Pruning smallest weights")
print("="*60)

# Try different pruning thresholds
thresholds = [0.05, 0.1, 0.15, 0.2, 0.25, 0.3]

def count_pruned(model, threshold):
    """Count how many weights would be pruned at given threshold."""
    total = 0
    pruned = 0
    for layer in model.layers:
        for param in [layer.L, layer.R, layer.D]:
            total += param.numel()
            pruned += (param.abs() < threshold).sum().item()
    return pruned, total

def test_accuracy(model):
    """Test model accuracy."""
    model.eval()
    with torch.no_grad():
        x = torch.randn(10000, n, device=device)
        targets = task_2nd_argmax(x)
        logits = model(x)
        return (logits.argmax(-1) == targets).float().mean().item()

print(f"{'Threshold':<12} {'Pruned':<10} {'Total':<10} {'% Pruned':<12} {'Acc (before)':<12}")
print("-" * 60)

for thresh in thresholds:
    pruned, total = count_pruned(model, thresh)

    # Test accuracy with pruning (temporary)
    model_copy = BilinearResidualRMSNorm(n, num_layers, rank, use_rmsnorm=True).to(device)
    model_copy.load_state_dict(model.state_dict())

    # Apply pruning
    with torch.no_grad():
        for layer in model_copy.layers:
            layer.L.data[layer.L.abs() < thresh] = 0
            layer.R.data[layer.R.abs() < thresh] = 0
            layer.D.data[layer.D.abs() < thresh] = 0

    acc = test_accuracy(model_copy)
    print(f"{thresh:<12.2f} {pruned:<10} {total:<10} {100*pruned/total:<12.1f} {acc:<12.3f}")

# %%
# Choose pruning threshold (aim for ~30-50% pruned while maintaining decent accuracy)
prune_threshold = 0.15
print(f"\nSelected pruning threshold: {prune_threshold}")

# Create pruned model
model_pruned = BilinearResidualRMSNorm(n, num_layers, rank, use_rmsnorm=True).to(device)
model_pruned.load_state_dict(model.state_dict())

# Create mask and apply pruning
masks = {}
with torch.no_grad():
    for i, layer in enumerate(model_pruned.layers):
        masks[f'L{i}'] = (layer.L.abs() >= prune_threshold).float()
        masks[f'R{i}'] = (layer.R.abs() >= prune_threshold).float()
        masks[f'D{i}'] = (layer.D.abs() >= prune_threshold).float()

        layer.L.data *= masks[f'L{i}']
        layer.R.data *= masks[f'R{i}']
        layer.D.data *= masks[f'D{i}']

# Count pruned
total_params = sum(m.numel() for m in masks.values())
remaining_params = sum(m.sum().item() for m in masks.values())
pruned_params = total_params - remaining_params

print(f"Pruned: {int(pruned_params)} / {int(total_params)} ({100*pruned_params/total_params:.1f}%)")
print(f"Remaining: {int(remaining_params)} ({100*remaining_params/total_params:.1f}%)")

acc_after_prune = test_accuracy(model_pruned)
print(f"Accuracy after pruning (before fine-tune): {acc_after_prune:.1%}")

# %%
# =============================================================================
# PHASE 3: Fine-tune with mask (keep pruned weights at 0)
# =============================================================================
print("\n" + "="*60)
print("PHASE 3: Fine-tuning with pruning mask")
print("="*60)

optimizer_ft = torch.optim.AdamW(model_pruned.parameters(), lr=0.005, weight_decay=0.001)
steps_phase3 = 5000

history_phase3 = {'steps': [], 'loss': [], 'eval_acc': []}

for step in range(steps_phase3 + 1):
    model_pruned.train()
    x = torch.randn(128, n, device=device)
    targets = task_2nd_argmax(x)
    logits = model_pruned(x)
    loss = nn.functional.cross_entropy(logits, targets)

    optimizer_ft.zero_grad()
    loss.backward()

    # Apply mask to gradients (zero out gradients for pruned weights)
    with torch.no_grad():
        for i, layer in enumerate(model_pruned.layers):
            if layer.L.grad is not None:
                layer.L.grad *= masks[f'L{i}']
            if layer.R.grad is not None:
                layer.R.grad *= masks[f'R{i}']
            if layer.D.grad is not None:
                layer.D.grad *= masks[f'D{i}']

    optimizer_ft.step()

    # Re-apply mask to weights (in case of numerical issues)
    with torch.no_grad():
        for i, layer in enumerate(model_pruned.layers):
            layer.L.data *= masks[f'L{i}']
            layer.R.data *= masks[f'R{i}']
            layer.D.data *= masks[f'D{i}']

    if step % 500 == 0:
        model_pruned.eval()
        with torch.no_grad():
            x_eval = torch.randn(10000, n, device=device)
            targets_eval = task_2nd_argmax(x_eval)
            logits_eval = model_pruned(x_eval)
            eval_acc = (logits_eval.argmax(-1) == targets_eval).float().mean().item()

        history_phase3['steps'].append(step)
        history_phase3['loss'].append(loss.item())
        history_phase3['eval_acc'].append(eval_acc)

        if step % 1000 == 0:
            print(f"Step {step:5d}: Loss={loss.item():.4f}, Acc={eval_acc:.3f}")

print(f"\nPhase 3 final accuracy: {history_phase3['eval_acc'][-1]:.1%}")

# %%
# =============================================================================
# SUMMARY AND VISUALIZATION
# =============================================================================
print("\n" + "="*60)
print("SUMMARY")
print("="*60)

print(f"Phase 1 (L1 training):     {history_phase1['eval_acc'][-1]:.1%}")
print(f"After pruning:             {acc_after_prune:.1%}")
print(f"Phase 3 (fine-tuned):      {history_phase3['eval_acc'][-1]:.1%}")
print(f"Weights pruned:            {int(pruned_params)} / {int(total_params)} ({100*pruned_params/total_params:.1f}%)")

# Plot training curves
fig, axes = plt.subplots(2, 2, figsize=(12, 10))

# Phase 1 accuracy
ax1 = axes[0, 0]
ax1.plot(history_phase1['steps'], history_phase1['eval_acc'], 'b-', linewidth=2)
ax1.set_xlabel('Step')
ax1.set_ylabel('Accuracy')
ax1.set_title('Phase 1: Training with L1 Penalty')
ax1.set_ylim(0, 1)
ax1.grid(True, alpha=0.3)

# Phase 1 losses
ax2 = axes[0, 1]
ax2.plot(history_phase1['steps'], history_phase1['ce_loss'], 'b-', linewidth=2, label='CE Loss')
ax2.plot(history_phase1['steps'], [l1_lambda * l for l in history_phase1['l1_loss']], 'r--', linewidth=2, label=f'L1 Loss (×{l1_lambda})')
ax2.set_xlabel('Step')
ax2.set_ylabel('Loss')
ax2.set_title('Phase 1: Loss Components')
ax2.legend()
ax2.grid(True, alpha=0.3)

# Phase 3 accuracy
ax3 = axes[1, 0]
ax3.plot(history_phase3['steps'], history_phase3['eval_acc'], 'g-', linewidth=2)
ax3.axhline(y=acc_after_prune, color='red', linestyle='--', label=f'After prune: {acc_after_prune:.1%}')
ax3.set_xlabel('Step')
ax3.set_ylabel('Accuracy')
ax3.set_title('Phase 3: Fine-tuning After Pruning')
ax3.set_ylim(0, 1)
ax3.legend()
ax3.grid(True, alpha=0.3)

# Final weight visualization
ax4 = axes[1, 1]
final_weights = []
for layer in model_pruned.layers:
    final_weights.extend([layer.L.detach().cpu().flatten(),
                          layer.R.detach().cpu().flatten(),
                          layer.D.detach().cpu().flatten()])
final_weights_flat = torch.cat(final_weights).numpy()
nonzero_weights = final_weights_flat[final_weights_flat != 0]

ax4.hist(nonzero_weights, bins=30, color='green', edgecolor='black', alpha=0.7, label=f'Non-zero ({len(nonzero_weights)})')
ax4.axvline(x=0, color='red', linestyle='--', linewidth=2)
ax4.set_xlabel('Weight value')
ax4.set_ylabel('Count')
ax4.set_title(f'Final Weight Distribution\n({len(nonzero_weights)}/{len(final_weights_flat)} non-zero)')
ax4.legend()

plt.tight_layout()
plt.show()

# %%
# Visualize the pruned weight matrices
print("\n" + "="*60)
print("PRUNED WEIGHT MATRICES")
print("="*60)

fig, axes = plt.subplots(2, 3, figsize=(12, 8))

for i, layer in enumerate(model_pruned.layers):
    for j, (name, param) in enumerate([('L', layer.L), ('R', layer.R), ('D', layer.D)]):
        ax = axes[i, j]
        mat = param.detach().cpu().numpy()
        vmax = max(abs(mat.min()), abs(mat.max())) if mat.max() != mat.min() else 1
        im = ax.imshow(mat, cmap='RdBu_r', vmin=-vmax, vmax=vmax, aspect='auto')

        # Add text
        for ii in range(mat.shape[0]):
            for jj in range(mat.shape[1]):
                val = mat[ii, jj]
                if val == 0:
                    ax.text(jj, ii, '0', ha='center', va='center', fontsize=9, color='gray')
                else:
                    color = 'white' if abs(val) > vmax * 0.6 else 'black'
                    ax.text(jj, ii, f'{val:.2f}', ha='center', va='center', fontsize=8, color=color)

        ax.set_title(f'Layer {i+1}: {name}')
        if j == 0:
            ax.set_ylabel(f'Layer {i+1}')

plt.suptitle(f'Pruned Weight Matrices (threshold={prune_threshold})', fontsize=12)
plt.tight_layout()
plt.show()

# Print sparsity per matrix
print("\nSparsity per weight matrix:")
for i, layer in enumerate(model_pruned.layers):
    for name, param in [('L', layer.L), ('R', layer.R), ('D', layer.D)]:
        total = param.numel()
        zeros = (param == 0).sum().item()
        print(f"  Layer {i+1} {name}: {zeros}/{total} zeros ({100*zeros/total:.0f}% sparse)")

# %%
# Save the pruned model
save_path = Path("pruned_model_l1.pt")
torch.save({
    'state_dict': model_pruned.state_dict(),
    'masks': {k: v.cpu() for k, v in masks.items()},
    'config': {'n': n, 'num_layers': num_layers, 'rank': rank},
    'prune_threshold': prune_threshold,
    'final_accuracy': history_phase3['eval_acc'][-1],
    'pruned_fraction': pruned_params / total_params,
}, save_path)
print(f"\nSaved pruned model to {save_path}")

# %%
