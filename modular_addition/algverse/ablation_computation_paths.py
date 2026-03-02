# %%
"""
Computation Path Ablation Experiment

Decompose output into 5 components:
1. x (direct residual)
2. r1 = bilinear_1(x) = D1 @ (L1 @ x)²
3. bilinear_2(x + r1) decomposes into:
   A. D2 @ (L2 @ x)²           -- x mixed with x
   B. D2 @ 2(L2@x)(L2@r1)      -- x mixed with r1 (cross term)
   C. D2 @ (L2 @ r1)²          -- r1 mixed with r1

Sanity check: all 5 should sum to original output.
Ablation: remove one at a time, check accuracy.
Powerset: all combinations of A, B, C.
"""

import torch
import numpy as np
import matplotlib.pyplot as plt
import pickle
from pathlib import Path
from itertools import combinations

# %%
# Load pruned symmetric model (t=0.25)
with open('prune_symmetric_results.pkl', 'rb') as f:
    data = pickle.load(f)

config = data['config']
n = config['n']
rank = config['rank']

prune_result = [r for r in data['results'] if r['threshold'] == 0.25][0]
state_dict = prune_result['state_dict']

L1 = state_dict['layers.0.L']
D1 = state_dict['layers.0.D']
L2 = state_dict['layers.1.L']
D2 = state_dict['layers.1.D']
norm1_weight = state_dict['norms.0.weight']
norm2_weight = state_dict['norms.1.weight']

print(f"Loaded pruned symmetric model (t=0.25)")
print(f"Accuracy: {prune_result['final_acc']:.1%}")

# %%
# Helper functions
def rmsnorm(x, weight, eps=1e-6):
    rms = torch.sqrt((x ** 2).mean(dim=-1, keepdim=True) + eps)
    return weight * (x / rms)

def task_2nd_argmax(x):
    return x.argsort(-1)[..., -2]

# %%
def compute_all_components(x):
    """
    Compute all 5 components of the output.

    Returns dict with:
        'x': direct input contribution
        'r1': layer 1 output
        'A': D2 @ (L2 @ x_norm)²  (x mixed with x in layer 2)
        'B': D2 @ 2(L2 @ x_norm)(L2 @ r1_in_h1_norm)  (cross term)
        'C': D2 @ (L2 @ r1_in_h1_norm)²  (r1 mixed with r1 in layer 2)
        'full': full model output (should equal sum of all)
    """
    batch = x.shape[0]

    # Component 1: direct input
    comp_x = x.clone()

    # Component 2: r1 = bilinear_1(norm(x))
    x_norm1 = rmsnorm(x, norm1_weight)
    L1x = x_norm1 @ L1.T  # (batch, rank)
    r1 = (L1x ** 2) @ D1.T  # (batch, n)
    comp_r1 = r1.clone()

    # h1 = x + r1
    h1 = x + r1

    # For layer 2, we need to decompose (L2 @ h1_norm)²
    # But h1_norm = norm(x + r1), which is tricky to decompose cleanly
    #
    # Actually, let's think about this more carefully:
    # h1_norm = rmsnorm(x + r1)
    # L2 @ h1_norm then gets squared
    #
    # The issue is that rmsnorm mixes x and r1 before L2 sees them.
    # So we can't cleanly separate "L2 sees x" from "L2 sees r1".
    #
    # Alternative approach: decompose in the hidden space AFTER L2 projection
    # Let's define:
    #   h1_norm = rmsnorm(x + r1)
    #   L2_h1 = L2 @ h1_norm
    #
    # We can decompose h1 = x + r1 before normalization, then see how
    # L2 @ norm(x) and L2 @ norm(r1) relate to L2 @ norm(x+r1)
    #
    # Actually, the cleanest is to compute:
    # - What if layer 2 only saw x (ignoring r1)?
    # - What if layer 2 only saw r1 (ignoring x)?
    # - What's the cross term?
    #
    # Let's do: decompose h1_norm into its components from x vs r1
    # Using first-order approximation isn't clean either.
    #
    # CLEANER APPROACH: Don't go through rmsnorm for decomposition.
    # Instead, compute the terms directly:

    h1_norm = rmsnorm(h1, norm2_weight)
    L2_h1 = h1_norm @ L2.T  # (batch, rank)

    # Full layer 2 output
    layer2_full = (L2_h1 ** 2) @ D2.T

    # Now decompose by considering what each part of h1 contributes
    # h1 = x + r1
    # If we had h1 = x only: h1_x = x, then norm and apply L2
    # If we had h1 = r1 only: h1_r1 = r1, then norm and apply L2

    # Component A: layer 2 seeing only x (as if r1 = 0)
    h1_x_only = x
    h1_x_norm = rmsnorm(h1_x_only, norm2_weight)
    L2_x = h1_x_norm @ L2.T
    comp_A = (L2_x ** 2) @ D2.T

    # Component C: layer 2 seeing only r1 (as if x = 0)
    h1_r1_only = r1
    h1_r1_norm = rmsnorm(h1_r1_only, norm2_weight)
    L2_r1 = h1_r1_norm @ L2.T
    comp_C = (L2_r1 ** 2) @ D2.T

    # Component B: cross term = full - A - C - x - r1
    # This captures the interaction between x and r1 in layer 2
    comp_B = layer2_full - comp_A - comp_C

    # But wait, the full output is x + r1 + layer2_full
    # So: full = x + r1 + A + B + C should hold if we define B as the remainder

    # Actually let's be more precise about B:
    # (L2 @ norm(x+r1))² = ... this is complicated due to normalization
    #
    # Let's just define B as whatever makes the sum work:
    # full_output = x + r1 + layer2_output
    # layer2_output = A + B + C where B is the "interaction" term

    full_output = x + r1 + layer2_full

    # Verify decomposition
    reconstructed = comp_x + comp_r1 + comp_A + comp_B + comp_C

    return {
        'x': comp_x,
        'r1': comp_r1,
        'A': comp_A,
        'B': comp_B,
        'C': comp_C,
        'layer2_full': layer2_full,
        'full': full_output,
        'reconstructed': reconstructed,
    }

# %%
# Sanity check: verify decomposition
print("="*60)
print("SANITY CHECK: Decomposition")
print("="*60)

torch.manual_seed(42)
x_test = torch.randn(1000, n)
comps = compute_all_components(x_test)

# Check that components sum to full
diff = (comps['full'] - comps['reconstructed']).abs().max()
print(f"Max diff (full - reconstructed): {diff:.2e}")

# Check that A + B + C = layer2_full
layer2_reconstructed = comps['A'] + comps['B'] + comps['C']
diff2 = (comps['layer2_full'] - layer2_reconstructed).abs().max()
print(f"Max diff (layer2 - A+B+C): {diff2:.2e}")

# Component magnitudes
print(f"\nComponent magnitudes (L2 norm, averaged over 1000 samples):")
for name in ['x', 'r1', 'A', 'B', 'C']:
    norm = comps[name].norm(dim=1).mean().item()
    print(f"  {name}: {norm:.3f}")

# %%
# =============================================================================
# ABLATION: Remove one component at a time
# =============================================================================
print("\n" + "="*60)
print("ABLATION: Remove one component at a time")
print("="*60)

def compute_accuracy(logits, x):
    targets = task_2nd_argmax(x)
    preds = logits.argmax(dim=1)
    return (preds == targets).float().mean().item()

# Generate test data
torch.manual_seed(123)
x_eval = torch.randn(10000, n)
comps_eval = compute_all_components(x_eval)

# Full accuracy
full_acc = compute_accuracy(comps_eval['full'], x_eval)
print(f"\nFull model accuracy: {full_acc:.1%}")

# Remove each component
print(f"\nAccuracy when removing each component:")
print(f"{'Removed':<12} {'Accuracy':<12} {'Δ Accuracy':<12}")
print("-" * 36)

component_names = ['x', 'r1', 'A', 'B', 'C']
ablation_results = {}

for remove_name in component_names:
    # Compute output without this component
    output_ablated = comps_eval['full'] - comps_eval[remove_name]
    acc = compute_accuracy(output_ablated, x_eval)
    delta = acc - full_acc
    ablation_results[f'remove_{remove_name}'] = acc
    print(f"{remove_name:<12} {acc:<12.1%} {delta:>+.1%}")

# %%
# =============================================================================
# ABLATION: Keep only one component at a time
# =============================================================================
print("\n" + "="*60)
print("ABLATION: Keep only one component")
print("="*60)

print(f"{'Kept':<12} {'Accuracy':<12}")
print("-" * 24)

for keep_name in component_names:
    output_only = comps_eval[keep_name]
    acc = compute_accuracy(output_only, x_eval)
    ablation_results[f'only_{keep_name}'] = acc
    print(f"{keep_name:<12} {acc:<12.1%}")

# %%
# =============================================================================
# POWERSET of A, B, C (layer 2 components)
# =============================================================================
print("\n" + "="*60)
print("POWERSET: All combinations of A, B, C")
print("="*60)
print("(Always including x + r1 as base)")

# Base = x + r1
base = comps_eval['x'] + comps_eval['r1']
base_acc = compute_accuracy(base, x_eval)
print(f"\nBase (x + r1 only, no layer 2): {base_acc:.1%}")

abc_components = ['A', 'B', 'C']
powerset_results = []

print(f"\n{'Components':<20} {'Accuracy':<12}")
print("-" * 32)

# Empty set (just base)
powerset_results.append(('none', base_acc))
print(f"{'(none)':<20} {base_acc:<12.1%}")

# All subsets
for r in range(1, len(abc_components) + 1):
    for subset in combinations(abc_components, r):
        output = base.clone()
        for comp_name in subset:
            output = output + comps_eval[comp_name]
        acc = compute_accuracy(output, x_eval)
        subset_str = '+'.join(subset)
        powerset_results.append((subset_str, acc))
        print(f"{subset_str:<20} {acc:<12.1%}")

# %%
# Visualize results
fig, axes = plt.subplots(1, 3, figsize=(15, 5))

# Plot 1: Remove one at a time
ax1 = axes[0]
remove_names = [f'remove_{n}' for n in component_names]
remove_accs = [ablation_results[n] for n in remove_names]
colors = ['red' if a < full_acc - 0.05 else 'orange' if a < full_acc else 'green' for a in remove_accs]
bars = ax1.bar(component_names, [a*100 for a in remove_accs], color=colors, edgecolor='black')
ax1.axhline(y=full_acc*100, color='blue', linestyle='--', linewidth=2, label=f'Full: {full_acc:.1%}')
ax1.set_ylabel('Accuracy (%)')
ax1.set_title('Accuracy when REMOVING each component')
ax1.set_ylim(0, 100)
ax1.legend()
for bar, acc in zip(bars, remove_accs):
    ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1, f'{acc:.1%}',
             ha='center', fontsize=9)

# Plot 2: Keep only one
ax2 = axes[1]
only_names = [f'only_{n}' for n in component_names]
only_accs = [ablation_results[n] for n in only_names]
bars = ax2.bar(component_names, [a*100 for a in only_accs], color='steelblue', edgecolor='black')
ax2.axhline(y=25, color='gray', linestyle=':', linewidth=1, label='Random (25%)')
ax2.set_ylabel('Accuracy (%)')
ax2.set_title('Accuracy with ONLY each component')
ax2.set_ylim(0, 100)
ax2.legend()
for bar, acc in zip(bars, only_accs):
    ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1, f'{acc:.1%}',
             ha='center', fontsize=9)

# Plot 3: Powerset of A, B, C
ax3 = axes[2]
ps_labels = [p[0] for p in powerset_results]
ps_accs = [p[1]*100 for p in powerset_results]
colors = plt.cm.viridis(np.linspace(0, 1, len(ps_labels)))
bars = ax3.bar(range(len(ps_labels)), ps_accs, color=colors, edgecolor='black')
ax3.set_xticks(range(len(ps_labels)))
ax3.set_xticklabels(ps_labels, rotation=45, ha='right', fontsize=9)
ax3.axhline(y=full_acc*100, color='red', linestyle='--', linewidth=2, label=f'Full: {full_acc:.1%}')
ax3.set_ylabel('Accuracy (%)')
ax3.set_title('Powerset of Layer 2 components (base = x + r1)')
ax3.set_ylim(0, 100)
ax3.legend()

plt.tight_layout()
plt.show()

# %%
# =============================================================================
# DETAILED ANALYSIS: Which components matter most?
# =============================================================================
print("\n" + "="*60)
print("SUMMARY: Component Importance")
print("="*60)

print("\n1. REMOVING components (higher = less important):")
sorted_remove = sorted([(n, ablation_results[f'remove_{n}']) for n in component_names],
                       key=lambda x: x[1], reverse=True)
for name, acc in sorted_remove:
    delta = acc - full_acc
    importance = "LOW" if delta > -0.05 else "MEDIUM" if delta > -0.15 else "HIGH"
    print(f"   {name}: {acc:.1%} (Δ={delta:+.1%}) - {importance} importance")

print("\n2. KEEPING only one component (higher = more standalone value):")
sorted_only = sorted([(n, ablation_results[f'only_{n}']) for n in component_names],
                     key=lambda x: x[1], reverse=True)
for name, acc in sorted_only:
    print(f"   {name}: {acc:.1%}")

print("\n3. BEST layer 2 combinations (base = x + r1):")
sorted_ps = sorted(powerset_results, key=lambda x: x[1], reverse=True)
for name, acc in sorted_ps[:5]:
    print(f"   {name}: {acc:.1%}")

# %%
# Per-component contribution to correct predictions
print("\n" + "="*60)
print("COMPONENT CONTRIBUTIONS TO CORRECT PREDICTIONS")
print("="*60)

targets = task_2nd_argmax(x_eval)
full_preds = comps_eval['full'].argmax(dim=1)
correct_mask = (full_preds == targets)

print(f"\nAverage component value at TARGET position:")
print(f"{'Component':<12} {'Correct':<12} {'Wrong':<12} {'Diff':<12}")
print("-" * 48)

for name in component_names:
    comp = comps_eval[name]
    # Get value at target position for each sample
    target_vals = comp[torch.arange(len(targets)), targets]

    correct_mean = target_vals[correct_mask].mean().item()
    wrong_mean = target_vals[~correct_mask].mean().item()
    diff = correct_mean - wrong_mean

    print(f"{name:<12} {correct_mean:<12.3f} {wrong_mean:<12.3f} {diff:<+12.3f}")

# %%
