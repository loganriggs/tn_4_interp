# %%
"""
Compare n=10, 3-layer models with and without input W matrix.

Without W (baseline from previous sweep):
  - Mean: 77.5%, Std: 4.3% (rank=10)
  - Grok rate: 5/10

With W: new experiment
"""

import torch
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))
from models import train

# %%
# =============================================================================
# CONFIG
# =============================================================================
N = 10
NUM_LAYERS = 3
RANK = 10
NUM_STEPS = 40000
SEEDS = list(range(10))

results_dir = Path(__file__).parent.parent / "sweep_results"
results_dir.mkdir(exist_ok=True)

# %%
# =============================================================================
# RUN WITH W MATRIX
# =============================================================================
print("=" * 60)
print(f"n={N}, {NUM_LAYERS}-layer, rank={RANK}, WITH W matrix")
print("=" * 60)

results_with_W = []

for seed in SEEDS:
    print(f"\nSeed {seed}:")
    acc, trajectory, state = train(
        n=N,
        num_layers=NUM_LAYERS,
        rank=RANK,
        seed=seed,
        num_steps=NUM_STEPS,
        weight_decay=0.01,
        use_W=True,
        verbose=True,
    )

    # Check for grokking (acc > 75% at any point)
    max_acc = max(trajectory['eval_acc'])
    grok_step = None
    for i, a in enumerate(trajectory['eval_acc']):
        if a > 0.75:
            grok_step = trajectory['steps'][i]
            break

    results_with_W.append({
        'seed': seed,
        'final_acc': acc,
        'max_acc': max_acc,
        'grok_step': grok_step,
        'trajectory': trajectory,
    })

    grok_str = f"grok@{grok_step}" if grok_step else "no grok"
    print(f"  Final: {acc:.1%}, Max: {max_acc:.1%}, {grok_str}")

# %%
# =============================================================================
# SUMMARY
# =============================================================================
print("\n" + "=" * 60)
print("SUMMARY: WITH W MATRIX")
print("=" * 60)

accs = [r['final_acc'] for r in results_with_W]
grok_count = sum(1 for r in results_with_W if r['grok_step'] is not None)

print(f"\nWith W matrix (n={N}, {NUM_LAYERS}L, r={RANK}):")
print(f"  Mean: {100*sum(accs)/len(accs):.1f}%")
print(f"  Std:  {100*torch.tensor(accs).std().item():.1f}%")
print(f"  Min:  {100*min(accs):.1f}%")
print(f"  Max:  {100*max(accs):.1f}%")
print(f"  Grok rate: {grok_count}/{len(SEEDS)}")

print("\nPer-seed results:")
print(f"{'Seed':<6} {'Final':<8} {'Max':<8} {'Grok Step':<12}")
print("-" * 36)
for r in results_with_W:
    grok = r['grok_step'] if r['grok_step'] else 'never'
    print(f"{r['seed']:<6} {r['final_acc']*100:<8.1f} {r['max_acc']*100:<8.1f} {grok:<12}")

print("\n" + "=" * 60)
print("BASELINE (without W, from previous sweep):")
print("=" * 60)
print(f"Without W (n={N}, {NUM_LAYERS}L, r={RANK}):")
print(f"  Mean: 77.5%")
print(f"  Std:  4.3%")
print(f"  Grok rate: 5/10")

# %%
# Save results
import pickle
save_path = results_dir / "sweep_W_comparison.pkl"
with open(save_path, 'wb') as f:
    pickle.dump({
        'config': {'n': N, 'num_layers': NUM_LAYERS, 'rank': RANK, 'num_steps': NUM_STEPS},
        'results_with_W': results_with_W,
    }, f)
print(f"\nSaved to {save_path}")

# %%
