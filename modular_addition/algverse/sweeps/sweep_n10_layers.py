# %%
"""
Sweep for n=10: 2/3/4 layers, rank=10 and rank=20, 10 seeds each

Using AdamW with wd=0.01 (best grokking setting).
40k steps (double the usual).
"""

import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import pickle

# %%
# =============================================================================
# MODEL DEFINITIONS
# =============================================================================

class RMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(1))

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
    def __init__(self, n: int, num_layers: int, rank: int):
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
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Device: {device}")

GROK_THRESHOLD = 0.80


def train_with_trajectory(n, num_layers, rank, seed, wd=0.01, num_steps=40000, eval_every=200):
    """Train and return trajectory."""
    torch.manual_seed(seed)
    model = SymmetricBilinearResidual(n, num_layers, rank).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.01, weight_decay=wd)

    trajectory = {
        'steps': [],
        'eval_acc': [],
        'train_loss': [],
        'grok_step': None,
    }

    for step in range(num_steps + 1):
        model.train()
        x = torch.randn(128, n, device=device)
        targets = task_2nd_argmax(x)
        logits = model(x)
        loss = nn.functional.cross_entropy(logits, targets)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if step % eval_every == 0:
            model.eval()
            with torch.no_grad():
                x_eval = torch.randn(10000, n, device=device)
                targets_eval = task_2nd_argmax(x_eval)
                logits_eval = model(x_eval)
                acc = (logits_eval.argmax(-1) == targets_eval).float().mean().item()

            trajectory['steps'].append(step)
            trajectory['eval_acc'].append(acc)
            trajectory['train_loss'].append(loss.item())

            if trajectory['grok_step'] is None and acc >= GROK_THRESHOLD:
                trajectory['grok_step'] = step

    final_acc = trajectory['eval_acc'][-1]
    return final_acc, trajectory


# %%
# =============================================================================
# RUN SWEEP
# =============================================================================
n = 10
layers_list = [2, 3, 4]
ranks = [10, 20]  # n and 2n
seeds = list(range(10))
wd = 0.01
num_steps = 40000

print(f"Config: n={n}, wd={wd}, steps={num_steps}")
print(f"Layers: {layers_list}")
print(f"Ranks: {ranks}")
print(f"Seeds: {seeds}")
print()

all_results = {}

for num_layers in layers_list:
    for rank in ranks:
        config_name = f"L{num_layers}_r{rank}"
        print(f"\n{'='*60}")
        print(f"Config: {num_layers} layers, rank={rank}")
        print(f"{'='*60}")

        results = {
            'final_accs': [],
            'grok_steps': [],
            'trajectories': [],
        }

        for seed in seeds:
            print(f"  Seed {seed}...", end=" ", flush=True)
            acc, traj = train_with_trajectory(n, num_layers, rank, seed, wd, num_steps)
            results['final_accs'].append(acc)
            results['grok_steps'].append(traj['grok_step'])
            results['trajectories'].append(traj)

            grok_str = f"grok@{traj['grok_step']}" if traj['grok_step'] else "no grok"
            print(f"{acc:.1%} ({grok_str})")

        accs = results['final_accs']
        grok_count = sum(1 for g in results['grok_steps'] if g is not None)
        print(f"\n  Summary: Mean={np.mean(accs):.1%}, Std={np.std(accs):.1%}, "
              f"Range={min(accs):.1%}-{max(accs):.1%}, Grok={grok_count}/{len(seeds)}")

        all_results[config_name] = {
            'config': {'n': n, 'num_layers': num_layers, 'rank': rank},
            'results': results,
        }

# %%
# =============================================================================
# SUMMARY TABLE
# =============================================================================
print("\n" + "="*80)
print("SUMMARY")
print("="*80)

print(f"\n{'Config':<15} {'Mean':<10} {'Std':<10} {'Min':<10} {'Max':<10} {'Grok Rate':<12}")
print("-" * 70)

for config_name, data in all_results.items():
    accs = data['results']['final_accs']
    grok_steps = data['results']['grok_steps']
    grok_count = sum(1 for g in grok_steps if g is not None)
    print(f"{config_name:<15} {np.mean(accs):<10.1%} {np.std(accs):<10.1%} "
          f"{min(accs):<10.1%} {max(accs):<10.1%} {grok_count}/{len(seeds):<12}")

# %%
# =============================================================================
# VISUALIZATION
# =============================================================================
fig, axes = plt.subplots(2, 3, figsize=(18, 10))

colors = plt.cm.tab10(np.linspace(0, 1, len(seeds)))

# Plot accuracy trajectories for each config
for idx, (config_name, data) in enumerate(all_results.items()):
    ax = axes[idx // 3, idx % 3]
    trajs = data['results']['trajectories']

    for i, traj in enumerate(trajs):
        ax.plot(traj['steps'], [a*100 for a in traj['eval_acc']],
                color=colors[i], alpha=0.7, linewidth=1, label=f's{i}' if idx == 0 else None)

    ax.axhline(y=GROK_THRESHOLD*100, color='red', linestyle='--', linewidth=1.5)
    ax.set_xlabel('Step')
    ax.set_ylabel('Accuracy (%)')
    ax.set_title(f'{config_name} (mean={np.mean(data["results"]["final_accs"]):.1%})')
    ax.grid(True, alpha=0.3)
    ax.set_ylim(0, 100)

# Add legend to first plot
axes[0, 0].legend(fontsize=7, loc='lower right', ncol=2)

plt.suptitle(f'n={n} Sweep: 2/3/4 layers × rank=10/20, wd={wd}, {num_steps} steps', fontsize=14)
plt.tight_layout()
plt.savefig('sweep_n10_layers.png', dpi=150)
plt.show()

# %%
# Bar chart comparison
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

config_names = list(all_results.keys())
means = [np.mean(all_results[c]['results']['final_accs']) for c in config_names]
stds = [np.std(all_results[c]['results']['final_accs']) for c in config_names]
grok_rates = [sum(1 for g in all_results[c]['results']['grok_steps'] if g) / len(seeds) for c in config_names]

# Mean accuracy
ax = axes[0]
x = np.arange(len(config_names))
bars = ax.bar(x, [m*100 for m in means], yerr=[s*100 for s in stds], capsize=5,
              color='steelblue', edgecolor='black')
ax.set_xticks(x)
ax.set_xticklabels(config_names, rotation=45, ha='right')
ax.set_ylabel('Accuracy (%)')
ax.set_title('Mean Accuracy (± std)')
ax.axhline(y=GROK_THRESHOLD*100, color='red', linestyle='--', label=f'Grok threshold')
ax.legend()
ax.set_ylim(0, 100)

# Grok rate
ax = axes[1]
bars = ax.bar(x, [g*100 for g in grok_rates], color='coral', edgecolor='black')
ax.set_xticks(x)
ax.set_xticklabels(config_names, rotation=45, ha='right')
ax.set_ylabel('Grok Rate (%)')
ax.set_title('Fraction of seeds that grok (>80%)')
ax.set_ylim(0, 100)
for i, (bar, rate) in enumerate(zip(bars, grok_rates)):
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 2,
            f'{int(rate*len(seeds))}/{len(seeds)}', ha='center', fontsize=10)

plt.tight_layout()
plt.savefig('sweep_n10_summary.png', dpi=150)
plt.show()

# %%
# Save results
save_path = Path("sweep_n10_layers_results.pkl")
with open(save_path, 'wb') as f:
    pickle.dump({
        'config': {'n': n, 'wd': wd, 'num_steps': num_steps},
        'layers_list': layers_list,
        'ranks': ranks,
        'seeds': seeds,
        'all_results': all_results,
        'grok_threshold': GROK_THRESHOLD,
    }, f)
print(f"\nSaved to {save_path}")

# %%
