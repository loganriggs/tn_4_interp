# %%
"""
Sweep for n=10: 1 layer, rank=10 and rank=20, 10 seeds each

Using AdamW with wd=0.01.
40k steps.
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


def train_with_trajectory(n, num_layers, rank, seed, wd=0.01, num_steps=40000, eval_every=200):
    """Train and return trajectory."""
    torch.manual_seed(seed)
    model = SymmetricBilinearResidual(n, num_layers, rank).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.01, weight_decay=wd)

    trajectory = {
        'steps': [],
        'eval_acc': [],
        'train_loss': [],
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

    final_acc = trajectory['eval_acc'][-1]
    return final_acc, trajectory


# %%
# =============================================================================
# RUN SWEEP
# =============================================================================
n = 10
num_layers = 1
ranks = [10, 20]
seeds = list(range(10))
wd = 0.01
num_steps = 40000

print(f"Config: n={n}, layers={num_layers}, wd={wd}, steps={num_steps}")
print(f"Ranks: {ranks}")
print(f"Seeds: {seeds}")
print()

all_results = {}

for rank in ranks:
    config_name = f"L{num_layers}_r{rank}"
    print(f"\n{'='*60}")
    print(f"Config: {num_layers} layer, rank={rank}")
    print(f"{'='*60}")

    results = {
        'final_accs': [],
        'trajectories': [],
    }

    for seed in seeds:
        print(f"  Seed {seed}...", end=" ", flush=True)
        acc, traj = train_with_trajectory(n, num_layers, rank, seed, wd, num_steps)
        results['final_accs'].append(acc)
        results['trajectories'].append(traj)
        print(f"{acc:.1%}")

    accs = results['final_accs']
    print(f"\n  Summary: Mean={np.mean(accs):.1%}, Std={np.std(accs):.1%}, "
          f"Range={min(accs):.1%}-{max(accs):.1%}")

    all_results[config_name] = {
        'config': {'n': n, 'num_layers': num_layers, 'rank': rank},
        'results': results,
    }

# %%
# =============================================================================
# SUMMARY
# =============================================================================
print("\n" + "="*60)
print("SUMMARY")
print("="*60)

print(f"\n{'Config':<15} {'Mean':<10} {'Std':<10} {'Min':<10} {'Max':<10}")
print("-" * 55)

for config_name, data in all_results.items():
    accs = data['results']['final_accs']
    print(f"{config_name:<15} {np.mean(accs):<10.1%} {np.std(accs):<10.1%} "
          f"{min(accs):<10.1%} {max(accs):<10.1%}")

# %%
# Visualization
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

colors = plt.cm.tab10(np.linspace(0, 1, len(seeds)))

for idx, (config_name, data) in enumerate(all_results.items()):
    ax = axes[idx]
    trajs = data['results']['trajectories']

    for i, traj in enumerate(trajs):
        ax.plot(traj['steps'], [a*100 for a in traj['eval_acc']],
                color=colors[i], alpha=0.7, linewidth=1.5, label=f's{i}')

    ax.set_xlabel('Step')
    ax.set_ylabel('Accuracy (%)')
    ax.set_title(f'{config_name} (mean={np.mean(data["results"]["final_accs"]):.1%})')
    ax.legend(fontsize=7, loc='lower right', ncol=2)
    ax.grid(True, alpha=0.3)
    ax.set_ylim(0, 100)

plt.suptitle(f'n={n}, 1-layer Sweep', fontsize=14)
plt.tight_layout()
plt.savefig('sweep_n10_1layer.png', dpi=150)
plt.show()

# %%
# Save
save_path = Path("sweep_n10_1layer_results.pkl")
with open(save_path, 'wb') as f:
    pickle.dump({
        'config': {'n': n, 'num_layers': num_layers, 'wd': wd, 'num_steps': num_steps},
        'ranks': ranks,
        'seeds': seeds,
        'all_results': all_results,
    }, f)
print(f"\nSaved to {save_path}")

# %%
