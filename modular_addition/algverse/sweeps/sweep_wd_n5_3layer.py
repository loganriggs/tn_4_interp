# %%
"""
Weight Decay Comparison for n=5, 3-layer

Compare 2 weight decay settings with 5 seeds each.
Save training trajectories to understand variance.
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
        self.weight = nn.Parameter(torch.ones(1))  # single scalar

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
# =============================================================================
# TRAINING WITH TRAJECTORY
# =============================================================================
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Device: {device}")


def train_with_trajectory(n, num_layers, rank, seed, weight_decay, num_steps=15000, eval_every=100):
    """Train and return full trajectory."""
    torch.manual_seed(seed)
    model = SymmetricBilinearResidual(n, num_layers, rank).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.01, weight_decay=weight_decay)

    trajectory = {
        'steps': [],
        'train_loss': [],
        'eval_acc': [],
        'weight_norms': [],  # track weight magnitudes
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

                # Track weight norms
                total_norm = 0
                for layer in model.layers:
                    total_norm += layer.L.norm().item() ** 2
                    total_norm += layer.D.norm().item() ** 2
                total_norm = np.sqrt(total_norm)

            trajectory['steps'].append(step)
            trajectory['train_loss'].append(loss.item())
            trajectory['eval_acc'].append(acc)
            trajectory['weight_norms'].append(total_norm)

    final_acc = trajectory['eval_acc'][-1]
    return final_acc, trajectory, model.state_dict()


# %%
# =============================================================================
# RUN EXPERIMENT
# =============================================================================
n = 5
num_layers = 3
rank = 5
seeds = [0, 1, 2, 3, 4]
weight_decays = [0.001, 0.01]  # low vs high

print(f"Config: n={n}, layers={num_layers}, rank={rank}")
print(f"Seeds: {seeds}")
print(f"Weight decays: {weight_decays}")
print()

results = {}

for wd in weight_decays:
    print(f"\n{'='*60}")
    print(f"Weight Decay = {wd}")
    print(f"{'='*60}")

    results[wd] = {
        'seeds': [],
        'final_accs': [],
        'trajectories': [],
        'state_dicts': [],
    }

    for seed in seeds:
        print(f"  Seed {seed}...", end=" ", flush=True)
        acc, traj, state = train_with_trajectory(n, num_layers, rank, seed, wd)
        results[wd]['seeds'].append(seed)
        results[wd]['final_accs'].append(acc)
        results[wd]['trajectories'].append(traj)
        results[wd]['state_dicts'].append({k: v.cpu() for k, v in state.items()})
        print(f"{acc:.1%}")

    accs = results[wd]['final_accs']
    print(f"\n  Mean: {np.mean(accs):.1%}, Std: {np.std(accs):.1%}")
    print(f"  Range: {min(accs):.1%} - {max(accs):.1%}")

# %%
# =============================================================================
# SUMMARY
# =============================================================================
print("\n" + "="*60)
print("SUMMARY")
print("="*60)

print(f"\n{'Weight Decay':<15} {'Mean':<10} {'Std':<10} {'Min':<10} {'Max':<10}")
print("-" * 55)
for wd in weight_decays:
    accs = results[wd]['final_accs']
    print(f"{wd:<15} {np.mean(accs):<10.1%} {np.std(accs):<10.1%} {min(accs):<10.1%} {max(accs):<10.1%}")

print("\nPer-seed results:")
print(f"{'Seed':<8}", end="")
for wd in weight_decays:
    print(f"{'wd='+str(wd):<12}", end="")
print()
print("-" * 35)
for i, seed in enumerate(seeds):
    print(f"{seed:<8}", end="")
    for wd in weight_decays:
        print(f"{results[wd]['final_accs'][i]:<12.1%}", end="")
    print()

# %%
# =============================================================================
# VISUALIZATION
# =============================================================================
fig, axes = plt.subplots(2, 3, figsize=(15, 10))

colors = plt.cm.tab10(np.linspace(0, 1, len(seeds)))

# Row 1: Accuracy trajectories
for col, wd in enumerate(weight_decays):
    ax = axes[0, col]
    for i, seed in enumerate(seeds):
        traj = results[wd]['trajectories'][i]
        ax.plot(traj['steps'], [a*100 for a in traj['eval_acc']],
                color=colors[i], label=f'seed {seed}', alpha=0.8)
    ax.set_xlabel('Step')
    ax.set_ylabel('Accuracy (%)')
    ax.set_title(f'Accuracy (wd={wd})')
    ax.legend(loc='lower right', fontsize=8)
    ax.grid(True, alpha=0.3)
    ax.set_ylim(0, 100)

# Row 1, Col 3: Final accuracy comparison
ax = axes[0, 2]
x_pos = np.arange(len(seeds))
width = 0.35
for j, wd in enumerate(weight_decays):
    accs = [a*100 for a in results[wd]['final_accs']]
    ax.bar(x_pos + j*width, accs, width, label=f'wd={wd}', alpha=0.8)
ax.set_xlabel('Seed')
ax.set_ylabel('Final Accuracy (%)')
ax.set_title('Final Accuracy by Seed')
ax.set_xticks(x_pos + width/2)
ax.set_xticklabels(seeds)
ax.legend()
ax.set_ylim(0, 100)

# Row 2: Loss and weight norm trajectories
for col, wd in enumerate(weight_decays):
    ax = axes[1, col]
    for i, seed in enumerate(seeds):
        traj = results[wd]['trajectories'][i]
        ax.plot(traj['steps'], traj['train_loss'],
                color=colors[i], label=f'seed {seed}', alpha=0.8)
    ax.set_xlabel('Step')
    ax.set_ylabel('Train Loss')
    ax.set_title(f'Loss (wd={wd})')
    ax.legend(loc='upper right', fontsize=8)
    ax.grid(True, alpha=0.3)
    ax.set_yscale('log')

# Row 2, Col 3: Weight norms
ax = axes[1, 2]
for j, wd in enumerate(weight_decays):
    linestyle = '-' if j == 0 else '--'
    for i, seed in enumerate(seeds):
        traj = results[wd]['trajectories'][i]
        ax.plot(traj['steps'], traj['weight_norms'],
                color=colors[i], linestyle=linestyle, alpha=0.7,
                label=f's{seed},wd={wd}' if i == 0 else None)
ax.set_xlabel('Step')
ax.set_ylabel('Total Weight Norm')
ax.set_title('Weight Norms (solid=low wd, dashed=high wd)')
ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('sweep_wd_n5_3layer.png', dpi=150)
plt.show()

# %%
# Save results
save_path = Path("sweep_wd_n5_3layer_results.pkl")
with open(save_path, 'wb') as f:
    pickle.dump({
        'config': {'n': n, 'num_layers': num_layers, 'rank': rank},
        'seeds': seeds,
        'weight_decays': weight_decays,
        'results': results,
    }, f)
print(f"\nSaved to {save_path}")

# %%
