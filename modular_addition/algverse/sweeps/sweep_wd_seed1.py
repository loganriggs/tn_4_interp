# %%
"""
Weight Decay Sweep for Seed 1 (n=5, 3-layer)

Investigate the accuracy jump at step 10k.
Try higher weight decays: 0.01, 0.03, 0.1, 0.3, 1.0
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
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Device: {device}")


def train_with_trajectory(n, num_layers, rank, seed, weight_decay, num_steps=20000, eval_every=50):
    """Train and return full trajectory with finer granularity."""
    torch.manual_seed(seed)
    model = SymmetricBilinearResidual(n, num_layers, rank).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.01, weight_decay=weight_decay)

    trajectory = {
        'steps': [],
        'train_loss': [],
        'eval_acc': [],
        'weight_norms': [],
        'layer_norms': [],  # per-layer norms
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

                # Track weight norms per layer
                layer_norms = []
                total_norm = 0
                for layer in model.layers:
                    ln = (layer.L.norm().item() ** 2 + layer.D.norm().item() ** 2) ** 0.5
                    layer_norms.append(ln)
                    total_norm += ln ** 2
                total_norm = np.sqrt(total_norm)

            trajectory['steps'].append(step)
            trajectory['train_loss'].append(loss.item())
            trajectory['eval_acc'].append(acc)
            trajectory['weight_norms'].append(total_norm)
            trajectory['layer_norms'].append(layer_norms)

    final_acc = trajectory['eval_acc'][-1]
    return final_acc, trajectory, model.state_dict()


# %%
# =============================================================================
# RUN EXPERIMENT
# =============================================================================
n = 5
num_layers = 3
rank = 5
seed = 1
weight_decays = [0.01, 0.03, 0.1, 0.3, 1.0]

print(f"Config: n={n}, layers={num_layers}, rank={rank}, seed={seed}")
print(f"Weight decays: {weight_decays}")
print()

results = {}

for wd in weight_decays:
    print(f"Training with wd={wd}...", end=" ", flush=True)
    acc, traj, state = train_with_trajectory(n, num_layers, rank, seed, wd)
    results[wd] = {
        'final_acc': acc,
        'trajectory': traj,
        'state_dict': {k: v.cpu() for k, v in state.items()},
    }
    print(f"{acc:.1%}")

# %%
# =============================================================================
# SUMMARY
# =============================================================================
print("\n" + "="*60)
print("SUMMARY")
print("="*60)

print(f"\n{'Weight Decay':<15} {'Final Acc':<12}")
print("-" * 30)
for wd in weight_decays:
    print(f"{wd:<15} {results[wd]['final_acc']:<12.1%}")

# %%
# =============================================================================
# VISUALIZATION
# =============================================================================
fig, axes = plt.subplots(2, 2, figsize=(14, 10))

colors = plt.cm.viridis(np.linspace(0, 1, len(weight_decays)))

# Plot 1: Accuracy trajectories
ax = axes[0, 0]
for i, wd in enumerate(weight_decays):
    traj = results[wd]['trajectory']
    ax.plot(traj['steps'], [a*100 for a in traj['eval_acc']],
            color=colors[i], label=f'wd={wd}', linewidth=2)
ax.set_xlabel('Step')
ax.set_ylabel('Accuracy (%)')
ax.set_title(f'Accuracy Trajectories (seed={seed})')
ax.legend()
ax.grid(True, alpha=0.3)
ax.set_ylim(0, 100)

# Plot 2: Loss trajectories
ax = axes[0, 1]
for i, wd in enumerate(weight_decays):
    traj = results[wd]['trajectory']
    ax.plot(traj['steps'], traj['train_loss'],
            color=colors[i], label=f'wd={wd}', linewidth=2)
ax.set_xlabel('Step')
ax.set_ylabel('Train Loss')
ax.set_title('Loss Trajectories')
ax.legend()
ax.grid(True, alpha=0.3)
ax.set_yscale('log')

# Plot 3: Total weight norm
ax = axes[1, 0]
for i, wd in enumerate(weight_decays):
    traj = results[wd]['trajectory']
    ax.plot(traj['steps'], traj['weight_norms'],
            color=colors[i], label=f'wd={wd}', linewidth=2)
ax.set_xlabel('Step')
ax.set_ylabel('Total Weight Norm')
ax.set_title('Weight Norm Trajectories')
ax.legend()
ax.grid(True, alpha=0.3)

# Plot 4: Per-layer norms for wd=0.01 (the one with the jump)
ax = axes[1, 1]
wd_focus = 0.01
traj = results[wd_focus]['trajectory']
layer_norms = np.array(traj['layer_norms'])
for layer_idx in range(num_layers):
    ax.plot(traj['steps'], layer_norms[:, layer_idx],
            label=f'Layer {layer_idx+1}', linewidth=2)
ax.set_xlabel('Step')
ax.set_ylabel('Layer Weight Norm')
ax.set_title(f'Per-Layer Norms (wd={wd_focus})')
ax.legend()
ax.grid(True, alpha=0.3)

plt.suptitle(f'Seed {seed}, n={n}, {num_layers}-layer', fontsize=14)
plt.tight_layout()
plt.savefig('sweep_wd_seed1.png', dpi=150)
plt.show()

# %%
# Detailed plot around the jump region for wd=0.01
fig, axes = plt.subplots(1, 3, figsize=(15, 4))

wd_focus = 0.01
traj = results[wd_focus]['trajectory']
steps = np.array(traj['steps'])
accs = np.array(traj['eval_acc']) * 100
losses = np.array(traj['train_loss'])
layer_norms = np.array(traj['layer_norms'])

# Find the jump: where accuracy increases most
acc_diff = np.diff(accs)
jump_idx = np.argmax(acc_diff)
jump_step = steps[jump_idx]
print(f"\nLargest accuracy jump at step {jump_step}: {accs[jump_idx]:.1f}% -> {accs[jump_idx+1]:.1f}%")

# Zoom around jump
window = 100  # steps around jump
mask = (steps >= jump_step - window * 50) & (steps <= jump_step + window * 50)

ax = axes[0]
ax.plot(steps[mask], accs[mask], 'b-', linewidth=2)
ax.axvline(x=jump_step, color='red', linestyle='--', label=f'Jump at {jump_step}')
ax.set_xlabel('Step')
ax.set_ylabel('Accuracy (%)')
ax.set_title(f'Accuracy around jump (wd={wd_focus})')
ax.legend()
ax.grid(True, alpha=0.3)

ax = axes[1]
ax.plot(steps[mask], losses[mask], 'g-', linewidth=2)
ax.axvline(x=jump_step, color='red', linestyle='--')
ax.set_xlabel('Step')
ax.set_ylabel('Loss')
ax.set_title('Loss around jump')
ax.grid(True, alpha=0.3)

ax = axes[2]
for layer_idx in range(num_layers):
    ax.plot(steps[mask], layer_norms[mask, layer_idx], label=f'Layer {layer_idx+1}', linewidth=2)
ax.axvline(x=jump_step, color='red', linestyle='--')
ax.set_xlabel('Step')
ax.set_ylabel('Layer Norm')
ax.set_title('Layer norms around jump')
ax.legend()
ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('sweep_wd_seed1_jump.png', dpi=150)
plt.show()

# %%
# Save results
save_path = Path("sweep_wd_seed1_results.pkl")
with open(save_path, 'wb') as f:
    pickle.dump({
        'config': {'n': n, 'num_layers': num_layers, 'rank': rank, 'seed': seed},
        'weight_decays': weight_decays,
        'results': results,
    }, f)
print(f"\nSaved to {save_path}")

# %%
