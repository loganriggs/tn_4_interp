# %%
"""
Grokking Investigation for Seed 1 (n=5, 3-layer)

Try different optimizers to understand/control the phase transition:
1. Adam with manual L2 (not AdamW's decoupled weight decay)
2. AdamW (current)
3. Muon optimizer
4. SGD with momentum
5. Different learning rates

Detect when accuracy crosses 80% ("grokking" threshold).
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

    def get_l2_reg(self):
        """Manual L2 regularization term."""
        l2 = 0
        for layer in self.layers:
            l2 += (layer.L ** 2).sum() + (layer.D ** 2).sum()
        return l2


def task_2nd_argmax(x):
    return x.argsort(-1)[..., -2]


# %%
# =============================================================================
# MUON OPTIMIZER (simplified version)
# =============================================================================

class Muon(torch.optim.Optimizer):
    """
    Muon optimizer: Momentum + Orthogonalization.
    Orthogonalizes gradient w.r.t. current weights for each parameter.
    """
    def __init__(self, params, lr=0.01, momentum=0.9):
        defaults = dict(lr=lr, momentum=momentum)
        super().__init__(params, defaults)

    @torch.no_grad()
    def step(self):
        for group in self.param_groups:
            lr = group['lr']
            momentum = group['momentum']

            for p in group['params']:
                if p.grad is None:
                    continue

                grad = p.grad
                state = self.state[p]

                # Initialize momentum buffer
                if 'momentum_buffer' not in state:
                    state['momentum_buffer'] = torch.zeros_like(p)

                buf = state['momentum_buffer']

                # Orthogonalize gradient w.r.t. parameter (for matrices)
                if p.dim() >= 2:
                    # Project out component parallel to p
                    p_flat = p.view(-1)
                    g_flat = grad.view(-1)
                    proj = (g_flat @ p_flat) / (p_flat @ p_flat + 1e-8)
                    grad_orth = grad - proj * p
                else:
                    grad_orth = grad

                # Momentum update
                buf.mul_(momentum).add_(grad_orth)

                # Update parameters
                p.add_(buf, alpha=-lr)


# %%
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Device: {device}")

GROK_THRESHOLD = 0.80  # Detect when accuracy crosses this


def train_experiment(n, num_layers, rank, seed, optimizer_type, lr, wd, num_steps=20000, eval_every=50):
    """
    Train with specified optimizer configuration.

    optimizer_type: 'adamw', 'adam_l2', 'muon', 'sgd'
    """
    torch.manual_seed(seed)
    model = SymmetricBilinearResidual(n, num_layers, rank).to(device)

    # Setup optimizer
    if optimizer_type == 'adamw':
        optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=wd)
        use_manual_l2 = False
    elif optimizer_type == 'adam_l2':
        optimizer = torch.optim.Adam(model.parameters(), lr=lr)
        use_manual_l2 = True
    elif optimizer_type == 'muon':
        optimizer = Muon(model.parameters(), lr=lr, momentum=0.9)
        use_manual_l2 = True
    elif optimizer_type == 'sgd':
        optimizer = torch.optim.SGD(model.parameters(), lr=lr, momentum=0.9)
        use_manual_l2 = True
    else:
        raise ValueError(f"Unknown optimizer: {optimizer_type}")

    trajectory = {
        'steps': [],
        'train_loss': [],
        'ce_loss': [],
        'eval_acc': [],
        'weight_norms': [],
        'grok_step': None,  # When accuracy first crosses threshold
    }

    for step in range(num_steps + 1):
        model.train()
        x = torch.randn(128, n, device=device)
        targets = task_2nd_argmax(x)
        logits = model(x)
        ce_loss = nn.functional.cross_entropy(logits, targets)

        if use_manual_l2:
            l2_loss = model.get_l2_reg()
            loss = ce_loss + wd * l2_loss
        else:
            loss = ce_loss

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

                total_norm = 0
                for layer in model.layers:
                    total_norm += layer.L.norm().item() ** 2
                    total_norm += layer.D.norm().item() ** 2
                total_norm = np.sqrt(total_norm)

            trajectory['steps'].append(step)
            trajectory['train_loss'].append(loss.item())
            trajectory['ce_loss'].append(ce_loss.item())
            trajectory['eval_acc'].append(acc)
            trajectory['weight_norms'].append(total_norm)

            # Detect grokking
            if trajectory['grok_step'] is None and acc >= GROK_THRESHOLD:
                trajectory['grok_step'] = step

    final_acc = trajectory['eval_acc'][-1]
    return final_acc, trajectory, model.state_dict()


# %%
# =============================================================================
# RUN EXPERIMENTS
# =============================================================================
n = 5
num_layers = 3
rank = 5
seed = 1

experiments = [
    # (name, optimizer_type, lr, wd)
    ('AdamW wd=0.01', 'adamw', 0.01, 0.01),
    ('AdamW wd=0.03', 'adamw', 0.01, 0.03),
    ('AdamW wd=0.1', 'adamw', 0.01, 0.1),
    ('Adam+L2 wd=0.01', 'adam_l2', 0.01, 0.01),
    ('Adam+L2 wd=0.03', 'adam_l2', 0.01, 0.03),
    ('Adam+L2 wd=0.1', 'adam_l2', 0.01, 0.1),
    ('Muon wd=0.01', 'muon', 0.01, 0.01),
    ('Muon wd=0.03', 'muon', 0.01, 0.03),
    ('Muon wd=0.1', 'muon', 0.01, 0.1),
    ('SGD wd=0.01', 'sgd', 0.01, 0.01),
    ('SGD wd=0.1', 'sgd', 0.01, 0.1),
]

print(f"Config: n={n}, layers={num_layers}, rank={rank}, seed={seed}")
print(f"Grok threshold: {GROK_THRESHOLD:.0%}")
print(f"Running {len(experiments)} experiments...\n")

results = {}

for name, opt_type, lr, wd in experiments:
    print(f"{name}...", end=" ", flush=True)
    try:
        acc, traj, state = train_experiment(n, num_layers, rank, seed, opt_type, lr, wd)
        grok_step = traj['grok_step']
        grok_str = f"step {grok_step}" if grok_step else "never"
        print(f"{acc:.1%} (grok: {grok_str})")
        results[name] = {
            'config': (opt_type, lr, wd),
            'final_acc': acc,
            'trajectory': traj,
            'state_dict': {k: v.cpu() for k, v in state.items()},
        }
    except Exception as e:
        print(f"FAILED: {e}")
        results[name] = None

# %%
# =============================================================================
# SUMMARY
# =============================================================================
print("\n" + "="*70)
print("SUMMARY")
print("="*70)

print(f"\n{'Experiment':<25} {'Final Acc':<12} {'Grok Step':<12} {'Grok?':<8}")
print("-" * 60)
for name in results:
    if results[name] is None:
        print(f"{name:<25} FAILED")
        continue
    acc = results[name]['final_acc']
    grok_step = results[name]['trajectory']['grok_step']
    grok_str = str(grok_step) if grok_step else "never"
    grok_yn = "YES" if grok_step else "no"
    print(f"{name:<25} {acc:<12.1%} {grok_str:<12} {grok_yn:<8}")

# %%
# =============================================================================
# VISUALIZATION
# =============================================================================
fig, axes = plt.subplots(2, 2, figsize=(14, 10))

# Group by optimizer type
optimizer_groups = {
    'AdamW': [n for n in results if n.startswith('AdamW')],
    'Adam+L2': [n for n in results if n.startswith('Adam+L2')],
    'Muon': [n for n in results if n.startswith('Muon')],
    'SGD': [n for n in results if n.startswith('SGD')],
}

colors = plt.cm.tab10(np.linspace(0, 1, 10))

# Plot 1: All accuracy trajectories
ax = axes[0, 0]
for i, name in enumerate(results):
    if results[name] is None:
        continue
    traj = results[name]['trajectory']
    ax.plot(traj['steps'], [a*100 for a in traj['eval_acc']],
            label=name, alpha=0.8, linewidth=1.5)
ax.axhline(y=GROK_THRESHOLD*100, color='red', linestyle='--', linewidth=2, label=f'Grok threshold ({GROK_THRESHOLD:.0%})')
ax.set_xlabel('Step')
ax.set_ylabel('Accuracy (%)')
ax.set_title('All Experiments - Accuracy')
ax.legend(fontsize=7, loc='lower right')
ax.grid(True, alpha=0.3)
ax.set_ylim(0, 100)

# Plot 2: AdamW vs Adam+L2 comparison
ax = axes[0, 1]
for name in optimizer_groups['AdamW']:
    if results[name] is None:
        continue
    traj = results[name]['trajectory']
    ax.plot(traj['steps'], [a*100 for a in traj['eval_acc']],
            label=name, linewidth=2)
for name in optimizer_groups['Adam+L2']:
    if results[name] is None:
        continue
    traj = results[name]['trajectory']
    ax.plot(traj['steps'], [a*100 for a in traj['eval_acc']],
            '--', label=name, linewidth=2)
ax.axhline(y=GROK_THRESHOLD*100, color='red', linestyle=':', linewidth=1)
ax.set_xlabel('Step')
ax.set_ylabel('Accuracy (%)')
ax.set_title('AdamW (solid) vs Adam+L2 (dashed)')
ax.legend(fontsize=8)
ax.grid(True, alpha=0.3)
ax.set_ylim(0, 100)

# Plot 3: Weight norms
ax = axes[1, 0]
for i, name in enumerate(results):
    if results[name] is None:
        continue
    traj = results[name]['trajectory']
    ax.plot(traj['steps'], traj['weight_norms'],
            label=name, alpha=0.8, linewidth=1.5)
ax.set_xlabel('Step')
ax.set_ylabel('Total Weight Norm')
ax.set_title('Weight Norms')
ax.legend(fontsize=7, loc='upper right')
ax.grid(True, alpha=0.3)

# Plot 4: Grok step vs final accuracy
ax = axes[1, 1]
grok_data = []
for name in results:
    if results[name] is None:
        continue
    grok_step = results[name]['trajectory']['grok_step']
    final_acc = results[name]['final_acc']
    if grok_step:
        grok_data.append((name, grok_step, final_acc))

if grok_data:
    names, steps, accs = zip(*grok_data)
    ax.scatter(steps, [a*100 for a in accs], s=100, c='blue', edgecolor='black')
    for i, name in enumerate(names):
        ax.annotate(name.replace(' ', '\n'), (steps[i], accs[i]*100),
                    fontsize=7, ha='center', va='bottom')
ax.set_xlabel('Grok Step')
ax.set_ylabel('Final Accuracy (%)')
ax.set_title('Grok Step vs Final Accuracy')
ax.grid(True, alpha=0.3)

plt.suptitle(f'Grokking Investigation - Seed {seed}, n={n}, {num_layers}-layer', fontsize=14)
plt.tight_layout()
plt.savefig('grokking_investigation.png', dpi=150)
plt.show()

# %%
# Save results
save_path = Path("grokking_investigation_results.pkl")
with open(save_path, 'wb') as f:
    pickle.dump({
        'config': {'n': n, 'num_layers': num_layers, 'rank': rank, 'seed': seed},
        'experiments': experiments,
        'results': results,
        'grok_threshold': GROK_THRESHOLD,
    }, f)
print(f"\nSaved to {save_path}")

# %%
