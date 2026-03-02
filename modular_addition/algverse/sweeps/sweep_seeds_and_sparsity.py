# %%
"""
Seed + Sparsity Sweep for Symmetric Bilinear Models

Configurations:
- n=5, rank=5, 2-layer
- n=6, rank=6, 2-layer
- n=5, rank=5, 3-layer
- n=6, rank=6, 3-layer

For each config:
1. Sweep seeds (0-9) to find best
2. L1 train + prune at various thresholds
3. Find max sparsity within -1% of baseline accuracy
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
# TRAINING FUNCTIONS
# =============================================================================
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Device: {device}")


def train_model(n, num_layers, rank, seed, verbose=False):
    """Train a model and return final accuracy."""
    torch.manual_seed(seed)
    model = SymmetricBilinearResidual(n, num_layers, rank).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.01, weight_decay=0.001)

    for step in range(10001):
        model.train()
        x = torch.randn(128, n, device=device)
        targets = task_2nd_argmax(x)
        logits = model(x)
        loss = nn.functional.cross_entropy(logits, targets)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    model.eval()
    with torch.no_grad():
        x_eval = torch.randn(10000, n, device=device)
        targets_eval = task_2nd_argmax(x_eval)
        logits_eval = model(x_eval)
        acc = (logits_eval.argmax(-1) == targets_eval).float().mean().item()

    if verbose:
        print(f"  Seed {seed}: {acc:.1%}")

    return acc, model.state_dict()


def train_with_l1_and_prune(n, num_layers, rank, seed, thresholds, verbose=False):
    """Train with L1, then prune at various thresholds."""
    # Phase 1a: Warm-up
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

    # Phase 1b: L1 penalty
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
            l1_loss += layer.L.abs().sum() + layer.D.abs().sum()

        loss = ce_loss + l1_lambda * l1_loss
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    # Eval baseline (after L1)
    model.eval()
    with torch.no_grad():
        x_eval = torch.randn(10000, n, device=device)
        targets_eval = task_2nd_argmax(x_eval)
        logits_eval = model(x_eval)
        baseline_acc = (logits_eval.argmax(-1) == targets_eval).float().mean().item()

    if verbose:
        print(f"  Baseline (after L1): {baseline_acc:.1%}")

    l1_state = {k: v.clone() for k, v in model.state_dict().items()}

    # Phase 2 & 3: Prune + fine-tune at each threshold
    results = []
    for thresh in thresholds:
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
                    if layer.D.grad is not None:
                        layer.D.grad *= masks[f'D{i}']
            optimizer_ft.step()
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

        results.append({
            'threshold': thresh,
            'pruned_frac': pruned_frac,
            'final_acc': final_acc,
            'acc_drop': baseline_acc - final_acc,
            'state_dict': {k: v.cpu() for k, v in model_pruned.state_dict().items()},
        })

        if verbose:
            print(f"    t={thresh}: {pruned_frac:.1%} pruned, {final_acc:.1%} acc (Δ={baseline_acc - final_acc:+.1%})")

    return baseline_acc, results


# %%
# =============================================================================
# MAIN SWEEP
# =============================================================================
configs = [
    {'n': 5, 'num_layers': 1, 'rank': 5},
    {'n': 6, 'num_layers': 1, 'rank': 6},
    {'n': 5, 'num_layers': 2, 'rank': 5},
    {'n': 6, 'num_layers': 2, 'rank': 6},
    {'n': 5, 'num_layers': 3, 'rank': 5},
    {'n': 6, 'num_layers': 3, 'rank': 6},
]

seeds = list(range(10))
thresholds = [0.05, 0.10, 0.15, 0.20, 0.25, 0.30]
acceptance_threshold = 0.01  # -1%

all_results = {}

for cfg in configs:
    n, num_layers, rank = cfg['n'], cfg['num_layers'], cfg['rank']
    cfg_name = f"n{n}_L{num_layers}_r{rank}"
    print(f"\n{'='*60}")
    print(f"Config: n={n}, layers={num_layers}, rank={rank}")
    print(f"{'='*60}")

    # Phase 1: Seed sweep
    print("\nPhase 1: Seed sweep")
    seed_results = []
    for seed in seeds:
        acc, state = train_model(n, num_layers, rank, seed, verbose=True)
        seed_results.append({'seed': seed, 'acc': acc, 'state': state})

    best_seed_result = max(seed_results, key=lambda x: x['acc'])
    best_seed = best_seed_result['seed']
    best_acc = best_seed_result['acc']
    print(f"\nBest seed: {best_seed} with {best_acc:.1%} accuracy")

    # Phase 2: L1 + sparsity sweep on best seed
    print(f"\nPhase 2: L1 + sparsity sweep (seed={best_seed})")
    baseline_acc, prune_results = train_with_l1_and_prune(
        n, num_layers, rank, best_seed, thresholds, verbose=True
    )

    # Find best sparsity within acceptance threshold
    acceptable = [r for r in prune_results if r['acc_drop'] <= acceptance_threshold]
    if acceptable:
        best_prune = max(acceptable, key=lambda x: x['pruned_frac'])
        print(f"\nBest within -{acceptance_threshold:.0%}: t={best_prune['threshold']}, "
              f"{best_prune['pruned_frac']:.1%} pruned, {best_prune['final_acc']:.1%} acc")
    else:
        best_prune = min(prune_results, key=lambda x: x['acc_drop'])
        print(f"\nNo threshold within -{acceptance_threshold:.0%}. Best: t={best_prune['threshold']}, "
              f"{best_prune['pruned_frac']:.1%} pruned, {best_prune['final_acc']:.1%} acc")

    all_results[cfg_name] = {
        'config': cfg,
        'seed_results': seed_results,
        'best_seed': best_seed,
        'best_seed_acc': best_acc,
        'baseline_acc': baseline_acc,
        'prune_results': prune_results,
        'best_prune': best_prune,
    }

# %%
# =============================================================================
# SUMMARY
# =============================================================================
print("\n" + "="*80)
print("SUMMARY")
print("="*80)

print(f"\n{'Config':<20} {'Best Seed':<12} {'Seed Acc':<12} {'L1 Acc':<12} {'Best Thresh':<12} {'Pruned':<12} {'Final Acc':<12}")
print("-" * 95)

for cfg_name, res in all_results.items():
    cfg = res['config']
    print(f"n={cfg['n']},L={cfg['num_layers']},r={cfg['rank']:<5} "
          f"{res['best_seed']:<12} "
          f"{res['best_seed_acc']:<12.1%} "
          f"{res['baseline_acc']:<12.1%} "
          f"{res['best_prune']['threshold']:<12} "
          f"{res['best_prune']['pruned_frac']:<12.1%} "
          f"{res['best_prune']['final_acc']:<12.1%}")

# %%
# =============================================================================
# VISUALIZATION
# =============================================================================
fig, axes = plt.subplots(2, 3, figsize=(18, 10))

for idx, (cfg_name, res) in enumerate(all_results.items()):
    ax = axes[idx // 3, idx % 3]
    cfg = res['config']

    thresh_vals = [r['threshold'] for r in res['prune_results']]
    pruned_fracs = [r['pruned_frac'] * 100 for r in res['prune_results']]
    final_accs = [r['final_acc'] * 100 for r in res['prune_results']]

    ax2 = ax.twinx()

    bars = ax.bar(thresh_vals, pruned_fracs, width=0.03, alpha=0.7, color='steelblue', label='Pruned %')
    line, = ax2.plot(thresh_vals, final_accs, 'ro-', markersize=8, linewidth=2, label='Accuracy')

    ax2.axhline(y=res['baseline_acc'] * 100, color='green', linestyle='--', linewidth=1.5, label=f'Baseline: {res["baseline_acc"]:.1%}')
    ax2.axhline(y=(res['baseline_acc'] - acceptance_threshold) * 100, color='orange', linestyle=':', linewidth=1.5, label=f'-{acceptance_threshold:.0%} threshold')

    ax.set_xlabel('Pruning Threshold')
    ax.set_ylabel('Pruned %', color='steelblue')
    ax2.set_ylabel('Accuracy %', color='red')
    ax.set_title(f"n={cfg['n']}, {cfg['num_layers']}-layer, rank={cfg['rank']} (seed={res['best_seed']})")

    # Mark best
    best_t = res['best_prune']['threshold']
    best_idx = thresh_vals.index(best_t)
    ax2.scatter([best_t], [res['best_prune']['final_acc'] * 100], s=200, marker='*', color='gold', zorder=10, edgecolor='black')

    lines1, labels1 = ax.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax.legend(lines1 + lines2, labels1 + labels2, loc='lower left', fontsize=8)

plt.tight_layout()
plt.savefig('sweep_results.png', dpi=150)
plt.show()

# %%
# Save results
save_path = Path("sweep_seeds_sparsity_results.pkl")
# Remove state_dicts to save space (can be regenerated)
save_results = {}
for cfg_name, res in all_results.items():
    save_results[cfg_name] = {
        'config': res['config'],
        'best_seed': res['best_seed'],
        'best_seed_acc': res['best_seed_acc'],
        'baseline_acc': res['baseline_acc'],
        'prune_summary': [
            {'threshold': r['threshold'], 'pruned_frac': r['pruned_frac'], 'final_acc': r['final_acc'], 'acc_drop': r['acc_drop']}
            for r in res['prune_results']
        ],
        'best_prune_threshold': res['best_prune']['threshold'],
        'best_prune_frac': res['best_prune']['pruned_frac'],
        'best_prune_acc': res['best_prune']['final_acc'],
    }

with open(save_path, 'wb') as f:
    pickle.dump(save_results, f)
print(f"\nSaved summary to {save_path}")

# %%
