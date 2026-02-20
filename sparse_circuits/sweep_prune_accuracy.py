# %% [markdown]
# # Accuracy vs Weight Ablation Pareto Frontier
#
# For each sparse circuit config (Phase 2, Phase 3 λ=1, λ=10),
# sweep over magnitude pruning thresholds and measure accuracy.
# Plots: accuracy vs % weights zeroed.

# %% Configuration
import sys, os

PROJECT_ROOT = "/workspace/tn_4_interp"
ODT_DIR = os.path.join(PROJECT_ROOT, "odt")
sys.path.insert(0, ODT_DIR)
os.chdir(ODT_DIR)

CHECKPOINT = "models/chi_net_L3_h256.pt"
ODT_CACHE = "models/odt_cache_{model_name}.pt"
DEVICE = "cuda"
RESULTS_DIR = os.path.join(PROJECT_ROOT, "sparse_circuits", "results")
os.makedirs(RESULTS_DIR, exist_ok=True)

# Pruning thresholds to sweep
PRUNE_PCTS = [0, 10, 20, 30, 40, 50, 60, 70, 80, 85, 90, 95, 98, 99]

# %% Imports
import torch
import numpy as np
import matplotlib.pyplot as plt
import math

from odt import run_odt, effective_bond_dim, truncate
from analyze import load_model, extract_cores, evaluate_truncated
from train import get_svhn_loaders

sys.path.insert(0, os.path.join(PROJECT_ROOT, "sparse_circuits"))
from sparse_circuits import run_phase3
from sparse_rotation import skew_to_rotation, rotate_core

# %% Load model + cached ODT + test data
print("Loading model...")
model, ckpt = load_model(CHECKPOINT, DEVICE)
model_name = f"{ckpt['model_type']}_L{ckpt['n_layers']}_h{ckpt['hidden_dim']}"

cache_path = ODT_CACHE.format(model_name=model_name)
if os.path.isfile(cache_path):
    print(f"Loading cached ODT from {cache_path}")
    odt_results = torch.load(cache_path, map_location='cpu', weights_only=False)
else:
    print("Running ODT...")
    embed, cores_raw, unembed = extract_cores(model.cpu())
    odt_results = run_odt(embed, cores_raw, unembed, verbose=True)
    torch.save(odt_results, cache_path)

ranks = effective_bond_dim(odt_results['eigenvalues'])
embed_trunc, cores_trunc, unembed_trunc = truncate(
    odt_results['embed_orth'], odt_results['cores_orth'],
    odt_results['unembed_mod'], odt_results['eigenvectors'], ranks)
print(f"Ranks: {ranks}")

print("Loading test data...")
_, test_loader, _ = get_svhn_loaders(
    data_root=os.path.join(PROJECT_ROOT, "modular_addition/dev_interp/notebooks/data"))


# %% Helper: prune cores by magnitude percentile and evaluate
def prune_and_eval(cores, embed, unembed, prune_pct, model, test_loader, device):
    """Zero out bottom prune_pct% of core entries by magnitude, evaluate accuracy."""
    if prune_pct == 0:
        return evaluate_truncated(embed, cores, unembed, model, test_loader, device)

    # Gather all magnitudes across cores
    all_vals = torch.cat([c.abs().flatten() for c in cores])
    threshold = torch.quantile(all_vals, prune_pct / 100.0).item()

    # Prune
    pruned_cores = []
    for c in cores:
        mask = c.abs() >= threshold
        pruned_cores.append(c * mask.float())

    return evaluate_truncated(embed, pruned_cores, unembed, model, test_loader, device)


def get_sparsity_at_threshold(cores, prune_pct):
    """Return actual % of zeros after pruning at given percentile."""
    all_vals = torch.cat([c.abs().flatten() for c in cores])
    if prune_pct == 0:
        return (all_vals == 0).float().mean().item() * 100
    threshold = torch.quantile(all_vals, prune_pct / 100.0).item()
    return (all_vals < threshold).float().mean().item() * 100


# %% Build configs to evaluate

# Original model accuracy (no ODT truncation)
print("\nOriginal model accuracy...")
orig_acc = ckpt.get('test_acc', None)
if orig_acc is None:
    # evaluate original model
    model.to(DEVICE)
    model.eval()
    correct = total = 0
    with torch.no_grad():
        for x, y in test_loader:
            x, y = x.to(DEVICE), y.to(DEVICE)
            logits = model(x)
            correct += (logits.argmax(-1) == y).sum().item()
            total += y.size(0)
    orig_acc = correct / total
print(f"  Original model: {orig_acc:.4f}")

# ODT truncated baseline (no rotation)
print("ODT truncated baseline...")
trunc_acc = evaluate_truncated(embed_trunc, cores_trunc, unembed_trunc,
                                model, test_loader, DEVICE)
print(f"  Truncated (no rotation): {trunc_acc:.4f}")

# Phase 2: rotation only
print("\nLoading Phase 2 (rotation only)...")
phase2 = torch.load(os.path.join(RESULTS_DIR, f'{model_name}_sparse_rotation.pt'),
                     map_location='cpu', weights_only=False)
phase2_cores = phase2['rotated_cores']
phase2_embed = phase2['embed_rot']
phase2_unembed = phase2['unembed_rot']

# Phase 3: λ=1 (more sparse, less faithful)
print("Running Phase 3 (λ=1, 1x)...")
result_lam1 = run_phase3(
    cores_trunc, ranks, DEVICE,
    lam=1.0, lr=0.005, n_steps=1000, log_every=200,
    init_rotations=phase2['rot_params'],
    early_stop_patience=100, early_stop_rtol=1e-4)

# Reconstruct embed/unembed for λ=1
device = torch.device(DEVICE)
with torch.no_grad():
    rots_lam1 = [skew_to_rotation(p.to(device), r)
                 for p, r in zip(result_lam1['rot_params'], ranks)]
    embed_lam1 = (rots_lam1[0] @ embed_trunc.to(device)).cpu()
    unembed_lam1 = (unembed_trunc.to(device) @ rots_lam1[-1].T).cpu()
cores_lam1 = result_lam1['cores_final']

# Phase 3: λ=10 (less sparse, more faithful)
print("\nRunning Phase 3 (λ=10, 1x)...")
result_lam10 = run_phase3(
    cores_trunc, ranks, DEVICE,
    lam=10.0, lr=0.005, n_steps=1000, log_every=200,
    init_rotations=phase2['rot_params'],
    early_stop_patience=100, early_stop_rtol=1e-4)

with torch.no_grad():
    rots_lam10 = [skew_to_rotation(p.to(device), r)
                  for p, r in zip(result_lam10['rot_params'], ranks)]
    embed_lam10 = (rots_lam10[0] @ embed_trunc.to(device)).cpu()
    unembed_lam10 = (unembed_trunc.to(device) @ rots_lam10[-1].T).cpu()
cores_lam10 = result_lam10['cores_final']

# %% Sweep pruning thresholds for each config
configs = {
    'ODT truncated (no rot)': (cores_trunc, embed_trunc, unembed_trunc),
    'Phase 2 (rotation only)': (phase2_cores, phase2_embed, phase2_unembed),
    'Phase 3 (λ=10)': (cores_lam10, embed_lam10, unembed_lam10),
    'Phase 3 (λ=1)': (cores_lam1, embed_lam1, unembed_lam1),
}

all_results = {}
for name, (cores, embed, unembed) in configs.items():
    print(f"\n--- {name} ---")
    accs = []
    sparsities = []
    for pct in PRUNE_PCTS:
        acc = prune_and_eval(cores, embed, unembed, pct, model, test_loader, DEVICE)
        sp = get_sparsity_at_threshold(cores, pct)
        accs.append(acc)
        sparsities.append(sp)
        print(f"  {pct:3d}% pruned -> {sp:5.1f}% zeros, acc={acc:.4f}")
    all_results[name] = {
        'prune_pcts': PRUNE_PCTS,
        'accuracies': accs,
        'sparsities': sparsities,
    }

# %% Save
save_path = os.path.join(RESULTS_DIR, f'{model_name}_prune_accuracy_sweep.pt')
torch.save({
    'configs': all_results,
    'orig_acc': orig_acc,
    'trunc_acc': trunc_acc,
    'model_name': model_name,
    'ranks': ranks,
}, save_path)
print(f"\nSaved to {save_path}")

# %% Plot: Accuracy vs % weights zeroed
fig, ax = plt.subplots(figsize=(10, 6))

colors = {
    'ODT truncated (no rot)': '#888888',
    'Phase 2 (rotation only)': '#1f77b4',
    'Phase 3 (λ=10)': '#ff7f0e',
    'Phase 3 (λ=1)': '#2ca02c',
}
markers = {
    'ODT truncated (no rot)': 'x',
    'Phase 2 (rotation only)': 'o',
    'Phase 3 (λ=10)': 's',
    'Phase 3 (λ=1)': 'D',
}

for name, data in all_results.items():
    ax.plot(data['sparsities'], data['accuracies'],
            color=colors[name], marker=markers[name],
            markersize=6, linewidth=2, label=name)

ax.axhline(orig_acc, color='k', ls='--', alpha=0.5, label=f'Original model ({orig_acc:.4f})')
ax.axhline(trunc_acc, color='gray', ls=':', alpha=0.5, label=f'ODT truncated ({trunc_acc:.4f})')

ax.set_xlabel('% of core weights zeroed', fontsize=12)
ax.set_ylabel('Test accuracy', fontsize=12)
ax.set_title(f'{model_name}: Accuracy vs Weight Ablation', fontsize=13)
ax.legend(fontsize=9, loc='lower left')
ax.set_xlim(-2, 100)
ax.grid(True, alpha=0.3)

plt.tight_layout()
save_fig = os.path.join(RESULTS_DIR, f'{model_name}_prune_accuracy.png')
plt.savefig(save_fig, dpi=150, bbox_inches='tight')
plt.close()
print(f"Saved plot to {save_fig}")

# %% Summary
summary_path = os.path.join(RESULTS_DIR, 'prune_accuracy_summary.txt')
with open(summary_path, 'w') as f:
    f.write(f"Accuracy vs Weight Ablation Summary\n")
    f.write(f"{'='*70}\n\n")
    f.write(f"Original model accuracy: {orig_acc:.4f}\n")
    f.write(f"ODT truncated accuracy:  {trunc_acc:.4f}\n\n")

    for name, data in all_results.items():
        f.write(f"--- {name} ---\n")
        f.write(f"{'% pruned':>10} {'% zeros':>10} {'accuracy':>10}\n")
        for pct, sp, acc in zip(data['prune_pcts'], data['sparsities'], data['accuracies']):
            f.write(f"{pct:>10d} {sp:>10.1f} {acc:>10.4f}\n")
        f.write(f"\n")

print(f"Summary written to {summary_path}")
