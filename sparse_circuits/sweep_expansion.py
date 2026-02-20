# %% [markdown]
# # Sweep: Fidelity Weight × Hidden Dimension Expansion
#
# For each expansion factor h' ∈ {1×, 2×, 4×} and λ ∈ {1, 10},
# run Phase 3 (joint rotation + perturbation) for 1k steps.
# Plot the Pareto frontier of (fidelity, sparsity) where 1=best for both.

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

EXPANSION_FACTORS = [1, 2, 4]
LAMBDAS = [1.0, 10.0]
N_STEPS = 1000
LR = 0.005
LOG_EVERY = 200

# %% Imports
import torch
import numpy as np
import matplotlib.pyplot as plt
import math

from odt import run_odt, effective_bond_dim, truncate
from analyze import load_model, extract_cores

sys.path.insert(0, os.path.join(PROJECT_ROOT, "sparse_circuits"))
from sparse_circuits import run_phase3, expand_cores
from sparse_rotation import sparsity_loss

# %% Load model + cached ODT
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

print(f"Ranks: {ranks}, core shapes: {[c.shape for c in cores_trunc]}")

# Load Phase 2 rotation params (for expansion_factor=1 initialization)
phase2_path = os.path.join(RESULTS_DIR, f'{model_name}_sparse_rotation.pt')
phase2 = torch.load(phase2_path, map_location='cpu', weights_only=False)

# %% Compute baselines
device = torch.device(DEVICE)

# Baseline: no rotation, no perturbation
cores_dev = [c.detach().to(device) for c in cores_trunc]
identity_rots = [torch.eye(r, device=device) for r in ranks]
baseline_loss, _ = sparsity_loss(cores_dev, identity_rots, ranks)
baseline_hoyer = 1.0 - baseline_loss.item()

# Phase 2 baseline: rotation only (exact, fidelity=1.0)
phase2_hoyer = 1.0 - phase2['final_loss']

print(f"\nBaselines:")
print(f"  No rotation:    hoyer={baseline_hoyer:.4f}, fidelity=1.0")
print(f"  Phase 2 (rot):  hoyer={phase2_hoyer:.4f}, fidelity=1.0")

# %% Run sweep
results = []

for exp_factor in EXPANSION_FACTORS:
    if exp_factor == 1:
        exp_cores = [c.detach() for c in cores_trunc]
        exp_ranks = list(ranks)
    else:
        exp_cores, exp_ranks = expand_cores(
            [c.detach() for c in cores_trunc], ranks, expansion_factor=exp_factor)

    print(f"\n{'='*60}")
    print(f"Expansion factor: {exp_factor}x -> ranks {exp_ranks}")
    print(f"Core shapes: {[c.shape for c in exp_cores]}")
    total_params = sum(c.numel() for c in exp_cores)
    print(f"Total core parameters: {total_params:,}")

    for lam in LAMBDAS:
        print(f"\n  lambda={lam}, {N_STEPS} steps...")

        # For expansion=1, init from Phase 2 rotations; otherwise identity
        init_rots = phase2['rot_params'] if exp_factor == 1 else None

        result = run_phase3(
            exp_cores, exp_ranks, DEVICE,
            lam=lam, lr=LR, n_steps=N_STEPS,
            log_every=LOG_EVERY,
            init_rotations=init_rots)

        final_sp = result['history']['sparsity'][-1]
        final_fi = result['history']['fidelity'][-1]
        hoyer_score = 1.0 - final_sp   # 1 = perfectly sparse
        similarity = 1.0 - final_fi     # 1 = identical to original

        entry = {
            'exp_factor': exp_factor,
            'lam': lam,
            'ranks': exp_ranks,
            'hoyer': hoyer_score,
            'similarity': similarity,
            'sparsity_loss': final_sp,
            'fidelity_loss': final_fi,
            'history': result['history'],
        }
        results.append(entry)
        print(f"  -> hoyer={hoyer_score:.4f}, similarity={similarity:.4f}")

# %% Save results
save_path = os.path.join(RESULTS_DIR, f'{model_name}_expansion_sweep.pt')
torch.save({
    'results': results,
    'baseline_hoyer': baseline_hoyer,
    'phase2_hoyer': phase2_hoyer,
    'model_name': model_name,
    'ranks': ranks,
}, save_path)
print(f"\nSaved sweep results to {save_path}")

# %% Plot Pareto frontier
fig, ax = plt.subplots(figsize=(8, 6))

# Baselines (fidelity=1.0 for both, on right edge)
ax.plot(1.0, baseline_hoyer, 'kx', markersize=12, markeredgewidth=2,
        label='No rotation', zorder=10)
ax.plot(1.0, phase2_hoyer, 'k*', markersize=14,
        label='Phase 2 (rotation only)', zorder=10)

# Sweep results
colors = {1: '#1f77b4', 2: '#ff7f0e', 4: '#2ca02c'}
markers = {1.0: 'o', 10.0: 's'}

for r in results:
    color = colors[r['exp_factor']]
    marker = markers[r['lam']]
    label = f"{r['exp_factor']}x, \u03bb={int(r['lam'])}"
    ax.plot(r['similarity'], r['hoyer'], marker=marker, color=color,
            markersize=10, label=label, zorder=5)

# Formatting: 1 is best for both axes (upper-right = ideal)
ax.set_xlabel('Fidelity (tensor similarity, 1 = identical)', fontsize=12)
ax.set_ylabel('Sparsity (Hoyer score, 1 = one-hot)', fontsize=12)
ax.set_title(f'{model_name}: Expansion \u00d7 \u03bb Pareto Frontier', fontsize=13)
ax.legend(fontsize=9, loc='lower left')
ax.set_xlim(right=1.02)
ax.grid(True, alpha=0.3)

# Arrow showing "better" direction
ax.annotate('better \u2192\nbetter \u2191', xy=(0.95, 0.95), xycoords='axes fraction',
            fontsize=9, ha='right', va='top', color='gray')

plt.tight_layout()
save_fig = os.path.join(RESULTS_DIR, f'{model_name}_pareto_frontier.png')
plt.savefig(save_fig, dpi=150, bbox_inches='tight')
plt.close()
print(f"Saved plot to {save_fig}")

# %% Summary table
print(f"\n{'='*70}")
print(f"{'Config':<20} {'Fidelity':>10} {'Sparsity':>10} {'Ranks'}")
print(f"{'-'*70}")
print(f"{'No rotation':<20} {'1.0000':>10} {baseline_hoyer:>10.4f} {ranks}")
print(f"{'Phase 2 (rot)':<20} {'1.0000':>10} {phase2_hoyer:>10.4f} {ranks}")
for r in results:
    name = f"{r['exp_factor']}x, \u03bb={int(r['lam'])}"
    print(f"{name:<20} {r['similarity']:>10.4f} {r['hoyer']:>10.4f} {r['ranks']}")
