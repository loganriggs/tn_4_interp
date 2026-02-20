# Fold unembed into last core, then: Phase 2 rotation → prune 90% → fine-tune.
#
# This makes Bond 3 features directly correspond to the 10 SVHN classes,
# eliminating the separate unembed matrix. The last-bond rotation is fixed
# to identity so class dimensions don't get mixed.

import sys, os

PROJECT_ROOT = "/workspace/tn_4_interp"
ODT_DIR = os.path.join(PROJECT_ROOT, "odt")
sys.path.insert(0, ODT_DIR)
os.chdir(ODT_DIR)

import torch
import torch.nn.functional as F
import numpy as np
import math

from odt import effective_bond_dim, truncate
from analyze import load_model, evaluate_truncated
from train import get_svhn_loaders

sys.path.insert(0, os.path.join(PROJECT_ROOT, "sparse_circuits"))
from sparse_rotation import skew_to_rotation, rotate_core, sparsity_loss, count_nonzeros

DEVICE = "cuda"
CHECKPOINT = "models/chi_net_L3_h256.pt"
ODT_CACHE = "models/odt_cache_{model_name}.pt"
RESULTS_DIR = os.path.join(PROJECT_ROOT, "sparse_circuits", "results")
DATA_ROOT = os.path.join(PROJECT_ROOT, "modular_addition/dev_interp/notebooks/data")

# Phase 2 (rotation) hyperparams
ROT_LR = 0.01
ROT_STEPS = 2000
ROT_LOG_EVERY = 100

# Pruning + fine-tuning hyperparams
PRUNE_PCT = 90
FT_LR = 1e-4
FT_STEPS = 5000
FT_EVAL_EVERY = 250
BATCH_SIZE = 512


def forward_odt(x, cores, embed, unembed):
    """Standard ODT bilinear forward pass."""
    ones = torch.ones(x.shape[0], 1, device=x.device)
    h = torch.cat([ones, x], dim=-1)
    h = h @ embed.T
    for core in cores:
        h = torch.einsum('oij,bi,bj->bo', core, h, h)
    return h @ unembed.T


# ================================================================
# 1. Load and truncate ODT
# ================================================================
print("=" * 60)
print("Step 1: Load model and ODT")
print("=" * 60)

model, ckpt = load_model(CHECKPOINT, DEVICE)
model_name = f"{ckpt['model_type']}_L{ckpt['n_layers']}_h{ckpt['hidden_dim']}"
device = torch.device(DEVICE)

odt_results = torch.load(
    ODT_CACHE.format(model_name=model_name), map_location='cpu', weights_only=False)
ranks = effective_bond_dim(odt_results['eigenvalues'])
print(f"Ranks: {ranks}")

embed_trunc, cores_trunc, unembed_trunc = truncate(
    odt_results['embed_orth'], odt_results['cores_orth'],
    odt_results['unembed_mod'], odt_results['eigenvectors'], ranks)

print(f"Core shapes: {[c.shape for c in cores_trunc]}")
print(f"Embed: {embed_trunc.shape}, Unembed: {unembed_trunc.shape}")

# Load test data
print("\nLoading data...")
train_loader, test_loader, _ = get_svhn_loaders(data_root=DATA_ROOT, batch_size=BATCH_SIZE)

# Baseline accuracy
acc_baseline = evaluate_truncated(
    embed_trunc, cores_trunc, unembed_trunc, model, test_loader, DEVICE)
print(f"Baseline (truncated) accuracy: {acc_baseline:.4f}")

# ================================================================
# 2. Fold unembed into last core
# ================================================================
print("\n" + "=" * 60)
print("Step 2: Fold unembed into last core")
print("=" * 60)

print(f"  Last core shape: {cores_trunc[-1].shape}")  # (10, 20, 20)
print(f"  Unembed shape: {unembed_trunc.shape}")       # (10, 10)

# T_folded[c,i,j] = sum_k unembed[c,k] * T[k,i,j]
cores_folded = [c.clone() for c in cores_trunc]
cores_folded[-1] = torch.einsum('ck,kij->cij', unembed_trunc, cores_trunc[-1])
unembed_folded = torch.eye(10)  # identity — logits come directly from last core output

print(f"  Folded last core shape: {cores_folded[-1].shape}")
print(f"  Unembed is now identity: {unembed_folded.shape}")

# Verify folding preserves accuracy
acc_folded = evaluate_truncated(
    embed_trunc, cores_folded, unembed_folded, model, test_loader, DEVICE)
print(f"  Accuracy after folding: {acc_folded:.4f} (should match {acc_baseline:.4f})")
assert abs(acc_folded - acc_baseline) < 1e-6, \
    f"Folding changed accuracy! {acc_folded:.4f} vs {acc_baseline:.4f}"

# ================================================================
# 3. Phase 2: Optimize rotations (last bond fixed to identity)
# ================================================================
print("\n" + "=" * 60)
print("Step 3: Phase 2 rotation optimization (Q_last = identity)")
print("=" * 60)

# Ranks for the folded model: same ranks, but now the "output" of the last core
# directly indexes classes. We still have the same bond dimensions.
# ranks = [47, 60, 20, 10] — 4 bonds for 3 cores
n_layers = len(cores_folded)
n_bonds = n_layers + 1  # bonds 0..3

cores_dev = [c.detach().to(device) for c in cores_folded]

# Create rotation parameters for bonds 0..L-1 (NOT the last bond)
rot_params = []
for i in range(n_bonds - 1):  # bonds 0, 1, 2 — skip bond 3
    r = ranks[i]
    n_params = r * (r - 1) // 2
    p = torch.zeros(n_params, device=device, requires_grad=True)
    rot_params.append(p)

print(f"  Optimizing {n_bonds-1} rotations (bonds 0..{n_bonds-2}), "
      f"bond {n_bonds-1} fixed to identity")
print(f"  Parameters per bond: {[r*(r-1)//2 for r in ranks[:-1]]}")
print(f"  Total rotation params: {sum(r*(r-1)//2 for r in ranks[:-1])}")

# Baseline sparsity with identity rotations
identity_rots = [torch.eye(r, device=device) for r in ranks]
baseline_loss, baseline_stats = sparsity_loss(cores_dev, identity_rots, ranks)
print(f"\n  Baseline sparsity loss: {baseline_loss.item():.4f}")
for ell in range(n_layers):
    s = baseline_stats[f'layer_{ell}']
    print(f"    Layer {ell}: loss={s['loss']:.4f}, shape={s['shape']}")

# Optimize
optimizer_rot = torch.optim.Adam(rot_params, lr=ROT_LR)
history_rot = {'loss': []}

print(f"\n  Optimizing ({ROT_STEPS} steps, lr={ROT_LR})...")
for step in range(ROT_STEPS):
    optimizer_rot.zero_grad()

    # Build full rotation list: optimized for bonds 0..L-1, identity for bond L
    rotations = [skew_to_rotation(p, ranks[i]) for i, p in enumerate(rot_params)]
    rotations.append(torch.eye(ranks[-1], device=device))  # fixed identity for last bond

    loss, stats = sparsity_loss(cores_dev, rotations, ranks)
    loss.backward()
    optimizer_rot.step()
    history_rot['loss'].append(loss.item())

    if step % ROT_LOG_EVERY == 0 or step == ROT_STEPS - 1:
        nnz = count_nonzeros(cores_dev, rotations)
        total_nnz = sum(v['nnz'] for v in nnz.values())
        total_entries = sum(v['total'] for v in nnz.values())
        print(f"    Step {step:4d}: loss={loss.item():.4f}, "
              f"nnz={total_nnz}/{total_entries} ({1-total_nnz/total_entries:.1%} sparse)")

# Apply final rotations
print("\n  Applying final rotations...")
with torch.no_grad():
    rotations_final = [skew_to_rotation(p, ranks[i]) for i, p in enumerate(rot_params)]
    rotations_final.append(torch.eye(ranks[-1], device=device))

    rotated_cores = []
    for ell in range(n_layers):
        Q_in = rotations_final[ell]
        Q_out = rotations_final[ell + 1]
        T_rot = rotate_core(cores_dev[ell], Q_in, Q_out)
        rotated_cores.append(T_rot)

    Q_first = rotations_final[0]
    embed_rot = Q_first @ embed_trunc.to(device)
    # Q_last is identity, so unembed_rot = unembed_folded @ I = unembed_folded
    unembed_rot = unembed_folded.to(device)

# Verify rotation preserves accuracy
acc_rotated = evaluate_truncated(
    embed_rot.cpu(), [c.cpu() for c in rotated_cores], unembed_rot.cpu(),
    model, test_loader, DEVICE)
print(f"  Accuracy after rotation: {acc_rotated:.4f} (should match {acc_folded:.4f})")
assert abs(acc_rotated - acc_folded) < 1e-3, \
    f"Rotation changed accuracy! {acc_rotated:.4f} vs {acc_folded:.4f}"

# ================================================================
# 4. Prune 90% by magnitude
# ================================================================
print("\n" + "=" * 60)
print(f"Step 4: Prune {PRUNE_PCT}% of weights by magnitude")
print("=" * 60)

all_vals = torch.cat([c.abs().flatten() for c in rotated_cores])
threshold = torch.quantile(all_vals, PRUNE_PCT / 100.0).item()
masks = [(c.abs() >= threshold).float() for c in rotated_cores]
cores_pruned = [(c * m).detach() for c, m in zip(rotated_cores, masks)]

for i, m in enumerate(masks):
    nnz = int(m.sum().item())
    tot = int(m.numel())
    print(f"  Layer {i}: {nnz}/{tot} nonzero ({100*(1-nnz/tot):.1f}% sparse)")

total_nnz = sum(m.sum().item() for m in masks)
total_params = sum(m.numel() for m in masks)
print(f"  Total: {int(total_nnz)}/{int(total_params)} nonzero "
      f"({100*(1-total_nnz/total_params):.1f}% sparse)")

acc_pruned = evaluate_truncated(
    embed_rot.cpu(), [c.cpu() for c in cores_pruned], unembed_rot.cpu(),
    model, test_loader, DEVICE)
print(f"  Accuracy after pruning: {acc_pruned:.4f}")

# ================================================================
# 5. Fine-tune with masked gradients
# ================================================================
print("\n" + "=" * 60)
print(f"Step 5: Fine-tune ({FT_STEPS} steps, lr={FT_LR})")
print("=" * 60)

core_params = [c.clone().requires_grad_(True) for c in cores_pruned]
embed_param = embed_rot.clone().requires_grad_(True)
# unembed is identity — we can choose to fine-tune it or not.
# Keeping it fixed at identity to maintain class alignment.
unembed_param = unembed_rot.clone()  # NOT trainable

optimizer_ft = torch.optim.Adam(core_params + [embed_param], lr=FT_LR)
history_ft = {'train_loss': [], 'test_acc': []}
train_iter = iter(train_loader)

for step in range(FT_STEPS):
    try:
        x, y = next(train_iter)
    except StopIteration:
        train_iter = iter(train_loader)
        x, y = next(train_iter)

    optimizer_ft.zero_grad()
    x, y = x.to(device), y.to(device)
    logits = forward_odt(x, core_params, embed_param, unembed_param)
    loss = F.cross_entropy(logits, y)
    loss.backward()

    # Mask core gradients + re-enforce masks
    with torch.no_grad():
        for p, m in zip(core_params, masks):
            if p.grad is not None:
                p.grad *= m
    optimizer_ft.step()
    with torch.no_grad():
        for p, m in zip(core_params, masks):
            p.data *= m

    history_ft['train_loss'].append(loss.item())
    if step % FT_EVAL_EVERY == 0 or step == FT_STEPS - 1:
        acc = evaluate_truncated(
            embed_param.detach().cpu(), [c.detach().cpu() for c in core_params],
            unembed_param.detach().cpu(), model, test_loader, DEVICE)
        history_ft['test_acc'].append(acc)
        print(f"  Step {step:4d}: loss={loss.item():.4f}, acc={acc:.4f}")

# ================================================================
# 6. Save checkpoint
# ================================================================
print("\n" + "=" * 60)
print("Step 6: Save checkpoint")
print("=" * 60)

save_path = os.path.join(RESULTS_DIR, f'{model_name}_folded_sparse.pt')
final_acc = history_ft['test_acc'][-1]

torch.save({
    'model_name': model_name,
    'ranks': ranks,
    'cores': [c.detach().cpu() for c in core_params],
    'embed': embed_param.detach().cpu(),
    'unembed': unembed_param.detach().cpu(),  # identity matrix
    'masks': [m.cpu() for m in masks],
    'prune_pct': PRUNE_PCT,
    'acc_baseline': acc_baseline,
    'acc_folded': acc_folded,
    'acc_rotated': acc_rotated,
    'acc_pruned': acc_pruned,
    'acc_finetuned': final_acc,
    'history_rot': history_rot,
    'history_ft': history_ft,
    'rot_params': [p.detach().cpu() for p in rot_params],
    'rotations': [Q.detach().cpu() for Q in rotations_final],
    # For reference
    'odt_eigenvalues': odt_results['eigenvalues'],
    'odt_eigenvectors': odt_results['eigenvectors'],
    'unembed_folded': True,  # flag: unembed is folded into last core
}, save_path)

print(f"Saved to {save_path}")
print(f"\nAccuracy pipeline:")
print(f"  Baseline (truncated ODT):  {acc_baseline:.4f}")
print(f"  After fold unembed:        {acc_folded:.4f}")
print(f"  After rotation:            {acc_rotated:.4f}")
print(f"  After {PRUNE_PCT}% prune:           {acc_pruned:.4f}")
print(f"  After fine-tune:           {final_acc:.4f}")
