# Train the best sparse model: Phase 2 (exact rotation) → prune 90% → fine-tune weights
# Saves checkpoint for feature card / circuit analysis.

import sys, os

PROJECT_ROOT = "/workspace/tn_4_interp"
ODT_DIR = os.path.join(PROJECT_ROOT, "odt")
sys.path.insert(0, ODT_DIR)
os.chdir(ODT_DIR)

import torch
import torch.nn.functional as F
import numpy as np

from odt import effective_bond_dim, truncate
from analyze import load_model, evaluate_truncated
from train import get_svhn_loaders

DEVICE = "cuda"
CHECKPOINT = "models/chi_net_L3_h256.pt"
ODT_CACHE = "models/odt_cache_{model_name}.pt"
RESULTS_DIR = os.path.join(PROJECT_ROOT, "sparse_circuits", "results")

PRUNE_PCT = 90
LR = 1e-4
N_STEPS = 5000
EVAL_EVERY = 250
BATCH_SIZE = 512


def forward_odt(x, cores, embed, unembed):
    ones = torch.ones(x.shape[0], 1, device=x.device)
    h = torch.cat([ones, x], dim=-1)
    h = h @ embed.T
    for core in cores:
        h = torch.einsum('oij,bi,bj->bo', core, h, h)
    return h @ unembed.T


# ── Load ──
print("Loading model + ODT cache...")
model, ckpt = load_model(CHECKPOINT, DEVICE)
model_name = f"{ckpt['model_type']}_L{ckpt['n_layers']}_h{ckpt['hidden_dim']}"
device = torch.device(DEVICE)

odt_results = torch.load(
    ODT_CACHE.format(model_name=model_name), map_location='cpu', weights_only=False)
ranks = effective_bond_dim(odt_results['eigenvalues'])
embed_trunc, cores_trunc, unembed_trunc = truncate(
    odt_results['embed_orth'], odt_results['cores_orth'],
    odt_results['unembed_mod'], odt_results['eigenvectors'], ranks)
print(f"Ranks: {ranks}")

print("Loading data...")
train_loader, test_loader, _ = get_svhn_loaders(
    data_root=os.path.join(PROJECT_ROOT, "modular_addition/dev_interp/notebooks/data"),
    batch_size=BATCH_SIZE)

phase2 = torch.load(os.path.join(RESULTS_DIR, f'{model_name}_sparse_rotation.pt'),
                     map_location='cpu', weights_only=False)

# ── Phase 2 rotated cores (exact, no accuracy loss) ──
cores_init = [c.to(device).detach() for c in phase2['rotated_cores']]
embed_init = phase2['embed_rot'].to(device).detach()
unembed_init = phase2['unembed_rot'].to(device).detach()

acc_pre_prune = evaluate_truncated(
    embed_init.cpu(), [c.cpu() for c in cores_init], unembed_init.cpu(),
    model, test_loader, DEVICE)
print(f"Pre-prune accuracy: {acc_pre_prune:.4f}")

# ── Prune 90% by magnitude ──
all_vals = torch.cat([c.abs().flatten() for c in cores_init])
threshold = torch.quantile(all_vals, PRUNE_PCT / 100.0).item()
masks = [(c.abs() >= threshold).float() for c in cores_init]
cores_pruned = [(c * m).detach() for c, m in zip(cores_init, masks)]

nnz = sum(m.sum().item() for m in masks)
total = sum(m.numel() for m in masks)
print(f"Pruned {PRUNE_PCT}%: {int(nnz)}/{int(total)} nonzero ({100-100*nnz/total:.1f}% sparse)")

acc_post_prune = evaluate_truncated(
    embed_init.cpu(), [c.cpu() for c in cores_pruned], unembed_init.cpu(),
    model, test_loader, DEVICE)
print(f"Post-prune accuracy: {acc_post_prune:.4f}")

# ── Fine-tune with masked gradients ──
print(f"\nFine-tuning {N_STEPS} steps, lr={LR}...")
core_params = [c.clone().requires_grad_(True) for c in cores_pruned]
embed_param = embed_init.clone().requires_grad_(True)
unembed_param = unembed_init.clone().requires_grad_(True)

optimizer = torch.optim.Adam(core_params + [embed_param, unembed_param], lr=LR)
history = {'train_loss': [], 'test_acc': []}
train_iter = iter(train_loader)

for step in range(N_STEPS):
    try:
        x, y = next(train_iter)
    except StopIteration:
        train_iter = iter(train_loader)
        x, y = next(train_iter)

    optimizer.zero_grad()
    x, y = x.to(device), y.to(device)
    logits = forward_odt(x, core_params, embed_param, unembed_param)
    loss = F.cross_entropy(logits, y)
    loss.backward()

    # Mask core gradients + re-enforce masks
    with torch.no_grad():
        for p, m in zip(core_params, masks):
            if p.grad is not None:
                p.grad *= m
    optimizer.step()
    with torch.no_grad():
        for p, m in zip(core_params, masks):
            p.data *= m

    history['train_loss'].append(loss.item())
    if step % EVAL_EVERY == 0 or step == N_STEPS - 1:
        acc = evaluate_truncated(
            embed_param.detach().cpu(), [c.detach().cpu() for c in core_params],
            unembed_param.detach().cpu(), model, test_loader, DEVICE)
        history['test_acc'].append(acc)
        print(f"  Step {step:4d}: loss={loss.item():.4f}, acc={acc:.4f}")

# ── Save checkpoint ──
save_path = os.path.join(RESULTS_DIR, f'{model_name}_sparse_finetuned.pt')
torch.save({
    'model_name': model_name,
    'ranks': ranks,
    'cores': [c.detach().cpu() for c in core_params],
    'embed': embed_param.detach().cpu(),
    'unembed': unembed_param.detach().cpu(),
    'masks': [m.cpu() for m in masks],
    'prune_pct': PRUNE_PCT,
    'acc_pre_prune': acc_pre_prune,
    'acc_post_prune': acc_post_prune,
    'acc_finetuned': history['test_acc'][-1],
    'history': history,
    'n_steps': N_STEPS,
    'lr': LR,
    # Also save ODT results reference for feature analysis
    'odt_eigenvalues': odt_results['eigenvalues'],
    'odt_eigenvectors': odt_results['eigenvectors'],
    'embed_trunc': embed_trunc,
    'unembed_trunc': unembed_trunc,
    'cores_trunc': cores_trunc,
}, save_path)
print(f"\nSaved to {save_path}")
print(f"Final accuracy: {history['test_acc'][-1]:.4f} ({PRUNE_PCT}% sparse)")
