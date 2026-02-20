# %% [markdown]
# # Fine-tune pruned models with data
#
# Two approaches compared:
# (A) Phase 3 λ=10 -> prune 90% -> fine-tune weights only
# (B) Phase 2 (exact) -> prune 90% -> fine-tune weights + rotation jointly
#
# Approach B should dominate because it starts from 79.7% (exact)
# instead of 72.1% (perturbed).

# %% Setup
import sys, os

PROJECT_ROOT = "/workspace/tn_4_interp"
ODT_DIR = os.path.join(PROJECT_ROOT, "odt")
sys.path.insert(0, ODT_DIR)
os.chdir(ODT_DIR)

import torch
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt

from odt import effective_bond_dim, truncate
from analyze import load_model, extract_cores, evaluate_truncated
from train import get_svhn_loaders

sys.path.insert(0, os.path.join(PROJECT_ROOT, "sparse_circuits"))
from sparse_circuits import run_phase3
from sparse_rotation import skew_to_rotation, rotate_core

DEVICE = "cuda"
CHECKPOINT = "models/chi_net_L3_h256.pt"
ODT_CACHE = "models/odt_cache_{model_name}.pt"
RESULTS_DIR = os.path.join(PROJECT_ROOT, "sparse_circuits", "results")

PRUNE_PCT = 90
LR = 1e-4
N_STEPS = 5000
EVAL_EVERY = 250
BATCH_SIZE = 512

# %% Load
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

print("Loading data...")
train_loader, test_loader, _ = get_svhn_loaders(
    data_root=os.path.join(PROJECT_ROOT, "modular_addition/dev_interp/notebooks/data"),
    batch_size=BATCH_SIZE)

phase2 = torch.load(os.path.join(RESULTS_DIR, f'{model_name}_sparse_rotation.pt'),
                     map_location='cpu', weights_only=False)


# %% Helpers
def forward_odt(x, cores, embed, unembed):
    """ODT bilinear forward pass."""
    ones = torch.ones(x.shape[0], 1, device=x.device)
    h = torch.cat([ones, x], dim=-1)
    h = h @ embed.T
    for core in cores:
        h = torch.einsum('oij,bi,bj->bo', core, h, h)
    return h @ unembed.T


def forward_with_rotation(x, cores_raw, rot_params, ranks, embed_trunc, unembed_trunc):
    """Forward pass: rotate raw cores on-the-fly, then run ODT forward."""
    rots = [skew_to_rotation(p, r) for p, r in zip(rot_params, ranks)]
    rotated = []
    for ell in range(len(cores_raw)):
        T_rot = rotate_core(cores_raw[ell], rots[ell], rots[ell + 1])
        rotated.append(T_rot)
    embed = rots[0] @ embed_trunc
    unembed = unembed_trunc @ rots[-1].T
    return forward_odt(x, rotated, embed, unembed)


def prune_cores(cores, prune_pct):
    """Magnitude-prune cores, return pruned cores + masks."""
    all_vals = torch.cat([c.abs().flatten() for c in cores])
    threshold = torch.quantile(all_vals, prune_pct / 100.0).item()
    masks = [(c.abs() >= threshold).float() for c in cores]
    pruned = [(c * m).detach() for c, m in zip(cores, masks)]
    total_nnz = sum(m.sum().item() for m in masks)
    total = sum(m.numel() for m in masks)
    print(f"  Pruned {prune_pct}%: {int(total_nnz)}/{int(total)} nonzero "
          f"({100-100*total_nnz/total:.1f}% sparse)")
    return pruned, masks


def train_loop(name, core_params, masks, embed_param, unembed_param,
               extra_params=None, forward_fn=None):
    """Generic training loop with masked gradients on cores."""
    all_params = list(core_params) + [embed_param, unembed_param]
    if extra_params:
        all_params += list(extra_params)
    optimizer = torch.optim.Adam(all_params, lr=LR)

    history = {'train_loss': [], 'test_acc': []}
    train_iter = iter(train_loader)

    def get_batch():
        nonlocal train_iter
        try:
            return next(train_iter)
        except StopIteration:
            train_iter = iter(train_loader)
            return next(train_iter)

    for step in range(N_STEPS):
        optimizer.zero_grad()
        x, y = get_batch()
        x, y = x.to(device), y.to(device)

        if forward_fn:
            logits = forward_fn(x)
        else:
            logits = forward_odt(x, core_params, embed_param, unembed_param)

        loss = F.cross_entropy(logits, y)
        loss.backward()

        # Mask core gradients
        with torch.no_grad():
            for p, m in zip(core_params, masks):
                if p.grad is not None:
                    p.grad *= m

        optimizer.step()

        # Re-enforce masks
        with torch.no_grad():
            for p, m in zip(core_params, masks):
                p.data *= m

        history['train_loss'].append(loss.item())

        if step % EVAL_EVERY == 0 or step == N_STEPS - 1:
            # Evaluate
            if forward_fn:
                # Need to materialize the rotated cores for eval
                with torch.no_grad():
                    if extra_params:  # rotation params
                        rots = [skew_to_rotation(p, r) for p, r in zip(extra_params, ranks)]
                        rot_cores = [rotate_core(c, rots[ell], rots[ell+1])
                                     for ell, c in enumerate(core_params)]
                        e = (rots[0] @ embed_trunc.to(device)).cpu()
                        u = (unembed_trunc.to(device) @ rots[-1].T).cpu()
                    test_acc = evaluate_truncated(
                        e, [c.detach().cpu() for c in rot_cores], u,
                        model, test_loader, DEVICE)
            else:
                test_acc = evaluate_truncated(
                    embed_param.detach().cpu(),
                    [c.detach().cpu() for c in core_params],
                    unembed_param.detach().cpu(),
                    model, test_loader, DEVICE)
            history['test_acc'].append(test_acc)
            print(f"  [{name}] Step {step:4d}: loss={loss.item():.4f}, test_acc={test_acc:.4f}")

    return history


# %% (A) Phase 3 λ=10 -> prune -> fine-tune weights only
print("\n" + "="*60)
print("(A) Phase 3 λ=10 -> prune 90% -> fine-tune weights")
print("="*60)

print("Running Phase 3 (λ=10)...")
result_lam10 = run_phase3(
    cores_trunc, ranks, DEVICE,
    lam=10.0, lr=0.005, n_steps=1000, log_every=500,
    init_rotations=phase2['rot_params'],
    early_stop_patience=100, early_stop_rtol=1e-4)

with torch.no_grad():
    rots = [skew_to_rotation(p.to(device), r) for p, r in zip(result_lam10['rot_params'], ranks)]
    embed_A = (rots[0] @ embed_trunc.to(device)).detach()
    unembed_A = (unembed_trunc.to(device) @ rots[-1].T).detach()
cores_A = [c.to(device).detach() for c in result_lam10['cores_final']]

acc_A_before = evaluate_truncated(
    embed_A.cpu(), [c.cpu() for c in cores_A], unembed_A.cpu(),
    model, test_loader, DEVICE)
print(f"Before pruning: {acc_A_before:.4f}")

cores_A_pruned, masks_A = prune_cores(cores_A, PRUNE_PCT)
acc_A_pruned = evaluate_truncated(
    embed_A.cpu(), [c.cpu() for c in cores_A_pruned], unembed_A.cpu(),
    model, test_loader, DEVICE)
print(f"After pruning:  {acc_A_pruned:.4f}")

core_params_A = [c.clone().requires_grad_(True) for c in cores_A_pruned]
embed_A_param = embed_A.clone().requires_grad_(True)
unembed_A_param = unembed_A.clone().requires_grad_(True)

history_A = train_loop("A", core_params_A, masks_A, embed_A_param, unembed_A_param)


# %% (B) Phase 2 (exact) -> prune -> fine-tune weights + rotation
print("\n" + "="*60)
print("(B) Phase 2 (exact) -> prune 90% -> fine-tune weights + rotation")
print("="*60)

# Start from Phase 2 rotated cores
cores_B = [c.to(device).detach() for c in phase2['rotated_cores']]
embed_B = phase2['embed_rot'].to(device).detach()
unembed_B = phase2['unembed_rot'].to(device).detach()

acc_B_before = evaluate_truncated(
    embed_B.cpu(), [c.cpu() for c in cores_B], unembed_B.cpu(),
    model, test_loader, DEVICE)
print(f"Before pruning: {acc_B_before:.4f} (exact rotation)")

cores_B_pruned, masks_B = prune_cores(cores_B, PRUNE_PCT)
acc_B_pruned = evaluate_truncated(
    embed_B.cpu(), [c.cpu() for c in cores_B_pruned], unembed_B.cpu(),
    model, test_loader, DEVICE)
print(f"After pruning:  {acc_B_pruned:.4f}")

core_params_B = [c.clone().requires_grad_(True) for c in cores_B_pruned]
embed_B_param = embed_B.clone().requires_grad_(True)
unembed_B_param = unembed_B.clone().requires_grad_(True)

history_B = train_loop("B-weights", core_params_B, masks_B, embed_B_param, unembed_B_param)


# %% (C) Phase 2 -> prune -> fine-tune weights + rotation jointly
print("\n" + "="*60)
print("(C) Phase 2 -> prune 90% -> fine-tune weights + rotation jointly")
print("="*60)

# Work in the UN-rotated basis: learn raw cores + rotation params together
# The masks apply in the rotated basis, so we rotate, mask, then forward
cores_raw_C = [c.detach().to(device) for c in cores_trunc]

# Initialize rotation from Phase 2
rot_params_C = [p.clone().to(device).requires_grad_(True) for p in phase2['rot_params']]

# Compute initial rotated cores to get masks
with torch.no_grad():
    rots_init = [skew_to_rotation(p, r) for p, r in zip(rot_params_C, ranks)]
    rotated_init = [rotate_core(c, rots_init[ell], rots_init[ell+1])
                    for ell, c in enumerate(cores_raw_C)]

_, masks_C = prune_cores(rotated_init, PRUNE_PCT)

# Learnable: perturbation to raw cores (applied before rotation)
deltas_C = [torch.zeros_like(c, requires_grad=True) for c in cores_raw_C]
embed_C = embed_trunc.to(device).detach()  # will be rotated on-the-fly
unembed_C = unembed_trunc.to(device).detach()

def forward_C(x):
    """Forward with joint rotation + masked cores."""
    rots = [skew_to_rotation(p, r) for p, r in zip(rot_params_C, ranks)]
    cores_pert = [c + d for c, d in zip(cores_raw_C, deltas_C)]
    rotated = []
    for ell in range(len(cores_pert)):
        T_rot = rotate_core(cores_pert[ell], rots[ell], rots[ell + 1])
        # Apply pruning mask in rotated basis
        T_rot = T_rot * masks_C[ell]
        rotated.append(T_rot)
    embed_rot = rots[0] @ embed_C
    unembed_rot = unembed_C @ rots[-1].T
    return forward_odt(x, rotated, embed_rot, unembed_rot)

# Need custom training loop since rotation changes the mask interpretation
all_params_C = deltas_C + rot_params_C
optimizer_C = torch.optim.Adam(all_params_C, lr=LR)
history_C = {'train_loss': [], 'test_acc': []}
train_iter_C = iter(train_loader)

def get_batch_C():
    global train_iter_C
    try:
        return next(train_iter_C)
    except StopIteration:
        train_iter_C = iter(train_loader)
        return next(train_iter_C)

for step in range(N_STEPS):
    optimizer_C.zero_grad()
    x, y = get_batch_C()
    x, y = x.to(device), y.to(device)

    logits = forward_C(x)
    loss = F.cross_entropy(logits, y)
    loss.backward()
    optimizer_C.step()

    history_C['train_loss'].append(loss.item())

    if step % EVAL_EVERY == 0 or step == N_STEPS - 1:
        with torch.no_grad():
            rots = [skew_to_rotation(p, r) for p, r in zip(rot_params_C, ranks)]
            cores_pert = [c + d for c, d in zip(cores_raw_C, deltas_C)]
            rot_cores = [rotate_core(c, rots[ell], rots[ell+1]) * masks_C[ell]
                         for ell, c in enumerate(cores_pert)]
            e = (rots[0] @ embed_C).cpu()
            u = (unembed_C @ rots[-1].T).cpu()
        test_acc = evaluate_truncated(
            e, [c.detach().cpu() for c in rot_cores], u,
            model, test_loader, DEVICE)
        history_C['test_acc'].append(test_acc)
        print(f"  [C-joint] Step {step:4d}: loss={loss.item():.4f}, test_acc={test_acc:.4f}")

# %% Sparsity verification
print("\n--- Sparsity verification ---")
for label, cores in [("A", core_params_A), ("B", core_params_B)]:
    nnz = sum((c.abs() > 1e-10).sum().item() for c in cores)
    total = sum(c.numel() for c in cores)
    print(f"  {label}: {nnz}/{total} nonzero ({100-100*nnz/total:.1f}% pruned)")

# C: check in rotated basis
with torch.no_grad():
    rots = [skew_to_rotation(p, r) for p, r in zip(rot_params_C, ranks)]
    cores_pert = [c + d for c, d in zip(cores_raw_C, deltas_C)]
    rot_cores_C = [rotate_core(c, rots[ell], rots[ell+1]) * masks_C[ell]
                   for ell, c in enumerate(cores_pert)]
    nnz = sum((c.abs() > 1e-10).sum().item() for c in rot_cores_C)
    total = sum(c.numel() for c in rot_cores_C)
    print(f"  C: {nnz}/{total} nonzero ({100-100*nnz/total:.1f}% pruned)")

# %% Plot
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

# Training loss (smoothed)
window = 50
for hist, label, color in [
    (history_A, 'A: Phase3 λ=10 + prune + ft', '#ff7f0e'),
    (history_B, 'B: Phase2 + prune + ft weights', '#1f77b4'),
    (history_C, 'C: Phase2 + prune + ft weights+rot', '#2ca02c'),
]:
    losses = hist['train_loss']
    if len(losses) > window:
        smooth = np.convolve(losses, np.ones(window)/window, mode='valid')
        ax1.plot(range(window-1, len(losses)), smooth, color=color, linewidth=2, label=label)
ax1.set_xlabel('Step')
ax1.set_ylabel('Cross-entropy loss')
ax1.set_title('Training loss (smoothed)')
ax1.legend(fontsize=8)
ax1.grid(True, alpha=0.3)
ax1.set_ylim(0, 10)

# Test accuracy
eval_steps = list(range(0, N_STEPS, EVAL_EVERY)) + [N_STEPS - 1]
for hist, label, color, marker in [
    (history_A, 'A: Phase3 λ=10 + prune + ft', '#ff7f0e', 's'),
    (history_B, 'B: Phase2 + prune + ft weights', '#1f77b4', 'o'),
    (history_C, 'C: Phase2 + prune + ft weights+rot', '#2ca02c', 'D'),
]:
    ax2.plot(eval_steps[:len(hist['test_acc'])], hist['test_acc'],
             f'{marker}-', color=color, markersize=5, linewidth=2, label=label)

ax2.axhline(0.7984, color='gray', ls=':', alpha=0.5, label='original (0.7984)')
ax2.axhline(acc_B_before, color='#1f77b4', ls='--', alpha=0.5,
             label=f'Phase2 pre-prune ({acc_B_before:.4f})')
ax2.axhline(acc_B_pruned, color='#1f77b4', ls=':', alpha=0.5,
             label=f'Phase2 post-prune ({acc_B_pruned:.4f})')
ax2.set_xlabel('Step')
ax2.set_ylabel('Test accuracy')
ax2.set_title(f'Fine-tuning at {PRUNE_PCT}% pruning (lr={LR})')
ax2.legend(fontsize=7, loc='lower right')
ax2.grid(True, alpha=0.3)

plt.suptitle(f'{model_name}: Fine-tune comparison at {PRUNE_PCT}% pruning', fontsize=13)
plt.tight_layout()
save_fig = os.path.join(RESULTS_DIR, f'{model_name}_finetune_comparison.png')
plt.savefig(save_fig, dpi=150, bbox_inches='tight')
plt.close()
print(f"\nSaved plot to {save_fig}")

# Summary
summary = f"""Fine-tune Comparison Summary
============================
All at {PRUNE_PCT}% magnitude pruning, lr={LR}, {N_STEPS} steps

                          Pre-prune   Post-prune  Fine-tuned
Original model:           0.7984
A (Phase3 λ=10 + ft):    {acc_A_before:.4f}      {acc_A_pruned:.4f}       {history_A['test_acc'][-1]:.4f}
B (Phase2 + ft weights):  {acc_B_before:.4f}      {acc_B_pruned:.4f}       {history_B['test_acc'][-1]:.4f}
C (Phase2 + ft w+rot):    {acc_B_before:.4f}      {acc_B_pruned:.4f}       {history_C['test_acc'][-1]:.4f}

Sparsity: all ~{PRUNE_PCT}% pruned
"""
summary_path = os.path.join(RESULTS_DIR, f'finetune_comparison_summary.txt')
with open(summary_path, 'w') as f:
    f.write(summary)
print(summary)
