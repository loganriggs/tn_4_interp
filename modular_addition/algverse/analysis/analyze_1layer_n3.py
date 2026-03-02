# %%
"""
1-Layer Bilinear Analysis for n=3 (2nd-argmax task)

Train, iteratively sparsify (remove bottom 10% each round), and visualize M matrices.
"""

import torch
import torch.nn as nn
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from pathlib import Path
import sys

PROJECT_ROOT = Path("/workspace/tn_4_interp/modular_addition/algverse")
sys.path.insert(0, str(PROJECT_ROOT))
from models import SymmetricBilinearResidual, task_2nd_argmax
from analysis.analysis_utils import (
    compute_quadratic_forms, plot_mat, plot_weights, plot_M_eigen, plot_M_only
)

checkpoint_dir = PROJECT_ROOT / "checkpoints"
images_dir = PROJECT_ROOT / "images" / "1layer_n3"
images_dir.mkdir(exist_ok=True, parents=True)

device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Device: {device}")

N = 3
RANK = 4

# %%
# =============================================================================
# PHASE 0: TRAIN BASE MODEL
# =============================================================================
print("=" * 60)
print("PHASE 0: TRAIN BASE MODEL (n=3, 1-layer, rank=3)")
print("=" * 60)

def evaluate(model, n_samples=10000):
    model.eval()
    with torch.no_grad():
        x = torch.randn(n_samples, N, device=device)
        targets = task_2nd_argmax(x)
        output = model(x)
        preds = output.argmax(dim=1)
        acc = (preds == targets).float().mean().item()
    return acc

def get_sparsity(model):
    total = 0
    zeros = 0
    for layer in model.layers:
        total += layer.L.numel() + layer.D.numel()
        zeros += (layer.L == 0).sum().item() + (layer.D == 0).sum().item()
    return zeros / total

torch.manual_seed(0)
model = SymmetricBilinearResidual(N, 1, RANK).to(device)
optimizer = torch.optim.AdamW(model.parameters(), lr=0.01, weight_decay=0.01)

model.train()
for step in range(5001):
    x = torch.randn(256, N, device=device)
    targets = task_2nd_argmax(x)
    loss = nn.functional.cross_entropy(model(x), targets)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    if step % 1000 == 0:
        acc = evaluate(model)
        print(f"Step {step}: loss={loss.item():.4f}, acc={acc*100:.1f}%")

baseline_acc = evaluate(model)
print(f"\nBaseline accuracy: {baseline_acc*100:.1f}%")

# %%
# =============================================================================
# ITERATIVE PRUNING: remove bottom 10%, fine-tune, repeat
# =============================================================================
print("\n" + "=" * 60)
print("ITERATIVE PRUNING (remove bottom 10% each round)")
print("=" * 60)

PRUNE_PERCENT = 0.10
MAX_ITERATIONS = 15
MAX_ACC_DROP = 0.03
FINETUNE_STEPS = 3000

def finetune_with_masks(model, masks, steps=3000):
    """Fine-tune while keeping pruned weights at zero."""
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.005, weight_decay=0.001)

    for step in range(steps):
        model.train()
        x = torch.randn(128, N, device=device)
        targets = task_2nd_argmax(x)
        loss = nn.functional.cross_entropy(model(x), targets)

        optimizer.zero_grad()
        loss.backward()

        # Mask gradients
        with torch.no_grad():
            for i, layer in enumerate(model.layers):
                if layer.L.grad is not None:
                    layer.L.grad *= masks[f'L{i}']
                if layer.D.grad is not None:
                    layer.D.grad *= masks[f'D{i}']

        optimizer.step()

        # Re-apply mask
        with torch.no_grad():
            for i, layer in enumerate(model.layers):
                layer.L.data *= masks[f'L{i}']
                layer.D.data *= masks[f'D{i}']

    return model

# Initialize masks (all ones)
masks = {}
for i, layer in enumerate(model.layers):
    masks[f'L{i}'] = torch.ones_like(layer.L)
    masks[f'D{i}'] = torch.ones_like(layer.D)

results = []
best_within_threshold = None

for iteration in range(MAX_ITERATIONS):
    print(f"\nIteration {iteration + 1}")
    print("-" * 40)

    # Collect all nonzero weight magnitudes
    all_weights = []
    for i, layer in enumerate(model.layers):
        L_masked = layer.L.detach() * masks[f'L{i}']
        D_masked = layer.D.detach() * masks[f'D{i}']
        all_weights.append(L_masked.abs().flatten())
        all_weights.append(D_masked.abs().flatten())

    all_weights = torch.cat(all_weights)
    nonzero_weights = all_weights[all_weights > 0]

    if len(nonzero_weights) == 0:
        print("All weights pruned!")
        break

    # Find threshold to prune bottom PRUNE_PERCENT
    k = max(int(len(nonzero_weights) * PRUNE_PERCENT), 1)
    threshold = torch.kthvalue(nonzero_weights, k).values.item()

    print(f"  Pruning threshold: {threshold:.6f}")
    print(f"  Remaining nonzero: {len(nonzero_weights)}, removing ~{k}")

    # Update masks
    with torch.no_grad():
        for i, layer in enumerate(model.layers):
            L_abs = layer.L.abs()
            D_abs = layer.D.abs()
            masks[f'L{i}'] = ((L_abs > threshold) | (L_abs == 0)).float() * masks[f'L{i}']
            masks[f'D{i}'] = ((D_abs > threshold) | (D_abs == 0)).float() * masks[f'D{i}']
            layer.L.data *= masks[f'L{i}']
            layer.D.data *= masks[f'D{i}']

    sparsity = get_sparsity(model)
    acc_before_ft = evaluate(model)
    print(f"  After pruning: acc={acc_before_ft:.1%}, sparsity={sparsity:.1%}")

    # Fine-tune
    model = finetune_with_masks(model, masks, FINETUNE_STEPS)
    acc_after_ft = evaluate(model)
    print(f"  After fine-tune: acc={acc_after_ft:.1%}")

    result = {
        'iteration': iteration + 1,
        'threshold': threshold,
        'sparsity': sparsity,
        'acc_before_ft': acc_before_ft,
        'acc_after_ft': acc_after_ft,
        'acc_drop': baseline_acc - acc_after_ft,
        'state_dict': {k: v.cpu().clone() for k, v in model.state_dict().items()},
    }
    results.append(result)

    if result['acc_drop'] <= 0.01:
        best_within_threshold = result
        print(f"  ** Within 1% threshold (drop={result['acc_drop']:.1%})")

    if result['acc_drop'] > MAX_ACC_DROP:
        print(f"  Stopping: accuracy drop {result['acc_drop']:.1%} > {MAX_ACC_DROP:.0%}")
        break

# =============================================================================
# SUMMARY TABLE
# =============================================================================
print("\n" + "=" * 60)
print("PRUNING SUMMARY")
print("=" * 60)

print(f"\nBaseline accuracy: {baseline_acc:.1%}")
print(f"\n{'Iter':>4} {'Acc(pre-ft)':>11} {'Acc(post-ft)':>12} {'Δ':>7} {'Sparsity':>9} {'Nonzero':>8}")
print("-" * 55)

total_params = sum(l.L.numel() + l.D.numel() for l in model.layers)
for r in results:
    nz = int(total_params * (1 - r['sparsity']))
    marker = " *" if best_within_threshold and r['iteration'] == best_within_threshold['iteration'] else ""
    print(f"  {r['iteration']:2d}   {r['acc_before_ft']:>9.1%}   {r['acc_after_ft']:>10.1%}  {-r['acc_drop']:>+6.1%}  {r['sparsity']:>8.1%}  {nz:>6}/{total_params}{marker}")

# Pick best model
if best_within_threshold:
    best = best_within_threshold
    print(f"\nBest within 1%: iteration {best['iteration']}, acc={best['acc_after_ft']:.1%}, sparsity={best['sparsity']:.1%}")
else:
    best = min(results, key=lambda r: r['acc_drop'])
    print(f"\nBest overall: iteration {best['iteration']}, acc={best['acc_after_ft']:.1%}, sparsity={best['sparsity']:.1%}")

# Load best state
model.load_state_dict({k: v.to(device) for k, v in best['state_dict'].items()})

# Save
save_path = checkpoint_dir / "1layer_n3_r4_seed0_sparse.pt"
torch.save({
    'state_dict': best['state_dict'],
    'config': {'n': N, 'num_layers': 1, 'rank': RANK, 'seed': 0},
    'accuracy': best['acc_after_ft'],
    'sparsity': best['sparsity'],
    'baseline_acc': baseline_acc,
    'iteration': best['iteration'],
    'all_iterations': [{k: v for k, v in r.items() if k != 'state_dict'} for r in results],
}, save_path)
print(f"Saved to {save_path}")

# %%
# =============================================================================
# EXTRACT WEIGHTS AND COMPUTE M
# =============================================================================
print("\n" + "=" * 60)
print("WEIGHT AND M MATRIX ANALYSIS")
print("=" * 60)

L = model.layers[0].L.detach().cpu()
D = model.layers[0].D.detach().cpu()
gamma = model.norms[0].weight.detach().cpu()
M = compute_quadratic_forms(L, D)

print(f"\nL ({L.shape[0]}×{L.shape[1]}):")
print(L.numpy())
print(f"\nD ({D.shape[0]}×{D.shape[1]}):")
print(D.numpy())
print(f"\nγ = {gamma.item():.4f}")

for i in range(N):
    M_i = M[i].numpy()
    eigenvalues = np.linalg.eigvalsh(M_i)
    rank = np.sum(np.abs(eigenvalues) > 1e-6)
    print(f"\nM[{i}] (rank={rank}):")
    print(M_i.round(4))
    print(f"  eigenvalues: {eigenvalues.round(4)}")
    print(f"  self-coeff M[{i},{i}]: {M_i[i,i]:.4f}")

# %%
# =============================================================================
# VISUALIZATIONS
# =============================================================================
print("\n" + "=" * 60)
print("GENERATING VISUALIZATIONS")
print("=" * 60)

final_acc = best['acc_after_ft']
final_sparsity = best['sparsity']

# Per-matrix sparsity
sp_L = (L == 0).sum().item() / L.numel() * 100
sp_D = (D == 0).sum().item() / D.numel() * 100

# --- Weight matrices ---
fig = plot_weights(
    [(L, D, 'Layer1', sp_L, sp_D)],
    title=f'n=3, 1-Layer Weights (Acc: {final_acc*100:.1f}%, Sparsity: {final_sparsity*100:.1f}%)',
    save_path=images_dir / 'weights.png',
)
plt.close()
print("Saved weights.png")

# --- M matrices only (side by side) ---
fig = plot_M_only(M, layer_name='M', save_path=images_dir / 'M_matrices.png')
plt.close()
print("Saved M_matrices.png")

# --- M matrices with eigendecomposition ---
fig = plot_M_eigen(M, 'M', list(range(N)), n_top=N,
                   save_path=images_dir / 'M_eigen.png')
plt.close()
print("Saved M_eigen.png")

# --- M hadamard view ---
outer_labels = [['a²', 'ab', 'ac'], ['ab', 'b²', 'bc'], ['ac', 'bc', 'c²']]

fig, axes = plt.subplots(N, 3, figsize=(16, 14))

for out_idx in range(N):
    M_i = M[out_idx].numpy()

    # Col 0: M matrix
    plot_mat(axes[out_idx, 0], M_i, f'M$^{{({out_idx})}}$', fontsize=14)
    axes[out_idx, 0].set_ylabel(f'Output {out_idx}', fontsize=13, fontweight='bold')

    # Col 1: hadamard symbol
    axes[out_idx, 1].set_xlim(0, 1)
    axes[out_idx, 1].set_ylim(0, 1)
    axes[out_idx, 1].text(0.5, 0.5, '⊙', fontsize=48, ha='center', va='center',
                          fontweight='bold')
    axes[out_idx, 1].axis('off')

    # Col 2: h outer h^T template
    template = np.array([[1.0, 0.6, 0.6], [0.6, 1.0, 0.6], [0.6, 0.6, 1.0]])
    im = axes[out_idx, 2].imshow(template, cmap='Purples', vmin=0, vmax=1.2,
                                  aspect='equal')
    axes[out_idx, 2].set_title('h outer h$^T$  (h = gx/rms)', fontsize=12, fontweight='bold')

    for i in range(N):
        for j in range(N):
            axes[out_idx, 2].text(j, i, outer_labels[i][j],
                                  ha='center', va='center', fontsize=16,
                                  fontweight='bold', color='black',
                                  fontstyle='italic')
    axes[out_idx, 2].set_xticks(range(N))
    axes[out_idx, 2].set_yticks(range(N))
    axes[out_idx, 2].set_xlabel('pos', fontsize=10)
    axes[out_idx, 2].set_ylabel('pos', fontsize=10)

plt.suptitle(
    f'n=3 1-Layer: bilinear_i = sum_jk M^(i)_jk h_j h_k'
    f'\nAcc: {final_acc*100:.1f}%, Sparsity: {final_sparsity*100:.1f}%, g = {gamma.item():.3f}',
    fontsize=13)
plt.tight_layout()
plt.savefig(images_dir / 'M_hadamard.png', dpi=150, bbox_inches='tight')
plt.close()
print("Saved M_hadamard.png")

# %%
# =============================================================================
# INTERPRETATION
# =============================================================================
print("\n" + "=" * 60)
print("INTERPRETATION")
print("=" * 60)

print(f"""
n=3 2nd-argmax task (1-layer, sparse)
======================================

Architecture: output_i = x_i + h^T M^(i) h,  h = γx/rms(x)
γ = {gamma.item():.4f}
Accuracy: {final_acc*100:.1f}%
Sparsity: {final_sparsity*100:.1f}%

Weights:
  L = {L.numpy().round(4)}
  D = {D.numpy().round(4)}

Quadratic Forms:""")

for i in range(N):
    M_i = M[i].numpy()
    eigenvalues, eigenvectors = np.linalg.eigh(M_i)
    abs_order = np.argsort(np.abs(eigenvalues))[::-1]
    print(f"\n  M[{i}] = {M_i.round(4)}")
    print(f"    self-coeff M[{i},{i}] = {M_i[i,i]:.4f}")
    others = [f'M[{i},{j}]={M_i[j,j]:.4f}' for j in range(N) if j != i]
    print(f"    other diag: {', '.join(others)}")
    print(f"    eigenvalues (by |λ|): {eigenvalues[abs_order].round(4)}")

print(f"\nVisualizations saved to {images_dir}/")

# %%
