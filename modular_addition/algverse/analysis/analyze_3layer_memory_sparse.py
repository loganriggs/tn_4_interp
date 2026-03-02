# %%
"""
Analysis of Sparse 3-Layer Memory Model

Analyze the structure of the pruned 3-layer model with 5 "memory" dimensions.
"""

import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import sys
import pickle
from mpl_toolkits.axes_grid1 import make_axes_locatable

# =============================================================================
# PROJECT ROOT
# =============================================================================
PROJECT_ROOT = Path("/workspace/tn_4_interp/modular_addition/algverse")
sys.path.insert(0, str(PROJECT_ROOT))
from models import SymmetricBilinearResidual, task_2nd_argmax
from analysis.analysis_utils import compute_quadratic_forms

checkpoint_dir = PROJECT_ROOT / "checkpoints"
images_dir = PROJECT_ROOT / "images" / "sparse_3layer_memory"
images_dir.mkdir(exist_ok=True, parents=True)

N_TASK = 10
N_MEMORY = 5
N_MODEL = N_TASK + N_MEMORY  # 15
device = "cuda" if torch.cuda.is_available() else "cpu"

# %%
# =============================================================================
# LOAD SPARSE MODEL
# =============================================================================
checkpoint_path = checkpoint_dir / "3layer_n10_memory5_seed2_sparse.pkl"
print(f"Loading sparse model from {checkpoint_path}")

with open(checkpoint_path, 'rb') as f:
    checkpoint = pickle.load(f)

state = checkpoint['state_dict']
config = checkpoint['config']

print(f"\nConfig: n_task={config['n_task']}, n_model={config['n_model']}, rank={config['rank']}")
print(f"Accuracy: {checkpoint['accuracy']*100:.1f}%")
print(f"Sparsity: {checkpoint['sparsity']*100:.1f}%")
print(f"Baseline: {checkpoint['baseline_acc']*100:.1f}%")
print(f"Iteration: {checkpoint['iteration']}")

# Extract weights
L1 = state['layers.0.L']
D1 = state['layers.0.D']
gamma1 = state['norms.0.weight']

L2 = state['layers.1.L']
D2 = state['layers.1.D']
gamma2 = state['norms.1.weight']

L3 = state['layers.2.L']
D3 = state['layers.2.D']
gamma3 = state['norms.2.weight']

print(f"\nWeight shapes:")
print(f"  L1: {L1.shape}, D1: {D1.shape}, gamma1: {gamma1.item():.4f}")
print(f"  L2: {L2.shape}, D2: {D2.shape}, gamma2: {gamma2.item():.4f}")
print(f"  L3: {L3.shape}, D3: {D3.shape}, gamma3: {gamma3.item():.4f}")

# %%
# =============================================================================
# SPARSITY ANALYSIS
# =============================================================================
print("\n" + "=" * 60)
print("SPARSITY ANALYSIS")
print("=" * 60)

def analyze_sparsity(tensor, name):
    zeros = (tensor == 0).sum().item()
    total = tensor.numel()
    sparsity = zeros / total * 100
    print(f"  {name}: {sparsity:.1f}% sparse ({zeros}/{total} zeros)")
    return sparsity

print("\nPer-layer sparsity:")
sp_L1 = analyze_sparsity(L1, "L1")
sp_D1 = analyze_sparsity(D1, "D1")
sp_L2 = analyze_sparsity(L2, "L2")
sp_D2 = analyze_sparsity(D2, "D2")
sp_L3 = analyze_sparsity(L3, "L3")
sp_D3 = analyze_sparsity(D3, "D3")

# %%
# =============================================================================
# MEMORY DIMENSION USAGE
# =============================================================================
print("\n" + "=" * 60)
print(f"MEMORY DIMENSIONS (pos {N_TASK}-{N_MODEL-1}) USAGE")
print("=" * 60)

# L1 memory columns (input memory -> ranks)
L1_mem_cols = L1[:, N_TASK:]
print(f"\nL1 memory columns (input -> ranks):")
print(f"  Non-zero per col: {[(L1[:, j] != 0).sum().item() for j in range(N_TASK, N_MODEL)]}")

# D1 memory rows (ranks -> memory output)
D1_mem_rows = D1[N_TASK:, :]
print(f"\nD1 memory rows (ranks -> memory output):")
print(f"  Non-zero per row: {[(D1[i, :] != 0).sum().item() for i in range(N_TASK, N_MODEL)]}")

# L2 memory columns
L2_mem_cols = L2[:, N_TASK:]
print(f"\nL2 memory columns:")
print(f"  Non-zero per col: {[(L2[:, j] != 0).sum().item() for j in range(N_TASK, N_MODEL)]}")

# D2 memory rows
D2_mem_rows = D2[N_TASK:, :]
print(f"\nD2 memory rows:")
print(f"  Non-zero per row: {[(D2[i, :] != 0).sum().item() for i in range(N_TASK, N_MODEL)]}")

# L3 memory columns
L3_mem_cols = L3[:, N_TASK:]
print(f"\nL3 memory columns:")
print(f"  Non-zero per col: {[(L3[:, j] != 0).sum().item() for j in range(N_TASK, N_MODEL)]}")

# D3 memory rows (should be all zero - not used in loss)
D3_mem_rows = D3[N_TASK:, :]
print(f"\nD3 memory rows (unused in loss):")
print(f"  Non-zero per row: {[(D3[i, :] != 0).sum().item() for i in range(N_TASK, N_MODEL)]}")

# %%
# =============================================================================
# QUADRATIC FORM ANALYSIS
# =============================================================================
print("\n" + "=" * 60)
print("QUADRATIC FORM ANALYSIS")
print("=" * 60)

M1 = compute_quadratic_forms(L1, D1)
M2 = compute_quadratic_forms(L2, D2)
M3 = compute_quadratic_forms(L3, D3)

print(f"\nM1 (layer 1 quadratic forms): {M1.shape}")
print(f"M2 (layer 2 quadratic forms): {M2.shape}")
print(f"M3 (layer 3 quadratic forms): {M3.shape}")

# Analyze M structure for task outputs (0-9)
for layer_idx, (M, name) in enumerate([(M1, 'M1'), (M2, 'M2'), (M3, 'M3')]):
    print(f"\n{name} structure (task outputs 0-9):")
    for i in range(N_TASK):
        M_i = M[i].numpy()
        eigenvalues = np.linalg.eigvalsh(M_i)
        rank = np.sum(np.abs(eigenvalues) > 1e-6)
        diag = np.diag(M_i)
        task_diag = diag[:N_TASK]
        print(f"  {name}[{i}]: self={M_i[i,i]:+.2f}, other_mean={np.delete(task_diag, i).mean():.2f}, rank={rank}")

# %%
# =============================================================================
# FORWARD PASS AND ABLATION
# =============================================================================
print("\n" + "=" * 60)
print("FORWARD PASS AND ABLATION")
print("=" * 60)

# Load model
model = SymmetricBilinearResidual(N_MODEL, 3, config['rank']).to(device)
model.load_state_dict({k: v.to(device) for k, v in state.items()})
model.eval()

def my_rmsnorm(x, gamma):
    rms = (x ** 2).mean(dim=-1, keepdim=True).sqrt()
    return gamma * x / rms, rms.squeeze(-1)

def forward_3layer_memory(x_task):
    """Forward pass with memory dimensions."""
    # Pad with memory (zeros)
    x = torch.cat([x_task, torch.zeros(x_task.shape[0], N_MEMORY, device=x_task.device)], dim=1)

    # Layer 1
    h1, rms1 = my_rmsnorm(x, gamma1.to(x.device))
    Lh1 = h1 @ L1.T.to(x.device)
    r1 = (Lh1 ** 2) @ D1.T.to(x.device)

    # Layer 2
    h_mid1 = x + r1
    h2, rms2 = my_rmsnorm(h_mid1, gamma2.to(x.device))
    Lh2 = h2 @ L2.T.to(x.device)
    r2 = (Lh2 ** 2) @ D2.T.to(x.device)

    # Layer 3
    h_mid2 = x + r1 + r2
    h3, rms3 = my_rmsnorm(h_mid2, gamma3.to(x.device))
    Lh3 = h3 @ L3.T.to(x.device)
    r3 = (Lh3 ** 2) @ D3.T.to(x.device)

    output = x + r1 + r2 + r3
    return output, r1, r2, r3

# Evaluate
torch.manual_seed(123)
x_task = torch.randn(10000, N_TASK, device=device)
targets = task_2nd_argmax(x_task)

output, r1, r2, r3 = forward_3layer_memory(x_task)

# Only use first N_TASK outputs for prediction
preds = output[:, :N_TASK].argmax(dim=1)
accuracy = (preds == targets).float().mean().item()

print(f"\nModel accuracy: {accuracy:.1%}")

# Analyze layer contributions
x_padded = torch.cat([x_task, torch.zeros(10000, N_MEMORY, device=device)], dim=1)
r1_task = r1[:, :N_TASK]
r2_task = r2[:, :N_TASK]
r3_task = r3[:, :N_TASK]
r1_mem = r1[:, N_TASK:]
r2_mem = r2[:, N_TASK:]
r3_mem = r3[:, N_TASK:]

print(f"\nNorm contributions (mean):")
print(f"  ||x_task||: {x_task.norm(dim=1).mean().item():.4f}")
print(f"  ||r1_task||: {r1_task.norm(dim=1).mean().item():.4f}")
print(f"  ||r2_task||: {r2_task.norm(dim=1).mean().item():.4f}")
print(f"  ||r3_task||: {r3_task.norm(dim=1).mean().item():.4f}")
print(f"  ||r1_mem||: {r1_mem.norm(dim=1).mean().item():.4f}")
print(f"  ||r2_mem||: {r2_mem.norm(dim=1).mean().item():.4f}")
print(f"  ||r3_mem||: {r3_mem.norm(dim=1).mean().item():.4f}")

print(f"\nProgressive accuracy:")
print(f"  x only:           {(x_task.argmax(dim=1) == targets).float().mean().item():.1%}")
print(f"  x + r1:           {((x_task + r1_task).argmax(dim=1) == targets).float().mean().item():.1%}")
print(f"  x + r1 + r2:      {((x_task + r1_task + r2_task).argmax(dim=1) == targets).float().mean().item():.1%}")
print(f"  x + r1 + r2 + r3: {accuracy:.1%}")

# Ablation
print(f"\nAblation (removing components):")
print(f"  Without r1: {((x_task + r2_task + r3_task).argmax(dim=1) == targets).float().mean().item():.1%}")
print(f"  Without r2: {((x_task + r1_task + r3_task).argmax(dim=1) == targets).float().mean().item():.1%}")
print(f"  Without r3: {((x_task + r1_task + r2_task).argmax(dim=1) == targets).float().mean().item():.1%}")

# %%
# =============================================================================
# PER-POSITION ACCURACY
# =============================================================================
print("\n" + "=" * 60)
print("PER-POSITION ACCURACY")
print("=" * 60)

# Confusion matrix
confusion = torch.zeros(N_TASK, N_TASK)
for t in range(N_TASK):
    mask = (targets == t)
    if mask.sum() > 0:
        preds_t = preds[mask]
        for p in range(N_TASK):
            confusion[t, p] = (preds_t == p).sum().item()

# Per-position accuracy
print("\nPer-position accuracy:")
for t in range(N_TASK):
    total = confusion[t].sum().item()
    correct = confusion[t, t].item()
    acc = correct / total if total > 0 else 0
    print(f"  Position {t}: {acc:.1%} ({int(correct)}/{int(total)})")

print(f"\nOverall: {accuracy:.1%}")

# %%
# =============================================================================
# R3 DECOMPOSITION INTO A, B, C COMPONENTS (h_prev vs r2 decomposition)
# =============================================================================
print("\n" + "=" * 60)
print("R3 DECOMPOSITION: A (h_prev×h_prev), B (h_prev×r2 cross), C (r2×r2)")
print("=" * 60)

# Layer 3 input: h_mid2 = x + r1 + r2 = h_prev + r2 where h_prev = x + r1
# After norm: h3 = γ3 * (h_prev + r2) / rms3
#
# Decompose:
#   h_prev_norm = γ3 * h_prev / rms3
#   r2_norm = γ3 * r2 / rms3
# So h3 = h_prev_norm + r2_norm
#
# Layer 3 bilinear: r3 = D3 @ (L3 @ h3)²
#                      = D3 @ (L3@h_prev_norm + L3@r2_norm)²
#
# A = D3 @ (L3@h_prev_norm)²           -- previous state × previous state
# B = 2 * D3 @ (L3@h_prev_norm * L3@r2_norm)  -- cross-term
# C = D3 @ (L3@r2_norm)²               -- r2 × r2

def compute_r3_ABC(x_task, L1, D1, L2, D2, L3, D3, gamma1, gamma2, gamma3):
    """
    Compute r3 decomposed into A, B, C based on h_prev vs r2.

    A = h_prev × h_prev contribution through layer 3
    B = h_prev × r2 cross-term through layer 3
    C = r2 × r2 contribution through layer 3

    Returns:
        r3_A, r3_B, r3_C: (batch, n_model) tensors
    """
    device = x_task.device
    batch_size = x_task.shape[0]

    # Pad input with memory dimensions (zeros)
    x = torch.cat([x_task, torch.zeros(batch_size, N_MEMORY, device=device)], dim=1)

    # Layer 1 forward
    rms1 = (x ** 2).mean(dim=-1, keepdim=True).sqrt()
    h1 = gamma1.to(device) * x / rms1
    Lh1 = h1 @ L1.T.to(device)
    r1 = (Lh1 ** 2) @ D1.T.to(device)

    # Layer 2 forward
    h_mid1 = x + r1
    rms2 = (h_mid1 ** 2).mean(dim=-1, keepdim=True).sqrt()
    h2 = gamma2.to(device) * h_mid1 / rms2
    Lh2 = h2 @ L2.T.to(device)
    r2 = (Lh2 ** 2) @ D2.T.to(device)

    # h_prev = x + r1 (state before r2 is added)
    h_prev = x + r1

    # Layer 3 input: h_mid2 = h_prev + r2
    h_mid2 = h_prev + r2
    rms3 = (h_mid2 ** 2).mean(dim=-1, keepdim=True).sqrt()

    # Normalized components
    h_prev_norm = gamma3.to(device) * h_prev / rms3
    r2_norm = gamma3.to(device) * r2 / rms3

    # L3 projections
    L3_dev = L3.to(device)
    D3_dev = D3.to(device)

    L_hprev = h_prev_norm @ L3_dev.T
    L_r2 = r2_norm @ L3_dev.T

    # A: h_prev × h_prev = D3 @ (L3 @ h_prev_norm)²
    r3_A = (L_hprev ** 2) @ D3_dev.T

    # B: h_prev × r2 cross = 2 * D3 @ (L3@h_prev_norm * L3@r2_norm)
    r3_B = (2 * L_hprev * L_r2) @ D3_dev.T

    # C: r2 × r2 = D3 @ (L3 @ r2_norm)²
    r3_C = (L_r2 ** 2) @ D3_dev.T

    return r3_A, r3_B, r3_C, r1, r2, h_prev_norm, r2_norm

# Compute decomposition
r3_A, r3_B, r3_C, r1_recomp, r2_recomp, h_prev_norm, r2_norm = compute_r3_ABC(
    x_task, L1, D1, L2, D2, L3, D3, gamma1, gamma2, gamma3
)

# Verify: A + B + C should equal r3
r3_reconstructed = r3_A + r3_B + r3_C
reconstruction_error = (r3 - r3_reconstructed).abs().max().item()
print(f"\nReconstruction check: max|r3 - (A+B+C)| = {reconstruction_error:.2e}")

# Component statistics
print(f"\nComponent magnitudes (mean over batch, task outputs only):")
print(f"  ||A_task|| (h_prev×h_prev): {r3_A[:, :N_TASK].norm(dim=1).mean().item():.4f}")
print(f"  ||B_task|| (h_prev×r2):     {r3_B[:, :N_TASK].norm(dim=1).mean().item():.4f}")
print(f"  ||C_task|| (r2×r2):         {r3_C[:, :N_TASK].norm(dim=1).mean().item():.4f}")
print(f"  ||r3_task|| (total):        {r3[:, :N_TASK].norm(dim=1).mean().item():.4f}")

print(f"\nPer-output mean magnitudes (task outputs):")
print(f"  A (h_prev²):  {r3_A[:, :N_TASK].abs().mean(dim=0).cpu().numpy()}")
print(f"  B (h_prev×r2): {r3_B[:, :N_TASK].abs().mean(dim=0).cpu().numpy()}")
print(f"  C (r2²):      {r3_C[:, :N_TASK].abs().mean(dim=0).cpu().numpy()}")

# Normalized contribution statistics
print(f"\nNormalized layer 3 input contributions:")
print(f"  ||h_prev_norm||: {h_prev_norm.norm(dim=1).mean().item():.4f}")
print(f"  ||r2_norm||:     {r2_norm.norm(dim=1).mean().item():.4f}")
h3_norm = (h_prev_norm + r2_norm).norm(dim=1).mean().item()
print(f"  ||h_prev_norm|| / ||h3||: {h_prev_norm.norm(dim=1).mean().item() / h3_norm:.4f}")
print(f"  ||r2_norm|| / ||h3||:     {r2_norm.norm(dim=1).mean().item() / h3_norm:.4f}")

# Ablation: accuracy with only each component
print(f"\nAblation - accuracy with different r3 components:")

# Full output without any r3
base_output = x_task + r1_task + r2_task

# Only A (h_prev×h_prev)
output_A = x_padded + r1 + r2 + r3_A
preds_A = output_A[:, :N_TASK].argmax(dim=1)
acc_A = (preds_A == targets).float().mean().item()
print(f"  r3 = A only (h_prev²):       {acc_A:.1%}")

# Only B (h_prev×r2)
output_B = x_padded + r1 + r2 + r3_B
preds_B = output_B[:, :N_TASK].argmax(dim=1)
acc_B = (preds_B == targets).float().mean().item()
print(f"  r3 = B only (h_prev×r2):     {acc_B:.1%}")

# Only C (r2×r2)
output_C = x_padded + r1 + r2 + r3_C
preds_C = output_C[:, :N_TASK].argmax(dim=1)
acc_C = (preds_C == targets).float().mean().item()
print(f"  r3 = C only (r2²):           {acc_C:.1%}")

# A + B (no r2×r2)
output_AB = x_padded + r1 + r2 + r3_A + r3_B
preds_AB = output_AB[:, :N_TASK].argmax(dim=1)
acc_AB = (preds_AB == targets).float().mean().item()
print(f"  r3 = A + B (no r2²):         {acc_AB:.1%}")

# A + C (no cross)
output_AC = x_padded + r1 + r2 + r3_A + r3_C
preds_AC = output_AC[:, :N_TASK].argmax(dim=1)
acc_AC = (preds_AC == targets).float().mean().item()
print(f"  r3 = A + C (no cross):       {acc_AC:.1%}")

# B + C (no h_prev²)
output_BC = x_padded + r1 + r2 + r3_B + r3_C
preds_BC = output_BC[:, :N_TASK].argmax(dim=1)
acc_BC = (preds_BC == targets).float().mean().item()
print(f"  r3 = B + C (no h_prev²):     {acc_BC:.1%}")

# Full r3 for comparison
print(f"  r3 = A + B + C (full):       {accuracy:.1%}")

# Without r3 for baseline
acc_no_r3 = ((x_task + r1_task + r2_task).argmax(dim=1) == targets).float().mean().item()
print(f"  r3 = 0 (no layer 3):         {acc_no_r3:.1%}")

# %%
# =============================================================================
# VISUALIZATIONS
# =============================================================================
print("\n" + "=" * 60)
print("GENERATING VISUALIZATIONS")
print("=" * 60)

# Weight matrices using shared function
from analysis.analysis_utils import plot_weights, plot_M_eigen

fig = plot_weights(
    [(L1, D1, 'Layer1', sp_L1, sp_D1),
     (L2, D2, 'Layer2', sp_L2, sp_D2),
     (L3, D3, 'Layer3', sp_L3, sp_D3)],
    title=f'Sparse 3-Layer Memory Model (Acc: {checkpoint["accuracy"]*100:.1f}%, Sparsity: {checkpoint["sparsity"]*100:.1f}%)\nGreen lines mark memory boundary (task: 0-9, memory: 10-14)',
    memory_boundary=N_TASK,
    save_path=images_dir / 'weights_all.png',
)
plt.show()

# Per-position accuracy bar chart
fig, ax = plt.subplots(figsize=(10, 6))
pos_accs = [confusion[t, t].item() / confusion[t].sum().item() for t in range(N_TASK)]
colors = ['green' if acc > 0.8 else 'orange' if acc > 0.6 else 'red' for acc in pos_accs]
bars = ax.bar(range(N_TASK), pos_accs, color=colors)
ax.axhline(y=accuracy, color='blue', linestyle='--', label=f'Overall: {accuracy:.1%}')
ax.set_xlabel('Position')
ax.set_ylabel('Accuracy')
ax.set_title('Per-Position Accuracy (3-Layer Memory Model)')
ax.set_xticks(range(N_TASK))
ax.legend()
for i, acc in enumerate(pos_accs):
    ax.text(i, acc + 0.02, f'{acc:.1%}', ha='center', fontsize=9)
plt.tight_layout()
plt.savefig(images_dir / 'per_position_accuracy.png', dpi=150)
plt.show()

# Confusion matrix
fig, ax = plt.subplots(figsize=(10, 8))
conf_np = confusion.numpy()
im = ax.imshow(conf_np, cmap='Blues')
ax.set_xlabel('Predicted')
ax.set_ylabel('True')
ax.set_title('Confusion Matrix (3-Layer Memory Model)')
ax.set_xticks(range(N_TASK))
ax.set_yticks(range(N_TASK))
plt.colorbar(im, ax=ax)
for i in range(N_TASK):
    for j in range(N_TASK):
        val = int(conf_np[i, j])
        color = 'white' if val > conf_np.max() * 0.5 else 'black'
        ax.text(j, i, val, ha='center', va='center', color=color, fontsize=8)
plt.tight_layout()
plt.savefig(images_dir / 'confusion_matrix.png', dpi=150)
plt.show()

print(f"\nVisualizations saved to {images_dir}/")

# %%
# =============================================================================
# M1, M2, M3 MATRIX VISUALIZATIONS WITH EIGENDECOMPOSITION
# =============================================================================
print("\n" + "=" * 60)
print("M1, M2, M3 MATRIX VISUALIZATIONS WITH EIGENDECOMPOSITION")
print("=" * 60)

# Plot M1, M2, M3 with eigendecomposition using shared function
for M, name in [(M1, 'M1'), (M2, 'M2'), (M3, 'M3')]:
    print(f"\nGenerating {name} visualizations (outputs 0-4)...")
    plot_M_eigen(M, name, list(range(5)), memory_boundary=N_TASK,
                 save_path=images_dir / f'{name}_eigen_analysis_0to4.png')
    plt.show()

    print(f"Generating {name} visualizations (outputs 5-9)...")
    plot_M_eigen(M, name, list(range(5, 10)), memory_boundary=N_TASK,
                 save_path=images_dir / f'{name}_eigen_analysis_5to9.png')
    plt.show()

    print(f"Generating {name} visualizations (memory outputs 10-14)...")
    plot_M_eigen(M, name, list(range(N_TASK, N_MODEL)), memory_boundary=N_TASK,
                 save_path=images_dir / f'{name}_eigen_analysis_mem.png')
    plt.show()

# Eigenvalue spectra summary
print("\nGenerating eigenvalue spectra summary...")
fig, axes = plt.subplots(3, 1, figsize=(14, 15))

for ax, (M, name) in zip(axes, [(M1, 'M1'), (M2, 'M2'), (M3, 'M3')]):
    for i in range(N_MODEL):
        M_i = M[i].numpy()
        eigenvalues, _ = np.linalg.eigh(M_i)
        label = f'out {i}' if i < N_TASK else f'mem {i - N_TASK}'
        linestyle = '-' if i < N_TASK else '--'
        ax.plot(range(len(eigenvalues)), eigenvalues, marker='o', label=label,
                alpha=0.7, markersize=4, linestyle=linestyle)
    ax.axhline(y=0, color='black', linestyle='-', linewidth=0.5)
    ax.set_xlabel('Eigenvalue index (sorted ascending)')
    ax.set_ylabel('λ')
    ax.set_title(f'{name} Eigenvalue Spectra')
    ax.legend(bbox_to_anchor=(1.02, 1), loc='upper left', fontsize=8)
    ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig(images_dir / 'eigenvalue_spectra.png', dpi=150, bbox_inches='tight')
plt.show()

print(f"\nM matrix visualizations saved to {images_dir}/")

# %%
# =============================================================================
# SUMMARY
# =============================================================================
print("\n" + "=" * 60)
print("SUMMARY")
print("=" * 60)

RANK = config['rank']
print(f"""
Sparse 3-Layer Memory Model
============================

Config: n_task={N_TASK}, n_model={N_MODEL}, n_memory={N_MEMORY}, rank={RANK}
Accuracy: {accuracy:.1%} (baseline: {checkpoint['baseline_acc']*100:.1f}%)
Sparsity: {checkpoint['sparsity']*100:.1f}%

Per-layer sparsity:
  L1: {sp_L1:.1f}%   D1: {sp_D1:.1f}%
  L2: {sp_L2:.1f}%   D2: {sp_D2:.1f}%
  L3: {sp_L3:.1f}%   D3: {sp_D3:.1f}%

Layer contributions:
  ||r1_task||: {r1_task.norm(dim=1).mean().item():.2f}
  ||r2_task||: {r2_task.norm(dim=1).mean().item():.2f}
  ||r3_task||: {r3_task.norm(dim=1).mean().item():.2f}

Progressive accuracy:
  x only:           {(x_task.argmax(dim=1) == targets).float().mean().item():.1%}
  x + r1:           {((x_task + r1_task).argmax(dim=1) == targets).float().mean().item():.1%}
  x + r1 + r2:      {((x_task + r1_task + r2_task).argmax(dim=1) == targets).float().mean().item():.1%}
  x + r1 + r2 + r3: {accuracy:.1%}
""")

# %%
# =============================================================================
# R3 DECOMPOSITION SUMMARY (printed at the end)
# =============================================================================
print("\n" + "=" * 60)
print("R3 DECOMPOSITION SUMMARY: A (h_prev²), B (h_prev×r2), C (r2²)")
print("=" * 60)

print(f"""
Layer 3 output r3 = D3 @ (L3 @ h3)² where h3 = norm(h_prev + r2), h_prev = x + r1

Decomposition:
  r3 = A + B + C where:
    A = D3 @ (L3 @ h_prev_norm)²           -- previous_state × previous_state
    B = 2 * D3 @ (L3@h_prev_norm * L3@r2_norm)  -- previous_state × r2 cross
    C = D3 @ (L3 @ r2_norm)²               -- r2 × r2

Component magnitudes (||·|| over task outputs):
  ||A|| (h_prev²):   {r3_A[:, :N_TASK].norm(dim=1).mean().item():.2f}
  ||B|| (h_prev×r2): {r3_B[:, :N_TASK].norm(dim=1).mean().item():.2f}
  ||C|| (r2²):       {r3_C[:, :N_TASK].norm(dim=1).mean().item():.2f}
  ||r3|| (total):    {r3[:, :N_TASK].norm(dim=1).mean().item():.2f}

Normalized input contributions:
  ||h_prev_norm|| / ||h3||: {h_prev_norm.norm(dim=1).mean().item() / h3_norm:.2%}
  ||r2_norm|| / ||h3||:     {r2_norm.norm(dim=1).mean().item() / h3_norm:.2%}

Ablation accuracy:
  A only (h_prev²):       {acc_A:.1%}
  B only (h_prev×r2):     {acc_B:.1%}
  C only (r2²):           {acc_C:.1%}
  A + B (no r2²):         {acc_AB:.1%}
  A + C (no cross):       {acc_AC:.1%}
  B + C (no h_prev²):     {acc_BC:.1%}
  Full A+B+C:             {accuracy:.1%}
  No r3 (baseline):       {acc_no_r3:.1%}
""")

# %%
