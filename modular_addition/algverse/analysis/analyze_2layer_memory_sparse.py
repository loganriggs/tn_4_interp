# %%
"""
Analysis of Sparse 2-Layer Memory Model

Analyze the structure of the pruned 2-layer model with extra "memory" dimension.
Key question: Does position 10 (memory) take over the role position 5 had?
"""

import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import sys
import pickle

# =============================================================================
# PROJECT ROOT
# =============================================================================
PROJECT_ROOT = Path("/workspace/tn_4_interp/modular_addition/algverse")
sys.path.insert(0, str(PROJECT_ROOT))
from models import SymmetricBilinearResidual, task_2nd_argmax
from analysis.analysis_utils import compute_quadratic_forms

checkpoint_dir = PROJECT_ROOT / "checkpoints"
images_dir = PROJECT_ROOT / "images" / "sparse_2layer_memory"
images_dir.mkdir(exist_ok=True, parents=True)

N_TASK = 10
N_MODEL = 11
device = "cuda" if torch.cuda.is_available() else "cpu"

# %%
# =============================================================================
# LOAD SPARSE MODEL
# =============================================================================
checkpoint_path = checkpoint_dir / "2layer_n10_memory_seed3_sparse.pkl"
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

print(f"\nWeight shapes:")
print(f"  L1: {L1.shape}, D1: {D1.shape}, gamma1: {gamma1.item():.4f}")
print(f"  L2: {L2.shape}, D2: {D2.shape}, gamma2: {gamma2.item():.4f}")

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

# %%
# =============================================================================
# MEMORY DIMENSION USAGE
# =============================================================================
print("\n" + "=" * 60)
print("MEMORY DIMENSION (pos 10) USAGE")
print("=" * 60)

# L1 column 10: how much does memory input contribute to each rank
L1_col10 = L1[:, 10]
print(f"\nL1 column 10 (memory input -> ranks):")
print(f"  Non-zero: {(L1_col10 != 0).sum().item()}/{len(L1_col10)}")
print(f"  Values: {L1_col10.numpy()}")

# D1 row 10: how much does each rank write to memory
D1_row10 = D1[10, :]
print(f"\nD1 row 10 (ranks -> memory output):")
print(f"  Non-zero: {(D1_row10 != 0).sum().item()}/{len(D1_row10)}")
print(f"  Values: {D1_row10.numpy()}")

# L2 column 10: how much does layer1 memory output contribute
L2_col10 = L2[:, 10]
print(f"\nL2 column 10 (L1 memory output -> L2 ranks):")
print(f"  Non-zero: {(L2_col10 != 0).sum().item()}/{len(L2_col10)}")
print(f"  Values: {L2_col10.numpy()}")

# D2 row 10: layer 2 output to memory (not used in loss)
D2_row10 = D2[10, :]
print(f"\nD2 row 10 (ranks -> final memory, unused):")
print(f"  Non-zero: {(D2_row10 != 0).sum().item()}/{len(D2_row10)}")

# %%
# =============================================================================
# QUADRATIC FORM ANALYSIS
# =============================================================================
print("\n" + "=" * 60)
print("QUADRATIC FORM ANALYSIS")
print("=" * 60)

M1 = compute_quadratic_forms(L1, D1)  # (11, 11, 11)
M2 = compute_quadratic_forms(L2, D2)

print(f"\nM1 (layer 1 quadratic forms): {M1.shape}")
print(f"M2 (layer 2 quadratic forms): {M2.shape}")

# Analyze M1 structure for task outputs (0-9)
print("\nM1 structure (first 10 outputs):")
for i in range(N_TASK):
    M1_i = M1[i].numpy()
    diag = np.diag(M1_i)
    task_diag = diag[:N_TASK]
    mem_diag = diag[N_TASK]  # diagonal at position 10
    other_task_diag = np.delete(task_diag, i)

    U, S, Vh = np.linalg.svd(M1_i)
    rank = np.sum(S > 1e-6)

    print(f"  M1[{i}]: self={M1_i[i,i]:+.3f}, other_mean={other_task_diag.mean():.3f}, mem_diag={mem_diag:.3f}, rank={rank}")

# Analyze M1[10] - the memory output
print(f"\nM1[10] (memory output):")
M1_mem = M1[10].numpy()
diag_mem = np.diag(M1_mem)
print(f"  Diag (task positions 0-9): {diag_mem[:N_TASK]}")
print(f"  Diag[10] (memory self): {diag_mem[10]:.4f}")

# Analyze M2 structure
print("\nM2 structure (first 10 outputs):")
for i in range(N_TASK):
    M2_i = M2[i].numpy()
    U, S, Vh = np.linalg.svd(M2_i)
    rank = np.sum(S > 1e-6)

    # Check memory involvement
    mem_row = M2_i[10, :]
    mem_col = M2_i[:, 10]
    mem_involvement = np.abs(mem_row).sum() + np.abs(mem_col).sum() - 2*abs(M2_i[10,10])

    print(f"  M2[{i}]: rank={rank}, mem_involvement={mem_involvement:.3f}, top_sv={S[0]:.3f}")

# %%
# =============================================================================
# FORWARD PASS AND ABLATION
# =============================================================================
print("\n" + "=" * 60)
print("FORWARD PASS AND ABLATION")
print("=" * 60)

# Load model
model = SymmetricBilinearResidual(N_MODEL, 2, 11).to(device)
model.load_state_dict({k: v.to(device) for k, v in state.items()})
model.eval()

def my_rmsnorm(x, gamma):
    rms = (x ** 2).mean(dim=-1, keepdim=True).sqrt()
    return gamma * x / rms, rms.squeeze(-1)

def forward_2layer_memory(x_task):
    """Forward pass with memory dimension."""
    # Pad with memory (zeros)
    x = torch.cat([x_task, torch.zeros(x_task.shape[0], 1, device=x_task.device)], dim=1)

    h1, rms1 = my_rmsnorm(x, gamma1.to(x.device))
    Lh1 = h1 @ L1.T.to(x.device)
    r1 = (Lh1 ** 2) @ D1.T.to(x.device)

    h_mid = x + r1
    h2, rms2 = my_rmsnorm(h_mid, gamma2.to(x.device))
    Lh2 = h2 @ L2.T.to(x.device)
    r2 = (Lh2 ** 2) @ D2.T.to(x.device)

    output = x + r1 + r2
    return output, r1, r2

# Evaluate
torch.manual_seed(123)
x_task = torch.randn(10000, N_TASK, device=device)
targets = task_2nd_argmax(x_task)

output, r1, r2 = forward_2layer_memory(x_task)

# Only use first 10 outputs for prediction
preds = output[:, :N_TASK].argmax(dim=1)
accuracy = (preds == targets).float().mean().item()

print(f"\nModel accuracy: {accuracy:.1%}")

# Analyze layer contributions (task dimensions only)
x_padded = torch.cat([x_task, torch.zeros(10000, 1, device=device)], dim=1)
r1_task = r1[:, :N_TASK]
r2_task = r2[:, :N_TASK]
r1_mem = r1[:, N_TASK]
r2_mem = r2[:, N_TASK]

print(f"\nNorm contributions (mean):")
print(f"  ||x_task||: {x_task.norm(dim=1).mean().item():.4f}")
print(f"  ||r1_task||: {r1_task.norm(dim=1).mean().item():.4f}")
print(f"  ||r2_task||: {r2_task.norm(dim=1).mean().item():.4f}")
print(f"  |r1_mem|: {r1_mem.abs().mean().item():.4f}")
print(f"  |r2_mem|: {r2_mem.abs().mean().item():.4f}")

print(f"\nProgressive accuracy:")
print(f"  x only:      {(x_task.argmax(dim=1) == targets).float().mean().item():.1%}")
print(f"  x + r1_task: {((x_task + r1_task).argmax(dim=1) == targets).float().mean().item():.1%}")
print(f"  x + r1 + r2: {accuracy:.1%}")

# Ablation
print(f"\nAblation (removing components):")
print(f"  Without r1 (x + r2_task):      {((x_task + r2_task).argmax(dim=1) == targets).float().mean().item():.1%}")
print(f"  Without r2 (x + r1_task):      {((x_task + r1_task).argmax(dim=1) == targets).float().mean().item():.1%}")

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

# Overall
print(f"\nOverall: {accuracy:.1%}")

# Compare to non-memory model position 5
print(f"\nComparison to baseline 2-layer n=10:")
print(f"  Baseline position 5: 27.4% (pathological)")
print(f"  Memory model position 5: {confusion[5, 5].item() / confusion[5].sum().item():.1%}")

# %%
# =============================================================================
# POSITION-SPECIFIC ABLATION
# =============================================================================
print("\n" + "=" * 60)
print("POSITION-SPECIFIC ABLATION")
print("=" * 60)

print("\nRemoving r2[i] for each position:")
for pos in range(N_TASK):
    r2_ablated = r2.clone()
    r2_ablated[:, pos] = 0
    output_ablated = x_padded + r1 + r2_ablated
    preds_abl = output_ablated[:, :N_TASK].argmax(dim=1)
    acc_abl = (preds_abl == targets).float().mean().item()
    print(f"  Remove r2[{pos}]: {acc_abl:.1%} (Δ={acc_abl - accuracy:+.1%})")

# Memory ablation
print("\nRemoving memory dimension entirely:")
r1_no_mem = r1.clone()
r1_no_mem[:, 10] = 0
r2_no_mem = r2.clone()
r2_no_mem[:, 10] = 0
output_no_mem = x_padded + r1_no_mem + r2_no_mem
preds_no_mem = output_no_mem[:, :N_TASK].argmax(dim=1)
acc_no_mem = (preds_no_mem == targets).float().mean().item()
print(f"  Without memory (r1[10]=0, r2[10]=0): {acc_no_mem:.1%} (Δ={acc_no_mem - accuracy:+.1%})")

# %%
# =============================================================================
# R2 DECOMPOSITION INTO A, B, C COMPONENTS (x vs r1 decomposition)
# =============================================================================
print("\n" + "=" * 60)
print("R2 DECOMPOSITION: A (x×x), B (x×r1 cross), C (r1×r1)")
print("=" * 60)

# Layer 2 input: h_mid = x + r1
# After norm: h2 = γ2 * (x + r1) / rms2
#
# Decompose the normalized contributions:
#   x_norm = γ2 * x / rms2
#   r1_norm = γ2 * r1 / rms2
# So h2 = x_norm + r1_norm
#
# Layer 2 bilinear: r2 = D2 @ (L2 @ h2)²
#                      = D2 @ (L2@x_norm + L2@r1_norm)²
#                      = D2 @ [(L2@x_norm)² + 2*(L2@x_norm)*(L2@r1_norm) + (L2@r1_norm)²]
#
# A = D2 @ (L2@x_norm)²           -- input × input through layer 2
# B = 2 * D2 @ (L2@x_norm * L2@r1_norm)  -- input × layer1_output cross-term
# C = D2 @ (L2@r1_norm)²          -- layer1_output × layer1_output (D × D)

def compute_r2_ABC_xr1(x_task, L1, D1, L2, D2, gamma1, gamma2):
    """
    Compute r2 decomposed into A, B, C based on x vs r1.

    A = x × x contribution through layer 2
    B = x × r1 cross-term through layer 2
    C = r1 × r1 contribution through layer 2

    Returns:
        r2_A, r2_B, r2_C: (batch, n_model) tensors
        r1: layer 1 output
        x_norm, r1_norm: normalized components at layer 2 input
    """
    device = x_task.device
    batch_size = x_task.shape[0]

    # Pad input with memory dimension (zeros)
    x = torch.cat([x_task, torch.zeros(batch_size, 1, device=device)], dim=1)

    # Layer 1 forward
    rms1 = (x ** 2).mean(dim=-1, keepdim=True).sqrt()
    h1 = gamma1.to(device) * x / rms1
    Lh1 = h1 @ L1.T.to(device)
    r1 = (Lh1 ** 2) @ D1.T.to(device)

    # Intermediate state before layer 2
    h_mid = x + r1

    # Layer 2 normalization - but decompose into x and r1 contributions
    rms2 = (h_mid ** 2).mean(dim=-1, keepdim=True).sqrt()

    # Normalized x and r1 contributions
    x_norm = gamma2.to(device) * x / rms2
    r1_norm = gamma2.to(device) * r1 / rms2

    # L2 projections of each component
    L2_dev = L2.to(device)
    D2_dev = D2.to(device)

    Lx = x_norm @ L2_dev.T   # (batch, rank)
    Lr1 = r1_norm @ L2_dev.T  # (batch, rank)

    # A: x × x = D2 @ (L2 @ x_norm)²
    r2_A = (Lx ** 2) @ D2_dev.T

    # B: x × r1 cross = 2 * D2 @ (L2@x_norm * L2@r1_norm)
    r2_B = (2 * Lx * Lr1) @ D2_dev.T

    # C: r1 × r1 = D2 @ (L2 @ r1_norm)²
    r2_C = (Lr1 ** 2) @ D2_dev.T

    return r2_A, r2_B, r2_C, r1, x_norm, r1_norm

# Compute decomposition
r2_A, r2_B, r2_C, r1_recomp, x_norm, r1_norm = compute_r2_ABC_xr1(
    x_task, L1, D1, L2, D2, gamma1, gamma2
)

# Verify: A + B + C should equal r2
r2_reconstructed = r2_A + r2_B + r2_C
reconstruction_error = (r2 - r2_reconstructed).abs().max().item()
print(f"\nReconstruction check: max|r2 - (A+B+C)| = {reconstruction_error:.2e}")

# Component statistics
print(f"\nComponent magnitudes (mean over batch, task outputs only):")
print(f"  ||A_task|| (x×x):     {r2_A[:, :N_TASK].norm(dim=1).mean().item():.4f}")
print(f"  ||B_task|| (x×r1):    {r2_B[:, :N_TASK].norm(dim=1).mean().item():.4f}")
print(f"  ||C_task|| (r1×r1):   {r2_C[:, :N_TASK].norm(dim=1).mean().item():.4f}")
print(f"  ||r2_task|| (total):  {r2[:, :N_TASK].norm(dim=1).mean().item():.4f}")

print(f"\nPer-output mean magnitudes (task outputs):")
print(f"  A (x×x):   {r2_A[:, :N_TASK].abs().mean(dim=0).cpu().numpy()}")
print(f"  B (x×r1):  {r2_B[:, :N_TASK].abs().mean(dim=0).cpu().numpy()}")
print(f"  C (r1×r1): {r2_C[:, :N_TASK].abs().mean(dim=0).cpu().numpy()}")

# Normalized contribution statistics
print(f"\nNormalized layer 2 input contributions:")
print(f"  ||x_norm||:  {x_norm.norm(dim=1).mean().item():.4f}")
print(f"  ||r1_norm||: {r1_norm.norm(dim=1).mean().item():.4f}")
print(f"  ||x_norm|| / ||h2||:  {x_norm.norm(dim=1).mean().item() / (x_norm + r1_norm).norm(dim=1).mean().item():.4f}")
print(f"  ||r1_norm|| / ||h2||: {r1_norm.norm(dim=1).mean().item() / (x_norm + r1_norm).norm(dim=1).mean().item():.4f}")

# Ablation: accuracy with only each component
print(f"\nAblation - accuracy with different r2 components:")

# Only A (x×x)
output_A = x_padded + r1 + r2_A
preds_A = output_A[:, :N_TASK].argmax(dim=1)
acc_A = (preds_A == targets).float().mean().item()
print(f"  r2 = A only (x×x):       {acc_A:.1%}")

# Only B (x×r1)
output_B = x_padded + r1 + r2_B
preds_B = output_B[:, :N_TASK].argmax(dim=1)
acc_B = (preds_B == targets).float().mean().item()
print(f"  r2 = B only (x×r1):      {acc_B:.1%}")

# Only C (r1×r1)
output_C = x_padded + r1 + r2_C
preds_C = output_C[:, :N_TASK].argmax(dim=1)
acc_C = (preds_C == targets).float().mean().item()
print(f"  r2 = C only (r1×r1):     {acc_C:.1%}")

# A + B (x×x + cross, no r1×r1)
output_AB = x_padded + r1 + r2_A + r2_B
preds_AB = output_AB[:, :N_TASK].argmax(dim=1)
acc_AB = (preds_AB == targets).float().mean().item()
print(f"  r2 = A + B (no r1×r1):   {acc_AB:.1%}")

# A + C (x×x + r1×r1, no cross)
output_AC = x_padded + r1 + r2_A + r2_C
preds_AC = output_AC[:, :N_TASK].argmax(dim=1)
acc_AC = (preds_AC == targets).float().mean().item()
print(f"  r2 = A + C (no cross):   {acc_AC:.1%}")

# B + C (cross + r1×r1, no x×x)
output_BC = x_padded + r1 + r2_B + r2_C
preds_BC = output_BC[:, :N_TASK].argmax(dim=1)
acc_BC = (preds_BC == targets).float().mean().item()
print(f"  r2 = B + C (no x×x):     {acc_BC:.1%}")

# Full r2 for comparison
print(f"  r2 = A + B + C (full):       {accuracy:.1%}")

# Without r2 for baseline
output_no_r2 = x_padded + r1
preds_no_r2 = output_no_r2[:, :N_TASK].argmax(dim=1)
acc_no_r2 = (preds_no_r2 == targets).float().mean().item()
print(f"  r2 = 0 (no layer 2):         {acc_no_r2:.1%}")

# %%
# =============================================================================
# VISUALIZATIONS
# =============================================================================
print("\n" + "=" * 60)
print("GENERATING VISUALIZATIONS")
print("=" * 60)

# Weight matrices
from analysis.analysis_utils import plot_weights, plot_M_eigen

fig = plot_weights(
    [(L1, D1, 'Layer1', sp_L1, sp_D1),
     (L2, D2, 'Layer2', sp_L2, sp_D2)],
    title=f'Sparse 2-Layer Memory Model (Acc: {checkpoint["accuracy"]*100:.1f}%, Sparsity: {checkpoint["sparsity"]*100:.1f}%)\nGreen lines mark memory dimension (pos 10)',
    memory_boundary=N_TASK,
    save_path=images_dir / 'weights_all.png',
)
plt.show()

# Per-position accuracy bar chart
fig, ax = plt.subplots(figsize=(10, 6))
pos_accs = [confusion[t, t].item() / confusion[t].sum().item() for t in range(N_TASK)]
colors = ['green' if acc > 0.6 else 'orange' if acc > 0.4 else 'red' for acc in pos_accs]
bars = ax.bar(range(N_TASK), pos_accs, color=colors)
ax.axhline(y=accuracy, color='blue', linestyle='--', label=f'Overall: {accuracy:.1%}')
ax.axhline(y=0.274, color='red', linestyle=':', label='Baseline pos 5: 27.4%')
ax.set_xlabel('Position')
ax.set_ylabel('Accuracy')
ax.set_title('Per-Position Accuracy (Memory Model)')
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
ax.set_title('Confusion Matrix (Memory Model)')
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
# M1 AND M2 MATRIX VISUALIZATIONS WITH EIGENDECOMPOSITION
# =============================================================================
print("\n" + "=" * 60)
print("M1 AND M2 MATRIX VISUALIZATIONS WITH EIGENDECOMPOSITION")
print("=" * 60)

# Plot M1 and M2 with eigendecomposition using shared function
for M, name in [(M1, 'M1'), (M2, 'M2')]:
    print(f"\nGenerating {name} visualizations (outputs 0-5)...")
    plot_M_eigen(M, name, list(range(6)), memory_boundary=N_TASK,
                 save_path=images_dir / f'{name}_eigen_analysis.png')
    plt.show()

    print(f"Generating {name} visualizations (outputs 6-10)...")
    plot_M_eigen(M, name, list(range(6, N_MODEL)), memory_boundary=N_TASK,
                 save_path=images_dir / f'{name}_eigen_analysis_6to10.png')
    plt.show()

# Summary plot: All eigenvalue spectra (line plots, linear scale)
print("\nGenerating eigenvalue summary...")
fig, axes = plt.subplots(2, 1, figsize=(14, 10))

# M1 eigenvalues
ax = axes[0]
for i in range(N_MODEL):
    M_i = M1[i].numpy()
    eigenvalues, _ = np.linalg.eigh(M_i)
    label = f'out {i}' if i < 10 else 'mem'
    ax.plot(range(len(eigenvalues)), eigenvalues, 'o-', label=label, alpha=0.7, markersize=4)
ax.axhline(y=0, color='black', linestyle='-', linewidth=0.5)
ax.set_xlabel('Eigenvalue index (sorted ascending)')
ax.set_ylabel('λ')
ax.set_title('M1 Eigenvalue Spectra (all outputs)')
ax.legend(bbox_to_anchor=(1.02, 1), loc='upper left', fontsize=8)
ax.grid(True, alpha=0.3)

# M2 eigenvalues
ax = axes[1]
for i in range(N_MODEL):
    M_i = M2[i].numpy()
    eigenvalues, _ = np.linalg.eigh(M_i)
    label = f'out {i}' if i < 10 else 'mem'
    ax.plot(range(len(eigenvalues)), eigenvalues, 'o-', label=label, alpha=0.7, markersize=4)
ax.axhline(y=0, color='black', linestyle='-', linewidth=0.5)
ax.set_xlabel('Eigenvalue index (sorted ascending)')
ax.set_ylabel('λ')
ax.set_title('M2 Eigenvalue Spectra (all outputs)')
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

pos5_acc = confusion[5, 5].item() / confusion[5].sum().item()

RANK = config['rank']
print(f"""
Sparse 2-Layer Memory Model
============================

Config: n_task={N_TASK}, n_model={N_MODEL}, rank={RANK}
Accuracy: {accuracy:.1%} (baseline: {checkpoint['baseline_acc']*100:.1f}%)
Sparsity: {checkpoint['sparsity']*100:.1f}%

Per-layer sparsity:
  L1: {sp_L1:.1f}%   D1: {sp_D1:.1f}%
  L2: {sp_L2:.1f}%   D2: {sp_D2:.1f}%

Memory dimension usage:
  r1[10] mean: {r1_mem.abs().mean().item():.4f}
  r2[10] mean: {r2_mem.abs().mean().item():.4f}
  L2[:,10] non-zero: {(L2_col10 != 0).sum().item()}/{len(L2_col10)}

Position 5 fix:
  Baseline (no memory): 27.4%
  Memory model: {pos5_acc:.1%}
  Improvement: {pos5_acc - 0.274:.1%}

Key findings:
  - Memory dimension allows extra computation space
  - Position 5 accuracy improved from 27.4% to {pos5_acc:.1%}
  - Overall accuracy improved from 61.3% to {accuracy:.1%}
  - Model prunes to similar sparsity ({checkpoint['sparsity']*100:.1f}% vs 40.8%)
""")

# %%
# =============================================================================
# A/B/C DECOMPOSITION SUMMARY (printed at the end)
# =============================================================================
print("\n" + "=" * 60)
print("R2 DECOMPOSITION SUMMARY: A (x×x), B (x×r1), C (r1×r1)")
print("=" * 60)

print(f"""
Layer 2 output r2 = D2 @ (L2 @ h2)² where h2 = norm(x + r1)

Decomposition:
  r2 = A + B + C where:
    A = D2 @ (L2 @ x_norm)²           -- input × input
    B = 2 * D2 @ (L2@x_norm * L2@r1_norm)  -- input × layer1_output cross
    C = D2 @ (L2 @ r1_norm)²          -- layer1_output × layer1_output

Component magnitudes (||·|| over task outputs):
  ||A|| (x×x):     {r2_A[:, :N_TASK].norm(dim=1).mean().item():.2f}
  ||B|| (x×r1):    {r2_B[:, :N_TASK].norm(dim=1).mean().item():.2f}
  ||C|| (r1×r1):   {r2_C[:, :N_TASK].norm(dim=1).mean().item():.2f}
  ||r2|| (total):  {r2[:, :N_TASK].norm(dim=1).mean().item():.2f}

Normalized input contributions:
  ||x_norm|| / ||h2||:  {x_norm.norm(dim=1).mean().item() / (x_norm + r1_norm).norm(dim=1).mean().item():.2%}
  ||r1_norm|| / ||h2||: {r1_norm.norm(dim=1).mean().item() / (x_norm + r1_norm).norm(dim=1).mean().item():.2%}

Ablation accuracy:
  A only (x×x):       {acc_A:.1%}
  B only (x×r1):      {acc_B:.1%}
  C only (r1×r1):     {acc_C:.1%}
  A + B (no r1×r1):   {acc_AB:.1%}
  A + C (no cross):   {acc_AC:.1%}
  B + C (no x×x):     {acc_BC:.1%}
  Full A+B+C:         {accuracy:.1%}
  No r2 (baseline):   {acc_no_r2:.1%}
""")

# %%
