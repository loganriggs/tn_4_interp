# %%
"""
2-Layer n=4 + 1 Memory Analysis

Loads the sparse 2-layer n=4 model with 1 memory dimension.
Visualizes weights, M matrices, and A/B/C decomposition.
"""

import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import sys
import pickle

PROJECT_ROOT = Path("/workspace/tn_4_interp/modular_addition/algverse")
sys.path.insert(0, str(PROJECT_ROOT))
from models import SymmetricBilinearResidual, task_2nd_argmax
from analysis.analysis_utils import (
    compute_quadratic_forms, plot_mat, plot_weights, plot_M_eigen, plot_M_only
)

checkpoint_dir = PROJECT_ROOT / "checkpoints"
images_dir = PROJECT_ROOT / "images" / "2layer_n4_memory"
images_dir.mkdir(exist_ok=True, parents=True)

N_TASK = 4
N_MODEL = 5
device = "cuda" if torch.cuda.is_available() else "cpu"

# %%
# =============================================================================
# LOAD SPARSE MODEL
# =============================================================================
checkpoint_path = checkpoint_dir / "2layer_n4_memory1_sparse.pkl"
print(f"Loading sparse model from {checkpoint_path}")

with open(checkpoint_path, 'rb') as f:
    checkpoint = pickle.load(f)

state = checkpoint['state_dict']
config = checkpoint['config']

print(f"\nConfig: n_task={config['n_task']}, n_model={config['n_model']}, rank={config['rank']}")
print(f"Accuracy: {checkpoint['accuracy']*100:.1f}%")
print(f"Sparsity: {checkpoint.get('sparsity', 0)*100:.1f}%")
print(f"Baseline: {checkpoint.get('baseline_acc', checkpoint['accuracy'])*100:.1f}%")

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
print("MEMORY DIMENSION (pos 4) USAGE")
print("=" * 60)

L1_col4 = L1[:, 4]
print(f"\nL1 column 4 (memory input -> ranks):")
print(f"  Non-zero: {(L1_col4 != 0).sum().item()}/{len(L1_col4)}")
print(f"  Values: {L1_col4.numpy()}")

D1_row4 = D1[4, :]
print(f"\nD1 row 4 (ranks -> memory output):")
print(f"  Non-zero: {(D1_row4 != 0).sum().item()}/{len(D1_row4)}")
print(f"  Values: {D1_row4.numpy()}")

L2_col4 = L2[:, 4]
print(f"\nL2 column 4 (L1 memory output -> L2 ranks):")
print(f"  Non-zero: {(L2_col4 != 0).sum().item()}/{len(L2_col4)}")
print(f"  Values: {L2_col4.numpy()}")

D2_row4 = D2[4, :]
print(f"\nD2 row 4 (ranks -> final memory, unused):")
print(f"  Non-zero: {(D2_row4 != 0).sum().item()}/{len(D2_row4)}")

# %%
# =============================================================================
# QUADRATIC FORM ANALYSIS
# =============================================================================
print("\n" + "=" * 60)
print("QUADRATIC FORM ANALYSIS")
print("=" * 60)

M1 = compute_quadratic_forms(L1, D1)
M2 = compute_quadratic_forms(L2, D2)

print(f"\nM1 (layer 1 quadratic forms): {M1.shape}")
print(f"M2 (layer 2 quadratic forms): {M2.shape}")

for M, name in [(M1, 'M1'), (M2, 'M2')]:
    print(f"\n{name} structure:")
    for i in range(N_MODEL):
        M_i = M[i].numpy()
        eigenvalues = np.linalg.eigvalsh(M_i)
        rank = np.sum(np.abs(eigenvalues) > 1e-6)
        label = f'mem' if i >= N_TASK else str(i)
        print(f"  {name}[{label}]: self={M_i[i,i]:+.3f}, rank={rank}, top_eig={eigenvalues[-1]:.3f}")

# %%
# =============================================================================
# FORWARD PASS AND ABLATION
# =============================================================================
print("\n" + "=" * 60)
print("FORWARD PASS AND ABLATION")
print("=" * 60)

model = SymmetricBilinearResidual(N_MODEL, 2, config['rank']).to(device)
model.load_state_dict({k: v.to(device) for k, v in state.items()})
model.eval()

def my_rmsnorm(x, gamma):
    rms = (x ** 2).mean(dim=-1, keepdim=True).sqrt()
    return gamma * x / rms, rms.squeeze(-1)

def forward_2layer_memory(x_task):
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

torch.manual_seed(123)
x_task = torch.randn(10000, N_TASK, device=device)
targets = task_2nd_argmax(x_task)

output, r1, r2 = forward_2layer_memory(x_task)
preds = output[:, :N_TASK].argmax(dim=1)
accuracy = (preds == targets).float().mean().item()

x_padded = torch.cat([x_task, torch.zeros(10000, 1, device=device)], dim=1)
r1_task = r1[:, :N_TASK]
r2_task = r2[:, :N_TASK]

print(f"\nModel accuracy: {accuracy:.1%}")

print(f"\nNorm contributions (mean):")
print(f"  ||x_task||: {x_task.norm(dim=1).mean().item():.4f}")
print(f"  ||r1_task||: {r1_task.norm(dim=1).mean().item():.4f}")
print(f"  ||r2_task||: {r2_task.norm(dim=1).mean().item():.4f}")
print(f"  |r1_mem|: {r1[:, N_TASK].abs().mean().item():.4f}")
print(f"  |r2_mem|: {r2[:, N_TASK].abs().mean().item():.4f}")

print(f"\nProgressive accuracy:")
print(f"  x only:      {(x_task.argmax(dim=1) == targets).float().mean().item():.1%}")
print(f"  x + r1_task: {((x_task + r1_task).argmax(dim=1) == targets).float().mean().item():.1%}")
print(f"  x + r1 + r2: {accuracy:.1%}")

print(f"\nAblation (removing components):")
print(f"  Without r1 (x + r2_task):  {((x_task + r2_task).argmax(dim=1) == targets).float().mean().item():.1%}")
print(f"  Without r2 (x + r1_task):  {((x_task + r1_task).argmax(dim=1) == targets).float().mean().item():.1%}")

# %%
# =============================================================================
# PER-POSITION ACCURACY
# =============================================================================
print("\n" + "=" * 60)
print("PER-POSITION ACCURACY")
print("=" * 60)

confusion = torch.zeros(N_TASK, N_TASK)
for t in range(N_TASK):
    mask = (targets == t)
    if mask.sum() > 0:
        preds_t = preds[mask]
        for p in range(N_TASK):
            confusion[t, p] = (preds_t == p).sum().item()

print("\nPer-position accuracy:")
for t in range(N_TASK):
    total = confusion[t].sum().item()
    correct = confusion[t, t].item()
    acc = correct / total if total > 0 else 0
    print(f"  Position {t}: {acc:.1%} ({int(correct)}/{int(total)})")
print(f"\nOverall: {accuracy:.1%}")

# %%
# =============================================================================
# R2 DECOMPOSITION INTO A, B, C
# =============================================================================
print("\n" + "=" * 60)
print("R2 DECOMPOSITION: A (x*x), B (x*r1 cross), C (r1*r1)")
print("=" * 60)

def compute_r2_ABC(x_task):
    device = x_task.device
    x = torch.cat([x_task, torch.zeros(x_task.shape[0], 1, device=device)], dim=1)
    rms1 = (x ** 2).mean(dim=-1, keepdim=True).sqrt()
    h1 = gamma1.to(device) * x / rms1
    Lh1 = h1 @ L1.T.to(device)
    r1 = (Lh1 ** 2) @ D1.T.to(device)
    h_mid = x + r1
    rms2 = (h_mid ** 2).mean(dim=-1, keepdim=True).sqrt()
    x_norm = gamma2.to(device) * x / rms2
    r1_norm = gamma2.to(device) * r1 / rms2
    L2_dev, D2_dev = L2.to(device), D2.to(device)
    Lx = x_norm @ L2_dev.T
    Lr1 = r1_norm @ L2_dev.T
    r2_A = (Lx ** 2) @ D2_dev.T
    r2_B = (2 * Lx * Lr1) @ D2_dev.T
    r2_C = (Lr1 ** 2) @ D2_dev.T
    return r2_A, r2_B, r2_C, r1

r2_A, r2_B, r2_C, _ = compute_r2_ABC(x_task)

reconstruction_error = (r2 - (r2_A + r2_B + r2_C)).abs().max().item()
print(f"\nReconstruction check: max|r2 - (A+B+C)| = {reconstruction_error:.2e}")

print(f"\nComponent magnitudes (task outputs):")
print(f"  ||A|| (x*x):     {r2_A[:, :N_TASK].norm(dim=1).mean().item():.4f}")
print(f"  ||B|| (x*r1):    {r2_B[:, :N_TASK].norm(dim=1).mean().item():.4f}")
print(f"  ||C|| (r1*r1):   {r2_C[:, :N_TASK].norm(dim=1).mean().item():.4f}")
print(f"  ||r2|| (total):  {r2[:, :N_TASK].norm(dim=1).mean().item():.4f}")

print(f"\nAblation - accuracy with different r2 components:")
for name, comp in [('A only', r2_A), ('B only', r2_B), ('C only', r2_C),
                    ('A+B', r2_A+r2_B), ('A+C', r2_A+r2_C), ('B+C', r2_B+r2_C),
                    ('A+B+C', r2_A+r2_B+r2_C)]:
    out = x_padded + r1 + comp
    acc = (out[:, :N_TASK].argmax(dim=1) == targets).float().mean().item()
    print(f"  r2 = {name:8s}: {acc:.1%}")

acc_no_r2 = ((x_task + r1_task).argmax(dim=1) == targets).float().mean().item()
print(f"  r2 = 0 (none):  {acc_no_r2:.1%}")

# %%
# =============================================================================
# VISUALIZATIONS
# =============================================================================
print("\n" + "=" * 60)
print("GENERATING VISUALIZATIONS")
print("=" * 60)

# Weight matrices
fig = plot_weights(
    [(L1, D1, 'Layer1', sp_L1, sp_D1),
     (L2, D2, 'Layer2', sp_L2, sp_D2)],
    title=f'2-Layer n=4+1mem (Acc: {checkpoint["accuracy"]*100:.1f}%, Sparsity: {checkpoint.get("sparsity",0)*100:.1f}%)\nGreen line marks memory dimension (pos 4)',
    memory_boundary=N_TASK,
    save_path=images_dir / 'weights_all.png',
)
plt.close()
print("Saved weights_all.png")

# M matrices with eigendecomposition
for M, name in [(M1, 'M1'), (M2, 'M2')]:
    fig = plot_M_eigen(M, name, list(range(N_MODEL)), n_top=3,
                       memory_boundary=N_TASK,
                       save_path=images_dir / f'{name}_eigen_analysis.png')
    plt.close()
    print(f"Saved {name}_eigen_analysis.png")

# M matrices only (side by side, task outputs)
for M, name in [(M1, 'M1'), (M2, 'M2')]:
    fig = plot_M_only(M, name, list(range(N_TASK)),
                      memory_boundary=N_TASK,
                      save_path=images_dir / f'{name}_matrices.png')
    plt.close()
    print(f"Saved {name}_matrices.png")

print(f"\nVisualizations saved to {images_dir}/")

# %%
# =============================================================================
# SUMMARY
# =============================================================================
print("\n" + "=" * 60)
print("SUMMARY")
print("=" * 60)

RANK = config['rank']
print(f"""
2-Layer n=4 + 1 Memory Model (Sparse)
=======================================

Config: n_task={N_TASK}, n_model={N_MODEL}, rank={RANK}
Accuracy: {accuracy:.1%} (baseline: {checkpoint.get('baseline_acc', checkpoint['accuracy'])*100:.1f}%)
Sparsity: {checkpoint.get('sparsity', 0)*100:.1f}%

Per-layer sparsity:
  L1: {sp_L1:.1f}%   D1: {sp_D1:.1f}%
  L2: {sp_L2:.1f}%   D2: {sp_D2:.1f}%

Memory dimension usage:
  |r1_mem| mean: {r1[:, N_TASK].abs().mean().item():.4f}
  |r2_mem| mean: {r2[:, N_TASK].abs().mean().item():.4f}

Progressive accuracy:
  x only:      {(x_task.argmax(dim=1) == targets).float().mean().item():.1%}
  x + r1:      {((x_task + r1_task).argmax(dim=1) == targets).float().mean().item():.1%}
  x + r1 + r2: {accuracy:.1%}

Comparison to 1-layer n=4: 70.0% -> {accuracy:.1%} (+{accuracy - 0.70:.1%})
""")

# %%
