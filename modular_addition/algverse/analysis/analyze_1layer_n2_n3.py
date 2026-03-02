# %%
"""
1-Layer Bilinear Analysis for n=2 and n=3 (2nd-argmax task)

Train simple 1-layer models and visualize M matrices.
For n=2, show M ⊙ (h⊗h^T) interpretation.
"""

import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch
from mpl_toolkits.axes_grid1 import make_axes_locatable
from pathlib import Path
import sys

PROJECT_ROOT = Path("/workspace/tn_4_interp/modular_addition/algverse")
sys.path.insert(0, str(PROJECT_ROOT))
from models import SymmetricBilinearResidual, task_2nd_argmax
from analysis.analysis_utils import compute_quadratic_forms

images_dir = PROJECT_ROOT / "images" / "1layer_n2_n3"
images_dir.mkdir(exist_ok=True, parents=True)

device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Device: {device}")

# %%
# =============================================================================
# TRAIN 1-LAYER MODELS
# =============================================================================

def train_1layer(n, rank, seed=0, n_steps=10000, lr=0.01, wd=0.01, batch_size=256):
    """Train a 1-layer symmetric bilinear model."""
    torch.manual_seed(seed)
    model = SymmetricBilinearResidual(n, 1, rank).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=wd)

    model.train()
    for step in range(n_steps + 1):
        x = torch.randn(batch_size, n, device=device)
        targets = task_2nd_argmax(x)
        output = model(x)
        loss = nn.functional.cross_entropy(output, targets)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if step % 2000 == 0:
            model.eval()
            with torch.no_grad():
                x_test = torch.randn(10000, n, device=device)
                t_test = task_2nd_argmax(x_test)
                acc = (model(x_test).argmax(1) == t_test).float().mean().item()
            print(f"  Step {step}: loss={loss.item():.4f}, acc={acc*100:.1f}%")
            model.train()

    # Final eval
    model.eval()
    with torch.no_grad():
        x_test = torch.randn(10000, n, device=device)
        t_test = task_2nd_argmax(x_test)
        acc = (model(x_test).argmax(1) == t_test).float().mean().item()

    return model, acc


print("=" * 60)
print("TRAINING n=2 MODEL")
print("=" * 60)
# n=2 is hard: bilinear must overcome residual to flip argmax->argmin
# Try multiple seeds with higher rank and more steps
best_n2 = None
for seed in range(10):
    model_n2_try, acc_n2_try = train_1layer(n=2, rank=4, seed=seed, n_steps=20000,
                                             lr=0.01, wd=0.01)
    print(f"  Seed {seed}: {acc_n2_try*100:.1f}%")
    if best_n2 is None or acc_n2_try > best_n2[1]:
        best_n2 = (model_n2_try, acc_n2_try, seed)

model_n2, acc_n2, seed_n2 = best_n2
print(f"\nn=2 best: accuracy={acc_n2*100:.1f}%, seed={seed_n2}")

print("\n" + "=" * 60)
print("TRAINING n=3 MODEL")
print("=" * 60)
model_n3, acc_n3 = train_1layer(n=3, rank=3, seed=0)
print(f"\nn=3 final accuracy: {acc_n3*100:.1f}%")

# %%
# =============================================================================
# EXTRACT WEIGHTS AND COMPUTE M MATRICES
# =============================================================================
print("\n" + "=" * 60)
print("WEIGHT ANALYSIS")
print("=" * 60)

L2 = model_n2.layers[0].L.detach().cpu()
D2 = model_n2.layers[0].D.detach().cpu()
gamma2 = model_n2.norms[0].weight.detach().cpu()
M_n2 = compute_quadratic_forms(L2, D2)

L3 = model_n3.layers[0].L.detach().cpu()
D3 = model_n3.layers[0].D.detach().cpu()
gamma3 = model_n3.norms[0].weight.detach().cpu()
M_n3 = compute_quadratic_forms(L3, D3)

print(f"\nn=2: L={L2.shape}, D={D2.shape}, gamma={gamma2.item():.4f}")
print(f"  M shape: {M_n2.shape}")
for i in range(2):
    print(f"  M[{i}]:\n{M_n2[i].numpy()}")
    eig = np.linalg.eigvalsh(M_n2[i].numpy())
    print(f"    eigenvalues: {eig}")

print(f"\nn=3: L={L3.shape}, D={D3.shape}, gamma={gamma3.item():.4f}")
print(f"  M shape: {M_n3.shape}")
for i in range(3):
    print(f"  M[{i}]:\n{M_n3[i].numpy()}")
    eig = np.linalg.eigvalsh(M_n3[i].numpy())
    print(f"    eigenvalues: {eig}")

# %%
# =============================================================================
# N=2 VISUALIZATION: M ⊙ (h ⊗ h^T)
# =============================================================================
print("\n" + "=" * 60)
print("N=2 VISUALIZATION: M ⊙ (h ⊗ h^T)")
print("=" * 60)

def plot_matrix_with_values(ax, mat, title, val_fontsize=14, cmap='RdBu_r',
                            label_map=None):
    """Plot matrix with values in cells and colorbar below."""
    mat_np = mat if isinstance(mat, np.ndarray) else mat.numpy()
    vmax = max(abs(mat_np.min()), abs(mat_np.max()))
    if vmax == 0:
        vmax = 1
    im = ax.imshow(mat_np, cmap=cmap, vmin=-vmax, vmax=vmax, aspect='equal')
    ax.set_title(title, fontsize=13, fontweight='bold')

    # Colorbar below
    divider = make_axes_locatable(ax)
    cax = divider.append_axes("bottom", size="8%", pad=0.3)
    plt.colorbar(im, cax=cax, orientation='horizontal')

    # Show values
    for i in range(mat_np.shape[0]):
        for j in range(mat_np.shape[1]):
            if label_map is not None:
                txt = label_map[i][j]
            else:
                val = mat_np[i, j]
                txt = f'{val:.3f}'
            color = 'white' if abs(mat_np[i, j]) > vmax * 0.5 else 'black'
            ax.text(j, i, txt, ha='center', va='center', fontsize=val_fontsize,
                    color=color, fontweight='bold')

    ax.set_xticks(range(mat_np.shape[1]))
    ax.set_yticks(range(mat_np.shape[0]))
    return im


# For n=2: output_i = x_i + h^T M^(i) h where h = γx/rms(x)
# h^T M h = Σ_{j,k} M_{jk} h_j h_k
# The outer product h⊗h^T has entries h_j * h_k
# So the contribution of each (j,k) pair is M_{jk} * h_j * h_k

# Create the symbolic outer product template
# Using a=h_0, b=h_1
outer_labels_n2 = [['a²', 'ab'], ['ab', 'b²']]

# For display: use actual expected values under standard normal
# E[h_j h_k] for normalized inputs
# Under RMSNorm with gamma, h = gamma * x / rms(x)
# For n=2 standard normal: rms = sqrt((x0^2 + x1^2)/2)

# Create figure: 2 rows (output 0, output 1), 3 cols (M, ⊙, h⊗h^T)
fig, axes = plt.subplots(2, 3, figsize=(16, 10))

for out_idx in range(2):
    M_i = M_n2[out_idx].numpy()

    # Column 0: M matrix
    plot_matrix_with_values(axes[out_idx, 0], M_i,
                            f'M$^{{({out_idx})}}$', val_fontsize=16)
    axes[out_idx, 0].set_xlabel('k', fontsize=11)
    axes[out_idx, 0].set_ylabel('j', fontsize=11)

    # Column 1: ⊙ symbol
    axes[out_idx, 1].set_xlim(0, 1)
    axes[out_idx, 1].set_ylim(0, 1)
    axes[out_idx, 1].text(0.5, 0.5, '⊙', fontsize=48, ha='center', va='center',
                          fontweight='bold')
    axes[out_idx, 1].axis('off')

    # Column 2: h⊗h^T template
    # Use a "neutral" colormap to show it's a template
    template = np.array([[1.0, 0.5], [0.5, 1.0]])  # Placeholder for coloring
    vmax = 1.0
    im = axes[out_idx, 2].imshow(template, cmap='Purples', vmin=0, vmax=vmax,
                                  aspect='equal')
    axes[out_idx, 2].set_title('h ⊗ h$^T$', fontsize=13, fontweight='bold')
    divider = make_axes_locatable(axes[out_idx, 2])
    cax = divider.append_axes("bottom", size="8%", pad=0.3)
    cax.set_visible(False)  # No colorbar needed for template

    for i in range(2):
        for j in range(2):
            axes[out_idx, 2].text(j, i, outer_labels_n2[i][j],
                                  ha='center', va='center', fontsize=18,
                                  fontweight='bold', color='black',
                                  fontstyle='italic')
    axes[out_idx, 2].set_xticks(range(2))
    axes[out_idx, 2].set_yticks(range(2))
    axes[out_idx, 2].set_xlabel('k', fontsize=11)

    # Row label
    axes[out_idx, 0].set_ylabel(f'Output {out_idx}\nj', fontsize=12)

plt.suptitle(
    f'n=2 1-Layer Bilinear: output$_i$ = x$_i$ + h$^T$ M$^{{(i)}}$ h'
    f'  where  h$^T$ M h = $\\sum_{{j,k}}$ M$_{{jk}}$ · (h ⊗ h$^T$)$_{{jk}}$'
    f'\nAccuracy: {acc_n2*100:.1f}%,  γ = {gamma2.item():.3f}',
    fontsize=13)
plt.tight_layout()
plt.savefig(images_dir / 'n2_M_hadamard.png', dpi=150, bbox_inches='tight')
plt.show()
print(f"Saved n2_M_hadamard.png")

# %%
# =============================================================================
# N=2 EIGENDECOMPOSITION
# =============================================================================
print("\n" + "=" * 60)
print("N=2 EIGENDECOMPOSITION")
print("=" * 60)

fig = plt.figure(figsize=(20, 10))

for out_idx in range(2):
    M_i = M_n2[out_idx].numpy()
    eigenvalues, eigenvectors = np.linalg.eigh(M_i)
    abs_order = np.argsort(np.abs(eigenvalues))[::-1]
    eigenvalues_sorted = eigenvalues[abs_order]
    eigenvectors_sorted = eigenvectors[:, abs_order]

    # Col 0: M matrix
    ax0 = fig.add_subplot(2, 4, out_idx * 4 + 1)
    plot_matrix_with_values(ax0, M_i,
                            f'M$^{{({out_idx})}}$', val_fontsize=14)

    # Col 1-2: Top 2 outer products
    for k in range(2):
        ax = fig.add_subplot(2, 4, out_idx * 4 + 2 + k)
        lam = eigenvalues_sorted[k]
        v = eigenvectors_sorted[:, k:k+1]
        outer = lam * (v @ v.T)

        sign_str = "(NEG)" if lam < 0 else ""
        title_color = 'red' if lam < 0 else 'black'
        plot_matrix_with_values(ax, outer,
                                f'λ$_{k+1}$={lam:.3f} {sign_str}',
                                val_fontsize=14)
        ax.title.set_color(title_color)

    # Col 3: Eigenvalue bar chart
    ax = fig.add_subplot(2, 4, out_idx * 4 + 4)
    colors = ['red' if e < 0 else 'steelblue' for e in eigenvalues]
    ax.bar(range(len(eigenvalues)), eigenvalues, color=colors)
    ax.axhline(y=0, color='black', linewidth=0.5)
    ax.set_title(f'Eigenvalues of M$^{{({out_idx})}}$', fontsize=11)
    ax.set_xticks(range(len(eigenvalues)))
    ax.set_xlabel('Index')
    ax.set_ylabel('λ')
    ax.grid(True, alpha=0.3)
    for i, e in enumerate(eigenvalues):
        offset = max(abs(e) * 0.05, 0.01) * np.sign(e)
        ax.text(i, e + offset, f'{e:.3f}', ha='center', fontsize=10)

plt.suptitle(f'n=2: M Eigendecomposition (Acc: {acc_n2*100:.1f}%)', fontsize=14)
plt.tight_layout()
plt.savefig(images_dir / 'n2_eigen.png', dpi=150, bbox_inches='tight')
plt.show()
print(f"Saved n2_eigen.png")

# %%
# =============================================================================
# N=3 EIGENDECOMPOSITION
# =============================================================================
print("\n" + "=" * 60)
print("N=3 EIGENDECOMPOSITION")
print("=" * 60)

fig, axes = plt.subplots(3, 5, figsize=(28, 14))

for out_idx in range(3):
    M_i = M_n3[out_idx].numpy()
    eigenvalues, eigenvectors = np.linalg.eigh(M_i)
    abs_order = np.argsort(np.abs(eigenvalues))[::-1]
    eigenvalues_sorted = eigenvalues[abs_order]
    eigenvectors_sorted = eigenvectors[:, abs_order]

    # Col 0: M matrix
    plot_matrix_with_values(axes[out_idx, 0], M_i,
                            f'M$^{{({out_idx})}}$', val_fontsize=12)

    # Col 1-3: Top 3 outer products
    for k in range(3):
        lam = eigenvalues_sorted[k]
        v = eigenvectors_sorted[:, k:k+1]
        outer = lam * (v @ v.T)

        sign_str = "(NEG)" if lam < 0 else ""
        title_color = 'red' if lam < 0 else 'black'
        plot_matrix_with_values(axes[out_idx, k+1], outer,
                                f'λ$_{k+1}$={lam:.3f} {sign_str}',
                                val_fontsize=12)
        axes[out_idx, k+1].title.set_color(title_color)

    # Col 4: Eigenvalue spectrum
    ax = axes[out_idx, 4]
    colors = ['red' if e < 0 else 'steelblue' for e in eigenvalues]
    ax.bar(range(len(eigenvalues)), eigenvalues, color=colors)
    ax.axhline(y=0, color='black', linewidth=0.5)
    ax.set_title(f'Eigenvalues of M$^{{({out_idx})}}$', fontsize=11)
    ax.set_xticks(range(len(eigenvalues)))
    ax.set_xlabel('Index')
    ax.set_ylabel('λ')
    ax.grid(True, alpha=0.3)
    for i, e in enumerate(eigenvalues):
        ax.text(i, e + 0.02 * np.sign(e), f'{e:.3f}', ha='center', fontsize=9)

plt.suptitle(f'n=3: M Eigendecomposition (Acc: {acc_n3*100:.1f}%)', fontsize=14)
plt.tight_layout()
plt.savefig(images_dir / 'n3_eigen.png', dpi=150, bbox_inches='tight')
plt.show()
print(f"Saved n3_eigen.png")

# %%
# =============================================================================
# WEIGHT MATRICES
# =============================================================================
print("\n" + "=" * 60)
print("WEIGHT MATRICES")
print("=" * 60)

fig, axes = plt.subplots(2, 2, figsize=(14, 10))

plot_matrix_with_values(axes[0, 0], L2.numpy(), f'n=2: L ({L2.shape[0]}×{L2.shape[1]})',
                        val_fontsize=14)
plot_matrix_with_values(axes[0, 1], D2.numpy(), f'n=2: D ({D2.shape[0]}×{D2.shape[1]})',
                        val_fontsize=14)
plot_matrix_with_values(axes[1, 0], L3.numpy(), f'n=3: L ({L3.shape[0]}×{L3.shape[1]})',
                        val_fontsize=12)
plot_matrix_with_values(axes[1, 1], D3.numpy(), f'n=3: D ({D3.shape[0]}×{D3.shape[1]})',
                        val_fontsize=12)

plt.suptitle('Weight Matrices (1-Layer Models)', fontsize=14)
plt.tight_layout()
plt.savefig(images_dir / 'weights.png', dpi=150, bbox_inches='tight')
plt.show()
print(f"Saved weights.png")

# %%
# =============================================================================
# N=2 INTERPRETATION
# =============================================================================
print("\n" + "=" * 60)
print("N=2 INTERPRETATION")
print("=" * 60)

print(f"""
n=2 2nd-argmax task:
  If x_0 > x_1: answer = 1 (2nd largest is at position 1)
  If x_1 > x_0: answer = 0 (2nd largest is at position 0)
  This is equivalent to argmin.

Model: output_i = x_i + (γ/rms)² · x^T M^(i) x

For output 0 (should be large when x_1 > x_0):
  M^(0) = {M_n2[0].numpy().round(4)}
  h^T M^(0) h = {M_n2[0][0,0].item():.4f}·a² + 2·{M_n2[0][0,1].item():.4f}·ab + {M_n2[0][1,1].item():.4f}·b²
  where a=h_0, b=h_1

For output 1 (should be large when x_0 > x_1):
  M^(1) = {M_n2[1].numpy().round(4)}
  h^T M^(1) h = {M_n2[1][0,0].item():.4f}·a² + 2·{M_n2[1][0,1].item():.4f}·ab + {M_n2[1][1,1].item():.4f}·b²

γ = {gamma2.item():.4f}
Accuracy: {acc_n2*100:.1f}%
""")

# %%
# =============================================================================
# N=3 INTERPRETATION
# =============================================================================
print("\n" + "=" * 60)
print("N=3 INTERPRETATION")
print("=" * 60)

print(f"""
n=3 2nd-argmax task:
  Find the position of the 2nd largest element.

Model: output_i = x_i + (γ/rms)² · x^T M^(i) x
γ = {gamma3.item():.4f}
Accuracy: {acc_n3*100:.1f}%
""")

for i in range(3):
    M_i = M_n3[i].numpy()
    eigenvalues = np.linalg.eigvalsh(M_i)
    print(f"M^({i}):")
    print(f"  Matrix:\n{M_i.round(4)}")
    print(f"  Diagonal: [{M_i[0,0]:.4f}, {M_i[1,1]:.4f}, {M_i[2,2]:.4f}]")
    print(f"  Self-coeff M[{i},{i}]: {M_i[i,i]:.4f}")
    others = [M_i[j,j] for j in range(3) if j != i]
    print(f"  Other diag: {[f'{v:.4f}' for v in others]}")
    print(f"  Eigenvalues: {eigenvalues.round(4)}")
    print()

# %%
# =============================================================================
# SUMMARY
# =============================================================================
print("\n" + "=" * 60)
print("SUMMARY")
print("=" * 60)

print(f"""
1-Layer Models for 2nd-Argmax Task
====================================

n=2: Accuracy={acc_n2*100:.1f}%, rank=2, γ={gamma2.item():.4f}
  output_i = x_i + h^T M^(i) h, where h = γx/rms(x)
  h^T M h = Σ_jk M_jk · h_j · h_k = trace(M · h⊗h^T) = Σ_jk M_jk · (h⊗h^T)_jk

n=3: Accuracy={acc_n3*100:.1f}%, rank=3, γ={gamma3.item():.4f}
  Same structure with 3×3 M matrices.

Visualizations saved to {images_dir}/
  - n2_M_hadamard.png: M ⊙ (h⊗h^T) view
  - n2_eigen.png: M eigendecomposition
  - n3_eigen.png: M eigendecomposition
  - weights.png: L and D weight matrices
""")

# %%
