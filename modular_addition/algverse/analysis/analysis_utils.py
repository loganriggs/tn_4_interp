"""
Analysis utilities for symmetric bilinear networks.

Key insight: The bilinear layer computes D @ (L @ h)² where ² is ELEMENTWISE.
This means the output is a QUADRATIC FORM, not a linear transform.

For output position i:
    bilinear_i = Σ_r D_ir (Σ_j L_rj h_j)²
              = Σ_r D_ir Σ_{j,k} L_rj L_rk h_j h_k
              = Σ_{j,k} T_ijk h_j h_k
              = h^T M^(i) h

where:
    T_ijk = Σ_r D_ir L_rj L_rk   (3rd-order tensor)
    M^(i)_jk = T_ijk             (n symmetric matrices, one per output)

The full output is:
    output_i = x_i + (γ/rms)² · x^T M^(i) x
"""

import torch
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from itertools import combinations
import pickle


# =============================================================================
# CHECKPOINT LOADING
# =============================================================================

def load_checkpoint(path, prefer_sparse=True):
    """
    Load checkpoint from .pt file or .pkl pruned results.

    Returns: dict with keys:
        - state_dict: model weights
        - config: {n, num_layers, rank, seed}
        - accuracy: float
        - sparsity: dict or None
    """
    path = Path(path)

    if path.suffix == '.pkl':
        with open(path, 'rb') as f:
            data = pickle.load(f)

        if 'results' in data:
            # Pruned results - pick best
            best = max(data['results'], key=lambda x: x.get('final_acc', x.get('acc', 0)))
            return {
                'state_dict': best['state_dict'],
                'config': data['config'],
                'accuracy': best.get('final_acc', best.get('acc')),
                'sparsity': best.get('sparsity'),
                'threshold': best.get('threshold'),
            }
        else:
            return {
                'state_dict': data.get('state_dict', data.get('l1_state')),
                'config': data['config'],
                'accuracy': data.get('accuracy', data.get('l1_acc')),
                'sparsity': None,
            }
    else:
        # .pt file
        data = torch.load(path, map_location='cpu', weights_only=False)
        return {
            'state_dict': data['state_dict'],
            'config': data['config'],
            'accuracy': data.get('accuracy', data.get('final_acc')),
            'sparsity': data.get('sparsity'),
        }


# =============================================================================
# QUADRATIC FORM COMPUTATION (THE CORRECT MATH)
# =============================================================================

def compute_quadratic_forms(L, D):
    """
    Compute the n quadratic form matrices M^(i) for a bilinear layer.

    The bilinear layer computes: bilinear_i = h^T M^(i) h
    where M^(i)_jk = Σ_r D_ir L_rj L_rk

    Args:
        L: (rank, n) projection matrix
        D: (n, rank) output matrix

    Returns:
        M: (n, n, n) tensor where M[i] is the quadratic form matrix for output i
        T: (n, n, n) tensor where T[i,j,k] = Σ_r D_ir L_rj L_rk (same as M, different view)
    """
    rank, n = L.shape

    # M^(i)_jk = Σ_r D_ir L_rj L_rk
    # Using einsum: M_ijk = D_ir L_rj L_rk summed over r
    M = torch.einsum('ir,rj,rk->ijk', D, L, L)

    return M


def compute_full_tensor(L, D):
    """
    Compute the 3rd-order tensor T where T_ijk = Σ_r D_ir L_rj L_rk.

    This is the same as compute_quadratic_forms but emphasizes the tensor view.
    """
    return torch.einsum('ir,rj,rk->ijk', D, L, L)


# =============================================================================
# FORWARD COMPUTATION
# =============================================================================

def rmsnorm(x, weight, eps=1e-6):
    """
    Apply RMSNorm.

    Args:
        x: input tensor (..., n)
        weight: scalar or (n,) vector
        eps: epsilon for numerical stability

    Returns:
        h: normalized output
        rms: the rms values used
    """
    rms = torch.sqrt((x ** 2).mean(dim=-1, keepdim=True) + eps)
    h = weight * (x / rms)
    return h, rms


def bilinear_forward(h, L, D):
    """
    Compute bilinear layer output: D @ (L @ h)²

    Returns:
        output: bilinear output
        Lh: intermediate L @ h values (useful for analysis)
    """
    Lh = h @ L.T  # (..., rank)
    output = (Lh ** 2) @ D.T  # (..., n)
    return output, Lh


def bilinear_as_quadratic(h, M):
    """
    Compute bilinear output using quadratic form matrices.

    bilinear_i = h^T M^(i) h

    This is mathematically equivalent to bilinear_forward but shows
    the quadratic structure explicitly.

    Args:
        h: (..., n) input
        M: (n, n, n) quadratic form matrices

    Returns:
        output: (..., n)
    """
    # For each output i: h^T M[i] h
    # Using einsum: out_i = h_j M_ijk h_k
    return torch.einsum('...j,ijk,...k->...i', h, M, h)


# =============================================================================
# LAYER-WISE COMPUTATION
# =============================================================================

def compute_layer_outputs(x, state_dict, num_layers):
    """
    Compute outputs from each layer.

    Returns dict with:
        - x: input
        - h1, h2, ...: normalized inputs to each layer
        - rms1, rms2, ...: rms values
        - Lh1, Lh2, ...: L @ h intermediate values
        - r1, r2, ...: layer outputs
        - output: final output
    """
    result = {'x': x}
    h = x

    for i in range(num_layers):
        L = state_dict[f'layers.{i}.L']
        D = state_dict[f'layers.{i}.D']
        norm_w = state_dict[f'norms.{i}.weight']

        # Normalize
        h_norm, rms = rmsnorm(h, norm_w)
        result[f'h{i+1}'] = h_norm
        result[f'rms{i+1}'] = rms

        # Bilinear
        r, Lh = bilinear_forward(h_norm, L, D)
        result[f'Lh{i+1}'] = Lh
        result[f'r{i+1}'] = r

        # Residual
        h = h + r

    result['output'] = h
    return result


# =============================================================================
# 2-LAYER DECOMPOSITION (A, B, C terms)
# =============================================================================

def compute_2layer_decomposition(x, state_dict):
    """
    Compute the 5 paths for 2-layer model: x, r1, A, B, C

    Where:
        A = contribution from x² through layer 2
        B = cross term (x × r1) through layer 2
        C = contribution from r1² through layer 2
    """
    L1 = state_dict['layers.0.L']
    D1 = state_dict['layers.0.D']
    norm1_w = state_dict['norms.0.weight']

    L2 = state_dict['layers.1.L']
    D2 = state_dict['layers.1.D']
    norm2_w = state_dict['norms.1.weight']

    # Layer 1
    h1, rms1 = rmsnorm(x, norm1_w)
    r1, Lh1 = bilinear_forward(h1, L1, D1)

    # Input to layer 2
    h_mid = x + r1
    rms2 = torch.sqrt((h_mid ** 2).mean(dim=-1, keepdim=True) + 1e-6)

    # Decompose the normalized input
    x_contrib = norm2_w * x / rms2
    r1_contrib = norm2_w * r1 / rms2

    # L2 projections
    Lx = x_contrib @ L2.T
    Lr1 = r1_contrib @ L2.T

    # A: (Lx)² term - x contribution squared
    A = (Lx ** 2) @ D2.T

    # B: 2 * Lx * Lr1 cross term
    B = (2 * Lx * Lr1) @ D2.T

    # C: (Lr1)² term - r1 contribution squared
    C = (Lr1 ** 2) @ D2.T

    # Full layer 2 output
    h2, _ = rmsnorm(h_mid, norm2_w)
    r2, _ = bilinear_forward(h2, L2, D2)

    return {
        'x': x,
        'r1': r1,
        'A': A,
        'B': B,
        'C': C,
        'r2': r2,
        'output': x + r1 + r2,
    }


# =============================================================================
# ABLATION
# =============================================================================

def powerset_ablation(components, component_names, targets):
    """
    Run ablation over all 2^n combinations of components.

    Args:
        components: dict mapping name -> tensor
        component_names: list of names to include in ablation
        targets: ground truth labels

    Returns:
        List of {components: tuple, accuracy: float}, sorted by accuracy desc
    """
    results = []
    n_components = len(component_names)

    for num_active in range(n_components + 1):
        for combo in combinations(range(n_components), num_active):
            # Build output from selected components
            if len(combo) == 0:
                output = torch.zeros_like(components[component_names[0]])
                names = ('none',)
            else:
                output = sum(components[component_names[i]] for i in combo)
                names = tuple(component_names[i] for i in combo)

            preds = output.argmax(dim=-1)
            acc = (preds == targets).float().mean().item()

            results.append({
                'components': names,
                'num_components': len(combo),
                'accuracy': acc,
            })

    results.sort(key=lambda x: -x['accuracy'])
    return results


def removal_ablation(components, component_names, targets):
    """
    Remove one component at a time from full model.

    Returns:
        Dict mapping removed_component -> accuracy_without_it
    """
    # Full accuracy
    full_output = sum(components[name] for name in component_names)
    full_acc = (full_output.argmax(dim=-1) == targets).float().mean().item()

    results = {'full': full_acc}

    for remove_name in component_names:
        remaining = [n for n in component_names if n != remove_name]
        output = sum(components[n] for n in remaining)
        acc = (output.argmax(dim=-1) == targets).float().mean().item()
        results[f'without_{remove_name}'] = acc
        results[f'delta_{remove_name}'] = acc - full_acc

    return results


# =============================================================================
# VISUALIZATION
# =============================================================================

def plot_weight_matrix(ax, mat, title, show_values=True, fontsize=7, cmap='RdBu_r'):
    """Plot weight matrix as heatmap with optional values."""
    if isinstance(mat, torch.Tensor):
        mat = mat.numpy()

    vmax = max(abs(mat.min()), abs(mat.max()))
    if vmax == 0:
        vmax = 1

    im = ax.imshow(mat, cmap=cmap, vmin=-vmax, vmax=vmax, aspect='auto')
    ax.set_title(title, fontsize=10)

    if show_values and mat.size <= 64:  # Only show values for small matrices
        for i in range(mat.shape[0]):
            for j in range(mat.shape[1]):
                val = mat[i, j]
                color = 'white' if abs(val) > vmax * 0.5 else 'black'
                ax.text(j, i, f'{val:.2f}', ha='center', va='center',
                       fontsize=fontsize, color=color)

    plt.colorbar(im, ax=ax, shrink=0.8)
    return im


def plot_quadratic_forms(M, title_prefix="M", save_path=None):
    """
    Plot the n quadratic form matrices M^(0), M^(1), ..., M^(n-1).

    Args:
        M: (n, n, n) tensor of quadratic form matrices
        title_prefix: prefix for subplot titles
        save_path: optional path to save figure
    """
    if isinstance(M, torch.Tensor):
        M = M.numpy()

    n = M.shape[0]
    fig, axes = plt.subplots(1, n, figsize=(4*n, 4))
    if n == 1:
        axes = [axes]

    # Global vmax for consistent coloring
    vmax = max(abs(M.min()), abs(M.max()))

    for i in range(n):
        plot_weight_matrix(axes[i], M[i], f'{title_prefix}^({i})', fontsize=7)
        axes[i].set_xlabel('k')
        axes[i].set_ylabel('j')

    plt.suptitle(f'Quadratic Form Matrices: bilinear_i = h^T {title_prefix}^(i) h', fontsize=12)
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150)

    return fig


def plot_heatmap_row(ax, data, title, mark_cols=None, vmax=None, fontsize=8):
    """
    Plot 1×n heatmap row with optional column markers.

    Args:
        mark_cols: dict mapping column_idx -> (color, linestyle)
    """
    if isinstance(data, torch.Tensor):
        data = data.numpy()

    if data.ndim == 1:
        data = data.reshape(1, -1)

    if vmax is None:
        vmax = max(abs(data.min()), abs(data.max()))
        if vmax == 0:
            vmax = 1

    im = ax.imshow(data, cmap='RdBu_r', vmin=-vmax, vmax=vmax, aspect='auto')
    ax.set_title(title, fontsize=9, loc='left')
    ax.set_xticks(range(data.shape[1]))
    ax.set_yticks([])

    for j in range(data.shape[1]):
        val = data[0, j]
        color = 'white' if abs(val) > vmax * 0.5 else 'black'
        ax.text(j, 0, f'{val:.2f}', ha='center', va='center', fontsize=fontsize, color=color)

    if mark_cols:
        for col, (color, style) in mark_cols.items():
            if col < data.shape[1]:
                rect = plt.Rectangle((col - 0.5, -0.5), 1, 1,
                                     fill=False, edgecolor=color, linewidth=2, linestyle=style)
                ax.add_patch(rect)

    return im


# =============================================================================
# ANALYSIS HELPERS
# =============================================================================

def analyze_quadratic_form(M, name="M"):
    """
    Analyze the structure of a quadratic form matrix.

    Returns dict with:
        - diagonal: diagonal entries
        - off_diagonal: off-diagonal entries
        - is_diag_dominant: True if diagonal dominates
        - eigenvalues: eigenvalues (for definiteness)
    """
    if isinstance(M, torch.Tensor):
        M = M.numpy()

    n = M.shape[0]
    diag = np.diag(M)
    off_diag = M[~np.eye(n, dtype=bool)]

    # Eigenvalue analysis
    eigenvalues = np.linalg.eigvalsh(M)

    return {
        'name': name,
        'diagonal_mean': diag.mean(),
        'diagonal_std': diag.std(),
        'off_diagonal_mean': off_diag.mean(),
        'off_diagonal_std': off_diag.std(),
        'is_diag_dominant': abs(diag.mean()) > abs(off_diag.mean()) * 2,
        'eigenvalues': eigenvalues,
        'is_positive_definite': all(eigenvalues > 0),
        'is_negative_definite': all(eigenvalues < 0),
    }


def print_quadratic_analysis(M, layer_name="Layer"):
    """Print analysis of all n quadratic form matrices."""
    if isinstance(M, torch.Tensor):
        M = M.numpy()

    n = M.shape[0]
    print(f"\n{layer_name} Quadratic Form Analysis:")
    print(f"  bilinear_i = h^T M^(i) h")
    print(f"  M^(i)_jk = Σ_r D_ir L_rj L_rk")
    print()

    for i in range(n):
        analysis = analyze_quadratic_form(M[i], f"M^({i})")
        print(f"  M^({i}):")
        print(f"    Diagonal mean: {analysis['diagonal_mean']:.3f}")
        print(f"    Off-diagonal mean: {analysis['off_diagonal_mean']:.3f}")
        print(f"    Eigenvalues: {analysis['eigenvalues'].round(3)}")
        definite = "positive" if analysis['is_positive_definite'] else \
                   "negative" if analysis['is_negative_definite'] else "indefinite"
        print(f"    Definiteness: {definite}")


def sparsity(tensor):
    """Compute sparsity (fraction of zeros) in a tensor."""
    return (tensor == 0).sum().item() / tensor.numel()
