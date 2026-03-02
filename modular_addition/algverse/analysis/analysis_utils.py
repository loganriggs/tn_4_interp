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


# =============================================================================
# SHARED PLOTTING FUNCTIONS (used across all analysis scripts)
# =============================================================================

def plot_mat(ax, mat, title, fontsize=None, cmap='RdBu_r', memory_boundary=None):
    """
    Plot matrix heatmap with values in cells. No colorbar.

    Args:
        ax: matplotlib axes
        mat: numpy array or torch tensor
        title: subplot title
        fontsize: font size for cell values (auto-scaled if None)
        cmap: colormap
        memory_boundary: if set, draw green dashed line at this position
    """
    if isinstance(mat, torch.Tensor):
        mat = mat.numpy()

    vmax = max(abs(mat.min()), abs(mat.max()))
    if vmax == 0:
        vmax = 1

    im = ax.imshow(mat, cmap=cmap, vmin=-vmax, vmax=vmax, aspect='equal')
    ax.set_title(title, fontsize=12, fontweight='bold')

    n = max(mat.shape)

    # Auto font size based on matrix dimension
    if fontsize is None:
        if n <= 3:
            fontsize = 14
        elif n <= 5:
            fontsize = 11
        elif n <= 10:
            fontsize = 9
        elif n <= 15:
            fontsize = 7
        else:
            fontsize = 5

    # Show values in each cell
    for i in range(mat.shape[0]):
        for j in range(mat.shape[1]):
            val = mat[i, j]
            color = 'white' if abs(val) > vmax * 0.5 else 'black'
            if abs(val) < 0.005:
                txt = '0'
            elif abs(val) < 10:
                txt = f'{val:.2f}' if n <= 5 else f'{val:.1f}'
            else:
                txt = f'{val:.0f}'
            ax.text(j, i, txt, ha='center', va='center',
                    fontsize=fontsize, color=color)

    # Memory boundary
    if memory_boundary is not None:
        ax.axhline(y=memory_boundary - 0.5, color='green', linestyle='--',
                    linewidth=2, alpha=0.7)
        ax.axvline(x=memory_boundary - 0.5, color='green', linestyle='--',
                    linewidth=2, alpha=0.7)

    ax.set_xticks(range(mat.shape[1]))
    ax.set_yticks(range(mat.shape[0]))
    ax.set_xlabel('pos', fontsize=10)
    ax.set_ylabel('pos', fontsize=10)
    return im


def plot_weights(weights_dict, title='', save_path=None, memory_boundary=None,
                 figsize=None):
    """
    Plot L and D weight matrices for all layers.

    Args:
        weights_dict: list of (L, D, layer_name, sparsity_L, sparsity_D) tuples
        title: suptitle
        save_path: path to save figure
        memory_boundary: position of memory boundary line
        figsize: optional figure size override
    """
    n_layers = len(weights_dict)
    if figsize is None:
        n = max(weights_dict[0][0].shape)
        w = max(16, n * 1.8)
        figsize = (w, 7 * n_layers)

    fig, axes = plt.subplots(n_layers, 2, figsize=figsize)
    if n_layers == 1:
        axes = axes.reshape(1, -1)

    for row, (L, D, name, sp_L, sp_D) in enumerate(weights_dict):
        L_np = L.numpy() if isinstance(L, torch.Tensor) else L
        D_np = D.numpy() if isinstance(D, torch.Tensor) else D
        plot_mat(axes[row, 0], L_np,
                 f'{name} L ({L_np.shape[0]}x{L_np.shape[1]}) - {sp_L:.0f}% sparse',
                 memory_boundary=memory_boundary)
        plot_mat(axes[row, 1], D_np,
                 f'{name} D ({D_np.shape[0]}x{D_np.shape[1]}) - {sp_D:.0f}% sparse',
                 memory_boundary=memory_boundary)

    if title:
        plt.suptitle(title, fontsize=14)
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
    return fig


def plot_M_eigen(M_tensor, layer_name, outputs_to_show, n_top=3,
                 memory_boundary=None, save_path=None, fontsize=None):
    """
    Plot M matrices with top eigenvalue outer products and eigenvalue spectrum.

    Args:
        M_tensor: (n_out, n, n) quadratic form matrices
        layer_name: e.g. 'M1', 'M2'
        outputs_to_show: list of output indices to plot
        n_top: number of top eigenvalue outer products to show
        memory_boundary: position of memory boundary line
        save_path: path to save figure
        fontsize: override font size for cell values
    """
    n_outputs = len(outputs_to_show)
    n_model = M_tensor.shape[1]
    n_cols = 1 + n_top + 1  # M + outer products + eigenvalue plot

    if isinstance(M_tensor, torch.Tensor):
        M_tensor = M_tensor.numpy()

    fig = plt.figure(figsize=(7 * n_cols, 7 * n_outputs))

    for idx, i in enumerate(outputs_to_show):
        M_i = M_tensor[i]
        eigenvalues, eigenvectors = np.linalg.eigh(M_i)
        abs_order = np.argsort(np.abs(eigenvalues))[::-1]
        eigenvalues_sorted = eigenvalues[abs_order]
        eigenvectors_sorted = eigenvectors[:, abs_order]

        # Column 0: M[i] matrix
        ax0 = fig.add_subplot(n_outputs, n_cols, idx * n_cols + 1)
        rank = np.sum(np.abs(eigenvalues) > 1e-6)
        output_label = f'mem{i - memory_boundary}' if memory_boundary and i >= memory_boundary else str(i)
        plot_mat(ax0, M_i, f'{layer_name}[{output_label}] (rank={rank})',
                 fontsize=fontsize, memory_boundary=memory_boundary)

        # Columns 1..n_top: outer products
        for k in range(n_top):
            ax = fig.add_subplot(n_outputs, n_cols, idx * n_cols + 2 + k)
            if k < len(eigenvalues_sorted):
                lam = eigenvalues_sorted[k]
                v = eigenvectors_sorted[:, k:k+1]
                outer = lam * (v @ v.T)
                sign_str = "(NEG)" if lam < 0 else ""
                plot_mat(ax, outer, f'\u03bb$_{k+1}$={lam:.2f} {sign_str}',
                         fontsize=fontsize, memory_boundary=memory_boundary)
                if lam < 0:
                    ax.title.set_color('red')
            else:
                ax.set_visible(False)

        # Last column: eigenvalue spectrum
        ax_eig = fig.add_subplot(n_outputs, n_cols, idx * n_cols + n_cols)
        ax_eig.axhline(y=0, color='black', linestyle='-', linewidth=0.5)
        ax_eig.plot(range(len(eigenvalues)), eigenvalues, marker='o',
                    linestyle='-', color='steelblue', markersize=5)
        neg_idx = np.where(eigenvalues < 0)[0]
        if len(neg_idx) > 0:
            ax_eig.scatter(neg_idx, eigenvalues[neg_idx], color='red', s=50,
                          zorder=5, label='Negative')
        pos_idx = np.where(eigenvalues >= 0)[0]
        if len(pos_idx) > 0:
            ax_eig.scatter(pos_idx, eigenvalues[pos_idx], color='blue', s=50,
                          zorder=5, label='Positive')
        ax_eig.set_title('Eigenvalues (sorted asc)', fontsize=10)
        ax_eig.set_xlabel('Index')
        ax_eig.set_ylabel('\u03bb')
        ax_eig.grid(True, alpha=0.3)
        if len(neg_idx) > 0 or len(pos_idx) > 0:
            ax_eig.legend(fontsize=8)

    plt.suptitle(
        f'{layer_name} Quadratic Forms: M and Top {n_top} Outer Products '
        f'(\u03bb\u00b7v\u00b7v\u1d40)',
        fontsize=14)
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
    return fig


def plot_M_only(M_tensor, layer_name='M', outputs_to_show=None,
                memory_boundary=None, save_path=None, fontsize=None):
    """
    Plot just the M matrices side by side (no eigendecomposition).

    Args:
        M_tensor: (n_out, n, n) quadratic form matrices
        layer_name: prefix for titles
        outputs_to_show: list of indices (default: all)
        memory_boundary: position of memory boundary line
        save_path: path to save figure
        fontsize: override font size
    """
    if isinstance(M_tensor, torch.Tensor):
        M_tensor = M_tensor.numpy()

    if outputs_to_show is None:
        outputs_to_show = list(range(M_tensor.shape[0]))

    n = len(outputs_to_show)
    fig, axes = plt.subplots(1, n, figsize=(5 * n, 5))
    if n == 1:
        axes = [axes]

    for idx, i in enumerate(outputs_to_show):
        output_label = f'mem{i - memory_boundary}' if memory_boundary and i >= memory_boundary else str(i)
        plot_mat(axes[idx], M_tensor[i],
                 f'{layer_name}$^{{({output_label})}}$',
                 fontsize=fontsize, memory_boundary=memory_boundary)

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
    return fig
