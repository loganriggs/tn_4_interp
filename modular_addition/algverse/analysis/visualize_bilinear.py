"""
Visualization utilities for symmetric bilinear networks.

Handles 1-4 layers, with or without memory dimensions.
"""

import torch
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import pickle
from typing import Dict, List, Optional, Tuple, Union

# =============================================================================
# CHECKPOINT LOADING
# =============================================================================

def load_model_weights(checkpoint_path: Union[str, Path]) -> Dict:
    """
    Load model weights from checkpoint file.

    Handles both .pt and .pkl formats.

    Returns:
        dict with keys:
            - weights: list of (L, D) tuples for each layer
            - gammas: list of gamma values for each layer
            - config: dict with n_task, n_model, rank, num_layers, has_memory
            - accuracy: float (if available)
            - sparsity: float (if available)
    """
    path = Path(checkpoint_path)

    if path.suffix == '.pkl':
        with open(path, 'rb') as f:
            data = pickle.load(f)

        state = data.get('state_dict', data.get('l1_state'))
        config = data.get('config', {})
        accuracy = data.get('accuracy', data.get('final_acc'))
        sparsity = data.get('sparsity')

    else:  # .pt file
        data = torch.load(path, map_location='cpu', weights_only=False)
        state = data['state_dict']
        config = data.get('config', {})
        accuracy = data.get('accuracy', data.get('final_acc'))
        sparsity = data.get('sparsity')

    # Count layers
    num_layers = 0
    while f'layers.{num_layers}.L' in state:
        num_layers += 1

    # Extract weights
    weights = []
    gammas = []
    for i in range(num_layers):
        L = state[f'layers.{i}.L']
        D = state[f'layers.{i}.D']
        # Check for asymmetric case (R != L)
        R = state.get(f'layers.{i}.R', L)  # Default to L if symmetric
        weights.append((L, D, R))

        gamma = state.get(f'norms.{i}.weight', torch.tensor(1.0))
        gammas.append(gamma)

    # Determine dimensions
    L0, D0, _ = weights[0]
    rank = L0.shape[0]
    n_model = L0.shape[1]

    # Determine n_task (task dimensions, excluding memory)
    n_task = config.get('n_task', config.get('n', n_model))
    has_memory = n_model > n_task
    n_memory = n_model - n_task if has_memory else 0

    return {
        'weights': weights,
        'gammas': gammas,
        'config': {
            'n_task': n_task,
            'n_model': n_model,
            'n_memory': n_memory,
            'rank': rank,
            'num_layers': num_layers,
            'has_memory': has_memory,
        },
        'accuracy': accuracy,
        'sparsity': sparsity,
        'state_dict': state,
    }


# =============================================================================
# QUADRATIC FORM COMPUTATION
# =============================================================================

def compute_quadratic_forms(L: torch.Tensor, D: torch.Tensor) -> torch.Tensor:
    """
    Compute the n quadratic form matrices M^(i) for a bilinear layer.

    M^(i)_jk = Σ_r D_ir L_rj L_rk

    Args:
        L: (rank, n) projection matrix
        D: (n, rank) output matrix

    Returns:
        M: (n, n, n) tensor where M[i] is the quadratic form matrix for output i
    """
    return torch.einsum('ir,rj,rk->ijk', D, L, L)


# =============================================================================
# WEIGHT MATRIX VISUALIZATION
# =============================================================================

def plot_weight_matrices(
    model_data: Dict,
    title: str = "Model Weights",
    save_path: Optional[Path] = None,
    figsize_per_layer: Tuple[float, float] = (18, 8),
) -> plt.Figure:
    """
    Plot L (and R if asymmetric) and D matrices for all layers.

    Args:
        model_data: dict from load_model_weights()
        title: figure title
        save_path: optional path to save figure
        figsize_per_layer: figure size per layer row

    Returns:
        matplotlib Figure
    """
    weights = model_data['weights']
    gammas = model_data['gammas']
    config = model_data['config']
    num_layers = config['num_layers']
    n_task = config['n_task']
    has_memory = config['has_memory']

    # Check if any layer is asymmetric
    is_asymmetric = any(not torch.equal(L, R) for L, _, R in weights)

    # Figure layout: one row per layer, 2 or 3 columns (L, [R], D)
    n_cols = 3 if is_asymmetric else 2
    fig, axes = plt.subplots(
        num_layers, n_cols,
        figsize=(figsize_per_layer[0], figsize_per_layer[1] * num_layers),
        squeeze=False
    )

    for layer_idx, (L, D, R) in enumerate(weights):
        L_np = L.numpy() if isinstance(L, torch.Tensor) else L
        D_np = D.numpy() if isinstance(D, torch.Tensor) else D
        R_np = R.numpy() if isinstance(R, torch.Tensor) else R

        gamma = gammas[layer_idx]
        gamma_val = gamma.item() if isinstance(gamma, torch.Tensor) else gamma

        # Compute sparsity
        L_sparsity = (L_np == 0).sum() / L_np.size * 100
        D_sparsity = (D_np == 0).sum() / D_np.size * 100

        col_idx = 0

        # Plot L
        ax = axes[layer_idx, col_idx]
        _plot_matrix_with_memory(ax, L_np, f'L{layer_idx+1} ({L_np.shape[0]}×{L_np.shape[1]}) - {L_sparsity:.0f}% sparse',
                                  n_task, has_memory, mark_memory_cols=True)
        col_idx += 1

        # Plot R if asymmetric
        if is_asymmetric:
            ax = axes[layer_idx, col_idx]
            R_sparsity = (R_np == 0).sum() / R_np.size * 100
            _plot_matrix_with_memory(ax, R_np, f'R{layer_idx+1} ({R_np.shape[0]}×{R_np.shape[1]}) - {R_sparsity:.0f}% sparse',
                                      n_task, has_memory, mark_memory_cols=True)
            col_idx += 1

        # Plot D
        ax = axes[layer_idx, col_idx]
        _plot_matrix_with_memory(ax, D_np, f'D{layer_idx+1} ({D_np.shape[0]}×{D_np.shape[1]}) - {D_sparsity:.0f}% sparse',
                                  n_task, has_memory, mark_memory_rows=True)

        # Add gamma annotation
        axes[layer_idx, 0].annotate(f'γ{layer_idx+1}={gamma_val:.4f}',
                                     xy=(0.02, 0.98), xycoords='axes fraction',
                                     fontsize=9, ha='left', va='top',
                                     bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

    accuracy_str = f", Acc: {model_data['accuracy']*100:.1f}%" if model_data['accuracy'] else ""
    sp = model_data['sparsity']
    if sp and isinstance(sp, (int, float)):
        sparsity_str = f", Sparsity: {sp*100:.1f}%"
    else:
        sparsity_str = ""
    mem_str = f" (memory={config['n_memory']})" if has_memory else ""

    plt.suptitle(f"{title}{mem_str}{accuracy_str}{sparsity_str}\nGreen lines mark memory dimensions", fontsize=12)
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')

    return fig


def _plot_matrix_with_memory(
    ax: plt.Axes,
    mat: np.ndarray,
    title: str,
    n_task: int,
    has_memory: bool,
    mark_memory_rows: bool = False,
    mark_memory_cols: bool = False,
    show_values: bool = True,
    fontsize: int = 7,
) -> None:
    """Helper to plot a matrix with memory dimension markers and values."""
    vmax = max(abs(mat.min()), abs(mat.max()))
    if vmax == 0:
        vmax = 1

    im = ax.imshow(mat, cmap='RdBu_r', vmin=-vmax, vmax=vmax, aspect='equal')
    ax.set_title(title, fontsize=10)

    # Colorbar below the image
    from mpl_toolkits.axes_grid1 import make_axes_locatable
    divider = make_axes_locatable(ax)
    cax = divider.append_axes("bottom", size="5%", pad=0.3)
    plt.colorbar(im, cax=cax, orientation='horizontal')

    # Show values in each cell
    if show_values:
        for i in range(mat.shape[0]):
            for j in range(mat.shape[1]):
                val = mat[i, j]
                # Choose text color based on background
                color = 'white' if abs(val) > vmax * 0.5 else 'black'
                # Format: show 1 decimal for small values, 0 for large
                if abs(val) < 10:
                    txt = f'{val:.1f}'
                else:
                    txt = f'{val:.0f}'
                ax.text(j, i, txt, ha='center', va='center', fontsize=fontsize, color=color)

    # Show zeros in gray (lighter overlay)
    zeros_mask = (mat == 0)
    if zeros_mask.any():
        ax.imshow(zeros_mask, cmap='Greys', alpha=0.3, aspect='auto')

    # Mark memory dimensions
    if has_memory:
        if mark_memory_cols and mat.shape[1] > n_task:
            ax.axvline(x=n_task - 0.5, color='green', linestyle='--', linewidth=2, alpha=0.7)
        if mark_memory_rows and mat.shape[0] > n_task:
            ax.axhline(y=n_task - 0.5, color='green', linestyle='--', linewidth=2, alpha=0.7)


# =============================================================================
# QUADRATIC FORM (M MATRIX) VISUALIZATION WITH EIGENDECOMPOSITION
# =============================================================================

def plot_M_matrices_with_eigen(
    model_data: Dict,
    layer_idx: int = 0,
    outputs_to_show: Optional[List[int]] = None,
    n_eigen: int = 3,
    title: str = None,
    save_path: Optional[Path] = None,
) -> plt.Figure:
    """
    Plot M matrices with top eigenvalue outer products and eigenvalue spectrum.

    For each output i:
    - M[i] matrix
    - Top n_eigen outer products: λ_k * v_k @ v_k.T (sorted by |λ|)
    - Eigenvalue line plot (linear scale, showing negative values)

    Args:
        model_data: dict from load_model_weights()
        layer_idx: which layer (0-indexed)
        outputs_to_show: list of output indices to show (default: all)
        n_eigen: number of top eigenvalues to show
        title: figure title
        save_path: optional path to save figure

    Returns:
        matplotlib Figure
    """
    L, D, _ = model_data['weights'][layer_idx]
    config = model_data['config']
    n_task = config['n_task']
    n_model = config['n_model']
    has_memory = config['has_memory']

    # Compute quadratic forms
    M = compute_quadratic_forms(L, D)

    if outputs_to_show is None:
        outputs_to_show = list(range(n_model))

    n_outputs = len(outputs_to_show)

    from mpl_toolkits.axes_grid1 import make_axes_locatable

    # Figure: n_eigen + 2 columns per row (M, outer1, ..., outerN, eigenvalues)
    # Larger size to accommodate values
    n_cols = n_eigen + 2
    cell_size = max(0.4, 4.0 / n_model)  # Scale cell size based on matrix dimension
    fig_width = (n_model * cell_size + 2) * n_cols
    fig_height = (n_model * cell_size + 2) * n_outputs
    fig = plt.figure(figsize=(fig_width, fig_height))

    # Font size for values - smaller for larger matrices
    val_fontsize = max(5, min(8, int(40 / n_model)))

    for idx, i in enumerate(outputs_to_show):
        M_i = M[i].numpy()

        # Eigendecomposition (M is symmetric)
        eigenvalues, eigenvectors = np.linalg.eigh(M_i)

        # Sort by absolute value (descending)
        abs_order = np.argsort(np.abs(eigenvalues))[::-1]
        eigenvalues_sorted = eigenvalues[abs_order]
        eigenvectors_sorted = eigenvectors[:, abs_order]

        row = idx

        # Column 1: M[i] matrix
        ax1 = fig.add_subplot(n_outputs, n_cols, row * n_cols + 1)
        vmax = max(abs(M_i.min()), abs(M_i.max()))
        if vmax == 0:
            vmax = 1
        im1 = ax1.imshow(M_i, cmap='RdBu_r', vmin=-vmax, vmax=vmax, aspect='equal')
        rank = np.sum(np.abs(eigenvalues) > 1e-6)
        output_label = f'mem' if i >= n_task else str(i)
        ax1.set_title(f'M{layer_idx+1}[{output_label}] (rank={rank})', fontsize=11)

        # Show values in cells
        for ii in range(M_i.shape[0]):
            for jj in range(M_i.shape[1]):
                val = M_i[ii, jj]
                color = 'white' if abs(val) > vmax * 0.5 else 'black'
                txt = f'{val:.1f}' if abs(val) < 10 else f'{val:.0f}'
                ax1.text(jj, ii, txt, ha='center', va='center', fontsize=val_fontsize, color=color)

        # Colorbar below
        divider = make_axes_locatable(ax1)
        cax = divider.append_axes("bottom", size="5%", pad=0.3)
        plt.colorbar(im1, cax=cax, orientation='horizontal')

        # Mark memory dimension
        if has_memory:
            ax1.axhline(y=n_task - 0.5, color='green', linestyle='--', linewidth=1, alpha=0.7)
            ax1.axvline(x=n_task - 0.5, color='green', linestyle='--', linewidth=1, alpha=0.7)

        # Columns 2 to n_eigen+1: Top eigenvalue outer products
        for k in range(n_eigen):
            ax = fig.add_subplot(n_outputs, n_cols, row * n_cols + 2 + k)
            if k < len(eigenvalues_sorted):
                lam = eigenvalues_sorted[k]
                v = eigenvectors_sorted[:, k:k+1]  # Column vector
                outer = lam * (v @ v.T)

                vmax_out = max(abs(outer.min()), abs(outer.max()))
                if vmax_out == 0:
                    vmax_out = 1
                im = ax.imshow(outer, cmap='RdBu_r', vmin=-vmax_out, vmax=vmax_out, aspect='equal')

                # Show values in cells
                for ii in range(outer.shape[0]):
                    for jj in range(outer.shape[1]):
                        val = outer[ii, jj]
                        color = 'white' if abs(val) > vmax_out * 0.5 else 'black'
                        txt = f'{val:.1f}' if abs(val) < 10 else f'{val:.0f}'
                        ax.text(jj, ii, txt, ha='center', va='center', fontsize=val_fontsize, color=color)

                # Mark if negative
                sign_str = "(NEG)" if lam < 0 else ""
                ax.set_title(f'λ_{k+1}={lam:.2f} {sign_str}', fontsize=10,
                            color='red' if lam < 0 else 'black')

                # Colorbar below
                divider = make_axes_locatable(ax)
                cax = divider.append_axes("bottom", size="5%", pad=0.3)
                plt.colorbar(im, cax=cax, orientation='horizontal')

                # Mark memory dimension
                if has_memory:
                    ax.axhline(y=n_task - 0.5, color='green', linestyle='--', linewidth=1, alpha=0.7)
                    ax.axvline(x=n_task - 0.5, color='green', linestyle='--', linewidth=1, alpha=0.7)
            else:
                ax.set_visible(False)

        # Last column: Eigenvalue spectrum (line plot, linear scale)
        ax_eig = fig.add_subplot(n_outputs, n_cols, row * n_cols + n_cols)
        eig_original_order = eigenvalues  # Already sorted ascending from eigh
        ax_eig.axhline(y=0, color='black', linestyle='-', linewidth=0.5)
        ax_eig.plot(range(len(eig_original_order)), eig_original_order, 'o-', color='steelblue', markersize=5)

        # Mark negative eigenvalues with red
        neg_idx = np.where(eig_original_order < 0)[0]
        if len(neg_idx) > 0:
            ax_eig.scatter(neg_idx, eig_original_order[neg_idx], color='red', s=50, zorder=5, label='Negative')
        pos_idx = np.where(eig_original_order >= 0)[0]
        if len(pos_idx) > 0:
            ax_eig.scatter(pos_idx, eig_original_order[pos_idx], color='blue', s=50, zorder=5, label='Positive')

        ax_eig.set_title('Eigenvalues', fontsize=10)
        ax_eig.set_xlabel('Index')
        ax_eig.set_ylabel('λ')
        ax_eig.grid(True, alpha=0.3)
        if len(neg_idx) > 0 or len(pos_idx) > 0:
            ax_eig.legend(fontsize=8, loc='best')

    if title is None:
        title = f'Layer {layer_idx+1} Quadratic Forms (M): bilinear_i = h^T M[i] h'

    mem_str = f" (green = memory boundary)" if has_memory else ""
    plt.suptitle(f'{title}{mem_str}', fontsize=14)
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')

    return fig


def plot_eigenvalue_spectra(
    model_data: Dict,
    layer_idx: int = 0,
    title: str = None,
    save_path: Optional[Path] = None,
) -> plt.Figure:
    """
    Plot eigenvalue spectra for all M matrices in a layer.

    Args:
        model_data: dict from load_model_weights()
        layer_idx: which layer (0-indexed)
        title: figure title
        save_path: optional path to save figure

    Returns:
        matplotlib Figure
    """
    L, D, _ = model_data['weights'][layer_idx]
    config = model_data['config']
    n_task = config['n_task']
    n_model = config['n_model']

    M = compute_quadratic_forms(L, D)

    fig, ax = plt.subplots(figsize=(12, 6))

    for i in range(n_model):
        M_i = M[i].numpy()
        eigenvalues, _ = np.linalg.eigh(M_i)
        label = f'out {i}' if i < n_task else f'mem {i - n_task}'
        linestyle = '-' if i < n_task else '--'
        ax.plot(range(len(eigenvalues)), eigenvalues, marker='o', label=label,
                alpha=0.7, markersize=4, linestyle=linestyle)

    ax.axhline(y=0, color='black', linestyle='-', linewidth=0.5)
    ax.set_xlabel('Eigenvalue index (sorted ascending)')
    ax.set_ylabel('λ')

    if title is None:
        title = f'Layer {layer_idx+1} Eigenvalue Spectra'
    ax.set_title(title)
    ax.legend(bbox_to_anchor=(1.02, 1), loc='upper left', fontsize=8)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')

    return fig


# =============================================================================
# FULL MODEL VISUALIZATION
# =============================================================================

def visualize_model(
    checkpoint_path: Union[str, Path],
    output_dir: Optional[Path] = None,
    show_plots: bool = True,
    max_outputs_per_fig: int = 6,
) -> Dict:
    """
    Generate full visualization for a model checkpoint.

    Creates:
    - Weight matrices plot (L, D for all layers)
    - M matrix + eigendecomposition plots for each layer
    - Eigenvalue spectra summary for each layer

    Args:
        checkpoint_path: path to checkpoint file
        output_dir: directory to save figures (optional)
        show_plots: whether to display plots
        max_outputs_per_fig: max outputs per M matrix figure

    Returns:
        dict with model_data and figure handles
    """
    checkpoint_path = Path(checkpoint_path)

    # Load model
    model_data = load_model_weights(checkpoint_path)
    config = model_data['config']
    num_layers = config['num_layers']
    n_model = config['n_model']

    print(f"Model: {checkpoint_path.name}")
    print(f"  Layers: {num_layers}")
    print(f"  n_task: {config['n_task']}, n_model: {config['n_model']}")
    print(f"  Memory: {config['n_memory']} dimensions" if config['has_memory'] else "  No memory")
    print(f"  Rank: {config['rank']}")
    if model_data['accuracy']:
        print(f"  Accuracy: {model_data['accuracy']*100:.1f}%")
    if model_data['sparsity']:
        sp = model_data['sparsity']
        if isinstance(sp, (int, float)):
            print(f"  Sparsity: {sp*100:.1f}%")
        elif isinstance(sp, dict):
            print(f"  Sparsity: {sp}")

    # Setup output directory
    if output_dir:
        output_dir = Path(output_dir)
        output_dir.mkdir(exist_ok=True, parents=True)

    figures = {}

    # 1. Weight matrices
    title = f"{num_layers}-Layer Model (n={config['n_task']}, rank={config['rank']})"
    fig_weights = plot_weight_matrices(
        model_data,
        title=title,
        save_path=output_dir / 'weights.png' if output_dir else None,
    )
    figures['weights'] = fig_weights
    if show_plots:
        plt.show()

    # 2. M matrices with eigendecomposition for each layer
    for layer_idx in range(num_layers):
        # Split into multiple figures if too many outputs
        all_outputs = list(range(n_model))

        for fig_idx, start in enumerate(range(0, n_model, max_outputs_per_fig)):
            outputs = all_outputs[start:start + max_outputs_per_fig]

            suffix = f"_{fig_idx}" if n_model > max_outputs_per_fig else ""

            fig_M = plot_M_matrices_with_eigen(
                model_data,
                layer_idx=layer_idx,
                outputs_to_show=outputs,
                title=f'Layer {layer_idx+1} M Matrices (outputs {outputs[0]}-{outputs[-1]})',
                save_path=output_dir / f'M{layer_idx+1}_eigen{suffix}.png' if output_dir else None,
            )
            figures[f'M{layer_idx+1}_eigen{suffix}'] = fig_M
            if show_plots:
                plt.show()

        # 3. Eigenvalue spectra summary
        fig_spectra = plot_eigenvalue_spectra(
            model_data,
            layer_idx=layer_idx,
            save_path=output_dir / f'M{layer_idx+1}_spectra.png' if output_dir else None,
        )
        figures[f'M{layer_idx+1}_spectra'] = fig_spectra
        if show_plots:
            plt.show()

    return {
        'model_data': model_data,
        'figures': figures,
    }


# =============================================================================
# MAIN: TEST WITH 1-4 LAYER MODELS
# =============================================================================

if __name__ == '__main__':
    import sys

    PROJECT_ROOT = Path(__file__).parent.parent
    checkpoint_dir = PROJECT_ROOT / 'checkpoints'
    images_dir = PROJECT_ROOT / 'images' / 'visualize_test'
    images_dir.mkdir(exist_ok=True, parents=True)

    # Test checkpoints: 1-4 layers, with and without memory
    test_checkpoints = [
        # 1-layer (no memory)
        ('1layer_n5_r5_seed9.pt', '1layer_n5'),
        # 2-layer without memory
        ('2layer_n10_r10_seed8_sparse.pkl', '2layer_n10_nomem'),
        # 2-layer with memory
        ('2layer_n10_memory_seed3_sparse.pkl', '2layer_n10_mem1'),
        # 3-layer with memory
        ('3layer_n10_memory5_seed2.pkl', '3layer_n10_mem5'),
        # 4-layer with memory
        ('4layer_n10_memory10_seed4.pkl', '4layer_n10_mem10'),
    ]

    for ckpt_name, test_name in test_checkpoints:
        ckpt_path = checkpoint_dir / ckpt_name
        if not ckpt_path.exists():
            print(f"\nSkipping {ckpt_name} (not found)")
            continue

        print(f"\n{'='*60}")
        print(f"Testing: {ckpt_name}")
        print('='*60)

        test_output_dir = images_dir / test_name

        try:
            result = visualize_model(
                ckpt_path,
                output_dir=test_output_dir,
                show_plots=False,  # Don't show in test mode
                max_outputs_per_fig=6,
            )
            print(f"  Saved to: {test_output_dir}")
            plt.close('all')
        except Exception as e:
            print(f"  ERROR: {e}")
            import traceback
            traceback.print_exc()

    print(f"\nAll tests complete. Output in: {images_dir}")
