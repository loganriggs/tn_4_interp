"""
JS Divergence computation based on frequency distributions from eigenvector FFT analysis.

This module provides caching wrappers and the full pipeline for JS divergence computation:
models -> interaction matrices -> eigendecomposition -> FFT -> frequency heatmap -> JS divergence
"""
import numpy as np
from pathlib import Path

from ..core.models import load_sweep_results
from ..core.interaction import compute_interaction_matrices_from_state
from ..core.frequency import (
    compute_eigendecomposition,
    compute_eigen_data,
    compute_eigenvector_fft,
    compute_frequency_distribution,
    compute_frequency_heatmap,
    compute_all_frequency_heatmaps,
    entropy_effective_rank,
    ratio_effective_rank,
    cumulative_explained_variance,
    components_for_variance_threshold,
)
from ..core.metrics import (
    JS_divergence,
    compute_average_js_divergence,
    compute_js_divergence_matrix as _compute_js_divergence_matrix,
)

# Re-export core functions
__all__ = [
    # Interaction matrices (kept for backwards compatibility)
    'compute_interaction_matrices',
    # Eigendecomposition
    'compute_eigendecomposition',
    'compute_eigen_data',
    # FFT and frequency
    'compute_eigenvector_fft',
    'compute_frequency_distribution',
    'compute_frequency_heatmap',
    'compute_all_frequency_heatmaps',
    # JS divergence
    'JS_divergence',
    'compute_average_js_divergence',
    'compute_js_divergence_matrix',
    'load_or_compute_js_divergence',
    # Effective rank utilities
    'analyze_eigenvector_fft',
    'entropy_effective_rank',
    'ratio_effective_rank',
    'cumulative_explained_variance',
    'components_for_variance_threshold',
]


# Backwards compatibility alias
def compute_interaction_matrices(models_state: dict, P: int, device=None) -> dict:
    """Compute interaction matrices for all models. Wrapper for backwards compatibility."""
    return compute_interaction_matrices_from_state(models_state, P, device)


def analyze_eigenvector_fft(evec: np.ndarray, P: int) -> dict:
    """Alias for compute_eigenvector_fft for backwards compatibility."""
    return compute_eigenvector_fft(evec, P)


def compute_js_divergence_matrix(
    models_state: dict,
    P: int,
    n_evecs: int = 4,
    device=None
) -> np.ndarray:
    """
    Compute the full pairwise JS divergence matrix between frequency distributions.

    Pipeline: models -> int_mats -> eigen -> frequency heatmap -> JS divergence

    Args:
        models_state: Dict mapping d_hidden -> model state dict
        P: Input dimension
        n_evecs: Number of eigenvectors for frequency analysis
        device: Device to use for computation

    Returns:
        (P, P) numpy array where entry [i, j] is summed JS divergence
        across all remainders between models i+1 and j+1
    """
    # Compute intermediate representations
    int_mats = compute_interaction_matrices_from_state(models_state, P, device)
    eigen_data = compute_eigen_data(int_mats, P)
    heatmaps = compute_all_frequency_heatmaps(eigen_data, P, n_evecs)

    # Compute pairwise JS divergence
    return _compute_js_divergence_matrix(heatmaps, P)


def load_or_compute_js_divergence(
    sweep_path: str | Path,
    cache_path: str | Path | None = None,
    n_evecs: int = 4,
    device=None,
    force_recompute: bool = False
) -> np.ndarray:
    """
    Load JS divergence matrix from cache, or compute and cache it.

    Args:
        sweep_path: Path to sweep results pickle file
        cache_path: Path to cache .npy file (default: comp_diagrams/js_divergence_matrix.npy)
        n_evecs: Number of eigenvectors for frequency analysis
        device: Device to use for computation
        force_recompute: If True, recompute even if cache exists

    Returns:
        (P, P) numpy array of pairwise JS divergences
    """
    sweep_path = Path(sweep_path)

    if cache_path is None:
        cache_path = sweep_path.parent / 'js_divergence_matrix.npy'
    else:
        cache_path = Path(cache_path)

    # Try to load from cache
    if not force_recompute and cache_path.exists():
        print(f"Loading JS divergence from cache: {cache_path}")
        return np.load(cache_path)

    # Compute
    print("Computing JS divergence matrix...")
    models_state, val_acc, P = load_sweep_results(sweep_path)
    js_matrix = compute_js_divergence_matrix(models_state, P, n_evecs, device)

    # Save to cache
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    np.save(cache_path, js_matrix)
    print(f"Saved JS divergence to cache: {cache_path}")

    return js_matrix


if __name__ == "__main__":
    # Example usage
    import os
    os.chdir(Path(__file__).parent)

    sweep_path = "comp_diagrams/sweep_results_0401.pkl"
    js_matrix = load_or_compute_js_divergence(sweep_path)
    print(f"JS divergence matrix shape: {js_matrix.shape}")
    print(f"JS divergence range: [{js_matrix.min():.4f}, {js_matrix.max():.4f}]")
