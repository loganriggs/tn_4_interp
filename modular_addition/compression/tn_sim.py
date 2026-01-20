"""
Tensor Network Similarity computation for compression analysis.

This module provides caching wrappers around the core similarity functions.
"""
import numpy as np
from pathlib import Path

from ..core.models import load_sweep_results
from ..core.similarity import (
    symmetric_inner,
    symmetric_similarity,
    compute_tn_similarity_matrix,
    compute_tn_inner_product_matrix,
)

# Re-export core functions
__all__ = [
    'symmetric_inner',
    'symmetric_similarity',
    'compute_tn_similarity_matrix',
    'compute_tn_inner_product_matrix',
    'load_or_compute_tn_similarity',
]


def load_or_compute_tn_similarity(
    sweep_path: str | Path,
    cache_path: str | Path | None = None,
    device=None,
    force_recompute: bool = False
) -> np.ndarray:
    """
    Load TN similarity matrix from cache, or compute and cache it.

    Args:
        sweep_path: Path to sweep results pickle file
        cache_path: Path to cache .npy file (default: comp_diagrams/tn_similarity_matrix.npy)
        device: Device to use for computation
        force_recompute: If True, recompute even if cache exists

    Returns:
        (P, P) numpy array of pairwise TN similarities
    """
    sweep_path = Path(sweep_path)

    if cache_path is None:
        cache_path = sweep_path.parent / 'tn_similarity_matrix.npy'
    else:
        cache_path = Path(cache_path)

    # Try to load from cache
    if not force_recompute and cache_path.exists():
        print(f"Loading TN similarity from cache: {cache_path}")
        return np.load(cache_path)

    # Compute
    print("Computing TN similarity matrix...")
    models_state, val_acc, P = load_sweep_results(sweep_path)
    sim_mat = compute_tn_similarity_matrix(models_state, P, device)

    # Save to cache
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    np.save(cache_path, sim_mat)
    print(f"Saved TN similarity to cache: {cache_path}")

    return sim_mat


if __name__ == "__main__":
    # Example usage
    import os
    os.chdir(Path(__file__).parent)

    sweep_path = "comp_diagrams/sweep_results_0401.pkl"
    sim_mat = load_or_compute_tn_similarity(sweep_path)
    print(f"TN similarity matrix shape: {sim_mat.shape}")
    print(f"Similarity range: [{sim_mat.min():.4f}, {sim_mat.max():.4f}]")
