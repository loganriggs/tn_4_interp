"""
Activation/Logit Cosine Similarity computation for compression analysis.

This module provides caching wrappers around the core similarity functions.
"""
import numpy as np
from pathlib import Path

from ..core.models import load_sweep_results
from ..core.dataset import create_full_dataset
from ..core.similarity import (
    compute_all_logits,
    pairwise_cosine_similarity,
    compute_activation_similarity,
    compute_act_similarity_matrix,
)

# Re-export core functions
__all__ = [
    'create_full_dataset',
    'compute_all_logits',
    'pairwise_cosine_similarity',
    'compute_activation_similarity',
    'compute_act_similarity_matrix',
    'load_or_compute_act_similarity',
]


def load_or_compute_act_similarity(
    sweep_path: str | Path,
    cache_path: str | Path | None = None,
    device=None,
    force_recompute: bool = False
) -> np.ndarray:
    """
    Load activation similarity matrix from cache, or compute and cache it.

    Args:
        sweep_path: Path to sweep results pickle file
        cache_path: Path to cache .npy file (default: comp_diagrams/act_similarity_matrix.npy)
        device: Device to use for computation
        force_recompute: If True, recompute even if cache exists

    Returns:
        (P, P) numpy array of pairwise logit cosine similarities
    """
    sweep_path = Path(sweep_path)

    if cache_path is None:
        cache_path = sweep_path.parent / 'act_similarity_matrix.npy'
    else:
        cache_path = Path(cache_path)

    # Try to load from cache
    if not force_recompute and cache_path.exists():
        print(f"Loading activation similarity from cache: {cache_path}")
        return np.load(cache_path)

    # Compute
    print("Computing activation similarity matrix...")
    models_state, val_acc, P = load_sweep_results(sweep_path)
    sim_mat = compute_act_similarity_matrix(models_state, P, device)

    # Save to cache
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    np.save(cache_path, sim_mat)
    print(f"Saved activation similarity to cache: {cache_path}")

    return sim_mat


if __name__ == "__main__":
    # Example usage
    import os
    os.chdir(Path(__file__).parent)

    sweep_path = "comp_diagrams/sweep_results_0401.pkl"
    sim_mat = load_or_compute_act_similarity(sweep_path)
    print(f"Activation similarity matrix shape: {sim_mat.shape}")
    print(f"Similarity range: [{sim_mat.min():.4f}, {sim_mat.max():.4f}]")
