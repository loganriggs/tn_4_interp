"""
Tensor Network Similarity computation for compression analysis.

Uses the symmetric inner product that matches the symmetrization used in interaction matrices.
"""
import torch
import numpy as np
from einops import einsum
from pathlib import Path
from tqdm import tqdm

try:
    from .models import Model, load_sweep_results, load_model_for_dim
except ImportError:
    from models import Model, load_sweep_results, load_model_for_dim


def symmetric_inner(model1: Model, model2: Model) -> torch.Tensor:
    """
    Compute symmetric inner product between two models.

    This inner product accounts for the symmetry in the bilinear layer
    (left*right vs right*left give same contribution).

    Args:
        model1: First model
        model2: Second model

    Returns:
        Scalar tensor with the inner product value
    """
    ll = einsum(model1.w_l, model2.w_l, "hid1 i, hid2 i -> hid1 hid2")
    rr = einsum(model1.w_r, model2.w_r, "hid1 i, hid2 i -> hid1 hid2")

    lr = einsum(model1.w_l, model2.w_r, "hid1 i, hid2 i -> hid1 hid2")
    rl = einsum(model1.w_r, model2.w_l, "hid1 i, hid2 i -> hid1 hid2")

    core = 0.5 * ((ll * rr) + (lr * rl))

    dd = einsum(model1.w_p, model2.w_p, "o hid1, o hid2 -> hid1 hid2")
    hid = einsum(core, dd, "hid1 hid2, hid1 hid3 -> hid2 hid3")
    return torch.trace(hid)


def symmetric_similarity(model1: Model, model2: Model) -> torch.Tensor:
    """
    Compute symmetric cosine similarity between two models.

    Args:
        model1: First model
        model2: Second model

    Returns:
        Scalar tensor with similarity value in [0, 1]
    """
    if model1 is model2:
        return torch.tensor(1.0)
    inner = symmetric_inner(model1, model2)
    norm1 = torch.sqrt(symmetric_inner(model1, model1))
    norm2 = torch.sqrt(symmetric_inner(model2, model2))
    return inner / (norm1 * norm2)


def compute_tn_similarity_matrix(models_state: dict, P: int, device: torch.device | None = None) -> np.ndarray:
    """
    Compute the full pairwise TN similarity matrix for all bottleneck dimensions.

    Args:
        models_state: Dict mapping d_hidden -> model state dict
        P: Input dimension
        device: Device to use for computation

    Returns:
        (P, P) numpy array where entry [i, j] is similarity between
        models with d_hidden=i+1 and d_hidden=j+1
    """
    if device is None:
        device = torch.device('cpu')

    torch.set_grad_enabled(False)

    # Pre-load all models
    models_cache = {}
    for d in tqdm(range(1, P + 1), desc="Loading models"):
        models_cache[d] = load_model_for_dim(d, models_state, P, device)

    # Compute pairwise similarities
    sim_mat = np.zeros((P, P), dtype=float)
    for i, di in enumerate(tqdm(range(1, P + 1), desc="Computing TN similarity")):
        for j, dj in enumerate(range(1, P + 1)):
            if j < i:
                # Matrix is symmetric
                sim_mat[i, j] = sim_mat[j, i]
            else:
                s = float(symmetric_similarity(models_cache[di], models_cache[dj]))
                sim_mat[i, j] = s

    return sim_mat


def load_or_compute_tn_similarity(
    sweep_path: str | Path,
    cache_path: str | Path | None = None,
    device: torch.device | None = None,
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
