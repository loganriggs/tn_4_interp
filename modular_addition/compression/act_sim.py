"""
Activation/Logit Cosine Similarity computation for compression analysis.

Compares models by computing the cosine similarity between their output logits
across the entire enumerable dataset (all P×P input pairs).
"""
import torch
import numpy as np
from pathlib import Path
from tqdm import tqdm

try:
    from .models import Model, init_model, load_sweep_results, get_device
except ImportError:
    from models import Model, init_model, load_sweep_results, get_device


def create_full_dataset(P: int, device: torch.device | None = None) -> torch.Tensor:
    """
    Create all P^2 input pairs as one-hot encoded vectors.

    Args:
        P: Input dimension (number of classes)
        device: Device to create tensor on

    Returns:
        Tensor of shape (P^2, 2*P) with one-hot encoded inputs
    """
    if device is None:
        device = torch.device('cpu')

    dataset = torch.zeros(P * P, 2 * P, device=device)
    for a in range(P):
        for b in range(P):
            idx = a * P + b
            dataset[idx, a] = 1.0  # one-hot for a
            dataset[idx, P + b] = 1.0  # one-hot for b
    return dataset


def compute_all_logits(model: Model, dataset: torch.Tensor) -> torch.Tensor:
    """
    Compute logits (before softmax) for all inputs in the dataset.

    Args:
        model: Model to evaluate
        dataset: Input tensor of shape (N, 2*P)

    Returns:
        Tensor of shape (N, P) with logits
    """
    model.eval()
    with torch.no_grad():
        logits = model(dataset)
    return logits


def pairwise_cosine_similarity(logits1: torch.Tensor, logits2: torch.Tensor) -> float:
    """
    Compute average cosine similarity between two sets of logits.

    Args:
        logits1: First logits tensor of shape (N, P)
        logits2: Second logits tensor of shape (N, P)

    Returns:
        Average cosine similarity across all samples
    """
    # Normalize each row
    norm1 = logits1 / (logits1.norm(dim=1, keepdim=True) + 1e-8)
    norm2 = logits2 / (logits2.norm(dim=1, keepdim=True) + 1e-8)

    # Compute cosine similarity for each sample
    cos_sim = (norm1 * norm2).sum(dim=1)  # (N,)

    return cos_sim.mean().item()


def compute_act_similarity_matrix(
    models_state: dict,
    P: int,
    device: torch.device | None = None
) -> np.ndarray:
    """
    Compute the full pairwise logit cosine similarity matrix.

    Args:
        models_state: Dict mapping d_hidden -> model state dict
        P: Input dimension
        device: Device to use for computation

    Returns:
        (P, P) numpy array where entry [i, j] is logit cosine similarity
        between models with d_hidden=i+1 and d_hidden=j+1
    """
    if device is None:
        device = get_device()

    # Create dataset
    dataset = create_full_dataset(P, device)

    # Compute logits for all models
    all_logits = {}
    for d_hidden in tqdm(range(1, P + 1), desc="Computing logits"):
        model = init_model(P, d_hidden).to(device)
        model.load_state_dict(models_state[d_hidden])
        all_logits[d_hidden] = compute_all_logits(model, dataset)

    # Compute pairwise similarities
    sim_mat = np.zeros((P, P), dtype=float)
    for i, di in enumerate(tqdm(range(1, P + 1), desc="Computing logit similarity")):
        for j, dj in enumerate(range(1, P + 1)):
            if j < i:
                # Matrix is symmetric
                sim_mat[i, j] = sim_mat[j, i]
            else:
                sim = pairwise_cosine_similarity(all_logits[di], all_logits[dj])
                sim_mat[i, j] = sim

    return sim_mat


def load_or_compute_act_similarity(
    sweep_path: str | Path,
    cache_path: str | Path | None = None,
    device: torch.device | None = None,
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
