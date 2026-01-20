"""
Similarity metrics for comparing bilinear models.

This module provides:
- TN (tensor network) inner product and similarity
- Activation/logit-based cosine similarity
- Utilities for computing pairwise similarity matrices

Two main perspectives on model similarity:
1. Weight-space (TN): Uses the symmetric structure of the bilinear layer
2. Output-space (activation): Compares model outputs on all inputs
"""
import torch
import numpy as np
from einops import einsum
from pathlib import Path
from tqdm import tqdm

from .models import Model, init_model, load_model_for_dim, get_device
from .dataset import create_full_dataset


# =============================================================================
# TN (Tensor Network) Similarity - Weight Space
# =============================================================================

def symmetric_inner(model1: Model, model2: Model) -> torch.Tensor:
    """
    Compute symmetric inner product between two models.

    This inner product accounts for the symmetry in the bilinear layer
    (left*right vs right*left give the same contribution). The formula is:

    ⟨M1|M2⟩ = Tr(core @ W_p1 @ W_p2.T)

    where core = 0.5 * ((W_l1.T @ W_l2) * (W_r1.T @ W_r2) +
                        (W_l1.T @ W_r2) * (W_r1.T @ W_l2))

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

    similarity = ⟨M1|M2⟩ / (||M1|| * ||M2||)

    Args:
        model1: First model
        model2: Second model

    Returns:
        Scalar tensor with similarity value in [0, 1] for similar models
    """
    if model1 is model2:
        return torch.tensor(1.0)
    inner = symmetric_inner(model1, model2)
    norm1 = torch.sqrt(symmetric_inner(model1, model1))
    norm2 = torch.sqrt(symmetric_inner(model2, model2))
    return inner / (norm1 * norm2)


def compute_tn_inner_product_matrix(
    models_state: dict,
    P: int,
    device: torch.device | None = None,
    show_progress: bool = True
) -> np.ndarray:
    """
    Compute the full TN inner product matrix for all models.

    Args:
        models_state: Dict mapping d_hidden -> model state dict
        P: Input dimension
        device: Device to use for computation
        show_progress: Whether to show progress bar

    Returns:
        (P, P) numpy array where entry [i, j] = ⟨model_{i+1} | model_{j+1}⟩
    """
    if device is None:
        device = torch.device('cpu')

    torch.set_grad_enabled(False)

    # Pre-load all models
    models_cache = {}
    iterator = range(1, P + 1)
    if show_progress:
        iterator = tqdm(iterator, desc="Loading models")
    for d in iterator:
        models_cache[d] = load_model_for_dim(d, models_state, P, device)

    # Compute pairwise inner products
    inner_mat = np.zeros((P, P), dtype=float)
    iterator = range(1, P + 1)
    if show_progress:
        iterator = tqdm(iterator, desc="Computing TN inner products")

    for i, di in enumerate(iterator):
        for j, dj in enumerate(range(1, P + 1)):
            if j < i:
                # Matrix is symmetric
                inner_mat[i, j] = inner_mat[j, i]
            else:
                inner_mat[i, j] = float(symmetric_inner(models_cache[di], models_cache[dj]))

    return inner_mat


def compute_tn_similarity_matrix(
    models_state: dict,
    P: int,
    device: torch.device | None = None,
    show_progress: bool = True
) -> np.ndarray:
    """
    Compute the full pairwise TN similarity matrix for all models.

    Args:
        models_state: Dict mapping d_hidden -> model state dict
        P: Input dimension
        device: Device to use for computation
        show_progress: Whether to show progress bar

    Returns:
        (P, P) numpy array where entry [i, j] is cosine similarity between
        models with d_hidden=i+1 and d_hidden=j+1
    """
    if device is None:
        device = torch.device('cpu')

    torch.set_grad_enabled(False)

    # Pre-load all models
    models_cache = {}
    iterator = range(1, P + 1)
    if show_progress:
        iterator = tqdm(iterator, desc="Loading models")
    for d in iterator:
        models_cache[d] = load_model_for_dim(d, models_state, P, device)

    # Compute pairwise similarities
    sim_mat = np.zeros((P, P), dtype=float)
    iterator = range(1, P + 1)
    if show_progress:
        iterator = tqdm(iterator, desc="Computing TN similarity")

    for i, di in enumerate(iterator):
        for j, dj in enumerate(range(1, P + 1)):
            if j < i:
                sim_mat[i, j] = sim_mat[j, i]
            else:
                s = float(symmetric_similarity(models_cache[di], models_cache[dj]))
                sim_mat[i, j] = s

    return sim_mat


# =============================================================================
# Activation/Logit Similarity - Output Space
# =============================================================================

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

    For each input sample, computes cosine similarity between the two
    models' logit vectors, then averages across all samples.

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


def compute_activation_similarity(
    model1: Model,
    model2: Model,
    dataset: torch.Tensor
) -> float:
    """
    Compute activation-based cosine similarity between two models.

    Args:
        model1: First model
        model2: Second model
        dataset: Full dataset tensor of shape (N, 2*P)

    Returns:
        Average cosine similarity of logits across all inputs
    """
    logits1 = compute_all_logits(model1, dataset)
    logits2 = compute_all_logits(model2, dataset)
    return pairwise_cosine_similarity(logits1, logits2)


def compute_act_similarity_matrix(
    models_state: dict,
    P: int,
    device: torch.device | None = None,
    show_progress: bool = True
) -> np.ndarray:
    """
    Compute the full pairwise logit cosine similarity matrix.

    Args:
        models_state: Dict mapping d_hidden -> model state dict
        P: Input dimension
        device: Device to use for computation
        show_progress: Whether to show progress bar

    Returns:
        (P, P) numpy array where entry [i, j] is logit cosine similarity
        between models with d_hidden=i+1 and d_hidden=j+1
    """
    if device is None:
        device = get_device()

    # Create dataset once
    dataset = create_full_dataset(P, device)

    # Compute logits for all models
    all_logits = {}
    iterator = range(1, P + 1)
    if show_progress:
        iterator = tqdm(iterator, desc="Computing logits")

    for d_hidden in iterator:
        model = init_model(P, d_hidden).to(device)
        model.load_state_dict(models_state[d_hidden])
        all_logits[d_hidden] = compute_all_logits(model, dataset)

    # Compute pairwise similarities
    sim_mat = np.zeros((P, P), dtype=float)
    iterator = range(1, P + 1)
    if show_progress:
        iterator = tqdm(iterator, desc="Computing logit similarity")

    for i, di in enumerate(iterator):
        for j, dj in enumerate(range(1, P + 1)):
            if j < i:
                sim_mat[i, j] = sim_mat[j, i]
            else:
                sim = pairwise_cosine_similarity(all_logits[di], all_logits[dj])
                sim_mat[i, j] = sim

    return sim_mat
