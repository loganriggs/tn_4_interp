"""
Symmetrized interaction matrix computation for bilinear models.

This module provides:
- Computing the full interaction tensor from model weights
- Symmetrization of interaction matrices
- Utilities for working with the CP decomposition structure

The interaction matrix B[p, i, j] represents the contribution to output p
from the interaction between input positions i and j:
    B[p, i, j] = Σ_h w_l[h, i] * w_r[h, j] * w_p[p, h]
"""
import torch
import numpy as np
from einops import einsum
from pathlib import Path
from tqdm import tqdm

from .models import Model, init_model, get_device


def compute_interaction_matrix(model: Model) -> np.ndarray:
    """
    Compute the symmetrized interaction matrix for a single model.

    The interaction matrix combines w_l, w_r, and w_p into a single tensor
    that represents the full bilinear transformation. It is then symmetrized
    to account for the equivalence of (a, b) and (b, a) contributions.

    Args:
        model: Model instance with w_l, w_r, w_p attributes

    Returns:
        (P, 2P, 2P) numpy array where P is the output dimension.
        B[p, i, j] gives the interaction weight between positions i and j
        for output class p.
    """
    w_l = model.w_l.detach()  # (d_hidden, 2*P)
    w_r = model.w_r.detach()  # (d_hidden, 2*P)
    w_p = model.w_p.detach()  # (P, d_hidden)

    # Compute interaction matrices via CP decomposition
    # B[p, i, j] = Σ_h w_l[h, i] * w_r[h, j] * w_p[p, h]
    b = einsum(w_l, w_r, w_p, "hid in1, hid in2, p hid -> p in1 in2").cpu().numpy()

    # Symmetrize: B_sym = (B + B^T) / 2
    return 0.5 * (b + b.transpose(0, 2, 1))


def compute_interaction_matrix_stack(model: Model) -> np.ndarray:
    """
    Alias for compute_interaction_matrix for clarity.

    Returns the stack of P interaction matrices, one per output class (remainder).
    """
    return compute_interaction_matrix(model)


def compute_interaction_matrices_from_state(
    models_state: dict,
    P: int,
    device: torch.device | None = None,
    show_progress: bool = True
) -> dict:
    """
    Compute interaction matrices for all models in a sweep.

    Args:
        models_state: Dict mapping d_hidden -> model state dict
        P: Input dimension
        device: Device to use for computation
        show_progress: Whether to show progress bar

    Returns:
        Dict mapping d_hidden -> (P, 2P, 2P) numpy array of interaction matrices
    """
    if device is None:
        device = get_device()

    int_mats = {}
    iterator = models_state.items()
    if show_progress:
        iterator = tqdm(iterator, desc="Computing interaction matrices")

    for d_hidden, model_state in iterator:
        model = init_model(p=P, d_hidden=d_hidden).to(device)
        model.load_state_dict(model_state)
        model.eval()
        int_mats[d_hidden] = compute_interaction_matrix(model)

    return int_mats


def extract_remainder_matrix(int_mat_stack: np.ndarray, remainder: int) -> np.ndarray:
    """
    Extract the interaction matrix for a specific remainder/output class.

    Args:
        int_mat_stack: (P, 2P, 2P) stack of interaction matrices
        remainder: The remainder/output class index

    Returns:
        (2P, 2P) symmetric interaction matrix for the given remainder
    """
    return int_mat_stack[remainder]


def compute_effective_rank(int_mat: np.ndarray, method: str = "entropy") -> float:
    """
    Compute effective rank of an interaction matrix.

    Args:
        int_mat: (2P, 2P) symmetric matrix
        method: "entropy" for entropy-based rank, "ratio" for nuclear/spectral ratio

    Returns:
        Effective rank value
    """
    eigenvalues = np.linalg.eigvalsh(int_mat)
    eigenvalues = np.abs(eigenvalues)

    if method == "entropy":
        # Entropy-based: exp(entropy of normalized |λ|²)
        evals_sq = eigenvalues ** 2
        total = evals_sq.sum()
        if total < 1e-12:
            return 0.0
        p = evals_sq / total
        p = p[p > 1e-12]
        entropy = -np.sum(p * np.log(p))
        return np.exp(entropy)
    elif method == "ratio":
        # Ratio-based: sum(|λ|) / max(|λ|)
        max_eval = eigenvalues.max()
        if max_eval < 1e-12:
            return 0.0
        return eigenvalues.sum() / max_eval
    else:
        raise ValueError(f"Unknown method: {method}")
