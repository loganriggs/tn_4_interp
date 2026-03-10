"""
TN-Sim: Tensor Network Similarity for Bilinear Networks

Computes analytic TN inner products and similarities for N-layer bilinear networks
with down projections: B(x) = D @ (Lx ⊙ Rx)

The composition of N bilinear layers gives a (2N+1)-order tensor.
"""

import torch
from typing import List, Tuple, Optional


def bilinear_core(L_a: torch.Tensor, R_a: torch.Tensor,
                  L_b: torch.Tensor, R_b: torch.Tensor,
                  symmetrize: bool = True) -> torch.Tensor:
    """
    Compute the bilinear core matrix for a single layer.

    For bilinear B(x) = D @ (Lx ⊙ Rx), the core captures the (j,k) contraction
    when computing <T_a|T_b>.

    Args:
        L_a, R_a: (d_hidden, d_input) weights from model a
        L_b, R_b: (d_hidden, d_input) weights from model b
        symmetrize: If True, symmetrize for x⊗x contraction

    Returns:
        core: (d_hidden_a, d_hidden_b) matrix
    """
    LL = L_a @ L_b.T  # (d_hidden_a, d_hidden_b)
    RR = R_a @ R_b.T  # (d_hidden_a, d_hidden_b)

    if symmetrize:
        LR = L_a @ R_b.T
        RL = R_a @ L_b.T
        core = 0.5 * (LL * RR + LR * RL)
    else:
        core = LL * RR

    return core


def tn_inner_1layer(L_a: torch.Tensor, R_a: torch.Tensor, D_a: torch.Tensor,
                    L_b: torch.Tensor, R_b: torch.Tensor, D_b: torch.Tensor,
                    symmetrize: bool = True) -> torch.Tensor:
    """
    TN inner product for 1-layer bilinear: B(x) = D @ (Lx ⊙ Rx)

    Tensor: T[i,j,k] = Σ_h D[i,h] L[h,j] R[h,k]

    <T_a|T_b> = (DD * core).sum()

    Args:
        L, R: (d_hidden, d_input)
        D: (d_output, d_hidden)
        symmetrize: Symmetrize core for x⊗x

    Returns:
        Scalar inner product
    """
    DD = D_a.T @ D_b  # (d_hidden, d_hidden)
    core = bilinear_core(L_a, R_a, L_b, R_b, symmetrize=symmetrize)

    return (DD * core).sum()


def tn_inner_2layer(L1_a: torch.Tensor, R1_a: torch.Tensor, D1_a: torch.Tensor,
                    L2_a: torch.Tensor, R2_a: torch.Tensor, D2_a: torch.Tensor,
                    L1_b: torch.Tensor, R1_b: torch.Tensor, D1_b: torch.Tensor,
                    L2_b: torch.Tensor, R2_b: torch.Tensor, D2_b: torch.Tensor,
                    symmetrize: bool = True) -> torch.Tensor:
    """
    TN inner product for 2-layer bilinear: B2(B1(x))

    5th order tensor:
    T[n,j,k,p,q] = Σ_{m,h,h'} D2[n,m] (L2@D1)[m,h] L1[h,j] R1[h,k] (R2@D1)[m,h'] L1[h',p] R1[h',q]

    Efficient computation:
    <T_a|T_b> = (DD2 * (A_a @ C1 @ A_b.T) * (B_a @ C1 @ B_b.T)).sum()

    Args:
        L1, R1: (d_hidden1, d_res) - layer 1 bilinear
        D1: (d_res, d_hidden1) - layer 1 down projection
        L2, R2: (d_hidden2, d_res) - layer 2 bilinear
        D2: (d_res, d_hidden2) - layer 2 down projection
        symmetrize: Symmetrize layer 1 core for x⊗x

    Returns:
        Scalar inner product
    """
    # Layer 1 core (symmetrized for x⊗x)
    C1 = bilinear_core(L1_a, R1_a, L1_b, R1_b, symmetrize=symmetrize)

    # Compositions: how layer 2 sees layer 1's output
    A_a = L2_a @ D1_a  # (d_hidden2, d_hidden1)
    A_b = L2_b @ D1_b
    B_a = R2_a @ D1_a
    B_b = R2_b @ D1_b

    # Layer 2 down projection contraction
    DD2 = D2_a.T @ D2_b  # (d_hidden2, d_hidden2)

    # Full contraction
    term_A = A_a @ C1 @ A_b.T  # (d_hidden2, d_hidden2)
    term_B = B_a @ C1 @ B_b.T  # (d_hidden2, d_hidden2)

    return (DD2 * term_A * term_B).sum()


def tn_inner_nlayer(weights_a: List[Tuple[torch.Tensor, torch.Tensor, torch.Tensor]],
                    weights_b: List[Tuple[torch.Tensor, torch.Tensor, torch.Tensor]],
                    symmetrize: bool = True) -> torch.Tensor:
    """
    TN inner product for N-layer bilinear: BN(...B2(B1(x))...)

    Each layer i has weights (Li, Ri, Di).
    Produces a (2N+1)-order tensor.

    The computation is recursive:
    - Start with layer 1 core C1
    - Each subsequent layer builds a new "effective core" that captures
      all previous layers

    Args:
        weights_a: List of (L, R, D) tuples for model a, from layer 1 to N
        weights_b: List of (L, R, D) tuples for model b
        symmetrize: Symmetrize layer 1 core

    Returns:
        Scalar inner product
    """
    n_layers = len(weights_a)
    assert len(weights_b) == n_layers, "Models must have same number of layers"

    if n_layers == 1:
        L_a, R_a, D_a = weights_a[0]
        L_b, R_b, D_b = weights_b[0]
        return tn_inner_1layer(L_a, R_a, D_a, L_b, R_b, D_b, symmetrize=symmetrize)

    if n_layers == 2:
        L1_a, R1_a, D1_a = weights_a[0]
        L2_a, R2_a, D2_a = weights_a[1]
        L1_b, R1_b, D1_b = weights_b[0]
        L2_b, R2_b, D2_b = weights_b[1]
        return tn_inner_2layer(L1_a, R1_a, D1_a, L2_a, R2_a, D2_a,
                               L1_b, R1_b, D1_b, L2_b, R2_b, D2_b,
                               symmetrize=symmetrize)

    # General N-layer case (N >= 3)
    # Build up from layer 1
    L1_a, R1_a, D1_a = weights_a[0]
    L1_b, R1_b, D1_b = weights_b[0]

    # Layer 1 core
    C = bilinear_core(L1_a, R1_a, L1_b, R1_b, symmetrize=symmetrize)

    # Process layers 2 to N-1, building up the effective core
    for i in range(1, n_layers - 1):
        Li_a, Ri_a, Di_a = weights_a[i]
        Li_b, Ri_b, Di_b = weights_b[i]

        # Compositions
        A_a = Li_a @ Di_a if i == 1 else Li_a @ weights_a[i-1][2]
        A_b = Li_b @ Di_b if i == 1 else Li_b @ weights_b[i-1][2]
        B_a = Ri_a @ Di_a if i == 1 else Ri_a @ weights_a[i-1][2]
        B_b = Ri_b @ Di_b if i == 1 else Ri_b @ weights_b[i-1][2]

        # Wait, this recursion is getting complex. Let me think more carefully.
        # For N layers, we need to track how each layer's output feeds into the next.

    # For now, raise NotImplementedError for N > 2
    # The pattern exists but requires careful bookkeeping
    raise NotImplementedError(
        f"N-layer TN inner product not yet implemented for N={n_layers} > 2. "
        "Use tn_inner_1layer or tn_inner_2layer directly."
    )


def tn_sim(weights_a: List[Tuple[torch.Tensor, torch.Tensor, torch.Tensor]],
           weights_b: List[Tuple[torch.Tensor, torch.Tensor, torch.Tensor]],
           symmetrize: bool = True) -> float:
    """
    Compute TN similarity (normalized inner product).

    TN-sim = <T_a|T_b> / (||T_a|| * ||T_b||)

    Args:
        weights_a: List of (L, R, D) tuples for model a
        weights_b: List of (L, R, D) tuples for model b
        symmetrize: Symmetrize for x⊗x contractions

    Returns:
        Similarity in [0, 1] (or negative if anti-correlated)
    """
    n_layers = len(weights_a)

    if n_layers == 1:
        inner_fn = lambda wa, wb: tn_inner_1layer(
            wa[0][0], wa[0][1], wa[0][2],
            wb[0][0], wb[0][1], wb[0][2],
            symmetrize=symmetrize
        )
    elif n_layers == 2:
        inner_fn = lambda wa, wb: tn_inner_2layer(
            wa[0][0], wa[0][1], wa[0][2], wa[1][0], wa[1][1], wa[1][2],
            wb[0][0], wb[0][1], wb[0][2], wb[1][0], wb[1][1], wb[1][2],
            symmetrize=symmetrize
        )
    else:
        inner_fn = lambda wa, wb: tn_inner_nlayer(wa, wb, symmetrize=symmetrize)

    inner_ab = inner_fn(weights_a, weights_b)
    inner_aa = inner_fn(weights_a, weights_a)
    inner_bb = inner_fn(weights_b, weights_b)

    norm_a = torch.sqrt(inner_aa)
    norm_b = torch.sqrt(inner_bb)

    sim = inner_ab / (norm_a * norm_b + 1e-10)
    return sim.item()


# =============================================================================
# CONVENIENCE FUNCTIONS
# =============================================================================

def tn_sim_1layer(L_a, R_a, D_a, L_b, R_b, D_b, symmetrize=True) -> float:
    """TN-sim for 1-layer bilinear."""
    weights_a = [(L_a, R_a, D_a)]
    weights_b = [(L_b, R_b, D_b)]
    return tn_sim(weights_a, weights_b, symmetrize=symmetrize)


def tn_sim_2layer(L1_a, R1_a, D1_a, L2_a, R2_a, D2_a,
                  L1_b, R1_b, D1_b, L2_b, R2_b, D2_b,
                  symmetrize=True) -> float:
    """TN-sim for 2-layer bilinear."""
    weights_a = [(L1_a, R1_a, D1_a), (L2_a, R2_a, D2_a)]
    weights_b = [(L1_b, R1_b, D1_b), (L2_b, R2_b, D2_b)]
    return tn_sim(weights_a, weights_b, symmetrize=symmetrize)


# =============================================================================
# AUGMENTED INPUT FOR RESIDUAL CONNECTIONS
# =============================================================================
# For residual: output = x + D @ (Lx ⊙ Rx)
#
# Key insight (Ozaru): Augment input x̂ = [x, 1], then:
#   x_i = 1 * x_i = (e_{n+1} · x̂) * (e_i · x̂)
# This is a bilinear in x̂ with L_res selecting the constant, R_res = I.
#
# Concatenate residual + bilinear weights along hidden dim, then use
# the standard TN inner product on augmented weights.
#
# NOTE: L=R=D=I does NOT give the identity — it gives x^2 (elementwise squared).
# The augmented trick is the correct way to tensorify the residual.


def make_augmented_weights_general(
    L: torch.Tensor, R: torch.Tensor, D: torch.Tensor, W_res: torch.Tensor
):
    """
    Convert (L, R, D, W_res) for y = W_res@x + D@(Lx⊙Rx) to augmented weights
    on x̂ = [x, 1] such that y = D_aug @ (L_aug @ x̂ ⊙ R_aug @ x̂).

    This is the general version where the residual W_res can be any matrix
    (not just identity). Useful when embed/head are folded into the weights.

    Residual decomposition: y_i = sum_j W_res[i,j] * 1 * x_j
      L_res[h,:] = e_{n+1} (selects constant 1), for h = 0..n_in-1
      R_res[h,:] = e_h (selects x_h), for h = 0..n_in-1
      D_res = W_res  (n_out × n_in)

    Args:
        L: (rank, n_in) left weights
        R: (rank, n_in) right weights
        D: (n_out, rank) down-projection
        W_res: (n_out, n_in) residual matrix (e.g. W_head @ W_embed)

    Returns:
        L_aug: (n_in + rank, n_in + 1)
        R_aug: (n_in + rank, n_in + 1)
        D_aug: (n_out, n_in + rank)
    """
    n_out, rank = D.shape
    n_in = L.shape[1]
    assert R.shape == (rank, n_in)
    assert W_res.shape == (n_out, n_in)
    device = L.device
    dtype = L.dtype

    # Residual part (rank = n_in hidden units)
    L_res = torch.zeros(n_in, n_in + 1, device=device, dtype=dtype)
    L_res[:, -1] = 1.0  # each row selects x̂_{n+1} = 1
    R_res = torch.zeros(n_in, n_in + 1, device=device, dtype=dtype)
    R_res[:, :n_in] = torch.eye(n_in, device=device, dtype=dtype)  # row h selects x_h
    D_res = W_res  # (n_out, n_in)

    # Bilinear part (pad with zero column for constant dim)
    L_bil = torch.cat([L, torch.zeros(rank, 1, device=device, dtype=dtype)], dim=1)
    R_bil = torch.cat([R, torch.zeros(rank, 1, device=device, dtype=dtype)], dim=1)
    D_bil = D

    # Concatenate along hidden dim
    L_aug = torch.cat([L_res, L_bil], dim=0)  # (n_in + rank, n_in + 1)
    R_aug = torch.cat([R_res, R_bil], dim=0)  # (n_in + rank, n_in + 1)
    D_aug = torch.cat([D_res, D_bil], dim=1)  # (n_out, n_in + rank)

    return L_aug, R_aug, D_aug


def make_augmented_weights(L: torch.Tensor, R: torch.Tensor, D: torch.Tensor):
    """
    Convert (L, R, D) for y = x + D@(Lx⊙Rx) to augmented weights
    on x̂ = [x, 1] such that y = D_aug @ (L_aug @ x̂ ⊙ R_aug @ x̂).

    Residual: x_i = 1 * x_i, represented as:
      L_res[h,:] = e_{n+1} (selects constant 1)
      R_res[h,:] = e_h (selects x_h)
      D_res = I_n

    Bilinear: D@(Lx⊙Rx), pad L,R with zero column:
      L_bil = [L | 0], R_bil = [R | 0], D_bil = D

    Concatenate along hidden dim.

    Args:
        L: (rank, n) left weights
        R: (rank, n) right weights
        D: (n, rank) down-projection (must be square output = input for residual)

    Returns:
        L_aug: (n + rank, n + 1)
        R_aug: (n + rank, n + 1)
        D_aug: (n, n + rank)
    """
    n_out, rank = D.shape
    n_in = L.shape[1]
    assert R.shape == (rank, n_in)
    assert n_out == n_in, "Residual requires d_output == d_input"
    n = n_out
    device = L.device
    dtype = L.dtype

    # Residual part (rank = n)
    L_res = torch.zeros(n, n + 1, device=device, dtype=dtype)
    L_res[:, -1] = 1.0  # each row selects x̂_{n+1} = 1
    R_res = torch.zeros(n, n + 1, device=device, dtype=dtype)
    R_res[:, :n] = torch.eye(n, device=device, dtype=dtype)  # row h selects x_h
    D_res = torch.eye(n, device=device, dtype=dtype)

    # Bilinear part (pad with zero column for constant dim)
    L_bil = torch.cat([L, torch.zeros(rank, 1, device=device, dtype=dtype)], dim=1)
    R_bil = torch.cat([R, torch.zeros(rank, 1, device=device, dtype=dtype)], dim=1)
    D_bil = D

    # Concatenate along hidden dim
    L_aug = torch.cat([L_res, L_bil], dim=0)  # (n + rank, n + 1)
    R_aug = torch.cat([R_res, R_bil], dim=0)  # (n + rank, n + 1)
    D_aug = torch.cat([D_res, D_bil], dim=1)  # (n, n + rank)

    return L_aug, R_aug, D_aug


def tn_inner_1layer_with_residual(
    L_a: torch.Tensor, R_a: torch.Tensor, D_a: torch.Tensor,
    L_b: torch.Tensor, R_b: torch.Tensor, D_b: torch.Tensor,
    symmetrize: bool = True
) -> torch.Tensor:
    """
    TN inner product for 1-layer bilinear WITH residual: y = x + D@(Lx⊙Rx).

    Uses the augmented input trick: x̂ = [x, 1], then the full residual+bilinear
    is a single bilinear on x̂. Standard TN inner product on augmented weights.
    """
    L_aug_a, R_aug_a, D_aug_a = make_augmented_weights(L_a, R_a, D_a)
    L_aug_b, R_aug_b, D_aug_b = make_augmented_weights(L_b, R_b, D_b)
    return tn_inner_1layer(L_aug_a, R_aug_a, D_aug_a, L_aug_b, R_aug_b, D_aug_b,
                           symmetrize=symmetrize)


def tn_sim_1layer_with_residual(
    L_a: torch.Tensor, R_a: torch.Tensor, D_a: torch.Tensor,
    L_b: torch.Tensor, R_b: torch.Tensor, D_b: torch.Tensor,
    symmetrize: bool = True
) -> float:
    """TN-sim for 1-layer bilinear with residual."""
    inner_ab = tn_inner_1layer_with_residual(L_a, R_a, D_a, L_b, R_b, D_b, symmetrize)
    inner_aa = tn_inner_1layer_with_residual(L_a, R_a, D_a, L_a, R_a, D_a, symmetrize)
    inner_bb = tn_inner_1layer_with_residual(L_b, R_b, D_b, L_b, R_b, D_b, symmetrize)

    norm_a = torch.sqrt(inner_aa)
    norm_b = torch.sqrt(inner_bb)

    sim = inner_ab / (norm_a * norm_b + 1e-10)
    return sim.item()


def tn_inner_2layer_with_residual(
    L1_a: torch.Tensor, R1_a: torch.Tensor, D1_a: torch.Tensor,
    L2_a: torch.Tensor, R2_a: torch.Tensor, D2_a: torch.Tensor,
    L1_b: torch.Tensor, R1_b: torch.Tensor, D1_b: torch.Tensor,
    L2_b: torch.Tensor, R2_b: torch.Tensor, D2_b: torch.Tensor,
    symmetrize: bool = True
) -> torch.Tensor:
    """
    TN inner product for 2-layer bilinear WITH residuals.

    Full output: out = x + B1(x) + B2(x + B1(x))

    Uses augmented layer 1 (x̂ = [x, 1]) to correctly represent the residual,
    then decomposes into degree terms. Same-parity cross terms (deg1×deg3,
    deg2×deg4) are small and omitted for efficiency.

    The expansion of B2(x + B1(x)) = B2(h1) where h1 = x + B1(x):
    - h1 is represented by augmented layer 1 weights
    - B2(h1) = D2@(L2·h1 ⊙ R2·h1) is a 2-layer composition with augmented layer 1
    - The layer-2 residual (adding h1) is the augmented layer 1 output itself

    Total = <h1_a|h1_b> + <B2(h1_a)|B2(h1_b)> + cross terms
    """
    n = D1_a.shape[0]

    # Augment layer 1 to include residual: h1 = x + B1(x)
    L1_aug_a, R1_aug_a, D1_aug_a = make_augmented_weights(L1_a, R1_a, D1_a)
    L1_aug_b, R1_aug_b, D1_aug_b = make_augmented_weights(L1_b, R1_b, D1_b)

    # =========================================================================
    # Term 1: <h1_a | h1_b> (augmented 1-layer = residual + B1)
    # =========================================================================
    inner_h1 = tn_inner_1layer(L1_aug_a, R1_aug_a, D1_aug_a,
                                L1_aug_b, R1_aug_b, D1_aug_b, symmetrize)

    # =========================================================================
    # Term 2: <B2(h1_a) | B2(h1_b)> (2-layer composition, augmented layer 1)
    # This is the standard 2-layer TN inner product with augmented L1, R1, D1
    # =========================================================================
    inner_b2b2 = tn_inner_2layer(L1_aug_a, R1_aug_a, D1_aug_a, L2_a, R2_a, D2_a,
                                  L1_aug_b, R1_aug_b, D1_aug_b, L2_b, R2_b, D2_b,
                                  symmetrize)

    # =========================================================================
    # Cross terms: <h1 | B2(h1)> and <B2(h1) | h1>
    # h1 is degree ≤ 2 in x, B2(h1) is degree ≤ 4 in x.
    # The Frobenius cross-term of tensors of different orders is zero
    # (they live in different tensor spaces), so these vanish exactly
    # in the Frobenius metric. We keep them as zero.
    # =========================================================================
    # (Cross terms are zero in tensor Frobenius inner product since the tensors
    # have different numbers of input indices: h1 has 2, B2(h1) has 4.)

    return inner_h1 + inner_b2b2


def tn_sim_2layer_with_residual(
    L1_a: torch.Tensor, R1_a: torch.Tensor, D1_a: torch.Tensor,
    L2_a: torch.Tensor, R2_a: torch.Tensor, D2_a: torch.Tensor,
    L1_b: torch.Tensor, R1_b: torch.Tensor, D1_b: torch.Tensor,
    L2_b: torch.Tensor, R2_b: torch.Tensor, D2_b: torch.Tensor,
    symmetrize: bool = True
) -> float:
    """TN-sim for 2-layer bilinear with residuals."""
    inner_ab = tn_inner_2layer_with_residual(
        L1_a, R1_a, D1_a, L2_a, R2_a, D2_a,
        L1_b, R1_b, D1_b, L2_b, R2_b, D2_b,
        symmetrize
    )
    inner_aa = tn_inner_2layer_with_residual(
        L1_a, R1_a, D1_a, L2_a, R2_a, D2_a,
        L1_a, R1_a, D1_a, L2_a, R2_a, D2_a,
        symmetrize
    )
    inner_bb = tn_inner_2layer_with_residual(
        L1_b, R1_b, D1_b, L2_b, R2_b, D2_b,
        L1_b, R1_b, D1_b, L2_b, R2_b, D2_b,
        symmetrize
    )

    norm_a = torch.sqrt(inner_aa)
    norm_b = torch.sqrt(inner_bb)

    sim = inner_ab / (norm_a * norm_b + 1e-10)
    return sim.item()
