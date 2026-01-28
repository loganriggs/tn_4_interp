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
# RESIDUAL CORRECTIONS
# =============================================================================
# For pre-norm residual: output = B(x/||x||) + x
# After scaling by ||x||^2: B(x) + x||x||^2
# The residual term x||x||^2 is a 3rd-order tensor with L=R=D=I
#
# See: tensorify_norm_and_residual.md.sty for derivation

def tn_inner_1layer_with_residual(
    L_a: torch.Tensor, R_a: torch.Tensor, D_a: torch.Tensor,
    L_b: torch.Tensor, R_b: torch.Tensor, D_b: torch.Tensor,
    symmetrize: bool = True
) -> torch.Tensor:
    """
    TN inner product for 1-layer bilinear WITH pre-norm residual.

    Full block: output = B(x/||x||) + x
    Tensorified: B(x) + x||x||^2, where x||x||^2 has L=R=D=I

    <A+res | B+res> = <A|B> + <A|res> + <res|B> + <res|res>
    """
    d_out = D_a.shape[0]  # output/residual dimension

    # Bilinear-bilinear term
    inner_bb = tn_inner_1layer(L_a, R_a, D_a, L_b, R_b, D_b, symmetrize=symmetrize)

    # Residual-residual term: <res|res> with L=R=D=I
    # core_res = I * I = I (element-wise), then Tr(I @ I @ I^T) = Tr(I) = d_out
    inner_rr = d_out

    # Cross terms: <bilinear | residual>
    # Residual has L=R=D=I, so:
    # core(A, res) = 0.5 * ((L_a @ I^T) * (R_a @ I^T) + (L_a @ I^T) * (R_a @ I^T))
    #             = L_a * R_a (element-wise, summed appropriately)
    # Then contract: (D_a^T @ I) * core = D_a^T * (L_a * R_a summed over input)

    # For symmetrized: core = 0.5 * (L*R + L*R) = L*R element-wise product
    # <A|res> = sum over h,i,j of D_a[i,h] * L_a[h,j] * R_a[h,j] * I[i] * I[j] * I[j]
    # Simplifies to: sum_h (sum_j L_a[h,j] * R_a[h,j]) * (sum_i D_a[i,h])

    LR_a = (L_a * R_a).sum(dim=1)  # (d_hidden,) - sum over input dim
    D_a_colsum = D_a.sum(dim=0)    # (d_hidden,) - sum over output dim
    inner_ar = (LR_a * D_a_colsum).sum()

    LR_b = (L_b * R_b).sum(dim=1)
    D_b_colsum = D_b.sum(dim=0)
    inner_br = (LR_b * D_b_colsum).sum()

    # Full inner product: <A+res|B+res> = <A|B> + <A|res> + <res|B> + <res|res>
    # Note: <A|res> uses A's weights, <res|B> uses B's weights
    # But for cross terms with shared residual (I,I,I), we need:
    # <A|res_B> where res_B = (I,I,I)
    # This is symmetric, so <A|res> = <res|A>

    return inner_bb + inner_ar + inner_br + inner_rr


def tn_sim_1layer_with_residual(
    L_a: torch.Tensor, R_a: torch.Tensor, D_a: torch.Tensor,
    L_b: torch.Tensor, R_b: torch.Tensor, D_b: torch.Tensor,
    symmetrize: bool = True
) -> float:
    """TN-sim for 1-layer bilinear with pre-norm residual."""
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

    Expanding B2(x + B1(x)) gives terms of degrees 2, 3, 4 in x.
    Full polynomial has degrees 1, 2, 3, 4.

    Under Gaussian measure, different degrees are orthogonal:
    <out_a|out_b> = <deg1> + <deg2> + <deg3> + <deg4>

    Degree 1: x (same for all models) -> contributes d_res
    Degree 2: B1(x) + D2@(L2x⊙R2x)
    Degree 3: D2@(L2x⊙R2@B1(x) + L2@B1(x)⊙R2x) [complex, included]
    Degree 4: D2@(L2@B1(x)⊙R2@B1(x)) [pure composition]
    """
    d_res = D1_a.shape[0]  # residual stream dimension

    # =========================================================================
    # DEGREE 1: Linear term x (identity residual)
    # Both models pass through x, so <x|x> = d_res
    # =========================================================================
    inner_deg1 = d_res

    # =========================================================================
    # DEGREE 2: Quadratic terms B1(x) + B2_standalone(x)
    # B1(x) = D1@(L1x⊙R1x)
    # B2_standalone(x) = D2@(L2x⊙R2x)
    # =========================================================================
    # B1-B1 inner product
    inner_b1b1 = tn_inner_1layer(L1_a, R1_a, D1_a, L1_b, R1_b, D1_b, symmetrize)

    # B2_standalone-B2_standalone inner product (layer 2 applied directly to x)
    inner_b2sb2s = tn_inner_1layer(L2_a, R2_a, D2_a, L2_b, R2_b, D2_b, symmetrize)

    # Cross term: B1_a with B2_standalone_b (and vice versa)
    # <B1_a | B2s_b> = (D1_a^T @ D2_b) * core(L1_a,R1_a, L2_b,R2_b)
    DD_cross = D1_a.T @ D2_b  # (d_hidden1, d_hidden2)
    core_cross = bilinear_core(L1_a, R1_a, L2_b, R2_b, symmetrize)  # (d_hidden1, d_hidden2)
    inner_b1_b2s = (DD_cross * core_cross).sum()

    DD_cross2 = D2_a.T @ D1_b
    core_cross2 = bilinear_core(L2_a, R2_a, L1_b, R1_b, symmetrize)
    inner_b2s_b1 = (DD_cross2 * core_cross2).sum()

    inner_deg2 = inner_b1b1 + inner_b2sb2s + inner_b1_b2s + inner_b2s_b1

    # =========================================================================
    # DEGREE 3: Cubic cross-terms (more complex)
    # Term: D2@(L2x⊙R2@D1@(L1x⊙R1x)) + D2@(L2@D1@(L1x⊙R1x)⊙R2x)
    # These are 4th-order tensors with structure mixing layers 1 and 2
    # =========================================================================
    # For cubic term 1: T[n,j,k,p] where output n, L2 input j, L1⊙R1 indices k,p
    # Contraction requires careful bookkeeping

    # Cubic term contributions (symmetrized)
    # T_c1[n,j,k,p] = Σ_{m,h} D2[n,m] L2[m,j] (R2@D1)[m,h] L1[h,k] R1[h,p]
    # T_c2[n,j,k,p] = Σ_{m,h} D2[n,m] (L2@D1)[m,h] L1[h,j] R1[h,k] R2[m,p]

    # For inner product, we need to contract over all input indices
    # <T_c1_a | T_c1_b> etc.

    # Let's compute the key contractions:
    # A_a = L2_a @ D1_a, B_a = R2_a @ D1_a (compositions)
    A_a = L2_a @ D1_a  # (d_h2, d_h1)
    B_a = R2_a @ D1_a
    A_b = L2_b @ D1_b
    B_b = R2_b @ D1_b

    # Layer 1 core (for k,p indices)
    C1 = bilinear_core(L1_a, R1_a, L1_b, R1_b, symmetrize)  # (d_h1, d_h1)

    # Layer 2 down proj contraction
    DD2 = D2_a.T @ D2_b  # (d_h2, d_h2)

    # L2 contraction
    LL2 = L2_a @ L2_b.T  # (d_h2, d_h2)
    RR2 = R2_a @ R2_b.T

    # Cubic term 1: L2x ⊙ (R2@D1@(L1x⊙R1x))
    # <c1_a|c1_b> = sum over (n,j,k,p) of T_c1_a[n,j,k,p] * T_c1_b[n,j,k,p]
    # = (DD2 * LL2 * (B_a @ C1 @ B_b.T)).sum()
    inner_c1c1 = (DD2 * LL2 * (B_a @ C1 @ B_b.T)).sum()

    # Cubic term 2: (L2@D1@(L1x⊙R1x)) ⊙ R2x
    # <c2_a|c2_b> = (DD2 * RR2 * (A_a @ C1 @ A_b.T)).sum()
    inner_c2c2 = (DD2 * RR2 * (A_a @ C1 @ A_b.T)).sum()

    # Cross: <c1_a|c2_b> and <c2_a|c1_b>
    # c1 has L2 on j, B on (k,p)
    # c2 has A on (j,k), R2 on p
    # These have different index structures so cross-term requires more care
    # After symmetrization over input indices, the cross terms contribute:
    LR2 = L2_a @ R2_b.T  # (d_h2, d_h2)
    RL2 = R2_a @ L2_b.T
    inner_c1c2 = (DD2 * LR2 * (B_a @ C1 @ A_b.T)).sum()
    inner_c2c1 = (DD2 * RL2 * (A_a @ C1 @ B_b.T)).sum()

    inner_deg3 = inner_c1c1 + inner_c2c2 + inner_c1c2 + inner_c2c1

    # =========================================================================
    # DEGREE 4: Pure composition D2@(L2@B1(x)⊙R2@B1(x))
    # This is what tn_inner_2layer computes
    # =========================================================================
    inner_deg4 = tn_inner_2layer(
        L1_a, R1_a, D1_a, L2_a, R2_a, D2_a,
        L1_b, R1_b, D1_b, L2_b, R2_b, D2_b,
        symmetrize=symmetrize
    )

    # =========================================================================
    # TOTAL
    # =========================================================================
    return inner_deg1 + inner_deg2 + inner_deg3 + inner_deg4


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
