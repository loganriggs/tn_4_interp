"""
ODT (Orthogonalization, Diagonalization, Truncation) algorithm.

Implements the algorithm from Dooms et al. 2025, adapted from the
hierarchical SVD for tree tensor networks.

The key insight: per-layer SVD misses global low-rank structure.
ODT finds it by computing Gram matrices that account for the full
tree structure, using isometric properties for efficiency.

The Gram matrix at bond H_i is NOT just the SVD of the individual
core tensor. It's the Gram matrix of the FULL network's output,
viewed as a multilinear function, matricized at the bond H_i in
the unfolded tree. Due to isometric orthogonalization, this can be
computed iteratively in O(h^4) per layer.
"""

import torch
import numpy as np
from copy import deepcopy


def matricize_core(core):
    """
    Matricize a 3rd-order core tensor for orthogonalization.

    core: shape (h_out, h_in, h_in)
    returns: shape (h_out, h_in^2)

    The two input indices are flattened into a single index.
    """
    h_out, h_in, _ = core.shape
    return core.reshape(h_out, h_in * h_in)


def unmatricize_core(mat, h_in):
    """Reshape matricized core back to 3-tensor."""
    h_out = mat.shape[0]
    return mat.reshape(h_out, h_in, h_in)


# ================================================================
# Step 1: Orthogonalization (bottom-up RQ)
# ================================================================

def orthogonalize(embed, cores, unembed):
    """
    Make all cores isometric via bottom-up reduced RQ decomposition.

    After orthogonalization:
    - embed and all cores have orthonormal rows (when matricized)
    - unembed absorbs all the non-orthogonal R factors
    - The network computes the same function

    Args:
        embed: (h, input_dim+1) embedding matrix
        cores: list of (h, h, h) symmetrized interaction tensors
        unembed: (output, h) unembedding matrix

    Returns:
        embed_orth, cores_orth, unembed_modified
    """
    embed = embed.clone()
    cores = [c.clone() for c in cores]
    unembed = unembed.clone()

    L = len(cores)

    # --- Orthogonalize embedding ---
    # embed shape: (h, input_dim+1)
    # RQ decomposition: embed = R @ Q, Q has orthonormal rows
    Q, R_factor = torch.linalg.qr(embed.T)  # embed.T = Q @ R -> embed = R.T @ Q.T
    # Correct approach: use torch QR on transpose, then flip
    # embed = R @ Q_orth where Q_orth has orthonormal rows
    # embed.T (input+1, h) = Q (input+1, h) @ R (h, h) via QR
    # So embed = R.T @ Q.T, meaning Q_orth = Q.T (h, input+1), R = R.T (h, h)
    R = R_factor.T  # (h, h)
    embed_orth = Q.T  # (h, input+1), has orthonormal rows

    # Push R into first core: absorb on both input legs
    # new_core[o,a,b] = sum_{i,j} core[o,i,j] * R[i,a] * R[j,b]
    cores[0] = torch.einsum('oij,ia,jb->oab', cores[0], R, R)

    # --- Orthogonalize each core ---
    for i in range(L):
        h_out, h_in, _ = cores[i].shape
        M = matricize_core(cores[i])  # (h_out, h_in^2)

        # RQ decomposition: M = R @ Q, Q has orthonormal rows
        Q, R_factor = torch.linalg.qr(M.T)
        R = R_factor.T    # (h_out, h_out)
        Q_orth = Q.T      # (h_out, h_in^2)

        # Store orthogonalized core
        cores[i] = unmatricize_core(Q_orth, h_in)

        # Push R into next layer
        if i < L - 1:
            # Absorb R into next core's input legs
            # next_core[o,a,b] = sum_{i,j} next_core[o,i,j] * R[i,a] * R[j,b]
            cores[i + 1] = torch.einsum('oij,ia,jb->oab', cores[i + 1], R, R)
        else:
            # Last core: push R into unembedding
            # unembed[o,i] = sum_j unembed[o,j] * R[j,i]
            unembed = unembed @ R

    return embed_orth, cores, unembed


def verify_isometric(embed, cores):
    """Verify that orthogonalized cores have orthonormal rows."""
    M = embed
    orth_error = (M @ M.T - torch.eye(M.shape[0], device=M.device)).norm()
    print(f"  Embed orthogonality error: {orth_error:.2e}")

    for i, core in enumerate(cores):
        M = matricize_core(core)
        orth_error = (M @ M.T - torch.eye(M.shape[0], device=M.device)).norm()
        print(f"  Core {i} orthogonality error: {orth_error:.2e}")


# ================================================================
# Step 2: Diagonalization (top-down Gram matrices)
# ================================================================

def compute_gram_matrices(cores_orth, unembed):
    """
    Compute Gram matrices at each bond, iterating top-down.

    The Gram matrix at bond H_i captures the singular value structure
    of the FULL unfolded tree, matricized at that bond. Due to the
    isometric property of the orthogonalized cores, this can be
    computed iteratively.

    The key formula: at each step, we compute
        F = Q_i^T @ G_{i+1} @ Q_i    (shape h_i^2 x h_i^2)
    then trace over the "cloned" index:
        G_i[a,b] = sum_c F[(a,c), (b,c)]

    This trace accounts for the cloning operator: the same vector h
    enters both inputs of the bilinear core. In the unfolded tree,
    the "other" input's subtree collapses to identity due to isometry,
    leaving a trace over that index.

    Args:
        cores_orth: list of orthogonalized 3-tensors (h, h, h)
        unembed: modified unembedding matrix (output, h)

    Returns:
        grams: list of Gram matrices, one per bond (including output bond)
        eigenvalues: list of eigenvalue arrays (sorted descending)
        eigenvectors: list of eigenvector matrices
    """
    # Initialize: Gram at output bond
    G = unembed.T @ unembed  # (h, h)

    grams = [G]
    eigenvalues = []
    eigenvectors = []

    # Top-down iteration through cores
    for core in reversed(cores_orth):
        h_out, h_in, _ = core.shape

        # Eigendecompose current Gram
        evals, evecs = torch.linalg.eigh(G)
        # Sort descending
        idx = evals.argsort(descending=True)
        evals = evals[idx]
        evecs = evecs[:, idx]
        eigenvalues.append(evals)
        eigenvectors.append(evecs)

        # Matricize core
        Q = matricize_core(core)  # (h_out, h_in^2)

        # Compute F = Q^T @ G @ Q, shape (h_in^2, h_in^2)
        F = Q.T @ G @ Q

        # Reshape to 4-tensor and trace over cloned index
        # F[(a,c), (b,d)] -> F[a,c,b,d] -> sum_c F[a,c,b,c] = G_new[a,b]
        F = F.reshape(h_in, h_in, h_in, h_in)
        G = torch.einsum('acbc->ab', F)

        grams.append(G)

    # Final Gram (at input bond, before embedding)
    evals, evecs = torch.linalg.eigh(G)
    idx = evals.argsort(descending=True)
    eigenvalues.append(evals[idx])
    eigenvectors.append(evecs[:, idx])

    # Reverse so bond 0 = embedding bond, bond L+1 = output bond
    grams.reverse()
    eigenvalues.reverse()
    eigenvectors.reverse()

    return grams, eigenvalues, eigenvectors


# ================================================================
# Step 3: Truncation
# ================================================================

def truncate(embed_orth, cores_orth, unembed, eigenvectors, ranks):
    """
    Truncate the network by projecting each bond onto its top
    singular vectors.

    Args:
        embed_orth: orthogonalized embedding (h, input+1)
        cores_orth: list of orthogonalized cores (h, h, h)
        unembed: modified unembedding (output, h)
        eigenvectors: list of eigenvector matrices from diagonalization
        ranks: list of truncation ranks, one per bond

    Returns:
        embed_trunc, cores_trunc, unembed_trunc
    """
    L = len(cores_orth)

    # eigenvectors[0] is for the bond between embed and core[0]
    # eigenvectors[i] for i=1..L is for the bond between core[i-1] and core[i]
    # eigenvectors[L+1] is for the bond between core[L-1] and unembed

    embed_trunc = embed_orth.clone()
    cores_trunc = [c.clone() for c in cores_orth]
    unembed_trunc = unembed.clone()

    for i, (V, r) in enumerate(zip(eigenvectors, ranks)):
        V_trunc = V[:, :r]  # (h, r) — top r eigenvectors

        if i == 0:
            # Bond between embed and first core
            # Project embed output: embed_new = V^T @ embed
            embed_trunc = V_trunc.T @ embed_trunc  # (r, input+1)
            # Project first core input (both legs): core[o,a,b] -> core[o, V^T a, V^T b]
            cores_trunc[0] = torch.einsum('oij,ia,jb->oab',
                                          cores_trunc[0], V_trunc, V_trunc)

        elif i <= L:
            # Bond between core[i-1] and core[i]
            # Project core[i-1] output
            cores_trunc[i - 1] = torch.einsum('oij,ok->kij',
                                              cores_trunc[i - 1], V_trunc)
            if i < L:
                # Project core[i] input (both legs)
                cores_trunc[i] = torch.einsum('oij,ia,jb->oab',
                                              cores_trunc[i], V_trunc, V_trunc)
            else:
                # Project unembed input
                unembed_trunc = unembed_trunc @ V_trunc

        else:
            # Bond between last core and unembed
            # Project last core output
            cores_trunc[-1] = torch.einsum('oij,ok->kij',
                                           cores_trunc[-1], V_trunc)
            # Project unembed input
            unembed_trunc = unembed_trunc @ V_trunc

    return embed_trunc, cores_trunc, unembed_trunc


# ================================================================
# Comparison: naive per-layer SVD
# ================================================================

def naive_svd_bond_dims(embed, cores, unembed):
    """
    Compute bond dimensions using naive per-layer SVD.

    This is what you get without ODT: just SVD each individual
    core's matricization. The paper shows this MISSES the global
    low-rank structure.

    Returns dict mapping bond name to singular values.
    """
    results = {}

    # Embedding bond
    sv = torch.linalg.svdvals(embed)
    results['bond_0 (embed)'] = sv

    # Core bonds (3 per core: D-bond, L-bond, R-bond)
    for i, core in enumerate(cores):
        h_out, h_in, _ = core.shape

        # D-bond: output vs both inputs
        mat = core.reshape(h_out, h_in * h_in)
        sv = torch.linalg.svdvals(mat)
        results[f'bond_{i+1} (core {i}, D-bond)'] = sv

    # Unembedding bond
    sv = torch.linalg.svdvals(unembed)
    results[f'bond_{len(cores)+1} (unembed)'] = sv

    return results


# ================================================================
# Full ODT pipeline
# ================================================================

def run_odt(embed, cores, unembed, verbose=True):
    """
    Run the full ODT pipeline: orthogonalize, diagonalize, report.

    Args:
        embed: (h, input_dim+1) embedding matrix
        cores: list of (h, h, h) symmetrized interaction tensors
        unembed: (output, h) unembedding matrix
        verbose: print results

    Returns:
        Dictionary with all results.
    """
    h = embed.shape[0]

    if verbose:
        print("=" * 60)
        print("ODT Analysis")
        print("=" * 60)

    # --- Naive SVD for comparison ---
    if verbose:
        print("\n--- Naive per-layer SVD ---")
    naive_results = naive_svd_bond_dims(embed, cores, unembed)
    if verbose:
        for name, sv in naive_results.items():
            eff_rank = (sv > sv[0] * 1e-6).sum().item()
            print(f"  {name}: effective rank = {eff_rank}")

    # --- Step 1: Orthogonalize ---
    if verbose:
        print("\n--- Step 1: Orthogonalization ---")
    embed_orth, cores_orth, unembed_mod = orthogonalize(embed, cores, unembed)

    if verbose:
        verify_isometric(embed_orth, cores_orth)

    # --- Step 2: Diagonalize ---
    if verbose:
        print("\n--- Step 2: Diagonalization (Gram matrices) ---")
    grams, eigenvalues, eigvecs = compute_gram_matrices(cores_orth, unembed_mod)

    if verbose:
        for i, evals in enumerate(eigenvalues):
            # Effective rank using relative threshold
            evals_pos = evals.clamp(min=0)  # numerical safety
            if evals_pos[0] > 0:
                eff_rank = (evals_pos > evals_pos[0] * 1e-6).sum().item()
            else:
                eff_rank = 0
            # L2-based effective rank (as in the paper)
            total = evals_pos.sum()
            if total > 0:
                cumsum = evals_pos.cumsum(0) / total
                l2_rank = (cumsum < 0.99).sum().item() + 1
            else:
                l2_rank = 0
            print(f"  Bond {i}: effective rank = {eff_rank}, "
                  f"L2 rank (99%) = {l2_rank}")
            print(f"    Top 10 eigenvalues: {evals_pos[:10].detach().numpy().round(4)}")

    return {
        'embed_orth': embed_orth,
        'cores_orth': cores_orth,
        'unembed_mod': unembed_mod,
        'grams': grams,
        'eigenvalues': eigenvalues,
        'eigenvectors': eigvecs,
        'naive_svd': naive_results,
    }


def effective_bond_dim(eigenvalues, threshold=1e-6):
    """
    Compute effective bond dimension for each bond.

    Uses relative threshold: count eigenvalues > max_eval * threshold.
    """
    dims = []
    for evals in eigenvalues:
        evals_pos = evals.clamp(min=0)
        if evals_pos[0] > 0:
            dims.append((evals_pos > evals_pos[0] * threshold).sum().item())
        else:
            dims.append(0)
    return dims
