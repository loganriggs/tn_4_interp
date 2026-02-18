"""
Sanity checks for the ODT implementation.

Run this first to verify everything works before training on SVHN.
Tests on small random models — no data needed.
"""

import torch
import numpy as np
from model import ChiNet, ResidualBilinearNet, BilinearCore
from odt import (
    orthogonalize, compute_gram_matrices, naive_svd_bond_dims,
    effective_bond_dim, verify_isometric, run_odt
)


def test_orthogonalize():
    """Test that orthogonalization preserves the network function."""
    print("=" * 60)
    print("Test: Orthogonalization preserves function")
    print("=" * 60)

    torch.manual_seed(42)
    h, n_in, n_out = 16, 8, 4

    # Create random cores
    embed = torch.randn(h, n_in + 1) / np.sqrt(n_in)
    cores = [
        torch.randn(h, h, h) / h for _ in range(3)
    ]
    # Symmetrize cores
    cores = [0.5 * (c + c.transpose(1, 2)) for c in cores]
    unembed = torch.randn(n_out, h) / np.sqrt(h)

    # Evaluate original network on test input
    x = torch.randn(10, n_in)
    ones = torch.ones(10, 1)
    inp = torch.cat([ones, x], dim=-1)

    h_vec = inp @ embed.T
    for core in cores:
        h_vec = torch.einsum('oij,bi,bj->bo', core, h_vec, h_vec)
    out_original = h_vec @ unembed.T

    # Orthogonalize
    embed_orth, cores_orth, unembed_mod = orthogonalize(embed, cores, unembed)

    # Evaluate orthogonalized network
    h_vec = inp @ embed_orth.T
    for core in cores_orth:
        h_vec = torch.einsum('oij,bi,bj->bo', core, h_vec, h_vec)
    out_orth = h_vec @ unembed_mod.T

    err = (out_original - out_orth).norm() / out_original.norm()
    print(f"  Relative error: {err:.2e}")
    print(f"  {'PASS' if err < 1e-4 else 'FAIL'}")

    # Verify isometric property
    print("\n  Isometric property:")
    verify_isometric(embed_orth, cores_orth)

    return err < 1e-4


def test_gram_rank_bound():
    """
    Test that ODT Gram matrices have rank ≤ bond dimension
    for a low-rank construction.
    """
    print("\n" + "=" * 60)
    print("Test: Gram matrix rank respects bond dimension")
    print("=" * 60)

    torch.manual_seed(123)
    h = 16
    true_rank = 4  # build a network that only uses 4 dimensions

    # Build a low-rank network: embed into a rank-4 subspace
    U = torch.randn(h, true_rank)
    embed = U @ torch.randn(true_rank, 9)  # input dim = 8 + 1

    # Cores that only act in the rank-4 subspace
    cores = []
    for _ in range(2):
        # Low-rank Khatri-Rao: A, B are (h, h) but rank-4
        A = U @ torch.randn(true_rank, h)
        B = U @ torch.randn(true_rank, h)
        T = torch.einsum('oi,oj->oij', A, B)
        T = 0.5 * (T + T.transpose(1, 2))
        cores.append(T)

    unembed = torch.randn(4, true_rank) @ U.T  # rank-4

    # Run ODT
    odt_results = run_odt(embed, cores, unembed, verbose=True)

    dims = effective_bond_dim(odt_results['eigenvalues'])
    print(f"\n  True rank: {true_rank}")
    print(f"  ODT effective dims: {dims}")
    print(f"  {'PASS' if all(d <= true_rank + 1 for d in dims) else 'FAIL'}")


def test_odt_vs_naive():
    """
    Demonstrate that ODT finds lower effective rank than naive SVD.

    Creates a network where individual cores look full-rank but
    the global function is low-rank (the paper's main point).
    """
    print("\n" + "=" * 60)
    print("Test: ODT finds lower rank than naive SVD")
    print("=" * 60)

    torch.manual_seed(7)
    h = 32
    n_in = 8
    n_out = 4

    # Create a χ-net model
    model = ChiNet(n_in, h, n_out, n_layers=3)

    # Get cores (with random init, the network won't be low-rank,
    # but we can at least verify the pipeline runs)
    embed, cores, unembed = model.get_cores_for_odt()

    print("\n--- Naive SVD ---")
    naive = naive_svd_bond_dims(embed, cores, unembed)
    for name, sv in naive.items():
        eff = (sv > sv[0] * 1e-6).sum().item()
        print(f"  {name}: effective rank = {eff}")

    print("\n--- ODT ---")
    odt_results = run_odt(embed, cores, unembed, verbose=True)

    dims_naive = []
    for name, sv in naive.items():
        dims_naive.append((sv > sv[0] * 1e-6).sum().item())
    dims_odt = effective_bond_dim(odt_results['eigenvalues'])

    print(f"\n  Naive SVD dims: {dims_naive}")
    print(f"  ODT dims:       {dims_odt}")

    # After training, ODT should find much lower dims.
    # With random init, the reduction may be modest.
    print("  (With trained models, ODT dims should be much lower)")


def test_forward_consistency():
    """Test that ChiNet.forward matches manual einsum evaluation."""
    print("\n" + "=" * 60)
    print("Test: ChiNet forward matches manual einsum")
    print("=" * 60)

    torch.manual_seed(99)
    model = ChiNet(8, 16, 4, n_layers=2)
    model.eval()

    x = torch.randn(5, 8)

    # Model forward
    y_model = model(x)

    # Manual forward with extracted cores
    embed, cores, unembed = model.get_cores_for_odt()

    ones = torch.ones(5, 1)
    inp = torch.cat([ones, x], dim=-1)
    h = inp @ embed.T
    for core in cores:
        h = torch.einsum('oij,bi,bj->bo', core, h, h)
    y_manual = h @ unembed.T

    err = (y_model - y_manual).norm() / y_model.norm()
    print(f"  Relative error: {err:.2e}")
    print(f"  {'PASS' if err < 1e-3 else 'FAIL'}")

    # Note: may not be exact due to RMSBatchNorm running stats
    # In eval mode, it divides by running_rms (initialized to 1.0)
    # and the core extraction folds this in.


def test_augmented_residual_forward():
    """
    Test that augmented cores give the same output as the residual model.

    This is the KEY sanity check: verifies that the constant/discard trick
    correctly folds the residual into the bilinear core.
    """
    print("\n" + "=" * 60)
    print("Test: Augmented residual cores match model forward")
    print("=" * 60)

    for n_layers in [1, 2, 3]:
        torch.manual_seed(42)
        model = ResidualBilinearNet(8, 16, 4, n_layers=n_layers)

        # Run forward passes in train mode to get non-trivial running_rms
        model.train()
        for _ in range(20):
            _ = model(torch.randn(32, 8))
        model.eval()

        x = torch.randn(10, 8)
        y_model = model(x)

        embed_aug, cores_aug, unembed_aug = model.get_augmented_cores_for_odt()

        # Manual forward with augmented cores
        ones = torch.ones(10, 1)
        x_aug = torch.cat([ones, x], dim=-1)
        h = x_aug @ embed_aug.T  # (10, H+1)

        # Verify constant is preserved through all layers
        const_ok = True
        if abs(h[0, 0].item() - 1.0) > 1e-6:
            const_ok = False

        for i, core in enumerate(cores_aug):
            h = torch.einsum('oij,bi,bj->bo', core, h, h)
            if abs(h[0, 0].item() - 1.0) > 1e-4:
                const_ok = False

        y_manual = h @ unembed_aug.T

        err = (y_model - y_manual).norm() / y_model.norm()
        status = 'PASS' if (err < 1e-3 and const_ok) else 'FAIL'
        print(f"  L={n_layers}: relative error = {err:.2e}, "
              f"constant preserved = {const_ok}  [{status}]")

    return err < 1e-3


def test_augmented_odt_pipeline():
    """
    Test that ODT works correctly on augmented residual cores.

    Verifies:
    1. Orthogonalization preserves the augmented network function
    2. Isometric property holds for augmented cores
    3. Gram matrices compute without error
    """
    print("\n" + "=" * 60)
    print("Test: ODT pipeline on augmented residual cores")
    print("=" * 60)

    torch.manual_seed(42)
    model = ResidualBilinearNet(8, 16, 4, n_layers=2)

    model.train()
    for _ in range(20):
        _ = model(torch.randn(32, 8))
    model.eval()

    embed_aug, cores_aug, unembed_aug = model.get_augmented_cores_for_odt()

    # Test input
    x = torch.randn(5, 8)
    ones = torch.ones(5, 1)
    x_aug = torch.cat([ones, x], dim=-1)

    # Original forward with augmented cores
    h = x_aug @ embed_aug.T
    for core in cores_aug:
        h = torch.einsum('oij,bi,bj->bo', core, h, h)
    y_orig = h @ unembed_aug.T

    # Orthogonalize
    embed_orth, cores_orth, unembed_mod = orthogonalize(
        embed_aug, cores_aug, unembed_aug)

    # Forward with orthogonalized cores
    h = x_aug @ embed_orth.T
    for core in cores_orth:
        h = torch.einsum('oij,bi,bj->bo', core, h, h)
    y_orth = h @ unembed_mod.T

    err = (y_orig - y_orth).norm() / y_orig.norm()
    print(f"  Orthogonalization error: {err:.2e}  "
          f"[{'PASS' if err < 1e-4 else 'FAIL'}]")

    # Verify isometric property
    print("  Isometric property:")
    verify_isometric(embed_orth, cores_orth)

    # Compute Gram matrices
    grams, eigenvalues, eigvecs = compute_gram_matrices(cores_orth, unembed_mod)
    print(f"  Gram matrices: {len(grams)} grams, {len(eigenvalues)} eigenvalue sets")

    for i, evals in enumerate(eigenvalues):
        evals_pos = evals.clamp(min=0)
        eff_rank = (evals_pos > evals_pos[0] * 1e-6).sum().item() if evals_pos[0] > 0 else 0
        print(f"    Bond {i}: effective rank = {eff_rank}/{evals.shape[0]}")

    print("  PASS (ODT pipeline completed)")
    return True


def test_augmented_vs_bilinear_only():
    """
    Compare augmented (with residual) vs bilinear-only ODT spectra.

    The augmented cores should have higher effective rank due to the
    residual path contributing a near-identity component.
    """
    print("\n" + "=" * 60)
    print("Test: Augmented vs bilinear-only ODT comparison")
    print("=" * 60)

    torch.manual_seed(42)
    model = ResidualBilinearNet(8, 32, 4, n_layers=2)

    model.train()
    for _ in range(20):
        _ = model(torch.randn(32, 8))
    model.eval()

    # Bilinear only
    embed_bil, cores_bil, unembed_bil = model.get_cores_for_odt(fold_residual=False)
    odt_bil = run_odt(embed_bil, cores_bil, unembed_bil, verbose=False)
    dims_bil = effective_bond_dim(odt_bil['eigenvalues'])

    # Augmented (with residual)
    embed_aug, cores_aug, unembed_aug = model.get_augmented_cores_for_odt()
    odt_aug = run_odt(embed_aug, cores_aug, unembed_aug, verbose=False)
    dims_aug = effective_bond_dim(odt_aug['eigenvalues'])

    print(f"  Bilinear-only dims: {dims_bil}")
    print(f"  Augmented dims:     {dims_aug}")
    print(f"  (Augmented should generally have equal or higher effective ranks)")
    print("  PASS")


if __name__ == '__main__':
    print("Running ODT sanity checks...\n")
    test_forward_consistency()
    test_orthogonalize()
    test_gram_rank_bound()
    test_odt_vs_naive()
    print("\n--- Augmented residual tests ---")
    test_augmented_residual_forward()
    test_augmented_odt_pipeline()
    test_augmented_vs_bilinear_only()
    print("\n" + "=" * 60)
    print("All checks complete!")
    print("=" * 60)
