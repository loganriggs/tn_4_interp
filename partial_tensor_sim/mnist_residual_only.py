"""
MNIST Pairwise TN-Sim: Part C only (Residual Bilinear)
Runs the residual experiment and saves results for later comparison.
"""
import torch
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from mnist_pairwise_tnsim import (
    ResidualBilinearMNIST, BilinearMNIST,
    train_full_model_residual, run_all_pairs_residual,
    compute_slice_spectral_residual, plot_pairwise_results,
    visualize_rank1_patterns, print_summary,
    tn_inner_1layer,
)

if __name__ == '__main__':
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Device: {device}")

    print("\n" + "=" * 70)
    print("PART C: 1-Layer Bilinear with Residual")
    print("=" * 70)

    print("\nC1. Training full residual model...")
    full_res = train_full_model_residual(d_hidden=64, d_rank=32, epochs=10, device=device)

    print("\nC2. Computing spectral baseline...")
    spectral_res = compute_slice_spectral_residual(full_res)

    print("\nC3. Optimizing rank-1 for all pairs...")
    sims_res, models_res = run_all_pairs_residual(
        full_res, steps=3000, n_seeds=5, device=device
    )
    print_summary(sims_res, "residual rank-1")

    print("\nC4. Plotting residual results...")
    plot_pairwise_results(sims_res, spectral_res, 'Residual', 'mnist_residual_tnsim.png')

    print("\nC5. Visualizing rank-1 patterns...")
    visualize_rank1_patterns(models_res, sims_res, n_show=10)

    # Save results for later comparison
    np.save('mnist_residual_sims.npy', dict(sims_res))
    print("\nSaved sims to mnist_residual_sims.npy")
