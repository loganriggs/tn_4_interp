"""
MNIST Pairwise TN-Sim: Part C (Residual) with BATCHED rank-1 optimization.
All 45 pairs × N seeds optimized simultaneously on GPU.
"""
import torch
import torch.nn.functional as F
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from itertools import combinations
from mnist_pairwise_tnsim import (
    ResidualBilinearMNIST, train_full_model_residual,
    compute_slice_spectral_residual, plot_pairwise_results,
    visualize_rank1_patterns, print_summary, BilinearMNIST,
    get_mnist_loaders, eval_accuracy,
)


def batched_tn_sim_1layer(w_l, w_r, w_p, target_L, target_R, target_D):
    """
    Batched TN-sim: w_l/w_r/w_p have leading batch dim B.
    target_L/R/D also have leading batch dim B (one target per batch element).

    w_l: (B, 1, input_dim)
    w_r: (B, 1, input_dim)
    w_p: (B, 2, 1)
    target_L: (B, hidden, input_dim)
    target_R: (B, hidden, input_dim)
    target_D: (B, 2, hidden)

    Returns: (B,) similarities
    """
    def batched_inner(l1, r1, p1, l2, r2, p2):
        # l1: (B, h1, d), l2: (B, h2, d)
        LL = torch.bmm(l1, l2.transpose(1, 2))  # (B, h1, h2)
        RR = torch.bmm(r1, r2.transpose(1, 2))
        LR = torch.bmm(l1, r2.transpose(1, 2))
        RL = torch.bmm(r1, l2.transpose(1, 2))
        core = 0.5 * (LL * RR + LR * RL)  # (B, h1, h2)
        DD = torch.bmm(p1.transpose(1, 2), p2)  # (B, h1, h2)
        return (core * DD).sum(dim=(1, 2))  # (B,)

    inner_ab = batched_inner(w_l, w_r, w_p, target_L, target_R, target_D)
    inner_aa = batched_inner(w_l, w_r, w_p, w_l, w_r, w_p)
    inner_bb = batched_inner(target_L, target_R, target_D,
                              target_L, target_R, target_D)

    norm_a = torch.sqrt(inner_aa.clamp(min=1e-12))
    norm_b = torch.sqrt(inner_bb.clamp(min=1e-12))
    return inner_ab / (norm_a * norm_b)


def run_batched_rank1_optimization(full_model, n_seeds=5, steps=3000, lr=1e-3,
                                    device='cuda'):
    """
    Run all 45 pairs × n_seeds simultaneously.
    """
    pairs = list(combinations(range(10), 2))
    n_pairs = len(pairs)
    B = n_pairs * n_seeds  # total batch size

    # Pre-compute all augmented target weights
    print(f"  Pre-computing augmented weights for {n_pairs} pairs...")
    all_L_aug, all_R_aug, all_D_aug = [], [], []
    for ci, cj in pairs:
        L_aug, R_aug, D_aug = full_model.get_augmented_weights_for_pair(ci, cj)
        all_L_aug.append(L_aug)
        all_R_aug.append(R_aug)
        all_D_aug.append(D_aug)

    # Stack targets: (n_pairs, hidden, input_dim) etc.
    target_L = torch.stack(all_L_aug)  # (45, 816, 785)
    target_R = torch.stack(all_R_aug)
    target_D = torch.stack(all_D_aug)  # (45, 2, 816)

    # Repeat for n_seeds: (45*n_seeds, ...)
    target_L = target_L.repeat(n_seeds, 1, 1)
    target_R = target_R.repeat(n_seeds, 1, 1)
    target_D = target_D.repeat(n_seeds, 1, 1)

    input_dim = target_L.shape[2]  # 785

    # Initialize rank-1 params for all B optimizations
    # w_l: (B, 1, 785), w_r: (B, 1, 785), w_p: (B, 2, 1)
    seeds = []
    for seed in range(n_seeds):
        for ci, cj in pairs:
            seeds.append(seed * 1000 + ci * 10 + cj)

    # Initialize with different seeds
    w_l = torch.zeros(B, 1, input_dim, device=device)
    w_r = torch.zeros(B, 1, input_dim, device=device)
    w_p = torch.zeros(B, 2, 1, device=device)
    for i, s in enumerate(seeds):
        torch.manual_seed(s)
        w_l[i] = torch.randn(1, input_dim) * 0.02
        w_r[i] = torch.randn(1, input_dim) * 0.02
        w_p[i] = torch.randn(2, 1) * 0.02

    w_l = w_l.requires_grad_(True)
    w_r = w_r.requires_grad_(True)
    w_p = w_p.requires_grad_(True)

    optimizer = torch.optim.Adam([w_l, w_r, w_p], lr=lr)

    print(f"  Optimizing {B} rank-1 models simultaneously ({n_seeds} seeds × {n_pairs} pairs)...")
    print(f"  Batch tensors: w_l {w_l.shape}, target_L {target_L.shape}")

    # Track per-step sims for convergence plot
    log_every = 10
    history = torch.zeros(steps // log_every + 1, B, device=device)

    for step in range(steps):
        optimizer.zero_grad()
        sims = batched_tn_sim_1layer(w_l, w_r, w_p, target_L, target_R, target_D)
        loss = -sims.mean()
        loss.backward()
        optimizer.step()

        if step % log_every == 0:
            history[step // log_every] = sims.detach()

        if step % 500 == 0 or step == steps - 1:
            with torch.no_grad():
                sim_vals = sims.detach()
                print(f"    Step {step:4d}: mean_sim={sim_vals.mean():.4f} "
                      f"min={sim_vals.min():.4f} max={sim_vals.max():.4f}")

    history[-1] = sims.detach()

    # Plot convergence for one example pair (pair 0 = classes 0-1)
    # showing all seeds
    history_np = history.cpu().numpy()  # (T, B)
    T = history_np.shape[0]
    steps_axis = np.arange(T) * log_every

    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    # Panel 1: all seeds for pair 0
    ax = axes[0]
    for s in range(n_seeds):
        b_idx = s * n_pairs + 0  # pair 0
        ax.plot(steps_axis, history_np[:, b_idx], alpha=0.7, label=f'seed {s}')
    ax.set_xlabel('Step')
    ax.set_ylabel('TN-Sim')
    ax.set_title(f'Pair {pairs[0][0]}-{pairs[0][1]}: convergence across seeds')
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    # Panel 2: mean ± std across all 225 runs
    ax = axes[1]
    mean_curve = history_np.mean(axis=1)
    std_curve = history_np.std(axis=1)
    ax.plot(steps_axis, mean_curve, 'b-', linewidth=2)
    ax.fill_between(steps_axis, mean_curve - std_curve, mean_curve + std_curve,
                     alpha=0.3, color='blue')
    ax.set_xlabel('Step')
    ax.set_ylabel('TN-Sim')
    ax.set_title(f'All {B} runs: mean ± std')
    ax.grid(True, alpha=0.3)

    # Panel 3: final sim distribution (histogram)
    ax = axes[2]
    final_all = history_np[-1, :]
    ax.hist(final_all, bins=30, edgecolor='k', alpha=0.7)
    ax.axvline(final_all.mean(), color='r', linestyle='--', label=f'mean={final_all.mean():.3f}')
    ax.set_xlabel('Final TN-Sim')
    ax.set_ylabel('Count')
    ax.set_title(f'Final sim distribution ({B} runs)')
    ax.legend()

    plt.tight_layout()
    plt.savefig('mnist_residual_convergence.png', dpi=150, bbox_inches='tight')
    print("  Saved convergence plot to mnist_residual_convergence.png")

    # Extract best per pair (across seeds)
    with torch.no_grad():
        final_sims = batched_tn_sim_1layer(w_l, w_r, w_p,
                                            target_L, target_R, target_D)
        # Reshape to (n_seeds, n_pairs)
        final_sims = final_sims.view(n_seeds, n_pairs)
        best_seed_idx = final_sims.argmax(dim=0)  # (n_pairs,)

        pair_sims = {}
        pair_models = {}
        for p_idx, (ci, cj) in enumerate(pairs):
            best_s = best_seed_idx[p_idx].item()
            b_idx = best_s * n_pairs + p_idx
            pair_sims[(ci, cj)] = final_sims[best_s, p_idx].item()

            # Create a BilinearMNIST model with these weights for visualization
            model = BilinearMNIST(input_dim, 1, 2).to(device)
            model.bi_linear.weight.data[:1] = w_l[b_idx].detach()
            model.bi_linear.weight.data[1:] = w_r[b_idx].detach()
            model.projection.weight.data = w_p[b_idx].detach()
            pair_models[(ci, cj)] = model

    return pair_sims, pair_models


if __name__ == '__main__':
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Device: {device}")

    print("\n" + "=" * 70)
    print("PART C: 1-Layer Bilinear with Residual (BATCHED)")
    print("=" * 70)

    print("\nC1. Training full residual model...")
    full_res = train_full_model_residual(d_hidden=64, d_rank=32, epochs=10, device=device)

    print("\nC2. Computing spectral baseline...")
    spectral_res = compute_slice_spectral_residual(full_res)

    print("\nC3. Batched rank-1 optimization for all pairs...")
    sims_res, models_res = run_batched_rank1_optimization(
        full_res, n_seeds=5, steps=3000, lr=1e-3, device=device
    )
    print_summary(sims_res, "residual rank-1")

    print("\nC4. Plotting residual results...")
    plot_pairwise_results(sims_res, spectral_res, 'Residual', 'mnist_residual_tnsim.png')

    # Rank-1 patterns are in 785-dim augmented space; truncate to 784 for visualization
    print("\nC5. Visualizing rank-1 patterns (first 784 dims = image)...")
    trunc_models = {}
    for key, m in models_res.items():
        m_trunc = BilinearMNIST(784, 1, 2).to(device)
        m_trunc.bi_linear.weight.data[:1] = m.bi_linear.weight.data[:1, :784]
        m_trunc.bi_linear.weight.data[1:] = m.bi_linear.weight.data[1:, :784]
        m_trunc.projection.weight.data = m.projection.weight.data
        trunc_models[key] = m_trunc
    visualize_rank1_patterns(trunc_models, sims_res, n_show=10)

    # Save for comparison
    torch.save({
        'sims': sims_res,
        'spectral': spectral_res,
        'model_state': full_res.state_dict(),
    }, 'mnist_residual_results.pt')
    print("\nSaved to mnist_residual_results.pt")
