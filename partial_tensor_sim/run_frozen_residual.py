"""
Frozen-residual rank-1 optimization.

Instead of optimizing a free rank-1 in augmented space, we freeze the exact
residual and only optimize the rank-1 bilinear correction on top.

Approximation = exact residual + trainable rank-1 bilinear:
  L_approx = [L_res; l]    (784+1, 785)
  R_approx = [R_res; r]    (784+1, 785)
  D_approx = [W_res, d]    (n_out, 784+1)

Only l, r, d are optimized. The residual is always captured perfectly.
The TN-sim now measures how well the rank-1 captures the bilinear correction.
"""
import torch
import torch.nn.functional as F
import numpy as np
import os
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from itertools import combinations
from mnist_pairwise_tnsim import (
    ResidualBilinearMNIST, BilinearMNIST,
    get_mnist_loaders, eval_accuracy,
    tn_inner_1layer, tn_sim_1layer,
    print_summary, plot_pairwise_results,
)
from run_residual import get_augmented_weights_for_class

IMG_DIR = os.path.join(os.path.dirname(__file__), 'images')
os.makedirs(IMG_DIR, exist_ok=True)


# =============================================================================
# BATCHED TN-SIM WITH FROZEN RESIDUAL
# =============================================================================

def batched_tn_inner_1layer(l1, r1, p1, l2, r2, p2):
    """Batched symmetrized TN inner product."""
    LL = torch.bmm(l1, l2.transpose(1, 2))
    RR = torch.bmm(r1, r2.transpose(1, 2))
    LR = torch.bmm(l1, r2.transpose(1, 2))
    RL = torch.bmm(r1, l2.transpose(1, 2))
    core = 0.5 * (LL * RR + LR * RL)
    DD = torch.bmm(p1.transpose(1, 2), p2)
    return (core * DD).sum(dim=(1, 2))


def batched_tn_sim_frozen_residual(l_rank1, r_rank1, d_rank1,
                                    L_res_batch, R_res_batch, D_res_batch,
                                    target_L, target_R, target_D):
    """
    TN-sim between (frozen_residual + rank1) and full target.

    l_rank1: (B, 1, 785) - trainable
    r_rank1: (B, 1, 785)
    d_rank1: (B, n_out, 1)
    L_res_batch: (B, 784, 785) - frozen
    R_res_batch: (B, 784, 785)
    D_res_batch: (B, n_out, 784)
    target_L/R/D: full augmented target
    """
    # Build approximation by concatenating frozen residual + trainable rank-1
    L_approx = torch.cat([L_res_batch, l_rank1], dim=1)  # (B, 785, 785)
    R_approx = torch.cat([R_res_batch, r_rank1], dim=1)
    D_approx = torch.cat([D_res_batch, d_rank1], dim=2)  # (B, n_out, 785)

    inner_ab = batched_tn_inner_1layer(L_approx, R_approx, D_approx,
                                        target_L, target_R, target_D)
    inner_aa = batched_tn_inner_1layer(L_approx, R_approx, D_approx,
                                        L_approx, R_approx, D_approx)
    inner_bb = batched_tn_inner_1layer(target_L, target_R, target_D,
                                        target_L, target_R, target_D)

    norm_a = torch.sqrt(inner_aa.clamp(min=1e-12))
    norm_b = torch.sqrt(inner_bb.clamp(min=1e-12))
    return inner_ab / (norm_a * norm_b)


def build_residual_parts(model, class_indices, device):
    """
    Build the frozen residual parts for a set of class indices.

    For pair (i,j): W_res = W_head[[i,j]] @ W_embed  (2, 784)
    For single class i: W_res = W_head[[i]] @ W_embed  (1, 784)

    Returns L_res, R_res, D_res (without batch dim).
    """
    with torch.no_grad():
        W_e = model.W_embed  # (d_hidden, 784)
        if isinstance(class_indices, (list, tuple)):
            W_h = model.W_head[list(class_indices)]
        else:
            W_h = model.W_head[[class_indices]]

        W_res = W_h @ W_e  # (n_out, 784)
        n_out = W_res.shape[0]
        n_in = 784

        L_res = torch.zeros(n_in, n_in + 1, device=device)
        L_res[:, -1] = 1.0
        R_res = torch.zeros(n_in, n_in + 1, device=device)
        R_res[:, :n_in] = torch.eye(n_in, device=device)
        D_res = W_res  # (n_out, 784)

    return L_res, R_res, D_res


def run_frozen_residual_optimization(model, targets, labels, n_seeds=5,
                                      steps=1000, lr=1e-3, device='cuda'):
    """
    Batched optimization with frozen residual.

    targets: list of (class_indices, L_aug, R_aug, D_aug) tuples
    labels: list of labels for each target
    """
    n_targets = len(targets)
    B = n_targets * n_seeds
    n_out = targets[0][3].shape[0]  # 2 for pairs, 1 for classes

    # Build frozen residual parts and full targets
    all_L_res, all_R_res, all_D_res = [], [], []
    all_tL, all_tR, all_tD = [], [], []

    for class_indices, L_aug, R_aug, D_aug in targets:
        L_res, R_res, D_res = build_residual_parts(model, class_indices, device)
        all_L_res.append(L_res)
        all_R_res.append(R_res)
        all_D_res.append(D_res)
        all_tL.append(L_aug)
        all_tR.append(R_aug)
        all_tD.append(D_aug)

    # Stack and repeat for seeds
    L_res_batch = torch.stack(all_L_res).repeat(n_seeds, 1, 1)  # (B, 784, 785)
    R_res_batch = torch.stack(all_R_res).repeat(n_seeds, 1, 1)
    D_res_batch = torch.stack(all_D_res).repeat(n_seeds, 1, 1)
    tL = torch.stack(all_tL).repeat(n_seeds, 1, 1)
    tR = torch.stack(all_tR).repeat(n_seeds, 1, 1)
    tD = torch.stack(all_tD).repeat(n_seeds, 1, 1)

    # Trainable rank-1 params
    l_rank1 = torch.zeros(B, 1, 785, device=device)
    r_rank1 = torch.zeros(B, 1, 785, device=device)
    d_rank1 = torch.zeros(B, n_out, 1, device=device)

    for i in range(B):
        torch.manual_seed(i * 7919 + 42)
        l_rank1[i] = torch.randn(1, 785) * 0.02
        r_rank1[i] = torch.randn(1, 785) * 0.02
        d_rank1[i] = torch.randn(n_out, 1) * 0.02

    l_rank1.requires_grad_(True)
    r_rank1.requires_grad_(True)
    d_rank1.requires_grad_(True)

    optimizer = torch.optim.Adam([l_rank1, r_rank1, d_rank1], lr=lr)

    print(f"  Optimizing {B} frozen-residual + rank-1 models "
          f"({n_seeds} seeds x {n_targets} targets, {steps} steps)...")

    # Also compute residual-only baseline (rank-1 = 0)
    with torch.no_grad():
        zero_l = torch.zeros(B, 1, 785, device=device)
        zero_r = torch.zeros(B, 1, 785, device=device)
        zero_d = torch.zeros(B, n_out, 1, device=device)
        baseline_sims = batched_tn_sim_frozen_residual(
            zero_l, zero_r, zero_d,
            L_res_batch, R_res_batch, D_res_batch,
            tL, tR, tD
        )
        print(f"  Residual-only baseline: mean={baseline_sims.mean():.4f} "
              f"min={baseline_sims.min():.4f} max={baseline_sims.max():.4f}")

    log_every = 10
    history = torch.zeros(steps // log_every + 1, B, device=device)

    for step in range(steps):
        optimizer.zero_grad()
        sims = batched_tn_sim_frozen_residual(
            l_rank1, r_rank1, d_rank1,
            L_res_batch, R_res_batch, D_res_batch,
            tL, tR, tD
        )
        loss = -sims.mean()
        loss.backward()
        optimizer.step()

        if step % log_every == 0:
            history[step // log_every] = sims.detach()

        if step % 200 == 0 or step == steps - 1:
            with torch.no_grad():
                s = sims.detach()
                print(f"    Step {step:4d}: mean={s.mean():.4f} min={s.min():.4f} max={s.max():.4f}")

    history[-1] = sims.detach()

    # Convergence plot
    history_np = history.cpu().numpy()
    T = history_np.shape[0]
    steps_axis = np.arange(T) * log_every
    baseline_np = baseline_sims.cpu().numpy()

    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    ax = axes[0]
    for s in range(n_seeds):
        ax.plot(steps_axis, history_np[:, s * n_targets], alpha=0.7, label=f'seed {s}')
    ax.axhline(baseline_np[0], color='k', linestyle='--', alpha=0.5, label='residual only')
    ax.set_xlabel('Step'); ax.set_ylabel('TN-Sim')
    ax.set_title(f'{labels[0]}: convergence')
    ax.legend(fontsize=7); ax.grid(True, alpha=0.3)

    ax = axes[1]
    mean_c = history_np.mean(axis=1)
    std_c = history_np.std(axis=1)
    ax.plot(steps_axis, mean_c, 'b-', linewidth=2)
    ax.fill_between(steps_axis, mean_c - std_c, mean_c + std_c, alpha=0.3, color='blue')
    ax.axhline(baseline_np.mean(), color='k', linestyle='--', alpha=0.5, label='residual only')
    ax.set_xlabel('Step'); ax.set_ylabel('TN-Sim')
    ax.set_title(f'All {B} runs: mean +/- std'); ax.legend(); ax.grid(True, alpha=0.3)

    ax = axes[2]
    final_all = history_np[-1]
    ax.hist(final_all, bins=30, edgecolor='k', alpha=0.7, label='frozen res + rank-1')
    ax.hist(baseline_np, bins=30, edgecolor='k', alpha=0.3, color='orange', label='residual only')
    ax.axvline(final_all.mean(), color='b', linestyle='--')
    ax.axvline(baseline_np.mean(), color='orange', linestyle='--')
    ax.set_xlabel('TN-Sim'); ax.set_ylabel('Count')
    ax.set_title('Final distribution'); ax.legend(fontsize=8)

    plt.tight_layout()

    # Extract best per target
    with torch.no_grad():
        final_sims = batched_tn_sim_frozen_residual(
            l_rank1, r_rank1, d_rank1,
            L_res_batch, R_res_batch, D_res_batch,
            tL, tR, tD
        ).view(n_seeds, n_targets)
        best_idx = final_sims.argmax(dim=0)

        # Also get residual-only sims per target
        baseline_per_target = baseline_sims.view(n_seeds, n_targets)[0]  # same across seeds

        result_sims = {}
        result_baseline = {}
        result_weights = {}
        for t_idx, label in enumerate(labels):
            bs = best_idx[t_idx].item()
            b_idx = bs * n_targets + t_idx
            result_sims[label] = final_sims[bs, t_idx].item()
            result_baseline[label] = baseline_per_target[t_idx].item()
            result_weights[label] = {
                'l': l_rank1[b_idx].detach().clone(),
                'r': r_rank1[b_idx].detach().clone(),
                'd': d_rank1[b_idx].detach().clone(),
            }

    return result_sims, result_baseline, result_weights, fig


# =============================================================================
# MAIN
# =============================================================================

if __name__ == '__main__':
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Device: {device}")

    results = torch.load(os.path.join(os.path.dirname(__file__), 'residual_results.pt'),
                         map_location=device, weights_only=False)
    model = ResidualBilinearMNIST(784, 64, 32, 10).to(device)
    model.load_state_dict(results['model_state'])
    model.eval()

    # ==== Per-class ====
    print("\n" + "=" * 70)
    print("Per-class: frozen residual + rank-1")
    print("=" * 70)

    class_targets = []
    class_labels = []
    for c in range(10):
        L, R, D = get_augmented_weights_for_class(model, c)
        class_targets.append((c, L, R, D))
        class_labels.append(c)

    class_sims, class_baseline, class_weights, fig_class = run_frozen_residual_optimization(
        model, class_targets, class_labels,
        n_seeds=5, steps=1000, lr=1e-3, device=device
    )
    fig_class.savefig(os.path.join(IMG_DIR, 'frozen_res_class_convergence.png'),
                      dpi=150, bbox_inches='tight')
    plt.close(fig_class)

    print("\n  Per-class results:")
    print(f"  {'Class':>6s}  {'Res-only':>9s}  {'Res+Rank1':>10s}  {'Improvement':>12s}")
    for c in range(10):
        bl = class_baseline[c]
        full = class_sims[c]
        print(f"  {c:>6d}  {bl:>9.4f}  {full:>10.4f}  {full - bl:>+12.4f}")

    # ==== Pairwise ====
    print("\n" + "=" * 70)
    print("Pairwise: frozen residual + rank-1")
    print("=" * 70)

    pairs = list(combinations(range(10), 2))
    pair_targets = []
    pair_labels = []
    for ci, cj in pairs:
        L, R, D = model.get_augmented_weights_for_pair(ci, cj)
        pair_targets.append(((ci, cj), L, R, D))
        pair_labels.append((ci, cj))

    pair_sims, pair_baseline, pair_weights, fig_pair = run_frozen_residual_optimization(
        model, pair_targets, pair_labels,
        n_seeds=5, steps=1000, lr=1e-3, device=device
    )
    fig_pair.savefig(os.path.join(IMG_DIR, 'frozen_res_pair_convergence.png'),
                     dpi=150, bbox_inches='tight')
    plt.close(fig_pair)

    sorted_pairs = sorted(pair_sims.items(), key=lambda x: x[1], reverse=True)
    print(f"\n  {'Pair':>6s}  {'Res-only':>9s}  {'Res+Rank1':>10s}  {'Improvement':>12s}")
    for (ci, cj), sim in sorted_pairs[:10]:
        bl = pair_baseline[(ci, cj)]
        print(f"  {ci}-{cj:>4d}  {bl:>9.4f}  {sim:>10.4f}  {sim - bl:>+12.4f}")
    print("  ...")
    for (ci, cj), sim in sorted_pairs[-5:]:
        bl = pair_baseline[(ci, cj)]
        print(f"  {ci}-{cj:>4d}  {bl:>9.4f}  {sim:>10.4f}  {sim - bl:>+12.4f}")

    # Summary comparison
    print(f"\n  Overall:")
    bl_mean = np.mean(list(pair_baseline.values()))
    full_mean = np.mean(list(pair_sims.values()))
    print(f"    Residual-only mean TN-sim: {bl_mean:.4f}")
    print(f"    Residual + rank-1 mean:    {full_mean:.4f}")
    print(f"    Improvement:               {full_mean - bl_mean:+.4f}")

    # Save
    torch.save({
        'class_sims': class_sims,
        'class_baseline': class_baseline,
        'class_weights': class_weights,
        'pair_sims': pair_sims,
        'pair_baseline': pair_baseline,
        'pair_weights': pair_weights,
    }, os.path.join(os.path.dirname(__file__), 'frozen_residual_results.pt'))
    print("\nSaved to frozen_residual_results.pt")
