"""
MNIST Partial Tensor Similarity: Residual Bilinear Model

Trains a 1-layer residual bilinear (embed + bilinear + skip + head) on MNIST,
then computes rank-1 TN-sim approximations for:
  - All 45 class pairs (2-class slices)
  - All 10 individual classes (1-class slices)

Uses batched optimization: all rank-1 models trained simultaneously on GPU.
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

IMG_DIR = os.path.join(os.path.dirname(__file__), 'images')
os.makedirs(IMG_DIR, exist_ok=True)


# =============================================================================
# BATCHED TN-SIM
# =============================================================================

def batched_tn_inner_1layer(l1, r1, p1, l2, r2, p2):
    """Batched symmetrized TN inner product. All inputs have leading batch dim B."""
    LL = torch.bmm(l1, l2.transpose(1, 2))
    RR = torch.bmm(r1, r2.transpose(1, 2))
    LR = torch.bmm(l1, r2.transpose(1, 2))
    RL = torch.bmm(r1, l2.transpose(1, 2))
    core = 0.5 * (LL * RR + LR * RL)
    DD = torch.bmm(p1.transpose(1, 2), p2)
    return (core * DD).sum(dim=(1, 2))


def batched_tn_sim_1layer(w_l, w_r, w_p, target_L, target_R, target_D):
    """Batched TN cosine similarity. Returns (B,) tensor."""
    inner_ab = batched_tn_inner_1layer(w_l, w_r, w_p, target_L, target_R, target_D)
    inner_aa = batched_tn_inner_1layer(w_l, w_r, w_p, w_l, w_r, w_p)
    inner_bb = batched_tn_inner_1layer(target_L, target_R, target_D,
                                        target_L, target_R, target_D)
    norm_a = torch.sqrt(inner_aa.clamp(min=1e-12))
    norm_b = torch.sqrt(inner_bb.clamp(min=1e-12))
    return inner_ab / (norm_a * norm_b)


# =============================================================================
# AUGMENTED WEIGHTS FOR SINGLE CLASS
# =============================================================================

def get_augmented_weights_for_class(model, class_i):
    """
    Like get_augmented_weights_for_pair but for a single class.
    Returns L_aug, R_aug, D_aug where D_aug has shape (1, 784+rank).
    """
    with torch.no_grad():
        W_e = model.W_embed
        W_h = model.W_head[[class_i]]  # (1, d_hidden)

        L_eff = model.L @ W_e
        R_eff = model.R @ W_e
        D_eff = W_h @ model.D
        W_res = W_h @ W_e

        n_in = W_e.shape[1]
        rank = model.d_rank
        device = W_e.device
        dtype = W_e.dtype

        L_res = torch.zeros(n_in, n_in + 1, device=device, dtype=dtype)
        L_res[:, -1] = 1.0
        R_res = torch.zeros(n_in, n_in + 1, device=device, dtype=dtype)
        R_res[:, :n_in] = torch.eye(n_in, device=device, dtype=dtype)

        L_bil = torch.cat([L_eff, torch.zeros(rank, 1, device=device, dtype=dtype)], dim=1)
        R_bil = torch.cat([R_eff, torch.zeros(rank, 1, device=device, dtype=dtype)], dim=1)

        L_aug = torch.cat([L_res, L_bil], dim=0)
        R_aug = torch.cat([R_res, R_bil], dim=0)
        D_aug = torch.cat([W_res, D_eff], dim=1)

    return L_aug, R_aug, D_aug


# =============================================================================
# BATCHED RANK-1 OPTIMIZATION
# =============================================================================

def run_batched_optimization(target_Ls, target_Rs, target_Ds, labels,
                              n_seeds=5, steps=1000, lr=1e-3, device='cuda',
                              title_prefix=''):
    """
    Generic batched rank-1 optimization.

    Args:
        target_Ls: list of (hidden, input_dim) tensors
        target_Rs: list of (hidden, input_dim) tensors
        target_Ds: list of (n_out, hidden) tensors
        labels: list of string labels for each target
        n_seeds: number of random seeds per target
        steps: optimization steps
    """
    n_targets = len(target_Ls)
    B = n_targets * n_seeds
    input_dim = target_Ls[0].shape[1]
    n_out = target_Ds[0].shape[0]

    # Stack and repeat for seeds
    tL = torch.stack(target_Ls).repeat(n_seeds, 1, 1)
    tR = torch.stack(target_Rs).repeat(n_seeds, 1, 1)
    tD = torch.stack(target_Ds).repeat(n_seeds, 1, 1)

    # Initialize
    w_l = torch.zeros(B, 1, input_dim, device=device)
    w_r = torch.zeros(B, 1, input_dim, device=device)
    w_p = torch.zeros(B, n_out, 1, device=device)
    for i in range(B):
        torch.manual_seed(i * 7919 + 42)
        w_l[i] = torch.randn(1, input_dim) * 0.02
        w_r[i] = torch.randn(1, input_dim) * 0.02
        w_p[i] = torch.randn(n_out, 1) * 0.02

    w_l.requires_grad_(True)
    w_r.requires_grad_(True)
    w_p.requires_grad_(True)

    optimizer = torch.optim.Adam([w_l, w_r, w_p], lr=lr)

    print(f"  Optimizing {B} rank-1 models ({n_seeds} seeds x {n_targets} targets, {steps} steps)...")

    log_every = 10
    history = torch.zeros(steps // log_every + 1, B, device=device)

    for step in range(steps):
        optimizer.zero_grad()
        sims = batched_tn_sim_1layer(w_l, w_r, w_p, tL, tR, tD)
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

    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    # Panel 1: all seeds for target 0
    ax = axes[0]
    for s in range(n_seeds):
        ax.plot(steps_axis, history_np[:, s * n_targets], alpha=0.7, label=f'seed {s}')
    ax.set_xlabel('Step'); ax.set_ylabel('TN-Sim')
    ax.set_title(f'{labels[0]}: convergence across seeds')
    ax.legend(fontsize=8); ax.grid(True, alpha=0.3)

    # Panel 2: mean ± std
    ax = axes[1]
    mean_c = history_np.mean(axis=1)
    std_c = history_np.std(axis=1)
    ax.plot(steps_axis, mean_c, 'b-', linewidth=2)
    ax.fill_between(steps_axis, mean_c - std_c, mean_c + std_c, alpha=0.3, color='blue')
    ax.set_xlabel('Step'); ax.set_ylabel('TN-Sim')
    ax.set_title(f'All {B} runs: mean +/- std'); ax.grid(True, alpha=0.3)

    # Panel 3: final histogram
    ax = axes[2]
    final_all = history_np[-1]
    ax.hist(final_all, bins=30, edgecolor='k', alpha=0.7)
    ax.axvline(final_all.mean(), color='r', linestyle='--', label=f'mean={final_all.mean():.3f}')
    ax.set_xlabel('Final TN-Sim'); ax.set_ylabel('Count')
    ax.set_title(f'Final distribution ({B} runs)'); ax.legend()

    plt.tight_layout()
    fname = os.path.join(IMG_DIR, f'{title_prefix}_convergence.png')
    plt.savefig(fname, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved {fname}")

    # Extract best per target
    with torch.no_grad():
        final_sims = batched_tn_sim_1layer(w_l, w_r, w_p, tL, tR, tD)
        final_sims = final_sims.view(n_seeds, n_targets)
        best_idx = final_sims.argmax(dim=0)

        result_sims = {}
        result_weights = {}
        for t_idx, label in enumerate(labels):
            bs = best_idx[t_idx].item()
            b_idx = bs * n_targets + t_idx
            result_sims[label] = final_sims[bs, t_idx].item()
            result_weights[label] = {
                'w_l': w_l[b_idx].detach().clone(),
                'w_r': w_r[b_idx].detach().clone(),
                'w_p': w_p[b_idx].detach().clone(),
            }

    return result_sims, result_weights


# =============================================================================
# VISUALIZATION
# =============================================================================

def visualize_per_class_patterns(class_sims, class_weights, device='cuda'):
    """Visualize rank-1 patterns for each of the 10 classes."""
    fig, axes = plt.subplots(10, 4, figsize=(10, 22))
    fig.suptitle('Per-Class Rank-1 Patterns: L, R, L+R, L-R', fontsize=14)

    sorted_classes = sorted(class_sims.items(), key=lambda x: x[1], reverse=True)

    for row, (cls, sim) in enumerate(sorted_classes):
        w = class_weights[cls]
        l = w['w_l'][0, :784].cpu().numpy()
        r = w['w_r'][0, :784].cpu().numpy()

        imgs = [l.reshape(28, 28), r.reshape(28, 28),
                (l + r).reshape(28, 28), (l - r).reshape(28, 28)]
        titles = ['L', 'R', 'L+R', 'L-R']

        for col in range(4):
            ax = axes[row, col]
            im = ax.imshow(imgs[col], cmap='RdBu_r', aspect='equal')
            vabs = max(abs(imgs[col].min()), abs(imgs[col].max()))
            im.set_clim(-vabs, vabs)
            if row == 0:
                ax.set_title(titles[col], fontsize=10)
            ax.set_xticks([]); ax.set_yticks([])
            if col == 0:
                ax.set_ylabel(f'Class {cls}\nsim={sim:.3f}', fontsize=9)

    plt.tight_layout()
    fname = os.path.join(IMG_DIR, 'residual_per_class_patterns.png')
    plt.savefig(fname, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved {fname}")


def visualize_pair_patterns(pair_sims_dict, pair_weights, n_show=10, device='cuda'):
    """Visualize rank-1 patterns for top and bottom pairs."""
    sorted_pairs = sorted(pair_sims_dict.items(), key=lambda x: x[1], reverse=True)
    show = [p for p, _ in sorted_pairs[:n_show//2]] + [p for p, _ in sorted_pairs[-n_show//2:]]

    fig, axes = plt.subplots(len(show), 5, figsize=(12, 2.2 * len(show)))
    fig.suptitle('Rank-1 Pair Patterns: L, R, (L+R), (L-R), D weights', fontsize=12)

    for row, key in enumerate(show):
        sim = pair_sims_dict[key]
        w = pair_weights[key]
        l = w['w_l'][0, :784].cpu().numpy()
        r = w['w_r'][0, :784].cpu().numpy()
        d = w['w_p'][:, 0].cpu().numpy()

        imgs = [l.reshape(28, 28), r.reshape(28, 28),
                (l + r).reshape(28, 28), (l - r).reshape(28, 28)]
        titles = ['L', 'R', 'L+R', 'L-R']

        ci, cj = key if isinstance(key, tuple) else (key, key)

        for col in range(4):
            ax = axes[row, col]
            im = ax.imshow(imgs[col], cmap='RdBu_r', aspect='equal')
            vabs = max(abs(imgs[col].min()), abs(imgs[col].max()))
            im.set_clim(-vabs, vabs)
            if row == 0:
                ax.set_title(titles[col], fontsize=10)
            ax.set_xticks([]); ax.set_yticks([])
            if col == 0:
                ax.set_ylabel(f'{ci}-{cj}\nsim={sim:.3f}', fontsize=9)

        ax = axes[row, 4]
        ax.barh(range(len(d)), d, color=['C0', 'C1'][:len(d)])
        ax.set_yticks(range(len(d)))
        ax.set_yticklabels([f'cls {ci}', f'cls {cj}'], fontsize=8)
        ax.axvline(0, color='k', linewidth=0.5)
        if row == 0:
            ax.set_title('D weights', fontsize=10)

    plt.tight_layout()
    fname = os.path.join(IMG_DIR, 'residual_pair_patterns.png')
    plt.savefig(fname, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved {fname}")


# =============================================================================
# MAIN
# =============================================================================

if __name__ == '__main__':
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Device: {device}")

    # ==== Train model ====
    print("\n" + "=" * 70)
    print("Training residual bilinear model")
    print("=" * 70)

    model = ResidualBilinearMNIST(784, 64, 32, 10).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=0.01)
    train_loader, test_loader = get_mnist_loaders(256)

    for epoch in range(10):
        model.train()
        total_loss = 0
        for data, target in train_loader:
            data, target = data.to(device), target.to(device)
            x = data.view(data.size(0), -1)
            loss = F.cross_entropy(model(x), target)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        if (epoch + 1) % 2 == 0 or epoch == 0:
            test_acc = eval_accuracy(model, test_loader, device)
            print(f"  Epoch {epoch+1}/10: loss={total_loss/len(train_loader):.4f} test={test_acc:.4f}")

    test_acc = eval_accuracy(model, test_loader, device)
    print(f"  Final test accuracy: {test_acc:.4f}")

    # ==== Per-class rank-1 ====
    print("\n" + "=" * 70)
    print("Per-class rank-1 optimization (10 classes)")
    print("=" * 70)

    class_Ls, class_Rs, class_Ds, class_labels = [], [], [], []
    for c in range(10):
        L, R, D = get_augmented_weights_for_class(model, c)
        class_Ls.append(L)
        class_Rs.append(R)
        class_Ds.append(D)
        class_labels.append(c)

    class_sims, class_weights = run_batched_optimization(
        class_Ls, class_Rs, class_Ds, class_labels,
        n_seeds=5, steps=1000, lr=1e-3, device=device,
        title_prefix='residual_per_class'
    )

    print("\n  Per-class rank-1 TN-sim:")
    for c in range(10):
        print(f"    Class {c}: {class_sims[c]:.4f}")

    visualize_per_class_patterns(class_sims, class_weights, device)

    # ==== Pairwise rank-1 ====
    print("\n" + "=" * 70)
    print("Pairwise rank-1 optimization (45 pairs)")
    print("=" * 70)

    pairs = list(combinations(range(10), 2))
    pair_Ls, pair_Rs, pair_Ds, pair_labels = [], [], [], []
    for ci, cj in pairs:
        L, R, D = model.get_augmented_weights_for_pair(ci, cj)
        pair_Ls.append(L)
        pair_Rs.append(R)
        pair_Ds.append(D)
        pair_labels.append((ci, cj))

    pair_sims, pair_weights = run_batched_optimization(
        pair_Ls, pair_Rs, pair_Ds, pair_labels,
        n_seeds=5, steps=1000, lr=1e-3, device=device,
        title_prefix='residual_pairwise'
    )

    # Print summary
    sorted_pairs = sorted(pair_sims.items(), key=lambda x: x[1], reverse=True)
    print("\n  Top 10 easiest pairs:")
    for (i, j), sim in sorted_pairs[:10]:
        print(f"    {i}-{j}: {sim:.4f}")
    print("  Bottom 10 hardest pairs:")
    for (i, j), sim in sorted_pairs[-10:]:
        print(f"    {i}-{j}: {sim:.4f}")

    # Spectral baseline
    spectral = {}
    with torch.no_grad():
        for idx, (ci, cj) in enumerate(pairs):
            L_aug, R_aug, D_aug = pair_Ls[idx], pair_Rs[idx], pair_Ds[idx]
            LL = L_aug @ L_aug.T; RR = R_aug @ R_aug.T; LR = L_aug @ R_aug.T
            core = 0.5 * (LL * RR + LR * LR.T)
            DD = D_aug.T @ D_aug
            M = DD * core
            eigvals = torch.linalg.eigvalsh(M).clamp(min=0)
            total = eigvals.sum().item()
            top = eigvals[-1].item()
            spectral[(ci, cj)] = {'top_eig_frac': top / (total + 1e-12),
                                   'norm': np.sqrt(max(total, 0))}

    plot_pairwise_results(pair_sims, spectral, 'Residual',
                          os.path.join(IMG_DIR, 'residual_pairwise_tnsim.png'))

    visualize_pair_patterns(pair_sims, pair_weights, n_show=10, device=device)

    # ==== Save ====
    torch.save({
        'model_state': model.state_dict(),
        'class_sims': class_sims,
        'pair_sims': pair_sims,
        'spectral': spectral,
    }, os.path.join(os.path.dirname(__file__), 'residual_results.pt'))
    print("\nSaved to residual_results.pt")
    print("Done!")
