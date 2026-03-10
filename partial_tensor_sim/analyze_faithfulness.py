"""
Behavioral faithfulness analysis for rank-1 TN-sim approximations.

For each rank-1 model, compare its output to the full model on actual MNIST images:
- KL divergence on relevant class logits
- Histograms of faithfulness
- Most/least faithful example images

The rank-1 models live in augmented 785-dim space (x̂ = [x, 1]).
Their output for a pair (i,j) is: logits = D @ ((L @ x̂) * (R @ x̂))  shape (2,)
The full model's output for that pair is: full_logits[[i,j]]  shape (2,)
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
)

IMG_DIR = os.path.join(os.path.dirname(__file__), 'images')
os.makedirs(IMG_DIR, exist_ok=True)


def compute_rank1_output(w_l, w_r, w_p, x_flat):
    """
    Compute rank-1 bilinear output on real images.
    w_l: (1, 785), w_r: (1, 785), w_p: (n_out, 1)
    x_flat: (N, 784)
    Returns: (N, n_out) logits
    """
    # Augment input
    x_hat = torch.cat([x_flat, torch.ones(x_flat.shape[0], 1, device=x_flat.device)], dim=1)
    left = x_hat @ w_l.T   # (N, 1)
    right = x_hat @ w_r.T  # (N, 1)
    hidden = left * right   # (N, 1)
    return hidden @ w_p.T  # (N, n_out)


def kl_div_per_sample(logits_full, logits_approx):
    """
    KL(full || approx) per sample. Both are (N, n_out) logits.
    Returns (N,) tensor.
    """
    p = F.softmax(logits_full, dim=1)
    log_p = F.log_softmax(logits_full, dim=1)
    log_q = F.log_softmax(logits_approx, dim=1)
    return (p * (log_p - log_q)).sum(dim=1)


def analyze_pairwise_faithfulness(full_model, pair_weights, pair_sims,
                                   test_loader, device, n_show=8):
    """
    For each class pair, compute KL divergence between full and rank-1 outputs
    on images from those two classes. Show most/least faithful examples.
    """
    # Collect all test images by class
    all_images = {c: [] for c in range(10)}
    for data, target in test_loader:
        for c in range(10):
            mask = target == c
            if mask.any():
                all_images[c].append(data[mask].view(-1, 784))
    for c in range(10):
        all_images[c] = torch.cat(all_images[c]).to(device)

    # Get full model outputs for all test images
    full_model.eval()
    all_flat = torch.cat([all_images[c] for c in range(10)])
    with torch.no_grad():
        full_logits_all = full_model(all_flat)

    # Build index mapping
    offsets = {}
    idx = 0
    for c in range(10):
        offsets[c] = (idx, idx + len(all_images[c]))
        idx += len(all_images[c])

    # Analyze top pairs + bottom pairs
    sorted_pairs = sorted(pair_sims.items(), key=lambda x: x[1], reverse=True)
    pairs_to_analyze = [p for p, _ in sorted_pairs[:5]] + [p for p, _ in sorted_pairs[-5:]]

    n_pairs = len(pairs_to_analyze)
    fig_faithful, axes_f = plt.subplots(n_pairs, n_show, figsize=(n_show * 1.5, n_pairs * 1.8))
    fig_unfaithful, axes_u = plt.subplots(n_pairs, n_show, figsize=(n_show * 1.5, n_pairs * 1.8))
    fig_hist, axes_h = plt.subplots(2, 5, figsize=(20, 8))

    for p_idx, (ci, cj) in enumerate(pairs_to_analyze):
        w = pair_weights[(ci, cj)]
        w_l, w_r, w_p = w['w_l'], w['w_r'], w['w_p']

        # Get images from classes ci and cj
        imgs_i = all_images[ci]
        imgs_j = all_images[cj]
        imgs = torch.cat([imgs_i, imgs_j])
        labels = torch.cat([torch.full((len(imgs_i),), ci, device=device),
                           torch.full((len(imgs_j),), cj, device=device)])

        # Full model logits for this pair
        with torch.no_grad():
            full_out = full_model(imgs)[:, [ci, cj]]  # (N, 2)
            rank1_out = compute_rank1_output(w_l, w_r, w_p, imgs)  # (N, 2)

        # KL divergence
        kl = kl_div_per_sample(full_out, rank1_out).cpu().numpy()

        # Histogram
        ax = axes_h[p_idx // 5, p_idx % 5]
        ax.hist(kl, bins=50, edgecolor='k', alpha=0.7, density=True)
        ax.axvline(np.median(kl), color='r', linestyle='--', alpha=0.7)
        ax.set_title(f'{ci}-{cj} (sim={pair_sims[(ci,cj)]:.3f})', fontsize=9)
        ax.set_xlabel('KL div', fontsize=8)
        if p_idx % 5 == 0:
            ax.set_ylabel('Density', fontsize=8)

        # Sort by KL
        order = np.argsort(kl)
        imgs_np = imgs.cpu().numpy()

        # Most faithful (lowest KL)
        for k in range(min(n_show, len(order))):
            idx = order[k]
            ax = axes_f[p_idx, k]
            ax.imshow(imgs_np[idx].reshape(28, 28), cmap='gray')
            ax.set_xticks([]); ax.set_yticks([])
            lbl = labels[idx].item()
            ax.set_title(f'cls={lbl}\nKL={kl[idx]:.3f}', fontsize=7)
            if k == 0:
                ax.set_ylabel(f'{ci}-{cj}\nsim={pair_sims[(ci,cj)]:.2f}', fontsize=8)

        # Least faithful (highest KL)
        for k in range(min(n_show, len(order))):
            idx = order[-(k+1)]
            ax = axes_u[p_idx, k]
            ax.imshow(imgs_np[idx].reshape(28, 28), cmap='gray')
            ax.set_xticks([]); ax.set_yticks([])
            lbl = labels[idx].item()
            ax.set_title(f'cls={lbl}\nKL={kl[idx]:.3f}', fontsize=7)
            if k == 0:
                ax.set_ylabel(f'{ci}-{cj}\nsim={pair_sims[(ci,cj)]:.2f}', fontsize=8)

    fig_faithful.suptitle('Most Faithful Examples (lowest KL)', fontsize=14)
    fig_faithful.tight_layout()
    fig_faithful.savefig(os.path.join(IMG_DIR, 'pair_most_faithful.png'), dpi=150, bbox_inches='tight')
    plt.close(fig_faithful)

    fig_unfaithful.suptitle('Least Faithful Examples (highest KL)', fontsize=14)
    fig_unfaithful.tight_layout()
    fig_unfaithful.savefig(os.path.join(IMG_DIR, 'pair_least_faithful.png'), dpi=150, bbox_inches='tight')
    plt.close(fig_unfaithful)

    fig_hist.suptitle('KL Divergence Distributions (top 5 + bottom 5 pairs)', fontsize=14)
    fig_hist.tight_layout()
    fig_hist.savefig(os.path.join(IMG_DIR, 'pair_kl_histograms.png'), dpi=150, bbox_inches='tight')
    plt.close(fig_hist)

    print(f"  Saved pair_most_faithful.png, pair_least_faithful.png, pair_kl_histograms.png")


def analyze_per_class_faithfulness(full_model, class_weights, class_sims,
                                    test_loader, device, n_show=8):
    """
    For each class, compute KL divergence between full model's class-c logit
    and rank-1 approximation on images from that class.

    The rank-1 output is a scalar (1 logit). We compare:
    - Full model: logit_c(x) for all x in class c
    - Rank-1: its single output

    Since we can't do KL on a single logit, we instead measure:
    - Correlation between full logit_c and rank1 output across images
    - Residual (full - rank1) distribution
    - Show images where rank1 captures it best/worst
    """
    # Collect test images by class
    all_images = {c: [] for c in range(10)}
    for data, target in test_loader:
        for c in range(10):
            mask = target == c
            if mask.any():
                all_images[c].append(data[mask].view(-1, 784))
    for c in range(10):
        all_images[c] = torch.cat(all_images[c]).to(device)

    full_model.eval()

    fig_grid, axes_grid = plt.subplots(10, 2 + n_show, figsize=(2 * (2 + n_show), 10 * 1.8))
    fig_corr, axes_corr = plt.subplots(2, 5, figsize=(20, 8))

    sorted_classes = sorted(class_sims.items(), key=lambda x: x[1], reverse=True)

    for row, (cls, sim) in enumerate(sorted_classes):
        w = class_weights[cls]
        w_l, w_r, w_p = w['w_l'], w['w_r'], w['w_p']

        imgs = all_images[cls]  # (N, 784)

        with torch.no_grad():
            full_logit = full_model(imgs)[:, cls]  # (N,) — class-c logit
            rank1_out = compute_rank1_output(w_l, w_r, w_p, imgs).squeeze(1)  # (N,)

        full_np = full_logit.cpu().numpy()
        rank1_np = rank1_out.cpu().numpy()

        # Correlation
        corr = np.corrcoef(full_np, rank1_np)[0, 1]

        # Residual
        residual = np.abs(full_np - rank1_np)
        order = np.argsort(residual)

        # Scatter plot
        ax = axes_corr[row // 5, row % 5]
        ax.scatter(full_np, rank1_np, s=3, alpha=0.3)
        ax.set_xlabel('Full logit', fontsize=8)
        ax.set_ylabel('Rank-1 output', fontsize=8)
        ax.set_title(f'Class {cls} (sim={sim:.3f}, r={corr:.3f})', fontsize=9)
        # Add diagonal
        mn = min(full_np.min(), rank1_np.min())
        mx = max(full_np.max(), rank1_np.max())
        ax.plot([mn, mx], [mn, mx], 'r--', alpha=0.5, linewidth=1)

        imgs_np = imgs.cpu().numpy()

        # Grid: [histogram, scatter_mini, top faithful..., bottom faithful...]
        # Histogram of residuals
        ax = axes_grid[row, 0]
        ax.hist(residual, bins=30, edgecolor='k', alpha=0.7, density=True)
        ax.axvline(np.median(residual), color='r', linestyle='--')
        ax.set_title(f'|residual|', fontsize=7)
        ax.set_ylabel(f'Cls {cls}\nsim={sim:.2f}', fontsize=8)

        # Mini scatter
        ax = axes_grid[row, 1]
        ax.scatter(full_np, rank1_np, s=1, alpha=0.2)
        ax.plot([mn, mx], [mn, mx], 'r--', alpha=0.5, linewidth=0.5)
        ax.set_title(f'r={corr:.2f}', fontsize=7)

        # Most faithful (n_show/2) and least faithful (n_show/2)
        half = n_show // 2
        for k in range(half):
            # Most faithful
            idx = order[k]
            ax = axes_grid[row, 2 + k]
            ax.imshow(imgs_np[idx].reshape(28, 28), cmap='gray')
            ax.set_xticks([]); ax.set_yticks([])
            ax.set_title(f'res={residual[idx]:.2f}', fontsize=6)
            if row == 0:
                ax.set_xlabel('faithful', fontsize=7, color='green')

            # Least faithful
            idx = order[-(k+1)]
            ax = axes_grid[row, 2 + half + k]
            ax.imshow(imgs_np[idx].reshape(28, 28), cmap='gray')
            ax.set_xticks([]); ax.set_yticks([])
            ax.set_title(f'res={residual[idx]:.2f}', fontsize=6)
            if row == 0:
                ax.set_xlabel('unfaithful', fontsize=7, color='red')

    fig_grid.suptitle('Per-Class Rank-1 Faithfulness: histogram, correlation, faithful/unfaithful examples', fontsize=12)
    fig_grid.tight_layout()
    fig_grid.savefig(os.path.join(IMG_DIR, 'class_faithfulness_grid.png'), dpi=150, bbox_inches='tight')
    plt.close(fig_grid)

    fig_corr.suptitle('Full Logit vs Rank-1 Output (per class)', fontsize=14)
    fig_corr.tight_layout()
    fig_corr.savefig(os.path.join(IMG_DIR, 'class_logit_correlation.png'), dpi=150, bbox_inches='tight')
    plt.close(fig_corr)

    print(f"  Saved class_faithfulness_grid.png, class_logit_correlation.png")


def analyze_cross_class_faithfulness(full_model, class_weights, class_sims,
                                      test_loader, device):
    """
    For each per-class rank-1 (e.g. class 7), check how well it captures
    the class-7 logit on images from OTHER classes.
    This reveals interference structure.
    """
    all_images = {c: [] for c in range(10)}
    for data, target in test_loader:
        for c in range(10):
            mask = target == c
            if mask.any():
                all_images[c].append(data[mask].view(-1, 784))
    for c in range(10):
        all_images[c] = torch.cat(all_images[c]).to(device)

    full_model.eval()

    # For each rank-1 class model, compute correlation with full logit on every class
    corr_matrix = np.zeros((10, 10))  # [rank1_class, image_class]

    sorted_classes = sorted(class_sims.items(), key=lambda x: x[1], reverse=True)

    for rank1_cls, sim in sorted_classes:
        w = class_weights[rank1_cls]
        w_l, w_r, w_p = w['w_l'], w['w_r'], w['w_p']

        for img_cls in range(10):
            imgs = all_images[img_cls]
            with torch.no_grad():
                full_logit = full_model(imgs)[:, rank1_cls].cpu().numpy()
                rank1_out = compute_rank1_output(w_l, w_r, w_p, imgs).squeeze(1).cpu().numpy()

            if full_logit.std() > 1e-8 and rank1_out.std() > 1e-8:
                corr_matrix[rank1_cls, img_cls] = np.corrcoef(full_logit, rank1_out)[0, 1]

    fig, ax = plt.subplots(figsize=(8, 7))
    im = ax.imshow(corr_matrix, cmap='RdBu_r', vmin=-1, vmax=1)
    ax.set_xlabel('Image class')
    ax.set_ylabel('Rank-1 model (class)')
    ax.set_xticks(range(10)); ax.set_yticks(range(10))
    ax.set_title('Cross-class correlation: rank-1 output vs full logit')
    plt.colorbar(im, ax=ax, shrink=0.8)

    for i in range(10):
        for j in range(10):
            ax.text(j, i, f'{corr_matrix[i,j]:.2f}', ha='center', va='center',
                    fontsize=7, color='white' if abs(corr_matrix[i,j]) > 0.5 else 'black')

    plt.tight_layout()
    plt.savefig(os.path.join(IMG_DIR, 'cross_class_correlation.png'), dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved cross_class_correlation.png")

    return corr_matrix


# =============================================================================
# MAIN
# =============================================================================

if __name__ == '__main__':
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Device: {device}")

    # Load saved results
    results = torch.load(os.path.join(os.path.dirname(__file__), 'residual_results.pt'),
                         map_location=device, weights_only=False)

    # Reconstruct model
    model = ResidualBilinearMNIST(784, 64, 32, 10).to(device)
    model.load_state_dict(results['model_state'])
    model.eval()

    class_sims = results['class_sims']
    pair_sims = results['pair_sims']

    print(f"Loaded model and results")
    print(f"  Per-class sims: {[f'{c}:{s:.3f}' for c, s in sorted(class_sims.items())]}")

    # We need to re-run the rank-1 optimization to get the weights
    # (they weren't saved in the .pt file — let's re-optimize quickly)
    from run_residual import (
        run_batched_optimization, get_augmented_weights_for_class,
    )

    _, test_loader = get_mnist_loaders(256)

    # Per-class rank-1
    print("\nRe-optimizing per-class rank-1 (fast)...")
    class_Ls, class_Rs, class_Ds, class_labels = [], [], [], []
    for c in range(10):
        L, R, D = get_augmented_weights_for_class(model, c)
        class_Ls.append(L); class_Rs.append(R); class_Ds.append(D)
        class_labels.append(c)

    class_sims_new, class_weights = run_batched_optimization(
        class_Ls, class_Rs, class_Ds, class_labels,
        n_seeds=5, steps=1000, lr=1e-3, device=device,
        title_prefix='_tmp_class'
    )

    # Pairwise rank-1
    print("\nRe-optimizing pairwise rank-1 (fast)...")
    pairs = list(combinations(range(10), 2))
    pair_Ls, pair_Rs, pair_Ds, pair_labels = [], [], [], []
    for ci, cj in pairs:
        L, R, D = model.get_augmented_weights_for_pair(ci, cj)
        pair_Ls.append(L); pair_Rs.append(R); pair_Ds.append(D)
        pair_labels.append((ci, cj))

    pair_sims_new, pair_weights = run_batched_optimization(
        pair_Ls, pair_Rs, pair_Ds, pair_labels,
        n_seeds=5, steps=1000, lr=1e-3, device=device,
        title_prefix='_tmp_pair'
    )

    # ============================================================
    # Faithfulness analysis
    # ============================================================
    print("\n" + "=" * 70)
    print("Pairwise faithfulness analysis")
    print("=" * 70)
    analyze_pairwise_faithfulness(model, pair_weights, pair_sims_new,
                                  test_loader, device, n_show=8)

    print("\n" + "=" * 70)
    print("Per-class faithfulness analysis")
    print("=" * 70)
    analyze_per_class_faithfulness(model, class_weights, class_sims_new,
                                   test_loader, device, n_show=8)

    print("\n" + "=" * 70)
    print("Cross-class interference analysis")
    print("=" * 70)
    corr_matrix = analyze_cross_class_faithfulness(model, class_weights, class_sims_new,
                                                    test_loader, device)

    # Clean up temp convergence plots
    for f in ['_tmp_class_convergence.png', '_tmp_pair_convergence.png']:
        p = os.path.join(IMG_DIR, f)
        if os.path.exists(p):
            os.remove(p)

    print("\nDone! All plots in images/")
