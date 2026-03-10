"""
Behavioral faithfulness analysis for FROZEN-RESIDUAL + rank-1 models.

The approximation is: exact residual + rank-1 bilinear correction
  f_approx(x) = W_res @ x + d @ ((l @ x̂) * (r @ x̂))

Compare to full model on actual MNIST images.
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
    ResidualBilinearMNIST, get_mnist_loaders,
)
from run_frozen_residual import build_residual_parts

IMG_DIR = os.path.join(os.path.dirname(__file__), 'images', '3_frozen_residual')
os.makedirs(IMG_DIR, exist_ok=True)


def compute_frozen_residual_output(W_res, l, r, d, x_flat):
    """
    Compute frozen-residual + rank-1 output on real images.
    W_res: (n_out, 784) - frozen residual weights
    l: (1, 785), r: (1, 785), d: (n_out, 1) - rank-1 correction
    x_flat: (N, 784)
    Returns: (N, n_out)
    """
    # Residual part
    residual = x_flat @ W_res.T  # (N, n_out)

    # Bilinear correction
    x_hat = torch.cat([x_flat, torch.ones(x_flat.shape[0], 1, device=x_flat.device)], dim=1)
    left = x_hat @ l.T    # (N, 1)
    right = x_hat @ r.T   # (N, 1)
    hidden = left * right  # (N, 1)
    bilinear = hidden @ d.T  # (N, n_out)

    return residual + bilinear


def optimal_rescale(full_out, approx_out):
    """Find optimal affine: alpha * approx + beta ≈ full. Per output dim."""
    rescaled = torch.zeros_like(approx_out)
    for d in range(full_out.shape[1]):
        y = full_out[:, d]
        x = approx_out[:, d]
        X = torch.stack([x, torch.ones_like(x)], dim=1)
        coeffs = torch.linalg.lstsq(X, y).solution
        rescaled[:, d] = coeffs[0] * x + coeffs[1]
    return rescaled


def kl_div_per_sample(logits_full, logits_approx):
    """KL(full || approx) per sample."""
    p = F.softmax(logits_full, dim=1)
    log_p = F.log_softmax(logits_full, dim=1)
    log_q = F.log_softmax(logits_approx, dim=1)
    return (p * (log_p - log_q)).sum(dim=1)


def get_test_images_by_class(test_loader, device):
    all_images = {c: [] for c in range(10)}
    for data, target in test_loader:
        for c in range(10):
            mask = target == c
            if mask.any():
                all_images[c].append(data[mask].view(-1, 784))
    for c in range(10):
        all_images[c] = torch.cat(all_images[c]).to(device)
    return all_images


# =============================================================================
# PAIRWISE FAITHFULNESS
# =============================================================================

def analyze_pairwise_faithfulness(full_model, pair_weights, pair_sims,
                                   test_loader, device, n_show=8):
    all_images = get_test_images_by_class(test_loader, device)
    full_model.eval()

    sorted_pairs = sorted(pair_sims.items(), key=lambda x: x[1], reverse=True)
    pairs_to_analyze = [p for p, _ in sorted_pairs[:5]] + [p for p, _ in sorted_pairs[-5:]]
    n_pairs = len(pairs_to_analyze)

    all_kl_rescaled = {}
    all_acc_match = {}

    fig_examples, axes_ex = plt.subplots(n_pairs, 2 + n_show,
                                          figsize=(2 * (2 + n_show), n_pairs * 2))
    fig_hist, axes_h = plt.subplots(2, 5, figsize=(20, 8))
    fig_scatter, axes_s = plt.subplots(2, 5, figsize=(20, 8))

    for p_idx, (ci, cj) in enumerate(pairs_to_analyze):
        w = pair_weights[(ci, cj)]
        l, r, d = w['l'], w['r'], w['d']

        # Get W_res for this pair
        W_res = (full_model.W_head[[ci, cj]] @ full_model.W_embed).detach()

        imgs = torch.cat([all_images[ci], all_images[cj]])
        labels = torch.cat([torch.full((len(all_images[ci]),), ci, device=device),
                           torch.full((len(all_images[cj]),), cj, device=device)])

        with torch.no_grad():
            full_out = full_model(imgs)[:, [ci, cj]]
            approx_out = compute_frozen_residual_output(W_res, l, r, d, imgs)
            rescaled_out = optimal_rescale(full_out, approx_out)

        kl_raw = kl_div_per_sample(full_out, approx_out).cpu().numpy()
        kl_rescaled = kl_div_per_sample(full_out, rescaled_out).cpu().numpy()

        full_pred = full_out.argmax(dim=1)
        rank1_pred = rescaled_out.argmax(dim=1)
        acc_match = (full_pred == rank1_pred).float().mean().item()

        all_kl_rescaled[(ci, cj)] = np.median(kl_rescaled)
        all_acc_match[(ci, cj)] = acc_match

        # Histogram
        ax = axes_h[p_idx // 5, p_idx % 5]
        ax.hist(kl_rescaled, bins=50, edgecolor='k', alpha=0.7, color='C0',
                density=True, label='rescaled')
        ax.hist(kl_raw, bins=50, edgecolor='k', alpha=0.3, color='C1',
                density=True, label='raw')
        ax.set_title(f'{ci}-{cj} (sim={pair_sims[(ci,cj)]:.3f})\nacc={acc_match:.1%}', fontsize=9)
        ax.set_xlabel('KL div', fontsize=8)
        ax.legend(fontsize=6)
        if p_idx % 5 == 0:
            ax.set_ylabel('Density', fontsize=8)

        # Scatter: logit diff
        full_diff = (full_out[:, 0] - full_out[:, 1]).cpu().numpy()
        rank1_diff = (rescaled_out[:, 0] - rescaled_out[:, 1]).cpu().numpy()
        corr = np.corrcoef(full_diff, rank1_diff)[0, 1]

        ax = axes_s[p_idx // 5, p_idx % 5]
        colors = ['C0' if l_ == ci else 'C1' for l_ in labels.cpu().numpy()]
        ax.scatter(full_diff, rank1_diff, s=3, alpha=0.3, c=colors)
        mn = min(full_diff.min(), rank1_diff.min())
        mx = max(full_diff.max(), rank1_diff.max())
        ax.plot([mn, mx], [mn, mx], 'r--', alpha=0.5, linewidth=1)
        ax.set_title(f'{ci}-{cj} (r={corr:.3f})', fontsize=9)
        ax.set_xlabel(f'Full: logit({ci})-logit({cj})', fontsize=7)
        ax.set_ylabel(f'Frozen-res+rank1 (rescaled)', fontsize=7)

        # Example images
        order = np.argsort(kl_rescaled)
        imgs_np = imgs.cpu().numpy()
        half = n_show // 2

        ax = axes_ex[p_idx, 0]
        ax.hist(kl_rescaled, bins=30, edgecolor='k', alpha=0.7, density=True)
        ax.axvline(np.median(kl_rescaled), color='r', linestyle='--')
        ax.set_title('KL(rescaled)', fontsize=6)
        ax.set_ylabel(f'{ci}-{cj}\nsim={pair_sims[(ci,cj)]:.2f}\nacc={acc_match:.0%}', fontsize=7)
        ax.tick_params(labelsize=5)

        ax = axes_ex[p_idx, 1]
        ax.scatter(full_diff, rank1_diff, s=1, alpha=0.2, c=colors)
        ax.plot([mn, mx], [mn, mx], 'r--', alpha=0.5, linewidth=0.5)
        ax.set_title(f'r={corr:.2f}', fontsize=6)
        ax.tick_params(labelsize=5)

        for k in range(half):
            idx = order[k]
            ax = axes_ex[p_idx, 2 + k]
            ax.imshow(imgs_np[idx].reshape(28, 28), cmap='gray')
            ax.set_xticks([]); ax.set_yticks([])
            lbl = labels[idx].item()
            ax.set_title(f'{lbl} KL={kl_rescaled[idx]:.3f}', fontsize=5, color='green')

        for k in range(half):
            idx = order[-(k+1)]
            ax = axes_ex[p_idx, 2 + half + k]
            ax.imshow(imgs_np[idx].reshape(28, 28), cmap='gray')
            ax.set_xticks([]); ax.set_yticks([])
            lbl = labels[idx].item()
            ax.set_title(f'{lbl} KL={kl_rescaled[idx]:.3f}', fontsize=5, color='red')

    fig_hist.suptitle('Frozen-Res+Rank1: KL Divergence (raw=orange, rescaled=blue)', fontsize=14)
    fig_hist.tight_layout()
    fig_hist.savefig(os.path.join(IMG_DIR, 'pair_kl_histograms.png'), dpi=150, bbox_inches='tight')
    plt.close(fig_hist)

    fig_scatter.suptitle('Frozen-Res+Rank1: Full vs Approx Logit Difference', fontsize=14)
    fig_scatter.tight_layout()
    fig_scatter.savefig(os.path.join(IMG_DIR, 'pair_logit_scatter.png'), dpi=150, bbox_inches='tight')
    plt.close(fig_scatter)

    fig_examples.suptitle('Frozen-Res+Rank1 Pairwise: KL | scatter | faithful (green) | unfaithful (red)', fontsize=11)
    fig_examples.tight_layout()
    fig_examples.savefig(os.path.join(IMG_DIR, 'pair_faithfulness_grid.png'), dpi=150, bbox_inches='tight')
    plt.close(fig_examples)

    print(f"  Saved pair_kl_histograms.png, pair_logit_scatter.png, pair_faithfulness_grid.png")

    print(f"\n  Pairwise faithfulness summary (rescaled):")
    print(f"  {'Pair':>6s}  {'TN-sim':>7s}  {'KL(med)':>8s}  {'Acc match':>10s}")
    for (ci, cj) in pairs_to_analyze:
        print(f"  {ci}-{cj:>4d}  {pair_sims[(ci,cj)]:>7.3f}  "
              f"{all_kl_rescaled[(ci,cj)]:>8.4f}  {all_acc_match[(ci,cj)]:>10.1%}")


# =============================================================================
# PER-CLASS FAITHFULNESS
# =============================================================================

def analyze_per_class_faithfulness(full_model, class_weights, class_sims,
                                    test_loader, device, n_show=8):
    all_images = get_test_images_by_class(test_loader, device)
    full_model.eval()

    sorted_classes = sorted(class_sims.items(), key=lambda x: x[1], reverse=True)
    half = n_show // 2

    fig_grid, axes_grid = plt.subplots(10, 2 + n_show, figsize=(2 * (2 + n_show), 10 * 2))
    fig_corr, axes_corr = plt.subplots(2, 5, figsize=(20, 8))

    for row, (cls, sim) in enumerate(sorted_classes):
        w = class_weights[cls]
        l, r, d = w['l'], w['r'], w['d']

        # W_res for single class
        W_res = (full_model.W_head[[cls]] @ full_model.W_embed).detach()

        imgs = all_images[cls]

        with torch.no_grad():
            full_logit = full_model(imgs)[:, cls]  # (N,)
            approx_out = compute_frozen_residual_output(W_res, l, r, d, imgs).squeeze(1)  # (N,)

        # Optimal rescaling
        X = torch.stack([approx_out, torch.ones_like(approx_out)], dim=1)
        coeffs = torch.linalg.lstsq(X, full_logit).solution
        rescaled = coeffs[0] * approx_out + coeffs[1]

        full_np = full_logit.cpu().numpy()
        rank1_np = rescaled.cpu().numpy()
        corr = np.corrcoef(full_np, rank1_np)[0, 1]
        residual = np.abs(full_np - rank1_np)
        order = np.argsort(residual)

        # Correlation scatter
        ax = axes_corr[row // 5, row % 5]
        ax.scatter(full_np, rank1_np, s=3, alpha=0.3)
        mn = min(full_np.min(), rank1_np.min())
        mx = max(full_np.max(), rank1_np.max())
        ax.plot([mn, mx], [mn, mx], 'r--', alpha=0.5, linewidth=1)
        ax.set_xlabel('Full logit', fontsize=8)
        ax.set_ylabel('Frozen-res+rank1 (rescaled)', fontsize=8)
        ax.set_title(f'Class {cls} (sim={sim:.3f}, r={corr:.3f})', fontsize=9)

        imgs_np = imgs.cpu().numpy()

        # Grid row
        ax = axes_grid[row, 0]
        ax.hist(residual, bins=30, edgecolor='k', alpha=0.7, density=True)
        ax.axvline(np.median(residual), color='r', linestyle='--')
        ax.set_title(f'|residual|', fontsize=7)
        ax.set_ylabel(f'Cls {cls}\nsim={sim:.2f}\nr={corr:.2f}', fontsize=8)

        ax = axes_grid[row, 1]
        ax.scatter(full_np, rank1_np, s=1, alpha=0.2)
        ax.plot([mn, mx], [mn, mx], 'r--', alpha=0.5, linewidth=0.5)
        ax.set_title(f'r={corr:.2f}', fontsize=7)

        for k in range(half):
            idx = order[k]
            ax = axes_grid[row, 2 + k]
            ax.imshow(imgs_np[idx].reshape(28, 28), cmap='gray')
            ax.set_xticks([]); ax.set_yticks([])
            ax.set_title(f'res={residual[idx]:.1f}', fontsize=6, color='green')

            idx = order[-(k+1)]
            ax = axes_grid[row, 2 + half + k]
            ax.imshow(imgs_np[idx].reshape(28, 28), cmap='gray')
            ax.set_xticks([]); ax.set_yticks([])
            ax.set_title(f'res={residual[idx]:.1f}', fontsize=6, color='red')

    fig_grid.suptitle('Frozen-Res+Rank1 Per-Class: histogram | scatter | faithful | unfaithful', fontsize=12)
    fig_grid.tight_layout()
    fig_grid.savefig(os.path.join(IMG_DIR, 'class_faithfulness_grid.png'), dpi=150, bbox_inches='tight')
    plt.close(fig_grid)

    fig_corr.suptitle('Frozen-Res+Rank1: Full Logit vs Approx (rescaled, per class)', fontsize=14)
    fig_corr.tight_layout()
    fig_corr.savefig(os.path.join(IMG_DIR, 'class_logit_correlation.png'), dpi=150, bbox_inches='tight')
    plt.close(fig_corr)

    print(f"  Saved class_faithfulness_grid.png, class_logit_correlation.png")


# =============================================================================
# CROSS-CLASS
# =============================================================================

def analyze_cross_class_faithfulness(full_model, class_weights, class_sims,
                                      test_loader, device):
    all_images = get_test_images_by_class(test_loader, device)
    full_model.eval()

    corr_matrix = np.zeros((10, 10))

    for rank1_cls in range(10):
        w = class_weights[rank1_cls]
        l, r, d = w['l'], w['r'], w['d']
        W_res = (full_model.W_head[[rank1_cls]] @ full_model.W_embed).detach()

        for img_cls in range(10):
            imgs = all_images[img_cls]
            with torch.no_grad():
                full_logit = full_model(imgs)[:, rank1_cls].cpu().numpy()
                approx_out = compute_frozen_residual_output(
                    W_res, l, r, d, imgs
                ).squeeze(1).cpu().numpy()

            if full_logit.std() > 1e-8 and approx_out.std() > 1e-8:
                corr_matrix[rank1_cls, img_cls] = np.corrcoef(full_logit, approx_out)[0, 1]

    fig, ax = plt.subplots(figsize=(8, 7))
    im = ax.imshow(corr_matrix, cmap='RdBu_r', vmin=-1, vmax=1)
    ax.set_xlabel('Image class')
    ax.set_ylabel('Rank-1 model (class)')
    ax.set_xticks(range(10)); ax.set_yticks(range(10))
    ax.set_title('Frozen-Res+Rank1: Cross-class correlation\n(approx output vs full logit on other classes)')
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
# MOST/LEAST FAITHFUL PAIRS
# =============================================================================

def plot_most_least_faithful_pairs(full_model, pair_weights, pair_sims,
                                    test_loader, device, n_show=5):
    """Show the most and least faithful pairs side by side."""
    all_images = get_test_images_by_class(test_loader, device)
    full_model.eval()

    sorted_pairs = sorted(pair_sims.items(), key=lambda x: x[1], reverse=True)
    most = sorted_pairs[:n_show]
    least = sorted_pairs[-n_show:]

    for tag, subset in [('most', most), ('least', least)]:
        fig, axes = plt.subplots(n_show, 4, figsize=(16, n_show * 3))
        for row, ((ci, cj), sim) in enumerate(subset):
            w = pair_weights[(ci, cj)]
            l, r, d = w['l'], w['r'], w['d']
            W_res = (full_model.W_head[[ci, cj]] @ full_model.W_embed).detach()

            imgs = torch.cat([all_images[ci], all_images[cj]])
            labels = torch.cat([torch.full((len(all_images[ci]),), ci, device=device),
                               torch.full((len(all_images[cj]),), cj, device=device)])

            with torch.no_grad():
                full_out = full_model(imgs)[:, [ci, cj]]
                approx_out = compute_frozen_residual_output(W_res, l, r, d, imgs)
                rescaled_out = optimal_rescale(full_out, approx_out)

            full_diff = (full_out[:, 0] - full_out[:, 1]).cpu().numpy()
            rank1_diff = (rescaled_out[:, 0] - rescaled_out[:, 1]).cpu().numpy()
            corr = np.corrcoef(full_diff, rank1_diff)[0, 1]

            kl = kl_div_per_sample(full_out, rescaled_out).cpu().numpy()
            acc = (full_out.argmax(1) == rescaled_out.argmax(1)).float().mean().item()

            colors = ['C0' if lb == ci else 'C1' for lb in labels.cpu().numpy()]

            # Scatter: logit diff
            ax = axes[row, 0]
            ax.scatter(full_diff, rank1_diff, s=3, alpha=0.3, c=colors)
            mn = min(full_diff.min(), rank1_diff.min())
            mx = max(full_diff.max(), rank1_diff.max())
            ax.plot([mn, mx], [mn, mx], 'r--', alpha=0.5)
            ax.set_title(f'{ci}-{cj}: r={corr:.3f}', fontsize=10)
            ax.set_ylabel(f'sim={sim:.3f}', fontsize=9)
            ax.set_xlabel('Full logit diff', fontsize=8)

            # Scatter: raw (no rescaling)
            ax = axes[row, 1]
            raw_diff = (approx_out[:, 0] - approx_out[:, 1]).cpu().numpy()
            corr_raw = np.corrcoef(full_diff, raw_diff)[0, 1]
            ax.scatter(full_diff, raw_diff, s=3, alpha=0.3, c=colors)
            ax.set_title(f'raw: r={corr_raw:.3f}', fontsize=10)
            ax.set_xlabel('Full logit diff', fontsize=8)
            ax.set_ylabel('Approx (raw)', fontsize=8)

            # KL histogram
            ax = axes[row, 2]
            ax.hist(kl, bins=50, edgecolor='k', alpha=0.7)
            ax.axvline(np.median(kl), color='r', linestyle='--')
            ax.set_title(f'KL: med={np.median(kl):.3f}', fontsize=10)
            ax.set_xlabel('KL div', fontsize=8)

            # Stats text
            ax = axes[row, 3]
            ax.axis('off')
            stats = (f"Pair: {ci}-{cj}\n"
                     f"TN-sim: {sim:.4f}\n"
                     f"Corr (rescaled): {corr:.4f}\n"
                     f"Corr (raw): {corr_raw:.4f}\n"
                     f"Acc match: {acc:.1%}\n"
                     f"KL median: {np.median(kl):.4f}\n"
                     f"KL mean: {np.mean(kl):.4f}")
            ax.text(0.1, 0.5, stats, transform=ax.transAxes, fontsize=11,
                    verticalalignment='center', fontfamily='monospace')

        fig.suptitle(f'Frozen-Res+Rank1: {tag.capitalize()} faithful pairs', fontsize=14)
        fig.tight_layout()
        fig.savefig(os.path.join(IMG_DIR, f'pair_{tag}_faithful.png'), dpi=150, bbox_inches='tight')
        plt.close(fig)

    print(f"  Saved pair_most_faithful.png, pair_least_faithful.png")


# =============================================================================
# MAIN
# =============================================================================

if __name__ == '__main__':
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Device: {device}")

    base_dir = os.path.dirname(__file__)

    # Load model
    model_results = torch.load(os.path.join(base_dir, 'residual_results.pt'),
                                map_location=device, weights_only=False)
    model = ResidualBilinearMNIST(784, 64, 32, 10).to(device)
    model.load_state_dict(model_results['model_state'])
    model.eval()

    # Load frozen residual results
    frozen = torch.load(os.path.join(base_dir, 'frozen_residual_results.pt'),
                        map_location=device, weights_only=False)

    class_sims = frozen['class_sims']
    class_weights = frozen['class_weights']
    pair_sims = frozen['pair_sims']
    pair_weights = frozen['pair_weights']

    _, test_loader = get_mnist_loaders(256)

    print("\n" + "=" * 70)
    print("Frozen-Res+Rank1: Pairwise faithfulness")
    print("=" * 70)
    analyze_pairwise_faithfulness(model, pair_weights, pair_sims,
                                  test_loader, device, n_show=8)

    print("\n" + "=" * 70)
    print("Frozen-Res+Rank1: Most/Least faithful pairs")
    print("=" * 70)
    plot_most_least_faithful_pairs(model, pair_weights, pair_sims,
                                   test_loader, device)

    print("\n" + "=" * 70)
    print("Frozen-Res+Rank1: Per-class faithfulness")
    print("=" * 70)
    analyze_per_class_faithfulness(model, class_weights, class_sims,
                                   test_loader, device, n_show=8)

    print("\n" + "=" * 70)
    print("Frozen-Res+Rank1: Cross-class interference")
    print("=" * 70)
    corr_matrix = analyze_cross_class_faithfulness(model, class_weights, class_sims,
                                                    test_loader, device)

    print("\nDone! All images saved to images/3_frozen_residual/")
