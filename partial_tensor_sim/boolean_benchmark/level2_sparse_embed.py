"""
Level 2: Sparse embedding analysis with staged pruning.

3 AND features (A=x0^x1, B=x2^x3, C=x4^x5), 8 classes = binary encoding of (A,B,C).
Train with learned embedding, then progressively prune to top-2 and top-1 per row.
At each stage: bilinear TN-sim, rank-1 optimization, ground truth correlation.
"""
import torch
import torch.nn.functional as F
import numpy as np
import os
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy.stats import pearsonr
from model import (
    BooleanBilinear1Layer, make_level2_dataset, train_model,
    tn_inner_1layer, tn_sim_1layer, optimize_rank1,
)

IMG_DIR = os.path.join(os.path.dirname(__file__), 'images', 'level2')
os.makedirs(IMG_DIR, exist_ok=True)
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

N_CLASSES = 8
CLASS_LABELS = ['\u2205', 'A', 'B', 'AB', 'C', 'AC', 'BC', 'ABC']
INPUT_LABELS = ['x0', 'x1', 'x2', 'x3', 'x4', 'x5']
FEATURES_MAP = {
    0: set(), 1: {'A'}, 2: {'B'}, 3: {'A', 'B'},
    4: {'C'}, 5: {'A', 'C'}, 6: {'B', 'C'}, 7: {'A', 'B', 'C'},
}


# =========================================================================
# HELPERS
# =========================================================================

def train_with_embed_mask(model, X, labels, embed_mask, epochs, lr, wd):
    """Train model while enforcing embed sparsity mask."""
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=wd)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, epochs)
    for epoch in range(epochs):
        logits = model(X)
        loss = F.cross_entropy(logits, labels)
        optimizer.zero_grad()
        loss.backward()
        if embed_mask is not None and model.embed.weight.grad is not None:
            model.embed.weight.grad *= embed_mask
        optimizer.step()
        with torch.no_grad():
            if embed_mask is not None:
                model.embed.weight.data *= embed_mask
        scheduler.step()
    with torch.no_grad():
        return (model(X).argmax(1) == labels).float().mean().item()


def prune_embed_topk(model, k):
    """Prune embedding to top-k nonzero entries per row."""
    embed_mask = torch.zeros_like(model.embed.weight)
    with torch.no_grad():
        W = model.embed.weight.abs()
        for h in range(W.shape[0]):
            if (W[h] > 1e-8).sum() > k:
                _, top_idx = W[h].topk(k)
                embed_mask[h, top_idx] = 1.0
            else:
                embed_mask[h] = (W[h].abs() > 1e-8).float()
        model.embed.weight.data *= embed_mask
    return embed_mask


def get_bilinear_effective_weights(model, c):
    """Bilinear-only effective weights in input space for class c."""
    with torch.no_grad():
        W_e = model.W_embed
        W_h = model.W_head
        L_eff = model.L @ W_e       # (rank, n_input)
        R_eff = model.R @ W_e       # (rank, n_input)
        D_eff = W_h[[c]] @ model.D  # (1, rank)
    return L_eff, R_eff, D_eff


def compute_ground_truth_jaccard():
    """8x8 Jaccard similarity from feature sets."""
    gt = np.zeros((N_CLASSES, N_CLASSES))
    for i in range(N_CLASSES):
        for j in range(N_CLASSES):
            shared = len(FEATURES_MAP[i] & FEATURES_MAP[j])
            total = max(len(FEATURES_MAP[i] | FEATURES_MAP[j]), 1)
            gt[i, j] = shared / total
    return gt


def compute_stage_metrics(model):
    """Compute bilinear TN-sim, rank-1 results, and projection matrices."""
    # Bilinear TN-sim
    bil_sim = np.zeros((N_CLASSES, N_CLASSES))
    bil_weights = {}
    for c in range(N_CLASSES):
        bil_weights[c] = get_bilinear_effective_weights(model, c)
    for i in range(N_CLASSES):
        Li, Ri, Di = bil_weights[i]
        for j in range(N_CLASSES):
            Lj, Rj, Dj = bil_weights[j]
            bil_sim[i, j] = tn_sim_1layer(Li, Ri, Di, Lj, Rj, Dj).item()

    # Rank-1 optimization per class
    rank1_results = {}
    for c in range(N_CLASSES):
        L_eff, R_eff, D_eff = bil_weights[c]
        result = optimize_rank1(L_eff, R_eff, D_eff, n_seeds=20, steps=2000,
                                lr=0.01, device=DEVICE)
        rank1_results[c] = result

    # Rank-1 cross-class projection: TN-sim(rank1_i, T_bil_j)
    proj_sim = np.zeros((N_CLASSES, N_CLASSES))
    for i in range(N_CLASSES):
        li, ri, di = rank1_results[i]['l'], rank1_results[i]['r'], rank1_results[i]['d']
        for j in range(N_CLASSES):
            Lj, Rj, Dj = bil_weights[j]
            proj_sim[i, j] = tn_sim_1layer(li, ri, di, Lj, Rj, Dj).item()

    return bil_sim, rank1_results, proj_sim


def compute_correlation(sim_mat, gt_mat):
    """Pearson correlation between upper-triangle entries."""
    mask = np.triu_indices(N_CLASSES, k=1)
    r, p = pearsonr(sim_mat[mask], gt_mat[mask])
    return r, p


def print_matrix(name, mat, labels_short):
    """Pretty-print a matrix."""
    print(f"\n  {name}:")
    print(f"  {'':>12s}", '  '.join(f'{l:>9s}' for l in labels_short))
    for i in range(N_CLASSES):
        print(f"  {labels_short[i]:>12s}",
              '  '.join(f'{mat[i,j]:+9.3f}' for j in range(N_CLASSES)))


def print_rank1_vectors(rank1_results, n_input):
    """Print rank-1 vectors with top input activations."""
    for c in range(N_CLASSES):
        l = rank1_results[c]['l'].detach().cpu().numpy().flatten()
        r = rank1_results[c]['r'].detach().cpu().numpy().flatten()
        l_abs = np.abs(l[:n_input])
        r_abs = np.abs(r[:n_input])
        l_top = np.argsort(l_abs)[::-1][:2]
        r_top = np.argsort(r_abs)[::-1][:2]
        print(f"    C{c}({CLASS_LABELS[c]:>3s}): sim={rank1_results[c]['sim']:.4f}  "
              f"l->{INPUT_LABELS[l_top[0]]}({l[l_top[0]]:+.3f}),{INPUT_LABELS[l_top[1]]}({l[l_top[1]]:+.3f})  "
              f"r->{INPUT_LABELS[r_top[0]]}({r[r_top[0]]:+.3f}),{INPUT_LABELS[r_top[1]]}({r[r_top[1]]:+.3f})")


# =========================================================================
# MAIN
# =========================================================================

def main():
    print("=" * 70)
    print("LEVEL 2: Sparse Embedding Analysis (3 AND features, 8 classes)")
    print("=" * 70)

    X, labels = make_level2_dataset(DEVICE)
    print(f"\nDataset: {len(X)} patterns, {N_CLASSES} classes")
    for c in range(N_CLASSES):
        count = (labels == c).sum().item()
        print(f"  C{c} ({CLASS_LABELS[c]:>3s}): {count} patterns")

    gt_jaccard = compute_ground_truth_jaccard()
    labels_short = [f'C{c}({CLASS_LABELS[c]})' for c in range(N_CLASSES)]

    # ================================================================
    # STAGE 1: Dense (no pruning)
    # ================================================================
    print("\n" + "=" * 60)
    print("STAGE 1: Dense embedding")
    print("=" * 60)

    model = BooleanBilinear1Layer(
        n_input=6, d_hidden=24, d_rank=12, n_classes=8
    ).to(DEVICE)

    acc_dense = train_model(model, X, labels, epochs=5000, lr=0.01, wd=0.01,
                            print_every=1000, device=DEVICE)
    print(f"  Dense accuracy: {acc_dense:.1%}")

    # Per-class accuracy
    with torch.no_grad():
        preds = model(X).argmax(1)
        for c in range(N_CLASSES):
            mask = labels == c
            if mask.sum() > 0:
                ca = (preds[mask] == c).float().mean().item()
                print(f"    C{c}({CLASS_LABELS[c]:>3s}): {ca:.1%}")

    # Embed sparsity
    with torch.no_grad():
        W = model.embed.weight
        nnz = (W.abs() > 1e-8).sum().item()
        total = W.numel()
        print(f"  Embed: {nnz}/{total} nonzero ({100*nnz/total:.1f}%)")

    bil_sim_dense, rank1_dense, proj_sim_dense = compute_stage_metrics(model)

    r_bil_dense, _ = compute_correlation(bil_sim_dense, gt_jaccard)
    r_proj_dense, _ = compute_correlation(proj_sim_dense, gt_jaccard)
    print(f"  Correlation with ground truth:")
    print(f"    Bilinear TN-sim: r = {r_bil_dense:.3f}")
    print(f"    Rank-1 proj:     r = {r_proj_dense:.3f}")

    print_matrix("Bilinear TN-sim", bil_sim_dense, labels_short)
    print_matrix("Rank-1 projection", proj_sim_dense, labels_short)
    print(f"\n  Rank-1 vectors:")
    print_rank1_vectors(rank1_dense, 6)

    # ================================================================
    # STAGE 2: Top-2 per row
    # ================================================================
    print("\n" + "=" * 60)
    print("STAGE 2: Top-2 per embedding row")
    print("=" * 60)

    embed_mask_2 = prune_embed_topk(model, k=2)
    with torch.no_grad():
        nnz = (model.embed.weight.abs() > 1e-8).sum().item()
        total = model.embed.weight.numel()
        print(f"  Embed after prune: {nnz}/{total} nonzero ({100*nnz/total:.1f}%)")
        acc_pre = (model(X).argmax(1) == labels).float().mean().item()
        print(f"  Accuracy before retrain: {acc_pre:.1%}")

    acc_top2 = train_with_embed_mask(model, X, labels, embed_mask_2,
                                     epochs=3000, lr=0.005, wd=0.01)
    print(f"  Accuracy after retrain: {acc_top2:.1%}")

    # Per-class accuracy
    with torch.no_grad():
        preds = model(X).argmax(1)
        for c in range(N_CLASSES):
            mask = labels == c
            if mask.sum() > 0:
                ca = (preds[mask] == c).float().mean().item()
                print(f"    C{c}({CLASS_LABELS[c]:>3s}): {ca:.1%}")

    bil_sim_top2, rank1_top2, proj_sim_top2 = compute_stage_metrics(model)

    r_bil_top2, _ = compute_correlation(bil_sim_top2, gt_jaccard)
    r_proj_top2, _ = compute_correlation(proj_sim_top2, gt_jaccard)
    print(f"  Correlation with ground truth:")
    print(f"    Bilinear TN-sim: r = {r_bil_top2:.3f}")
    print(f"    Rank-1 proj:     r = {r_proj_top2:.3f}")

    print_matrix("Bilinear TN-sim", bil_sim_top2, labels_short)
    print_matrix("Rank-1 projection", proj_sim_top2, labels_short)
    print(f"\n  Rank-1 vectors:")
    print_rank1_vectors(rank1_top2, 6)

    # ================================================================
    # STAGE 3: Top-1 per row
    # ================================================================
    print("\n" + "=" * 60)
    print("STAGE 3: Top-1 per embedding row")
    print("=" * 60)

    embed_mask_1 = prune_embed_topk(model, k=1)
    with torch.no_grad():
        nnz = (model.embed.weight.abs() > 1e-8).sum().item()
        total = model.embed.weight.numel()
        print(f"  Embed after prune: {nnz}/{total} nonzero ({100*nnz/total:.1f}%)")
        acc_pre = (model(X).argmax(1) == labels).float().mean().item()
        print(f"  Accuracy before retrain: {acc_pre:.1%}")

    acc_top1 = train_with_embed_mask(model, X, labels, embed_mask_1,
                                     epochs=3000, lr=0.005, wd=0.01)
    print(f"  Accuracy after retrain: {acc_top1:.1%}")

    # Per-class accuracy
    with torch.no_grad():
        preds = model(X).argmax(1)
        for c in range(N_CLASSES):
            mask = labels == c
            if mask.sum() > 0:
                ca = (preds[mask] == c).float().mean().item()
                print(f"    C{c}({CLASS_LABELS[c]:>3s}): {ca:.1%}")

    bil_sim_top1, rank1_top1, proj_sim_top1 = compute_stage_metrics(model)

    r_bil_top1, _ = compute_correlation(bil_sim_top1, gt_jaccard)
    r_proj_top1, _ = compute_correlation(proj_sim_top1, gt_jaccard)
    print(f"  Correlation with ground truth:")
    print(f"    Bilinear TN-sim: r = {r_bil_top1:.3f}")
    print(f"    Rank-1 proj:     r = {r_proj_top1:.3f}")

    print_matrix("Bilinear TN-sim", bil_sim_top1, labels_short)
    print_matrix("Rank-1 projection", proj_sim_top1, labels_short)
    print(f"\n  Rank-1 vectors:")
    print_rank1_vectors(rank1_top1, 6)

    # ================================================================
    # SUMMARY
    # ================================================================
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    print(f"  {'Stage':>10s}  {'Acc':>6s}  {'r(bil,gt)':>10s}  {'r(proj,gt)':>10s}")
    print(f"  {'Dense':>10s}  {acc_dense:>6.1%}  {r_bil_dense:>10.3f}  {r_proj_dense:>10.3f}")
    print(f"  {'Top-2':>10s}  {acc_top2:>6.1%}  {r_bil_top2:>10.3f}  {r_proj_top2:>10.3f}")
    print(f"  {'Top-1':>10s}  {acc_top1:>6.1%}  {r_bil_top1:>10.3f}  {r_proj_top1:>10.3f}")

    # ================================================================
    # PLOT: 3 rows (dense, top-2, top-1) x 3 cols (bil, proj, gt)
    # ================================================================
    print("\nGenerating plot...")

    fig, axes = plt.subplots(3, 3, figsize=(21, 18))

    stages = [
        (0, bil_sim_dense, proj_sim_dense, acc_dense, r_bil_dense, r_proj_dense, 'Dense'),
        (1, bil_sim_top2, proj_sim_top2, acc_top2, r_bil_top2, r_proj_top2, 'Top-2'),
        (2, bil_sim_top1, proj_sim_top1, acc_top1, r_bil_top1, r_proj_top1, 'Top-1'),
    ]

    for row, bil_sim, proj_sim, acc, r_bil, r_proj, stage_name in stages:
        for col, (mat, title, corr) in enumerate([
            (bil_sim, f'Bilinear TN-sim ({stage_name})\nr={r_bil:.3f}', r_bil),
            (proj_sim, f'Rank-1 projection ({stage_name})\nr={r_proj:.3f}', r_proj),
            (gt_jaccard, f'Ground truth Jaccard\nacc={acc:.1%}', None),
        ]):
            ax = axes[row, col]
            if col == 2:  # ground truth
                im = ax.imshow(mat, cmap='Blues', vmin=0, vmax=1)
            else:
                im = ax.imshow(mat, cmap='RdBu_r', vmin=-1, vmax=1)
            ax.set_xticks(range(N_CLASSES))
            ax.set_yticks(range(N_CLASSES))
            ax.set_xticklabels(labels_short, fontsize=7, rotation=45, ha='right')
            ax.set_yticklabels(labels_short, fontsize=7)
            ax.set_title(title, fontsize=10)
            for i in range(N_CLASSES):
                for j in range(N_CLASSES):
                    ax.text(j, i, f'{mat[i,j]:+.2f}', ha='center', va='center',
                            fontsize=6, fontweight='bold',
                            color='white' if abs(mat[i, j]) > 0.5 else 'black')
            plt.colorbar(im, ax=ax, shrink=0.8)

    plt.suptitle('Level 2: Sparse Embedding Analysis (3 AND features, 8 classes)',
                 fontsize=14, y=1.01)
    plt.tight_layout()
    plt.savefig(os.path.join(IMG_DIR, 'sparse_embed.png'), dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved to {IMG_DIR}/sparse_embed.png")

    # ================================================================
    # SAVE RESULTS
    # ================================================================
    save_path = os.path.join(os.path.dirname(__file__), 'level2_sparse_embed_results.pt')
    torch.save({
        'model_state': model.state_dict(),
        'stages': {
            'dense': {
                'accuracy': acc_dense,
                'bil_sim': bil_sim_dense,
                'proj_sim': proj_sim_dense,
                'rank1_results': rank1_dense,
                'corr_bil': r_bil_dense,
                'corr_proj': r_proj_dense,
            },
            'top2': {
                'accuracy': acc_top2,
                'bil_sim': bil_sim_top2,
                'proj_sim': proj_sim_top2,
                'rank1_results': rank1_top2,
                'corr_bil': r_bil_top2,
                'corr_proj': r_proj_top2,
            },
            'top1': {
                'accuracy': acc_top1,
                'bil_sim': bil_sim_top1,
                'proj_sim': proj_sim_top1,
                'rank1_results': rank1_top1,
                'corr_bil': r_bil_top1,
                'corr_proj': r_proj_top1,
            },
        },
        'gt_jaccard': gt_jaccard,
    }, save_path)
    print(f"  Saved to {save_path}")


if __name__ == '__main__':
    main()
