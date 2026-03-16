"""
Level 1: Iterative per-row embedding pruning with bilinear rank-1 analysis.

Stages: dense -> top-2 per row -> top-1 per row.
At each stage: rank-1 optimize bilinear-only effective weights, compute cross-class
TN-sim, and visualize comparison with ground truth Jaccard.
"""
import torch
import torch.nn.functional as F
import numpy as np
import os
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from model import (
    BooleanBilinear1Layer, make_level1_dataset, train_model,
    tn_inner_1layer, tn_sim_1layer, optimize_rank1,
)

IMG_DIR = os.path.join(os.path.dirname(__file__), 'images', 'level1')
os.makedirs(IMG_DIR, exist_ok=True)
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

CLASS_LABELS = ['∅', 'A', 'B', 'AB']
INPUT_LABELS = ['x0', 'x1', 'x2', 'x3']

# Ground truth features
features = {0: set(), 1: {'A'}, 2: {'B'}, 3: {'A', 'B'}}


# =============================================================================
# HELPERS
# =============================================================================

def train_with_embed_mask(model, X, labels, embed_mask, epochs, lr, wd):
    """Train model with embedding mask applied to gradients and weights."""
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
    """Prune embedding to top-k entries per row by magnitude."""
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


def get_bilinear_effective_weights(model, c, embed_mask=None):
    """Get bilinear-only effective weights in input space for class c.
    L_eff = L @ W_embed_masked, R_eff = R @ W_embed_masked, D_eff = W_head[c] @ D.
    """
    with torch.no_grad():
        W_e = model.W_embed.clone()
        if embed_mask is not None:
            W_e = W_e * embed_mask
        L_eff = model.L @ W_e      # (rank, n_input)
        R_eff = model.R @ W_e      # (rank, n_input)
        D_eff = model.W_head[[c]] @ model.D  # (1, rank)
    return L_eff, R_eff, D_eff


def compute_cross_class_tnsim(model, embed_mask=None):
    """Compute 4x4 bilinear cross-class TN-sim matrix."""
    sim = np.zeros((4, 4))
    for i in range(4):
        Li, Ri, Di = get_bilinear_effective_weights(model, i, embed_mask)
        for j in range(4):
            Lj, Rj, Dj = get_bilinear_effective_weights(model, j, embed_mask)
            sim[i, j] = tn_sim_1layer(Li, Ri, Di, Lj, Rj, Dj).item()
    return sim


def compute_rank1_all_classes(model, embed_mask=None):
    """Optimize rank-1 approximation for all 4 classes."""
    results = {}
    for c in range(4):
        L_eff, R_eff, D_eff = get_bilinear_effective_weights(model, c, embed_mask)
        result = optimize_rank1(L_eff, R_eff, D_eff, n_seeds=20, steps=2000,
                                lr=0.01, device=DEVICE)
        results[c] = result
    return results


def compute_rank1_cross_proj(rank1_results, model, embed_mask=None):
    """Compute rank-1 cross-class projection matrix: TN-sim(rank1_i, T_bil_j)."""
    proj = np.zeros((4, 4))
    for i in range(4):
        li, ri, di = rank1_results[i]['l'], rank1_results[i]['r'], rank1_results[i]['d']
        for j in range(4):
            Lj, Rj, Dj = get_bilinear_effective_weights(model, j, embed_mask)
            proj[i, j] = tn_sim_1layer(li, ri, di, Lj, Rj, Dj).item()
    return proj


def compute_gt_jaccard():
    """Compute ground truth Jaccard similarity matrix."""
    gt = np.zeros((4, 4))
    for i in range(4):
        for j in range(4):
            shared = len(features[i] & features[j])
            total = max(len(features[i] | features[j]), 1)
            gt[i, j] = shared / total
    return gt


def print_rank1_vectors(rank1_results, stage_name):
    """Print rank-1 vectors with input labels."""
    print(f"\nRank-1 vectors ({stage_name}) [x0, x1, x2, x3]:")
    for c in range(4):
        l = rank1_results[c]['l'].detach().cpu().numpy().flatten()
        r = rank1_results[c]['r'].detach().cpu().numpy().flatten()
        d = rank1_results[c]['d'].detach().cpu().numpy().flatten()
        sim = rank1_results[c]['sim']
        print(f"  C{c}({CLASS_LABELS[c]}): sim={sim:.4f}")
        print(f"    l = [{', '.join(f'{v:+.3f}' for v in l)}]")
        print(f"    r = [{', '.join(f'{v:+.3f}' for v in r)}]")
        print(f"    d = {d[0]:+.4f}")


# =============================================================================
# MAIN
# =============================================================================

def main():
    X, labels = make_level1_dataset(DEVICE)
    print(f"Dataset: {len(X)} patterns, 4 classes")
    print(f"Device: {DEVICE}\n")

    # ================================================================
    # STAGE 1: Train dense model
    # ================================================================
    print("=" * 60)
    print("STAGE 1: Dense embedding")
    print("=" * 60)
    model = BooleanBilinear1Layer(n_input=4, d_hidden=16, d_rank=8, n_classes=4).to(DEVICE)
    acc_dense = train_model(model, X, labels, epochs=5000, lr=0.01, wd=0.005,
                            print_every=2500, device=DEVICE)
    print(f"Dense accuracy: {acc_dense:.1%}")

    tnsim_dense = compute_cross_class_tnsim(model)
    rank1_dense = compute_rank1_all_classes(model)
    proj_dense = compute_rank1_cross_proj(rank1_dense, model)
    print_rank1_vectors(rank1_dense, "dense")

    # ================================================================
    # STAGE 2: Prune to top-2 per row, retrain
    # ================================================================
    print("\n" + "=" * 60)
    print("STAGE 2: Top-2 per row")
    print("=" * 60)
    embed_mask_2 = prune_embed_topk(model, k=2)
    nnz = (embed_mask_2 > 0).sum().item()
    total = embed_mask_2.numel()
    print(f"Embed sparsity: {nnz}/{total} nonzero ({100*nnz/total:.0f}%)")

    acc_top2 = train_with_embed_mask(model, X, labels, embed_mask_2,
                                     epochs=3000, lr=0.005, wd=0.005)
    print(f"Top-2 accuracy: {acc_top2:.1%}")

    tnsim_top2 = compute_cross_class_tnsim(model, embed_mask_2)
    rank1_top2 = compute_rank1_all_classes(model, embed_mask_2)
    proj_top2 = compute_rank1_cross_proj(rank1_top2, model, embed_mask_2)
    print_rank1_vectors(rank1_top2, "top-2")

    # ================================================================
    # STAGE 3: Prune to top-1 per row, retrain
    # ================================================================
    print("\n" + "=" * 60)
    print("STAGE 3: Top-1 per row")
    print("=" * 60)
    embed_mask_1 = prune_embed_topk(model, k=1)
    nnz = (embed_mask_1 > 0).sum().item()
    total = embed_mask_1.numel()
    print(f"Embed sparsity: {nnz}/{total} nonzero ({100*nnz/total:.0f}%)")

    acc_top1 = train_with_embed_mask(model, X, labels, embed_mask_1,
                                     epochs=3000, lr=0.005, wd=0.005)
    print(f"Top-1 accuracy: {acc_top1:.1%}")

    tnsim_top1 = compute_cross_class_tnsim(model, embed_mask_1)
    rank1_top1 = compute_rank1_all_classes(model, embed_mask_1)
    proj_top1 = compute_rank1_cross_proj(rank1_top1, model, embed_mask_1)
    print_rank1_vectors(rank1_top1, "top-1")

    # ================================================================
    # SUMMARY
    # ================================================================
    print("\n" + "=" * 60)
    print("ACCURACY SUMMARY")
    print("=" * 60)
    print(f"  Dense:  {acc_dense:.1%}")
    print(f"  Top-2:  {acc_top2:.1%}")
    print(f"  Top-1:  {acc_top1:.1%}")

    # ================================================================
    # PLOT: 3 rows x 4 cols
    # ================================================================
    gt_jaccard = compute_gt_jaccard()
    labels_short = [f'C{c}({CLASS_LABELS[c]})' for c in range(4)]
    n = model.n_input

    stages = [
        ('Dense', acc_dense, tnsim_dense, proj_dense, rank1_dense),
        ('Top-2', acc_top2, tnsim_top2, proj_top2, rank1_top2),
        ('Top-1', acc_top1, tnsim_top1, proj_top1, rank1_top1),
    ]

    fig, axes = plt.subplots(3, 4, figsize=(22, 15))

    for row, (stage_name, acc, tnsim, proj, rank1) in enumerate(stages):
        # Col 0: Bilinear cross-class TN-sim
        ax = axes[row, 0]
        im = ax.imshow(tnsim, cmap='RdBu_r', vmin=-1, vmax=1)
        ax.set_xticks(range(4)); ax.set_yticks(range(4))
        ax.set_xticklabels(labels_short, fontsize=8, rotation=45, ha='right')
        ax.set_yticklabels(labels_short, fontsize=8)
        ax.set_title(f'{stage_name} ({acc:.0%})\nBilinear TN-sim', fontsize=10)
        for i in range(4):
            for j in range(4):
                ax.text(j, i, f'{tnsim[i,j]:+.2f}', ha='center', va='center',
                        fontsize=8, fontweight='bold',
                        color='white' if abs(tnsim[i,j]) > 0.5 else 'black')
        plt.colorbar(im, ax=ax, shrink=0.8)

        # Col 1: Rank-1 cross-class projection
        ax = axes[row, 1]
        im = ax.imshow(proj, cmap='RdBu_r', vmin=-1, vmax=1)
        ax.set_xticks(range(4)); ax.set_yticks(range(4))
        ax.set_xticklabels(labels_short, fontsize=8, rotation=45, ha='right')
        ax.set_yticklabels(labels_short, fontsize=8)
        ax.set_title(f'{stage_name}\nRank-1 cross-class proj', fontsize=10)
        for i in range(4):
            for j in range(4):
                ax.text(j, i, f'{proj[i,j]:+.2f}', ha='center', va='center',
                        fontsize=8, fontweight='bold',
                        color='white' if abs(proj[i,j]) > 0.5 else 'black')
        plt.colorbar(im, ax=ax, shrink=0.8)

        # Col 2: Rank-1 l/r vectors heatmap
        ax = axes[row, 2]
        vec_data = np.zeros((4, 2 * n))
        for c in range(4):
            l = rank1[c]['l'].detach().cpu().numpy().flatten()
            r = rank1[c]['r'].detach().cpu().numpy().flatten()
            vec_data[c, :n] = l
            vec_data[c, n:] = r
        vmax_vec = max(np.abs(vec_data).max(), 1e-6)
        im = ax.imshow(vec_data, cmap='RdBu_r', vmin=-vmax_vec, vmax=vmax_vec)
        ax.set_yticks(range(4))
        ax.set_yticklabels(labels_short, fontsize=8)
        xlabels = [f'l:{INPUT_LABELS[i]}' for i in range(n)] + \
                  [f'r:{INPUT_LABELS[i]}' for i in range(n)]
        ax.set_xticks(range(2 * n))
        ax.set_xticklabels(xlabels, fontsize=7, rotation=45, ha='right')
        ax.set_title(f'{stage_name}\nRank-1 l/r vectors', fontsize=10)
        for i in range(4):
            for j in range(2 * n):
                ax.text(j, i, f'{vec_data[i,j]:+.2f}', ha='center', va='center',
                        fontsize=7,
                        color='white' if abs(vec_data[i,j]) > vmax_vec * 0.4 else 'black')
        plt.colorbar(im, ax=ax, shrink=0.8)

        # Col 3: Ground truth Jaccard (same for all rows)
        ax = axes[row, 3]
        im = ax.imshow(gt_jaccard, cmap='Blues', vmin=0, vmax=1)
        ax.set_xticks(range(4)); ax.set_yticks(range(4))
        ax.set_xticklabels(labels_short, fontsize=8, rotation=45, ha='right')
        ax.set_yticklabels(labels_short, fontsize=8)
        ax.set_title(f'Ground truth\nJaccard similarity', fontsize=10)
        for i in range(4):
            for j in range(4):
                ax.text(j, i, f'{gt_jaccard[i,j]:.2f}', ha='center', va='center',
                        fontsize=8, fontweight='bold',
                        color='white' if gt_jaccard[i,j] > 0.5 else 'black')
        plt.colorbar(im, ax=ax, shrink=0.8)

    plt.suptitle('Level 1: Sparse Embedding Pruning (dense -> top-2 -> top-1)',
                 fontsize=14, y=1.01)
    plt.tight_layout()
    save_path = os.path.join(IMG_DIR, 'sparse_embed.png')
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"\nSaved plot to {save_path}")

    # ================================================================
    # SAVE RESULTS
    # ================================================================
    results_path = os.path.join(os.path.dirname(__file__), 'level1_sparse_embed_results.pt')
    torch.save({
        'model_state': model.state_dict(),
        'stages': {
            'dense': {
                'accuracy': acc_dense,
                'tnsim': tnsim_dense,
                'rank1': rank1_dense,
                'proj': proj_dense,
                'embed_mask': None,
            },
            'top2': {
                'accuracy': acc_top2,
                'tnsim': tnsim_top2,
                'rank1': rank1_top2,
                'proj': proj_top2,
                'embed_mask': embed_mask_2.cpu(),
            },
            'top1': {
                'accuracy': acc_top1,
                'tnsim': tnsim_top1,
                'rank1': rank1_top1,
                'proj': proj_top1,
                'embed_mask': embed_mask_1.cpu(),
            },
        },
        'gt_jaccard': gt_jaccard,
    }, results_path)
    print(f"Saved results to {results_path}")


if __name__ == '__main__':
    main()
