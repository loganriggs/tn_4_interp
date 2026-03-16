"""
Level 1 rerun with frozen identity embedding.
This makes L, R vectors directly interpretable as input selectors.

d_hidden = n_input = 4, embed = identity (frozen).
"""
import torch
import torch.nn as nn
import numpy as np
import os
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from model import (
    BooleanBilinear1Layer, make_level1_dataset, train_model,
    tn_inner_1layer, tn_sim_1layer, compute_relative_norms,
    optimize_rank1, project_rank1_onto_class,
)

IMG_DIR = os.path.join(os.path.dirname(__file__), 'images', 'level1_frozen')
os.makedirs(IMG_DIR, exist_ok=True)
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

CLASS_LABELS = ['∅', 'A', 'B', 'AB']
INPUT_LABELS = ['x0', 'x1', 'x2', 'x3']


def freeze_embed_to_identity(model):
    """Set embedding to identity and freeze it."""
    with torch.no_grad():
        model.embed.weight.copy_(torch.eye(model.n_input, device=model.embed.weight.device))
    model.embed.weight.requires_grad_(False)


def analyze_level1_frozen():
    print("=" * 70)
    print("LEVEL 1: Frozen identity embedding (d_hidden = n_input = 4)")
    print("=" * 70)

    X, labels = make_level1_dataset(DEVICE)
    print(f"\nDataset: {len(X)} patterns, {labels.max().item()+1} classes")
    for c in range(4):
        count = (labels == c).sum().item()
        print(f"  Class {c} ({CLASS_LABELS[c]}): {count} patterns")

    # d_hidden = n_input = 4, so embed is identity
    model = BooleanBilinear1Layer(n_input=4, d_hidden=4, d_rank=8, n_classes=4).to(DEVICE)
    freeze_embed_to_identity(model)

    acc = train_model(model, X, labels, epochs=5000, lr=0.01, wd=0.005,
                      print_every=1000, device=DEVICE)
    print(f"Final accuracy: {acc:.1%}")

    # Verify embedding stayed identity
    with torch.no_grad():
        embed_err = (model.W_embed - torch.eye(4, device=DEVICE)).abs().max().item()
        print(f"Embed deviation from identity: {embed_err:.2e}")

    # Per-class accuracy
    with torch.no_grad():
        preds = model(X).argmax(dim=1)
        for c in range(4):
            mask = labels == c
            if mask.sum() > 0:
                ca = (preds[mask] == c).float().mean().item()
                print(f"  Class {c} ({CLASS_LABELS[c]}): {ca:.1%} ({mask.sum().item()} samples)")

    # ================================================================
    # RELATIVE NORMS
    # ================================================================
    print("\nRelative norms:")
    class_norms = {}
    for c in range(4):
        r = compute_relative_norms(model, c, DEVICE)
        class_norms[c] = r
        print(f"  C{c} ({CLASS_LABELS[c]}): ρ_res={r['rho_res']:.4f}  ρ_bil={r['rho_bil']:.4f}")

    # ================================================================
    # RANK-1 OPTIMIZATION
    # ================================================================
    print("\nRank-1 optimization:")
    rank1_results = {}
    for c in range(4):
        L_T, R_T, D_T = model.get_augmented_weights_for_class(c)
        result = optimize_rank1(L_T, R_T, D_T, n_seeds=20, steps=2000,
                                lr=0.01, device=DEVICE)
        rank1_results[c] = result
        print(f"  C{c} ({CLASS_LABELS[c]}): TN-sim = {result['sim']:.4f}")

    # ================================================================
    # CROSS-CLASS TN-SIM (all 4x4 combinations)
    # ================================================================
    # Full tensor pairwise
    full_sim = np.zeros((4, 4))
    for i in range(4):
        Li, Ri, Di = model.get_augmented_weights_for_class(i)
        for j in range(4):
            Lj, Rj, Dj = model.get_augmented_weights_for_class(j)
            full_sim[i, j] = tn_sim_1layer(Li, Ri, Di, Lj, Rj, Dj).item()

    # Bilinear-only pairwise
    bil_sim = np.zeros((4, 4))
    for i in range(4):
        Li, Ri, Di = model.get_bilinear_only_weights_for_class(i)
        for j in range(4):
            Lj, Rj, Dj = model.get_bilinear_only_weights_for_class(j)
            bil_sim[i, j] = tn_sim_1layer(Li, Ri, Di, Lj, Rj, Dj).item()

    # Rank-1 projection: TN-sim(rank1_i, T_j)
    proj_sim = np.zeros((4, 4))
    for i in range(4):
        li, ri, di = rank1_results[i]['l'], rank1_results[i]['r'], rank1_results[i]['d']
        for j in range(4):
            Lj, Rj, Dj = model.get_augmented_weights_for_class(j)
            proj_sim[i, j] = tn_sim_1layer(li, ri, di, Lj, Rj, Dj).item()

    print("\nFull TN-sim (T_i vs T_j):")
    labels_short = [f'C{c}({CLASS_LABELS[c]})' for c in range(4)]
    print(f"  {'':>12s}", '  '.join(f'{l:>10s}' for l in labels_short))
    for i in range(4):
        print(f"  {labels_short[i]:>12s}", '  '.join(f'{full_sim[i,j]:+10.3f}' for j in range(4)))

    print("\nBilinear-only TN-sim:")
    print(f"  {'':>12s}", '  '.join(f'{l:>10s}' for l in labels_short))
    for i in range(4):
        print(f"  {labels_short[i]:>12s}", '  '.join(f'{bil_sim[i,j]:+10.3f}' for j in range(4)))

    print("\nRank-1 projection TN-sim (rank1_i → T_j):")
    print(f"  {'':>12s}", '  '.join(f'{l:>10s}' for l in labels_short))
    for i in range(4):
        print(f"  {labels_short[i]:>12s}", '  '.join(f'{proj_sim[i,j]:+10.3f}' for j in range(4)))

    # ================================================================
    # RANK-1 VECTORS (directly interpretable now!)
    # ================================================================
    print("\nRank-1 vectors (in augmented x̂ = [x0, x1, x2, x3, 1] space):")
    for c in range(4):
        l = rank1_results[c]['l'].detach().cpu().numpy().flatten()
        r = rank1_results[c]['r'].detach().cpu().numpy().flatten()
        d = rank1_results[c]['d'].detach().cpu().numpy().flatten()
        print(f"\n  C{c} ({CLASS_LABELS[c]}):")
        print(f"    l = [{', '.join(f'{v:+.3f}' for v in l)}]")
        print(f"    r = [{', '.join(f'{v:+.3f}' for v in r)}]")
        print(f"    d = [{', '.join(f'{v:+.3f}' for v in d)}]")

        # Which inputs dominate?
        l_abs = np.abs(l[:4])  # exclude constant dim
        r_abs = np.abs(r[:4])
        l_top = np.argsort(l_abs)[::-1]
        r_top = np.argsort(r_abs)[::-1]
        print(f"    l top inputs: x{l_top[0]}({l_abs[l_top[0]]:.3f}), x{l_top[1]}({l_abs[l_top[1]]:.3f})")
        print(f"    r top inputs: x{r_top[0]}({r_abs[r_top[0]]:.3f}), x{r_top[1]}({r_abs[r_top[1]]:.3f})")

    # ================================================================
    # GROUND TRUTH COMPARISON
    # ================================================================
    print("\n" + "=" * 70)
    print("GROUND TRUTH COMPARISON")
    print("=" * 70)

    # Expected feature sharing (Jaccard)
    features = {0: set(), 1: {'A'}, 2: {'B'}, 3: {'A', 'B'}}
    gt_sharing = np.zeros((4, 4))
    for i in range(4):
        for j in range(4):
            shared = len(features[i] & features[j])
            total = max(len(features[i] | features[j]), 1)
            gt_sharing[i, j] = shared / total

    print("\nGround truth Jaccard:")
    print(f"  {'':>12s}", '  '.join(f'{l:>10s}' for l in labels_short))
    for i in range(4):
        print(f"  {labels_short[i]:>12s}", '  '.join(f'{gt_sharing[i,j]:+10.3f}' for j in range(4)))

    # Correlation
    mask = np.triu_indices(4, k=1)
    gt_flat = gt_sharing[mask]
    full_flat = full_sim[mask]
    bil_flat = bil_sim[mask]
    from scipy.stats import pearsonr
    r_full, _ = pearsonr(gt_flat, full_flat)
    r_bil, _ = pearsonr(gt_flat, bil_flat)
    print(f"\nCorrelation with ground truth:")
    print(f"  Full TN-sim:     r = {r_full:.3f}")
    print(f"  Bilinear TN-sim: r = {r_bil:.3f}")

    # ================================================================
    # PLOTS
    # ================================================================
    print("\nGenerating plots...")

    fig, axes = plt.subplots(2, 3, figsize=(18, 11))

    # Row 1: Three similarity matrices
    for ax, mat, title in [
        (axes[0, 0], full_sim, 'Full TN-sim (T_i vs T_j)'),
        (axes[0, 1], bil_sim, 'Bilinear-only TN-sim'),
        (axes[0, 2], proj_sim, 'Rank-1 projection\nTN-sim(rank1_i, T_j)'),
    ]:
        im = ax.imshow(mat, cmap='RdBu_r', vmin=-1, vmax=1)
        ax.set_xticks(range(4)); ax.set_yticks(range(4))
        ax.set_xticklabels(labels_short, fontsize=9, rotation=45, ha='right')
        ax.set_yticklabels(labels_short, fontsize=9)
        ax.set_title(title, fontsize=11)
        for i in range(4):
            for j in range(4):
                ax.text(j, i, f'{mat[i,j]:+.2f}', ha='center', va='center',
                        fontsize=10, color='white' if abs(mat[i,j]) > 0.5 else 'black')
        plt.colorbar(im, ax=ax, shrink=0.8)

    # Row 2 left: Ground truth
    ax = axes[1, 0]
    im = ax.imshow(gt_sharing, cmap='Blues', vmin=0, vmax=1)
    ax.set_xticks(range(4)); ax.set_yticks(range(4))
    ax.set_xticklabels(labels_short, fontsize=9, rotation=45, ha='right')
    ax.set_yticklabels(labels_short, fontsize=9)
    ax.set_title('Ground truth Jaccard', fontsize=11)
    for i in range(4):
        for j in range(4):
            ax.text(j, i, f'{gt_sharing[i,j]:.2f}', ha='center', va='center',
                    fontsize=10, color='white' if gt_sharing[i,j] > 0.5 else 'black')
    plt.colorbar(im, ax=ax, shrink=0.8)

    # Row 2 middle: Relative norms
    ax = axes[1, 1]
    x_pos = np.arange(4)
    rho_res = [class_norms[c]['rho_res'] for c in range(4)]
    rho_bil = [class_norms[c]['rho_bil'] for c in range(4)]
    ax.bar(x_pos - 0.15, rho_res, 0.3, label='residual', color='C0')
    ax.bar(x_pos + 0.15, rho_bil, 0.3, label='bilinear', color='C1')
    ax.set_xticks(x_pos)
    ax.set_xticklabels(labels_short, fontsize=9)
    ax.set_ylabel('ρ (relative norm fraction)')
    ax.set_title('Relative norms', fontsize=11)
    ax.legend()
    ax.set_ylim(-0.1, 1.1)
    ax.grid(True, alpha=0.3)

    # Row 2 right: Rank-1 vectors visualization
    ax = axes[1, 2]
    # Show |l| and |r| vectors for each class as a heatmap
    vec_data = np.zeros((4, 10))  # 4 classes, 5 dims for l + 5 for r
    for c in range(4):
        l = rank1_results[c]['l'].detach().cpu().numpy().flatten()
        r = rank1_results[c]['r'].detach().cpu().numpy().flatten()
        vec_data[c, :5] = l
        vec_data[c, 5:] = r
    im = ax.imshow(vec_data, cmap='RdBu_r', vmin=-np.abs(vec_data).max(), vmax=np.abs(vec_data).max())
    ax.set_yticks(range(4))
    ax.set_yticklabels(labels_short, fontsize=9)
    ax.set_xticks(range(10))
    ax.set_xticklabels(['l:x0','l:x1','l:x2','l:x3','l:1',
                         'r:x0','r:x1','r:x2','r:x3','r:1'], fontsize=7, rotation=45, ha='right')
    ax.set_title('Rank-1 vectors (l, r)', fontsize=11)
    for i in range(4):
        for j in range(10):
            ax.text(j, i, f'{vec_data[i,j]:+.2f}', ha='center', va='center',
                    fontsize=7, color='white' if abs(vec_data[i,j]) > np.abs(vec_data).max()*0.4 else 'black')
    plt.colorbar(im, ax=ax, shrink=0.8)

    plt.suptitle('Level 1: Frozen Identity Embedding (d_hidden = n_input = 4)', fontsize=13, y=1.01)
    plt.tight_layout()
    plt.savefig(os.path.join(IMG_DIR, 'level1_frozen_embed.png'), dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved to {IMG_DIR}/level1_frozen_embed.png")

    # Save results
    torch.save({
        'model_state': model.state_dict(),
        'class_norms': class_norms,
        'rank1_results': rank1_results,
        'full_sim': full_sim,
        'bil_sim': bil_sim,
        'proj_sim': proj_sim,
        'gt_sharing': gt_sharing,
        'accuracy': acc,
    }, os.path.join(os.path.dirname(__file__), 'level1_frozen_results.pt'))
    print("  Saved level1_frozen_results.pt")


if __name__ == '__main__':
    analyze_level1_frozen()
