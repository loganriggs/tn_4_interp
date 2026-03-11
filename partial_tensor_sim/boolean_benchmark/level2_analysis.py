"""
Level 2: Three AND features, 8 classes.
  A = x0 AND x1, B = x2 AND x3, C = x4 AND x5
  class = 4*C + 2*B + A (binary encoding: class 5 = A+C, class 7 = A+B+C, etc.)

Feature sharing matrix:
              A    B    C
  class 1:    ✓              (only A)
  class 2:         ✓         (only B)
  class 3:    ✓    ✓         (A+B)
  class 4:              ✓    (only C)
  class 5:    ✓         ✓    (A+C)
  class 6:         ✓    ✓    (B+C)
  class 7:    ✓    ✓    ✓    (all)
"""
import torch
import numpy as np
import os
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from model import (
    BooleanBilinear1Layer, make_level2_dataset, train_model,
    tn_inner_1layer, tn_sim_1layer, compute_relative_norms,
    optimize_rank1, project_rank1_onto_class,
)

IMG_DIR = os.path.join(os.path.dirname(__file__), 'images', 'level2')
os.makedirs(IMG_DIR, exist_ok=True)
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

# Feature labels for each class
CLASS_LABELS = ['∅', 'A', 'B', 'AB', 'C', 'AC', 'BC', 'ABC']
# Which features each class uses
FEATURES = {
    0: set(),
    1: {'A'},
    2: {'B'},
    3: {'A', 'B'},
    4: {'C'},
    5: {'A', 'C'},
    6: {'B', 'C'},
    7: {'A', 'B', 'C'},
}


def analyze_level2():
    print("=" * 70)
    print("LEVEL 2: Three AND features, 8 classes")
    print("=" * 70)

    # ================================================================
    # 1. TRAIN
    # ================================================================
    print("\n1. Training model...")
    X, labels = make_level2_dataset(DEVICE)
    print(f"   Dataset: {len(X)} patterns, {labels.max().item()+1} classes")
    for c in range(8):
        count = (labels == c).sum().item()
        print(f"   Class {c} ({CLASS_LABELS[c]}): {count} patterns")

    model = BooleanBilinear1Layer(n_input=6, d_hidden=24, d_rank=12, n_classes=8).to(DEVICE)
    acc = train_model(model, X, labels, epochs=5000, lr=0.01, wd=0.005,
                      print_every=1000, device=DEVICE)
    print(f"   Final accuracy: {acc:.1%}")

    # ================================================================
    # 2. RELATIVE NORMS
    # ================================================================
    print("\n2. Relative norms per class...")
    print(f"   {'Class':>5s} {'Label':>5s}  {'ρ(res)':>8s}  {'ρ(bil)':>8s}  {'#feat':>5s}")
    class_norms = {}
    for c in range(8):
        r = compute_relative_norms(model, c, DEVICE)
        class_norms[c] = r
        nf = len(FEATURES[c])
        print(f"   {c:>5d} {CLASS_LABELS[c]:>5s}  {r['rho_res']:>8.4f}  {r['rho_bil']:>8.4f}  {nf:>5d}")

    # ================================================================
    # 3. RANK-1 OPTIMIZATION
    # ================================================================
    print("\n3. Rank-1 optimization per class...")
    rank1_results = {}
    for c in range(8):
        L_T, R_T, D_T = model.get_augmented_weights_for_class(c)
        result = optimize_rank1(L_T, R_T, D_T, n_seeds=20, steps=2000,
                                lr=0.01, device=DEVICE)
        rank1_results[c] = result
        print(f"   Class {c} ({CLASS_LABELS[c]}): TN-sim = {result['sim']:.4f}")

    # ================================================================
    # 4. CROSS-CLASS TN-SIM (rank-1 of class i vs full tensor of class j)
    # ================================================================
    print("\n4. Cross-class TN-sim: sim(rank1_i, T_j)")
    cross_sim = np.zeros((8, 8))
    for i in range(8):
        li, ri, di = rank1_results[i]['l'], rank1_results[i]['r'], rank1_results[i]['d']
        for j in range(8):
            Lj, Rj, Dj = model.get_augmented_weights_for_class(j)
            cross_sim[i, j] = tn_sim_1layer(li, ri, di, Lj, Rj, Dj).item()

    # Print key pairs: single-feature classes projected onto multi-feature classes
    print("\n   Single-feature rank-1 → multi-feature classes:")
    for src in [1, 2, 4]:  # single feature classes (A, B, C)
        feat = list(FEATURES[src])[0]
        for tgt in range(8):
            if len(FEATURES[tgt]) >= 2 and feat in FEATURES[tgt]:
                print(f"     C{src}({CLASS_LABELS[src]}) → C{tgt}({CLASS_LABELS[tgt]}): "
                      f"{cross_sim[src, tgt]:+.3f} (shares {feat})")
        # Also show one non-sharing target
        for tgt in range(8):
            if len(FEATURES[tgt]) >= 1 and feat not in FEATURES[tgt] and tgt != 0:
                print(f"     C{src}({CLASS_LABELS[src]}) → C{tgt}({CLASS_LABELS[tgt]}): "
                      f"{cross_sim[src, tgt]:+.3f} (no shared feature)")
                break

    # ================================================================
    # 5. PAIRWISE TN-SIM (full tensors)
    # ================================================================
    print("\n5. Full pairwise TN-sim matrix...")
    full_sim = np.zeros((8, 8))
    for i in range(8):
        Li, Ri, Di = model.get_augmented_weights_for_class(i)
        for j in range(8):
            Lj, Rj, Dj = model.get_augmented_weights_for_class(j)
            full_sim[i, j] = tn_sim_1layer(Li, Ri, Di, Lj, Rj, Dj).item()

    # Bilinear-only pairwise
    bil_sim = np.zeros((8, 8))
    for i in range(8):
        Li, Ri, Di = model.get_bilinear_only_weights_for_class(i)
        for j in range(8):
            Lj, Rj, Dj = model.get_bilinear_only_weights_for_class(j)
            bil_sim[i, j] = tn_sim_1layer(Li, Ri, Di, Lj, Rj, Dj).item()

    # ================================================================
    # 6. GROUND TRUTH FEATURE SHARING
    # ================================================================
    gt_sharing = np.zeros((8, 8))
    for i in range(8):
        for j in range(8):
            shared = len(FEATURES[i] & FEATURES[j])
            total = max(len(FEATURES[i] | FEATURES[j]), 1)
            gt_sharing[i, j] = shared / total  # Jaccard similarity

    # ================================================================
    # 7. PLOTS
    # ================================================================
    print("\n6. Generating plots...")

    # Plot 1: Relative norms
    fig, ax = plt.subplots(figsize=(10, 5))
    x_pos = np.arange(8)
    rho_res = [class_norms[c]['rho_res'] for c in range(8)]
    rho_bil = [class_norms[c]['rho_bil'] for c in range(8)]
    ax.bar(x_pos - 0.15, rho_res, 0.3, label='residual', color='C0')
    ax.bar(x_pos + 0.15, rho_bil, 0.3, label='bilinear', color='C1')
    ax.set_xticks(x_pos)
    ax.set_xticklabels([f'C{c}\n({CLASS_LABELS[c]})' for c in range(8)])
    ax.set_ylabel('ρ (relative norm fraction)')
    ax.set_title('Level 2: Relative norms per class')
    ax.legend()
    ax.set_ylim(-0.1, 1.1)
    ax.axhline(0, color='k', linewidth=0.5)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(IMG_DIR, 'relative_norms.png'), dpi=150, bbox_inches='tight')
    plt.close()

    # Plot 2: 2x2 grid — full sim, bilinear sim, cross-class rank1 sim, ground truth
    fig, axes = plt.subplots(2, 2, figsize=(14, 12))
    labels_short = [f'C{c}({CLASS_LABELS[c]})' for c in range(8)]

    for ax, mat, title in [
        (axes[0, 0], full_sim, 'Full tensor TN-sim'),
        (axes[0, 1], bil_sim, 'Bilinear-only TN-sim'),
        (axes[1, 0], cross_sim, 'TN-sim(rank1_i, T_j)'),
        (axes[1, 1], gt_sharing, 'Ground truth: Jaccard feature similarity'),
    ]:
        vmin, vmax = (-1, 1) if 'Ground' not in title else (0, 1)
        cmap = 'RdBu_r' if 'Ground' not in title else 'Blues'
        im = ax.imshow(mat, cmap=cmap, vmin=vmin, vmax=vmax)
        ax.set_xticks(range(8)); ax.set_yticks(range(8))
        ax.set_xticklabels(labels_short, fontsize=7, rotation=45, ha='right')
        ax.set_yticklabels(labels_short, fontsize=7)
        ax.set_title(title, fontsize=11)
        for i in range(8):
            for j in range(8):
                ax.text(j, i, f'{mat[i,j]:.2f}', ha='center', va='center',
                        fontsize=6, color='white' if abs(mat[i,j]) > 0.5 else 'black')
        plt.colorbar(im, ax=ax, shrink=0.8)

    plt.tight_layout()
    plt.savefig(os.path.join(IMG_DIR, 'similarity_matrices.png'), dpi=150, bbox_inches='tight')
    plt.close()

    # Plot 3: Rank-1 TN-sim vs number of features
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    ax = axes[0]
    n_feats = [len(FEATURES[c]) for c in range(8)]
    sims = [rank1_results[c]['sim'] for c in range(8)]
    for c in range(8):
        ax.scatter(n_feats[c], sims[c], s=100, zorder=5)
        ax.annotate(f'C{c}', (n_feats[c], sims[c]), textcoords="offset points",
                    xytext=(5, 5), fontsize=8)
    ax.set_xlabel('Number of features')
    ax.set_ylabel('Rank-1 TN-sim')
    ax.set_title('Rank-1 quality vs feature count')
    ax.grid(True, alpha=0.3)

    ax = axes[1]
    for c in range(8):
        ax.scatter(class_norms[c]['rho_bil'], sims[c], s=100, zorder=5)
        ax.annotate(f'C{c}({CLASS_LABELS[c]})', (class_norms[c]['rho_bil'], sims[c]),
                    textcoords="offset points", xytext=(5, 5), fontsize=7)
    ax.set_xlabel('ρ(bilinear)')
    ax.set_ylabel('Rank-1 TN-sim')
    ax.set_title('Rank-1 quality vs bilinear fraction')
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(os.path.join(IMG_DIR, 'rank1_vs_features.png'), dpi=150, bbox_inches='tight')
    plt.close()

    # Plot 4: Correlation between cross-class TN-sim and ground truth sharing
    fig, ax = plt.subplots(figsize=(7, 6))
    gt_flat = gt_sharing[np.triu_indices(8, k=1)]
    cs_flat = cross_sim[np.triu_indices(8, k=1)]
    fs_flat = full_sim[np.triu_indices(8, k=1)]
    bs_flat = bil_sim[np.triu_indices(8, k=1)]

    ax.scatter(gt_flat, fs_flat, s=40, alpha=0.7, label='Full TN-sim')
    ax.scatter(gt_flat, bs_flat, s=40, alpha=0.7, label='Bilinear TN-sim')
    ax.set_xlabel('Ground truth Jaccard similarity')
    ax.set_ylabel('TN-sim')
    ax.set_title('TN-sim vs ground truth feature sharing')
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Add correlation values
    from scipy.stats import pearsonr
    r_full, _ = pearsonr(gt_flat, fs_flat)
    r_bil, _ = pearsonr(gt_flat, bs_flat)
    ax.text(0.05, 0.95, f'r(full)={r_full:.3f}\nr(bil)={r_bil:.3f}',
            transform=ax.transAxes, fontsize=10, verticalalignment='top',
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))

    plt.tight_layout()
    plt.savefig(os.path.join(IMG_DIR, 'tnsim_vs_groundtruth.png'), dpi=150, bbox_inches='tight')
    plt.close()

    print(f"   Saved plots to {IMG_DIR}/")

    # ================================================================
    # SUMMARY
    # ================================================================
    print("\n" + "=" * 70)
    print("LEVEL 2 SUMMARY")
    print("=" * 70)
    print(f"  Accuracy: {acc:.1%}")
    print(f"\n  Relative norms:")
    for c in range(8):
        r = class_norms[c]
        print(f"    C{c} ({CLASS_LABELS[c]:>3s}): res={r['rho_res']:.3f} bil={r['rho_bil']:.3f}")
    print(f"\n  Rank-1 TN-sim:")
    for c in range(8):
        print(f"    C{c} ({CLASS_LABELS[c]:>3s}): {rank1_results[c]['sim']:.4f}")
    print(f"\n  Feature sharing detection (single-feat rank-1 → multi-feat classes):")
    for src in [1, 2, 4]:
        feat = list(FEATURES[src])[0]
        targets_sharing = [t for t in range(8) if feat in FEATURES[t] and t != src and t != 0]
        targets_not = [t for t in range(8) if feat not in FEATURES[t] and t != 0 and len(FEATURES[t]) > 0]
        avg_sharing = np.mean([cross_sim[src, t] for t in targets_sharing]) if targets_sharing else 0
        avg_not = np.mean([cross_sim[src, t] for t in targets_not]) if targets_not else 0
        print(f"    C{src}({CLASS_LABELS[src]}): avg sim with shared={avg_sharing:+.3f}, "
              f"not shared={avg_not:+.3f}, gap={avg_sharing-avg_not:+.3f}")

    # TN-sim vs ground truth correlation
    r_full, _ = pearsonr(gt_flat, fs_flat)
    r_bil, _ = pearsonr(gt_flat, bs_flat)
    print(f"\n  Correlation with ground truth Jaccard:")
    print(f"    Full TN-sim: r={r_full:.3f}")
    print(f"    Bilinear TN-sim: r={r_bil:.3f}")

    # Save
    torch.save({
        'model_state': model.state_dict(),
        'class_norms': class_norms,
        'rank1_results': rank1_results,
        'cross_sim': cross_sim,
        'full_sim': full_sim,
        'bil_sim': bil_sim,
        'gt_sharing': gt_sharing,
        'accuracy': acc,
    }, os.path.join(os.path.dirname(__file__), 'level2_results.pt'))
    print(f"\n  Saved results to level2_results.pt")


if __name__ == '__main__':
    analyze_level2()
