"""
Level 3: Mixed degrees with 2-layer model.
  Features: A = x0∧x1, B = x2∧x3, C = x4∧x5
  P = A AND B (degree 4, needs 2 layers)

  Classes (mutually exclusive):
    class 0: P=1              (degree 4)
    class 1: A=1, B=0         (degree 2)
    class 2: B=1, A=0         (degree 2)
    class 3: C=1, A=0, B=0    (degree 2)
    class 4: otherwise         (default)

Tests: Can we distinguish degree-2 vs degree-4 classes?
Does the 2-layer model outperform 1-layer on the degree-4 class?
"""
import torch
import numpy as np
import os
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from model import (
    BooleanBilinear1Layer, BooleanBilinear2Layer,
    make_level3_dataset, train_model,
    tn_inner_1layer, tn_sim_1layer, compute_relative_norms,
    optimize_rank1,
)

IMG_DIR = os.path.join(os.path.dirname(__file__), 'images', 'level3')
os.makedirs(IMG_DIR, exist_ok=True)
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

CLASS_LABELS = ['P(d4)', 'A(d2)', 'B(d2)', 'C(d2)', 'def']
CLASS_DEGREES = [4, 2, 2, 2, 0]


def analyze_1layer():
    """Train and analyze 1-layer model (can't do degree-4)."""
    print("\n" + "-" * 50)
    print("1-LAYER MODEL")
    print("-" * 50)

    X, labels = make_level3_dataset(DEVICE)
    model = BooleanBilinear1Layer(n_input=6, d_hidden=24, d_rank=12, n_classes=5).to(DEVICE)
    acc = train_model(model, X, labels, epochs=5000, lr=0.01, wd=0.005,
                      print_every=2000, device=DEVICE)
    print(f"  Accuracy: {acc:.1%}")

    # Per-class accuracy
    with torch.no_grad():
        preds = model(X).argmax(dim=1)
        for c in range(5):
            mask = labels == c
            if mask.sum() > 0:
                ca = (preds[mask] == c).float().mean().item()
                print(f"    Class {c} ({CLASS_LABELS[c]}): {ca:.1%} ({mask.sum().item()} samples)")

    # Relative norms
    class_norms = {}
    for c in range(5):
        r = compute_relative_norms(model, c, DEVICE)
        class_norms[c] = r

    # Rank-1
    rank1_results = {}
    for c in range(5):
        L_T, R_T, D_T = model.get_augmented_weights_for_class(c)
        result = optimize_rank1(L_T, R_T, D_T, n_seeds=20, steps=2000,
                                lr=0.01, device=DEVICE)
        rank1_results[c] = result

    # Cross-class TN-sim
    cross_sim = np.zeros((5, 5))
    for i in range(5):
        li, ri, di = rank1_results[i]['l'], rank1_results[i]['r'], rank1_results[i]['d']
        for j in range(5):
            Lj, Rj, Dj = model.get_augmented_weights_for_class(j)
            cross_sim[i, j] = tn_sim_1layer(li, ri, di, Lj, Rj, Dj).item()

    return model, acc, class_norms, rank1_results, cross_sim


def analyze_2layer():
    """Train and analyze 2-layer model (can do degree-4)."""
    print("\n" + "-" * 50)
    print("2-LAYER MODEL")
    print("-" * 50)

    X, labels = make_level3_dataset(DEVICE)
    model = BooleanBilinear2Layer(n_input=6, d_hidden=24, d_rank=12, n_classes=5).to(DEVICE)
    acc = train_model(model, X, labels, epochs=8000, lr=0.01, wd=0.005,
                      print_every=2000, device=DEVICE)
    print(f"  Accuracy: {acc:.1%}")

    with torch.no_grad():
        preds = model(X).argmax(dim=1)
        for c in range(5):
            mask = labels == c
            if mask.sum() > 0:
                ca = (preds[mask] == c).float().mean().item()
                print(f"    Class {c} ({CLASS_LABELS[c]}): {ca:.1%} ({mask.sum().item()} samples)")

    return model, acc


def analyze_level3():
    print("=" * 70)
    print("LEVEL 3: Mixed degrees (degree 2 + degree 4)")
    print("=" * 70)

    X, labels = make_level3_dataset(DEVICE)
    print(f"\nDataset: {len(X)} patterns, {labels.max().item()+1} classes")
    for c in range(5):
        count = (labels == c).sum().item()
        print(f"  Class {c} ({CLASS_LABELS[c]}, degree {CLASS_DEGREES[c]}): {count} patterns")

    # ================================================================
    # 1-LAYER ANALYSIS
    # ================================================================
    model_1L, acc_1L, norms_1L, rank1_1L, cross_1L = analyze_1layer()

    print(f"\n  1-Layer relative norms:")
    print(f"  {'Class':>5s} {'Label':>8s} {'Degree':>6s}  {'ρ(res)':>8s}  {'ρ(bil)':>8s}  {'R1 sim':>7s}")
    for c in range(5):
        r = norms_1L[c]
        print(f"  {c:>5d} {CLASS_LABELS[c]:>8s} {CLASS_DEGREES[c]:>6d}  "
              f"{r['rho_res']:>8.4f}  {r['rho_bil']:>8.4f}  {rank1_1L[c]['sim']:>7.4f}")

    # ================================================================
    # 2-LAYER ANALYSIS
    # ================================================================
    model_2L, acc_2L = analyze_2layer()

    # ================================================================
    # COMPARISON: 1L vs 2L on degree-4 class
    # ================================================================
    print("\n" + "-" * 50)
    print("COMPARISON: 1-Layer vs 2-Layer")
    print("-" * 50)

    with torch.no_grad():
        X_test = X
        preds_1L = model_1L(X_test).argmax(dim=1)
        preds_2L = model_2L(X_test).argmax(dim=1)

        for c in range(5):
            mask = labels == c
            if mask.sum() > 0:
                acc1 = (preds_1L[mask] == c).float().mean().item()
                acc2 = (preds_2L[mask] == c).float().mean().item()
                print(f"  Class {c} ({CLASS_LABELS[c]}, deg {CLASS_DEGREES[c]}): "
                      f"1L={acc1:.0%} 2L={acc2:.0%}")

    # ================================================================
    # PLOTS
    # ================================================================
    print("\nGenerating plots...")

    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    # Plot 1: Relative norms by degree
    ax = axes[0]
    x_pos = np.arange(5)
    rho_res = [norms_1L[c]['rho_res'] for c in range(5)]
    rho_bil = [norms_1L[c]['rho_bil'] for c in range(5)]
    colors_res = ['C3' if CLASS_DEGREES[c] == 4 else 'C0' for c in range(5)]
    colors_bil = ['C3' if CLASS_DEGREES[c] == 4 else 'C1' for c in range(5)]
    ax.bar(x_pos - 0.15, rho_res, 0.3, label='residual', color='C0')
    ax.bar(x_pos + 0.15, rho_bil, 0.3, label='bilinear', color='C1')
    ax.set_xticks(x_pos)
    ax.set_xticklabels([f'C{c}\n{CLASS_LABELS[c]}\ndeg{CLASS_DEGREES[c]}' for c in range(5)],
                       fontsize=8)
    ax.set_ylabel('ρ (relative norm fraction)')
    ax.set_title('1-Layer: Relative norms (degree-4 class in red if present)')
    ax.legend()
    ax.set_ylim(-0.1, 1.1)
    ax.grid(True, alpha=0.3)

    # Plot 2: Cross-class TN-sim (1-layer)
    ax = axes[1]
    im = ax.imshow(cross_1L, cmap='RdBu_r', vmin=-1, vmax=1)
    ax.set_xticks(range(5)); ax.set_yticks(range(5))
    ax.set_xticklabels([f'C{c}({CLASS_LABELS[c]})' for c in range(5)], fontsize=7, rotation=45, ha='right')
    ax.set_yticklabels([f'C{c}({CLASS_LABELS[c]})' for c in range(5)], fontsize=7)
    ax.set_title('1-Layer: TN-sim(rank1_i, T_j)')
    for i in range(5):
        for j in range(5):
            ax.text(j, i, f'{cross_1L[i,j]:.2f}', ha='center', va='center',
                    fontsize=8, color='white' if abs(cross_1L[i,j]) > 0.5 else 'black')
    plt.colorbar(im, ax=ax, shrink=0.8)

    # Plot 3: Rank-1 TN-sim vs degree
    ax = axes[2]
    sims = [rank1_1L[c]['sim'] for c in range(5)]
    colors = ['C3' if CLASS_DEGREES[c] == 4 else 'C0' for c in range(5)]
    ax.bar(x_pos, sims, color=colors)
    ax.set_xticks(x_pos)
    ax.set_xticklabels([f'C{c}\n{CLASS_LABELS[c]}\ndeg{CLASS_DEGREES[c]}' for c in range(5)],
                       fontsize=8)
    ax.set_ylabel('Rank-1 TN-sim')
    ax.set_title('1-Layer: Rank-1 quality (red = degree-4)')
    ax.set_ylim(0.5, 1.05)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(os.path.join(IMG_DIR, 'level3_1layer.png'), dpi=150, bbox_inches='tight')
    plt.close()

    # Comparison bar chart
    fig, ax = plt.subplots(figsize=(10, 5))
    with torch.no_grad():
        accs_1L = []
        accs_2L = []
        for c in range(5):
            mask = labels == c
            if mask.sum() > 0:
                accs_1L.append((preds_1L[mask] == c).float().mean().item())
                accs_2L.append((preds_2L[mask] == c).float().mean().item())
            else:
                accs_1L.append(0)
                accs_2L.append(0)

    x_pos = np.arange(5)
    ax.bar(x_pos - 0.15, accs_1L, 0.3, label='1-Layer', color='C0')
    ax.bar(x_pos + 0.15, accs_2L, 0.3, label='2-Layer', color='C1')
    ax.set_xticks(x_pos)
    ax.set_xticklabels([f'C{c}\n{CLASS_LABELS[c]}\ndeg{CLASS_DEGREES[c]}' for c in range(5)],
                       fontsize=8)
    ax.set_ylabel('Per-class accuracy')
    ax.set_title('1-Layer vs 2-Layer: Per-class accuracy')
    ax.legend()
    ax.set_ylim(0, 1.1)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(IMG_DIR, 'level3_comparison.png'), dpi=150, bbox_inches='tight')
    plt.close()

    print(f"  Saved plots to {IMG_DIR}/")

    # ================================================================
    # SUMMARY
    # ================================================================
    print("\n" + "=" * 70)
    print("LEVEL 3 SUMMARY")
    print("=" * 70)
    print(f"  1-Layer accuracy: {acc_1L:.1%}")
    print(f"  2-Layer accuracy: {acc_2L:.1%}")
    print(f"\n  Key question: Can 1-layer handle degree-4 (class 0 = P = A∧B)?")
    print(f"    1-Layer class 0 acc: {accs_1L[0]:.0%}")
    print(f"    2-Layer class 0 acc: {accs_2L[0]:.0%}")
    print(f"\n  1-Layer relative norms and rank-1 quality:")
    for c in range(5):
        r = norms_1L[c]
        print(f"    C{c} ({CLASS_LABELS[c]}, deg{CLASS_DEGREES[c]}): "
              f"ρ_res={r['rho_res']:.3f} ρ_bil={r['rho_bil']:.3f} "
              f"R1_sim={rank1_1L[c]['sim']:.4f}")

    # Save
    torch.save({
        'model_1L_state': model_1L.state_dict(),
        'model_2L_state': model_2L.state_dict(),
        'acc_1L': acc_1L,
        'acc_2L': acc_2L,
        'norms_1L': norms_1L,
        'rank1_1L': rank1_1L,
        'cross_1L': cross_1L,
        'accs_1L': accs_1L,
        'accs_2L': accs_2L,
    }, os.path.join(os.path.dirname(__file__), 'level3_results.pt'))
    print(f"\n  Saved results to level3_results.pt")


if __name__ == '__main__':
    analyze_level3()
