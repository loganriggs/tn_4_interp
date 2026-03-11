"""
Level 1: Two AND features, 4 classes.
  A = x0 AND x1, B = x2 AND x3
  class 0: A=0, B=0 | class 1: A=1, B=0 | class 2: A=0, B=1 | class 3: A=1, B=1

Ground truth:
  - class 1 uses feature A = x0*x1 (rank-1 bilinear)
  - class 2 uses feature B = x2*x3 (rank-1 bilinear)
  - class 3 uses both A and B
  - class 0 is default (neither)
  - classes 1,3 share feature A
  - classes 2,3 share feature B
"""
import torch
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

IMG_DIR = os.path.join(os.path.dirname(__file__), 'images', 'level1')
os.makedirs(IMG_DIR, exist_ok=True)
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'


def analyze_level1():
    print("=" * 70)
    print("LEVEL 1: Two AND features, 4 classes")
    print("=" * 70)

    # ================================================================
    # 1. TRAIN MODEL
    # ================================================================
    print("\n1. Training model...")
    X, labels = make_level1_dataset(DEVICE)
    print(f"   Dataset: {len(X)} patterns, {labels.max().item()+1} classes")
    for c in range(4):
        count = (labels == c).sum().item()
        print(f"   Class {c}: {count} patterns")

    model = BooleanBilinear1Layer(n_input=4, d_hidden=16, d_rank=8, n_classes=4).to(DEVICE)
    acc = train_model(model, X, labels, epochs=3000, lr=0.01, wd=0.01,
                      print_every=1000, device=DEVICE)
    print(f"   Final accuracy: {acc:.1%}")
    assert acc > 0.95, f"Model didn't learn the task! acc={acc:.1%}"

    # ================================================================
    # 2. RELATIVE NORMS (residual vs bilinear fraction)
    # ================================================================
    print("\n2. Relative norms per class...")
    print(f"   {'Class':>5s}  {'ρ(res)':>8s}  {'ρ(bil)':>8s}  {'sum':>6s}")
    class_norms = {}
    for c in range(4):
        r = compute_relative_norms(model, c, DEVICE)
        class_norms[c] = r
        print(f"   {c:>5d}  {r['rho_res']:>8.4f}  {r['rho_bil']:>8.4f}  "
              f"{r['rho_res']+r['rho_bil']:>6.4f}")

    # ================================================================
    # 3. RANK-1 OPTIMIZATION PER CLASS
    # ================================================================
    print("\n3. Rank-1 optimization per class...")
    rank1_results = {}
    for c in range(4):
        L_T, R_T, D_T = model.get_augmented_weights_for_class(c)
        result = optimize_rank1(L_T, R_T, D_T, n_seeds=20, steps=2000,
                                lr=0.01, device=DEVICE)
        rank1_results[c] = result
        print(f"   Class {c}: TN-sim = {result['sim']:.4f}")

    # ================================================================
    # 4. INSPECT RANK-1 VECTORS (compare to ground truth)
    # ================================================================
    print("\n4. Rank-1 vector inspection (should recover AND gate inputs)...")
    print("   Ground truth: class 1 → x0*x1, class 2 → x2*x3")
    for c in range(4):
        l = rank1_results[c]['l'].squeeze().cpu().numpy()  # (n+1,)
        r = rank1_results[c]['r'].squeeze().cpu().numpy()
        d = rank1_results[c]['d'].squeeze().cpu().numpy()
        # Show which input dims have largest magnitude
        l_input = l[:-1]  # drop constant dim
        r_input = r[:-1]
        top_l = np.argsort(np.abs(l_input))[::-1][:2]
        top_r = np.argsort(np.abs(r_input))[::-1][:2]
        print(f"   Class {c}: sim={rank1_results[c]['sim']:.3f}")
        print(f"     l top dims: {top_l} (vals: {l_input[top_l]})")
        print(f"     r top dims: {top_r} (vals: {r_input[top_r]})")
        print(f"     l[const]={l[-1]:.3f}, r[const]={r[-1]:.3f}, d={d}")

    # ================================================================
    # 5. PAIRWISE TN-SIM MATRIX
    # ================================================================
    print("\n5. Pairwise TN-sim matrix...")
    sim_matrix = np.zeros((4, 4))
    for ci in range(4):
        for cj in range(4):
            Li, Ri, Di = model.get_augmented_weights_for_class(ci)
            Lj, Rj, Dj = model.get_augmented_weights_for_class(cj)
            sim_matrix[ci, cj] = tn_sim_1layer(Li, Ri, Di, Lj, Rj, Dj).item()
    print("   Full tensor TN-sim:")
    print("         ", "  ".join(f"C{c}" for c in range(4)))
    for ci in range(4):
        vals = "  ".join(f"{sim_matrix[ci,cj]:.3f}" for cj in range(4))
        print(f"   C{ci}:  {vals}")

    # Also bilinear-only TN-sim
    bil_sim_matrix = np.zeros((4, 4))
    for ci in range(4):
        for cj in range(4):
            Li, Ri, Di = model.get_bilinear_only_weights_for_class(ci)
            Lj, Rj, Dj = model.get_bilinear_only_weights_for_class(cj)
            bil_sim_matrix[ci, cj] = tn_sim_1layer(Li, Ri, Di, Lj, Rj, Dj).item()
    print("\n   Bilinear-only TN-sim:")
    print("         ", "  ".join(f"C{c}" for c in range(4)))
    for ci in range(4):
        vals = "  ".join(f"{bil_sim_matrix[ci,cj]:.3f}" for cj in range(4))
        print(f"   C{ci}:  {vals}")

    # ================================================================
    # 6. CROSS-CLASS PROJECTION (feature sharing detection)
    # ================================================================
    print("\n6. Cross-class TN-sim: sim(rank1_i, T_j)")
    print("   (How well does class i's rank-1 align with class j's tensor?)")
    proj_matrix = np.zeros((4, 4))
    for i in range(4):
        li, ri, di = rank1_results[i]['l'], rank1_results[i]['r'], rank1_results[i]['d']
        for j in range(4):
            Lj, Rj, Dj = model.get_augmented_weights_for_class(j)
            proj_matrix[i, j] = tn_sim_1layer(li, ri, di, Lj, Rj, Dj).item()
    print("            target class")
    print("   rank1    ", "  ".join(f"C{c}" for c in range(4)))
    for i in range(4):
        vals = "  ".join(f"{proj_matrix[i,j]:+.3f}" for j in range(4))
        print(f"   C{i}:  {vals}")

    print("\n   Expected: C1's rank-1 (feature A) has high sim with C3 (uses A)")
    print("             C2's rank-1 (feature B) has high sim with C3 (uses B)")
    print("             C1 and C2 rank-1s should NOT align (disjoint features)")

    # ================================================================
    # 7. PLOTS
    # ================================================================
    print("\n7. Generating plots...")

    # Plot 1: Relative norms bar chart
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    ax = axes[0]
    classes = range(4)
    rho_res = [class_norms[c]['rho_res'] for c in classes]
    rho_bil = [class_norms[c]['rho_bil'] for c in classes]
    x_pos = np.arange(4)
    ax.bar(x_pos - 0.15, rho_res, 0.3, label='residual', color='C0')
    ax.bar(x_pos + 0.15, rho_bil, 0.3, label='bilinear', color='C1')
    ax.set_xticks(x_pos)
    ax.set_xticklabels([f'C{c}\n({["∅","A","B","AB"][c]})' for c in range(4)])
    ax.set_ylabel('ρ (relative norm fraction)')
    ax.set_title('Relative norms per class')
    ax.legend()
    ax.set_ylim(-0.1, 1.1)
    ax.axhline(0, color='k', linewidth=0.5)
    ax.grid(True, alpha=0.3)

    # Plot 2: TN-sim matrices (full and bilinear-only)
    ax = axes[1]
    im = ax.imshow(sim_matrix, cmap='RdBu_r', vmin=-1, vmax=1)
    ax.set_xticks(range(4)); ax.set_yticks(range(4))
    ax.set_xticklabels([f'C{c}' for c in range(4)])
    ax.set_yticklabels([f'C{c}' for c in range(4)])
    ax.set_title('Full tensor TN-sim')
    for i in range(4):
        for j in range(4):
            ax.text(j, i, f'{sim_matrix[i,j]:.2f}', ha='center', va='center',
                    fontsize=9, color='white' if abs(sim_matrix[i,j]) > 0.5 else 'black')
    plt.colorbar(im, ax=ax, shrink=0.8)

    ax = axes[2]
    im = ax.imshow(bil_sim_matrix, cmap='RdBu_r', vmin=-1, vmax=1)
    ax.set_xticks(range(4)); ax.set_yticks(range(4))
    ax.set_xticklabels([f'C{c}' for c in range(4)])
    ax.set_yticklabels([f'C{c}' for c in range(4)])
    ax.set_title('Bilinear-only TN-sim')
    for i in range(4):
        for j in range(4):
            ax.text(j, i, f'{bil_sim_matrix[i,j]:.2f}', ha='center', va='center',
                    fontsize=9, color='white' if abs(bil_sim_matrix[i,j]) > 0.5 else 'black')
    plt.colorbar(im, ax=ax, shrink=0.8)

    plt.tight_layout()
    plt.savefig(os.path.join(IMG_DIR, 'norms_and_tnsim.png'), dpi=150, bbox_inches='tight')
    plt.close()

    # Plot 3: Projection matrix
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    ax = axes[0]
    im = ax.imshow(proj_matrix, cmap='RdBu_r', vmin=-0.5, vmax=0.5)
    ax.set_xlabel('Target class')
    ax.set_ylabel('Rank-1 source class')
    ax.set_xticks(range(4)); ax.set_yticks(range(4))
    ax.set_xticklabels([f'C{c}' for c in range(4)])
    ax.set_yticklabels([f'C{c}' for c in range(4)])
    ax.set_title('TN-sim(rank1_source, T_target)')
    for i in range(4):
        for j in range(4):
            ax.text(j, i, f'{proj_matrix[i,j]:.3f}', ha='center', va='center',
                    fontsize=9, color='white' if abs(proj_matrix[i,j]) > 0.25 else 'black')
    plt.colorbar(im, ax=ax, shrink=0.8)

    # Ground truth feature sharing matrix
    ax = axes[1]
    gt_sharing = np.array([
        #  C0  C1  C2  C3
        [0, 0, 0, 0],  # C0 shares nothing
        [0, 1, 0, 1],  # C1 has A, shares with C3
        [0, 0, 1, 1],  # C2 has B, shares with C3
        [0, 1, 1, 1],  # C3 has A+B, shares with C1 and C2
    ], dtype=float)
    im = ax.imshow(gt_sharing, cmap='Blues', vmin=0, vmax=1)
    ax.set_xlabel('Class')
    ax.set_ylabel('Class')
    ax.set_xticks(range(4)); ax.set_yticks(range(4))
    ax.set_xticklabels([f'C{c}' for c in range(4)])
    ax.set_yticklabels([f'C{c}' for c in range(4)])
    ax.set_title('Ground truth: feature sharing')
    for i in range(4):
        for j in range(4):
            ax.text(j, i, f'{gt_sharing[i,j]:.0f}', ha='center', va='center',
                    fontsize=12)
    plt.colorbar(im, ax=ax, shrink=0.8)

    plt.tight_layout()
    plt.savefig(os.path.join(IMG_DIR, 'projection_matrix.png'), dpi=150, bbox_inches='tight')
    plt.close()

    # Plot 4: Rank-1 vectors visualization
    fig, axes = plt.subplots(4, 3, figsize=(12, 12))
    for c in range(4):
        l = rank1_results[c]['l'].squeeze().cpu().numpy()
        r = rank1_results[c]['r'].squeeze().cpu().numpy()
        d = rank1_results[c]['d'].squeeze().cpu().numpy()

        ax = axes[c, 0]
        ax.bar(range(len(l)), l)
        ax.set_title(f'C{c}: l vector (sim={rank1_results[c]["sim"]:.3f})')
        ax.set_xticks(range(len(l)))
        ax.set_xticklabels([f'x{i}' for i in range(4)] + ['const'])
        ax.grid(True, alpha=0.3)

        ax = axes[c, 1]
        ax.bar(range(len(r)), r)
        ax.set_title(f'C{c}: r vector')
        ax.set_xticks(range(len(r)))
        ax.set_xticklabels([f'x{i}' for i in range(4)] + ['const'])
        ax.grid(True, alpha=0.3)

        ax = axes[c, 2]
        d_arr = np.atleast_1d(d)
        ax.bar(range(len(d_arr)), d_arr)
        ax.set_title(f'C{c}: d vector')
        ax.grid(True, alpha=0.3)

    plt.suptitle('Level 1: Rank-1 vectors per class\n'
                 'C1 should select x0,x1 | C2 should select x2,x3', fontsize=13)
    plt.tight_layout()
    plt.savefig(os.path.join(IMG_DIR, 'rank1_vectors.png'), dpi=150, bbox_inches='tight')
    plt.close()

    print(f"   Saved plots to {IMG_DIR}/")

    # ================================================================
    # 8. SUMMARY
    # ================================================================
    print("\n" + "=" * 70)
    print("LEVEL 1 SUMMARY")
    print("=" * 70)
    print(f"  Model accuracy: {acc:.1%}")
    print(f"\n  Relative norms (ρ_residual / ρ_bilinear):")
    for c in range(4):
        r = class_norms[c]
        print(f"    C{c} ({['∅','A','B','AB'][c]}): {r['rho_res']:.3f} / {r['rho_bil']:.3f}")
    print(f"\n  Rank-1 TN-sim per class:")
    for c in range(4):
        print(f"    C{c}: {rank1_results[c]['sim']:.4f}")
    print(f"\n  Feature sharing (TN-sim of C1's rank-1 with other classes):")
    print(f"    C1→C0: {proj_matrix[1,0]:+.3f} (should be ~0 or negative)")
    print(f"    C1→C1: {proj_matrix[1,1]:+.3f} (self, high)")
    print(f"    C1→C2: {proj_matrix[1,2]:+.3f} (should be low, disjoint features)")
    print(f"    C1→C3: {proj_matrix[1,3]:+.3f} (should be positive, shared feature A)")
    print(f"\n  Feature sharing (TN-sim of C2's rank-1 with other classes):")
    print(f"    C2→C0: {proj_matrix[2,0]:+.3f} (should be ~0 or negative)")
    print(f"    C2→C1: {proj_matrix[2,1]:+.3f} (should be low, disjoint features)")
    print(f"    C2→C2: {proj_matrix[2,2]:+.3f} (self, high)")
    print(f"    C2→C3: {proj_matrix[2,3]:+.3f} (should be positive, shared feature B)")

    # Save results
    torch.save({
        'model_state': model.state_dict(),
        'class_norms': class_norms,
        'rank1_results': rank1_results,
        'sim_matrix': sim_matrix,
        'bil_sim_matrix': bil_sim_matrix,
        'proj_matrix': proj_matrix,
        'accuracy': acc,
    }, os.path.join(os.path.dirname(__file__), 'level1_results.pt'))
    print(f"\n  Saved results to level1_results.pt")


if __name__ == '__main__':
    analyze_level1()
