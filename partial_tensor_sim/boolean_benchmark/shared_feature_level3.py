"""
Shared feature extraction on Level 3 (mixed degrees).

Method: For a pair (i, j), find rank-1 S such that TN-sim(T_i - S, T_j) -> 0.
This extracts the component of T_i that is shared with T_j.

Level 3 classes:
  C0: P=A∧B (degree 4)
  C1: A=1,B=0 (degree 2)
  C2: B=1,A=0 (degree 2)
  C3: C=1,A=0,B=0 (degree 2)
  C4: otherwise (default)
"""
import torch
import torch.nn.functional as F
import numpy as np
import os
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from model import (
    BooleanBilinear1Layer, make_level3_dataset, train_model,
    tn_inner_1layer, tn_sim_1layer, batched_tn_inner,
)

BASE_DIR = os.path.dirname(__file__)
IMG_DIR = os.path.join(BASE_DIR, 'images', 'level3')
os.makedirs(IMG_DIR, exist_ok=True)
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

CLASS_LABELS = ['P(d4)', 'A(d2)', 'B(d2)', 'C(d2)', 'def']


def optimize_shared_feature(L_i, R_i, D_i, L_j, R_j, D_j,
                            n_seeds=20, steps=3000, lr=1e-2, device='cuda'):
    """
    Find rank-1 S that minimizes TN-sim(T_i - S, T_j)^2.

    T_i - S is represented by:
      L = [L_i; l_s],  R = [R_i; r_s],  D = [D_i, -d_s]

    We optimize l_s (1, d_in), r_s (1, d_in), d_s (n_out, 1).
    """
    B = n_seeds
    rank_i = L_i.shape[0]
    d_in = L_i.shape[1]
    n_out = D_i.shape[0]

    # Expand targets for batch
    bL_i = L_i.unsqueeze(0).expand(B, -1, -1)
    bR_i = R_i.unsqueeze(0).expand(B, -1, -1)
    bD_i = D_i.unsqueeze(0).expand(B, -1, -1)
    bL_j = L_j.unsqueeze(0).expand(B, -1, -1)
    bR_j = R_j.unsqueeze(0).expand(B, -1, -1)
    bD_j = D_j.unsqueeze(0).expand(B, -1, -1)

    # Init rank-1 params
    w_l = torch.zeros(B, 1, d_in, device=device)
    w_r = torch.zeros(B, 1, d_in, device=device)
    w_d = torch.zeros(B, n_out, 1, device=device)
    for s in range(B):
        torch.manual_seed(s * 137 + 42)
        w_l[s] = torch.randn(1, d_in) * 0.1
        w_r[s] = torch.randn(1, d_in) * 0.1
        w_d[s] = torch.randn(n_out, 1) * 0.1

    w_l.requires_grad_(True)
    w_r.requires_grad_(True)
    w_d.requires_grad_(True)

    optimizer = torch.optim.Adam([w_l, w_r, w_d], lr=lr)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, steps)

    for step in range(steps):
        optimizer.zero_grad()

        # Build T_i - S: L_diff = [L_i; l_s], R_diff = [R_i; r_s], D_diff = [D_i, -d_s]
        L_diff = torch.cat([bL_i, w_l], dim=1)
        R_diff = torch.cat([bR_i, w_r], dim=1)
        D_diff = torch.cat([bD_i, -w_d], dim=2)

        # TN-sim(T_i - S, T_j)^2
        ab = batched_tn_inner(L_diff, R_diff, D_diff, bL_j, bR_j, bD_j)
        aa = batched_tn_inner(L_diff, R_diff, D_diff, L_diff, R_diff, D_diff)
        bb = batched_tn_inner(bL_j, bR_j, bD_j, bL_j, bR_j, bD_j)
        sim_sq = ab ** 2 / (aa.clamp(min=1e-12) * bb.clamp(min=1e-12))

        loss = sim_sq.mean()
        loss.backward()
        optimizer.step()
        scheduler.step()

    # Evaluate final
    with torch.no_grad():
        L_diff = torch.cat([bL_i, w_l], dim=1)
        R_diff = torch.cat([bR_i, w_r], dim=1)
        D_diff = torch.cat([bD_i, -w_d], dim=2)

        ab = batched_tn_inner(L_diff, R_diff, D_diff, bL_j, bR_j, bD_j)
        aa = batched_tn_inner(L_diff, R_diff, D_diff, L_diff, R_diff, D_diff)
        bb = batched_tn_inner(bL_j, bR_j, bD_j, bL_j, bR_j, bD_j)
        final_sim_sq = ab ** 2 / (aa.clamp(min=1e-12) * bb.clamp(min=1e-12))
        final_sim = ab / (torch.sqrt(aa.clamp(min=1e-12)) * torch.sqrt(bb.clamp(min=1e-12)))

        best = final_sim_sq.argmin().item()

    return {
        'l': w_l[best].detach(),
        'r': w_r[best].detach(),
        'd': w_d[best].detach(),
        'final_sim': final_sim[best].item(),
        'final_sim_sq': final_sim_sq[best].item(),
        'all_final_sims': final_sim.detach().cpu(),
    }


def initial_tn_sim(model, ci, cj):
    """TN-sim(T_ci, T_cj) before extraction."""
    L_i, R_i, D_i = model.get_augmented_weights_for_class(ci)
    L_j, R_j, D_j = model.get_augmented_weights_for_class(cj)
    return tn_sim_1layer(L_i, R_i, D_i, L_j, R_j, D_j).item()


def tn_sim_rank1_vs_class(l, r, d, model, c):
    """TN-sim(S, T_c)."""
    L_c, R_c, D_c = model.get_augmented_weights_for_class(c)
    return tn_sim_1layer(l, r, d, L_c, R_c, D_c).item()


def run_pair(model, ci, cj, n_seeds=20):
    """Run shared feature extraction for pair (ci, cj)."""
    L_i, R_i, D_i = model.get_augmented_weights_for_class(ci)
    L_j, R_j, D_j = model.get_augmented_weights_for_class(cj)

    init_sim = tn_sim_1layer(L_i, R_i, D_i, L_j, R_j, D_j).item()

    result = optimize_shared_feature(
        L_i, R_i, D_i, L_j, R_j, D_j,
        n_seeds=n_seeds, steps=3000, lr=1e-2, device=DEVICE
    )

    # S's similarity with all classes
    l, r, d = result['l'], result['r'], result['d']
    s_vs_all = [tn_sim_rank1_vs_class(l, r, d, model, c) for c in range(5)]

    # Normalized l and r
    l_norm = l / (l.norm() + 1e-12)
    r_norm = r / (r.norm() + 1e-12)

    return {
        'ci': ci, 'cj': cj,
        'init_sim': init_sim,
        'final_sim': result['final_sim'],
        'final_sim_sq': result['final_sim_sq'],
        's_vs_all': s_vs_all,
        'l': l.cpu(), 'r': r.cpu(), 'd': d.cpu(),
        'l_norm': l_norm.cpu(), 'r_norm': r_norm.cpu(),
    }


def main():
    print("=" * 70)
    print("SHARED FEATURE EXTRACTION: Level 3 (mixed degrees)")
    print("=" * 70)
    print(f"Device: {DEVICE}")

    # Train model
    X, labels = make_level3_dataset(DEVICE)
    print(f"\nDataset: {len(X)} patterns, 5 classes")
    for c in range(5):
        count = (labels == c).sum().item()
        print(f"  C{c} ({CLASS_LABELS[c]}): {count} patterns")

    model = BooleanBilinear1Layer(n_input=6, d_hidden=24, d_rank=12, n_classes=5).to(DEVICE)
    print("\nTraining 1-layer model...")
    acc = train_model(model, X, labels, epochs=5000, lr=0.01, wd=0.005,
                      print_every=1000, device=DEVICE)
    print(f"Final accuracy: {acc:.1%}")

    # Per-class accuracy
    with torch.no_grad():
        preds = model(X).argmax(dim=1)
        for c in range(5):
            mask = labels == c
            if mask.sum() > 0:
                ca = (preds[mask] == c).float().mean().item()
                print(f"  C{c} ({CLASS_LABELS[c]}): {ca:.1%}")

    # Test pairs
    pairs = [
        (1, 0, "C1(A)->C0(P): P uses A"),
        (2, 0, "C2(B)->C0(P): P uses B"),
        (1, 2, "C1(A)->C2(B): disjoint"),
        (1, 3, "C1(A)->C3(C): disjoint"),
        (3, 0, "C3(C)->C0(P): disjoint"),
        (0, 4, "C0(P)->C4(def): deg4 vs default"),
    ]

    results = []
    for ci, cj, desc in pairs:
        print(f"\n{'='*60}")
        print(f"Pair: {desc}")
        print(f"{'='*60}")

        r = run_pair(model, ci, cj, n_seeds=20)
        results.append(r)

        print(f"  Initial TN-sim(T_{ci}, T_{cj}): {r['init_sim']:.4f}")
        print(f"  Final TN-sim(T_{ci}-S, T_{cj}): {r['final_sim']:.4f}")
        print(f"  Final TN-sim^2:                  {r['final_sim_sq']:.6f}")
        print(f"  TN-sim(S, T_k) for all k:")
        for k in range(5):
            marker = " <--" if k == ci or k == cj else ""
            print(f"    k={k} ({CLASS_LABELS[k]}): {r['s_vs_all'][k]:.4f}{marker}")
        print(f"  S's l_norm: {r['l_norm'].squeeze().numpy()}")
        print(f"  S's r_norm: {r['r_norm'].squeeze().numpy()}")

    # ================================================================
    # PLOT
    # ================================================================
    print("\nGenerating plots...")

    n_pairs = len(pairs)
    fig, axes = plt.subplots(2, 3, figsize=(18, 11))

    for idx, (r, (ci, cj, desc)) in enumerate(zip(results, pairs)):
        ax = axes[idx // 3, idx % 3]

        # Bar chart: TN-sim(S, T_k) for all k
        x_pos = np.arange(5)
        sims = r['s_vs_all']
        colors = []
        for k in range(5):
            if k == ci:
                colors.append('C0')  # source
            elif k == cj:
                colors.append('C1')  # target
            else:
                colors.append('C7')  # gray
        bars = ax.bar(x_pos, sims, color=colors, edgecolor='black', linewidth=0.5)

        ax.set_xticks(x_pos)
        ax.set_xticklabels([f'C{k}\n{CLASS_LABELS[k]}' for k in range(5)], fontsize=7)
        ax.set_ylabel('TN-sim(S, T_k)')
        ax.axhline(0, color='black', linewidth=0.5)
        ax.set_ylim(-1.05, 1.05)
        ax.grid(True, alpha=0.3, axis='y')

        short_desc = f"C{ci}({CLASS_LABELS[ci]})->C{cj}({CLASS_LABELS[cj]})"
        ax.set_title(
            f"{short_desc}\n"
            f"init={r['init_sim']:.3f} final={r['final_sim']:.3f}",
            fontsize=9
        )

        # Annotate bars
        for k, v in enumerate(sims):
            ax.text(k, v + 0.03 * np.sign(v), f'{v:.2f}', ha='center', va='bottom' if v >= 0 else 'top',
                    fontsize=7)

    fig.suptitle(
        "Shared Feature Extraction: Level 3 (mixed degrees)\n"
        "Blue=source(i), Orange=target(j), Gray=other | S minimizes TN-sim(T_i - S, T_j)",
        fontsize=11, fontweight='bold'
    )
    plt.tight_layout(rect=[0, 0, 1, 0.93])
    save_path = os.path.join(IMG_DIR, 'shared_features.png')
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved plot to {save_path}")

    # ================================================================
    # SUMMARY TABLE
    # ================================================================
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    print(f"{'Pair':<25s} {'Init sim':>10s} {'Final sim':>10s} {'S~C0':>7s} {'S~C1':>7s} {'S~C2':>7s} {'S~C3':>7s} {'S~C4':>7s}")
    for r, (ci, cj, desc) in zip(results, pairs):
        pair_str = f"C{ci}->C{cj}"
        s_strs = [f"{r['s_vs_all'][k]:.3f}" for k in range(5)]
        print(f"{pair_str:<25s} {r['init_sim']:>10.4f} {r['final_sim']:>10.4f} {'  '.join(s_strs)}")

    print("\nKey question: When extracting S from C1(A)->C0(P), does S capture feature A?")
    r_a_to_p = results[0]
    print(f"  S from C1(A)->C0(P) similarities:")
    for k in range(5):
        print(f"    T_{k} ({CLASS_LABELS[k]}): {r_a_to_p['s_vs_all'][k]:.4f}")

    r_b_to_p = results[1]
    print(f"\n  S from C2(B)->C0(P) similarities:")
    for k in range(5):
        print(f"    T_{k} ({CLASS_LABELS[k]}): {r_b_to_p['s_vs_all'][k]:.4f}")

    r_disjoint_ab = results[2]
    r_disjoint_ac = results[3]
    r_disjoint_cp = results[4]
    print(f"\n  Disjoint pairs (expect near-zero extraction):")
    print(f"    C1(A)->C2(B) final sim: {r_disjoint_ab['final_sim']:.4f}")
    print(f"    C1(A)->C3(C) final sim: {r_disjoint_ac['final_sim']:.4f}")
    print(f"    C3(C)->C0(P) final sim: {r_disjoint_cp['final_sim']:.4f}")

    # Save
    save_results = {
        'model_state': model.state_dict(),
        'accuracy': acc,
        'pairs': [(ci, cj, desc) for ci, cj, desc in pairs],
        'results': results,
    }
    results_path = os.path.join(BASE_DIR, 'shared_feature_level3_results.pt')
    torch.save(save_results, results_path)
    print(f"\n  Saved results to {results_path}")


if __name__ == '__main__':
    main()
