"""
Shared feature extraction via rank-1 tensor subtraction.

For a pair of classes (i, j), find rank-1 tensor S such that TN-sim(T_i - S, T_j) → 0.
S captures the shared feature between classes i and j.

Level 2: 3 AND features (A, B, C), 8 classes.
"""
import sys
import os
import torch
import torch.nn.functional as F
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from model import (BooleanBilinear1Layer, make_level2_dataset, train_model,
                   tn_inner_1layer, tn_sim_1layer, batched_tn_inner)

CLASS_LABELS = ['∅', 'A', 'B', 'AB', 'C', 'AC', 'BC', 'ABC']
FEATURES = {
    0: set(), 1: {'A'}, 2: {'B'}, 3: {'A','B'},
    4: {'C'}, 5: {'A','C'}, 6: {'B','C'}, 7: {'A','B','C'}
}

PAIRS = [
    # Sharing pairs
    (1, 3, 'A'),   # C1(A) → C3(AB): share A
    (1, 5, 'A'),   # C1(A) → C5(AC): share A
    (2, 3, 'B'),   # C2(B) → C3(AB): share B
    (4, 5, 'C'),   # C4(C) → C5(AC): share C
    (1, 7, 'A'),   # C1(A) → C7(ABC): share A
    # Non-sharing pairs
    (1, 2, None),   # C1(A) → C2(B): disjoint
    (1, 4, None),   # C1(A) → C4(C): disjoint
]


def optimize_shared_feature(model, ci, cj, n_seeds=20, steps=3000, lr=1e-2, device='cuda'):
    """
    Find rank-1 S minimizing TN-sim(T_i - S, T_j)^2.

    T_i - S represented as: L=[L_i; l_s], R=[R_i; r_s], D=[D_i, -d_s]
    """
    # Get augmented weights for both classes
    L_i, R_i, D_i = model.get_augmented_weights_for_class(ci)
    L_j, R_j, D_j = model.get_augmented_weights_for_class(cj)

    L_i, R_i, D_i = L_i.to(device), R_i.to(device), D_i.to(device)
    L_j, R_j, D_j = L_j.to(device), R_j.to(device), D_j.to(device)

    rank_i = L_i.shape[0]
    d_in = L_i.shape[1]  # n+1 = 7
    n_out = D_i.shape[0]  # 1

    B = n_seeds

    # Initialize rank-1 params for S: l_s (1, d_in), r_s (1, d_in), d_s (n_out, 1)
    w_l = torch.randn(B, 1, d_in, device=device) * 0.1
    w_r = torch.randn(B, 1, d_in, device=device) * 0.1
    w_d = torch.randn(B, n_out, 1, device=device) * 0.1

    for i in range(B):
        torch.manual_seed(i * 137 + 42)
        w_l[i] = torch.randn(1, d_in) * 0.1
        w_r[i] = torch.randn(1, d_in) * 0.1
        w_d[i] = torch.randn(n_out, 1) * 0.1

    w_l = w_l.clone().requires_grad_(True)
    w_r = w_r.clone().requires_grad_(True)
    w_d = w_d.clone().requires_grad_(True)

    optimizer = torch.optim.Adam([w_l, w_r, w_d], lr=lr)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, steps)

    # Batch the target weights
    bL_i = L_i.unsqueeze(0).expand(B, -1, -1)
    bR_i = R_i.unsqueeze(0).expand(B, -1, -1)
    bD_i = D_i.unsqueeze(0).expand(B, -1, -1)
    bL_j = L_j.unsqueeze(0).expand(B, -1, -1)
    bR_j = R_j.unsqueeze(0).expand(B, -1, -1)
    bD_j = D_j.unsqueeze(0).expand(B, -1, -1)

    for step in range(steps):
        optimizer.zero_grad()

        # Construct T_i - S: concatenate L_i with l_s, R_i with r_s, D_i with -d_s
        L_diff = torch.cat([bL_i, w_l], dim=1)      # (B, rank_i+1, d_in)
        R_diff = torch.cat([bR_i, w_r], dim=1)      # (B, rank_i+1, d_in)
        D_diff = torch.cat([bD_i, -w_d], dim=2)     # (B, n_out, rank_i+1)

        # TN-sim(T_i - S, T_j)^2
        ab = batched_tn_inner(L_diff, R_diff, D_diff, bL_j, bR_j, bD_j)
        aa = batched_tn_inner(L_diff, R_diff, D_diff, L_diff, R_diff, D_diff)
        bb = batched_tn_inner(bL_j, bR_j, bD_j, bL_j, bR_j, bD_j)

        sim_sq = (ab ** 2) / (aa.clamp(min=1e-12) * bb.clamp(min=1e-12))
        loss = sim_sq.mean()

        loss.backward()
        optimizer.step()
        scheduler.step()

    # Find best seed
    with torch.no_grad():
        L_diff = torch.cat([bL_i, w_l], dim=1)
        R_diff = torch.cat([bR_i, w_r], dim=1)
        D_diff = torch.cat([bD_i, -w_d], dim=2)

        ab = batched_tn_inner(L_diff, R_diff, D_diff, bL_j, bR_j, bD_j)
        aa = batched_tn_inner(L_diff, R_diff, D_diff, L_diff, R_diff, D_diff)
        bb = batched_tn_inner(bL_j, bR_j, bD_j, bL_j, bR_j, bD_j)

        final_sims = ab / (torch.sqrt(aa.clamp(min=1e-12)) * torch.sqrt(bb.clamp(min=1e-12)))
        best = final_sims.abs().argmin().item()

    return {
        'l_s': w_l[best].detach().cpu(),
        'r_s': w_r[best].detach().cpu(),
        'd_s': w_d[best].detach().cpu(),
        'final_sim': final_sims[best].item(),
        'all_final_sims': final_sims.cpu(),
    }


def compute_S_sim_with_all_classes(model, l_s, r_s, d_s, device='cuda'):
    """Compute TN-sim(S, T_k) for all classes k."""
    l_s, r_s, d_s = l_s.to(device), r_s.to(device), d_s.to(device)
    sims = []
    for k in range(model.n_classes):
        L_k, R_k, D_k = model.get_augmented_weights_for_class(k)
        L_k, R_k, D_k = L_k.to(device), R_k.to(device), D_k.to(device)
        sim = tn_sim_1layer(l_s, r_s, d_s, L_k, R_k, D_k).item()
        sims.append(sim)
    return sims


def main():
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Device: {device}")

    # Train model
    print("\n=== Training Level 2 Model ===")
    torch.manual_seed(42)
    model = BooleanBilinear1Layer(n_input=6, d_hidden=24, d_rank=12, n_classes=8)
    X, labels = make_level2_dataset(device=device)
    model.to(device)
    acc = train_model(model, X, labels, epochs=3000, lr=0.01, wd=0.01,
                      print_every=500, device=device)
    print(f"Final accuracy: {acc:.1%}")
    assert acc > 0.99, f"Model didn't converge: {acc:.1%}"

    # Compute baseline TN-sim matrix
    print("\n=== Baseline TN-sim Matrix ===")
    sim_matrix = torch.zeros(8, 8)
    for i in range(8):
        for j in range(8):
            L_i, R_i, D_i = model.get_augmented_weights_for_class(i)
            L_j, R_j, D_j = model.get_augmented_weights_for_class(j)
            L_i, R_i, D_i = L_i.to(device), R_i.to(device), D_i.to(device)
            L_j, R_j, D_j = L_j.to(device), R_j.to(device), D_j.to(device)
            sim_matrix[i, j] = tn_sim_1layer(L_i, R_i, D_i, L_j, R_j, D_j).item()

    print("     " + "  ".join(f"{CLASS_LABELS[j]:>5s}" for j in range(8)))
    for i in range(8):
        row = "  ".join(f"{sim_matrix[i,j]:+.3f}" for j in range(8))
        print(f"{CLASS_LABELS[i]:>3s}: {row}")

    # Run shared feature extraction for each pair
    print("\n=== Shared Feature Extraction ===")
    results = {}

    for ci, cj, shared_feat in PAIRS:
        label = f"C{ci}({CLASS_LABELS[ci]})→C{cj}({CLASS_LABELS[cj]})"
        if shared_feat:
            label += f" [share {shared_feat}]"
        else:
            label += " [disjoint]"
        print(f"\n--- {label} ---")

        # Initial TN-sim
        L_i, R_i, D_i = model.get_augmented_weights_for_class(ci)
        L_j, R_j, D_j = model.get_augmented_weights_for_class(cj)
        L_i, R_i, D_i = L_i.to(device), R_i.to(device), D_i.to(device)
        L_j, R_j, D_j = L_j.to(device), R_j.to(device), D_j.to(device)
        initial_sim = tn_sim_1layer(L_i, R_i, D_i, L_j, R_j, D_j).item()
        print(f"  Initial TN-sim(T_{ci}, T_{cj}) = {initial_sim:+.4f}")

        # Optimize
        opt_result = optimize_shared_feature(model, ci, cj, n_seeds=20, steps=3000,
                                              lr=1e-2, device=device)
        print(f"  Final TN-sim(T_{ci}-S, T_{cj}) = {opt_result['final_sim']:+.6f}")

        # TN-sim of S with all classes
        s_sims = compute_S_sim_with_all_classes(model, opt_result['l_s'],
                                                 opt_result['r_s'], opt_result['d_s'],
                                                 device=device)
        print(f"  TN-sim(S, T_k) for k=0..7:")
        for k in range(8):
            marker = ""
            if shared_feat and shared_feat in FEATURES.get(k, set()):
                marker = " ← has " + shared_feat
            print(f"    C{k}({CLASS_LABELS[k]:>3s}): {s_sims[k]:+.4f}{marker}")

        # Normalized l, r vectors
        l_norm = opt_result['l_s'].squeeze() / opt_result['l_s'].norm()
        r_norm = opt_result['r_s'].squeeze() / opt_result['r_s'].norm()
        print(f"  l_s (normalized): [{', '.join(f'{v:.3f}' for v in l_norm.tolist())}]")
        print(f"  r_s (normalized): [{', '.join(f'{v:.3f}' for v in r_norm.tolist())}]")

        results[(ci, cj)] = {
            'ci': ci, 'cj': cj,
            'shared_feat': shared_feat,
            'label': label,
            'initial_sim': initial_sim,
            'final_sim': opt_result['final_sim'],
            'S_sims': s_sims,
            'l_s': opt_result['l_s'],
            'r_s': opt_result['r_s'],
            'd_s': opt_result['d_s'],
            'l_norm': l_norm,
            'r_norm': r_norm,
        }

    # Save results
    save_path = os.path.join(os.path.dirname(__file__), 'shared_feature_level2_results.pt')
    torch.save(results, save_path)
    print(f"\nResults saved to {save_path}")

    # Plot
    print("\n=== Generating Plot ===")
    n_pairs = len(PAIRS)
    fig, axes = plt.subplots(1, n_pairs, figsize=(3.5 * n_pairs, 3.5))

    for idx, (ci, cj, shared_feat) in enumerate(PAIRS):
        ax = axes[idx]
        r = results[(ci, cj)]
        s_sims = r['S_sims']

        # 1x8 heatmap (S similarity with each class)
        data = np.array(s_sims).reshape(1, 8)
        im = ax.imshow(data, cmap='RdBu_r', vmin=-1, vmax=1, aspect='auto')

        ax.set_xticks(range(8))
        ax.set_xticklabels(CLASS_LABELS, fontsize=7, rotation=45)
        ax.set_yticks([])

        title = f"C{ci}→C{cj}"
        if shared_feat:
            title += f"\n(share {shared_feat})"
        else:
            title += "\n(disjoint)"
        ax.set_title(title, fontsize=8)

        # Annotate values
        for k in range(8):
            color = 'white' if abs(s_sims[k]) > 0.5 else 'black'
            ax.text(k, 0, f"{s_sims[k]:.2f}", ha='center', va='center',
                    fontsize=7, color=color)

    fig.suptitle("TN-sim(S, T_k) — Extracted Shared Feature S for Each Pair", fontsize=11)
    plt.colorbar(im, ax=axes.tolist(), shrink=0.6, label='TN-sim')
    plt.tight_layout(rect=[0, 0, 0.92, 0.92])

    img_dir = os.path.join(os.path.dirname(__file__), 'images', 'level2')
    os.makedirs(img_dir, exist_ok=True)
    img_path = os.path.join(img_dir, 'shared_features.png')
    plt.savefig(img_path, dpi=150, bbox_inches='tight')
    print(f"Plot saved to {img_path}")

    # SUMMARY
    print("\n" + "=" * 70)
    print("SUMMARY: Does S correctly identify shared features?")
    print("=" * 70)

    for ci, cj, shared_feat in PAIRS:
        r = results[(ci, cj)]
        s_sims = r['S_sims']
        print(f"\n{r['label']}")
        print(f"  Residual sim after subtraction: {r['final_sim']:+.6f}")

        if shared_feat:
            # Find classes that share this feature
            sharing_classes = [k for k in range(8) if shared_feat in FEATURES[k]]
            non_sharing = [k for k in range(8) if shared_feat not in FEATURES[k]]

            avg_sharing = np.mean([abs(s_sims[k]) for k in sharing_classes])
            avg_non_sharing = np.mean([abs(s_sims[k]) for k in non_sharing])

            print(f"  Classes with feature {shared_feat}: {[f'C{k}({CLASS_LABELS[k]})' for k in sharing_classes]}")
            print(f"    Avg |TN-sim(S, T_k)|: {avg_sharing:.4f}")
            print(f"  Classes without feature {shared_feat}: {[f'C{k}({CLASS_LABELS[k]})' for k in non_sharing]}")
            print(f"    Avg |TN-sim(S, T_k)|: {avg_non_sharing:.4f}")

            # Check: does S have higher sim with sharing classes?
            correct = avg_sharing > avg_non_sharing
            print(f"  VERDICT: {'YES - S correctly identifies feature ' + shared_feat if correct else 'NO - S does not clearly identify the feature'}")

            # Detailed: which classes have |sim| > 0.3?
            high_sim = [k for k in range(8) if abs(s_sims[k]) > 0.3]
            print(f"  Classes with |sim(S,T_k)| > 0.3: {[f'C{k}({CLASS_LABELS[k]})' for k in high_sim]}")
        else:
            print(f"  (Disjoint pair — S captures anti-correlation structure)")
            high_sim = [k for k in range(8) if abs(s_sims[k]) > 0.3]
            print(f"  Classes with |sim(S,T_k)| > 0.3: {[f'C{k}({CLASS_LABELS[k]})' for k in high_sim]}")


if __name__ == '__main__':
    main()
