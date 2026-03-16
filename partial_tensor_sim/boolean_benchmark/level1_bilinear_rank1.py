"""
Level 1: Rank-1 optimization on bilinear-only part (embed + head folded in).

The residual W_head[c] @ W_embed @ x is understood — we skip it.
The bilinear part D_eff_c @ ((L_eff @ x) ⊙ (R_eff @ x)) is what we decompose.
"""
import torch
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


def get_bilinear_effective_weights(model, c):
    """Get bilinear-only effective weights in input space for class c.
    L_eff = L @ W_embed, R_eff = R @ W_embed, D_eff = W_head[c] @ D.
    No augmented space, no residual.
    """
    with torch.no_grad():
        W_e = model.W_embed
        W_h = model.W_head
        L_eff = model.L @ W_e      # (rank, n_input)
        R_eff = model.R @ W_e      # (rank, n_input)
        D_eff = W_h[[c]] @ model.D  # (1, rank)
    return L_eff, R_eff, D_eff


def main():
    X, labels = make_level1_dataset(DEVICE)
    print(f"Dataset: {len(X)} patterns, 4 classes")

    # Train with learned embedding (d_hidden=16)
    model = BooleanBilinear1Layer(n_input=4, d_hidden=16, d_rank=8, n_classes=4).to(DEVICE)
    acc = train_model(model, X, labels, epochs=5000, lr=0.01, wd=0.005,
                      print_every=2500, device=DEVICE)
    print(f"Accuracy: {acc:.1%}\n")

    # ================================================================
    # RANK-1 ON BILINEAR-ONLY EFFECTIVE WEIGHTS
    # ================================================================
    print("Rank-1 on bilinear-only (embed+head folded in):")
    rank1_results = {}
    for c in range(4):
        L_eff, R_eff, D_eff = get_bilinear_effective_weights(model, c)
        result = optimize_rank1(L_eff, R_eff, D_eff, n_seeds=20, steps=2000,
                                lr=0.01, device=DEVICE)
        rank1_results[c] = result
        print(f"  C{c}({CLASS_LABELS[c]}): TN-sim = {result['sim']:.4f}")

    # ================================================================
    # RANK-1 VECTORS — directly in input space
    # ================================================================
    print("\nRank-1 vectors (input space [x0, x1, x2, x3]):")
    for c in range(4):
        l = rank1_results[c]['l'].detach().cpu().numpy().flatten()
        r = rank1_results[c]['r'].detach().cpu().numpy().flatten()
        d = rank1_results[c]['d'].detach().cpu().numpy().flatten()
        print(f"  C{c}({CLASS_LABELS[c]}):")
        print(f"    l = [{', '.join(f'{v:+.3f}' for v in l)}]")
        print(f"    r = [{', '.join(f'{v:+.3f}' for v in r)}]")
        print(f"    d = {d[0]:+.4f}")

    # ================================================================
    # CROSS-CLASS TN-SIM (bilinear-only tensors)
    # ================================================================
    print("\nBilinear-only TN-sim (T_bil_i vs T_bil_j):")
    bil_sim = np.zeros((4, 4))
    for i in range(4):
        Li, Ri, Di = get_bilinear_effective_weights(model, i)
        for j in range(4):
            Lj, Rj, Dj = get_bilinear_effective_weights(model, j)
            bil_sim[i, j] = tn_sim_1layer(Li, Ri, Di, Lj, Rj, Dj).item()

    labels_short = [f'C{c}({CLASS_LABELS[c]})' for c in range(4)]
    print(f"  {'':>12s}", '  '.join(f'{l:>9s}' for l in labels_short))
    for i in range(4):
        print(f"  {labels_short[i]:>12s}", '  '.join(f'{bil_sim[i,j]:+9.3f}' for j in range(4)))

    # ================================================================
    # RANK-1 CROSS-CLASS: TN-sim(rank1_i, T_bil_j)
    # ================================================================
    print("\nRank-1 cross-class TN-sim (rank1_i → T_bil_j):")
    proj_sim = np.zeros((4, 4))
    for i in range(4):
        li, ri, di = rank1_results[i]['l'], rank1_results[i]['r'], rank1_results[i]['d']
        for j in range(4):
            Lj, Rj, Dj = get_bilinear_effective_weights(model, j)
            proj_sim[i, j] = tn_sim_1layer(li, ri, di, Lj, Rj, Dj).item()

    print(f"  {'':>12s}", '  '.join(f'{l:>9s}' for l in labels_short))
    for i in range(4):
        print(f"  {labels_short[i]:>12s}", '  '.join(f'{proj_sim[i,j]:+9.3f}' for j in range(4)))

    # ================================================================
    # GROUND TRUTH
    # ================================================================
    features = {0: set(), 1: {'A'}, 2: {'B'}, 3: {'A', 'B'}}
    gt_sharing = np.zeros((4, 4))
    for i in range(4):
        for j in range(4):
            shared = len(features[i] & features[j])
            total = max(len(features[i] | features[j]), 1)
            gt_sharing[i, j] = shared / total

    # ================================================================
    # PLOT
    # ================================================================
    fig, axes = plt.subplots(2, 3, figsize=(18, 11))

    # Row 1: Similarity matrices
    for ax, mat, title in [
        (axes[0, 0], bil_sim, 'Bilinear TN-sim\n(T_bil_i vs T_bil_j)'),
        (axes[0, 1], proj_sim, 'Rank-1 cross-class\nTN-sim(rank1_i, T_bil_j)'),
        (axes[0, 2], gt_sharing, 'Ground truth\nJaccard similarity'),
    ]:
        vmin, vmax = (-1, 1) if 'Ground' not in title else (0, 1)
        cmap = 'RdBu_r' if 'Ground' not in title else 'Blues'
        im = ax.imshow(mat, cmap=cmap, vmin=vmin, vmax=vmax)
        ax.set_xticks(range(4)); ax.set_yticks(range(4))
        ax.set_xticklabels(labels_short, fontsize=9, rotation=45, ha='right')
        ax.set_yticklabels(labels_short, fontsize=9)
        ax.set_title(title, fontsize=11)
        for i in range(4):
            for j in range(4):
                ax.text(j, i, f'{mat[i,j]:+.2f}', ha='center', va='center',
                        fontsize=10, fontweight='bold',
                        color='white' if abs(mat[i,j]) > 0.5 else 'black')
        plt.colorbar(im, ax=ax, shrink=0.8)

    # Row 2 left: Rank-1 TN-sim bar chart
    ax = axes[1, 0]
    sims = [rank1_results[c]['sim'] for c in range(4)]
    colors = ['C3' if c == 3 else 'C0' for c in range(4)]
    ax.bar(range(4), sims, color=colors)
    ax.set_xticks(range(4))
    ax.set_xticklabels(labels_short, fontsize=9)
    ax.set_ylabel('Rank-1 TN-sim')
    ax.set_title('Rank-1 approximation quality\n(red = multi-feature)', fontsize=11)
    ax.set_ylim(0.5, 1.05)
    ax.grid(True, alpha=0.3)
    for c in range(4):
        ax.text(c, sims[c] + 0.01, f'{sims[c]:.3f}', ha='center', fontsize=9)

    # Row 2 middle: Rank-1 l,r vectors heatmap
    ax = axes[1, 1]
    n = model.n_input
    vec_data = np.zeros((4, 2 * n))
    for c in range(4):
        l = rank1_results[c]['l'].detach().cpu().numpy().flatten()
        r = rank1_results[c]['r'].detach().cpu().numpy().flatten()
        vec_data[c, :n] = l
        vec_data[c, n:] = r
    vmax = np.abs(vec_data).max()
    im = ax.imshow(vec_data, cmap='RdBu_r', vmin=-vmax, vmax=vmax)
    ax.set_yticks(range(4))
    ax.set_yticklabels(labels_short, fontsize=9)
    xlabels = [f'l:{INPUT_LABELS[i]}' for i in range(n)] + [f'r:{INPUT_LABELS[i]}' for i in range(n)]
    ax.set_xticks(range(2 * n))
    ax.set_xticklabels(xlabels, fontsize=8, rotation=45, ha='right')
    ax.set_title('Rank-1 vectors\n(input space)', fontsize=11)
    for i in range(4):
        for j in range(2 * n):
            ax.text(j, i, f'{vec_data[i,j]:+.2f}', ha='center', va='center',
                    fontsize=8, color='white' if abs(vec_data[i,j]) > vmax * 0.4 else 'black')
    plt.colorbar(im, ax=ax, shrink=0.8)

    # Row 2 right: Verify by computing bilinear output on all inputs
    ax = axes[1, 2]
    with torch.no_grad():
        X_np = X.cpu().numpy()
        outputs = np.zeros((16, 4))  # 16 inputs, 4 classes
        for c in range(4):
            l = rank1_results[c]['l']
            r = rank1_results[c]['r']
            d = rank1_results[c]['d']
            # rank-1 output: d * (l @ x) * (r @ x)
            lx = (X.float() @ l.T).squeeze()  # (16,)
            rx = (X.float() @ r.T).squeeze()
            outputs[:, c] = (d.item() * lx * rx).cpu().numpy()

        # Show output for each class's rank-1 on each input pattern
        # Sort by label
        order = labels.cpu().numpy().argsort()
        sorted_labels = labels.cpu().numpy()[order]
        sorted_outputs = outputs[order]

        im = ax.imshow(sorted_outputs, cmap='RdBu_r', aspect='auto',
                       vmin=-np.abs(sorted_outputs).max(), vmax=np.abs(sorted_outputs).max())
        ax.set_xlabel('Class rank-1')
        ax.set_ylabel('Input pattern (sorted by true class)')
        ax.set_xticks(range(4))
        ax.set_xticklabels(labels_short, fontsize=9)
        # Mark class boundaries
        for boundary in [9, 12, 15]:
            ax.axhline(boundary - 0.5, color='k', linewidth=1)
        ax.set_title('Rank-1 outputs on all inputs\n(sorted by true class)', fontsize=11)
        plt.colorbar(im, ax=ax, shrink=0.8)

    plt.suptitle('Level 1: Bilinear-only rank-1 (embed+head folded in)', fontsize=13, y=1.01)
    plt.tight_layout()
    plt.savefig(os.path.join(IMG_DIR, 'bilinear_rank1.png'), dpi=150, bbox_inches='tight')
    plt.close()
    print(f"\nSaved to {IMG_DIR}/bilinear_rank1.png")

    # Save
    torch.save({
        'model_state': model.state_dict(),
        'rank1_results': rank1_results,
        'bil_sim': bil_sim,
        'proj_sim': proj_sim,
        'accuracy': acc,
    }, os.path.join(os.path.dirname(__file__), 'level1_bilinear_rank1_results.pt'))


if __name__ == '__main__':
    main()
