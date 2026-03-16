"""
Level 1: Compare learned embedding vs frozen identity embedding.
Shows full 4x4 TN-sim matrices for both.
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
    optimize_rank1,
)

IMG_DIR = os.path.join(os.path.dirname(__file__), 'images', 'level1')
os.makedirs(IMG_DIR, exist_ok=True)
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

CLASS_LABELS = ['∅', 'A', 'B', 'AB']


def freeze_embed_to_identity(model):
    with torch.no_grad():
        model.embed.weight.copy_(torch.eye(model.n_input, device=model.embed.weight.device))
    model.embed.weight.requires_grad_(False)


def run_analysis(model, X, labels, tag):
    """Run full analysis pipeline, return dict of results."""
    print(f"\n{'='*50}")
    print(f"  {tag}")
    print(f"{'='*50}")

    acc = train_model(model, X, labels, epochs=5000, lr=0.01, wd=0.005,
                      print_every=2500, device=DEVICE)
    print(f"  Accuracy: {acc:.1%}")

    # Relative norms
    class_norms = {}
    for c in range(4):
        r = compute_relative_norms(model, c, DEVICE)
        class_norms[c] = r

    # Rank-1
    rank1_results = {}
    for c in range(4):
        L_T, R_T, D_T = model.get_augmented_weights_for_class(c)
        result = optimize_rank1(L_T, R_T, D_T, n_seeds=20, steps=2000,
                                lr=0.01, device=DEVICE)
        rank1_results[c] = result

    # Full TN-sim
    full_sim = np.zeros((4, 4))
    for i in range(4):
        Li, Ri, Di = model.get_augmented_weights_for_class(i)
        for j in range(4):
            Lj, Rj, Dj = model.get_augmented_weights_for_class(j)
            full_sim[i, j] = tn_sim_1layer(Li, Ri, Di, Lj, Rj, Dj).item()

    # Bilinear-only TN-sim
    bil_sim = np.zeros((4, 4))
    for i in range(4):
        Li, Ri, Di = model.get_bilinear_only_weights_for_class(i)
        for j in range(4):
            Lj, Rj, Dj = model.get_bilinear_only_weights_for_class(j)
            bil_sim[i, j] = tn_sim_1layer(Li, Ri, Di, Lj, Rj, Dj).item()

    # Rank-1 cross-class: TN-sim(rank1_i, T_j)
    proj_sim = np.zeros((4, 4))
    for i in range(4):
        li, ri, di = rank1_results[i]['l'], rank1_results[i]['r'], rank1_results[i]['d']
        for j in range(4):
            Lj, Rj, Dj = model.get_augmented_weights_for_class(j)
            proj_sim[i, j] = tn_sim_1layer(li, ri, di, Lj, Rj, Dj).item()

    # Print
    labels_short = [f'C{c}({CLASS_LABELS[c]})' for c in range(4)]
    def print_mat(name, mat):
        print(f"\n  {name}:")
        print(f"  {'':>12s}", '  '.join(f'{l:>9s}' for l in labels_short))
        for i in range(4):
            print(f"  {labels_short[i]:>12s}", '  '.join(f'{mat[i,j]:+9.3f}' for j in range(4)))

    print_mat("Full TN-sim", full_sim)
    print_mat("Bilinear-only TN-sim", bil_sim)
    print_mat("Rank-1 projection", proj_sim)

    print(f"\n  Relative norms & rank-1 sim:")
    for c in range(4):
        r = class_norms[c]
        print(f"    C{c}({CLASS_LABELS[c]}): ρ_res={r['rho_res']:.3f} ρ_bil={r['rho_bil']:.3f} "
              f"R1_sim={rank1_results[c]['sim']:.4f}")

    # Rank-1 vectors
    print(f"\n  Rank-1 vectors (augmented space):")
    n_input = model.n_input
    for c in range(4):
        l = rank1_results[c]['l'].detach().cpu().numpy().flatten()
        r = rank1_results[c]['r'].detach().cpu().numpy().flatten()
        # Show top 2 input dims for l and r (excluding constant)
        l_abs = np.abs(l[:n_input])
        r_abs = np.abs(r[:n_input])
        l_top = np.argsort(l_abs)[::-1][:2]
        r_top = np.argsort(r_abs)[::-1][:2]
        print(f"    C{c}({CLASS_LABELS[c]}): l→x{l_top[0]}({l[l_top[0]]:+.3f}),x{l_top[1]}({l[l_top[1]]:+.3f})  "
              f"r→x{r_top[0]}({r[r_top[0]]:+.3f}),x{r_top[1]}({r[r_top[1]]:+.3f})  "
              f"const: l={l[-1]:+.3f} r={r[-1]:+.3f}")

    return {
        'model': model,
        'acc': acc,
        'class_norms': class_norms,
        'rank1_results': rank1_results,
        'full_sim': full_sim,
        'bil_sim': bil_sim,
        'proj_sim': proj_sim,
    }


def plot_comparison(res_learned, res_frozen):
    """Side-by-side comparison plot."""
    labels_short = [f'C{c}({CLASS_LABELS[c]})' for c in range(4)]

    fig, axes = plt.subplots(2, 4, figsize=(22, 10))

    configs = [
        (0, res_learned, 'Learned embed (d_h=16)'),
        (1, res_frozen, 'Frozen identity (d_h=4)'),
    ]

    for row, res, row_title in configs:
        for col, (mat, title) in enumerate([
            (res['full_sim'], 'Full TN-sim'),
            (res['bil_sim'], 'Bilinear-only TN-sim'),
            (res['proj_sim'], 'Rank-1 → T_j'),
            (None, 'Relative norms'),
        ]):
            ax = axes[row, col]

            if mat is not None:
                im = ax.imshow(mat, cmap='RdBu_r', vmin=-1, vmax=1)
                ax.set_xticks(range(4)); ax.set_yticks(range(4))
                ax.set_xticklabels(labels_short, fontsize=8, rotation=45, ha='right')
                ax.set_yticklabels(labels_short, fontsize=8)
                for i in range(4):
                    for j in range(4):
                        ax.text(j, i, f'{mat[i,j]:+.2f}', ha='center', va='center',
                                fontsize=9, fontweight='bold',
                                color='white' if abs(mat[i,j]) > 0.5 else 'black')
                plt.colorbar(im, ax=ax, shrink=0.8)
            else:
                # Bar chart for relative norms
                x_pos = np.arange(4)
                rho_res = [res['class_norms'][c]['rho_res'] for c in range(4)]
                rho_bil = [res['class_norms'][c]['rho_bil'] for c in range(4)]
                r1_sim = [res['rank1_results'][c]['sim'] for c in range(4)]
                ax.bar(x_pos - 0.2, rho_res, 0.2, label='ρ_res', color='C0')
                ax.bar(x_pos, rho_bil, 0.2, label='ρ_bil', color='C1')
                ax.bar(x_pos + 0.2, r1_sim, 0.2, label='R1 sim', color='C2')
                ax.set_xticks(x_pos)
                ax.set_xticklabels(labels_short, fontsize=8)
                ax.legend(fontsize=7)
                ax.set_ylim(0, 1.1)
                ax.grid(True, alpha=0.3)

            ax.set_title(f'{row_title}\n{title}', fontsize=10)

    plt.tight_layout()
    plt.savefig(os.path.join(IMG_DIR, 'embed_comparison.png'), dpi=150, bbox_inches='tight')
    plt.close()
    print(f"\nSaved to {IMG_DIR}/embed_comparison.png")


def main():
    X, labels = make_level1_dataset(DEVICE)
    print(f"Dataset: {len(X)} patterns")
    for c in range(4):
        print(f"  C{c}({CLASS_LABELS[c]}): {(labels==c).sum().item()} patterns")

    # Run 1: Learned embedding, d_hidden=16
    model_learned = BooleanBilinear1Layer(n_input=4, d_hidden=16, d_rank=8, n_classes=4).to(DEVICE)
    res_learned = run_analysis(model_learned, X, labels, "LEARNED EMBED (d_hidden=16)")

    # Run 2: Frozen identity embedding, d_hidden=4
    model_frozen = BooleanBilinear1Layer(n_input=4, d_hidden=4, d_rank=8, n_classes=4).to(DEVICE)
    freeze_embed_to_identity(model_frozen)
    res_frozen = run_analysis(model_frozen, X, labels, "FROZEN IDENTITY (d_hidden=4)")

    # Comparison
    print(f"\n{'='*70}")
    print("COMPARISON")
    print(f"{'='*70}")
    print(f"  {'':>20s}  {'Learned':>10s}  {'Frozen':>10s}")
    print(f"  {'Accuracy':>20s}  {res_learned['acc']:>10.1%}  {res_frozen['acc']:>10.1%}")
    for c in range(4):
        rl = res_learned['class_norms'][c]
        rf = res_frozen['class_norms'][c]
        print(f"  C{c} ρ_res / ρ_bil     {rl['rho_res']:>5.3f}/{rl['rho_bil']:>.3f}  "
              f"{rf['rho_res']:>5.3f}/{rf['rho_bil']:>.3f}")
    for c in range(4):
        print(f"  C{c} R1 sim            {res_learned['rank1_results'][c]['sim']:>10.4f}  "
              f"{res_frozen['rank1_results'][c]['sim']:>10.4f}")

    plot_comparison(res_learned, res_frozen)


if __name__ == '__main__':
    main()
