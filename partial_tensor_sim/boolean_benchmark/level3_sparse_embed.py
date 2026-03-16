"""
Level 3: Sparse embedding analysis on mixed-degree classes.

Staged embedding pruning (1-layer only): dense -> top-2 -> top-1 per hidden dim.
Key question: does the degree-4 class (C0 = P = A AND B) suffer more from
sparse embedding in 1L than degree-2 classes?

Classes:
  C0: P=1              (degree 4, 4 patterns)
  C1: A=1, B=0         (degree 2, 12 patterns)
  C2: B=1, A=0         (degree 2, 12 patterns)
  C3: C=1, A=0, B=0    (degree 2, 9 patterns)
  C4: otherwise         (default, 27 patterns)
"""
import torch
import torch.nn.functional as F
import numpy as np
import os
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from model import (
    BooleanBilinear1Layer, BooleanBilinear2Layer,
    make_level3_dataset, train_model,
    tn_inner_1layer, tn_sim_1layer, optimize_rank1,
    compute_relative_norms,
)

IMG_DIR = os.path.join(os.path.dirname(__file__), 'images', 'level3')
os.makedirs(IMG_DIR, exist_ok=True)
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

CLASS_LABELS = ['P(d4)', 'A(d2)', 'B(d2)', 'C(d2)', 'def']
CLASS_DEGREES = [4, 2, 2, 2, 0]
INPUT_LABELS = ['x0', 'x1', 'x2', 'x3', 'x4', 'x5']
N_CLASSES = 5


# =============================================================================
# HELPERS
# =============================================================================

def train_with_embed_mask(model, X, labels, embed_mask, epochs, lr, wd):
    """Train with optional embedding mask (gradient + weight zeroing)."""
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
    """Per-row pruning: keep top-k inputs per hidden dim by magnitude."""
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
    """Get bilinear-only effective weights in input space for class c."""
    with torch.no_grad():
        W_e = model.W_embed
        W_h = model.W_head
        L_eff = model.L @ W_e      # (rank, n_input)
        R_eff = model.R @ W_e      # (rank, n_input)
        D_eff = W_h[[c]] @ model.D  # (1, rank)
    return L_eff, R_eff, D_eff


def compute_cross_sim_bilinear(model, n_classes):
    """5x5 bilinear-only TN-sim matrix."""
    sim = np.zeros((n_classes, n_classes))
    for i in range(n_classes):
        Li, Ri, Di = get_bilinear_effective_weights(model, i)
        for j in range(n_classes):
            Lj, Rj, Dj = get_bilinear_effective_weights(model, j)
            sim[i, j] = tn_sim_1layer(Li, Ri, Di, Lj, Rj, Dj).item()
    return sim


def compute_rank1_all(model, n_classes, device):
    """Optimize rank-1 for bilinear-only weights of each class."""
    rank1_results = {}
    for c in range(n_classes):
        L_eff, R_eff, D_eff = get_bilinear_effective_weights(model, c)
        result = optimize_rank1(L_eff, R_eff, D_eff, n_seeds=20, steps=2000,
                                lr=0.01, device=device)
        rank1_results[c] = result
    return rank1_results


def compute_rank1_cross_proj(rank1_results, model, n_classes):
    """5x5 rank-1 cross-class projection: TN-sim(rank1_i, T_bil_j)."""
    proj = np.zeros((n_classes, n_classes))
    for i in range(n_classes):
        li, ri, di = rank1_results[i]['l'], rank1_results[i]['r'], rank1_results[i]['d']
        for j in range(n_classes):
            Lj, Rj, Dj = get_bilinear_effective_weights(model, j)
            proj[i, j] = tn_sim_1layer(li, ri, di, Lj, Rj, Dj).item()
    return proj


def compute_per_class_acc(model, X, labels, n_classes):
    """Per-class accuracy dict."""
    accs = {}
    with torch.no_grad():
        preds = model(X).argmax(dim=1)
        for c in range(n_classes):
            mask = labels == c
            if mask.sum() > 0:
                accs[c] = (preds[mask] == c).float().mean().item()
            else:
                accs[c] = 0.0
    return accs


def print_rank1_vectors(rank1_results, n_input, label=""):
    """Print rank-1 vectors with input labels showing which inputs are selected."""
    if label:
        print(f"\n  Rank-1 vectors ({label}):")
    else:
        print(f"\n  Rank-1 vectors:")
    for c in range(N_CLASSES):
        l = rank1_results[c]['l'].detach().cpu().numpy().flatten()
        r = rank1_results[c]['r'].detach().cpu().numpy().flatten()
        d = rank1_results[c]['d'].detach().cpu().numpy().flatten()
        sim = rank1_results[c]['sim']

        # Top inputs by magnitude
        l_abs = np.abs(l[:n_input])
        r_abs = np.abs(r[:n_input])
        l_top = np.argsort(l_abs)[::-1][:3]
        r_top = np.argsort(r_abs)[::-1][:3]

        l_str = ', '.join(f'{INPUT_LABELS[i]}({l[i]:+.3f})' for i in l_top if l_abs[i] > 0.01)
        r_str = ', '.join(f'{INPUT_LABELS[i]}({r[i]:+.3f})' for i in r_top if r_abs[i] > 0.01)

        print(f"    C{c}({CLASS_LABELS[c]}): sim={sim:.4f}  d={d[0]:+.4f}")
        print(f"      l selects: [{l_str}]")
        print(f"      r selects: [{r_str}]")


def compute_norms_all(model, n_classes, device):
    """Relative norms for all classes."""
    norms = {}
    for c in range(n_classes):
        norms[c] = compute_relative_norms(model, c, device)
    return norms


# =============================================================================
# ANALYSIS STAGES
# =============================================================================

def analyze_1layer_stage(model, X, labels, stage_name, device):
    """Full analysis for one pruning stage of the 1L model."""
    print(f"\n  --- {stage_name} ---")

    # Per-class accuracy
    accs = compute_per_class_acc(model, X, labels, N_CLASSES)
    overall = sum(accs[c] * (labels == c).sum().item() for c in range(N_CLASSES)) / len(labels)
    print(f"  Overall accuracy: {overall:.1%}")
    for c in range(N_CLASSES):
        n_samples = (labels == c).sum().item()
        print(f"    C{c}({CLASS_LABELS[c]}): {accs[c]:.1%} ({n_samples} samples)")

    # Embedding sparsity
    with torch.no_grad():
        W = model.embed.weight
        nnz = (W.abs() > 1e-8).sum().item()
        total = W.numel()
        print(f"  Embed sparsity: {1 - nnz/total:.1%} ({nnz}/{total} nonzero)")

    # Bilinear cross-class TN-sim
    bil_sim = compute_cross_sim_bilinear(model, N_CLASSES)

    # Rank-1 optimization
    rank1_results = compute_rank1_all(model, N_CLASSES, device)

    # Rank-1 cross-class projection
    rank1_proj = compute_rank1_cross_proj(rank1_results, model, N_CLASSES)

    # Relative norms
    norms = compute_norms_all(model, N_CLASSES, device)

    # Print rank-1 vectors
    print_rank1_vectors(rank1_results, model.n_input, label=stage_name)

    # Print relative norms
    print(f"\n  Relative norms ({stage_name}):")
    print(f"  {'Class':>5s} {'Label':>8s} {'Deg':>4s}  {'rho_res':>8s}  {'rho_bil':>8s}  {'R1 sim':>7s}")
    for c in range(N_CLASSES):
        r = norms[c]
        print(f"  {c:>5d} {CLASS_LABELS[c]:>8s} {CLASS_DEGREES[c]:>4d}  "
              f"{r['rho_res']:>8.4f}  {r['rho_bil']:>8.4f}  {rank1_results[c]['sim']:>7.4f}")

    return {
        'accs': accs,
        'overall_acc': overall,
        'bil_sim': bil_sim,
        'rank1_results': rank1_results,
        'rank1_proj': rank1_proj,
        'norms': norms,
    }


def main():
    print("=" * 70)
    print("LEVEL 3: Sparse Embedding Analysis (mixed degrees)")
    print("=" * 70)

    X, labels = make_level3_dataset(DEVICE)
    print(f"\nDataset: {len(X)} patterns, {N_CLASSES} classes")
    for c in range(N_CLASSES):
        count = (labels == c).sum().item()
        print(f"  C{c} ({CLASS_LABELS[c]}, degree {CLASS_DEGREES[c]}): {count} patterns")

    # ==================================================================
    # 1-LAYER: STAGED SPARSE EMBEDDING
    # ==================================================================
    print("\n" + "=" * 70)
    print("1-LAYER MODEL: Staged embedding pruning")
    print("=" * 70)

    # Stage 0: Dense training
    print("\n[Stage 0] Dense training...")
    model_1L = BooleanBilinear1Layer(n_input=6, d_hidden=24, d_rank=12, n_classes=5).to(DEVICE)
    acc_dense = train_model(model_1L, X, labels, epochs=5000, lr=0.01, wd=0.005,
                            print_every=2500, device=DEVICE)
    stage_dense = analyze_1layer_stage(model_1L, X, labels, "1L Dense", DEVICE)

    # Stage 1: Top-2 pruning + retrain
    print("\n[Stage 1] Prune to top-2 per hidden dim + retrain...")
    embed_mask_2 = prune_embed_topk(model_1L, k=2)
    acc_top2 = train_with_embed_mask(model_1L, X, labels, embed_mask_2,
                                     epochs=3000, lr=0.005, wd=0.005)
    stage_top2 = analyze_1layer_stage(model_1L, X, labels, "1L Top-2", DEVICE)

    # Stage 2: Top-1 pruning + retrain
    print("\n[Stage 2] Prune to top-1 per hidden dim + retrain...")
    embed_mask_1 = prune_embed_topk(model_1L, k=1)
    acc_top1 = train_with_embed_mask(model_1L, X, labels, embed_mask_1,
                                     epochs=3000, lr=0.005, wd=0.005)
    stage_top1 = analyze_1layer_stage(model_1L, X, labels, "1L Top-1", DEVICE)

    # ==================================================================
    # 2-LAYER MODEL (no pruning, just accuracy reference)
    # ==================================================================
    print("\n" + "=" * 70)
    print("2-LAYER MODEL (reference)")
    print("=" * 70)

    model_2L = BooleanBilinear2Layer(n_input=6, d_hidden=24, d_rank=12, n_classes=5).to(DEVICE)
    acc_2L = train_model(model_2L, X, labels, epochs=8000, lr=0.01, wd=0.005,
                         print_every=4000, device=DEVICE)
    accs_2L = compute_per_class_acc(model_2L, X, labels, N_CLASSES)
    print(f"  2L accuracy: {acc_2L:.1%}")
    for c in range(N_CLASSES):
        n_samples = (labels == c).sum().item()
        print(f"    C{c}({CLASS_LABELS[c]}): {accs_2L[c]:.1%} ({n_samples} samples)")

    # ==================================================================
    # COMPARISON: per-class accuracy across all configs
    # ==================================================================
    print("\n" + "=" * 70)
    print("COMPARISON: Per-class accuracy")
    print("=" * 70)
    print(f"  {'Class':>8s} {'Deg':>4s}  {'1L-Dense':>9s}  {'1L-Top2':>9s}  {'1L-Top1':>9s}  {'2L':>9s}")
    for c in range(N_CLASSES):
        print(f"  {CLASS_LABELS[c]:>8s} {CLASS_DEGREES[c]:>4d}  "
              f"{stage_dense['accs'][c]:>9.1%}  "
              f"{stage_top2['accs'][c]:>9.1%}  "
              f"{stage_top1['accs'][c]:>9.1%}  "
              f"{accs_2L[c]:>9.1%}")
    print(f"  {'Overall':>8s} {'':>4s}  "
          f"{stage_dense['overall_acc']:>9.1%}  "
          f"{stage_top2['overall_acc']:>9.1%}  "
          f"{stage_top1['overall_acc']:>9.1%}  "
          f"{acc_2L:>9.1%}")

    # Key question
    print(f"\n  Key: Does degree-4 class (C0) suffer more from sparse embedding?")
    d4_drop_top2 = stage_dense['accs'][0] - stage_top2['accs'][0]
    d4_drop_top1 = stage_dense['accs'][0] - stage_top1['accs'][0]
    d2_drops_top2 = [stage_dense['accs'][c] - stage_top2['accs'][c] for c in [1, 2, 3]]
    d2_drops_top1 = [stage_dense['accs'][c] - stage_top1['accs'][c] for c in [1, 2, 3]]
    print(f"    C0(d4) accuracy drop: top2={d4_drop_top2:+.1%}, top1={d4_drop_top1:+.1%}")
    print(f"    C1-C3(d2) avg drop:   top2={np.mean(d2_drops_top2):+.1%}, top1={np.mean(d2_drops_top1):+.1%}")

    # ==================================================================
    # PLOTS
    # ==================================================================
    print("\nGenerating plots...")

    fig, axes = plt.subplots(2, 3, figsize=(18, 11))

    stages_to_plot = [
        (0, stage_dense, '1L Dense'),
        (1, stage_top1, '1L Top-1 Sparse'),
    ]

    labels_short = [f'C{c}({CLASS_LABELS[c]})' for c in range(N_CLASSES)]

    for row, stage, row_title in stages_to_plot:
        # Col 0: Bilinear TN-sim
        ax = axes[row, 0]
        mat = stage['bil_sim']
        im = ax.imshow(mat, cmap='RdBu_r', vmin=-1, vmax=1)
        ax.set_xticks(range(N_CLASSES))
        ax.set_yticks(range(N_CLASSES))
        ax.set_xticklabels(labels_short, fontsize=7, rotation=45, ha='right')
        ax.set_yticklabels(labels_short, fontsize=7)
        ax.set_title(f'{row_title}\nBilinear TN-sim', fontsize=10)
        for i in range(N_CLASSES):
            for j in range(N_CLASSES):
                ax.text(j, i, f'{mat[i,j]:+.2f}', ha='center', va='center',
                        fontsize=8, fontweight='bold',
                        color='white' if abs(mat[i, j]) > 0.5 else 'black')
        plt.colorbar(im, ax=ax, shrink=0.8)

        # Col 1: Rank-1 cross-class projection
        ax = axes[row, 1]
        mat = stage['rank1_proj']
        im = ax.imshow(mat, cmap='RdBu_r', vmin=-1, vmax=1)
        ax.set_xticks(range(N_CLASSES))
        ax.set_yticks(range(N_CLASSES))
        ax.set_xticklabels(labels_short, fontsize=7, rotation=45, ha='right')
        ax.set_yticklabels(labels_short, fontsize=7)
        ax.set_title(f'{row_title}\nRank-1 cross-class TN-sim', fontsize=10)
        for i in range(N_CLASSES):
            for j in range(N_CLASSES):
                ax.text(j, i, f'{mat[i,j]:+.2f}', ha='center', va='center',
                        fontsize=8, fontweight='bold',
                        color='white' if abs(mat[i, j]) > 0.5 else 'black')
        plt.colorbar(im, ax=ax, shrink=0.8)

        # Col 2: Relative norms bar chart
        ax = axes[row, 2]
        x_pos = np.arange(N_CLASSES)
        rho_res = [stage['norms'][c]['rho_res'] for c in range(N_CLASSES)]
        rho_bil = [stage['norms'][c]['rho_bil'] for c in range(N_CLASSES)]
        r1_sim = [stage['rank1_results'][c]['sim'] for c in range(N_CLASSES)]
        ax.bar(x_pos - 0.2, rho_res, 0.2, label='rho_res', color='C0')
        ax.bar(x_pos, rho_bil, 0.2, label='rho_bil', color='C1')
        ax.bar(x_pos + 0.2, r1_sim, 0.2, label='R1 sim', color='C2')
        ax.set_xticks(x_pos)
        ax.set_xticklabels([f'C{c}\n{CLASS_LABELS[c]}\ndeg{CLASS_DEGREES[c]}'
                            for c in range(N_CLASSES)], fontsize=7)
        ax.set_ylabel('Value')
        ax.set_title(f'{row_title}\nRelative norms + R1 quality', fontsize=10)
        ax.legend(fontsize=7)
        ax.set_ylim(-0.2, 1.15)
        ax.grid(True, alpha=0.3)
        # Annotate bars
        for c in range(N_CLASSES):
            ax.text(c + 0.2, r1_sim[c] + 0.02, f'{r1_sim[c]:.2f}', ha='center',
                    fontsize=6, color='C2')

    plt.suptitle('Level 3: Sparse Embedding Analysis (1L Dense vs 1L Top-1)',
                 fontsize=13, y=1.01)
    plt.tight_layout()
    plt.savefig(os.path.join(IMG_DIR, 'sparse_embed.png'), dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved plot to {IMG_DIR}/sparse_embed.png")

    # ==================================================================
    # SAVE RESULTS
    # ==================================================================
    save_path = os.path.join(os.path.dirname(__file__), 'level3_sparse_embed_results.pt')
    torch.save({
        'model_1L_state': model_1L.state_dict(),
        'model_2L_state': model_2L.state_dict(),
        'stage_dense': {k: v for k, v in stage_dense.items() if k != 'rank1_results'},
        'stage_top2': {k: v for k, v in stage_top2.items() if k != 'rank1_results'},
        'stage_top1': {k: v for k, v in stage_top1.items() if k != 'rank1_results'},
        'rank1_dense': {c: {'l': stage_dense['rank1_results'][c]['l'].cpu(),
                            'r': stage_dense['rank1_results'][c]['r'].cpu(),
                            'd': stage_dense['rank1_results'][c]['d'].cpu(),
                            'sim': stage_dense['rank1_results'][c]['sim']}
                        for c in range(N_CLASSES)},
        'rank1_top2': {c: {'l': stage_top2['rank1_results'][c]['l'].cpu(),
                           'r': stage_top2['rank1_results'][c]['r'].cpu(),
                           'd': stage_top2['rank1_results'][c]['d'].cpu(),
                           'sim': stage_top2['rank1_results'][c]['sim']}
                       for c in range(N_CLASSES)},
        'rank1_top1': {c: {'l': stage_top1['rank1_results'][c]['l'].cpu(),
                           'r': stage_top1['rank1_results'][c]['r'].cpu(),
                           'd': stage_top1['rank1_results'][c]['d'].cpu(),
                           'sim': stage_top1['rank1_results'][c]['sim']}
                       for c in range(N_CLASSES)},
        'accs_2L': accs_2L,
        'acc_2L': acc_2L,
    }, save_path)
    print(f"  Saved results to {save_path}")

    # ==================================================================
    # SUMMARY
    # ==================================================================
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    print(f"  1L Dense:  {stage_dense['overall_acc']:.1%}")
    print(f"  1L Top-2:  {stage_top2['overall_acc']:.1%}")
    print(f"  1L Top-1:  {stage_top1['overall_acc']:.1%}")
    print(f"  2L:        {acc_2L:.1%}")
    print(f"\n  Degree-4 class (C0=P) accuracy:")
    print(f"    Dense={stage_dense['accs'][0]:.0%}  Top2={stage_top2['accs'][0]:.0%}  "
          f"Top1={stage_top1['accs'][0]:.0%}  2L={accs_2L[0]:.0%}")


if __name__ == '__main__':
    main()
