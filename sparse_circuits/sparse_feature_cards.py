# Feature cards + circuit visualization for sparse pruned ODT model
#
# Adapted from odt/feature_cards.py for sparse (90% pruned) models.
# Key difference: no eigenvector basis change — the rotation defines the feature
# basis directly. Sparsity makes the circuit graph much cleaner.

import sys, os

PROJECT_ROOT = "/workspace/tn_4_interp"
ODT_DIR = os.path.join(PROJECT_ROOT, "odt")
sys.path.insert(0, ODT_DIR)
os.chdir(ODT_DIR)

import torch
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from dataclasses import dataclass, field
from typing import Optional
from collections import Counter

from analyze import load_model, evaluate_truncated
from train import get_svhn_loaders

DEVICE = "cuda"
RESULTS_DIR = os.path.join(PROJECT_ROOT, "sparse_circuits", "results")
SAVE_DIR = os.path.join(RESULTS_DIR, "feature_cards")
os.makedirs(SAVE_DIR, exist_ok=True)

DATA_ROOT = os.path.join(PROJECT_ROOT, "modular_addition/dev_interp/notebooks/data")
MAX_SAMPLES = 5000
N_EXAMPLES = 10
IMG_SHAPE = (32, 32)
ACTIVATION_THRESHOLD = 0.4  # threshold for counting pos/neg activating samples


# ================================================================
# 1. Core computation functions
# ================================================================

def forward_odt(x, cores, embed, unembed):
    """Standard ODT bilinear forward."""
    ones = torch.ones(x.shape[0], 1, device=x.device)
    h = torch.cat([ones, x], dim=-1)
    h = h @ embed.T
    for core in cores:
        h = torch.einsum('oij,bi,bj->bo', core, h, h)
    return h @ unembed.T


@torch.no_grad()
def compute_sparse_activations(cores, embed, unembed, data_loader, device,
                                max_samples=5000):
    """Forward pass through sparse network, recording h at every bond.

    Returns:
        activations: list of (n_samples, r) arrays, one per bond
        labels: (n_samples,) int array
        images: (n_samples, 1024) array
    """
    embed_d = embed.to(device)
    cores_d = [c.to(device) for c in cores]

    n_bonds = len(cores) + 1
    all_acts = [[] for _ in range(n_bonds)]
    all_labels = []
    all_images = []
    total = 0

    for x, y in data_loader:
        x, y = x.to(device), y.to(device)
        ones = torch.ones(x.shape[0], 1, device=device)
        h = torch.cat([ones, x], dim=-1)
        h = h @ embed_d.T
        all_acts[0].append(h.cpu())

        for i, core in enumerate(cores_d):
            h = torch.einsum('oij,bi,bj->bo', core, h, h)
            all_acts[i + 1].append(h.cpu())

        all_labels.append(y.cpu())
        all_images.append(x.cpu())
        total += x.shape[0]
        if total >= max_samples:
            break

    activations = [torch.cat(a, dim=0).numpy()[:max_samples] for a in all_acts]
    labels = torch.cat(all_labels).numpy()[:max_samples]
    images = torch.cat(all_images).numpy()[:max_samples]
    return activations, labels, images


def compute_class_effects(unembed):
    """How each feature at the last bond affects each class logit.

    Returns: (n_classes, r_last) numpy array
    """
    return unembed.detach().numpy()


def build_sparse_circuit(cores, threshold_pctile=75):
    """Build circuit graph from sparse cores.

    Only includes edges above the threshold_pctile of nonzero weights
    per layer. Default 75th percentile keeps the top 25% of surviving
    weights — enough to see the important structure without clutter.

    Returns:
        nodes: dict {bond_idx: list of feature indices}
        edges: list of dicts with layer, src_i, src_j, dst_k, weight
    """
    edges = []
    nodes = {}

    for ell, core in enumerate(cores):
        T = 0.5 * (core + core.transpose(-1, -2))
        r_out, r_in, _ = T.shape

        # Compute threshold as percentile of nonzero values in this layer
        all_abs = T.abs().flatten()
        nonzero_abs = all_abs[all_abs > 1e-10]
        if len(nonzero_abs) == 0:
            threshold = 0
        else:
            threshold = torch.quantile(nonzero_abs, threshold_pctile / 100.0).item()

        if ell == 0:
            nodes[ell] = list(range(r_in))
        nodes[ell + 1] = list(range(r_out))

        for k in range(r_out):
            for i in range(r_in):
                for j in range(i, r_in):
                    w = T[k, i, j].item()
                    if i != j:
                        w += T[k, j, i].item()
                    if abs(w) > threshold:
                        edges.append({
                            'layer': ell,
                            'src_i': i, 'src_j': j,
                            'dst_k': k,
                            'weight': w,
                        })
    return nodes, edges


def trace_feature_to_input(cores, embed, bond, feature_idx, img_shape=(32, 32)):
    """Trace a feature backward to pixel space via dominant eigenvector.

    Starting from unit vector e_{feature_idx} at the given bond,
    trace backward through cores using dominant eigenvector at each layer.
    """
    h_dim = cores[0].shape[1] if bond > 0 else embed.shape[0]
    w = torch.zeros(cores[bond - 1].shape[0] if bond > 0 else embed.shape[0])
    w[feature_idx] = 1.0

    for layer_idx in range(bond - 1, -1, -1):
        core = cores[layer_idx]
        M = torch.einsum('o,oij->ij', w, core)
        M = 0.5 * (M + M.T)
        try:
            evals, evecs = torch.linalg.eigh(M)
            idx = evals.abs().argsort(descending=True)[0]
            w = evecs[:, idx]
        except torch._C._LinAlgError:
            # Degenerate matrix — use SVD fallback
            U, S, _ = torch.linalg.svd(M)
            w = U[:, 0]

    # Project through embedding to pixel space
    input_pattern = embed.T @ w  # (input_dim+1,)
    input_pattern = input_pattern[1:]  # skip constant
    n_pixels = img_shape[0] * img_shape[1]
    return input_pattern[:n_pixels].reshape(img_shape).detach().numpy()


def decompose_sparse_interaction(core, feature_idx):
    """Eigendecompose the interaction matrix T[feature_idx, :, :].

    Returns:
        eigenvalues, eigenvectors, interaction_matrix (all numpy)
    """
    M = core[feature_idx]
    M = 0.5 * (M + M.T)
    try:
        evals, evecs = torch.linalg.eigh(M)
        idx = evals.abs().argsort(descending=True)
        return evals[idx].numpy(), evecs[:, idx].numpy(), M.numpy()
    except torch._C._LinAlgError:
        # Degenerate matrix — use SVD
        U, S, Vh = torch.linalg.svd(M)
        return S.numpy(), U.numpy(), M.numpy()


# ================================================================
# 2. Feature card assembly
# ================================================================

@dataclass
class SparseFeatureCard:
    """Information about one feature in the sparse network."""
    bond: int
    feature_idx: int
    act_variance: float  # how selective: high = fires strongly for some, weakly for others
    act_mean: float

    # Top activating examples (positive and negative)
    top_pos_indices: np.ndarray = field(default_factory=lambda: np.array([]))
    top_pos_activations: np.ndarray = field(default_factory=lambda: np.array([]))
    top_pos_labels: np.ndarray = field(default_factory=lambda: np.array([]))
    top_neg_indices: np.ndarray = field(default_factory=lambda: np.array([]))
    top_neg_activations: np.ndarray = field(default_factory=lambda: np.array([]))
    top_neg_labels: np.ndarray = field(default_factory=lambda: np.array([]))

    # Activation frequency
    all_activations: Optional[np.ndarray] = None  # full activation vector over dataset
    frac_pos: float = 0.0          # % of dataset above +threshold
    frac_neg: float = 0.0          # % of dataset below -threshold
    frac_inactive: float = 0.0     # % within [-threshold, +threshold]

    # Low-activation counterexamples (images closest to zero activation)
    low_act_indices: np.ndarray = field(default_factory=lambda: np.array([], dtype=int))
    low_act_activations: np.ndarray = field(default_factory=lambda: np.array([]))
    low_act_labels: np.ndarray = field(default_factory=lambda: np.array([], dtype=int))

    # Class breakdown
    pos_class_counts: Optional[np.ndarray] = None   # (n_classes,) count above threshold
    neg_class_counts: Optional[np.ndarray] = None   # (n_classes,) count below -threshold
    class_act_pct: Optional[np.ndarray] = None       # (n_classes, 2) % of each class pos/neg
    class_selectivity: Optional[np.ndarray] = None   # (n_classes,) mean activation per class

    # Input attribution pattern
    input_pattern: Optional[np.ndarray] = None

    # Computation (for bonds > 0)
    mode_eigenvalues: Optional[np.ndarray] = None
    mode_eigenvectors: Optional[np.ndarray] = None
    interaction_matrix: Optional[np.ndarray] = None
    n_nonzero_interactions: int = 0

    # Output effect (last bond)
    class_effects: Optional[np.ndarray] = None
    class_label: Optional[int] = None  # set when unembed is identity (folded)

    # Circuit
    incoming_edges: list = field(default_factory=list)
    outgoing_edges: list = field(default_factory=list)


def generate_sparse_feature_card(bond, feature_idx, activations, labels, images,
                                  cores, embed, unembed, edges, n_examples=10,
                                  threshold=ACTIVATION_THRESHOLD):
    """Assemble a feature card for one sparse feature."""
    acts = activations[bond][:, feature_idx]
    n_total = len(acts)
    n_classes = int(labels.max()) + 1
    card = SparseFeatureCard(
        bond=bond,
        feature_idx=feature_idx,
        act_variance=float(np.var(acts)),
        act_mean=float(np.mean(acts)),
    )

    # ── Top activating examples ──
    pos_idx = np.argsort(acts)[::-1][:n_examples]
    neg_idx = np.argsort(acts)[:n_examples]
    card.top_pos_indices = pos_idx
    card.top_pos_activations = acts[pos_idx]
    card.top_pos_labels = labels[pos_idx]
    card.top_neg_indices = neg_idx
    card.top_neg_activations = acts[neg_idx]
    card.top_neg_labels = labels[neg_idx]

    # ── Activation frequency stats ──
    card.all_activations = acts
    above = acts > threshold
    below = acts < -threshold
    card.frac_pos = above.sum() / n_total
    card.frac_neg = below.sum() / n_total
    card.frac_inactive = 1.0 - card.frac_pos - card.frac_neg

    # Low-activation counterexamples (closest to zero)
    abs_acts = np.abs(acts)
    low_idx = np.argsort(abs_acts)[:5]
    card.low_act_indices = low_idx
    card.low_act_activations = acts[low_idx]
    card.low_act_labels = labels[low_idx]

    # ── Class breakdown for pos/neg activators ──
    card.pos_class_counts = np.zeros(n_classes)
    card.neg_class_counts = np.zeros(n_classes)
    card.class_act_pct = np.zeros((n_classes, 2))  # col 0=pos%, col 1=neg%
    for c in range(n_classes):
        mask_c = labels == c
        n_c = mask_c.sum()
        card.pos_class_counts[c] = (above & mask_c).sum()
        card.neg_class_counts[c] = (below & mask_c).sum()
        if n_c > 0:
            card.class_act_pct[c, 0] = card.pos_class_counts[c] / n_c * 100
            card.class_act_pct[c, 1] = card.neg_class_counts[c] / n_c * 100

    # ── Class selectivity: mean activation per class ──
    class_sel = np.zeros(n_classes)
    for c in range(n_classes):
        mask = labels == c
        if mask.any():
            class_sel[c] = acts[mask].mean()
    card.class_selectivity = class_sel

    # ── Input attribution ──
    card.input_pattern = trace_feature_to_input(cores, embed, bond, feature_idx)

    # ── Computation modes (bonds > 0) ──
    if bond > 0:
        core = cores[bond - 1]
        evals, evecs, M = decompose_sparse_interaction(core, feature_idx)
        card.mode_eigenvalues = evals
        card.mode_eigenvectors = evecs
        card.interaction_matrix = M
        card.n_nonzero_interactions = int((np.abs(M) > 1e-10).sum())

    # ── Class effects (last bond) ──
    n_bonds = len(cores) + 1
    if bond == n_bonds - 1:
        effects = unembed[:, feature_idx].detach().numpy()
        card.class_effects = effects
        # Check if unembed is identity (folded case): feature IS the class
        is_identity = (torch.eye(unembed.shape[0]) - unembed).abs().max().item() < 1e-6
        if is_identity:
            card.class_label = feature_idx  # this feature = this class

    # ── Circuit edges ──
    card.incoming_edges = [e for e in edges
                           if e['dst_k'] == feature_idx and e['layer'] == bond - 1]
    card.outgoing_edges = [e for e in edges
                           if (e['src_i'] == feature_idx or e['src_j'] == feature_idx)
                           and e['layer'] == bond]

    return card


# ================================================================
# 3. Visualization functions
# ================================================================

def plot_sparse_feature_card(card, images, save_path, labels=None):
    """Plot a single sparse feature card.

    Layout (3 rows):
      Row 1: Top positive examples | Input pattern + low-act | Top negative examples
      Row 2: Activation histogram | Class breakdown | Top-3 class histograms
      Row 3: Computation modes | Sparse interaction matrix | Class effects + act % heatmap
    """
    from matplotlib.colors import LinearSegmentedColormap
    n_pos = len(card.top_pos_indices)
    n_neg = len(card.top_neg_indices)
    has_modes = card.mode_eigenvalues is not None

    fig = plt.figure(figsize=(16, 13))
    gs = gridspec.GridSpec(3, 3, height_ratios=[1, 1, 1],
                           width_ratios=[2, 1, 2], hspace=0.4, wspace=0.3)

    # ==== Row 1: Top examples + input pattern ====

    n_cols = min(5, n_pos)
    if n_cols > 0:
        gs_pos = gs[0, 0].subgridspec(2, n_cols, hspace=0.1, wspace=0.05)
        for idx in range(min(10, n_pos)):
            row, col = idx // n_cols, idx % n_cols
            if row >= 2 or col >= n_cols:
                continue
            ax = fig.add_subplot(gs_pos[row, col])
            ax.imshow(images[card.top_pos_indices[idx]].reshape(IMG_SHAPE),
                     cmap='gray', vmin=0, vmax=1)
            ax.set_title(f'{card.top_pos_activations[idx]:.1f}\n(y={card.top_pos_labels[idx]})',
                         fontsize=6)
            ax.axis('off')

    # Center: input pattern + low-activation counterexamples
    gs_center0 = gs[0, 1].subgridspec(2, 1, height_ratios=[3, 1], hspace=0.3)
    ax_inp = fig.add_subplot(gs_center0[0])
    if card.input_pattern is not None:
        vmax = np.abs(card.input_pattern).max()
        ax_inp.imshow(card.input_pattern, cmap='RdBu_r',
                     vmin=-vmax if vmax > 0 else -1, vmax=vmax if vmax > 0 else 1)
    ax_inp.set_title('Input pattern', fontsize=9)
    ax_inp.axis('off')

    n_low = len(card.low_act_indices)
    if n_low > 0:
        gs_low = gs_center0[1].subgridspec(1, min(5, n_low), wspace=0.05)
        for idx in range(min(5, n_low)):
            ax = fig.add_subplot(gs_low[0, idx])
            ax.imshow(images[card.low_act_indices[idx]].reshape(IMG_SHAPE),
                     cmap='gray', vmin=0, vmax=1)
            ax.set_title(f'{card.low_act_activations[idx]:.2f}', fontsize=5)
            ax.axis('off')

    n_cols_neg = min(5, n_neg)
    if n_cols_neg > 0:
        gs_neg = gs[0, 2].subgridspec(2, n_cols_neg, hspace=0.1, wspace=0.05)
        for idx in range(min(10, n_neg)):
            row, col = idx // n_cols_neg, idx % n_cols_neg
            if row >= 2 or col >= n_cols_neg:
                continue
            ax = fig.add_subplot(gs_neg[row, col])
            ax.imshow(images[card.top_neg_indices[idx]].reshape(IMG_SHAPE),
                     cmap='gray', vmin=0, vmax=1)
            ax.set_title(f'{card.top_neg_activations[idx]:.1f}\n(y={card.top_neg_labels[idx]})',
                         fontsize=6)
            ax.axis('off')

    # ==== Row 2: Activation histogram + class breakdown + top-3 class hists ====

    ax_hist = fig.add_subplot(gs[1, 0])
    if card.all_activations is not None:
        ax_hist.hist(card.all_activations, bins=80, color='steelblue', alpha=0.7,
                     edgecolor='none', density=True)
        thr = ACTIVATION_THRESHOLD
        ax_hist.axvline(thr, color='#d62728', ls='--', lw=1.2, label=f'+{thr}')
        ax_hist.axvline(-thr, color='#1f77b4', ls='--', lw=1.2, label=f'-{thr}')
        ax_hist.set_xlabel('Activation', fontsize=8)
        ax_hist.set_ylabel('Density', fontsize=8)
        freq_str = (f'+:{card.frac_pos:.1%}  -:{card.frac_neg:.1%}  '
                    f'inactive:{card.frac_inactive:.1%}')
        ax_hist.set_title(f'Activation dist  [{freq_str}]', fontsize=8)
        ax_hist.legend(fontsize=6, loc='upper right')

    if card.pos_class_counts is not None:
        ax_brk = fig.add_subplot(gs[1, 1])
        n_cls = len(card.pos_class_counts)
        pos_total = card.pos_class_counts.sum()
        neg_total = card.neg_class_counts.sum()
        pos_pct = card.pos_class_counts / pos_total * 100 if pos_total > 0 else np.zeros(n_cls)
        neg_pct = card.neg_class_counts / neg_total * 100 if neg_total > 0 else np.zeros(n_cls)

        x = np.arange(n_cls)
        ax_brk.bar(x, pos_pct, color='#d62728', alpha=0.7, label=f'pos (n={int(pos_total)})')
        ax_brk.bar(x, -neg_pct, color='#1f77b4', alpha=0.7, label=f'neg (n={int(neg_total)})')
        ax_brk.axhline(0, color='k', lw=0.5)
        ax_brk.set_xticks(x)
        ax_brk.set_xticklabels([str(c) for c in range(n_cls)], fontsize=6)
        ax_brk.set_xlabel('Class', fontsize=7)
        ax_brk.set_ylabel('% of activators', fontsize=7)
        ax_brk.set_title('Class breakdown', fontsize=9)
        ax_brk.legend(fontsize=5, loc='upper right')

    if card.pos_class_counts is not None and card.all_activations is not None and labels is not None:
        gs_top3 = gs[1, 2].subgridspec(2, 1, hspace=0.5)
        class_colors = plt.cm.tab10(np.linspace(0, 1, 10))

        pos_top3 = np.argsort(card.pos_class_counts)[::-1][:3]
        ax_p3 = fig.add_subplot(gs_top3[0])
        above_mask = card.all_activations > ACTIVATION_THRESHOLD
        for c in pos_top3:
            if card.pos_class_counts[c] < 1:
                continue
            mask_c = (labels == c) & above_mask
            if mask_c.any():
                ax_p3.hist(card.all_activations[mask_c], bins=40,
                          color=class_colors[c], alpha=0.5,
                          label=f'c{c} ({card.pos_class_counts[c]:.0f})',
                          density=True, edgecolor='none')
        ax_p3.set_title('Top-3 pos classes', fontsize=8)
        ax_p3.set_xlabel('Activation', fontsize=6)
        ax_p3.legend(fontsize=5, loc='upper right')
        ax_p3.tick_params(labelsize=6)

        neg_top3 = np.argsort(card.neg_class_counts)[::-1][:3]
        ax_n3 = fig.add_subplot(gs_top3[1])
        below_mask = card.all_activations < -ACTIVATION_THRESHOLD
        for c in neg_top3:
            if card.neg_class_counts[c] < 1:
                continue
            mask_c = (labels == c) & below_mask
            if mask_c.any():
                ax_n3.hist(card.all_activations[mask_c], bins=40,
                          color=class_colors[c], alpha=0.5,
                          label=f'c{c} ({card.neg_class_counts[c]:.0f})',
                          density=True, edgecolor='none')
        ax_n3.set_title('Top-3 neg classes', fontsize=8)
        ax_n3.set_xlabel('Activation', fontsize=6)
        ax_n3.legend(fontsize=5, loc='upper left')
        ax_n3.tick_params(labelsize=6)

    # ==== Row 3: Computation modes + T' matrix + class info ====

    if has_modes:
        ax_modes = fig.add_subplot(gs[2, 0])
        all_evals = card.mode_eigenvalues
        sig_mask = np.abs(all_evals) > 0.05 * np.abs(all_evals[0]) if len(all_evals) > 0 else np.array([])
        n_sig = max(1, sig_mask.sum()) if len(sig_mask) > 0 else 1
        n_show = min(n_sig, len(all_evals))
        evals = all_evals[:n_show]
        colors = ['#d62728' if v > 0 else '#1f77b4' for v in evals]
        ax_modes.barh(range(n_show), evals, color=colors)
        ax_modes.set_yticks(range(n_show))
        ax_modes.set_yticklabels([f'm{i}' for i in range(n_show)], fontsize=7)
        ax_modes.set_xlabel('Eigenvalue', fontsize=8)
        ax_modes.set_title(f'Modes ({n_show} sig) | {card.n_nonzero_interactions} nonzero entries',
                           fontsize=9)
        ax_modes.invert_yaxis()

    if has_modes and card.interaction_matrix is not None:
        ax_mat = fig.add_subplot(gs[2, 1])
        M = card.interaction_matrix
        n_in = M.shape[0]
        vmax = np.abs(M).max()
        ax_mat.imshow(M, cmap='RdBu_r',
                     vmin=-vmax if vmax > 0 else -1, vmax=vmax if vmax > 0 else 1,
                     aspect='equal')
        ax_mat.set_title(f"T'[{card.feature_idx},:,:]  ({n_in}x{n_in})", fontsize=9)
        fs = 7 if n_in <= 12 else 5 if n_in <= 20 else 4
        ax_mat.set_xticks(range(n_in))
        ax_mat.set_yticks(range(n_in))
        ax_mat.set_xticklabels(range(n_in), fontsize=fs)
        ax_mat.set_yticklabels(range(n_in), fontsize=fs)
        ax_mat.set_xlabel(f'Input feat (bond {card.bond-1})', fontsize=7)

    # Class effect bar (top) + class activation % heatmap (bottom)
    gs_cls = gs[2, 2].subgridspec(2, 1, height_ratios=[1, 1.2], hspace=0.4)

    ax_cls = fig.add_subplot(gs_cls[0])
    if card.class_effects is not None and card.class_label is None:
        # Non-folded: show unembed effect
        n_cls = len(card.class_effects)
        colors_cls = ['#2ca02c' if v > 0 else '#d62728' for v in card.class_effects]
        ax_cls.bar(range(n_cls), card.class_effects, color=colors_cls)
        ax_cls.set_title('Class effect (unembed)', fontsize=8)
        ax_cls.set_ylabel('Effect', fontsize=7)
    elif card.class_label is not None:
        # Folded: this feature IS the class logit
        n_cls = 10
        colors_cls = ['#2ca02c' if i == card.class_label else '#dddddd'
                      for i in range(n_cls)]
        vals = [1 if i == card.class_label else 0 for i in range(n_cls)]
        ax_cls.bar(range(n_cls), vals, color=colors_cls)
        ax_cls.set_title(f'This feature = Class {card.class_label} logit', fontsize=8)
        ax_cls.set_ylabel('', fontsize=7)
    elif card.class_selectivity is not None:
        n_cls = len(card.class_selectivity)
        colors_cls = ['#2ca02c' if v > 0 else '#d62728' for v in card.class_selectivity]
        ax_cls.bar(range(n_cls), card.class_selectivity, color=colors_cls)
        ax_cls.set_title('Class selectivity (mean act)', fontsize=8)
        ax_cls.set_ylabel('Mean act', fontsize=7)
    if card.class_effects is not None or card.class_selectivity is not None:
        ax_cls.set_xticks(range(n_cls))
        ax_cls.set_xlabel('Class', fontsize=7)
        ax_cls.tick_params(labelsize=6)

    # Class activation % heatmap
    if card.class_act_pct is not None:
        ax_pct = fig.add_subplot(gs_cls[1])
        n_cls = card.class_act_pct.shape[0]
        wg = LinearSegmentedColormap.from_list('wg', ['white', '#2ca02c'])
        pct_max = max(card.class_act_pct.max(), 1)
        ax_pct.imshow(card.class_act_pct, cmap=wg, vmin=0, vmax=pct_max, aspect='auto')
        for r in range(n_cls):
            for col in range(2):
                val = card.class_act_pct[r, col]
                color = 'white' if val > pct_max * 0.6 else 'black'
                ax_pct.text(col, r, f'{val:.1f}%', ha='center', va='center',
                           fontsize=6, color=color, fontweight='bold')
        ax_pct.set_yticks(range(n_cls))
        ax_pct.set_yticklabels([str(c) for c in range(n_cls)], fontsize=6)
        ax_pct.set_xticks([0, 1])
        ax_pct.set_xticklabels(['pos', 'neg'], fontsize=7)
        ax_pct.set_ylabel('Class', fontsize=7)
        ax_pct.set_title('% of class activated', fontsize=8)

    if card.class_label is not None:
        title = (f'Sparse Feature Card: Bond {card.bond}, Feature {card.feature_idx} '
                 f'= CLASS {card.class_label} (var={card.act_variance:.4f})')
    else:
        title = (f'Sparse Feature Card: Bond {card.bond}, Feature {card.feature_idx} '
                 f'(var={card.act_variance:.4f})')
    if card.incoming_edges or card.outgoing_edges:
        title += f' | {len(card.incoming_edges)} in, {len(card.outgoing_edges)} out'
    fig.suptitle(title, fontsize=11, y=0.98)

    os.makedirs(os.path.dirname(save_path) if os.path.dirname(save_path) else '.', exist_ok=True)
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()


def plot_sparse_circuit(nodes, edges, ranks, save_path, title='Sparse Circuit',
                        max_edges_per_layer=50):
    """Circuit graph with active features highlighted."""
    layers = sorted(nodes.keys())
    n_bonds = len(layers)

    # Identify active features (those with any nonzero connection)
    active = set()
    for e in edges:
        active.add((e['layer'], e['src_i']))
        active.add((e['layer'], e['src_j']))
        active.add((e['layer'] + 1, e['dst_k']))

    # Per-layer top edges
    selected = []
    for ell in range(n_bonds - 1):
        layer_edges = sorted([e for e in edges if e['layer'] == ell],
                             key=lambda e: abs(e['weight']), reverse=True)
        selected.extend(layer_edges[:max_edges_per_layer])

    fig, ax = plt.subplots(figsize=(max(10, n_bonds * 4), 8))

    # Draw nodes
    for layer_idx in layers:
        r = len(nodes[layer_idx])
        for f in nodes[layer_idx]:
            y = f / max(1, r - 1) if r > 1 else 0.5
            is_active = (layer_idx, f) in active
            color = 'black' if is_active else '#cccccc'
            size = 7 if is_active else 4
            ax.plot(layer_idx, y, 'o', color=color, markersize=size, zorder=5)
            if is_active:
                ax.annotate(f'{f}', (layer_idx, y), fontsize=5,
                           ha='center', va='bottom', xytext=(0, 4),
                           textcoords='offset points')

    # Draw edges
    if selected:
        max_w = max(abs(e['weight']) for e in selected)
        for e in selected:
            r_in = len(nodes[e['layer']])
            r_out = len(nodes[e['layer'] + 1])
            y_i = e['src_i'] / max(1, r_in - 1) if r_in > 1 else 0.5
            y_j = e['src_j'] / max(1, r_in - 1) if r_in > 1 else 0.5
            y_src = (y_i + y_j) / 2
            y_dst = e['dst_k'] / max(1, r_out - 1) if r_out > 1 else 0.5

            color = '#d62728' if e['weight'] > 0 else '#1f77b4'
            alpha = min(1.0, 0.3 + 0.7 * abs(e['weight']) / max_w)
            lw = 0.5 + 2.5 * abs(e['weight']) / max_w
            ax.annotate('', xy=(e['layer'] + 1, y_dst), xytext=(e['layer'], y_src),
                        arrowprops=dict(arrowstyle='->', color=color,
                                       alpha=alpha, lw=lw))

    # Stats annotation
    for layer_idx in layers:
        r = len(nodes[layer_idx])
        n_active = sum(1 for f in nodes[layer_idx] if (layer_idx, f) in active)
        ax.text(layer_idx, -0.15, f'Bond {layer_idx}\nr={r}, active={n_active}',
                ha='center', fontsize=8)

    ax.set_xlim(-0.5, max(layers) + 0.5)
    ax.set_ylim(-0.25, 1.1)
    ax.set_xticks([])
    ax.set_title(title, fontsize=12)
    ax.set_ylabel('Feature index (normalized)')
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()


def plot_feature_effect_matrix(effect_matrix, save_path, top_r=None):
    """Heatmap of class effects for features at the last bond."""
    if top_r is not None:
        mat = effect_matrix[:, :top_r]
    else:
        mat = effect_matrix
    vmax = np.abs(mat).max()

    fig, ax = plt.subplots(figsize=(max(6, mat.shape[1] * 0.5), 4))
    im = ax.imshow(mat, cmap='RdBu_r', vmin=-vmax, vmax=vmax, aspect='auto')
    ax.set_xlabel('Feature index')
    ax.set_ylabel('Class')
    ax.set_yticks(range(mat.shape[0]))
    ax.set_title(f'Class effect matrix ({mat.shape[1]} features)')
    plt.colorbar(im, ax=ax, label='Effect on class logit')
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()


def plot_feature_overview(all_cards, save_path):
    """Summary: one row per bond, showing feature importance (variance) + class."""
    bonds = sorted(all_cards.keys())
    n_bonds = len(bonds)
    max_feats = max(len(all_cards[b]) for b in bonds)

    fig, axes = plt.subplots(n_bonds, 1, figsize=(max(12, max_feats * 0.3),
                                                    n_bonds * 2.5))
    if n_bonds == 1:
        axes = [axes]

    for row, bond in enumerate(bonds):
        ax = axes[row]
        cards = all_cards[bond]
        variances = [c.act_variance for c in cards]

        # Color by dominant class
        colors = []
        cmap = plt.cm.tab10
        for c in cards:
            if c.class_effects is not None:
                dom_cls = int(np.argmax(np.abs(c.class_effects)))
            elif c.class_selectivity is not None:
                dom_cls = int(np.argmax(np.abs(c.class_selectivity)))
            else:
                dom_cls = 0
            colors.append(cmap(dom_cls / 10))

        ax.bar(range(len(cards)), variances, color=colors, alpha=0.8)

        # Annotate top features
        for i, c in enumerate(cards):
            if c.class_effects is not None:
                dom_cls = int(np.argmax(np.abs(c.class_effects)))
                ax.annotate(f'{dom_cls}', (i, variances[i]),
                           fontsize=5, ha='center', va='bottom')

        n_active = sum(1 for c in cards if len(c.incoming_edges) > 0 or len(c.outgoing_edges) > 0)
        ax.set_ylabel(f'Bond {bond}\nVariance', fontsize=8)
        ax.set_xlabel('Feature index', fontsize=8)
        ax.set_title(f'Bond {bond}: {len(cards)} features, {n_active} active', fontsize=9)
        ax.tick_params(labelsize=5)

    fig.suptitle('Sparse Feature Overview (color = dominant class)', fontsize=12)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()


def plot_circuit_stats(nodes, edges, save_path):
    """Fan-in, fan-out, edge weight distributions."""
    layers = sorted(nodes.keys())

    # Fan-in/out
    fan_in = {}
    fan_out = {}
    for layer_idx, node_list in nodes.items():
        for n in node_list:
            fan_in[(layer_idx, n)] = 0
            fan_out[(layer_idx, n)] = 0
    for e in edges:
        fan_in[(e['layer'] + 1, e['dst_k'])] = fan_in.get((e['layer'] + 1, e['dst_k']), 0) + 1
        fan_out[(e['layer'], e['src_i'])] = fan_out.get((e['layer'], e['src_i']), 0) + 1
        if e['src_i'] != e['src_j']:
            fan_out[(e['layer'], e['src_j'])] = fan_out.get((e['layer'], e['src_j']), 0) + 1

    fig, axes = plt.subplots(1, 3, figsize=(15, 4))

    fi_vals = [v for k, v in fan_in.items() if k[0] > 0]
    axes[0].hist(fi_vals, bins=range(max(fi_vals or [1]) + 2),
                 color='steelblue', edgecolor='white', alpha=0.8)
    axes[0].set_xlabel('Fan-in')
    axes[0].set_ylabel('Count')
    axes[0].set_title(f'Fan-in (mean={np.mean(fi_vals):.1f})')

    fo_vals = [v for k, v in fan_out.items() if k[0] < max(layers)]
    axes[1].hist(fo_vals, bins=range(max(fo_vals or [1]) + 2),
                 color='#d62728', edgecolor='white', alpha=0.8)
    axes[1].set_xlabel('Fan-out')
    axes[1].set_ylabel('Count')
    axes[1].set_title(f'Fan-out (mean={np.mean(fo_vals):.1f})')

    weights = [abs(e['weight']) for e in edges]
    if weights:
        axes[2].hist(weights, bins=50, color='#2ca02c', edgecolor='none', alpha=0.8)
        axes[2].set_yscale('log')
    axes[2].set_xlabel('|weight|')
    axes[2].set_ylabel('Count')
    axes[2].set_title(f'Edge weights (n={len(edges)})')

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()


def plot_sparsity_per_layer(cores, masks, save_path):
    """Bar chart of sparsity and nonzero count per layer."""
    n_layers = len(cores)
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))

    nnz_per_layer = []
    total_per_layer = []
    for i, (c, m) in enumerate(zip(cores, masks)):
        nnz = int(m.sum().item())
        tot = int(m.numel())
        nnz_per_layer.append(nnz)
        total_per_layer.append(tot)

    sparsity = [100 * (1 - nnz / tot) for nnz, tot in zip(nnz_per_layer, total_per_layer)]

    axes[0].bar(range(n_layers), nnz_per_layer, color='steelblue')
    for i in range(n_layers):
        axes[0].text(i, nnz_per_layer[i], f'{nnz_per_layer[i]}', ha='center', va='bottom', fontsize=8)
    axes[0].set_xlabel('Layer')
    axes[0].set_ylabel('Nonzero weights')
    axes[0].set_title('Nonzero weights per layer')
    axes[0].set_xticks(range(n_layers))

    axes[1].bar(range(n_layers), sparsity, color='#d62728')
    for i in range(n_layers):
        axes[1].text(i, sparsity[i], f'{sparsity[i]:.1f}%', ha='center', va='bottom', fontsize=8)
    axes[1].set_xlabel('Layer')
    axes[1].set_ylabel('% sparse')
    axes[1].set_title('Sparsity per layer')
    axes[1].set_xticks(range(n_layers))
    axes[1].set_ylim(0, 100)

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()


# ================================================================
# 4. Pipeline
# ================================================================

def run_sparse_feature_cards(sparse_ckpt_path, device='cuda', save_dir=SAVE_DIR,
                              data_root=DATA_ROOT, max_samples=MAX_SAMPLES,
                              n_examples=N_EXAMPLES, top_k=None):
    """Full pipeline: load sparse model, compute activations, generate feature cards.

    Args:
        sparse_ckpt_path: path to sparse finetuned checkpoint
        device: torch device
        save_dir: output directory
        data_root: SVHN data path
        max_samples: max test samples for activations
        n_examples: top examples per card
        top_k: features per bond to generate cards for (None = all active features)
    """
    os.makedirs(save_dir, exist_ok=True)

    # ── Load sparse model ──
    print("Loading sparse model...")
    ckpt = torch.load(sparse_ckpt_path, map_location='cpu', weights_only=False)
    model_name = ckpt['model_name']
    ranks = ckpt['ranks']
    cores = ckpt['cores']
    embed = ckpt['embed']
    unembed = ckpt['unembed']
    masks = ckpt['masks']

    print(f"  Model: {model_name}")
    print(f"  Ranks: {ranks}")
    acc_finetuned = ckpt.get('acc_finetuned', ckpt.get('acc_pruned', 0))
    if 'acc_pre_prune' in ckpt:
        print(f"  Accuracy: {ckpt['acc_pre_prune']:.4f} -> {ckpt['acc_post_prune']:.4f} "
              f"-> {acc_finetuned:.4f} (pre/post prune/finetuned)")
    else:
        print(f"  Accuracy: {ckpt.get('acc_baseline', 0):.4f} -> {ckpt.get('acc_pruned', 0):.4f} "
              f"-> {acc_finetuned:.4f} (baseline/pruned/finetuned)")
    is_folded = ckpt.get('unembed_folded', False)
    if is_folded:
        print(f"  Unembed: FOLDED into last core (features = classes)")
    print(f"  Core shapes: {[c.shape for c in cores]}")

    # ── Sparsity stats ──
    total_nnz = sum(m.sum().item() for m in masks)
    total_params = sum(m.numel() for m in masks)
    print(f"  Sparsity: {100*(1-total_nnz/total_params):.1f}% ({int(total_nnz)}/{int(total_params)})")

    # ── Load test data ──
    print("Loading test data...")
    _, test_loader, _ = get_svhn_loaders(data_root=data_root)

    # ── Verify accuracy ──
    print("Verifying accuracy...")
    model, _ = load_model(f"models/{model_name}.pt", device)
    acc = evaluate_truncated(embed, cores, unembed, model, test_loader, device)
    print(f"  Verified accuracy: {acc:.4f}")

    # ── Compute activations ──
    print("Computing activations through sparse network...")
    activations, labels, images = compute_sparse_activations(
        cores, embed, unembed, test_loader, device, max_samples)
    print(f"  {len(labels)} samples, bonds: {[a.shape for a in activations]}")

    # ── Build circuit graph ──
    print("Building sparse circuit graph...")
    nodes, edges = build_sparse_circuit(cores)
    print(f"  {len(edges)} edges")

    # Identify active features
    active_features = {}
    for bond in range(len(cores) + 1):
        r = ranks[bond] if bond < len(ranks) else cores[-1].shape[0]
        active_at_bond = set()
        for e in edges:
            if e['layer'] + 1 == bond:
                active_at_bond.add(e['dst_k'])
            if e['layer'] == bond:
                active_at_bond.add(e['src_i'])
                active_at_bond.add(e['src_j'])
        active_features[bond] = sorted(active_at_bond)
        print(f"  Bond {bond}: {len(active_at_bond)} active out of {activations[bond].shape[1]} total")

    # ── Class effects ──
    effect_matrix = compute_class_effects(unembed)

    # ── Generate feature cards ──
    print("\nGenerating feature cards...")
    all_cards = {}
    n_bonds = len(cores) + 1

    for bond in range(n_bonds):
        # Decide which features to make cards for
        if top_k is not None:
            # Top-k by activation variance
            variances = np.var(activations[bond], axis=0)
            feat_indices = np.argsort(variances)[::-1][:top_k].tolist()
        else:
            # All features at this bond
            feat_indices = list(range(activations[bond].shape[1]))

        cards_at_bond = []
        for feat in feat_indices:
            card = generate_sparse_feature_card(
                bond, feat, activations, labels, images,
                cores, embed, unembed, edges, n_examples)
            cards_at_bond.append(card)

        all_cards[bond] = cards_at_bond

    # ── Plot individual feature cards (only for active features) ──
    print("Plotting feature cards for active features...")
    n_plotted = 0
    for bond in range(n_bonds):
        for card in all_cards[bond]:
            is_active = (len(card.incoming_edges) > 0 or len(card.outgoing_edges) > 0
                         or bond == 0 or bond == n_bonds - 1)
            if not is_active:
                continue
            card_path = os.path.join(save_dir,
                                      f'card_bond{bond}_feat{card.feature_idx}.png')
            plot_sparse_feature_card(card, images, card_path, labels=labels)
            n_plotted += 1
    print(f"  Plotted {n_plotted} feature cards")

    # ── Summary plots ──
    print("Generating summary plots...")

    # Circuit graph
    circuit_path = os.path.join(save_dir, 'circuit_graph.png')
    prune_pct = ckpt.get('prune_pct', 90)
    plot_sparse_circuit(nodes, edges, ranks, circuit_path,
                        title=f'{model_name}: Sparse Circuit ({prune_pct}% pruned, '
                              f'acc={acc_finetuned:.4f})')
    print(f"  {circuit_path}")

    # Circuit stats
    stats_path = os.path.join(save_dir, 'circuit_stats.png')
    plot_circuit_stats(nodes, edges, stats_path)
    print(f"  {stats_path}")

    # Effect matrix
    effect_path = os.path.join(save_dir, 'effect_matrix.png')
    plot_feature_effect_matrix(effect_matrix, effect_path)
    print(f"  {effect_path}")

    # Feature overview
    overview_path = os.path.join(save_dir, 'feature_overview.png')
    plot_feature_overview(all_cards, overview_path)
    print(f"  {overview_path}")

    # Sparsity per layer
    sparsity_path = os.path.join(save_dir, 'sparsity_per_layer.png')
    plot_sparsity_per_layer(cores, masks, sparsity_path)
    print(f"  {sparsity_path}")

    # ── Text summary ──
    summary_lines = [
        f"Sparse Feature Cards Summary",
        f"{'=' * 50}",
        f"Model: {model_name}",
        f"Ranks: {ranks}",
        f"Accuracy: {acc_finetuned:.4f} (finetuned)",
        f"Sparsity: {100*(1-total_nnz/total_params):.1f}%",
        f"Circuit: {len(edges)} edges",
        f"",
    ]
    for bond in range(n_bonds):
        cards = all_cards[bond]
        n_active = len(active_features.get(bond, []))
        summary_lines.append(f"Bond {bond}: {len(cards)} features, {n_active} active")
        # Top 5 by variance
        top5 = sorted(cards, key=lambda c: c.act_variance, reverse=True)[:5]
        for c in top5:
            cls_info = ""
            if c.class_effects is not None:
                dom = int(np.argmax(np.abs(c.class_effects)))
                sign = '+' if c.class_effects[dom] > 0 else '-'
                cls_info = f", class {sign}{dom}"
            elif c.class_selectivity is not None:
                dom = int(np.argmax(np.abs(c.class_selectivity)))
                cls_info = f", selective for {dom}"
            summary_lines.append(
                f"  feat {c.feature_idx}: var={c.act_variance:.4f}, "
                f"{len(c.incoming_edges)} in, {len(c.outgoing_edges)} out{cls_info}")
        summary_lines.append("")

    summary = "\n".join(summary_lines)
    summary_path = os.path.join(save_dir, 'summary.txt')
    with open(summary_path, 'w') as f:
        f.write(summary)
    print(f"\n{summary}")
    print(f"All outputs saved to {save_dir}/")

    return {
        'all_cards': all_cards,
        'effect_matrix': effect_matrix,
        'edges': edges,
        'nodes': nodes,
        'activations': activations,
        'labels': labels,
        'images': images,
        'active_features': active_features,
    }


# ================================================================
# CLI
# ================================================================

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--checkpoint', default=None,
                        help='Path to sparse checkpoint (default: folded sparse)')
    parser.add_argument('--save_dir', default=None,
                        help='Output directory (default: results/feature_cards_folded)')
    args = parser.parse_args()

    if args.checkpoint is None:
        # Default: folded sparse (unembed folded into last core)
        sparse_ckpt = os.path.join(RESULTS_DIR, 'chi_net_L3_h256_folded_sparse.pt')
        save_dir = args.save_dir or os.path.join(RESULTS_DIR, 'feature_cards_folded')
    else:
        sparse_ckpt = args.checkpoint
        save_dir = args.save_dir or SAVE_DIR

    run_sparse_feature_cards(
        sparse_ckpt_path=sparse_ckpt,
        device=DEVICE,
        save_dir=save_dir,
        data_root=DATA_ROOT,
        top_k=None,  # all features
    )
