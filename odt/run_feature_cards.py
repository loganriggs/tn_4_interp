# %% [markdown]
# # Feature Cards for ODT-Analyzed Bilinear Networks
#
# Produces a "feature card" for each principal ODT feature at each bond.
# Run cells top-to-bottom, or use "Run All" in VS Code / PyCharm.

# %% Configuration — edit these
import sys, os

# >>> SET THIS TO YOUR PROJECT ROOT <<<
PROJECT_ROOT = "/workspace/tn_4_interp"

ODT_DIR = os.path.join(PROJECT_ROOT, "odt")
sys.path.insert(0, ODT_DIR)
os.chdir(ODT_DIR)

CHECKPOINT = "models/chi_net_L3_h256.pt"  # or "models/residual_L3_h256.pt"
DEVICE = "cuda"          # "cpu" if no GPU
TOP_K = 10               # features per bond (None = all L2-rank features)
DATA_ROOT = os.path.join(PROJECT_ROOT, "modular_addition/dev_interp/notebooks/data")
SAVE_DIR = "images_features"
N_EXAMPLES = 10          # top pos/neg examples per card
IMG_SHAPE = (32, 32)

# Where to cache ODT results (set None to always recompute)
ODT_CACHE = "models/odt_cache_{model_name}.pt"  # {model_name} gets formatted later

# %% Imports
import torch
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from collections import Counter
from dataclasses import dataclass, field
from typing import Optional

from model import ChiNet, ResidualBilinearNet
from odt import run_odt, effective_bond_dim, truncate
from analyze import load_model, extract_cores, evaluate_truncated
from train import get_svhn_loaders

os.makedirs(SAVE_DIR, exist_ok=True)

# %% [markdown]
# ## 1. Core computation functions

# %%
@torch.no_grad()
def compute_odt_activations(odt_results, data_loader, device, max_samples=5000):
    """Forward pass through orth cores, project onto Gram eigvecs at each bond."""
    embed = odt_results['embed_orth'].to(device)
    cores = [c.to(device) for c in odt_results['cores_orth']]
    eigvecs = [v.to(device) for v in odt_results['eigenvectors']]

    all_projs = [[] for _ in range(len(eigvecs))]
    all_labels, all_images = [], []
    total = 0

    for x, y in data_loader:
        x, y = x.to(device), y.to(device)
        batch = x.shape[0]
        ones = torch.ones(batch, 1, device=device)
        h = torch.cat([ones, x], dim=-1)
        h = h @ embed.T
        all_projs[0].append((h @ eigvecs[0]).cpu())

        for i, core in enumerate(cores):
            h = torch.einsum('oij,bi,bj->bo', core, h, h)
            all_projs[i + 1].append((h @ eigvecs[i + 1]).cpu())

        all_labels.append(y.cpu())
        all_images.append(x.cpu())
        total += batch
        if total >= max_samples:
            break

    activations = [torch.cat(p, dim=0).numpy()[:max_samples] for p in all_projs]
    labels = torch.cat(all_labels).numpy()[:max_samples]
    images = torch.cat(all_images).numpy()[:max_samples]
    return activations, labels, images


def compute_feature_effect_matrix(odt_results):
    """effect[c, i] = unembed[c] . eigvecs[last_bond][:, i]"""
    unembed = odt_results['unembed_mod']
    eigvecs_last = odt_results['eigenvectors'][-1]
    return (unembed @ eigvecs_last).detach().numpy()


def decompose_core_in_odt_basis(odt_results, bond, feature_idx, rank=None):
    """Eigendecompose T'[feature_idx, :, :] into excitatory/inhibitory modes."""
    core = odt_results['cores_orth'][bond]
    V_in = odt_results['eigenvectors'][bond]
    V_out = odt_results['eigenvectors'][bond + 1]
    if rank is not None:
        V_in = V_in[:, :rank]

    T_prime = torch.einsum('oij,ok,ia,jb->kab', core, V_out, V_in, V_in)
    M = T_prime[feature_idx]
    M = 0.5 * (M + M.T)

    evals, evecs = torch.linalg.eigh(M)
    idx = evals.abs().argsort(descending=True)
    return evals[idx].detach().numpy(), evecs[:, idx].detach().numpy(), M.detach().numpy()


def compute_feature_inheritance(odt_results, ranks=None):
    """Linear path inheritance: H[k, i] = T'[k, i, 0] + T'[k, 0, i]"""
    cores = odt_results['cores_orth']
    eigvecs = odt_results['eigenvectors']
    inheritance = []
    for bond in range(len(cores)):
        V_in = eigvecs[bond]
        V_out = eigvecs[bond + 1]
        r_in = ranks[bond] if ranks else V_in.shape[1]
        r_out = ranks[bond + 1] if ranks else V_out.shape[1]

        T_prime = torch.einsum('oij,ok,ia,jb->kab',
                               cores[bond], V_out[:, :r_out], V_in[:, :r_in], V_in[:, :r_in])
        H = T_prime[:r_out, :r_in, 0] + T_prime[:r_out, 0, :r_in]
        inheritance.append(H.detach().numpy())
    return inheritance


def build_circuit_graph(odt_results, ranks):
    """Directed graph: (i,j)@bond -> k@bond+1 with weight from T'[k,i,j]."""
    cores = odt_results['cores_orth']
    eigvecs = odt_results['eigenvectors']
    edges = []
    for bond in range(len(cores)):
        r_in, r_out = ranks[bond], ranks[bond + 1]
        T_prime = torch.einsum('oij,ok,ia,jb->kab',
                               cores[bond], eigvecs[bond+1][:, :r_out],
                               eigvecs[bond][:, :r_in], eigvecs[bond][:, :r_in])
        for k in range(r_out):
            for i in range(r_in):
                for j in range(i, r_in):
                    w = T_prime[k, i, j].item()
                    if i != j:
                        w += T_prime[k, j, i].item()
                    if abs(w) > 1e-10:
                        edges.append(dict(layer=bond, src_i=i, src_j=j,
                                          dst_k=k, weight=w,
                                          sign='+' if w > 0 else '-'))
    return edges


def trace_feature_to_input(odt_results, bond, feature_idx, img_shape=(32, 32)):
    """Trace feature's dominant eigvec backward to 32x32 pixel space."""
    cores = odt_results['cores_orth']
    embed = odt_results['embed_orth']
    w = odt_results['eigenvectors'][bond][:, feature_idx]

    for layer_idx in range(bond - 1, -1, -1):
        M = torch.einsum('o,oij->ij', w, cores[layer_idx])
        M = 0.5 * (M + M.T)
        evals, evecs = torch.linalg.eigh(M)
        w = evecs[:, evals.abs().argsort(descending=True)[0]]

    pat = (embed.T @ w)[1:]  # skip constant
    n_px = img_shape[0] * img_shape[1]
    return pat[:n_px].reshape(img_shape).detach().numpy()


def get_l2_ranks(eigenvalues, threshold=0.99):
    """L2-based truncation ranks (capturing threshold of total eigenvalue mass)."""
    ranks = []
    for evals in eigenvalues:
        evals_pos = evals.clamp(min=0)
        total = evals_pos.sum()
        if total > 0:
            r = (evals_pos.cumsum(0) / total < threshold).sum().item() + 1
        else:
            r = 1
        ranks.append(r)
    return ranks


# %% [markdown]
# ## 2. Feature card dataclass

# %%
ACTIVATION_THRESHOLD = 0.4

@dataclass
class FeatureCard:
    """All information about a single ODT feature."""
    bond: int
    feature_idx: int
    eigenvalue: float

    top_pos_indices: np.ndarray = field(default_factory=lambda: np.array([]))
    top_pos_activations: np.ndarray = field(default_factory=lambda: np.array([]))
    top_pos_labels: np.ndarray = field(default_factory=lambda: np.array([]))
    top_neg_indices: np.ndarray = field(default_factory=lambda: np.array([]))
    top_neg_activations: np.ndarray = field(default_factory=lambda: np.array([]))
    top_neg_labels: np.ndarray = field(default_factory=lambda: np.array([]))

    # Activation frequency
    all_activations: Optional[np.ndarray] = None     # full activation vector
    frac_pos: float = 0.0                            # % above +threshold
    frac_neg: float = 0.0                            # % below -threshold
    frac_inactive: float = 0.0                       # % within [-threshold, +threshold]
    low_act_indices: np.ndarray = field(default_factory=lambda: np.array([], dtype=int))
    low_act_activations: np.ndarray = field(default_factory=lambda: np.array([]))
    low_act_labels: np.ndarray = field(default_factory=lambda: np.array([], dtype=int))

    # Class breakdown for pos/neg activators
    pos_class_counts: Optional[np.ndarray] = None    # (n_classes,) counts above threshold
    neg_class_counts: Optional[np.ndarray] = None    # (n_classes,) counts below -threshold

    input_pattern: Optional[np.ndarray] = None
    mode_eigenvalues: Optional[np.ndarray] = None
    mode_eigenvectors: Optional[np.ndarray] = None
    interaction_matrix: Optional[np.ndarray] = None
    class_effects: Optional[np.ndarray] = None      # last bond: unembed projection
    class_selectivity: Optional[np.ndarray] = None   # all bonds: mean activation per class

    incoming_edges: list = field(default_factory=list)
    outgoing_edges: list = field(default_factory=list)


def generate_feature_card(bond, feature_idx, activations, labels, images,
                          odt_results, effect_matrix, edges, n_examples=10,
                          img_shape=(32, 32), threshold=ACTIVATION_THRESHOLD):
    """Assemble all info for one feature into a FeatureCard."""
    gram_ev = odt_results['eigenvalues'][bond][feature_idx].item()
    card = FeatureCard(bond=bond, feature_idx=feature_idx, eigenvalue=gram_ev)

    acts = activations[bond][:, feature_idx]
    n_total = len(acts)

    # Top pos/neg examples
    pos_idx = np.argsort(acts)[::-1][:n_examples]
    neg_idx = np.argsort(acts)[:n_examples]
    card.top_pos_indices = pos_idx
    card.top_pos_activations = acts[pos_idx]
    card.top_pos_labels = labels[pos_idx]
    card.top_neg_indices = neg_idx
    card.top_neg_activations = acts[neg_idx]
    card.top_neg_labels = labels[neg_idx]

    # Activation frequency stats
    card.all_activations = acts
    above = acts > threshold
    below = acts < -threshold
    card.frac_pos = above.sum() / n_total
    card.frac_neg = below.sum() / n_total
    card.frac_inactive = 1.0 - card.frac_pos - card.frac_neg

    # Low-activating counterexamples (closest to 0)
    abs_acts = np.abs(acts)
    low_idx = np.argsort(abs_acts)[:5]
    card.low_act_indices = low_idx
    card.low_act_activations = acts[low_idx]
    card.low_act_labels = labels[low_idx]

    # Class breakdown for pos/neg activators
    n_classes = effect_matrix.shape[0]
    card.pos_class_counts = np.zeros(n_classes)
    card.neg_class_counts = np.zeros(n_classes)
    for c in range(n_classes):
        mask_c = labels == c
        card.pos_class_counts[c] = (above & mask_c).sum()
        card.neg_class_counts[c] = (below & mask_c).sum()

    card.input_pattern = trace_feature_to_input(odt_results, bond, feature_idx, img_shape)

    if bond > 0:
        ev, evec, M = decompose_core_in_odt_basis(odt_results, bond - 1, feature_idx)
        card.mode_eigenvalues = ev
        card.mode_eigenvectors = evec
        card.interaction_matrix = M

    # Class selectivity: mean activation per class (works for ALL bonds)
    class_sel = np.zeros(n_classes)
    for c in range(n_classes):
        mask = labels == c
        if mask.any():
            class_sel[c] = acts[mask].mean()
    card.class_selectivity = class_sel

    # Direct class effects (last bond only: projection through unembed)
    n_bonds = len(odt_results['eigenvalues'])
    if bond == n_bonds - 1:
        card.class_effects = effect_matrix[:, feature_idx]

    card.incoming_edges = [e for e in edges if e['dst_k'] == feature_idx and e['layer'] == bond - 1]
    card.outgoing_edges = [e for e in edges if (e['src_i'] == feature_idx or e['src_j'] == feature_idx) and e['layer'] == bond]
    return card


# %% [markdown]
# ## 3. Plotting functions

# %%
def plot_feature_card(card, images, save_path, img_shape=(32, 32), labels=None):
    """Multi-panel figure: examples, input pattern, modes, class effects,
    activation frequency, and class breakdown."""
    n_pos = len(card.top_pos_indices)
    n_neg = len(card.top_neg_indices)

    fig = plt.figure(figsize=(16, 13))
    gs = gridspec.GridSpec(3, 3, height_ratios=[1, 1, 1],
                           width_ratios=[2, 1, 2], hspace=0.4, wspace=0.3)

    # ---- Row 1: Top examples + input pattern ----

    # Row 1 left: top positive examples
    n_cols = min(5, n_pos)
    if n_cols > 0:
        gs_pos = gs[0, 0].subgridspec(2, n_cols, hspace=0.1, wspace=0.05)
        for idx in range(min(10, n_pos)):
            row, col = idx // 5, idx % 5
            if col >= n_cols:
                continue
            ax = fig.add_subplot(gs_pos[row, col])
            ax.imshow(images[card.top_pos_indices[idx]].reshape(img_shape),
                     cmap='gray', vmin=0, vmax=1)
            ax.set_title(f'{card.top_pos_activations[idx]:.1f}\n(y={card.top_pos_labels[idx]})',
                         fontsize=6)
            ax.axis('off')

    # Row 1 center: input pattern + low-activating counterexamples
    gs_center0 = gs[0, 1].subgridspec(2, 1, height_ratios=[3, 1], hspace=0.3)
    ax_inp = fig.add_subplot(gs_center0[0])
    if card.input_pattern is not None:
        vmax = np.abs(card.input_pattern).max()
        ax_inp.imshow(card.input_pattern, cmap='RdBu_r',
                     vmin=-vmax if vmax > 0 else -1, vmax=vmax if vmax > 0 else 1)
    ax_inp.set_title('Input pattern', fontsize=9)
    ax_inp.axis('off')

    # Low-activating counterexamples
    n_low = len(card.low_act_indices)
    if n_low > 0:
        gs_low = gs_center0[1].subgridspec(1, min(5, n_low), wspace=0.05)
        for idx in range(min(5, n_low)):
            ax = fig.add_subplot(gs_low[0, idx])
            ax.imshow(images[card.low_act_indices[idx]].reshape(img_shape),
                     cmap='gray', vmin=0, vmax=1)
            ax.set_title(f'{card.low_act_activations[idx]:.2f}', fontsize=5)
            ax.axis('off')
        # Label the row
        fig.text(gs_center0[1].get_position(fig).x0 - 0.01,
                 gs_center0[1].get_position(fig).y0 + gs_center0[1].get_position(fig).height / 2,
                 'Low act', fontsize=6, rotation=90, va='center', ha='right')

    # Row 1 right: top negative examples
    n_cols_neg = min(5, n_neg)
    if n_cols_neg > 0:
        gs_neg = gs[0, 2].subgridspec(2, n_cols_neg, hspace=0.1, wspace=0.05)
        for idx in range(min(10, n_neg)):
            row, col = idx // 5, idx % 5
            if col >= n_cols_neg:
                continue
            ax = fig.add_subplot(gs_neg[row, col])
            ax.imshow(images[card.top_neg_indices[idx]].reshape(img_shape),
                     cmap='gray', vmin=0, vmax=1)
            ax.set_title(f'{card.top_neg_activations[idx]:.1f}\n(y={card.top_neg_labels[idx]})',
                         fontsize=6)
            ax.axis('off')

    # ---- Row 2: Computation modes + interaction matrix + class info ----

    # Row 2 left: computation modes
    if card.mode_eigenvalues is not None:
        ax_modes = fig.add_subplot(gs[1, 0])
        n_show = min(10, len(card.mode_eigenvalues))
        evals = card.mode_eigenvalues[:n_show]
        colors = ['#d62728' if v > 0 else '#1f77b4' for v in evals]
        ax_modes.barh(range(n_show), evals, color=colors)
        ax_modes.set_yticks(range(n_show))
        ax_modes.set_yticklabels([f'm{i}' for i in range(n_show)], fontsize=7)
        ax_modes.set_xlabel('Eigenvalue', fontsize=8)
        ax_modes.set_title('Computation modes (red=excit, blue=inhib)', fontsize=9)
        ax_modes.invert_yaxis()

    # Row 2 center: interaction matrix
    if card.interaction_matrix is not None:
        ax_mat = fig.add_subplot(gs[1, 1])
        M = card.interaction_matrix
        ns = min(20, M.shape[0])
        vmax = np.abs(M[:ns, :ns]).max()
        ax_mat.imshow(M[:ns, :ns], cmap='RdBu_r',
                     vmin=-vmax if vmax > 0 else -1, vmax=vmax if vmax > 0 else 1,
                     aspect='equal')
        ax_mat.set_title(f"T'[{card.feature_idx},:,:]", fontsize=9)
        ax_mat.set_xlabel('Input feat', fontsize=7)
        ax_mat.set_ylabel('Input feat', fontsize=7)

    # Row 2 right: class info
    ax_cls = fig.add_subplot(gs[1, 2])
    if card.class_effects is not None:
        n_cls = len(card.class_effects)
        colors_cls = ['#2ca02c' if v > 0 else '#d62728' for v in card.class_effects]
        ax_cls.bar(range(n_cls), card.class_effects, color=colors_cls)
        ax_cls.set_title('Class effect (unembed)', fontsize=9)
        ax_cls.set_ylabel('Effect', fontsize=8)
    elif card.class_selectivity is not None:
        n_cls = len(card.class_selectivity)
        colors_cls = ['#2ca02c' if v > 0 else '#d62728' for v in card.class_selectivity]
        ax_cls.bar(range(n_cls), card.class_selectivity, color=colors_cls)
        ax_cls.set_title('Class selectivity (mean act)', fontsize=9)
        ax_cls.set_ylabel('Mean activation', fontsize=8)
    if card.class_effects is not None or card.class_selectivity is not None:
        ax_cls.set_xticks(range(n_cls))
        ax_cls.set_xlabel('Class', fontsize=8)

    # ---- Row 3: Activation histogram + class breakdown ----

    # Row 3 left: activation histogram with threshold lines + frequency annotation
    ax_hist = fig.add_subplot(gs[2, 0])
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

    # Row 3 center: class breakdown bar charts (pos above, neg below)
    if card.pos_class_counts is not None:
        ax_brk = fig.add_subplot(gs[2, 1])
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

    # Row 3 right: top-3 class histograms for pos & neg
    if card.pos_class_counts is not None and card.all_activations is not None:
        gs_top3 = gs[2, 2].subgridspec(2, 1, hspace=0.5)

        # Top-3 positive classes
        pos_top3 = np.argsort(card.pos_class_counts)[::-1][:3]
        ax_p3 = fig.add_subplot(gs_top3[0])
        above_mask = card.all_activations > ACTIVATION_THRESHOLD
        thr = ACTIVATION_THRESHOLD
        class_colors = plt.cm.tab10(np.linspace(0, 1, 10))
        for rank_i, c in enumerate(pos_top3):
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

        # Top-3 negative classes
        neg_top3 = np.argsort(card.neg_class_counts)[::-1][:3]
        ax_n3 = fig.add_subplot(gs_top3[1])
        below_mask = card.all_activations < -ACTIVATION_THRESHOLD
        for rank_i, c in enumerate(neg_top3):
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

    fig.suptitle(f'Feature Card: Bond {card.bond}, Feature {card.feature_idx} '
                 f'(\u03bb={card.eigenvalue:.4f})', fontsize=12, y=0.98)
    os.makedirs(os.path.dirname(save_path) if os.path.dirname(save_path) else '.', exist_ok=True)
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()


def plot_feature_effect_matrix(effect_matrix, eigenvalues_last, save_path, top_r=None):
    """Heatmap of (n_classes x top_r) feature effects on each class."""
    if top_r is None:
        evals = eigenvalues_last.clamp(min=0).detach().numpy()
        top_r = max(1, int((evals > evals[0] * 1e-4).sum()))
    top_r = min(top_r, effect_matrix.shape[1])
    mat = effect_matrix[:, :top_r]
    vmax = np.abs(mat).max()

    fig, ax = plt.subplots(figsize=(max(6, top_r * 0.4), 4))
    im = ax.imshow(mat, cmap='RdBu_r', vmin=-vmax, vmax=vmax, aspect='auto')
    ax.set_xlabel('Feature index'); ax.set_ylabel('Class')
    ax.set_yticks(range(mat.shape[0]))
    ax.set_title(f'Feature effect matrix (top {top_r} features)')
    plt.colorbar(im, ax=ax, label='Effect on class logit')
    plt.tight_layout()
    plt.savefig(save_path, dpi=150); plt.close()


def plot_circuit_graph(edges, ranks, save_path, top_per_bond=20,
                       effect_matrix=None, n_classes=10, top_class_edges=20):
    """Network diagram: nodes at each layer, edges colored by sign/weight.
    Uses per-bond top-N selection so class edges don't dominate.
    If effect_matrix is provided, adds output class nodes at the end."""
    if not edges:
        return
    n_bonds = len(ranks)
    has_classes = effect_matrix is not None

    # Select top edges PER BOND (not globally)
    selected_edges = []
    for bond in range(n_bonds - 1):
        bond_edges = [e for e in edges if e['layer'] == bond]
        bond_edges.sort(key=lambda e: abs(e['weight']), reverse=True)
        selected_edges.extend(bond_edges[:top_per_bond])

    # Helper: y position for feature f out of r total
    def feat_y(f, r):
        return f / max(1, r - 1) if r > 1 else 0.5

    fig, ax = plt.subplots(figsize=(max(8, (n_bonds + has_classes) * 3), 6))

    # Draw feature nodes at each bond
    for bond_idx in range(n_bonds):
        r = ranks[bond_idx]
        for feat in range(r):
            y = feat_y(feat, r)
            ax.plot(bond_idx, y, 'ko', markersize=6, zorder=5)
            ax.annotate(f'{feat}', (bond_idx, y), fontsize=5,
                       ha='center', va='bottom', xytext=(0, 3), textcoords='offset points')

    # Draw class nodes at the end
    if has_classes:
        cx = n_bonds  # x position for class column
        for c in range(n_classes):
            y = feat_y(c, n_classes)
            ax.plot(cx, y, 's', color='#7f7f7f', markersize=8, zorder=5)
            ax.annotate(f'c{c}', (cx, y), fontsize=6, fontweight='bold',
                       ha='center', va='bottom', xytext=(0, 3), textcoords='offset points')

    # Draw bilinear edges between bonds (per-bond normalization)
    if selected_edges:
        max_w = max(abs(e['weight']) for e in selected_edges)
        for e in selected_edges:
            bond = e['layer']
            r_in, r_out = ranks[bond], ranks[bond + 1]
            y_src = feat_y(e['src_i'], r_in)
            y_dst = feat_y(e['dst_k'], r_out)
            color = '#d62728' if e['weight'] > 0 else '#1f77b4'
            alpha = min(1.0, 0.3 + 0.7 * abs(e['weight']) / max_w)
            lw = 0.5 + 2.5 * abs(e['weight']) / max_w
            ax.annotate('', xy=(bond + 1, y_dst), xytext=(bond, y_src),
                        arrowprops=dict(arrowstyle='->', color=color, alpha=alpha, lw=lw))

    # Draw edges from last bond to class nodes
    if has_classes:
        r_last = ranks[-1]
        cls_edges = []
        for c in range(n_classes):
            for f in range(min(r_last, effect_matrix.shape[1])):
                w = effect_matrix[c, f]
                if abs(w) > 1e-10:
                    cls_edges.append((f, c, w))
        cls_edges.sort(key=lambda x: abs(x[2]), reverse=True)
        top_cls = cls_edges[:top_class_edges]
        if top_cls:
            max_cw = max(abs(e[2]) for e in top_cls)
            for f, c, w in top_cls:
                y_src = feat_y(f, r_last)
                y_dst = feat_y(c, n_classes)
                color = '#2ca02c' if w > 0 else '#d62728'
                alpha = min(1.0, 0.3 + 0.7 * abs(w) / max_cw)
                lw = 0.5 + 2.5 * abs(w) / max_cw
                ax.annotate('', xy=(cx, y_dst), xytext=(n_bonds - 1, y_src),
                            arrowprops=dict(arrowstyle='->', color=color, alpha=alpha, lw=lw))

    n_bilinear = len(selected_edges)
    n_cls_shown = len(top_cls) if has_classes else 0
    x_max = n_bonds + has_classes - 1
    ax.set_xlim(-0.5, x_max + 0.5); ax.set_ylim(-0.1, 1.1)
    xticks = list(range(n_bonds))
    xlabels = [f'Bond {i}\n(r={ranks[i]})' for i in range(n_bonds)]
    if has_classes:
        xticks.append(cx)
        xlabels.append(f'Classes\n(n={n_classes})')
    ax.set_xticks(xticks); ax.set_xticklabels(xlabels)
    ax.set_title(f'Circuit graph ({top_per_bond}/bond bilinear, {n_cls_shown} class edges, '
                 f'red=excit, blue=inhib' + (', green=class+)' if has_classes else ')'))
    ax.set_ylabel('Feature position')
    plt.tight_layout(); plt.savefig(save_path, dpi=150); plt.close()


def plot_computation_modes(card, save_path, n_modes=6):
    """Eigenvalue bar chart + top mode patterns in input feature space."""
    if card.mode_eigenvalues is None:
        return
    n_modes = min(n_modes, len(card.mode_eigenvalues))
    evals = card.mode_eigenvalues[:n_modes]
    evecs = card.mode_eigenvectors[:, :n_modes]

    fig, axes = plt.subplots(1, n_modes + 1, figsize=(2.5 * (n_modes + 1), 3))
    ax = axes[0]
    colors = ['#d62728' if v > 0 else '#1f77b4' for v in evals]
    ax.barh(range(n_modes), evals, color=colors)
    ax.set_yticks(range(n_modes))
    ax.set_yticklabels([f'm{i}' for i in range(n_modes)])
    ax.invert_yaxis(); ax.set_xlabel('\u03bb'); ax.set_title('Mode eigenvalues')

    for m in range(n_modes):
        ax = axes[m + 1]
        v = evecs[:, m]
        n_show = min(15, len(v))
        top_idx = np.argsort(np.abs(v))[::-1][:n_show]
        vals = v[top_idx]
        ax.barh(range(n_show), vals,
               color=['#d62728' if x > 0 else '#1f77b4' for x in vals])
        ax.set_yticks(range(n_show))
        ax.set_yticklabels([f'f{i}' for i in top_idx], fontsize=6)
        ax.invert_yaxis()
        ax.set_title(f'Mode {m} ({"+" if evals[m] > 0 else "-"})', fontsize=9)

    fig.suptitle(f'Bond {card.bond}, Feature {card.feature_idx}: computation modes', fontsize=11)
    plt.tight_layout(); plt.savefig(save_path, dpi=150, bbox_inches='tight'); plt.close()


def plot_feature_overview(all_cards, save_path):
    """Summary grid: one row per bond, eigenvalue bar + class annotations."""
    bonds = sorted(all_cards.keys())
    n_bonds = len(bonds)
    max_feats = max(len(all_cards[b]) for b in bonds)

    fig, axes = plt.subplots(n_bonds, 1,
                              figsize=(max(10, max_feats * 0.8), n_bonds * 2.5))
    if n_bonds == 1:
        axes = [axes]

    for row, bond in enumerate(bonds):
        ax = axes[row]
        cards = all_cards[bond]
        eigenvalues = [c.eigenvalue for c in cards]
        ax.bar(range(len(cards)), eigenvalues, color='steelblue', alpha=0.7)

        for i, c in enumerate(cards):
            if c.class_effects is not None:
                dom = np.argmax(np.abs(c.class_effects))
                sign = '+' if c.class_effects[dom] > 0 else '-'
                ax.annotate(f'{sign}{dom}', (i, eigenvalues[i]),
                           fontsize=6, ha='center', va='bottom')
            if len(c.top_pos_labels) > 0:
                top_cls = Counter(c.top_pos_labels).most_common(1)[0][0]
                ax.annotate(f'\u2191{top_cls}', (i, 0), fontsize=5,
                           ha='center', va='top', color='green',
                           xytext=(0, -5), textcoords='offset points')

        ax.set_ylabel(f'Bond {bond}\neigenvalue', fontsize=8)
        ax.set_xlabel('Feature index', fontsize=8)
        ax.set_xticks(range(len(cards)))
        ax.tick_params(labelsize=6)

    fig.suptitle('Feature overview: eigenvalue spectrum per bond', fontsize=12)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight'); plt.close()


# %% [markdown]
# ## 4. Load model and run ODT (cached)

# %%
print("Loading model...")
model, ckpt = load_model(CHECKPOINT, DEVICE)
model_name = f"{ckpt['model_type']}_L{ckpt['n_layers']}_h{ckpt['hidden_dim']}"
model.to(DEVICE)

# Check for cached ODT results
cache_path = ODT_CACHE.format(model_name=model_name) if ODT_CACHE else None

if cache_path and os.path.isfile(cache_path):
    print(f"Loading cached ODT results from {cache_path}")
    odt_results = torch.load(cache_path, map_location='cpu', weights_only=False)
else:
    print("Extracting cores...")
    embed, cores, unembed = extract_cores(model.cpu())
    print(f"  embed {embed.shape}, cores {[c.shape for c in cores]}, unembed {unembed.shape}")

    print("Running ODT...")
    odt_results = run_odt(embed, cores, unembed, verbose=True)

    if cache_path:
        os.makedirs(os.path.dirname(cache_path) if os.path.dirname(cache_path) else '.', exist_ok=True)
        torch.save(odt_results, cache_path)
        print(f"Saved ODT cache to {cache_path}")

l2_ranks = get_l2_ranks(odt_results['eigenvalues'])
eff_ranks = effective_bond_dim(odt_results['eigenvalues'])
print(f"\nL2 ranks (99%): {l2_ranks}")
print(f"Effective ranks: {eff_ranks}")

ranks = l2_ranks
card_counts = [min(TOP_K, r) for r in ranks] if TOP_K is not None else ranks

# %% [markdown]
# ## 5. Load data and compute activations

# %%
print("Loading SVHN test data...")
_, test_loader, _ = get_svhn_loaders(data_root=DATA_ROOT)

print("Computing ODT activations...")
activations, labels, images = compute_odt_activations(odt_results, test_loader, DEVICE)
print(f"  {len(labels)} samples, bonds: {[a.shape for a in activations]}")

# %%
print("Computing feature effect matrix...")
effect_matrix = compute_feature_effect_matrix(odt_results)

print("Building circuit graph...")
edges = build_circuit_graph(odt_results, ranks)
print(f"  {len(edges)} circuit edges")

# %% [markdown]
# ## 6. Sanity checks

# %%
print("--- Sanity checks ---")
acc_full = evaluate_truncated(
    odt_results['embed_orth'], odt_results['cores_orth'],
    odt_results['unembed_mod'], model, test_loader, DEVICE)
print(f"  Full orth model accuracy: {acc_full:.4f}")

e_trunc, c_trunc, u_trunc = truncate(
    odt_results['embed_orth'], odt_results['cores_orth'],
    odt_results['unembed_mod'], odt_results['eigenvectors'], ranks)
acc_trunc = evaluate_truncated(e_trunc, c_trunc, u_trunc, model, test_loader, DEVICE)
print(f"  Truncated (L2 ranks={ranks}) accuracy: {acc_trunc:.4f}")

# %% [markdown]
# ## 7. Generate feature cards

# %%
all_cards = {}
n_bonds = len(odt_results['eigenvalues'])

for bond in range(n_bonds):
    n_feats = card_counts[bond]
    cards_at_bond = []
    for feat in range(n_feats):
        print(f"  Bond {bond}, Feature {feat}/{n_feats}...")
        card = generate_feature_card(
            bond, feat, activations, labels, images,
            odt_results, effect_matrix, edges,
            n_examples=N_EXAMPLES, img_shape=IMG_SHAPE)
        cards_at_bond.append(card)

        card_path = os.path.join(SAVE_DIR, f'{model_name}_bond{bond}_feat{feat}.png')
        plot_feature_card(card, images, card_path, IMG_SHAPE, labels=labels)

        if card.mode_eigenvalues is not None:
            modes_path = os.path.join(SAVE_DIR, f'{model_name}_bond{bond}_feat{feat}_modes.png')
            plot_computation_modes(card, modes_path)

    all_cards[bond] = cards_at_bond

print(f"Generated {sum(len(v) for v in all_cards.values())} feature cards")

# %% [markdown]
# ## 8. Summary plots

# %%
effect_path = os.path.join(SAVE_DIR, f'{model_name}_effect_matrix.png')
plot_feature_effect_matrix(effect_matrix, odt_results['eigenvalues'][-1], effect_path)
print(f"Saved {effect_path}")

circuit_path = os.path.join(SAVE_DIR, f'{model_name}_circuit.png')
plot_circuit_graph(edges, ranks, circuit_path, effect_matrix=effect_matrix)
print(f"Saved {circuit_path}")

overview_path = os.path.join(SAVE_DIR, f'{model_name}_overview.png')
plot_feature_overview(all_cards, overview_path)
print(f"Saved {overview_path}")

# %%
print("Computing feature inheritance...")
inheritance = compute_feature_inheritance(odt_results, ranks)
for i, H in enumerate(inheritance):
    print(f"  Layer {i}: shape {H.shape}, max |H| = {np.abs(H).max():.4f}")

print(f"\nDone! Outputs in {SAVE_DIR}/")

# %% [markdown]
# ## 9. Inspect a single card interactively
#
# Uncomment and edit the cell below to look at a specific feature.

# %%
# # Pick a card to inspect
# card = all_cards[3][0]  # Bond 3, Feature 0
# print(f"Bond {card.bond}, Feature {card.feature_idx}, lambda={card.eigenvalue:.4f}")
# print(f"Top pos labels: {card.top_pos_labels}")
# print(f"Top neg labels: {card.top_neg_labels}")
# if card.class_effects is not None:
#     print(f"Class effects: {card.class_effects.round(3)}")
# if card.mode_eigenvalues is not None:
#     print(f"Top 5 mode eigenvalues: {card.mode_eigenvalues[:5].round(4)}")
# print(f"Incoming edges: {len(card.incoming_edges)}, Outgoing: {len(card.outgoing_edges)}")
#
# # Show the input pattern inline
# fig, ax = plt.subplots(figsize=(3, 3))
# vmax = np.abs(card.input_pattern).max()
# ax.imshow(card.input_pattern, cmap='RdBu_r', vmin=-vmax, vmax=vmax)
# ax.set_title(f'Bond {card.bond} Feature {card.feature_idx}')
# ax.axis('off')
# plt.show()
