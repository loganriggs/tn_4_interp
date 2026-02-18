"""
Feature cards pipeline for ODT-analyzed bilinear networks.

Produces a "feature card" for each principal ODT feature at each bond,
containing:
  - Top positive and negative activating examples
  - Input attribution pattern (traced to 32x32 pixel space)
  - Computation modes: excitatory (lambda>0) and inhibitory (lambda<0)
  - Output class effect vector
  - Circuit edges (incoming + outgoing)

Key advantage over standard transformer interp: bilinear structure means
all feature interactions are exact quadratic forms, not local linear
approximations.

Usage:
    python feature_cards.py --checkpoint models/chi_net_L3_h256.pt \
        --device cuda --top_k 10 --data_root ./data
"""

import torch
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import argparse
import os
from dataclasses import dataclass, field
from typing import Optional

from model import ChiNet, ResidualBilinearNet
from odt import run_odt, effective_bond_dim, truncate
from analyze import load_model, extract_cores


# ================================================================
# 1. Core computation functions
# ================================================================

@torch.no_grad()
def compute_odt_activations(odt_results, data_loader, device,
                            max_samples=5000):
    """
    Run forward pass through orthogonalized cores on data, recording
    the projection of hidden states onto Gram eigenvectors at each bond.

    Args:
        odt_results: output of run_odt()
        data_loader: test data loader (SVHN format: (batch, 1024))
        device: torch device
        max_samples: cap on number of samples

    Returns:
        activations: list of (n_samples, h) arrays, one per bond
        labels: (n_samples,) int array
        images: (n_samples, 1024) array of raw inputs
    """
    embed = odt_results['embed_orth'].to(device)
    cores = [c.to(device) for c in odt_results['cores_orth']]
    eigvecs = [v.to(device) for v in odt_results['eigenvectors']]

    all_projs = [[] for _ in range(len(eigvecs))]
    all_labels = []
    all_images = []
    total = 0

    for x, y in data_loader:
        x, y = x.to(device), y.to(device)
        batch = x.shape[0]

        # Append constant
        ones = torch.ones(batch, 1, device=device)
        h = torch.cat([ones, x], dim=-1)

        # Embedding
        h = h @ embed.T  # (batch, h_dim)

        # Project at bond 0
        all_projs[0].append((h @ eigvecs[0]).cpu())

        # Forward through cores
        for i, core in enumerate(cores):
            h = torch.einsum('oij,bi,bj->bo', core, h, h)
            all_projs[i + 1].append((h @ eigvecs[i + 1]).cpu())

        all_labels.append(y.cpu())
        all_images.append(x.cpu())
        total += batch
        if total >= max_samples:
            break

    activations = [torch.cat(p, dim=0).numpy()[:max_samples]
                   for p in all_projs]
    labels = torch.cat(all_labels).numpy()[:max_samples]
    images = torch.cat(all_images).numpy()[:max_samples]

    return activations, labels, images


def compute_feature_effect_matrix(odt_results):
    """
    Project unembedding into eigenvector basis at the last bond.

    effect[c, i] = unembed[c] . eigvecs[last_bond][:, i]

    Returns:
        effect: (n_classes, h) matrix
    """
    unembed = odt_results['unembed_mod']  # (n_classes, h)
    eigvecs_last = odt_results['eigenvectors'][-1]  # (h, h)
    effect = unembed @ eigvecs_last  # (n_classes, h)
    return effect.detach().numpy()


def decompose_core_in_odt_basis(odt_results, bond, feature_idx, rank=None):
    """
    Decompose how a feature at bond (bond+1) is computed from features
    at bond (bond) via the core at layer bond.

    Projects core[bond] into the Gram eigenvector bases and extracts
    the interaction matrix T'[feature_idx, :, :] for the target feature.
    Eigendecomposes it into excitatory (lambda>0) and inhibitory (lambda<0) modes.

    Args:
        odt_results: output of run_odt()
        bond: layer index (0-indexed, core[bond] connects bond to bond+1)
        feature_idx: which feature at bond+1 to decompose
        rank: truncation rank for input basis (None = all)

    Returns:
        eigenvalues: 1D array of mode eigenvalues (sorted by |lambda|)
        eigenvectors: 2D array (r_in, n_modes) of mode vectors in input feature space
        T_prime: the full (r_in, r_in) interaction matrix for this output feature
    """
    core = odt_results['cores_orth'][bond]  # (h_out, h_in, h_in)
    V_in = odt_results['eigenvectors'][bond]  # (h, r) at input bond
    V_out = odt_results['eigenvectors'][bond + 1]  # (h, r) at output bond

    if rank is not None:
        V_in = V_in[:, :rank]

    # Project core into eigenvector bases
    # T'[k, a, b] = sum_o,i,j core[o,i,j] * V_out[o,k] * V_in[i,a] * V_in[j,b]
    T_prime = torch.einsum('oij,ok,ia,jb->kab',
                           core, V_out, V_in, V_in)

    # Extract the interaction matrix for this output feature
    M = T_prime[feature_idx]  # (r_in, r_in)
    # Symmetrize (should already be symmetric from core symmetrization)
    M = 0.5 * (M + M.T)

    # Eigendecompose
    evals, evecs = torch.linalg.eigh(M)
    # Sort by absolute magnitude
    idx = evals.abs().argsort(descending=True)
    evals = evals[idx]
    evecs = evecs[:, idx]

    return (evals.detach().numpy(),
            evecs.detach().numpy(),
            M.detach().numpy())


def compute_feature_inheritance(odt_results, ranks=None):
    """
    Compute the "linear path" inheritance matrix at each layer.

    For augmented cores (residual models), the constant dimension (idx 0)
    carries the residual. The inheritance from feature i at bond ℓ to
    feature k at bond ℓ+1 via the linear path is:
        H[k, i] = T'[k, i, 0] + T'[k, 0, i]

    For chi-nets (no constant dimension), this is typically near-zero.

    Args:
        odt_results: output of run_odt()
        ranks: truncation ranks per bond (None = use all)

    Returns:
        inheritance: list of (r_out, r_in) arrays, one per layer
    """
    cores = odt_results['cores_orth']
    eigvecs = odt_results['eigenvectors']
    inheritance = []

    for bond in range(len(cores)):
        V_in = eigvecs[bond]
        V_out = eigvecs[bond + 1]

        r_in = ranks[bond] if ranks else V_in.shape[1]
        r_out = ranks[bond + 1] if ranks else V_out.shape[1]
        V_in = V_in[:, :r_in]
        V_out = V_out[:, :r_out]

        core = cores[bond]
        T_prime = torch.einsum('oij,ok,ia,jb->kab',
                               core, V_out, V_in, V_in)

        # Linear path: contract one input against constant (feature 0)
        H = T_prime[:r_out, :r_in, 0] + T_prime[:r_out, 0, :r_in]
        inheritance.append(H.detach().numpy())

    return inheritance


def build_circuit_graph(odt_results, ranks):
    """
    Build a directed circuit graph through the ODT features.

    Truncates cores to top-r features at each bond and extracts the
    strongest edges (quadratic interactions).

    Args:
        odt_results: output of run_odt()
        ranks: list of truncation ranks, one per bond

    Returns:
        edges: list of dicts with keys:
            layer, src_i, src_j, dst_k, weight, sign
    """
    cores = odt_results['cores_orth']
    eigvecs = odt_results['eigenvectors']
    edges = []

    for bond in range(len(cores)):
        r_in = ranks[bond]
        r_out = ranks[bond + 1]
        V_in = eigvecs[bond][:, :r_in]
        V_out = eigvecs[bond + 1][:, :r_out]

        core = cores[bond]
        T_prime = torch.einsum('oij,ok,ia,jb->kab',
                               core, V_out, V_in, V_in)

        # Extract all edges (including symmetric pairs i,j and j,i)
        for k in range(r_out):
            for i in range(r_in):
                for j in range(i, r_in):  # upper triangle + diagonal
                    w = T_prime[k, i, j].item()
                    if i != j:
                        w += T_prime[k, j, i].item()  # symmetrize
                    if abs(w) > 1e-10:
                        edges.append({
                            'layer': bond,
                            'src_i': i,
                            'src_j': j,
                            'dst_k': k,
                            'weight': w,
                            'sign': '+' if w > 0 else '-',
                        })

    return edges


def trace_feature_to_input(odt_results, bond, feature_idx,
                           img_shape=(32, 32)):
    """
    Trace a single feature's dominant eigenvector backward to input space.

    Starting from eigenvector[bond][:, feature_idx], traces backward
    through cores using the dominant eigenvector at each layer.

    Args:
        odt_results: output of run_odt()
        bond: which bond the feature lives at
        feature_idx: index of the feature at that bond
        img_shape: spatial dims of input

    Returns:
        input_pattern: (H, W) array in pixel space
    """
    cores = odt_results['cores_orth']
    embed = odt_results['embed_orth']
    eigvecs = odt_results['eigenvectors']

    # Start with the feature's eigenvector in the hidden space at this bond
    w = eigvecs[bond][:, feature_idx]

    # Trace backward through cores
    for layer_idx in range(bond - 1, -1, -1):
        core = cores[layer_idx]  # (h_out, h_in, h_in)
        # Contract w with core's output dimension
        M = torch.einsum('o,oij->ij', w, core)
        M = 0.5 * (M + M.T)

        # Take dominant eigenvector
        evals, evecs = torch.linalg.eigh(M)
        idx = evals.abs().argsort(descending=True)[0]
        w = evecs[:, idx]

    # Project through embedding to input space
    input_pattern = embed.T @ w  # (input+1,)
    input_pattern = input_pattern[1:]  # skip constant
    n_pixels = img_shape[0] * img_shape[1]
    input_pattern = input_pattern[:n_pixels].reshape(img_shape)

    return input_pattern.detach().numpy()


# ================================================================
# 2. Feature card assembly
# ================================================================

@dataclass
class FeatureCard:
    """All information about a single ODT feature."""
    bond: int
    feature_idx: int
    eigenvalue: float  # Gram eigenvalue (importance)

    # Top activating examples
    top_pos_indices: np.ndarray = field(default_factory=lambda: np.array([]))
    top_pos_activations: np.ndarray = field(default_factory=lambda: np.array([]))
    top_pos_labels: np.ndarray = field(default_factory=lambda: np.array([]))
    top_neg_indices: np.ndarray = field(default_factory=lambda: np.array([]))
    top_neg_activations: np.ndarray = field(default_factory=lambda: np.array([]))
    top_neg_labels: np.ndarray = field(default_factory=lambda: np.array([]))

    # Input attribution
    input_pattern: Optional[np.ndarray] = None  # (32, 32)

    # Computation modes
    mode_eigenvalues: Optional[np.ndarray] = None
    mode_eigenvectors: Optional[np.ndarray] = None
    interaction_matrix: Optional[np.ndarray] = None

    # Output effect
    class_effects: Optional[np.ndarray] = None  # (n_classes,)

    # Circuit connections
    incoming_edges: list = field(default_factory=list)
    outgoing_edges: list = field(default_factory=list)


def generate_feature_card(bond, feature_idx, activations, labels, images,
                          odt_results, effect_matrix, edges, n_examples=10,
                          img_shape=(32, 32)):
    """
    Assemble a complete feature card for one ODT feature.

    Args:
        bond: bond index
        feature_idx: feature index at this bond
        activations: list of (n_samples, h) arrays from compute_odt_activations
        labels: (n_samples,) label array
        images: (n_samples, 1024) raw input array
        odt_results: output of run_odt()
        effect_matrix: (n_classes, h) from compute_feature_effect_matrix
        edges: circuit graph edges from build_circuit_graph
        n_examples: number of top positive/negative examples
        img_shape: spatial dims

    Returns:
        FeatureCard
    """
    evals = odt_results['eigenvalues'][bond]
    gram_eigenvalue = evals[feature_idx].item()

    card = FeatureCard(
        bond=bond,
        feature_idx=feature_idx,
        eigenvalue=gram_eigenvalue,
    )

    # --- Top activating examples ---
    acts = activations[bond][:, feature_idx]  # (n_samples,)
    pos_idx = np.argsort(acts)[::-1][:n_examples]
    neg_idx = np.argsort(acts)[:n_examples]

    card.top_pos_indices = pos_idx
    card.top_pos_activations = acts[pos_idx]
    card.top_pos_labels = labels[pos_idx]
    card.top_neg_indices = neg_idx
    card.top_neg_activations = acts[neg_idx]
    card.top_neg_labels = labels[neg_idx]

    # --- Input attribution ---
    card.input_pattern = trace_feature_to_input(
        odt_results, bond, feature_idx, img_shape)

    # --- Computation modes (for bonds > 0) ---
    if bond > 0:
        mode_evals, mode_evecs, M = decompose_core_in_odt_basis(
            odt_results, bond - 1, feature_idx)
        card.mode_eigenvalues = mode_evals
        card.mode_eigenvectors = mode_evecs
        card.interaction_matrix = M

    # --- Output effect (for last bond only) ---
    n_bonds = len(odt_results['eigenvalues'])
    if bond == n_bonds - 1:
        card.class_effects = effect_matrix[:, feature_idx]

    # --- Circuit edges ---
    card.incoming_edges = [e for e in edges
                           if e['dst_k'] == feature_idx
                           and e['layer'] == bond - 1]
    card.outgoing_edges = [e for e in edges
                           if (e['src_i'] == feature_idx or
                               e['src_j'] == feature_idx)
                           and e['layer'] == bond]

    return card


# ================================================================
# 3. Visualization functions
# ================================================================

def plot_feature_card(card, images, save_path, img_shape=(32, 32)):
    """
    Plot a single feature card as a multi-panel figure.

    Layout:
      Row 1: Top positive examples | Input pattern | Top negative examples
      Row 2: Computation modes (bar + patterns) | Class effects
    """
    n_pos = len(card.top_pos_indices)
    n_neg = len(card.top_neg_indices)
    has_modes = card.mode_eigenvalues is not None
    has_effects = card.class_effects is not None

    fig = plt.figure(figsize=(16, 8))
    gs = gridspec.GridSpec(2, 3, height_ratios=[1, 1],
                           width_ratios=[2, 1, 2], hspace=0.35, wspace=0.3)

    # --- Row 1 left: top positive examples ---
    gs_pos = gs[0, 0].subgridspec(2, min(5, n_pos), hspace=0.1, wspace=0.05)
    for idx in range(min(10, n_pos)):
        row, col = idx // 5, idx % 5
        if col >= min(5, n_pos):
            continue
        ax = fig.add_subplot(gs_pos[row, col])
        img = images[card.top_pos_indices[idx]].reshape(img_shape)
        ax.imshow(img, cmap='gray', vmin=0, vmax=1)
        ax.set_title(f'{card.top_pos_activations[idx]:.1f}\n(y={card.top_pos_labels[idx]})',
                     fontsize=6)
        ax.axis('off')

    # --- Row 1 center: input pattern ---
    ax_inp = fig.add_subplot(gs[0, 1])
    if card.input_pattern is not None:
        vmax = np.abs(card.input_pattern).max()
        if vmax > 0:
            ax_inp.imshow(card.input_pattern, cmap='RdBu_r',
                         vmin=-vmax, vmax=vmax)
        else:
            ax_inp.imshow(card.input_pattern, cmap='RdBu_r')
    ax_inp.set_title('Input pattern', fontsize=9)
    ax_inp.axis('off')

    # --- Row 1 right: top negative examples ---
    gs_neg = gs[0, 2].subgridspec(2, min(5, n_neg), hspace=0.1, wspace=0.05)
    for idx in range(min(10, n_neg)):
        row, col = idx // 5, idx % 5
        if col >= min(5, n_neg):
            continue
        ax = fig.add_subplot(gs_neg[row, col])
        img = images[card.top_neg_indices[idx]].reshape(img_shape)
        ax.imshow(img, cmap='gray', vmin=0, vmax=1)
        ax.set_title(f'{card.top_neg_activations[idx]:.1f}\n(y={card.top_neg_labels[idx]})',
                     fontsize=6)
        ax.axis('off')

    # --- Row 2 left: computation modes ---
    if has_modes:
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

    # --- Row 2 center: interaction matrix ---
    if has_modes and card.interaction_matrix is not None:
        ax_mat = fig.add_subplot(gs[1, 1])
        M = card.interaction_matrix
        n_show_mat = min(20, M.shape[0])
        vmax = np.abs(M[:n_show_mat, :n_show_mat]).max()
        if vmax > 0:
            ax_mat.imshow(M[:n_show_mat, :n_show_mat], cmap='RdBu_r',
                         vmin=-vmax, vmax=vmax, aspect='equal')
        else:
            ax_mat.imshow(M[:n_show_mat, :n_show_mat], cmap='RdBu_r',
                         aspect='equal')
        ax_mat.set_title(f'T\'[{card.feature_idx},:,:]', fontsize=9)
        ax_mat.set_xlabel('Input feat', fontsize=7)
        ax_mat.set_ylabel('Input feat', fontsize=7)

    # --- Row 2 right: class effects ---
    if has_effects:
        ax_cls = fig.add_subplot(gs[1, 2])
        n_classes = len(card.class_effects)
        colors_cls = ['#2ca02c' if v > 0 else '#d62728'
                      for v in card.class_effects]
        ax_cls.bar(range(n_classes), card.class_effects, color=colors_cls)
        ax_cls.set_xticks(range(n_classes))
        ax_cls.set_xlabel('Class', fontsize=8)
        ax_cls.set_ylabel('Effect', fontsize=8)
        ax_cls.set_title('Class effect vector', fontsize=9)

    fig.suptitle(f'Feature Card: Bond {card.bond}, Feature {card.feature_idx} '
                 f'(λ={card.eigenvalue:.4f})', fontsize=12, y=0.98)

    os.makedirs(os.path.dirname(save_path) if os.path.dirname(save_path) else '.', exist_ok=True)
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()


def plot_feature_effect_matrix(effect_matrix, eigenvalues_last, save_path,
                               top_r=None):
    """
    Heatmap of (n_classes x top_r) feature effects on each class.

    Args:
        effect_matrix: (n_classes, h) from compute_feature_effect_matrix
        eigenvalues_last: eigenvalues at last bond (for importance weighting)
        save_path: where to save
        top_r: number of features to show (default: all with significant eigenvalue)
    """
    if top_r is None:
        evals = eigenvalues_last.clamp(min=0).detach().numpy()
        top_r = max(1, int((evals > evals[0] * 1e-4).sum()))
    top_r = min(top_r, effect_matrix.shape[1])

    mat = effect_matrix[:, :top_r]
    vmax = np.abs(mat).max()

    fig, ax = plt.subplots(figsize=(max(6, top_r * 0.4), 4))
    im = ax.imshow(mat, cmap='RdBu_r', vmin=-vmax, vmax=vmax, aspect='auto')
    ax.set_xlabel('Feature index')
    ax.set_ylabel('Class')
    ax.set_yticks(range(mat.shape[0]))
    ax.set_title(f'Feature effect matrix (top {top_r} features)')
    plt.colorbar(im, ax=ax, label='Effect on class logit')
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()


def plot_circuit_graph(edges, ranks, save_path, top_n=50):
    """
    Network diagram: nodes at each layer, edges colored by sign/weight.

    Args:
        edges: from build_circuit_graph()
        ranks: truncation ranks per bond
        save_path: where to save
        top_n: maximum number of edges to display
    """
    if not edges:
        return

    # Sort by absolute weight, take top_n
    sorted_edges = sorted(edges, key=lambda e: abs(e['weight']), reverse=True)
    top_edges = sorted_edges[:top_n]

    n_bonds = len(ranks)
    fig, ax = plt.subplots(figsize=(max(8, n_bonds * 3), 6))

    # Draw nodes
    for bond_idx in range(n_bonds):
        r = ranks[bond_idx]
        for feat in range(r):
            y = feat / max(1, r - 1) if r > 1 else 0.5
            ax.plot(bond_idx, y, 'ko', markersize=6, zorder=5)
            ax.annotate(f'{feat}', (bond_idx, y), fontsize=5,
                       ha='center', va='bottom', xytext=(0, 3),
                       textcoords='offset points')

    # Draw edges
    max_w = max(abs(e['weight']) for e in top_edges) if top_edges else 1
    for e in top_edges:
        bond = e['layer']
        r_in = ranks[bond]
        r_out = ranks[bond + 1]

        # Source node positions (midpoint of the pair)
        y_i = e['src_i'] / max(1, r_in - 1) if r_in > 1 else 0.5
        y_j = e['src_j'] / max(1, r_in - 1) if r_in > 1 else 0.5
        y_src = (y_i + y_j) / 2

        y_dst = e['dst_k'] / max(1, r_out - 1) if r_out > 1 else 0.5

        color = '#d62728' if e['weight'] > 0 else '#1f77b4'
        alpha = min(1.0, 0.3 + 0.7 * abs(e['weight']) / max_w)
        lw = 0.5 + 2.5 * abs(e['weight']) / max_w

        ax.annotate('', xy=(bond + 1, y_dst), xytext=(bond, y_src),
                    arrowprops=dict(arrowstyle='->', color=color,
                                   alpha=alpha, lw=lw))

    ax.set_xlim(-0.5, n_bonds - 0.5)
    ax.set_ylim(-0.1, 1.1)
    ax.set_xticks(range(n_bonds))
    ax.set_xticklabels([f'Bond {i}\n(r={ranks[i]})' for i in range(n_bonds)])
    ax.set_title(f'Circuit graph (top {len(top_edges)} edges, '
                 f'red=excit, blue=inhib)')
    ax.set_ylabel('Feature position')
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()


def plot_computation_modes(card, odt_results, save_path, n_modes=6):
    """
    Detailed view of computation modes for one feature.

    Shows eigenvalue bar chart and top mode patterns projected to
    previous bond's feature space.

    Args:
        card: FeatureCard with mode_eigenvalues/mode_eigenvectors
        odt_results: for tracing patterns to input
        save_path: where to save
        n_modes: number of top modes to show
    """
    if card.mode_eigenvalues is None:
        return

    n_modes = min(n_modes, len(card.mode_eigenvalues))
    evals = card.mode_eigenvalues[:n_modes]
    evecs = card.mode_eigenvectors[:, :n_modes]

    fig, axes = plt.subplots(1, n_modes + 1,
                              figsize=(2.5 * (n_modes + 1), 3))

    # Eigenvalue bar chart
    ax = axes[0]
    colors = ['#d62728' if v > 0 else '#1f77b4' for v in evals]
    ax.barh(range(n_modes), evals, color=colors)
    ax.set_yticks(range(n_modes))
    ax.set_yticklabels([f'm{i}' for i in range(n_modes)])
    ax.invert_yaxis()
    ax.set_xlabel('λ')
    ax.set_title('Mode eigenvalues')

    # Mode patterns: bar chart of components in input feature space
    for m in range(n_modes):
        ax = axes[m + 1]
        v = evecs[:, m]
        n_show = min(15, len(v))
        top_idx = np.argsort(np.abs(v))[::-1][:n_show]
        vals = v[top_idx]
        colors_v = ['#d62728' if x > 0 else '#1f77b4' for x in vals]
        ax.barh(range(n_show), vals, color=colors_v)
        ax.set_yticks(range(n_show))
        ax.set_yticklabels([f'f{i}' for i in top_idx], fontsize=6)
        ax.invert_yaxis()
        sign = '+' if evals[m] > 0 else '-'
        ax.set_title(f'Mode {m} ({sign})', fontsize=9)

    fig.suptitle(f'Bond {card.bond}, Feature {card.feature_idx}: '
                 f'computation modes', fontsize=11)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()


def plot_feature_overview(all_cards, odt_results, save_path):
    """
    Summary grid: one row per bond, showing all features with their
    eigenvalues and dominant class associations.

    Args:
        all_cards: dict {bond: [FeatureCard, ...]}
        odt_results: for eigenvalue info
        save_path: where to save
    """
    bonds = sorted(all_cards.keys())
    n_bonds = len(bonds)
    max_feats = max(len(all_cards[b]) for b in bonds)

    fig, axes = plt.subplots(n_bonds, 1, figsize=(max(10, max_feats * 0.8),
                                                    n_bonds * 2.5))
    if n_bonds == 1:
        axes = [axes]

    for row, bond in enumerate(bonds):
        ax = axes[row]
        cards = all_cards[bond]
        n_f = len(cards)

        eigenvalues = [c.eigenvalue for c in cards]
        ax.bar(range(n_f), eigenvalues, color='steelblue', alpha=0.7)

        # Annotate with dominant class if available
        for i, c in enumerate(cards):
            if c.class_effects is not None:
                dom_cls = np.argmax(np.abs(c.class_effects))
                sign = '+' if c.class_effects[dom_cls] > 0 else '-'
                ax.annotate(f'{sign}{dom_cls}', (i, eigenvalues[i]),
                           fontsize=6, ha='center', va='bottom')
            # Label with pos/neg class distribution
            if len(c.top_pos_labels) > 0:
                from collections import Counter
                top_cls = Counter(c.top_pos_labels).most_common(1)[0][0]
                ax.annotate(f'↑{top_cls}', (i, 0), fontsize=5,
                           ha='center', va='top', color='green',
                           xytext=(0, -5), textcoords='offset points')

        ax.set_ylabel(f'Bond {bond}\neigenvalue', fontsize=8)
        ax.set_xlabel('Feature index', fontsize=8)
        ax.set_xticks(range(n_f))
        ax.tick_params(labelsize=6)

    fig.suptitle('Feature overview: eigenvalue spectrum per bond', fontsize=12)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()


# ================================================================
# 4. Pipeline runner
# ================================================================

def get_l2_ranks(eigenvalues, threshold=0.99):
    """Compute L2-based truncation ranks (capturing threshold of total eigenvalue mass)."""
    ranks = []
    for evals in eigenvalues:
        evals_pos = evals.clamp(min=0)
        total = evals_pos.sum()
        if total > 0:
            cumsum = evals_pos.cumsum(0) / total
            r = (cumsum < threshold).sum().item() + 1
        else:
            r = 1
        ranks.append(r)
    return ranks


def run_feature_cards(checkpoint_path, device='cpu', top_k=None,
                      data_root='./data', save_dir='images_features',
                      n_examples=10, img_shape=(32, 32)):
    """
    Full pipeline: load model, run ODT, compute activations, generate
    and plot feature cards.

    Args:
        checkpoint_path: path to trained model checkpoint
        device: torch device
        top_k: features per bond to generate cards for (None = L2 rank)
        data_root: path to SVHN data
        save_dir: output directory
        n_examples: examples per card
        img_shape: spatial dims
    """
    os.makedirs(save_dir, exist_ok=True)

    # --- Load model and run ODT ---
    print("Loading model...")
    model, ckpt = load_model(checkpoint_path, device)
    model_name = f"{ckpt['model_type']}_L{ckpt['n_layers']}_h{ckpt['hidden_dim']}"
    model.to(device)

    print("Extracting cores...")
    embed, cores, unembed = extract_cores(model.cpu())
    print(f"  Core shapes: embed {embed.shape}, "
          f"cores {[c.shape for c in cores]}, unembed {unembed.shape}")

    print("Running ODT...")
    odt_results = run_odt(embed, cores, unembed, verbose=True)

    # --- Compute ranks ---
    l2_ranks = get_l2_ranks(odt_results['eigenvalues'])
    eff_ranks = effective_bond_dim(odt_results['eigenvalues'])
    print(f"\nL2 ranks (99%): {l2_ranks}")
    print(f"Effective ranks: {eff_ranks}")

    # Use L2 ranks for truncation, but generate cards for top_k features
    ranks = l2_ranks
    if top_k is not None:
        card_counts = [min(top_k, r) for r in ranks]
    else:
        card_counts = ranks

    # --- Load data and compute activations ---
    print("\nLoading SVHN test data...")
    from train import get_svhn_loaders
    _, test_loader, _ = get_svhn_loaders(data_root=data_root)

    print("Computing ODT activations...")
    activations, labels, images = compute_odt_activations(
        odt_results, test_loader, device)
    print(f"  Got {len(labels)} samples, "
          f"bonds: {[a.shape for a in activations]}")

    # --- Compute global quantities ---
    print("Computing feature effect matrix...")
    effect_matrix = compute_feature_effect_matrix(odt_results)

    print("Building circuit graph...")
    edges = build_circuit_graph(odt_results, ranks)
    print(f"  {len(edges)} circuit edges")

    # --- Sanity checks ---
    print("\n--- Sanity checks ---")
    # Check: forward pass through orth cores matches
    from analyze import evaluate_truncated
    acc_full = evaluate_truncated(
        odt_results['embed_orth'], odt_results['cores_orth'],
        odt_results['unembed_mod'], model, test_loader, device)
    print(f"  Full orth model accuracy: {acc_full:.4f}")

    # Check: truncated forward pass
    e_trunc, c_trunc, u_trunc = truncate(
        odt_results['embed_orth'], odt_results['cores_orth'],
        odt_results['unembed_mod'], odt_results['eigenvectors'], ranks)
    acc_trunc = evaluate_truncated(
        e_trunc, c_trunc, u_trunc, model, test_loader, device)
    print(f"  Truncated (L2 ranks={ranks}) accuracy: {acc_trunc:.4f}")

    # --- Generate feature cards ---
    print(f"\nGenerating feature cards...")
    all_cards = {}
    n_bonds = len(odt_results['eigenvalues'])

    for bond in range(n_bonds):
        n_feats = card_counts[bond]
        # Skip constant dimension for augmented models
        h_dim = odt_results['eigenvalues'][bond].shape[0]
        is_augmented = (h_dim != ckpt['hidden_dim'])
        start_feat = 0  # feature 0 may be constant for augmented, but still interesting to see

        cards_at_bond = []
        for feat in range(start_feat, n_feats):
            print(f"  Bond {bond}, Feature {feat}/{n_feats}...")
            card = generate_feature_card(
                bond, feat, activations, labels, images,
                odt_results, effect_matrix, edges,
                n_examples=n_examples, img_shape=img_shape)
            cards_at_bond.append(card)

            # Plot individual card
            card_path = os.path.join(save_dir,
                                      f'{model_name}_bond{bond}_feat{feat}.png')
            plot_feature_card(card, images, card_path, img_shape)

            # Plot computation modes if available
            if card.mode_eigenvalues is not None:
                modes_path = os.path.join(save_dir,
                                           f'{model_name}_bond{bond}_feat{feat}_modes.png')
                plot_computation_modes(card, odt_results, modes_path)

        all_cards[bond] = cards_at_bond

    # --- Summary plots ---
    print("\nGenerating summary plots...")

    # Feature effect matrix
    effect_path = os.path.join(save_dir, f'{model_name}_effect_matrix.png')
    plot_feature_effect_matrix(effect_matrix, odt_results['eigenvalues'][-1],
                               effect_path)
    print(f"  Saved {effect_path}")

    # Circuit graph
    circuit_path = os.path.join(save_dir, f'{model_name}_circuit.png')
    plot_circuit_graph(edges, ranks, circuit_path)
    print(f"  Saved {circuit_path}")

    # Feature overview
    overview_path = os.path.join(save_dir, f'{model_name}_overview.png')
    plot_feature_overview(all_cards, odt_results, overview_path)
    print(f"  Saved {overview_path}")

    # --- Inheritance analysis ---
    print("\nComputing feature inheritance...")
    inheritance = compute_feature_inheritance(odt_results, ranks)
    for i, H in enumerate(inheritance):
        print(f"  Layer {i}: inheritance matrix shape {H.shape}, "
              f"max |H| = {np.abs(H).max():.4f}")

    print(f"\nDone! All outputs saved to {save_dir}/")
    return {
        'odt_results': odt_results,
        'all_cards': all_cards,
        'effect_matrix': effect_matrix,
        'edges': edges,
        'ranks': ranks,
        'activations': activations,
        'labels': labels,
        'images': images,
        'inheritance': inheritance,
    }


# ================================================================
# CLI
# ================================================================

if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Generate feature cards for ODT-analyzed bilinear networks')
    parser.add_argument('--checkpoint', type=str, required=True,
                        help='Path to trained model checkpoint')
    parser.add_argument('--device', type=str, default='cpu')
    parser.add_argument('--top_k', type=int, default=None,
                        help='Features per bond (default: L2 rank)')
    parser.add_argument('--data_root', type=str, default='./data',
                        help='Path to SVHN data')
    parser.add_argument('--save_dir', type=str, default='images_features',
                        help='Output directory')
    parser.add_argument('--n_examples', type=int, default=10,
                        help='Top examples per card')
    args = parser.parse_args()

    run_feature_cards(
        checkpoint_path=args.checkpoint,
        device=args.device,
        top_k=args.top_k,
        data_root=args.data_root,
        save_dir=args.save_dir,
        n_examples=args.n_examples,
    )
