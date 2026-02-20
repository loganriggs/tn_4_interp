# %% [markdown]
# # Circuit Graph Visualization for Sparse Bilinear Networks
#
# Visualizes the sparse circuit graph after rotation optimization.
# Each hidden node computes one quadratic operation s_m = (l_m . x)(r_m . x).

# %% Configuration
import sys, os

PROJECT_ROOT = "/workspace/tn_4_interp"
ODT_DIR = os.path.join(PROJECT_ROOT, "odt")
sys.path.insert(0, ODT_DIR)
os.chdir(ODT_DIR)

RESULTS_DIR = os.path.join(PROJECT_ROOT, "sparse_circuits", "results")
DEVICE = "cpu"

# %% Imports
import torch
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

from analyze import load_model
from train import get_svhn_loaders


# %%
def load_sparse_rotation(model_name, results_dir=RESULTS_DIR):
    """Load saved sparse rotation results."""
    path = os.path.join(results_dir, f'{model_name}_sparse_rotation.pt')
    return torch.load(path, map_location='cpu', weights_only=False)


def build_sparse_circuit(rotated_cores, threshold_frac=1e-3):
    """Build a sparse circuit graph from rotated cores.

    For each layer, extracts the nonzero entries of the symmetrized
    interaction matrices T'[k,:,:].

    Returns:
        nodes: dict {layer: list of node indices}
        edges: list of dicts with layer, src_i, src_j, dst_k, weight
    """
    edges = []
    nodes = {}

    for ell, core in enumerate(rotated_cores):
        T = 0.5 * (core + core.transpose(-1, -2))
        r_out, r_in, _ = T.shape
        max_val = T.abs().max()
        threshold = threshold_frac * max_val

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


def compute_fan_in_out(edges, nodes):
    """Compute fan-in and fan-out for each node."""
    fan_in = {}   # how many edges target this node
    fan_out = {}  # how many edges originate from this node

    for layer_idx, node_list in nodes.items():
        for n in node_list:
            fan_in[(layer_idx, n)] = 0
            fan_out[(layer_idx, n)] = 0

    for e in edges:
        fan_in[(e['layer'] + 1, e['dst_k'])] = fan_in.get((e['layer'] + 1, e['dst_k']), 0) + 1
        fan_out[(e['layer'], e['src_i'])] = fan_out.get((e['layer'], e['src_i']), 0) + 1
        if e['src_i'] != e['src_j']:
            fan_out[(e['layer'], e['src_j'])] = fan_out.get((e['layer'], e['src_j']), 0) + 1

    return fan_in, fan_out


def plot_sparse_circuit(nodes, edges, save_path, title='Sparse Circuit Graph',
                        max_edges_per_layer=30):
    """Plot the sparse circuit graph with per-layer edge selection."""
    n_bonds = len(nodes)
    layers = sorted(nodes.keys())

    # Per-layer edge selection
    selected_edges = []
    for ell in range(n_bonds - 1):
        layer_edges = [e for e in edges if e['layer'] == ell]
        layer_edges.sort(key=lambda e: abs(e['weight']), reverse=True)
        selected_edges.extend(layer_edges[:max_edges_per_layer])

    def feat_y(f, r):
        return f / max(1, r - 1) if r > 1 else 0.5

    fig, ax = plt.subplots(figsize=(max(8, n_bonds * 3), 6))

    # Draw nodes
    for layer_idx in layers:
        node_list = nodes[layer_idx]
        r = len(node_list)
        for f in node_list:
            y = feat_y(f, r)
            ax.plot(layer_idx, y, 'ko', markersize=7, zorder=5)
            ax.annotate(f'{f}', (layer_idx, y), fontsize=6,
                       ha='center', va='bottom', xytext=(0, 4),
                       textcoords='offset points')

    # Draw edges
    if selected_edges:
        max_w = max(abs(e['weight']) for e in selected_edges)
        for e in selected_edges:
            r_in = len(nodes[e['layer']])
            r_out = len(nodes[e['layer'] + 1])
            y_src = feat_y(e['src_i'], r_in)
            y_dst = feat_y(e['dst_k'], r_out)
            color = '#d62728' if e['weight'] > 0 else '#1f77b4'
            alpha = min(1.0, 0.3 + 0.7 * abs(e['weight']) / max_w)
            lw = 0.5 + 2.5 * abs(e['weight']) / max_w
            ax.annotate('', xy=(e['layer'] + 1, y_dst),
                        xytext=(e['layer'], y_src),
                        arrowprops=dict(arrowstyle='->', color=color,
                                       alpha=alpha, lw=lw))

    ax.set_xlim(-0.5, max(layers) + 0.5)
    ax.set_ylim(-0.1, 1.1)
    ax.set_xticks(layers)
    ax.set_xticklabels([f'Bond {i}\n(r={len(nodes[i])})' for i in layers])
    ax.set_title(title)
    ax.set_ylabel('Feature')
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()


def plot_circuit_stats(nodes, edges, save_path):
    """Plot circuit statistics: fan-in/out distributions, edge weight distribution."""
    fan_in, fan_out = compute_fan_in_out(edges, nodes)
    layers = sorted(nodes.keys())

    fig, axes = plt.subplots(1, 3, figsize=(14, 4))

    # Fan-in distribution
    fi_vals = [v for k, v in fan_in.items() if k[0] > 0]
    axes[0].hist(fi_vals, bins=range(max(fi_vals) + 2), color='steelblue',
                 edgecolor='white', alpha=0.8)
    axes[0].set_xlabel('Fan-in')
    axes[0].set_ylabel('Count')
    axes[0].set_title(f'Fan-in distribution (mean={np.mean(fi_vals):.1f})')

    # Fan-out distribution
    fo_vals = [v for k, v in fan_out.items() if k[0] < max(layers)]
    axes[1].hist(fo_vals, bins=range(max(fo_vals) + 2), color='#d62728',
                 edgecolor='white', alpha=0.8)
    axes[1].set_xlabel('Fan-out')
    axes[1].set_ylabel('Count')
    axes[1].set_title(f'Fan-out distribution (mean={np.mean(fo_vals):.1f})')

    # Edge weight distribution
    weights = [abs(e['weight']) for e in edges]
    axes[2].hist(weights, bins=50, color='#2ca02c', edgecolor='none', alpha=0.8)
    axes[2].set_xlabel('|weight|')
    axes[2].set_ylabel('Count')
    axes[2].set_title(f'Edge weight dist (n={len(edges)})')
    axes[2].set_yscale('log')

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()


# %% [markdown]
# ## Load and visualize
#
# Uncomment and run after sparse_rotation.py has been executed.

# %%
# model_name = "chi_net_L3_h256"
# results = load_sparse_rotation(model_name)
# print(f"Loaded {model_name}: loss {results['baseline_loss']:.4f} -> {results['final_loss']:.4f}")
# print(f"Accuracy: {results['acc_before']:.4f} -> {results['acc_after']:.4f}")
#
# nodes, edges = build_sparse_circuit(results['rotated_cores'])
# print(f"Circuit: {sum(len(v) for v in nodes.values())} nodes, {len(edges)} edges")
#
# plot_sparse_circuit(nodes, edges,
#     os.path.join(RESULTS_DIR, f'{model_name}_sparse_circuit.png'),
#     title=f'{model_name}: Sparse Circuit Graph')
#
# plot_circuit_stats(nodes, edges,
#     os.path.join(RESULTS_DIR, f'{model_name}_circuit_stats.png'))
