"""
Per-class ODT analysis and polynomial decomposition.

Implements the two research directions from RESEARCH_DIRECTIONS.md:
1. Per-class Gram propagation (class-specific rank and features)
2. Per-class degree decomposition (polynomial complexity per class)

Also checks forward cascade bounds to detect "phantom dimensions"
where backward ODT overestimates the true effective rank.
"""

import torch
import numpy as np
import matplotlib.pyplot as plt
import os
from model import ChiNet, ResidualBilinearNet
from odt import (
    orthogonalize, compute_gram_matrices, effective_bond_dim,
    run_odt, matricize_core
)


# ================================================================
# 1. Per-Class Gram Propagation
# ================================================================

def compute_per_class_grams(cores_orth, unembed):
    """
    Propagate per-class Gram matrices G_c = w_c w_c^T backward
    through the orthogonalized cores.

    Unlike the global ODT which uses G = W^T W (mixing all classes),
    this gives class-specific principal dimensions at each bond.

    Args:
        cores_orth: list of orthogonalized cores from ODT
        unembed: modified unembedding matrix (n_classes, h)

    Returns:
        per_class_eigenvalues: dict {class_idx: [evals_per_bond]}
        per_class_eigenvectors: dict {class_idx: [evecs_per_bond]}
    """
    n_classes = unembed.shape[0]
    results = {}

    for c in range(n_classes):
        w_c = unembed[c:c+1, :]  # (1, h)
        G_c = w_c.T @ w_c  # (h, h), rank-1

        eigenvalues_c = []
        eigenvectors_c = []

        for core in reversed(cores_orth):
            h_out, h_in, _ = core.shape

            # Eigendecompose current Gram
            evals, evecs = torch.linalg.eigh(G_c)
            idx = evals.argsort(descending=True)
            eigenvalues_c.append(evals[idx])
            eigenvectors_c.append(evecs[:, idx])

            # Propagate backward through this core
            Q = matricize_core(core)  # (h_out, h_in^2)
            F = Q.T @ G_c @ Q  # (h_in^2, h_in^2)
            F = F.reshape(h_in, h_in, h_in, h_in)
            G_c = torch.einsum('acbc->ab', F)  # cloning trace

        # Final Gram (at input bond)
        evals, evecs = torch.linalg.eigh(G_c)
        idx = evals.argsort(descending=True)
        eigenvalues_c.append(evals[idx])
        eigenvectors_c.append(evecs[:, idx])

        # Reverse so bond 0 = input, bond L = output
        eigenvalues_c.reverse()
        eigenvectors_c.reverse()

        results[c] = {
            'eigenvalues': eigenvalues_c,
            'eigenvectors': eigenvectors_c,
        }

    return results


def per_class_effective_ranks(per_class_results, threshold=1e-6):
    """
    Compute per-class effective rank at each bond.

    Returns: (n_classes, n_bonds) array of effective ranks.
    """
    n_classes = len(per_class_results)
    n_bonds = len(per_class_results[0]['eigenvalues'])
    ranks = np.zeros((n_classes, n_bonds), dtype=int)

    for c in range(n_classes):
        for b in range(n_bonds):
            evals = per_class_results[c]['eigenvalues'][b].clamp(min=0)
            if evals[0] > 0:
                ranks[c, b] = (evals > evals[0] * threshold).sum().item()
    return ranks


def cross_class_similarity(per_class_results, bond_idx, top_k=None):
    """
    Compute cross-class subspace similarity at a given bond.

    Uses the principal angles between the top-k eigenvector subspaces
    of each pair of classes. Returns cosine similarity matrix.

    Args:
        per_class_results: output of compute_per_class_grams
        bond_idx: which bond to analyze
        top_k: number of top eigenvectors to use (None = use effective rank)

    Returns:
        similarity: (n_classes, n_classes) cosine similarity matrix
    """
    n_classes = len(per_class_results)
    similarity = np.zeros((n_classes, n_classes))

    # Get subspace bases for each class
    bases = []
    for c in range(n_classes):
        evals = per_class_results[c]['eigenvalues'][bond_idx].clamp(min=0)
        evecs = per_class_results[c]['eigenvectors'][bond_idx]

        if top_k is None:
            if evals[0] > 0:
                k = max(1, (evals > evals[0] * 1e-6).sum().item())
            else:
                k = 1
        else:
            k = min(top_k, evecs.shape[1])

        bases.append(evecs[:, :k])  # (h, k)

    # Compute pairwise subspace similarity via principal angles
    for c1 in range(n_classes):
        for c2 in range(c1, n_classes):
            # Cosine of principal angles = singular values of B1^T @ B2
            cos_angles = torch.linalg.svdvals(bases[c1].T @ bases[c2])
            # Average cosine similarity
            sim = cos_angles.mean().item()
            similarity[c1, c2] = sim
            similarity[c2, c1] = sim

    return similarity


# ================================================================
# Forward Cascade Bound (Phantom Dimension Detection)
# ================================================================

def forward_cascade_bounds(global_eigenvalues, threshold=1e-6):
    """
    Compute the forward cascade bound on effective rank at each bond.

    Given ODT backward ranks r_0, r_1, ..., the forward cascade says:
    - Input produces r_0 features
    - After cloning + bilinear: at most r_0*(r_0+1)/2 features
    - Layer 1 core selects min(r_1, r_0*(r_0+1)/2)
    - After next cloning: at most that * (that+1)/2
    - etc.

    The true effective rank is min(backward_rank, forward_bound).
    Violations (backward > forward) indicate phantom dimensions.

    Args:
        global_eigenvalues: list of eigenvalue tensors from global ODT

    Returns:
        backward_ranks: list of backward (ODT) effective ranks
        forward_bounds: list of forward cascade upper bounds
        true_ranks: list of min(backward, forward)
        phantoms: list of (backward - true) phantom dimensions
    """
    backward_ranks = effective_bond_dim(global_eigenvalues, threshold)
    n_bonds = len(backward_ranks)

    forward_bounds = [backward_ranks[0]]  # bond 0: input rank
    for i in range(1, n_bonds):
        prev = forward_bounds[-1]
        # Cloning produces at most prev*(prev+1)/2 symmetric monomials
        max_from_cloning = prev * (prev + 1) // 2
        # But also bounded by the hidden dim at this bond
        forward_bounds.append(min(max_from_cloning, backward_ranks[i]))

    true_ranks = [min(b, f) for b, f in zip(backward_ranks, forward_bounds)]
    phantoms = [b - t for b, t in zip(backward_ranks, true_ranks)]

    return backward_ranks, forward_bounds, true_ranks, phantoms


# ================================================================
# 2. Per-Class Degree Decomposition
# ================================================================

@torch.no_grad()
def per_class_degree_spectrum(model, test_loader, device, max_degree=8,
                               n_epsilon=20):
    """
    Compute per-class degree spectrum by evaluating f(eps*x) for
    multiple epsilon values and fitting polynomial coefficients.

    For each class c, the logit f_c(eps*x) = sum_d alpha_d * eps^d.
    We fit alpha_d via least squares on the epsilon grid.

    Args:
        model: trained model (ChiNet or ResidualBilinearNet)
        test_loader: test data
        device: torch device
        max_degree: maximum polynomial degree to fit
        n_epsilon: number of epsilon values to use

    Returns:
        degree_power: (n_classes, max_degree+1) array of degree contributions
            Each entry is E[alpha_d^2 | label=c], the average squared
            coefficient for degree d, class c.
    """
    model.eval()
    model.to(device)

    epsilons = torch.linspace(0.01, 2.0, n_epsilon)
    n_classes = 10

    # Collect per-class, per-epsilon outputs
    # class_outputs[c][eps_idx] = list of logit values for class c
    class_logits = {c: {i: [] for i in range(n_epsilon)} for c in range(n_classes)}
    class_counts = {c: 0 for c in range(n_classes)}

    # Only need a few batches
    max_samples = 2000
    total = 0

    for x, y in test_loader:
        x = x.to(device)
        y = y.to(device)

        for eps_idx, eps in enumerate(epsilons):
            out = model(eps * x)  # (batch, n_classes)
            for c in range(n_classes):
                mask = (y == c)
                if mask.any():
                    class_logits[c][eps_idx].append(out[mask, c].cpu())

        total += x.shape[0]
        if total >= max_samples:
            break

    # Fit polynomial per class
    # For each class c and each sample, fit f_c(eps) = sum_d alpha_d * eps^d
    degree_power = np.zeros((n_classes, max_degree + 1))

    eps_np = epsilons.numpy()
    # Vandermonde matrix: V[i, d] = eps[i]^d
    V = np.vander(eps_np, N=max_degree + 1, increasing=True)

    for c in range(n_classes):
        # Concatenate all logits for this class
        per_eps_vals = []
        for eps_idx in range(n_epsilon):
            if class_logits[c][eps_idx]:
                vals = torch.cat(class_logits[c][eps_idx]).numpy()
                per_eps_vals.append(vals.mean())  # average over samples
            else:
                per_eps_vals.append(0.0)

        per_eps_vals = np.array(per_eps_vals)

        # Least squares fit: V @ alpha = per_eps_vals
        alpha, _, _, _ = np.linalg.lstsq(V, per_eps_vals, rcond=None)

        # Power at each degree = alpha_d^2
        degree_power[c] = alpha ** 2

    return degree_power


def per_class_interaction_ranks(odt_results, n_classes=10):
    """
    Compute per-class interaction matrix rank at each layer.

    At each layer, contract the unembedding weight (projected to truncated
    basis) with the core to get a per-class interaction matrix M_c.
    The rank of M_c tells how many independent quadratic features
    class c uses at that layer.

    Args:
        odt_results: output of run_odt (with orthogonalized cores)

    Returns:
        interaction_ranks: (n_classes, n_layers) array
    """
    cores = odt_results['cores_orth']
    unembed = odt_results['unembed_mod']
    n_layers = len(cores)

    ranks = np.zeros((n_classes, n_layers), dtype=int)

    for c in range(n_classes):
        # Start from the output weight for class c
        w = unembed[c]  # (h,)

        # Work backward through layers
        for layer_idx in range(n_layers - 1, -1, -1):
            core = cores[layer_idx]  # (h_out, h_in, h_in)

            # Contract w with core's output dimension to get interaction matrix
            M_c = torch.einsum('o,oij->ij', w, core)  # (h_in, h_in)
            M_c = 0.5 * (M_c + M_c.T)  # symmetrize

            # Rank of M_c = number of significant eigenvalues
            evals = torch.linalg.eigvalsh(M_c)
            evals_abs = evals.abs()
            if evals_abs.max() > 0:
                ranks[c, layer_idx] = (evals_abs > evals_abs.max() * 1e-6).sum().item()

            # For next iteration, w becomes the top eigenvector of M_c
            # (the direction that contributes most to class c)
            _, evecs = torch.linalg.eigh(M_c)
            idx = evals.abs().argsort(descending=True)[0]
            w = evecs[:, idx]

    return ranks


# ================================================================
# Dominant Monomial Path Visualization
# ================================================================

def trace_dominant_paths(odt_results, embed, n_classes=10,
                         img_shape=(32, 32)):
    """
    Trace the dominant monomial path from output to input for each class.

    At each layer, find the top eigenvector of the per-class interaction
    matrix, and trace it backward through the tree to input space.

    Returns:
        paths: dict {class_idx: input_pattern (as image)}
    """
    cores = odt_results['cores_orth']
    unembed = odt_results['unembed_mod']
    n_layers = len(cores)

    paths = {}

    for c in range(n_classes):
        w = unembed[c]  # (h,)
        directions = []

        # Backward through layers: get dominant direction at each bond
        for layer_idx in range(n_layers - 1, -1, -1):
            core = cores[layer_idx]
            M_c = torch.einsum('o,oij->ij', w, core)
            M_c = 0.5 * (M_c + M_c.T)

            evals, evecs = torch.linalg.eigh(M_c)
            idx = evals.abs().argsort(descending=True)[0]
            top_evec = evecs[:, idx]
            top_eval = evals[idx]

            directions.append((top_evec, top_eval))
            w = top_evec  # pass through to next layer

        # Final direction is in the embedding space
        # Project through embedding to input space
        input_pattern = embed.T @ w  # (input+1,)
        input_pattern = input_pattern[1:]  # skip constant
        if input_pattern.numel() >= img_shape[0] * img_shape[1]:
            input_pattern = input_pattern[:img_shape[0]*img_shape[1]]
            input_pattern = input_pattern.reshape(img_shape)

        paths[c] = input_pattern.detach()

    return paths


# ================================================================
# Plotting Functions
# ================================================================

def plot_per_class_ranks(ranks, model_name, save_dir='images'):
    """Plot per-class effective rank table as heatmap."""
    os.makedirs(save_dir, exist_ok=True)

    n_classes, n_bonds = ranks.shape
    fig, ax = plt.subplots(figsize=(max(6, n_bonds * 1.5), 6))

    im = ax.imshow(ranks, cmap='YlOrRd', aspect='auto')
    ax.set_xlabel('Bond')
    ax.set_ylabel('Class')
    ax.set_xticks(range(n_bonds))
    ax.set_xticklabels([f'Bond {i}' for i in range(n_bonds)])
    ax.set_yticks(range(n_classes))
    ax.set_yticklabels([str(i) for i in range(n_classes)])

    # Annotate cells
    for i in range(n_classes):
        for j in range(n_bonds):
            ax.text(j, i, str(ranks[i, j]), ha='center', va='center',
                    fontsize=9, color='black' if ranks[i, j] < ranks.max() * 0.7 else 'white')

    plt.colorbar(im, ax=ax, label='Effective rank')
    ax.set_title(f'{model_name}: Per-class effective rank at each bond')
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, f'{model_name}_per_class_ranks.png'), dpi=150)
    plt.close()
    print(f"Saved per-class ranks")


def plot_cross_class_similarity(per_class_results, model_name,
                                 save_dir='images'):
    """Plot cross-class similarity heatmaps at each bond."""
    os.makedirs(save_dir, exist_ok=True)

    n_bonds = len(per_class_results[0]['eigenvalues'])

    fig, axes = plt.subplots(1, n_bonds, figsize=(4 * n_bonds + 1, 4.5))
    if n_bonds == 1:
        axes = [axes]

    for b in range(n_bonds):
        sim = cross_class_similarity(per_class_results, b)
        ax = axes[b]
        im = ax.imshow(sim, cmap='RdBu_r', vmin=0, vmax=1)
        ax.set_title(f'Bond {b}', fontsize=10)
        ax.set_xticks(range(10))
        ax.set_yticks(range(10))
        ax.set_xticklabels(range(10), fontsize=7)
        ax.set_yticklabels(range(10), fontsize=7)
        if b == 0:
            ax.set_ylabel('Class')
        ax.set_xlabel('Class')

    fig.colorbar(im, ax=axes, label='Subspace similarity', shrink=0.8,
                 pad=0.02)
    fig.suptitle(f'{model_name}: Cross-class subspace similarity', fontsize=12,
                 y=1.02)
    fig.savefig(os.path.join(save_dir, f'{model_name}_cross_class_sim.png'),
                dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved cross-class similarity")


def plot_phantom_dimensions(backward_ranks, forward_bounds, true_ranks,
                             phantoms, model_name, save_dir='images'):
    """Visualize forward vs backward ranks and phantom dimensions."""
    os.makedirs(save_dir, exist_ok=True)

    n_bonds = len(backward_ranks)
    x = np.arange(n_bonds)

    fig, ax = plt.subplots(figsize=(8, 5))
    width = 0.25
    ax.bar(x - width, backward_ranks, width, label='Backward (ODT)', color='steelblue')
    ax.bar(x, forward_bounds, width, label='Forward bound', color='coral')
    ax.bar(x + width, true_ranks, width, label='True (min)', color='forestgreen')

    # Annotate phantoms
    for i, p in enumerate(phantoms):
        if p > 0:
            ax.annotate(f'+{p} phantom', (i - width, backward_ranks[i]),
                       textcoords="offset points", xytext=(0, 5),
                       ha='center', fontsize=8, color='red')

    ax.set_xlabel('Bond')
    ax.set_ylabel('Effective rank')
    ax.set_xticks(x)
    ax.set_xticklabels([f'Bond {i}' for i in range(n_bonds)])
    ax.legend()
    ax.set_title(f'{model_name}: Forward vs backward rank bounds')
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, f'{model_name}_phantom_dims.png'), dpi=150)
    plt.close()
    print(f"Saved phantom dimension analysis")


def plot_degree_spectrum(degree_power, model_name, save_dir='images'):
    """Plot per-class degree spectrum as heatmap."""
    os.makedirs(save_dir, exist_ok=True)

    n_classes, n_degrees = degree_power.shape

    # Normalize per class (fraction of total power)
    row_sums = degree_power.sum(axis=1, keepdims=True)
    row_sums[row_sums == 0] = 1
    degree_frac = degree_power / row_sums

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Raw power (log scale)
    ax = axes[0]
    im = ax.imshow(np.log10(degree_power + 1e-20), cmap='viridis', aspect='auto')
    ax.set_xlabel('Degree')
    ax.set_ylabel('Class')
    ax.set_xticks(range(n_degrees))
    ax.set_yticks(range(n_classes))
    ax.set_yticklabels(range(n_classes))
    ax.set_title('Log power by degree')
    plt.colorbar(im, ax=ax, label='log10(power)')

    # Fraction of power
    ax = axes[1]
    im = ax.imshow(degree_frac, cmap='YlOrRd', aspect='auto', vmin=0, vmax=1)
    ax.set_xlabel('Degree')
    ax.set_ylabel('Class')
    ax.set_xticks(range(n_degrees))
    ax.set_yticks(range(n_classes))
    ax.set_yticklabels(range(n_classes))
    ax.set_title('Fraction of power by degree')
    plt.colorbar(im, ax=ax, label='Fraction')

    fig.suptitle(f'{model_name}: Per-class degree spectrum', fontsize=12)
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, f'{model_name}_degree_spectrum.png'), dpi=150)
    plt.close()
    print(f"Saved degree spectrum")


def plot_dominant_paths(paths, model_name, img_shape=(32, 32),
                        save_dir='images'):
    """Visualize dominant monomial paths for each class."""
    os.makedirs(save_dir, exist_ok=True)

    fig, axes = plt.subplots(2, 5, figsize=(12, 5))
    for c in range(10):
        ax = axes[c // 5, c % 5]
        pattern = paths[c]
        vmax = pattern.abs().max()
        if vmax > 0:
            ax.imshow(pattern.numpy(), cmap='RdBu_r', vmin=-vmax, vmax=vmax)
        else:
            ax.imshow(pattern.numpy(), cmap='RdBu_r')
        ax.set_title(f'Class {c}', fontsize=10)
        ax.axis('off')

    fig.suptitle(f'{model_name}: Dominant monomial paths (traced to input)',
                 fontsize=12)
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, f'{model_name}_dominant_paths.png'), dpi=150)
    plt.close()
    print(f"Saved dominant monomial paths")


def plot_interaction_ranks(ranks, model_name, save_dir='images'):
    """Plot per-class interaction rank at each layer."""
    os.makedirs(save_dir, exist_ok=True)

    n_classes, n_layers = ranks.shape
    fig, ax = plt.subplots(figsize=(max(6, n_layers * 2), 5))

    im = ax.imshow(ranks, cmap='YlOrRd', aspect='auto')
    ax.set_xlabel('Layer')
    ax.set_ylabel('Class')
    ax.set_xticks(range(n_layers))
    ax.set_xticklabels([f'Layer {i}' for i in range(n_layers)])
    ax.set_yticks(range(n_classes))
    ax.set_yticklabels(range(n_classes))

    for i in range(n_classes):
        for j in range(n_layers):
            ax.text(j, i, str(ranks[i, j]), ha='center', va='center',
                    fontsize=9, color='black' if ranks[i, j] < ranks.max() * 0.7 else 'white')

    plt.colorbar(im, ax=ax, label='Interaction rank')
    ax.set_title(f'{model_name}: Per-class interaction rank at each layer')
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, f'{model_name}_interaction_ranks.png'), dpi=150)
    plt.close()
    print(f"Saved interaction ranks")


# ================================================================
# Main Analysis Pipeline
# ================================================================

def run_per_class_analysis(checkpoint_path, test_loader=None, device='cpu'):
    """Run the full per-class analysis on a single model."""
    from analyze import load_model

    model, ckpt = load_model(checkpoint_path, device)
    model_name = f"{ckpt['model_type']}_L{ckpt['n_layers']}_h{ckpt['hidden_dim']}"
    model.to(device)

    print(f"\n{'='*60}")
    print(f"Per-class analysis: {model_name}")
    print(f"{'='*60}")

    # Extract cores and run global ODT
    if isinstance(model, ResidualBilinearNet):
        embed, cores, unembed = model.cpu().get_augmented_cores_for_odt()
        print(f"Using augmented cores (dim {embed.shape[0]})")
    else:
        embed, cores, unembed = model.cpu().get_cores_for_odt()

    odt_results = run_odt(embed, cores, unembed, verbose=False)

    # --- 1. Per-class Gram propagation ---
    print("\n--- Per-class Gram propagation ---")
    per_class = compute_per_class_grams(
        odt_results['cores_orth'], odt_results['unembed_mod'])

    # Per-class ranks
    ranks = per_class_effective_ranks(per_class)
    print(f"\nPer-class effective ranks (threshold=1e-6):")
    print(f"{'Class':<8}" + "".join(f"{'Bond '+str(b):<10}" for b in range(ranks.shape[1])))
    print("-" * (8 + 10 * ranks.shape[1]))
    for c in range(10):
        print(f"{c:<8}" + "".join(f"{ranks[c, b]:<10}" for b in range(ranks.shape[1])))

    # Sum of per-class ranks vs global rank
    global_ranks = effective_bond_dim(odt_results['eigenvalues'])
    rank_sum = ranks.sum(axis=0)
    print(f"\n{'Sum':<8}" + "".join(f"{rank_sum[b]:<10}" for b in range(ranks.shape[1])))
    print(f"{'Global':<8}" + "".join(f"{global_ranks[b]:<10}" for b in range(len(global_ranks))))
    print(f"{'Sharing':<8}" + "".join(
        f"{1 - global_ranks[b]/max(1,rank_sum[b]):<10.2f}"
        for b in range(min(len(global_ranks), ranks.shape[1]))))

    plot_per_class_ranks(ranks, model_name)
    plot_cross_class_similarity(per_class, model_name)

    # --- 2. Phantom dimension check ---
    print("\n--- Forward cascade bound (phantom dimensions) ---")
    backward, forward, true, phantoms = forward_cascade_bounds(
        odt_results['eigenvalues'])

    print(f"{'Bond':<8} {'Backward':<10} {'Forward':<10} {'True':<10} {'Phantom':<10}")
    print("-" * 48)
    for i in range(len(backward)):
        flag = " <<<" if phantoms[i] > 0 else ""
        print(f"Bond {i:<3} {backward[i]:<10} {forward[i]:<10} {true[i]:<10} {phantoms[i]:<10}{flag}")

    plot_phantom_dimensions(backward, forward, true, phantoms, model_name)

    # --- 3. Per-class interaction ranks ---
    print("\n--- Per-class interaction ranks ---")
    int_ranks = per_class_interaction_ranks(odt_results)
    print(f"{'Class':<8}" + "".join(f"{'Layer '+str(l):<10}" for l in range(int_ranks.shape[1])))
    print("-" * (8 + 10 * int_ranks.shape[1]))
    for c in range(10):
        print(f"{c:<8}" + "".join(f"{int_ranks[c, l]:<10}" for l in range(int_ranks.shape[1])))

    plot_interaction_ranks(int_ranks, model_name)

    # --- 4. Dominant monomial paths ---
    print("\n--- Dominant monomial paths ---")
    paths = trace_dominant_paths(odt_results, odt_results['embed_orth'])
    plot_dominant_paths(paths, model_name)

    # --- 5. Per-class degree spectrum ---
    if test_loader is not None:
        print("\n--- Per-class degree spectrum ---")
        model.to(device)
        degree_power = per_class_degree_spectrum(model, test_loader, device)

        # Normalize and print dominant degree per class
        row_sums = degree_power.sum(axis=1, keepdims=True)
        row_sums[row_sums == 0] = 1
        degree_frac = degree_power / row_sums

        print(f"{'Class':<8} {'Dom. deg':<10} {'Frac':<10}")
        print("-" * 28)
        for c in range(10):
            dom_deg = np.argmax(degree_frac[c])
            dom_frac = degree_frac[c, dom_deg]
            print(f"{c:<8} {dom_deg:<10} {dom_frac:<10.3f}")

        plot_degree_spectrum(degree_power, model_name)

    print(f"\n{'='*60}")
    print(f"Per-class analysis complete for {model_name}")
    print(f"{'='*60}")

    return {
        'per_class': per_class,
        'ranks': ranks,
        'phantom': (backward, forward, true, phantoms),
        'interaction_ranks': int_ranks,
        'paths': paths,
        'odt_results': odt_results,
    }


def compare_per_class(results_a, results_b, name_a, name_b, save_dir='images'):
    """
    Side-by-side comparison of per-class results for two models.
    Generates:
    1. Dominant paths comparison (2 rows of 10)
    2. Degree spectrum comparison (2 rows)
    3. Interaction rank comparison (2 rows)
    """
    os.makedirs(save_dir, exist_ok=True)

    # --- Dominant paths side-by-side ---
    fig, axes = plt.subplots(2, 10, figsize=(20, 5))
    for row, (results, name) in enumerate([(results_a, name_a), (results_b, name_b)]):
        for c in range(10):
            ax = axes[row, c]
            pattern = results['paths'][c]
            vmax = pattern.abs().max()
            if vmax > 0:
                ax.imshow(pattern.numpy(), cmap='RdBu_r', vmin=-vmax, vmax=vmax)
            else:
                ax.imshow(pattern.numpy(), cmap='RdBu_r')
            ax.axis('off')
            if row == 0:
                ax.set_title(f'{c}', fontsize=10)
        axes[row, 0].set_ylabel(name, fontsize=9, rotation=0, labelpad=50,
                                 va='center')
    fig.suptitle('Dominant monomial paths: chi_net vs residual', fontsize=13)
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, 'comparison_dominant_paths.png'),
                dpi=150, bbox_inches='tight')
    plt.close()
    print("Saved comparison_dominant_paths.png")

    # --- Interaction rank comparison ---
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    for ax, (results, name) in zip(axes, [(results_a, name_a), (results_b, name_b)]):
        ranks = results['interaction_ranks']
        n_classes, n_layers = ranks.shape
        im = ax.imshow(ranks, cmap='YlOrRd', aspect='auto')
        ax.set_xlabel('Layer')
        ax.set_ylabel('Class')
        ax.set_xticks(range(n_layers))
        ax.set_xticklabels([f'L{i}' for i in range(n_layers)])
        ax.set_yticks(range(n_classes))
        ax.set_yticklabels(range(n_classes))
        for i in range(n_classes):
            for j in range(n_layers):
                ax.text(j, i, str(ranks[i, j]), ha='center', va='center',
                        fontsize=8,
                        color='black' if ranks[i, j] < ranks.max() * 0.7 else 'white')
        ax.set_title(name)
    fig.colorbar(im, ax=axes, label='Interaction rank', shrink=0.8)
    fig.suptitle('Per-class interaction ranks comparison', fontsize=13)
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, 'comparison_interaction_ranks.png'),
                dpi=150, bbox_inches='tight')
    plt.close()
    print("Saved comparison_interaction_ranks.png")

    # --- Per-class rank comparison ---
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    for ax, (results, name) in zip(axes, [(results_a, name_a), (results_b, name_b)]):
        ranks = results['ranks']
        n_classes, n_bonds = ranks.shape
        im = ax.imshow(ranks, cmap='YlOrRd', aspect='auto')
        ax.set_xlabel('Bond')
        ax.set_ylabel('Class')
        ax.set_xticks(range(n_bonds))
        ax.set_xticklabels([f'B{i}' for i in range(n_bonds)])
        ax.set_yticks(range(n_classes))
        ax.set_yticklabels(range(n_classes))
        for i in range(n_classes):
            for j in range(n_bonds):
                ax.text(j, i, str(ranks[i, j]), ha='center', va='center',
                        fontsize=8,
                        color='black' if ranks[i, j] < ranks.max() * 0.7 else 'white')
        ax.set_title(name)
    fig.colorbar(im, ax=axes, label='Effective rank', shrink=0.8)
    fig.suptitle('Per-class effective rank comparison', fontsize=13)
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, 'comparison_per_class_ranks.png'),
                dpi=150, bbox_inches='tight')
    plt.close()
    print("Saved comparison_per_class_ranks.png")


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--checkpoint', type=str, default=None)
    parser.add_argument('--compare', action='store_true',
                        help='Compare chi_net and residual models')
    parser.add_argument('--device', type=str, default='cpu')
    parser.add_argument('--with_data', action='store_true')
    parser.add_argument('--data_root', type=str, default='../modular_addition/dev_interp/notebooks/data')
    args = parser.parse_args()

    test_loader = None
    if args.with_data:
        from train import get_svhn_loaders
        _, test_loader, _ = get_svhn_loaders(data_root=args.data_root)

    if args.compare:
        res_chi = run_per_class_analysis(
            'models/chi_net_L3_h256.pt', test_loader, args.device)
        res_res = run_per_class_analysis(
            'models/residual_L3_h256.pt', test_loader, args.device)
        compare_per_class(res_chi, res_res, 'chi_net', 'residual')
    elif args.checkpoint:
        run_per_class_analysis(args.checkpoint, test_loader, args.device)
    else:
        print("Provide --checkpoint PATH or --compare")
