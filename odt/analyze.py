"""
Analysis script: run ODT on trained models, compare spectra, visualize.

Usage:
    python analyze.py --checkpoint checkpoints/chi_net_L3_h256.pt
    python analyze.py --checkpoint checkpoints/chi_net_L3_h256.pt checkpoints/residual_L3_h256.pt --compare

Produces:
    1. SVD vs ODT singular value comparison (replicates Figure 7)
    2. Truncation curve: accuracy vs dimensions removed (replicates Figure 2)
    3. Atom visualization (replicates Figure 3)
    4. Per-digit eigenvector visualization (replicates Figure 4)
    5. Spectral gap analysis for residual models
"""

import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
import argparse
import os
from model import ChiNet, ResidualBilinearNet
from odt import (
    run_odt, naive_svd_bond_dims, orthogonalize,
    compute_gram_matrices, effective_bond_dim, truncate
)


def load_model(checkpoint_path, device='cpu'):
    """Load a trained model from checkpoint."""
    ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)

    model_type = ckpt['model_type']
    n_layers = ckpt['n_layers']
    hidden_dim = ckpt['hidden_dim']
    input_dim = ckpt['input_dim']
    output_dim = ckpt['output_dim']

    if model_type == 'chi_net':
        model = ChiNet(input_dim, hidden_dim, output_dim, n_layers)
    else:
        model = ResidualBilinearNet(input_dim, hidden_dim, output_dim, n_layers)

    model.load_state_dict(ckpt['model_state_dict'])
    model.eval()

    print(f"Loaded {model_type} (L={n_layers}, h={hidden_dim}), "
          f"test acc={ckpt.get('test_acc', 'N/A'):.4f}")

    return model, ckpt


def extract_cores(model, use_augmented=True):
    """Extract embedding, cores, unembedding from a trained model.

    For residual models with use_augmented=True, returns augmented cores
    of shape (H+1, H+1, H+1) that correctly fold the residual path
    via the constant/discard trick.
    """
    if isinstance(model, ChiNet):
        return model.get_cores_for_odt()
    elif use_augmented:
        return model.get_augmented_cores_for_odt()
    else:
        return model.get_cores_for_odt(fold_residual=False)


# ================================================================
# Plot 1: SVD vs ODT singular value spectra (Figure 7)
# ================================================================

def plot_svd_vs_odt(odt_results, model_name, save_dir='images'):
    """
    Compare naive SVD and ODT singular value spectra.

    Replicates Figure 7 from the paper: SVD shows no clear low-rank
    structure, while ODT reveals sharp rank drops.
    """
    os.makedirs(save_dir, exist_ok=True)

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # --- Naive SVD ---
    ax = axes[0]
    for name, sv in odt_results['naive_svd'].items():
        sv_norm = sv / sv[0]
        ax.plot(sv_norm.detach().numpy(), label=name, alpha=0.7)
    ax.set_xlabel('Dimension')
    ax.set_ylabel('Normalized singular value')
    ax.set_title('Naive per-layer SVD')
    ax.legend(fontsize=7)
    ax.set_ylim(-0.05, 1.05)

    # --- ODT Gram eigenvalues ---
    ax = axes[1]
    for i, evals in enumerate(odt_results['eigenvalues']):
        evals_pos = evals.clamp(min=0)
        if evals_pos[0] > 0:
            ev_norm = evals_pos / evals_pos[0]
        else:
            ev_norm = evals_pos
        ax.plot(ev_norm.detach().numpy(), label=f'Bond {i}', alpha=0.7)
    ax.set_xlabel('Dimension')
    ax.set_ylabel('Normalized eigenvalue')
    ax.set_title('ODT Gram eigenvalues')
    ax.legend()
    ax.set_ylim(-0.05, 1.05)

    fig.suptitle(f'{model_name}: SVD vs ODT', fontsize=14)
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, f'{model_name}_svd_vs_odt.png'), dpi=150)
    plt.close()
    print(f"Saved SVD vs ODT plot")


# ================================================================
# Plot 2: Truncation curve (Figure 2)
# ================================================================

def plot_truncation_curve(model, odt_results, test_loader, device,
                          model_name, save_dir='images'):
    """
    Plot accuracy as a function of dimensions removed.

    Replicates Figure 2: ~70% of dimensions can be removed without
    accuracy loss.
    """
    os.makedirs(save_dir, exist_ok=True)

    embed_orth = odt_results['embed_orth']
    cores_orth = odt_results['cores_orth']
    unembed_mod = odt_results['unembed_mod']
    eigvecs = odt_results['eigenvectors']
    eigenvalues = odt_results['eigenvalues']

    h = embed_orth.shape[0]
    n_bonds = len(eigenvalues)

    # Try different truncation levels
    fractions = np.linspace(0, 0.98, 30)
    accuracies = []

    for frac in fractions:
        # Compute ranks: keep (1-frac) of dimensions at each bond
        ranks = []
        for evals in eigenvalues:
            evals_pos = evals.clamp(min=0)
            max_rank = (evals_pos > evals_pos[0] * 1e-10).sum().item()
            r = max(1, int(max_rank * (1 - frac)))
            ranks.append(r)

        # Truncate
        e_trunc, c_trunc, u_trunc = truncate(
            embed_orth, cores_orth, unembed_mod, eigvecs, ranks)

        # Evaluate truncated model
        acc = evaluate_truncated(e_trunc, c_trunc, u_trunc,
                                 model, test_loader, device)
        accuracies.append(acc)

        print(f"  Frac removed: {frac:.2f}, ranks: {ranks}, acc: {acc:.4f}")

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(fractions * 100, [a * 100 for a in accuracies], 'b-o', markersize=3)
    ax.set_xlabel('Dimensions removed (%)')
    ax.set_ylabel('Accuracy (%)')
    ax.set_title(f'{model_name}: Truncation curve')
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, f'{model_name}_truncation.png'), dpi=150)
    plt.close()
    print(f"Saved truncation curve")


@torch.no_grad()
def evaluate_truncated(embed_trunc, cores_trunc, unembed_trunc,
                       original_model, test_loader, device):
    """
    Evaluate a truncated model by running the χ-net forward pass
    with the modified weights.
    """
    total_correct = 0
    total_samples = 0

    embed_trunc = embed_trunc.to(device)
    cores_trunc = [c.to(device) for c in cores_trunc]
    unembed_trunc = unembed_trunc.to(device)

    for x, y in test_loader:
        x = x.to(device)
        y = y.to(device)

        # Append constant
        ones = torch.ones(x.shape[0], 1, device=device)
        h = torch.cat([ones, x], dim=-1)

        # Embedding
        h = h @ embed_trunc.T

        # Cores (bilinear with cloning)
        for core in cores_trunc:
            # core shape: (h_out, h_in, h_in)
            # h shape: (batch, h_in)
            # out = einsum('oij, bi, bj -> bo', core, h, h)
            h = torch.einsum('oij,bi,bj->bo', core, h, h)

        # Unembedding
        logits = h @ unembed_trunc.T

        total_correct += (logits.argmax(dim=-1) == y).sum().item()
        total_samples += x.size(0)

    return total_correct / total_samples


# ================================================================
# Plot 3: Atom visualization (Figure 3)
# ================================================================

def plot_atoms(odt_results, model_name, n_atoms=8,
               img_shape=(32, 32), save_dir='images'):
    """
    Visualize the most important atoms (embedding rows) and
    interaction matrices after ODT.

    Replicates Figure 3: atoms show edge detectors and proto-digits,
    interaction matrices are sparse (especially early layers).
    """
    os.makedirs(save_dir, exist_ok=True)

    eigenvalues = odt_results['eigenvalues']
    eigvecs = odt_results['eigenvectors']
    embed = odt_results['embed_orth']

    # Atoms are rows of the (projected) embedding.
    # Project embed into the diagonalized basis.
    V0 = eigvecs[0]  # eigenvectors at bond 0 (embed -> core 0)
    embed_proj = V0.T @ embed  # (h, input+1)

    # Sort atoms by importance (eigenvalue magnitude)
    evals = eigenvalues[0].clamp(min=0)
    top_idx = evals.argsort(descending=True)[:n_atoms]

    fig, axes = plt.subplots(2, n_atoms // 2, figsize=(n_atoms * 1.5, 6))
    axes = axes.flatten()

    for idx, atom_idx in enumerate(top_idx):
        atom = embed_proj[atom_idx, 1:]  # skip the constant (index 0)
        atom = atom.reshape(img_shape)
        vmax = atom.abs().max()

        axes[idx].imshow(atom.detach().numpy(), cmap='RdBu_r', vmin=-vmax, vmax=vmax)
        axes[idx].set_title(f'Atom {atom_idx}\nλ={evals[atom_idx]:.2f}', fontsize=8)
        axes[idx].axis('off')

    fig.suptitle(f'{model_name}: Top atoms (embedding rows)', fontsize=12)
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, f'{model_name}_atoms.png'), dpi=150)
    plt.close()
    print(f"Saved atom visualization")


# ================================================================
# Plot 4: Per-digit eigenvectors (Figure 4)
# ================================================================

def plot_digit_eigenvectors(odt_results, model_name,
                            img_shape=(32, 32), save_dir='images'):
    """
    Extract and visualize the most important eigenvector per digit class.

    Following the paper: take the root core (contracted with unembedding),
    compute eigenvectors for each output class, and trace them linearly
    through the previous layers onto the input space.

    This works because the first layers are near-linear (ODT finding).
    """
    os.makedirs(save_dir, exist_ok=True)

    embed = odt_results['embed_orth']
    cores = odt_results['cores_orth']
    unembed = odt_results['unembed_mod']
    eigenvalues = odt_results['eigenvalues']
    eigvecs = odt_results['eigenvectors']

    # Contract unembedding into the last core
    # last_core shape: (h, h, h), unembed shape: (10, h)
    # root = einsum('ij, jkl -> ikl', unembed, last_core)
    root = torch.einsum('ij,jkl->ikl', unembed, cores[-1])
    # root shape: (10, h, h) — one (h,h) matrix per output class

    # For each digit, get the dominant eigenvector of its interaction matrix
    fig, axes = plt.subplots(2, 5, figsize=(12, 5))

    for digit in range(10):
        interaction = root[digit]  # (h, h)
        # Symmetrize
        interaction = 0.5 * (interaction + interaction.T)

        evals, evecs = torch.linalg.eigh(interaction)
        # Top eigenvector (largest magnitude eigenvalue)
        idx = evals.abs().argsort(descending=True)[0]
        top_evec = evecs[:, idx]  # (h,)

        # Trace through previous layers linearly
        # Since early layers are near-linear, the dominant path is through
        # the constant (bias) entries of the interaction matrices.
        # The "linear trace" is: v -> cores[L-2][:,:,0] @ v (constant column)
        # then repeat for each layer down to the embedding.
        v = top_evec
        for core in reversed(cores[:-1]):
            # The "linear" part of the bilinear core is the slice where
            # one input is the constant (index 0):
            # core[:,j,0] maps h_j -> output via linear path
            linear_map = core[:, :, 0] + core[:, 0, :]  # both ways, symmetrized
            linear_map = 0.5 * linear_map  # avoid double-counting diagonal
            v = linear_map.T @ v

        # Project through embedding to input space
        input_pattern = embed.T @ v  # (input+1,)
        input_pattern = input_pattern[1:]  # skip constant
        input_pattern = input_pattern.reshape(img_shape)

        ax = axes[digit // 5, digit % 5]
        vmax = input_pattern.abs().max()
        ax.imshow(input_pattern.detach().numpy(), cmap='RdBu_r', vmin=-vmax, vmax=vmax)
        ax.set_title(f'Digit {digit}', fontsize=10)
        ax.axis('off')

    fig.suptitle(f'{model_name}: Per-digit eigenvectors (traced to input)',
                 fontsize=12)
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, f'{model_name}_digits.png'), dpi=150)
    plt.close()
    print(f"Saved digit eigenvector visualization")


# ================================================================
# Plot 5: Spectral gap analysis (residual models)
# ================================================================

def plot_spectral_gap(odt_results_chi, odt_results_res,
                      save_dir='images'):
    """
    Compare Gram eigenvalue spectra between χ-net and residual model.

    Key question: does the residual model show a spectral gap
    separating the full-rank residual subspace from the low-rank
    bilinear corrections?
    """
    os.makedirs(save_dir, exist_ok=True)

    n_bonds = len(odt_results_chi['eigenvalues'])

    fig, axes = plt.subplots(2, n_bonds, figsize=(4 * n_bonds, 8))

    for i in range(n_bonds):
        # χ-net
        evals_chi = odt_results_chi['eigenvalues'][i].clamp(min=0)
        ax = axes[0, i]
        ax.semilogy(evals_chi.detach().numpy() + 1e-20, 'g-', alpha=0.8)
        ax.set_title(f'Bond {i} (χ-net)', fontsize=9)
        ax.set_xlabel('Index')
        if i == 0:
            ax.set_ylabel('Eigenvalue (log scale)')
        ax.grid(True, alpha=0.3)

        # Residual
        evals_res = odt_results_res['eigenvalues'][i].clamp(min=0)
        ax = axes[1, i]
        ax.semilogy(evals_res.detach().numpy() + 1e-20, 'r-', alpha=0.8)
        ax.set_title(f'Bond {i} (residual)', fontsize=9)
        ax.set_xlabel('Index')
        if i == 0:
            ax.set_ylabel('Eigenvalue (log scale)')
        ax.grid(True, alpha=0.3)

    fig.suptitle('Spectral gap analysis: χ-net vs residual', fontsize=14)
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, 'spectral_gap_comparison.png'), dpi=150)
    plt.close()
    print(f"Saved spectral gap comparison")


# ================================================================
# Degree decomposition analysis
# ================================================================

def degree_analysis(model, test_loader, device, model_name,
                    save_dir='images'):
    """
    Estimate the norm contribution of each polynomial degree.

    Method: evaluate f(ε*x) for multiple ε values, fit polynomial
    in ε to separate degree components.

    For a 3-layer χ-net (no residual), the function is degree 2^3 = 8.
    For a residual model, the function has degrees 2 through 8.
    """
    os.makedirs(save_dir, exist_ok=True)
    model.eval()

    # Collect outputs at different scales
    epsilons = torch.linspace(0.01, 2.0, 20)
    output_norms = {eps.item(): [] for eps in epsilons}

    with torch.no_grad():
        for x, y in test_loader:
            x = x.to(device)
            for eps in epsilons:
                out = model(eps * x)
                output_norms[eps.item()].append(out.norm(dim=-1).mean().item())
            break  # one batch is enough

    # For each epsilon, average the output norms
    eps_vals = np.array([e for e in epsilons.numpy()])
    norm_vals = np.array([np.mean(output_norms[e]) for e in epsilons.tolist()])

    # Fit polynomial: log(||f(eps*x)||) ≈ k * log(eps) + const
    # to find the dominant degree k
    log_eps = np.log(eps_vals[eps_vals > 0.05])
    log_norm = np.log(norm_vals[eps_vals > 0.05] + 1e-20)

    # Fit in segments to see if degree changes
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.loglog(eps_vals, norm_vals, 'b-o', markersize=4)

    # Fit overall slope
    coeffs = np.polyfit(log_eps, log_norm, 1)
    ax.set_xlabel('ε (input scale)')
    ax.set_ylabel('||f(ε·x)|| (output norm)')
    ax.set_title(f'{model_name}: Degree analysis\n'
                 f'Dominant degree ≈ {coeffs[0]:.2f}')
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, f'{model_name}_degree.png'), dpi=150)
    plt.close()
    print(f"Saved degree analysis (dominant degree ≈ {coeffs[0]:.2f})")


# ================================================================
# Main analysis pipeline
# ================================================================

def analyze_model(checkpoint_path, test_loader=None, device='cpu',
                  use_augmented=True):
    """Run full analysis on a single model."""
    model, ckpt = load_model(checkpoint_path, device)
    model.to(device)
    model_name = f"{ckpt['model_type']}_L{ckpt['n_layers']}_h{ckpt['hidden_dim']}"

    # Extract cores (on CPU for ODT computation)
    if isinstance(model, ResidualBilinearNet):
        if use_augmented:
            embed, cores, unembed = model.cpu().get_augmented_cores_for_odt()
            print(f"\nResidual model, using augmented cores (H+1 = {embed.shape[0]})")
        else:
            embed, cores, unembed = model.cpu().get_cores_for_odt(fold_residual=False)
            print(f"\nResidual model, bilinear only (no residual)")
    else:
        embed, cores, unembed = model.cpu().get_cores_for_odt()

    # Run ODT (on CPU)
    odt_results = run_odt(embed, cores, unembed, verbose=True)

    # Plots
    plot_svd_vs_odt(odt_results, model_name)
    plot_atoms(odt_results, model_name)
    plot_digit_eigenvectors(odt_results, model_name)

    if test_loader is not None:
        plot_truncation_curve(model, odt_results, test_loader, device, model_name)
        model.to(device)  # move back for degree analysis
        degree_analysis(model, test_loader, device, model_name)

    return odt_results


def compare_models(chi_path, res_path, device='cpu'):
    """Compare χ-net and residual model ODT spectra."""
    print("\n" + "=" * 60)
    print("χ-net analysis")
    print("=" * 60)
    model_chi, _ = load_model(chi_path, device)
    embed_chi, cores_chi, unembed_chi = model_chi.get_cores_for_odt()
    odt_chi = run_odt(embed_chi, cores_chi, unembed_chi)

    print("\n" + "=" * 60)
    print("Residual model analysis (augmented cores, correct residual)")
    print("=" * 60)
    model_res, _ = load_model(res_path, device)
    embed_res, cores_res, unembed_res = model_res.get_augmented_cores_for_odt()
    print(f"  Augmented hidden dim: {embed_res.shape[0]} (H+1)")
    odt_res = run_odt(embed_res, cores_res, unembed_res)

    print("\n" + "=" * 60)
    print("Residual model analysis (bilinear only, no residual)")
    print("=" * 60)
    embed_res_nores, cores_res_nores, unembed_res_nores = model_res.get_cores_for_odt(
        fold_residual=False)
    odt_res_nores = run_odt(embed_res_nores, cores_res_nores, unembed_res_nores)

    # Spectral gap comparison
    plot_spectral_gap(odt_chi, odt_res)

    # Summary table
    print("\n" + "=" * 60)
    print("SUMMARY: Effective bond dimensions")
    print("=" * 60)
    dims_chi = effective_bond_dim(odt_chi['eigenvalues'])
    dims_res = effective_bond_dim(odt_res['eigenvalues'])
    dims_res_nores = effective_bond_dim(odt_res_nores['eigenvalues'])

    print(f"{'Bond':<10} {'χ-net':<10} {'Res(aug)':<10} {'Res(bil only)':<15}")
    print("-" * 45)
    max_bonds = max(len(dims_chi), len(dims_res))
    for i in range(max_bonds):
        chi_val = dims_chi[i] if i < len(dims_chi) else '-'
        res_val = dims_res[i] if i < len(dims_res) else '-'
        nores_val = dims_res_nores[i] if i < len(dims_res_nores) else '-'
        print(f"Bond {i:<5} {str(chi_val):<10} {str(res_val):<10} {str(nores_val):<15}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--checkpoint', type=str, nargs='+', required=True)
    parser.add_argument('--compare', action='store_true',
                        help='Compare two models (first=chi_net, second=residual)')
    parser.add_argument('--device', type=str, default='cpu')
    parser.add_argument('--with_data', action='store_true',
                        help='Load SVHN test data for truncation curve + degree analysis')
    parser.add_argument('--data_root', type=str, default='./data')
    args = parser.parse_args()

    test_loader = None
    if args.with_data:
        from train import get_svhn_loaders
        _, test_loader, _ = get_svhn_loaders(data_root=args.data_root)

    if args.compare and len(args.checkpoint) >= 2:
        compare_models(args.checkpoint[0], args.checkpoint[1], args.device)
    else:
        for ckpt in args.checkpoint:
            analyze_model(ckpt, test_loader, args.device)
