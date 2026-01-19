"""
JS Divergence computation based on frequency distributions from eigenvector FFT analysis.

Pipeline: models -> interaction matrices -> eigendecomposition -> FFT -> frequency heatmap -> JS divergence
"""
import torch
import numpy as np
from einops import einsum
from pathlib import Path
from tqdm import tqdm

try:
    from .models import Model, init_model, load_sweep_results, get_device
except ImportError:
    from models import Model, init_model, load_sweep_results, get_device


def compute_interaction_matrices(models_state: dict, P: int, device: torch.device | None = None) -> dict:
    """
    Compute full interaction matrices for all bottleneck dimensions.

    The interaction matrix combines w_l, w_r, and w_p into a single tensor
    that represents the full bilinear transformation.

    Args:
        models_state: Dict mapping d_hidden -> model state dict
        P: Input dimension
        device: Device to use for computation

    Returns:
        Dict mapping d_hidden -> (P, 2P, 2P) numpy array of interaction matrices
    """
    if device is None:
        device = get_device()

    int_mats = {}
    for d_hidden, model_state in tqdm(models_state.items(), desc="Computing interaction matrices"):
        model = init_model(p=P, d_hidden=d_hidden).to(device)
        model.load_state_dict(model_state)
        w_l = model.w_l.detach()  # (d_hidden, 2*P)
        w_r = model.w_r.detach()  # (d_hidden, 2*P)
        w_p = model.w_p.detach()  # (P, d_hidden)

        # Compute interaction matrices via CP decomposition
        b = einsum(w_l, w_r, w_p, "hid in1, hid in2, p hid -> p in1 in2").cpu().numpy()
        int_mats[d_hidden] = 0.5 * (b + b.transpose(0, 2, 1))  # Symmetrize

    return int_mats


def compute_eigen_data(int_mats: dict, P: int) -> dict:
    """
    Compute eigendecomposition for all interaction matrices.

    For each bottleneck size and each remainder, computes eigenvalues and
    eigenvectors of the symmetric interaction matrix.

    Args:
        int_mats: Dict mapping d_hidden -> (P, 2P, 2P) interaction matrices
        P: Input dimension

    Returns:
        Dict mapping d_hidden -> {'eigenvalues': (P, 2P), 'eigenvectors': (P, 2P, 2P)}
        Eigenvalues/vectors are sorted by absolute magnitude (descending)
    """
    eigen_data = {}

    for d_hidden in tqdm(range(1, P + 1), desc="Computing eigendecomposition"):
        int_mat = int_mats[d_hidden]  # shape: (P, 2P, 2P)
        eigenvalues = []
        eigenvectors = []

        for remainder in range(P):
            mat = int_mat[remainder]  # shape: (2P, 2P), symmetric
            evals, evecs = np.linalg.eigh(mat)  # eigh for symmetric matrices
            # Sort by absolute magnitude (descending)
            sort_idx = np.argsort(np.abs(evals))[::-1]
            evals = evals[sort_idx]
            evecs = evecs[:, sort_idx]
            eigenvalues.append(evals)
            eigenvectors.append(evecs)

        eigen_data[d_hidden] = {
            'eigenvalues': np.array(eigenvalues),  # (P, 2P)
            'eigenvectors': np.array(eigenvectors)  # (P, 2P, 2P)
        }

    return eigen_data


def compute_frequency_heatmap(eigen_data: dict, d_hidden: int, P: int, n_evecs: int = 4) -> np.ndarray:
    """
    Compute frequency heatmap for a given bottleneck size.

    For each remainder, sums FFT magnitudes of top eigenvectors weighted
    by their eigenvalue magnitudes, then normalizes.

    Args:
        eigen_data: Dict from compute_eigen_data
        d_hidden: Bottleneck dimension to analyze
        P: Input dimension
        n_evecs: Number of top eigenvectors to include

    Returns:
        (P, P//2+1) array where [r, f] = normalized FFT magnitude at frequency f for remainder r
    """
    evecs = eigen_data[d_hidden]['eigenvectors']  # (P, 2P, 2P)
    evals = eigen_data[d_hidden]['eigenvalues']  # (P, 2P)

    n_freqs = P // 2 + 1
    heatmap = np.zeros((P, n_freqs))

    for r in range(P):
        for i in range(min(n_evecs, evecs.shape[2])):
            evec = evecs[r, :, i]
            # Weight by eigenvalue magnitude
            weight = np.abs(evals[r, i])

            # FFT of input a component (first P elements)
            fft_a = np.abs(np.fft.rfft(evec[:P]))
            heatmap[r] += weight * fft_a

    # Normalize each row to get probability distribution
    norm = heatmap.sum(axis=1, keepdims=True)
    norm[norm < 1e-12] = 1.0  # Avoid division by zero
    return heatmap / norm


def JS_divergence(p: np.ndarray, q: np.ndarray) -> float:
    """
    Compute Jensen-Shannon divergence between two probability distributions.

    Args:
        p: First probability distribution
        q: Second probability distribution

    Returns:
        JS divergence value (0 = identical, higher = more different)
    """
    # Normalize to ensure valid distributions
    p = np.asarray(p, dtype=float)
    q = np.asarray(q, dtype=float)
    p = p / (p.sum() + 1e-12)
    q = q / (q.sum() + 1e-12)

    m = 0.5 * (p + q)

    def kl_divergence(a, b):
        # Only compute where a > 0 to avoid log(0)
        mask = a > 1e-12
        a_masked = a[mask]
        b_masked = b[mask]
        # Also ensure b > 0
        b_masked = np.maximum(b_masked, 1e-12)
        return np.sum(a_masked * np.log(a_masked / b_masked))

    return 0.5 * (kl_divergence(p, m) + kl_divergence(q, m))


def compute_js_divergence_matrix(
    models_state: dict,
    P: int,
    n_evecs: int = 4,
    device: torch.device | None = None
) -> np.ndarray:
    """
    Compute the full pairwise JS divergence matrix between frequency distributions.

    Pipeline: models -> int_mats -> eigen -> frequency heatmap -> JS divergence

    Args:
        models_state: Dict mapping d_hidden -> model state dict
        P: Input dimension
        n_evecs: Number of eigenvectors for frequency analysis
        device: Device to use for computation

    Returns:
        (P, P) numpy array where entry [i, j] is summed JS divergence
        across all remainders between models i+1 and j+1
    """
    # Compute intermediate representations
    int_mats = compute_interaction_matrices(models_state, P, device)
    eigen_data = compute_eigen_data(int_mats, P)

    # Pre-compute frequency heatmaps for all bottleneck sizes
    heatmaps = {}
    for d in tqdm(range(1, P + 1), desc="Computing frequency heatmaps"):
        heatmaps[d] = compute_frequency_heatmap(eigen_data, d, P, n_evecs)

    # Compute pairwise JS divergence
    js_matrix = np.zeros((P, P), dtype=float)
    for i, di in enumerate(tqdm(range(1, P + 1), desc="Computing JS divergence")):
        heatmap_i = heatmaps[di]
        for j, dj in enumerate(range(1, P + 1)):
            if j < i:
                js_matrix[i, j] = js_matrix[j, i]
            else:
                heatmap_j = heatmaps[dj]
                js_sum = 0.0
                for r in range(P):
                    js = JS_divergence(heatmap_i[r], heatmap_j[r])
                    js_sum += js
                js_matrix[i, j] = js_sum

    return js_matrix


def load_or_compute_js_divergence(
    sweep_path: str | Path,
    cache_path: str | Path | None = None,
    n_evecs: int = 4,
    device: torch.device | None = None,
    force_recompute: bool = False
) -> np.ndarray:
    """
    Load JS divergence matrix from cache, or compute and cache it.

    Args:
        sweep_path: Path to sweep results pickle file
        cache_path: Path to cache .npy file (default: comp_diagrams/js_divergence_matrix.npy)
        n_evecs: Number of eigenvectors for frequency analysis
        device: Device to use for computation
        force_recompute: If True, recompute even if cache exists

    Returns:
        (P, P) numpy array of pairwise JS divergences
    """
    sweep_path = Path(sweep_path)

    if cache_path is None:
        cache_path = sweep_path.parent / 'js_divergence_matrix.npy'
    else:
        cache_path = Path(cache_path)

    # Try to load from cache
    if not force_recompute and cache_path.exists():
        print(f"Loading JS divergence from cache: {cache_path}")
        return np.load(cache_path)

    # Compute
    print("Computing JS divergence matrix...")
    models_state, val_acc, P = load_sweep_results(sweep_path)
    js_matrix = compute_js_divergence_matrix(models_state, P, n_evecs, device)

    # Save to cache
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    np.save(cache_path, js_matrix)
    print(f"Saved JS divergence to cache: {cache_path}")

    return js_matrix


# Additional utility functions for frequency analysis notebooks


def analyze_eigenvector_fft(evec: np.ndarray, P: int) -> dict:
    """
    Compute FFT of an eigenvector and return frequency analysis.

    Args:
        evec: (2P,) eigenvector
        P: Input dimension

    Returns:
        Dict with FFT magnitudes for first half, second half, and combined
    """
    evec_a = evec[:P]  # first input
    evec_b = evec[P:]  # second input

    # Compute FFT (real signal, so use rfft)
    fft_a = np.abs(np.fft.rfft(evec_a))
    fft_b = np.abs(np.fft.rfft(evec_b))
    fft_full = np.abs(np.fft.rfft(evec))

    return {
        'fft_a': fft_a,
        'fft_b': fft_b,
        'fft_full': fft_full,
        'freqs_a': np.fft.rfftfreq(P, d=1.0),
        'freqs_b': np.fft.rfftfreq(P, d=1.0),
        'freqs_full': np.fft.rfftfreq(2 * P, d=1.0)
    }


def entropy_effective_rank(eigenvalues: np.ndarray) -> float:
    """Entropy-based effective rank: exp(entropy of normalized |λ|²)"""
    evals_sq = eigenvalues ** 2
    total = evals_sq.sum()
    if total < 1e-12:
        return 0.0
    p = evals_sq / total
    p = p[p > 1e-12]  # Filter out zeros
    entropy = -np.sum(p * np.log(p))
    return np.exp(entropy)


def ratio_effective_rank(eigenvalues: np.ndarray) -> float:
    """Ratio-based effective rank: sum(|λ|) / max(|λ|) (nuclear/spectral norm ratio)"""
    abs_evals = np.abs(eigenvalues)
    max_eval = abs_evals.max()
    if max_eval < 1e-12:
        return 0.0
    return abs_evals.sum() / max_eval


def cumulative_explained_variance(eigenvalues: np.ndarray) -> np.ndarray:
    """Compute cumulative explained variance for sorted eigenvalues."""
    evals_sq = eigenvalues ** 2
    total = evals_sq.sum()
    if total < 1e-12:
        return np.zeros_like(evals_sq)
    return np.cumsum(evals_sq) / total


def components_for_variance_threshold(eigenvalues: np.ndarray, threshold: float = 0.99) -> int:
    """Return number of components needed to exceed variance threshold."""
    cev = cumulative_explained_variance(eigenvalues)
    idx = np.searchsorted(cev, threshold)
    return min(idx + 1, len(eigenvalues))


if __name__ == "__main__":
    # Example usage
    import os
    os.chdir(Path(__file__).parent)

    sweep_path = "comp_diagrams/sweep_results_0401.pkl"
    js_matrix = load_or_compute_js_divergence(sweep_path)
    print(f"JS divergence matrix shape: {js_matrix.shape}")
    print(f"JS divergence range: [{js_matrix.min():.4f}, {js_matrix.max():.4f}]")
