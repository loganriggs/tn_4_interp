"""
Eigendecomposition and frequency analysis for interaction matrices.

This module provides:
- Eigendecomposition of interaction matrices
- FFT-based frequency extraction from eigenvectors
- Computing p(frequency | remainder) distributions
- Effective rank and explained variance utilities

The key insight is that eigenvectors of the interaction matrix often have
clean frequency structure when analyzed via FFT, which relates to the
Fourier components used to solve modular addition.
"""
import numpy as np
from tqdm import tqdm


def compute_eigendecomposition(int_mat: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """
    Compute eigendecomposition of a symmetric interaction matrix.

    Args:
        int_mat: (2P, 2P) symmetric matrix

    Returns:
        Tuple of (eigenvalues, eigenvectors) sorted by absolute magnitude (descending).
        eigenvalues: (2P,) array
        eigenvectors: (2P, 2P) array where eigenvectors[:, i] is the i-th eigenvector
    """
    evals, evecs = np.linalg.eigh(int_mat)  # eigh for symmetric matrices

    # Sort by absolute magnitude (descending)
    sort_idx = np.argsort(np.abs(evals))[::-1]
    evals = evals[sort_idx]
    evecs = evecs[:, sort_idx]

    return evals, evecs


def compute_eigen_data(int_mats: dict, P: int, show_progress: bool = True) -> dict:
    """
    Compute eigendecomposition for all interaction matrices.

    For each bottleneck size and each remainder, computes eigenvalues and
    eigenvectors of the symmetric interaction matrix.

    Args:
        int_mats: Dict mapping d_hidden -> (P, 2P, 2P) interaction matrices
        P: Input dimension
        show_progress: Whether to show progress bar

    Returns:
        Dict mapping d_hidden -> {
            'eigenvalues': (P, 2P) array,
            'eigenvectors': (P, 2P, 2P) array
        }
        Eigenvalues/vectors are sorted by absolute magnitude (descending).
    """
    eigen_data = {}

    iterator = range(1, P + 1)
    if show_progress:
        iterator = tqdm(iterator, desc="Computing eigendecomposition")

    for d_hidden in iterator:
        if d_hidden not in int_mats:
            continue

        int_mat = int_mats[d_hidden]  # shape: (P, 2P, 2P)
        eigenvalues = []
        eigenvectors = []

        for remainder in range(P):
            mat = int_mat[remainder]  # shape: (2P, 2P), symmetric
            evals, evecs = compute_eigendecomposition(mat)
            eigenvalues.append(evals)
            eigenvectors.append(evecs)

        eigen_data[d_hidden] = {
            'eigenvalues': np.array(eigenvalues),    # (P, 2P)
            'eigenvectors': np.array(eigenvectors)   # (P, 2P, 2P)
        }

    return eigen_data


def compute_eigenvector_fft(evec: np.ndarray, P: int) -> dict:
    """
    Compute FFT of an eigenvector and return frequency components.

    The eigenvector is split into two halves (corresponding to the two
    input positions in modular addition), and FFT is computed on each.

    Args:
        evec: (2P,) eigenvector
        P: Input dimension

    Returns:
        Dict with:
        - 'fft_a': FFT magnitudes for first input component (P//2+1,)
        - 'fft_b': FFT magnitudes for second input component (P//2+1,)
        - 'fft_full': FFT magnitudes for full vector (P+1,)
        - 'freqs_a': frequency bins for component a
        - 'freqs_b': frequency bins for component b
        - 'freqs_full': frequency bins for full vector
    """
    evec_a = evec[:P]   # first input component
    evec_b = evec[P:]   # second input component

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


def compute_frequency_distribution(
    eigenvalues: np.ndarray,
    eigenvectors: np.ndarray,
    P: int,
    n_evecs: int = 4
) -> np.ndarray:
    """
    Compute p(frequency | remainder) distribution from eigen data.

    For each remainder, sums FFT magnitudes of top eigenvectors weighted
    by their eigenvalue magnitudes, then normalizes to a probability distribution.

    Args:
        eigenvalues: (2P,) eigenvalues for one remainder
        eigenvectors: (2P, 2P) eigenvectors for one remainder
        P: Input dimension
        n_evecs: Number of top eigenvectors to include

    Returns:
        (P//2+1,) probability distribution over frequencies
    """
    n_freqs = P // 2 + 1
    freq_dist = np.zeros(n_freqs)

    for i in range(min(n_evecs, eigenvectors.shape[1])):
        evec = eigenvectors[:, i]
        weight = np.abs(eigenvalues[i])

        # FFT of input a component (first P elements)
        fft_a = np.abs(np.fft.rfft(evec[:P]))
        freq_dist += weight * fft_a

    # Normalize to probability distribution
    total = freq_dist.sum()
    if total > 1e-12:
        freq_dist /= total

    return freq_dist


def compute_frequency_heatmap(
    eigen_data: dict,
    d_hidden: int,
    P: int,
    n_evecs: int = 4
) -> np.ndarray:
    """
    Compute frequency heatmap p(frequency | remainder) for all remainders.

    Args:
        eigen_data: Dict from compute_eigen_data
        d_hidden: Bottleneck dimension to analyze
        P: Input dimension
        n_evecs: Number of top eigenvectors to include

    Returns:
        (P, P//2+1) array where [r, f] = p(frequency=f | remainder=r)
    """
    evecs = eigen_data[d_hidden]['eigenvectors']  # (P, 2P, 2P)
    evals = eigen_data[d_hidden]['eigenvalues']   # (P, 2P)

    n_freqs = P // 2 + 1
    heatmap = np.zeros((P, n_freqs))

    for r in range(P):
        heatmap[r] = compute_frequency_distribution(
            evals[r], evecs[r], P, n_evecs
        )

    return heatmap


def compute_all_frequency_heatmaps(
    eigen_data: dict,
    P: int,
    n_evecs: int = 4,
    show_progress: bool = True
) -> dict:
    """
    Compute frequency heatmaps for all bottleneck dimensions.

    Args:
        eigen_data: Dict from compute_eigen_data
        P: Input dimension
        n_evecs: Number of top eigenvectors to include
        show_progress: Whether to show progress bar

    Returns:
        Dict mapping d_hidden -> (P, P//2+1) frequency heatmap
    """
    heatmaps = {}

    iterator = range(1, P + 1)
    if show_progress:
        iterator = tqdm(iterator, desc="Computing frequency heatmaps")

    for d_hidden in iterator:
        if d_hidden not in eigen_data:
            continue
        heatmaps[d_hidden] = compute_frequency_heatmap(
            eigen_data, d_hidden, P, n_evecs
        )

    return heatmaps


def entropy_effective_rank(eigenvalues: np.ndarray) -> float:
    """
    Compute entropy-based effective rank.

    Effective rank = exp(entropy of normalized |λ|²)

    This measures how "spread out" the eigenvalue spectrum is.
    A rank-1 matrix has effective rank 1, while a matrix with
    uniform eigenvalues has effective rank equal to its dimension.

    Args:
        eigenvalues: Array of eigenvalues

    Returns:
        Effective rank value
    """
    evals_sq = eigenvalues ** 2
    total = evals_sq.sum()
    if total < 1e-12:
        return 0.0
    p = evals_sq / total
    p = p[p > 1e-12]  # Filter out zeros
    entropy = -np.sum(p * np.log(p))
    return np.exp(entropy)


def ratio_effective_rank(eigenvalues: np.ndarray) -> float:
    """
    Compute ratio-based effective rank (nuclear/spectral norm ratio).

    Effective rank = sum(|λ|) / max(|λ|)

    Args:
        eigenvalues: Array of eigenvalues

    Returns:
        Effective rank value
    """
    abs_evals = np.abs(eigenvalues)
    max_eval = abs_evals.max()
    if max_eval < 1e-12:
        return 0.0
    return abs_evals.sum() / max_eval


def cumulative_explained_variance(eigenvalues: np.ndarray) -> np.ndarray:
    """
    Compute cumulative explained variance for sorted eigenvalues.

    Args:
        eigenvalues: Array of eigenvalues (should be sorted by magnitude)

    Returns:
        Array of cumulative explained variance ratios
    """
    evals_sq = eigenvalues ** 2
    total = evals_sq.sum()
    if total < 1e-12:
        return np.zeros_like(evals_sq)
    return np.cumsum(evals_sq) / total


def components_for_variance_threshold(
    eigenvalues: np.ndarray,
    threshold: float = 0.99
) -> int:
    """
    Return number of components needed to exceed variance threshold.

    Args:
        eigenvalues: Array of eigenvalues (should be sorted by magnitude)
        threshold: Variance threshold (e.g., 0.99 for 99%)

    Returns:
        Number of components needed
    """
    cev = cumulative_explained_variance(eigenvalues)
    idx = np.searchsorted(cev, threshold)
    return min(idx + 1, len(eigenvalues))
