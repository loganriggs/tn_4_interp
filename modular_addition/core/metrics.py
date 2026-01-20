"""
Metric comparison tools and divergence measures.

This module provides:
- Jensen-Shannon divergence computation
- Average JS divergence between frequency distributions
- Conversion from inner products/similarities to proper distance metrics
- Global metric comparison (correlation, stress)
- Neighborhood preservation metrics (k-NN overlap, Jaccard, trustworthiness, continuity)
"""
import numpy as np
from typing import Callable
from tqdm import tqdm


# =============================================================================
# Jensen-Shannon Divergence
# =============================================================================

def JS_divergence(p: np.ndarray, q: np.ndarray) -> float:
    """
    Compute Jensen-Shannon divergence between two probability distributions.

    JS(p, q) = 0.5 * KL(p || m) + 0.5 * KL(q || m)
    where m = 0.5 * (p + q)

    JS divergence is symmetric and bounded: 0 ≤ JS ≤ ln(2) ≈ 0.693

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


def compute_average_js_divergence(
    heatmap1: np.ndarray,
    heatmap2: np.ndarray
) -> float:
    """
    Compute average JS divergence between two frequency heatmaps.

    Each heatmap has shape (P, n_freqs) where rows are p(freq | remainder).
    Computes JS divergence for each remainder and returns the sum.

    Args:
        heatmap1: (P, n_freqs) frequency distribution for model 1
        heatmap2: (P, n_freqs) frequency distribution for model 2

    Returns:
        Sum of JS divergences across all remainders
    """
    assert heatmap1.shape == heatmap2.shape
    P = heatmap1.shape[0]

    js_sum = 0.0
    for r in range(P):
        js_sum += JS_divergence(heatmap1[r], heatmap2[r])

    return js_sum


def compute_js_divergence_matrix(
    heatmaps: dict,
    P: int,
    show_progress: bool = True
) -> np.ndarray:
    """
    Compute pairwise JS divergence matrix between all frequency heatmaps.

    Args:
        heatmaps: Dict mapping d_hidden -> (P, n_freqs) frequency heatmap
        P: Number of models (bottleneck dimensions 1 to P)
        show_progress: Whether to show progress bar

    Returns:
        (P, P) numpy array where entry [i, j] is summed JS divergence
        between models i+1 and j+1
    """
    js_matrix = np.zeros((P, P), dtype=float)

    iterator = range(1, P + 1)
    if show_progress:
        iterator = tqdm(iterator, desc="Computing JS divergence")

    for i, di in enumerate(iterator):
        heatmap_i = heatmaps[di]
        for j, dj in enumerate(range(1, P + 1)):
            if j < i:
                js_matrix[i, j] = js_matrix[j, i]
            else:
                heatmap_j = heatmaps[dj]
                js_matrix[i, j] = compute_average_js_divergence(heatmap_i, heatmap_j)

    return js_matrix


# =============================================================================
# Converting inner products/similarities to distance metrics
# =============================================================================

def inner_product_to_metric(inner_mat: np.ndarray) -> np.ndarray:
    """
    Convert inner product matrix to distance matrix.

    d(A,B)² = ||A||² + ||B||² - 2⟨A|B⟩
            = ⟨A|A⟩ + ⟨B|B⟩ - 2⟨A|B⟩

    Args:
        inner_mat: (N, N) matrix of inner products ⟨i|j⟩

    Returns:
        (N, N) distance matrix
    """
    norms_sq = np.diag(inner_mat)  # ||i||² = ⟨i|i⟩

    # d²[i,j] = ||i||² + ||j||² - 2⟨i|j⟩
    dist_sq = norms_sq[:, None] + norms_sq[None, :] - 2 * inner_mat

    # Numerical cleanup: ensure non-negative and zero diagonal
    dist_sq = np.maximum(dist_sq, 0)
    np.fill_diagonal(dist_sq, 0)

    return np.sqrt(dist_sq)


def cosine_similarity_to_metric(sim_mat: np.ndarray) -> np.ndarray:
    """
    Convert cosine similarity matrix to distance matrix.

    For normalized vectors: d² = 2(1 - cos_sim)
    This gives d ∈ [0, 2] where d=0 means identical, d=2 means opposite.

    Args:
        sim_mat: (N, N) matrix of cosine similarities in [-1, 1]

    Returns:
        (N, N) distance matrix
    """
    # Clip to valid range
    sim_mat = np.clip(sim_mat, -1, 1)

    dist_sq = 2 * (1 - sim_mat)
    dist_sq = np.maximum(dist_sq, 0)
    np.fill_diagonal(dist_sq, 0)

    return np.sqrt(dist_sq)


def divergence_to_metric(div_mat: np.ndarray) -> np.ndarray:
    """
    Convert divergence matrix to distance matrix.

    JS divergence is already distance-like. We take sqrt to make it
    satisfy the triangle inequality (sqrt(JS) is a proper metric).

    Args:
        div_mat: (N, N) matrix of JS divergences (non-negative)

    Returns:
        (N, N) distance matrix
    """
    dist_mat = np.sqrt(np.maximum(div_mat, 0))
    np.fill_diagonal(dist_mat, 0)
    return dist_mat


# =============================================================================
# Global metric comparison
# =============================================================================

def get_upper_triangular(mat: np.ndarray, k: int = 1) -> np.ndarray:
    """Extract upper triangular elements (excluding diagonal by default)."""
    return mat[np.triu_indices(mat.shape[0], k=k)]


def pearson_correlation(D1: np.ndarray, D2: np.ndarray) -> float:
    """
    Compute Pearson correlation between two distance matrices.

    Uses upper triangular elements only (excluding diagonal).

    Args:
        D1: First distance matrix (N, N)
        D2: Second distance matrix (N, N)

    Returns:
        Pearson correlation coefficient
    """
    d1_flat = get_upper_triangular(D1)
    d2_flat = get_upper_triangular(D2)

    return np.corrcoef(d1_flat, d2_flat)[0, 1]


def compute_optimal_scale(D: np.ndarray, D_prime: np.ndarray) -> float:
    """
    Compute optimal scaling factor α that minimizes ||D - αD'||².

    α = (D · D') / (D' · D')

    Args:
        D: Target distance matrix
        D_prime: Approximating distance matrix

    Returns:
        Optimal scale factor
    """
    d_flat = get_upper_triangular(D)
    dp_flat = get_upper_triangular(D_prime)

    numerator = np.dot(d_flat, dp_flat)
    denominator = np.dot(dp_flat, dp_flat)

    if denominator < 1e-12:
        return 1.0

    return numerator / denominator


def compute_stress(D: np.ndarray, D_prime: np.ndarray, normalize: bool = True) -> float:
    """
    Compute Procrustes stress between two distance matrices.

    stress(D, D') = sqrt(Σ(D_ij - αD'_ij)² / Σ D_ij²)

    where α is the optimal scale factor (if normalize=True).

    Args:
        D: Target distance matrix (N, N)
        D_prime: Approximating distance matrix (N, N)
        normalize: If True, find optimal scale α; if False, use α=1

    Returns:
        Stress value (0 = perfect match, higher = worse)
    """
    d_flat = get_upper_triangular(D)
    dp_flat = get_upper_triangular(D_prime)

    if normalize:
        alpha = compute_optimal_scale(D, D_prime)
    else:
        alpha = 1.0

    residuals_sq = (d_flat - alpha * dp_flat) ** 2
    normalization = np.sum(d_flat ** 2)

    if normalization < 1e-12:
        return 0.0

    return np.sqrt(np.sum(residuals_sq) / normalization)


# =============================================================================
# Neighborhood preservation metrics
# =============================================================================

def get_knn_indices(D: np.ndarray, k: int) -> np.ndarray:
    """
    Get k-nearest neighbor indices for each point.

    Args:
        D: Distance matrix (N, N)
        k: Number of neighbors

    Returns:
        (N, k) array of neighbor indices
    """
    N = D.shape[0]
    k = min(k, N - 1)  # Can't have more neighbors than other points

    # For each row, get indices that would sort it, skip first (self)
    knn = np.zeros((N, k), dtype=int)
    for i in range(N):
        sorted_indices = np.argsort(D[i])
        # Skip index 0 which is self (distance 0)
        knn[i] = sorted_indices[1:k+1]

    return knn


def compute_knn_overlap(D1: np.ndarray, D2: np.ndarray, k: int) -> float:
    """
    Compute k-NN overlap between two distance matrices.

    overlap = (1/N) Σ_i |NN_k^D1(i) ∩ NN_k^D2(i)| / k

    Args:
        D1: First distance matrix (N, N)
        D2: Second distance matrix (N, N)
        k: Number of neighbors

    Returns:
        Average overlap in [0, 1]
    """
    N = D1.shape[0]
    k = min(k, N - 1)

    knn1 = get_knn_indices(D1, k)
    knn2 = get_knn_indices(D2, k)

    overlaps = []
    for i in range(N):
        set1 = set(knn1[i])
        set2 = set(knn2[i])
        overlap = len(set1 & set2) / k
        overlaps.append(overlap)

    return np.mean(overlaps)


def compute_jaccard_index(D1: np.ndarray, D2: np.ndarray, k: int) -> float:
    """
    Compute average Jaccard index for k-NN sets.

    Jaccard(A, B) = |A ∩ B| / |A ∪ B|

    Args:
        D1: First distance matrix (N, N)
        D2: Second distance matrix (N, N)
        k: Number of neighbors

    Returns:
        Average Jaccard index in [0, 1]
    """
    N = D1.shape[0]
    k = min(k, N - 1)

    knn1 = get_knn_indices(D1, k)
    knn2 = get_knn_indices(D2, k)

    jaccards = []
    for i in range(N):
        set1 = set(knn1[i])
        set2 = set(knn2[i])
        intersection = len(set1 & set2)
        union = len(set1 | set2)
        jaccard = intersection / union if union > 0 else 0
        jaccards.append(jaccard)

    return np.mean(jaccards)


def compute_trustworthiness(D_high: np.ndarray, D_low: np.ndarray, k: int) -> float:
    """
    Compute trustworthiness: penalizes points that become neighbors in D_low
    but were not neighbors in D_high (false neighbors).

    T(k) = 1 - (2 / (Nk(2N-3k-1))) Σ_i Σ_{j ∈ U_k(i)} (r(i,j) - k)

    where U_k(i) are points in k-NN of i in D_low but not in D_high,
    and r(i,j) is the rank of j in D_high from i.

    Args:
        D_high: "True" distance matrix (original space)
        D_low: "Embedded" distance matrix (approximation)
        k: Number of neighbors

    Returns:
        Trustworthiness in [0, 1], higher is better
    """
    N = D_high.shape[0]
    k = min(k, N - 1)

    # Get k-NN in both spaces
    knn_high = get_knn_indices(D_high, k)
    knn_low = get_knn_indices(D_low, k)

    # Get ranks in high-dimensional space
    ranks_high = np.zeros((N, N), dtype=int)
    for i in range(N):
        sorted_indices = np.argsort(D_high[i])
        for rank, j in enumerate(sorted_indices):
            ranks_high[i, j] = rank

    # Compute penalty
    penalty = 0
    for i in range(N):
        set_high = set(knn_high[i])
        set_low = set(knn_low[i])
        # False neighbors: in low but not in high
        false_neighbors = set_low - set_high
        for j in false_neighbors:
            penalty += ranks_high[i, j] - k

    # Normalize
    normalization = N * k * (2 * N - 3 * k - 1)
    if normalization <= 0:
        return 1.0

    return 1 - (2 / normalization) * penalty


def compute_continuity(D_high: np.ndarray, D_low: np.ndarray, k: int) -> float:
    """
    Compute continuity: penalizes points that were neighbors in D_high
    but are not neighbors in D_low (missing neighbors).

    C(k) = 1 - (2 / (Nk(2N-3k-1))) Σ_i Σ_{j ∈ V_k(i)} (r̂(i,j) - k)

    where V_k(i) are points in k-NN of i in D_high but not in D_low,
    and r̂(i,j) is the rank of j in D_low from i.

    Args:
        D_high: "True" distance matrix (original space)
        D_low: "Embedded" distance matrix (approximation)
        k: Number of neighbors

    Returns:
        Continuity in [0, 1], higher is better
    """
    N = D_high.shape[0]
    k = min(k, N - 1)

    # Get k-NN in both spaces
    knn_high = get_knn_indices(D_high, k)
    knn_low = get_knn_indices(D_low, k)

    # Get ranks in low-dimensional space
    ranks_low = np.zeros((N, N), dtype=int)
    for i in range(N):
        sorted_indices = np.argsort(D_low[i])
        for rank, j in enumerate(sorted_indices):
            ranks_low[i, j] = rank

    # Compute penalty
    penalty = 0
    for i in range(N):
        set_high = set(knn_high[i])
        set_low = set(knn_low[i])
        # Missing neighbors: in high but not in low
        missing_neighbors = set_high - set_low
        for j in missing_neighbors:
            penalty += ranks_low[i, j] - k

    # Normalize
    normalization = N * k * (2 * N - 3 * k - 1)
    if normalization <= 0:
        return 1.0

    return 1 - (2 / normalization) * penalty


# =============================================================================
# Comprehensive comparison
# =============================================================================

def compare_metrics(
    D1: np.ndarray,
    D2: np.ndarray,
    name1: str = "D1",
    name2: str = "D2",
    k_values: list[int] | None = None
) -> dict:
    """
    Comprehensive comparison between two distance matrices.

    Args:
        D1: First distance matrix
        D2: Second distance matrix
        name1: Name for first metric
        name2: Name for second metric
        k_values: List of k values for neighborhood analysis

    Returns:
        Dictionary with all comparison metrics
    """
    N = D1.shape[0]

    if k_values is None:
        k_values = [1, 2, 3, 5, 10, 15, 20, 30]
    k_values = [k for k in k_values if k < N]

    results = {
        'name1': name1,
        'name2': name2,
        'pearson': pearson_correlation(D1, D2),
        'stress': compute_stress(D1, D2, normalize=True),
        'stress_unnorm': compute_stress(D1, D2, normalize=False),
        'optimal_scale': compute_optimal_scale(D1, D2),
        'k_values': k_values,
        'knn_overlap': [compute_knn_overlap(D1, D2, k) for k in k_values],
        'jaccard': [compute_jaccard_index(D1, D2, k) for k in k_values],
        'trustworthiness': [compute_trustworthiness(D1, D2, k) for k in k_values],
        'continuity': [compute_continuity(D1, D2, k) for k in k_values],
    }

    return results


def print_comparison_results(results: dict) -> None:
    """Print comparison results in a readable format."""
    print(f"\n{'='*60}")
    print(f"Metric Comparison: {results['name1']} vs {results['name2']}")
    print(f"{'='*60}")

    print(f"\nGlobal Measures:")
    print(f"  Pearson correlation: {results['pearson']:.4f}")
    print(f"  Procrustes stress:   {results['stress']:.4f}")
    print(f"  Optimal scale α:     {results['optimal_scale']:.4f}")

    print(f"\nNeighborhood Preservation (k-NN overlap):")
    for k, overlap in zip(results['k_values'], results['knn_overlap']):
        print(f"  k={k:2d}: {overlap:.4f}")

    print(f"\nJaccard Index:")
    for k, jaccard in zip(results['k_values'], results['jaccard']):
        print(f"  k={k:2d}: {jaccard:.4f}")
