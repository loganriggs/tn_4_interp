"""
Metric comparison tools for compression analysis.

This module re-exports general metric comparison tools from the core module
and adds compression-specific utilities.
"""
import numpy as np

# Import all metric comparison functions from core
from ..core.metrics import (
    inner_product_to_metric,
    cosine_similarity_to_metric,
    divergence_to_metric,
    get_upper_triangular,
    pearson_correlation,
    compute_optimal_scale,
    compute_stress,
    get_knn_indices,
    compute_knn_overlap,
    compute_jaccard_index,
    compute_trustworthiness,
    compute_continuity,
    compare_metrics,
    print_comparison_results,
)

# Import compression-specific functions
from .tn_sim import symmetric_inner
from .models import load_sweep_results, load_model_for_dim
from ..core.similarity import compute_tn_inner_product_matrix


# Export everything
__all__ = [
    # General functions (from core)
    'inner_product_to_metric',
    'cosine_similarity_to_metric',
    'divergence_to_metric',
    'get_upper_triangular',
    'pearson_correlation',
    'compute_optimal_scale',
    'compute_stress',
    'get_knn_indices',
    'compute_knn_overlap',
    'compute_jaccard_index',
    'compute_trustworthiness',
    'compute_continuity',
    'compare_metrics',
    'print_comparison_results',
    # Compression-specific
    'compute_tn_inner_product_matrix',
]
