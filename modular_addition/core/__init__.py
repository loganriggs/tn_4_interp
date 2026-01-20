"""
Core utilities for modular addition experiments.

This package provides shared functionality used across different experiments
(compression, developmental interpretability, etc.):

Modules:
- models: Model architecture (Bilinear, Model) and loading utilities
- dataset: Dataset creation and training utilities
- interaction: Symmetrized interaction matrix computation
- frequency: Eigendecomposition and frequency analysis
- similarity: TN and activation-based similarity metrics
- metrics: JS divergence and metric comparison tools
"""

# Models
from .models import (
    Bilinear,
    ModelConfig,
    Model,
    init_model,
    get_device,
    load_sweep_results,
    load_model_for_dim,
)

# Dataset and training
from .dataset import (
    create_full_dataset,
    create_labels,
    create_train_val_split,
    make_dataloaders,
    compute_accuracy,
    train_step,
    train_model,
)

# Interaction matrices
from .interaction import (
    compute_interaction_matrix,
    compute_interaction_matrix_stack,
    compute_interaction_matrices_from_state,
    extract_remainder_matrix,
    compute_effective_rank,
)

# Frequency analysis
from .frequency import (
    compute_eigendecomposition,
    compute_eigen_data,
    compute_eigenvector_fft,
    compute_frequency_distribution,
    compute_frequency_heatmap,
    compute_all_frequency_heatmaps,
    entropy_effective_rank,
    ratio_effective_rank,
    cumulative_explained_variance,
    components_for_variance_threshold,
)

# Similarity metrics
from .similarity import (
    symmetric_inner,
    symmetric_similarity,
    compute_tn_inner_product_matrix,
    compute_tn_similarity_matrix,
    compute_all_logits,
    pairwise_cosine_similarity,
    compute_activation_similarity,
    compute_act_similarity_matrix,
)

# Metrics and comparison
from .metrics import (
    JS_divergence,
    compute_average_js_divergence,
    compute_js_divergence_matrix,
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

__all__ = [
    # Models
    'Bilinear',
    'ModelConfig',
    'Model',
    'init_model',
    'get_device',
    'load_sweep_results',
    'load_model_for_dim',
    # Dataset
    'create_full_dataset',
    'create_labels',
    'create_train_val_split',
    'make_dataloaders',
    'compute_accuracy',
    'train_step',
    'train_model',
    # Interaction matrices
    'compute_interaction_matrix',
    'compute_interaction_matrix_stack',
    'compute_interaction_matrices_from_state',
    'extract_remainder_matrix',
    'compute_effective_rank',
    # Frequency
    'compute_eigendecomposition',
    'compute_eigen_data',
    'compute_eigenvector_fft',
    'compute_frequency_distribution',
    'compute_frequency_heatmap',
    'compute_all_frequency_heatmaps',
    'entropy_effective_rank',
    'ratio_effective_rank',
    'cumulative_explained_variance',
    'components_for_variance_threshold',
    # Similarity
    'symmetric_inner',
    'symmetric_similarity',
    'compute_tn_inner_product_matrix',
    'compute_tn_similarity_matrix',
    'compute_all_logits',
    'pairwise_cosine_similarity',
    'compute_activation_similarity',
    'compute_act_similarity_matrix',
    # Metrics
    'JS_divergence',
    'compute_average_js_divergence',
    'compute_js_divergence_matrix',
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
]
