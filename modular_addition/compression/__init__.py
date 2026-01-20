"""
Compression analysis tools for modular addition models.

This package provides tools for computing and analyzing similarity metrics
between compressed neural network models with different bottleneck dimensions.

All core functionality is imported from the core/ module. This package adds:
- Caching wrappers for expensive computations
- Compression-specific analysis pipelines

Modules:
- models: Model definitions and loading utilities (from core)
- tn_sim: TN similarity computation with caching
- act_sim: Activation/logit similarity computation with caching
- js_div: JS divergence on frequency distributions with caching
"""

# Import from local modules which re-export from core
from .models import (
    Bilinear,
    ModelConfig,
    Model,
    init_model,
    load_sweep_results,
    load_model_for_dim,
    get_device,
)

from .tn_sim import (
    symmetric_inner,
    symmetric_similarity,
    compute_tn_similarity_matrix,
    load_or_compute_tn_similarity,
)

from .act_sim import (
    create_full_dataset,
    compute_all_logits,
    pairwise_cosine_similarity,
    compute_act_similarity_matrix,
    load_or_compute_act_similarity,
)

from .js_div import (
    compute_interaction_matrices,
    compute_eigen_data,
    compute_frequency_heatmap,
    JS_divergence,
    compute_js_divergence_matrix,
    load_or_compute_js_divergence,
)

__all__ = [
    # models
    'Bilinear',
    'ModelConfig',
    'Model',
    'init_model',
    'load_sweep_results',
    'load_model_for_dim',
    'get_device',
    # tn_sim
    'symmetric_inner',
    'symmetric_similarity',
    'compute_tn_similarity_matrix',
    'load_or_compute_tn_similarity',
    # act_sim
    'create_full_dataset',
    'compute_all_logits',
    'pairwise_cosine_similarity',
    'compute_act_similarity_matrix',
    'load_or_compute_act_similarity',
    # js_div
    'compute_interaction_matrices',
    'compute_eigen_data',
    'compute_frequency_heatmap',
    'JS_divergence',
    'compute_js_divergence_matrix',
    'load_or_compute_js_divergence',
]
