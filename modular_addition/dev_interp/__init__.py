"""
Developmental interpretability tools for modular addition experiments.

This module provides tools for analyzing model development during training,
computing similarity measures between different training stages of the same
model to track phenomena like grokking, phase transitions, and representation
changes over time.

Key features:
- TN similarity across training checkpoints
- Activation similarity evolution
- Frequency structure changes during training
- Grokking detection via metric comparisons

All core functionality is available via the core module. Example usage:

```python
from modular_addition.core import (
    # Models
    Model, init_model, load_sweep_results,
    # Similarity
    symmetric_similarity, compute_activation_similarity,
    # Frequency analysis
    compute_interaction_matrix, compute_eigendecomposition,
    compute_frequency_heatmap,
    # Metrics
    JS_divergence, compare_metrics,
)
```
"""

# Re-export commonly used functions from core for convenience
from ..core import (
    # Models
    Model,
    ModelConfig,
    init_model,
    get_device,
    load_sweep_results,
    load_model_for_dim,
    # Dataset
    create_full_dataset,
    create_labels,
    create_train_val_split,
    compute_accuracy,
    train_model,
    # Similarity
    symmetric_inner,
    symmetric_similarity,
    compute_activation_similarity,
    # Interaction & Frequency
    compute_interaction_matrix,
    compute_eigendecomposition,
    compute_frequency_heatmap,
    # Metrics
    JS_divergence,
    cosine_similarity_to_metric,
    compare_metrics,
)

__all__ = [
    'Model',
    'ModelConfig',
    'init_model',
    'get_device',
    'load_sweep_results',
    'load_model_for_dim',
    'create_full_dataset',
    'create_labels',
    'create_train_val_split',
    'compute_accuracy',
    'train_model',
    'symmetric_inner',
    'symmetric_similarity',
    'compute_activation_similarity',
    'compute_interaction_matrix',
    'compute_eigendecomposition',
    'compute_frequency_heatmap',
    'JS_divergence',
    'cosine_similarity_to_metric',
    'compare_metrics',
]
