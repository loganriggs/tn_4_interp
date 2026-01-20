# Compression Experiments Guide

This module studies how bottleneck dimension affects learned representations in modular addition models.

[← Back to main guide](../GUIDE.md) | [Core API](../core/GUIDE.md)

## Research Question

**How does the bottleneck dimension affect what representations a model learns?**

By training models with different hidden dimensions (1 to 64 for P=64), we can study:
- At what dimension do models fail to generalize?
- How similar are models with different bottleneck sizes?
- Do compressed models learn different frequency structures?

## Experimental Setup

- **Task:** Modular addition with `P = 64`
- **Bottleneck dimensions:** 1 to 64 (64 models total)
- **Training:** 75/25 train/val split, AdamW optimizer
- **Early stopping:** At 100% validation accuracy

All trained models are saved in `comp_diagrams/sweep_results_0401.pkl`.

## Three Similarity Perspectives

We compare models using three complementary metrics:

### 1. TN Similarity (Weight-Space)

Measures structural similarity via the tensor network inner product.

**Interpretation:** High TN similarity means models have similar weight structure, accounting for the bilinear layer's symmetry.

```python
from modular_addition.compression import load_or_compute_tn_similarity

tn_sim = load_or_compute_tn_similarity("comp_diagrams/sweep_results_0401.pkl")
```

### 2. Activation Similarity (Output-Space)

Measures functional equivalence via cosine similarity of logits.

**Interpretation:** High activation similarity means models produce similar outputs on all inputs, regardless of internal structure.

```python
from modular_addition.compression import load_or_compute_act_similarity

act_sim = load_or_compute_act_similarity("comp_diagrams/sweep_results_0401.pkl")
```

### 3. JS Divergence (Frequency Structure)

Compares the p(frequency | remainder) distributions extracted from eigendecomposition.

**Interpretation:** Low JS divergence means models use similar Fourier frequencies to solve each remainder class.

```python
from modular_addition.compression import load_or_compute_js_divergence

js_div = load_or_compute_js_divergence("comp_diagrams/sweep_results_0401.pkl")
```

## Cached Data

All expensive computations are cached to `.npy` files in `comp_diagrams/`:

| File | Description |
|------|-------------|
| `sweep_results_0401.pkl` | Trained model state dicts and validation accuracies |
| `tn_similarity.npy` | (64, 64) TN similarity matrix |
| `act_similarity.npy` | (64, 64) activation similarity matrix |
| `js_divergence.npy` | (64, 64) JS divergence matrix |

The `load_or_compute_*` functions automatically load from cache if available.

## Notebooks

### `training.ipynb`

Train a sweep of models with different bottleneck dimensions.

```python
# Example: train models for d_hidden in [1, 2, 4, 8, 16, 32, 64]
results = run_sweep(P=64, dimensions=[1, 2, 4, 8, 16, 32, 64])
save_sweep_results(results, "comp_diagrams/my_sweep.pkl")
```

### `similarity_analysis.ipynb`

Load similarity matrices, visualize heatmaps, compare metrics.

```python
from modular_addition.compression import (
    load_or_compute_tn_similarity,
    load_or_compute_act_similarity,
    load_or_compute_js_divergence
)
from modular_addition.core import compare_metrics, cosine_similarity_to_metric

# Load all three metrics
tn_sim = load_or_compute_tn_similarity(sweep_path)
act_sim = load_or_compute_act_similarity(sweep_path)
js_div = load_or_compute_js_divergence(sweep_path)

# Convert to distance matrices for comparison
D_tn = cosine_similarity_to_metric(tn_sim)
D_act = cosine_similarity_to_metric(act_sim)

# Compare TN vs Activation
results = compare_metrics(D_tn, D_act, name1="TN", name2="Activation")
```

### `frequency_analysis.ipynb`

Deep dive into eigendecomposition and FFT analysis.

```python
from modular_addition.compression import (
    load_sweep_results,
    compute_interaction_matrices,
    compute_eigen_data,
    compute_frequency_heatmap
)

# Load models and compute interaction matrices
models_state, val_acc, P = load_sweep_results(sweep_path)
int_mats = compute_interaction_matrices(models_state, P)
eigen_data = compute_eigen_data(int_mats, P)

# Visualize frequency structure for d_hidden=32
heatmap = compute_frequency_heatmap(eigen_data, d_hidden=32, P=64)
plt.imshow(heatmap, aspect='auto')
plt.xlabel('Frequency')
plt.ylabel('Remainder')
```

## Key Findings

*(Results from analysis)*

- Models with `d_hidden >= 10` achieve 100% accuracy
- TN similarity shows clear block structure based on bottleneck size
- Activation similarity is high across all models that achieve perfect accuracy
- JS divergence reveals which models use similar frequency structures

## Quick Start

```python
from modular_addition.compression import (
    load_sweep_results,
    load_or_compute_tn_similarity,
    load_or_compute_act_similarity
)
from modular_addition.core import compare_metrics, cosine_similarity_to_metric

sweep_path = "modular_addition/compression/comp_diagrams/sweep_results_0401.pkl"

# Load pre-trained models
models_state, val_acc, P = load_sweep_results(sweep_path)

# Get similarity matrices (cached)
tn_sim = load_or_compute_tn_similarity(sweep_path)
act_sim = load_or_compute_act_similarity(sweep_path)

# Compare the two metrics
D_tn = cosine_similarity_to_metric(tn_sim)
D_act = cosine_similarity_to_metric(act_sim)
results = compare_metrics(D_tn, D_act, name1="TN", name2="Activation")
```
