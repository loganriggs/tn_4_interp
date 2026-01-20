# Core Module Guide

The `core/` module provides all shared utilities for modular addition experiments. Import from here for any experiment.

[← Back to main guide](../GUIDE.md)

## Overview

```
core/
├── models.py       Model architecture (Bilinear, Model)
├── dataset.py      Dataset creation and training
├── interaction.py  Symmetrized interaction matrices
├── frequency.py    Eigendecomposition and FFT analysis
├── similarity.py   TN and activation similarity metrics
└── metrics.py      JS divergence, metric comparison
```

## Model Architecture (`models.py`)

### Bilinear Layer

The `Bilinear` layer implements a rank-constrained bilinear form:

```python
class Bilinear(nn.Linear):
    """
    For input x of dimension d_in, produces output of dimension d_out by:
    1. Linear transformation to 2*d_out dimensions
    2. Split into left and right halves
    3. Element-wise multiplication: output = left * right
    """
```

Properties:
- `w_l` — Left weight matrix (first half of rows)
- `w_r` — Right weight matrix (second half of rows)

### Model Class

Two-layer architecture: `input (2P one-hot) → Bilinear (d_hidden) → Linear (P) → output`

```python
from modular_addition.core import Model, init_model

model = init_model(p=64, d_hidden=32)  # P=64 classes, 32-dim bottleneck
```

Properties:
- `w_l` — Left weights from bilinear layer: `(d_hidden, 2P)`
- `w_r` — Right weights from bilinear layer: `(d_hidden, 2P)`
- `w_p` — Projection weights: `(P, d_hidden)`

### Training with `model.fit()`

```python
history = model.fit(
    train_loader,           # DataLoader for training
    val_loader,             # DataLoader for validation
    optimizer,              # Pre-initialized optimizer
    epochs=500,             # Maximum epochs
    loss_fn=None,           # Default: CrossEntropyLoss
    device=None,            # Auto-detect if None
    early_stop_acc=1.0,     # Stop at this accuracy (None to disable)
    verbose=True,
    log_every=50
)
# Returns: {'train_loss': [...], 'val_acc': [...], 'stopped_epoch': int}
```

### Loading Trained Models

```python
from modular_addition.core import load_sweep_results, load_model_for_dim

# Load sweep results
models_state, val_acc, P = load_sweep_results("path/to/sweep.pkl")

# Load a specific model
model = load_model_for_dim(d=32, models_state=models_state, P=P)
```

## Dataset Utilities (`dataset.py`)

```python
from modular_addition.core import (
    create_full_dataset,     # All P² input pairs as one-hot
    create_labels,           # Ground truth (a+b) mod P
    create_train_val_split,  # Raw tensors for train/val
    make_dataloaders         # DataLoaders (recommended)
)

# Create full dataset
dataset = create_full_dataset(P=64)  # (4096, 128) tensor
labels = create_labels(P=64)         # (4096,) tensor

# Create DataLoaders (recommended for model.fit())
train_loader, val_loader = make_dataloaders(
    P=64,
    train_fraction=0.75,
    batch_size=64
)
```

## Interaction Matrices (`interaction.py`)

Each interaction matrix gives full and interpretable insight how feature pairs combine in the bilinear layer to output the chosen output dimension.

**Mathematical definition:**
```
B[p, i, j] = Σ_h w_l[h, i] × w_r[h, j] × w_p[p, h]
```

This is symmetrized: `B_sym = (B + B^T) / 2`

```python
from modular_addition.core import (
    compute_interaction_matrix,
    compute_interaction_matrices_from_state,
    compute_effective_rank
)

# Single model
int_mat = compute_interaction_matrix(model)  # (P, 2P, 2P)

# All models in a sweep
int_mats = compute_interaction_matrices_from_state(models_state, P)
# Returns: {d_hidden: (P, 2P, 2P) array, ...}

# Compute effective rank
rank = compute_effective_rank(int_mat[0], method="entropy")  # or "ratio"
```

**Interpretation:** Each `int_mat[r]` is a `(2P, 2P)` symmetric matrix representing how input positions interact to produce output class `r` (the remainder).

## Frequency Analysis (`frequency.py`)

Eigenvectors of interaction matrices can be analyzed for Fourier structure via FFT.

### Eigendecomposition

```python
from modular_addition.core import (
    compute_eigendecomposition,
    compute_eigen_data
)

# Single matrix
eigenvalues, eigenvectors = compute_eigendecomposition(int_mat[0])
# eigenvalues: (2P,), sorted by |λ| descending
# eigenvectors: (2P, 2P), columns are eigenvectors

# All models and remainders
eigen_data = compute_eigen_data(int_mats, P)
# Returns: {d_hidden: {'eigenvalues': (P, 2P), 'eigenvectors': (P, 2P, 2P)}, ...}
```

### FFT of Eigenvectors

```python
from modular_addition.core import (
    compute_eigenvector_fft,
    compute_frequency_distribution,
    compute_frequency_heatmap
)

# FFT of single eigenvector
fft_result = compute_eigenvector_fft(eigenvectors[:, 0], P)
# Keys: 'fft_a', 'fft_b', 'fft_full', 'freqs_a', 'freqs_b', 'freqs_full'

# p(frequency | remainder) distribution
heatmap = compute_frequency_heatmap(eigen_data, d_hidden=32, P=64, n_evecs=4)
# Shape: (P, P//2+1) where [r, f] = p(frequency=f | remainder=r)
```


### JS Divergence

Jensen-Shannon divergence between probability distributions is implemented. To compare extracted frequency distributions: 
$$ p(f|r) = \frac{1}{Z}\sum_{(\lambda, v)\in\text{eigen}[r]}|\underbrace{\text{fft}(v)(f)}_{\text{amplitude of fft of }v\text{ at }f}|*|\lambda|$$

The final heatmap then returns for a pair of models $i$ and $j$: $$\mathrm{heatmap}[i,j]=\sum_{r=1}^P\mathrm{JS}\left[p_i(f|r)|p_j(f|r)\right].$$



```python
from modular_addition.core import (
    JS_divergence,
    compute_js_divergence_matrix
)

js = JS_divergence(p, q)  # Scalar in [0, 1]

# Matrix of JS divergences between frequency heatmaps
js_mat = compute_js_divergence_matrix(heatmaps, P)  # (P, P)
```

### Effective Rank Measures

```python
from modular_addition.core import (
    entropy_effective_rank,    # exp(entropy of normalized |λ|²)
    ratio_effective_rank,      # sum(|λ|) / max(|λ|)
    cumulative_explained_variance,
    components_for_variance_threshold
)

eff_rank = entropy_effective_rank(eigenvalues)
n_components = components_for_variance_threshold(eigenvalues, threshold=0.99)
```

## Similarity Metrics (`similarity.py`)

Two perspectives on model similarity:

### TN Similarity (Weight-Space)

Uses the multi-linear structure of the bilinear layer to define an inner product. Only the symmetric part of the interaction matrix contributes to the output.

```
⟨M1|M2⟩ = Tr(core @ W_p1 @ W_p2.T)

where core = 0.5 × ((W_l1.T @ W_l2) × (W_r1.T @ W_r2) +
                    (W_l1.T @ W_r2) × (W_r1.T @ W_l2))
```

```python
from modular_addition.core import (
    symmetric_inner,           # Raw inner product
    symmetric_similarity,      # Normalized (cosine similarity)
    compute_tn_similarity_matrix
)

inner = symmetric_inner(model1, model2)      # Scalar
sim = symmetric_similarity(model1, model2)   # In [0, 1]

# Full pairwise matrix
tn_sim_mat = compute_tn_similarity_matrix(models_state, P)  # (P, P)
```

### Activation Similarity (Output-Space)

Compares model outputs (logits) on all inputs:

```python
from modular_addition.core import (
    compute_all_logits,
    pairwise_cosine_similarity,
    compute_activation_similarity,
    compute_act_similarity_matrix
)

# Compute logits for all P² inputs
logits = compute_all_logits(model, dataset)  # (P², P)

# Cosine similarity between two models' outputs
act_sim = compute_activation_similarity(model1, model2, dataset)

# Full pairwise matrix
act_sim_mat = compute_act_similarity_matrix(models_state, P)  # (P, P)
```

**How to interpet these two:**
- **TN similarity:** Measures structural similarity in weight space. Good for understanding whether models learned similar internal representations. Out-of-distribution behaviour is also captured with TN similarity.
- **Activation similarity:** Measures functional equivalence on the given data distribution. High similarity means models produce similar outputs. Given the toy nature of the problem, it is fair to assume that using all data on this measure is the standard to compare the other metrics to. For more relativistic data, this would capture model functinoal behaviour well on and only on the data distribution used for this measure.

## Metric Comparison Tools (`metrics.py`)


### Distance Conversions

```python
from modular_addition.core import (
    cosine_similarity_to_metric,  # D = sqrt(2(1 - cos_sim))
    inner_product_to_metric,      # D = sqrt(⟨x|x⟩ + ⟨y|y⟩ - 2⟨x|y⟩)
    divergence_to_metric          # D = sqrt(divergence)
)

D = cosine_similarity_to_metric(tn_sim_mat)  # (P, P) distance matrix
```

### Global Metrics

```python
from modular_addition.core import (
    pearson_correlation,  # Correlation between upper triangular elements
    compute_stress        # Kruskal stress (MDS-style distortion)
)

corr = pearson_correlation(D1, D2)
stress = compute_stress(D1, D2)  # Lower is better
```

### Local Metrics (Neighborhood Preservation)

```python
from modular_addition.core import (
    compute_knn_overlap,      # Fraction of shared k-nearest neighbors
    compute_jaccard_index,    # Jaccard similarity of neighbor sets
    compute_trustworthiness,  # Are new neighbors actually close?
    compute_continuity        # Are old neighbors still close?
)

knn_overlap = compute_knn_overlap(D1, D2, k=5)
trust = compute_trustworthiness(D1, D2, k=5)
cont = compute_continuity(D1, D2, k=5)
```

### Compare Two Metrics

```python
from modular_addition.core import compare_metrics, print_comparison_results

results = compare_metrics(D1, D2, name1="TN", name2="Activation", k=5)
print_comparison_results(results)
```

Output includes: correlation, stress, kNN overlap, Jaccard index, trustworthiness, continuity.

## API Quick Reference

| Module | Key Functions |
|--------|---------------|
| `models` | `init_model`, `load_sweep_results`, `load_model_for_dim`, `get_device` |
| `dataset` | `create_full_dataset`, `create_labels`, `make_dataloaders` |
| `interaction` | `compute_interaction_matrix`, `compute_effective_rank` |
| `frequency` | `compute_eigendecomposition`, `compute_frequency_heatmap`, `entropy_effective_rank`, `JS_divergence` |
| `similarity` | `symmetric_similarity`, `compute_tn_similarity_matrix`, `compute_act_similarity_matrix` |
| `metrics` | `cosine_similarity_to_metric`, `compare_metrics` |
