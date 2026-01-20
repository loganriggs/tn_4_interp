# Developmental Interpretability Guide

This module provides tools for analyzing how model representations evolve during training.

[← Back to main guide](../GUIDE.md) | [Core API](../core/GUIDE.md)

## Research Question

**How do model representations change during training, and can we detect phase transitions?**

This relates to the phenomenon of **grokking**: models may memorize training data early, then later generalize by discovering algorithmic solutions (like the Fourier algorithm for modular addition).

## Grokking Phenomenon

Grokking occurs when:
1. Training loss drops quickly
2. Validation accuracy remains low for many epochs (memorization)
3. Suddenly, validation accuracy jumps to near-perfect (generalization)

**Hypothesis:** During grokking, the model's internal representations may undergo a phase transition from memorization circuits to algorithmic circuits. This codebase provides tools to investigate whether this occurs in bilinear tensor networks.

We can try to detect this transition using:
- TN similarity
- Output logit similarity
- Frequency structure evolution 
- Effective rank changes

## Available Tools

All core functionality is available via imports:

```python
from modular_addition.dev_interp import (
    # Models
    Model, init_model, train_model, get_device,

    # Similarity
    symmetric_similarity, compute_activation_similarity,

    # Interaction & Frequency
    compute_interaction_matrix, compute_eigendecomposition,
    compute_frequency_heatmap,

    # Metrics
    JS_divergence, compare_metrics
)
```

## Planned Experiments

### 1. Checkpoint Similarity Tracking

Save model checkpoints during training and compute TN similarity between consecutive checkpoints:

```python
import torch
from modular_addition.dev_interp import (
    init_model, make_dataloaders, get_device,
    symmetric_similarity
)

P = 64
device = get_device()
model = init_model(p=P, d_hidden=32).to(device)
train_loader, val_loader = make_dataloaders(P)
optimizer = torch.optim.AdamW(model.parameters(), lr=3e-3, weight_decay=1e-4)

checkpoints = []
for epoch in range(1000):
    # Training step
    model.train()
    for xb, yb in train_loader:
        logits = model(xb.to(device))
        loss = torch.nn.functional.cross_entropy(logits, yb.to(device))
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    # Save checkpoint every 50 epochs
    if epoch % 50 == 0:
        checkpoints.append({
            'epoch': epoch,
            'state_dict': {k: v.clone() for k, v in model.state_dict().items()}
        })

# Compute similarity between consecutive checkpoints
similarities = []
for i in range(1, len(checkpoints)):
    model1 = load_checkpoint(checkpoints[i-1])
    model2 = load_checkpoint(checkpoints[i])
    sim = symmetric_similarity(model1, model2)
    similarities.append(sim.item())

# Plot: similarity vs epoch
# Hypothesis: low similarity between checkpoints may indicate rapid representation change
```

### 2. Frequency Structure Evolution

Track how the frequency heatmap p(frequency | remainder) changes during training:

```python
from modular_addition.dev_interp import (
    compute_interaction_matrix,
    compute_eigendecomposition,
    compute_frequency_heatmap,
    JS_divergence
)

# For each checkpoint, compute frequency heatmap
heatmaps = []
for ckpt in checkpoints:
    model = load_checkpoint(ckpt)
    int_mat = compute_interaction_matrix(model)
    eigen_data = {32: {
        'eigenvalues': ...,
        'eigenvectors': ...
    }}
    heatmap = compute_frequency_heatmap(eigen_data, d_hidden=32, P=64)
    heatmaps.append(heatmap)

# Compute JS divergence between consecutive heatmaps
js_trajectory = []
for i in range(1, len(heatmaps)):
    # Average JS divergence across all remainders
    js = np.mean([JS_divergence(heatmaps[i-1][r], heatmaps[i][r])
                  for r in range(P)])
    js_trajectory.append(js)

# Spikes in JS divergence may indicate phase transitions
```

### 3. Effective Rank Trajectory

Track the effective rank of interaction matrices:

```python
from modular_addition.dev_interp import (
    compute_interaction_matrix,
    compute_effective_rank
)

ranks = []
for ckpt in checkpoints:
    model = load_checkpoint(ckpt)
    int_mat = compute_interaction_matrix(model)
    # Average effective rank across remainders
    avg_rank = np.mean([compute_effective_rank(int_mat[r]) for r in range(P)])
    ranks.append(avg_rank)

# Plot: effective rank vs epoch
# Decreasing rank may indicate transition to simpler Fourier solution
```

## Detecting Grokking

Hypothesized signatures of grokking in the metrics (to be validated):

| Metric | Before Grokking | During Transition | After Grokking |
|--------|-----------------|-------------------|----------------|
| Val accuracy | Low | Rapidly increasing | High |
| TN similarity (consecutive) | High | Low (rapid change) | High |
| Frequency JS divergence | Varies | High (spikes) | Low (stable) |
| Effective rank | High | Decreasing | Low (few frequencies) |

## Getting Started

```python
import torch
from modular_addition.dev_interp import (
    init_model, make_dataloaders, get_device,
    symmetric_similarity, compute_interaction_matrix
)

# Setup
P = 64
device = get_device()

# Initialize model
model = init_model(p=P, d_hidden=32).to(device)
train_loader, val_loader = make_dataloaders(P)
optimizer = torch.optim.AdamW(model.parameters(), lr=3e-3, weight_decay=1e-4)

# Training loop with checkpoint saving
checkpoints = []
for epoch in range(500):
    # ... training code ...

    if epoch % 50 == 0:
        checkpoints.append(model.state_dict().copy())

# Analyze checkpoints
for i, ckpt in enumerate(checkpoints):
    model.load_state_dict(ckpt)
    int_mat = compute_interaction_matrix(model)
    print(f"Epoch {i*50}: interaction matrix shape = {int_mat.shape}")
```

## References

- [Progress measures for grokking via mechanistic interpretability](https://arxiv.org/abs/2301.05217) — Defines progress measures that predict grokking before it occurs
