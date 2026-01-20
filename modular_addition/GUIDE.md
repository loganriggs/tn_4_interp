# Modular Addition Experiments Guide

This codebase provides tools for studying how tensor networks learn to solve modular addition, and for analyzing learned representations and circuits. This demonstrates the interpretability advantages of tensor networks.

## Background: How Models Solve Modular Addition

### The Problem

Modular addition asks: given two numbers `a` and `b` in the range `[0, P-1]`, compute `(a + b) mod P`. For example, with `P = 64`:
- `25 + 30 = 55` (ordinary addition)
- `40 + 50 = 26` (wraps around: 90 mod 64 = 26)

This is "clock arithmetic" - addition on a circle rather than a line.

### The Fourier Solution

Neural networks (and tensor networks as well) learn to solve this task using Fourier analysis. The key insight is that circular structure is naturally represented by complex exponentials.

**Core idea:** Each position on the circle corresponds to a point on the unit circle in the complex plane:

```
position a  →  e^(2πika/P)  for frequency k
```

For frequency `k`, the solution uses:
1. Embed input `a` as `e^(2πika/P)` (complex number on unit circle)
2. Embed input `b` as `e^(2πikb/P)`
3. Multiply: `e^(2πika/P) × e^(2πikb/P) = e^(2πik(a+b)/P)`
4. The result encodes `(a+b) mod P` in its phase

**Key formulas:**

The Discrete Fourier Transform (DFT) of a signal `x[n]` is:
```
X[k] = Σ_{n=0}^{P-1} x[n] × e^(-2πikn/P)
```

For modular addition, if we represent the inputs as one-hot vectors, the learned weights contain Fourier components. The network computes (approximately):
```
logit[r] ∝ Σ_k |F[k]|² × cos(2πk(a+b-r)/P)
```

where `F[k]` are the learned Fourier coefficients. The argmax over `r` gives `(a+b) mod P`.

### Prior Work: Why Neural Networks Learn Fourier Solutions

Prior work on neural networks trained on modular addition found that gradient descent discovers Fourier-based solutions. Key observations from Nanda et al. (2023):
- Fourier bases are eigenfunctions of circular convolution
- The solution is parameter-efficient: a few Fourier frequencies suffice
- This relates to **grokking**: networks may memorize training data first, then later generalize by discovering the Fourier algorithm

**Reference:** [Progress measures for grokking via mechanistic interpretability](https://arxiv.org/abs/2301.05217) (Neel Nanda, Lawrence Chan, Tom Lieberum, Jess Smith, Jacob Steinhardt, 2023)

### This Codebase: Bilinear Tensor Networks

This codebase studies whether bilinear tensor networks exhibit similar Fourier structure, and develops tools to analyze their learned representations. The bilinear architecture provides interpretability advantages: interaction matrices can be directly extracted and analyzed via eigendecomposition.

## Codebase Architecture

This codebase follows a **hub-and-spoke** architecture:

```
modular_addition/
│
├── core/                  ← Hub: Shared utilities
│   ├── models.py          Model architecture
│   ├── dataset.py         Data generation & training
│   ├── interaction.py     Symmetrized interaction matrices
│   ├── frequency.py       Eigendecomposition & FFT analysis
│   ├── similarity.py      TN and activation similarity
│   └── metrics.py         JS divergence, metric comparison
│
├── compression/           ← Spoke: Bottleneck compression study
│   └── (caching wrappers + notebooks)
│
└── dev_interp/            ← Spoke: Developmental interpretability study
    └── (training dynamics analysis)
```

**Core** provides all shared functionality. The spoke modules import from core and add experiment-specific caching, configuration, and notebooks.

## Quick Navigation

| If you want to... | Go to... |
|-------------------|----------|
| Understand the core API | [core/GUIDE.md](core/GUIDE.md) |
| Run compression experiments | [compression/GUIDE.md](compression/GUIDE.md) |
| Study training dynamics | [dev_interp/GUIDE.md](dev_interp/GUIDE.md) |

## Quick Start

```python
import torch
from modular_addition.core import (
    init_model, make_dataloaders, get_device,
    compute_interaction_matrix, symmetric_similarity
)

# Setup
P = 64
device = get_device()

# Initialize and train model
model = init_model(p=P, d_hidden=32).to(device)
train_loader, val_loader = make_dataloaders(P, train_fraction=0.75, batch_size=64)
optimizer = torch.optim.AdamW(model.parameters(), lr=3e-3, weight_decay=1e-4)

history = model.fit(train_loader, val_loader, optimizer, epochs=500)
print(f"Final val accuracy: {history['val_acc'][-1]:.4f}")

# Analyze: extract (symmetric part of) interaction matrices
int_mat = compute_interaction_matrix(model)  # (P, 2P, 2P)
print(f"Interaction matrix shape: {int_mat.shape}")
```

## Environment

```bash
conda activate tn_interp
```

Required packages: torch, numpy, einops, matplotlib, ipywidgets, tqdm
