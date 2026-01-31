# Analysis Code Refactoring Plan

## Status: PARTIALLY COMPLETE

### Completed:
- [x] Created `analysis_utils.py` with correct quadratic form mathematics
- [x] Updated `analyze_symmetric_1layer.py` to use analysis_utils
- [x] Updated `analyze_2layer_n4.py` to use analysis_utils
- [x] Updated `pattern_1layer_n4.md` with correct quadratic form explanation
- [x] Updated `pattern_2layer_n4.md` with correct quadratic form explanation
- [x] Created `__init__.py` for analysis package

### Pending:
- [ ] Create general `analyze_nlayer.py` for any configuration
- [ ] Test with n=5, n=10 configurations
- [ ] Add 3-layer and 4-layer analysis

## Goal
Consolidate analysis code into reusable `analysis_utils.py` that works for 1, 2, 3, 4+ layer models.

## CRITICAL MATHEMATICAL CORRECTION

**D @ L is NOT the "effective matrix"!**

The bilinear layer computes `D @ (L @ h)²` where the **square is ELEMENTWISE**.
This creates a **QUADRATIC FORM**, not a linear transformation:

```
bilinear_i = Σ_r D_ir (Σ_j L_rj h_j)²
           = Σ_r D_ir Σ_{j,k} L_rj L_rk h_j h_k
           = Σ_{j,k} M^(i)_jk h_j h_k
           = h^T M^(i) h
```

Where `M^(i)_jk = Σ_r D_ir L_rj L_rk` is a **symmetric matrix** for each output i.

The full output is:
```
output_i = x_i + (γ/rms)² · x^T M^(i) x
```

---

## Current Code Inventory

### analyze_symmetric_1layer.py
- `plot_weights(ax, mat, title)` - heatmap with values
- `compute_components(x)` - returns {x, bilinear, full}
- `get_predictions(comps, x)` - returns targets, preds, correctness
- `plot_example_flow()` - 1-layer specific visualization

### analyze_2layer_n4.py
- `plot_weights(ax, mat, title)` - **DUPLICATE**
- `compute_5_paths(x)` - returns {x, r1, A, B, C, output}
- `plot_example_2layer()` - 2-layer specific visualization
- Powerset ablation logic

---

## What Can Be Unified

### 1. Checkpoint Loading
```python
def load_checkpoint(path):
    """Load from .pt file or .pkl pruned results."""
    # Handle both formats
    # Return: state_dict, config, accuracy, sparsity_info
```

### 2. Weight Visualization
```python
def plot_weight_matrix(ax, mat, title, show_values=True, fontsize=7):
    """Plot single weight matrix as heatmap with values."""
    # Identical in both files

def plot_all_weights(state_dict, num_layers, n, rank):
    """Plot L, D, D@L for all layers."""
    # Generalize for any number of layers
```

### 3. RMSNorm Computation
```python
def rmsnorm(x, weight, eps=1e-6):
    """Apply RMSNorm with given weight (scalar or vector)."""
    rms = torch.sqrt((x ** 2).mean(dim=-1, keepdim=True) + eps)
    return weight * (x / rms), rms
```

### 4. Bilinear Layer Forward
```python
def bilinear_forward(h, L, D):
    """Compute D @ (L @ h)²"""
    Lh = h @ L.T
    return (Lh ** 2) @ D.T, Lh
```

### 5. Path Computation (GENERALIZE)

**Key insight**: For L layers, we have:
- `x` (input)
- `r1, r2, ..., rL` (layer outputs)
- For layer i > 1, we can decompose its output into terms based on what went in

**1-layer paths**: `[x, r1]` → 2 components
**2-layer paths**: `[x, r1, A2, B2, C2]` → 5 components
  - A2 = x² term through layer 2
  - B2 = cross term (x × r1)
  - C2 = r1² term through layer 2

**3-layer paths**: `[x, r1, A2, B2, C2, A3, B3_xr1, B3_xA2, B3_xB2, ...]`
  - Gets complicated! Many cross terms.

**Simpler approach for 3+ layers**: Just decompose into layer outputs
- `[x, r1, r2, r3]` → 4 components for 3-layer
- Then do ablation on these

```python
def compute_layer_outputs(x, weights, num_layers):
    """
    Compute all layer outputs.

    Returns dict with:
      - x: input
      - r1, r2, ..., rL: layer outputs
      - h1, h2, ..., hL: normalized inputs to each layer
      - output: final output (x + r1 + r2 + ...)
    """
```

For detailed decomposition (like A, B, C in 2-layer):
```python
def decompose_layer2_bilinear(x, r1, L2, D2, norm2_w, rms2):
    """
    Decompose layer 2's bilinear into A, B, C terms.
    Only makes sense for 2-layer analysis.
    """
```

### 6. Ablation
```python
def powerset_ablation(components_dict, component_names, targets):
    """
    Run ablation over all 2^n combinations.

    Args:
        components_dict: {name: tensor} for each component
        component_names: list of names to ablate
        targets: ground truth labels

    Returns: sorted list of {components, accuracy}
    """

def removal_ablation(components_dict, component_names, targets):
    """
    Remove one component at a time from full.

    Returns: {component: accuracy_without}
    """
```

### 7. Visualization
```python
def plot_heatmap_row(ax, data, title, mark_cols=None, vmax=None):
    """Plot 1×n heatmap row with optional column markers."""

def plot_component_flow(x_i, components_dict, component_names, target, pred, title):
    """
    Generic visualization of computation flow.
    Shows each component as a row.
    """
```

---

## Proposed analysis_utils.py Structure

```python
"""
Analysis utilities for symmetric bilinear networks.
Works for 1, 2, 3, 4+ layer models.
"""

import torch
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from itertools import combinations
import pickle

# =============================================================================
# CHECKPOINT LOADING
# =============================================================================
def load_checkpoint(path, prefer_sparse=True): ...

# =============================================================================
# FORWARD COMPUTATION
# =============================================================================
def rmsnorm(x, weight, eps=1e-6): ...
def bilinear_forward(h, L, D): ...
def compute_layer_outputs(x, state_dict, num_layers): ...

# =============================================================================
# PATH DECOMPOSITION (for detailed analysis)
# =============================================================================
def compute_1layer_paths(x, state_dict): ...
def compute_2layer_paths(x, state_dict): ...
def compute_nlayer_paths(x, state_dict, num_layers): ...

# =============================================================================
# ABLATION
# =============================================================================
def powerset_ablation(components, component_names, targets): ...
def removal_ablation(components, component_names, targets): ...
def print_ablation_results(results, top_k=10): ...

# =============================================================================
# VISUALIZATION
# =============================================================================
def plot_weight_matrix(ax, mat, title, show_values=True): ...
def plot_all_weights(state_dict, num_layers, n, rank, save_path=None): ...
def plot_heatmap_row(ax, data, title, mark_cols=None, vmax=None): ...
def plot_component_flow(x_i, components, names, target, pred, title): ...

# =============================================================================
# EVALUATION
# =============================================================================
def evaluate_accuracy(output, targets): ...
def get_example_indices(preds, targets, category='correct', n=5): ...
```

---

## Migration Plan

1. **Create `analysis_utils.py`** with core functions
2. **Test with 1-layer** - verify `analyze_symmetric_1layer.py` still works
3. **Test with 2-layer** - verify `analyze_2layer_n4.py` still works
4. **Simplify analysis scripts** - import from utils, reduce duplication
5. **Create `analyze_nlayer.py`** - general script that works for any n, layers

---

## Open Questions

1. **How deep to go with decomposition?**
   - 2-layer A/B/C decomposition is useful
   - 3+ layer decomposition gets exponentially complex
   - Maybe just do layer-wise ablation for 3+ layers?

2. **Visualization for 3+ layers?**
   - Showing all paths becomes unwieldy
   - Maybe just show: x, r1, r2, r3, output

3. **Different checkpoint formats?**
   - .pt files (direct state dict)
   - .pkl files (pruned results with multiple thresholds)
   - Need unified interface

---

## Next Steps

1. Write `analysis_utils.py` with core functions
2. Refactor `analyze_symmetric_1layer.py` to use utils
3. Refactor `analyze_2layer_n4.py` to use utils
4. Verify both still work
5. Create general `analyze_model.py` for any configuration
