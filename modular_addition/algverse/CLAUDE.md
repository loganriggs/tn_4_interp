# Claude Code Project Guide: Symmetric Bilinear Networks

This document contains key procedures and conventions for this project. Claude Code should read this at the start of each session.

---

## Project Overview

This project studies **symmetric bilinear networks** for the 2nd-argmax task:
- Input: x ~ N(0, I) of dimension n
- Output: position of the 2nd largest element
- Architecture: `output = x + Σ_layers D @ (L @ norm(x))²`

---

## Key Mathematical Insight

**The bilinear layer creates QUADRATIC FORMS, not linear transforms!**

```
bilinear_i = D @ (L @ h)²    where ² is ELEMENTWISE
           = Σ_r D_ir (Σ_j L_rj h_j)²
           = Σ_{j,k} M^(i)_jk h_j h_k
           = h^T M^(i) h
```

Where `M^(i)_jk = Σ_r D_ir L_rj L_rk` is a symmetric matrix for each output i.

**D @ L is NOT an "effective matrix"** - this is a common mistake. The correct view is through the quadratic form matrices M^(i).

---

## Proper Training Procedure

### Standard Training
```python
optimizer = torch.optim.AdamW(model.parameters(), lr=0.01, weight_decay=0.01)
batch_size = 128 or 256
```

### Sparsity Training (L1 Pruning)

**Phase 1a: Warm-up (no L1)** - ~5000 steps
```python
optimizer = AdamW(lr=0.01, weight_decay=0.001)
# Train normally
```

**Phase 1b: L1 Penalty Training** - ~10000 steps
```python
optimizer = AdamW(lr=0.005, weight_decay=0.001)
l1_lambda = 0.001

for step in range(10001):
    ce_loss = cross_entropy(model(x), targets)
    l1_loss = sum(layer.L.abs().sum() + layer.D.abs().sum() for layer in model.layers)
    loss = ce_loss + l1_lambda * l1_loss
    # backward, step
```

**Phase 2: Threshold Pruning**
```python
threshold = 0.1  # try 0.05, 0.1, 0.15, 0.2
for layer in model.layers:
    mask_L = (layer.L.abs() >= threshold).float()
    mask_D = (layer.D.abs() >= threshold).float()
    layer.L.data *= mask_L
    layer.D.data *= mask_D
```

**Phase 3: Fine-tune with Masked Gradients** - ~5000 steps
```python
optimizer = AdamW(lr=0.005, weight_decay=0.001)

for step in range(5001):
    loss = cross_entropy(model(x), targets)
    optimizer.zero_grad()
    loss.backward()

    # Mask gradients to keep pruned weights at zero
    with torch.no_grad():
        for i, layer in enumerate(model.layers):
            layer.L.grad *= masks[f'L{i}']
            layer.D.grad *= masks[f'D{i}']

    optimizer.step()

    # Re-apply mask after step
    with torch.no_grad():
        for i, layer in enumerate(model.layers):
            layer.L.data *= masks[f'L{i}']
            layer.D.data *= masks[f'D{i}']
```

---

## TN-Sim (Tensor Network Similarity)

### 1-Layer Symmetric Bilinear
For `output = D @ (L @ x)²`:

```python
def tn_inner_1layer(L1, D1, L2, D2):
    """TN inner product for symmetric bilinear (R=L)."""
    LL = L1 @ L2.T  # (rank, rank)
    C = LL * LL     # Element-wise square since R=L
    DD = D1.T @ D2  # (rank, rank)
    return (DD * C).sum()
```

### 2-Layer Symmetric Bilinear
For composition B2(B1(x)):

```python
def tn_inner_2layer_symmetric(L1_a, D1_a, L2_a, D2_a,
                               L1_b, D1_b, L2_b, D2_b):
    """
    TN inner product for 2-layer symmetric bilinear.
    For symmetric case, R = L.
    """
    # Layer 1 contractions
    LL1 = L1_a @ L1_b.T
    C1 = LL1 * LL1  # Element-wise square (R=L)

    # Layer 2 compositions with layer 1
    A_a = L2_a @ D1_a
    A_b = L2_b @ D1_b

    # Layer 2 contraction
    DD2 = D2_a.T @ D2_b

    # Full contraction
    term = A_a @ C1 @ A_b.T
    return (DD2 * term * term).sum()


def tn_sim_2layer(L1_a, D1_a, L2_a, D2_a, L1_b, D1_b, L2_b, D2_b):
    """Normalized TN similarity."""
    inner_ab = tn_inner_2layer_symmetric(...)
    inner_aa = tn_inner_2layer_symmetric(L1_a, D1_a, L2_a, D2_a,
                                          L1_a, D1_a, L2_a, D2_a)
    inner_bb = tn_inner_2layer_symmetric(L1_b, D1_b, L2_b, D2_b,
                                          L1_b, D1_b, L2_b, D2_b)
    return inner_ab / (sqrt(inner_aa) * sqrt(inner_bb))
```

### Asymmetric Case (L ≠ R)
If L and R are different, use the general formula:
```python
LL = L1 @ L2.T
RR = R1 @ R2.T
LR = L1 @ R2.T
RL = R1 @ L2.T
C = 0.5 * (LL * RR + LR * RL)  # Symmetrized
```

---

## Quadratic Form Analysis

To compute and analyze the quadratic form matrices:

```python
from analysis.analysis_utils import compute_quadratic_forms

M = compute_quadratic_forms(L, D)  # Returns (n, n, n) tensor
# M[i] is the quadratic form matrix for output i
# bilinear_i = h^T M[i] h
```

---

## File Locations

- **Models**: `models.py`
- **Analysis utilities**: `analysis/analysis_utils.py`
- **Pattern documentation**: `analysis/pattern_*.md`
- **Training scripts**: `training/`
- **Pruning scripts**: `prune_*.py`
- **Checkpoints**: `checkpoints/`

---

## Common Hyperparameters

| Setting | Value |
|---------|-------|
| Learning rate | 0.01 |
| Weight decay | 0.01 (grokking), 0.001 (L1 training) |
| Batch size | 128 or 256 |
| L1 lambda | 0.001 |
| Pruning thresholds | 0.05, 0.1, 0.15, 0.2 |

---

## Typical Results

| Config | Accuracy | Notes |
|--------|----------|-------|
| n=4, 1-layer | ~70% | Simple task |
| n=4, 2-layer | ~88% | With L1 pruning |
| n=10, 2-layer | ~65% | Harder task |
| n=10, 3-layer | ~80% | Better for n=10 |
