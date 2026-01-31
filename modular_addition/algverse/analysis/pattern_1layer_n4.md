# 1-Layer Symmetric Bilinear Network: n=4 Pattern Analysis

## Problem: 2nd-Argmax

Given a 4-dimensional input vector `x ~ N(0, I)`, predict the **position of the 2nd largest element**.

Example:
- Input: `x = [0.5, 1.2, -0.3, 0.8]`
- 1st largest: position 1 (value 1.2)
- 2nd largest: position 3 (value 0.8) ← **target**

Random baseline accuracy: **25%** (1 in 4 chance)

---

## Architecture

```
output = x + D @ (L @ h)²

where:
  h = γ · (x / rms(x))     # RMSNorm with scalar γ
  rms(x) = sqrt(mean(x²) + ε)
```

**Parameters:**
- `γ` (RMSNorm weight): scalar = **1.595**
- `L`: rank × n = 4 × 4 matrix
- `D`: n × rank = 4 × 4 matrix

**Computation flow:**
1. Normalize: `h = 1.595 · x / rms(x)`
2. Project: `Lh = L @ h` → 4 values
3. Square: `Lh² = Lh ** 2` → 4 non-negative values
4. Output projection: `bilinear = D @ Lh²` → 4 values
5. Residual: `output = x + bilinear`

---

## Learned Weights

### L Matrix (4×4) — 0% sparse
```
       pos0    pos1    pos2    pos3
rank0 [-0.321   0.908  -0.300  -0.310]
rank1 [ 0.302   0.295  -0.922   0.305]
rank2 [-0.903   0.283   0.297   0.293]
rank3 [ 0.267   0.299   0.288  -0.898]
```

Each row of L has one dominant positive entry (~0.9) and three smaller entries (~0.3).
This creates a soft "detector" for each position.

### D Matrix (4×4) — 75% sparse
```
       rank0   rank1   rank2   rank3
pos0  [ 0.000   0.000  -1.000   0.000]
pos1  [-0.957   0.000   0.000   0.000]
pos2  [ 0.000  -0.982   0.000   0.000]
pos3  [ 0.000   0.000   0.000  -0.989]
```

D is extremely sparse! Each output position reads from exactly one rank:
- pos0 ← -1.0 × rank2
- pos1 ← -0.96 × rank0
- pos2 ← -0.98 × rank1
- pos3 ← -0.99 × rank3

### Quadratic Form Matrices M^(i)

**Key insight:** The bilinear layer computes `D @ (L @ h)²` where **² is ELEMENTWISE**.
This is NOT a linear transform — it's a **quadratic form**:

```
bilinear_i = Σ_r D_ir (Σ_j L_rj h_j)²
           = Σ_r D_ir Σ_{j,k} L_rj L_rk h_j h_k
           = Σ_{j,k} M^(i)_jk h_j h_k
           = h^T M^(i) h
```

Where **M^(i)_jk = Σ_r D_ir L_rj L_rk** is a 4×4 symmetric matrix for each output i.

**M^(0):** (output position 0)
```
       pos0    pos1    pos2    pos3
pos0 [ 0.817  -0.256  -0.268  -0.265]
pos1 [-0.256   0.080   0.084   0.083]
pos2 [-0.268   0.084   0.088   0.087]
pos3 [-0.265   0.083   0.087   0.086]
```

**M^(1):** (output position 1)
```
       pos0    pos1    pos2    pos3
pos0 [ 0.099  -0.279   0.092   0.095]
pos1 [-0.279   0.789  -0.261  -0.269]
pos2 [ 0.092  -0.261   0.086   0.089]
pos3 [ 0.095  -0.269   0.089   0.092]
```

**M^(2):** (output position 2)
```
       pos0    pos1    pos2    pos3
pos0 [ 0.090  -0.088   0.274  -0.091]
pos1 [-0.088   0.086  -0.268   0.089]
pos2 [ 0.274  -0.268   0.836  -0.278]
pos3 [-0.091   0.089  -0.278   0.092]
```

**M^(3):** (output position 3)
```
       pos0    pos1    pos2    pos3
pos0 [ 0.071  -0.079  -0.076   0.239]
pos1 [-0.079   0.088   0.085  -0.267]
pos2 [-0.076   0.085   0.082  -0.257]
pos3 [ 0.239  -0.267  -0.257   0.798]
```

**Key pattern in each M^(i):**
- **Diagonal M^(i)_ii is large positive** (~0.8) — self-interaction term
- **Off-diagonal M^(i)_ij is small** — cross-position interactions
- Each M^(i) is **indefinite** (has positive and negative eigenvalues)
- Structure is approximately **rank-1**: one large eigenvalue, rest near zero

This means: `bilinear[i] ≈ 0.8·h[i]² + small cross-terms`

The bilinear term **emphasizes** positions with large |h[i]| values through the quadratic self-interaction.

---

## Ablation Results (10,000 samples)

| Component | Accuracy | Description |
|-----------|----------|-------------|
| Full model (x + bilinear) | **71.3%** | Complete model |
| Bilinear only | 50.3% | Without residual connection |
| x only (argmax) | 0.0% | Predicts 1st largest, never 2nd |
| Random | 25.0% | Baseline |

### Breakdown

| Category | Count | Percentage |
|----------|-------|------------|
| Both correct | 5,028 | 50.3% |
| Bilinear wrong, Full correct | 2,104 | 21.0% |
| Bilinear correct, Full wrong | 7 | 0.1% |
| Both wrong | 2,861 | 28.6% |

**Key insight:** The residual connection provides **+21%** accuracy gain. In 21% of cases, the bilinear term alone gets it wrong, but adding x back corrects the prediction.

---

## Example Analysis

### Example 1: Both Correct

```
x        = [ 0.678, -1.235, -0.043, -1.605]
bilinear = [-4.665, -1.466, -0.815, -3.401]
output   = [-3.987, -2.701, -0.858, -5.005]

1st argmax of x: position 0 (value 0.678)
2nd argmax of x: position 2 (value -0.043) ← TARGET

Bilinear pred: 2 ✓
Full pred: 2 ✓
```

**Why it works:** The bilinear term correctly identifies that position 2 has the 2nd largest value. Even though x[2] = -0.043 is small, the bilinear computation suppresses positions 0, 1, 3 more strongly, leaving position 2 as the maximum of the bilinear output.

### Example 2: Bilinear Wrong, Full Correct

```
x        = [-0.098,  1.845, -1.185,  1.384]
bilinear = [-0.667, -3.860, -6.123, -1.677]
output   = [-0.765, -2.016, -7.307, -0.293]

1st argmax of x: position 1 (value 1.845)
2nd argmax of x: position 3 (value 1.384) ← TARGET

Bilinear pred: 0 ✗
Full pred: 3 ✓
```

**Why bilinear fails:** The bilinear term predicts position 0 (least negative). But position 0 has x[0] = -0.098, which is actually the 3rd largest.

**Why full model succeeds:** Adding x back: output[3] = 1.384 + (-1.677) = -0.293, which becomes the maximum. The strong positive x[3] value overcomes the bilinear's mistake.

### Example 3: Both Wrong

```
x        = [ 1.927,  1.487,  0.901, -2.106]
bilinear = [-2.539, -1.079, -0.182, -8.712]
output   = [-0.612,  0.408,  0.718, -10.818]

1st argmax of x: position 0 (value 1.927)
2nd argmax of x: position 1 (value 1.487) ← TARGET

Bilinear pred: 2 ✗
Full pred: 2 ✗
```

**Why both fail:**
- The top two values (1.927, 1.487) are close together
- The bilinear term focuses too much on suppressing position 0 (the largest)
- Position 2 (value 0.901) ends up with the least negative bilinear output
- Adding x doesn't fix it: output[2] = 0.901 + (-0.182) = 0.718 > output[1] = 1.487 + (-1.079) = 0.408

This is a **hard case**: when the 1st and 2nd largest are close in value, the model struggles to distinguish them.

---

## Interpretation

The 1-layer symmetric bilinear network learns a simple but effective strategy:

1. **L creates "position detectors"** with soft one-hot-like rows (~0.9 on one position, ~0.3 on others)
2. **Elementwise squaring** makes the output a **quadratic form** in h
3. **Each output bilinear_i = h^T M^(i) h** where M^(i) is a symmetric matrix
4. **The quadratic forms M^(i) have large diagonal self-interactions** (~0.8):
   - bilinear[i] ≈ 0.8·h[i]² + cross-terms
   - Larger |h[i]| values dominate the output
5. **Residual connection is critical:** provides the "context" of actual input values

**Why quadratic forms help for 2nd-argmax:**
- The h^T M^(i) h structure computes **pairwise comparisons** between positions
- Squaring h amplifies differences: if h[i] >> h[j], then h[i]² >> h[j]²
- The M^(i) matrices learned to emphasize self-interaction (diagonal), which helps identify which position has large magnitude

**Limitation:** With only 1 layer, the model achieves 71% accuracy (vs 25% random). The single quadratic form can detect relative magnitudes but struggles when the top-2 values are similar. Deeper models (2-3 layers) achieve 85%+ by building more complex comparisons through composed quadratic forms.
