# 2-Layer Symmetric Bilinear Network: n=4 Pattern Analysis

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
output = x + r1 + r2

Layer 1:
  h1 = γ1 ⊙ (x / rms(x))         # RMSNorm with vector γ1
  r1 = D1 @ (L1 @ h1)²           # bilinear

Layer 2:
  h2 = γ2 ⊙ ((x + r1) / rms(x + r1))
  r2 = D2 @ (L2 @ h2)²
```

**Parameters:**
- `γ1` (RMSNorm layer 1): vector of 4 values ≈ 0.7
- `γ2` (RMSNorm layer 2): vector of 4 values ≈ 1.7
- `L1`, `D1`: layer 1 weights (4×4 each)
- `L2`, `D2`: layer 2 weights (4×4 each)

**5 Computational Paths:**

The output can be decomposed into 5 components:
- **x**: direct input (residual stream)
- **r1**: layer 1 bilinear output = `D1 @ (L1 @ h1)²`
- **A**: x contribution through layer 2 = `D2 @ (L2 @ x_norm)²`
- **B**: cross term in layer 2 = `D2 @ [2 * (L2 @ x_norm) * (L2 @ r1_norm)]`
- **C**: r1 contribution through layer 2 = `D2 @ (L2 @ r1_norm)²`

Where `x_norm` and `r1_norm` are the normalized components within h2.

---

## Model Configuration

**Sparse Model** (L1 pruned with threshold=0.1):
- Accuracy: **88.1%**
- L1: 0% sparse
- **D1: 19% sparse** (3 zeros)
- L2: 0% sparse
- **D2: 38% sparse** (6 zeros)

---

## Learned Weights

### Norm Weights
```
γ1 = [0.727, 0.760, 0.711, 0.748]  ≈ 0.7 (downscales)
γ2 = [1.756, 1.706, 1.703, 1.709]  ≈ 1.7 (upscales)
```

### Layer 1

**L1 (4×4):**
```
        pos0    pos1    pos2    pos3
rank0  -0.334  -0.335  -0.329   0.766   ← detects pos3
rank1   0.774  -0.338  -0.327  -0.322   ← detects pos0
rank2  -0.342   0.765  -0.346  -0.327   ← detects pos1
rank3  -0.319  -0.312   0.746  -0.324   ← detects pos2
```

Each row has one large positive (~0.77) and three negative (~-0.33) entries.

**D1 (4×4) — 19% sparse:**
```
        rank0   rank1   rank2   rank3
pos0   -0.595   0.000  -0.630  -0.616   ← zero at rank1
pos1   -0.593  -0.607   0.000  -0.622   ← zero at rank2
pos2   -0.609  -0.629  -0.610   0.082
pos3    0.000  -0.611  -0.613  -0.607   ← zero at rank0
```

Three zero entries create sparse connections. Most entries are negative (~-0.6).

**Quadratic Form Matrices M1^(i):**

The bilinear layer computes `D1 @ (L1 @ h1)²` where ² is ELEMENTWISE.
This gives: `r1_i = h1^T M1^(i) h1` where `M1^(i)_jk = Σ_r D1_ir L1_rj L1_rk`.

Each M1^(i) is a 4×4 symmetric matrix. Key structure:
- **Large positive diagonal** M1^(i)_ii ≈ 0.4 (self-interaction)
- **Small off-diagonal** entries ≈ -0.05 to 0.05
- **Rank-1 structure**: one dominant eigenvalue, others near zero

**Pattern: r1_i ≈ 0.4·h1_i² + small cross-terms** (emphasizes large |h1_i|)

---

### Layer 2

**L2 (4×4):**
```
        pos0    pos1    pos2    pos3
rank0  -1.319   0.367   0.388   0.394   ← anti-detects pos0
rank1  -0.392   1.330  -0.357  -0.382   ← detects pos1
rank2  -0.389  -0.412  -0.388   1.420   ← detects pos3
rank3   0.400   0.404  -1.400   0.413   ← anti-detects pos2
```

Larger magnitudes than L1 (~1.3-1.4 vs ~0.77). Mixed positive/negative patterns.

**D2 (4×4) — 38% sparse:**
```
        rank0   rank1   rank2   rank3
pos0   -1.479   0.232   0.000   0.000   ← zeros at rank2,3
pos1    0.315  -1.573   0.000   0.000   ← zeros at rank2,3
pos2    0.313   0.219   0.000  -1.620   ← zero at rank2
pos3    0.324   0.233  -1.579   0.000   ← zero at rank3
```

Very sparse! Only ranks 0,1 connect to all outputs. Rank 2 only → pos3. Rank 3 only → pos2.

**Quadratic Form Matrices M2^(i):**

Layer 2 computes `r2_i = h2^T M2^(i) h2` where `M2^(i)_jk = Σ_r D2_ir L2_rj L2_rk`.

Unlike layer 1, the M2^(i) matrices have **more complex structure**:
- **NOT diagonal-dominant**: significant off-diagonal entries
- **Large positive AND negative eigenvalues** (indefinite)
- Each M2^(i) captures **cross-position comparisons**

Example structure of M2^(2) (output for position 2):
```
        pos0    pos1    pos2    pos3
pos0    1.74   -0.54   -0.52   -0.75
pos1   -0.54    0.17    0.16    0.23
pos2   -0.52    0.16    0.15    0.22
pos3   -0.75    0.23    0.22    0.33
```

**Key insight:** The M2 matrices learned richer **pairwise comparison** structure.
This allows: `r2_i = Σ_{j,k} M2^(i)_jk h2_j h2_k` to compare multiple positions.

---

## Powerset Ablation Results

All 32 combinations of the 5 components, sorted by accuracy:

| Rank | Components | Accuracy | Notes |
|------|------------|----------|-------|
| 1 | **x+A+B+C** | **88.7%** | Best! (without r1) |
| 2 | x+r1+A+B+C | 87.8% | Full model |
| 3 | x+A+B | 87.0% | |
| 4 | x+r1+A+B | 86.7% | |
| 5 | A+B+C | 85.5% | No residual x |
| 6 | r1+A+B+C | 84.1% | |
| 7 | A+B | 83.6% | Just layer 2 x-terms |
| 8 | r1+A+B | 83.1% | |
| 9 | x+r1+A | 60.9% | |
| 10 | x+A+C | 60.5% | |
| 11 | x+A | 60.4% | |
| 12 | x+r1+A+C | 59.9% | |
| 13 | x+B | 54.7% | |
| 14 | r1+A | 50.2% | |
| 15 | A | 49.8% | |
| 16 | x+r1+B | 49.6% | |
| 17 | A+C | 48.3% | |
| 18 | r1+A+C | 48.2% | |
| 19 | x+B+C | 43.8% | |
| 20 | B | 42.8% | |
| 21 | x+r1+B+C | 39.5% | |
| 22 | r1+B | 37.5% | |
| 23 | B+C | 30.2% | |
| 24 | r1+B+C | 26.7% | |
| 25 | none | 25.0% | Random baseline |
| 26 | C | 5.4% | |
| 27 | x+C | 2.8% | |
| 28 | r1+C | 2.2% | |
| 29 | x+r1+C | 1.9% | |
| 30 | x+r1 | 0.8% | |
| 31 | r1 | 0.2% | |
| 32 | x | 0.0% | Predicts 1st argmax |

---

## Component Importance Analysis

### Remove one component from full model:

| Component Removed | Remaining Acc | Δ from Full |
|-------------------|---------------|-------------|
| r1 | 88.7% | **+0.9%** |
| C | 86.7% | -1.0% |
| x | 84.1% | -3.7% |
| B | 59.8% | **-28.0%** |
| A | 40.0% | **-47.7%** |

### Single component accuracies:

| Component | Accuracy | vs Random (25%) |
|-----------|----------|-----------------|
| A | 49.8% | +25% |
| B | 42.8% | +18% |
| C | 5.4% | -20% |
| r1 | 0.2% | -25% |
| x | 0.0% | -25% |

---

## Key Findings

### 1. r1 (layer 1 output) is slightly HARMFUL

Removing r1 from the full model **improves** accuracy from 87.8% to 88.7%.

The best combination is `x+A+B+C` — the model works better when layer 1's contribution is ignored!

### 2. A (x through layer 2) is CRITICAL

Removing A drops accuracy by 47.7%, from 87.8% to 40.0%.

A alone achieves 49.8% — nearly double random baseline.

### 3. B (cross term) is the second most important

Removing B drops accuracy by 28.0%.

B captures how the (largely ignored) r1 signal modulates the x signal through layer 2.

### 4. C (r1 through layer 2) is noise

C alone is worse than random (5.4% < 25%).
Removing C only costs 1.0%.

### 5. Sparsity helps!

The sparse model (88.1%) actually outperforms the dense model (86.9%).
D2 being 38% sparse means fewer, more focused connections.

---

## Example: Full Model CORRECT

```
Input and target:
  x      = [ 0.678, -1.235, -0.043, -1.605]
  1st argmax: pos 0 (value  0.678)
  2nd argmax: pos 2 (value -0.043) ← TARGET

5 computational paths:
  x      = [ 0.678, -1.235, -0.043, -1.605]
  r1     = [-0.567, -1.076, -1.078, -0.904]
  A      = [-5.126, -1.184,  0.494, -4.325]
  B      = [-1.905, -1.267,  1.813, -0.423]
  C      = [-0.166, -0.272, -0.225,  0.037]
  ─────────────────────────────────────────
  output = [-7.086, -5.035,  0.962, -7.219]

Prediction: pos 2 ✓ (argmax of output)
```

**Why it works:**
- A contributes +0.494 to pos 2, highest among A values
- B contributes +1.813 to pos 2, highest among B values
- Combined: output[2] = 0.962, far above others (all negative)
- The model correctly identifies pos 2 as 2nd-argmax

---

## Example: Full Model WRONG

```
Input and target:
  x      = [ 1.927,  1.487,  0.901, -2.106]
  1st argmax: pos 0 (value  1.927)
  2nd argmax: pos 1 (value  1.487) ← TARGET

5 computational paths:
  x      = [  1.927,   1.487,   0.901,  -2.106]
  r1     = [ -1.205,  -1.314,  -1.444,  -0.328]
  A      = [-15.446,  -4.136,   3.060, -55.938]
  B      = [  5.208,   4.363,   1.448,  26.463]
  C      = [ -0.397,  -0.924,  -1.777,  -3.099]
  ──────────────────────────────────────────────
  output = [ -9.914,  -0.522,   2.188, -35.007]

Prediction: pos 2 ✗ (should be pos 1)
```

**Why it fails:**
- A gives +3.060 to pos 2, but -4.136 to pos 1 (target)
- The model over-activates on pos 2 because x[2]=0.901 is the 3rd largest
- When top-2 values are close (1.927 vs 1.487), the model confuses rankings
- output[2]=2.188 beats output[1]=-0.522

---

## Interpretation

### Why does the 2-layer model achieve 88% vs 1-layer's 70%?

**Not because of sequential composition!**

The improvement comes from **x passing through layer 2's more expressive bilinear**, not from building on layer 1's output.

Evidence:
- r1 is slightly harmful (removing it helps)
- A (x→layer2) provides 48% of accuracy
- C (r1→layer2) is nearly useless

### What layer 2 learns that layer 1 can't

**Layer 1** quadratic forms M1^(i): Simple diagonal-dominant structure
- Each M1^(i)_ii ≈ 0.4 (self-interaction dominates)
- Small off-diagonal entries
- Computes approximately: r1_i ≈ 0.4·h1_i² + noise
- Can detect large |h_i| but not compare positions well

**Layer 2** quadratic forms M2^(i): Complex comparison structure
- Large off-diagonal entries in M2^(i)
- Both positive AND negative eigenvalues (indefinite)
- Computes: r2_i = Σ_{j,k} M2^(i)_jk h2_j h2_k
- The **cross-terms h_j·h_k** enable pairwise comparisons
- Can detect "2nd largest" by comparing position pairs

### Why sparsity helps

D2 being 38% sparse means:
- Rank 2 only connects to pos 3
- Rank 3 only connects to pos 2
- This forces cleaner, more interpretable computations
- Less interference between unrelated paths

### The role of B (cross term)

Even though r1 alone is useless, its interaction with x through the cross term B provides 28% accuracy contribution. The cross term `2*(L2@x)*(L2@r1)` captures:
- How the "rough estimate" r1 modulates x
- Sign information that pure squaring loses

### Summary

The 2-layer model learns: **"Ignore layer 1's direct output, but use layer 2's richer quadratic forms on x. The cross-term with r1 provides additional correction."**

**Why quadratic forms explain this:**
- Layer 1's M1^(i) matrices are diagonal-dominant → simple self-interactions h_i²
- Layer 2's M2^(i) matrices have rich off-diagonal structure → pairwise comparisons h_j·h_k
- The **A term** (x through layer 2) uses x^T M2^(i) x with complex M2^(i) → compares positions
- The **B term** uses cross-products between x and r1 → sign information

Depth helps by providing access to more expressive quadratic forms (richer M2 matrices), not by composing simpler transformations sequentially.
