# Sparse 3-Layer N=10 Model Analysis

Detailed analysis of the iteratively pruned 3-layer symmetric bilinear network on the 2nd-argmax task.

## Model Configuration

| Parameter | Value |
|-----------|-------|
| Input dimension (n) | 10 |
| Rank | 10 |
| Layers | 3 |
| Seed | 8 |

## Pruning Results

| Metric | Value |
|--------|-------|
| Baseline accuracy | 82.9% |
| Sparse accuracy | **83.1%** |
| Accuracy change | **+0.2%** (improved!) |
| Total sparsity | 27.0% |
| Pruning iteration | 3 |

### Per-Layer Sparsity

| Layer | L Sparsity | D Sparsity |
|-------|------------|------------|
| 1 | 4.0% | **37.0%** |
| 2 | 6.0% | 1.0% |
| 3 | **48.0%** | **66.0%** |

**Key observation**: Sparsity concentrates in D1 (37%), L3 (48%), and D3 (66%). Layer 2 remains nearly dense.

### Normalization Scales

| Layer | γ |
|-------|---|
| 1 | 0.773 |
| 2 | 1.324 |
| 3 | **2.971** |

Layer 3 has strong amplification (γ3 ≈ 3), compensating for its sparse weights.

---

## Layer 1 Weights

### L1 Matrix (10×10) - 4% sparse

```
       col0    col1    col2    col3    col4    col5    col6    col7    col8    col9
row0  -0.095  -0.093  +0.963  -0.000  -0.029  -0.129  -0.112  -0.098  -0.096  -0.136
row1  +0.117  -1.027  +0.090  +0.087  +0.108  +0.119  +0.110  +0.111  +0.116  +0.155
row2  -0.144  -0.109  -0.083  -0.121  -0.105  -0.121  +1.020  -0.096  -0.110  -0.162
row3  +0.126  +0.112  +0.081  +0.079  +0.086  +0.115  +0.116  +0.107  -1.024  +0.148
row4  -0.129  -0.096  -0.000  +0.883  -0.000  -0.088  -0.089  -0.100  -0.079  -0.148
row5  -0.108  -0.095  -0.000  -0.028  +0.967  -0.110  -0.089  -0.104  -0.098  -0.137
row6  +0.110  +0.108  +0.085  +0.102  +0.083  -1.023  +0.121  +0.107  +0.124  +0.186
row7  -1.002  +0.100  +0.100  +0.082  +0.080  +0.094  +0.122  +0.119  +0.103  +0.178
row8  +0.135  +0.116  +0.135  +0.161  +0.142  +0.115  +0.124  +0.118  +0.104  -1.015
row9  -0.094  -0.118  -0.093  -0.079  -0.094  -0.102  -0.119  +0.965  -0.108  -0.187
```

**Structure**: Permutation-like with one dominant weight (~±1.0) per row:

| Row | Dominant Col | Value | Pattern |
|-----|--------------|-------|---------|
| 0 | 2 | +0.963 | Selects input 2 |
| 1 | 1 | -1.027 | Selects input 1 |
| 2 | 6 | +1.020 | Selects input 6 |
| 3 | 8 | -1.024 | Selects input 8 |
| 4 | 3 | +0.883 | Selects input 3 |
| 5 | 4 | +0.967 | Selects input 4 |
| 6 | 5 | -1.023 | Selects input 5 |
| 7 | 0 | -1.002 | Selects input 0 |
| 8 | 9 | -1.015 | Selects input 9 |
| 9 | 7 | +0.965 | Selects input 7 |

This is a **scrambled permutation**: each L1 row primarily responds to one input dimension.

### D1 Matrix (10×10) - 37% sparse

```
       col0    col1    col2    col3    col4    col5    col6    col7    col8    col9
row0  -0.000  -0.000  -0.000  -0.000  -0.019  -0.000  -0.000  +0.554  -0.107  -0.051
row1  -0.000  +0.538  -0.000  -0.000  -0.000  -0.000  -0.000  -0.000  -0.125  -0.023
row2  +0.559  -0.236  -0.252  -0.222  -0.259  -0.233  -0.258  -0.225  -0.278  -0.296
row3  -0.232  -0.256  -0.267  -0.234  +0.595  -0.222  -0.251  -0.254  -0.282  -0.286
row4  -0.222  -0.222  -0.238  -0.233  -0.280  +0.544  -0.225  -0.238  -0.291  -0.269
row5  -0.000  -0.000  -0.000  -0.000  -0.011  -0.000  +0.553  -0.000  -0.110  -0.035
row6  -0.000  -0.000  +0.559  -0.000  -0.030  -0.000  -0.020  +0.007  -0.123  -0.000
row7  -0.000  -0.000  -0.024  -0.000  -0.000  -0.000  -0.000  -0.000  -0.110  +0.561
row8  -0.000  -0.000  -0.000  +0.520  -0.023  -0.000  -0.000  -0.000  -0.139  -0.026
row9  +0.116  +0.092  +0.086  +0.123  +0.104  +0.108  +0.137  +0.105  +0.584  +0.121
```

**Two patterns emerge**:

**Pattern A - Sparse rows (outputs 0,1,5,6,7,8)**: 3-5 non-zero entries
| Output | Positive col | Negative cols |
|--------|--------------|---------------|
| 0 | 7 (+0.55) | 4,8,9 |
| 1 | 1 (+0.54) | 8,9 |
| 5 | 6 (+0.55) | 4,8,9 |
| 6 | 2,7 (+0.56,+0.01) | 4,6,8 |
| 7 | 9 (+0.56) | 2,8 |
| 8 | 3 (+0.52) | 4,8,9 |

**Pattern B - Dense rows (outputs 2,3,4,9)**: Full "leave-one-out" structure
| Output | Large positive col | Others |
|--------|-------------------|--------|
| 2 | 0 (+0.56) | All negative (~-0.25) |
| 3 | 4 (+0.60) | All negative (~-0.25) |
| 4 | 5 (+0.54) | All negative (~-0.25) |
| 9 | 8 (+0.58) | All positive (~+0.1) - different! |

---

## Layer 2 Weights (Nearly Dense)

### L2 Matrix (10×10) - 6% sparse

```
       col0    col1    col2    col3    col4    col5    col6    col7    col8    col9
row0  -0.985  +0.098  +0.087  +0.107  +0.115  +0.095  +0.117  +0.088  +0.105  +0.210
row1  +0.123  +0.110  +0.108  +0.122  +0.088  +0.098  +0.064  -0.975  +0.111  +0.229
row2  -0.091  +0.985  -0.110  -0.104  -0.105  -0.106  -0.093  -0.090  -0.090  -0.247
row3  -0.073  -0.070  -0.325  -0.281  +0.934  -0.044  -0.059  -0.075  -0.091  +0.000
row4  +0.203  +0.191  +0.000  +0.000  +0.000  +0.205  +0.182  +0.196  +0.185  +0.326
row5  -0.060  -0.074  -0.297  +0.934  -0.330  -0.065  -0.075  -0.069  -0.064  +0.000
row6  -0.068  -0.080  +0.946  -0.314  -0.295  -0.074  -0.082  -0.072  -0.063  -0.000
row7  -0.105  -0.105  -0.106  -0.110  -0.091  -0.094  +0.994  -0.116  -0.090  -0.233
row8  -0.108  -0.099  -0.119  -0.107  -0.092  +0.976  -0.114  -0.101  -0.097  -0.222
row9  -0.093  -0.093  -0.118  -0.125  -0.103  -0.107  -0.086  -0.113  +0.986  -0.236
```

Similar permutation structure to L1, with dominant weights ~±1.0.

### D2 Matrix (10×10) - 1% sparse (nearly full!)

```
       col0    col1    col2    col3    col4    col5    col6    col7    col8    col9
row0  +0.368  -0.353  -0.351  -0.191  -0.134  -0.197  -0.216  -0.358  -0.356  -0.365
row1  -0.361  -0.364  +0.366  -0.189  -0.122  -0.201  -0.212  -0.354  -0.358  -0.351
row2  -0.097  -0.093  -0.100  -0.303  -0.330  -0.306  -0.726  -0.089  -0.088  -0.092
row3  -0.078  -0.086  -0.081  -0.305  -0.319  -0.726  -0.316  -0.076  -0.076  -0.084
row4  -0.079  -0.084  -0.077  -0.725  -0.332  -0.291  -0.306  -0.071  -0.089  -0.094
row5  -0.349  -0.357  -0.354  -0.204  -0.117  -0.208  -0.196  -0.347  +0.351  -0.363
row6  -0.352  -0.346  -0.349  -0.199  -0.104  -0.181  -0.210  +0.349  -0.352  -0.369
row7  -0.343  +0.347  -0.352  -0.214  -0.128  -0.189  -0.193  -0.353  -0.360  -0.348
row8  -0.360  -0.357  -0.359  -0.203  -0.133  -0.207  -0.207  -0.370  -0.335  +0.349
row9  -0.361  -0.359  -0.384  -0.157  +0.000  -0.155  -0.152  -0.375  -0.366  -0.379
```

**Structure**: Each row has one positive entry (~+0.35 to +0.37), rest negative.

**Grouped structure**:
- Rows 0,1,5,6,7,8: positive ~+0.35, negative ~-0.35 (high contrast)
- Rows 2,3,4: positive absent or small, large negative ~-0.73 at cols 3,4,5,6
- Row 9: All negative

---

## Layer 3 Weights (Sparse)

### L3 Matrix (10×10) - 48% sparse

```
       col0    col1    col2    col3    col4    col5    col6    col7    col8    col9
row0  -0.216  -0.197  -0.429  +1.173  -0.423  -0.211  -0.211  -0.208  -0.209  +0.030
row1  -0.200  -0.208  +1.162  -0.456  -0.383  -0.203  -0.196  -0.192  -0.201  +0.000
row2   0.000   0.000   0.000   0.000   0.000   0.000   0.000  -1.352   0.000  +0.330
row3   0.000  -1.335   0.000   0.000   0.000   0.000   0.000   0.000   0.000  +0.338
row4   0.000   0.000   0.000   0.000   0.000  +1.320   0.000   0.000   0.000  -0.323
row5  -0.186  -0.196  -0.436  -0.393  +1.156  -0.187  -0.203  -0.192  -0.211  +0.034
row6   0.000   0.000   0.000   0.000   0.000   0.000  +1.367   0.000   0.000  -0.323
row7   0.000   0.000   0.000   0.000   0.000  +0.018   0.000   0.000  -1.307  +0.310
row8  +0.194  +0.223  -0.210  -0.225  -0.211  +0.194  +0.198  +0.219  +0.201  -1.561
row9  -1.368   0.000   0.000   0.000   0.000   0.000   0.000   0.000   0.000  +0.347
```

**Two types of rows**:

**Dense rows (0,1,5,8)**: ~10 non-zero entries, complex patterns
**Sparse rows (2,3,4,6,7,9)**: Only 2-3 non-zero entries

| Row | Non-zeros | Pattern |
|-----|-----------|---------|
| 0 | 10 | Dense, dominant +1.17 at col 3 |
| 1 | 9 | Dense, dominant +1.16 at col 2 |
| 2 | 2 | **Sparse**: cols [7,9] = [-1.35, +0.33] |
| 3 | 2 | **Sparse**: cols [1,9] = [-1.34, +0.34] |
| 4 | 2 | **Sparse**: cols [5,9] = [+1.32, -0.32] |
| 5 | 10 | Dense, dominant +1.16 at col 4 |
| 6 | 2 | **Sparse**: cols [6,9] = [+1.37, -0.32] |
| 7 | 3 | **Sparse**: cols [5,8,9] = [+0.02, -1.31, +0.31] |
| 8 | 10 | Dense, dominant -1.56 at col 9 |
| 9 | 2 | **Sparse**: cols [0,9] = [-1.37, +0.35] |

### D3 Matrix (10×10) - 66% sparse

```
       col0    col1    col2    col3    col4    col5    col6    col7    col8    col9
row0   0.000   0.000   0.000   0.000   0.000   0.000   0.000   0.000   0.000  -1.084
row1   0.000   0.000   0.000  -1.105   0.000   0.000   0.000   0.000   0.000   0.000
row2   0.000  +0.766  -0.105  -0.103  -0.119  -0.070  -0.113  -0.112  -0.071  -0.098
row3  +0.752  -0.067  -0.123  -0.120  -0.113  -0.046  -0.103  -0.112  -0.065  -0.105
row4   0.000   0.000  -0.108  -0.112  -0.125  +0.749  -0.120  -0.113  -0.076  -0.121
row5   0.000   0.000   0.000   0.000  -1.074   0.000   0.000   0.000   0.000   0.000
row6   0.000   0.000   0.000   0.000   0.000   0.000  -1.033   0.000   0.000   0.000
row7   0.000   0.000  -1.078   0.000   0.000   0.000   0.000   0.000   0.000   0.000
row8   0.000   0.000   0.000   0.000   0.000   0.000   0.000  -1.125   0.000   0.000
row9   0.000   0.000   0.000   0.000   0.000   0.000   0.000   0.000  -1.427   0.000
```

**Two patterns**:

**Pattern A - Single entry (outputs 0,1,5,6,7,8,9)**: One large negative ~-1.0
| Output | D3 entry | Value |
|--------|----------|-------|
| 0 | col 9 | -1.084 |
| 1 | col 3 | -1.105 |
| 5 | col 4 | -1.074 |
| 6 | col 6 | -1.033 |
| 7 | col 2 | -1.078 |
| 8 | col 7 | -1.125 |
| 9 | col 8 | -1.427 |

**Pattern B - Dense rows (outputs 2,3,4)**: One positive ~+0.75, rest negative ~-0.1
| Output | Positive col | Pattern |
|--------|--------------|---------|
| 2 | col 1 (+0.77) | Leave-one-out |
| 3 | col 0 (+0.75) | Leave-one-out |
| 4 | col 5 (+0.75) | Leave-one-out |

---

## Quadratic Form Analysis

### M1 Quadratic Forms (Layer 1)

| Output | Self M1[i][i,i] | Other diag mean | Rank | Pattern |
|--------|-----------------|-----------------|------|---------|
| 0 | +0.553 | -0.014 | 4 | Boost self (sparse D1) |
| 1 | +0.566 | -0.012 | 3 | Boost self (sparse D1) |
| 2 | +0.502 | -0.268 | 10 | Strong leave-one-out |
| 3 | +0.443 | -0.276 | 10 | Strong leave-one-out |
| 4 | +0.490 | -0.263 | 10 | Strong leave-one-out |
| 5 | +0.576 | -0.012 | 4 | Boost self (sparse D1) |
| 6 | +0.580 | -0.013 | 5 | Boost self (sparse D1) |
| 7 | +0.521 | -0.010 | 3 | Boost self (sparse D1) |
| 8 | +0.544 | -0.017 | 4 | Boost self (sparse D1) |
| 9 | +0.628 | +0.126 | 10 | All positive (special) |

**Two groups**:
- **Outputs 0,1,5,6,7,8**: Low-rank M1 (3-5), weak suppression
- **Outputs 2,3,4**: Full-rank M1 (10), strong leave-one-out suppression
- **Output 9**: Full-rank, all positive - anomalous

### M2 Quadratic Forms (Layer 2)

All M2 matrices are **full rank (9-10)** with:
- Diagonal mean ≈ -0.22
- Top 3 singular values similar magnitude

Layer 2 performs **dense quadratic mixing** - no sparsity-induced simplification.

### M3 Quadratic Forms (Layer 3) - Key Finding!

| Output | D3 non-zeros | M3 Rank | Top singular value |
|--------|--------------|---------|-------------------|
| 0 | 1 (col 9) | **1** | 2.157 |
| 1 | 1 (col 3) | **1** | 2.094 |
| 2 | 9 cols | 9 | 1.458 |
| 3 | 10 cols | 10 | 1.457 |
| 4 | 8 cols | 8 | 1.408 |
| 5 | 1 (col 4) | **1** | 1.984 |
| 6 | 1 (col 6) | **1** | 2.037 |
| 7 | 1 (col 2) | **1** | 2.089 |
| 8 | 1 (col 7) | **1** | 2.030 |
| 9 | 1 (col 8) | **1** | 4.036 |

**7 of 10 M3 matrices are rank-1!** (outputs 0,1,5,6,7,8,9)

**Example M3[0] (rank-1)**:
```
       col0    col1    ...    col9
row0  -2.027   0.000   ...   +0.514
row1   0.000   0.000   ...    0.000
...
row9  +0.514   0.000   ...   -0.130
```

Only positions (0,0), (0,9), (9,0), (9,9) are non-zero. This comes from:
- D3[0] has single entry at col 9
- L3[9] has entries at cols [0, 9]
- M3[0] = D3[0,9] * L3[9] ⊗ L3[9]

**Example M3[5] (rank-1)**:
```
       col0    ...    col5    ...    col9
row5   0.000   ...   -1.872   ...   +0.457
...
row9   0.000   ...   +0.457   ...   -0.112
```

Only positions involving rows/cols {5,9} are non-zero.

---

## Layer Contribution Ablation

### Norm Contributions

| Component | Mean Norm |
|-----------|-----------|
| ||x|| | 3.09 |
| ||r1|| | 2.81 |
| ||r2|| | 11.42 |
| ||r3|| | **39.43** |

Layer 3 dominates by a large margin.

### Progressive Accuracy

| Output | Accuracy |
|--------|----------|
| x only | 0.0% |
| x + r1 | 17.4% |
| x + r1 + r2 | 22.7% |
| x + r1 + r2 + r3 | **83.5%** |

### Ablation (removing one layer)

| Removed | Accuracy | Impact |
|---------|----------|--------|
| None | 83.5% | Baseline |
| r1 | 79.8% | -3.7% |
| r2 | 71.1% | **-12.4%** |
| r3 | 22.7% | **-60.8%** |

**r3 is critical** - removing it drops accuracy by 60%!

### Single Layer Contributions

| Layer | x + rᵢ only |
|-------|-------------|
| r1 | 17.4% |
| r2 | 26.4% |
| r3 | **68.4%** |

r3 alone achieves 68.4% - nearly as good as the full model!

---

## Interpretation: The 3-Stage Algorithm

### Stage 1 (Layer 1): Feature Extraction

**Two parallel strategies**:

1. **Sparse pathway (outputs 0,1,5,6,7,8)**:
   - Low-rank M1 (3-5)
   - Weak "boost self" pattern
   - Focused feature detection

2. **Dense pathway (outputs 2,3,4,9)**:
   - Full-rank M1 (10)
   - Strong "leave-one-out" suppression
   - Comprehensive comparison

### Stage 2 (Layer 2): Dense Mixing

- All M2 matrices are full-rank
- Performs dense quadratic combinations
- Mixes Stage 1 features thoroughly
- Acts as an **information bottleneck** - compresses and combines

### Stage 3 (Layer 3): Sparse Selection

**For outputs 0,1,5,6,7,8,9** (7 outputs):
- **Rank-1 M3** - single quadratic form
- D3 has exactly 1 non-zero entry
- L3 row has only 2-3 entries
- Final decision is a **single sparse quadratic combination**

**For outputs 2,3,4** (3 outputs):
- Higher-rank M3 (8-10)
- D3 has leave-one-out structure
- More complex decision boundary needed

### Connection to r3 Components

The user asked to break r3 into its 3 components. Given the rank-1 structure:

For output i with rank-1 M3 (using D3[i,k] as the single non-zero):
```
r3_i = γ3 * h3^T M3[i] h3
     = γ3 * D3[i,k] * (L3[k] · h3)²
```

This is the square of a single linear combination of the Layer 2 output!

The sparse L3 rows (e.g., L3[9] = [-1.37, 0, 0, 0, 0, 0, 0, 0, 0, +0.35]) mean:
```
r3_0 = γ3 * D3[0,9] * (-1.37*h3[0] + 0.35*h3[9])²
```

**Each rank-1 output computes the square of a 2-term linear combination!**

---

## Summary

| Property | Layer 1 | Layer 2 | Layer 3 |
|----------|---------|---------|---------|
| L sparsity | 4% | 6% | **48%** |
| D sparsity | **37%** | 1% | **66%** |
| M rank | Mixed (3-10) | Full (10) | Mostly 1 |
| Role | Feature extraction | Dense mixing | Sparse selection |
| Contribution | 17.4% | +5.3% | **+60.8%** |

### Key Findings

1. **Layer 3 is critical**: Removing r3 drops accuracy by 60%
2. **7/10 outputs use rank-1 M3**: Simple sparse quadratic selection
3. **Layer 2 stays dense**: Acts as mixing/bottleneck layer
4. **Layer 1 has two strategies**: Sparse (6 outputs) vs dense (4 outputs)
5. **Accuracy improved with pruning**: 83.1% vs 82.9% baseline!

The 3-layer network achieves better accuracy with 27% sparsity by:
1. Maintaining dense Layer 2 for thorough mixing
2. Collapsing Layer 3 to mostly rank-1 operations
3. Using Layer 1 sparsity to create focused features for specific outputs

---

## Computation Examples

Three detailed examples showing the step-by-step computation through the 3-layer network. For r3, we decompose into 6 terms:
- **A**: x*x term (h3_x^T M3 h3_x)
- **B**: r1*r1 term (h3_r1^T M3 h3_r1)
- **C**: r2*r2 term (h3_r2^T M3 h3_r2)
- **D**: 2·x·r1 cross-term (2·h3_x^T M3 h3_r1)
- **E**: 2·x·r2 cross-term (2·h3_x^T M3 h3_r2)
- **F**: 2·r1·r2 cross-term (2·h3_r1^T M3 h3_r2)

### Example 1: Target = position 9 ✓

**Input:**
```
x     = [ +1.927  +1.487  +0.901  -2.106  +0.678  -1.235  -0.043  -1.605  -0.752  +1.649]
         pos 0   pos 1   pos 2   pos 3   pos 4   pos 5   pos 6   pos 7   pos 8   pos 9

Max at position 0 (1.927), 2nd max at position 9 (1.649) → Target = 9
```

**Layer 1:**
```
rms1 = 1.3789, gamma1 = 0.7728
h1    = [ +1.080  +0.834  +0.505  -1.180  +0.380  -0.692  -0.024  -0.899  -0.422  +0.924]
r1    = [ +0.434  +0.223  -1.667  -0.570  -1.714  +0.268  -0.163  +0.583  -0.017  +1.248]
```

**Layer 2:**
```
rms2 = 1.6830, gamma2 = 1.3235
h2    = [ +1.857  +1.345  -0.603  -2.104  -0.815  -0.760  -0.162  -0.804  -0.605  +2.278]
r2    = [ -1.061  -2.462  -1.847  -2.732  -1.681  -2.667  -3.224  -2.473  -2.997  -3.324]
```

**Layer 3:**
```
rms3 = 3.1185, gamma3 = 2.9712
h3    = [ +1.239  -0.716  -2.490  -5.152  -2.589  -3.461  -3.268  -3.330  -3.588  -0.407]
```

**R3 Decomposition (6 terms):**
```
A     = [ -4.192  -2.042  -0.357  +3.507  -1.402  -4.556  -0.328  -7.211  -2.209  -7.952]
        (x*x term)
B     = [ -0.026  -0.015  +0.932  -0.086  +0.916  -0.002  -0.367  -0.138  -0.174  -0.889]
        (r1*r1 term)
C     = [ -0.088  -4.682  +0.342  -3.889  +0.652  -5.843 -10.412  -4.938  -8.230 -15.807]
        (r2*r2 term) ← Large suppression
D     = [ -0.657  +0.355  -3.270  -2.047  -2.593  -0.203  -0.695  +1.997  -1.241  -5.317]
        (2*x*r1 term)
E     = [ +1.218  +6.184  +4.462  -8.153  +3.284 -10.319  -3.698 -11.934  -8.529 +22.423]
        (2*x*r2 term) ← DOMINANT: strongly boosts position 9!
F     = [ +0.095  -0.538  -4.628  +1.485  -4.849  -0.230  -3.911  +1.652  -2.396  +7.496]
        (2*r1*r2 term)

r3    = [ -3.649  -0.738  -2.519  -9.184  -3.991 -21.153 -19.411 -20.571 -22.779  -0.045]
        (full r3 = A+B+C+D+E+F)
```

**Final Output:**
```
out   = [ -2.349  -1.489  -5.133 -14.592  -6.708 -24.786 -22.841 -24.066 -26.545  -0.472]
        prediction: 9 (target: 9) CORRECT
```

**Analysis:** The E term (2·x·r2 cross-term) provides a massive +22.423 boost to position 9, far exceeding any other position. Despite the C term trying to suppress position 9 (-15.807), the E term wins. Position 9 has final output -0.472, clearly the maximum.

---

### Example 2: Target = position 5 ✓

**Input:**
```
x     = [ -0.392  -1.404  -0.728  -0.559  -0.769  +0.762  +1.642  -0.160  -0.497  +0.440]

Max at position 6 (1.642), 2nd max at position 5 (0.762) → Target = 5
```

**Layer 1:**
```
rms1 = 0.8546, gamma1 = 0.7728
h1    = [ -0.355  -1.269  -0.658  -0.506  -0.695  +0.690  +1.485  -0.144  -0.450  +0.398]
r1    = [ +0.004  +0.886  -1.504  -1.790  -1.440  +0.361  +1.754  -0.127  +0.023  +0.974]
```

**Layer 2:**
```
rms2 = 1.7578, gamma2 = 1.3235
h2    = [ -0.292  -0.390  -1.681  -1.769  -1.663  +0.846  +2.557  -0.216  -0.357  +1.064]
r2    = [ -3.686  -3.609  -1.698  -1.596  -1.466  -2.977  +2.100  -3.683  -3.783  -3.743]
```

**Layer 3:**
```
rms3 = 3.8871, gamma3 = 2.9712
h3    = [ -3.114  -3.154  -3.004  -3.016  -2.809  -1.417  +4.201  -3.034  -3.255  -1.781]
```

**R3 Decomposition (6 terms):**
```
A     = [ -0.301  -2.639  -0.642  -0.691  -0.672  -0.469  -2.667  -0.082  -0.421  -0.073]
        (x*x term)
B     = [ -0.070  -0.469  +0.027  +0.482  -0.065  -0.016  -2.616  -0.153  -0.051  -0.007]
        (r1*r1 term)
C     = [ -8.880  -8.134  -3.036  -3.067  -3.008  -4.650 -10.044  -8.830  -9.152 -10.918]
        (r2*r2 term) ← Strong suppression everywhere
D     = [ -0.289  +2.225  -0.255  -0.403  -0.244  -0.176  -5.283  -0.224  -0.293  +0.047]
        (2*x*r1 term)
E     = [ -3.268  -9.265  -2.965  -2.485  -3.370  +2.954 -10.352  -1.703  -3.924  +1.787]
        (2*x*r2 term) ← Only position 5 gets positive boost!
F     = [ -1.573  +3.906  -3.178  -4.072  -3.218  +0.553 -10.251  -2.326  -1.366  -0.572]
        (2*r1*r2 term)

r3    = [-14.381 -14.376 -10.049 -10.237 -10.578  -1.804 -41.212 -13.318 -15.206  -9.737]
        (full r3 = A+B+C+D+E+F)
```

**Final Output:**
```
out   = [-18.455 -18.503 -13.979 -14.182 -14.252  -3.658 -35.716 -17.287 -19.464 -12.067]
        prediction: 5 (target: 5) CORRECT
```

**Analysis:** The E term provides a selective +2.954 boost to position 5 (the only positive E value). Combined with the r3 structure leaving position 5 at only -1.804 suppression (vs -10 to -41 for others), position 5 clearly wins with final output -3.658.

---

### Example 3: Target = position 7 ✗

**Input:**
```
x     = [ -0.758  +1.078  +0.801  +1.681  +1.279  +1.296  +0.610  +1.335  -0.232  +0.042]

Max at position 3 (1.681), 2nd max at position 7 (1.335) → Target = 7
```

**Layer 1:**
```
rms1 = 1.0358, gamma1 = 0.7728
h1    = [ -0.566  +0.805  +0.597  +1.254  +0.954  +0.967  +0.455  +0.996  -0.173  +0.031]
r1    = [ +0.640  +0.009  -0.982  -0.404  -0.751  +0.105  -0.075  +0.111  +0.167  +0.684]
```

**Layer 2:**
```
rms2 = 0.8949, gamma2 = 1.3235
h2    = [ -0.174  +1.608  -0.268  +1.888  +0.782  +2.073  +0.791  +2.138  -0.095  +1.073]
r2    = [ -2.207  -3.113  -3.604  -3.048  -2.525  -2.460  -3.225  -2.331  -2.190  -2.929]
```

**Layer 3:**
```
rms3 = 2.2368, gamma3 = 2.9712
h3    = [ -3.088  -2.690  -5.028  -2.353  -2.652  -1.407  -3.573  -1.176  -2.995  -2.926]
```

**R3 Decomposition (6 terms):**
```
A     = [ -2.114  -3.958  -0.591  -1.954  -2.013  -5.464  -1.228  -6.103  -0.228  -0.074]
        (x*x term)
B     = [ -0.780  -0.094  +0.853  -0.222  +0.109  -0.013  -0.192  -0.011  -0.000  -0.449]
        (r1*r1 term)
C     = [ -7.675 -19.502  -7.783  -3.642  +0.640 -10.054 -21.823  -9.083  -7.250 -28.444]
        (r2*r2 term)
D     = [ +2.568  +1.219  +2.831  +0.452  +0.819  +0.528  +0.970  +0.518  +0.006  -0.363]
        (2*x*r1 term)
E     = [ -8.056 +17.571  +1.710  +8.039  +3.914 +14.823 +10.352 +14.891  -2.574  +2.893]
        (2*x*r2 term) ← Boosts positions 1,3,5,7 similarly
F     = [ +4.893  -2.705  -2.715  +0.887  -3.208  -0.716  -4.089  -0.632  +0.036  +7.146]
        (2*r1*r2 term)

r3    = [-11.164  -7.469  -5.696  +3.560  +0.261  -0.895 -16.009  -0.420 -10.010 -19.292]
        (full r3 = A+B+C+D+E+F)
```

**Final Output:**
```
out   = [-13.488  -9.494  -9.481  +1.789  -1.735  -1.954 -18.699  -1.306 -12.264 -21.495]
        prediction: 3 (target: 7) WRONG
```

**Analysis:** The E term provides similar boosts to both positions 3 (+8.039) and 7 (+14.891). However, position 3 benefits from being the argmax initially, and the combination of terms leaves r3[3] = +3.560 (positive!) while r3[7] = -0.420. The model incorrectly predicts the argmax (position 3) instead of the 2nd-argmax (position 7).

---

### Key Observations from Examples

1. **E term (2·x·r2) is critical in layer 3**: This cross-term between the input and layer 2's output provides the most position-specific signal. It's the primary mechanism for identifying the 2nd-argmax.

2. **C term (r2*r2) provides global suppression**: The r2*r2 term tends to be large and negative across all positions, acting as a normalizing suppressor.

3. **Layer 3's rank-1 structure is selective**: For rank-1 M3 matrices, each output position computes essentially `D3[i,k] * (L3[k] · h3)²`, a sparse quadratic form. This creates sharp, selective decisions.

4. **The 3-layer model handles position 5 correctly**: Unlike the 2-layer model which struggles with position 5, the 3-layer model correctly predicts it in Example 2. The additional layer provides better separation.

5. **Argmax confusion in hard cases**: In Example 3, the model confuses the argmax (position 3) with the 2nd-argmax (position 7) when they're relatively close in value (1.681 vs 1.335). The E term boosts both, and the argmax wins.

---

## Files

- Checkpoint: `checkpoints/3layer_n10_r10_seed8_sparse.pkl`
- Analysis script: `analysis/analyze_3layer_n10_sparse.py`
- Visualizations: `images/sparse_3layer_n10/`
  - `weights_all.png` - All 6 weight matrices (L1, D1, L2, D2, L3, D3)
  - `M1_all.png` - All 10 M1 quadratic forms with rank
  - `M2_all.png` - All 10 M2 quadratic forms (all full rank)
  - `M3_all.png` - All 10 M3 quadratic forms (7/10 are rank-1!)
  - `M3_rank1_examples.png` - Detailed M3[0] and M3[5]
