# Sparse 2-Layer N=10 Model Analysis

Detailed analysis of the iteratively pruned 2-layer symmetric bilinear network on the 2nd-argmax task.

## Model Configuration

| Parameter | Value |
|-----------|-------|
| Input dimension (n) | 10 |
| Rank | 10 |
| Layers | 2 |
| Seed | 8 |

## Pruning Results

| Metric | Value |
|--------|-------|
| Baseline accuracy | 63.7% |
| Sparse accuracy | 61.3% |
| Accuracy drop | 2.4% |
| Total sparsity | 40.8% |
| Pruning iteration | 5 |

### Per-Layer Sparsity

| Layer | Sparsity | Zeros/Total |
|-------|----------|-------------|
| L1 | 7.0% | 7/100 |
| D1 | 1.0% | 1/100 |
| L2 | **69.0%** | 69/100 |
| D2 | **86.0%** | 86/100 |

---

## Layer 1 Weights (Dense)

### L1 Matrix (10×10)

```
       col0    col1    col2    col3    col4    col5    col6    col7    col8    col9
row0  -0.165  -0.161   0.627  -0.137  -0.153  -0.000  -0.119  -0.155  -0.126  -0.161
row1   0.132  -0.662   0.132   0.117   0.159   0.125   0.112   0.139   0.140   0.138
row2  -0.105  -0.137  -0.163  -0.115  -0.143  -0.000   0.641  -0.155  -0.135  -0.156
row3   0.161   0.174   0.154   0.143   0.150   0.000   0.143   0.138  -0.624   0.133
row4  -0.132  -0.154  -0.148   0.612  -0.159  -0.000  -0.116  -0.170  -0.141  -0.200
row5  -0.149  -0.134  -0.114  -0.000  -0.134  -0.160  -0.091  -0.114  -0.153   0.657
row6   0.080   0.109   0.214   0.210   0.066  -0.638   0.225   0.234   0.241   0.081
row7  -0.696   0.099   0.132   0.135   0.110   0.149   0.143   0.116   0.121   0.100
row8   0.133   0.103   0.110   0.117  -0.679   0.155   0.000   0.114   0.113   0.123
row9  -0.161  -0.148  -0.135  -0.142  -0.140  -0.000  -0.142   0.616  -0.130  -0.136
```

**Structure**: Each row has one large-magnitude element (~±0.6-0.7) and many small elements (~±0.1-0.2). The large element appears at different column positions for each row.

**Pattern**: Row r has its dominant weight at a specific column c(r):
- Row 0 → col 2 (+0.627)
- Row 1 → col 1 (-0.662)
- Row 2 → col 6 (+0.641)
- Row 3 → col 8 (-0.624)
- Row 4 → col 3 (+0.612)
- Row 5 → col 9 (+0.657)
- Row 6 → col 5 (-0.638)
- Row 7 → col 0 (-0.696)
- Row 8 → col 4 (-0.679)
- Row 9 → col 7 (+0.616)

This forms a **permutation-like structure** where each row "selects" a different input dimension as its primary feature.

### D1 Matrix (10×10)

```
       col0    col1    col2    col3    col4    col5    col6    col7    col8    col9
row0  -0.193  -0.175  -0.243  -0.198  -0.203  -0.160  -0.229   0.407  -0.205  -0.219
row1  -0.189   0.407  -0.152  -0.171  -0.186  -0.172  -0.209  -0.136  -0.149  -0.189
row2   0.311  -0.116  -0.136  -0.184  -0.163  -0.106  -0.218  -0.071  -0.079  -0.170
row3  -0.171  -0.103  -0.152  -0.140   0.347  -0.122  -0.238  -0.077  -0.086  -0.185
row4  -0.205  -0.137  -0.193  -0.201  -0.210  -0.176  -0.208  -0.145   0.414  -0.191
row5   0.020   0.332   0.128   0.038   0.147   0.319  -0.157   0.462   0.335   0.048
row6  -0.180  -0.113   0.383  -0.167  -0.173  -0.109  -0.259  -0.000  -0.110  -0.161
row7  -0.170  -0.106  -0.148  -0.178  -0.125  -0.105  -0.223  -0.081  -0.095   0.317
row8  -0.185  -0.117  -0.152   0.323  -0.159  -0.091  -0.210  -0.069  -0.091  -0.155
row9  -0.219  -0.183  -0.155  -0.198  -0.149   0.393  -0.229  -0.161  -0.174  -0.196
```

**Structure**: Each row has one positive element (~0.3-0.5) and many negative elements (~-0.1 to -0.2).

**Pattern** (output i → positive D1 entry at column r):
- Output 0 → col 7 (+0.407)
- Output 1 → col 1 (+0.407)
- Output 2 → col 0 (+0.311)
- Output 3 → col 4 (+0.347)
- Output 4 → col 8 (+0.414)
- Output 5 → cols 1,5,7,8 all positive (row 5 is special!)
- Output 6 → col 2 (+0.383)
- Output 7 → col 9 (+0.317)
- Output 8 → col 3 (+0.323)
- Output 9 → col 5 (+0.393)

This is reminiscent of the **"leave-one-out"** pattern from n=4, but more complex.

---

## Layer 2 Weights (Sparse)

### L2 Matrix (10×10) - 69% sparse

```
       col0    col1    col2    col3    col4    col5    col6    col7    col8    col9
row0  -0.968   0.069   0.000   0.000   0.000   0.013   0.000   0.000   0.000   0.085
row1   0.000   0.000   0.000   0.000   0.000  -0.207   0.000   0.000   1.061  -0.066
row2   0.000   0.000   0.000  -0.062   0.000  -0.220   0.000   1.091   0.000   0.000
row3   0.000  -0.087   0.000   0.000   0.961  -0.032   0.000   0.000   0.000   0.000
row4   0.000   0.000  -1.121   0.000   0.000   0.202   0.081   0.000   0.000   0.000
row5   0.000   0.000   0.000   1.079   0.000  -0.215  -0.074   0.000  -0.046   0.000
row6   0.000   0.000   0.000   0.000   0.000  -0.365   0.000   0.000   0.000   0.000
row7   0.000  -0.978   0.000   0.000   0.000   0.064   0.056   0.000   0.000   0.000
row8  -0.098   0.000   0.000   0.000   0.000  -0.263   1.049   0.000   0.000   0.000
row9   0.000   0.000   0.000   0.136   0.000   0.053   0.057   0.000   0.000  -0.972
```

**Non-zero entries per row**:
| Row | Non-zeros | Columns | Dominant Value |
|-----|-----------|---------|----------------|
| 0 | 4 | [0,1,5,9] | -0.968 at col 0 |
| 1 | 3 | [5,8,9] | +1.061 at col 8 |
| 2 | 3 | [3,5,7] | +1.091 at col 7 |
| 3 | 3 | [1,4,5] | +0.961 at col 4 |
| 4 | 3 | [2,5,6] | -1.121 at col 2 |
| 5 | 4 | [3,5,6,8] | +1.079 at col 3 |
| 6 | **1** | [5] | -0.365 at col 5 |
| 7 | 3 | [1,5,6] | -0.978 at col 1 |
| 8 | 3 | [0,5,6] | +1.049 at col 6 |
| 9 | 4 | [3,5,6,9] | -0.972 at col 9 |

**Key observation**: Column 5 appears in nearly every row! This rank dimension is shared across all outputs.

### D2 Matrix (10×10) - 86% sparse

```
       col0    col1    col2    col3    col4    col5    col6    col7    col8    col9
row0  -0.466   0.000   0.000   0.000   0.000   0.000   0.000   0.000   0.000   0.000
row1   0.000   0.000   0.000   0.000   0.000   0.000   0.000  -0.536   0.000   0.000
row2   0.000   0.000   0.000   0.000  -0.563   0.000   0.000   0.000   0.000   0.000
row3   0.000   0.000   0.000   0.000   0.000  -0.594   0.000   0.000   0.000   0.000
row4   0.000   0.000   0.000  -0.466   0.000   0.000   0.000   0.000   0.000   0.000
row5  -0.116   0.000   0.000  -0.104   0.000   0.000  -0.247  -0.101   0.000  -0.088
row6   0.000   0.000   0.000   0.000   0.000   0.000   0.000   0.000  -0.587   0.000
row7   0.000   0.000  -0.543   0.000   0.000   0.000   0.000   0.000   0.000   0.000
row8   0.000  -0.597   0.000   0.000   0.000   0.000   0.000   0.000   0.000   0.000
row9   0.000   0.000   0.000   0.000   0.000   0.000   0.000   0.000   0.000  -0.509
```

**All 14 non-zero entries**:
| Entry | Value | Note |
|-------|-------|------|
| D2[0,0] | -0.466 | Output 0 uses rank 0 |
| D2[1,7] | -0.536 | Output 1 uses rank 7 |
| D2[2,4] | -0.563 | Output 2 uses rank 4 |
| D2[3,5] | -0.594 | Output 3 uses rank 5 |
| D2[4,3] | -0.466 | Output 4 uses rank 3 |
| D2[5,0] | -0.116 | Output 5 uses rank 0,3,6,7,9 |
| D2[5,3] | -0.104 | (5 entries - special!) |
| D2[5,6] | -0.247 | |
| D2[5,7] | -0.101 | |
| D2[5,9] | -0.088 | |
| D2[6,8] | -0.587 | Output 6 uses rank 8 |
| D2[7,2] | -0.543 | Output 7 uses rank 2 |
| D2[8,1] | -0.597 | Output 8 uses rank 1 |
| D2[9,9] | -0.509 | Output 9 uses rank 9 |

**Critical finding**: **All D2 values are negative!** This means Layer 2 performs suppression/inhibition.

**Structure**: 9 of 10 outputs have exactly 1 non-zero D2 entry. Output 5 has 5 non-zero entries.

---

## Quadratic Form Analysis

The bilinear layer computes: `output_i = h^T M^(i) h` where `M^(i)_jk = Σ_r D[i,r] * L[r,j] * L[r,k]`

### Layer 1 Quadratic Forms (M1)

**Pattern**: "Boost self, suppress others" - similar to n=4!

For each output i, M1[i] has:
- **Positive self-diagonal**: M1[i][i,i] ≈ +0.09 to +0.16
- **Negative other-diagonals**: M1[i][j,j] ≈ -0.05 to -0.12 for j≠i
- **Near-zero off-diagonals**: M1[i][j,k] ≈ 0 for j≠k

| Output | Self M1[i][i,i] | Other diag mean | Off-diag mean |
|--------|-----------------|-----------------|---------------|
| 0 | +0.163 | -0.108 | +0.004 |
| 1 | +0.148 | -0.086 | +0.004 |
| 2 | +0.092 | -0.068 | +0.003 |
| 3 | +0.104 | -0.074 | +0.003 |
| 4 | +0.160 | -0.100 | +0.003 |
| 5 | **-0.032** | +0.111 | -0.008 |
| 6 | +0.130 | -0.072 | +0.002 |
| 7 | +0.088 | -0.072 | +0.003 |
| 8 | +0.096 | -0.071 | +0.003 |
| 9 | +0.138 | -0.100 | +0.003 |

**Output 5 is anomalous**: It has negative self-term and positive others - the opposite pattern!

**Example: M1[0]**
```
       j=0     j=1     j=2     j=3     j=4     j=5     j=6     j=7     j=8     j=9
k=0  [+0.163] -0.044  -0.048  -0.048  -0.043  -0.041  -0.049  -0.042  -0.045  -0.044
k=1   -0.044 [-0.106]  0.010   0.013   0.008   0.030   0.016   0.010   0.014   0.004
k=2   -0.048   0.010 [-0.106]  0.013   0.011   0.030   0.020   0.008   0.007   0.007
...
```
The bracketed diagonal entries show: M1[0][0,0] = +0.163 (boost), M1[0][1,1] = -0.106 (suppress), etc.

### Layer 2 Quadratic Forms (M2) - Almost All Rank-1!

Because D2 is so sparse (most rows have only 1 non-zero entry), most M2[i] are **rank-1 matrices**.

| Output | D2 non-zeros | M2 Rank | Singular Values |
|--------|--------------|---------|-----------------|
| 0 | 1 (col 0) | **1** | [0.442, 0, 0, ...] |
| 1 | 1 (col 7) | **1** | [0.517, 0, 0, ...] |
| 2 | 1 (col 4) | **1** | [0.734, 0, 0, ...] |
| 3 | 1 (col 5) | **1** | [0.724, 0, 0, ...] |
| 4 | 1 (col 3) | **1** | [0.435, 0, 0, ...] |
| 5 | 5 (cols 0,3,6,7,9) | **5** | [0.116, 0.103, 0.088, 0.083, 0.032] |
| 6 | 1 (col 8) | **1** | [0.692, 0, 0, ...] |
| 7 | 1 (col 2) | **1** | [0.675, 0, 0, ...] |
| 8 | 1 (col 1) | **1** | [0.700, 0, 0, ...] |
| 9 | 1 (col 9) | **1** | [0.493, 0, 0, ...] |

**Key finding**: 9 of 10 M2 matrices are rank-1! Each can be written as M2[i] = σ * v * v^T for some vector v.

**Example: M2[0] (rank-1)**
```
       j=0     j=1     j=2     j=3     j=4     j=5     j=6     j=7     j=8     j=9
k=0  [-0.437]  0.031   0.000   0.000   0.000   0.006   0.000   0.000   0.000   0.038
k=1    0.031  -0.002   0.000   0.000   0.000  -0.000   0.000   0.000   0.000  -0.003
k=2    0.000   0.000   0.000   0.000   0.000   0.000   0.000   0.000   0.000   0.000
...
k=9    0.038  -0.003   0.000   0.000   0.000  -0.001   0.000   0.000   0.000  -0.003
```

Only entries involving columns {0,1,5,9} are non-zero, matching L2[0]'s non-zero columns.

**Example: M2[5] (rank-5, the exception)**
```
       j=0     j=1     j=2     j=3     j=4     j=5     j=6     j=7     j=8     j=9
k=0  [-0.109]  0.008   0.000   0.000   0.000   0.001   0.000   0.000   0.000   0.010
k=1    0.008 [-0.098]  0.000   0.000   0.009   0.006   0.006   0.000   0.000  -0.001
...
k=9    0.010  -0.001   0.000   0.012   0.000   0.004   0.005   0.000   0.000 [-0.084]
```

M2[5] has structure across more dimensions because D2[5] has 5 non-zero entries.

---

## Layer Contribution Ablation

### Norm Contributions

| Component | Mean Norm |
|-----------|-----------|
| ||x|| | 3.09 |
| ||r1|| | 4.00 |
| ||r2|| | **22.05** |

Layer 2 output dominates by a large margin (5x larger than r1).

### Progressive Accuracy

| Output | Accuracy |
|--------|----------|
| x only | 0.0% |
| x + r1 | 9.4% |
| x + r1 + r2 | **60.5%** |

### Ablation (removing one layer)

| Removed | Accuracy | Impact |
|---------|----------|--------|
| None | 60.5% | Baseline |
| r1 (x + r2 only) | **61.7%** | +1.2% (improves!) |
| r2 (x + r1 only) | 9.4% | **-51.1%** |

**Key finding**: Removing r1 actually *improves* accuracy slightly! Layer 2 alone is sufficient.

### Single Layer Contributions

| Layer | x + rᵢ only |
|-------|-------------|
| r1 | 9.4% |
| r2 | **61.7%** |

**r2 is critical and sufficient** - it achieves the full model's accuracy by itself.

### R2 Component Decomposition (A/B/C)

Since layer 2's input is `h2 ∝ (x + r1)`, and bilinear is quadratic, we can decompose:

```
r2_i = h2^T M2[i] h2 = (x + r1)^T M2[i] (x + r1)
     = x^T M2[i] x        [A: x with x]
     + 2 x^T M2[i] r1     [B: x with r1 cross-term]
     + r1^T M2[i] r1      [C: r1 with r1]
```

**Component ablation:**

| Ablation | Accuracy | Δ |
|----------|----------|---|
| Full model | 60.5% | - |
| Without A (x*x) | 40.1% | -20.4% |
| **Without B (x*r1)** | **19.6%** | **-40.9%** |
| Without C (r1*r1) | 37.6% | -22.8% |

**B (the cross-term) is the most important component!**

| Single component | Accuracy |
|------------------|----------|
| x + r1 + A only | 12.1% |
| **x + r1 + B only** | **46.9%** |
| x + r1 + C only | 8.7% |
| x + B only (no r1) | 48.7% |

The cross-term B alone achieves 47-49% accuracy - nearly as good as the full model!

**Norm contributions:**
- ||r2_A|| (x*x): 12.6
- ||r2_B|| (x*r1): 16.4
- ||r2_C|| (r1*r1): 15.4
- ||r2_full||: 22.0

### Position-Specific Ablation

| Position | Remove r1[i] | Remove r2[i] | Remove both |
|----------|--------------|--------------|-------------|
| 0 | 60.0% | 53.4% | 34.7% |
| 1 | 59.5% | 51.7% | 34.5% |
| 2 | 58.8% | 49.7% | 33.5% |
| 3 | 59.1% | 50.1% | 34.2% |
| 4 | 58.7% | 52.9% | 34.5% |
| **5** | **60.5%** | **14.4%** | 36.6% |
| 6 | 59.0% | 49.3% | 33.8% |
| 7 | 58.7% | 50.4% | 34.4% |
| 8 | 58.6% | 49.1% | 34.5% |
| 9 | 58.6% | 52.2% | 34.6% |

**Position 5 is catastrophic to remove from r2!** (60.5% → 14.4%)

### Position 5's Global Effect

When we remove position 5 from r2, how does it affect predictions for each TRUE position?

| True pos | Baseline | No r2[5] | Δ |
|----------|----------|----------|---|
| 0 | 50.3% | 3.7% | -46.6% |
| 1 | 61.9% | 4.6% | -57.3% |
| 2 | 66.9% | 5.7% | -61.2% |
| 3 | 67.9% | 5.3% | -62.6% |
| 4 | 57.9% | 4.6% | -53.3% |
| **5** | **27.4%** | **100.0%** | **+72.6%** |
| 6 | 70.4% | 4.4% | -66.0% |
| 7 | 71.6% | 6.2% | -65.4% |
| 8 | 72.6% | 6.1% | -66.5% |
| 9 | 58.7% | 3.7% | -55.0% |

**Critical finding: r2[5] is a GLOBAL SUPPRESSOR!**

- Removing r2[5] makes true_pos=5 **perfect** (100%)
- But it **destroys** all other positions (drops to 3-6%)
- Position 5 output is suppressing the correct answers for positions 0-4,6-9

This explains the poor accuracy for position 5 (27.4%): its own large magnitude suppresses the correct prediction.

### Position 2 (Control) - Normal Behavior

| True pos | Baseline | No r2[2] | Δ |
|----------|----------|----------|---|
| 0 | 50.3% | 35.9% | -14.4% |
| 1 | 61.9% | 46.5% | -15.4% |
| **2** | **66.9%** | **89.2%** | **+22.3%** |
| 3 | 67.9% | 55.0% | -12.9% |
| ... | ... | ... | ... |

Position 2 shows **localized** effect: removing r2[2] helps position 2, modestly hurts others. This is the expected behavior.

**Position 5 is fundamentally different** - it's the only output where:
- M1[5] has opposite "boost self" pattern
- M2[5] is rank-5 (others are rank-1)
- D2[5] has 5 non-zero entries (others have 1)
- r2[5] acts as a global suppressor

### Confusion Matrix by Position

| True | p=0 | p=1 | p=2 | p=3 | p=4 | p=5 | p=6 | p=7 | p=8 | p=9 | Acc |
|------|-----|-----|-----|-----|-----|-----|-----|-----|-----|-----|-----|
| t=0 | **526** | 42 | 49 | 65 | 30 | 110 | 68 | 54 | 62 | 40 | 50.3% |
| t=1 | 17 | **643** | 38 | 44 | 40 | 60 | 61 | 54 | 52 | 29 | 61.9% |
| t=2 | 24 | 33 | **660** | 46 | 42 | 30 | 33 | 40 | 40 | 39 | 66.9% |
| t=3 | 19 | 27 | 42 | **667** | 37 | 29 | 33 | 60 | 38 | 30 | 67.9% |
| t=4 | 24 | 32 | 57 | 53 | **575** | 84 | 46 | 41 | 55 | 26 | 57.9% |
| t=5 | 55 | 73 | 75 | 94 | 76 | **274** | 99 | 70 | 107 | 76 | **27.4%** |
| t=6 | 26 | 25 | 40 | 46 | 35 | 24 | **722** | 38 | 41 | 29 | 70.4% |
| t=7 | 22 | 40 | 33 | 33 | 27 | 27 | 39 | **707** | 36 | 24 | 71.6% |
| t=8 | 23 | 29 | 37 | 30 | 29 | 23 | 32 | 42 | **701** | 19 | 72.6% |
| t=9 | 25 | 43 | 47 | 36 | 38 | 67 | 41 | 50 | 56 | **573** | 58.7% |

### Per-Position Accuracy

| Position | Accuracy | Notes |
|----------|----------|-------|
| 0 | 50.3% | Below average |
| 1 | 61.9% | Average |
| 2 | 66.9% | Good |
| 3 | 67.9% | Good |
| 4 | 57.9% | Below average |
| **5** | **27.4%** | **Very poor!** |
| 6 | 70.4% | Good |
| 7 | 71.6% | Good |
| 8 | 72.6% | Best |
| 9 | 58.7% | Below average |

**Position 5 is dramatically worse** (27.4% vs 60.5% overall). This aligns with earlier findings:
- M1[5] has the **opposite** pattern (negative self, positive others)
- M2[5] is **rank-5** (the only non-rank-1 M2)
- D2[5] has 5 non-zero entries vs 1 for other outputs

### Top Confusion Pairs

| True → Pred | Count | % of true class |
|-------------|-------|-----------------|
| 5 → 8 | 107 | 10.7% |
| 0 → 5 | 110 | 10.5% |
| 5 → 6 | 99 | 9.9% |
| 5 → 3 | 94 | 9.4% |
| 4 → 5 | 84 | 8.5% |

Position 5 errors dominate - both true=5 getting mispredicted everywhere, and other positions getting mispredicted as 5.

### Interpretation

Unlike the 3-layer model where layers build on each other:
- **Layer 1 provides minimal value** (only 9.4% alone, and removing it helps!)
- **Layer 2 does all the work** (61.7% alone, same as full model)
- The h2 input to Layer 2 includes r1, but the model could work with just x

This suggests the 2-layer sparse model has essentially learned to **bypass Layer 1** and rely entirely on Layer 2's sparse rank-1 quadratic forms.

---

## Interpretation

### Why This Sparsity Pattern?

**Layer 1 must stay dense** because:
1. It processes raw input - needs all 10 input dimensions
2. The M1 "boost self, suppress others" pattern requires full-rank structure
3. Sparse L1 would lose information about some input positions

**Layer 2 can be sparse** because:
1. It operates on Layer 1's processed features, not raw input
2. Each output only needs to "select" one rank dimension from L2
3. Rank-1 M2 matrices are sufficient for the final classification

### The Algorithm

**Stage 1 (Layer 1)**: Dense quadratic feature extraction
- Computes `r1_i = h^T M1[i] h` for each output i
- M1[i] boosts squared contribution of input i, suppresses others
- Creates 10 intermediate features capturing "how much does position i stand out?"

**Stage 2 (Layer 2)**: Sparse selection
- For outputs 0,1,2,3,4,6,7,8,9: Uses a **single rank-1 quadratic form**
  - M2[i] = -|d| * (L2[r] ⊗ L2[r]) for some specific rank r
  - Computes a single quadratic combination of Layer 1 features
- For output 5: Uses rank-5 combination (more complex decision boundary)

### Connection to N=4 Patterns

The M1 "boost self, suppress others" is analogous to the n=4 structure:
- N=4: M1[i][i,i] positive, M1[i][j,j] negative (j≠i), off-diag ~0
- N=10: Same pattern! M1[i][i,i] ≈ +0.1, others ≈ -0.08, off-diag ≈ 0

The main difference: At n=10, the network needs more capacity, but pruning reveals that Layer 2 can collapse to mostly rank-1 operations.

---

## Summary

| Property | Layer 1 | Layer 2 |
|----------|---------|---------|
| Sparsity | Low (4%) | High (78%) |
| Role | Feature extraction | Selection |
| M structure | Full rank, "boost self" | Mostly rank-1 |
| Required for | Input coverage | Output decision |

The sparse network achieves 61.3% accuracy (vs 63.7% baseline) by:
1. Keeping Layer 1 dense for full input processing
2. Collapsing Layer 2 to mostly rank-1 quadratic forms
3. Each output position needs only a single "selection" operation from the processed features

---

## Computation Examples

Three detailed examples showing the step-by-step computation through the network, with r2 decomposed into its A (x*x), B (x*r1 cross-term), and C (r1*r1) components.

### Example 1: Target = position 9

**Input:**
```
x     = [ +1.927  +1.487  +0.901  -2.106  +0.678  -1.235  -0.043  -1.605  -0.752  +1.649]
         pos 0   pos 1   pos 2   pos 3   pos 4   pos 5   pos 6   pos 7   pos 8   pos 9

Max at position 0 (1.927), 2nd max at position 9 (1.649) → Target = 9
```

**Layer 1:**
```
rms1 = 1.3789
h1    = [ +1.769  +1.366  +0.827  -1.933  +0.623  -1.134  -0.040  -1.474  -0.691  +1.514]
r1    = [ -0.773  -1.274  -1.394  +0.215  -1.778  +2.134  -1.233  -0.320  -0.965  -1.371]
```

**Layer 2:**
```
rms2 = 1.2478
h2    = [ +2.914  +0.540  -1.245  -4.776  -2.777  +2.271  -3.222  -4.862  -4.338  +0.700]
```

**R2 Decomposition:**
```
r2_A  = [ -7.985  -8.075  -5.730 -14.678  -0.938  -6.191  -0.030  -6.304  -1.614 -12.428]
        (x*x term: h2_x^T M2 h2_x)

r2_B  = [ +5.572 +13.803 +17.186  -1.378  +5.564  +9.424  +1.202  -7.782  -6.818 +17.842]
        (x*r1 cross-term: 2 * h2_x^T M2 h2_r1) ← DOMINANT TERM

r2_C  = [ -0.972  -5.899 -12.887  -0.032  -8.251  -5.261 -11.844  -2.402  -7.200  -6.404]
        (r1*r1 term: h2_r1^T M2 h2_r1)

r2    = [ -3.385  -0.170  -1.431 -16.089  -3.625  -2.028 -10.673 -16.488 -15.632  -0.990]
        (full r2 = A + B + C)
```

**Final Output:**
```
out   = [ -2.231  +0.043  -1.924 -17.980  -4.724  -1.129 -11.949 -18.413 -17.350  -0.712]
        prediction: 1 (target: 9) WRONG
```

**Analysis:** The B term (cross-term) strongly boosts positions 1, 2, and 9. However, position 1 ends up winning due to x[1] being relatively high. The model incorrectly predicts position 1.

---

### Example 2: Target = position 5

**Input:**
```
x     = [ -0.392  -1.404  -0.728  -0.559  -0.769  +0.762  +1.642  -0.160  -0.497  +0.440]

Max at position 6 (1.642), 2nd max at position 5 (0.762) → Target = 5
```

**Layer 1:**
```
rms1 = 0.8546
h1    = [ -0.582  -2.080  -1.079  -0.829  -1.139  +1.130  +2.434  -0.236  -0.737  +0.651]
r1    = [ -2.066  -0.658  -1.306  -1.437  -1.647  +1.355  +1.550  -1.334  -1.401  -1.259]
```

**Layer 2:**
```
rms2 = 2.1315
h2    = [ -3.635  -3.048  -3.008  -2.952  -3.573  +3.130  +4.720  -2.209  -2.808  -1.211]
```

**R2 Decomposition:**
```
r2_A  = [ -0.111  -2.689  -1.499  -0.974  -0.419  -0.695  -3.128  -0.112  -0.666  -0.152]
r2_B  = [ -1.257  -2.902  -5.065  -4.254  -2.050  -1.297  -5.902  -1.215  -3.139  +0.977]
r2_C  = [ -3.544  -0.783  -4.278  -4.647  -2.510  -1.997  -2.783  -3.293  -3.702  -1.572]
r2    = [ -4.912  -6.374 -10.842  -9.874  -4.979  -3.988 -11.813  -4.621  -7.507  -0.747]
```

**Final Output:**
```
out   = [ -7.371  -8.435 -12.876 -11.870  -7.395  -1.871  -8.621  -6.115  -9.406  -1.566]
        prediction: 9 (target: 5) WRONG
```

**Analysis:** Position 5 is the target, but r2[5] only suppresses to -3.988 while r2[9] is less negative at -0.747. The B term at position 9 is actually positive (+0.977), boosting it. The model incorrectly predicts position 9. This illustrates position 5's problematic behavior.

---

### Example 3: Target = position 7 ✓

**Input:**
```
x     = [ -0.758  +1.078  +0.801  +1.681  +1.279  +1.296  +0.610  +1.335  -0.232  +0.042]

Max at position 3 (1.681), 2nd max at position 7 (1.335) → Target = 7
```

**Layer 1:**
```
rms1 = 1.0358
h1    = [ -0.927  +1.318  +0.979  +2.054  +1.564  +1.585  +0.746  +1.632  -0.283  +0.051]
r1    = [ +0.825  -0.973  -0.683  -0.582  -1.028  +1.932  -0.353  -0.712  +0.119  -0.758]
```

**Layer 2:**
```
rms2 = 1.1270
h2    = [ +0.186  +0.295  +0.331  +3.073  +0.703  +9.028  +0.721  +1.742  -0.315  -2.004]
```

**R2 Decomposition:**
```
r2_A  = [ -2.499  -3.683  -1.510 -10.450  -4.366  -2.783  -0.642  -4.835  -1.245  -0.337]
        (x*x: Large negative at position 3 because max is there)

r2_B  = [ +5.463  +8.297  +5.815 +14.258  +7.696  +3.057  +3.294 +10.569  -1.076  -1.714]
        (cross-term: Positive boost across many positions, especially 3 and 7)

r2_C  = [ -2.986  -4.673  -5.598  -4.863  -3.391  -3.719  -4.222  -5.775  -0.233  -2.178]
        (r1*r1: Mostly negative suppression)

r2    = [ -0.022  -0.059  -1.293  -1.055  -0.062  -3.445  -1.570  -0.042  -2.554  -4.229]
```

**Final Output:**
```
out   = [ +0.045  +0.047  -1.175  +0.043  +0.190  -0.217  -1.312  +0.581  -2.667  -4.945]
        prediction: 7 (target: 7) CORRECT
```

**Analysis:** The B term provides a strong boost to position 7 (+10.569), and the combined r2 leaves position 7 near zero (-0.042) while other positions get more suppressed. The final output correctly predicts position 7.

---

### Key Observations from Examples

1. **B (cross-term) dominates**: In all examples, the B term (2·h2_x^T M2 h2_r1) has the largest magnitude and most position-specific structure.

2. **A term (x*x) tends to suppress the max**: Because the maximum input has large squared value, r2_A tends to be most negative at the argmax position, helping to eliminate it.

3. **Cancellation is critical**: The final r2 depends on delicate cancellation between A, B, and C. When this cancellation fails (as in examples 1 and 2), the model makes errors.

4. **Position 5 pathology**: Example 2 shows how position 5's unusual structure (rank-5 M2[5], global suppressor behavior) leads to incorrect predictions when target=5.

---

## Files

- Checkpoint: `checkpoints/2layer_n10_r10_seed8_sparse.pkl`
- Analysis script: `analysis/analyze_2layer_n10_sparse.py`
- Visualizations: `images/sparse_2layer_n10/`
  - `weights_L1_D1_L2_D2.png` - All weight matrices with values
  - `M1_all.png` - All 10 M1 quadratic forms
  - `M2_all.png` - All 10 M2 quadratic forms with rank
  - `M1_M2_output0_detailed.png` - Detailed M1[0] and M2[0]
  - `confusion_matrix.png` - Confusion matrix (counts and %)
  - `per_position_accuracy.png` - Per-position accuracy bar chart
