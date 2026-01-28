# TN-Sim Analysis for Bilinear Networks

Analytic Tensor Network Similarity (TN-Sim) analysis for N-layer residual bilinear networks trained on SVHN.

## Architecture

```
x → Embed → [RMSNorm → Bilinear → (+residual)] × N → Projection → output
```

Where each bilinear layer is: `B(x) = D @ (Lx ⊙ Rx)`
- `L, R`: (d_hidden, d_res) - expand to hidden dimension
- `D`: (d_res, d_hidden) - project back to residual stream
- `⊙`: element-wise (Hadamard) product

## Key Files

### Core Analysis
| File | Description |
|------|-------------|
| `tnsim.py` | Analytic TN-sim module for 1-layer and 2-layer bilinear networks |
| `run_tnsim_1layer.py` | 1-layer analysis with effective Tucker ranks |
| `run_tnsim_2layer.py` | 2-layer analysis with proper 5th-order Tucker ranks |
| `analyze_1to4_layers_aug.py` | Multi-model comparison (1-4 layers) |

### Training
| File | Description |
|------|-------------|
| `train_1to4_layers_aug.py` | Train 1-4 layer models with augmentation on SVHN |

### Documentation
| File | Description |
|------|-------------|
| `PLOT_REMINDERS.md` | Plotting conventions (x-axis alignment, colorbars) |
| `tn_analysis_checkpoints/tensorify_norm_and_residual.md.sty` | Theory: tensorifying RMSNorm and residuals |

## TN-Sim Formulas

### 1-Layer Bilinear
3rd-order tensor: `T[i,j,k] = Σ_h D[i,h] L[h,j] R[h,k]`

Inner product: `<T_a|T_b> = (D_a.T @ D_b) * core).sum()`
where `core = 0.5 * (LL * RR + LR * RL)` (symmetrized)

### 2-Layer Bilinear
5th-order tensor from composition `B2(B1(x))`.

Inner product: `<T_a|T_b> = (DD2 * (A_a @ C1 @ A_b.T) * (B_a @ C1 @ B_b.T)).sum()`
where:
- `C1 = bilinear_core(L1, R1)` - layer 1 core
- `A = L2 @ D1`, `B = R2 @ D1` - layer 2 compositions
- `DD2 = D2_a.T @ D2_b` - layer 2 output contraction

### With Residual
For pre-norm residual `output = B(x/||x||) + x`, the residual term `x||x||^2` is a 3rd-order tensor with `L=R=D=I`. The full polynomial has terms of degrees 1, 2, 3, 4 (for 2-layer).

## Tucker Ranks

### 1-Layer (3 modes)
- Mode-1: output dimension
- Mode-2: input-L dimension
- Mode-3: input-R dimension

### 2-Layer (5 modes)
- Mode-1: output (n)
- Mode-2,4: input-L (j, p) - symmetric
- Mode-3,5: input-R (k, q) - symmetric

Gram matrices computed efficiently without materializing full tensor.

## Usage

```bash
# Train models
python train_1to4_layers_aug.py

# Run TN-sim analysis
python run_tnsim_1layer.py
python run_tnsim_2layer.py
```

## Key Findings

1. **Effective rank U-shape**: Per-layer effective ranks show compression (high→low→medium) characteristic of grokking
2. **5th-order Tucker ranks**: Grow monotonically as the composed function gains complexity
3. **Residual anchoring**: Identity residual provides stable baseline, masking early learning in TN-sim
