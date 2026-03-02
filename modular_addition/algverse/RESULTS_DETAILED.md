# Symmetric Bilinear Networks: Detailed Results

Task: 2nd-argmax (find position of 2nd largest element)
Model: `output = x + sum_i D_i @ (L_i @ norm(x))²`

---

## 1. Seed Sweeps (sweep_seeds_and_sparsity.py)

### n=5, 1-Layer, rank=5
| Seed | Accuracy |
|------|----------|
| 0 | 57.3% |
| 1 | 55.6% |
| 2 | 57.0% |
| 3 | 53.4% |
| 4 | 53.3% |
| 5 | 56.5% |
| 6 | 56.2% |
| 7 | 55.5% |
| 8 | 54.4% |
| **9** | **59.6%** |

**Best: seed 9, 59.6%**
- After L1+prune (t=0.15): 52% pruned, 56.6% acc

### n=6, 1-Layer, rank=6
| Seed | Accuracy |
|------|----------|
| 0 | 51.8% |
| 1 | 54.9% |
| 2 | 55.1% |
| 3 | 55.6% |
| **4** | **57.1%** |
| 5 | 52.4% |
| 6 | 52.3% |
| 7 | 51.6% |
| 8 | 56.9% |
| 9 | 54.9% |

**Best: seed 4, 57.1%**
- After L1+prune (t=0.05): 41.7% pruned, 55.0% acc

### n=5, 2-Layer, rank=5
| Seed | Accuracy |
|------|----------|
| 0 | 67.4% |
| 1 | 65.9% |
| **2** | **73.0%** |
| 3 | 66.5% |
| 4 | 64.0% |
| 5 | 64.9% |
| 6 | 69.0% |
| 7 | 64.4% |
| 8 | 72.3% |
| 9 | 62.8% |

**Best: seed 2, 73.0%**
- After L1+prune (t=0.10): 14% pruned, 72.9% acc

### n=6, 2-Layer, rank=6
| Seed | Accuracy |
|------|----------|
| 0 | 68.7% |
| 1 | 68.5% |
| 2 | 63.4% |
| 3 | 63.8% |
| **4** | **68.9%** |
| 5 | 68.4% |
| 6 | 64.7% |
| 7 | 65.1% |
| 8 | 59.3% |
| 9 | 65.8% |

**Best: seed 4, 68.9%**
- After L1+prune (t=0.15): 29.9% pruned, 68.1% acc

### n=5, 3-Layer, rank=5
| Seed | Accuracy |
|------|----------|
| 0 | 69.9% |
| 1 | 72.7% |
| **2** | **74.9%** |
| 3 | 66.1% |
| 4 | 68.0% |
| 5 | 66.5% |
| 6 | 71.1% |
| 7 | 71.2% |
| 8 | 73.1% |
| 9 | 66.8% |

**Best: seed 2, 74.9%**
- Mean: 70.0%, Std: 3.0%
- After L1+prune (t=0.15): 22% pruned, 74.8% acc

### n=6, 3-Layer, rank=6
| Seed | Accuracy |
|------|----------|
| 0 | 74.7% |
| 1 | 63.4% |
| **2** | **80.5%** |
| 3 | 61.1% |
| 4 | 77.2% |
| 5 | 64.9% |
| 6 | 75.3% |
| 7 | 66.1% |
| 8 | 58.1% |
| 9 | 66.2% |

**Best: seed 2, 80.5%**
- Mean: 68.8%, Std: 7.3% (high variance!)
- After L1+prune (t=0.05): 12.5% pruned, 82.6% acc

---

## 2. Weight Decay Study (sweep_wd_n5_3layer.py)

Config: n=5, 3-layer, rank=5

### wd=0.001
| Seed | Accuracy |
|------|----------|
| 0 | 70.9% |
| 1 | 71.2% |
| 2 | 77.2% |
| 3 | 69.9% |
| 4 | 71.5% |

**Mean: 72.1%, Std: 2.6%**

### wd=0.01
| Seed | Accuracy |
|------|----------|
| 0 | 69.5% |
| **1** | **90.2%** |
| 2 | 74.7% |
| 3 | 68.0% |
| 4 | 69.3% |

**Mean: 74.3%, Std: 8.2%**
**Seed 1 grokked to 90.2%!**

### wd sweep on seed 1
| Weight Decay | Final Acc |
|--------------|-----------|
| 0.01 | **90.7%** |
| 0.03 | 67.6% |
| 0.1 | 88.3% |
| 0.3 | 85.5% |
| 1.0 | 51.3% |

Note: wd=0.03 is a "dead zone" - worse than both lower and higher values.

---

## 3. Optimizer Comparison (grokking_investigation.py)

Config: n=5, 3-layer, rank=5, seed=1

| Optimizer | wd | Final Acc | Grok Step |
|-----------|-----|-----------|-----------|
| **AdamW** | 0.01 | **90.7%** | 19050 |
| AdamW | 0.03 | 67.6% | never |
| **AdamW** | 0.1 | **88.3%** | 9550 |
| Adam+L2 | 0.01 | 63.6% | never |
| Adam+L2 | 0.03 | 0.0% | never |
| Adam+L2 | 0.1 | 0.0% | never |
| Muon | any | 47.2% | never |
| SGD | 0.01 | 62.1% | never |
| SGD | 0.1 | 0.0% | never |

**Key finding: Only AdamW enables grokking!**

---

## 4. n=10 Sweep (sweep_n10_layers.py)

Settings: wd=0.01, 40k steps, 10 seeds each

### Summary Table
| Layers | Rank | Mean | Std | Min | Max | Grok Rate |
|--------|------|------|-----|-----|-----|-----------|
| 1 | 10 | 10.6% | 1.7% | 8.6% | 13.9% | 0/10 |
| 1 | 20 | 10.6% | 1.6% | 8.5% | 13.7% | 0/10 |
| 2 | 10 | 59.8% | 3.8% | 52.1% | 63.7% | 0/10 |
| 2 | 20 | 63.2% | 4.3% | 53.6% | 67.4% | 0/10 |
| 3 | 10 | 77.5% | 4.3% | 71.0% | 83.0% | 5/10 |
| 3 | 20 | 80.1% | 6.6% | 68.9% | 87.5% | 7/10 |
| 4 | 10 | 81.8% | 3.2% | 76.1% | 85.6% | 9/10 |
| **4** | **20** | **84.7%** | 2.9% | 79.5% | **89.2%** | **10/10** |

### 4-Layer, rank=20 (Best config for n=10)
| Seed | Accuracy | Grok Step |
|------|----------|-----------|
| 0 | 83.3% | 1800 |
| 1 | 89.1% | 600 |
| 2 | 83.3% | 1200 |
| 3 | 83.3% | 1000 |
| 4 | 82.1% | 2200 |
| 5 | 79.5% | 6400 |
| 6 | 85.1% | 800 |
| **7** | **89.2%** | **600** |
| 8 | 84.6% | 1400 |
| 9 | 87.3% | 1000 |

**Best: seed 7, 89.2% (grok@600)**

### 3-Layer, rank=10 (grokking details)
| Seed | Accuracy | Grok Step |
|------|----------|-----------|
| 0 | 73.0% | never |
| 1 | 71.0% | never |
| 2 | 76.8% | never |
| 3 | 75.4% | never |
| 4 | 78.8% | 28600 |
| 5 | 81.3% | 5800 |
| 6 | 82.0% | 7400 |
| 7 | 72.1% | never |
| **8** | **83.0%** | **3200** |
| 9 | 81.9% | 24400 |

---

## 5. L1 Pruning Results

### Pruning Threshold Sweep (within -1% accuracy)

| Config | Best Thresh | Pruned % | Pre-Prune Acc | Post-Prune Acc |
|--------|-------------|----------|---------------|----------------|
| n=5, 1L | t=0.15 | 52% | 54.8% | 56.6% |
| n=6, 1L | t=0.05 | 42% | 53.1% | 55.0% |
| n=5, 2L | t=0.10 | 14% | 72.9% | 72.9% |
| n=6, 2L | t=0.15 | 30% | 67.9% | 68.1% |
| n=5, 3L | t=0.15 | 22% | 75.6% | 74.8% |
| n=6, 3L | t=0.05 | 13% | 82.0% | 82.6% |

### Per-Matrix Sparsity (n=4, 2-layer, t=0.15)
| Matrix | Sparsity |
|--------|----------|
| L1 | 6% |
| D1 | 44% |
| L2 | 0% |
| D2 | 56% |

Observation: D matrices are more compressible than L matrices.

---

## 6. Computation Path Ablation (n=4, 2-layer)

Decomposition: `output = x + r1 + A + B + C`
- x: direct input
- r1: layer 1 output
- A: D2 @ (L2 @ x)² (x×x term)
- B: cross term (x×r1)
- C: D2 @ (L2 @ r1)² (r1×r1 term)

| Component Removed | Accuracy |
|-------------------|----------|
| None (full) | 87.4% |
| Remove x | 68.0% |
| Remove r1 | 86.8% |
| Remove A | 14.0% |
| Remove B | 56.3% |
| Remove C | 87.3% |

**Key finding: Component A (layer 2 seeing x directly) is critical.**

---

## 7. Checkpoints Available

| File | Config | Accuracy |
|------|--------|----------|
| 1layer_n3_r3_seed0.pt | n=3, 1L, r=3 | 99.5% |
| 1layer_n4_r4_seed4.pt | n=4, 1L, r=4 | 70.0% |
| 1layer_n5_r5_seed9.pt | n=5, 1L, r=5 | 41.9% |
| symmetric_bilinear_seed4.pt | n=4, 2L, r=4 | 86.8% |
| prune_symmetric_results.pkl | n=4, 2L pruned | various |

---

## 8. Known Good Seeds Reference

```python
GOOD_SEEDS = {
    (3, 1): 0,   # 99.5%
    (4, 1): 4,   # 70.0%
    (5, 1): 9,   # 59.6%
    (6, 1): 4,   # 57.1%
    (5, 2): 2,   # 73.0%
    (6, 2): 4,   # 68.9%
    (5, 3): 2,   # 74.9% (or seed 1 with wd=0.01 for 90.2%)
    (6, 3): 2,   # 80.5%
    (10, 2): 8,  # 63.7%
    (10, 3): 8,  # 83.0%
    (10, 4): 7,  # 89.2% (r=20)
}
```
