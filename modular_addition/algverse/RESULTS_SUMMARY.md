# Symmetric Bilinear Networks: Best Results Summary

Task: 2nd-argmax (find position of 2nd largest element in n-dimensional input)

## Best Configurations by n

| n | Layers | Rank | Best Seed | Accuracy | Notes |
|---|--------|------|-----------|----------|-------|
| 3 | 1 | 3 | 0 | 99.5% | Trivial task |
| 4 | 2 | 4 | 4 | 87.7% | From prune_seed4 |
| 5 | 3 | 5 | 1 | **90.2%** | Requires wd=0.01, grokking |
| 6 | 3 | 6 | 2 | 82.6% | After L1+prune |
| 10 | 4 | 20 | 7 | **89.2%** | Grok@600 steps |

## Key Findings

1. **Grokking requires ≥3 layers** for n≥5
   - 2-layer models plateau at 60-70% and never grok
   - 3+ layer models can grok to 80%+

2. **AdamW weight decay is essential**
   - Only AdamW triggers grokking
   - Adam + manual L2 loss fails completely
   - Optimal wd ≈ 0.01

3. **Doubling rank helps**
   - rank=2n consistently outperforms rank=n
   - Especially for deeper models

4. **1-layer capacity**
   - n=3: 99.5% (trivial)
   - n=4: 70%
   - n=5: 60%
   - n=10: 10% (random)

## Recommended Training Settings

```python
optimizer = torch.optim.AdamW(model.parameters(), lr=0.01, weight_decay=0.01)
num_steps = 40000  # for grokking
batch_size = 128
```

## Sparsity (L1 + Prune)

Best pruning thresholds (within -1% accuracy):

| Config | Threshold | Pruned % | Final Acc |
|--------|-----------|----------|-----------|
| n=5, 2L | t=0.10 | 14% | 72.9% |
| n=6, 2L | t=0.15 | 30% | 68.1% |
| n=5, 3L | t=0.15 | 22% | 74.8% |
| n=6, 3L | t=0.05 | 13% | 82.6% |
