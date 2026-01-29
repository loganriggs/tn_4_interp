import torch
import torch.nn as nn
from itertools import permutations

"""
Try to find ANY degree-2 polynomial that solves 2nd argmax for n=4.
Spoiler: it won't work.
"""

# Parameterize a general quadratic: logit[i] = Σ_j a[i,j] x[j] + Σ_{j,k} b[i,j,k] x[j] x[k]
# For simplicity, use bilinear form: logit = D @ (L @ x * R @ x)

n = 5
rank = 20  # Give it plenty of capacity

L = nn.Parameter(torch.randn(rank, n) * 0.1)
R = nn.Parameter(torch.randn(rank, n) * 0.1)
D = nn.Parameter(torch.randn(n, rank) * 0.1)
L2 = nn.Parameter(torch.randn(rank, n) * 0.1)
R2 = nn.Parameter(torch.randn(rank, n) * 0.1)
D2 = nn.Parameter(torch.randn(n, rank) * 0.1)
# Also add linear term for full generality
W = nn.Parameter(torch.randn(n, n) * 0.1)
W2 = nn.Parameter(torch.randn(n, n) * 0.1)

# bias = nn.Parameter(torch.randn(n) * 0.1)

def model(x):
    # bilinear = (x @ L.T) * (x @ R.T)  # (batch, rank)
    bilinear = (x @ L.T).square()  # (batch, rank)

    return x @ W + bilinear @ (D.T)   # (batch, n)
    # return bilinear @ D.T   # (batch, n)
def model2(x):
    bilinear = (x @ L.T) * (x @ R.T)  # (batch, rank)
    # bilinear = (x @ L.T).square()  # (batch, rank)
    x = x @ W + bilinear @ (D.T)
    bilinear_2 = (x @ L.T) * (x @ R.T)
    return  x @ W2 + bilinear_2 @ (D2.T)
    # return bilinear @ D.T   # (batch, n)

def task_2nd_argmax(x):
    return x.argsort(-1)[..., -2]

optimizer = torch.optim.Adam([L, R, D, W, L2, R2, D2, W2], lr=0.01)

for step in range(10_000):
    x = torch.randn(512, n)
    targets = task_2nd_argmax(x)
    logits = model2(x)
    loss = nn.functional.cross_entropy(logits, targets)
    
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    
    if step % 2000 == 0:
        acc = (logits.argmax(-1) == targets).float().mean()
        print(f"Step {step}: loss={loss.item():.3f}, acc={acc.item():.3f}")

# Final test
with torch.no_grad():
    x = torch.randn(10000, n)
    targets = task_2nd_argmax(x)
    logits = model2(x)
    acc = (logits.argmax(-1) == targets).float().mean()
    print(f"\nFinal accuracy: {acc.item():.3f}")
    print(f"Random baseline: {1/n:.3f}")

# ```

# Running this:
# ```
# Step 0: loss=1.405, acc=0.238
# Step 2000: loss=1.155, acc=0.387
# Step 4000: loss=1.145, acc=0.385
# ...
# Step 18000: loss=1.143, acc=0.391

# Final accuracy: 0.386
# Random baseline: 0.250