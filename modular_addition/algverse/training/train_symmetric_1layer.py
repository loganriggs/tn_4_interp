# %%
"""
1-Layer Symmetric Bilinear Model: D @ (L @ x)²

Compare with 2-layer model to understand what each layer contributes.
"""

import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt

# %%
# =============================================================================
# MODEL DEFINITION
# =============================================================================

class RMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        rms = torch.sqrt((x ** 2).mean(dim=-1, keepdim=True) + self.eps)
        return self.weight * (x / rms)


class SymmetricBilinear1Layer(nn.Module):
    def __init__(self, n: int, rank: int = 4):
        super().__init__()
        self.n = n
        self.rank = rank
        self.norm = RMSNorm(n)
        self.L = nn.Parameter(torch.randn(rank, n) * 0.1)
        self.D = nn.Parameter(torch.randn(n, rank) * 0.1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.norm(x)
        Lh = h @ self.L.T  # (batch, rank)
        return x + (Lh ** 2) @ self.D.T  # (batch, n)


def task_2nd_argmax(x):
    return x.argsort(-1)[..., -2]


# %%
# Configuration
n = 4
rank = 4
seed = 4
device = "cuda" if torch.cuda.is_available() else "cpu"

print(f"Config: n={n}, rank={rank}, seed={seed}")
print(f"Device: {device}")

# %%
# =============================================================================
# TRAINING
# =============================================================================
print("\n" + "="*60)
print("TRAINING 1-LAYER SYMMETRIC BILINEAR")
print("="*60)

torch.manual_seed(seed)
model = SymmetricBilinear1Layer(n, rank).to(device)
optimizer = torch.optim.AdamW(model.parameters(), lr=0.01, weight_decay=0.001)

history = {'steps': [], 'loss': [], 'eval_acc': []}

for step in range(10001):
    model.train()
    x = torch.randn(128, n, device=device)
    targets = task_2nd_argmax(x)
    logits = model(x)
    loss = nn.functional.cross_entropy(logits, targets)

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    if step % 500 == 0:
        model.eval()
        with torch.no_grad():
            x_eval = torch.randn(10000, n, device=device)
            targets_eval = task_2nd_argmax(x_eval)
            logits_eval = model(x_eval)
            eval_acc = (logits_eval.argmax(-1) == targets_eval).float().mean().item()

        history['steps'].append(step)
        history['loss'].append(loss.item())
        history['eval_acc'].append(eval_acc)

        if step % 2000 == 0:
            print(f"Step {step:5d}: Loss={loss.item():.4f}, Acc={eval_acc:.1%}")

final_acc = eval_acc
print(f"\nFinal accuracy: {final_acc:.1%}")

# %%
# =============================================================================
# WEIGHT ANALYSIS
# =============================================================================
print("\n" + "="*60)
print("WEIGHT ANALYSIS")
print("="*60)

L = model.L.detach().cpu()
D = model.D.detach().cpu()
norm_weight = model.norm.weight.detach().cpu()

print(f"\nRMSNorm weights: {norm_weight.numpy()}")
print(f"\nL matrix (rank x n = {rank} x {n}):")
print(L.numpy().round(3))
print(f"\nD matrix (n x rank = {n} x {rank}):")
print(D.numpy().round(3))

# %%
# Compute effective bilinear tensor T[i,j,k] = sum_c D[i,c] * L[c,j] * L[c,k]
print("\n" + "="*60)
print("BILINEAR TENSOR")
print("="*60)

# T_ijk = sum_c D_ic L_cj L_ck
T = torch.einsum('ic,cj,ck->ijk', D, L, L)
print(f"Shape: {T.shape}")

# For 2nd-argmax, we want:
# output[i] high when x[i] is the 2nd largest
# This requires detecting "there exists j > i such that x[j] > x[i]" and
# "for all k != i,j, x[k] < x[i]"

# Let's look at the diagonal and off-diagonal structure
print("\nDiagonal terms T[i,i,i] (self-cubed contribution):")
for i in range(n):
    print(f"  T[{i},{i},{i}] = {T[i,i,i].item():.3f}")

print("\nT[i,j,j] terms (output i from input j squared):")
for i in range(n):
    row = [T[i,j,j].item() for j in range(n)]
    print(f"  output {i}: {[f'{v:.3f}' for v in row]}")

# %%
# Visualize weights
fig, axes = plt.subplots(1, 3, figsize=(15, 4))

# L matrix
ax1 = axes[0]
vmax = max(abs(L.min()), abs(L.max()))
im1 = ax1.imshow(L, cmap='RdBu_r', vmin=-vmax, vmax=vmax, aspect='auto')
ax1.set_xlabel('Input dimension')
ax1.set_ylabel('Channel')
ax1.set_title('L matrix')
for i in range(rank):
    for j in range(n):
        ax1.text(j, i, f'{L[i,j]:.2f}', ha='center', va='center', fontsize=9)
plt.colorbar(im1, ax=ax1, shrink=0.8)

# D matrix
ax2 = axes[1]
vmax = max(abs(D.min()), abs(D.max()))
im2 = ax2.imshow(D, cmap='RdBu_r', vmin=-vmax, vmax=vmax, aspect='auto')
ax2.set_xlabel('Channel')
ax2.set_ylabel('Output dimension')
ax2.set_title('D matrix')
for i in range(n):
    for j in range(rank):
        ax2.text(j, i, f'{D[i,j]:.2f}', ha='center', va='center', fontsize=9)
plt.colorbar(im2, ax=ax2, shrink=0.8)

# D @ L composition (what each output sees from each input squared)
DL = D @ L  # (n, n) - output i's weight on input j (through squaring)
ax3 = axes[2]
vmax = max(abs(DL.min()), abs(DL.max()))
im3 = ax3.imshow(DL, cmap='RdBu_r', vmin=-vmax, vmax=vmax, aspect='auto')
ax3.set_xlabel('Input dimension')
ax3.set_ylabel('Output dimension')
ax3.set_title('D @ L (effective linear on L@x)')
for i in range(n):
    for j in range(n):
        ax3.text(j, i, f'{DL[i,j]:.2f}', ha='center', va='center', fontsize=9)
plt.colorbar(im3, ax=ax3, shrink=0.8)

plt.suptitle(f'1-Layer Symmetric Bilinear Weights (acc={final_acc:.1%})')
plt.tight_layout()
plt.show()

# %%
# =============================================================================
# OPTIMAL INPUTS
# =============================================================================
print("\n" + "="*60)
print("OPTIMAL INPUTS FOR EACH CHANNEL")
print("="*60)

# For symmetric bilinear, optimal input for channel c is L[c] (normalized)
print("\nOptimal inputs (L rows, normalized):")
for c in range(rank):
    L_c = L[c]
    L_c_norm = L_c / L_c.norm()
    print(f"  Channel {c}: {L_c_norm.numpy().round(3)}")

# %%
# =============================================================================
# COMPARISON WITH 2-LAYER ABLATION
# =============================================================================
print("\n" + "="*60)
print("COMPARISON WITH 2-LAYER ABLATION")
print("="*60)

print("""
In ablation_computation_paths.py, component "A" represents:
  A = D2 @ (L2 @ x_norm)²

This is structurally identical to a 1-layer symmetric bilinear model.
However, the ablation "A only" achieved ~50% accuracy while this
1-layer model achieves 70.8%.

KEY DIFFERENCE:
- Ablation "A": Uses weights trained as part of 2-layer model
  (L2/D2 were trained to work WITH r1 via the cross-term B)
- 1-layer model: Weights trained specifically for 1-layer setting

This suggests that L2/D2 in the 2-layer model specialize for
a different computation than what's optimal for x-only processing.
The 2-layer model delegates some "x detection" to layer 1 (via r1),
allowing layer 2 to focus on combining x with r1.
""")

# %%
# Save model
save_path = 'symmetric_bilinear_1layer_seed4.pt'
torch.save({
    'state_dict': model.state_dict(),
    'config': {'n': n, 'rank': rank, 'seed': seed},
    'history': history,
    'final_acc': final_acc,
}, save_path)
print(f"\nSaved to {save_path}")

# %%
