# %%
"""
Train symmetric bilinear model: D @ (L @ x)²

No R matrix - activation is always positive (squared).
This makes interpretation much cleaner.
"""

import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import pickle

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


class SymmetricBilinearLayer(nn.Module):
    """
    Symmetric bilinear layer: D @ (L @ x)²

    Activation is always non-negative (squared).
    """
    def __init__(self, dim: int, rank: int):
        super().__init__()
        self.dim = dim
        self.rank = rank
        self.L = nn.Parameter(torch.randn(rank, dim) * 0.1)
        self.D = nn.Parameter(torch.randn(dim, rank) * 0.1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        Lx = x @ self.L.T  # (batch, rank)
        return (Lx ** 2) @ self.D.T  # (batch, dim)


class SymmetricBilinearResidual(nn.Module):
    """
    Symmetric bilinear residual network.
    h_new = h + D @ (L @ norm(h))²
    """
    def __init__(self, n: int, num_layers: int = 2, rank: int = 4):
        super().__init__()
        self.n = n
        self.num_layers = num_layers
        self.rank = rank

        self.norms = nn.ModuleList([RMSNorm(n) for _ in range(num_layers)])
        self.layers = nn.ModuleList([
            SymmetricBilinearLayer(n, rank) for _ in range(num_layers)
        ])

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = x
        for i in range(self.num_layers):
            h_norm = self.norms[i](h)
            h = h + self.layers[i](h_norm)
        return h


def task_2nd_argmax(x: torch.Tensor) -> torch.Tensor:
    return x.argsort(-1)[..., -2]


# %%
# =============================================================================
# TRAINING
# =============================================================================

n = 4
num_layers = 2
rank = 4
seed = 4
device = "cuda" if torch.cuda.is_available() else "cpu"

print(f"Config: n={n}, layers={num_layers}, rank={rank}, seed={seed}")
print(f"Model: Symmetric bilinear D @ (L @ x)²")
print(f"Device: {device}")

torch.manual_seed(seed)
model = SymmetricBilinearResidual(n, num_layers, rank).to(device)
optimizer = torch.optim.AdamW(model.parameters(), lr=0.01, weight_decay=0.001)

n_params = sum(p.numel() for p in model.parameters())
print(f"Parameters: {n_params}")

# Training
steps = 10000
history = {'steps': [], 'loss': [], 'eval_acc': []}

print("\nTraining...")
for step in range(steps + 1):
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
            print(f"Step {step:5d}: Loss={loss.item():.4f}, Acc={eval_acc:.3f}")

print(f"\nFinal accuracy: {history['eval_acc'][-1]:.1%}")

# %%
# Plot training curves
fig, axes = plt.subplots(1, 2, figsize=(12, 4))

ax1 = axes[0]
ax1.plot(history['steps'], history['eval_acc'], 'b-', linewidth=2)
ax1.set_xlabel('Step')
ax1.set_ylabel('Accuracy')
ax1.set_title('Symmetric Bilinear: Accuracy')
ax1.set_ylim(0, 1)
ax1.grid(True, alpha=0.3)

ax2 = axes[1]
ax2.plot(history['steps'], history['loss'], 'r-', linewidth=2)
ax2.set_xlabel('Step')
ax2.set_ylabel('Loss')
ax2.set_title('Symmetric Bilinear: Loss')
ax2.grid(True, alpha=0.3)

plt.tight_layout()
plt.show()

# %%
# Save model
save_path = Path("symmetric_bilinear_seed4.pt")
torch.save({
    'state_dict': {k: v.cpu() for k, v in model.state_dict().items()},
    'config': {'n': n, 'num_layers': num_layers, 'rank': rank, 'seed': seed},
    'history': history,
    'final_acc': history['eval_acc'][-1],
}, save_path)
print(f"Saved to {save_path}")

# %%
# =============================================================================
# ANALYZE WEIGHTS
# =============================================================================
print("\n" + "="*60)
print("WEIGHT ANALYSIS")
print("="*60)

state_dict = {k: v.cpu() for k, v in model.state_dict().items()}

L1 = state_dict['layers.0.L']
D1 = state_dict['layers.0.D']
L2 = state_dict['layers.1.L']
D2 = state_dict['layers.1.D']

print(f"L1: {L1.shape}, D1: {D1.shape}")
print(f"L2: {L2.shape}, D2: {D2.shape}")

# Visualize weights
fig, axes = plt.subplots(2, 2, figsize=(10, 8))

def plot_mat(ax, mat, title):
    vmax = max(abs(mat.min()), abs(mat.max()))
    im = ax.imshow(mat.numpy(), cmap='RdBu_r', vmin=-vmax, vmax=vmax, aspect='auto')
    ax.set_title(title)
    for i in range(mat.shape[0]):
        for j in range(mat.shape[1]):
            val = mat[i, j].item()
            color = 'white' if abs(val) > vmax * 0.6 else 'black'
            ax.text(j, i, f'{val:.2f}', ha='center', va='center', fontsize=8, color=color)
    plt.colorbar(im, ax=ax, shrink=0.8)

plot_mat(axes[0, 0], L1, 'L1 (rank x n)')
plot_mat(axes[0, 1], D1, 'D1 (n x rank)')
plot_mat(axes[1, 0], L2, 'L2 (rank x n)')
plot_mat(axes[1, 1], D2, 'D2 (n x rank)')

plt.suptitle('Symmetric Bilinear Weights', fontsize=12)
plt.tight_layout()
plt.show()

# %%
# =============================================================================
# OPTIMAL INPUTS (much simpler now!)
# =============================================================================
print("\n" + "="*60)
print("OPTIMAL INPUTS FOR LAYER 1")
print("="*60)
print("""
For symmetric bilinear: activation_h = (L[h] · x)²

Optimal input for channel h: x* = ±L[h] / ||L[h]||
(Both + and - give the same positive activation)
""")

# Optimal inputs are just the L vectors (normalized)
x_opt_L1 = L1 / (L1.norm(dim=1, keepdim=True) + 1e-8)

print("Optimal inputs (= normalized L vectors):")
for h in range(rank):
    print(f"  Ch {h}: {x_opt_L1[h].numpy().round(3)}")

# Visualize
fig, axes = plt.subplots(1, rank, figsize=(4*rank, 4))

for h in range(rank):
    ax = axes[h]
    x_opt = x_opt_L1[h].numpy()
    colors = ['green' if v > 0 else 'red' for v in x_opt]
    ax.bar(range(n), x_opt, color=colors, edgecolor='black')
    ax.set_title(f'L1 Ch {h}: x* = L[{h}]', fontsize=11)
    ax.axhline(y=0, color='k', linewidth=0.5)
    ax.set_ylim(-1, 1)
    ax.set_xlabel('Position')

plt.suptitle('Layer 1: Optimal Inputs (normalized L vectors)', fontsize=12)
plt.tight_layout()
plt.show()

# %%
# What do these optimal inputs produce?
print("\n" + "="*60)
print("ACTIVATIONS FROM OPTIMAL INPUTS")
print("="*60)

def rmsnorm(x, weight, eps=1e-6):
    rms = torch.sqrt((x ** 2).mean(dim=-1, keepdim=True) + eps)
    return weight * (x / rms)

norm1_weight = state_dict['norms.0.weight']
norm2_weight = state_dict['norms.1.weight']

def get_L1_activations(x):
    x_norm = rmsnorm(x, norm1_weight)
    Lx = x_norm @ L1.T
    return Lx ** 2  # Always positive!

acts = get_L1_activations(x_opt_L1)
print("L1 activations (all positive now!):")
print(f"{'Input':<10} {'Act[0]':<10} {'Act[1]':<10} {'Act[2]':<10} {'Act[3]':<10}")
print("-" * 50)
for h in range(rank):
    a = acts[h].numpy()
    print(f"x* ch{h:<5} {a[0]:<10.3f} {a[1]:<10.3f} {a[2]:<10.3f} {a[3]:<10.3f}")

# %%
# Full forward pass
print("\n" + "="*60)
print("MODEL PREDICTIONS FOR OPTIMAL INPUTS")
print("="*60)

def forward(x):
    h = x
    h_norm = rmsnorm(h, norm1_weight)
    Lh = h_norm @ L1.T
    h = h + (Lh ** 2) @ D1.T

    h_norm = rmsnorm(h, norm2_weight)
    Lh = h_norm @ L2.T
    h = h + (Lh ** 2) @ D2.T
    return h

for h in range(rank):
    x_opt = x_opt_L1[h:h+1]
    logits = forward(x_opt)
    pred = logits.argmax().item()
    target = task_2nd_argmax(x_opt).item()
    status = '✓' if pred == target else '✗'
    logits_str = ', '.join([f'{v:.2f}' for v in logits[0].numpy()])
    print(f"Ch {h} optimal: pred={pred}, target={target} {status}, logits=[{logits_str}]")

# %%
# D matrix interpretation
print("\n" + "="*60)
print("D MATRIX INTERPRETATION")
print("="*60)
print("""
D1[i, h] = how much L1 channel h contributes to output position i
D2[i, h] = how much L2 channel h contributes to output position i

Since activations are always positive:
- Positive D means channel activation increases that output
- Negative D means channel activation decreases that output
""")

print("\nD1 (Layer 1 -> output contribution):")
print("        ", end="")
for i in range(n):
    print(f"pos{i:<6}", end="")
print()
for h in range(rank):
    print(f"Ch {h}:  ", end="")
    for i in range(n):
        val = D1[i, h].item()
        print(f"{val:>+7.3f} ", end="")
    print()

print("\nD2 (Layer 2 -> output contribution):")
print("        ", end="")
for i in range(n):
    print(f"pos{i:<6}", end="")
print()
for h in range(rank):
    print(f"Ch {h}:  ", end="")
    for i in range(n):
        val = D2[i, h].item()
        print(f"{val:>+7.3f} ", end="")
    print()

# %%
