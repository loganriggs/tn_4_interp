# %%
"""
Hand-Coded Weight Verification

Compare idealized weight patterns against learned weights:
1. Accuracy overlap (do they agree on same examples?)
2. Output similarity (cosine similarity of logits)
3. Tensor similarity of weight matrices
"""

import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import sys
import pickle

# =============================================================================
# PROJECT ROOT
# =============================================================================
PROJECT_ROOT = Path("/workspace/tn_4_interp/modular_addition/algverse")
sys.path.insert(0, str(PROJECT_ROOT))
from models import task_2nd_argmax
from analysis.analysis_utils import (
    compute_quadratic_forms, bilinear_forward, rmsnorm,
    bilinear_as_quadratic
)

checkpoint_dir = PROJECT_ROOT / "checkpoints"
images_dir = PROJECT_ROOT / "images"

# %%
# =============================================================================
# DEFINE IDEALIZED WEIGHT PATTERNS
# =============================================================================
n = 4
rank = 4

# --- Layer 1 ---
# L1: "Deviation from mean" detectors
# Each row has +1 on diagonal, -1/3 elsewhere
L1_ideal = torch.zeros(rank, n)
for i in range(rank):
    for j in range(n):
        if i == j:
            L1_ideal[i, j] = 1.0
        else:
            L1_ideal[i, j] = -1/3

# D1: "Leave-one-out" pattern
# -1 everywhere except 0 on diagonal: -1 * (11^T - I)
D1_ideal = torch.zeros(n, rank)
for i in range(n):
    for j in range(rank):
        if i == j:
            D1_ideal[i, j] = 0.0
        else:
            D1_ideal[i, j] = -1.0

# γ1: downscale layer 1
# TUNED: 0.5 works better than 0.7
gamma1_ideal = 0.5

# --- Layer 2 ---
# L2: Same as L1, scaled by α
# TUNED: α=1.0 (no scaling) works better than 1.3
alpha = 1.0
L2_ideal = alpha * L1_ideal.clone()

# D2: Large negative diagonal, small positive off-diagonal
# -β*I + γ*(11^T - I)
# TUNED: β=1.3, γ=0.2 works better than β=1.5, γ=0.3
beta = 1.3
gamma_off = 0.2
D2_ideal = torch.zeros(n, rank)
for i in range(n):
    for j in range(rank):
        if i == j:
            D2_ideal[i, j] = -beta
        else:
            D2_ideal[i, j] = gamma_off

# γ2: upscale layer 2
# TUNED: 1.5 works better than 1.7
gamma2_ideal = 1.5

print("Idealized Weight Patterns:")
print("=" * 60)
print(f"\nL1 (deviation from mean):\n{L1_ideal}")
print(f"\nD1 (leave-one-out):\n{D1_ideal}")
print(f"\nγ1 = {gamma1_ideal}")
print(f"\nL2 (scaled deviation, α={alpha}):\n{L2_ideal}")
print(f"\nD2 (β={beta}, γ={gamma_off}):\n{D2_ideal}")
print(f"\nγ2 = {gamma2_ideal}")

# %%
# =============================================================================
# LOAD LEARNED WEIGHTS
# =============================================================================
prune_path = checkpoint_dir / "prune_symmetric_results.pkl"
print(f"\nLoading learned weights from {prune_path}")

with open(prune_path, 'rb') as f:
    prune_data = pickle.load(f)

best_result = max(prune_data['results'], key=lambda x: x['final_acc'])
cfg = prune_data['config']
state = best_result['state_dict']

L1_learned = state['layers.0.L']
D1_learned = state['layers.0.D']
gamma1_learned = state['norms.0.weight']

L2_learned = state['layers.1.L']
D2_learned = state['layers.1.D']
gamma2_learned = state['norms.1.weight']

print(f"\nLearned weights loaded:")
print(f"  L1: {L1_learned.shape}, D1: {D1_learned.shape}")
print(f"  γ1: {gamma1_learned}")
print(f"  L2: {L2_learned.shape}, D2: {D2_learned.shape}")
print(f"  γ2: {gamma2_learned}")

# %%
# =============================================================================
# FORWARD PASS FUNCTIONS
# =============================================================================
def forward_2layer(x, L1, D1, gamma1, L2, D2, gamma2):
    """
    Run 2-layer symmetric bilinear forward pass.

    Returns:
        output: final output
        intermediates: dict with r1, r2, h1, h2
    """
    # Handle scalar vs vector gamma
    if isinstance(gamma1, (int, float)):
        gamma1 = torch.tensor(gamma1)
    if isinstance(gamma2, (int, float)):
        gamma2 = torch.tensor(gamma2)

    # Layer 1
    h1, rms1 = rmsnorm(x, gamma1)
    r1, Lh1 = bilinear_forward(h1, L1, D1)

    # Layer 2
    h_mid = x + r1
    h2, rms2 = rmsnorm(h_mid, gamma2)
    r2, Lh2 = bilinear_forward(h2, L2, D2)

    output = x + r1 + r2

    return output, {
        'h1': h1, 'r1': r1, 'Lh1': Lh1,
        'h2': h2, 'r2': r2, 'Lh2': Lh2,
    }

# %%
# =============================================================================
# COMPARE ON EVALUATION DATA
# =============================================================================
print("\n" + "=" * 60)
print("COMPARISON ON EVALUATION DATA")
print("=" * 60)

torch.manual_seed(123)
x_eval = torch.randn(10000, n)
targets = task_2nd_argmax(x_eval)

# Forward pass with both models
output_ideal, inter_ideal = forward_2layer(
    x_eval, L1_ideal, D1_ideal, gamma1_ideal, L2_ideal, D2_ideal, gamma2_ideal
)
output_learned, inter_learned = forward_2layer(
    x_eval, L1_learned, D1_learned, gamma1_learned, L2_learned, D2_learned, gamma2_learned
)

# %%
# =============================================================================
# 1. ACCURACY COMPARISON
# =============================================================================
print("\n1. ACCURACY COMPARISON")
print("-" * 40)

preds_ideal = output_ideal.argmax(dim=1)
preds_learned = output_learned.argmax(dim=1)

acc_ideal = (preds_ideal == targets).float().mean().item()
acc_learned = (preds_learned == targets).float().mean().item()

# Agreement: do they predict the same thing?
agreement = (preds_ideal == preds_learned).float().mean().item()

# Both correct
both_correct = ((preds_ideal == targets) & (preds_learned == targets)).float().mean().item()

# Ideal correct, learned wrong
ideal_only = ((preds_ideal == targets) & (preds_learned != targets)).float().mean().item()

# Learned correct, ideal wrong
learned_only = ((preds_ideal != targets) & (preds_learned == targets)).float().mean().item()

# Both wrong
both_wrong = ((preds_ideal != targets) & (preds_learned != targets)).float().mean().item()

print(f"Ideal model accuracy:   {acc_ideal:.1%}")
print(f"Learned model accuracy: {acc_learned:.1%}")
print(f"")
print(f"Prediction agreement:   {agreement:.1%}")
print(f"")
print(f"Both correct:           {both_correct:.1%}")
print(f"Only ideal correct:     {ideal_only:.1%}")
print(f"Only learned correct:   {learned_only:.1%}")
print(f"Both wrong:             {both_wrong:.1%}")

# %%
# =============================================================================
# 2. OUTPUT SIMILARITY (Cosine Similarity)
# =============================================================================
print("\n2. OUTPUT SIMILARITY")
print("-" * 40)

def cosine_similarity(a, b):
    """Compute cosine similarity between two tensors (per sample)."""
    a_norm = a / (a.norm(dim=-1, keepdim=True) + 1e-8)
    b_norm = b / (b.norm(dim=-1, keepdim=True) + 1e-8)
    return (a_norm * b_norm).sum(dim=-1)

# Output cosine similarity
output_cossim = cosine_similarity(output_ideal, output_learned)
print(f"Output cosine similarity:")
print(f"  Mean: {output_cossim.mean().item():.4f}")
print(f"  Std:  {output_cossim.std().item():.4f}")
print(f"  Min:  {output_cossim.min().item():.4f}")
print(f"  Max:  {output_cossim.max().item():.4f}")

# Layer-wise similarity
r1_cossim = cosine_similarity(inter_ideal['r1'], inter_learned['r1'])
r2_cossim = cosine_similarity(inter_ideal['r2'], inter_learned['r2'])

print(f"\nLayer 1 output (r1) cosine similarity:")
print(f"  Mean: {r1_cossim.mean().item():.4f}")

print(f"\nLayer 2 output (r2) cosine similarity:")
print(f"  Mean: {r2_cossim.mean().item():.4f}")

# %%
# =============================================================================
# 3. TN-SIM FOR 2-LAYER SYMMETRIC BILINEAR
# =============================================================================
print("\n3. TN-SIM (2-Layer Symmetric Bilinear)")
print("-" * 40)

def tn_inner_2layer_symmetric(L1_a, D1_a, L2_a, D2_a,
                               L1_b, D1_b, L2_b, D2_b):
    """
    Compute TN inner product for 2-layer symmetric bilinear: B2(B1(x)).

    For symmetric bilinear, R = L, so we set R1=L1, R2=L2.

    The 5th order tensor inner product:
    <T_a|T_b> = (DD2 * (A_a @ C1 @ A_b.T) * (B_a @ C1 @ B_b.T)).sum()

    Where for symmetric case (R=L):
      C1 = (L1_a @ L1_b.T) * (L1_a @ L1_b.T) = (L1_a @ L1_b.T)²  (element-wise)
      A_a = L2_a @ D1_a, B_a = L2_a @ D1_a  (same since R2=L2)
    """
    # Layer 1 contractions (symmetric: R=L)
    LL1 = L1_a @ L1_b.T  # (d_hidden1, d_hidden1)
    C1 = LL1 * LL1       # Element-wise square (since R=L)

    # Layer 2 compositions with layer 1 (symmetric: R=L)
    A_a = L2_a @ D1_a    # (d_hidden2, d_hidden1)
    A_b = L2_b @ D1_b
    # B_a = R2_a @ D1_a = L2_a @ D1_a = A_a (symmetric)

    # Layer 2 contraction
    DD2 = D2_a.T @ D2_b  # (d_hidden2, d_hidden2)

    # Full contraction (A=B for symmetric)
    term = A_a @ C1 @ A_b.T  # (d_hidden2, d_hidden2)

    inner = (DD2 * term * term).sum()  # term² since A=B
    return inner


def tn_sim_2layer_symmetric(L1_a, D1_a, L2_a, D2_a,
                             L1_b, D1_b, L2_b, D2_b):
    """
    Compute TN similarity for 2-layer symmetric bilinear.
    TN-sim = <T_a|T_b> / (||T_a|| * ||T_b||)
    """
    inner_ab = tn_inner_2layer_symmetric(L1_a, D1_a, L2_a, D2_a,
                                          L1_b, D1_b, L2_b, D2_b)

    inner_aa = tn_inner_2layer_symmetric(L1_a, D1_a, L2_a, D2_a,
                                          L1_a, D1_a, L2_a, D2_a)

    inner_bb = tn_inner_2layer_symmetric(L1_b, D1_b, L2_b, D2_b,
                                          L1_b, D1_b, L2_b, D2_b)

    norm_a = torch.sqrt(inner_aa)
    norm_b = torch.sqrt(inner_bb)

    sim = inner_ab / (norm_a * norm_b + 1e-10)
    return sim.item()


# Compute TN-sim between ideal and learned (bilinear layers only, no RMSNorm/residual)
tnsim_bilinear = tn_sim_2layer_symmetric(
    L1_ideal, D1_ideal, L2_ideal, D2_ideal,
    L1_learned, D1_learned, L2_learned, D2_learned
)

print(f"TN-Sim (2-layer bilinear only): {tnsim_bilinear:.4f}")

# Also compute quadratic form similarities for reference
M1_ideal = compute_quadratic_forms(L1_ideal, D1_ideal)
M1_learned = compute_quadratic_forms(L1_learned, D1_learned)
M2_ideal = compute_quadratic_forms(L2_ideal, D2_ideal)
M2_learned = compute_quadratic_forms(L2_learned, D2_learned)

def frobenius_sim(A, B):
    """Frobenius cosine similarity."""
    A_flat = A.flatten()
    B_flat = B.flatten()
    inner = (A_flat * B_flat).sum()
    return (inner / (A_flat.norm() * B_flat.norm() + 1e-8)).item()

M1_sim = frobenius_sim(M1_ideal, M1_learned)
M2_sim = frobenius_sim(M2_ideal, M2_learned)

print(f"\nQuadratic form similarities (for reference):")
print(f"  M1 (layer 1): {M1_sim:.4f}")
print(f"  M2 (layer 2): {M2_sim:.4f}")

# %%
# =============================================================================
# 4. VISUALIZE WEIGHT COMPARISONS
# =============================================================================
print("\n4. WEIGHT VISUALIZATIONS")
print("-" * 40)

fig, axes = plt.subplots(4, 3, figsize=(15, 16))

def plot_comparison(ax, ideal, learned, title):
    """Plot ideal vs learned as side-by-side comparison."""
    combined = torch.stack([ideal.flatten(), learned.flatten()], dim=0)
    vmax = max(abs(combined.min().item()), abs(combined.max().item()))
    if vmax == 0:
        vmax = 1

    # Show learned
    mat = learned.numpy()
    im = ax.imshow(mat, cmap='RdBu_r', vmin=-vmax, vmax=vmax, aspect='auto')
    ax.set_title(title, fontsize=10)

    for i in range(mat.shape[0]):
        for j in range(mat.shape[1]):
            val = mat[i, j]
            color = 'white' if abs(val) > vmax * 0.5 else 'black'
            ax.text(j, i, f'{val:.2f}', ha='center', va='center', fontsize=7, color=color)

    plt.colorbar(im, ax=ax, shrink=0.8)

# Row 0: L1
plot_comparison(axes[0, 0], L1_ideal, L1_ideal, f'L1 Ideal')
plot_comparison(axes[0, 1], L1_ideal, L1_learned, f'L1 Learned')
axes[0, 2].text(0.5, 0.5, f'M1 Frob-Sim:\n{M1_sim:.3f}',
                ha='center', va='center', fontsize=14, transform=axes[0, 2].transAxes)
axes[0, 2].axis('off')

# Row 1: D1
plot_comparison(axes[1, 0], D1_ideal, D1_ideal, f'D1 Ideal')
plot_comparison(axes[1, 1], D1_ideal, D1_learned, f'D1 Learned')
axes[1, 2].axis('off')

# Row 2: L2
plot_comparison(axes[2, 0], L2_ideal, L2_ideal, f'L2 Ideal')
plot_comparison(axes[2, 1], L2_ideal, L2_learned, f'L2 Learned')
axes[2, 2].text(0.5, 0.5, f'M2 Frob-Sim:\n{M2_sim:.3f}',
                ha='center', va='center', fontsize=14, transform=axes[2, 2].transAxes)
axes[2, 2].axis('off')

# Row 3: D2
plot_comparison(axes[3, 0], D2_ideal, D2_ideal, f'D2 Ideal')
plot_comparison(axes[3, 1], D2_ideal, D2_learned, f'D2 Learned')
axes[3, 2].text(0.5, 0.5, f'2-Layer TN-Sim:\n{tnsim_bilinear:.3f}',
                ha='center', va='center', fontsize=14, transform=axes[3, 2].transAxes)
axes[3, 2].axis('off')

plt.suptitle('Ideal vs Learned Weights', fontsize=14)
plt.tight_layout()
plt.savefig(images_dir / 'hand_coded_weight_comparison.png', dpi=150)
plt.show()

# %%
# =============================================================================
# 5. QUADRATIC FORM COMPARISON
# =============================================================================
print("\n5. QUADRATIC FORM COMPARISON")
print("-" * 40)

# Layout: 4 rows x n cols
# Row 0: Layer 1 Learned
# Row 1: Layer 1 Ideal
# Row 2: Layer 2 Learned
# Row 3: Layer 2 Ideal
fig, axes = plt.subplots(4, n, figsize=(4*n, 14))

for i in range(n):
    M1_i_ideal = M1_ideal[i].numpy()
    M1_i_learned = M1_learned[i].numpy()
    M2_i_ideal = M2_ideal[i].numpy()
    M2_i_learned = M2_learned[i].numpy()

    # Use same vmax for ideal vs learned comparison
    vmax_M1 = max(abs(M1_i_ideal).max(), abs(M1_i_learned).max())
    vmax_M2 = max(abs(M2_i_ideal).max(), abs(M2_i_learned).max())

    # Row 0: Layer 1 Learned
    im = axes[0, i].imshow(M1_i_learned, cmap='RdBu_r', vmin=-vmax_M1, vmax=vmax_M1)
    axes[0, i].set_title(f'M1^({i})', fontsize=10)
    plt.colorbar(im, ax=axes[0, i], shrink=0.7)

    # Row 1: Layer 1 Ideal
    im = axes[1, i].imshow(M1_i_ideal, cmap='RdBu_r', vmin=-vmax_M1, vmax=vmax_M1)
    plt.colorbar(im, ax=axes[1, i], shrink=0.7)

    # Row 2: Layer 2 Learned
    im = axes[2, i].imshow(M2_i_learned, cmap='RdBu_r', vmin=-vmax_M2, vmax=vmax_M2)
    axes[2, i].set_title(f'M2^({i})', fontsize=10)
    plt.colorbar(im, ax=axes[2, i], shrink=0.7)

    # Row 3: Layer 2 Ideal
    im = axes[3, i].imshow(M2_i_ideal, cmap='RdBu_r', vmin=-vmax_M2, vmax=vmax_M2)
    plt.colorbar(im, ax=axes[3, i], shrink=0.7)

axes[0, 0].set_ylabel('L1 Learned', fontsize=11, fontweight='bold')
axes[1, 0].set_ylabel('L1 Ideal', fontsize=11, fontweight='bold')
axes[2, 0].set_ylabel('L2 Learned', fontsize=11, fontweight='bold')
axes[3, 0].set_ylabel('L2 Ideal', fontsize=11, fontweight='bold')

plt.suptitle(f'Quadratic Form Matrices: Learned vs Ideal\n'
             f'M1 Frob-Sim: {M1_sim:.3f}, M2 Frob-Sim: {M2_sim:.3f}, '
             f'2-Layer TN-Sim: {tnsim_bilinear:.3f}', fontsize=12)
plt.tight_layout()
plt.savefig(images_dir / 'hand_coded_quadform_comparison.png', dpi=150)
plt.show()

# %%
# =============================================================================
# 6. SUMMARY
# =============================================================================
print("\n" + "=" * 60)
print("SUMMARY")
print("=" * 60)

print(f"""
Ideal Model (hand-coded weights):
  L1: "Deviation from mean" pattern (1 on diag, -1/3 off-diag)
  D1: "Leave-one-out" pattern (0 on diag, -1 off-diag)
  γ1 = {gamma1_ideal}

  L2: Scaled deviation (α={alpha})
  D2: Negative diagonal (β={beta}), positive off-diag (γ={gamma_off})
  γ2 = {gamma2_ideal}

Accuracy:
  Ideal (tuned):  {acc_ideal:.1%}
  Learned:        {acc_learned:.1%}

Agreement: {agreement:.1%} (predictions match)

Output Similarity:
  Cosine sim: {output_cossim.mean().item():.4f} (mean)

TN-Sim (2-layer symmetric bilinear):
  Full model: {tnsim_bilinear:.4f}

Quadratic Form Similarities (Frobenius):
  M1: {M1_sim:.4f}
  M2: {M2_sim:.4f}

KEY FINDING:
  TN-Sim between ideal and learned: {tnsim_bilinear:.4f}

  This means: The learned model has found weights that produce
  a similar tensor to the hand-coded idealized patterns.

  With tuned constants (γ1={gamma1_ideal}, α={alpha}, β={beta}, γ={gamma_off}, γ2={gamma2_ideal}),
  the hand-coded model achieves {acc_ideal:.1%} vs learned {acc_learned:.1%}!
""")

# %%
