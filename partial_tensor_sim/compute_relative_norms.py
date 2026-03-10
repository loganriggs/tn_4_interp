"""
Compute relative norm fractions: what fraction of the tensor is residual vs bilinear?

Uses the projection decomposition:
  ρ(A) = <T|A> / <T|T>     (residual fraction)
  ρ(B) = <T|B> / <T|T>     (bilinear fraction)
  ρ(A) + ρ(B) = 1          (exact, by linearity)

Also verifies the Pythagorean identity (since A ⊥ B in our construction):
  ||T||² = ||A||² + ||B||²
"""
import torch
import numpy as np
import os
from itertools import combinations
from mnist_pairwise_tnsim import (
    ResidualBilinearMNIST,
    tn_inner_1layer,
)
from run_residual import get_augmented_weights_for_class
from run_frozen_residual import build_residual_parts


def compute_relative_norms(model, class_indices, device):
    """
    For a given output slice (single class or pair), compute:
      - <T|T>, <T|A>, <T|B>, <A|A>, <B|B>, <A|B>
      - ρ(A) = <T|A>/<T|T>, ρ(B) = <T|B>/<T|T>

    Returns dict with all quantities.
    """
    # Get full augmented target T
    if isinstance(class_indices, int):
        L_T, R_T, D_T = get_augmented_weights_for_class(model, class_indices)
    else:
        ci, cj = class_indices
        L_T, R_T, D_T = model.get_augmented_weights_for_pair(ci, cj)

    # Get residual-only part A
    L_A, R_A, D_A = build_residual_parts(model, class_indices, device)

    # Get bilinear-only part B
    with torch.no_grad():
        W_e = model.W_embed
        if isinstance(class_indices, int):
            W_h = model.W_head[[class_indices]]
        else:
            W_h = model.W_head[list(class_indices)]

        L_eff = model.L @ W_e   # (rank, 784)
        R_eff = model.R @ W_e
        D_eff = W_h @ model.D  # (n_out, rank)

        rank = L_eff.shape[0]
        L_B = torch.zeros(rank, 785, device=device)
        L_B[:, :784] = L_eff
        R_B = torch.zeros(rank, 785, device=device)
        R_B[:, :784] = R_eff
        D_B = D_eff

    # Compute all inner products
    TT = tn_inner_1layer(L_T, R_T, D_T, L_T, R_T, D_T).item()
    TA = tn_inner_1layer(L_T, R_T, D_T, L_A, R_A, D_A).item()
    TB = tn_inner_1layer(L_T, R_T, D_T, L_B, R_B, D_B).item()
    AA = tn_inner_1layer(L_A, R_A, D_A, L_A, R_A, D_A).item()
    BB = tn_inner_1layer(L_B, R_B, D_B, L_B, R_B, D_B).item()
    AB = tn_inner_1layer(L_A, R_A, D_A, L_B, R_B, D_B).item()

    rho_A = TA / TT
    rho_B = TB / TT

    return {
        'TT': TT, 'TA': TA, 'TB': TB,
        'AA': AA, 'BB': BB, 'AB': AB,
        'rho_A': rho_A, 'rho_B': rho_B,
        'rho_sum': rho_A + rho_B,
        'pythagorean_check': (AA + BB) / TT,  # should be 1.0
        'orthogonality_check': AB,              # should be 0.0
    }


if __name__ == '__main__':
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Device: {device}")

    base_dir = os.path.dirname(__file__)
    results = torch.load(os.path.join(base_dir, 'residual_results.pt'),
                         map_location=device, weights_only=False)
    model = ResidualBilinearMNIST(784, 64, 32, 10).to(device)
    model.load_state_dict(results['model_state'])
    model.eval()

    # ====== Per-class relative norms ======
    print("\n" + "=" * 70)
    print("PER-CLASS RELATIVE NORMS")
    print("=" * 70)
    print(f"  {'Class':>5s}  {'ρ(res)':>8s}  {'ρ(bil)':>8s}  {'sum':>6s}  "
          f"{'||A||²+||B||²/||T||²':>20s}  {'<A|B>':>10s}")

    class_rhos = []
    for c in range(10):
        r = compute_relative_norms(model, c, device)
        class_rhos.append(r)
        print(f"  {c:>5d}  {r['rho_A']:>8.4f}  {r['rho_B']:>8.4f}  {r['rho_sum']:>6.4f}  "
              f"{r['pythagorean_check']:>20.6f}  {r['orthogonality_check']:>10.2e}")

    mean_rho_A = np.mean([r['rho_A'] for r in class_rhos])
    mean_rho_B = np.mean([r['rho_B'] for r in class_rhos])
    print(f"\n  Mean:  ρ(residual) = {mean_rho_A:.4f}   ρ(bilinear) = {mean_rho_B:.4f}")
    print(f"  → {mean_rho_A*100:.1f}% of the tensor norm is the residual stream")
    print(f"  → {mean_rho_B*100:.1f}% of the tensor norm is the bilinear computation")

    # ====== Pairwise relative norms ======
    print("\n" + "=" * 70)
    print("PAIRWISE RELATIVE NORMS")
    print("=" * 70)

    pairs = list(combinations(range(10), 2))
    pair_rhos = []
    for ci, cj in pairs:
        r = compute_relative_norms(model, (ci, cj), device)
        pair_rhos.append(r)

    # Sort by bilinear fraction
    sorted_idx = np.argsort([r['rho_B'] for r in pair_rhos])[::-1]

    print(f"  {'Pair':>6s}  {'ρ(res)':>8s}  {'ρ(bil)':>8s}  {'sum':>6s}  "
          f"{'<A|B>':>10s}")
    for idx in sorted_idx[:10]:
        ci, cj = pairs[idx]
        r = pair_rhos[idx]
        print(f"  {ci}-{cj:>4d}  {r['rho_A']:>8.4f}  {r['rho_B']:>8.4f}  "
              f"{r['rho_sum']:>6.4f}  {r['orthogonality_check']:>10.2e}")
    print("  ...")
    for idx in sorted_idx[-5:]:
        ci, cj = pairs[idx]
        r = pair_rhos[idx]
        print(f"  {ci}-{cj:>4d}  {r['rho_A']:>8.4f}  {r['rho_B']:>8.4f}  "
              f"{r['rho_sum']:>6.4f}  {r['orthogonality_check']:>10.2e}")

    mean_rho_A_p = np.mean([r['rho_A'] for r in pair_rhos])
    mean_rho_B_p = np.mean([r['rho_B'] for r in pair_rhos])
    print(f"\n  Mean:  ρ(residual) = {mean_rho_A_p:.4f}   ρ(bilinear) = {mean_rho_B_p:.4f}")
    print(f"  → {mean_rho_A_p*100:.1f}% residual, {mean_rho_B_p*100:.1f}% bilinear")

    # ====== Verify Pythagorean TN-sim identity ======
    print("\n" + "=" * 70)
    print("PYTHAGOREAN TN-SIM IDENTITY CHECK")
    print("=" * 70)
    print("  TN-sim(A,T)² + TN-sim(B,T)² should = 1")
    print()

    # Load frozen residual results for the measured TN-sim values
    frozen = torch.load(os.path.join(base_dir, 'frozen_residual_results.pt'),
                        map_location=device, weights_only=False)

    print(f"  {'Class':>5s}  {'ρ(A)':>7s}  {'ρ(B)':>7s}  {'sim(A,T)':>9s}  "
          f"{'√ρ(A)':>7s}  {'sim²(A)+sim²(B)':>16s}")
    for c in range(10):
        r = class_rhos[c]
        sim_A = np.sqrt(r['rho_A'])  # predicted TN-sim(A,T)
        sim_B = np.sqrt(r['rho_B'])
        # Compare with measured baseline sim
        measured_sim_A = frozen['class_baseline'].get(c, None)
        if measured_sim_A is not None:
            print(f"  {c:>5d}  {r['rho_A']:>7.4f}  {r['rho_B']:>7.4f}  "
                  f"{measured_sim_A:>9.4f}  {sim_A:>7.4f}  "
                  f"{measured_sim_A**2 + sim_B**2:>16.6f}")
        else:
            print(f"  {c:>5d}  {r['rho_A']:>7.4f}  {r['rho_B']:>7.4f}  "
                  f"{'N/A':>9s}  {sim_A:>7.4f}")

    # ====== Analytic check: ||A||² = 0.5 * ||W_res||²_F ======
    print("\n" + "=" * 70)
    print("ANALYTIC CHECK: ||A||² = 0.5 * ||W_res||²_F")
    print("=" * 70)

    for c in range(10):
        with torch.no_grad():
            W_res = model.W_head[[c]] @ model.W_embed  # (1, 784)
            frobenius_sq = (W_res ** 2).sum().item()
        predicted_AA = 0.5 * frobenius_sq
        actual_AA = class_rhos[c]['AA']
        print(f"  Class {c}: ||A||² = {actual_AA:.4f}, 0.5*||W_res||²_F = {predicted_AA:.4f}, "
              f"ratio = {actual_AA/predicted_AA:.8f}")

    print("\nDone!")
