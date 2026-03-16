"""
Adversarial tests for shared feature extraction via rank-1 TN-sim minimization.

Method under test: find rank-1 S such that TN-sim(T_i - S, T_j) -> 0.
Concern: does this always find an S that drives similarity to zero, even when
there is no meaningful shared structure?

Five tests:
  1. Random tensors (no shared structure)
  2. Known shared feature + noise
  3. Known disjoint features (orthogonal inputs)
  4. Partially overlapping features
  5. Spurious correlation (positive TN-sim for wrong reason)
"""
import torch
import numpy as np
import os
import sys

sys.path.insert(0, os.path.dirname(__file__))
from model import batched_tn_inner, tn_inner_1layer, tn_sim_1layer

DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
RESULTS_FILE = os.path.join(os.path.dirname(__file__), 'adversarial_test_results.txt')

# Optimization hyperparams
N_SEEDS = 20
STEPS = 3000
LR = 0.01


# =============================================================================
# HELPERS
# =============================================================================

def make_rank1_tensor(l_vec, r_vec, d_val, d_in):
    """Create rank-1 bilinear tensor from vectors."""
    L = l_vec.unsqueeze(0).to(DEVICE)   # (1, d_in)
    R = r_vec.unsqueeze(0).to(DEVICE)   # (1, d_in)
    D = torch.tensor([[d_val]], device=DEVICE)  # (1, 1)
    return L, R, D


def concat_rank1s(tensors):
    """Combine multiple rank-1 tensors into a higher-rank tensor.
    Each tensor is (L, R, D) with L: (1, d_in), R: (1, d_in), D: (n_out, 1).
    Returns combined (L_cat, R_cat, D_cat).
    """
    Ls, Rs, Ds = zip(*tensors)
    L_cat = torch.cat(Ls, dim=0)  # (rank, d_in)
    R_cat = torch.cat(Rs, dim=0)  # (rank, d_in)
    # D columns concatenated: each D_i is (n_out, 1)
    D_cat = torch.cat(Ds, dim=1)  # (n_out, rank)
    return L_cat, R_cat, D_cat


def make_random_tensor(d_in, rank, n_out=1, scale=1.0, seed=None):
    """Create a random tensor of given rank."""
    if seed is not None:
        torch.manual_seed(seed)
    L = torch.randn(rank, d_in, device=DEVICE) * scale
    R = torch.randn(rank, d_in, device=DEVICE) * scale
    D = torch.randn(n_out, rank, device=DEVICE) * scale
    return L, R, D


def tn_norm(L, R, D):
    """TN norm squared."""
    return tn_inner_1layer(L, R, D, L, R, D).item()


def subtract_rank1(L_T, R_T, D_T, l_s, r_s, d_s):
    """Compute T - S by concatenating with negated S."""
    L_out = torch.cat([L_T, l_s], dim=0)
    R_out = torch.cat([R_T, r_s], dim=0)
    D_out = torch.cat([D_T, -d_s], dim=1)
    return L_out, R_out, D_out


def optimize_shared_feature(L_i, R_i, D_i, L_j, R_j, D_j,
                             n_seeds=N_SEEDS, steps=STEPS, lr=LR):
    """
    Optimize rank-1 S to minimize TN-sim(T_i - S, T_j)^2.

    Returns dict with best S (l, r, d), initial and final TN-sim values.
    """
    B = n_seeds
    d_in = L_i.shape[1]
    n_out = D_i.shape[0]

    # Compute initial TN-sim
    init_sim = tn_sim_1layer(L_i, R_i, D_i, L_j, R_j, D_j).item()

    # Repeat targets for batch
    tLi = L_i.unsqueeze(0).expand(B, -1, -1)
    tRi = R_i.unsqueeze(0).expand(B, -1, -1)
    tDi = D_i.unsqueeze(0).expand(B, -1, -1)
    tLj = L_j.unsqueeze(0).expand(B, -1, -1)
    tRj = R_j.unsqueeze(0).expand(B, -1, -1)
    tDj = D_j.unsqueeze(0).expand(B, -1, -1)

    # Initialize S params
    w_l = torch.zeros(B, 1, d_in, device=DEVICE)
    w_r = torch.zeros(B, 1, d_in, device=DEVICE)
    w_d = torch.zeros(B, n_out, 1, device=DEVICE)
    for i in range(B):
        torch.manual_seed(i * 137 + 42)
        w_l[i] = torch.randn(1, d_in) * 0.1
        w_r[i] = torch.randn(1, d_in) * 0.1
        w_d[i] = torch.randn(n_out, 1) * 0.1

    w_l.requires_grad_(True)
    w_r.requires_grad_(True)
    w_d.requires_grad_(True)

    optimizer = torch.optim.Adam([w_l, w_r, w_d], lr=lr)

    for step in range(steps):
        optimizer.zero_grad()

        # T_i - S: concatenate with negated S
        # Batched: L_diff = [L_i; l_s], R_diff = [R_i; r_s], D_diff = [D_i, -d_s]
        L_diff = torch.cat([tLi, w_l], dim=1)
        R_diff = torch.cat([tRi, w_r], dim=1)
        D_diff = torch.cat([tDi, -w_d], dim=2)

        # TN-sim(T_i - S, T_j)
        ab = batched_tn_inner(L_diff, R_diff, D_diff, tLj, tRj, tDj)
        aa = batched_tn_inner(L_diff, R_diff, D_diff, L_diff, R_diff, D_diff)
        bb = batched_tn_inner(tLj, tRj, tDj, tLj, tRj, tDj)
        sims = ab / (torch.sqrt(aa.clamp(min=1e-12)) * torch.sqrt(bb.clamp(min=1e-12)))

        loss = sims.pow(2).mean()
        loss.backward()
        optimizer.step()

    # Evaluate final
    with torch.no_grad():
        L_diff = torch.cat([tLi, w_l], dim=1)
        R_diff = torch.cat([tRi, w_r], dim=1)
        D_diff = torch.cat([tDi, -w_d], dim=2)

        ab = batched_tn_inner(L_diff, R_diff, D_diff, tLj, tRj, tDj)
        aa = batched_tn_inner(L_diff, R_diff, D_diff, L_diff, R_diff, D_diff)
        bb = batched_tn_inner(tLj, tRj, tDj, tLj, tRj, tDj)
        final_sims = ab / (torch.sqrt(aa.clamp(min=1e-12)) * torch.sqrt(bb.clamp(min=1e-12)))

        # Pick best (lowest |sim|)
        best = final_sims.abs().argmin().item()

        # Compute S norm
        s_norm_sq = batched_tn_inner(w_l, w_r, w_d, w_l, w_r, w_d)

    return {
        'l': w_l[best].detach(),
        'r': w_r[best].detach(),
        'd': w_d[best].detach(),
        'init_sim': init_sim,
        'final_sim': final_sims[best].item(),
        's_norm_sq': s_norm_sq[best].item(),
        'all_final_sims': final_sims.detach().cpu().numpy(),
    }


def print_and_log(msg, log_lines):
    print(msg)
    log_lines.append(msg)


# =============================================================================
# TEST 1: Random tensors (no shared structure)
# =============================================================================

def test1_random_tensors(log):
    print_and_log("\n" + "=" * 70, log)
    print_and_log("TEST 1: Random tensors (no shared structure)", log)
    print_and_log("=" * 70, log)

    d_in = 6
    n_trials = 5
    results = []

    for trial in range(n_trials):
        L_i, R_i, D_i = make_random_tensor(d_in, rank=4, seed=trial * 2)
        L_j, R_j, D_j = make_random_tensor(d_in, rank=4, seed=trial * 2 + 1)

        Ti_norm = tn_norm(L_i, R_i, D_i)
        Tj_norm = tn_norm(L_j, R_j, D_j)

        res = optimize_shared_feature(L_i, R_i, D_i, L_j, R_j, D_j)

        s_ratio = np.sqrt(abs(res['s_norm_sq'])) / np.sqrt(abs(Ti_norm)) if Ti_norm > 1e-12 else float('inf')

        results.append({
            'init_sim': res['init_sim'],
            'final_sim': res['final_sim'],
            's_ratio': s_ratio,
        })

        print_and_log(f"  Trial {trial}: init_sim={res['init_sim']:+.4f}  "
                      f"final_sim={res['final_sim']:+.4f}  "
                      f"||S||/||T_i||={s_ratio:.4f}", log)

    # Assessment
    avg_final = np.mean([r['final_sim'] for r in results])
    avg_ratio = np.mean([r['s_ratio'] for r in results])
    always_zero = all(abs(r['final_sim']) < 0.05 for r in results)

    print_and_log(f"\n  Average final |sim|: {abs(avg_final):.4f}", log)
    print_and_log(f"  Average ||S||/||T_i||: {avg_ratio:.4f}", log)

    if always_zero and avg_ratio > 2.0:
        verdict = "FAIL: Method brute-forces orthogonality with large S even for random tensors"
    elif always_zero and avg_ratio < 0.5:
        verdict = "WARN: Method always finds small S that zeros out sim — suspicious"
    elif not always_zero:
        verdict = "PASS: Method does NOT always drive sim to zero for random tensors"
    else:
        verdict = "AMBIGUOUS: Drives sim to zero with moderate S — needs investigation"

    print_and_log(f"  Verdict: {verdict}", log)
    return results, verdict


# =============================================================================
# TEST 2: Known shared feature + noise
# =============================================================================

def test2_known_shared_feature(log):
    print_and_log("\n" + "=" * 70, log)
    print_and_log("TEST 2: Known shared feature + noise", log)
    print_and_log("=" * 70, log)

    d_in = 6

    # Feature A = x0 * x1 (rank-1)
    l_A = torch.zeros(d_in, device=DEVICE); l_A[0] = 1.0
    r_A = torch.zeros(d_in, device=DEVICE); r_A[1] = 1.0
    L_A, R_A, D_A = make_rank1_tensor(l_A, r_A, 1.0, d_in)

    # T_i = feature_A + noise_i (rank 4 total)
    noise_i = make_random_tensor(d_in, rank=3, seed=100, scale=0.3)
    L_i, R_i, D_i = concat_rank1s([(L_A, R_A, D_A)] + [(noise_i[0][[k]], noise_i[1][[k]], noise_i[2][:, [k]]) for k in range(3)])

    # T_j = feature_A + noise_j (rank 4 total)
    noise_j = make_random_tensor(d_in, rank=3, seed=200, scale=0.3)
    L_j, R_j, D_j = concat_rank1s([(L_A, R_A, D_A)] + [(noise_j[0][[k]], noise_j[1][[k]], noise_j[2][:, [k]]) for k in range(3)])

    Ti_norm = tn_norm(L_i, R_i, D_i)
    A_norm = tn_norm(L_A, R_A, D_A)

    print_and_log(f"  ||feature_A||^2 = {A_norm:.4f}", log)
    print_and_log(f"  ||T_i||^2 = {Ti_norm:.4f}", log)
    print_and_log(f"  ||T_j||^2 = {tn_norm(L_j, R_j, D_j):.4f}", log)

    res = optimize_shared_feature(L_i, R_i, D_i, L_j, R_j, D_j)

    # Check if S recovers feature_A
    s_sim_A = tn_sim_1layer(res['l'], res['r'], res['d'], L_A, R_A, D_A).item()
    s_ratio = np.sqrt(abs(res['s_norm_sq'])) / np.sqrt(abs(Ti_norm)) if Ti_norm > 1e-12 else float('inf')

    print_and_log(f"  init_sim(T_i, T_j) = {res['init_sim']:+.4f}", log)
    print_and_log(f"  final_sim(T_i-S, T_j) = {res['final_sim']:+.4f}", log)
    print_and_log(f"  TN-sim(S, feature_A) = {s_sim_A:+.4f}", log)
    print_and_log(f"  ||S||/||T_i|| = {s_ratio:.4f}", log)

    # Inspect recovered vectors
    l_s = res['l'].cpu().numpy().flatten()
    r_s = res['r'].cpu().numpy().flatten()
    print_and_log(f"  S.l = [{', '.join(f'{v:+.3f}' for v in l_s)}]", log)
    print_and_log(f"  S.r = [{', '.join(f'{v:+.3f}' for v in r_s)}]", log)
    print_and_log(f"  (Ground truth: l=[+1,0,0,0,0,0], r=[0,+1,0,0,0,0])", log)

    if abs(s_sim_A) > 0.8:
        verdict = "PASS: Recovered the true shared feature (TN-sim > 0.8)"
    elif abs(s_sim_A) > 0.5:
        verdict = "PARTIAL: Some recovery of shared feature (0.5 < TN-sim < 0.8)"
    else:
        verdict = "FAIL: Did NOT recover the shared feature"

    print_and_log(f"  Verdict: {verdict}", log)
    return res, verdict


# =============================================================================
# TEST 3: Known disjoint features (orthogonal inputs)
# =============================================================================

def test3_disjoint_features(log):
    print_and_log("\n" + "=" * 70, log)
    print_and_log("TEST 3: Disjoint features (orthogonal inputs)", log)
    print_and_log("=" * 70, log)

    d_in = 6

    # Feature A = x0 * x1
    l_A = torch.zeros(d_in, device=DEVICE); l_A[0] = 1.0
    r_A = torch.zeros(d_in, device=DEVICE); r_A[1] = 1.0
    L_A, R_A, D_A = make_rank1_tensor(l_A, r_A, 1.0, d_in)

    # Feature B = x2 * x3
    l_B = torch.zeros(d_in, device=DEVICE); l_B[2] = 1.0
    r_B = torch.zeros(d_in, device=DEVICE); r_B[3] = 1.0
    L_B, R_B, D_B = make_rank1_tensor(l_B, r_B, 1.0, d_in)

    init_sim = tn_sim_1layer(L_A, R_A, D_A, L_B, R_B, D_B).item()
    print_and_log(f"  init_sim(A, B) = {init_sim:+.6f} (should be ~0, orthogonal)", log)

    res = optimize_shared_feature(L_A, R_A, D_A, L_B, R_B, D_B)

    s_ratio = np.sqrt(abs(res['s_norm_sq'])) / np.sqrt(tn_norm(L_A, R_A, D_A)) if tn_norm(L_A, R_A, D_A) > 1e-12 else float('inf')

    print_and_log(f"  final_sim(A-S, B) = {res['final_sim']:+.6f}", log)
    print_and_log(f"  ||S||/||T_A|| = {s_ratio:.6f}", log)
    print_and_log(f"  ||S||^2 = {res['s_norm_sq']:.6f}", log)

    if s_ratio < 0.1 and abs(res['final_sim']) < 0.01:
        verdict = "PASS: S is near-zero for already-orthogonal tensors"
    elif s_ratio < 0.3:
        verdict = "PASS: S is small for orthogonal tensors"
    else:
        verdict = "FAIL: S is large even though tensors are already orthogonal"

    print_and_log(f"  Verdict: {verdict}", log)
    return res, verdict


# =============================================================================
# TEST 4: Partially overlapping features
# =============================================================================

def test4_partial_overlap(log):
    print_and_log("\n" + "=" * 70, log)
    print_and_log("TEST 4: Partially overlapping features", log)
    print_and_log("=" * 70, log)

    d_in = 6

    # Feature A = x0 * x1
    l_A = torch.zeros(d_in, device=DEVICE); l_A[0] = 1.0
    r_A = torch.zeros(d_in, device=DEVICE); r_A[1] = 1.0
    L_A, R_A, D_A = make_rank1_tensor(l_A, r_A, 1.0, d_in)

    # Feature B = x2 * x3
    l_B = torch.zeros(d_in, device=DEVICE); l_B[2] = 1.0
    r_B = torch.zeros(d_in, device=DEVICE); r_B[3] = 1.0
    L_B, R_B, D_B = make_rank1_tensor(l_B, r_B, 1.0, d_in)

    # Feature C = x4 * x5
    l_C = torch.zeros(d_in, device=DEVICE); l_C[4] = 1.0
    r_C = torch.zeros(d_in, device=DEVICE); r_C[5] = 1.0
    L_C, R_C, D_C = make_rank1_tensor(l_C, r_C, 1.0, d_in)

    # T_i = A + B (rank-2)
    L_i, R_i, D_i = concat_rank1s([(L_A, R_A, D_A), (L_B, R_B, D_B)])
    # T_j = A + C (rank-2)
    L_j, R_j, D_j = concat_rank1s([(L_A, R_A, D_A), (L_C, R_C, D_C)])

    Ti_norm = tn_norm(L_i, R_i, D_i)
    A_norm = tn_norm(L_A, R_A, D_A)

    print_and_log(f"  ||feature_A||^2 = {A_norm:.4f}", log)
    print_and_log(f"  ||T_i||^2 = {Ti_norm:.4f}  (A+B)", log)
    print_and_log(f"  ||T_j||^2 = {tn_norm(L_j, R_j, D_j):.4f}  (A+C)", log)

    res = optimize_shared_feature(L_i, R_i, D_i, L_j, R_j, D_j)

    # Check S recovery
    s_sim_A = tn_sim_1layer(res['l'], res['r'], res['d'], L_A, R_A, D_A).item()
    s_sim_B = tn_sim_1layer(res['l'], res['r'], res['d'], L_B, R_B, D_B).item()
    s_sim_C = tn_sim_1layer(res['l'], res['r'], res['d'], L_C, R_C, D_C).item()
    s_ratio = np.sqrt(abs(res['s_norm_sq'])) / np.sqrt(abs(Ti_norm)) if Ti_norm > 1e-12 else float('inf')

    print_and_log(f"  init_sim(T_i, T_j) = {res['init_sim']:+.4f}", log)
    print_and_log(f"  final_sim(T_i-S, T_j) = {res['final_sim']:+.4f}", log)
    print_and_log(f"  TN-sim(S, feature_A) = {s_sim_A:+.4f}  (shared — should be high)", log)
    print_and_log(f"  TN-sim(S, feature_B) = {s_sim_B:+.4f}  (T_i only — should be low)", log)
    print_and_log(f"  TN-sim(S, feature_C) = {s_sim_C:+.4f}  (T_j only — should be low)", log)
    print_and_log(f"  ||S||/||T_i|| = {s_ratio:.4f}", log)

    l_s = res['l'].cpu().numpy().flatten()
    r_s = res['r'].cpu().numpy().flatten()
    print_and_log(f"  S.l = [{', '.join(f'{v:+.3f}' for v in l_s)}]", log)
    print_and_log(f"  S.r = [{', '.join(f'{v:+.3f}' for v in r_s)}]", log)
    print_and_log(f"  (Ground truth A: l=[+1,0,0,0,0,0], r=[0,+1,0,0,0,0])", log)

    if abs(s_sim_A) > 0.8 and abs(s_sim_B) < 0.3 and abs(s_sim_C) < 0.3:
        verdict = "PASS: Correctly extracted shared feature A, ignoring B and C"
    elif abs(s_sim_A) > 0.5:
        verdict = "PARTIAL: Some preference for shared feature A"
    else:
        verdict = "FAIL: Did NOT extract the shared feature"

    print_and_log(f"  Verdict: {verdict}", log)
    return res, verdict


# =============================================================================
# TEST 5: Spurious correlation
# =============================================================================

def test5_spurious_correlation(log):
    print_and_log("\n" + "=" * 70, log)
    print_and_log("TEST 5: Spurious correlation (positive TN-sim, different features)", log)
    print_and_log("=" * 70, log)

    d_in = 6

    # T_i = x0 * x1 (feature A)
    l_A = torch.zeros(d_in, device=DEVICE); l_A[0] = 1.0
    r_A = torch.zeros(d_in, device=DEVICE); r_A[1] = 1.0
    L_A, R_A, D_A = make_rank1_tensor(l_A, r_A, 1.0, d_in)

    # T_j = x0 * x2 (feature D — shares x0 with A but different second input)
    l_D = torch.zeros(d_in, device=DEVICE); l_D[0] = 1.0
    r_D = torch.zeros(d_in, device=DEVICE); r_D[2] = 1.0
    L_D, R_D, D_D = make_rank1_tensor(l_D, r_D, 1.0, d_in)

    init_sim = tn_sim_1layer(L_A, R_A, D_A, L_D, R_D, D_D).item()
    print_and_log(f"  T_i = x0*x1 (feature A)", log)
    print_and_log(f"  T_j = x0*x2 (feature D — shares x0 dimension)", log)
    print_and_log(f"  init_sim(A, D) = {init_sim:+.4f} (nonzero due to shared x0)", log)

    res = optimize_shared_feature(L_A, R_A, D_A, L_D, R_D, D_D)

    s_ratio = np.sqrt(abs(res['s_norm_sq'])) / np.sqrt(tn_norm(L_A, R_A, D_A)) if tn_norm(L_A, R_A, D_A) > 1e-12 else float('inf')

    # What does S look like?
    l_s = res['l'].cpu().numpy().flatten()
    r_s = res['r'].cpu().numpy().flatten()
    d_s = res['d'].cpu().numpy().flatten()

    print_and_log(f"  final_sim(A-S, D) = {res['final_sim']:+.4f}", log)
    print_and_log(f"  ||S||/||T_A|| = {s_ratio:.4f}", log)
    print_and_log(f"  S.l = [{', '.join(f'{v:+.3f}' for v in l_s)}]", log)
    print_and_log(f"  S.r = [{', '.join(f'{v:+.3f}' for v in r_s)}]", log)
    print_and_log(f"  S.d = {d_s[0]:+.4f}", log)

    # Check: does S correspond to a meaningful shared feature?
    # There IS a shared aspect (x0 dimension), so S might pick up on it.
    # The question is whether S is interpretable or just an artifact.

    # Verify: compute what A - S looks like as a function
    # If S captures x0 component, then A - S should only use x1
    s_sim_A = tn_sim_1layer(res['l'], res['r'], res['d'], L_A, R_A, D_A).item()
    s_sim_D = tn_sim_1layer(res['l'], res['r'], res['d'], L_D, R_D, D_D).item()

    print_and_log(f"  TN-sim(S, A=x0*x1) = {s_sim_A:+.4f}", log)
    print_and_log(f"  TN-sim(S, D=x0*x2) = {s_sim_D:+.4f}", log)

    # Additional test: completely unrelated but with spurious correlation
    # via non-orthogonal random directions
    print_and_log("\n  --- Sub-test: non-orthogonal random features ---", log)

    torch.manual_seed(999)
    # Two features that happen to share a random direction
    v_shared = torch.randn(d_in, device=DEVICE)
    v_shared = v_shared / v_shared.norm()
    v1 = torch.randn(d_in, device=DEVICE); v1 = v1 / v1.norm()
    v2 = torch.randn(d_in, device=DEVICE); v2 = v2 / v2.norm()

    # T_i = v_shared * v1, T_j = v_shared * v2
    L_i, R_i, D_i = make_rank1_tensor(v_shared, v1, 1.0, d_in)
    L_j, R_j, D_j = make_rank1_tensor(v_shared, v2, 1.0, d_in)

    init_sim2 = tn_sim_1layer(L_i, R_i, D_i, L_j, R_j, D_j).item()
    print_and_log(f"  init_sim = {init_sim2:+.4f}", log)

    res2 = optimize_shared_feature(L_i, R_i, D_i, L_j, R_j, D_j)

    s_ratio2 = np.sqrt(abs(res2['s_norm_sq'])) / np.sqrt(tn_norm(L_i, R_i, D_i)) if tn_norm(L_i, R_i, D_i) > 1e-12 else float('inf')
    print_and_log(f"  final_sim = {res2['final_sim']:+.4f}", log)
    print_and_log(f"  ||S||/||T_i|| = {s_ratio2:.4f}", log)

    # These features share a left vector but have different right vectors
    # A meaningful S would capture the shared structure
    s_sim_i = tn_sim_1layer(res2['l'], res2['r'], res2['d'], L_i, R_i, D_i).item()
    s_sim_j = tn_sim_1layer(res2['l'], res2['r'], res2['d'], L_j, R_j, D_j).item()
    print_and_log(f"  TN-sim(S, T_i) = {s_sim_i:+.4f}", log)
    print_and_log(f"  TN-sim(S, T_j) = {s_sim_j:+.4f}", log)

    if s_ratio > 1.5:
        verdict = "WARN: Large S for spurious correlation — method may overfit"
    elif abs(res['final_sim']) < 0.05 and s_ratio < 0.8:
        verdict = "INFO: Method found some shared structure (x0 IS genuinely shared)"
    else:
        verdict = "AMBIGUOUS: Needs further investigation"

    print_and_log(f"  Verdict: {verdict}", log)
    return (res, res2), verdict


# =============================================================================
# MAIN
# =============================================================================

def main():
    log = []
    print_and_log("ADVERSARIAL TESTS FOR SHARED FEATURE EXTRACTION", log)
    print_and_log(f"Device: {DEVICE}", log)
    print_and_log(f"Optimization: n_seeds={N_SEEDS}, steps={STEPS}, lr={LR}", log)

    verdicts = {}

    _, verdicts['test1'] = test1_random_tensors(log)
    _, verdicts['test2'] = test2_known_shared_feature(log)
    _, verdicts['test3'] = test3_disjoint_features(log)
    _, verdicts['test4'] = test4_partial_overlap(log)
    _, verdicts['test5'] = test5_spurious_correlation(log)

    # Summary
    print_and_log("\n" + "=" * 70, log)
    print_and_log("SUMMARY", log)
    print_and_log("=" * 70, log)
    for name, v in verdicts.items():
        print_and_log(f"  {name}: {v}", log)

    # Overall assessment
    print_and_log("\n" + "-" * 70, log)
    passes = sum(1 for v in verdicts.values() if v.startswith("PASS"))
    fails = sum(1 for v in verdicts.values() if v.startswith("FAIL"))
    print_and_log(f"  {passes} PASS, {fails} FAIL, {len(verdicts) - passes - fails} other", log)

    if fails == 0 and passes >= 3:
        print_and_log("  OVERALL: Method appears to be finding genuine shared features", log)
    elif fails >= 2:
        print_and_log("  OVERALL: Method has significant issues — may find spurious shared features", log)
    else:
        print_and_log("  OVERALL: Mixed results — method needs careful interpretation", log)

    # Save
    with open(RESULTS_FILE, 'w') as f:
        f.write('\n'.join(log) + '\n')
    print(f"\nResults saved to {RESULTS_FILE}")


if __name__ == '__main__':
    main()
