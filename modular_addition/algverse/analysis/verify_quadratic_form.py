import numpy as np
np.set_printoptions(precision=3, suppress=True)

print("=" * 70)
print("PART 1: HOW I KNOW THE SPARSITY PATTERN")
print("=" * 70)

L1 = np.array([
    [-0.334, -0.335, -0.329,  0.766],  # rank0
    [ 0.774, -0.338, -0.327, -0.322],  # rank1
    [-0.342,  0.765, -0.346, -0.327],  # rank2
    [-0.319, -0.312,  0.746, -0.324],  # rank3
])

D1 = np.array([
    [-0.595,  0.000, -0.630, -0.616],  # pos0
    [-0.593, -0.607,  0.000, -0.622],  # pos1
    [-0.609, -0.629, -0.610,  0.082],  # pos2
    [ 0.000, -0.611, -0.613, -0.607],  # pos3
])

print("\nStep 1: Which L1 row detects which position?")
print("-" * 50)
for r in range(4):
    detected_pos = np.argmax(np.abs(L1[r]))
    print(f"  L1[{r}] has max magnitude at pos{detected_pos} → detects pos{detected_pos}")

print("\nStep 2: Where are the zeros in D1?")
print("-" * 50)
for i in range(4):
    zeros = [r for r in range(4) if abs(D1[i, r]) < 0.01]
    print(f"  D1[{i},:] has zeros at ranks: {zeros}")

print("\nStep 3: Match them up!")
print("-" * 50)
detector_map = {0: 3, 1: 0, 2: 1, 3: 2}  # rank r detects position detector_map[r]
for i in range(4):
    zeros = [r for r in range(4) if abs(D1[i, r]) < 0.01]
    for r in zeros:
        detected = detector_map[r]
        print(f"  r1[{i}] has D1[{i},{r}]=0, and L1[{r}] detects pos{detected}")
        if detected == i:
            print(f"    → r1[{i}] does NOT use its own detector! ✓")

print("\n" + "=" * 70)
print("PART 2: VERIFY QUADRATIC FORM = ORIGINAL COMPUTATION")
print("=" * 70)

def test_equivalence(n, rank, seed=42):
    """Test that M^(i) quadratic form equals D @ (L @ h)²"""
    np.random.seed(seed)
    
    # Random weights
    L = np.random.randn(rank, n)
    D = np.random.randn(n, rank)
    
    # Compute the quadratic form matrices M^(i)
    # M^(i)_{jk} = sum_r D_{ir} * L_{rj} * L_{rk}
    M = np.zeros((n, n, n))  # M[i] is n×n matrix for output i
    for i in range(n):
        for j in range(n):
            for k in range(n):
                M[i, j, k] = sum(D[i, r] * L[r, j] * L[r, k] for r in range(rank))
    
    # Test on random inputs
    n_tests = 5
    print(f"\nTesting with n={n}, rank={rank}:")
    print("-" * 50)
    
    all_match = True
    for t in range(n_tests):
        x = np.random.randn(n)
        
        # Original computation: D @ (L @ x)²
        Lx = L @ x
        Lx_sq = Lx ** 2
        original = D @ Lx_sq
        
        # Quadratic form computation: x^T M^(i) x for each i
        quadratic = np.array([x @ M[i] @ x for i in range(n)])
        
        match = np.allclose(original, quadratic)
        all_match = all_match and match
        
        if t < 2:  # Show first 2 examples
            print(f"  Test {t+1}:")
            print(f"    x = {x[:4]}{'...' if n > 4 else ''}")
            print(f"    Original D@(Lx)²:  {original[:4]}{'...' if n > 4 else ''}")
            print(f"    Quadratic x'M^(i)x: {quadratic[:4]}{'...' if n > 4 else ''}")
            print(f"    Match: {match}")
    
    print(f"  All {n_tests} tests passed: {all_match}")
    return all_match

# Test various sizes
test_equivalence(n=4, rank=4)
test_equivalence(n=4, rank=2)  # Low rank D
test_equivalence(n=8, rank=8)
test_equivalence(n=4, rank=6)  # Rank > n

print("\n" + "=" * 70)
print("PART 3: NON-SYMMETRIC BILINEAR LAYER")
print("=" * 70)

print("""
For symmetric: y = D @ (L @ x)²  = D @ (L @ x ⊙ L @ x)
For non-symmetric: y = D @ (L @ x ⊙ R @ x)  where L ≠ R

The tensor decomposition still works! For output i:

  output_i = Σ_r D_{ir} (L_r · x)(R_r · x)
           = Σ_r D_{ir} Σ_{j,k} L_{rj} R_{rk} x_j x_k
           = Σ_{j,k} [Σ_r D_{ir} L_{rj} R_{rk}] x_j x_k
           = x^T M^(i) x

where M^(i)_{jk} = Σ_r D_{ir} L_{rj} R_{rk}

KEY DIFFERENCE: M^(i) is NOT symmetric when L ≠ R!

But here's the cool part: for any matrix M,
  x^T M x = x^T [(M + M^T)/2] x

because the antisymmetric part (M - M^T)/2 contributes zero to x^T M x.

So we can always work with the SYMMETRIZED version: M_sym = (M + M^T)/2
""")

def test_nonsymmetric_equivalence(n, rank, seed=42):
    """Test quadratic form equivalence for non-symmetric bilinear layer"""
    np.random.seed(seed)
    
    # Random weights with L ≠ R
    L = np.random.randn(rank, n)
    R = np.random.randn(rank, n)  # Different from L!
    D = np.random.randn(n, rank)
    
    # Compute the (non-symmetric) quadratic form matrices M^(i)
    # M^(i)_{jk} = sum_r D_{ir} * L_{rj} * R_{rk}
    M = np.zeros((n, n, n))
    for i in range(n):
        for j in range(n):
            for k in range(n):
                M[i, j, k] = sum(D[i, r] * L[r, j] * R[r, k] for r in range(rank))
    
    # Check symmetry
    print(f"\nTesting NON-SYMMETRIC bilinear with n={n}, rank={rank}:")
    print("-" * 50)
    
    print(f"  Is M^(0) symmetric? {np.allclose(M[0], M[0].T)}")
    print(f"  Max asymmetry in M^(0): {np.max(np.abs(M[0] - M[0].T)):.4f}")
    
    # Test equivalence
    n_tests = 5
    all_match = True
    for t in range(n_tests):
        x = np.random.randn(n)
        
        # Original computation: D @ (L @ x ⊙ R @ x)
        Lx = L @ x
        Rx = R @ x
        original = D @ (Lx * Rx)
        
        # Quadratic form computation: x^T M^(i) x
        quadratic = np.array([x @ M[i] @ x for i in range(n)])
        
        # Also test with symmetrized M
        M_sym = np.array([(M[i] + M[i].T) / 2 for i in range(n)])
        quadratic_sym = np.array([x @ M_sym[i] @ x for i in range(n)])
        
        match = np.allclose(original, quadratic)
        match_sym = np.allclose(original, quadratic_sym)
        all_match = all_match and match
        
        if t < 2:
            print(f"  Test {t+1}:")
            print(f"    Original D@(Lx⊙Rx):      {original[:3]}...")
            print(f"    Quadratic x'M^(i)x:      {quadratic[:3]}...")
            print(f"    Symmetrized x'M_sym^(i)x: {quadratic_sym[:3]}...")
            print(f"    All match: {match and match_sym}")
    
    print(f"  All {n_tests} tests passed: {all_match}")
    return all_match

test_nonsymmetric_equivalence(n=4, rank=4)
test_nonsymmetric_equivalence(n=6, rank=4)

print("\n" + "=" * 70)
print("SUMMARY: THE GENERAL BILINEAR → QUADRATIC FORM EQUIVALENCE")
print("=" * 70)

print("""
For ANY bilinear layer y = D @ (L @ x ⊙ R @ x):

  y_i = x^T M^(i) x

where:
  M^(i)_{jk} = Σ_r D_{ir} L_{rj} R_{rk}

Properties:
  • If L = R (symmetric): M^(i) is symmetric
  • If L ≠ R (non-symmetric): M^(i) is NOT symmetric, but can use M_sym = (M + M^T)/2
  
  • If D is sparse (k non-zeros per row): M^(i) has rank ≤ k
  • If D has one non-zero per row: M^(i) is RANK-1 = d_i × L_{r_i} ⊗ R_{r_i}

The quadratic form view is ALWAYS equivalent to the original computation,
and often reveals structure (like rank) that's not obvious from the weights.
""")