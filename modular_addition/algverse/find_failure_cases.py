import torch
import torch.nn.functional as F

def task_2nd_argmax(x):
    return x.argsort(-1)[..., -2]

def test_algorithm(fn, name, n=4):
    """Test an algorithm on random Gaussian inputs."""
    x = torch.randn(100000, n)
    targets = task_2nd_argmax(x)
    logits = fn(x)
    preds = logits.argmax(-1)
    acc = (preds == targets).float().mean()
    print(f"{name}: {acc.item():.1%}")
    return acc.item()

# ============================================================
# ATTEMPT 1: Just linear (deviation from mean = argmax)
# ============================================================
def algo_linear(x):
    """Just prefer larger values. Should give argmax, not 2nd."""
    return x  # or equivalently: x - x.mean(dim=-1, keepdim=True)

# ============================================================
# ATTEMPT 2: Parabola (penalize large values)
# ============================================================
def algo_parabola(x, beta=0.5):
    """
    logit[i] = x[i] - beta * x[i]^2
    
    Parabola peaks at x = 1/(2*beta). 
    Values larger than the peak get penalized.
    """
    return x - beta * x ** 2

# ============================================================
# ATTEMPT 3: Penalize deviation from mean
# ============================================================
def algo_penalize_deviation(x, beta=1.0):
    """
    logit[i] = x[i] - beta * (x[i] - mean)^2
    
    Penalizes being far from the mean (either direction).
    """
    mean = x.mean(dim=-1, keepdim=True)
    return x - beta * (x - mean) ** 2

# ============================================================
# ATTEMPT 4: Penalize being larger than mean (one-sided)
# ============================================================
def algo_penalize_above_mean(x, beta=1.0):
    """
    logit[i] = x[i] - beta * relu(x[i] - mean)^2
    
    Only penalize values ABOVE the mean.
    But wait, we can't use relu in pure polynomial...
    """
    mean = x.mean(dim=-1, keepdim=True)
    above = x - mean
    return x - beta * above * above.abs()  # Cheating with abs!

# ============================================================
# ATTEMPT 5: Use cross-terms x[i] * sum(x)
# ============================================================
def algo_cross_sum(x, alpha=1.0, beta=0.3):
    """
    logit[i] = alpha * x[i] - beta * x[i] * sum(x)
    
    The x[i] * sum(x) term penalizes high values more when
    the overall sum is high.
    """
    s = x.sum(dim=-1, keepdim=True)
    return alpha * x - beta * x * s

# ============================================================
# ATTEMPT 6: x[i] * (sum - x[i]) = x[i] * sum_of_others
# ============================================================
def algo_cross_others(x, alpha=1.0, beta=0.3):
    """
    logit[i] = alpha * x[i] - beta * x[i] * sum(x[j] for j != i)
    
    Penalizes x[i] proportionally to "how positive are the others".
    If others are large and positive, being large yourself is penalized.
    """
    s = x.sum(dim=-1, keepdim=True)
    sum_others = s - x  # sum of all except x[i]
    return alpha * x - beta * x * sum_others

# ============================================================
# ATTEMPT 7: Quadratic "ranking" score
# ============================================================
def algo_quadratic_rank(x, alpha=1.0, beta=0.5):
    """
    logit[i] = alpha * x[i] - beta * x[i] * max(x)
    
    Penalize proportionally to the max value.
    The max itself gets the biggest penalty.
    
    But we can't compute max with polynomials...
    Let's approximate max with sum of squares?
    """
    # Soft approximation of max: sqrt(sum of x^2) ~ max for large values
    # Or just use sum of positive parts... but that needs relu
    
    # Pure quadratic approximation: use x[i] * (sum of x^2)
    # Larger x[i] with larger overall x^2 gets penalized
    sum_sq = (x ** 2).sum(dim=-1, keepdim=True)
    return alpha * x - beta * x * sum_sq / x.shape[-1]

# ============================================================
# TEST ALL
# ============================================================
print("="*50)
print("Testing hand-coded algorithms for n=4")
print("="*50)
print(f"Random baseline: {1/4:.1%}")
print()

test_algorithm(algo_linear, "Linear only (=argmax)")
print()

for beta in [0.3, 0.5, 0.7, 1.0]:
    test_algorithm(lambda x: algo_parabola(x, beta), f"Parabola (β={beta})")
print()

for beta in [0.5, 1.0, 2.0]:
    test_algorithm(lambda x: algo_penalize_deviation(x, beta), f"Penalize deviation (β={beta})")
print()

for beta in [0.1, 0.2, 0.3, 0.5]:
    test_algorithm(lambda x: algo_cross_sum(x, 1.0, beta), f"Cross-sum (β={beta})")
print()

for beta in [0.1, 0.2, 0.3, 0.5]:
    test_algorithm(lambda x: algo_cross_others(x, 1.0, beta), f"Cross-others (β={beta})")
print()

for beta in [0.3, 0.5, 1.0]:
    test_algorithm(lambda x: algo_quadratic_rank(x, 1.0, beta), f"Quadratic rank (β={beta})")

# ============================================================
# ANALYSIS: What does each term contribute?
# ============================================================
def analyze_contributions(x):
    """Break down the algorithm into interpretable parts."""
    n = x.shape[-1]
    targets = task_2nd_argmax(x)
    
    s = x.sum(dim=-1, keepdim=True)
    
    # Component contributions
    linear = x
    quadratic = -x ** 2
    cross = -x * (s - x)
    
    # Which component alone does best?
    print("Component analysis:")
    test_algorithm(lambda x: x, "  Linear only")
    test_algorithm(lambda x: -x**2, "  -x² only")
    test_algorithm(lambda x: -x*(x.sum(-1,keepdim=True)-x), "  -x*sum_others only")
    
    # Combinations
    print("\nCombinations:")
    test_algorithm(lambda x: x - 0.5*x**2, "  x - 0.5x²")
    test_algorithm(lambda x: x - 0.3*x*(x.sum(-1,keepdim=True)-x), "  x - 0.3*x*sum_others")
    test_algorithm(lambda x: x - 0.3*x**2 - 0.15*x*(x.sum(-1,keepdim=True)-x), 
                   "  x - 0.3x² - 0.15*x*sum_others")
def hand_coded_2nd_argmax(x):
    """
    logit[i] = a*x[i] - b*x[i]² - c*x[i]*sum(others)
    
    Interpretation:
    - a*x[i]: Prefer larger values (like argmax)
    - b*x[i]²: Penalize very large values (hurts the max most)  
    - c*x[i]*sum(others): Penalize when BOTH x[i] is large AND others are large
                          (the max tends to have large value when sum is large)
    """
    s = x.sum(dim=-1, keepdim=True)
    answer = x - 0.3 * x**2 - 0.15 * x * (s - x)
    print("handed coded answer:", answer)
    return answer
import torch

def construct_bilinear_weights(n=4, beta=2.0):
    """
    Construct weights for: logit[i] = x[i] - β*(x[i] - mean)²
    
    In bilinear form: y = Wx + D(Lx ⊙ Rx)
    """
    # Linear part: W = I (just pass through x)
    # But we can also absorb the cross-term x[i]*mean into W
    # W[i,j] = δ[i,j] + (2β/n) for the cross term contribution
    
    W = torch.eye(n) + (2 * beta / n) * torch.ones(n, n)
    
    # Quadratic part: -β * x[i]²
    # This is D @ (L @ x * R @ x) where we want just diagonal x[i]*x[i] terms
    # Use rank-1: L = R = I (identity), D = -β * I
    
    # But actually, let's use rank-n to capture x[i]² for each i
    L = torch.eye(n)  # L @ x = x, so (L @ x)[i] = x[i]
    R = torch.eye(n)  # R @ x = x
    D = -beta * torch.eye(n)  # Only diagonal: D[i,i] = -β
    
    return W, L, R, D

def algo_penalize_deviation_expanded(x, beta=2.0):
    """
    Expanded form showing it's a quadratic polynomial.
    
    logit[i] = x[i] - β*(x[i] - mean)²
             = x[i] - β*(x[i]² - 2*x[i]*mean + mean²)
             = x[i] - β*x[i]² + 2β*x[i]*mean - β*mean²
             
    The mean² term is same for all i, doesn't affect argmax.
    So effectively:
    
    logit[i] ∝ x[i] + 2β*mean*x[i] - β*x[i]²
             = x[i]*(1 + 2β*mean) - β*x[i]²
             = x[i]*(1 + 2β*mean - β*x[i])
    """
    n = x.shape[-1]
    mean = x.mean(dim=-1, keepdim=True)  # This is (1/n) * sum(x)
    
    # Pure polynomial form (no "mean" abstraction)
    sum_x = x.sum(dim=-1, keepdim=True)
    
    # logit[i] = x[i] - β*x[i]² + (2β/n)*x[i]*sum(x) - β*(sum(x)/n)²
    linear = x
    quadratic = -beta * x**2
    cross = (2 * beta / n) * x * sum_x  # x[i] * sum(x[j])
    # constant term (same for all i, ignored)
    
    return linear + quadratic + cross

# Test it
test_algorithm(lambda x: algo_penalize_deviation_expanded(x, 2.0), 
               "Penalize deviation (expanded)")

def bilinear_model(x, W, L, R, D):
    """Apply: y = Wx + D(Lx ⊙ Rx)"""
    linear = x @ W.T
    bilinear = (x @ L.T) * (x @ R.T) @ D.T
    return linear + bilinear

# Test it
n = 4
beta = 2.0
W, L, R, D = construct_bilinear_weights(n, beta)

print("Constructed weights:")
print(f"W =\n{W}")
print(f"L =\n{L}")
print(f"R =\n{R}")
print(f"D =\n{D}")

test_algorithm(lambda x: bilinear_model(x, W, L, R, D), 
               "Hand-constructed bilinear")

import torch

def construct_bilinear_correct(n=4, beta=2.0):
    """
    Construct y = Wx + D(Lx ⊙ Rx) for:
    logit[i] = x[i] - β*x[i]² + (2β/n)*x[i]*sum(x)
    
    Use rank = 2n:
    - First n components: capture x[i] * sum(x)
    - Next n components: capture x[i]²
    """
    rank = 2 * n
    
    # W = I (linear term)
    W = torch.eye(n)
    
    # L and R of shape (rank, n)
    L = torch.zeros(rank, n)
    R = torch.zeros(rank, n)
    
    # First n components: x[i] * sum(x)
    # L[k, k] = 1 -> (Lx)[k] = x[k]
    # R[k, :] = 1 -> (Rx)[k] = sum(x)
    for k in range(n):
        L[k, k] = 1.0
        R[k, :] = 1.0
    
    # Next n components: x[i]²
    # L[n+k, k] = 1, R[n+k, k] = 1 -> gives x[k]²
    for k in range(n):
        L[n + k, k] = 1.0
        R[n + k, k] = 1.0
    
    # D of shape (n, rank)
    # D[i, i] = 2β/n (coefficient for x[i]*sum(x))
    # D[i, n+i] = -β (extra coefficient for x[i]² to get net -β + 2β/n)
    D = torch.zeros(n, rank)
    for i in range(n):
        D[i, i] = 2 * beta / n        # x[i]*sum(x) term
        D[i, n + i] = -beta           # x[i]² term (adds to the 2β/n from above)
    
    return W, L, R, D


def bilinear_model(x, W, L, R, D):
    """Apply: y = Wx + D @ (Lx ⊙ Rx)"""
    linear = x @ W.T                        # (batch, n)
    Lx = x @ L.T                            # (batch, rank)
    Rx = x @ R.T                            # (batch, rank)
    bilinear = (Lx * Rx) @ D.T              # (batch, n)
    return linear + bilinear


def task_2nd_argmax(x):
    return x.argsort(-1)[..., -2]


def test_algorithm(fn, name, n=4):
    x = torch.randn(100000, n)
    targets = task_2nd_argmax(x)
    logits = fn(x)
    preds = logits.argmax(-1)
    acc = (preds == targets).float().mean()
    print(f"{name}: {acc.item():.1%}")
    return acc.item()


# Test
n = 4
beta = 2.0
W, L, R, D = construct_bilinear_correct(n, beta)

print("Constructed weights:")
print(f"W ({W.shape}):\n{W}\n")
print(f"L ({L.shape}):\n{L}\n")
print(f"R ({R.shape}):\n{R}\n")
print(f"D ({D.shape}):\n{D}\n")

# Verify coefficients
print("Verification of quadratic coefficients:")
print(f"  x[i]² coeff should be: β*(2-n)/n = {beta*(2-n)/n}")
print(f"  x[i]*x[j] coeff should be: 2β/n = {2*beta/n}")
print()

test_algorithm(lambda x: algo_penalize_deviation(x, beta), "Original algorithm")
test_algorithm(lambda x: bilinear_model(x, W, L, R, D), "Hand-constructed bilinear")

import torch

def construct_bilinear_minimal(n=4, beta=2.0):
    """
    Minimal rank = n construction for:
    logit[i] = x[i] - β*(x[i] - mean)²
    
    Key insight: (x[i] - mean)² = (x[i] - mean) * (x[i] - mean)
    
    So we set L = R = "subtract mean" projection
    """
    # L = R = I - (1/n)*ones
    # This computes: (Lx)[i] = x[i] - mean(x)
    L = torch.eye(n) - torch.ones(n, n) / n
    R = L.clone()
    
    # D = -β * I
    # So D @ (Lx * Rx) = -β * (x - mean)²
    D = -beta * torch.eye(n)
    
    # W = I (linear term)
    W = torch.eye(n)
    
    return W, L, R, D


def bilinear_model(x, W, L, R, D):
    """Apply: y = Wx + D @ (Lx ⊙ Rx)"""
    linear = x @ W.T
    Lx = x @ L.T
    Rx = x @ R.T
    bilinear = (Lx * Rx) @ D.T
    return linear + bilinear


def task_2nd_argmax(x):
    return x.argsort(-1)[..., -2]


def test_algorithm(fn, name, n=4):
    x = torch.randn(100000, n)
    targets = task_2nd_argmax(x)
    logits = fn(x)
    preds = logits.argmax(-1)
    acc = (preds == targets).float().mean()
    print(f"{name}: {acc.item():.1%}")


# Test
n = 4
beta = 2.0
W, L, R, D = construct_bilinear_minimal(n, beta)

print("Minimal rank-n construction:")
print(f"W =\n{W}\n")
print(f"L = R =\n{L}\n")
print(f"D =\n{D}\n")

print(f"Rank = {n}")
print()

test_algorithm(lambda x: bilinear_model(x, W, L, R, D), "Hand-constructed (rank n)")
