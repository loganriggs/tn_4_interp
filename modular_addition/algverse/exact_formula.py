import torch
import torch.nn as nn

# Train a good bilinear network for n=4
class BilinearStack(nn.Module):
    def __init__(self, n, num_layers=3, rank=32, use_linear=True):
        super().__init__()
        self.n = n
        self.num_layers = num_layers
        
        self.Ls = nn.ParameterList()
        self.Rs = nn.ParameterList()
        self.Ds = nn.ParameterList()
        self.Ws = nn.ParameterList() if use_linear else None
        
        for _ in range(num_layers):
            self.Ls.append(nn.Parameter(torch.randn(rank, n) * 0.1))
            self.Rs.append(nn.Parameter(torch.randn(rank, n) * 0.1))
            self.Ds.append(nn.Parameter(torch.randn(n, rank) * 0.1))
            if use_linear:
                self.Ws.append(nn.Parameter(torch.randn(n, n) * 0.1))
    
    def forward(self, x):
        h = x
        for i in range(self.num_layers):
            Lh = h @ self.Ls[i].T
            Rh = h @ self.Rs[i].T
            bilinear = (Lh * Rh) @ self.Ds[i].T
            if self.Ws:
                h = h + bilinear + h @ self.Ws[i].T
            else:
                h = h + bilinear
        return h


def task_2nd_argmax(x):
    return x.argsort(-1)[..., -2]


# Train
n = 4
model = BilinearStack(n, num_layers=3, rank=32, use_linear=True)
optimizer = torch.optim.Adam(model.parameters(), lr=0.01)

for step in range(10000):
    x = torch.randn(512, n)
    targets = task_2nd_argmax(x)
    logits = model(x)
    loss = nn.functional.cross_entropy(logits, targets)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

model.eval()

# Test accuracy
with torch.no_grad():
    x = torch.randn(100000, n)
    targets = task_2nd_argmax(x)
    logits = model(x)
    acc = (logits.argmax(-1) == targets).float().mean()
    print(f"Model accuracy: {acc.item():.1%}")


# Now probe: what does the network compute?
# Look at logit differences as a function of input

def probe_network(model, n=4):
    """
    Probe what the network has learned.
    """
    print("\n=== Probing learned network ===\n")
    
    # Test on sorted inputs with known 2nd argmax
    print("Test on canonical orderings:")
    for perm in [[0,1,2,3], [3,2,1,0], [1,3,2,0], [2,0,3,1]]:
        x = torch.tensor([[float(i) for i in perm]])  # values = positions
        logits = model(x)
        pred = logits.argmax(-1).item()
        target = task_2nd_argmax(x).item()
        print(f"  x={perm}, logits={logits[0].detach().numpy().round(2)}, pred={pred}, target={target}")
    
    # What features matter?
    # Compute gradient of logit[i] w.r.t. input
    print("\nGradient analysis (d logit[i] / d x[j]):")
    x = torch.randn(1, n, requires_grad=True)
    logits = model(x)
    
    for i in range(n):
        logits[0, i].backward(retain_graph=True)
        grad = x.grad.clone()
        x.grad.zero_()
        print(f"  d logit[{i}] / dx = {grad[0].detach().numpy().round(3)}")
    
    # Compare behavior on edge cases
    print("\nEdge case analysis:")
    
    # Case 1: Two values very close
    x = torch.tensor([[1.0, 1.001, 0.0, 2.0]])  # 2nd should be pos 0 or 1
    logits = model(x)
    target = task_2nd_argmax(x).item()
    print(f"  Close values: x={x[0].tolist()}, pred={logits.argmax(-1).item()}, target={target}")
    
    # Case 2: One extreme outlier
    x = torch.tensor([[0.0, 0.1, 0.2, 10.0]])  # 2nd should be pos 2
    logits = model(x)
    target = task_2nd_argmax(x).item()
    print(f"  Outlier max: x={x[0].tolist()}, pred={logits.argmax(-1).item()}, target={target}")
    
    # Case 3: Negative outlier
    x = torch.tensor([[-10.0, 0.0, 0.1, 0.2]])  # 2nd should be pos 2
    logits = model(x)
    target = task_2nd_argmax(x).item()
    print(f"  Outlier min: x={x[0].tolist()}, pred={logits.argmax(-1).item()}, target={target}")


probe_network(model, n)

#%%
import torch
import torch.nn as nn

def analyze_failure_modes(model, n=4, num_samples=100000):
    """
    Categorize where the model fails.
    """
    model.eval()
    
    x = torch.randn(num_samples, n)
    targets = task_2nd_argmax(x)
    
    with torch.no_grad():
        logits = model(x)
        preds = logits.argmax(-1)
    
    # Find failures
    failures = x[preds != targets]
    failure_targets = targets[preds != targets]
    failure_preds = preds[preds != targets]
    
    print(f"Total failures: {len(failures)} / {num_samples} ({len(failures)/num_samples:.1%})")
    
    # Analyze failure patterns
    print("\nFailure pattern analysis:")
    
    # What position does it predict vs actual?
    print("\nConfusion (pred vs target):")
    for pred_pos in range(n):
        for target_pos in range(n):
            count = ((failure_preds == pred_pos) & (failure_targets == target_pos)).sum().item()
            if count > 0:
                print(f"  Predicted {pred_pos}, actual {target_pos}: {count} ({count/len(failures):.1%})")
    
    # Analyze statistics of failures
    print("\nStatistics of failure cases vs successes:")
    
    successes = x[preds == targets]
    
    # Range (max - min)
    fail_range = (failures.max(dim=1).values - failures.min(dim=1).values).mean()
    succ_range = (successes.max(dim=1).values - successes.min(dim=1).values).mean()
    print(f"  Mean range: failures={fail_range:.2f}, successes={succ_range:.2f}")
    
    # Std
    fail_std = failures.std(dim=1).mean()
    succ_std = successes.std(dim=1).mean()
    print(f"  Mean std: failures={fail_std:.2f}, successes={succ_std:.2f}")
    
    # Gap between 2nd and 3rd
    sorted_fail = failures.sort(dim=1).values
    sorted_succ = successes.sort(dim=1).values
    
    fail_gap_23 = (sorted_fail[:, -2] - sorted_fail[:, -3]).mean()  # 2nd - 3rd
    succ_gap_23 = (sorted_succ[:, -2] - sorted_succ[:, -3]).mean()
    print(f"  Mean gap (2nd-3rd): failures={fail_gap_23:.2f}, successes={succ_gap_23:.2f}")
    
    fail_gap_12 = (sorted_fail[:, -1] - sorted_fail[:, -2]).mean()  # 1st - 2nd
    succ_gap_12 = (sorted_succ[:, -1] - sorted_succ[:, -2]).mean()
    print(f"  Mean gap (1st-2nd): failures={fail_gap_12:.2f}, successes={succ_gap_12:.2f}")
    
    # Look at some specific failures
    print("\nSample failures (sorted for clarity):")
    for i in range(min(10, len(failures))):
        xi = failures[i]
        target = failure_targets[i].item()
        pred = failure_preds[i].item()
        
        # Sort to understand structure
        sorted_vals, sorted_idx = xi.sort()
        second_largest_pos = sorted_idx[-2].item()
        
        print(f"  x_sorted={sorted_vals.numpy().round(2)}, pred_pos={pred}, target_pos={target}")


def test_distribution_dependence(model, n=4):
    """
    Does the model only work on Gaussian inputs?
    """
    print("\n=== Distribution dependence ===\n")
    
    distributions = {
        'Gaussian': lambda: torch.randn(10000, n),
        'Uniform[-1,1]': lambda: torch.rand(10000, n) * 2 - 1,
        'Uniform[0,1]': lambda: torch.rand(10000, n),
        'Laplace': lambda: torch.distributions.Laplace(0, 1).sample((10000, n)),
        'Exponential': lambda: torch.distributions.Exponential(1).sample((10000, n)),
        'Mixture': lambda: torch.randn(10000, n) + 3 * (torch.rand(10000, n) > 0.5).float(),
    }
    
    for name, sampler in distributions.items():
        x = sampler()
        targets = task_2nd_argmax(x)
        
        with torch.no_grad():
            logits = model(x)
            preds = logits.argmax(-1)
            acc = (preds == targets).float().mean()
        
        print(f"{name:20s}: {acc.item():.1%}")


# Run analysis
print("="*60)
print("FAILURE MODE ANALYSIS")
print("="*60)
analyze_failure_modes(model, n=4)

print("\n" + "="*60)
print("DISTRIBUTION DEPENDENCE")
print("="*60)
test_distribution_dependence(model, n=4)

# %%
import torch

def test_shifted_distributions(model, n=4):
    """
    Test if the issue is shift or spread.
    """
    print("=== Shift and Scale Analysis ===\n")
    
    tests = {
        'Gaussian (original)': lambda: torch.randn(10000, n),
        'Gaussian + 0.5': lambda: torch.randn(10000, n) + 0.5,
        'Gaussian + 2': lambda: torch.randn(10000, n) + 2,
        'Gaussian * 0.5': lambda: torch.randn(10000, n) * 0.5,
        'Gaussian * 0.25': lambda: torch.randn(10000, n) * 0.25,
        'Uniform[-1,1]': lambda: torch.rand(10000, n) * 2 - 1,
        'Uniform[0,1]': lambda: torch.rand(10000, n),
        'Uniform[0,1] centered': lambda: torch.rand(10000, n) - 0.5,  # Same as Uniform[-0.5, 0.5]
        'Uniform[-2,2]': lambda: torch.rand(10000, n) * 4 - 2,
    }
    
    for name, sampler in tests.items():
        x = sampler()
        targets = task_2nd_argmax(x)
        
        with torch.no_grad():
            logits = model(x)
            preds = logits.argmax(-1)
            acc = (preds == targets).float().mean()
        
        # Also compute stats
        mean_val = x.mean().item()
        std_val = x.std().item()
        
        print(f"{name:25s}: {acc.item():.1%}  (mean={mean_val:.2f}, std={std_val:.2f})")


def task_2nd_argmax(x):
    return x.argsort(-1)[..., -2]


test_shifted_distributions(model, n=4)

#%%
import torch
import torch.nn as nn

def test_with_normalization(model, n=4):
    """
    What if we normalize inputs before feeding to model?
    """
    print("=== With Input Normalization ===\n")
    
    def normalize(x):
        """Normalize each sample to mean=0, std=1"""
        mean = x.mean(dim=1, keepdim=True)
        std = x.std(dim=1, keepdim=True) + 1e-8
        return (x - mean) / std
    
    tests = {
        'Gaussian (original)': lambda: torch.randn(10000, n),
        'Gaussian * 0.25': lambda: torch.randn(10000, n) * 0.25,
        'Uniform[0,1]': lambda: torch.rand(10000, n),
        'Uniform[0,1] centered': lambda: torch.rand(10000, n) - 0.5,
        'Exponential': lambda: torch.distributions.Exponential(1).sample((10000, n)),
    }
    
    for name, sampler in tests.items():
        x = sampler()
        x_norm = normalize(x)
        targets = task_2nd_argmax(x)  # Target based on original x
        
        with torch.no_grad():
            # Without normalization
            logits_raw = model(x)
            acc_raw = (logits_raw.argmax(-1) == targets).float().mean()
            
            # With normalization
            logits_norm = model(x_norm)
            acc_norm = (logits_norm.argmax(-1) == targets).float().mean()
        
        print(f"{name:25s}: raw={acc_raw.item():.1%}, normalized={acc_norm.item():.1%}")


def task_2nd_argmax(x):
    return x.argsort(-1)[..., -2]


test_with_normalization(model, n=4)

#%%
import torch
import torch.nn as nn
import numpy as np

def mechanistic_analysis(model, n=4):
    """
    Extract and analyze the learned weights mechanistically.
    """
    print("="*60)
    print("MECHANISTIC WEIGHT ANALYSIS")
    print("="*60)
    
    # Extract all weights
    print("\n=== Layer Shapes ===")
    for i in range(model.num_layers):
        L = model.Ls[i].data
        R = model.Rs[i].data
        D = model.Ds[i].data
        W = model.Ws[i].data if model.Ws else None
        
        print(f"\nLayer {i}:")
        print(f"  L: {tuple(L.shape)}, R: {tuple(R.shape)}, D: {tuple(D.shape)}", end="")
        if W is not None:
            print(f", W: {tuple(W.shape)}")
        else:
            print()
    
    # Analyze Layer 0 in detail (most interpretable)
    print("\n" + "="*60)
    print("LAYER 0 DETAILED ANALYSIS")
    print("="*60)
    
    L0 = model.Ls[0].data
    R0 = model.Rs[0].data
    D0 = model.Ds[0].data
    W0 = model.Ws[0].data if model.Ws else torch.zeros(n, n)
    
    rank = L0.shape[0]
    
    # The bilinear term computes: D0 @ (L0 @ x * R0 @ x)
    # Each rank component k contributes: D0[:, k] * (L0[k] @ x) * (R0[k] @ x)
    
    # This is equivalent to a quadratic form:
    # output[i] = sum_k D0[i,k] * (sum_j L0[k,j] x[j]) * (sum_m R0[k,m] x[m])
    #           = sum_{j,m} (sum_k D0[i,k] L0[k,j] R0[k,m]) x[j] x[m]
    #           = sum_{j,m} Q[i,j,m] x[j] x[m]
    
    # Compute the effective quadratic tensor Q[i,j,m]
    # Q[i,j,m] = sum_k D0[i,k] L0[k,j] R0[k,m]
    
    Q = torch.einsum('ik,kj,km->ijm', D0, L0, R0)
    print(f"\nQuadratic tensor Q[i,j,m] shape: {Q.shape}")
    print("(Q[i,j,m] is coefficient of x[j]*x[m] in output[i])")
    
    # Decompose Q into symmetric and antisymmetric parts
    # x[j]*x[m] + x[m]*x[j] = 2*x[j]*x[m], so effective coefficient is (Q[i,j,m] + Q[i,m,j])/2
    Q_sym = (Q + Q.transpose(1, 2)) / 2
    
    print("\n--- Effective Quadratic Coefficients (symmetrized) ---")
    print("Q_sym[i,j,m] = coefficient of x[j]*x[m] in logit[i]")
    
    for i in range(n):
        print(f"\nOutput position {i}:")
        print(Q_sym[i].numpy().round(3))
    
    # Extract diagonal (x[j]^2 terms) and off-diagonal (x[j]*x[m] terms)
    print("\n--- Diagonal vs Off-diagonal ---")
    for i in range(n):
        diag = torch.diag(Q_sym[i])
        off_diag_sum = Q_sym[i].sum() - diag.sum()
        print(f"Output {i}: diag={diag.numpy().round(2)}, off_diag_sum={off_diag_sum.item():.2f}")
    
    # Analyze the linear term W0
    print("\n--- Linear Term W0 ---")
    print(W0.numpy().round(3))
    
    # Check if W0 has structure (e.g., W0 ≈ a*I + b*ones)
    W0_diag_mean = torch.diag(W0).mean().item()
    W0_offdiag_mean = (W0.sum() - torch.diag(W0).sum()) / (n*n - n)
    print(f"\nW0 structure: diag_mean={W0_diag_mean:.3f}, offdiag_mean={W0_offdiag_mean:.3f}")
    print(f"Approx: W0 ≈ {W0_diag_mean:.2f}*I + {W0_offdiag_mean:.2f}*ones")
    
    # Total first-layer transformation
    print("\n--- Combined Layer 0 Output ---")
    print("h = x + W0 @ x + D0 @ (L0 @ x * R0 @ x)")
    print("  = (I + W0) @ x + quadratic_terms")
    
    effective_linear = torch.eye(n) + W0
    print(f"\nEffective linear (I + W0):")
    print(effective_linear.numpy().round(3))
    
    return Q_sym, W0


def analyze_what_quadratic_computes(Q_sym, n=4):
    """
    Try to interpret the quadratic tensor.
    """
    print("\n" + "="*60)
    print("INTERPRETING THE QUADRATIC FORM")
    print("="*60)
    
    # For 2nd argmax, we hypothesized: logit[i] ∝ -(x[i] - mean)²
    # Let's check if Q_sym matches this pattern
    
    # (x[i] - mean)² = x[i]² - 2*x[i]*mean + mean²
    #                = x[i]² - (2/n)*x[i]*sum(x) + (1/n²)*sum(x)²
    #                = x[i]² - (2/n)*sum_j(x[i]*x[j]) + (1/n²)*sum_{j,m}(x[j]*x[m])
    
    # Coefficients for -(x[i] - mean)²:
    # - x[i]²: -1
    # - x[i]*x[j] (j≠i): +2/n
    # - x[j]*x[m] (j,m≠i): -1/n²
    
    print("\nExpected pattern for -(x[i] - mean)²:")
    print(f"  x[i]² coeff: -1")
    print(f"  x[i]*x[j] coeff: +{2/n:.3f}")
    print(f"  x[j]*x[m] coeff: -{1/n**2:.3f}")
    
    print("\nActual Q_sym patterns:")
    for i in range(n):
        self_coeff = Q_sym[i, i, i].item()
        cross_i_coeffs = [Q_sym[i, i, j].item() for j in range(n) if j != i]
        cross_i_mean = np.mean(cross_i_coeffs)
        other_coeffs = [Q_sym[i, j, m].item() for j in range(n) for m in range(n) if j != i and m != i]
        other_mean = np.mean(other_coeffs) if other_coeffs else 0
        
        print(f"  Output {i}: x[{i}]²={self_coeff:.3f}, x[{i}]*x[j]={cross_i_mean:.3f}, x[j]*x[m]={other_mean:.3f}")
    
    # Compare to theoretical
    print("\nRatio check (if following -(x[i]-mean)² pattern):")
    print("  x[i]² / x[i]*x[j] should be ≈", -1 / (2/n), f"= {-n/2}")
    for i in range(n):
        self_coeff = Q_sym[i, i, i].item()
        cross_i_coeffs = [Q_sym[i, i, j].item() + Q_sym[i, j, i].item() for j in range(n) if j != i]
        cross_i_mean = np.mean(cross_i_coeffs) / 2  # Divide by 2 because we summed both orderings
        if abs(cross_i_mean) > 0.01:
            ratio = self_coeff / cross_i_mean
            print(f"  Output {i}: ratio = {ratio:.2f}")


def trace_forward_pass(model, x, n=4):
    """
    Trace through a specific input to see what each layer contributes.
    """
    print("\n" + "="*60)
    print("FORWARD PASS TRACE")
    print("="*60)
    
    print(f"\nInput x = {x.numpy().round(3)}")
    print(f"Target 2nd argmax position: {task_2nd_argmax(x.unsqueeze(0)).item()}")
    
    h = x.clone()
    
    for layer_idx in range(model.num_layers):
        L = model.Ls[layer_idx].data
        R = model.Rs[layer_idx].data
        D = model.Ds[layer_idx].data
        W = model.Ws[layer_idx].data if model.Ws else None
        
        Lh = h @ L.T
        Rh = h @ R.T
        bilinear = (Lh * Rh) @ D.T
        
        if W is not None:
            linear = h @ W.T
            h_new = h + bilinear + linear
        else:
            linear = torch.zeros(n)
            h_new = h + bilinear
        
        print(f"\n--- Layer {layer_idx} ---")
        print(f"  Input h = {h.numpy().round(3)}")
        print(f"  Lh = {Lh.numpy().round(3)}")
        print(f"  Rh = {Rh.numpy().round(3)}")
        print(f"  Lh * Rh = {(Lh * Rh).numpy().round(3)}")
        print(f"  Bilinear = {bilinear.numpy().round(3)}")
        print(f"  Linear = {linear.numpy().round(3)}")
        print(f"  Output h = {h_new.numpy().round(3)}")
        
        h = h_new
    
    print(f"\n--- Final ---")
    print(f"Logits = {h.numpy().round(3)}")
    print(f"Prediction = {h.argmax().item()}")


def task_2nd_argmax(x):
    return x.argsort(-1)[..., -2]


# Run all analyses
Q_sym, W0 = mechanistic_analysis(model, n=4)
analyze_what_quadratic_computes(Q_sym, n=4)

# Trace a few examples
print("\n" + "="*60)
print("EXAMPLE TRACES")
print("="*60)

# Simple ordered case
trace_forward_pass(model, torch.tensor([0., 1., 2., 3.]), n=4)

# Reversed
trace_forward_pass(model, torch.tensor([3., 2., 1., 0.]), n=4)

# Random
trace_forward_pass(model, torch.tensor([0.5, -1.2, 0.8, -0.3]), n=4)

#%% 
import torch

def analyze_amplification_mechanism(model, n=4):
    """
    Understand how later layers amplify the signal.
    """
    print("="*60)
    print("AMPLIFICATION ANALYSIS")
    print("="*60)
    
    # Test on many inputs and track layer-by-layer statistics
    x = torch.randn(10000, n)
    targets = task_2nd_argmax(x)
    
    # Track activations through layers
    h = x.clone()
    
    for layer_idx in range(model.num_layers):
        L = model.Ls[layer_idx].data
        R = model.Rs[layer_idx].data
        D = model.Ds[layer_idx].data
        W = model.Ws[layer_idx].data if model.Ws else None
        
        Lh = h @ L.T
        Rh = h @ R.T
        bilinear = (Lh * Rh) @ D.T
        linear = h @ W.T if W is not None else torch.zeros_like(h)
        
        h_new = h + bilinear + linear
        
        # Compute statistics
        h_std = h.std(dim=1).mean().item()
        h_range = (h.max(dim=1).values - h.min(dim=1).values).mean().item()
        bilinear_norm = bilinear.abs().mean().item()
        linear_norm = linear.abs().mean().item()
        
        # How often is argmax of h the correct 2nd argmax?
        h_pred = h.argmax(dim=1)
        h_acc = (h_pred == targets).float().mean().item()
        
        # Gap between top-2
        sorted_h = h.sort(dim=1, descending=True).values
        gap_12 = (sorted_h[:, 0] - sorted_h[:, 1]).mean().item()
        
        print(f"\nAfter Layer {layer_idx}:")
        print(f"  h std: {h_std:.2f}, range: {h_range:.2f}")
        print(f"  |bilinear|: {bilinear_norm:.2f}, |linear|: {linear_norm:.2f}")
        print(f"  Current accuracy: {h_acc:.1%}")
        print(f"  Gap (1st - 2nd): {gap_12:.2f}")
        
        h = h_new
    
    # Final
    final_acc = (h.argmax(dim=1) == targets).float().mean().item()
    print(f"\nFinal accuracy: {final_acc:.1%}")


def analyze_layer_roles(model, n=4):
    """
    What does each layer contribute to the computation?
    """
    print("\n" + "="*60)
    print("LAYER ROLE ANALYSIS")
    print("="*60)
    
    x = torch.randn(10000, n)
    targets = task_2nd_argmax(x)
    
    # Test with different numbers of layers
    for num_active_layers in range(model.num_layers + 1):
        h = x.clone()
        
        for layer_idx in range(num_active_layers):
            L = model.Ls[layer_idx].data
            R = model.Rs[layer_idx].data
            D = model.Ds[layer_idx].data
            W = model.Ws[layer_idx].data if model.Ws else None
            
            bilinear = (h @ L.T) * (h @ R.T) @ D.T
            linear = h @ W.T if W is not None else torch.zeros_like(h)
            h = h + bilinear + linear
        
        acc = (h.argmax(dim=1) == targets).float().mean().item()
        print(f"Using {num_active_layers} layers: {acc:.1%}")


def analyze_bilinear_vs_linear_contribution(model, n=4):
    """
    How much does bilinear vs linear contribute at each layer?
    """
    print("\n" + "="*60)
    print("BILINEAR VS LINEAR CONTRIBUTION")
    print("="*60)
    
    x = torch.randn(10000, n)
    targets = task_2nd_argmax(x)
    
    # Test: only linear (no bilinear)
    print("\nLinear only (skip + W, no bilinear):")
    h = x.clone()
    for layer_idx in range(model.num_layers):
        W = model.Ws[layer_idx].data if model.Ws else torch.zeros(n, n)
        h = h + h @ W.T  # Skip + linear, no bilinear
    acc = (h.argmax(dim=1) == targets).float().mean().item()
    print(f"  Accuracy: {acc:.1%}")
    
    # Test: only bilinear (no linear W)
    print("\nBilinear only (skip + bilinear, no W):")
    h = x.clone()
    for layer_idx in range(model.num_layers):
        L = model.Ls[layer_idx].data
        R = model.Rs[layer_idx].data
        D = model.Ds[layer_idx].data
        bilinear = (h @ L.T) * (h @ R.T) @ D.T
        h = h + bilinear  # Skip + bilinear, no W
    acc = (h.argmax(dim=1) == targets).float().mean().item()
    print(f"  Accuracy: {acc:.1%}")
    
    # Test: full model
    print("\nFull model (skip + bilinear + W):")
    h = x.clone()
    for layer_idx in range(model.num_layers):
        L = model.Ls[layer_idx].data
        R = model.Rs[layer_idx].data
        D = model.Ds[layer_idx].data
        W = model.Ws[layer_idx].data if model.Ws else torch.zeros(n, n)
        bilinear = (h @ L.T) * (h @ R.T) @ D.T
        h = h + bilinear + h @ W.T
    acc = (h.argmax(dim=1) == targets).float().mean().item()
    print(f"  Accuracy: {acc:.1%}")


def task_2nd_argmax(x):
    return x.argsort(-1)[..., -2]


# Run analyses
analyze_amplification_mechanism(model, n=4)
analyze_layer_roles(model, n=4)
analyze_bilinear_vs_linear_contribution(model, n=4)