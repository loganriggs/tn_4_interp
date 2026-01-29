"""
================================================================================
TENSOR NETWORKS FOR INTERPRETABLE ALGORITHMIC LEARNING
================================================================================

Project Overview and Implementation Guide

This project explores whether tensor network / bilinear architectures can solve
simple algorithmic tasks in a more interpretable way than traditional RNNs.

================================================================================
TABLE OF CONTENTS
================================================================================
1. Motivation
2. The AlgZoo Benchmark
3. Why Tensor Networks / Bilinear Layers?
4. Architectures
5. Experiments
6. Interpretability Analysis
7. Success Criteria
================================================================================
"""

import torch
import torch.nn as nn
from typing import Callable


# ==============================================================================
# 1. MOTIVATION
# ==============================================================================
"""
THE INTERPRETABILITY CRISIS AT SMALL SCALE
------------------------------------------

The Alignment Research Center (ARC) trained tiny RNNs (<1500 params) on simple
algorithmic tasks like "find the 2nd largest number in a sequence." Shockingly,
even their 432-parameter model remains poorly understood:

    - They identified some features (running max, leave-one-out-max)
    - But couldn't produce a mechanistic estimate of accuracy competitive 
      with random sampling
    - The model achieves 95% accuracy, but WHY remains elusive

If we can't interpret 432 parameters, how will we interpret billions?

THE HOPE
--------
RNNs create interpretability challenges because:
    1. Sequential processing creates complex state evolution
    2. ReLU activations create exponentially many piecewise-linear regions
    3. The hidden state mixes information in opaque ways

Tensor networks / bilinear architectures might help because:
    1. They explicitly represent polynomial interactions: y = Σ w_ijk x_i x_j x_k
    2. The polynomial structure is mathematically explicit
    3. No activation functions = no piecewise regions to enumerate
    4. Cross-terms (x_i * x_j) directly correspond to "comparisons"

For 2nd argmax, we need to compute something like:
    "position i has the 2nd largest value" 
    = "exactly one other position j has x_j > x_i"
    
This is fundamentally about pairwise comparisons, which bilinear terms capture!
"""


# ==============================================================================
# 2. THE ALGZOO BENCHMARK
# ==============================================================================

def task_2nd_argmax(x: torch.Tensor) -> torch.Tensor:
    """
    Find the position of the second-largest element.
    
    Input:  x of shape (batch, seq_len) - sequence of real numbers
    Output: indices of shape (batch,) - position of 2nd largest
    
    Example:
        x = [0.3, 1.7, -0.5, 2.1, 0.8]
              ^         ^
           2nd max   1st max
           (pos 1)   (pos 3)
        
        output = 1
    """
    return x.argsort(-1)[..., -2]


def task_argmax(x: torch.Tensor) -> torch.Tensor:
    """Find position of largest element (easier baseline)."""
    return x.argsort(-1)[..., -1]


def task_argmedian(x: torch.Tensor) -> torch.Tensor:
    """Find position of median element."""
    return x.argsort(-1)[..., x.shape[-1] // 2]


# Data generation: just sample from standard Gaussian
def generate_batch(batch_size: int, seq_len: int) -> tuple[torch.Tensor, torch.Tensor]:
    x = torch.randn(batch_size, seq_len)
    targets = task_2nd_argmax(x)
    return x, targets


"""
THE ALGZOO RNN BASELINE
-----------------------
Architecture: 1-layer ReLU RNN with scalar inputs
    
    h_t = ReLU(W_hh @ h_{t-1} + W_hi * x_t)
    logits = W_oh @ h_T

Parameters (for hidden=16, seq_len=10):
    W_hi: (16, 1)   =  16 params
    W_hh: (16, 16)  = 256 params  <- majority!
    W_oh: (10, 16)  = 160 params
    Total:            432 params

Performance: ~95% accuracy on 2nd argmax (seq_len=10)
"""


# ==============================================================================
# 3. WHY TENSOR NETWORKS / BILINEAR LAYERS?
# ==============================================================================
"""
BILINEAR LAYERS EXPLAINED
-------------------------

Standard linear layer:
    y = Wx + b
    
Bilinear layer:
    y = D(Lx ⊙ Rx)
    
    where ⊙ is elementwise multiplication (Hadamard product)
    
Expanded form:
    y_i = Σ_k D_ik (Σ_j L_jk x_j)(Σ_m R_mk x_m)
        = Σ_jmk D_ik L_jk R_mk x_j x_m
        
This explicitly creates QUADRATIC terms x_j * x_m !

For comparisons like "is x_i > x_j?", we want terms involving both x_i and x_j.
Bilinear layers give us these directly.


THE MU-LAYER (from MONet)
-------------------------

    y = C[(Ax) ⊙ (BDx) + Ax]
    
    - Ax:    linear projection (1st degree)
    - BDx:   low-rank linear projection  
    - Ax⊙BDx: quadratic interaction (2nd degree)
    - +Ax:   skip connection preserves 1st degree terms
    - C:     output projection
    
Stacking N layers creates up to 2^N degree polynomials.


MLP-MIXER PARADIGM
------------------

Traditional MLP-Mixer alternates:
    1. Token mixing: information flows ACROSS positions
    2. Channel mixing: information flows WITHIN each position

For our task:
    - Token mixing lets position i "see" position j's value
    - Channel mixing combines these into useful features

We can make both bilinear instead of MLP+activation.
"""


# ==============================================================================
# 4. ARCHITECTURES
# ==============================================================================

# ------------------------------------------------------------------------------
# 4.1 BILINEAR MLP-MIXER
# ------------------------------------------------------------------------------

class BilinearMixerBlock(nn.Module):
    """
    One block of bilinear MLP-Mixer.
    
    Token mixing:   operates across seq_len (shared for all channels)
    Channel mixing: operates across hidden_dim (shared for all positions)
    """
    def __init__(self, seq_len: int, hidden_dim: int, token_rank: int, channel_rank: int):
        super().__init__()
        # Token mixing: y = D(Lx ⊙ Rx) on seq dimension
        self.token_L = nn.Linear(seq_len, token_rank, bias=False)
        self.token_R = nn.Linear(seq_len, token_rank, bias=False)
        self.token_D = nn.Linear(token_rank, seq_len, bias=False)
        
        # Channel mixing: y = D(Lx ⊙ Rx) on hidden dimension
        self.channel_L = nn.Linear(hidden_dim, channel_rank, bias=False)
        self.channel_R = nn.Linear(hidden_dim, channel_rank, bias=False)
        self.channel_D = nn.Linear(channel_rank, hidden_dim, bias=False)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (batch, seq_len, hidden_dim)
        
        # Token mixing (transpose to put seq_len last)
        y = x.transpose(1, 2)  # (batch, hidden_dim, seq_len)
        y = self.token_D(self.token_L(y) * self.token_R(y))
        x = x + y.transpose(1, 2)
        
        # Channel mixing
        y = self.channel_D(self.channel_L(x) * self.channel_R(x))
        x = x + y
        
        return x


class BilinearMLPMixer(nn.Module):
    """
    Full bilinear MLP-Mixer for sequence classification.
    
    Params = hidden_dim + num_layers * (3*seq*tr + 3*hid*cr) + hidden_dim
    
    For seq=10, hid=8, tr=4, cr=4, layers=2:
        8 + 2*(120 + 96) + 8 = 448 params
    """
    def __init__(self, seq_len: int, hidden_dim: int = 8, 
                 token_rank: int = 4, channel_rank: int = 4, num_layers: int = 2):
        super().__init__()
        self.embed = nn.Linear(1, hidden_dim, bias=False)
        self.layers = nn.ModuleList([
            BilinearMixerBlock(seq_len, hidden_dim, token_rank, channel_rank)
            for _ in range(num_layers)
        ])
        self.head = nn.Linear(hidden_dim, 1, bias=False)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (batch, seq_len)
        x = self.embed(x.unsqueeze(-1))  # (batch, seq_len, hidden_dim)
        for layer in self.layers:
            x = layer(x)
        return self.head(x).squeeze(-1)  # (batch, seq_len) logits


# ------------------------------------------------------------------------------
# 4.2 MONET-STYLE (Spatial Shift + Mu-Layer)
# ------------------------------------------------------------------------------

class MuLayer(nn.Module):
    """
    The Mu-Layer from MONet: y = C[(Ax) ⊙ (BDx) + Ax]
    
    Captures multiplicative interactions with a skip connection.
    """
    def __init__(self, dim: int, rank: int):
        super().__init__()
        self.A = nn.Linear(dim, dim, bias=False)
        self.D = nn.Linear(dim, rank, bias=False)
        self.B = nn.Linear(rank, dim, bias=False)
        self.C = nn.Linear(dim, dim, bias=False)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        Ax = self.A(x)
        BDx = self.B(self.D(x))
        return self.C(Ax * BDx + Ax)


class MONetBlock(nn.Module):
    """
    MONet-style block: spatial shift + Mu-Layer.
    
    Spatial shift provides FREE cross-position communication.
    Mu-Layer does channel mixing with multiplicative interactions.
    """
    def __init__(self, seq_len: int, hidden_dim: int, mu_rank: int):
        super().__init__()
        self.seq_len = seq_len
        self.hidden_dim = hidden_dim
        self.mu_layer = MuLayer(hidden_dim, mu_rank)
    
    def spatial_shift(self, x: torch.Tensor) -> torch.Tensor:
        """
        Shift different channel groups by different amounts.
        After shift, each position contains info from neighbors.
        
        x: (batch, seq_len, hidden_dim)
        """
        H = self.hidden_dim
        g = H // 4
        if g == 0:
            return x
        
        x = x.clone()
        x[:, :, 0:g] = torch.roll(x[:, :, 0:g], shifts=1, dims=1)      # right 1
        x[:, :, g:2*g] = torch.roll(x[:, :, g:2*g], shifts=-1, dims=1) # left 1
        x[:, :, 2*g:3*g] = torch.roll(x[:, :, 2*g:3*g], shifts=2, dims=1)  # right 2
        x[:, :, 3*g:4*g] = torch.roll(x[:, :, 3*g:4*g], shifts=-2, dims=1) # left 2
        return x
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x_shifted = self.spatial_shift(x)
        return x + self.mu_layer(x_shifted)


class MONetStyle(nn.Module):
    """
    MONet-inspired architecture.
    
    Params = hidden_dim + num_layers * (2*hid^2 + 2*hid*rank) + hidden_dim
    
    For hid=8, rank=5, layers=2:
        8 + 2*(128 + 80) + 8 = 432 params
    """
    def __init__(self, seq_len: int, hidden_dim: int = 8, 
                 mu_rank: int = 5, num_layers: int = 2):
        super().__init__()
        self.embed = nn.Linear(1, hidden_dim, bias=False)
        self.layers = nn.ModuleList([
            MONetBlock(seq_len, hidden_dim, mu_rank)
            for _ in range(num_layers)
        ])
        self.head = nn.Linear(hidden_dim, 1, bias=False)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.embed(x.unsqueeze(-1))
        for layer in self.layers:
            x = layer(x)
        return self.head(x).squeeze(-1)


# ------------------------------------------------------------------------------
# 4.3 PURE BILINEAR TOKEN MIXER (no channel mixing)
# ------------------------------------------------------------------------------

class BilinearTokenOnly(nn.Module):
    """
    Simplest architecture: only bilinear token mixing.
    
    Question: is channel mixing necessary, or can pure token mixing
    with bilinear interactions suffice?
    """
    def __init__(self, seq_len: int, hidden_dim: int = 6, 
                 rank: int = 7, num_layers: int = 2):
        super().__init__()
        self.embed = nn.Linear(1, hidden_dim, bias=False)
        
        self.layers = nn.ModuleList()
        for _ in range(num_layers):
            self.layers.append(nn.ModuleDict({
                'L': nn.Linear(seq_len, rank, bias=False),
                'R': nn.Linear(seq_len, rank, bias=False),
                'D': nn.Linear(rank, seq_len, bias=False),
            }))
        
        self.head = nn.Linear(hidden_dim, 1, bias=False)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.embed(x.unsqueeze(-1))
        
        for layer in self.layers:
            y = x.transpose(1, 2)
            y = layer['D'](layer['L'](y) * layer['R'](y))
            x = x + y.transpose(1, 2)
        
        return self.head(x).squeeze(-1)


# ------------------------------------------------------------------------------
# 4.4 BASELINE: Original AlgZoo RNN
# ------------------------------------------------------------------------------

class AlgZooRNN(nn.Module):
    """
    The original AlgZoo architecture for comparison.
    
    1-layer ReLU RNN with scalar inputs.
    """
    def __init__(self, seq_len: int, hidden_size: int = 16):
        super().__init__()
        self.rnn = nn.RNN(
            input_size=1,
            hidden_size=hidden_size,
            nonlinearity="relu",
            bias=False,
            batch_first=True,
        )
        self.linear = nn.Linear(hidden_size, seq_len, bias=False)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        _, final_state = self.rnn(x.unsqueeze(-1), None)
        return self.linear(final_state.squeeze(0))


# ==============================================================================
# 5. EXPERIMENTS
# ==============================================================================

def count_params(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters())


def train_model(
    model: nn.Module,
    task_fn: Callable = task_2nd_argmax,
    seq_len: int = 10,
    steps: int = 50_000,
    batch_size: int = 128,
    lr: float = 0.01,
    weight_decay: float = 0.001,
    log_every: int = 1000,
    device: str = "cpu",
) -> dict:
    """Train a model and return metrics."""
    model = model.to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    
    history = {'loss': [], 'acc': []}
    
    for step in range(steps):
        x = torch.randn(batch_size, seq_len, device=device)
        targets = task_fn(x)
        
        logits = model(x)
        loss = nn.functional.cross_entropy(logits, targets)
        acc = (logits.argmax(-1) == targets).float().mean()
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        if step % log_every == 0:
            history['loss'].append(loss.item())
            history['acc'].append(acc.item())
            print(f"Step {step:5d}: loss={loss.item():.3f}, acc={acc.item():.3f}")
    
    # Final evaluation
    model.eval()
    with torch.no_grad():
        x = torch.randn(10000, seq_len, device=device)
        targets = task_fn(x)
        logits = model(x)
        final_acc = (logits.argmax(-1) == targets).float().mean().item()
    
    print(f"Final accuracy: {final_acc:.3f}")
    return {'history': history, 'final_acc': final_acc}


def run_comparison(seq_len: int = 10, target_params: int = 432):
    """Run comparison of all architectures."""
    
    print("\n" + "="*70)
    print("ARCHITECTURE COMPARISON")
    print(f"Task: 2nd argmax, seq_len={seq_len}, target params≈{target_params}")
    print("="*70)
    
    models = {
        "AlgZoo RNN (baseline)": AlgZooRNN(seq_len=seq_len, hidden_size=16),
        "Bilinear MLP-Mixer": BilinearMLPMixer(seq_len=seq_len, hidden_dim=8, 
                                                token_rank=4, channel_rank=4, num_layers=2),
        "MONet-style": MONetStyle(seq_len=seq_len, hidden_dim=8, mu_rank=5, num_layers=2),
        "Bilinear Token-Only": BilinearTokenOnly(seq_len=seq_len, hidden_dim=6, 
                                                  rank=7, num_layers=2),
    }
    
    results = {}
    for name, model in models.items():
        print(f"\n{'='*70}")
        print(f"{name}: {count_params(model)} parameters")
        print("="*70)
        results[name] = train_model(model, seq_len=seq_len)
    
    # Summary
    print("\n" + "="*70)
    print("SUMMARY")
    print("="*70)
    for name, result in results.items():
        print(f"{name:30s}: {result['final_acc']:.3f}")
    
    return results


# ==============================================================================
# 6. INTERPRETABILITY ANALYSIS
# ==============================================================================
"""
ANALYZING TRAINED BILINEAR MODELS
---------------------------------

The key advantage: we can write out the exact polynomial computed!

For a 1-layer bilinear token mixer:
    
    output[i] = Σ_c head[c] * (embed[c] * x[i] + 
                               Σ_jmk D[i,k] L[j,k] R[m,k] embed[c] embed[c] x[j] x[m])

This is a polynomial in the inputs x[0], ..., x[n-1] that we can expand and examine.

INTERPRETABILITY QUESTIONS
--------------------------
1. What polynomial does the model compute for each output logit?

2. Do the cross-terms (x[i] * x[j]) have interpretable coefficients?
   - Positive coefficient: "i is 2nd argmax when x[i] and x[j] are both large"?
   - Negative coefficient: "penalize when both are large" (can't both be 2nd)?

3. Can we identify "max" and "2nd max" circuits?
   - Max might look like: large positive self-term (x[i]^2)
   - 2nd max: positive cross-terms with specific structure?

4. How does the polynomial structure compare to the RNN's piecewise-linear regions?
"""


def extract_polynomial_coefficients(model: BilinearTokenOnly, seq_len: int) -> dict:
    """
    Extract the polynomial coefficients from a trained bilinear model.
    
    For a simple 1-layer model, compute the effective quadratic form.
    
    Returns dict with:
        - linear: (seq_len, seq_len) linear coefficients for each output
        - quadratic: (seq_len, seq_len, seq_len) quadratic coefficients
    """
    model.eval()
    
    # Get weights
    embed = model.embed.weight.data  # (hidden_dim, 1)
    head = model.head.weight.data    # (1, hidden_dim)
    
    # For 1 layer: output[i] = head @ (embed * x[i] + bilinear_term)
    # This is a simplification; full extraction is more complex for multiple layers
    
    # Probe the model with basis vectors to extract coefficients
    coeffs = {'linear': torch.zeros(seq_len, seq_len),
              'quadratic': torch.zeros(seq_len, seq_len, seq_len)}
    
    # Linear terms: derivative at x=0
    with torch.no_grad():
        for i in range(seq_len):
            x = torch.zeros(1, seq_len)
            x[0, i] = 1.0
            # Approximate derivative
            eps = 0.001
            x_plus = x.clone()
            x_plus[0, i] += eps
            x_minus = x.clone()
            x_minus[0, i] -= eps
            deriv = (model(x_plus) - model(x_minus)) / (2 * eps)
            coeffs['linear'][:, i] = deriv.squeeze()
    
    # Quadratic terms would require second derivatives or polynomial fitting
    # (Left as exercise for detailed analysis)
    
    return coeffs


def visualize_coefficients(coeffs: dict, output_idx: int = 0):
    """Visualize the polynomial coefficients for interpretability."""
    import matplotlib.pyplot as plt
    
    linear = coeffs['linear'][output_idx].numpy()
    
    plt.figure(figsize=(10, 4))
    plt.bar(range(len(linear)), linear)
    plt.xlabel('Input position')
    plt.ylabel('Linear coefficient')
    plt.title(f'Linear coefficients for output position {output_idx}')
    plt.axhline(y=0, color='k', linestyle='-', linewidth=0.5)
    plt.show()


# ==============================================================================
# 7. SUCCESS CRITERIA
# ==============================================================================
"""
WHAT DOES SUCCESS LOOK LIKE?
----------------------------

LEVEL 1: Competitive Performance
    - Match RNN accuracy (~95%) with similar parameter count (~432)
    - This proves bilinear architectures can solve the task
    
LEVEL 2: Basic Interpretability  
    - Extract the polynomial computed by the model
    - Identify which cross-terms (x_i * x_j) have large coefficients
    - Verify the polynomial "makes sense" for 2nd argmax
    
LEVEL 3: Mechanistic Understanding
    - Produce a "mechanistic estimate" of accuracy (ARC's criterion)
    - Explain WHY the model achieves its accuracy based on the polynomial
    - Identify failure modes from the polynomial structure
    
LEVEL 4: Full Understanding
    - Construct a hand-crafted bilinear model that achieves high accuracy
    - This proves we understand the "algorithm" the model implements
    - Match ARC's "surprise accounting" criterion

COMPARISON WITH RNN INTERPRETABILITY
------------------------------------
The RNN creates ~2^(depth * hidden) piecewise-linear regions.
The bilinear model creates a polynomial of degree 2^(num_layers).

For 2 layers:
    - RNN: ~2^32 regions to analyze (intractable)
    - Bilinear: degree-4 polynomial (tractable to write out!)

Even if the bilinear polynomial has many terms, we can:
    1. Group terms by degree
    2. Identify dominant terms
    3. Check mathematical properties (symmetry, etc.)
    
This is fundamentally easier than enumerating piecewise-linear regions.
"""


# ==============================================================================
# MAIN
# ==============================================================================

if __name__ == "__main__":
    # Run the comparison
    results = run_comparison(seq_len=10, target_params=432)