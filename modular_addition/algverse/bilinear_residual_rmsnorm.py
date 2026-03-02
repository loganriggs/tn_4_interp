"""
Bilinear Residual Model with RMSNorm for 2nd-argmax task.

Clean implementation:
- RMSNorm applied before each bilinear layer (pre-norm)
- Residual connections: h = h + bilinear(norm(h))
- Residual stream dimension stays at n
- Rank controls bilinear hidden dimension

TN-sim is computed analytically from weights (see tnsim.py).
"""

import torch
import torch.nn as nn
from pathlib import Path
import sys

# Add parent path for tnsim import
sys.path.insert(0, str(Path(__file__).parent.parent / "dev_interp" / "notebooks"))
from tnsim import tn_sim_1layer_with_residual, tn_sim_2layer_with_residual


# =============================================================================
# MODEL DEFINITION
# =============================================================================

class RMSNorm(nn.Module):
    """Root Mean Square Layer Normalization."""
    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        rms = torch.sqrt((x ** 2).mean(dim=-1, keepdim=True) + self.eps)
        return self.weight * (x / rms)


class BilinearLayer(nn.Module):
    """
    Single bilinear layer: D @ [(L @ x) * (R @ x)]

    Args:
        dim: Input/output dimension
        rank: Hidden dimension for bilinear operation
    """
    def __init__(self, dim: int, rank: int):
        super().__init__()
        self.dim = dim
        self.rank = rank

        # L, R: (rank, dim) - project to bilinear hidden
        # D: (dim, rank) - project back
        self.L = nn.Parameter(torch.randn(rank, dim) * 0.1)
        self.R = nn.Parameter(torch.randn(rank, dim) * 0.1)
        self.D = nn.Parameter(torch.randn(dim, rank) * 0.1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (batch, dim)
        Returns:
            (batch, dim)
        """
        Lx = x @ self.L.T  # (batch, rank)
        Rx = x @ self.R.T  # (batch, rank)
        return (Lx * Rx) @ self.D.T  # (batch, dim)


class BilinearResidualRMSNorm(nn.Module):
    """
    Bilinear residual network with RMSNorm (pre-norm style).

    Architecture per layer:
        h_new = h + bilinear(norm(h))

    Where bilinear(x) = D @ [(L @ x) * (R @ x)]

    Args:
        n: Input/output dimension (sequence length for 2nd-argmax)
        num_layers: Number of bilinear layers
        rank: Rank of bilinear decomposition (hidden dimension)
        use_rmsnorm: Whether to use RMSNorm (default True)
    """
    def __init__(self, n: int, num_layers: int = 2, rank: int = 32,
                 use_rmsnorm: bool = True):
        super().__init__()
        self.n = n
        self.num_layers = num_layers
        self.rank = rank
        self.use_rmsnorm = use_rmsnorm

        # RMSNorm layers (one per bilinear layer)
        if use_rmsnorm:
            self.norms = nn.ModuleList([RMSNorm(n) for _ in range(num_layers)])
        else:
            self.norms = None

        # Bilinear layers
        self.layers = nn.ModuleList([
            BilinearLayer(n, rank) for _ in range(num_layers)
        ])

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.

        Args:
            x: Input tensor of shape (batch, n)

        Returns:
            Logits of shape (batch, n)
        """
        h = x

        for i in range(self.num_layers):
            # Apply RMSNorm (pre-norm)
            if self.use_rmsnorm:
                h_norm = self.norms[i](h)
            else:
                h_norm = h

            # Bilinear + residual
            h = h + self.layers[i](h_norm)

        return h

    def get_weights(self):
        """Extract (L, R, D) tuples for TN-sim computation."""
        return [(layer.L, layer.R, layer.D) for layer in self.layers]


# =============================================================================
# TN-SIM (WEIGHT-BASED, ANALYTIC)
# =============================================================================

def tn_sim(model_a: BilinearResidualRMSNorm, model_b: BilinearResidualRMSNorm) -> float:
    """
    Compute TN-sim between two models (weight-based, analytic).

    Uses the formulas from tnsim.py for 1-layer and 2-layer with residuals.
    """
    assert model_a.num_layers == model_b.num_layers, "Models must have same number of layers"

    if model_a.num_layers == 1:
        L_a, R_a, D_a = model_a.get_weights()[0]
        L_b, R_b, D_b = model_b.get_weights()[0]
        return tn_sim_1layer_with_residual(L_a, R_a, D_a, L_b, R_b, D_b)

    elif model_a.num_layers == 2:
        (L1_a, R1_a, D1_a), (L2_a, R2_a, D2_a) = model_a.get_weights()
        (L1_b, R1_b, D1_b), (L2_b, R2_b, D2_b) = model_b.get_weights()
        return tn_sim_2layer_with_residual(
            L1_a, R1_a, D1_a, L2_a, R2_a, D2_a,
            L1_b, R1_b, D1_b, L2_b, R2_b, D2_b
        )
    else:
        raise NotImplementedError(f"TN-sim not implemented for {model_a.num_layers} layers")


# =============================================================================
# TASK AND TRAINING
# =============================================================================

def task_2nd_argmax(x: torch.Tensor) -> torch.Tensor:
    """Find position of 2nd largest element."""
    return x.argsort(-1)[..., -2]


def train_model(
    n: int = 4,
    num_layers: int = 2,
    rank: int = 32,
    use_rmsnorm: bool = True,
    steps: int = 10_000,
    batch_size: int = 128,
    lr: float = 0.01,
    weight_decay: float = 0.001,
    log_every: int = 1000,
    device: str = "cuda" if torch.cuda.is_available() else "cpu",
) -> tuple[BilinearResidualRMSNorm, dict]:
    """
    Train model and return (model, history).
    """
    model = BilinearResidualRMSNorm(n, num_layers, rank, use_rmsnorm).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)

    history = {
        'steps': [],
        'loss': [],
        'train_acc': [],
        'eval_acc': [],
    }

    n_params = sum(p.numel() for p in model.parameters())
    print(f"Model: n={n}, layers={num_layers}, rank={rank}, rmsnorm={use_rmsnorm}")
    print(f"Parameters: {n_params}")
    print(f"Device: {device}")

    for step in range(steps + 1):
        model.train()

        # Generate data
        x = torch.randn(batch_size, n, device=device)
        targets = task_2nd_argmax(x)

        # Forward
        logits = model(x)
        loss = nn.functional.cross_entropy(logits, targets)
        train_acc = (logits.argmax(-1) == targets).float().mean()

        # Backward
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # Log
        if step % log_every == 0:
            model.eval()
            with torch.no_grad():
                x_eval = torch.randn(10000, n, device=device)
                targets_eval = task_2nd_argmax(x_eval)
                logits_eval = model(x_eval)
                eval_acc = (logits_eval.argmax(-1) == targets_eval).float().mean().item()

            history['steps'].append(step)
            history['loss'].append(loss.item())
            history['train_acc'].append(train_acc.item())
            history['eval_acc'].append(eval_acc)

            print(f"Step {step:5d}: loss={loss.item():.4f}, train_acc={train_acc.item():.3f}, eval_acc={eval_acc:.3f}")

    return model, history


def test_model(model: BilinearResidualRMSNorm, n_samples: int = 100000) -> float:
    """Test model accuracy."""
    device = next(model.parameters()).device
    model.eval()

    with torch.no_grad():
        x = torch.randn(n_samples, model.n, device=device)
        targets = task_2nd_argmax(x)
        logits = model(x)
        acc = (logits.argmax(-1) == targets).float().mean().item()

    return acc


# =============================================================================
# RANK SWEEP
# =============================================================================

def sweep_ranks(
    n: int = 4,
    num_layers: int = 2,
    ranks: list[int] = None,
    steps: int = 10_000,
    use_rmsnorm: bool = True,
):
    """Sweep over different ranks and report accuracy."""
    if ranks is None:
        ranks = [4, 8, 16, 32, 64, 128]

    print("=" * 60)
    print(f"Rank Sweep: n={n}, layers={num_layers}, rmsnorm={use_rmsnorm}")
    print("=" * 60)

    results = []

    for rank in ranks:
        print(f"\n--- Rank {rank} ---")
        model, history = train_model(
            n=n,
            num_layers=num_layers,
            rank=rank,
            use_rmsnorm=use_rmsnorm,
            steps=steps,
            log_every=2000,
        )

        final_acc = test_model(model)
        n_params = sum(p.numel() for p in model.parameters())

        results.append({
            'rank': rank,
            'params': n_params,
            'final_acc': final_acc,
            'best_acc': max(history['eval_acc']),
        })

        print(f"Rank {rank}: {n_params} params, final_acc={final_acc:.3f}")

    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    print(f"{'Rank':>6} {'Params':>8} {'Final Acc':>10} {'Best Acc':>10}")
    print("-" * 40)
    for r in results:
        print(f"{r['rank']:>6} {r['params']:>8} {r['final_acc']:>10.3f} {r['best_acc']:>10.3f}")

    return results


# =============================================================================
# MAIN
# =============================================================================

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--n", type=int, default=4, help="Sequence length")
    parser.add_argument("--num_layers", type=int, default=2, help="Number of layers")
    parser.add_argument("--rank", type=int, default=32, help="Bilinear rank")
    parser.add_argument("--steps", type=int, default=10000, help="Training steps")
    parser.add_argument("--no_rmsnorm", action="store_true", help="Disable RMSNorm")
    parser.add_argument("--sweep", action="store_true", help="Run rank sweep")
    args = parser.parse_args()

    if args.sweep:
        sweep_ranks(
            n=args.n,
            num_layers=args.num_layers,
            steps=args.steps,
            use_rmsnorm=not args.no_rmsnorm,
        )
    else:
        print("=" * 60)
        print(f"Training BilinearResidualRMSNorm for 2nd-argmax (n={args.n})")
        print("=" * 60)

        model, history = train_model(
            n=args.n,
            num_layers=args.num_layers,
            rank=args.rank,
            use_rmsnorm=not args.no_rmsnorm,
            steps=args.steps,
        )

        # Final test
        final_acc = test_model(model)
        print(f"\nFinal accuracy: {final_acc:.1%}")
