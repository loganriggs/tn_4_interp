# %%
"""
Core model definitions and training functions for symmetric bilinear networks.

All training/sweep/analysis scripts should import from here.
"""

import torch
import torch.nn as nn
import numpy as np


# =============================================================================
# MODEL DEFINITIONS
# =============================================================================

class RMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(1))  # single scalar

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        rms = torch.sqrt((x ** 2).mean(dim=-1, keepdim=True) + self.eps)
        return self.weight * (x / rms)


class SymmetricBilinearLayer(nn.Module):
    def __init__(self, dim: int, rank: int):
        super().__init__()
        self.L = nn.Parameter(torch.randn(rank, dim) * 0.1)
        self.D = nn.Parameter(torch.randn(dim, rank) * 0.1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        Lx = x @ self.L.T
        return (Lx ** 2) @ self.D.T


class SymmetricBilinearResidual(nn.Module):
    """Residual bilinear network: output = x + sum of bilinear layers."""
    def __init__(self, n: int, num_layers: int, rank: int):
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


class SymmetricBilinearResidualWithW(nn.Module):
    """Residual bilinear network with input projection: output = Wx + bilinear(norm(Wx))."""
    def __init__(self, n: int, num_layers: int, rank: int):
        super().__init__()
        self.n = n
        self.num_layers = num_layers
        self.rank = rank
        self.W = nn.Parameter(torch.eye(n) + torch.randn(n, n) * 0.1)  # init near identity
        self.norms = nn.ModuleList([RMSNorm(n) for _ in range(num_layers)])
        self.layers = nn.ModuleList([
            SymmetricBilinearLayer(n, rank) for _ in range(num_layers)
        ])

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = x @ self.W.T  # input projection
        for i in range(self.num_layers):
            h_norm = self.norms[i](h)
            h = h + self.layers[i](h_norm)
        return h


# =============================================================================
# TASK
# =============================================================================

def task_2nd_argmax(x):
    """Return position of 2nd largest element."""
    return x.argsort(-1)[..., -2]


# =============================================================================
# TRAINING FUNCTIONS
# =============================================================================

def train(
    n: int,
    num_layers: int,
    rank: int,
    seed: int,
    num_steps: int = 10000,
    lr: float = 0.01,
    weight_decay: float = 0.01,
    batch_size: int = 128,
    eval_every: int = 500,
    device: str = None,
    verbose: bool = True,
    use_W: bool = False,
):
    """
    Train a symmetric bilinear model.

    Args:
        use_W: If True, use model with input projection W matrix.

    Returns: (final_acc, trajectory, state_dict)
    """
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    torch.manual_seed(seed)
    if use_W:
        model = SymmetricBilinearResidualWithW(n, num_layers, rank).to(device)
    else:
        model = SymmetricBilinearResidual(n, num_layers, rank).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)

    trajectory = {'steps': [], 'eval_acc': [], 'train_loss': []}

    for step in range(num_steps + 1):
        model.train()
        x = torch.randn(batch_size, n, device=device)
        targets = task_2nd_argmax(x)
        logits = model(x)
        loss = nn.functional.cross_entropy(logits, targets)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if step % eval_every == 0:
            model.eval()
            with torch.no_grad():
                x_eval = torch.randn(10000, n, device=device)
                targets_eval = task_2nd_argmax(x_eval)
                logits_eval = model(x_eval)
                acc = (logits_eval.argmax(-1) == targets_eval).float().mean().item()

            trajectory['steps'].append(step)
            trajectory['eval_acc'].append(acc)
            trajectory['train_loss'].append(loss.item())

            if verbose and step % (eval_every * 4) == 0:
                print(f"  Step {step}: acc={acc:.1%}")

    final_acc = trajectory['eval_acc'][-1]
    state_dict = {k: v.cpu() for k, v in model.state_dict().items()}

    return final_acc, trajectory, state_dict


def train_with_l1_and_prune(
    n: int,
    num_layers: int,
    rank: int,
    seed: int,
    thresh: float = 0.20,
    warmup_steps: int = 5000,
    l1_steps: int = 10000,
    finetune_steps: int = 5000,
    l1_lambda: float = 0.001,
    device: str = None,
    verbose: bool = True,
):
    """
    Train with L1 penalty, prune, and fine-tune.

    Returns: (final_acc, state_dict, sparsity_info)
    """
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    torch.manual_seed(seed)
    model = SymmetricBilinearResidual(n, num_layers, rank).to(device)

    # Phase 1a: Warm-up
    if verbose:
        print("  Phase 1a: Warm-up...")
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.01, weight_decay=0.001)
    for step in range(warmup_steps + 1):
        model.train()
        x = torch.randn(128, n, device=device)
        targets = task_2nd_argmax(x)
        logits = model(x)
        loss = nn.functional.cross_entropy(logits, targets)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    # Phase 1b: L1 penalty
    if verbose:
        print("  Phase 1b: L1 training...")
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.005, weight_decay=0.001)
    for step in range(l1_steps + 1):
        model.train()
        x = torch.randn(128, n, device=device)
        targets = task_2nd_argmax(x)
        logits = model(x)
        ce_loss = nn.functional.cross_entropy(logits, targets)

        l1_loss = 0
        for layer in model.layers:
            l1_loss += layer.L.abs().sum() + layer.D.abs().sum()

        loss = ce_loss + l1_lambda * l1_loss
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    # Phase 2: Prune
    if verbose:
        print(f"  Phase 2: Pruning (t={thresh})...")
    masks = {}
    with torch.no_grad():
        for i, layer in enumerate(model.layers):
            masks[f'L{i}'] = (layer.L.abs() >= thresh).float()
            masks[f'D{i}'] = (layer.D.abs() >= thresh).float()
            layer.L.data *= masks[f'L{i}']
            layer.D.data *= masks[f'D{i}']

    total_params = sum(m.numel() for m in masks.values())
    remaining_params = sum(m.sum().item() for m in masks.values())
    pruned_frac = 1 - remaining_params / total_params

    # Phase 3: Fine-tune
    if verbose:
        print("  Phase 3: Fine-tuning...")
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.005, weight_decay=0.001)
    for step in range(finetune_steps + 1):
        model.train()
        x = torch.randn(128, n, device=device)
        targets = task_2nd_argmax(x)
        logits = model(x)
        loss = nn.functional.cross_entropy(logits, targets)
        optimizer.zero_grad()
        loss.backward()

        with torch.no_grad():
            for i, layer in enumerate(model.layers):
                if layer.L.grad is not None:
                    layer.L.grad *= masks[f'L{i}']
                if layer.D.grad is not None:
                    layer.D.grad *= masks[f'D{i}']

        optimizer.step()

        with torch.no_grad():
            for i, layer in enumerate(model.layers):
                layer.L.data *= masks[f'L{i}']
                layer.D.data *= masks[f'D{i}']

    # Eval
    model.eval()
    with torch.no_grad():
        x_eval = torch.randn(10000, n, device=device)
        targets_eval = task_2nd_argmax(x_eval)
        logits_eval = model(x_eval)
        final_acc = (logits_eval.argmax(-1) == targets_eval).float().mean().item()

    if verbose:
        print(f"  Final: acc={final_acc:.1%}, pruned={pruned_frac:.1%}")

    state_dict = {k: v.cpu() for k, v in model.state_dict().items()}

    # Compute per-layer sparsity
    sparsity = {}
    for i, layer in enumerate(model.layers):
        L = layer.L.detach().cpu()
        D = layer.D.detach().cpu()
        sparsity[f'L{i}'] = (L == 0).sum().item() / L.numel()
        sparsity[f'D{i}'] = (D == 0).sum().item() / D.numel()

    return final_acc, state_dict, {'pruned_frac': pruned_frac, 'sparsity': sparsity}


# =============================================================================
# KNOWN GOOD SEEDS (from sweeps)
# =============================================================================

# Format: (n, num_layers): best_seed
GOOD_SEEDS = {
    # 1-layer (from sweep_seeds_and_sparsity)
    (3, 1): 0,   # n=3 is easy, any seed works
    (4, 1): 4,   # from earlier experiments
    (5, 1): 9,   # from sweep: 59.6%
    (6, 1): 4,   # from sweep: 57.1%

    # 2-layer (from sweep_seeds_and_sparsity)
    (5, 2): 2,   # from sweep: 73.0%
    (6, 2): 4,   # from sweep: 68.9%

    # 3-layer (from sweep_seeds_and_sparsity)
    (5, 3): 2,   # from sweep: 74.9%
    (6, 3): 2,   # from sweep: 80.5%

    # n=10 (from sweep_n10_layers)
    (10, 2): 8,  # 63.7%
    (10, 3): 8,  # 83.0%
    (10, 4): 9,  # 85.6%
}


def get_good_seed(n: int, num_layers: int) -> int:
    """Get a known good seed for this config, or default to 0."""
    return GOOD_SEEDS.get((n, num_layers), 0)
