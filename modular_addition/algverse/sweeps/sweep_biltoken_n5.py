"""
Sweep BilinearTokenOnly on 2nd argmax with seq_len=5.

Vary: num_layers (1, 2), rank, residual (True/False)
"""

import os
import json
from pathlib import Path
from datetime import datetime
import torch
import torch.nn as nn
from dataclasses import dataclass, asdict

from argmax2nd import task_2nd_argmax, count_params


# =============================================================================
# BilinearTokenOnly with configurable residual
# =============================================================================

class BilinearTokenOnlyConfigurable(nn.Module):
    """
    BilinearTokenOnly with configurable residual connection.

    residual=True:  x = x + D(Lx ⊙ Rx)
    residual=False: x = D(Lx ⊙ Rx)
    """
    def __init__(self, seq_len: int, hidden_dim: int = 6,
                 rank: int = 7, num_layers: int = 2, residual: bool = True):
        super().__init__()
        self.residual = residual
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
            if self.residual:
                x = x + y.transpose(1, 2)
            else:
                x = y.transpose(1, 2)

        return self.head(x).squeeze(-1)


# =============================================================================
# CONFIG
# =============================================================================

@dataclass
class SweepConfig:
    seq_len: int = 5
    task_name: str = "2nd_argmax"
    steps: int = 50_000
    batch_size: int = 32
    lr: float = 0.01
    weight_decay: float = 0.001
    log_every: int = 100
    save_every: int = 5000
    eval_samples: int = 10000
    device: str = "cuda" if torch.cuda.is_available() else "cpu"


# =============================================================================
# TRAINING
# =============================================================================

def train_with_checkpoints(
    model: nn.Module,
    config: SweepConfig,
    save_dir: Path,
    model_name: str,
) -> dict:
    """Train model with periodic checkpointing."""
    device = torch.device(config.device)
    model = model.to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=config.lr,
        weight_decay=config.weight_decay
    )

    history = {
        'steps': [],
        'loss': [],
        'train_acc': [],
        'eval_acc': [],
        'config': asdict(config),
        'model_name': model_name,
        'num_params': count_params(model),
    }

    best_acc = 0.0

    for step in range(config.steps + 1):
        model.train()

        # Generate data on GPU
        x = torch.randn(config.batch_size, config.seq_len, device=device)
        targets = task_2nd_argmax(x)

        logits = model(x)
        loss = nn.functional.cross_entropy(logits, targets)
        train_acc = (logits.argmax(-1) == targets).float().mean()

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if step % config.log_every == 0:
            model.eval()
            with torch.no_grad():
                x_eval = torch.randn(config.eval_samples, config.seq_len, device=device)
                targets_eval = task_2nd_argmax(x_eval)
                logits_eval = model(x_eval)
                eval_acc = (logits_eval.argmax(-1) == targets_eval).float().mean().item()

            history['steps'].append(step)
            history['loss'].append(loss.item())
            history['train_acc'].append(train_acc.item())
            history['eval_acc'].append(eval_acc)

            if eval_acc > best_acc:
                best_acc = eval_acc

            if step % (config.log_every * 10) == 0:
                print(f"  Step {step:6d}: loss={loss.item():.4f}, "
                      f"train_acc={train_acc.item():.4f}, eval_acc={eval_acc:.4f}")

        if step > 0 and step % config.save_every == 0:
            ckpt_path = save_dir / f"{model_name}_step{step}.pt"
            torch.save({
                'step': step,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'history': history,
            }, ckpt_path)

    history['best_acc'] = best_acc
    history['final_acc'] = history['eval_acc'][-1] if history['eval_acc'] else 0.0

    final_path = save_dir / f"{model_name}_final.pt"
    torch.save({
        'model_state_dict': model.state_dict(),
        'history': history,
    }, final_path)

    history_path = save_dir / f"{model_name}_history.json"
    with open(history_path, 'w') as f:
        json.dump(history, f, indent=2)

    return history


# =============================================================================
# MODEL CONFIGS
# =============================================================================

def get_biltoken_configs(seq_len: int) -> list[dict]:
    """
    Generate BilinearTokenOnly configs varying:
    - num_layers: 1, 2
    - rank: 2, 4, 6, 8, 10, 12
    - hidden_dim: 2, 4, 6, 8
    - residual: True, False
    """
    configs = []

    for num_layers in [1, 2]:
        for rank in [2, 4, 6, 8, 10, 12]:
            for hidden_dim in [2, 4, 6, 8]:
                for residual in [True, False]:
                    configs.append({
                        'arch': 'BilinearTokenOnly',
                        'kwargs': {
                            'seq_len': seq_len,
                            'hidden_dim': hidden_dim,
                            'rank': rank,
                            'num_layers': num_layers,
                            'residual': residual,
                        }
                    })

    return configs


def create_model(config: dict) -> nn.Module:
    """Create model from config dict."""
    return BilinearTokenOnlyConfigurable(**config['kwargs'])


def config_to_name(config: dict) -> str:
    """Generate descriptive name from config."""
    kwargs = config['kwargs']
    res_str = "res" if kwargs['residual'] else "nores"
    return f"BilToken_h{kwargs['hidden_dim']}_r{kwargs['rank']}_L{kwargs['num_layers']}_{res_str}"


# =============================================================================
# MAIN SWEEP
# =============================================================================

def run_sweep(
    seq_len: int = 5,
    target_acc: float = 0.95,
    steps: int = 50_000,
    batch_size: int = 32,
):
    """Run sweep over BilinearTokenOnly configurations."""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    save_dir = Path(f"sweep_results/biltoken_n{seq_len}_{timestamp}")
    save_dir.mkdir(parents=True, exist_ok=True)

    config = SweepConfig(
        seq_len=seq_len,
        steps=steps,
        batch_size=batch_size,
    )

    model_configs = get_biltoken_configs(seq_len)

    print(f"="*70)
    print(f"BILTOKEN SWEEP: seq_len={seq_len}, {len(model_configs)} configs")
    print(f"Target accuracy: {target_acc}")
    print(f"Save directory: {save_dir}")
    print(f"Device: {config.device}")
    print(f"="*70)

    results = []
    successful = []

    for i, model_cfg in enumerate(model_configs):
        model = create_model(model_cfg)
        n_params = count_params(model)
        model_name = config_to_name(model_cfg)

        print(f"\n[{i+1}/{len(model_configs)}] {model_name}: {n_params} params")

        try:
            history = train_with_checkpoints(model, config, save_dir, model_name)

            result = {
                'model_name': model_name,
                'config': model_cfg,
                'num_params': n_params,
                'final_acc': history['final_acc'],
                'best_acc': history['best_acc'],
            }
            results.append(result)

            if history['best_acc'] >= target_acc:
                successful.append(result)
                print(f"  *** SUCCESS: {history['best_acc']:.4f} >= {target_acc} ***")

        except Exception as e:
            print(f"  ERROR: {e}")
            results.append({
                'model_name': model_name,
                'config': model_cfg,
                'num_params': n_params,
                'error': str(e),
            })

        del model
        torch.cuda.empty_cache() if torch.cuda.is_available() else None

    # Save summary
    summary = {
        'seq_len': seq_len,
        'target_acc': target_acc,
        'total_configs': len(model_configs),
        'successful': len(successful),
        'results': results,
    }

    summary_path = save_dir / "summary.json"
    with open(summary_path, 'w') as f:
        json.dump(summary, f, indent=2)

    # Print summary
    print("\n" + "="*70)
    print("SUMMARY")
    print("="*70)

    results_sorted = sorted([r for r in results if 'final_acc' in r],
                           key=lambda x: x['num_params'])

    print(f"\nTop 10 by params (all):")
    for r in results_sorted[:10]:
        print(f"  {r['num_params']:4d} params | acc={r['best_acc']:.4f} | {r['model_name']}")

    if successful:
        successful_sorted = sorted(successful, key=lambda x: x['num_params'])
        print(f"\nSmallest successful models (acc >= {target_acc}):")
        for r in successful_sorted[:10]:
            print(f"  {r['num_params']:4d} params | acc={r['best_acc']:.4f} | {r['model_name']}")
    else:
        print(f"\nNo models achieved target accuracy {target_acc}")
        print("Best results:")
        best_by_acc = sorted([r for r in results if 'final_acc' in r],
                            key=lambda x: -x['best_acc'])[:10]
        for r in best_by_acc:
            print(f"  {r['num_params']:4d} params | acc={r['best_acc']:.4f} | {r['model_name']}")

    # Compare residual vs no-residual
    print("\n" + "="*70)
    print("RESIDUAL vs NO-RESIDUAL COMPARISON")
    print("="*70)

    res_results = [r for r in results if 'final_acc' in r and '_res' in r['model_name'] and '_nores' not in r['model_name']]
    nores_results = [r for r in results if 'final_acc' in r and '_nores' in r['model_name']]

    if res_results:
        avg_res = sum(r['best_acc'] for r in res_results) / len(res_results)
        best_res = max(r['best_acc'] for r in res_results)
        print(f"With residual:    avg={avg_res:.4f}, best={best_res:.4f}, count={len(res_results)}")

    if nores_results:
        avg_nores = sum(r['best_acc'] for r in nores_results) / len(nores_results)
        best_nores = max(r['best_acc'] for r in nores_results)
        print(f"Without residual: avg={avg_nores:.4f}, best={best_nores:.4f}, count={len(nores_results)}")

    return summary


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--seq_len", type=int, default=5, help="Sequence length")
    parser.add_argument("--steps", type=int, default=50_000, help="Training steps")
    parser.add_argument("--batch_size", type=int, default=32, help="Batch size")
    parser.add_argument("--target_acc", type=float, default=0.95, help="Target accuracy")
    args = parser.parse_args()

    run_sweep(
        seq_len=args.seq_len,
        target_acc=args.target_acc,
        steps=args.steps,
        batch_size=args.batch_size,
    )
