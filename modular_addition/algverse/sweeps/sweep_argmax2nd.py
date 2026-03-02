"""
Sweep to find smallest bilinear models that can solve 2nd argmax task.

Focus on small seq_len (n=2, 3) and find minimum parameter counts.
"""

import os
import json
import pickle
from pathlib import Path
from datetime import datetime
import torch
import torch.nn as nn
from typing import Optional
from dataclasses import dataclass, asdict

# Import architectures from argmax2nd.py
from argmax2nd import (
    BilinearMLPMixer,
    MONetStyle,
    BilinearTokenOnly,
    AlgZooRNN,
    task_2nd_argmax,
    count_params,
)


# =============================================================================
# CONFIG
# =============================================================================

@dataclass
class SweepConfig:
    # Task
    seq_len: int = 3
    task_name: str = "2nd_argmax"

    # Training
    steps: int = 100_000
    batch_size: int = 32
    lr: float = 0.01
    weight_decay: float = 0.001

    # Logging
    log_every: int = 100
    save_every: int = 5000
    eval_samples: int = 10000

    # Device
    device: str = "cuda" if torch.cuda.is_available() else "cpu"


# =============================================================================
# TRAINING WITH CHECKPOINTING
# =============================================================================

def train_with_checkpoints(
    model: nn.Module,
    config: SweepConfig,
    save_dir: Path,
    model_name: str,
) -> dict:
    """
    Train model with periodic checkpointing and accuracy logging.

    Data is generated directly on GPU to reduce CPU load.
    """
    device = torch.device(config.device)
    model = model.to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=config.lr,
        weight_decay=config.weight_decay
    )

    # History tracking
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

        # Generate data directly on GPU (reduces CPU→GPU transfer)
        x = torch.randn(config.batch_size, config.seq_len, device=device)
        targets = task_2nd_argmax(x)

        logits = model(x)
        loss = nn.functional.cross_entropy(logits, targets)
        train_acc = (logits.argmax(-1) == targets).float().mean()

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # Log
        if step % config.log_every == 0:
            # Eval accuracy on fresh samples
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

        # Save checkpoint
        if step > 0 and step % config.save_every == 0:
            ckpt_path = save_dir / f"{model_name}_step{step}.pt"
            torch.save({
                'step': step,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'history': history,
            }, ckpt_path)

    # Final save
    history['best_acc'] = best_acc
    history['final_acc'] = history['eval_acc'][-1] if history['eval_acc'] else 0.0

    final_path = save_dir / f"{model_name}_final.pt"
    torch.save({
        'model_state_dict': model.state_dict(),
        'history': history,
    }, final_path)

    # Save history as JSON for easy analysis
    history_path = save_dir / f"{model_name}_history.json"
    with open(history_path, 'w') as f:
        json.dump(history, f, indent=2)

    return history


# =============================================================================
# MODEL CONFIGURATIONS TO SWEEP
# =============================================================================

def get_model_configs(seq_len: int) -> list[dict]:
    """
    Generate model configurations to sweep.

    For small seq_len, we want to find the SMALLEST models that work.
    """
    configs = []

    # Bilinear MLP-Mixer variations
    for hidden_dim in [2, 4, 6, 8]:
        for token_rank in [2, 4, 6]:
            for channel_rank in [2, 4, 6]:
                for num_layers in [1, 2, 3]:
                    configs.append({
                        'arch': 'BilinearMLPMixer',
                        'kwargs': {
                            'seq_len': seq_len,
                            'hidden_dim': hidden_dim,
                            'token_rank': token_rank,
                            'channel_rank': channel_rank,
                            'num_layers': num_layers,
                        }
                    })

    # MONet-style variations
    for hidden_dim in [4, 8, 12, 16]:
        for mu_rank in [2, 4, 6, 8]:
            for num_layers in [1, 2, 3]:
                configs.append({
                    'arch': 'MONetStyle',
                    'kwargs': {
                        'seq_len': seq_len,
                        'hidden_dim': hidden_dim,
                        'mu_rank': mu_rank,
                        'num_layers': num_layers,
                    }
                })

    # Bilinear Token-Only variations
    for hidden_dim in [2, 4, 6, 8]:
        for rank in [2, 4, 6, 8, 10]:
            for num_layers in [1, 2, 3]:
                configs.append({
                    'arch': 'BilinearTokenOnly',
                    'kwargs': {
                        'seq_len': seq_len,
                        'hidden_dim': hidden_dim,
                        'rank': rank,
                        'num_layers': num_layers,
                    }
                })

    # RNN baseline variations
    for hidden_size in [4, 8, 12, 16, 24]:
        configs.append({
            'arch': 'AlgZooRNN',
            'kwargs': {
                'seq_len': seq_len,
                'hidden_size': hidden_size,
            }
        })

    return configs


def create_model(config: dict) -> nn.Module:
    """Create model from config dict."""
    arch = config['arch']
    kwargs = config['kwargs']

    if arch == 'BilinearMLPMixer':
        return BilinearMLPMixer(**kwargs)
    elif arch == 'MONetStyle':
        return MONetStyle(**kwargs)
    elif arch == 'BilinearTokenOnly':
        return BilinearTokenOnly(**kwargs)
    elif arch == 'AlgZooRNN':
        return AlgZooRNN(**kwargs)
    else:
        raise ValueError(f"Unknown architecture: {arch}")


def config_to_name(config: dict) -> str:
    """Generate a descriptive name from config."""
    arch = config['arch']
    kwargs = config['kwargs']

    if arch == 'BilinearMLPMixer':
        return f"BilMixer_h{kwargs['hidden_dim']}_tr{kwargs['token_rank']}_cr{kwargs['channel_rank']}_L{kwargs['num_layers']}"
    elif arch == 'MONetStyle':
        return f"MONet_h{kwargs['hidden_dim']}_r{kwargs['mu_rank']}_L{kwargs['num_layers']}"
    elif arch == 'BilinearTokenOnly':
        return f"BilToken_h{kwargs['hidden_dim']}_r{kwargs['rank']}_L{kwargs['num_layers']}"
    elif arch == 'AlgZooRNN':
        return f"RNN_h{kwargs['hidden_size']}"
    else:
        return f"{arch}_{hash(str(kwargs)) % 10000}"


# =============================================================================
# QUICK SWEEP (fewer configs, quick iteration)
# =============================================================================

def get_quick_configs(seq_len: int) -> list[dict]:
    """Smaller set of configs for quick testing."""
    configs = []

    # A few Bilinear MLP-Mixer
    for hidden_dim in [4, 8]:
        for token_rank in [4]:
            for channel_rank in [4]:
                for num_layers in [1, 2]:
                    configs.append({
                        'arch': 'BilinearMLPMixer',
                        'kwargs': {
                            'seq_len': seq_len,
                            'hidden_dim': hidden_dim,
                            'token_rank': token_rank,
                            'channel_rank': channel_rank,
                            'num_layers': num_layers,
                        }
                    })

    # A few MONet
    for hidden_dim in [8]:
        for mu_rank in [4, 6]:
            for num_layers in [1, 2]:
                configs.append({
                    'arch': 'MONetStyle',
                    'kwargs': {
                        'seq_len': seq_len,
                        'hidden_dim': hidden_dim,
                        'mu_rank': mu_rank,
                        'num_layers': num_layers,
                    }
                })

    # A few Token-Only
    for hidden_dim in [4, 6]:
        for rank in [4, 6]:
            for num_layers in [1, 2]:
                configs.append({
                    'arch': 'BilinearTokenOnly',
                    'kwargs': {
                        'seq_len': seq_len,
                        'hidden_dim': hidden_dim,
                        'rank': rank,
                        'num_layers': num_layers,
                    }
                })

    # RNN baselines
    for hidden_size in [8, 16]:
        configs.append({
            'arch': 'AlgZooRNN',
            'kwargs': {
                'seq_len': seq_len,
                'hidden_size': hidden_size,
            }
        })

    return configs


# =============================================================================
# MAIN SWEEP
# =============================================================================

def run_sweep(
    seq_len: int = 3,
    quick: bool = True,
    max_params: Optional[int] = None,
    target_acc: float = 0.95,
    steps: int = 50_000,
    batch_size: int = 32,
):
    """
    Run sweep over model configurations.

    Args:
        seq_len: Sequence length (task difficulty)
        quick: Use smaller config set for quick iteration
        max_params: Skip models with more params than this
        target_acc: Target accuracy to consider "solved"
        steps: Training steps per model
        batch_size: Batch size
    """
    # Setup
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    save_dir = Path(f"sweep_results/seqlen{seq_len}_{timestamp}")
    save_dir.mkdir(parents=True, exist_ok=True)

    config = SweepConfig(
        seq_len=seq_len,
        steps=steps,
        batch_size=batch_size,
    )

    # Get model configs
    if quick:
        model_configs = get_quick_configs(seq_len)
    else:
        model_configs = get_model_configs(seq_len)

    # Filter by max params if specified
    if max_params:
        filtered = []
        for cfg in model_configs:
            model = create_model(cfg)
            if count_params(model) <= max_params:
                filtered.append(cfg)
            del model
        model_configs = filtered

    print(f"="*70)
    print(f"SWEEP: seq_len={seq_len}, {len(model_configs)} configs")
    print(f"Target accuracy: {target_acc}")
    print(f"Save directory: {save_dir}")
    print(f"Device: {config.device}")
    print(f"="*70)

    # Results tracking
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

    # Sort by params
    results_sorted = sorted([r for r in results if 'final_acc' in r],
                           key=lambda x: x['num_params'])

    print(f"\nTop 10 by params (all):")
    for r in results_sorted[:10]:
        print(f"  {r['num_params']:4d} params | acc={r['best_acc']:.4f} | {r['model_name']}")

    if successful:
        successful_sorted = sorted(successful, key=lambda x: x['num_params'])
        print(f"\nSmallest successful models (acc >= {target_acc}):")
        for r in successful_sorted[:5]:
            print(f"  {r['num_params']:4d} params | acc={r['best_acc']:.4f} | {r['model_name']}")
    else:
        print(f"\nNo models achieved target accuracy {target_acc}")
        print("Best results:")
        best_by_acc = sorted([r for r in results if 'final_acc' in r],
                            key=lambda x: -x['best_acc'])[:5]
        for r in best_by_acc:
            print(f"  {r['num_params']:4d} params | acc={r['best_acc']:.4f} | {r['model_name']}")

    return summary


# =============================================================================
# SINGLE MODEL TRAINING (for debugging)
# =============================================================================

def train_single(
    arch: str = "BilinearMLPMixer",
    seq_len: int = 3,
    steps: int = 50_000,
    **model_kwargs
):
    """Train a single model for debugging."""
    save_dir = Path("sweep_results/debug")
    save_dir.mkdir(parents=True, exist_ok=True)

    config = SweepConfig(seq_len=seq_len, steps=steps)

    model_cfg = {'arch': arch, 'kwargs': {'seq_len': seq_len, **model_kwargs}}
    model = create_model(model_cfg)
    model_name = config_to_name(model_cfg)

    print(f"Training {model_name}: {count_params(model)} params")
    print(f"Device: {config.device}")

    history = train_with_checkpoints(model, config, save_dir, model_name)

    return model, history


# =============================================================================
# MAIN
# =============================================================================

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--seq_len", type=int, default=3, help="Sequence length")
    parser.add_argument("--quick", action="store_true", help="Quick sweep with fewer configs")
    parser.add_argument("--full", action="store_true", help="Full sweep with all configs")
    parser.add_argument("--steps", type=int, default=50_000, help="Training steps")
    parser.add_argument("--batch_size", type=int, default=32, help="Batch size")
    parser.add_argument("--target_acc", type=float, default=0.95, help="Target accuracy")
    parser.add_argument("--max_params", type=int, default=None, help="Max params filter")
    parser.add_argument("--single", action="store_true", help="Train single model for debugging")
    args = parser.parse_args()

    if args.single:
        # Debug single model
        model, history = train_single(
            arch="BilinearMLPMixer",
            seq_len=args.seq_len,
            steps=args.steps,
            hidden_dim=8,
            token_rank=4,
            channel_rank=4,
            num_layers=2,
        )
    else:
        # Run sweep
        run_sweep(
            seq_len=args.seq_len,
            quick=not args.full,
            max_params=args.max_params,
            target_acc=args.target_acc,
            steps=args.steps,
            batch_size=args.batch_size,
        )
