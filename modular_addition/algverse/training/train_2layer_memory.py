# %%
"""
Train 2-layer model with extra "memory" dimension.

The idea: Add an 11th dimension that's always 0 on input but can be used
by the bilinear layers as scratch space for intermediate computation.

- Input: x ∈ R^11, where x[0:10] is normal input, x[10] = 0
- Output: logits ∈ R^11, we only use logits[0:10] for predictions
- Task: still 2nd-argmax of the first 10 dimensions
"""

import torch
import torch.nn as nn
import numpy as np
from pathlib import Path
import sys
import pickle

PROJECT_ROOT = Path("/workspace/tn_4_interp/modular_addition/algverse")
sys.path.insert(0, str(PROJECT_ROOT))
from models import SymmetricBilinearResidual, task_2nd_argmax

checkpoint_dir = PROJECT_ROOT / "checkpoints"
checkpoint_dir.mkdir(exist_ok=True)

# =============================================================================
# CONFIG
# =============================================================================
N_TASK = 10        # Task dimension (2nd-argmax of 10 elements)
N_MODEL = 11       # Model dimension (10 + 1 memory slot)
NUM_LAYERS = 2
RANK = 11          # Also increase rank to match
NUM_STEPS = 15000
LR = 0.01
WEIGHT_DECAY = 0.01
BATCH_SIZE = 256
EVAL_EVERY = 500
SEEDS = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]

device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Device: {device}")

# =============================================================================
# TRAINING FUNCTION
# =============================================================================
def train_memory_model(seed, verbose=True):
    """Train 2-layer model with memory dimension."""
    torch.manual_seed(seed)

    # Model operates on n=11
    model = SymmetricBilinearResidual(N_MODEL, NUM_LAYERS, RANK).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)

    trajectory = {'steps': [], 'eval_acc': [], 'train_loss': []}

    for step in range(NUM_STEPS + 1):
        model.train()

        # Generate input: (batch, 10) then pad to (batch, 11) with 0
        x_task = torch.randn(BATCH_SIZE, N_TASK, device=device)
        x = torch.cat([x_task, torch.zeros(BATCH_SIZE, 1, device=device)], dim=1)

        # Target is 2nd-argmax of the 10-dim input (positions 0-9)
        targets = task_2nd_argmax(x_task)

        # Forward pass
        logits_full = model(x)  # (batch, 11)
        logits = logits_full[:, :N_TASK]  # Only first 10 for loss

        loss = nn.functional.cross_entropy(logits, targets)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if step % EVAL_EVERY == 0:
            model.eval()
            with torch.no_grad():
                x_task_eval = torch.randn(10000, N_TASK, device=device)
                x_eval = torch.cat([x_task_eval, torch.zeros(10000, 1, device=device)], dim=1)
                targets_eval = task_2nd_argmax(x_task_eval)
                logits_eval = model(x_eval)[:, :N_TASK]
                acc = (logits_eval.argmax(-1) == targets_eval).float().mean().item()

            trajectory['steps'].append(step)
            trajectory['eval_acc'].append(acc)
            trajectory['train_loss'].append(loss.item())

            if verbose and step % (EVAL_EVERY * 4) == 0:
                print(f"  Step {step}: acc={acc:.1%}, loss={loss.item():.4f}")

    final_acc = trajectory['eval_acc'][-1]
    state_dict = {k: v.cpu() for k, v in model.state_dict().items()}

    return final_acc, trajectory, state_dict


# =============================================================================
# MAIN
# =============================================================================
if __name__ == "__main__":
    print(f"Training 2-layer memory model (n_task={N_TASK}, n_model={N_MODEL}, rank={RANK})")
    print(f"Seeds: {SEEDS}")
    print("=" * 60)

    results = {}
    best_acc = 0
    best_seed = None

    for seed in SEEDS:
        print(f"\nSeed {seed}:")
        acc, traj, state = train_memory_model(seed)
        results[seed] = {'acc': acc, 'trajectory': traj, 'state_dict': state}

        if acc > best_acc:
            best_acc = acc
            best_seed = seed

        print(f"  Final accuracy: {acc:.1%}")

    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    print(f"\nAccuracies by seed:")
    for seed in SEEDS:
        print(f"  Seed {seed}: {results[seed]['acc']:.1%}")

    print(f"\nBest: seed {best_seed} with {best_acc:.1%}")
    print(f"Mean: {np.mean([r['acc'] for r in results.values()]):.1%}")
    print(f"Std:  {np.std([r['acc'] for r in results.values()]):.1%}")

    # Compare to baseline (no memory)
    print(f"\nBaseline 2-layer n=10 (seed 8): ~63.7%")
    print(f"Memory model best: {best_acc:.1%} ({'better' if best_acc > 0.637 else 'worse'})")

    # Save best model
    best_path = checkpoint_dir / f"2layer_n10_memory_seed{best_seed}.pkl"
    with open(best_path, 'wb') as f:
        pickle.dump({
            'state_dict': results[best_seed]['state_dict'],
            'trajectory': results[best_seed]['trajectory'],
            'accuracy': best_acc,
            'config': {
                'n_task': N_TASK,
                'n_model': N_MODEL,
                'num_layers': NUM_LAYERS,
                'rank': RANK,
                'seed': best_seed,
            },
            'all_results': {s: {'acc': r['acc']} for s, r in results.items()},
        }, f)
    print(f"\nSaved best model to {best_path}")

    # Also save full results
    full_path = checkpoint_dir / "2layer_n10_memory_all_seeds.pkl"
    with open(full_path, 'wb') as f:
        pickle.dump({
            'results': {s: {'acc': r['acc'], 'trajectory': r['trajectory']} for s, r in results.items()},
            'config': {
                'n_task': N_TASK,
                'n_model': N_MODEL,
                'num_layers': NUM_LAYERS,
                'rank': RANK,
            },
        }, f)
    print(f"Saved all results to {full_path}")

# %%
