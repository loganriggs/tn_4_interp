# %%
"""
Train 3-layer models with extra "memory" dimensions.

Variants:
1. 1 memory slot: n_model=11 (positions 0-9 task, position 10 memory)
2. 2 memory slots: n_model=12 (positions 0-9 task, positions 10-11 memory)
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
NUM_LAYERS = 3
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
def train_memory_model(n_memory, seed, verbose=True):
    """Train 3-layer model with memory dimensions."""
    n_model = N_TASK + n_memory
    rank = n_model  # Match rank to model dimension

    torch.manual_seed(seed)
    model = SymmetricBilinearResidual(n_model, NUM_LAYERS, rank).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)

    trajectory = {'steps': [], 'eval_acc': [], 'train_loss': []}

    for step in range(NUM_STEPS + 1):
        model.train()

        # Generate input: (batch, 10) then pad with zeros for memory
        x_task = torch.randn(BATCH_SIZE, N_TASK, device=device)
        x = torch.cat([x_task, torch.zeros(BATCH_SIZE, n_memory, device=device)], dim=1)

        # Target is 2nd-argmax of the 10-dim input
        targets = task_2nd_argmax(x_task)

        # Forward pass
        logits_full = model(x)
        logits = logits_full[:, :N_TASK]  # Only first 10 for loss

        loss = nn.functional.cross_entropy(logits, targets)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if step % EVAL_EVERY == 0:
            model.eval()
            with torch.no_grad():
                x_task_eval = torch.randn(10000, N_TASK, device=device)
                x_eval = torch.cat([x_task_eval, torch.zeros(10000, n_memory, device=device)], dim=1)
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

    return final_acc, trajectory, state_dict, n_model, rank


# =============================================================================
# MAIN
# =============================================================================
if __name__ == "__main__":
    all_results = {}

    for n_memory in [1, 2]:
        print("\n" + "=" * 70)
        print(f"TRAINING 3-LAYER MODEL WITH {n_memory} MEMORY SLOT(S)")
        print(f"n_task={N_TASK}, n_model={N_TASK + n_memory}, rank={N_TASK + n_memory}")
        print("=" * 70)

        results = {}
        best_acc = 0
        best_seed = None

        for seed in SEEDS:
            print(f"\nSeed {seed}:")
            acc, traj, state, n_model, rank = train_memory_model(n_memory, seed)
            results[seed] = {'acc': acc, 'trajectory': traj, 'state_dict': state}

            if acc > best_acc:
                best_acc = acc
                best_seed = seed

            print(f"  Final accuracy: {acc:.1%}")

        print("\n" + "-" * 60)
        print(f"SUMMARY ({n_memory} memory slot(s))")
        print("-" * 60)
        print(f"\nAccuracies by seed:")
        for seed in SEEDS:
            print(f"  Seed {seed}: {results[seed]['acc']:.1%}")

        print(f"\nBest: seed {best_seed} with {best_acc:.1%}")
        print(f"Mean: {np.mean([r['acc'] for r in results.values()]):.1%}")
        print(f"Std:  {np.std([r['acc'] for r in results.values()]):.1%}")

        # Compare to baseline
        print(f"\nBaseline 3-layer n=10 (seed 8): ~83.0%")
        print(f"Memory model best: {best_acc:.1%} ({'better' if best_acc > 0.83 else 'similar/worse'})")

        # Save best model
        best_path = checkpoint_dir / f"3layer_n10_memory{n_memory}_seed{best_seed}.pkl"
        with open(best_path, 'wb') as f:
            pickle.dump({
                'state_dict': results[best_seed]['state_dict'],
                'trajectory': results[best_seed]['trajectory'],
                'accuracy': best_acc,
                'config': {
                    'n_task': N_TASK,
                    'n_model': n_model,
                    'n_memory': n_memory,
                    'num_layers': NUM_LAYERS,
                    'rank': rank,
                    'seed': best_seed,
                },
                'all_results': {s: {'acc': r['acc']} for s, r in results.items()},
            }, f)
        print(f"\nSaved best model to {best_path}")

        # Store for comparison
        all_results[n_memory] = {
            'best_acc': best_acc,
            'best_seed': best_seed,
            'mean_acc': np.mean([r['acc'] for r in results.values()]),
            'std_acc': np.std([r['acc'] for r in results.values()]),
        }

    # Final comparison
    print("\n" + "=" * 70)
    print("FINAL COMPARISON")
    print("=" * 70)
    print(f"\nBaseline 3-layer n=10: ~83.0%")
    for n_mem, res in all_results.items():
        print(f"{n_mem} memory slot(s): best={res['best_acc']:.1%}, mean={res['mean_acc']:.1%} ± {res['std_acc']:.1%}")

# %%
