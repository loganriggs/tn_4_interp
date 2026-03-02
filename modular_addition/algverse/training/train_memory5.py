# %%
"""
Train 2-layer and 3-layer models with 5 memory slots.
n_model = 15 (positions 0-9 task, positions 10-14 memory)
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
N_TASK = 10
N_MEMORY = 5
N_MODEL = N_TASK + N_MEMORY  # 15
RANK = N_MODEL
NUM_STEPS = 15000
LR = 0.01
WEIGHT_DECAY = 0.01
BATCH_SIZE = 256
EVAL_EVERY = 500
SEEDS = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]

device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Device: {device}")
print(f"n_task={N_TASK}, n_memory={N_MEMORY}, n_model={N_MODEL}")

# =============================================================================
# TRAINING FUNCTION
# =============================================================================
def train_memory_model(num_layers, seed, verbose=True):
    """Train model with 5 memory slots."""
    torch.manual_seed(seed)
    model = SymmetricBilinearResidual(N_MODEL, num_layers, RANK).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)

    trajectory = {'steps': [], 'eval_acc': [], 'train_loss': []}

    for step in range(NUM_STEPS + 1):
        model.train()

        # Generate input: (batch, 10) then pad with 5 zeros for memory
        x_task = torch.randn(BATCH_SIZE, N_TASK, device=device)
        x = torch.cat([x_task, torch.zeros(BATCH_SIZE, N_MEMORY, device=device)], dim=1)

        targets = task_2nd_argmax(x_task)
        logits_full = model(x)
        logits = logits_full[:, :N_TASK]

        loss = nn.functional.cross_entropy(logits, targets)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if step % EVAL_EVERY == 0:
            model.eval()
            with torch.no_grad():
                x_task_eval = torch.randn(10000, N_TASK, device=device)
                x_eval = torch.cat([x_task_eval, torch.zeros(10000, N_MEMORY, device=device)], dim=1)
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
    all_results = {}

    for num_layers in [2, 3]:
        print("\n" + "=" * 70)
        print(f"TRAINING {num_layers}-LAYER MODEL WITH 5 MEMORY SLOTS")
        print(f"n_model={N_MODEL}, rank={RANK}")
        print("=" * 70)

        results = {}
        best_acc = 0
        best_seed = None

        for seed in SEEDS:
            print(f"\nSeed {seed}:")
            acc, traj, state = train_memory_model(num_layers, seed)
            results[seed] = {'acc': acc, 'trajectory': traj, 'state_dict': state}

            if acc > best_acc:
                best_acc = acc
                best_seed = seed

            print(f"  Final accuracy: {acc:.1%}")

        print("\n" + "-" * 60)
        print(f"SUMMARY ({num_layers}-layer, 5 memory slots)")
        print("-" * 60)
        print(f"\nAccuracies by seed:")
        for seed in SEEDS:
            print(f"  Seed {seed}: {results[seed]['acc']:.1%}")

        print(f"\nBest: seed {best_seed} with {best_acc:.1%}")
        print(f"Mean: {np.mean([r['acc'] for r in results.values()]):.1%}")
        print(f"Std:  {np.std([r['acc'] for r in results.values()]):.1%}")

        # Baselines
        if num_layers == 2:
            baseline = 63.7
            mem1_best = 71.8
        else:
            baseline = 83.0
            mem1_best = 85.0
            mem2_best = 87.6

        print(f"\nBaseline {num_layers}-layer n=10: ~{baseline}%")
        if num_layers == 2:
            print(f"1 memory slot best: {mem1_best}%")
        else:
            print(f"1 memory slot best: {mem1_best}%")
            print(f"2 memory slots best: {mem2_best}%")
        print(f"5 memory slots best: {best_acc:.1%}")

        # Save best model
        best_path = checkpoint_dir / f"{num_layers}layer_n10_memory5_seed{best_seed}.pkl"
        with open(best_path, 'wb') as f:
            pickle.dump({
                'state_dict': results[best_seed]['state_dict'],
                'trajectory': results[best_seed]['trajectory'],
                'accuracy': best_acc,
                'config': {
                    'n_task': N_TASK,
                    'n_model': N_MODEL,
                    'n_memory': N_MEMORY,
                    'num_layers': num_layers,
                    'rank': RANK,
                    'seed': best_seed,
                },
                'all_results': {s: {'acc': r['acc']} for s, r in results.items()},
            }, f)
        print(f"\nSaved best model to {best_path}")

        all_results[num_layers] = {
            'best_acc': best_acc,
            'best_seed': best_seed,
            'mean_acc': np.mean([r['acc'] for r in results.values()]),
            'std_acc': np.std([r['acc'] for r in results.values()]),
        }

    # Final comparison
    print("\n" + "=" * 70)
    print("FINAL COMPARISON (5 memory slots)")
    print("=" * 70)
    print(f"\n2-layer:")
    print(f"  Baseline: ~63.7%")
    print(f"  1 memory: 71.8%")
    print(f"  5 memory: {all_results[2]['best_acc']:.1%} (mean={all_results[2]['mean_acc']:.1%})")
    print(f"\n3-layer:")
    print(f"  Baseline: ~83.0%")
    print(f"  1 memory: 85.0%")
    print(f"  2 memory: 87.6%")
    print(f"  5 memory: {all_results[3]['best_acc']:.1%} (mean={all_results[3]['mean_acc']:.1%})")

# %%
