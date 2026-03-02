# %%
"""
Memory sweep:
1. 2-layer and 3-layer with 10 memory slots
2. 4-layer with 1, 2, 5, 10 memory slots
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
NUM_STEPS = 15000
LR = 0.01
WEIGHT_DECAY = 0.01
BATCH_SIZE = 256
EVAL_EVERY = 500
SEEDS = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]

device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Device: {device}")

# Configurations to run
CONFIGS = [
    # (num_layers, n_memory)
    (2, 10),
    (3, 10),
    (4, 1),
    (4, 2),
    (4, 5),
    (4, 10),
]

# =============================================================================
# TRAINING FUNCTION
# =============================================================================
def train_memory_model(num_layers, n_memory, seed, verbose=True):
    """Train model with memory slots."""
    n_model = N_TASK + n_memory
    rank = n_model

    torch.manual_seed(seed)
    model = SymmetricBilinearResidual(n_model, num_layers, rank).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)

    trajectory = {'steps': [], 'eval_acc': [], 'train_loss': []}

    for step in range(NUM_STEPS + 1):
        model.train()

        x_task = torch.randn(BATCH_SIZE, N_TASK, device=device)
        x = torch.cat([x_task, torch.zeros(BATCH_SIZE, n_memory, device=device)], dim=1)

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

    return final_acc, trajectory, state_dict


# =============================================================================
# MAIN
# =============================================================================
if __name__ == "__main__":
    all_results = {}

    for num_layers, n_memory in CONFIGS:
        n_model = N_TASK + n_memory
        print("\n" + "=" * 70)
        print(f"TRAINING {num_layers}-LAYER MODEL WITH {n_memory} MEMORY SLOT(S)")
        print(f"n_model={n_model}, rank={n_model}")
        print("=" * 70)

        results = {}
        best_acc = 0
        best_seed = None

        for seed in SEEDS:
            print(f"\nSeed {seed}:")
            acc, traj, state = train_memory_model(num_layers, n_memory, seed)
            results[seed] = {'acc': acc, 'trajectory': traj, 'state_dict': state}

            if acc > best_acc:
                best_acc = acc
                best_seed = seed

            print(f"  Final accuracy: {acc:.1%}")

        mean_acc = np.mean([r['acc'] for r in results.values()])
        std_acc = np.std([r['acc'] for r in results.values()])

        print("\n" + "-" * 60)
        print(f"SUMMARY ({num_layers}-layer, {n_memory} memory)")
        print("-" * 60)
        print(f"Best: seed {best_seed} with {best_acc:.1%}")
        print(f"Mean: {mean_acc:.1%} ± {std_acc:.1%}")

        # Save best model
        best_path = checkpoint_dir / f"{num_layers}layer_n10_memory{n_memory}_seed{best_seed}.pkl"
        with open(best_path, 'wb') as f:
            pickle.dump({
                'state_dict': results[best_seed]['state_dict'],
                'trajectory': results[best_seed]['trajectory'],
                'accuracy': best_acc,
                'config': {
                    'n_task': N_TASK,
                    'n_model': n_model,
                    'n_memory': n_memory,
                    'num_layers': num_layers,
                    'rank': n_model,
                    'seed': best_seed,
                },
                'all_results': {s: {'acc': r['acc']} for s, r in results.items()},
            }, f)
        print(f"Saved to {best_path}")

        all_results[(num_layers, n_memory)] = {
            'best_acc': best_acc,
            'best_seed': best_seed,
            'mean_acc': mean_acc,
            'std_acc': std_acc,
        }

    # Final comparison
    print("\n" + "=" * 70)
    print("FINAL SUMMARY")
    print("=" * 70)

    # Known baselines
    baselines = {2: 63.7, 3: 83.0, 4: 85.6}

    print("\n2-layer:")
    print(f"  Baseline: ~63.7%")
    print(f"  1 memory:  71.8%")
    print(f"  5 memory:  74.1%")
    if (2, 10) in all_results:
        r = all_results[(2, 10)]
        print(f"  10 memory: {r['best_acc']:.1%} (mean={r['mean_acc']:.1%})")

    print("\n3-layer:")
    print(f"  Baseline: ~83.0%")
    print(f"  1 memory:  85.0%")
    print(f"  2 memory:  87.6%")
    print(f"  5 memory:  90.7%")
    if (3, 10) in all_results:
        r = all_results[(3, 10)]
        print(f"  10 memory: {r['best_acc']:.1%} (mean={r['mean_acc']:.1%})")

    print("\n4-layer:")
    print(f"  Baseline: ~85.6%")
    for n_mem in [1, 2, 5, 10]:
        if (4, n_mem) in all_results:
            r = all_results[(4, n_mem)]
            print(f"  {n_mem} memory:  {r['best_acc']:.1%} (mean={r['mean_acc']:.1%})")

# %%
