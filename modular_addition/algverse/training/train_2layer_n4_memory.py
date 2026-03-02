# %%
"""
Train 2-layer model with 1 extra memory dimension for n=4 task.

- n_task = 4 (2nd-argmax of 4 elements)
- n_model = 5 (4 + 1 memory slot, always 0 on input)
- Output: use only first 4 logits for loss/accuracy
- rank = 5 (match n_model)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
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
N_TASK = 4         # Task dimension (2nd-argmax of 4 elements)
N_MODEL = 5        # Model dimension (4 + 1 memory slot)
NUM_LAYERS = 2
RANK = 5           # Match n_model
NUM_STEPS = 10000
LR = 0.01
WEIGHT_DECAY = 0.01
BATCH_SIZE = 256
EVAL_EVERY = 500
SEEDS = list(range(10))

device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Device: {device}")

# =============================================================================
# TRAINING FUNCTION
# =============================================================================
def train_memory_model(seed, verbose=True):
    """Train 2-layer model with 1 memory dimension for n=4 task."""
    torch.manual_seed(seed)

    model = SymmetricBilinearResidual(N_MODEL, NUM_LAYERS, RANK).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)

    trajectory = {"steps": [], "eval_acc": [], "train_loss": []}

    for step in range(NUM_STEPS + 1):
        model.train()

        # Generate input: (batch, 4) then pad to (batch, 5) with 0
        x_task = torch.randn(BATCH_SIZE, N_TASK, device=device)
        targets = task_2nd_argmax(x_task)
        x = torch.cat([x_task, torch.zeros(BATCH_SIZE, 1, device=device)], dim=1)

        # Forward pass - only use first 4 logits
        output = model(x)
        loss = F.cross_entropy(output[:, :N_TASK], targets)

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

            trajectory["steps"].append(step)
            trajectory["eval_acc"].append(acc)
            trajectory["train_loss"].append(loss.item())

            if verbose and step % 2000 == 0:
                print(f"  Step {step}: acc={acc:.1%}, loss={loss.item():.4f}")

    final_acc = trajectory["eval_acc"][-1]
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
        print(f"
Seed {seed}:")
        acc, traj, state = train_memory_model(seed)
        results[seed] = {"acc": acc, "trajectory": traj, "state_dict": state}

        if acc > best_acc:
            best_acc = acc
            best_seed = seed

        print(f"  Final accuracy: {acc:.1%}")

    print("
" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    print(f"
Accuracies by seed:")
    for seed in SEEDS:
        marker = " <-- best" if seed == best_seed else ""
        print(f"  Seed {seed}: {results[seed]["acc"]:.1%}{marker}")

    print(f"
Best: seed {best_seed} with {best_acc:.1%}")
    print(f"Mean: {np.mean([r["acc"] for r in results.values()]):.1%}")
    print(f"Std:  {np.std([r["acc"] for r in results.values()]):.1%}")

    # Save best model
    best_path = checkpoint_dir / "2layer_n4_memory1_best.pkl"
    with open(best_path, "wb") as f:
        pickle.dump({
            "state_dict": results[best_seed]["state_dict"],
            "trajectory": results[best_seed]["trajectory"],
            "accuracy": best_acc,
            "config": {
                "n_task": N_TASK,
                "n_model": N_MODEL,
                "num_layers": NUM_LAYERS,
                "rank": RANK,
                "seed": best_seed,
                "num_steps": NUM_STEPS,
                "lr": LR,
                "weight_decay": WEIGHT_DECAY,
                "batch_size": BATCH_SIZE,
            },
            "all_results": {s: {"acc": r["acc"]} for s, r in results.items()},
        }, f)
    print(f"Saved best model to {best_path}")

# %%
