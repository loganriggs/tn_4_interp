# %%
"""
Iterative pruning of the 3-layer model with 5 memory slots.
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
device = "cuda" if torch.cuda.is_available() else "cpu"

# =============================================================================
# CONFIG
# =============================================================================
N_TASK = 10
N_MEMORY = 5
N_MODEL = N_TASK + N_MEMORY  # 15
NUM_LAYERS = 3
RANK = N_MODEL
SEED = 2  # Best seed from training (90.7%)

PRUNE_PERCENT = 0.10  # Prune 10% of remaining weights each iteration
MAX_ITERATIONS = 15
MAX_ACC_DROP = 0.03   # Stop if accuracy drops more than 3%
FINETUNE_STEPS = 3000

print(f"Device: {device}")
print(f"Iterative pruning of 3-layer model with 5 memory slots (seed {SEED})")

# =============================================================================
# HELPER FUNCTIONS
# =============================================================================
def eval_model(model):
    """Evaluate model accuracy."""
    model.eval()
    with torch.no_grad():
        x_task = torch.randn(10000, N_TASK, device=device)
        x = torch.cat([x_task, torch.zeros(10000, N_MEMORY, device=device)], dim=1)
        targets = task_2nd_argmax(x_task)
        logits = model(x)[:, :N_TASK]
        acc = (logits.argmax(-1) == targets).float().mean().item()
    return acc


def get_sparsity(model):
    """Compute overall sparsity."""
    total = 0
    zeros = 0
    for layer in model.layers:
        total += layer.L.numel() + layer.D.numel()
        zeros += (layer.L == 0).sum().item() + (layer.D == 0).sum().item()
    return zeros / total


def finetune_with_masks(model, masks, steps=3000):
    """Fine-tune while keeping pruned weights at zero."""
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.005, weight_decay=0.001)

    for step in range(steps):
        model.train()
        x_task = torch.randn(128, N_TASK, device=device)
        x = torch.cat([x_task, torch.zeros(128, N_MEMORY, device=device)], dim=1)
        targets = task_2nd_argmax(x_task)
        logits = model(x)[:, :N_TASK]
        loss = nn.functional.cross_entropy(logits, targets)

        optimizer.zero_grad()
        loss.backward()

        # Mask gradients
        with torch.no_grad():
            for i, layer in enumerate(model.layers):
                if layer.L.grad is not None:
                    layer.L.grad *= masks[f'L{i}']
                if layer.D.grad is not None:
                    layer.D.grad *= masks[f'D{i}']

        optimizer.step()

        # Re-apply mask
        with torch.no_grad():
            for i, layer in enumerate(model.layers):
                layer.L.data *= masks[f'L{i}']
                layer.D.data *= masks[f'D{i}']

    return model


# =============================================================================
# LOAD MODEL
# =============================================================================
ckpt_path = checkpoint_dir / f"3layer_n10_memory5_seed{SEED}.pkl"
print(f"Loading model from {ckpt_path}")

with open(ckpt_path, 'rb') as f:
    ckpt = pickle.load(f)

model = SymmetricBilinearResidual(N_MODEL, NUM_LAYERS, RANK).to(device)
model.load_state_dict({k: v.to(device) for k, v in ckpt['state_dict'].items()})

baseline_acc = eval_model(model)
print(f"Baseline accuracy: {baseline_acc:.1%}")

# Initialize masks (all ones)
masks = {}
for i, layer in enumerate(model.layers):
    masks[f'L{i}'] = torch.ones_like(layer.L)
    masks[f'D{i}'] = torch.ones_like(layer.D)

# =============================================================================
# ITERATIVE PRUNING
# =============================================================================
print("\n" + "=" * 60)
print("ITERATIVE PRUNING")
print("=" * 60)

results = []
best_within_threshold = None

for iteration in range(MAX_ITERATIONS):
    print(f"\nIteration {iteration + 1}")
    print("-" * 40)

    # Collect all weights (magnitudes)
    all_weights = []
    for i, layer in enumerate(model.layers):
        L_masked = layer.L.detach() * masks[f'L{i}']
        D_masked = layer.D.detach() * masks[f'D{i}']
        all_weights.append(L_masked.abs().flatten())
        all_weights.append(D_masked.abs().flatten())

    all_weights = torch.cat(all_weights)
    nonzero_weights = all_weights[all_weights > 0]

    if len(nonzero_weights) == 0:
        print("All weights pruned!")
        break

    # Find threshold to prune bottom PRUNE_PERCENT
    k = int(len(nonzero_weights) * PRUNE_PERCENT)
    if k == 0:
        k = 1
    threshold = torch.kthvalue(nonzero_weights, k).values.item()

    print(f"  Pruning threshold: {threshold:.6f}")
    print(f"  Weights below threshold: {k}")

    # Update masks
    with torch.no_grad():
        for i, layer in enumerate(model.layers):
            L_abs = layer.L.abs()
            D_abs = layer.D.abs()
            masks[f'L{i}'] = ((L_abs >= threshold) | (L_abs == 0)).float() * masks[f'L{i}']
            masks[f'D{i}'] = ((D_abs >= threshold) | (D_abs == 0)).float() * masks[f'D{i}']

            # Apply masks
            layer.L.data *= masks[f'L{i}']
            layer.D.data *= masks[f'D{i}']

    sparsity = get_sparsity(model)
    acc_before_ft = eval_model(model)
    print(f"  After pruning: acc={acc_before_ft:.1%}, sparsity={sparsity:.1%}")

    # Fine-tune
    model = finetune_with_masks(model, masks, FINETUNE_STEPS)
    acc_after_ft = eval_model(model)
    print(f"  After fine-tune: acc={acc_after_ft:.1%}")

    # Record
    result = {
        'iteration': iteration + 1,
        'threshold': threshold,
        'sparsity': sparsity,
        'acc_before_ft': acc_before_ft,
        'acc_after_ft': acc_after_ft,
        'acc_drop': baseline_acc - acc_after_ft,
        'state_dict': {k: v.cpu().clone() for k, v in model.state_dict().items()},
        'masks': {k: v.cpu().clone() for k, v in masks.items()},
    }
    results.append(result)

    # Track best within 1% threshold
    if result['acc_drop'] <= 0.01:
        best_within_threshold = result
        print(f"  ** Within 1% threshold (drop={result['acc_drop']:.1%})")

    # Check stopping condition
    if result['acc_drop'] > MAX_ACC_DROP:
        print(f"  Stopping: accuracy drop {result['acc_drop']:.1%} > {MAX_ACC_DROP:.0%}")
        break

# =============================================================================
# SUMMARY
# =============================================================================
print("\n" + "=" * 60)
print("SUMMARY")
print("=" * 60)

print(f"\nBaseline accuracy: {baseline_acc:.1%}")
print(f"\nIteration results:")
for r in results:
    marker = " *" if best_within_threshold and r['iteration'] == best_within_threshold['iteration'] else ""
    print(f"  {r['iteration']:2d}: acc={r['acc_after_ft']:.1%} (Δ={-r['acc_drop']:+.1%}), sparsity={r['sparsity']:.1%}{marker}")

if best_within_threshold:
    print(f"\nBest within 1% threshold: iteration {best_within_threshold['iteration']}")
    print(f"  Accuracy: {best_within_threshold['acc_after_ft']:.1%}")
    print(f"  Sparsity: {best_within_threshold['sparsity']:.1%}")

    # Save
    save_path = checkpoint_dir / f"3layer_n10_memory5_seed{SEED}_sparse.pkl"
    with open(save_path, 'wb') as f:
        pickle.dump({
            'state_dict': best_within_threshold['state_dict'],
            'masks': best_within_threshold['masks'],
            'accuracy': best_within_threshold['acc_after_ft'],
            'sparsity': best_within_threshold['sparsity'],
            'baseline_acc': baseline_acc,
            'iteration': best_within_threshold['iteration'],
            'config': {
                'n_task': N_TASK,
                'n_model': N_MODEL,
                'n_memory': N_MEMORY,
                'num_layers': NUM_LAYERS,
                'rank': RANK,
                'seed': SEED,
            },
            'all_iterations': [{k: v for k, v in r.items() if k != 'state_dict' and k != 'masks'}
                               for r in results],
        }, f)
    print(f"\nSaved to {save_path}")
else:
    print("\nNo iteration within 1% threshold - saving last good result")
    # Find best result overall
    best_result = min(results, key=lambda r: r['acc_drop'])
    save_path = checkpoint_dir / f"3layer_n10_memory5_seed{SEED}_sparse.pkl"
    with open(save_path, 'wb') as f:
        pickle.dump({
            'state_dict': best_result['state_dict'],
            'masks': best_result['masks'],
            'accuracy': best_result['acc_after_ft'],
            'sparsity': best_result['sparsity'],
            'baseline_acc': baseline_acc,
            'iteration': best_result['iteration'],
            'config': {
                'n_task': N_TASK,
                'n_model': N_MODEL,
                'n_memory': N_MEMORY,
                'num_layers': NUM_LAYERS,
                'rank': RANK,
                'seed': SEED,
            },
            'all_iterations': [{k: v for k, v in r.items() if k != 'state_dict' and k != 'masks'}
                               for r in results],
        }, f)
    print(f"  Saved iteration {best_result['iteration']} with acc={best_result['acc_after_ft']:.1%}")

# %%
