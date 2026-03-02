# %%
"""
Prune 3-Layer Memory10 Model

Apply L1 pruning to the trained 3-layer model with 10 memory dimensions.
"""

import torch
import torch.nn as nn
import numpy as np
import pickle
from pathlib import Path
import sys

PROJECT_ROOT = Path("/workspace/tn_4_interp/modular_addition/algverse")
sys.path.insert(0, str(PROJECT_ROOT))
from models import SymmetricBilinearResidual, task_2nd_argmax

checkpoint_dir = PROJECT_ROOT / "checkpoints"
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Device: {device}")

N_TASK = 10
N_MEMORY = 10
N_MODEL = N_TASK + N_MEMORY  # 20

# %%
# =============================================================================
# LOAD BASE MODEL
# =============================================================================
checkpoint_path = checkpoint_dir / "3layer_n10_memory10_seed4.pkl"
print(f"Loading base model from {checkpoint_path}")

with open(checkpoint_path, 'rb') as f:
    checkpoint = pickle.load(f)

config = checkpoint['config']
print(f"Config: {config}")
print(f"Base accuracy: {checkpoint['accuracy']*100:.1f}%")

model = SymmetricBilinearResidual(N_MODEL, 3, config['rank']).to(device)
model.load_state_dict({k: v.to(device) for k, v in checkpoint['state_dict'].items()})

# %%
# =============================================================================
# EVALUATION FUNCTION
# =============================================================================
def evaluate(model, n_samples=10000):
    model.eval()
    with torch.no_grad():
        x_task = torch.randn(n_samples, N_TASK, device=device)
        x = torch.cat([x_task, torch.zeros(n_samples, N_MEMORY, device=device)], dim=1)
        targets = task_2nd_argmax(x_task)
        output = model(x)
        preds = output[:, :N_TASK].argmax(dim=1)
        accuracy = (preds == targets).float().mean().item()
    return accuracy

def compute_sparsity(model):
    total_zeros = 0
    total_params = 0
    for layer in model.layers:
        total_zeros += (layer.L == 0).sum().item()
        total_zeros += (layer.D == 0).sum().item()
        total_params += layer.L.numel()
        total_params += layer.D.numel()
    return total_zeros / total_params

print(f"Initial accuracy: {evaluate(model)*100:.1f}%")
print(f"Initial sparsity: {compute_sparsity(model)*100:.1f}%")

# %%
# =============================================================================
# PHASE 1: L1 PENALTY TRAINING
# =============================================================================
print("\n" + "=" * 60)
print("PHASE 1: L1 PENALTY TRAINING")
print("=" * 60)

optimizer = torch.optim.AdamW(model.parameters(), lr=0.005, weight_decay=0.001)
l1_lambda = 0.001
n_steps = 10000
batch_size = 256

model.train()
for step in range(n_steps + 1):
    x_task = torch.randn(batch_size, N_TASK, device=device)
    x = torch.cat([x_task, torch.zeros(batch_size, N_MEMORY, device=device)], dim=1)
    targets = task_2nd_argmax(x_task)

    output = model(x)
    ce_loss = nn.functional.cross_entropy(output[:, :N_TASK], targets)

    # L1 penalty on L and D weights
    l1_loss = sum(layer.L.abs().sum() + layer.D.abs().sum() for layer in model.layers)
    loss = ce_loss + l1_lambda * l1_loss

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    if step % 1000 == 0:
        acc = evaluate(model)
        sp = compute_sparsity(model)
        print(f"Step {step}: CE={ce_loss.item():.4f}, L1={l1_loss.item():.1f}, Acc={acc*100:.1f}%, Sparsity={sp*100:.1f}%")

print(f"\nAfter L1 training: Acc={evaluate(model)*100:.1f}%, Sparsity={compute_sparsity(model)*100:.1f}%")

# %%
# =============================================================================
# PHASE 2: THRESHOLD PRUNING
# =============================================================================
print("\n" + "=" * 60)
print("PHASE 2: THRESHOLD PRUNING")
print("=" * 60)

# Try different thresholds
thresholds = [0.05, 0.1, 0.15, 0.2]
best_threshold = None
best_acc = 0
best_sparsity = 0

for threshold in thresholds:
    # Create a copy of the model state
    temp_state = {k: v.clone() for k, v in model.state_dict().items()}

    # Apply threshold
    for i, layer in enumerate(model.layers):
        mask_L = (layer.L.abs() >= threshold).float()
        mask_D = (layer.D.abs() >= threshold).float()
        layer.L.data *= mask_L
        layer.D.data *= mask_D

    acc = evaluate(model)
    sp = compute_sparsity(model)
    print(f"Threshold {threshold}: Acc={acc*100:.1f}%, Sparsity={sp*100:.1f}%")

    if acc > best_acc or (acc == best_acc and sp > best_sparsity):
        best_acc = acc
        best_sparsity = sp
        best_threshold = threshold

    # Restore model
    model.load_state_dict(temp_state)

print(f"\nBest threshold: {best_threshold} (Acc={best_acc*100:.1f}%, Sparsity={best_sparsity*100:.1f}%)")

# Apply best threshold
for layer in model.layers:
    mask_L = (layer.L.abs() >= best_threshold).float()
    mask_D = (layer.D.abs() >= best_threshold).float()
    layer.L.data *= mask_L
    layer.D.data *= mask_D

# Store masks for fine-tuning
masks = {}
for i, layer in enumerate(model.layers):
    masks[f'L{i}'] = (layer.L != 0).float()
    masks[f'D{i}'] = (layer.D != 0).float()

print(f"\nAfter pruning: Acc={evaluate(model)*100:.1f}%, Sparsity={compute_sparsity(model)*100:.1f}%")

# %%
# =============================================================================
# PHASE 3: FINE-TUNE WITH MASKED GRADIENTS
# =============================================================================
print("\n" + "=" * 60)
print("PHASE 3: FINE-TUNE WITH MASKED GRADIENTS")
print("=" * 60)

optimizer = torch.optim.AdamW(model.parameters(), lr=0.005, weight_decay=0.001)
n_finetune_steps = 5000

model.train()
for step in range(n_finetune_steps + 1):
    x_task = torch.randn(batch_size, N_TASK, device=device)
    x = torch.cat([x_task, torch.zeros(batch_size, N_MEMORY, device=device)], dim=1)
    targets = task_2nd_argmax(x_task)

    output = model(x)
    loss = nn.functional.cross_entropy(output[:, :N_TASK], targets)

    optimizer.zero_grad()
    loss.backward()

    # Mask gradients to keep pruned weights at zero
    with torch.no_grad():
        for i, layer in enumerate(model.layers):
            layer.L.grad *= masks[f'L{i}']
            layer.D.grad *= masks[f'D{i}']

    optimizer.step()

    # Re-apply mask after step
    with torch.no_grad():
        for i, layer in enumerate(model.layers):
            layer.L.data *= masks[f'L{i}']
            layer.D.data *= masks[f'D{i}']

    if step % 500 == 0:
        acc = evaluate(model)
        sp = compute_sparsity(model)
        print(f"Step {step}: Loss={loss.item():.4f}, Acc={acc*100:.1f}%, Sparsity={sp*100:.1f}%")

final_acc = evaluate(model)
final_sparsity = compute_sparsity(model)
print(f"\nFinal: Acc={final_acc*100:.1f}%, Sparsity={final_sparsity*100:.1f}%")

# %%
# =============================================================================
# SAVE SPARSE MODEL
# =============================================================================
print("\n" + "=" * 60)
print("SAVING SPARSE MODEL")
print("=" * 60)

save_path = checkpoint_dir / "3layer_n10_memory10_seed4_sparse.pkl"

sparse_checkpoint = {
    'state_dict': {k: v.cpu() for k, v in model.state_dict().items()},
    'config': config,
    'accuracy': final_acc,
    'sparsity': final_sparsity,
    'baseline_acc': checkpoint['accuracy'],
    'threshold': best_threshold,
    'iteration': 'sparse_from_seed4',
}

with open(save_path, 'wb') as f:
    pickle.dump(sparse_checkpoint, f)

print(f"Saved sparse model to {save_path}")
print(f"  Accuracy: {final_acc*100:.1f}% (baseline: {checkpoint['accuracy']*100:.1f}%)")
print(f"  Sparsity: {final_sparsity*100:.1f}%")
print(f"  Threshold: {best_threshold}")

# %%
# =============================================================================
# PER-LAYER SPARSITY
# =============================================================================
print("\n" + "=" * 60)
print("PER-LAYER SPARSITY")
print("=" * 60)

for i, layer in enumerate(model.layers):
    L_sp = (layer.L == 0).sum().item() / layer.L.numel() * 100
    D_sp = (layer.D == 0).sum().item() / layer.D.numel() * 100
    print(f"Layer {i+1}: L={L_sp:.1f}% sparse, D={D_sp:.1f}% sparse")

print("\nDone!")
