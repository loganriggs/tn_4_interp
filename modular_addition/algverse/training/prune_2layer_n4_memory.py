"""Iterative prune the 2-layer n=4 memory model (seed 2)."""
import torch, torch.nn as nn, pickle, sys
sys.path.insert(0, '/workspace/tn_4_interp/modular_addition/algverse')
from models import SymmetricBilinearResidual, task_2nd_argmax

N_TASK = 4; N_MODEL = 5; NUM_LAYERS = 2; RANK = 5; device = 'cuda'

# Load best model
with open('checkpoints/2layer_n4_memory1_best.pkl', 'rb') as f:
    ckpt = pickle.load(f)
seed = ckpt['config']['seed']
baseline_acc = ckpt['accuracy']

model = SymmetricBilinearResidual(N_MODEL, NUM_LAYERS, RANK).to(device)
model.load_state_dict({k: v.to(device) for k, v in ckpt['state_dict'].items()})

def evaluate(model):
    model.eval()
    with torch.no_grad():
        x_task = torch.randn(10000, N_TASK, device=device)
        targets = task_2nd_argmax(x_task)
        x = torch.cat([x_task, torch.zeros(10000, 1, device=device)], dim=1)
        return (model(x)[:, :N_TASK].argmax(dim=1) == targets).float().mean().item()

print(f'Baseline: seed={seed}, acc={baseline_acc:.1%}')
print(f'Verified: {evaluate(model):.1%}')

# Iterative pruning
masks = {}
for i, layer in enumerate(model.layers):
    masks[f'L{i}'] = torch.ones_like(layer.L)
    masks[f'D{i}'] = torch.ones_like(layer.D)

def get_sparsity(model):
    total = sum(l.L.numel() + l.D.numel() for l in model.layers)
    zeros = sum((l.L == 0).sum().item() + (l.D == 0).sum().item() for l in model.layers)
    return zeros / total

results = []
best_within = None

for iteration in range(30):
    all_w = []
    for i, layer in enumerate(model.layers):
        all_w.append((layer.L.detach() * masks[f'L{i}']).abs().flatten())
        all_w.append((layer.D.detach() * masks[f'D{i}']).abs().flatten())
    all_w = torch.cat(all_w)
    nonzero = all_w[all_w > 0]
    if len(nonzero) == 0: break

    k = max(int(len(nonzero) * 0.10), 1)
    threshold = torch.kthvalue(nonzero, k).values.item()

    with torch.no_grad():
        for i, layer in enumerate(model.layers):
            masks[f'L{i}'] = ((layer.L.abs() > threshold) | (layer.L.abs() == 0)).float() * masks[f'L{i}']
            masks[f'D{i}'] = ((layer.D.abs() > threshold) | (layer.D.abs() == 0)).float() * masks[f'D{i}']
            layer.L.data *= masks[f'L{i}']
            layer.D.data *= masks[f'D{i}']

    sparsity = get_sparsity(model)
    acc_pre = evaluate(model)

    optimizer = torch.optim.AdamW(model.parameters(), lr=0.005, weight_decay=0.001)
    for step in range(3000):
        model.train()
        x_task = torch.randn(128, N_TASK, device=device)
        targets = task_2nd_argmax(x_task)
        x = torch.cat([x_task, torch.zeros(128, 1, device=device)], dim=1)
        loss = nn.functional.cross_entropy(model(x)[:, :N_TASK], targets)
        optimizer.zero_grad(); loss.backward()
        with torch.no_grad():
            for i, layer in enumerate(model.layers):
                if layer.L.grad is not None: layer.L.grad *= masks[f'L{i}']
                if layer.D.grad is not None: layer.D.grad *= masks[f'D{i}']
        optimizer.step()
        with torch.no_grad():
            for i, layer in enumerate(model.layers):
                layer.L.data *= masks[f'L{i}']
                layer.D.data *= masks[f'D{i}']

    acc_post = evaluate(model)
    drop = baseline_acc - acc_post
    total_params = sum(l.L.numel() + l.D.numel() for l in model.layers)
    nz = int(total_params * (1 - sparsity))
    print(f'  Iter {iteration+1:2d}: pre={acc_pre:.1%}, post={acc_post:.1%}, drop={drop:+.1%}, sparsity={sparsity:.1%} ({nz}/{total_params})')

    results.append({'iteration': iteration+1, 'sparsity': sparsity, 'acc_post': acc_post, 'acc_drop': drop,
                    'state_dict': {k: v.cpu().clone() for k, v in model.state_dict().items()}})
    if drop <= 0.01:
        best_within = results[-1]
    if drop > 0.03:
        print(f'  Stopping: drop {drop:.1%} > 3%')
        break

if best_within is None:
    best_within = min(results, key=lambda r: r['acc_drop'])

best = best_within
print(f'\nBest sparse: acc={best["acc_post"]:.1%}, sparsity={best["sparsity"]:.1%}')

save_path = 'checkpoints/2layer_n4_memory1_sparse.pkl'
with open(save_path, 'wb') as f:
    pickle.dump({
        'state_dict': best['state_dict'],
        'config': {'n_task': N_TASK, 'n_model': N_MODEL, 'num_layers': NUM_LAYERS, 'rank': RANK, 'seed': seed},
        'accuracy': best['acc_post'], 'sparsity': best['sparsity'],
        'baseline_acc': baseline_acc, 'iteration': best['iteration'],
    }, f)
print(f'Saved to {save_path}')
