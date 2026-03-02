# %%
"""
Train 1-Layer models for n=3, 4, 5 using known good seeds.
Saves to checkpoints folder.
"""

import torch
import sys
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))
from models import train_with_l1_and_prune, get_good_seed

# %%
checkpoint_dir = Path(__file__).parent.parent / "checkpoints"
checkpoint_dir.mkdir(exist_ok=True)

configs = [
    {'n': 3, 'rank': 3, 'seed': 0},       # n=3 is easy
    {'n': 4, 'rank': 4, 'seed': 4},       # known from earlier
    {'n': 5, 'rank': 5, 'seed': 9},       # from sweep: 59.6%
]

for cfg in configs:
    n, rank, seed = cfg['n'], cfg['rank'], cfg['seed']
    print(f"\n{'='*60}")
    print(f"Training n={n}, rank={rank}, seed={seed}")
    print(f"{'='*60}")

    acc, state, sparsity_info = train_with_l1_and_prune(
        n=n, num_layers=1, rank=rank, seed=seed, thresh=0.20
    )

    save_path = checkpoint_dir / f"1layer_n{n}_r{rank}_seed{seed}.pt"
    torch.save({
        'state_dict': state,
        'config': {'n': n, 'num_layers': 1, 'rank': rank, 'seed': seed},
        'accuracy': acc,
        'sparsity': sparsity_info,
    }, save_path)
    print(f"Saved to {save_path}")

print("\nDone!")
# %%
