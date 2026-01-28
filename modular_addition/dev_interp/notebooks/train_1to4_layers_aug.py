# %% [markdown]
# # Train 1-4 Layer Residual Bilinear with Down Projection + Augmentation
#
# Best method from ablation: augmentation prevents overfitting and divergence.
#
# REMINDER: Always align x-axis of all plots with the TN-Sim matrix x-axis!

# %%
import pickle
from pathlib import Path
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from tqdm import tqdm

# %%
# =============================================================================
# MODELS
# =============================================================================

class RMSNorm(nn.Module):
    def __init__(self, dim, eps=1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x):
        rms = torch.sqrt(torch.mean(x ** 2, dim=-1, keepdim=True) + self.eps)
        return self.weight * (x / rms)


class BilinearLayerWithDownProj(nn.Module):
    """B(x) = D @ (Lx ⊙ Rx)"""
    def __init__(self, input_dim, hidden_dim, output_dim, init_std=0.01):
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        self.LR = nn.Linear(input_dim, 2 * hidden_dim, bias=False)
        nn.init.normal_(self.LR.weight, std=init_std)
        self.D = nn.Linear(hidden_dim, output_dim, bias=False)
        nn.init.normal_(self.D.weight, std=init_std)

    @property
    def L(self):
        return self.LR.weight[:self.hidden_dim]

    @property
    def R(self):
        return self.LR.weight[self.hidden_dim:]

    def forward(self, x):
        combined = self.LR(x)
        left, right = combined.chunk(2, dim=-1)
        h = left * right
        return self.D(h)


class ResidualBilinearNLayersDownProj(nn.Module):
    """
    N-layer residual bilinear with RMSNorm and down projections.
    """
    def __init__(self, n_layers, input_dim=3072, d_res=128, d_hidden=256, output_dim=10, init_std=0.01):
        super().__init__()
        self.n_layers = n_layers
        self.input_dim = input_dim
        self.d_res = d_res
        self.d_hidden = d_hidden
        self.output_dim = output_dim

        self.embed = nn.Linear(input_dim, d_res, bias=False)
        nn.init.normal_(self.embed.weight, std=init_std)

        self.norms = nn.ModuleList([RMSNorm(d_res) for _ in range(n_layers)])
        self.bilinears = nn.ModuleList([
            BilinearLayerWithDownProj(d_res, d_hidden, d_res, init_std)
            for _ in range(n_layers)
        ])

        self.projection = nn.Linear(d_res, output_dim, bias=False)
        nn.init.normal_(self.projection.weight, std=init_std)

    def forward(self, x):
        if x.dim() == 4:
            x = x.view(x.size(0), -1)

        r = self.embed(x)
        for norm, bilinear in zip(self.norms, self.bilinears):
            r = r + bilinear(norm(r))

        return self.projection(r)


# %%
# =============================================================================
# DATA WITH AUGMENTATION
# =============================================================================

def get_svhn_loaders_augmented(batch_size=128, num_workers=4):
    """SVHN loaders with augmentations (best method from ablation)."""

    train_transform = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
        transforms.ToTensor(),
        transforms.Normalize((0.4377, 0.4438, 0.4728), (0.1980, 0.2010, 0.1970)),
        transforms.RandomErasing(p=0.5, scale=(0.02, 0.2), ratio=(0.3, 3.3)),
    ])

    test_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4377, 0.4438, 0.4728), (0.1980, 0.2010, 0.1970))
    ])

    train = datasets.SVHN('./data', split='train', download=True, transform=train_transform)
    test = datasets.SVHN('./data', split='test', download=True, transform=test_transform)

    return (
        DataLoader(train, batch_size=batch_size, shuffle=True,
                   num_workers=num_workers, pin_memory=True, persistent_workers=True),
        DataLoader(test, batch_size=batch_size, shuffle=False,
                   num_workers=num_workers, pin_memory=True, persistent_workers=True)
    )


# %%
# =============================================================================
# TRAINING
# =============================================================================

def train_and_save(model, model_name, output_dir, n_epochs=60, lr=0.001,
                   weight_decay=0.01, grad_clip=1.0, device=None):
    """Train model and save checkpoints."""
    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    model = model.to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)

    train_loader, test_loader = get_svhn_loaders_augmented(num_workers=4)
    steps_per_epoch = len(train_loader)
    total_steps = n_epochs * steps_per_epoch

    milestones = [int(0.8 * total_steps), int(0.9 * total_steps)]
    scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones, gamma=0.1)

    # Checkpoint schedule
    checkpoint_steps = set([0])
    checkpoint_steps.update(range(1, 21))
    checkpoint_steps.update(range(20, 201, 10))
    checkpoint_steps.update(range(200, 1001, 50))
    checkpoint_steps.update(range(1000, 5001, 200))
    checkpoint_steps.update(range(5000, 15001, 500))
    checkpoint_steps.update(range(15000, total_steps + 1, 1000))
    checkpoint_steps.add(total_steps)
    checkpoint_steps = sorted([s for s in checkpoint_steps if s <= total_steps])

    print(f"Training {model_name} for {n_epochs} epochs ({total_steps} steps)")
    print(f"n_layers={model.n_layers}, d_res={model.d_res}, d_hidden={model.d_hidden}")
    print(f"Will save {len(checkpoint_steps)} checkpoints")

    checkpoints = {}
    history = {
        'steps': [],
        'train_loss': [],
        'train_acc': [],
        'val_loss': [],
        'val_acc': [],
    }

    # Evaluation loader (no augmentation)
    eval_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4377, 0.4438, 0.4728), (0.1980, 0.2010, 0.1970))
    ])
    eval_train = datasets.SVHN('./data', split='train', download=True, transform=eval_transform)
    eval_train_loader = DataLoader(eval_train, batch_size=128, shuffle=False, num_workers=2)

    def evaluate():
        model.eval()
        with torch.no_grad():
            train_loss, train_correct, train_total = 0, 0, 0
            for i, (data, target) in enumerate(eval_train_loader):
                if i >= 10:
                    break
                data, target = data.to(device), target.to(device)
                output = model(data)
                train_loss += F.cross_entropy(output, target, reduction='sum').item()
                train_correct += (output.argmax(1) == target).sum().item()
                train_total += target.size(0)

            val_loss, val_correct, val_total = 0, 0, 0
            for data, target in test_loader:
                data, target = data.to(device), target.to(device)
                output = model(data)
                val_loss += F.cross_entropy(output, target, reduction='sum').item()
                val_correct += (output.argmax(1) == target).sum().item()
                val_total += target.size(0)

        model.train()
        return {
            'train_loss': train_loss / train_total,
            'train_acc': train_correct / train_total,
            'val_loss': val_loss / val_total,
            'val_acc': val_correct / val_total,
        }

    # Initial checkpoint
    checkpoints[0] = {k: v.cpu().clone() for k, v in model.state_dict().items()}
    metrics = evaluate()
    history['steps'].append(0)
    for k in ['train_loss', 'train_acc', 'val_loss', 'val_acc']:
        history[k].append(metrics[k])
    print(f"Step 0: val_acc={metrics['val_acc']:.4f}")

    # Training
    global_step = 0
    model.train()

    for epoch in range(n_epochs):
        pbar = tqdm(train_loader, desc=f"{model_name} Epoch {epoch+1}/{n_epochs}")
        for data, target in pbar:
            global_step += 1
            data, target = data.to(device), target.to(device)

            optimizer.zero_grad()
            output = model(data)
            loss = F.cross_entropy(output, target)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
            optimizer.step()
            scheduler.step()

            if global_step in checkpoint_steps:
                checkpoints[global_step] = {k: v.cpu().clone() for k, v in model.state_dict().items()}
                metrics = evaluate()
                history['steps'].append(global_step)
                for k in ['train_loss', 'train_acc', 'val_loss', 'val_acc']:
                    history[k].append(metrics[k])
                pbar.set_postfix({'val_acc': f"{metrics['val_acc']:.3f}"})

    # Save
    save_data = {
        'checkpoints': checkpoints,
        'history': history,
        'config': {
            'model_name': model_name,
            'n_layers': model.n_layers,
            'input_dim': model.input_dim,
            'd_res': model.d_res,
            'd_hidden': model.d_hidden,
            'output_dim': model.output_dim,
            'n_epochs': n_epochs,
            'lr': lr,
            'weight_decay': weight_decay,
            'grad_clip': grad_clip,
            'total_steps': global_step,
            'checkpoint_steps': list(checkpoints.keys()),
            'augmentation': 'RandomCrop+ColorJitter+RandomErasing',
        }
    }

    save_path = output_dir / f"{model_name}.pkl"
    with open(save_path, 'wb') as f:
        pickle.dump(save_data, f)

    print(f"\nSaved {model_name} to {save_path}")
    print(f"  {len(checkpoints)} checkpoints")
    print(f"  Final val_acc: {history['val_acc'][-1]:.4f}")

    return save_data


# %%
# =============================================================================
# MAIN
# =============================================================================

if __name__ == "__main__":
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")

    output_dir = Path("tn_analysis_checkpoints")
    output_dir.mkdir(exist_ok=True)

    d_res = 128
    d_hidden = 256
    n_epochs = 60

    results = {}

    for n_layers in [1, 2, 3, 4]:
        print("\n" + "="*60)
        print(f"Training: {n_layers}-Layer Residual Bilinear + Down Proj + Aug")
        print(f"d_res={d_res}, d_hidden={d_hidden}")
        print("="*60)

        torch.manual_seed(42)
        model = ResidualBilinearNLayersDownProj(
            n_layers=n_layers,
            input_dim=3072,
            d_res=d_res,
            d_hidden=d_hidden,
            output_dim=10
        )

        n_params = sum(p.numel() for p in model.parameters())
        print(f"Total parameters: {n_params:,}")

        result = train_and_save(
            model,
            model_name=f"residual_{n_layers}layer_downproj_aug",
            output_dir=output_dir,
            n_epochs=n_epochs,
            device=device,
        )
        results[n_layers] = result

    # Summary
    print("\n" + "="*60)
    print("SUMMARY")
    print("="*60)
    for n_layers, result in results.items():
        h = result['history']
        print(f"{n_layers}-Layer: {len(result['checkpoints'])} checkpoints, "
              f"final val_acc={h['val_acc'][-1]:.4f}, val_loss={h['val_loss'][-1]:.4f}")
