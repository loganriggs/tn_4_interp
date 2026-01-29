# %% [markdown]
# # Train 2-Layer Residual Bilinear with Down-Projection
#
# Architecture:
# ```
# r0 = W_embed @ x
# r1 = r0 + D1 @ (L1 @ norm1(r0) ⊙ R1 @ norm1(r0))
# r2 = r1 + D2 @ (L2 @ norm2(r1) ⊙ R2 @ norm2(r1))
# y = W_unembed @ r2
# ```
#
# Each bilinear layer has: L (d_res -> rank), R (d_res -> rank), D (rank -> d_res)

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
import matplotlib.pyplot as plt

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


class BilinearDownProj(nn.Module):
    """
    Bilinear layer with down-projection: output = D @ (Lx ⊙ Rx)

    L: (rank, input_dim) - left projection
    R: (rank, input_dim) - right projection
    D: (output_dim, rank) - down projection
    """
    def __init__(self, input_dim, output_dim, rank, init_std=0.01):
        super().__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.rank = rank

        # L and R project to rank dimensions
        self.L = nn.Linear(input_dim, rank, bias=False)
        self.R = nn.Linear(input_dim, rank, bias=False)
        # D projects back to output_dim
        self.D = nn.Linear(rank, output_dim, bias=False)

        nn.init.normal_(self.L.weight, std=init_std)
        nn.init.normal_(self.R.weight, std=init_std)
        nn.init.normal_(self.D.weight, std=init_std)

    def forward(self, x):
        # x: (batch, input_dim)
        Lx = self.L(x)  # (batch, rank)
        Rx = self.R(x)  # (batch, rank)
        return self.D(Lx * Rx)  # (batch, output_dim)


class ResidualBilinearDownProj(nn.Module):
    """
    2-layer residual bilinear WITHOUT RMSNorm, WITH down-projection.

    r0 = W_embed @ x
    r1 = r0 + D1 @ (L1 @ r0 ⊙ R1 @ r0)
    r2 = r1 + D2 @ (L2 @ r1 ⊙ R2 @ r1)
    y = W_unembed @ r2
    """
    def __init__(self, input_dim=3072, d_res=128, output_dim=10, rank=64, init_std=0.01):
        super().__init__()
        self.input_dim = input_dim
        self.d_res = d_res
        self.output_dim = output_dim
        self.rank = rank

        self.embed = nn.Linear(input_dim, d_res, bias=False)
        nn.init.normal_(self.embed.weight, std=init_std)

        self.bilinear1 = BilinearDownProj(d_res, d_res, rank, init_std)
        self.bilinear2 = BilinearDownProj(d_res, d_res, rank, init_std)

        self.unembed = nn.Linear(d_res, output_dim, bias=False)
        nn.init.normal_(self.unembed.weight, std=init_std)

    def forward(self, x):
        if x.dim() == 4:
            x = x.view(x.size(0), -1)
        r0 = self.embed(x)
        r1 = r0 + self.bilinear1(r0)
        r2 = r1 + self.bilinear2(r1)
        return self.unembed(r2)

    def forward_with_intermediates(self, x):
        if x.dim() == 4:
            x = x.view(x.size(0), -1)
        r0 = self.embed(x)
        b1 = self.bilinear1(r0)
        r1 = r0 + b1
        b2 = self.bilinear2(r1)
        r2 = r1 + b2
        logits = self.unembed(r2)
        return {'x': x, 'r0': r0, 'b1': b1, 'r1': r1, 'b2': b2, 'r2': r2, 'logits': logits}


class ResidualBilinearDownProjRMSNorm(nn.Module):
    """
    2-layer residual bilinear WITH RMSNorm and down-projection.

    r0 = W_embed @ x
    r1 = r0 + D1 @ (L1 @ norm1(r0) ⊙ R1 @ norm1(r0))
    r2 = r1 + D2 @ (L2 @ norm2(r1) ⊙ R2 @ norm2(r1))
    y = W_unembed @ r2
    """
    def __init__(self, input_dim=3072, d_res=128, output_dim=10, rank=64, init_std=0.01):
        super().__init__()
        self.input_dim = input_dim
        self.d_res = d_res
        self.output_dim = output_dim
        self.rank = rank

        self.embed = nn.Linear(input_dim, d_res, bias=False)
        nn.init.normal_(self.embed.weight, std=init_std)

        self.norm1 = RMSNorm(d_res)
        self.bilinear1 = BilinearDownProj(d_res, d_res, rank, init_std)

        self.norm2 = RMSNorm(d_res)
        self.bilinear2 = BilinearDownProj(d_res, d_res, rank, init_std)

        self.unembed = nn.Linear(d_res, output_dim, bias=False)
        nn.init.normal_(self.unembed.weight, std=init_std)

    def forward(self, x):
        if x.dim() == 4:
            x = x.view(x.size(0), -1)
        r0 = self.embed(x)
        r1 = r0 + self.bilinear1(self.norm1(r0))
        r2 = r1 + self.bilinear2(self.norm2(r1))
        return self.unembed(r2)

    def forward_with_intermediates(self, x):
        if x.dim() == 4:
            x = x.view(x.size(0), -1)
        r0 = self.embed(x)
        n1 = self.norm1(r0)
        b1 = self.bilinear1(n1)
        r1 = r0 + b1
        n2 = self.norm2(r1)
        b2 = self.bilinear2(n2)
        r2 = r1 + b2
        logits = self.unembed(r2)
        return {'x': x, 'r0': r0, 'n1': n1, 'b1': b1, 'r1': r1, 'n2': n2, 'b2': b2, 'r2': r2, 'logits': logits}


# %%
# =============================================================================
# DATA
# =============================================================================

SCRIPT_DIR = Path(__file__).parent.resolve() if '__file__' in dir() else Path.cwd()
DATA_DIR = SCRIPT_DIR / "data"

def get_svhn_loaders(batch_size=128):
    # Training augmentation
    train_transform = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomRotation(10),
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
        transforms.ToTensor(),
        transforms.Normalize((0.4377, 0.4438, 0.4728), (0.1980, 0.2010, 0.1970))
    ])
    # No augmentation for test
    test_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4377, 0.4438, 0.4728), (0.1980, 0.2010, 0.1970))
    ])
    train = datasets.SVHN(str(DATA_DIR), split='train', download=True, transform=train_transform)
    test = datasets.SVHN(str(DATA_DIR), split='test', download=True, transform=test_transform)
    return (DataLoader(train, batch_size=batch_size, shuffle=True, num_workers=2),
            DataLoader(test, batch_size=batch_size, shuffle=False, num_workers=2))


# %%
# =============================================================================
# TRAINING
# =============================================================================

def count_params(model):
    return sum(p.numel() for p in model.parameters())


def train_model(model, model_name, n_epochs=30, lr=0.001, weight_decay=0.01,
                grad_clip=1.0, device=None, verbose=True):
    """Train model and return history."""
    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    model = model.to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)

    train_loader, test_loader = get_svhn_loaders()
    total_steps = n_epochs * len(train_loader)

    # LR schedule
    milestones = [int(0.8 * total_steps), int(0.9 * total_steps)]
    scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones, gamma=0.1)

    history = {'train_loss': [], 'train_acc': [], 'val_loss': [], 'val_acc': [], 'steps': []}
    checkpoints = {}

    # Checkpoint schedule
    checkpoint_epochs = [0, 1, 2, 5, 10, 15, 20, 25, n_epochs-1]
    checkpoint_epochs = [e for e in checkpoint_epochs if e < n_epochs]

    def evaluate():
        model.eval()
        with torch.no_grad():
            val_loss, val_correct, val_total = 0, 0, 0
            for data, target in test_loader:
                data, target = data.to(device), target.to(device)
                output = model(data)
                val_loss += F.cross_entropy(output, target, reduction='sum').item()
                val_correct += (output.argmax(1) == target).sum().item()
                val_total += target.size(0)
        model.train()
        return val_loss / val_total, val_correct / val_total

    # Initial eval
    val_loss, val_acc = evaluate()
    history['steps'].append(0)
    history['train_loss'].append(float('nan'))
    history['train_acc'].append(float('nan'))
    history['val_loss'].append(val_loss)
    history['val_acc'].append(val_acc)
    if verbose:
        print(f"  Initial: val_acc={val_acc:.4f}")

    if 0 in checkpoint_epochs:
        checkpoints[0] = {k: v.cpu().clone() for k, v in model.state_dict().items()}

    global_step = 0
    model.train()

    for epoch in range(n_epochs):
        epoch_loss, epoch_correct, epoch_total = 0, 0, 0

        pbar = tqdm(train_loader, desc=f"{model_name} Epoch {epoch+1}/{n_epochs}", disable=not verbose)
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

            epoch_loss += loss.item() * target.size(0)
            epoch_correct += (output.argmax(1) == target).sum().item()
            epoch_total += target.size(0)

        # End of epoch eval
        val_loss, val_acc = evaluate()
        train_loss = epoch_loss / epoch_total
        train_acc = epoch_correct / epoch_total

        history['steps'].append(global_step)
        history['train_loss'].append(train_loss)
        history['train_acc'].append(train_acc)
        history['val_loss'].append(val_loss)
        history['val_acc'].append(val_acc)

        if verbose:
            print(f"  Epoch {epoch+1}: train_acc={train_acc:.4f}, val_acc={val_acc:.4f}")

        if epoch in checkpoint_epochs:
            checkpoints[epoch] = {k: v.cpu().clone() for k, v in model.state_dict().items()}

    # Always save final
    checkpoints[n_epochs-1] = {k: v.cpu().clone() for k, v in model.state_dict().items()}

    return history, checkpoints


# %%
# =============================================================================
# MAIN
# =============================================================================

if __name__ == "__main__":
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")

    output_dir = Path("tn_analysis_checkpoints")
    output_dir.mkdir(exist_ok=True)

    # Config
    d_res = 128
    rank = 64  # bilinear rank (L, R project to this, D projects back)
    n_epochs = 30

    results = {}

    # =========================================================================
    # Model 1: WITHOUT RMSNorm
    # =========================================================================
    print("\n" + "="*60)
    print("Training: Residual Bilinear DownProj (NO RMSNorm)")
    print("="*60)

    torch.manual_seed(42)
    model_no_norm = ResidualBilinearDownProj(
        input_dim=3072, d_res=d_res, output_dim=10, rank=rank
    )
    print(f"Parameters: {count_params(model_no_norm):,}")

    history_no_norm, ckpts_no_norm = train_model(
        model_no_norm, "NoRMSNorm", n_epochs=n_epochs, device=device
    )
    results['no_rmsnorm'] = {
        'history': history_no_norm,
        'checkpoints': ckpts_no_norm,
        'config': {'d_res': d_res, 'rank': rank, 'rmsnorm': False}
    }

    # =========================================================================
    # Model 2: WITH RMSNorm
    # =========================================================================
    print("\n" + "="*60)
    print("Training: Residual Bilinear DownProj (WITH RMSNorm)")
    print("="*60)

    torch.manual_seed(42)
    model_with_norm = ResidualBilinearDownProjRMSNorm(
        input_dim=3072, d_res=d_res, output_dim=10, rank=rank
    )
    print(f"Parameters: {count_params(model_with_norm):,}")

    history_with_norm, ckpts_with_norm = train_model(
        model_with_norm, "WithRMSNorm", n_epochs=n_epochs, device=device
    )
    results['with_rmsnorm'] = {
        'history': history_with_norm,
        'checkpoints': ckpts_with_norm,
        'config': {'d_res': d_res, 'rank': rank, 'rmsnorm': True}
    }

    # =========================================================================
    # Save results
    # =========================================================================
    save_path = output_dir / "residual_downproj_comparison.pkl"
    with open(save_path, 'wb') as f:
        pickle.dump(results, f)
    print(f"\nSaved to {save_path}")

    # =========================================================================
    # Summary
    # =========================================================================
    print("\n" + "="*60)
    print("SUMMARY")
    print("="*60)
    print(f"\nArchitecture: d_res={d_res}, rank={rank}")
    print(f"  bilinear(x) = D @ (Lx ⊙ Rx)")
    print(f"  L, R: ({d_res} -> {rank}), D: ({rank} -> {d_res})")
    print(f"\nNo RMSNorm:   final val_acc = {history_no_norm['val_acc'][-1]:.4f}")
    print(f"With RMSNorm: final val_acc = {history_with_norm['val_acc'][-1]:.4f}")

    # =========================================================================
    # Plot comparison
    # =========================================================================
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    # Accuracy
    ax = axes[0]
    epochs = range(len(history_no_norm['val_acc']))
    ax.plot(epochs, history_no_norm['val_acc'], 'b-o', label='No RMSNorm', markersize=3)
    ax.plot(epochs, history_with_norm['val_acc'], 'r-o', label='With RMSNorm', markersize=3)
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Val Accuracy')
    ax.set_title('Validation Accuracy')
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Loss
    ax = axes[1]
    ax.semilogy(epochs[1:], history_no_norm['val_loss'][1:], 'b-o', label='No RMSNorm', markersize=3)
    ax.semilogy(epochs[1:], history_with_norm['val_loss'][1:], 'r-o', label='With RMSNorm', markersize=3)
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Val Loss')
    ax.set_title('Validation Loss')
    ax.legend()
    ax.grid(True, alpha=0.3)

    plt.suptitle(f'Residual Bilinear with Down-Projection\n(d_res={d_res}, rank={rank})', fontsize=12)
    plt.tight_layout()
    plt.savefig(output_dir / 'residual_downproj_comparison.png', dpi=150)
    plt.show()

    print(f"\nSaved plot to {output_dir / 'residual_downproj_comparison.png'}")
