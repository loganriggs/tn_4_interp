"""
Train χ-net and residual bilinear network on SVHN.

Replicates the setup from Dooms et al. 2025:
- SVHN dataset (grayscale, with input noise)
- 3 hidden bilinear layers, hidden dim 256
- Bias via appending 1 to input
- RMSBatchNorm
- AdamW, lr=0.001, weight_decay=1.0, cosine schedule, 20 epochs
- Batch size 2048

Expected: ~85% test accuracy for χ-net, slightly higher for residual variant.
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, ConcatDataset
import torchvision
import torchvision.transforms as transforms
import os
import argparse
from model import ChiNet, ResidualBilinearNet


def get_svhn_loaders(batch_size=2048, input_noise=0.3, num_workers=4,
                     data_root='./data'):
    """
    Load SVHN dataset: grayscale, concatenate train+extra, add noise.
    """
    # Grayscale transform
    transform = transforms.Compose([
        transforms.Grayscale(num_output_channels=1),
        transforms.ToTensor(),
        transforms.Lambda(lambda x: x.flatten()),  # (1, 32, 32) -> (1024,)
    ])

    # Load train set (don't download, assume data exists)
    train_set = torchvision.datasets.SVHN(
        root=data_root, split='train', download=False, transform=transform)

    # Try to load extra set, skip if not available
    try:
        extra_set = torchvision.datasets.SVHN(
            root=data_root, split='extra', download=False, transform=transform)
        full_train = ConcatDataset([train_set, extra_set])
        print(f"  Using train ({len(train_set)}) + extra ({len(extra_set)}) = {len(full_train)} samples")
    except Exception:
        full_train = train_set
        print(f"  Using train only: {len(train_set)} samples (extra set not available)")

    test_set = torchvision.datasets.SVHN(
        root=data_root, split='test', download=False, transform=transform)

    train_loader = DataLoader(
        full_train, batch_size=batch_size, shuffle=True,
        num_workers=num_workers, pin_memory=True)
    test_loader = DataLoader(
        test_set, batch_size=batch_size, shuffle=False,
        num_workers=num_workers, pin_memory=True)

    return train_loader, test_loader, input_noise


def train_epoch(model, loader, optimizer, scheduler, device, input_noise=0.0):
    model.train()
    total_loss = 0
    total_correct = 0
    total_samples = 0

    criterion = nn.CrossEntropyLoss()

    for x, y in loader:
        x, y = x.to(device), y.to(device)

        # Add input noise during training
        if input_noise > 0:
            x = x + input_noise * torch.randn_like(x)

        logits = model(x)
        loss = criterion(logits, y)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        scheduler.step()

        total_loss += loss.item() * x.size(0)
        total_correct += (logits.argmax(dim=-1) == y).sum().item()
        total_samples += x.size(0)

    return total_loss / total_samples, total_correct / total_samples


@torch.no_grad()
def evaluate(model, loader, device):
    model.eval()
    total_loss = 0
    total_correct = 0
    total_samples = 0

    criterion = nn.CrossEntropyLoss()

    for x, y in loader:
        x, y = x.to(device), y.to(device)
        logits = model(x)
        loss = criterion(logits, y)

        total_loss += loss.item() * x.size(0)
        total_correct += (logits.argmax(dim=-1) == y).sum().item()
        total_samples += x.size(0)

    return total_loss / total_samples, total_correct / total_samples


def train(model_type='chi_net', n_layers=3, hidden_dim=256,
          epochs=20, lr=0.001, weight_decay=1.0, batch_size=2048,
          input_noise=0.3, device='cuda', save_dir='models',
          data_root='./data'):
    """
    Train a model and save checkpoint.

    Args:
        model_type: 'chi_net' or 'residual'
    """
    os.makedirs(save_dir, exist_ok=True)
    device = torch.device(device if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # Data
    train_loader, test_loader, noise = get_svhn_loaders(batch_size, input_noise,
                                                         data_root=data_root)
    input_dim = 32 * 32  # grayscale SVHN
    output_dim = 10

    # Model
    if model_type == 'chi_net':
        model = ChiNet(input_dim, hidden_dim, output_dim, n_layers)
    elif model_type == 'residual':
        model = ResidualBilinearNet(input_dim, hidden_dim, output_dim, n_layers)
    else:
        raise ValueError(f"Unknown model type: {model_type}")

    model = model.to(device)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"Model: {model_type}, layers={n_layers}, hidden={hidden_dim}")
    print(f"Parameters: {n_params:,}")

    # Optimizer
    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    total_steps = epochs * len(train_loader)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=total_steps)

    # Training loop
    for epoch in range(epochs):
        train_loss, train_acc = train_epoch(
            model, train_loader, optimizer, scheduler, device, noise)
        test_loss, test_acc = evaluate(model, test_loader, device)

        print(f"Epoch {epoch+1:2d}/{epochs} | "
              f"Train loss: {train_loss:.4f}, acc: {train_acc:.4f} | "
              f"Test loss: {test_loss:.4f}, acc: {test_acc:.4f}")

    # Save
    save_path = os.path.join(save_dir, f'{model_type}_L{n_layers}_h{hidden_dim}.pt')
    torch.save({
        'model_state_dict': model.state_dict(),
        'model_type': model_type,
        'n_layers': n_layers,
        'hidden_dim': hidden_dim,
        'input_dim': input_dim,
        'output_dim': output_dim,
        'test_acc': test_acc,
        'test_loss': test_loss,
    }, save_path)
    print(f"\nSaved to {save_path}")
    print(f"Final test accuracy: {test_acc:.4f}")

    return model


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str, default='chi_net',
                        choices=['chi_net', 'residual', 'both'])
    parser.add_argument('--layers', type=int, default=3)
    parser.add_argument('--hidden', type=int, default=256)
    parser.add_argument('--epochs', type=int, default=20)
    parser.add_argument('--lr', type=float, default=0.001)
    parser.add_argument('--wd', type=float, default=1.0)
    parser.add_argument('--batch_size', type=int, default=2048)
    parser.add_argument('--noise', type=float, default=0.3)
    parser.add_argument('--device', type=str, default='cuda')
    parser.add_argument('--data_root', type=str, default='./data')
    args = parser.parse_args()

    if args.model == 'both':
        for mt in ['chi_net', 'residual']:
            print(f"\n{'='*60}")
            print(f"Training {mt}")
            print(f"{'='*60}")
            train(mt, args.layers, args.hidden, args.epochs,
                  args.lr, args.wd, args.batch_size, args.noise, args.device,
                  data_root=args.data_root)
    else:
        train(args.model, args.layers, args.hidden, args.epochs,
              args.lr, args.wd, args.batch_size, args.noise, args.device,
              data_root=args.data_root)
