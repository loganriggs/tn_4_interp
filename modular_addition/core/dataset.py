"""
Dataset creation and training utilities for modular addition experiments.

This module provides:
- Full dataset creation (all P² input pairs)
- One-hot encoding for modular addition inputs
- DataLoader creation utilities
- Basic training loop utilities
"""
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
from typing import Callable


def create_full_dataset(P: int, device: torch.device | None = None) -> torch.Tensor:
    """
    Create all P² input pairs as one-hot encoded vectors.

    For modular addition (a + b) mod P, creates all pairs (a, b) where a, b ∈ [0, P).
    Each input is encoded as a 2*P dimensional vector with one-hot encoding
    for each of the two positions.

    Args:
        P: Input dimension (number of classes)
        device: Device to create tensor on

    Returns:
        Tensor of shape (P², 2*P) with one-hot encoded inputs.
        Row index a*P + b corresponds to input pair (a, b).
    """
    if device is None:
        device = torch.device('cpu')

    dataset = torch.zeros(P * P, 2 * P, device=device)
    for a in range(P):
        for b in range(P):
            idx = a * P + b
            dataset[idx, a] = 1.0        # one-hot for a in first P positions
            dataset[idx, P + b] = 1.0    # one-hot for b in second P positions
    return dataset


def create_labels(P: int, device: torch.device | None = None) -> torch.Tensor:
    """
    Create labels for the full modular addition dataset.

    Args:
        P: Input dimension (number of classes)
        device: Device to create tensor on

    Returns:
        Tensor of shape (P²,) with labels (a + b) mod P
    """
    if device is None:
        device = torch.device('cpu')

    labels = torch.zeros(P * P, dtype=torch.long, device=device)
    for a in range(P):
        for b in range(P):
            idx = a * P + b
            labels[idx] = (a + b) % P
    return labels


def create_train_val_split(
    P: int,
    train_fraction: float = 0.5,
    seed: int | None = None,
    device: torch.device | None = None
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Create train/validation split of the modular addition dataset.

    Args:
        P: Input dimension (number of classes)
        train_fraction: Fraction of data to use for training
        seed: Random seed for reproducibility
        device: Device to create tensors on

    Returns:
        Tuple of (train_data, train_labels, val_data, val_labels)
    """
    if seed is not None:
        torch.manual_seed(seed)
        np.random.seed(seed)

    dataset = create_full_dataset(P, device)
    labels = create_labels(P, device)

    n_total = P * P
    n_train = int(n_total * train_fraction)

    # Random permutation of indices
    perm = torch.randperm(n_total, device=device)
    train_idx = perm[:n_train]
    val_idx = perm[n_train:]

    return (
        dataset[train_idx],
        labels[train_idx],
        dataset[val_idx],
        labels[val_idx]
    )


def make_dataloaders(
    P: int,
    train_fraction: float = 0.75,
    batch_size: int = 64,
    seed: int | None = None,
    shuffle_train: bool = True,
    drop_last: bool = True,
) -> tuple[DataLoader, DataLoader]:
    """
    Create train and validation DataLoaders for modular addition.

    This is the recommended way to prepare data for model.fit().

    Args:
        P: Input dimension (number of classes)
        train_fraction: Fraction of data to use for training
        batch_size: Batch size for both loaders
        seed: Random seed for reproducibility
        shuffle_train: Whether to shuffle training data each epoch
        drop_last: Whether to drop the last incomplete batch in training

    Returns:
        Tuple of (train_loader, val_loader)

    Example:
        >>> train_loader, val_loader = make_dataloaders(P=64)
        >>> model = init_model(p=64, d_hidden=32)
        >>> optimizer = torch.optim.AdamW(model.parameters(), lr=3e-3)
        >>> history = model.fit(train_loader, val_loader, optimizer)
    """
    train_data, train_labels, val_data, val_labels = create_train_val_split(
        P, train_fraction=train_fraction, seed=seed
    )

    train_ds = TensorDataset(train_data, train_labels)
    val_ds = TensorDataset(val_data, val_labels)

    train_loader = DataLoader(
        train_ds,
        batch_size=batch_size,
        shuffle=shuffle_train,
        drop_last=drop_last
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=batch_size,
        shuffle=False
    )

    return train_loader, val_loader


def compute_accuracy(
    model: nn.Module,
    data: torch.Tensor,
    labels: torch.Tensor
) -> float:
    """
    Compute classification accuracy.

    Args:
        model: Model to evaluate
        data: Input data tensor
        labels: Ground truth labels

    Returns:
        Accuracy as a float in [0, 1]
    """
    model.eval()
    with torch.no_grad():
        logits = model(data)
        predictions = logits.argmax(dim=-1)
        correct = (predictions == labels).sum().item()
    return correct / len(labels)


def train_step(
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    data: torch.Tensor,
    labels: torch.Tensor,
    loss_fn: Callable | None = None
) -> float:
    """
    Perform a single training step.

    Args:
        model: Model to train
        optimizer: Optimizer
        data: Input data batch
        labels: Target labels
        loss_fn: Loss function (default: CrossEntropyLoss)

    Returns:
        Loss value for this step
    """
    if loss_fn is None:
        loss_fn = nn.CrossEntropyLoss()

    model.train()
    optimizer.zero_grad()
    logits = model(data)
    loss = loss_fn(logits, labels)
    loss.backward()
    optimizer.step()
    return loss.item()


def train_model(
    model: nn.Module,
    train_data: torch.Tensor,
    train_labels: torch.Tensor,
    val_data: torch.Tensor | None = None,
    val_labels: torch.Tensor | None = None,
    epochs: int = 1000,
    lr: float = 1e-2,
    weight_decay: float = 1e-4,
    verbose: bool = True,
    log_every: int = 100
) -> dict:
    """
    Train a model on modular addition.

    Args:
        model: Model to train
        train_data: Training inputs
        train_labels: Training labels
        val_data: Validation inputs (optional)
        val_labels: Validation labels (optional)
        epochs: Number of training epochs
        lr: Learning rate
        weight_decay: Weight decay for regularization
        verbose: Whether to print progress
        log_every: Print progress every N epochs

    Returns:
        Dictionary with training history:
        - 'train_loss': list of training losses
        - 'train_acc': list of training accuracies
        - 'val_acc': list of validation accuracies (if val data provided)
    """
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    loss_fn = nn.CrossEntropyLoss()

    history = {
        'train_loss': [],
        'train_acc': [],
        'val_acc': []
    }

    for epoch in range(epochs):
        # Training step
        loss = train_step(model, optimizer, train_data, train_labels, loss_fn)
        train_acc = compute_accuracy(model, train_data, train_labels)

        history['train_loss'].append(loss)
        history['train_acc'].append(train_acc)

        # Validation
        if val_data is not None and val_labels is not None:
            val_acc = compute_accuracy(model, val_data, val_labels)
            history['val_acc'].append(val_acc)
        else:
            val_acc = None

        # Logging
        if verbose and (epoch + 1) % log_every == 0:
            msg = f"Epoch {epoch + 1}/{epochs}: loss={loss:.4f}, train_acc={train_acc:.4f}"
            if val_acc is not None:
                msg += f", val_acc={val_acc:.4f}"
            print(msg)

    return history
