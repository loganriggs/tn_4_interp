"""
Model definitions and loading utilities for modular addition experiments.

This module provides:
- Bilinear layer implementation
- Model architecture (bilinear + projection head)
- Model configuration and initialization
- Utilities for loading trained models from sweep results
"""
import torch
from torch import nn
from torch.utils.data import DataLoader
from dataclasses import dataclass
import pickle
from pathlib import Path
from typing import Callable
from tqdm import tqdm


class Bilinear(nn.Linear):
    """
    Bilinear layer: splits linear output into left/right halves and multiplies element-wise.

    For input x of dimension d_in, produces output of dimension d_out by:
    1. Computing linear transformation to 2*d_out dimensions
    2. Splitting result into left and right halves
    3. Element-wise multiplication: output = left * right

    This implements a rank-constrained bilinear form.
    """

    def __init__(self, d_in: int, d_out: int, bias: bool = False) -> None:
        super().__init__(d_in, 2 * d_out, bias=bias)

    def forward(self, x):
        left, right = super().forward(x).chunk(2, dim=-1)
        return left * right

    @property
    def w_l(self) -> torch.Tensor:
        """Left weight matrix: first half of rows."""
        return self.weight.chunk(2, dim=0)[0]

    @property
    def w_r(self) -> torch.Tensor:
        """Right weight matrix: second half of rows."""
        return self.weight.chunk(2, dim=0)[1]


@dataclass
class ModelConfig:
    """Configuration for the bilinear model."""
    p: int = 64              # Input dimension (number of classes for modular addition)
    d_hidden: int | None = None  # Bottleneck/hidden dimension
    bias: bool = False       # Whether to use bias terms


class Model(nn.Module):
    """
    Two-layer bilinear model for modular addition.

    Architecture:
        input (2*P one-hot) -> Bilinear (d_hidden) -> Linear (P) -> output

    The bilinear layer implements a low-rank approximation of the full
    interaction tensor between the two input positions.
    """

    def __init__(self, cfg: ModelConfig):
        super().__init__()
        self.cfg = cfg
        self.bi_linear = Bilinear(d_in=2 * cfg.p, d_out=cfg.d_hidden, bias=cfg.bias)
        self.projection = nn.Linear(cfg.d_hidden, cfg.p, bias=cfg.bias)

    def forward(self, x):
        return self.projection(self.bi_linear(x))

    @property
    def w_l(self) -> torch.Tensor:
        """Left weight matrix from bilinear layer: shape (d_hidden, 2*P)."""
        return self.bi_linear.w_l

    @property
    def w_r(self) -> torch.Tensor:
        """Right weight matrix from bilinear layer: shape (d_hidden, 2*P)."""
        return self.bi_linear.w_r

    @property
    def w_p(self) -> torch.Tensor:
        """Projection weight matrix: shape (P, d_hidden)."""
        return self.projection.weight

    def fit(
        self,
        train_loader: DataLoader,
        val_loader: DataLoader,
        optimizer: torch.optim.Optimizer,
        epochs: int = 500,
        loss_fn: Callable | None = None,
        device: torch.device | None = None,
        early_stop_acc: float | None = 1.0,
        verbose: bool = True,
        log_every: int = 50,
    ) -> dict:
        """
        Train the model using provided data loaders and optimizer.

        Args:
            train_loader: DataLoader for training data
            val_loader: DataLoader for validation data
            optimizer: Pre-initialized optimizer (e.g., torch.optim.AdamW)
            epochs: Maximum number of training epochs
            loss_fn: Loss function (default: CrossEntropyLoss)
            device: Device to train on (default: auto-detect)
            early_stop_acc: Stop training if validation accuracy reaches this value.
                            Set to None to disable early stopping.
            verbose: Whether to print progress
            log_every: Print progress every N epochs (if verbose)

        Returns:
            Dictionary with training history:
            - 'train_loss': list of average training losses per epoch
            - 'val_acc': list of validation accuracies per epoch
            - 'stopped_epoch': epoch at which training stopped
        """
        if device is None:
            device = next(self.parameters()).device

        if loss_fn is None:
            loss_fn = nn.CrossEntropyLoss()

        self.to(device)

        history = {
            'train_loss': [],
            'val_acc': [],
            'stopped_epoch': epochs
        }

        for epoch in tqdm(range(epochs), desc='Training', disable=not verbose):
            # Training phase
            self.train()
            epoch_losses = []

            for xb, yb in train_loader:
                xb = xb.to(device)
                yb = yb.to(device)

                logits = self(xb)
                loss = loss_fn(logits, yb)

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                epoch_losses.append(loss.item())

            avg_loss = sum(epoch_losses) / len(epoch_losses) if epoch_losses else 0.0
            history['train_loss'].append(avg_loss)

            # Validation phase
            self.eval()
            correct = 0
            total = 0

            with torch.no_grad():
                for xb, yb in val_loader:
                    xb = xb.to(device)
                    yb = yb.to(device)

                    logits = self(xb)
                    preds = logits.argmax(dim=-1)
                    correct += (preds == yb).sum().item()
                    total += yb.numel()

            val_acc = correct / total if total > 0 else 0.0
            history['val_acc'].append(val_acc)

            # Logging
            if verbose and (epoch + 1) % log_every == 0:
                print(f"Epoch {epoch + 1}/{epochs}: loss={avg_loss:.4f}, val_acc={val_acc:.4f}")

            # Early stopping on perfect accuracy
            if early_stop_acc is not None and val_acc >= early_stop_acc:
                if verbose:
                    print(f"Early stopping at epoch {epoch + 1} (val_acc={val_acc:.4f})")
                history['stopped_epoch'] = epoch + 1
                break

        return history


def init_model(p: int, d_hidden: int, bias: bool = False) -> Model:
    """
    Initialize a model with given input dimension and hidden dimension.

    Args:
        p: Input dimension (number of classes)
        d_hidden: Hidden/bottleneck dimension
        bias: Whether to use bias terms

    Returns:
        Initialized Model instance
    """
    cfg = ModelConfig(p=p, d_hidden=d_hidden, bias=bias)
    return Model(cfg)


def get_device() -> torch.device:
    """Get the best available device (CUDA if available, else CPU)."""
    return torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def load_sweep_results(path: str | Path) -> tuple[dict, dict, int]:
    """
    Load sweep results from pickle file.

    Expected format: {'models': {d_hidden: state_dict, ...},
                      'val_accs': {d_hidden: accuracy, ...},
                      'P': int}

    Args:
        path: Path to the pickle file

    Returns:
        Tuple of (models_state, val_acc, P) where:
        - models_state: dict mapping d_hidden -> model state dict
        - val_acc: dict mapping d_hidden -> validation accuracy
        - P: the input dimension (number of classes)
    """
    path = Path(path)
    with open(path, 'rb') as f:
        data = pickle.load(f)

    models_state = data.get('models', data.get('models_state', {}))
    val_acc = data.get('val_accs', {})
    P = data.get('P', 64)

    return models_state, val_acc, P


def load_model_for_dim(
    d: int,
    models_state: dict,
    P: int,
    device: torch.device | None = None
) -> Model:
    """
    Instantiate and load weights for a model with given hidden dimension.

    Args:
        d: Hidden dimension (bottleneck size)
        models_state: Dict mapping d_hidden -> model state dict
        P: Input dimension
        device: Device to load model to (default: CPU)

    Returns:
        Loaded model in eval mode
    """
    if device is None:
        device = torch.device('cpu')

    model = init_model(P, d)
    model.load_state_dict(models_state[d])
    model = model.to(device)
    model.eval()
    return model
