"""
Shared model definitions and loading utilities for compression analysis.
"""
import torch
from torch import nn
from dataclasses import dataclass
import pickle
from pathlib import Path


class Bilinear(nn.Linear):
    """Bilinear layer: splits linear output into left/right and multiplies element-wise."""

    def __init__(self, d_in: int, d_out: int, bias=False) -> None:
        super().__init__(d_in, 2 * d_out, bias=bias)

    def forward(self, x):
        left, right = super().forward(x).chunk(2, dim=-1)
        return left * right

    @property
    def w_l(self):
        return self.weight.chunk(2, dim=0)[0]

    @property
    def w_r(self):
        return self.weight.chunk(2, dim=0)[1]


@dataclass
class ModelConfig:
    """Configuration for compression model."""
    p: int = 64
    d_hidden: int | None = None
    bias: bool = False


class Model(nn.Module):
    """Two-layer bilinear model for modular addition."""

    def __init__(self, cfg: ModelConfig):
        super().__init__()
        self.cfg = cfg
        self.bi_linear = Bilinear(d_in=2*cfg.p, d_out=cfg.d_hidden, bias=cfg.bias)
        self.projection = nn.Linear(cfg.d_hidden, cfg.p, bias=cfg.bias)

    def forward(self, x):
        return self.projection(self.bi_linear(x))

    @property
    def w_l(self):
        return self.bi_linear.w_l

    @property
    def w_r(self):
        return self.bi_linear.w_r

    @property
    def w_p(self):
        return self.projection.weight


def init_model(p: int, d_hidden: int, bias: bool = False) -> Model:
    """Initialize a model with given input dimension p and hidden dimension."""
    cfg = ModelConfig(p=p, d_hidden=d_hidden, bias=bias)
    return Model(cfg)


def load_sweep_results(path: str | Path) -> tuple[dict, dict, int]:
    """
    Load sweep results from pickle file.

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

    # Determine P from the filename or data
    if 'sweep_results_0401.pkl' in str(path):
        P = 64
    else:
        P = data.get('P', 64)

    return models_state, val_acc, P


def load_model_for_dim(d: int, models_state: dict, P: int, device: torch.device | None = None) -> Model:
    """
    Instantiate and load weights for a model with given hidden dimension.

    Args:
        d: Hidden dimension (bottleneck size)
        models_state: Dict mapping d_hidden -> model state dict
        P: Input dimension
        device: Device to load model to (default: CPU)

    Returns:
        Loaded model
    """
    if device is None:
        device = torch.device('cpu')

    model = init_model(P, d)
    model.load_state_dict(models_state[d])
    model = model.to(device)
    return model


def get_device() -> torch.device:
    """Get the best available device (CUDA if available, else CPU)."""
    return torch.device('cuda' if torch.cuda.is_available() else 'cpu')
