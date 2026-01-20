"""
Model definitions for compression analysis.

This module re-exports model definitions from the core module
for backwards compatibility with existing code.
"""
from ..core.models import (
    Bilinear,
    ModelConfig,
    Model,
    init_model,
    get_device,
    load_sweep_results,
    load_model_for_dim,
)

__all__ = [
    'Bilinear',
    'ModelConfig',
    'Model',
    'init_model',
    'get_device',
    'load_sweep_results',
    'load_model_for_dim',
]
