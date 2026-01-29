# %% [markdown]
# # 2-Layer Residual Bilinear SVHN Interpretability
#
# Interpretability analysis for residual bilinear networks with down-projection.
#
# Architecture:
# ```
# r0 = W_embed @ x
# r1 = r0 + D1 @ (L1 @ r0 ⊙ R1 @ r0)
# r2 = r1 + D2 @ (L2 @ r1 ⊙ R2 @ r1)
# y = W_unembed @ r2
# ```
#
# Each bilinear layer: output = D @ (Lx ⊙ Rx)
# - L, R: d_res -> rank (128 -> 64)
# - D: rank -> d_res (64 -> 128)

# %%
import pickle
from pathlib import Path
import sys
from dataclasses import dataclass
from typing import Optional, Dict, List, Tuple, Any

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import matplotlib.pyplot as plt
from tqdm import tqdm


# =============================================================================
# CONFIGURATION
# =============================================================================

@dataclass
class AnalysisConfig:
    """Configuration for interpretability analysis."""
    # Paths
    notebook_dir: Path = Path("/workspace/tn_4_interp/modular_addition/dev_interp/notebooks")
    checkpoint_dir: Optional[Path] = None
    results_dir: Optional[Path] = None
    data_dir: Optional[Path] = None

    # Model settings
    use_rmsnorm: bool = False

    # Analysis settings
    cossim_threshold: float = 0.7
    min_group_size: int = 2
    layer: int = 1

    def __post_init__(self):
        if self.checkpoint_dir is None:
            self.checkpoint_dir = self.notebook_dir / "tn_analysis_checkpoints"
        if self.results_dir is None:
            self.results_dir = self.notebook_dir / "svhn_2layer_results"
        if self.data_dir is None:
            self.data_dir = self.notebook_dir / "data"
        self.results_dir.mkdir(exist_ok=True)


# =============================================================================
# MODEL LOADING
# =============================================================================

def get_device() -> torch.device:
    """Get the best available device."""
    if torch.cuda.is_available():
        return torch.device('cuda')
    elif torch.backends.mps.is_available():
        return torch.device('mps')
    return torch.device('cpu')


def load_model(config: AnalysisConfig):
    """Load the trained residual bilinear model with down-projection."""
    # Import model definitions
    sys.path.insert(0, str(config.notebook_dir))
    from train_residual_downproj import (
        ResidualBilinearDownProj,
        ResidualBilinearDownProjRMSNorm,
    )

    checkpoint_path = config.checkpoint_dir / "residual_downproj_comparison.pkl"
    with open(checkpoint_path, 'rb') as f:
        data = pickle.load(f)

    model_key = 'with_rmsnorm' if config.use_rmsnorm else 'no_rmsnorm'
    model_data = data[model_key]

    model_config = model_data['config']
    checkpoints = model_data['checkpoints']
    history = model_data['history']

    final_epoch = max(checkpoints.keys())
    final_state = checkpoints[final_epoch]

    model_config.setdefault('input_dim', 3072)
    model_config.setdefault('output_dim', 10)

    print(f"Loading model: {model_key} (epoch {final_epoch})")
    print(f"Config: d_res={model_config['d_res']}, rank={model_config['rank']}")

    if config.use_rmsnorm:
        model = ResidualBilinearDownProjRMSNorm(
            input_dim=model_config['input_dim'],
            d_res=model_config['d_res'],
            output_dim=model_config['output_dim'],
            rank=model_config['rank'],
        )
    else:
        model = ResidualBilinearDownProj(
            input_dim=model_config['input_dim'],
            d_res=model_config['d_res'],
            output_dim=model_config['output_dim'],
            rank=model_config['rank'],
        )
    model.load_state_dict(final_state)
    model.eval()

    return model, model_config, history


def get_svhn_test_loader(config: AnalysisConfig, batch_size: int = 128):
    """Load SVHN test set."""
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4377, 0.4438, 0.4728), (0.1980, 0.2010, 0.1970))
    ])
    test_dataset = datasets.SVHN(
        str(config.data_dir), split='test', download=True, transform=transform
    )
    test_loader = DataLoader(
        test_dataset, batch_size=batch_size, shuffle=False, num_workers=2
    )
    return test_loader, test_dataset


# =============================================================================
# WEIGHT EXTRACTION UTILITIES
# =============================================================================

def get_layer_weights(model, layer: int = 1) -> Dict[str, torch.Tensor]:
    """Extract weights from a bilinear layer."""
    if layer == 1:
        bilinear = model.bilinear1
    else:
        bilinear = model.bilinear2

    return {
        'L': bilinear.L.weight.detach().cpu(),  # (rank, d_res)
        'R': bilinear.R.weight.detach().cpu(),  # (rank, d_res)
        'D': bilinear.D.weight.detach().cpu(),  # (d_res, rank)
    }


def get_all_weights(model) -> Dict[str, torch.Tensor]:
    """Extract all model weights."""
    weights = {
        'W_embed': model.embed.weight.detach().cpu(),      # (d_res, input_dim)
        'W_unembed': model.unembed.weight.detach().cpu(),  # (output_dim, d_res)
    }
    weights.update({f'L1': model.bilinear1.L.weight.detach().cpu()})
    weights.update({f'R1': model.bilinear1.R.weight.detach().cpu()})
    weights.update({f'D1': model.bilinear1.D.weight.detach().cpu()})
    weights.update({f'L2': model.bilinear2.L.weight.detach().cpu()})
    weights.update({f'R2': model.bilinear2.R.weight.detach().cpu()})
    weights.update({f'D2': model.bilinear2.D.weight.detach().cpu()})
    return weights


# =============================================================================
# BILINEAR CHANNEL ANALYSIS
# =============================================================================

def analyze_bilinear_activations(
    model,
    r0: torch.Tensor,
    device: torch.device,
    correct_class: Optional[int] = None
) -> Dict[str, torch.Tensor]:
    """
    Analyze activations of bilinear channels for a given residual stream state.

    Args:
        model: The trained model
        r0: Residual stream state after embedding (d_res,)
        device: Computation device
        correct_class: If provided, compute contribution to this class

    Returns:
        Dictionary with:
        - rank_activations: Raw activation of each channel
        - weighted_activations: Activation × ||D[:,i]|| (impact on residual)
        - channel_contributions: (10, rank) contribution to each class
        - contrib_to_correct: Contribution to correct class
        - total_contrib_magnitude: Sum of |contribution| across classes
    """
    model.eval()

    L = model.bilinear1.L.weight.detach()
    R = model.bilinear1.R.weight.detach()
    D = model.bilinear1.D.weight.detach()
    W_unembed = model.unembed.weight.detach()

    r0_dev = r0.to(device)

    # Compute rank activations: (L @ r0) * (R @ r0)
    Lr0 = L @ r0_dev
    Rr0 = R @ r0_dev
    rank_activations = Lr0 * Rr0

    # Weighted activation: impact on residual stream
    D_col_norms = D.norm(dim=0)
    weighted_activations = rank_activations.abs() * D_col_norms

    # Bilinear output
    b1 = D @ rank_activations

    # Channel contributions to each class
    WD = W_unembed @ D  # (10, rank)
    channel_contributions = WD * rank_activations.unsqueeze(0)  # (10, rank)

    # Contribution to correct class
    contrib_to_correct = None
    if correct_class is not None:
        contrib_to_correct = channel_contributions[correct_class]

    # Total contribution magnitude
    total_contrib_magnitude = channel_contributions.abs().sum(dim=0)

    return {
        'rank_activations': rank_activations.cpu(),
        'weighted_activations': weighted_activations.cpu(),
        'D_col_norms': D_col_norms.cpu(),
        'channel_contributions': channel_contributions.cpu(),
        'contrib_to_correct': contrib_to_correct.cpu() if contrib_to_correct is not None else None,
        'total_contrib_magnitude': total_contrib_magnitude.cpu(),
        'b1': b1.cpu(),
        'WD': WD.cpu(),
    }


# =============================================================================
# OPTIMAL INPUT COMPUTATION
# =============================================================================

def compute_channel_optimal_input(
    model,
    channel_idx: int,
    layer: int = 1
) -> Dict[str, torch.Tensor]:
    """
    Compute optimal input for a single bilinear channel.

    For channel i: h_i = (L[i] @ x) * (R[i] @ x) = x^T S_i x
    where S_i = (1/2)(L[i] @ R[i]^T + R[i] @ L[i]^T)

    Optimal input is top eigenvector of S_i, projected to image space.

    Args:
        model: The trained model
        channel_idx: Index of the bilinear channel
        layer: Which bilinear layer (1 or 2)

    Returns:
        Dictionary with optimal images and eigenvalues
    """
    weights = get_layer_weights(model, layer)
    L, R = weights['L'], weights['R']
    W_embed = model.embed.weight.detach().cpu()

    Li = L[channel_idx]
    Ri = R[channel_idx]

    # Symmetric matrix for this channel
    S_i = 0.5 * (torch.outer(Li, Ri) + torch.outer(Ri, Li))

    # Eigendecomposition (ascending order, flip to descending)
    eigenvalues, eigenvectors = torch.linalg.eigh(S_i)
    eigenvalues = eigenvalues.flip(0)
    eigenvectors = eigenvectors.flip(1)

    x_max = eigenvectors[:, 0]
    x_min = eigenvectors[:, -1]

    # Project to image space
    W_embed_pinv = torch.linalg.pinv(W_embed)
    img_max = W_embed_pinv @ x_max
    img_min = W_embed_pinv @ x_min

    # Analytical solution: L_hat + R_hat
    L_hat = Li / (Li.norm() + 1e-8)
    R_hat = Ri / (Ri.norm() + 1e-8)
    x_analytical = L_hat + R_hat
    x_analytical = x_analytical / (x_analytical.norm() + 1e-8)
    img_analytical = W_embed_pinv @ x_analytical

    # Effective weights on pixels
    l_eff = Li @ W_embed
    r_eff = Ri @ W_embed

    return {
        'channel_idx': channel_idx,
        'S_i': S_i,
        'eigenvalues': eigenvalues,
        'x_max': x_max,
        'x_min': x_min,
        'img_max': img_max,
        'img_min': img_min,
        'img_analytical': img_analytical,
        'lambda_max': eigenvalues[0],
        'lambda_min': eigenvalues[-1],
        'l_eff': l_eff,
        'r_eff': r_eff,
    }


def compute_channel_pixel_contributions(
    model,
    channel_idx: int,
    input_image: torch.Tensor,
    layer: int = 1
) -> Dict[str, Any]:
    """
    Compute which pixels in the input contributed most to a channel's activation.

    For channel i: h_i = (L[i] @ x) * (R[i] @ x)
    where x = W_embed @ img

    The gradient ∂h_i/∂img = (R[i] @ x) * l_eff + (L[i] @ x) * r_eff

    Args:
        model: The trained model
        channel_idx: Index of the bilinear channel
        input_image: Input image tensor (C, H, W)
        layer: Which bilinear layer

    Returns:
        Dictionary with gradient and contribution maps
    """
    weights = get_layer_weights(model, layer)
    L, R = weights['L'], weights['R']
    W_embed = model.embed.weight.detach().cpu()

    Li = L[channel_idx]
    Ri = R[channel_idx]

    img_flat = input_image.flatten()
    x = W_embed @ img_flat

    Lx = (Li @ x).item()
    Rx = (Ri @ x).item()
    activation = Lx * Rx

    l_eff = Li @ W_embed
    r_eff = Ri @ W_embed

    # Gradient: ∂h_i/∂img
    gradient = Rx * l_eff + Lx * r_eff

    # Attribution: input × gradient
    input_x_grad = gradient * img_flat

    # Element-wise contributions to L and R projections
    contrib_L = l_eff * img_flat
    contrib_R = r_eff * img_flat

    return {
        'channel_idx': channel_idx,
        'activation': activation,
        'Lx': Lx,
        'Rx': Rx,
        'gradient': gradient,
        'input_x_grad': input_x_grad,
        'contrib_L': contrib_L,
        'contrib_R': contrib_R,
        'l_eff': l_eff,
        'r_eff': r_eff,
    }


def compute_combined_optimal_input(
    model,
    channel_indices: List[int],
    weights: Optional[torch.Tensor] = None,
    layer: int = 1
) -> Dict[str, torch.Tensor]:
    """
    Compute combined optimal input for a group of channels.

    S_combined = (1/2) * sum_i w_i * (L[i] @ R[i].T + R[i] @ L[i].T)

    Args:
        model: The trained model
        channel_indices: List of channel indices to combine
        weights: Optional weights for each channel (default: uniform)
        layer: Which bilinear layer

    Returns:
        Dictionary with combined optimal images
    """
    layer_weights = get_layer_weights(model, layer)
    L, R = layer_weights['L'], layer_weights['R']
    W_embed = model.embed.weight.detach().cpu()
    d_res = L.shape[1]

    if weights is None:
        weights = torch.ones(len(channel_indices))
    weights = weights / weights.sum()

    S_combined = torch.zeros(d_res, d_res)
    for idx, ch_idx in enumerate(channel_indices):
        Li = L[ch_idx]
        Ri = R[ch_idx]
        S_i = 0.5 * (torch.outer(Li, Ri) + torch.outer(Ri, Li))
        S_combined += weights[idx] * S_i

    eigenvalues, eigenvectors = torch.linalg.eigh(S_combined)
    eigenvalues = eigenvalues.flip(0)
    eigenvectors = eigenvectors.flip(1)

    x_max = eigenvectors[:, 0]
    x_min = eigenvectors[:, -1]

    W_embed_pinv = torch.linalg.pinv(W_embed)
    img_max = W_embed_pinv @ x_max
    img_min = W_embed_pinv @ x_min

    return {
        'channel_indices': channel_indices,
        'weights': weights,
        'S_combined': S_combined,
        'eigenvalues': eigenvalues,
        'x_max': x_max,
        'x_min': x_min,
        'img_max': img_max,
        'img_min': img_min,
        'lambda_max': eigenvalues[0],
        'lambda_min': eigenvalues[-1],
    }


def compute_class_optimal_input(
    model,
    class_idx: int,
    layer: int = 1
) -> Dict[str, torch.Tensor]:
    """
    Compute optimal input that maximizes a specific class through a bilinear layer.

    For class p:
        w_p = W_unembed[p, :] @ D_layer  (how class p reads from rank dims)
        S_p = (1/2) * sum_i w_p[i] * (L[i] @ R[i].T + R[i] @ L[i].T)

    Args:
        model: The trained model
        class_idx: Target class index
        layer: Which bilinear layer

    Returns:
        Dictionary with class-specific optimal images
    """
    layer_weights = get_layer_weights(model, layer)
    L, R, D = layer_weights['L'], layer_weights['R'], layer_weights['D']
    W_unembed = model.unembed.weight.detach().cpu()
    W_embed = model.embed.weight.detach().cpu()

    # How class reads from each rank dimension
    w_p = W_unembed[class_idx] @ D

    d_res = L.shape[1]
    rank = L.shape[0]

    S_p = torch.zeros(d_res, d_res)
    for i in range(rank):
        Li = L[i]
        Ri = R[i]
        S_i = 0.5 * (torch.outer(Li, Ri) + torch.outer(Ri, Li))
        S_p += w_p[i] * S_i

    eigenvalues, eigenvectors = torch.linalg.eigh(S_p)
    eigenvalues = eigenvalues.flip(0)
    eigenvectors = eigenvectors.flip(1)

    x_optimal = eigenvectors[:, 0]
    x_minimal = eigenvectors[:, -1]

    W_embed_pinv = torch.linalg.pinv(W_embed)
    img_optimal = W_embed_pinv @ x_optimal
    img_minimal = W_embed_pinv @ x_minimal

    return {
        'class_idx': class_idx,
        'class_weights': w_p,
        'S_p': S_p,
        'eigenvalues': eigenvalues,
        'x_optimal': x_optimal,
        'x_minimal': x_minimal,
        'img_optimal': img_optimal,
        'img_minimal': img_minimal,
        'lambda_max': eigenvalues[0],
        'lambda_min': eigenvalues[-1],
    }


# =============================================================================
# COMPUTATIONAL PATH ANALYSIS
# =============================================================================

def analyze_computational_paths(
    model,
    dataloader,
    device: torch.device,
    n_samples: int = 10000
) -> Dict[str, Any]:
    """
    Analyze when the correct answer emerges through the residual stream.

    Three paths:
    1. embed → unembed: W_unembed @ r0 (linear only)
    2. after layer 1: W_unembed @ r1 (after first residual block)
    3. full model: W_unembed @ r2 (after second residual block)
    """
    model.eval()
    D = model.unembed.weight.detach()

    results = {
        'embed_unembed': 0,
        'after_layer1': 0,
        'full_model': 0,
        'total': 0,
        'credit_r0': 0,
        'credit_layer1': 0,
        'credit_layer2': 0,
    }

    for data, target in tqdm(dataloader, desc="Analyzing paths"):
        if results['total'] >= n_samples:
            break

        data, target = data.to(device), target.to(device)

        with torch.no_grad():
            intermediates = model.forward_with_intermediates(data)

            r0 = intermediates['r0']
            r1 = intermediates['r1']
            r2 = intermediates['r2']

            logits_r0 = r0 @ D.T
            logits_r1 = r1 @ D.T
            logits_r2 = r2 @ D.T

            correct_r0 = (logits_r0.argmax(dim=1) == target)
            correct_r1 = (logits_r1.argmax(dim=1) == target)
            correct_r2 = (logits_r2.argmax(dim=1) == target)

            results['embed_unembed'] += correct_r0.sum().item()
            results['after_layer1'] += correct_r1.sum().item()
            results['full_model'] += correct_r2.sum().item()
            results['total'] += target.size(0)

            results['credit_r0'] += correct_r0.sum().item()
            results['credit_layer1'] += ((~correct_r0) & correct_r1).sum().item()
            results['credit_layer2'] += ((~correct_r0) & (~correct_r1) & correct_r2).sum().item()

    # Convert to percentages
    n = results['total']
    results['embed_unembed_pct'] = 100 * results['embed_unembed'] / n
    results['after_layer1_pct'] = 100 * results['after_layer1'] / n
    results['full_model_pct'] = 100 * results['full_model'] / n

    total_correct = results['full_model']
    if total_correct > 0:
        results['credit_r0_pct'] = 100 * results['credit_r0'] / total_correct
        results['credit_layer1_pct'] = 100 * results['credit_layer1'] / total_correct
        results['credit_layer2_pct'] = 100 * results['credit_layer2'] / total_correct

    return results


def find_layer1_fixes_examples(
    model,
    dataloader,
    device: torch.device,
    n_to_find: int = 5
) -> List[Dict[str, Any]]:
    """
    Find examples where embedding gets it wrong but layer 1 fixes it.

    Returns list of example dictionaries with image, labels, predictions,
    and intermediate activations.
    """
    model.eval()
    D = model.unembed.weight.detach()

    examples = []

    for data, target in dataloader:
        if len(examples) >= n_to_find:
            break

        data, target = data.to(device), target.to(device)

        with torch.no_grad():
            inter = model.forward_with_intermediates(data)

            logits_r0 = inter['r0'] @ D.T
            logits_r1 = inter['r1'] @ D.T
            logits_r2 = inter['r2'] @ D.T

            pred_r0 = logits_r0.argmax(dim=1)
            pred_r1 = logits_r1.argmax(dim=1)
            pred_r2 = logits_r2.argmax(dim=1)

            # Find examples where r0 wrong, r1 correct
            mask = (pred_r0 != target) & (pred_r1 == target)

            for i in torch.where(mask)[0]:
                if len(examples) >= n_to_find:
                    break

                idx = i.item()
                examples.append({
                    'image': data[idx].cpu(),
                    'label': target[idx].item(),
                    'pred_r0': pred_r0[idx].item(),
                    'pred_r1': pred_r1[idx].item(),
                    'pred_r2': pred_r2[idx].item(),
                    'logits_r0': logits_r0[idx].cpu(),
                    'logits_r1': logits_r1[idx].cpu(),
                    'logits_r2': logits_r2[idx].cpu(),
                    'r0': inter['r0'][idx].cpu(),
                    'r1': inter['r1'][idx].cpu(),
                    'b1': inter['b1'][idx].cpu(),
                })

    return examples


# =============================================================================
# GRAM MATRIX ANALYSIS
# =============================================================================

def compute_gram_matrices(model) -> Dict[str, torch.Tensor]:
    """
    Compute gram matrices for analyzing weight structure.

    Returns cosine similarity matrices for D1, D2 columns,
    and class/rank interaction matrices.
    """
    weights = get_all_weights(model)
    D1 = weights['D1']
    D2 = weights['D2']
    W_unembed = weights['W_unembed']

    results = {}

    # Cosine similarity between rank dimensions
    D1_normalized = D1 / (D1.norm(dim=0, keepdim=True) + 1e-8)
    D2_normalized = D2 / (D2.norm(dim=0, keepdim=True) + 1e-8)
    results['D1_gram'] = D1_normalized.T @ D1_normalized
    results['D2_gram'] = D2_normalized.T @ D2_normalized

    # Compose D with W_unembed
    WD1 = W_unembed @ D1
    WD2 = W_unembed @ D2
    results['WD1_class_gram'] = WD1 @ WD1.T
    results['WD2_class_gram'] = WD2 @ WD2.T
    results['WD1_rank_gram'] = WD1.T @ WD1
    results['WD2_rank_gram'] = WD2.T @ WD2

    # Unembed gram
    results['unembed_gram'] = W_unembed.T @ W_unembed

    results['D1'] = D1
    results['D2'] = D2
    results['W_unembed'] = W_unembed
    results['WD1'] = WD1
    results['WD2'] = WD2

    return results


def find_similar_channel_groups(
    model,
    layer: int = 1,
    threshold: float = 0.7,
    min_size: int = 2
) -> Tuple[List[Tuple[List[int], float]], torch.Tensor]:
    """
    Find groups of channels with similar D projections (|cos-sim| > threshold).

    Returns list of (group_indices, mean_cossim) tuples and the cossim matrix.
    """
    weights = get_layer_weights(model, layer)
    D = weights['D']

    D_normalized = D / (D.norm(dim=0, keepdim=True) + 1e-8)
    cossim = D_normalized.T @ D_normalized

    rank = D.shape[1]
    used = set()
    groups = []

    abs_cossim = cossim.abs()

    for i in range(rank):
        if i in used:
            continue

        similar = torch.where(abs_cossim[i] > threshold)[0].tolist()
        similar = [j for j in similar if j not in used]

        if len(similar) >= min_size:
            group_cossim = abs_cossim[similar][:, similar]
            mask = torch.triu(torch.ones_like(group_cossim), diagonal=1) > 0
            mean_sim = group_cossim[mask].mean().item()

            groups.append((similar, mean_sim))
            used.update(similar)

    groups.sort(key=lambda x: -len(x[0]))

    return groups, cossim


# =============================================================================
# VISUALIZATION UTILITIES
# =============================================================================

def tensor_to_image(tensor: torch.Tensor, shape: Tuple[int, ...] = (3, 32, 32)) -> np.ndarray:
    """Convert flattened tensor to displayable image (mean across channels)."""
    img = tensor.numpy().reshape(shape).mean(axis=0)
    return img


def plot_heatmap(
    ax,
    data: np.ndarray,
    title: str = "",
    cmap: str = 'RdBu_r',
    symmetric: bool = True
):
    """Plot a heatmap with colorbar."""
    if symmetric:
        vmax = np.abs(data).max()
        vmin = -vmax
    else:
        vmin, vmax = data.min(), data.max()

    im = ax.imshow(data, cmap=cmap, vmin=vmin, vmax=vmax)
    ax.set_title(title, fontsize=10)
    ax.axis('off')
    plt.colorbar(im, ax=ax, fraction=0.046)
    return im


def denormalize_svhn_image(img: torch.Tensor) -> np.ndarray:
    """Convert normalized SVHN image back to displayable format."""
    img_np = img.numpy().transpose(1, 2, 0)
    img_np = img_np * np.array([0.1980, 0.2010, 0.1970]) + np.array([0.4377, 0.4438, 0.4728])
    return np.clip(img_np, 0, 1)


# =============================================================================
# MAIN VISUALIZATION FUNCTIONS
# =============================================================================

def visualize_layer1_fix_example(
    example: Dict[str, Any],
    model,
    device: torch.device
) -> plt.Figure:
    """
    Create detailed visualization of how layer 1 fixes an embedding mistake.
    """
    fig = plt.figure(figsize=(20, 14))

    label = example['label']
    pred_r0 = example['pred_r0']
    pred_r1 = example['pred_r1']
    pred_r2 = example['pred_r2']

    analysis = analyze_bilinear_activations(model, example['r0'], device, correct_class=label)

    x = np.arange(10)
    width = 0.25

    # Panel 1: The image
    ax1 = fig.add_subplot(3, 3, 1)
    ax1.imshow(denormalize_svhn_image(example['image']))
    ax1.set_title(f'True: {label}\nPred r0: {pred_r0} → r1: {pred_r1} → r2: {pred_r2}', fontsize=12)
    ax1.axis('off')

    # Panel 2: Logits at each stage
    ax2 = fig.add_subplot(3, 3, 2)
    logits_r0 = example['logits_r0'].numpy()
    logits_r1 = example['logits_r1'].numpy()
    logits_r2 = example['logits_r2'].numpy()

    ax2.bar(x - width, logits_r0, width, label='r0 (embed)', alpha=0.8, color='#3498db')
    ax2.bar(x, logits_r1, width, label='r1 (after L1)', alpha=0.8, color='#2ecc71')
    ax2.bar(x + width, logits_r2, width, label='r2 (final)', alpha=0.8, color='#e74c3c')
    ax2.axvline(x=label, color='black', linestyle='--', linewidth=2, label='True class')
    ax2.set_xlabel('Class')
    ax2.set_ylabel('Logit')
    ax2.set_title('Logits at Each Stage')
    ax2.set_xticks(x)
    ax2.legend(fontsize=8)

    # Panel 3: Logit decomposition
    ax3 = fig.add_subplot(3, 3, 3)
    W_unembed = model.unembed.weight.detach().cpu()
    contrib_r0 = (W_unembed @ example['r0']).numpy()
    contrib_b1 = (W_unembed @ example['b1']).numpy()

    ax3.bar(x - width/2, contrib_r0, width, label='From r0 (embed)', alpha=0.8, color='#3498db')
    ax3.bar(x + width/2, contrib_b1, width, label='From b1 (bilinear)', alpha=0.8, color='#9b59b6')
    ax3.axvline(x=label, color='black', linestyle='--', linewidth=2)
    ax3.set_xlabel('Class')
    ax3.set_ylabel('Logit contribution')
    ax3.set_title('Logit Decomposition: r0 vs b1')
    ax3.set_xticks(x)
    ax3.legend(fontsize=8)

    # Panel 4: Weighted activations
    ax4 = fig.add_subplot(3, 3, 4)
    weighted_acts = analysis['weighted_activations'].numpy()
    rank_acts = analysis['rank_activations'].numpy()
    colors = ['#2ecc71' if a > 0 else '#e74c3c' for a in rank_acts]
    ax4.bar(range(len(weighted_acts)), weighted_acts, color=colors, alpha=0.7)
    ax4.set_xlabel('Rank dimension')
    ax4.set_ylabel('|Activation| × ||D[:,i]||')
    ax4.set_title('Weighted Activations\n(impact on residual stream)')

    # Panel 5: Top channels by contribution to correct class
    ax5 = fig.add_subplot(3, 3, 5)
    contrib_correct = analysis['contrib_to_correct'].numpy()
    top_k = 10

    sorted_by_contrib = np.argsort(contrib_correct)
    top_positive = sorted_by_contrib[-top_k//2:][::-1]
    top_negative = sorted_by_contrib[:top_k//2]
    top_by_correct = np.concatenate([top_positive, top_negative])

    contrib_vals = contrib_correct[top_by_correct]
    colors = ['#2ecc71' if c > 0 else '#e74c3c' for c in contrib_vals]
    ax5.barh(range(len(top_by_correct)), contrib_vals, color=colors, alpha=0.8)
    ax5.set_yticks(range(len(top_by_correct)))
    ax5.set_yticklabels([f'Ch {i}' for i in top_by_correct])
    ax5.axvline(x=0, color='black', linestyle='-', linewidth=0.5)
    ax5.set_xlabel(f'Contribution to class {label}')
    ax5.set_title(f'Top Channels by Contribution to Correct Class ({label})')
    ax5.invert_yaxis()

    # Panel 6: Top channels by total contribution magnitude
    ax6 = fig.add_subplot(3, 3, 6)
    total_contrib = analysis['total_contrib_magnitude'].numpy()
    top_by_total = np.argsort(total_contrib)[-top_k:][::-1]

    total_vals = total_contrib[top_by_total]
    colors = ['#2ecc71' if contrib_correct[i] > 0 else '#e74c3c' for i in top_by_total]
    ax6.barh(range(top_k), total_vals, color=colors, alpha=0.8)
    ax6.set_yticks(range(top_k))
    ax6.set_yticklabels([f'Ch {i}' for i in top_by_total])
    ax6.set_xlabel('Σ|contribution| across all classes')
    ax6.set_title(f'Top {top_k} by Total Impact\n(green=helps class {label}, red=hurts)')
    ax6.invert_yaxis()

    # Panel 7: Heatmap sorted by correct class contribution
    ax7 = fig.add_subplot(3, 3, 7)
    channel_contribs = analysis['channel_contributions'].numpy()
    contrib_matrix = channel_contribs[:, top_by_correct].T

    im = ax7.imshow(contrib_matrix, cmap='RdBu_r', aspect='auto',
                    vmin=-np.abs(contrib_matrix).max(), vmax=np.abs(contrib_matrix).max())
    ax7.set_yticks(range(len(top_by_correct)))
    ax7.set_yticklabels([f'Ch {i}' for i in top_by_correct])
    ax7.set_xticks(range(10))
    ax7.set_xlabel('Class')
    ax7.set_ylabel('Channel')
    ax7.set_title(f'Contributions (sorted by class {label} impact)')
    plt.colorbar(im, ax=ax7)
    ax7.axvline(x=label - 0.5, color='black', linestyle='--', linewidth=2)
    ax7.axvline(x=label + 0.5, color='black', linestyle='--', linewidth=2)

    # Panel 8: Heatmap sorted by total impact
    ax8 = fig.add_subplot(3, 3, 8)
    contrib_matrix_total = channel_contribs[:, top_by_total].T

    im = ax8.imshow(contrib_matrix_total, cmap='RdBu_r', aspect='auto',
                    vmin=-np.abs(contrib_matrix_total).max(), vmax=np.abs(contrib_matrix_total).max())
    ax8.set_yticks(range(top_k))
    ax8.set_yticklabels([f'Ch {i}' for i in top_by_total])
    ax8.set_xticks(range(10))
    ax8.set_xlabel('Class')
    ax8.set_ylabel('Channel')
    ax8.set_title('Contributions (sorted by total impact)')
    plt.colorbar(im, ax=ax8)
    ax8.axvline(x=label - 0.5, color='black', linestyle='--', linewidth=2)
    ax8.axvline(x=label + 0.5, color='black', linestyle='--', linewidth=2)

    # Panel 9: Summary
    ax9 = fig.add_subplot(3, 3, 9)
    ax9.axis('off')

    total_b1_to_correct = contrib_b1[label]
    total_b1_to_wrong = contrib_b1[pred_r0]
    top_helper = top_by_correct[0]
    top_hurter = top_by_correct[-1]

    summary_text = f"""
    Summary:
    ────────────────────────────────
    b1 contribution to class {label} (correct): {total_b1_to_correct:+.3f}
    b1 contribution to class {pred_r0} (r0 pred): {total_b1_to_wrong:+.3f}

    Top helper: Channel {top_helper}
      → contributes {contrib_correct[top_helper]:+.3f} to class {label}

    Top hurter: Channel {top_hurter}
      → contributes {contrib_correct[top_hurter]:+.3f} to class {label}

    Total channels helping class {label}: {(contrib_correct > 0).sum()}
    Total channels hurting class {label}: {(contrib_correct < 0).sum()}
    """
    ax9.text(0.1, 0.5, summary_text, transform=ax9.transAxes, fontsize=11,
             verticalalignment='center', fontfamily='monospace',
             bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

    plt.suptitle(f'Layer 1 Fix Example: True={label}, r0→{pred_r0}, r1→{pred_r1}', fontsize=14)
    plt.tight_layout()

    return fig


def visualize_channel_interpretation(
    example: Dict[str, Any],
    model,
    device: torch.device,
    n_channels: int = 5
) -> Tuple[plt.Figure, plt.Figure]:
    """
    Visualize optimal inputs and pixel contributions for important channels.

    Returns two figures:
    1. Optimal vs actual pixel contributions for top channels
    2. Detailed view of the top channel
    """
    label = example['label']
    input_image = example['image']

    analysis = analyze_bilinear_activations(model, example['r0'], device, correct_class=label)
    contrib_correct = analysis['channel_contributions'].numpy()[label]
    top_idx = np.argsort(contrib_correct)[-n_channels:][::-1]

    # Figure 1: All top channels
    fig1, axes1 = plt.subplots(4, n_channels, figsize=(3.5 * n_channels, 12))

    for col, ch_idx in enumerate(top_idx):
        ch_opt = compute_channel_optimal_input(model, ch_idx, layer=1)
        ch_contrib = compute_channel_pixel_contributions(model, ch_idx, input_image, layer=1)

        # Row 0: Optimal input
        ax = axes1[0, col]
        img = tensor_to_image(ch_opt['img_max'])
        plot_heatmap(ax, img, f'Ch {ch_idx}\ncontrib={contrib_correct[ch_idx]:.3f}\nact={ch_contrib["activation"]:.2f}')
        if col == 0:
            ax.set_ylabel('Optimal\ninput', fontsize=9)

        # Row 1: Input × Gradient
        ax = axes1[1, col]
        ixg = tensor_to_image(ch_contrib['input_x_grad'])
        plot_heatmap(ax, ixg)
        if col == 0:
            ax.set_ylabel('Input ×\nGradient', fontsize=9)

        # Row 2: Gradient
        ax = axes1[2, col]
        grad = tensor_to_image(ch_contrib['gradient'])
        plot_heatmap(ax, grad)
        if col == 0:
            ax.set_ylabel('Gradient\n(sensitivity)', fontsize=9)

        # Row 3: L weights
        ax = axes1[3, col]
        l_eff = tensor_to_image(ch_opt['l_eff'])
        plot_heatmap(ax, l_eff, f'Lx={ch_contrib["Lx"]:.2f}, Rx={ch_contrib["Rx"]:.2f}')
        if col == 0:
            ax.set_ylabel('L weights\non pixels', fontsize=9)

    fig1.suptitle(f'Top {n_channels} Channels for Class {label}: Optimal vs Actual Pixel Contributions', fontsize=12)
    plt.tight_layout()

    # Figure 2: Detail of top channel
    top_ch = top_idx[0]
    ch_opt = compute_channel_optimal_input(model, top_ch, layer=1)
    ch_contrib = compute_channel_pixel_contributions(model, top_ch, input_image, layer=1)

    fig2, axes2 = plt.subplots(2, 4, figsize=(16, 8))

    # Row 0: Weights
    plot_heatmap(axes2[0, 0], tensor_to_image(ch_opt['img_max']),
                 f'Optimal Max\nλ+={ch_opt["lambda_max"]:.3f}')
    plot_heatmap(axes2[0, 1], tensor_to_image(ch_opt['l_eff']),
                 f'L weights\nLx={ch_contrib["Lx"]:.3f}')
    plot_heatmap(axes2[0, 2], tensor_to_image(ch_opt['r_eff']),
                 f'R weights\nRx={ch_contrib["Rx"]:.3f}')

    l_eff = tensor_to_image(ch_opt['l_eff'])
    r_eff = tensor_to_image(ch_opt['r_eff'])
    lr_norm = l_eff / (np.abs(l_eff).max() + 1e-8) + r_eff / (np.abs(r_eff).max() + 1e-8)
    plot_heatmap(axes2[0, 3], lr_norm, 'L̂+R̂ (normalized)')

    # Row 1: Actual contributions
    axes2[1, 0].imshow(denormalize_svhn_image(input_image))
    axes2[1, 0].set_title(f'Actual Input\nTrue: {label}', fontsize=10)
    axes2[1, 0].axis('off')

    plot_heatmap(axes2[1, 1], tensor_to_image(ch_contrib['contrib_L']),
                 'Pixel contrib to L\n(l_eff × input)')
    plot_heatmap(axes2[1, 2], tensor_to_image(ch_contrib['contrib_R']),
                 'Pixel contrib to R\n(r_eff × input)')
    plot_heatmap(axes2[1, 3], tensor_to_image(ch_contrib['input_x_grad']),
                 f'Input × Gradient\nActivation={ch_contrib["activation"]:.3f}')

    fig2.suptitle(f'Channel {top_ch} Detail: h = (Lx)(Rx) = ({ch_contrib["Lx"]:.2f})({ch_contrib["Rx"]:.2f}) = {ch_contrib["activation"]:.2f}', fontsize=12)
    plt.tight_layout()

    return fig1, fig2


# =============================================================================
# MAIN ANALYSIS RUNNER
# =============================================================================

def run_full_analysis(config: Optional[AnalysisConfig] = None):
    """Run the complete interpretability analysis."""
    if config is None:
        config = AnalysisConfig()

    device = get_device()
    print(f"Device: {device}")

    # Load model and data
    model, model_config, history = load_model(config)
    model = model.to(device)
    print(f"Final val_acc: {history['val_acc'][-1]:.4f}")

    test_loader, test_dataset = get_svhn_test_loader(config)
    print(f"Test set size: {len(test_dataset)}")

    # Find examples where layer 1 fixes embedding
    print("\nFinding examples where layer 1 fixes embedding mistakes...")
    examples = find_layer1_fixes_examples(model, test_loader, device, n_to_find=5)
    print(f"Found {len(examples)} examples")

    for i, ex in enumerate(examples):
        print(f"  {i}: True={ex['label']}, r0={ex['pred_r0']}, r1={ex['pred_r1']}, r2={ex['pred_r2']}")

    if len(examples) > 0:
        # Main analysis figure
        print("\nGenerating main analysis figure...")
        fig_main = visualize_layer1_fix_example(examples[0], model, device)
        fig_main.savefig(config.results_dir / 'example_layer1_fixes.png', dpi=150)

        # Channel interpretation figures
        print("Generating channel interpretation figures...")
        fig_channels, fig_detail = visualize_channel_interpretation(examples[0], model, device, n_channels=5)
        fig_channels.savefig(config.results_dir / 'example_pixel_contributions.png', dpi=150)

        # Get top channel for naming
        analysis = analyze_bilinear_activations(model, examples[0]['r0'], device, correct_class=examples[0]['label'])
        top_ch = np.argsort(analysis['channel_contributions'].numpy()[examples[0]['label']])[-1]
        fig_detail.savefig(config.results_dir / f'example_channel{top_ch}_detail.png', dpi=150)

        plt.close('all')
        print(f"\nSaved all figures to {config.results_dir}")

    return model, examples


# =============================================================================
# SCRIPT ENTRY POINT
# =============================================================================

if __name__ == "__main__":
    config = AnalysisConfig()
    model, examples = run_full_analysis(config)
