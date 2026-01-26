"""
Grokking trajectory experiment for modular addition.

Trains a bilinear model on modular addition (P=113) and tracks TN similarity,
activation similarity, and JS divergence across 200 checkpoints to visualize
the grokking trajectory.

Usage:
    python -m modular_addition.dev_interp.grokking_experiment
"""
import json
import pickle
from dataclasses import dataclass, asdict
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm

from modular_addition.core import (
    # Models
    init_model,
    get_device,
    Model,
    # Dataset
    create_full_dataset,
    create_labels,
    create_train_val_split,
    compute_accuracy,
    # Interaction matrices
    compute_interaction_matrix,
    # Eigendecomposition & frequency
    compute_eigendecomposition,
    compute_frequency_distribution,
    # Similarity
    symmetric_similarity,
    compute_all_logits,
    pairwise_cosine_similarity,
    # Metrics
    compute_average_js_divergence,
)


@dataclass
class ExperimentConfig:
    """Configuration for the grokking trajectory experiment."""
    P: int = 113
    d_hidden: int = 64
    lr: float = 1e-3
    weight_decay: float = 1e-2
    batch_size: int = 512
    total_steps: int = 100_000
    checkpoint_every: int = 500  # Used as fallback for late training
    val_fraction: float = 0.4
    n_evecs: int = 4
    seed: int = 1337
    output_dir: str = "modular_addition/dev_interp/results"


def compute_checkpoint_schedule(
    total_steps: int,
    steps_per_epoch: int,
    dense_epochs: int = 5,
    log_checkpoints: int = 150
) -> set[int]:
    """
    Generate checkpoint schedule with dense early sampling and log-spaced later epochs.

    Schedule:
    - Epochs 1 to dense_epochs: dense per-step sampling
      (Epoch 1: every step, Epoch 2: every 2 steps, etc.)
    - After dense_epochs: logarithmically spaced epoch checkpoints

    Args:
        total_steps: Total training steps
        steps_per_epoch: Number of steps per epoch
        dense_epochs: Number of epochs with dense per-step sampling
        log_checkpoints: Approximate number of checkpoints in log phase

    Returns:
        Set of step numbers to checkpoint at
    """
    checkpoint_steps = {0}  # Always include random init
    total_epochs = total_steps // steps_per_epoch

    # Phase 1: Dense per-step sampling for first few epochs
    for epoch in range(1, min(dense_epochs + 1, total_epochs + 1)):
        epoch_start = (epoch - 1) * steps_per_epoch
        epoch_end = min(epoch * steps_per_epoch, total_steps)
        interval = epoch  # Epoch 1: every step, Epoch 2: every 2, etc.

        for s in range(epoch_start, epoch_end, interval):
            if s > 0:
                checkpoint_steps.add(s)
        checkpoint_steps.add(epoch_end)

    # Phase 2: Logarithmically spaced epochs after dense phase
    if total_epochs > dense_epochs:
        log_epochs = np.unique(np.logspace(
            np.log10(dense_epochs + 1),
            np.log10(total_epochs),
            num=log_checkpoints
        ).astype(int))

        for epoch in log_epochs:
            step = epoch * steps_per_epoch
            if step <= total_steps:
                checkpoint_steps.add(step)

    # Always include final step
    checkpoint_steps.add(total_steps)

    return checkpoint_steps


def train_with_checkpoints(config: ExperimentConfig) -> tuple[dict, dict]:
    """
    Train model and save checkpoints with dense early sampling.

    Uses a schedule that captures more checkpoints early in training:
    - Epoch 1: every step (including random init at step 0)
    - Epoch 2: every 2 steps
    - Epoch N: every N steps (capped at 1 per epoch)

    Args:
        config: Experiment configuration

    Returns:
        checkpoints: {step: state_dict}
        history: {step: {'train_loss', 'val_loss', 'val_acc'}}
    """
    device = get_device()
    print(f"Using device: {device}")

    # Set seeds for reproducibility
    torch.manual_seed(config.seed)
    np.random.seed(config.seed)

    # Create model
    model = init_model(p=config.P, d_hidden=config.d_hidden).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=config.lr,
        weight_decay=config.weight_decay
    )
    loss_fn = nn.CrossEntropyLoss()

    # Create data split
    train_data, train_labels, val_data, val_labels = create_train_val_split(
        P=config.P,
        train_fraction=1.0 - config.val_fraction,
        seed=config.seed,
        device=device
    )

    print(f"Train size: {len(train_data)}, Val size: {len(val_data)}")

    # Create train dataloader
    train_ds = TensorDataset(train_data, train_labels)
    train_loader = DataLoader(
        train_ds,
        batch_size=config.batch_size,
        shuffle=True,
        drop_last=True
    )

    # Compute steps per epoch and checkpoint schedule
    steps_per_epoch = len(train_data) // config.batch_size
    checkpoint_schedule = compute_checkpoint_schedule(config.total_steps, steps_per_epoch)
    print(f"Steps per epoch: {steps_per_epoch}")
    print(f"Total checkpoints scheduled: {len(checkpoint_schedule)}")

    # Storage
    checkpoints = {}
    history = {}

    def save_checkpoint(step: int):
        """Save checkpoint and compute metrics at given step."""
        checkpoints[step] = {
            k: v.cpu().clone() for k, v in model.state_dict().items()
        }
        model.eval()
        with torch.no_grad():
            train_logits = model(train_data)
            train_loss = loss_fn(train_logits, train_labels).item()
            val_logits = model(val_data)
            val_loss = loss_fn(val_logits, val_labels).item()
            val_acc = compute_accuracy(model, val_data, val_labels)
        history[step] = {
            'train_loss': train_loss,
            'val_loss': val_loss,
            'val_acc': val_acc
        }
        return train_loss, val_loss, val_acc

    # Save random init (step 0)
    if 0 in checkpoint_schedule:
        train_loss, val_loss, val_acc = save_checkpoint(0)
        print(f"Step 0 (random init): train_loss={train_loss:.4f}, val_acc={val_acc:.4f}")

    # Training loop with step tracking
    global_step = 0
    pbar = tqdm(total=config.total_steps, desc="Training")

    while global_step < config.total_steps:
        for batch_data, batch_labels in train_loader:
            if global_step >= config.total_steps:
                break

            # Training step
            model.train()
            optimizer.zero_grad()
            logits = model(batch_data)
            loss = loss_fn(logits, batch_labels)
            loss.backward()
            optimizer.step()

            global_step += 1
            pbar.update(1)

            # Checkpoint if in schedule
            if global_step in checkpoint_schedule:
                train_loss, val_loss, val_acc = save_checkpoint(global_step)
                pbar.set_postfix({
                    'ckpts': len(checkpoints),
                    'train_loss': f'{train_loss:.4f}',
                    'val_acc': f'{val_acc:.4f}'
                })

    pbar.close()
    print(f"Saved {len(checkpoints)} checkpoints")
    return checkpoints, history


def compute_frequency_heatmap_from_model(
    model: Model,
    P: int,
    n_evecs: int = 4
) -> np.ndarray:
    """
    Compute frequency heatmap directly from a model.

    Args:
        model: The trained model
        P: Input dimension
        n_evecs: Number of top eigenvectors to use

    Returns:
        (P, P//2+1) frequency heatmap
    """
    # Get interaction matrix
    int_mat = compute_interaction_matrix(model)  # (P, 2P, 2P)

    n_freqs = P // 2 + 1
    heatmap = np.zeros((P, n_freqs))

    for r in range(P):
        # Eigendecomposition for this remainder
        evals, evecs = compute_eigendecomposition(int_mat[r])
        # Compute frequency distribution
        heatmap[r] = compute_frequency_distribution(evals, evecs, P, n_evecs)

    return heatmap


def compute_similarity_matrices(
    checkpoints: dict[int, dict],
    config: ExperimentConfig
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Compute all pairwise similarity matrices.

    Args:
        checkpoints: {step: state_dict}
        config: Experiment configuration

    Returns:
        tn_sim: (N, N) TN similarity matrix
        act_sim: (N, N) activation similarity matrix
        js_div: (N, N) JS divergence matrix
    """
    device = get_device()
    steps = sorted(checkpoints.keys())
    N = len(steps)

    print(f"Computing similarity matrices for {N} checkpoints...")

    # Create dataset for activation similarity
    dataset = create_full_dataset(config.P, device)

    # Precompute all data needed for pairwise comparisons
    print("Precomputing logits and frequency heatmaps...")
    all_logits = {}
    all_heatmaps = {}

    for step in tqdm(steps, desc="Precomputing"):
        model = init_model(p=config.P, d_hidden=config.d_hidden).to(device)
        model.load_state_dict(checkpoints[step])
        model.eval()

        # Logits for activation similarity
        all_logits[step] = compute_all_logits(model, dataset)

        # Frequency heatmap for JS divergence
        all_heatmaps[step] = compute_frequency_heatmap_from_model(
            model, config.P, config.n_evecs
        )

    # Compute pairwise TN similarity
    print("Computing TN similarity matrix...")
    tn_sim = np.zeros((N, N), dtype=float)

    with torch.no_grad():
        for i in tqdm(range(N), desc="TN similarity"):
            step_i = steps[i]
            model_i = init_model(p=config.P, d_hidden=config.d_hidden).to(device)
            model_i.load_state_dict(checkpoints[step_i])
            model_i.eval()

            for j in range(i, N):
                step_j = steps[j]
                model_j = init_model(p=config.P, d_hidden=config.d_hidden).to(device)
                model_j.load_state_dict(checkpoints[step_j])
                model_j.eval()

                sim = float(symmetric_similarity(model_i, model_j))
                tn_sim[i, j] = sim
                tn_sim[j, i] = sim

    # Compute pairwise activation similarity
    print("Computing activation similarity matrix...")
    act_sim = np.zeros((N, N), dtype=float)

    for i in tqdm(range(N), desc="Activation similarity"):
        for j in range(i, N):
            sim = pairwise_cosine_similarity(
                all_logits[steps[i]],
                all_logits[steps[j]]
            )
            act_sim[i, j] = sim
            act_sim[j, i] = sim

    # Compute pairwise JS divergence
    print("Computing JS divergence matrix...")
    js_div = np.zeros((N, N), dtype=float)

    for i in tqdm(range(N), desc="JS divergence"):
        for j in range(i, N):
            div = compute_average_js_divergence(
                all_heatmaps[steps[i]],
                all_heatmaps[steps[j]]
            )
            js_div[i, j] = div
            js_div[j, i] = div

    return tn_sim, act_sim, js_div


def save_results(
    checkpoints: dict,
    history: dict,
    tn_sim: np.ndarray,
    act_sim: np.ndarray,
    js_div: np.ndarray,
    config: ExperimentConfig
) -> None:
    """Save all results to disk."""
    output_dir = Path(config.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    steps = np.array(sorted(checkpoints.keys()))

    # Save checkpoints (for potential reanalysis)
    with open(output_dir / "checkpoints.pkl", "wb") as f:
        pickle.dump(checkpoints, f)
    print(f"Saved checkpoints.pkl ({(output_dir / 'checkpoints.pkl').stat().st_size / 1024 / 1024:.2f} MB)")

    # Save training history
    with open(output_dir / "training_history.json", "w") as f:
        # Convert int keys to strings for JSON
        json.dump({str(k): v for k, v in history.items()}, f, indent=2)
    print("Saved training_history.json")

    # Save matrices
    np.save(output_dir / "tn_similarity.npy", tn_sim)
    np.save(output_dir / "act_similarity.npy", act_sim)
    np.save(output_dir / "js_divergence.npy", js_div)
    np.save(output_dir / "checkpoint_steps.npy", steps)
    print("Saved similarity matrices")

    # Save config
    with open(output_dir / "config.json", "w") as f:
        json.dump(asdict(config), f, indent=2)
    print("Saved config.json")

    print(f"\nAll results saved to {output_dir}/")
    print("Note: Delete checkpoints.pkl manually after analysis to save space.")


def main():
    """Run the grokking trajectory experiment."""
    config = ExperimentConfig()

    # Compute expected schedule for display
    train_samples = int(config.P * config.P * (1 - config.val_fraction))
    steps_per_epoch = train_samples // config.batch_size
    schedule = compute_checkpoint_schedule(config.total_steps, steps_per_epoch)

    print("=" * 60)
    print("Grokking Trajectory Experiment")
    print("=" * 60)
    print(f"P={config.P}, d_hidden={config.d_hidden}")
    print(f"Total steps={config.total_steps}")
    print(f"Steps per epoch: {steps_per_epoch}")
    print(f"Expected checkpoints: {len(schedule)} (dense early, sparse late)")
    print(f"First 20 checkpoint steps: {sorted(schedule)[:20]}")
    print("=" * 60)

    # Phase 1: Train and checkpoint
    print("\nPhase 1: Training with checkpoints...")
    checkpoints, history = train_with_checkpoints(config)

    # Phase 2: Compute similarity matrices
    print("\nPhase 2: Computing similarity matrices...")
    tn_sim, act_sim, js_div = compute_similarity_matrices(checkpoints, config)

    # Phase 3: Save results
    print("\nPhase 3: Saving results...")
    save_results(checkpoints, history, tn_sim, act_sim, js_div, config)

    # Summary
    print("\n" + "=" * 60)
    print("Experiment Complete!")
    print("=" * 60)
    print(f"Matrices shape: {tn_sim.shape}")
    print(f"Final validation accuracy: {history[max(history.keys())]['val_acc']:.4f}")

    # Quick grokking detection
    steps = sorted(history.keys())
    val_accs = [history[s]['val_acc'] for s in steps]
    grok_threshold = 0.95
    grok_step = next((s for s, acc in zip(steps, val_accs) if acc >= grok_threshold), None)
    if grok_step:
        print(f"Reached {grok_threshold*100}% val accuracy at step {grok_step}")
    else:
        print(f"Did not reach {grok_threshold*100}% val accuracy")


if __name__ == "__main__":
    main()
