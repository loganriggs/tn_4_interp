# %% [markdown]
# # Analyze 1-4 Layer Models with Down Projection + Augmentation
#
# Compute TN-sim matrices across checkpoints for all models.
#
# REMINDER: Always align x-axis of all plots with the TN-Sim matrix x-axis!

# %%
import pickle
from pathlib import Path
import numpy as np
import torch
import matplotlib.pyplot as plt

# %%
output_dir = Path("tn_analysis_checkpoints")

# %%
# =============================================================================
# LOAD MODELS
# =============================================================================

def load_model(n_layers):
    path = output_dir / f"residual_{n_layers}layer_downproj_aug.pkl"
    if not path.exists():
        print(f"Not found: {path}")
        return None
    with open(path, 'rb') as f:
        return pickle.load(f)

models = {}
for n_layers in [1, 2, 3, 4]:
    data = load_model(n_layers)
    if data:
        models[n_layers] = data
        print(f"{n_layers}-Layer: {len(data['checkpoints'])} checkpoints, "
              f"final val_acc={data['history']['val_acc'][-1]:.4f}")

if len(models) == 0:
    print("No models found. Training may still be in progress.")
    exit()

# %%
# =============================================================================
# TN-SIM (Monte Carlo - full model)
# =============================================================================

def extract_weights(state_dict, n_layers, d_res=128, d_hidden=256, device='cuda'):
    """Extract weights from checkpoint with down projection."""
    weights = {
        'embed': state_dict['embed.weight'].to(device),
        'proj': state_dict['projection.weight'].to(device),
        'n_layers': n_layers,
    }
    for i in range(n_layers):
        weights[f'L{i}'] = state_dict[f'bilinears.{i}.LR.weight'][:d_hidden].to(device)
        weights[f'R{i}'] = state_dict[f'bilinears.{i}.LR.weight'][d_hidden:].to(device)
        weights[f'D{i}'] = state_dict[f'bilinears.{i}.D.weight'].to(device)
        weights[f'norm{i}_weight'] = state_dict[f'norms.{i}.weight'].to(device)
    return weights


def forward_pass(weights, x):
    """Forward pass through N-layer residual bilinear with RMSNorm and down proj."""
    n_layers = weights['n_layers']

    r = x @ weights['embed'].T

    for i in range(n_layers):
        # RMSNorm
        rms = torch.sqrt((r ** 2).mean(dim=1, keepdim=True) + 1e-6)
        n = weights[f'norm{i}_weight'] * (r / rms)
        # Bilinear with down projection
        h = (n @ weights[f'L{i}'].T) * (n @ weights[f'R{i}'].T)
        h = h @ weights[f'D{i}'].T
        # Residual
        r = r + h

    return r @ weights['proj'].T


def mc_tnsim(weights1, weights2, x):
    """Monte Carlo TN-sim."""
    out1 = forward_pass(weights1, x)
    out2 = forward_pass(weights2, x)

    inner = (out1 * out2).sum(dim=1).mean()
    norm1_sq = (out1 * out1).sum(dim=1).mean()
    norm2_sq = (out2 * out2).sum(dim=1).mean()

    sim = inner / (torch.sqrt(norm1_sq * norm2_sq) + 1e-10)
    return sim.item()


# %%
# =============================================================================
# COMPUTE TN-SIM MATRICES
# =============================================================================

if __name__ == "__main__":
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")

    # Sample inputs
    n_samples = 5000
    d_input = 3072
    torch.manual_seed(42)
    x = torch.randn(n_samples, d_input, device=device)
    x = x / x.norm(dim=1, keepdim=True)

    # For each model, compute TN-sim matrix
    tnsim_matrices = {}

    for n_layers, data in models.items():
        print(f"\n{'='*60}")
        print(f"Computing TN-Sim Matrix for {n_layers}-Layer Model")
        print('='*60)

        checkpoints = data['checkpoints']
        steps = sorted(checkpoints.keys())

        # Sample steps for matrix
        if len(steps) > 15:
            indices = np.linspace(0, len(steps)-1, 15, dtype=int)
            sample_steps = [steps[i] for i in indices]
        else:
            sample_steps = steps

        n = len(sample_steps)
        sim_matrix = np.zeros((n, n))

        print(f"Computing {n}x{n} matrix for steps: {sample_steps}")

        for i, s1 in enumerate(sample_steps):
            w1 = extract_weights(checkpoints[s1], n_layers, device=device)
            for j, s2 in enumerate(sample_steps):
                if j < i:
                    sim_matrix[i, j] = sim_matrix[j, i]
                else:
                    w2 = extract_weights(checkpoints[s2], n_layers, device=device)
                    sim_matrix[i, j] = mc_tnsim(w1, w2, x)
            print(f"  Row {i+1}/{n} done")

        tnsim_matrices[n_layers] = {
            'matrix': sim_matrix,
            'steps': sample_steps,
        }

    # =========================================================================
    # PLOTTING
    # =========================================================================

    # IMPORTANT: Use consistent x-axis steps across all plots
    # Get sample_steps from first available model's TN-sim matrix
    reference_steps = list(tnsim_matrices.values())[0]['steps']

    def get_values_at_steps(history_steps, history_values, target_steps):
        """Interpolate history values at target steps."""
        return np.interp(target_steps, history_steps, history_values)

    def set_aligned_xticks(ax, steps=reference_steps):
        """Helper to align x-axis with TN-Sim matrix steps (evenly spaced)."""
        ax.set_xticks(range(len(steps)))
        ax.set_xticklabels([str(s) for s in steps], rotation=45, ha='right', fontsize=7)
        ax.set_xlim([-0.5, len(steps) - 0.5])

    n_models = len(models)
    fig, axes = plt.subplots(3, max(4, n_models), figsize=(20, 15))

    # Row 1: TN-Sim matrices (colorbar at bottom)
    for i, n_layers in enumerate([1, 2, 3, 4]):
        ax = axes[0, i]
        if n_layers in tnsim_matrices:
            data = tnsim_matrices[n_layers]
            sim_matrix = data['matrix']
            sample_steps = data['steps']
            n = len(sample_steps)

            im = ax.imshow(sim_matrix, cmap='viridis', vmin=0, vmax=1)
            ax.set_xticks(range(n))
            ax.set_xticklabels([str(s) for s in sample_steps], rotation=45, ha='right', fontsize=7)
            ax.set_yticks(range(n))
            ax.set_yticklabels([str(s) for s in sample_steps], fontsize=7)
            ax.set_title(f'{n_layers}-Layer TN-Sim Matrix')
            plt.colorbar(im, ax=ax, orientation='horizontal', shrink=0.8, pad=0.15)
        else:
            ax.text(0.5, 0.5, 'Not Available', ha='center', va='center', transform=ax.transAxes)
            ax.set_title(f'{n_layers}-Layer (N/A)')

    # Row 2: Training curves (x-axis evenly spaced, aligned with TN-Sim matrix)
    x_indices = np.arange(len(reference_steps))

    ax = axes[1, 0]
    for n_layers, data in models.items():
        h = data['history']
        vals = get_values_at_steps(h['steps'], h['val_acc'], reference_steps)
        ax.plot(x_indices, vals, 'o-', label=f'{n_layers}-Layer', linewidth=2, markersize=4)
    ax.set_xlabel('Step')
    ax.set_ylabel('Val Accuracy')
    ax.set_title('Validation Accuracy')
    ax.legend()
    ax.grid(True, alpha=0.3)
    set_aligned_xticks(ax)

    ax = axes[1, 1]
    for n_layers, data in models.items():
        h = data['history']
        vals = get_values_at_steps(h['steps'], h['val_loss'], reference_steps)
        ax.semilogy(x_indices, vals, 'o-', label=f'{n_layers}-Layer', linewidth=2, markersize=4)
    ax.set_xlabel('Step')
    ax.set_ylabel('Val Loss (log)')
    ax.set_title('Validation Loss')
    ax.legend()
    ax.grid(True, alpha=0.3)
    set_aligned_xticks(ax)

    # Similarity to final (evenly spaced)
    ax = axes[1, 2]
    for n_layers, data in tnsim_matrices.items():
        sim_matrix = data['matrix']
        final_sims = sim_matrix[:, -1]
        ax.plot(x_indices, final_sims, 'o-', label=f'{n_layers}-Layer', linewidth=2, markersize=4)
    ax.set_xlabel('Step')
    ax.set_ylabel('TN-Sim to Final')
    ax.set_title('Similarity to Final Checkpoint')
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.set_ylim([0, 1.1])
    set_aligned_xticks(ax)

    # Adjacent checkpoint similarity (evenly spaced, at midpoints)
    ax = axes[1, 3]
    x_midpoints = x_indices[:-1] + 0.5
    for n_layers, data in tnsim_matrices.items():
        sim_matrix = data['matrix']
        adjacent_sims = np.diag(sim_matrix, k=1)
        ax.plot(x_midpoints, adjacent_sims, 'o-', label=f'{n_layers}-Layer', linewidth=2, markersize=4)
    ax.set_xlabel('Step')
    ax.set_ylabel('TN-Sim')
    ax.set_title('Adjacent Checkpoint Similarity')
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.set_ylim([0, 1.1])
    set_aligned_xticks(ax)

    # Row 3: Overfitting and init-final similarity (x-axis evenly spaced)
    ax = axes[2, 0]
    for n_layers, data in models.items():
        h = data['history']
        train_vals = get_values_at_steps(h['steps'], h['train_acc'], reference_steps)
        val_vals = get_values_at_steps(h['steps'], h['val_acc'], reference_steps)
        gap = train_vals - val_vals
        ax.plot(x_indices, gap, 'o-', label=f'{n_layers}-Layer', linewidth=2, markersize=4)
    ax.set_xlabel('Step')
    ax.set_ylabel('Train - Val Acc')
    ax.set_title('Overfitting Gap')
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.axhline(y=0, color='black', linestyle='--', alpha=0.5)
    set_aligned_xticks(ax)

    ax = axes[2, 1]
    for n_layers, data in models.items():
        h = data['history']
        vals = get_values_at_steps(h['steps'], h['train_loss'], reference_steps)
        ax.semilogy(x_indices, vals, 'o-', label=f'{n_layers}-Layer', linewidth=2, markersize=4)
    ax.set_xlabel('Step')
    ax.set_ylabel('Train Loss (log)')
    ax.set_title('Training Loss')
    ax.legend()
    ax.grid(True, alpha=0.3)
    set_aligned_xticks(ax)

    # Bar chart: final metrics
    ax = axes[2, 2]
    layers_list = sorted(models.keys())
    val_accs = [models[n]['history']['val_acc'][-1] for n in layers_list]
    init_final_sims = [tnsim_matrices[n]['matrix'][0, -1] if n in tnsim_matrices else 0 for n in layers_list]
    x_pos = np.arange(len(layers_list))
    width = 0.35
    ax.bar(x_pos - width/2, val_accs, width, label='Val Acc')
    ax.bar(x_pos + width/2, init_final_sims, width, label='Init-Final TN-Sim')
    ax.set_xlabel('Layers')
    ax.set_ylabel('Value')
    ax.set_title('Final Metrics Comparison')
    ax.set_xticks(x_pos)
    ax.set_xticklabels([f'{n}L' for n in layers_list])
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Hide unused subplot
    axes[2, 3].axis('off')

    plt.tight_layout()
    plt.savefig(output_dir / "analyze_1to4_layers_aug.png", dpi=150, bbox_inches='tight')
    plt.show()

    print(f"\nSaved to {output_dir / 'analyze_1to4_layers_aug.png'}")

    # =========================================================================
    # SUMMARY
    # =========================================================================

    print("\n" + "="*70)
    print("SUMMARY")
    print("="*70)

    print(f"\n{'Layers':<10} | {'Final Val Acc':>14} | {'Final Val Loss':>14} | {'Init-Final TN-Sim':>18}")
    print("-"*70)

    for n_layers in [1, 2, 3, 4]:
        if n_layers in models:
            h = models[n_layers]['history']
            val_acc = h['val_acc'][-1]
            val_loss = h['val_loss'][-1]
            if n_layers in tnsim_matrices:
                init_final_sim = tnsim_matrices[n_layers]['matrix'][0, -1]
            else:
                init_final_sim = float('nan')
            print(f"{n_layers:<10} | {val_acc:>14.4f} | {val_loss:>14.4f} | {init_final_sim:>18.4f}")
