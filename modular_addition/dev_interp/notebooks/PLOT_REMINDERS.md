# Plot Reminders

## X-Axis Alignment with TN-Sim Matrix

**ALWAYS align the x-axis of all plots with the TN-Sim matrix x-axis!**

### Key Rules
1. **Evenly spaced** - Use indices (0, 1, 2, ...) not actual step values
2. **Same labels** - Show step numbers as tick labels
3. **Interpolate data** - Sample training curves at reference_steps

### Code Template

```python
# At the start of plotting section
reference_steps = list(tnsim_matrices.values())[0]['steps']
x_indices = np.arange(len(reference_steps))

def get_values_at_steps(history_steps, history_values, target_steps):
    """Interpolate history values at target steps."""
    return np.interp(target_steps, history_steps, history_values)

def set_aligned_xticks(ax, steps=reference_steps):
    """Helper to align x-axis with TN-Sim matrix steps (evenly spaced)."""
    ax.set_xticks(range(len(steps)))
    ax.set_xticklabels([str(s) for s in steps], rotation=45, ha='right', fontsize=7)
    ax.set_xlim([-0.5, len(steps) - 0.5])

# For training curves - interpolate and plot at x_indices:
vals = get_values_at_steps(h['steps'], h['val_acc'], reference_steps)
ax.plot(x_indices, vals, 'o-', label=f'{n_layers}-Layer', linewidth=2, markersize=4)
set_aligned_xticks(ax)

# For TN-Sim derived data (already at reference_steps):
ax.plot(x_indices, final_sims, 'o-', ...)
set_aligned_xticks(ax)
```

### TN-Sim Matrix Colorbar
Put colorbar at bottom:
```python
plt.colorbar(im, ax=ax, orientation='horizontal', shrink=0.8, pad=0.15)
```

### Checklist for New Analysis Scripts
- [ ] Define `reference_steps` from TN-Sim matrix sample_steps
- [ ] Define `x_indices = np.arange(len(reference_steps))`
- [ ] Create `get_values_at_steps()` for interpolation
- [ ] Create `set_aligned_xticks()` helper function
- [ ] Plot using `x_indices` not actual step values
- [ ] Use `orientation='horizontal'` for TN-Sim matrix colorbars
