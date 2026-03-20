# Training

## Running a Training Job

VersatIL trains policies via the `versatil.endpoints.train` module. Each training run is configured by an end-to-end YAML config that composes reusable config groups.

```bash
python -m versatil.endpoints.train --config-name end_to_end_training_runs/bowel_retraction/act
```

End-to-end configs live in `hydra_configs/end_to_end_training_runs/` and are organized by dataset:

```bash
# ACT on bowel retraction
python -m versatil.endpoints.train --config-name end_to_end_training_runs/bowel_retraction/act

# ACT on LIBERO
python -m versatil.endpoints.train --config-name end_to_end_training_runs/libero_hdf5/act
```

## CLI Overrides

Hydra allows overriding any configuration parameter from the command line without editing YAML files.

### Scalar Overrides

Override individual values using dot-separated paths:

```bash
# Change batch size
python -m versatil.endpoints.train \
    --config-name end_to_end_training_runs/bowel_retraction/act \
    task.dataloader.batch_size=64

# Change learning rate
python -m versatil.endpoints.train \
    --config-name end_to_end_training_runs/bowel_retraction/act \
    training.optimizer.lr=1e-4

# Disable EMA
python -m versatil.endpoints.train \
    --config-name end_to_end_training_runs/bowel_retraction/act \
    training.use_ema=false
```

### Config Group Overrides

Swap entire config groups using the `group=option` syntax:

```bash
# Use MMD loss instead of default
python -m versatil.endpoints.train \
    --config-name end_to_end_training_runs/bowel_retraction/act \
    policy/loss=regression_gripper_MMD
```

This replaces the loss config block while keeping everything else intact.

## Resume from Checkpoint

To resume a training run from a saved checkpoint:

```bash
python -m versatil.endpoints.train \
    --config-name end_to_end_training_runs/bowel_retraction/act \
    experiment.resume_from=/path/to/checkpoint.ckpt
```

This restores model weights, optimizer state, and training progress.

## Checkpointing

VersatIL saves checkpoints in the directory specified by `experiment.checkpoint_folder`, which typically resolves from `VERSATIL_CHECKPOINT_DIR` via the `${checkpoint_dir:...}` resolver.

### Best Model Checkpoints

Saved based on `val_loss`:

```
checkpoints/experiment_name/best-epoch=XX-val_loss=Y.YYYY.ckpt
```

### Periodic Checkpoints

Saved every `experiment.checkpoint_every` epochs (default: 100):

```
checkpoints/experiment_name/latest-epoch=XX.ckpt
checkpoints/experiment_name/last.ckpt
```

### Relevant ExperimentConfig Fields

| Field | Default | Description |
|-------|---------|-------------|
| `checkpoint_folder` | Required | Directory for checkpoint storage |
| `checkpoint_every` | 100 | Save periodic checkpoint every N epochs |
| `val_every` | 1 | Run validation every N epochs |
| `resume_from` | None | Path to checkpoint for resuming |

## WandB Integration

VersatIL uses [Weights & Biases](https://wandb.ai/) for experiment tracking. Logged metrics include:

- Train/validation loss curves
- Learning rate schedules
- Gradient norms (pre/post clipping)
- EMA decay values
- Model-specific metrics (e.g., phase confusion matrices)

### Setup

Set your API key as an environment variable:

```bash
export WANDB_API_KEY=your_key_here
```

For persistent configuration, add it to `~/.bashrc` or set it in your `.env` file.

### Configuration

WandB is enabled by default (`experiment.use_wandb=true`). Configure project and entity either via `.env`:

```bash
WANDB_PROJECT=versatil
WANDB_ENTITY=your-team
```

Or via CLI overrides:

```bash
python -m versatil.endpoints.train \
    --config-name end_to_end_training_runs/bowel_retraction/act \
    experiment.wandb_project=my_project \
    experiment.wandb_entity=my_team
```

To disable WandB for a run:

```bash
python -m versatil.endpoints.train \
    --config-name end_to_end_training_runs/bowel_retraction/act \
    experiment.use_wandb=false
```

## Training Configuration Reference

Key fields in `TrainingConfig`:

| Field | Default | Description |
|-------|---------|-------------|
| `num_epochs` | 100 | Total training epochs |
| `optimizer` | AdamW | Optimizer config (AdamW, Adam, SGD) |
| `optimizer.lr` | 1e-4 | Base learning rate |
| `clip_gradient_norm` | false | Enable gradient clipping |
| `clip_max_norm` | 0.1 | Max gradient norm (when clipping enabled) |
| `lr_schedule` | None | LR schedule: `"cosine"`, `"linear"`, or None |
| `lr_warmup_steps` | 5000 | Warmup steps for LR schedule |
| `use_ema` | true | Exponential Moving Average of weights |
| `ema_power` | 0.75 | EMA decay power |
| `early_stopping_patience` | 10 | Validation checks without improvement before stopping |
| `gradient_accumulate_every` | 1 | Steps between gradient updates |

## Distributed Training

!!! note
    Distributed training via SLURM is not yet supported in the current workspace. Set `export NCCL_P2P_DISABLE=1` to avoid NCCL issues on multi-GPU clusters.

## Troubleshooting

### CUDA Issues

Verify CUDA 12.8+ with `nvidia-smi` and check that `torch.cuda.is_available()` returns `True`.

### Data Loading

- Verify Zarr dataset paths match `VERSATIL_ZARR_DIR` in `.env`
- Ensure the dataset schema config matches your raw data format
- Check sufficient disk space for Zarr cache (created automatically on first run)
