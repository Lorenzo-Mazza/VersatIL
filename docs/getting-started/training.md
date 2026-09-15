# Training

## Running a Training Job

VersatIL trains policies via the `versatil.endpoints.train` module. Each training run is configured by an end-to-end YAML config that composes reusable config groups.

```bash
python -m versatil.endpoints.train --config-name end_to_end_training_runs/bowel_retraction/act
```

End-to-end configs live in `src/versatil/hydra_configs/end_to_end_training_runs/` and are organized by dataset:

```bash
# ACT on bowel retraction
python -m versatil.endpoints.train --config-name end_to_end_training_runs/bowel_retraction/act

# ACT on LIBERO
python -m versatil.endpoints.train --config-name end_to_end_training_runs/libero_hdf5/act
```

## Endoscope Guidance

The `endoscope_guidance_phantom_5hz` presets train on LeRobot V3 datasets with
stereo RGB images, camera-frame tip position, and scalar pivot roll. Set
`VERSATIL_ENDOSCOPE_GUIDANCE_PHANTOM_5HZ_LEROBOT_DIR` in `.env` to the dataset
directory, alongside the cache, Zarr, and checkpoint paths in `.env.example`.

The dataset schema expects `observation.images.left` and
`observation.images.right`, an `observation.state` vector containing three
position values followed by roll, and an `action` vector containing the
corresponding four deltas. It also reads language instructions from the task
metadata and the scalar `task_phase` column; phase-conditioned presets use six
phase classes.

```bash
python -m versatil.endpoints.train \
    --config-name end_to_end_training_runs/endoscope_guidance_phantom_5hz/act_proprio_history_5

python -m versatil.endpoints.train \
    --config-name end_to_end_training_runs/endoscope_guidance_phantom_5hz/smolvla_lora_language_proprio_history_5
```

Presets cover ACT, phase ACT, BC transformers, flow MMDiT, GPT with FAST actions,
pi0-FAST, and SmolVLA. The `_proprio_history_1`, `_proprio_history_2`, and
`_proprio_history_5` variants select the observation history; each predicts ten
action steps. The SmolVLA LoRA variants adapt vision backbones and connectors
while keeping text-model weights frozen.

CSV schema groups are also available as `endoscope_guidance_phantom_5hz` and
`endoscope_guidance_phantom_5hz_phase`. They use
`VERSATIL_ENDOSCOPE_GUIDANCE_PHANTOM_5HZ_DIR`, stereo image paths, camera-frame
position columns, and `relative_pivot_roll`. Compose them with the matching CSV
observation spaces, on-the-fly action spaces, action heads, and losses; their
action keys follow the source observation keys rather than LeRobot's
`position`/`orientation` keys.

For custom configs, scalar roll uses `RELATIVE_PIVOT_ROLL` and the `ROLL`
orientation representation. The old TSO Cartesian orientation enum members
have been removed. The shared `stereo_rgb_language_encoder` now produces full
language-model features; set `encoders.instruction.use_embeddings_only: true`
on that pipeline to retain embedding-only behavior in an existing experiment.

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
    policy/loss=regression_gripper_mmd
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

### Relevant [`ExperimentConfig`][versatil.configs.experiment.ExperimentConfig] Fields

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

Key fields in the default training config (`src/versatil/hydra_configs/training/default.yaml`):

| Field | Default | Description |
|-------|---------|-------------|
| `num_epochs` | 100 | Total training epochs |
| `optimizer` | AdamW | Optimizer config (AdamW, Adam, SGD) |
| `optimizer.lr` | 5e-5 | Base learning rate |
| `clip_gradient_norm` | false | Enable gradient clipping |
| `clip_max_norm` | 1.0 | Max gradient norm (when clipping enabled) |
| `lr_schedule` | None | LR schedule: `"cosine"`, `"linear"`, or None |
| `lr_warmup_steps` | 1000 | Warmup steps for LR schedule |
| `use_ema` | false | Exponential Moving Average of weights |
| `ema_power` | 0.75 | EMA decay power |
| `early_stopping_patience` | 200 | Validation checks without improvement before stopping |
| `gradient_accumulate_every` | 1 | Steps between gradient updates |
| `stages` | [] | Ordered training regimes for freezing groups, optimizer overrides, and loss weights |

The dataclass schema in `src/versatil/configs/training.py` has fallback values,
but normal Hydra training runs compose the YAML defaults above unless an
end-to-end config overrides them.

### Training Stages

`training.stages` defines ordered, epoch-indexed deltas over the base training
regime. A stage may independently override:

- parameter trainability, by optimizer group name
- optimizer learning rate / weight decay, by optimizer group name
- loss weights, as a nested patch matching `policy.loss_module.weights`
- module mode handling for fully frozen submodules

Stages must be listed in strictly increasing `start_epoch` order and may leave
gaps. The base optimizer, trainability, and loss configuration applies before
the first stage, between stage intervals, and after a stage with `end_epoch`
has ended. Each stage is applied from the cached base configuration, not from
the previous stage.

Stage fields:

| Field | Required | Description |
|-------|----------|-------------|
| `name` | yes | Human-readable stage name used in logs and validation errors |
| `start_epoch` | yes | Inclusive epoch where the stage becomes active |
| `end_epoch` | no | Exclusive epoch upper bound; if omitted, the stage runs until the next stage starts or forever if it is last |
| `trainable_groups` | no | Optimizer groups forced to `requires_grad=True` |
| `frozen_groups` | no | Optimizer groups forced to `requires_grad=False` |
| `group_lrs` | no | Per-group learning-rate overrides |
| `group_weight_decays` | no | Per-group weight-decay overrides |
| `loss_weights` | no | Nested partial tree merged onto `policy.loss_module.weights` |
| `eval_frozen_modules` | no | If `true`, fully frozen modules are put in eval mode |

Validation runs before training starts and rejects:

- duplicate stage names
- non-increasing `start_epoch`
- overlapping stage intervals
- unknown optimizer group names
- invalid `loss_weights` paths or dict/scalar shape mismatches

Named optimizer groups come from `training.optimizer.param_groups`. Parameters
that match no configured pattern are assigned to the implicit reserved group
`unmatched`, which must not be used as a custom group name.

`loss_weights` is not a flat scalar map anymore. It must match the public loss
weight tree. For a scalar-weight leaf such as [`PriorDenoisingLoss`][versatil.metrics.losses.prior_denoising.PriorDenoisingLoss], the patch
shape is:

```yaml
loss_weights:
  denoising_prior:
    weight: 0.0
```

When `group_lrs` is used together with `lr_schedule`, the staged values are
treated as new scheduler base learning rates. The current scheduler multiplier
is preserved; stage transitions do not reset scheduler progress.

`training.stages` is incompatible with `reduce_lr_on_plateau`.

```yaml
training:
  optimizer:
    lr: 5.0e-4
    weight_decay: 2.0e-2
    param_groups:
      - name: prior
        lr: 2.0e-4
        params_pattern: "^algorithm\\.prior\\."
      - name: decoder
        lr: 2.0e-4
        params_pattern: "^decoder\\."
  stages:
    - name: vae
      start_epoch: 0
      trainable_groups: ["decoder", "unmatched"]
      frozen_groups: ["prior"]
      loss_weights:
        denoising_prior:
          weight: 0.0
    - name: prior
      start_epoch: 500
      end_epoch: 1000
      trainable_groups: ["prior"]
      frozen_groups: ["decoder", "unmatched"]
      group_lrs:
        prior: 2.0e-4
      group_weight_decays:
        prior: 1.0e-3
      loss_weights:
        denoising_prior:
          weight: 0.03
```

## Distributed Training

!!! note
    Distributed training via SLURM is not yet supported in the current workspace. Set `export NCCL_P2P_DISABLE=1` to avoid NCCL issues on multi-GPU clusters.

## Troubleshooting

### CUDA Issues

Verify the installed driver supports the pinned CUDA 13.0 PyTorch wheels with `nvidia-smi`, then check that `torch.cuda.is_available()` returns `True`.

### Data Loading

- Verify Zarr dataset paths match `VERSATIL_ZARR_DIR` in `.env`
- Ensure the dataset schema config matches your raw data format
- Check sufficient disk space for Zarr cache (created automatically on first run)

### Python 3.14 Compatibility

If Hydra or torchao crash on Python 3.14, see [Known Issues](../known-issues.md) for active workarounds.
