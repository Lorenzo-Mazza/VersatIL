# Experiments Configuration

This directory contains modular Hydra/OmegaConf configuration files for running experiments with the VersatIL imitation learning framework.

## Overview

The configuration system is designed to be **modular and composable**. Instead of monolithic JSON files, each aspect of an experiment 
(task, training, policy architecture, loss) is defined in separate YAML files that can be mixed and matched.

## Directory Structure

```
experiments/
├── README.md                           # This file
├── act_bowel_retraction.yaml           # Complete ACT experiment config
├── phase_act_bowel_retraction.yaml     # Complete Phase ACT experiment config
├── ... (any other experiment config)
│
├── experiment/                         # Experiment tracking & infrastructure
│   ├── bowel_retraction_act.yaml
│   └── bowel_retraction_phase_act.yaml
│
├── task/                               # Task definitions
│   ├── action_space/                   # What actions to predict
│   │   └── br_position_gripper_deltas_cf.yaml
│   ├── observation_space/              # What observations to use
│   │   ├── br_rgb_only.yaml
│   │   └── br_rgb_depth.yaml
│   ├── dataset/                        # Dataset schemas
│   │   └── bowel_retraction_v2.yaml
│   └── dataloader/                     # Data loading settings
│       ├── act_default.yaml
│       ├── phase_act_default.yaml
│       └── ...
├── training/                           # Training hyperparameters
│   ├── act_default.yaml
│   ├── phase_act_default.yaml
│   └── ...
├── policy/                             # Policy architecture
│   ├── encoding/                       # Observation encoding pipelines
│   │   ├── rgb_resnet18.yaml
│   │   ├── rgb_depth_geometric.yaml
│   │   └── ...
│   ├── decoder/                        # Action decoder architectures
│   │   ├── act_default.yaml
│   │   └── phase_act_default.yaml
│   │   └── ...
│   ├── algorithm/                      # Learning algorithms
│   │   ├── behavioral_cloning.yaml
│   │   ├── behavioral_cloning_vae.yaml
│   │   └── ...
│   └── loss/                           # Loss functions
│       ├── act_default.yaml
│       ├── act_with_vae.yaml
│       ├── act_with_vae.yaml
│       └── ...
└── inference/                          # Inference settings
    └── act_default.yaml
```

## Configuration Groups

### 1. Experiment (`experiment/`)
Defines experiment tracking, checkpointing, and infrastructure:
- Experiment name
- Checkpoint folder
- WandB integration
- Device (CPU/GPU)
- Checkpointing frequency

### 2. Task (`task/`)
Defines the learning task:

#### Action Space (`task/action_space/`)
What actions the model predicts:
- Position (dim, frame)
- Orientation (representation)
- Gripper (binary/continuous)
- Deltas vs absolute poses
- Phase prediction (for Phase ACT)

#### Observation Space (`task/observation_space/`)
What observations the model receives:
- Camera keys (RGB/depth)
- Proprioceptive data (robot/camera frame)
- Language instructions
- Gripper state

#### Dataset (`task/dataset/`)
Dataset location and preprocessing:
- Zarr path
- Image dimensions
- Train/val split
- Episode filtering

#### Dataloader (`task/dataloader/`)
Data loading configuration:
- Batch size, workers
- Sampling mode (random_chunk, overlapping)
- Image normalization
- Augmentations
- Action shifting (latency compensation)

### 3. Training (`training/`)
Training hyperparameters:
- Number of epochs
- Optimizer (type, LR, weight decay, betas)
- Parameter groups (different LRs for backbone)
- Gradient clipping
- LR schedule (cosine, linear)
- EMA

### 4. Policy (`policy/`)
Policy architecture composed of:

#### Encoding (`policy/encoding/`)
Observation encoding pipeline:
- Encoders for each modality (RGB, depth, proprio)
- Fusion stages (concat, attention, MLP)

#### Decoder (`policy/decoder/`)
Action decoder architecture:
- ACT (DETR-style transformer)
- Phase ACT (with phase prediction)
- Future: UNet, DP-Transformer, MLP

#### Algorithm (`policy/algorithm/`)
Learning algorithm:
- Behavioral Cloning (direct supervision)
- Behavioral Cloning + Variational Inference for multi-modality
- Additional: Diffusion, Flow Matching, VFM

#### Loss (`policy/loss/`)
Loss function configuration:
- Action reconstruction (MSE, L1, gripper BCE)
- KL divergence (VAE)
- Trajectory regularization (length, smoothness)
- Phase classification (cross-entropy, entropy)

### 5. Inference (`inference/`)
Inference-time settings:
- Temporal aggregation
- Control frequency

## Using Configurations

### Running an Experiment

Use Hydra to compose configs:

```bash
# Run ACT experiment
python -m versatil.endpoints.train --config-path experiments --config-name act_bowel_retraction

# Run Phase ACT experiment
python -m versatil.endpoints.train --config-path experiments --config-name phase_act_bowel_retraction
```

### Overriding Parameters

Override any parameter from the command line:

```bash
# Change batch size
python -m versatil.endpoints.train \
    --config-path experiments \
    --config-name act_bowel_retraction \
    task.dataloader.batch_size=64

# Change learning rate
python -m versatil.endpoints.train \
    --config-path experiments \
    --config-name act_bowel_retraction \
    training.optimizer.learning_rate=5e-5

# Use different encoder
python -m versatil.endpoints.train \
    --config-path experiments \
    --config-name act_bowel_retraction \
    policy/encoding=rgb_depth_geometric
```

### Creating Custom Experiments

Mix and match existing configs:

```yaml
# experiments/my_custom_experiment.yaml
defaults:
  - experiment: bowel_retraction_act  # Base experiment
  - task/dataset: bowel_retraction_v2
  - task/dataloader: act_default
  - task/action_space: br_position_gripper_deltas_cf
  - task/observation_space: br_rgb_depth  # Use depth instead of RGB-only
  - training: act_default
  - policy/encoding: rgb_depth_geometric  # Match observation space
  - policy/decoder: act_default
  - policy/algorithm: behavioral_cloning_vae  # Add VAE
  - policy/loss: act_with_vae  # Match algorithm
  - inference: act_default
  - _self_  # Keep this last

# Override specific settings
experiment:
  name: my_custom_act_experiment

task:
  prediction_horizon: 16  # Longer chunks

training:
  optimizer:
    learning_rate: 1.0e-5  # Lower LR
```

## Mapping from Old Configs

The modular configs replicate the old JSON configs from `src/endpoints/default_configs/`:

### ACT (`act_br_cf_new_loss.json`)
Maps to: `experiments/act_bowel_retraction.yaml`
- Observation: RGB-only (ResNet18)
- Action: Position deltas + gripper in camera frame
- Loss: MSE (0.8), gripper BCE (0.05), length (0.001)
- Training: LR 1e-4, WD 1e-4, cosine schedule
- Prediction horizon: 10

### Phase ACT (`phase_act.json`)
Maps to: `experiments/phase_act_bowel_retraction.yaml`
- Observation: RGB + Depth (DFormerV2)
- Action: Position deltas + gripper + phase labels
- Loss: L1 (0.8), KL (100), phase CE (0.1), entropy (0.01)
- Training: LR 5e-5, WD 1e-3, no schedule
- Prediction horizon: 5

## Key Design Principles

1. **Modularity**: Each component in a separate file instead of monolithic JSON
2. **Composition**: Mix and match configs via Hydra defaults
3. **Type Safety**: Configs validated against dataclasses at load time
4. **Separate encoding/decoding**: Clearer separation of concerns
5. **Algorithm vs Architecture**: Learning algorithm (BC, diffusion) decoupled from network architecture (transformer, UNet)

## Adding New Configurations

To add a new configuration component:

1. **Create YAML file** in appropriate directory
2. **Use `_target_`** to specify the Python class to instantiate
3. **Reference other configs** using Hydra interpolation: `${task.observation_space}`
4. **Add to defaults** in a top-level experiment YAML

Example - new encoder:

```yaml
# experiments/policy/encoding/my_encoder.yaml
encoders:
  my_encoder:
    _target_: versatil.models.encoding.encoders.my_module.MyEncoder
    input_keys: left_image
    my_param: 42

fusion_stages:
  - _target_: versatil.models.encoding.fusion.concat.ConcatFusion
    input_features:
      - my_encoder_features
    output_name: visual_features
```

## Hydra Features Used

- **Config Groups**: Organized by category (task, training, policy)
- **Defaults List**: Compose configs from multiple groups
- **Interpolation**: `${task.observation_space}` references another config
- **Overrides**: Command-line and YAML overrides
- **Structured Configs**: Type-checked via dataclasses

## References

- [Hydra Documentation](https://hydra.cc/docs/intro/)
- [OmegaConf Documentation](https://omegaconf.readthedocs.io/)
