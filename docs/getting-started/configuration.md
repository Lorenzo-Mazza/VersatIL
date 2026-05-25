# Configuration System

VersatIL uses [Hydra](https://hydra.cc/) and [OmegaConf](https://omegaconf.readthedocs.io/) for experiment configuration. This combination provides hierarchical composition of reusable config blocks, typed validation via Python dataclasses, and CLI overrides without code changes.

## Hydra Composition Pattern

Instead of monolithic config files, VersatIL composes small, reusable config blocks stored in `hydra_configs/`. An end-to-end training config declares which blocks to combine via `defaults:`:

```yaml
# hydra_configs/end_to_end_training_runs/bowel_retraction/act.yaml

# Required in all end-to-end configs: declares the package scope and the target dataclass.
# @package _global_
_target_: versatil.configs.main.MainConfig

defaults:
  - /task: base
  - /policy: base
  - /experiment: default
  - /task/dataset_schema: bowel_retraction_v2
  - /task/dataloader: bowel_retraction
  - /task/action_space: deltas_cf_pos_gripper_phase
  - /task/observation_space: stereo_rgb
  - /training: default
  - /policy/encoding_pipeline: stereo_rgb
  - /policy/decoder: act_default
  - /policy/algorithm: bc_with_vae_gaussian
  - /policy/loss: regression_gripper_KL
  - /inference: default
  - _self_
```

Each line references a YAML file under `hydra_configs/`. For example, `/policy/algorithm: bc_with_vae_gaussian` loads `hydra_configs/policy/algorithm/bc_with_vae_gaussian.yaml`.

The `_self_` entry at the end means that any values defined directly in this file override values from the defaults.

### Config Group Hierarchy

```
hydra_configs/
├── end_to_end_training_runs/    # Complete training configs
│   ├── bowel_retraction/
│   ├── libero_hdf5/
│   └── ...
├── experiment/                  # Experiment settings
├── training/                    # Training hyperparameters
├── task/                        # Task definitions
│   ├── action_space/
│   ├── observation_space/
│   ├── dataloader/
│   └── dataset_schema/
├── policy/                      # Policy components
│   ├── encoding_pipeline/       # Encoder configs
│   ├── decoder/                 # Architecture configs
│   ├── algorithm/               # Algorithm configs
│   └── loss/                    # Loss configs
└── inference/                   # Inference settings
```

### Creating a New Experiment

To create a new experiment, write a YAML file that references existing config blocks and overrides only the parameters that differ:

```yaml
# hydra_configs/end_to_end_training_runs/my_dataset/diffusion.yaml
defaults:
  - /task: base
  - /policy: base
  - /experiment: default
  - /task/dataset_schema: my_dataset
  - /task/dataloader: my_dataset
  - /task/action_space: my_action_space
  - /task/observation_space: stereo_rgb
  - /training: default
  - /policy/encoding_pipeline: stereo_rgb
  - /policy/decoder: dit_block_transformer
  - /policy/algorithm: diffusion
  - /policy/loss: regression_gripper
  - /inference: default
  - _self_

experiment:
  checkpoint_folder: ${checkpoint_dir:my_dataset}

task:
  observation_horizon: 2
  prediction_horizon: 16
```

No need to redefine encoder architectures, optimizer settings, or other parameters that stay at their defaults.

## OmegaConf Typed Dataclasses

Every config file maps to a Python `dataclass` defined in `src/versatil/configs/`. This provides type safety: if a YAML value has the wrong type or a required field is missing, the run fails immediately at startup.

### [`MainConfig`][versatil.configs.main.MainConfig] Structure

The root config composes all top-level sections:

```python
@dataclass
class MainConfig:
    experiment: ExperimentConfig
    task: TaskSpaceConfig
    training: TrainingConfig
    policy: PolicyConfig
    inference: InferenceConfig
    quantization: Any = None
```

### Example: [`PolicyConfig`][versatil.configs.policy.PolicyConfig]

```python
@dataclass
class PolicyConfig:
    _target_: str = "versatil.models.policy.Policy"
    encoding_pipeline: EncodingPipelineConfig = MISSING
    algorithm: Any = MISSING
    decoder: Any = MISSING
    observation_space: ObservationSpaceConfig = "${task.observation_space}"
    action_space: ActionSpaceConfig = "${task.action_space}"
    prediction_horizon: int = "${task.prediction_horizon}"
    observation_horizon: int = "${task.observation_horizon}"
    device: str = "${experiment.device}"
    loss: CompositeLossConfig = MISSING
    validate_loss_keys: bool = True
```

- Fields set to `MISSING` are required and must be provided by a config file or CLI override.
- Fields with defaults are optional and filled automatically during config merging.
- Fields using `${...}` interpolation reference values from other config sections.

### Adding a New Config

When adding a new component:

1. Define a `dataclass` in `src/versatil/configs/`:

    ```python
    @dataclass
    class MyComponentConfig:
        _target_: str = "versatil.models.my_module.MyComponent"
        feature_dim: int = 256
        dropout: float = 0.1
    ```

2. Register it in the ConfigStore (`src/versatil/configs/__init__.py`):

    ```python
    cs.store(group="policy/decoder", name="my_component", node=MyComponentConfig)
    ```

3. Create the YAML file in `hydra_configs/`:

    ```yaml
    # hydra_configs/policy/decoder/my_component.yaml
    _target_: versatil.models.my_module.MyComponent
    feature_dim: 256
    dropout: 0.1
    ```

The ConfigStore validates that YAML values match the dataclass types at startup.

## Interpolation

OmegaConf interpolation allows referencing values from other parts of the config tree using `${}` syntax. This keeps configs DRY and prevents inconsistencies.

### Cross-References

```yaml
policy:
  prediction_horizon: ${task.prediction_horizon}
  observation_space: ${task.observation_space}
  device: ${experiment.device}
```

Changing `task.prediction_horizon` automatically propagates to every field that references it.

### Custom Resolvers

VersatIL registers custom resolvers in `src/versatil/configs/__init__.py` for accessing enum values and environment variables in YAML:

```yaml
# Enum resolvers — access Python Enum.value from YAML
input_keys: ${cameras:LEFT}              # Resolves to Cameras.LEFT.value ("left")
backbone: ${rgb_backbone:RESNET18}       # Resolves to RGBBackboneType.RESNET18.value
activation: ${activation_function:GELU}  # Resolves to ActivationFunction.GELU.value

# Path resolvers — read from environment variables
checkpoint_folder: ${checkpoint_dir:bowel_retraction}  # VERSATIL_CHECKPOINT_DIR/bowel_retraction
zarr_path: ${zarr_dir:my_dataset}                      # VERSATIL_ZARR_DIR/my_dataset
cache: ${cache_dir:}                                    # VERSATIL_CACHE_DIR

# Environment variable resolver
api_key: ${env:WANDB_API_KEY}            # Direct env var access
```

Available enum resolvers are available at `src/versatil/configs/__init__.py`.

## Config Validation

VersatIL validates configurations at multiple levels to catch errors before training begins.

### Type Validation (OmegaConf)

Typed dataclasses catch type mismatches and missing fields at config load time:

```yaml
# This fails at startup: num_epochs expects int
training:
  num_epochs: "one hundred"
```

### Feature Validation ([`ExperimentValidator`][versatil.validation.ExperimentValidator])

The [`ExperimentValidator`][versatil.validation.ExperimentValidator] in `src/versatil/validation.py` validates that encoder outputs match decoder input requirements during model construction:

```python
# src/versatil/validation.py
def validate_decoder_encoder_compatibility(self):
    available_features = self.encoding_pipeline.get_features()
    self.decoder.decoder_input.validate_feature_types(
        available_features=available_features
    )
```

This is simplified -- the full method also validates individual feature keys. It catches feature name mismatches (e.g., decoder expects `"fused_visual"` but no fusion produces it) and feature type mismatches (e.g., decoder expects FLAT features but receives SPATIAL) before any data is loaded.

### Loss Key Validation

When `experiment.validate_loss_keys=true` (default), the experiment validates that all loss module keys correspond to valid action space keys, preventing silent misconfiguration of loss components.

## CLI Usage

### Override Values

```bash
python -m versatil.endpoints.train \
    --config-name end_to_end_training_runs/bowel_retraction/act \
    training.optimizer.lr=1e-4 \
    task.dataloader.batch_size=64 \
    training.num_epochs=200
```

### Swap Config Groups

Replace an entire config group from the command line:

```bash
python -m versatil.endpoints.train \
    --config-name end_to_end_training_runs/bowel_retraction/act \
    policy/loss=regression_gripper_MMD \
    policy/algorithm=diffusion
```

### View Resolved Config

Use Hydra's `--cfg` flag to inspect the final merged config without running training:

```bash
python -m versatil.endpoints.train \
    --config-name end_to_end_training_runs/bowel_retraction/act \
    --cfg job
```
