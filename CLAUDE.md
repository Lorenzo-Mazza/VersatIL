# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

VersatIL: Imitation Learning framework for robotic manipulation. The codebase provides a modular architecture in `src/versatil/`. All new development should target the versatil package.

**Goal**: Develop all new code in the modular design in `src/versatil/`.

## Environment Setup

```bash
# Create environment (Mamba recommended for faster installation)
mamba env create -f environment.yml
mamba activate versatil
UV_PROJECT_ENVIRONMENT=$CONDA_PREFIX uv sync

# Install pre-commit hooks (required for all contributors)
pre-commit install
```

Requirements: Python 3.13+, CUDA 12.8+

## Common Commands

### Running Tests

```bash
# Run unit tests only (default)
pytest

# Run all tests including integration tests
pytest -m ""

# Run specific test file
pytest tests/data/test_dataloader.py

# Run tests by marker
pytest -m "unit"           # Fast unit tests with mocked dependencies
pytest -m "integration"    # Slower tests with real model downloads
pytest -m "requires_gpu"   # GPU-required tests
pytest -m "not slow"       # Skip slow tests
```

### Training

```bash
# Train with an end-to-end config (Hydra)
python -m versatil.endpoints.train --config-name end_to_end_training_runs/bowel_retraction/act

# Override parameters from CLI
python -m versatil.endpoints.train \
    --config-name end_to_end_training_runs/bowel_retraction/act \
    task.dataloader.batch_size=64 training.optimizer.lr=1e-4

# Resume from checkpoint
python -m versatil.endpoints.train \
    --config-name end_to_end_training_runs/bowel_retraction/act \
    experiment.resume_from=/path/to/checkpoint.ckpt
```

### Code Formatting and Linting

```bash
# Format code with Ruff (line length 88, Python 3.13)
ruff format src/ tests/

# Check formatting
ruff format --check src/ tests/

# Lint
ruff check src/ tests/

# Lint and auto-fix
ruff check --fix src/ tests/
```

## VersatIL Architecture (`src/versatil/`)

The modular design separates concerns into composable components configured via Hydra.

### Core Design Philosophy

**Policy = EncodingPipeline + Action Decoder + Loss**

Where:
- **EncodingPipeline**: Multi-modal observation encoding with hierarchical fusion
- **Decoder**: Algorithm (e.g., diffusion, flow matching) + Architecture (e.g., transformer)
- **Loss**: Composable loss modules

### Directory Structure

```
src/versatil/
├── configs/           # Hydra configuration dataclasses
│   ├── main.py       # MainConfig composes all configs
│   ├── experiment.py # Experiment tracking, checkpointing, WandB
│   ├── training.py   # Optimizer, LR schedule, EMA, gradient clipping
│   ├── policy.py     # Policy = encoding + decoder + loss
│   ├── inference.py  # Inference-specific settings (rotate_images, etc.)
│   ├── loss.py       # Loss composition configs
│   ├── data/         # Data configuration
│   │   ├── task.py           # ActionSpace, ObservationSpace, TaskConfig
│   │   ├── dataloader.py     # Batch size, num workers, augmentation config
│   │   ├── augmentations.py  # Augmentation pipeline config
│   │   ├── metadata.py       # Metadata config dataclasses
│   │   ├── tokenizer.py      # Tokenizer config
│   │   └── raw/              # Raw dataset schema configs
│   │       ├── schema.py
│   │       └── zarr_meta.py
│   ├── encoding/     # Encoder and fusion configs
│   │   ├── pipeline.py       # EncodingPipelineConfig
│   │   ├── encoder.py        # Base encoder configs
│   │   └── fusion.py         # Fusion module configs
│   └── decoding/     # Decoder configs
│       ├── decoder.py        # DecoderConfig
│       ├── algorithm.py      # Algorithm configs (Diffusion, FlowMatching, Variational)
│       ├── action_head.py    # Action head configs (single, gaussian, MoE)
│       └── latent.py         # Latent variable configs (VAE, priors)
│
├── models/           # Neural network implementations
│   ├── policy.py             # Policy orchestrates encoding → decoding → loss
│   ├── encoding/
│   │   ├── pipeline.py       # EncodingPipeline: encoder orchestration + fusion
│   │   ├── encoders/
│   │   │   ├── encoder.py            # Base encoder interface
│   │   │   ├── conditional.py        # ConditionalEncoder (e.g., FiLM)
│   │   │   ├── rgb/                  # timm CNN, HF ViT, Custom Conditional CNN (FiLM)
│   │   │   ├── depth/                # timm CNN, DFormerV2, Custom Geometric Encoder
│   │   │   ├── proprioceptive/       # MLP-based encoder
│   │   │   ├── language/             # HF Transformers language encoders
│   │   │   └── multimodal/           # HF Transformers VLM encoders
│   │   └── fusion/
│   │       ├── base.py               # Base fusion interface
│   │       ├── concat.py, mlp.py, attention.py
│   │       └── constants.py
│   ├── decoding/
│   │   ├── decoders/
│   │   │   ├── base.py               # Base decoder with algorithm + architecture + heads
│   │   │   └── factory/              # Pre-configured decoder factories (ACT, Action Transformer, Conditional Action U-Net, CrossAttention/MMDiT, Discrete-DETR, DiT Block, Free Transformer, GPT, LACT, MoDE-ACT, MoE Free Transformer, Phase-ACT)
│   │   ├── algorithm/
│   │   │   ├── base.py               # Algorithm interface (forward/predict)
│   │   │   ├── behavior_cloning.py
│   │   │   ├── diffusion.py
│   │   │   ├── flow_matching.py
│   │   │   └── variational.py        # VariationalAlgorithm wrapper
│   │   ├── action_heads/             # Action head implementations (single, gaussian, MoE)
│   │   ├── action_masking.py
│   │   └── constants.py
│   ├── layers/               # Reusable layer implementations
│   │   ├── mlp.py, activation.py, swiglu.py, drop_path.py
│   │   ├── transformer/             # Encoder/decoder layers, GPT, bidirectional, KV cache
│   │   ├── positional_encoding/     # Sinusoidal, Learned, Rotary
│   │   ├── pooling/                 # AttentionPooling, SpatialSoftmax
│   │   ├── detr_transformer/        # DETR encoder/decoder
│   │   ├── diffusion_transformer/   # DiT blocks, MMDiT, cross-attention DiT
│   │   ├── geometric_attention/     # Depth-aware attention mechanisms
│   │   ├── free_transformer/        # FreeTransformer, BinaryMapper
│   │   ├── modulation/              # FiLM, AdaLN, conditional residual blocks
│   │   ├── normalization/           # AdaNorm, RMSNorm, FrozenBatchNorm
│   │   ├── denoising/              # Diffusion schedulers, ODE solvers, timestep sampling
│   │   └── convolution/            # Conv1D, depthwise Conv2D
│
├── data/             # Data loading and preprocessing
│   ├── constants.py          # Data keys, enums (re-exports from versatil_constants)
│   ├── metadata.py           # Observation/action metadata classes
│   ├── task.py               # ActionSpace, ObservationSpace
│   ├── episodic_dataset.py   # EpisodicDataset: loads from Zarr
│   ├── dataloader.py         # get_dataloaders() factory
│   ├── sample_builder.py     # SampleBuilder: constructs training samples
│   ├── action_processor.py   # ActionProcessor: computes actions
│   ├── transform.py          # Data transforms
│   ├── transform_builder.py  # Transform pipeline builder
│   ├── augmentation/         # Image augmentation pipeline
│   ├── preprocessing/
│   │   ├── replay_buffer.py          # ReplayBuffer: episode → Zarr
│   │   ├── sampler.py                # Sampling strategies (uniform, balanced)
│   │   ├── create_zarr_from_csv.py   # CSV → Zarr
│   │   ├── create_zarr_from_hdf5.py  # HDF5 → Zarr (Libero/robomimic)
│   │   └── create_zarr_from_lerobot.py # LeRobot → Zarr
│   ├── normalization/
│   │   ├── normalizer.py             # LinearNormalizer
│   │   └── image_normalizer.py       # Image-specific normalization
│   ├── tokenization/                 # Action/observation tokenization
│   │   ├── tokenizer.py, action_tokenizer.py
│   │   ├── binning_tokenizer.py, observation_tokenizer.py
│   └── raw/                  # Raw dataset schemas and metadata
│       ├── zarr_meta.py              # DatasetMetadata (camera mapping validation)
│       └── schemas/                  # Per-format schema definitions (CSV, HDF5, LeRobot)
│
├── inference/        # Inference client and deployment
│   ├── protocol.py           # ObservationTransport, ActionTransport (typing.Protocol)
│   ├── socket_transport.py   # ZMQ socket transport implementations
│   ├── inference_client.py   # Unified client: orchestrates observe → infer → act loop
│   ├── observation_preprocessor.py  # Response parsing, image transforms, depth clamping
│   ├── action_postprocessor.py      # Structured actions, gripper sigmoid, denoising
│   ├── policy_loader.py     # Checkpoint loading, autocast inference, normalizer access
│   ├── observation_buffer.py # Per-environment temporal observation buffer
│   └── temporal_aggregation.py  # Exponential-weighted action averaging
│
├── metrics/          # Loss functions and metrics
│   ├── base.py               # LossOutput dataclass
│   ├── components.py         # Individual loss components (regression, classification)
│   ├── composite.py          # ComposableLoss: weighted sum of components
│   ├── kernels.py            # MMD kernels
│   ├── ot_loss.py            # Optimal transport loss
│   └── accumulators.py       # Metric accumulation for logging
│
├── training/         # Training infrastructure
│   ├── lightning_policy.py   # LightningModule wrapping Policy
│   ├── workspace.py          # Training workspace (checkpoint, logging, dataloaders)
│   ├── constants.py          # PrecisionType, MAP_PRECISION_TO_DTYPE
│   └── callbacks/            # Lightning callbacks
│
├── common/           # Shared utilities
│   ├── tensor_ops.py         # Tensor manipulation helpers
│   ├── dict_of_tensor_mixin.py
│   ├── module_attr_mixin.py
│   ├── omegaconf_ops.py      # OmegaConf resolvers
│   └── set_cache_dir.py      # HuggingFace cache directory
│
├── explain/          # Model explanation (GradCAM, etc.)
│   ├── explainer.py
│   └── constants.py
│
├── endpoints/        # Training and inference entry points
│   ├── train.py              # Hydra training endpoint
│   ├── test.py               # Inference/evaluation endpoint
│   └── explain.py            # Explanation endpoint
│
└── validation.py     # Experiment config validation
```

### Key Architectural Concepts

#### 1. Feature Flow and Validation

**EncodingPipeline inputs**: Encoder `input_keys` must use appropriate constants from `src/versatil/data/constants.py`:
  - **RGB/Depth encoders**: Use `Cameras` enum values
    - `Cameras.LEFT.value` ("left"), `Cameras.RIGHT.value` ("right"), `Cameras.DEPTH.value` ("depth")
    - Example: `input_keys: ${cameras:LEFT}` resolves to `"left"` via Hydra resolver
  - **Proprioceptive encoders**: Use observation key constants
    - `PROPRIO_OBS_ROBOT_FRAME_KEY` ("proprio_robot_frame")
    - `PROPRIO_OBS_CAMERA_FRAME_KEY` ("proprio_camera_frame")
    - `GRIPPER_STATE_OBS_KEY` ("gripper_state_obs")
  - **Language encoders**: Use `LANGUAGE_KEY` ("language_instruction")

**EncodingPipeline** produces named features:
- Each encoder registers output features with dimensions (e.g., `rgb_cnn_features: (C, H, W)`)
- Fusion stages combine features and register new ones (e.g., `fused_visual: (D,)`)
- Features are prefixed with encoder name to avoid collisions

**Decoder** specifies input requirements via `DecoderInput`:
- `keys`: List of feature names it expects
- `required`: Must-have features
- `required_types`: Must have at least one feature of type (SPATIAL/SEQUENTIAL/FLAT)
- `requires_actions`: Whether ground-truth actions are needed during forward pass

**Validation** happens at Policy instantiation:
```python
# src/versatil/models/policy.py:97-119
def validate_decoder(self):
    available_features_to_dims = self.encoding_pipeline.get_final_features_to_dimensions()
    # Check all required features are available
    # Validate feature types (spatial, flat, sequential)
    self.decoder.decoder_input.validate_feature_types(
        available_features_to_dims=available_features_to_dims
    )
```

This ensures configuration errors are caught early, not during training.

#### 2. Algorithm / Architecture / Loss Separation

**Algorithm** defines the learning paradigm (how to train/predict):
- Behavioral Cloning: direct supervision
- Diffusion: iterative denoising
- Flow Matching: continuous normalizing flows

**Architecture** defines the neural network structure:
- Transformer, MLP, UNet, DETR, etc.

**Loss** defines the training objective (composable loss modules).

They compose independently:
```python
Decoder(
    algorithm=DiffusionAlgorithm(...),
    architecture=TransformerArchitecture(...),
)
```

#### 3. Variational Inference Pattern (NEW)

**VariationalAlgorithm** provides compositional variational inference for multi-modal action prediction.

**Design**: Any algorithm can be wrapped with variational latent variables:
```python
VariationalAlgorithm(
    base_algorithm=<any algorithm>,    # BC, FlowMatching, Diffusion, etc.
    posterior_encoder=<latent encoder>, # q(z|a,s) - encodes actions during training
    prior=<latent prior>,               # p(z|s) - samples latents during inference
)
```

**Components**:
- **Posterior Encoder** (e.g., VAETransformerEncoder): Encodes actions into latent z during training via q(z|a,s)
- **Prior** (GaussianPrior or DiTPrior): Samples latent z during inference via p(z|s)
  - `GaussianPrior`: Simple N(0,I) prior (auto-created if prior=None)
  - `DiTPrior`: Learned diffusion-based prior (more expressive)
- **Base Algorithm**: Any decoding algorithm (BC, FlowMatching, Diffusion, etc.)

**Training Flow**:
1. Encode actions via posterior: z ~ q(z|a,s)
2. Train prior to match posterior samples (if learned prior)
3. Augment features with latent: features = {**features, 'latent': z}
4. Decode actions via base algorithm: p(a|z,s)

**Inference Flow**:
1. Sample latent from prior: z ~ p(z|s)
2. Augment features with latent
3. Decode actions via base algorithm: p(a|z,s)

**Example Combinations**:
```python
# BC with VAE (replaces old BehavioralCloning(latent_encoder=VAE))
VariationalAlgorithm(
    base_algorithm=BehavioralCloning(),
    posterior_encoder=VAETransformerEncoder(...),
    prior=None  # Auto-creates GaussianPrior
)

# Flow Matching with learned prior (replaces VariationalFlowMatching)
VariationalAlgorithm(
    base_algorithm=FlowMatching(...),
    posterior_encoder=VAETransformerEncoder(...),
    prior=DiTPrior(...)
)

# NEW: Variational Diffusion (previously impossible)
VariationalAlgorithm(
    base_algorithm=Diffusion(...),
    posterior_encoder=VAETransformerEncoder(...),
    prior=DiTPrior(...)
)
```

**Configs** (`hydra_configs/policy/algorithm/`):
- `bc_with_vae_gaussian.yaml`: BC + VAE + Gaussian prior
- `bc_with_learned_prior.yaml`: BC + VAE + DiT prior

**⚠️ IMPORTANT - No Backward Compatibility**:
The old variational APIs have been **completely removed** (no deprecation warnings):
- ❌ `BehavioralCloning(latent_encoder=...)` - Removed
- ❌ `VariationalFlowMatching` class - Removed
- ❌ `VariationalFlowMatchingConfig` - Removed
- ✅ Use `VariationalAlgorithm` for all variational inference

All algorithms are now **pure** (no latent variables). Use `VariationalAlgorithm` wrapper for variational inference.

#### 4. Observation and Action Spaces

**TaskConfig** defines what data the experiment uses at runtime:

**ObservationSpace** (`src/versatil/data/task.py:74-104`):
- Which cameras to use (RGB/depth)
- Whether to use proprioceptive data (robot/camera frame)
- Language instructions
- Gripper state
- Returns required Zarr keys via `get_required_zarr_keys()`

**ActionSpace** (`src/versatil/data/task.py:18-70`):
- Position (dim, camera frame vs robot frame)
- Orientation (representation: roll/euler/quaternion)
- Gripper (binary vs continuous)
- Whether to predict deltas or absolute poses
- Whether task has phases (for PhaseACT)
- Returns required Zarr keys and total action dimension

#### 5. Data Pipeline Flow

```
Raw Episodes (CSV)
  → ReplayBuffer.create_zarr()
  → Zarr Dataset (.zarr files)
  → EpisodicDataset.__getitem__()
  → SampleBuilder.build_sample()
    - ActionProcessor computes actions
    - AugmentationPipeline applies transforms
    - Padding masks computed
  → DataLoader (batching, normalization)
  → Policy
```

**Key Classes**:
- **ReplayBuffer** (`src/versatil/data/preprocessing/replay_buffer.py`): Converts episodes to Zarr
- **EpisodicDataset** (`src/versatil/data/episodic_dataset.py`): Loads temporal windows from Zarr
- **SampleBuilder** (`src/versatil/data/sample_builder.py`): Constructs samples with obs/action
- **ActionProcessor** (`src/versatil/data/action_processor.py`): Computes actions from proprioceptive data
- **Normalizer** (`src/versatil/data/normalization/normalizer.py`): Per-key min-max normalization

#### 6. Hydra Configuration System

Configs use OmegaConf with variable interpolation:

```python
@dataclass
class PolicyConfig:
    observation_space: ObservationSpace = "${task.observation_space}"  # Reference
    prediction_horizon: int = "${task.prediction_horizon}"
    encoding_pipeline: EncodingPipelineConfig = MISSING  # Must be set
    decoder: DecoderConfig = MISSING
```

Use `hydra.utils.instantiate()` to build objects from configs:
```python
encoder = instantiate(encoder_config)
```

#### 7. Inference Architecture

The inference package connects trained policies to environments (simulation or real robot) via pluggable transports.

**Design**: `InferenceClient` orchestrates the loop. Preprocessing and postprocessing are separate classes.

```
ObservationTransport.receive()
  → ObservationPreprocessor.parse_response()        # decompress, rotate, parse single/multi env
  → ObservationPreprocessor.transform_camera_observations()  # albumentations, depth clamping, RGB normalization
  → PolicyLoader.run_inference()                     # autocast + no_grad
  → ActionPostprocessor.format_action()              # structured dict, gripper sigmoid, denoising
  → ActionTransport.send()                           # structured actions + metadata
```

**Structured Actions**: Actions are sent as dicts keyed by `ActionComponent` (from `versatil_constants.shared`):
```python
{
    "position": [dx, dy, dz],
    "orientation": [roll],
    "gripper": [1.0],
}
```
Plus a separate `action_metadata` dict with `ActionMetadataField` entries (dimension, frame, orientation_representation, gripper_type, action_type).

**Transport Protocols** (`protocol.py`): `ObservationTransport` and `ActionTransport` are `typing.Protocol` classes. `socket_transport.py` provides ZMQ implementations. Any transport (HTTP, shared memory, direct function call) can satisfy the protocol.

**Key properties on PolicyLoader**:
- `denoising_thresholds`: Per-action-key thresholds from policy checkpoint, zeroes small deltas
- `depth_clamp_range`: Min/max from normalizer stats for depth images

**External packages used**:
- `tso-robotics-sockets`: Generic socket transport + protocol keys (`ServerRoute`, `InferenceRequestKey`, etc.)
- `versatil-constants`: Shared domain constants (`ActionComponent`, `ActionMetadataField`, `TSOCamera`, `ObsKey`, etc.)

### Adding New Components

**When adding new components**:
1. Identify the component type (encoder, decoder, fusion, etc.)
2. Create corresponding config dataclass in `src/versatil/configs/`
3. Implement module in `src/versatil/models/` following base class interfaces
4. Add tests in `tests/` with appropriate markers
5. Update this documentation

## Code Style Requirements

- **Always use Google-style docstrings**: Keep concise, avoid LLM patterns (no numbered lists, no excessive words/examples)
- **Add type hints to all function signatures**
- **Never use inline imports** (all imports at module top)
- **Examine whole codebase for context before changes**
- **Minimal comments**: Only for tensor shapes or when logic is non-obvious.
- Use English words as variables, avoid abbreviations.
- Use kwargs in function calls.
- **Never use `**kwargs` or `*args`** in function signatures. Always use explicit named parameters.
- Avoid Assertions and use Raise ... instead.
- Avoid try catch blocks.
- Use double quotes for strings: "foo" and not 'foo'.
- Avoid plain hardcoded strings. Use constant string values through Enum.value

Additional standards:
- Ruff formatter and linter (line length 88, Python 3.13 target). Configuration in `pyproject.toml`.
- Shared domain constants (`ActionComponent`, `ActionMetadataField`, `ObsKey`, `GripperType`, `OrientationRepresentation`, etc.) come from the `versatil-constants` PyPI package. Import directly from `versatil_constants.shared`, `versatil_constants.tso`, `versatil_constants.libero`, or `versatil_constants.metaworld`. VersatIL-internal enums (`Cameras`, `ProprioceptiveType`, `TokenizerType`, etc.) live in `versatil.data.constants`.
- Socket protocol keys (`ServerRoute`, `InferenceRequestKey`, `CompressionType`, etc.) come from the `tso-robotics-sockets` PyPI package.
- Prefer dataclasses for configurations
- Use `dict[str, torch.Tensor]` for observation/action dictionaries

## Testing

**Before writing or modifying any test, read `tests/CLAUDE.md` for mandatory testing guidelines.**

Test structure mirrors source code:
```
tests/
├── conftest.py                      # Shared fixtures (metadata factories, rng, device)
├── data/                            # Mirror versatil.data
│   ├── test_episodic_dataset.py
│   ├── normalize/
│   └── preprocess/
├── models/                          # Mirror versatil.models
│   ├── encoding/
│   └── layers/
└── inference/                       # Mirror versatil.inference
    ├── test_inference_client.py
    ├── test_observation_preprocessor.py
    ├── test_action_postprocessor.py
    ├── test_socket_transport.py
    ├── test_observation_buffer.py
    ├── test_temporal_aggregation.py
    ├── test_policy_loader.py
    └── test_integration.py          # Real ZMQ socket end-to-end tests
```

**Test markers** (defined in `pyproject.toml`):
- `@pytest.mark.unit`: Fast tests with mocked dependencies (default)
- `@pytest.mark.integration`: Slower tests with real model downloads
- `@pytest.mark.slow`: Very slow tests
- `@pytest.mark.requires_gpu`: GPU-required tests

## Implementation Patterns

### Adding a New Encoder

1. **Define config** in `src/versatil/configs/encoding/encoder.py`:
```python
@dataclass
class MyEncoderConfig(EncoderConfig):
    _target_: str = "versatil.models.encoding.encoders.my_encoder.MyEncoder"
    feature_dim: int = 256
```

2. **Implement encoder** in `src/versatil/models/encoding/encoders/my_encoder.py`:

```python
from versatil.models.encoding.encoders.unconditional import Encoder, EncoderOutput


class MyEncoder(Encoder):

    def get_output_specification(self) -> EncoderOutput:
        return EncoderOutput(
            features=["embedding"],
            dimensions={"embedding": self.feature_dim}
        )


    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        return {"embedding": self.encode(x)}
```

3. **Add tests** in `tests/models/encoding/test_my_encoder.py`

### Adding a New Decoder Architecture

1. **Implement architecture** inheriting from `Architecture` base class
2. **Define `DecoderInput`** specifying required features and types
3. **Implement forward pass** that processes features dict
4. **Create config** in `src/versatil/configs/decoding/architecture.py`

### Adding a New Algorithm

1. **Inherit from `Algorithm`** (`src/versatil/models/decoding/algorithm/base.py`)
2. **Implement `forward()`** for training (with actions)
3. **Implement `predict()`** for inference (without actions)
4. **Create config** in `src/versatil/configs/decoding/algorithm.py`

## WandB Integration

Set `WANDB_API_KEY` environment variable. The workspace logs:
- Train/val loss curves
- Learning rate schedules
- Gradient norms (pre/post clipping)
- Model-specific metrics (e.g., phase confusion matrices)

## Distributed Training (SLURM)

Not yet supported. Needs to be re-integrated with the current workspace.
Set `export NCCL_P2P_DISABLE=1` to avoid NCCL issues on some clusters.

## Common Pitfalls

1. **Feature name mismatches**: Encoder outputs are prefixed (e.g., `rgb_encoder_features`), decoder must request full name
2. **Feature type mismatches**: Decoder expecting SPATIAL features but encoder outputs FLAT
3. **Normalizer keys**: Binary gripper actions and language are NOT normalized
4. **Zarr keys**: ObservationSpace and ActionSpace must specify correct keys via `get_required_zarr_keys()`
5. **Config references**: Use `"${task.observation_space}"` not direct assignment for Hydra interpolation
6. **Renaming classes/configs**: When renaming a class, config, or loss module, you MUST also update:
   - The corresponding `*Config` dataclass in `src/versatil/configs/`
   - The `__init__.py` exports in both `src/versatil/configs/` and relevant model packages
   - The ConfigStore registration in `src/versatil/configs/__init__.py`
   - **ALL YAML files** in `hydra_configs/` that reference the old name (use `grep -r "OldName" hydra_configs/`)
   - Rename YAML files if the filename contains the old name

## TODOs
Fixes:
- Add input shape validation to `EncodingMixin` — all image encoders silently accept wrong-dimensioned tensors (e.g. no batch dim). Add a shared `_unpack_temporal` method that validates 4D/5D and handles the `(B*T, C, H, W)` reshape, replacing the duplicated `if img.dim() == 5` pattern in every encoder's `forward`.
Extensions:
- The explainer is buggy and hardcoded. It needs a refactoring to fit into the new architecture as modular component: the explain endpoint should be agnostic of the data format (right now it assumes CSV Schema).
- Distributed training needs to be re-integrated with the new workspace.
- Quantize package needs to be developed.
- Integrate history buffer of proprioceptive observation only + uniform masking for causal confusion (https://arxiv.org/pdf/1905.11979)
- Introduce pre-commit hooks.
- Update README Code Style section to reference Ruff instead of Black.

## For future versions
- **Implement LoRA config for parameter-efficient fine-tuning**:
  - Add `LoRAConfig` dataclass with `rank`, `alpha`, `dropout`, `target_modules` parameters
  - Use `peft` library for HuggingFace encoders: `get_peft_model(model, LoraConfig(...))`
  - For custom models (DFormer, custom CNNs), implement custom LoRA layers for attention/linear layers
  - Add LoRA config to all encoder configs (optional, enabled=False by default)
  - Benefits: Fine-tune large frozen models with <1% of original parameters
- Create a synthetic dataset schema for 1D and 2D vanilla tasks. 
- Introduce support for Pointcloud data and 3D encoders-decoders like RVT
- Implement memory based encoders like V-JEPA and Masked Autoencoders.
- Implement two-stage training somehow?
