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

### Post-Training Compression

```bash
# Compress a trained checkpoint with default x86 PT2E config
python -m versatil.endpoints.post_training_compress \
    --config-name end_to_end_ptq/unstructured_prune_x86.yaml \
    checkpoint_path=/path/to/training/checkpoint

# Override pruning amount and calibration steps
python -m versatil.endpoints.post_training_compress \
    --config-name end_to_end_ptq/unstructured_prune_x86 \
    checkpoint_path=/path/to/checkpoint \
    calibration_steps=32 \
    generate_report=true

# Run compressed model inference
python -m versatil.endpoints.test \
    --checkpoint_path /path/to/checkpoint/compressed/<timestamp> \
    --device cpu \
    --model_server_address 10.0.0.1 \
    --model_server_port 5556
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
│   ├── post_training_compression.py  # PTC configs (CompressionTargetConfig, PostTrainingCompressorConfig)
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
│   ├── exportable_policy.py  # ExportablePolicy: dict→positional wrapper for torch.export
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
│   ├── policy_loading/       # Policy loaders for float and compressed checkpoints
│   │   ├── base.py                  # BasePolicyLoader: config, tokenizer, shared properties
│   │   ├── float_loader.py          # PolicyLoader: training checkpoint → eager inference
│   │   └── compressed_loader.py     # CompressedPolicyLoader: .pt2 → compiled inference
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
├── post_training_compression/  # Post-training model compression
│   ├── compressor.py          # PostTrainingCompressor: orchestrates full pipeline
│   ├── compression_target.py  # CompressionTarget: per-module prep/prune/quantize config
│   ├── constants.py           # Enums (QuantizationStrategy, CompressionFilename, etc.)
│   ├── export.py              # torch.export utilities (build_example_inputs, export_policy)
│   ├── report.py              # QuantizationReport: size/speed/divergence analysis
│   ├── serialization.py       # Save/load compressed .pt2 checkpoints with metadata
│   ├── preparation/           # Pre-quantization model surgery
│   │   ├── batchnorm.py              # FrozenBN → standard BN replacement
│   │   └── fusion.py                 # Conv2d + BatchNorm2d folding
│   └── pruning/               # Weight pruning strategies
│       ├── base.py                    # BasePruner interface + sparsity computation
│       ├── unstructured.py            # Global L1 unstructured pruning
│       └── structured.py              # Per-layer Ln structured pruning
│
├── quantization/      # torchao quantization bridge
│   ├── strategies.py          # PT2EStrategy, QuantizeApiStrategy
│   ├── quantize.py            # apply_pt2e_quantization, apply_quantize_api
│   ├── calibration.py         # CalibrationDataProvider for static quantization
│   ├── constants.py           # FXNodePattern, QuantizableOperatorType
│   ├── torch_patches.py       # Monkey-patches for torchao/torch bugs
│   └── backends/              # Hardware-specific PT2E backends
│       ├── base.py                    # BasePT2EBackend interface
│       └── x86_inductor.py            # X86InductorBackend: quantizer + lowering + env
│
├── endpoints/        # Training and inference entry points
│   ├── train.py              # Hydra training endpoint
│   ├── test.py               # Inference/evaluation endpoint
│   ├── post_training_compress.py  # Hydra PTC endpoint (thin wrapper)
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

#### 8. Post-Training Compression

The post-training compression (PTC) package reduces model size and improves CPU inference efficiency for deployment on edge or resource-constrained hardware, without retraining.

**Pipeline** (`PostTrainingCompressor.compress()`):
```
Load policy (CPU)
  → Resolve compression targets (per-module or global fallback)
  → Validate module paths and strategy compatibility
  → Per-target: BN replacement → Conv+BN fusion → Pruning (sequential list)
  → Export to FX graph via torch.export (CPU, dynamic batch)
  → Quantize: PT2E (static/dynamic) or quantize_() API
  → Save .pt2 archive + normalizer + metadata → compressed/<timestamp>/
  → (Optional) Generate report: op coverage, size reduction, output divergence
```

**Two quantization paths**:

| Path | API | When to use | Calibration |
|------|-----|-------------|-------------|
| **PT2E** | `prepare_pt2e` → calibrate → `convert_pt2e` | Static quantization, per-module targeting, conv+linear fusion | Required for static, optional for dynamic |
| **quantize_()** | `torchao.quantization.quantize_()` | Dynamic weight-only quantization (e.g., int8 dynamic, int4 weight-only) | Not needed |

PT2E and quantize_() cannot be combined in a single run — PT2E operates on the exported FX graph while quantize_() modifies the eager model.

**Compression targets** (`CompressionTarget`):
Each target specifies a `module_path` (dotted path to a submodule, or `""` for the whole policy) and optional `preparation`, `pruning` (list of pruners, applied sequentially), and `quantization` strategy. When the global `modules` list is empty, `resolve_modules()` creates a single root target from the top-level config fields.

**Pruning** is composable: structured and unstructured pruners can be applied sequentially to the same module. Each pruner in the list runs in order, and sparsity accumulates:
```yaml
pruning:
  - _target_: versatil.post_training_compression.pruning.UnstructuredPruner
    amount: 0.3
  - _target_: versatil.post_training_compression.pruning.StructuredPruner
    amount: 0.2
```

**Compressed inference** (`CompressedPolicyLoader`):
Loads `.pt2` archives, applies `torch.compile` with backend-specific environment (freezing, cpp_wrapper), and runs compiled inference. The backend environment is activated permanently (not via context manager) because `torch.compile` is lazy — the actual inductor compilation happens on the first forward pass.

**Supported backends**: Currently X86InductorBackend for x86 CPUs. Additional torchao-supported backends can be added by implementing `BasePT2EBackend`. The quantize_() API path is backend-agnostic and supports CUDA, though some torchao configs have batch size constraints on CUDA (see [pytorch/ao#2376](https://github.com/pytorch/ao/issues/2376)).

**Known limitations**:
- **PT2E export and calibration must run on CPU**: `torch.export` bakes device metadata (`_to_copy(device='cpu')`, `_assert_tensor_metadata(device='cpu')`) into the FX graph. Moving the prepared model to CUDA causes runtime device mismatches.
- **Dynamic batch dimension**: `torch.export` with `batch=1` specializes to a constant. Always use `batch>=2` for dynamic dims.
- **First inference latency**: `torch.compile` with inductor backend generates and compiles C++ kernels on the first forward pass. Compilation time depends on model size and quantized op count.

**Running PTC**:
```bash
python -m versatil.endpoints.post_training_compress \
    --config-name end_to_end_ptq/unstructured_prune_x86.yaml \
    checkpoint_path=/path/to/training/checkpoint \
    checkpoint_name=last.ckpt
```

Override calibration steps and report generation from CLI:
```bash
python -m versatil.endpoints.post_training_compress \
    --config-name end_to_end_ptq/unstructured_prune_x86.yaml \
    checkpoint_path=/path/to/checkpoint \
    calibration_steps=32 \
    generate_report=true
```

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
- **Never use `object` as a type annotation** for return types or parameters. Use the actual type, a protocol, or a union.

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

1. **Inherit from `DecodingAlgorithm`** (`src/versatil/models/decoding/algorithm/base.py`)
2. **Implement `forward()`** for training (with actions)
3. **Implement `predict()`** for inference (without actions)
4. **Override `get_targets()`** if the loss target differs from raw ground-truth actions (e.g., velocity field for flow matching, noise for diffusion epsilon mode). The default returns ground-truth actions (correct for BC). `Policy.compute_loss` calls `algorithm.get_targets()` to obtain the correct regression targets for the loss module.
5. **Create config** in `src/versatil/configs/decoding/algorithm.py`

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
- ~~Quantize package needs to be developed.~~ **Done**: `post_training_compression/` and `quantization/` packages.
- Migrate from MkDocs to [ProperDocs](https://properdocs.org/) before MkDocs 2.0 breaks all plugins/themes.
- Integrate history buffer of proprioceptive observation only + uniform masking for causal confusion (https://arxiv.org/pdf/1905.11979)
- Introduce pre-commit hooks.
- Update README Code Style section to reference Ruff instead of Black.

## Data Loading Optimization
- **Selective preloading**: Add `preload_keys` parameter to `ReplayBuffer.copy_from_path` to preload only non-image keys (proprio, actions, language) into RAM (~20 MB) while keeping images lazy on disk. Eliminates ~33% of network I/O latency per sample at negligible memory cost. Useful for large datasets that don't fit in RAM.
- **Zarr rechunking**: Current image chunks are `(16, H, W, 3)` = 3.1 MB. For `obs_horizon=1`, rechunking to `(1, H, W, 3)` gives 1.7x faster random reads. Add a `rechunk_for_training` utility that sets optimal chunk_t based on obs_horizon.
- **uint8 transfer**: Keep images as uint8 in dataloader workers, do float conversion after collation on GPU. Reduces IPC data volume by 4x (uint8 vs float32).

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
