# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Surg-IL: Imitation Learning framework for robotic manipulation. The codebase is undergoing a **major refactoring** from legacy monolithic code to a new modular architecture in `src/refactoring/`. All new development should target the refactored package.

**Goal**: Refactor all legacy code into the new modular design in `src/refactoring/`.

## Environment Setup

```bash
# Create environment (Mamba recommended for faster installation)
mamba env create -f environment.yml
mamba activate surg-il
UV_PROJECT_ENVIRONMENT=$CONDA_PREFIX uv sync
```

Requirements: Python 3.11+, CUDA 12.4+

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

### Training (Legacy Endpoints - Being Replaced)

```bash
# Current training endpoints (legacy - will be deprecated)
python src/endpoints/diffusion_endpoint.py
python src/endpoints/flow_matching_endpoint.py
python src/endpoints/act_endpoint.py
python src/endpoints/phase_act_endpoint.py

# Training with custom JSON config
python src/endpoints/start_training.py --custom_config_path="/path/to/config.json"

# Distributed training via SLURM
sbatch run_distributed.sh  # Set export NCCL_P2P_DISABLE=1
```

### Code Formatting

```bash
# Format code with Black (line length 88, Python 3.11)
black src/ tests/

# Check formatting
black --check src/ tests/
```

## Refactored Architecture (`src/refactoring/`)

The new modular design separates concerns into composable components configured via Hydra.

### Core Design Philosophy

**Policy = EncodingPipeline + Decoder + Loss**

Where:
- **EncodingPipeline**: Multi-modal observation encoding with hierarchical fusion
- **Decoder**: Algorithm (e.g., diffusion, flow matching) + Architecture (e.g., transformer)
- **Loss**: Composable loss modules

### Directory Structure

```
src/refactoring/
├── configs/           # Hydra configuration dataclasses
│   ├── main.py       # MainConfig composes all configs
│   ├── experiment.py # Experiment tracking, checkpointing, WandB
│   ├── training.py   # Optimizer, LR schedule, EMA, gradient clipping
│   ├── policy.py     # Policy = encoding + decoder + loss
│   ├── inference.py  # Inference-specific settings
│   ├── task/         # Task definitions
│   │   ├── task.py           # ActionSpace, ObservationSpace, TaskConfig
│   │   ├── dataloader.py     # Batch size, num workers, augmentation config
│   │   └── dataset/          # Dataset schema definitions
│   ├── encoding/     # Encoder and fusion configs
│   │   ├── pipeline.py       # EncodingPipelineConfig
│   │   ├── encoder.py        # Base encoder configs
│   │   ├── image.py          # Image encoder configs
│   │   └── fusion.py         # Fusion module configs
│   ├── decoding/     # Decoder configs
│   │   ├── decoder.py        # DecoderConfig
│   │   ├── algorithm.py      # Algorithm configs (Diffusion, FlowMatching, etc.)
│   │   └── architecture.py   # Architecture configs (Transformer, MLP, etc.)
│   └── loss.py       # Loss composition configs
│
├── models/           # Neural network implementations
│   ├── policy.py             # Policy orchestrates encoding → decoding → loss
│   ├── encoding/
│   │   ├── pipeline.py       # EncodingPipeline: encoder orchestration + fusion
│   │   ├── encoders/
│   │   │   ├── encoder.py            # Base encoder interface
│   │   │   ├── conditional.py        # ConditionalEncoder (e.g., FiLM)
│   │   │   ├── rgb/                  # CNN, ViT, ConditionalCNN
│   │   │   ├── depth/                # CNN, DFormerV2, LightGeometric
│   │   │   ├── proprioceptive/       # MLP-based encoders
│   │   │   ├── language/             # Text encoders
│   │   │   └── multimodal/           # VLM encoders
│   │   └── fusion/
│   │       ├── base.py               # Base fusion interface
│   │       ├── concat.py, mlp.py, attention.py
│   │       ├── sequential.py, spatial.py
│   ├── decoding/
│   │   ├── decoder.py        # Decoder: algorithm + architecture + heads
│   │   ├── algorithm/
│   │   │   └── base.py               # Algorithm interface (forward/predict)
│   │   └── architecture/
│   │       ├── base.py               # Architecture + DecoderInput validation
│   │       └── action_chunking_transformer.py
│   ├── layers/               # Reusable layer implementations
│   │   ├── transformer.py, mlp.py
│   │   ├── positional_encoding/      # Sinusoidal, Learned, Rotary
│   │   ├── pooling/                  # AttentionPooling, SpatialSoftmax
│   │   ├── detr_transformer/         # DETR encoder/decoder
│   │   ├── geometric_attention/      # Depth-aware attention mechanisms
│   │   ├── conditional_modulation.py # FiLM layers
│   │   └── ...
│
├── data/             # Data loading and preprocessing
│   ├── episodic_dataset.py   # EpisodicDataset: loads from Zarr
│   ├── dataloader.py         # get_dataloaders() factory
│   ├── sample_builder.py     # SampleBuilder: constructs training samples
│   ├── action_processor.py   # ActionProcessor: computes actions
│   ├── augmentation_pipeline.py  # Image augmentation
│   ├── preprocessing/
│   │   ├── replay_buffer.py  # ReplayBuffer: episode → Zarr
│   │   ├── sampler.py        # Sampling strategies (uniform, balanced)
│   │   └── create_zarr.py    # Zarr creation utilities
│   ├── normalize/
│   │   ├── normalizer.py             # LinearNormalizer
│   │   ├── normalizer_builder.py     # Build normalizer from dataset
│   │   └── image_normalizer.py       # Image-specific normalization
│   └── schemas/              # Dataset schema definitions
│       ├── base.py
│       └── bowel_retraction.py
│
├── common/           # Shared utilities
│   ├── tensor_utils.py       # Tensor manipulation helpers
│   ├── dict_of_tensor_mixin.py
│   ├── module_attr_mixin.py
│   └── set_cache_dir.py      # HuggingFace cache directory
│
├── constants/        # Constants and enums
│   ├── data.py               # Data keys, enums (OrientationRepresentation, GripperType, Cameras)
│   ├── models/
│   │   ├── encoders.py       # Encoder type constants
│   │   ├── decoders.py       # Decoder constants, FeatureType enum
│   │   ├── fusion.py         # Fusion type constants
│   │   └── layers.py         # Layer type constants
│   └── validator.py          # Validation error messages
│
├── workspace.py      # TODO: Refactored training workspace (not implemented)
└── loss.py           # TODO: Composable loss modules (not implemented)
```

### Key Architectural Concepts

#### 1. Feature Flow and Validation

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
# src/refactoring/models/policy.py:41-55
def validate_decoder(self):
    available_features_to_dims = self.encoding_pipeline.get_features_to_dimensions()
    decoder_feature_specifications = self.decoder.architecture.input_specification.feature_keys_to_types
    # Check all required features are available
    # Check feature types match (spatial vs flat vs sequential)
```

This ensures configuration errors are caught early, not during training.

#### 2. Algorithm vs Architecture Separation

**Algorithm** defines the learning paradigm (how to train/predict):
- Behavioral Cloning: direct supervision
- Diffusion: iterative denoising
- Flow Matching: continuous normalizing flows

**Architecture** defines the neural network structure:
- Transformer, MLP, UNet, DETR, etc.

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
- **Prior** (GaussianPrior or DiffusionPrior): Samples latent z during inference via p(z|s)
  - `GaussianPrior`: Simple N(0,I) prior (auto-created if prior=None)
  - `DiffusionPrior`: Learned diffusion-based prior (more expressive)
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
    prior=DiffusionPrior(...)
)

# NEW: Variational Diffusion (previously impossible)
VariationalAlgorithm(
    base_algorithm=Diffusion(...),
    posterior_encoder=VAETransformerEncoder(...),
    prior=DiffusionPrior(...)
)
```

**Configs** (`experiments/policy/algorithm/`):
- `bc_with_vae_gaussian.yaml`: BC + VAE + Gaussian prior
- `bc_with_learned_prior.yaml`: BC + VAE + Diffusion prior
- `variational_diffusion.yaml`: Diffusion + VAE + Diffusion prior

**⚠️ IMPORTANT - No Backward Compatibility**:
The old variational APIs have been **completely removed** (no deprecation warnings):
- ❌ `BehavioralCloning(latent_encoder=...)` - Removed
- ❌ `VariationalFlowMatching` class - Removed
- ❌ `VariationalFlowMatchingConfig` - Removed
- ✅ Use `VariationalAlgorithm` for all variational inference

All algorithms are now **pure** (no latent variables). Use `VariationalAlgorithm` wrapper for variational inference.

#### 4. Observation and Action Spaces

**TaskConfig** defines what data the experiment uses at runtime:

**ObservationSpace** (`src/refactoring/configs/task/task.py:74-104`):
- Which cameras to use (RGB/depth)
- Whether to use proprioceptive data (robot/camera frame)
- Language instructions
- Gripper state
- Returns required Zarr keys via `get_required_zarr_keys()`

**ActionSpace** (`src/refactoring/configs/task/task.py:18-70`):
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
- **ReplayBuffer** (`src/refactoring/data/preprocessing/replay_buffer.py`): Converts episodes to Zarr
- **EpisodicDataset** (`src/refactoring/data/episodic_dataset.py`): Loads temporal windows from Zarr
- **SampleBuilder** (`src/refactoring/data/sample_builder.py`): Constructs samples with obs/action
- **ActionProcessor** (`src/refactoring/data/action_processor.py`): Computes actions from proprioceptive data
- **Normalizer** (`src/refactoring/data/normalize/normalizer.py`): Per-key min-max normalization

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

### Legacy Code (To Be Replaced)

The following are legacy implementations that will be deprecated:

- `src/workspace.py` → Will be replaced by `src/refactoring/workspace.py` (TODO)
- `src/endpoints/*_endpoint.py` → Will use new configs and workspace
- `src/model/` → Models being ported to `src/refactoring/models/`
- `src/dataset/` → Ported to `src/refactoring/data/`
- `src/legacy/` → Old implementations kept for reference
- `src/refactoring/configs/legacy_config.py` → Temporary bridge to old system

**When porting legacy code**:
1. Identify the component type (encoder, decoder, fusion, etc.)
2. Create corresponding config dataclass in `src/refactoring/configs/`
3. Implement module in `src/refactoring/models/` following base class interfaces
4. Add tests in `tests/` with appropriate markers
5. Update this documentation

## Code Style Requirements

From `.github/copilot-instructions.md`:
- **Always use Google-style docstrings**
- **Add type hints to all function signatures**
- **Never use inline imports** (all imports at module top)
- **Examine whole codebase for context before changes**

Additional standards:
- Black formatter (line length 88, Python 3.11 target)
- Use enums from `refactoring.constants.data` for data keys
- Prefer dataclasses for configurations
- Use `Dict[str, torch.Tensor]` for observation/action dictionaries

## Testing

Test structure mirrors source code:
```
tests/
├── conftest.py                      # Shared fixtures
├── data/                            # Mirror src/refactoring/data/
│   ├── test_episodic_dataset.py
│   ├── normalize/
│   └── preprocess/
└── models/                          # Mirror src/refactoring/models/
    ├── encoding/
    └── layers/
```

**Test markers** (`tests/pytest.in`):
- `@pytest.mark.unit`: Fast tests with mocked dependencies (default)
- `@pytest.mark.integration`: Slower tests with real model downloads
- `@pytest.mark.slow`: Very slow tests
- `@pytest.mark.requires_gpu`: GPU-required tests

## Implementation Patterns

### Adding a New Encoder

1. **Define config** in `src/refactoring/configs/encoding/encoder.py`:
```python
@dataclass
class MyEncoderConfig(EncoderConfig):
    _target_: str = "refactoring.models.encoding.encoders.my_encoder.MyEncoder"
    feature_dim: int = 256
```

2. **Implement encoder** in `src/refactoring/models/encoding/encoders/my_encoder.py`:

```python
from refactoring.models.encoding.encoders.unconditional import Encoder, EncoderOutput


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
4. **Create config** in `src/refactoring/configs/decoding/architecture.py`

### Adding a New Algorithm

1. **Inherit from `Algorithm`** (`src/refactoring/models/decoding/algorithm/base.py`)
2. **Implement `forward()`** for training (with actions)
3. **Implement `predict()`** for inference (without actions)
4. **Create config** in `src/refactoring/configs/decoding/algorithm.py`

## WandB Integration

Set `WANDB_API_KEY` environment variable. The workspace logs:
- Train/val loss curves
- Learning rate schedules
- Gradient norms (pre/post clipping)
- Model-specific metrics (e.g., phase confusion matrices)

## Distributed Training (SLURM)

Environment variables parsed by workspace:
- `WORLD_SIZE`: Total processes
- `SLURM_PROCID`: Global rank
- `SLURM_GPUS_ON_NODE`: GPUs per node
- `SLURM_CPUS_PER_TASK`: Workers per GPU

Set `export NCCL_P2P_DISABLE=1` to avoid NCCL issues on some clusters.

## Common Pitfalls

1. **Feature name mismatches**: Encoder outputs are prefixed (e.g., `rgb_encoder_features`), decoder must request full name
2. **Feature type mismatches**: Decoder expecting SPATIAL features but encoder outputs FLAT
3. **Normalizer keys**: Binary gripper actions and language are NOT normalized
4. **Zarr keys**: ObservationSpace and ActionSpace must specify correct keys via `get_required_zarr_keys()`
5. **Config references**: Use `"${task.observation_space}"` not direct assignment for Hydra interpolation

## TODOs
- Check optimizers in training config. I think we can introduce a set of optimizers configs targeting the torch objects and instantiate them w Hydra.
- The explainer looks buggy and hardcoded. It will probably need a huge refactor to fit into the new architecture.
3. Introduce the diffusion transformer and the conditional-unet decoders in the decoders package, based on the old models.diffusion_policy code.
4. Introduce the DiT action decoder.
5. Update the experiment yamls, they are probably not working with the new architecture right now.
7. Refactor the codebase files: old files go into legacy, new files go outside of the refactoring folder. Update the paths accordingly.
8. Add tests for all the layers package.
9. Add tests for all new modules
10. Verify compliance of tests to ruff
11. Introduce pre-commit hooks
12. Write proper readme, changelog, etc.

## For future versions
- Implement a LORA config for finetuning hugging face encoders with low rank adaptation. Implement a custom finetune logic for the Dformer and the custom models.
- Implement memory based encoders like V-JEPA and Masked Autoencoders.
- Implement two-stage training somehow?
