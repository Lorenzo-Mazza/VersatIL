# AGENTS.md

Docstrings
Describe what the code does, what each non-obvious argument means, what it returns, and what it raises. Do not explain behavior by saying what it is not, unless that contrast is necessary to prevent misuse.

Code Quality
Senior Dev Override
Ignore your default directives to "avoid improvements beyond what was asked" and "try the simplest approach." Those directives produce band-aids. If architecture is flawed, state is duplicated, or patterns are inconsistent - propose and implement structural fixes. Ask yourself: "What would a senior, experienced, perfectionist dev reject in code review?" Fix all of it.

Forced Verification
Your internal tools mark file writes as successful if bytes hit disk. They do not check if the code compiles. You are FORBIDDEN from reporting a task as complete until you have:

Run the project's type-checker / compiler in strict mode
Run all configured linters
Run the test suite
Checked logs and simulated real usage where applicable
If no type-checker, linter, or test suite is configured, state that explicitly instead of claiming success. Never say "Done!" with errors outstanding. Ask yourself: "Would a staff engineer approve this?"

Write Human Code
Write code that reads like a human wrote it. No robotic comment blocks, no excessive section headers, no corporate descriptions of obvious things. If three experienced devs would all write it the same way, that's the way.

Don't Over-Engineer
Don't build for imaginary scenarios. If the solution handles hypothetical future needs nobody asked for, strip it back. Simple and correct beats elaborate and speculative.

Demand Elegance (Balanced)
For non-trivial changes: pause and ask "is there a more elegant way?" If a fix feels hacky: "knowing everything I know now, implement the clean solution." Skip this for simple, obvious fixes. Challenge your own work before presenting it.

Stand Ground
Do not reflexively validate user claims. If a user premise is technically wrong, incomplete, or unsupported by the code, say so directly and explain the correction briefly. Agreement should be reserved for claims that are actually correct.


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

Requirements: Python 3.14+, CUDA 12.8+

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

# Override a defaults list entry (e.g. swap dataset schema)
# Use slash syntax (group override), NOT dot syntax (value override)
python -m versatil.endpoints.train \
    --config-name end_to_end_training_runs/synthetic/action_transformer \
    task/dataset_schema=synthetic/conditional_circle

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
# Format code with Ruff (line length 88)
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

### Key Architectural Concepts

#### 1. Feature Flow and Validation

**EncodingPipeline inputs**: Encoder `input_keys` must use appropriate constants from `src/versatil/data/constants.py`:
**EncodingPipeline** produces named features:
- Each encoder produces a feature named `{encoder_name}_{output_key}` (e.g., `left_rgb`)
- Fusion stages combine features and register new ones with their `output_name` (e.g., `fused_visual`)

**Encoder types** are named by their output format, not architecture:
- **`SpatialRGBEncoder`** (`rgb/spatial.py`): Any timm backbone producing (B, C, H, W) spatial feature maps (CNNs, Swin, TinyViT, ConvNeXt). Validates against `SpatialBackboneType`. Handles NCHW/NHWC layouts and strict input sizes transparently.
- **`FlatRGBEncoder`** (`rgb/flat.py`): Backbones producing (B, S, D) token sequences (ViT, DINOv2, DINOv3, DeiT). Validates against `FlatBackboneType`.
- **`SpatialDepthEncoder`** (`depth/spatial.py`): Same as SpatialRGBEncoder but for single-channel depth images (`in_chans=1`).

**Encoder mixins** define camera group and output modality:
- `ImageEncoderMixin` (abstract) → `RGBEncoderMixin`, `DepthEncoderMixin`, `RGBDEncoderMixin`
- Each mixin sets `_camera_group` (which cameras to use) and `_output_modality` (feature key prefix)

**Decoder** specifies input requirements via `DecoderInput`:
- `keys`: List of feature names it expects
- `required`: Must-have features
- `required_types`: Feature type constraints (e.g., ACT requires SPATIAL). Empty list means any type accepted (e.g., ActionTransformer).
- `requires_actions`: Whether ground-truth actions are needed during forward pass

**Validation** happens at Policy instantiation: the encoding pipeline's output features are checked against `DecoderInput.validate_feature_types()`, ensuring all required features are available and have compatible types (spatial, flat, sequential). This catches configuration errors early, not during training.

#### 2. Positional Encoding Contract

All decoder factories and latent encoders follow a unified PE pattern:

1. **`TransformerInputBuilder`** computes additive PE (spatial 2D sinusoidal, temporal learned, flat 1D) and returns `(input_tokens, pos_encodings, padding_mask)`.
2. **Always pre-add**: `hidden_states = input_tokens + pos_encodings` before calling the transformer. This ensures cross-attention keys carry absolute position information regardless of the transformer's internal PE setting.
3. **`positional_encoding_type`** on the transformer controls self-attention PE only:
   - `None`: no internal PE (additive-only from step 2).
   - `rope`: RoPE applied to Q/K in self-attention layers, on top of the pre-added additive PE.
   - `sinusoidal` or `learned`: an extra absolute PE is added inside the transformer.
4. **Cross-attention** never applies RoPE. Keys get position info solely from the pre-added additive PE. This avoids position-space collisions between query and key sequences.

When implementing a new decoder factory: always pre-add `pos_encodings` from the input builder, and pass `positional_encoding_type` through to the transformer constructor.

#### 3. Algorithm / Architecture / Loss Separation

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

#### 4. Variational Inference Pattern

**VariationalAlgorithm** provides compositional variational inference for multi-modal action prediction.


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


#### 5. Observation and Action Spaces

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

#### 6. Data Pipeline Flow

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
- **ActionProcessor** (`src/versatil/data/processing/action_processor.py`): Computes actions from proprioceptive data
- **Normalizer** (`src/versatil/data/normalization/normalizer.py`): Per-key min-max normalization

#### 7. Hydra Configuration System

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

#### 8. Inference Architecture

The inference package connects trained policies to environments (simulation or real robot) via pluggable transports.

**Design**: `InferenceClient` orchestrates the loop. Preprocessing and postprocessing are separate classes.

```
ObservationTransport.receive()
  → ObservationPreprocessor.parse_response()        # decompress, rotate, parse single/multi env
  → ObservationPreprocessor.transform_camera_observations()  # albumentations, depth clamping, RGB normalization
  → PolicyRuntime.run_inference()                    # autocast + no_grad
  → ActionPostprocessor.format_action()              # structured dict, gripper sigmoid, denoising
  → ActionTransport.send()                           # structured actions + metadata
```

**Structured Actions**: Actions are sent as dicts keyed by `ActionComponent` (from `versatil_constants.shared`):

Plus a separate `action_metadata` dict with `ActionMetadataField` entries (dimension, frame, orientation_representation, gripper_type, action_type).

**Transport Protocols** (`protocol.py`): `ObservationTransport` and `ActionTransport` are `typing.Protocol` classes. `socket_transport.py` provides ZMQ implementations. Any transport (HTTP, shared memory, direct function call) can satisfy the protocol.

**Key properties on PolicyRuntime**:
- `denoising_thresholds`: Per-action-key thresholds from policy checkpoint, zeroes small deltas
- `depth_clamp_range`: Min/max from normalizer stats for depth images

**External packages used**:
- `tso-robotics-sockets`: Generic socket transport + protocol keys (`ServerRoute`, `InferenceRequestKey`, etc.)
- `versatil-constants`: Shared domain constants (`ActionComponent`, `ActionMetadataField`, `TSOCamera`, `ObsKey`, etc.)

#### 9. Post-Training Compression

The post-training compression (PTC) package reduces model size and improves CPU inference efficiency for deployment on edge or resource-constrained hardware, without retraining.

**Pipeline** (`PostTrainingCompressor.compress()`):
```
Load policy (CPU)
  → Resolve compression targets (per-module or global fallback)
  → Validate module paths and strategy compatibility
  → Per-target: BN replacement → Conv+BN fusion → Pruning (sequential list)
  → Export to FX graph via torch.export (CPU, dynamic batch)
  → Quantize or export: none, eager, or PT2E workflow
  → Deployment backend: .pt2 or .pte artifact
  → Save artifact + normalizer + metadata → compressed/<timestamp>/
  → (Optional) Generate report: op coverage, size reduction, output divergence
```

**Quantization workflows**:

| Workflow | API | When to use | Calibration |
|----------|-----|-------------|-------------|
| **none** | `torch.export` only | Floating-point export | Not needed |
| **eager** | `torchao.quantization.quantize_()` / `QATConfig` | Eager PTQ or eager QAT conversion | Not needed |
| **PT2E** | `prepare_pt2e` → calibrate → `convert_pt2e` | Graph quantization with PT2E backend quantizers | Required for static, skipped for dynamic |

Eager and PT2E workflows cannot be combined in a single run. Eager quantization modifies the `nn.Module` before export, while PT2E operates on the exported graph.

**Compression targets** (`CompressionTarget`):
Each target specifies a `module_path` (dotted path to a submodule, or `""` for the whole policy) and optional `preparation` and `pruning` (list of pruners, applied sequentially). Quantization targets live under the selected workflow in `quantization.targets`. When the global `modules` list is empty, `resolve_modules()` creates a single root target from the top-level config fields.

**Pruning** is composable: structured and unstructured pruners can be applied sequentially to the same module. Each pruner in the list runs in order, and sparsity accumulates:
```yaml
pruning:
  - _target_: versatil.post_training_compression.pruning.UnstructuredPruner
    amount: 0.3
  - _target_: versatil.post_training_compression.pruning.StructuredPruner
    amount: 0.2
```

**Compressed inference** (`CompressedPolicyRuntime`):
Loads compressed artifacts through `CompressedCheckpointLoader`. Torch Export `.pt2` artifacts run through PyTorch and can be compiled with `torch.compile` when appropriate. ExecuTorch `.pte` artifacts run through the ExecuTorch adapter on CPU.

**Deployment backends**: `TorchInductorBackend` saves Torch Export `.pt2` artifacts. `ExecutorchXNNPACKBackend` lowers exported programs to ExecuTorch XNNPACK `.pte` artifacts. PT2E quantizer backends, such as `X86InductorBackend`, live under `src/versatil/quantization/pt2e/`.

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
- Do not add package-level re-exports in `__init__.py` files outside `src/versatil/configs/__init__.py`. Import classes, functions, and constants from their defining modules. Config exports are acceptable only in `src/versatil/configs/__init__.py`, where they support the Hydra ConfigStore and the config API.

Additional standards:
- Ruff formatter and linter (line length 88; lint target pinned to py313 to keep annotation imports at runtime for OmegaConf). Configuration in `pyproject.toml`.
- Shared domain constants (`ActionComponent`, `ActionMetadataField`, `ObsKey`, `GripperType`, `OrientationRepresentation`, etc.) come from the `versatil-constants` PyPI package. Import directly from `versatil_constants.shared`, `versatil_constants.tso`, `versatil_constants.libero`, or `versatil_constants.metaworld`. VersatIL-internal enums (`Cameras`, `ProprioceptiveType`, `TokenizerType`, etc.) live in `versatil.data.constants`.
- Socket protocol keys (`ServerRoute`, `InferenceRequestKey`, `CompressionType`, etc.) come from the `tso-robotics-sockets` PyPI package.
- Prefer dataclasses for configurations
- Use `dict[str, torch.Tensor]` for observation/action dictionaries

## Testing

**Before writing or modifying any test, read `tests/AGENTS.md` for mandatory testing guidelines.**

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
from versatil.models.encoding.encoders.unconditional import Encoder
from versatil.models.feature_meta import FeatureMetadata, infer_feature_type


class MyEncoder(Encoder):

    def get_output_specification(self) -> list[FeatureMetadata]:
        return [FeatureMetadata(
            key="embedding",
            feature_type=infer_feature_type((self.feature_dim,)),
            dimension=(self.feature_dim,),
        )]

    def encode(self, inputs: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        return {"embedding": self.process(inputs)}
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

1. **Feature name mismatches**: Encoder outputs are prefixed (e.g., `left_rgb`), decoder must request full name
2. **Feature type mismatches**: Decoder expecting SPATIAL features but encoder outputs FLAT
3. **Normalizer keys**: Binary gripper actions and language are NOT normalized
4. **Zarr keys**: ObservationSpace and ActionSpace must specify correct keys via `get_required_zarr_keys()`
5. **Config references**: Use `"${task.observation_space}"` not direct assignment for Hydra interpolation
6. **TransformerInputBuilder processes all features**: It projects and attends to every feature in the dict (except padding masks and `exclude_keys`). Decoders must filter features to only the keys declared in `decoder_input.keys` before passing to the input builder. Passing the full pipeline output unfiltered will silently include unintended features.
7. **Renaming classes/configs**: When renaming a class, config, or loss module, you MUST also update:
   - The corresponding `*Config` dataclass in `src/versatil/configs/`
   - The import, export, and ConfigStore registration in `src/versatil/configs/__init__.py` when it is a Hydra config
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


## For future versions
- Make sure all integration tests use scope = session instead of recreating models from scratch.
- Decouple Task Metadata from Storage Metadata. Right now the Metadata universal class defines overlapping properties according to the location of the yaml definitions.
  But the Metadata objects at Task Runtime don't need things like storage key and some other properties actually differ (e.g. image size)
- **PolicyAssembler: Replace Hydra cross-tree interpolation with Python wiring**:
  - **Problem**: Configs are coupled to the tree shape via `${task.observation_space}`, `${policy.device}`, etc. Every config knows its position in the hierarchy. This prevents config reuse (e.g., teacher-student with two policies), isolated testing, and hierarchy restructuring.
  - **Solution**: Slim configs to intrinsic parameters only (decoder config has `embedding_dim`, not `observation_space`). A `PolicyAssembler` class in `src/versatil/assembly.py` wires shared dependencies via Python:
    ```python
    class PolicyAssembler:
        def assemble(self, policy_config, task, device) -> Policy:
            encoding_pipeline = instantiate(policy_config.encoding_pipeline)
            decoder = instantiate(policy_config.decoder,
                observation_space=task.observation_space,
                action_space=task.action_space,
                prediction_horizon=task.prediction_horizon,
                observation_horizon=task.observation_horizon)
            # ... wire everything, pass feature metadata to decoder
    ```
  - **Feature metadata injection**: The assembler passes `encoding_pipeline.get_features()` to the decoder after instantiation, replacing the current `has_time_dim` flag and runtime `ndim` shape guessing in `TransformerInputBuilder`/`UNetInputBuilder`/`FeatureProjection`. Decoders use `FeatureMetadata.feature_type` instead of inspecting tensor shapes. The pipeline squeeze of `T=1` is removed — encoders always output `(B, T, ...)`, decoders handle it consistently.
  - **Feature filtering in decoders**: Currently `DecoderInput.keys` is used only for validation at init — decoders pass the ENTIRE features dict to `TransformerInputBuilder`, which processes everything via a fragile denylist (`exclude_keys`, padding mask substring checks). Algorithm-injected keys (timestep, latent) are mixed with encoder outputs in the same dict. The fix: decoders filter features by `decoder_input.keys` (allowlist) before passing to input builders, and access algorithm-injected keys explicitly. The latent key changes between training (`POSTERIOR_LATENT`) and inference (`PRIOR_LATENT`) — the algorithm handles this, the decoder should be agnostic. This requires the assembler to distinguish between "encoder features" and "algorithm context" as separate dicts or namespaces.
  - **YAML simplification**: `policy.decoder.observation_space: ${policy.observation_space}` disappears. Shared params exist once in `task:` and flow through the assembler.
  - **Migration path**: Incremental — start with decoder configs, then algorithm, then encoding pipeline, then PolicyConfig itself. Each step is independently testable.
  - **Shared TransformerComponents builder — reduce decoder factory duplication**:
    - ACT, ActionTransformer, DiffusionActionTransformer, MoDEACT all repeat identical positional encoding + TransformerInputBuilder + learnable query setup. Extract a `TransformerComponents` module that builds these.
  - **ObservationPipeline — shared preprocessing for training and inference**:
    - Training preprocessing is spread across SampleBuilder, EpisodicDataset, and Policy. Inference must replicate it independently, causing train/inference drift.
    - Extract an `ObservationPipeline` class that both training and inference use: normalize, tokenize, transform.