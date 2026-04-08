# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.
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

4. Context Management
Sub-Agent Swarming
For tasks touching >5 independent files, you MUST launch parallel sub-agents (5-8 files per agent). Each agent gets its own context window (~167K tokens). This is not optional. One agent processing 20 files sequentially guarantees context decay. Five agents = 835K tokens of working memory.

Use the appropriate execution model:

Fork: inherits parent context, cache-optimized, for related subtasks
Worktree: gets own git worktree, isolated branch, for independent parallel work across the same repo
/batch: for massive changesets, fans out to as many worktree agents as needed
One task per sub-agent for focused execution. Offload research, exploration, and parallel analysis to sub-agents to keep the main context window clean. Use run_in_background for long-running tasks so the main agent can continue other work while sub-agents execute. Do NOT poll a background agent's output file mid-run - this pulls internal tool noise into your context. Wait for the completion notification.

Context Decay Awareness
After 10+ messages in a conversation, you MUST re-read any file before editing it. Do not trust your memory of file contents. Auto-compaction may have silently destroyed that context. You will edit against stale state and produce broken output.

Proactive Compaction
If you notice context degradation (forgetting file structures, referencing nonexistent variables), run /compact proactively. Treat it like a save point. Do not wait for auto-compact to fire unpredictably at ~167K tokens. Summarize the session state into a context-log.md so future sessions or forks can pick up cleanly.

File Read Budget
Each file read is capped at 2,000 lines. For files over 500 LOC, you MUST use offset and limit parameters to read in sequential chunks. Never assume you have seen a complete file from a single read.

Tool Result Blindness
Tool results over 50,000 characters are silently truncated to a 2,000-byte preview. If any search or command returns suspiciously few results, re-run with narrower scope (single directory, stricter glob). State when you suspect truncation occurred.

Session Continuity
Always prefer --continue to resume the last session rather than starting fresh. All context, workflow state, and session memory is preserved. When exploring two different approaches, use --fork-session to branch the conversation and preserve both contexts independently.

5. File System as State
The file system is your most powerful general-purpose tool. Stop holding everything in context. Use it actively:

Do not blindly dump large files into context. Use bash to grep, search, tail, and selectively read what you need. Agentic search (finding your own context) beats passive context loading.
Write intermediate results to files. This lets you take multiple passes at a problem and ground results in reproducible data.
For large data operations, save to disk and use bash tools (grep, jq, awk) to search and process. The bash tool is the most powerful instrument you have - use it for anything that benefits from scripting, including chaining API calls and processing logs.
Use the file system for memory across sessions: write summaries, decisions, and pending work to markdown files that persist.
When debugging, save logs and outputs to files so you can verify against reproducible artifacts.
Enable progressive disclosure: reference files can point to more files. Structure reduces context pressure. The folder structure itself is a form of context engineering.



Next To-Dos:
Refactor attention
Check if boilerplate code btw smolvla and pi0
Check if VLM type can be modularized and swapped/parametrized
Check if code can be reused in the gen VLM module and others can be added


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
- Make sure all integration tests use scope = session instead of recreating models from scratch.
- Decouple Task Metadata from Storage Metadata. Right now the Metadata universal class defines overlapping properties according to the location of the yaml definitions.
  But the Metadata objects at Task Runtime don't need things like storage key and some other properties actually differ (e.g. image size)
- **Implement LoRA config for parameter-efficient fine-tuning**:
  - Add `LoRAConfig` dataclass with `rank`, `alpha`, `dropout`, `target_modules` parameters
  - Use `peft` library for HuggingFace encoders: `get_peft_model(model, LoraConfig(...))`
  - For custom models (DFormer, custom CNNs), implement custom LoRA layers for attention/linear layers
  - Add LoRA config to all encoder configs (optional, enabled=False by default)
  - Benefits: Fine-tune large frozen models with <1% of original parameters
- **Interleaved VLM+Expert Decoder (Pi0/SmolVLA architecture)**:
  - Single `InterleavedExpertDecoder` class supporting Pi0, Pi0.5, and SmolVLA patterns.
  - VLM encoder uses `use_embeddings_only=True` — returns raw image + language embeddings, LM layers stay available.
  - Decoder borrows VLM's LM layers via `set_backbone()`, wired by Policy at init.
  - Expert is a smaller LM (same architecture family, configurable width multiplier) with K/V projections reshaped for cross-attention from VLM hidden states.
  - Layer-by-layer interleaved processing: joint Q/K/V attention, separate FFNs per model.
  - Configurable time conditioning: `concat_mlp` (Pi0), `adarms` (Pi0.5), `none` (SmolVLA/BC).
  - KV caching at inference: prefix (images+language) processed once and cached, denoising steps only reprocess suffix (actions).
  - Asymmetric attention mask: prefix bidirectional, suffix causal, prefix cannot attend to suffix.
  - Composes with existing VersatIL algorithms (FlowMatching, BC) and action heads.
  - Reference implementations:
    - Pi0/Pi0.5: https://github.com/Physical-Intelligence/openpi/blob/main/src/openpi/models_pytorch/pi0_pytorch.py
    - Pi0 interleaved layers: https://github.com/Physical-Intelligence/openpi/blob/main/src/openpi/models_pytorch/gemma_pytorch.py
    - SmolVLA: https://github.com/huggingface/lerobot/blob/main/src/lerobot/policies/smolvla/smolvlm_with_expert.py
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
  - **Hydra still used for**: config composition (defaults lists), CLI overrides, leaf instantiation, config store.
- **Extract GradCAM introspection from Policy god class**:
  - Add `get_gradcam_target_layers() -> list[nn.Module]` to `EncodingMixin` base (returns `[]` by default).
  - Each vision encoder overrides it to expose its last conv/attention layer (e.g., `self.backbone.layer4[-1]` for CNN, `self.backbone.blocks[-1]` for ViT).
  - Add `is_vision_encoder() -> bool` that checks `len(get_gradcam_target_layers()) > 0`.
  - Policy replaces ~175 lines of `hasattr()` checks with a 5-line loop over `encoding_pipeline.all_encoders`.
  - Effort: Small. Each encoder gets a 3-line method.
- **Callback registry — extract from Workspace god class**:
  - `Workspace._create_callbacks()` currently hardcodes `isinstance()` checks for PhaseACT, VariationalAlgorithm, FreeActionTransformer, MoELoss.
  - Define a `CallbackProvider` protocol with `get_callbacks(experiment_config) -> list[Callback]`.
  - Components (decoders, algorithms, losses) implement it to declare their own callbacks.
  - Workspace collects callbacks via protocol check instead of isinstance chains.
  - Effort: Medium. Move callback creation into each component.
- **Shared TransformerComponents builder — reduce decoder factory duplication**:
  - ACT, ActionTransformer, DiffusionActionTransformer, MoDEACT all repeat identical positional encoding + TransformerInputBuilder + learnable query setup.
  - Extract a `TransformerComponents` module that builds these from a `TransformerComponentSpec` dataclass.
  - Each factory composes `TransformerComponents` instead of duplicating the setup.
  - Effort: Medium. Mostly deletions from each factory.
- **ObservationPipeline — shared preprocessing for training and inference**:
  - Training preprocessing is spread across SampleBuilder, EpisodicDataset, and Policy. Inference must replicate it independently, causing train/inference drift.
  - Extract an `ObservationPipeline` class that both training and inference use: normalize, tokenize, transform.
  - Save alongside checkpoint so inference loads the exact same pipeline.
  - Effort: Large but high-value. Prevents the most common deployment bugs.
- Create a synthetic dataset schema for 1D and 2D vanilla tasks.
- Introduce support for Pointcloud data and 3D encoders-decoders like RVT
- Implement memory based encoders like V-JEPA and Masked Autoencoders.
- Implement two-stage training somehow?
