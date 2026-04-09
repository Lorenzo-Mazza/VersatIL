# Changelog

All notable changes to VersatIL will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added

#### VLA Decoder Support (Pi0, SmolVLA)
- **Pi0Decoder** and **SmolVLADecoder** factories — interleaved VLM-expert joint attention architectures where a pretrained VLM backbone is paired 1:1 with learned expert layers. Pi0 fuses timestep via MLP, Pi0.5 via adaptive normalization. SmolVLA alternates cross-attention and joint self-attention layers.
- **`GenerativeVLMEncoder`** abstract base for single-stream VLMs (embed images → embed text → concat → LM). Thin subclasses: `PaliGemmaEncoder`, `SmolVLMEncoder`. Replaces the monolithic `multimodal/vlm.py`.
- **`TwoTowerVLMEncoder`** — CLIP-style separate vision/language towers with `ImageEncoderMixin` + `LanguageEncoderMixin`.

#### Multi-Camera Encoding
- **`ImageEncoderMixin`** — shared multi-camera dispatch with dotted feature naming (`rgb.left`, `rgb.right`). Encoders implement `_encode_single_image()`; the mixin handles iteration, resize, and feature registration. Mixed into CNN, ViT, Swin, ConditionalCNN, TwoTowerVLM.
- **Per-camera image sizes** — encoding pipeline sets dimensions from `CameraMetadata` in observation space, not hardcoded in encoder configs. `set_image_size()` hook on encoders.
- **`LanguageEncoderMixin`** — shared tokenized text pad/truncate, attention mask construction, and output padding mask. Mixed into LanguageEncoder, TwoTowerVLM, PaliGemma, SmolVLM.

#### Feature Metadata
- **`FeatureMetadata`** frozen dataclass `(key, feature_type, dimension)` with `FeatureType` enum (SPATIAL, SEQUENTIAL, FLAT). Replaces `EncoderOutput`. Travels from encoder through fusion to decoder validation — explicit typing over runtime shape guessing.

#### KV Cache by Information Role
- **`GenerationCache` / `GenerationLayerCache`** — append-only cache for the main sequence during autoregressive generation. Grows token-by-token.
- **`ConditioningCache` / `ConditioningLayerCache`** — write-once cache for static context (observations, encoder features). Stores K/V and optionally Q for bidirectional conditioning (Pi0 joint attention). Mechanism-agnostic: works for cross-attention, joint attention, and prefix caching.
- Cache presence implies behavior — no `use_cache` boolean. Passing `GenerationCache` triggers caching, `None` means no caching.
- **Cross-attention caching for diffusion decoders** — `DiffusionActionTransformer` precomputes conditioning K/V once and reuses across all denoising steps.

#### Transformer Package
- **Decomposed into `attention/`, `block/`, `layer/`, `cache/` sub-packages** — each module has single responsibility. ~1600 lines of duplicated diffusion transformer internals deleted.
- **`TransformerMixin`** — shared weight init (`_init_weights` with overridable `_total_residual_streams` property), positional encoding setup/application with `offset` for cached generation, padding mask expansion. `FreeTransformer` now inherits from it.
- **Attention modules**: `CachedAttention`, `JointAttention`, `JointAttentionBase`, `PrecomputedJointAttention`, `QueryKeyNorm`
- **Blocks**: `SelfAttentionBlock`, `CrossAttentionBlock`, `FeedforwardBlock`, `DualStreamAttentionBlock`, `PrecomputedDualStreamAttentionBlock`, `PrecomputedCrossAttentionBlock`
- **Layers**: `TransformerDecoderLayer`, `TransformerEncoderLayer`, `DualStreamLayer`, `PrecomputedDualStreamLayer`, `PrecomputedKVCrossAttentionLayer`

#### Normalization & Activation
- **`BlockNormalization`** type (`AdaNorm | UnconditionedNorm`) — uniform `(x, condition) → (normed, gate)` interface. `UnconditionedNorm` wraps plain LayerNorm/RMSNorm, ignores condition, returns `gate=1`. Eliminates conditioning branches in transformer blocks.
- **`create_block_normalization`** factory — returns `AdaNorm` when `condition_dim` is set (with optional gating for AdaLN-Zero), `UnconditionedNorm` otherwise.
- **`GatedLinearUnit`** generalizes `SwiGLU` — base class with configurable gate activation, plus `SwiGLU` (SiLU gate) and `GeGLU` (GELU-tanh gate) subclasses.

#### New Encoders
- **`SwinEncoder`** — Swin Transformer via timm with spatial feature map output and configurable pooling.
- **`GeometricRGBDEncoder`** — replaces `LightGeometricEncoder`.

#### Image Processing
- **`ImageProcessor`** — per-camera processing extracted into standalone class (resize, color augmentation, spatial augmentation, normalization). Uses `CameraMetadata` for per-camera interpolation (nearest for depth, bilinear for RGB).

#### Training Infrastructure
- **`CallbackProvider` protocol** — `@runtime_checkable` protocol with `get_callbacks()`. Decoders/algorithms declare their own callbacks; Workspace collects via protocol check instead of `isinstance` chains.
- **Trainer logging** — `train/epoch_time_seconds` and `train/gpu_memory_peak_gb` per epoch.
- **LR scheduler** — now uses HuggingFace `transformers.get_scheduler` with `lr_scheduler_kwargs` passthrough.
- **Action masking** — `make_attention_mask` supports `causal_actions` flag and `causal_prefix_suffix_length` for VLA prefix-suffix patterns.

#### Configs
- New Hydra configs for Pi0, SmolVLA, Swin, PaliGemma, SmolVLM, TwoTowerVLM.
- New OmegaConf resolvers: `vlm_model`, `sample_key`, `time_conditioning`, `token_padding`.

### Changed
- `DataLoaderConfig` no longer hardcodes `image_height`/`image_width` — dimensions come from per-camera `CameraMetadata`.
- `ObservationPreprocessor` uses `ImageProcessor` + `CameraMetadata` instead of raw albumentations transforms.
- `PolicyLoader` reads precision from checkpoint config instead of requiring a `precision` parameter.
- `FlowMatchingConfig` adds `reverse_flow_convention` flag.
- `data/augmentation/` restructured to `data/processing/` — `ActionProcessor` and `TransformBuilder` moved alongside `ImageProcessor`.
- `depth/dformerv2` and `depth/light_geometric` encoders moved to `cross_modal/rgbd/`.
- `multimodal/` package renamed by `cross_modal/`, `vlm.py` is now a `vision_language` package.
- Loss `_target_` paths updated across all YAML configs for metrics package restructure (removed `__init__.py` exports).
- Diffusion transformer high-level classes (`CrossAttentionDiT`, `MMDiTTransformer`, `DiTBlock`) now delegate to shared `transformer/` blocks and layers
  instead of maintaining duplicated implementations.
- All `__init__.py` re-exports removed across transformer, training, and inference packages — consumers import from concrete modules.


## [0.2.0] - 2026-04-09

### Fixed
- **Critical: Flow Matching and Diffusion loss targets** — `Policy.compute_loss` was comparing network predictions against raw ground-truth actions instead of the algorithm-specific targets (velocity field for flow matching, noise for diffusion epsilon mode). This caused flow matching models to learn the wrong objective entirely, producing noisy and unstable rollouts. Introduced `DecodingAlgorithm.get_targets()` so each algorithm provides the correct regression target to the loss module. Behavioral Cloning (default) returns ground-truth actions; Flow Matching returns the target velocity; Diffusion returns noise/sample/velocity depending on `prediction_type`. `VariationalAlgorithm` delegates to its wrapped base algorithm.
- **Loss-algorithm compatibility validation** — Added `DecodingAlgorithm.predicts_in_action_space` property and `BaseLoss.requires_action_space_targets` property to detect incompatible pairings at init (e.g. BCE gripper loss with flow matching). `ExperimentValidator.validate_loss_algorithm_compatibility()` raises on conflict.

### Added
- **Post-training compression pipeline** (`post_training_compression/` + `quantization/`):
  - `PostTrainingCompressor` with `compress()` method orchestrating load → prepare → prune → export → quantize → save
  - `CompressionTarget` for per-module or global compression configuration
  - BatchNorm preparation (frozen BN replacement) and Conv+BN weight folding, device-agnostic
  - Composable pruning: `UnstructuredPruner` (global L1) and `StructuredPruner` (per-channel Lp-norm) applied as a sequential list
  - PT2E quantization via `X86InductorBackend` (static/dynamic) with calibration data provider
  - quantize_() API path for dynamic/weight-only quantization via torchao
  - `ExportablePolicy`: dict→positional tensor wrapper for `torch.export` compatibility
  - `CompressedPolicyLoader`: loads `.pt2` archives with `torch.compile` + backend env activation
  - Timestamped output directories (`compressed/<YYYYMMDD_HHMMSS>/`)
  - Optional `QuantizationReport` (op coverage, size reduction, output divergence)
  - Hydra configs for PTQ under `hydra_configs/end_to_end_ptq/`
- `activate_environment()` on `BasePT2EBackend` for lazy `torch.compile` (env must persist past the compile call)
- Python 3.14 compatibility patch for torchao `Union.__module__` assignment (applied in `versatil/__init__.py`)
- Silenced `httpx` INFO logs and non-writable buffer warnings

### Changed
- `inference/__init__.py` and `inference/policy_loading/__init__.py` no longer re-export submodules (prevents circular imports)
- All consumers import from concrete modules (e.g., `from versatil.inference.policy_loading.float_loader import PolicyLoader`)

## [0.1.1] - 2026-03-20

Migrate to Python 3.13+ and PyTorch 2.10 with CUDA 12.8.

### Changed
- Minimum Python version raised from 3.11 to 3.13
- PyTorch upgraded to 2.10 with CUDA 12.8 (was 2.4 with CUDA 12.4)
- HuggingFace Transformers bumped to 5.x, with compatibility fixes for FAST tokenizer and SigLIP VLM encoder
- NumPy dependency bumped for Python 3.14 compatibility
- Ruff target version updated to `py313`
- `str, enum.Enum` replaced with `enum.StrEnum` (PEP 659, Python 3.11+)
- `torch.load` now uses `weights_only=False` by default to match PyTorch 2.10 behavior
- CI/CD Docker image updated to Python 3.14, pipelines now trigger only on merge requests

### Removed
- `flash-attn` dependency — PyTorch 2.10 SDPA natively dispatches to FlashAttention kernels
- `FLASH_ATTENTION_2` attention implementation type (replaced by `SDPA`)

### Fixed
- `asyncio.get_event_loop()` replaced with `asyncio.run()` for Python 3.14 compatibility in WebP codec
- OpenCV dependency conflict resolved
- CI/CD pipeline no longer passes silently when unit tests fail

## [0.1.0] - 2026-03-19

Initial release of VersatIL — a modular Imitation Learning framework for robotic manipulation.

### Added

#### Core Architecture
- **Policy = EncodingPipeline + Algorithm + Action Decoder + Loss** — composable, config-driven policy design
- **EncodingPipeline** with hierarchical multi-modal observation encoding and fusion
- **Algorithm/Architecture/Loss separation** — algorithms compose flexibly with action decoder architectures and loss functions

#### Algorithms
- Behavioral Cloning
- Diffusion-based action prediction
- Flow Matching
- VariationalAlgorithm — compositional variational inference wrapping any base algorithm with posterior encoders and learned/Gaussian priors

#### Encoders
- RGB: Any kind of vision encoder from `timm` library, Custom Conditional CNN (FiLM conditioning)
- Depth: Any kind of CNN from `timm` library, DFormerV2, Custom Geometric Encoder
- Proprioceptive: MLP-based encoder
- Language: Any kind of language encoder from `huggingface transformers` library
- Multimodal: Any kind of vision-language encoder from `huggingface transformers` library

#### Fusion Modules
- Concatenation, MLP, and Attention fusion modules for custom feature fusion

#### Decoder Factories
- ACT, Action Transformer, Conditional Action U-Net, Diffusion Action Transformer (Cross-Attention and MultiModal variants), Discrete-DETR Action Transformer, DiT-Block Action Transformer, Free Action Transformer, GPT Action Transformer, Latent Action Transformer (LACT), Mixture-Of-Density Action Transformer (MoDE-ACT), MoE Decoder, MoE Free Action Transformer, Phase-ACT

#### Action Heads
- Single-Output head, Gaussian head (mean and log-variance), Mixture of Experts (MoE) head

#### Data Pipeline
- Zarr-based episodic store construction
- Support for CSV (TSO), HDF5, and LeRobot raw data formats
- Action and observation pre-processing
- Image augmentation pipeline 
- Normalization and tokenization pipeline

#### Inference
- Pluggable transport protocol (ZMQ socket implementation)
- Observation and action preprocessing
- Temporal aggregation
- Unified inference client for simulation and on-hardware, through `versatil_constants` and `tso_robotics_sockets` libraries.

#### Training Infrastructure
- PyTorch Lightning training loop
- Configuration management with Hydra and OmegaConf
- WandB for experiment tracking
- Custom training callbacks and checkpoint management

#### Metrics and Losses
- Composable loss system through `torch.nn.Module` composition
- Regression and classification losses
- Probability measures: KL divergence, Optimal Transport / Sinkhorn divergence loss, Maximum Mean Discrepancy (MMD) with configurable kernels

#### CI/CD and testing
- GitLab CI and GitHub Actions pipelines
- Unit and integration test suites with pytest with >90% coverage
- Ruff formatting and linting

### Main Dependencies
- Python 3.13+
- CUDA 12.8
- PyTorch, PyTorch Lightning, Hydra, HuggingFace Transformers/Diffusers, Albumentations, OpenCV, ZMQ