# Changelog

All notable changes to VersatIL will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

## [0.4.0] - 2026-07-05

### Added
- Vision-language-action policies: `AutoregressiveVLADecoder` (OpenVLA,
  pi0-FAST), `OpenVLAOFTDecoder`, and interleaved-attention decoders for Pi0,
  Pi0.5, and SmolVLA, with LIBERO end-to-end presets for all six. Decoders own
  their VLM backbones (Prismatic, PaliGemma, SmolVLM) through decoder-local
  `vision_language_model` config groups, with LoRA enabled by default.
- `PrismaticVLM` with raw TRI-ML checkpoint loading and DINOv2+SigLIP visual
  prefixes, plus a reusable `DinoV2SigLIPRGBEncoder` and a `VLMEncoder` for
  policies that need image-text embeddings without a generative backbone.
- PEFT LoRA adaptation for HuggingFace language encoders, VLM encoders, timm
  image encoders, and generative VLM backbones.
- Action tokenization split into `ActionDiscretizer` (`fast`, `binned`) and
  `ActionTokenIdMapping` (`identity`, `language_vocabulary`), with uniform and
  quantile binning strategies and a `DiscreteDecoder` base for tokenized-action
  decoders.
- Quantization-aware training through torchao's eager `QATConfig`, with
  ready-to-compose `qat_int8_dynamic_intx_int4`, `qat_int4_weight_only`, and
  `qat_int2_weight_only` presets and QAT-aware checkpoint loading.
- Standalone quantization workflows (`none`, `eager`, `pt2e`) with per-module
  targets, an XNNPACK PT2E backend, and an `ExecutorchXNNPACKBackend` that
  lowers exported programs to ExecuTorch `.pte` artifacts.
- Explainability package (`versatil.explainability`): Grad-CAM, Grad-CAM++,
  and Ablation-CAM attribution over dataset samples or live inference, with
  per-decoder explanation targets, heatmap rendering, and the
  `versatil.endpoints.explain` endpoint.
- `RGBCameraMetadata`/`DepthCameraMetadata` with semantic camera-modality
  validation and metadata-driven pixel scaling via `max_pixel_value`.
- `trust_remote_code` support for HuggingFace models that ship custom code,
  and an `infer_constant_prior` flag on `GaussianPrior` for deterministic
  deployment latents.

### Changed
- Every feature crossing the encoding-pipeline boundary carries a canonical
  `(B, T, ...)` layout, even for a single observation frame. Rank alone
  identifies the feature kind (5D spatial, 4D token sequence, 3D vector,
  2D algorithm context); the `has_time_dim` flag and runtime shape guessing
  are gone. Custom encoders and decoders must follow this contract.
- Action keys follow action-space metadata declaration order everywhere
  (policy outputs, tokenization, exported and compressed artifacts),
  replacing the mix of alphabetical and insertion orders.
- Public parameters standardized on explicit long forms
  (`embedding_dimension`, `number_of_heads`, `input_dimension`,
  `hidden_dimension(s)`, `conditioning_dimension`,
  `maximum_sequence_length`, `epsilon`) across layers, decoders, encoders,
  configs, and YAMLs.
- Losses moved to `versatil.metrics.losses` with one module per family.
  Checkpoints saved before this change need the `_target_` paths rewritten
  in their stored `config.yaml`.
- The twelve per-dataset directory resolvers are replaced by
  `${dataset_dir:ENV_VAR,subpath}`.
- The deployment endpoint is Hydra-based and renamed from
  `versatil.endpoints.test` to `versatil.endpoints.deploy`, with client
  settings under the `client.` group shared with online explainability, and a
  `request_timeout_seconds` that raises instead of blocking on a dead server.
- Hydra configs ship inside the wheel (`versatil.hydra_configs`), so the
  documented CLI works for pip installs; training-run recipes ship as
  examples.
- Checkpoints load with `weights_only=True` under an explicit allowlist, so
  third-party checkpoints cannot execute pickled code.
- A zarr replay buffer is rebuilt only when structurally corrupt; key
  mismatches raise unless `recreate_zarr_on_missing_keys` opts in.
- torchao compatibility patches apply in memory through an import hook
  instead of editing installed site-packages files.
- Every config dataclass documents its fields, so the API reference renders
  complete config documentation.
- Dependencies: PyTorch 2.12 (`cu130` index), torchao 0.17, timm 1.0.27,
  transformers 5.9, hydra-core 1.4.0.dev5/omegaconf 2.4.0.dev12 for Python
  3.14. Dev tooling moved to a uv `dev` dependency group; torchaudio removed.

### Fixed
- EMA checkpoints stored averaged weights as the policy state, so resumed
  trainings continued from EMA weights; raw weights are now preserved and
  EMA respects gradient-accumulation boundaries.
- Image normalization broadcast per-channel statistics against the wrong
  axis, producing striped images.
- Padding-aware loss reduction scaled `(B, T, D)` losses by the action
  dimension relative to unmasked losses; both paths now average over valid
  elements.
- Binary gripper inference thresholded the raw logit instead of the sigmoid
  probability, biasing deployment toward the closed state.
- MoE experts ran on unsynchronized CUDA streams, nondeterministically
  corrupting GPU outputs and gradients.
- Geometric attention mixed head and spatial dimensions in a reshape,
  scrambling head contents.
- The DFormerv2 encoder now reproduces the reference implementation exactly
  and loads the official pretrained checkpoints (mirrored on HuggingFace),
  verified tensor-for-tensor against the reference forward.
- Roll-angle action deltas were not wrapped at the ±π discontinuity.
- Fitted action statistics dropped each episode's final row for precomputed
  actions, letting served actions normalize outside the fitted range; action
  padding masks now cover mixed precomputed/on-the-fly spaces.
- Raw dataset import resized every camera to the first camera's resolution;
  each camera now uses its own configured size, and depth clamping at
  inference uses each camera's own normalizer range.
- Per-sample image augmentations drew independent spatial transforms per
  camera; cameras within a sample now share replayed parameters.
- Epoch metrics averaged per-batch instead of per-sample, deflating
  components that appear in only some batches and skewing partial batches.
- `ReduceLROnPlateau` never stepped without a validation loader, was silently
  undone when combined with per-step LR schedules (now rejected), and lost
  its state on checkpoint resume.
- Phase-classification metrics included edge-padded steps.
- `ConditionalUnet1D` injected up-path local conditioning at half temporal
  resolution.
- Variational training leaked decoder-only latent jitter into prior targets
  and logged latents.
- Post-training compression: reports compared quantized models against
  already-mutated baselines, denoising thresholds were dropped from
  artifacts, Conv+BN fusion missed `Sequential`-wrapped pairs, exports
  specialized the batch dimension, compressed eager artifacts never moved
  off CPU, and unstructured pruning defaulted to every weight parameter
  including norm scales and embeddings (now convolution and linear layers).
- Stricter validation throughout: degenerate data (constant depth,
  single-row fits, unordered winsorization quantiles), misconfigured modules
  (invalid routing, temperatures, head counts), and mismatched tensor
  layouts fail loudly instead of training silently wrong.

### Removed
- Legacy experimental decoders `DiscreteDETRActionTransformer`,
  `FreeActionTransformer`, and `MoEFreeActionTransformer`, with their Hydra
  configs.
- Dead public API surface: `Workspace.load_checkpoint`/`predict`,
  `LossOutput.__add__`, the unused `validate_loss_keys` policy-constructor
  parameter, and decorative constructor flags.
- Obsolete torch 2.10/torchao 0.16 source-partition monkey patch for
  X86Inductor PT2E quantization.

## [0.3.0] - 2026-05-11

### Added
- End-to-end experiment config families for PushT, Block Pushing, Kitchen, Multimodal Ant, UR3 Block Push, and synthetic multimodal benchmarks, including state/RGB variants where supported.
- LIBERO/LIBERO+ configs for GPT-style action transformers, Pi0/SmolVLA-style VLA policies, and vision/language encoder sweeps.
- Dataset schema, Zarr metadata, observation/action-space configs, OmegaConf path resolvers, and `.env` variables for the new local/LeRobot-compatible benchmark families.
- Synthetic multimodal benchmark generation, presets, rollout metrics, and training/evaluation configs for mode-recovery experiments.
- Variational latent-modeling components for multimodal action distributions: VQ posterior encoder, uniform/learned codebook priors, DiT latent prior configs, conditional MMD losses, and relaxed conditional Sinkhorn/OT losses.
- Staged training support with epoch-indexed trainability, optimizer, and loss-weight overrides, plus prior-target standardization and richer latent/synthetic rollout callbacks.
- `Pi0Decoder` and `SmolVLADecoder` factories — interleaved VLM-expert joint attention architectures where a pretrained VLM backbone is paired 1:1 with learned expert layers. Pi0 fuses timestep via MLP, Pi0.5 via adaptive normalization. SmolVLA alternates cross-attention and joint self-attention layers.
- `GenerativeVLMEncoder` abstract base for single-stream VLMs (embed images → embed text → concat → LM). Thin subclasses: `PaliGemmaEncoder`, `SmolVLMEncoder`. Replaces the monolithic `multimodal/vlm.py`.
- `TwoTowerVLMEncoder` — CLIP-style separate vision/language towers with `ImageEncoderMixin` + `LanguageEncoderMixin`.
- `ImageEncoderMixin` — abstract base class for multi-camera dispatch with per-camera feature naming (`rgb:left`, `rgb:right`). Subclassed by `RGBEncoderMixin`, `DepthEncoderMixin`, `RGBDEncoderMixin`.
- Per-camera image sizes — encoding pipeline sets dimensions from `CameraMetadata` in observation space. `set_image_size()` hook on encoders.
- `LanguageEncoderMixin` — shared tokenized text pad/truncate, attention mask construction, and output padding mask.
- `FeatureMetadata` frozen dataclass `(key, feature_type, dimension)` with `FeatureType` enum (SPATIAL, SEQUENTIAL, FLAT). Replaces `EncoderOutput`.
- `GenerationCache` / `GenerationLayerCache` — append-only cache for autoregressive generation. Grows token-by-token.
- `ConditioningCache` / `ConditioningLayerCache` — write-once cache for static context (observations, encoder features). Stores K/V and optionally Q for bidirectional conditioning. Cache presence implies behavior — no `use_cache` boolean.
- Cross-attention caching for diffusion decoders — `DiffusionActionTransformer` precomputes conditioning K/V once and reuses across all denoising steps.
- Transformer package decomposed into `attention/`, `block/`, `layer/`, `cache/` sub-packages — ~1600 lines of duplicated diffusion transformer internals deleted.
- `TransformerMixin` — shared weight init, positional encoding setup/application with `offset` for cached generation, padding mask expansion.
- `BlockNormalization` type (`AdaNorm | UnconditionedNorm`) — uniform `(x, condition) → (normed, gate)` interface. Eliminates conditioning branches in transformer blocks.
- `GatedLinearUnit` generalizes `SwiGLU` — base class with configurable gate activation, plus `SwiGLU` (SiLU gate) and `GeGLU` (GELU-tanh gate) subclasses.
- Encoder refactoring — encoders renamed by output format (spatial vs flat), not architecture. `CNNEncoder` + `SwinEncoder` → `SpatialRGBEncoder`, `ViTEncoder` → `FlatRGBEncoder`, `DepthCNNEncoder` → `SpatialDepthEncoder`.
- New backbones: ConvNeXtV2-Nano, TinyViT-21M, DINOv3-ConvNeXt-Small.
- `exclude_cls` → `num_prefix_tokens` on `TokenPoolingHead` — handles CLS + register tokens.
- `ImageProcessor` — per-camera image processing extracted into standalone class (resize, augmentation, normalization).
- `CallbackProvider` protocol for training callbacks — decoders/algorithms declare their own callbacks; Workspace collects via protocol check instead of `isinstance` chains.
- `action_execution_horizon` parameter on `InferenceClient` — controls how many actions from each predicted chunk to execute before re-querying. Defaults to `prediction_horizon`.
- `make_attention_mask` supports `causal_actions` flag and `causal_prefix_suffix_length` for VLA prefix-suffix patterns.
- LR scheduler now uses HuggingFace `transformers.get_scheduler` with `lr_scheduler_kwargs` passthrough.

### Changed
- `ActionTransformer` no longer requires spatial features — accepts any feature type.
- Fusion modules report correct output feature types (SEQUENTIAL when inputs are sequential).
- `LanguageEncoder` detects CLS token via `AutoTokenizer.cls_token_id` instead of hardcoding.
- Encoding pipeline YAML defaults cleaned up: removed redundant `_target_`, added missing config group references.
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

### Fixed
- GPT Action Transformer tokenized head had a 64-dim MLP bottleneck before predicting 1025 vocab classes with weight tying in the same space. Removed bottleneck — direct `embedding_dimension` → vocab projection.
- `DiTPrior` always targeted noise regardless of `prediction_type` — now branches correctly for epsilon, sample, and velocity.
- `GaussianPrior` used stale `self.device` after `.to()` — now uses `_device_tracker` buffer.
- `ConditionalCNNEncoder` pooling head not re-frozen after `set_image_size()` when `frozen=True`.
- EMA callback `optimization_step` incremented per batch instead of per optimizer step — replaced with `trainer.global_step`.
- Inference client discarded predicted action chunks — without temporal aggregation, only first action was sent. Now sends `action_execution_horizon` actions per inference call.
- Post-training compression `validate()` skipped in global mode — iterated empty `self.modules` instead of resolved targets.
- `DictOfTensorMixin.device` raised `StopIteration` when `params_dict` empty — now uses `_device_tracker` buffer.
- DataLoader `persistent_workers=True` crashed with `num_workers=0` — now conditionally set.
- `TemporalAggregator` crashed with `IndexError` when exceeding `max_timesteps` — now raises `RuntimeError`.
- `TrajectoryLengthLoss` zeroed padded timesteps before computing diffs, creating spurious jumps at boundaries.
- `TrajectorySmoothness` padding mask only checked `is_pad[:, 2:]` but acceleration depends on 3 positions.
- `DiscreteDETRActionTransformer` had `requires_actions=True` but ignores actions — changed to `False`.
- `SmolVLADecoder` missing `proprioceptive_projection = None` init.
- Phantom batch in metrics accumulator from lazy module init — reset after dummy forward pass.
- SigLIP2 tokenizer fallback when `attention_mask` not returned.

### Removed
- Legacy encoder files: `rgb/cnn.py`, `rgb/swin.py`, `rgb/vit.py`, `depth/cnn.py` and their tests — superseded by `SpatialRGBEncoder`, `FlatRGBEncoder`, `SpatialDepthEncoder`.

## [0.2.0] - 2026-04-09

### Fixed
- `Policy.compute_loss` was comparing network predictions against raw ground-truth actions instead of the algorithm-specific targets (velocity field for flow matching, noise for diffusion epsilon mode). Introduced `DecodingAlgorithm.get_targets()` so each algorithm provides the correct regression target to the loss module.
- Added `DecodingAlgorithm.predicts_in_action_space` and `BaseLoss.requires_action_space_targets` to detect incompatible loss-algorithm pairings at init.

### Added
- Post-training compression pipeline (`post_training_compression/` + `quantization/`): `PostTrainingCompressor` orchestrates load → prepare → prune → export → quantize → save. Supports per-module `CompressionTarget`, composable pruning (unstructured + structured), PT2E quantization via `X86InductorBackend`, and quantize_() API for dynamic/weight-only quantization. `CompressedPolicyLoader` loads `.pt2` archives with `torch.compile`.
- Python 3.14 compatibility patch for torchao `Union.__module__` assignment.

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
- Composable policy architecture: Policy = EncodingPipeline + Algorithm + Action Decoder + Loss.
- Algorithms: Behavioral Cloning, Diffusion, Flow Matching, VariationalAlgorithm (compositional variational inference wrapping any base algorithm).
- Encoders: RGB/Depth (timm backbones), Conditional CNN (FiLM), DFormerV2, Geometric RGBD, Proprioceptive (MLP), Language (HuggingFace transformers), Vision-Language (HuggingFace multimodal).
- Fusion modules: Concatenation, MLP, Attention.
- Decoder factories: ACT, Action Transformer, Conditional Action U-Net, Diffusion Action Transformer (Cross-Attention and MultiModal), Discrete-DETR, DiT-Block, Free Action Transformer, GPT Action Transformer, LACT, MoDE-ACT, MoE Decoder, MoE Free Action Transformer, Phase-ACT.
- Action heads: Single-Output, Gaussian (mean + log-variance), Mixture of Experts (MoE).
- Data pipeline: Zarr episodic store, CSV/HDF5/LeRobot raw formats, action/observation processing, image augmentation, normalization, tokenization.
- Inference: pluggable transport protocol (ZMQ), observation/action preprocessing, temporal aggregation, unified client for simulation and on-hardware.
- Training: PyTorch Lightning loop, Hydra/OmegaConf configs, WandB tracking, custom callbacks and checkpoint management.
- Losses: composable system via `nn.Module` composition — regression, classification, KL divergence, Sinkhorn divergence, MMD.
- CI/CD: GitLab CI and GitHub Actions, unit and integration tests with >90% coverage, Ruff formatting/linting.
