# Changelog

All notable changes to VersatIL will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added
- `AutoregressiveVLADecoder`, the VLM-backed autoregressive action-token
  decoder used by the OpenVLA and pi0-FAST presets.
- `OpenVLAOFTDecoder`, a VLM-backed continuous action-chunk decoder with
  OpenVLA-OFT-style action slots, denoising-algorithm support, and joint
  action-head output.
- LIBERO end-to-end VLA presets for OpenVLA, OpenVLA-OFT, pi0-FAST, Pi0,
  Pi0.5, and SmolVLA.
- Decoder-local `vision_language_model` Hydra config groups for Prismatic,
  PaliGemma, and SmolVLM backbones.
- `PrismaticVLM`, including raw TRI-ML Prismatic checkpoint loading,
  Prismatic projector construction, and DINOv2+SigLIP visual prefix support.
- `GenerativeVLM` and `HuggingFaceGenerativeVLM` base classes so raw Prismatic
  VLMs and HuggingFace AutoModel-backed VLMs share only the common generative
  VLM contract.
- `DinoV2SigLIPRGBEncoder`, a reusable flat RGB encoder that concatenates
  DINOv2 and SigLIP patch features.
- `VLMEncoder`, an encoder-pipeline module for policies that need image-text
  VLM embeddings without owning a generative VLA backbone.
- PEFT LoRA adaptation package, config store entries, and presets for
  HuggingFace language encoders, VLM encoders, timm image encoders, and
  generative VLM backbones.
- `RGBCameraMetadata`, `DepthCameraMetadata`, and `CameraModality` for
  semantic camera-modality validation.
- `max_pixel_value` on camera metadata, used by image preprocessing for
  metadata-driven pixel scaling.
- Action-tokenization primitives split into `ActionDiscretizer` implementations
  (`fast`, `binned`) and `ActionTokenIdMapping` implementations (`identity`,
  `language_vocabulary`).
- Shared `BinnedValueDiscretizer` for both observation tokenization and binned
  action-token discretization.
- `DiscreteDecoder` for tokenized-action decoders and
  `AutoregressiveDecoderMixin` with cached generation state for GPT-style and
  VLA autoregressive decoders.
- Action-head layouts (`none`, `component`, `joint`, `vocabulary`) plus
  conditional action heads with adaptive-normalization blocks.
- Quantization-aware training through torchao's eager `QATConfig`:
  `EagerQuantizationWorkflow` handles fake-quant preparation before training
  and conversion afterwards, QAT checkpoints load through their own policy
  context, and `quantization/qat_int8_dynamic_intx_int4`,
  `qat_int4_weight_only`, and `qat_int2_weight_only` presets ship ready to
  compose onto any training config.
- Standalone quantization workflows (`none`, `eager`, `pt2e`) with per-module
  targets, decoupled from pruning and the post-training compression
  orchestrator. Eager targets skip linears whose `in_features` don't divide
  the configured group size, and overlapping module paths are rejected up
  front.
- XNNPACK PT2E quantizer backend and an `ExecutorchXNNPACKBackend` that lowers
  exported programs to ExecuTorch `.pte` artifacts, plus build instructions
  and an optional `executorch` extra on Python 3.13.
- Explainability package (`versatil.explainability`): gradient- and
  ablation-based attribution, activation capture, saliency map construction
  with per-decoder explanation targets, dataset and online observation
  sources, heatmap rendering, and an `ExplainabilityRunner` exposed through
  the `versatil.endpoints.explain` endpoint with its own Hydra config family.
- Binning strategies on `BinnedValueDiscretizer`: equal-width `uniform` bins
  over a configurable `[min_value, max_value]` range (the default, matching
  common VLA practice) alongside the existing `quantile` strategy. Quantile
  decode now returns per-bin data means, so duplicate bin edges no longer
  collapse decoded values.
- Graceful FAST action-token decoding: out-of-range tokens are clipped and
  DCT coefficient sequences padded or truncated to the expected length, with
  a warning instead of a crash.
- `trust_remote_code` option on `LanguageEncoder` and `ObservationTokenizer`
  for models that ship custom HuggingFace code, such as
  nvidia/llama-nemotron-embed. The flag is persisted in the tokenizer state
  dict and enabled in the nemotron LIBERO+ language-sweep config.
- `infer_constant_prior` flag on `GaussianPrior` to sample a constant zero
  latent at deployment while keeping training and validation stochastic.

### Changed
- Public parameter names standardized on explicit long forms across layers,
  decoders, encoders, configs, and YAMLs: `embedding_dimension`,
  `number_of_heads`, `input_dimension`, `hidden_dimension(s)`,
  `conditioning_dimension`, `maximum_sequence_length`, and `epsilon`.
  Parameters mirroring external APIs (torch optimizers/schedulers,
  `nn.MultiheadAttention`, `nn.Embedding`) keep their upstream names.
- `versatil.endpoints.test` renamed to `versatil.endpoints.deploy`; the
  deployment CLI is `python -m versatil.endpoints.deploy`.
- Removed dead API surface before freezing the public interface: the unused
  `validate_loss_keys` policy-constructor parameter (the experiment-level
  validation flag is unchanged), the decorative `autoregressive` flag on
  transformer decoder layers (masks control causality), `FusionInput`
  count fields, `LossOutput.__add__`, `Workspace.load_checkpoint`/`predict`,
  and the unused split-half rotary frequency builder.
- Hydra configs moved into the package (`src/versatil/hydra_configs/`) and
  ship in the wheel; endpoints resolve them through `importlib.resources`, so
  the documented CLI works for pip installs, not just source checkouts. The
  `end_to_end_training_runs/` recipes ship as examples.
- Socket transports accept `request_timeout_seconds` and the deployment
  endpoint gains `--request_timeout`: requests raise `TimeoutError` instead
  of blocking forever on a dead environment server, with the REQ socket
  rebuilt for retries. Requires tso-robotics-sockets >= 0.2.0, now the
  dependency floor.
- A zarr replay buffer is deleted and rebuilt only when structurally corrupt;
  key mismatches raise (opt into rebuilding with
  `task.dataloader.recreate_zarr_on_missing_keys`), and transient load
  failures propagate instead of destroying the store.
- Renamed the previous encoder-pipeline image-text embedding module to
  `VLMEncoder` and moved its Hydra config to
  `policy/encoding_pipeline/encoder/vlm/vlm_encoder`.
- Moved generative VLM components out of the encoding package. VLA decoders now
  depend on dedicated generative language-model components instead of treating
  generative models as ordinary encoders.
- Pi0 and SmolVLA decoders now own their VLM backbones and request raw
  normalized/tokenized observations from `Policy`, instead of consuming VLM
  embeddings produced by the encoding pipeline.
- OpenVLA, OpenVLA-OFT, pi0-FAST, Pi0, Pi0.5, and SmolVLA configs now declare
  their VLM backbones through decoder-local `vision_language_model` config
  groups and default to LoRA-enabled HuggingFace/Prismatic backbones.
- Pi0 and SmolVLA use the generic `policy/encoding_pipeline/proprio` preset for
  proprioceptive features instead of the old LIBERO-specific VLA filename.
- Encoder input specifications now declare semantic camera-modality
  requirements. The validation layer checks those requirements against
  observation-space metadata instead of relying on drift-prone RGB/depth key
  lists inside encoder constructors.
- RGB, depth, RGBD, and VLM encoders now use camera metadata modality checks for
  generic compatibility, while keeping encoder-local validation for
  architecture-specific constraints.
- Action tokenizer configuration now names the action discretizer and token-ID
  mapping separately, so FAST tokens, binned action tokens, identity IDs, and
  language-vocabulary IDs can be mixed without encoding that choice in one
  tokenizer type.
- OpenVLA and pi0-FAST presets use `AutoregressiveVLADecoder` with discrete
  action-token targets; OpenVLA maps binned actions into the Prismatic language
  vocabulary, while pi0-FAST uses FAST action tokens in the VLM vocabulary.
- OpenVLA-OFT uses a joint L1 regression head for LIBERO by default, with
  action-slot head input dimensions validated against the configured slot
  layout.
- Prismatic visual towers are built through the reusable DINOv2+SigLIP RGB
  encoder path instead of duplicating separate tower code inside the VLM.
- `ActionHead` validation is driven by explicit head layouts, separating
  component-wise, joint, vocabulary, and no-head decoder contracts.
- GPT-style and VLA autoregressive decoders share cached-generation control
  flow, while discrete tokenizer/vocabulary concerns live in `DiscreteDecoder`.
- PyTorch dependency upgraded to 2.12.0 using the `cu130` wheel index,
  torchao upgraded to 0.17.0, timm upgraded to 1.0.27, transformers upgraded
  to 5.9.0, torchvision left unpinned against the PyTorch index, and unused
  torchaudio dependency removed.
- Python 3.14 PT2E import workaround kept for the torchao 0.17 wheel because
  the clean wheel still mutates immutable Union aliases at import time.
- hydra-core pinned to 1.4.0.dev5 and omegaconf to 2.4.0.dev12 from PyPI for
  Python 3.14 support, replacing the git direct reference that blocked PyPI
  publishing. Resolver registration migrated from the deprecated
  `register_new_resolver` to `register_resolver`.
- Dev tooling (pytest, pytest-cov, ruff, pre-commit) moved from runtime
  dependencies into a uv `dev` dependency group; install docs updated for
  conda- and venv-based `uv sync` flows.
- Loss modules reorganized into a `versatil.metrics.losses` subpackage with
  one module per loss family, replacing the 2.5k-line `metrics/components.py`.
  `CompositeLoss` and the optimal-transport losses moved in as well, and every
  loss `_target_` path in the Hydra configs was updated. Checkpoints saved
  before this change need the same path rewrite in their stored `config.yaml`
  before they can be re-instantiated.
- Training runs now save checkpoints under the config file name instead of
  mirroring the full `end_to_end_training_runs/...` config path, which nested
  the dataset directory twice. WandB run names keep the full path.
- The OpenVLA LIBERO preset discretizes actions with uniform bins over
  min-max-normalized actions instead of quantile bins.
- `gpt_action_transformer` and `mixture_density_act` decoder configs declare
  grouped-query attention, matching their 8-head/2-KV-head layout.

### Fixed
- DinoV2SigLIP forwarded its resolved torch.dtype to the tower constructors,
  which validate the raw precision string, so every instantiation with a real
  precision setting crashed.
- The PT2E compression report compared the quantized model against its own
  in-place-mutated graph; the float baseline is now a genuine pre-quantization
  copy.
- The DFormerv2 encoder now reproduces the reference implementation
  exactly and loads the official pretrained checkpoints (mirrored at
  https://huggingface.co/bbynku/DFormerv2): rotary encoding rotates by
  flattened raster grid positions with endpoint-spaced frequencies, the
  feed-forward network carries the reference's inner depthwise convolution,
  FFN ratios follow the per-stage [4, 4, 3, 3] schedule, the final stage
  uses full attention, patch merging matches the reference conv+BatchNorm,
  and the depth map keeps its original resolution with per-block
  interpolation. Checkpoint loading translates the reference key names and
  raises when any tensor fails to match, instead of silently training from
  random weights. Checkpoints are downloaded from the HuggingFace mirror
  automatically — ``pretrained_weights`` selects the ImageNet backbone or the
  NYU/SUNRGBD finetuned models, replacing ``checkpoint_path`` — and LoRA
  adapters are supported through ``lora_config``. Verified
  numerically: all 780 reference tensors load and encoder outputs match the
  reference forward to float32 precision.
- PaliGemma prefixes were scaled twice: transformers 5.x moved the Gemma
  sqrt(hidden) embedding scale into the embedding module, so the manual
  multiply on top of it blew text tokens up by the full hidden size and
  image tokens carried a scale the HF reference never applies. Pretrained
  PaliGemma backbones (Pi0, Pi0.5, pi0-FAST, OpenVLA-OFT) now start from
  faithful reference activations.
- Prefix/suffix attention masks reached HF language models as boolean 4D
  tensors, which eager attention adds to logits and effectively ignores;
  they are converted to additive float masks, and fully-visible masks stay
  explicit so HF cannot fall back to causal attention.
- Binary gripper inference thresholded the raw logit at 0.5 instead of the
  sigmoid probability, biasing deployment toward the closed state.
- MoE experts ran on unsynchronized CUDA side streams, nondeterministically
  corrupting outputs and gradients on GPU; experts run sequentially now.
- AttentionFusion crashed on token-sequence inputs, which reach fusion as
  (B, T, S, D) before the pipeline's time squeeze.
- DEFAULT (CLS) pooling is rejected for flat RGB backbones and language
  models without a class token, where it silently returned an arbitrary
  first token; SigLIP timm backbones and CLS-less embedding models need
  AVERAGE or NONE pooling.
- Zero-initialized gated (SwiGLU/GeGLU) modulation projections had
  identically zero gradients, freezing DiT-style conditioning forever;
  only the value branch is zeroed now.
- Rebuilt strict-image-size backbones (Swin) crashed under half-precision
  model dtypes during the post-rebuild shape probe.
- Pi0 silently dropped a configured proprioceptive feature under
  adaptive-norm time conditioning.
- Batched inference observations mixing language lists with time-dimmed
  tensors produced corrupted prompts or crashes in the observation
  tokenizer; padded action chunks could poison the FAST decode shape, and
  pretrained FAST tokenizers saved without one could never decode after
  checkpoint load.
- reduce_lr_on_plateau combined with a per-step LR schedule was silently
  undone every optimizer step; the combination is rejected, plateau state
  survives checkpoint resumes, EMA includes the epoch-final flushed
  optimizer step, and LR schedules no longer end early under gradient
  accumulation.
- Compressed eager-workflow artifacts never moved off CPU at inference;
  overlapping compression targets compounded pruning sparsity silently;
  compression reports compared parameter-only float bytes against
  parameter-plus-buffer quantized bytes.
- Fitted action statistics dropped each episode's final row even for
  precomputed actions, letting served actions normalize outside the
  fitted range.
- Padding-aware loss reduction averaged over valid timesteps while the
  unmasked path averages over elements, which scaled padded `(B, T, D)`
  regression losses by the action dimension relative to their configured
  weights. Both paths now average over valid elements.
- Image normalization produced striped outputs: per-channel stats were
  broadcast against the flattened pixel dimension instead of the channel
  axis. Stat alignment is now shape-aware and raises on unalignable inputs.
- Geometric attention scrambled head contents through a reshape that mixed
  head and spatial dimensions; heads are now independent.
- Fully-unmasked attention masks were converted to `None` as an optimization,
  which HuggingFace decoder-only models interpret as causal masking —
  silently breaking bidirectional prefix attention in VLA backbones. Explicit
  masks are now always passed through.
- Fixed-length autoregressive action generation could emit EOS mid-payload;
  EOS is now excluded from the valid token set when the action token length
  is known.
- Joint dual-stream attention applied QK-normalization after RoPE and
  restarted the secondary stream's positions at zero; normalization now
  precedes rotation and the secondary stream continues the primary stream's
  position space.
- EMA callback saved averaged weights as the policy state without stashing
  the raw weights, so resumed trainings continued from EMA weights. Raw
  weights are now stored under a dedicated checkpoint key and restored on
  resume, and EMA updates respect gradient accumulation boundaries.
- `ReduceLROnPlateau` never stepped for runs without a validation loader; it
  now steps on train epoch ends in that case, and the best-checkpoint monitor
  falls back to `train_loss`.
- Phase-classification metrics included edge-padded steps, inflating the
  final phase in accuracy and confusion matrices.
- Variational training leaked decoder-only latent jitter into prior training
  targets and logged latents; predictions now keep the clean posterior
  sample.
- `ConditionalUnet1D` injected up-path local conditioning at half resolution
  inside the upsampling loop; it now joins after the final upsample at full
  temporal resolution, mirroring the down path.
- Roll angle action deltas were not wrapped, producing ±2π jumps at the
  discontinuity; deltas are wrapped via `arctan2(sin, cos)`.
- Eager PTQ exported the float baseline after `quantize_()` had already
  mutated the policy, so compression reports compared the quantized model
  against itself.
- PT2E export used single calibration batches, so `torch.export` specialized
  the batch dimension to a constant; export now always builds synthetic
  example inputs with batch size 2 and dynamic batch dims.
- Conv+BN fusion missed pairs wrapped in `Sequential` containers and could
  fuse mismatched channel counts.
- Metric accumulation averaged per-component sums over the global batch
  count, deflating components that only appear in some batches.
- Per-sample image augmentations drew independent spatial transforms per
  camera; cameras in one sample now share replayed spatial parameters.
- Observation tokenizer fitting attempted to fit non-numerical and
  non-predicted keys, and dataset statistics ignored episode selection masks.
- Denoising thresholds were dropped during post-training compression; they
  are now serialized into the artifact metadata and restored by the
  compressed-checkpoint loader.

### Removed
- Legacy experimental model classes `DiscreteDETRActionTransformer`, `MoEFreeActionTransformer`,
  `FreeActionTransformer`, and their Hydra configs/tests.
- Obsolete torch 2.10/torchao 0.16 source-partition monkey patch for
  X86Inductor PT2E quantization.

## [0.3.0] - 2026-05-11

### Added
- End-to-end experiment config families for PushT, Block Pushing, Kitchen, Multimodal Ant, UR3 Block Push, Multimodal Peg Transfer, and synthetic multimodal benchmarks, including state/RGB variants where supported.
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
