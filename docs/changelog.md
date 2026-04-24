# Changelog

All notable changes to VersatIL will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

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
- **[`Policy`][versatil.models.policy.Policy] = [`EncodingPipeline`][versatil.models.encoding.pipeline.EncodingPipeline] + Algorithm + Action Decoder + Loss** — composable, config-driven policy design
- **[`EncodingPipeline`][versatil.models.encoding.pipeline.EncodingPipeline]** with hierarchical multi-modal observation encoding and fusion
- **Algorithm/Architecture/Loss separation** — algorithms compose flexibly with action decoder architectures and loss functions

#### Algorithms
- [`BehavioralCloning`][versatil.models.decoding.algorithm.behavior_cloning.BehavioralCloning]
- [`Diffusion`][versatil.models.decoding.algorithm.diffusion.Diffusion]-based action prediction
- [`FlowMatching`][versatil.models.decoding.algorithm.flow_matching.FlowMatching]
- [`VariationalAlgorithm`][versatil.models.decoding.algorithm.variational.VariationalAlgorithm] — compositional variational inference wrapping any base algorithm with posterior encoders and learned/Gaussian priors

#### Encoders
- RGB: Any kind of vision encoder from `timm` library, Custom Conditional CNN (FiLM conditioning)
- Depth: Any kind of CNN from `timm` library, DFormerV2, Custom Geometric Encoder
- Proprioceptive: MLP-based encoder
- Language: Any kind of language encoder from `huggingface transformers` library
- Multimodal: Any kind of vision-language encoder from `huggingface transformers` library

#### Fusion Modules
- Concatenation, MLP, and Attention fusion modules for custom feature fusion

#### Decoder Factories
- [`ACT`][versatil.models.decoding.decoders.factory.act.ACT], [`ActionTransformer`][versatil.models.decoding.decoders.factory.action_transformer.ActionTransformer], [`ConditionalActionUNet`][versatil.models.decoding.decoders.factory.conditional_action_unet.ConditionalActionUNet], [`DiffusionActionTransformer`][versatil.models.decoding.decoders.factory.diffusion_action_transformer.DiffusionActionTransformer] (Cross-Attention and MultiModal variants), [`DiscreteDETRActionTransformer`][versatil.models.decoding.decoders.factory.discrete_detr_action_transformer.DiscreteDETRActionTransformer], [`DiTBlockActionTransformer`][versatil.models.decoding.decoders.factory.dit_block_action_transformer.DiTBlockActionTransformer], [`FreeActionTransformer`][versatil.models.decoding.decoders.factory.free_action_transformer.FreeActionTransformer], [`GPTActionTransformer`][versatil.models.decoding.decoders.factory.gpt_action_transformer.GPTActionTransformer], Latent Action Transformer ([`LACT`][versatil.models.decoding.decoders.factory.lact.LACT]), Mixture-Of-Density Action Transformer ([`MoDE-ACT`][versatil.models.decoding.decoders.factory.mode_act.MixtureOfDensitiesActionTransformer]), [`MoEDecoder`][versatil.models.decoding.decoders.moe.MoEDecoder], [`MoEFreeActionTransformer`][versatil.models.decoding.decoders.factory.moe_free_action_transformer.MoEFreeActionTransformer], [`PhaseACT`][versatil.models.decoding.decoders.factory.phase_act.PhaseACT]

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
