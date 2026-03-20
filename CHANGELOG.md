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