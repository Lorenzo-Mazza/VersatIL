# VersatIL: Imitation Learning for Any Robot Policy

[![pipeline status](https://gitlab.com/nct_tso_public/versatil/badges/main/pipeline.svg)](https://gitlab.com/nct_tso_public/versatil/-/commits/main)
[![coverage report](https://gitlab.com/nct_tso_public/versatil/badges/main/coverage.svg)](https://gitlab.com/nct_tso_public/versatil/-/commits/main)
[![Ruff](https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/astral-sh/ruff/main/assets/badge/v2.json)](https://github.com/astral-sh/ruff)
[![Python 3.13+](https://img.shields.io/badge/python-3.13+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![PyPI](https://img.shields.io/pypi/v/versatil.svg)](https://pypi.org/project/versatil/)
[![Docs](https://img.shields.io/badge/docs-online-blue)](https://lorenzo-mazza.github.io/VersatIL)

![VersatIL Logo](media/VersatIL_logo.png)

### 🤯 The Paradox of Research Code
Have you ever found yourself wondering: *"How would this Robot Policy perform if I simply swapped that ResNet18 for an EfficientNet, or just changed one term in the loss function?"*

So you clone the repo to try it out. You wrestle with a `requirements.txt` from 2018 that demands CUDA 9.0 and a version of PyTorch that seemingly only exists on a floppy disk in a basement. You finally get the environment running, only to discover that the loss function implementation is tightly coupled to a string variable named `"dataset_v2_final_final"` deep in the training loop.


Or perhaps you have wandered through State-Of-The-Art codebases, staring blankly at lines like:
`b = d.unsqueeze(-1).view(b, -1, h//16, w//16).permute(0, 3, 1, 2).contiguous()`
...wondering what unholy things are happening to those poor tensors?

### This ends with VersatIL. ⚡

VersatIL is a modular, composable framework built with PyTorch that decouples the three pillars of imitation learning:
**Data**, **Algorithm**, and **Architecture** into clean, reusable components.

Swap Behavioral Cloning for Diffusion or Flow Matching, replace a ResNet with a ViT or VLM backbone, or run your policy on a completely new dataset format. Compatible components can be freely swapped with config changes, no source code rewrites.

Rapid experimentation, cleaner code, and true reusability across projects.

### Core Principles
- 🧑‍🔬 **Research-First Flexibility** — Unlike frameworks that focus on reimplementing and distributing specific SOTA policies, VersatIL gives you the modular building blocks to **create and benchmark your own novel architectures and algorithms** on any dataset.
- 🔄 **Mix & Match** You are free to swap any robot policy component for easy benchmarking.
- 🧱 **Modularity** Each component is self-contained and reusable.
- ⚡ **Modern Dependency Management** – Dependencies managed with [uv](https://github.com/astral-sh/uv) and `pyproject.toml` for modern and fast installation.
- ♻️ **Don't Reinvent the Wheel** We rely on industry-standard libraries:
    * **[Timm](https://github.com/huggingface/pytorch-image-models)** for vision backbones.
    * **[HuggingFace Transformers](https://github.com/huggingface/transformers)** for Language encoders, VLMs, and tokenizers.
    * **[HuggingFace Diffusers](https://github.com/huggingface/diffusers)** for diffusion schedulers.
    * **[TorchCFM](https://github.com/atong01/conditional-flow-matching)** for Flow Matching schedulers.
    * **[Albumentations](https://albumentations.ai/)** for image augmentations.
    * **[torchao](https://github.com/pytorch/ao)** for post-training quantization (PT2E and quantize_() APIs).
- 💡 **Invent What Matters** For performance-critical components, we wrote a custom `models/layers` package in pure PyTorch. This includes optimized implementations of:
    * [Attention](https://docs.pytorch.org/docs/stable/generated/torch.nn.functional.scaled_dot_product_attention.html) (PyTorch built-in SDPA Flash kernel dispatch).
    * Positional Encodings (Sinusoidal, Learned, Rotary).
    * Transformer variants (DETR, GPT, BERT, Free Transformer).
    * Modular Deep Neural Networks layers such as normalization, modulation, convolution, etc
    * *Note: These are policy-agnostic and reusable in other projects.*
- 🔒 **Explainability & Safety** – Strict interfaces, full type hints, Google-style docstrings, and runtime config validation.
- 🧪 **Testing** – Comprehensive unit and integration tests for every module.


---

## ⏳ Workflow

### 📋 Before Training

 
#### ⚙️ Configuration Management
VersatIL uses **[Hydra](https://hydra.cc/)** and **[OmegaConf](https://omegaconf.readthedocs.io/)** for the management of experiment configurations.

Hydra provides:
- **Hierarchical composition** — Combine reusable config groups via `defaults:` without constant YAML editing.
- **CLI overrides & sweeps** — Change parameters or run hyperparameter searches directly from the command line.

OmegaConf provides:
- **Config Safety:** All configs are typed Dataclasses. If you pass a string where an int is expected, or forget a mandatory field, the run fails immediately at startup—not 2 hours into training.
- **Smart merging** — Config classes define defaults that automatically fill missing values when merging multiple files.

Together, this means you can create new experiments by overriding only the parameters that change — no need to write complete YAML files for every run.

---
#### 🗄️ Dataset Schema (Ingestion)
Raw data formats vary wildly (Rosbags, CSVs, HDF5). We don't force you to convert your raw files manually. \
Instead, VersatIL handles this with a two-stage approach:
1. **DatasetSchema (how your raw data is structured)**  
   A pluggable class that maps any raw format to a standardized **Zarr** store.

   | Schema                                                | Class | Raw Format              |
   |-------------------------------------------------------|---|-------------------------|
   | [HuggingFace LeRobot](https://huggingface.co/lerobot) | `LeRobotDatasetSchemaV30` | Parquet + MP4/images    |
   | HDF5                                                  | `Hdf5DatasetSchema` | HDF5 archive            |
   | CSV                                                   | `CsvDatasetSchema` | CSV + raw image folders |
   | Custom                                                | Subclass `DatasetSchema` | Any                     |
   
2. **Zarr Store Creation**
   Zarr [https://zarr.readthedocs.io/en/stable/]  provides fast, compressed, chunked storage with NumPy-like access.
   - Created **automatically** on first training run if missing — no separate preprocessing script needed.
   - Decouples raw storage from training-optimized layout.
   - Raw keys vs pipeline keys: Raw data formats use their own naming (e.g., LIBERO LeRobot dataset uses `observation.images.image`, LIBERO original HDF5 dataset uses `agentview_image`). During zarr creation, these *raw camera keys* (`RawCameraKey`) are remapped to standardized *pipeline camera keys* (`Cameras`) via `RAW_TO_CAMERA_MAPPING`. After zarr creation, only pipeline keys exist — the rest of the codebase (training, inference, validation) never sees raw format keys. This separation is defined in `src/versatil/data/constants.py` and ensures that adding a new raw data format only requires a new `RawCameraKey` entry and its mapping, with zero changes to the training or inference pipeline.

---


### 🏋️‍♂️ During Training
#### 🎯 Task Definition

The **TaskSpace** selects what subset of the Zarr data to use, allowing multiple tasks from a single dataset without duplication:

- **Observation space** — Choose which observations will be given to the robot policy as state, e.g. which cameras, proprioception, depth, language instructions, etc.
- **Action space** — Choose which ground-truth actions will be given to your robot policy and in which format, e.g. deltas or absolute positions, gripper states, end-effector orientation, etc.
- **Temporal horizons** — Observation and prediction temporal windows.

*Example:* The same Zarr store can power a pure-vision task, a state-only task, and a vision-language task simultaneously.

---
#### 🚚 Data Loading Pipeline
Uniform across all Zarr datasets:
- Fast episodic loading from Zarr
- Temporal chunking (observation windows + action sequences + masks)
- Preprocessing (normalization, augmentation, tokenization)
- Batching via PyTorch DataLoader

Actions can be **precomputed** (stored in Zarr) or computed **on-the-fly** during batching(e.g., deltas from consecutive states).

---



#### 🧠 Policy Composition
A robot policy is built from four decoupled components, orchestrated by the `Policy` class:
1.  👁️ **Encoding Pipeline:** A pipeline of multi-modal encoders that extract features from raw observations plus an optional fusion module that combines the features
into a unified representation.
2.  🧮 **Algorithm:** The learning paradigm that defines how to train the policy. This can be:
- Standard Behavioral Cloning (supervised learning of actions given observations)
- Generative approaches through Denoising Score Matching such as Diffusion and Flow Matching. 
- Variational approaches that add a learned latent variables to any base algorithm. The latent variable can be learned through different kinds of prior-posterior schemes,
 which will determine the nature of the latent space. 
3) 🕹️ **Action Decoder:** The neural architecture that decodes the features into robot actions. This can be a Transformer-based architecture or a UNet-based architecture.
We provide a set of standard decoders such as the Action Chunking Transformer (ACT) or the DiT-Block Policy from the literature. More information on the available decoders can be found below.
We additionally support a Mixture-Of-Experts (MoE) wrapper, which can be used on top of any decoder to copy the architecture across multiple experts and learn a gating network to select which expert to use at inference time.
4.  📉 **Loss Module:** A composable loss module that defines the objective function to optimize during training. This can be a simple regression loss (MSE) or a more complex loss that combines multiple terms (e.g. action regression + KL divergence for variational algorithms).

---


#### ⚡ Training Engine
Powered by **PyTorch Lightning**:
* Automatic handling of loops, distributed training, and checkpointing.
* **WandB Integration:** Tracks metrics, gradients, EMA decay, and latent visualizations.
* **Callbacks:** EMA weights, gradient norm logging, t-SNE plots.

---
### 🚀 Post-Training

#### 🔌 Inference

The inference pipeline is transport-agnostic: communication with any environment server (real robot, simulation, or custom) is abstracted behind `ObservationTransport` and `ActionTransport` Python protocols. Any object satisfying these protocols works — ZMQ, HTTP, etc.

The built-in ZMQ implementation uses our two PyPI packages:
- [**tso-robotics-sockets**](https://pypi.org/project/tso-robotics-sockets/): Generic ZMQ socket client/server with protocol keys (`ServerRoute`, `InferenceRequestKey`, `CompressionType`).
- [**versatil-constants**](https://pypi.org/project/versatil-constants/): Shared domain constants for action/observation message passing (`ActionComponent`, `ActionMetadataField`, `ObsKey`, `GripperType`, `OrientationRepresentation`).

Both libraries are server-agnostic — they define the message format, not the server implementation. Any server that speaks the protocol can be integrated by implementing the transport protocols. 

The built-in ZMQ transport works for both simulation and real hardware — the dataset format is fully decoupled from the transport layer. For custom setups, implement the `ObservationTransport` and `ActionTransport` protocols with any transport mechanism.

##### Simulation Servers

We provide custom ZMQ server wrappers for popular robot learning simulation environments, enabling seamless rollout of VersatIL policies:

| Simulator | Original | ZMQ Server Wrapper |
|---|---|---|
| [LIBERO / LIBERO-PRO](https://github.com/Zxy-MLlab/LIBERO-PRO/tree/master) | [GitHub](https://github.com/Zxy-MLlab/LIBERO-PRO/tree/master) | Coming soon |
| [LIBERO+](https://github.com/sylvestf/LIBERO-plus) | [GitHub](https://github.com/sylvestf/LIBERO-plus) | Coming soon |
| [MetaWorld](https://meta-world.github.io/) | [GitHub](https://github.com/Farama-Foundation/Metaworld) | Coming soon |

---

#### 🔍 Explainability
We provide tools for model interpretability, such as visualization of the feature maps from the trained policy vision encoders.
We currently support Grad-CAM, Grad-CAM++, Ablation-CAM and Integrated Gradients for visual explanations of the model's predictions.

---


#### 📦 Post-Training Compression

VersatIL includes a post-training compression (PTC) pipeline that reduces model size and improves CPU inference efficiency for deployment on edge or resource-constrained hardware where GPU acceleration is unavailable, without retraining.

**What is post-training quantization?**
Post-training quantization (PTQ) converts trained floating-point model weights and activations to lower-precision integer representations (e.g., INT8). This reduces memory footprint, improves cache utilization, and enables hardware-accelerated integer arithmetic — typically achieving inference speedup on x86 CPUs with minimal accuracy loss. Unlike quantization-aware training (QAT), PTQ is done after training. Static quantization uses a small calibration dataset to determine optimal activation ranges per layer; dynamic quantization computes ranges on-the-fly at inference time and needs no calibration.

**How VersatIL implements PTC:**

The compression pipeline is configurable via Hydra and supports three complementary techniques applied sequentially:

1. **Preparation**: Frozen BatchNorm replacement and Conv+BN weight folding — standard pre-quantization model surgery that merges batch normalization parameters into convolution weights.

2. **Pruning**: Weight pruning to introduce sparsity before quantization. Supports both unstructured (global L1 magnitude) and structured (per-channel Lp-norm) pruning, composable as a list — e.g., structured pruning followed by unstructured pruning on the same module.

3. **Quantization**: Two paths via [torchao](https://github.com/pytorch/ao), PyTorch's quantization library:
   - **PT2E** (PyTorch 2 Export): The graph-based quantization flow. The trained policy is exported to an FX graph via `torch.export`, then quantized using hardware-specific quantizers (e.g., X86InductorQuantizer for x86 CPUs). Static quantization requires a calibration pass over training data to determine activation ranges. This path supports per-module targeting, conv+linear fusion, and operator-level quantization control.
   - **quantize_() API**: The eager-mode dynamic quantization flow. Applies weight-only or dynamic activation quantization (e.g., INT8 dynamic, INT4 weight-only) directly on the eager model before export. Simpler to use but less granular than PT2E.

**Compressed inference:**

Compressed models are saved as `.pt2` archives and loaded by `CompressedPolicyLoader`, which applies `torch.compile` with the Inductor backend for optimized execution. The first inference call triggers kernel compilation, after which subsequent calls run at full speed. Compressed checkpoints include the normalizer, tokenizer, training config, and compression metadata for self-contained deployment.

**Per-module targeting:**

Compression targets can be specified globally (applied to the entire policy) or per-module (targeting specific submodules like individual encoder backbones or the decoder). This allows, for example, aggressively quantizing the vision backbones while leaving the language encoder or decoder at higher precision.

**Roadmap:** We plan to extend support to Quantization-Aware Training (QAT), where simulated quantization is inserted into the forward pass during training so the optimizer learns weights that are natively quantization-friendly, yielding higher accuracy than PTQ alone.

---

## 🚀 Quick Start

### Installation

**Prerequisites:**
- Python 3.13+
- CUDA 12.8+ (required for training)
- Git credentials for private repositories

**Setup:**
###### Install Conda/Mamba from miniforge
Follow the instructions here https://github.com/conda-forge/miniforge

```bash
# 1. Clone repository
git clone https://gitlab.com/nct_tso_public/versatil.git
cd versatil

# 2. Create environment (use Mamba for faster installation)
mamba env create -f environment.yml
mamba activate versatil

# 3. Install dependencies with uv
UV_PROJECT_ENVIRONMENT=$CONDA_PREFIX uv sync

# 4. Install pre-commit hooks (formatting + linting on every commit)
pre-commit install
```

### Environment Configuration

VersatIL uses a `.env` file to configure machine-specific paths. Copy the example and customize:

```bash
cp .env.example .env
```

Edit `.env` with your paths:

```bash
# Storage paths
VERSATIL_CHECKPOINT_DIR=/path/to/checkpoints      # Where model checkpoints are saved
VERSATIL_ZARR_DIR=/path/to/zarr                   # Preprocessed Zarr datasets
VERSATIL_CACHE_DIR=/path/to/cache                 # HuggingFace/torch model cache

# Dataset paths (set only the ones you use)
VERSATIL_BOWEL_RETRACTION_DIR=/path/to/bowel_retraction
VERSATIL_LIBERO_HDF5_DIR=/path/to/libero/datasets
VERSATIL_LIBERO_LEROBOT_DIR=/path/to/libero_lerobot
VERSATIL_METAWORLD_LEROBOT_DIR=/path/to/metaworld_lerobot

# Weights & Biases (optional)
WANDB_PROJECT=versatil
WANDB_ENTITY=your-team
```

These variables are referenced in Hydra configs via OmegaConf resolvers (e.g., `${checkpoint_dir:bowel_retraction}`).

### Available Training Configs

Ready-to-use end-to-end configs are organized by dataset under `hydra_configs/end_to_end_training_runs/`:

| Dataset | Path | Data Link | Notes                                                                   |
|---|---|---|-------------------------------------------------------------------------|
| [Bowel Retraction](https://arxiv.org/abs/2601.21971) | `bowel_retraction/` | Coming soon | Real-world UR5e surgical robotics demonstrations. Language and depth variants included. |
| [LIBERO](https://libero-project.github.io/datasets) (HDF5) | `libero_hdf5/` | [libero-project.github.io](https://libero-project.github.io/datasets) | Original HDF5 format with 128x128 (flipped) images.                     |
| [LIBERO](https://huggingface.co/datasets/lerobot/libero) (LeRobot) | `libero_lerobot/` | [HF Hub](https://huggingface.co/datasets/lerobot/libero) | LeRobot format with OpenVLA filtered demonstrations and 256x256 images. |
| [LIBERO+](https://huggingface.co/datasets/Sylvest/libero_plus_lerobot) | `libero_plus/` | [HF Hub](https://huggingface.co/datasets/Sylvest/libero_plus_lerobot) | Extended LIBERO dataset.                                                |
| [MetaWorld MT50](https://huggingface.co/datasets/lerobot/metaworld_mt50) | `metaworld/` | [HF Hub](https://huggingface.co/datasets/lerobot/metaworld_mt50) | Multi-task benchmark (MT50 variant).                                    |

Each config is self-contained — just point to your data path and run.

### Training Your First Model

**1. Default Training:**
```bash
# Train ACT on bowel retraction dataset
python -m versatil.endpoints.train --config-name end_to_end_training_runs/bowel_retraction/act

# Train ACT on LIBERO dataset
python -m versatil.endpoints.train --config-name end_to_end_training_runs/libero_hdf5/act
```

**2. Override Configuration:**
```bash
# Change batch size
python -m versatil.endpoints.train \
    --config-name end_to_end_training_runs/bowel_retraction/act \
    task.dataloader.batch_size=64

# Disable EMA
python -m versatil.endpoints.train \
    --config-name end_to_end_training_runs/bowel_retraction/act \
    training.use_ema=false

# Change learning rate
python -m versatil.endpoints.train \
    --config-name end_to_end_training_runs/bowel_retraction/act \
    training.optimizer.lr=1e-4
```

**3. Distributed Training (SLURM):**
Not supported yet

---

## 🏗️ Architecture

### Policy Composition

The `Policy` class orchestrates three stages:

```python
# 1. Encode observations
features = encoding_pipeline(observations)  # Multi-modal → unified representation

# 2. Decode actions (algorithm orchestrates the decoder internally)
predictions = algorithm.forward(
    network=decoder,       # Algorithm receives decoder as a callable
    features=features,
    actions=ground_truth,  # During training
)

# 3. Compute loss
loss = loss_module(predictions, targets)
```


### Feature Naming Contract

VersatIL relies on strict naming conventions to wire encoders to decoders automatically. Instead of manually passing tensors, we match strings.

**The Rule:** `feature_name = "{encoder_name}_{output_key}"`

If you define an RGB encoder named `left_eye`, it produces:
* `left_eye_rgb` (The spatial features)

If you define a proprioception encoder named `robot_state`, it produces:
* `robot_state_proprio` (The flat features)

For multimodal encoders that produce multiple outputs (e.g. Vision-Language models), each output gets its own prefixed name.
If you define a VLM encoder named `vlm_model`, it produces:
* `vlm_model_rgb` (Image features)
* `vlm_model_language` (Text features)

For multi-camera encoders that share weights across cameras, the modality is followed by the camera key separated by a colon.
If you define an RGB encoder named `stereo` with two input cameras keyed `key_1` and `key_2`, it produces:
* `stereo_rgb:key_1` (Features from camera `key_1`)
* `stereo_rgb:key_2` (Features from camera `key_2`)

**Why strict naming?**
It prevents shape mismatches silently propagating. The `Policy` class validates shapes at initialization. If your Decoder expects a **FLAT** feature (1D)
but you feed it **SPATIAL** (3D) features from a ResNet, the code raises a `ValueError` immediately—before training starts.


**Fusion outputs** specify `output_name` directly, due to their multi-input nature.
```python
fusion = AttentionFusion(
    input_features=["left_eye_rgb", "right_eye_rgb"],  # Use encoder feature names
    output_name="fused_visual"  # Direct name (no prefix)
)
```


**Decoder inputs** require `input_keys` from the encoders or fusion outputs.


## 🧩 Available Components

### Encoders

- **RGB** via [timm](https://github.com/huggingface/pytorch-image-models)
  - SpatialRGBEncoder: spatial feature maps — ResNet, EfficientNet, ConvNeXt, MobileNet, EdgeNeXt, Swin, TinyViT, ...
  - FlatRGBEncoder: token sequences — ViT, DINOv2, DINOv3, ...
  - ConditionalCNNEncoder: ResNet with FiLM conditioning
- **Depth** via [timm](https://github.com/huggingface/pytorch-image-models)
  - SpatialDepthEncoder: single-channel spatial feature maps
- **Cross-Modal RGBD**
  - DFormerEncoder: RGB-D encoder with Geometric Attention ([paper](https://arxiv.org/abs/2504.04701))
  - GeometricRGBDEncoder: Custom lightweight geometric depth encoder
- **Cross-Modal Vision-Language** via [HF Transformers](https://github.com/huggingface/transformers):
  - TwoTowerVLMEncoder: dual vision/language towers e.g. CLIP, SigLIP.
  - GenerativeVLMEncoder: PaliGemma2, SmolVLM
- **Language** via [HF Transformers](https://github.com/huggingface/transformers): BERT, DistilBERT, MiniLM, Gemma, Qwen, ALBERT, RoBERTa, GPT2, DeBERTa, Phi, Llama, ...
- **Proprioceptive**: ProprioceptiveEncoder — MLP for robot state

Available backbones are listed in `src/versatil/models/encoding/encoders/constants.py` (`SpatialBackboneType`, `FlatBackboneType`, `LanguageEncoderType`, `ImageTextModelType`, `PaliGemmaModelType`, `SmolVLMModelType`).
They can be easily extended by either:
- Adding new Enum values that map to timm or HF Transformers model names.
- Implementing custom encoder classes that subclass `Encoder` (or `ConditionalEncoder` for conditioned encoders).

### Fusion

- `ConcatFusion` - Concatenation
- `MLPFusion` - MLP projection after concat
- `AttentionFusion` - Cross-attention

### Algorithms

- `BehavioralCloning` - Optimizes likelihood of expert actions via supervised learning
- `Diffusion` - Generative modeling via Denoising Score Matching through Diffusion ([paper](https://arxiv.org/abs/2011.13456))
- `FlowMatching` - Flow-Based Generative Modeling via Continuous Normalizing Flows ([paper](https://arxiv.org/abs/2209.03003))
- `VariationalAlgorithm` - Variational Inference wrapper to learn a latent space to use for any base algorithm

### Variational Framework

The `VariationalAlgorithm` wraps any base algorithm with a VAE-style latent space:

- **Posterior Network** q(z|a,s): Encodes actions into latent z during training
- **Prior Network** p(z|s): Samples latent z during inference (no access to actions)

**Posterior Network types:**
- `VAETransformerEncoder` - Transformer encoder that learns a CLS token to predict latent mean and logvar of a conditional Gaussian posterior

**Prior Network types:**
- `GaussianPrior` - Fixed Gaussian N(0,I) (standard VAE prior)
- `PriorTransformerEncoder` - Learned conditional gaussian prior using a transformer encoder
- `DiTPrior` - Multimodal prior trained via diffusion/flow matching
- `VampPrior` - Mixture of posteriors ([paper](https://arxiv.org/abs/1705.07120))

Each decoder can customize how it integrates the latent `z` token into its architecture (e.g., prepended token, cross-attention, FiLM).

### Decoders

- `ActionTransformer` - Bidirectional Transformer Decoder with any configurable positional encoding, normalization, and activation layers.
- `ACT` - Action Chunking Transformer ([paper](https://arxiv.org/abs/2304.13705))
- `LACT` - **Novel** Latent Action Transformer
- `PhaseACT` - Phase-aware ACT with surgical phase prediction ([paper](https://arxiv.org/abs/2601.21971))
- `FreeTransformer` - **Novel** Free Transformer action decoder inspired by ([paper](https://arxiv.org/pdf/2510.17558))
- `MoEFreeActionTransformer` - Mixture of Experts on top of Free Transformer
- `GPTActionTransformer` - **Novel** autoregressive GPT-style decoder with tokenized actions in the style of ([pi0-FAST](https://www.physicalintelligence.company/blog/pi0-fast))
- `DiscreteDETRActionTransformer` - DETR-style decoder (https://arxiv.org/abs/2005.12872) with tokenized actions
- `ConditionalActionUNet` - U-Net for Diffusion Policy ([paper](https://arxiv.org/abs/2303.04137))
- `DiTBlockActionTransformer` - DiT-Block Action Transformer (from [paper](https://arxiv.org/html/2410.10088v1))
- `DiffusionActionTransformer` - **Novel** Diffusion Action Transformer supporting two different architectures:
    - With cross-attention to encoder tokens, using an architecture inspired by PixArt ([paper](https://arxiv.org/abs/2310.00426))
    - With a dual-attention stream, using the MultiModal DiT architecture from SD3   ([paper](https://arxiv.org/abs/2403.03206))
- `MoDE-ACT` - Mixture Density Network Transformer with K Gaussian expert heads
- `Pi0Decoder` - Interleaved VLM-expert joint attention ([Pi0](https://arxiv.org/abs/2410.24164), [Pi0.5](https://arxiv.org/abs/2504.16054))
- `SmolVLADecoder` - Interleaved cross-attention and joint self-attention with VLM backbone ([SmolVLA](https://arxiv.org/abs/2506.01844))
- `MoEDecoder` - Mixture of Experts wrapper applicable on top of any decoder

You can easily extend the available decoders by implementing new classes that subclass `versatil.models.decoding.decoders.base.ActionDecoder`.

---

## Configuration System

**The Composition Pattern:**
Instead of massive monolithic config files, we mix and match small, reusable blocks, which are located in `hydra_configs/`.
An end-to-end training config just points to the blocks it wants to use:

```yaml
# hydra_configs/end_to_end_training_runs/bowel_retraction/act.yaml
# @package _global_
_target_: versatil.configs.main.MainConfig

defaults:
  - /task: base
  - /policy: base
  - /experiment: default
  - /task/dataset_schema: bowel_retraction_v2   # Raw data format
  - /task/dataloader: bowel_retraction           # Batch size, workers, augmentation
  - /task/action_space: deltas_cf_pos_gripper_phase
  - /task/observation_space: stereo_rgb
  - /training: default
  - /policy/encoding_pipeline: stereo_rgb        # Encoder + fusion config
  - /policy/decoder: act_default                 # Action decoder architecture
  - /policy/algorithm: bc_with_vae_gaussian      # Learning algorithm
  - /policy/loss: regression_gripper_KL          # Loss composition
  - /inference: default
  - _self_
```

### Config Validation

This is enforced at runtime using OmegaConf's typed Dataclasses.
The OmegaConf store is defined in `src/versatil/configs/__init__.py`.
Whenever you create a new config file, define a matching Dataclass in `src/versatil/configs/` to enforce types and defaults, and register it in the store.
This prevents silent errors from typos, wrong or missing parameters.

### Interpolation

You can reference other config values using `${}`:

```yaml
policy:
  prediction_horizon: ${task.prediction_horizon}
  observation_space: ${task.observation_space}
  device: ${experiment.device}
```
You can also define custom interpolation resolvers in `src/versatil/configs/__init__.py`, to interpolate e.g. Enum.values.

---

## 📊 Monitoring & Logging

### WandB Integration

Set environment variable:
```bash
export WANDB_API_KEY=your_key_here
```
Or add it to your `.bashrc` profile, for persistent settings.


### Checkpointing

**Best models** (based on `val_loss`):
```
checkpoints/experiment_name/best-epoch=XX-val_loss=Y.YYYY.ckpt
```

**Latest checkpoints** (every N epochs):
```
checkpoints/experiment_name/latest-epoch=XX.ckpt
checkpoints/experiment_name/last.ckpt
```

**Resume training:**
```bash
python -m versatil.endpoints.train \
    --config-name end_to_end_training_runs/bowel_retraction/act \
    experiment.resume_from=/path/to/checkpoint.ckpt
```

---

## 🧪 Testing

```bash
# Run unit tests only (default)
pytest

# Run all tests including integration tests
pytest -m ""

# Run specific test file
pytest tests/models/test_policy.py

# Run tests by marker
pytest -m "unit"           # Fast unit tests with mocked dependencies
pytest -m "integration"    # Slower tests with real model downloads
pytest -m "requires_gpu"   # GPU-required tests
pytest -m "not slow"       # Skip slow tests
```

---

## 📝 Code Style

- **Docstrings**: Google-style, concise (avoid LLM patterns like numbered lists or excessive words)
- **Type hints**: Required for all function signatures
- **Formatter/Linter**: [Ruff](https://docs.astral.sh/ruff/) (line length 88, Python 3.13)
- **No inline imports**: All imports at module top
- **Minimal comments**: Only for tensor shapes or non-obvious logic
- **Variables**: Use English words, avoid abbreviations
- **Function calls**: Use kwargs
- **Error handling**: Use `raise`, avoid assertions and try/catch blocks
- **Strings**: Use double quotes (`"foo"` not `'foo'`)
- **Constants**: Avoid hardcoded strings, use `Enum.MY_ENUM.value`
- **No wildcard imports**: Avoid `from module import *`
- Avoid `**kwargs` and `*args` signatures: Explicit is better than implicit

```bash
# Format code
ruff format src/ tests/

# Check formatting
ruff format --check src/ tests/

# Lint
ruff check src/ tests/

# Lint and auto-fix
ruff check --fix src/ tests/
```

Pre-commit hooks run ruff automatically on every `git commit`.

---

## 🐛 Troubleshooting

### CUDA Issues
- Verify CUDA 12.8 with `nvidia-smi`
- Check `torch.cuda.is_available()` returns `True`

### SLURM/NCCL Errors
- Set `export NCCL_P2P_DISABLE=1` in SLURM script
- Check `WORLD_SIZE` and `SLURM_PROCID` environment variables

### Data Loading
- Verify Zarr dataset paths and permissions
- Check dataset schema matches your data
- Ensure sufficient disk space for Zarr cache

---

## Known Issues

### Hydra 1.3.2 + Python 3.14: `ValueError: badly formed help string`

Python 3.14 added eager help-string validation in `argparse._ActionsContainer._check_help` which breaks Hydra's `LazyCompletionHelp` (only defines `__repr__`, not `__contains__`). Every `@hydra.main` endpoint fails at startup.

**Status:** Fixed upstream in [facebookresearch/hydra#3090](https://github.com/facebookresearch/hydra/pull/3090) (merged to `main`, targets 1.4.0.dev), but no PyPI release yet ([facebookresearch/hydra#3125](https://github.com/facebookresearch/hydra/issues/3125)). See [facebookresearch/hydra#3121](https://github.com/facebookresearch/hydra/issues/3121) for the bug report.

**Workaround:** `src/versatil/common/argparse_compat.py` monkey-patches `_check_help` to skip the eager validation for non-string help values. Imported in `train.py` and `post_training_compress.py`. Remove once `hydra-core >= 1.4` ships on PyPI.

### torchao 0.16 + Python 3.14: `Union.__module__` assignment crash in PT2E

Python 3.14 merged `typing.Union` with `types.UnionType`, making Union objects immutable. torchao 0.16 assigns `__module__` to Union aliases at import time in `torchao.quantization.pt2e`, which crashes on 3.14+.

**Status:** Fixed upstream in [pytorch/ao#3657](https://github.com/pytorch/ao/pull/3657). See [pytorch/ao#3619](https://github.com/pytorch/ao/issues/3619) for the bug report.

**Workaround:** `src/versatil/quantization/torch_patches.py` patches the installed torchao `.py` files on disk, replacing the crashing `__module__` assignments with `pass`. Called automatically before any PT2E import. Idempotent.

### torchao 0.16: `X86InductorQuantizer` silently quantizes 0 ops

`get_source_partitions` compares `source_fn_name` strings (e.g. `"linear"`) against class objects (e.g. `torch.nn.Linear`) in `wanted_sources`, which never matches. The quantizer reports success but quantizes nothing.

**Status:** See [pytorch/ao#3914](https://github.com/pytorch/ao/issues/3914).

**Workaround:** `src/versatil/quantization/torch_patches.py` monkey-patches `get_source_partitions` with a fallback that matches `source_fn_name` against each class's `__name__` (case-insensitive). Applied automatically, idempotent, version-gated to torch <= 2.10 + torchao <= 0.16.
