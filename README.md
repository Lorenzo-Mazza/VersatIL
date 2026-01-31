# VersatIL: Imitation Learning for Any Robot Policy

![VersatIL Logo](media/VersatIL_logo.png)

### 🤯 The Paradox of Research Code
Have you ever found yourself wondering: *"How would this Robot Policy perform if I simply swapped that ResNet18 for an EfficientNet, or just changed one term in the loss function?"*

So you clone the repo to try it out. You wrestle with a `requirements.txt` from 2018 that demands CUDA 9.0 and a version of PyTorch that seemingly only exists on a floppy disk in a basement. You finally get the environment running, only to discover that the loss function implementation is tightly coupled to a string variable named `"dataset_v2_final_final"` deep in the training loop.


Or perhaps you have wandered through State-Of-The-Art codebases, staring blankly at lines like:
`b = d.unsqueeze(-1).view(b, -1, h//16, w//16).permute(0, 3, 1, 2).contiguous()`
...wondering what unholy things are happening to those poor tensors?

### This ends with VersatIL. ⚡

VersatIL is a modular, composable framework built with PyTorch fully decouples the three pillars of imitation learning: 
**Data**,
 **Algorithm**, and **Architecture** into clean, reusable components.


Swap Behavioral Cloning for Diffusion or Flow Matching, replace a ResNet with a ViT or VLM backbone, or run your policy on a completely new dataset format — all with just config changes, no code rewrites.

Rapid experimentation, cleaner code, and true reusability across projects.

### Core Principles
- 🧑‍🔬 **Research-First Flexibility** — Unlike frameworks that focus on reimplementing and distributing specific SOTA policies, VersatIL gives you the modular building blocks to **create and benchmark your own novel architectures and algorithms** on any dataset.
- 🔄 **Mix & Match** You are free to swap any robot policy component for easy benchmarking.
- 🧱 **Modularity** Each component is self-contained and reusable.
- ⚡ **Modern Dependency Management** – Dependencies managed with [uv](https://github.com/astral-sh/uv) and `pyproject.toml` for modern and fast installation.
- ♻️ **Don't Reinvent the Wheel** We rely on industry-standard libraries:
    * **Timm** for vision backbones.
    * **HuggingFace Transformers** for Language encoders and VLMs.
    * **HuggingFace Diffusers** for diffusion schedulers.
    * **TorchCFM** for Flow Matching schedulers.
- 💡 **Invent What Matters** For performance-critical components, we wrote a custom `models/layers` package. This includes optimized implementations of:
    * Attention (FlashAttention).
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
   Built-in support for:
   - HuggingFace LeRobot datasets
   - LIBERO-style HDF5
   - Custom CSV + image folders (TSO Lab format)  
   Extend by subclassing `DatasetSchema` for new formats.
   
2. **Zarr Store Creation**  
   Zarr [https://zarr.readthedocs.io/en/stable/]  provides fast, compressed, chunked storage with NumPy-like access.  
   - Created **automatically** on first training run if missing — no separate preprocessing script needed.
   - Decouples raw storage from training-optimized layout.

---


### 🏋️‍♂️ During Training
#### 🎯 Task Definition

The **TaskConfig** selects what subset of the Zarr data to use, allowing multiple tasks from a single dataset without duplication:

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

#### 🔌 Inference (ZMQ)

We provide ZMQ-based inference clients for TSO Lab Robot Testbed server, LIBERO simulation server and Metaworlds simulation server. 
The clients handle 
communication with the server through sockets, requesting for observations and sending back actions predicted by the trained policy. This design allows us to 
decouple the policy training environment from the robot/simulation server implementation, enabling easy integration with different robot platforms or simulation
environments.

---

#### 🔍 Explainability
We provide tools for model interpretability, such as visualization of the feature maps from the trained policy vision encoders.
We currently support Grad-CAM, Grad-CAM++, Ablation-CAM and Integrated Gradients for visual explanations of the model's predictions.

---


#### 📦 Quantization
We plan to add support for post-training quantization of the trained policies, to enable deployment on edge devices with limited computational resources.


---

## 🚀 Quick Start

### Installation

**Prerequisites:**
- Python 3.11+
- CUDA 12.4+ (required for training)
- Git credentials for private repositories

**Setup:**
###### Install Conda/Mamba from miniforge
Follow the instructions here https://github.com/conda-forge/miniforge

```bash
# 1. Clone repository
git clone https://gitlab.com/nct_tso_public/versatil.git
cd versatil

# 2. Configure git credentials
git config --global credential.helper store
# 3. Create environment (use Mamba for faster installation)
mamba env create -f environment.yml
mamba activate versatil

# 4. Install dependencies with uv
UV_PROJECT_ENVIRONMENT=$CONDA_PREFIX uv sync
```

NB: The above installation requires a machine with a GPU with CUDA installed.
This is needed so that flash-attn can find the CUDA runtime libraries.
To install VersatIL from our computing cluster, you need to run:
```bash
srun --gres=gpu:1 --cpus-per-task=1 --pty bash
# Then run installation commands above
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

# 2. Decode actions
predictions = algorithm.forward(
    decoder(features),  # Algorithm uses decoder architecture
    actions=ground_truth  # During training
)

# 3. Compute loss
loss = loss_module(predictions, targets)
```


### Feature Naming Contract

VersatIL relies on strict naming conventions to wire encoders to decoders automatically. Instead of manually passing tensors, we match strings.

**The Rule:** `feature_name = "{encoder_name}_{type}"`

If you define an RGB encoder named `left_eye`, it produces:
* `left_eye_image` (The spatial features)

If you define a proprioception encoder named `robot_state`, it produces:
* `robot_state_proprio` (The flat features)

For multimodal encoders that produce multiple features, such as Vision-Language models, we use a dot separator to select the feature.
If you define a VLM encoder named `vlm_model`, it produces:
* `vlm_model.image` (Image features)
* `vlm_model.language` (Text features)

**Why strict naming?**
It prevents shape mismatches silently propagating. The `Policy` class validates shapes at initialization. If your Decoder expects a **FLAT** feature (1D)
but you feed it **SPATIAL** (3D) features from a ResNet, the code raises a `ValueError` immediately—before training starts.


**Fusion outputs** specify `output_name` directly, due to their multi-input nature.
```python
fusion = AttentionFusion(
    input_features=["left_eye_image", "right_eye_image"],  # Use encoder feature names
    output_name="fused_visual"  # Direct name (no prefix)
)
```


**Decoder inputs** require `input_keys` from the encoders or fusion outputs.


## 🧩 Available Components

### Encoders

- **RGB** via [timm](https://github.com/huggingface/pytorch-image-models) 
  - CNNEncoder: ResNet, EfficientNet, EdgeNeXt
  - ConditionalCNNEncoder: CNN with FiLM conditioning
  - ViTEncoder: ViT Base, DINOv2, DINOv3
- **Depth**
  - DepthCNNEncoder: timm backbones adapted for single-channel depth
  - DFormerV2: RGB-D encoder with Geometric Attention ([paper](https://arxiv.org/abs/2504.04701))
  - LightGeometric: Custom lightweight geometric depth encoder
- **Language** via [transformers](https://github.com/huggingface/transformers): BERT Base, DistilBERT Base, MiniLM-L6, Gemma 2B, Qwen2-1.5B
- **Proprioceptive**: ProprioEncoder - MLP for robot state
- **VLM** via [transformers](https://github.com/huggingface/transformers): CLIP, SigLIP

Available backbones are listed in `src/versatil/models/encoding/encoders/constants.py` (`RGBBackboneType`, `LanguageEncoderType`, `ImageTextModelType`).
They can be easily extended by either:
- adding new Enum values that map to timm or transformers model names. 
- Implementing custom encoder classes that subclass `EncoderBase`.

### Fusion

- `ConcatFusion` - Concatenation
- `MLPFusion` - MLP projection after concat
- `AttentionFusion` - Cross-attention

### Algorithms

- `BehavioralCloning` - Optimizes likelihood of expert actions via supervised learning
- `Diffusion` - Generative modeling via Denoising Score Matching through Diffusion ([paper](https://arxiv.org/abs/2011.13456))
- `FlowMatching` - Flow-Based Generative Modeling via Continuous Normalizing Flows ([paper](https://arxiv.org/abs/2209.03003))
- `VariationalAlgorithm` - Variational Inference wrapper to learn a latent space to use for any base algorithm

### Decoders

- `ActionTransformer` - Action Transformer Decoder, using modern components s.a. Rotary Positional Embeddings and RMSNorm.
- `ACT` - Action Chunking Transformer ([paper](https://arxiv.org/abs/2304.13705))
- `LACT` - Latent ACtion Transformer
- `PhaseACT` - Phase-aware ACT with surgical phase prediction ([paper](https://arxiv.org/abs/2601.21971))
- `FreeTransformer` - Free Transformer adapted as action decoder ([paper](https://arxiv.org/pdf/2510.17558))
- `MoEFreeTransformer` - Mixture of Experts on top of Free Transformer
- `FASTGPTDecoder` - Autoregressive GPT-style decoder with tokenized actions in the style of ([pi0-FAST](https://www.physicalintelligence.company/blog/pi0-fast))
- `FASTDETRDecoder` - DETR-style decoder (https://arxiv.org/abs/2005.12872) with tokenized actions
- `ConditionalUNet` - U-Net for Diffusion Policy ([paper](https://arxiv.org/abs/2303.04137))
- `DiTBlockActionTransformer` - DiT-Block Action Transformer (from [paper](https://arxiv.org/html/2410.10088v1))
- `DiffusionActionTransformer` - Diffusion Action Transformer supporting two different architectures:
    - With cross-attention to encoder tokens, using an architecture inspired by PixArt ([paper](https://arxiv.org/abs/2310.00426))
    - With a dual-attention stream, using the MultiModal DiT architecture from SD3   ([paper](https://arxiv.org/abs/2403.03206))
- `MoEDecoderWrapper` - Mixture of Experts wrapper to use on top of any decoder



---

## Configuration System

**The Composition Pattern:**
Instead of massive monolithic config files, we mix and match small, reusable blocks, which are located in `hydra_configs/`.
An end-to-end training config just points to the blocks it wants to use:

```yaml
# hydra_configs/end_to_end_training_runs/bowel_retraction/act.yaml
defaults:
  - /dataset_schema: bowel_retraction    # Loads raw dataset schema
  - /task: bowel_retraction     # Loads task definition
  - /policy: act                # Loads ACT architecture
  - ...
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

- **Docstrings**: Google-style
- **Type hints**: Required for all functions
- **Formatter**: Black (line length 88, Python 3.11)
- **No inline imports**: All imports at module top

```bash
# Format code
black src/ tests/

# Check formatting
black --check src/ tests/
```

---

## 🐛 Troubleshooting

### CUDA Issues
- Verify CUDA 12.4 with `nvidia-smi`
- Check `torch.cuda.is_available()` returns `True`

### SLURM/NCCL Errors
- Set `export NCCL_P2P_DISABLE=1` in SLURM script
- Check `WORLD_SIZE` and `SLURM_PROCID` environment variables

### Data Loading
- Verify Zarr dataset paths and permissions
- Check dataset schema matches your data
- Ensure sufficient disk space for Zarr cache


