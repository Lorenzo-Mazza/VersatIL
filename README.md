# VersatIL: Imitation Learning for Any Robot Policy

[![CI](https://github.com/Lorenzo-Mazza/VersatIL/actions/workflows/ci.yml/badge.svg?branch=main)](https://github.com/Lorenzo-Mazza/VersatIL/actions/workflows/ci.yml)
[![codecov](https://codecov.io/gh/Lorenzo-Mazza/VersatIL/branch/main/graph/badge.svg)](https://codecov.io/gh/Lorenzo-Mazza/VersatIL)
[![Ruff](https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/astral-sh/ruff/main/assets/badge/v2.json)](https://github.com/astral-sh/ruff)
[![Python 3.13/3.14](https://img.shields.io/badge/python-3.13%20%7C%203.14-blue.svg)](https://www.python.org/downloads/)
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
    * **[Albumentations](https://albumentations.ai/)** for image augmentations.
    * **[torchao](https://github.com/pytorch/ao)** for eager and PT2E quantization workflows, for both quantization-aware training and post-training quantization.
- 💡 **Invent What Matters** For performance-critical components, we wrote a custom `src/versatil/models/layers` package in pure PyTorch. This includes optimized implementations of:
    * [Attention](https://docs.pytorch.org/docs/stable/generated/torch.nn.functional.scaled_dot_product_attention.html) (PyTorch built-in SDPA Flash kernel dispatch).
    * Conditional Flow Matching utilities and ODE integration.
    * Positional Encodings (Sinusoidal, Learned, Rotary).
    * Transformer variants (DETR, GPT, BERT, DiT).
    * Modular Deep Neural Networks layers such as normalization, modulation, convolution, etc
    * *Note: These are policy-agnostic and reusable in other projects.*
- 🔒 **Explainability & Safety** – Strict interfaces, full type hints, Google-style docstrings, and runtime config validation.
- 🧪 **Testing** – Unit and integration tests for every module, run in CI.

### Paper Reproducibility Instructions

Paper-specific instructions for reproducing reported experiments are collected in
the [Papers Reproducibility Instructions](docs/papers-reproducibility-instructions/index.md)
docs section. These notes list the datasets, local path setup, environment
variables, and Hydra configs used for each paper.

---

## 🚀 Installation

**Prerequisites:**
- Python 3.13 or 3.14 — supported by `pyproject.toml` (`requires-python = ">=3.13,<3.15"`).
- NVIDIA driver compatible with CUDA 13.0 PyTorch wheels, only when installing the `gpu` extra.

**Setup:**
### Option A: Install from PyPI

Create a Python 3.13/3.14 environment with your preferred manager and install
the package:

```bash
# With uv
uv venv --python 3.14
source .venv/bin/activate
uv pip install versatil

# Or with mamba/conda
mamba create -n versatil python=3.14 pip
mamba activate versatil
pip install versatil
```

The default PyPI PyTorch wheel runs on both CPU-only and CUDA machines. The
dedicated CPU-only or CUDA 13.0 wheel sets are selected through the
`--extra cpu` / `--extra gpu` flags of the source installs below.

### Option B: Install from source into a Miniforge/Mamba environment
Use a source install when you want to develop VersatIL itself or run the test
suite. Follow the Miniforge instructions at https://github.com/conda-forge/miniforge

```bash
# 1. Clone repository
git clone https://github.com/Lorenzo-Mazza/VersatIL.git
cd VersatIL

# 2. Create environment (use Mamba for faster installation)
mamba env create -f environment.yml
mamba activate versatil
# To force Python 3.13 instead:
# mamba create -n versatil python=3.13 pip
# mamba activate versatil
# python -m pip install uv

# 3. Install dependencies into the active conda environment
PYTHON_VERSION=3.14
UV_PROJECT_ENVIRONMENT=$CONDA_PREFIX uv sync --python "$PYTHON_VERSION" --extra gpu
# For CPU-only environments:
# UV_PROJECT_ENVIRONMENT=$CONDA_PREFIX uv sync --python "$PYTHON_VERSION" --extra cpu
# For Python 3.13, set PYTHON_VERSION=3.13.

# 4. Install pre-commit hooks (formatting + linting on every commit)
pre-commit install
```

### Option C: Install from source with uv directly
This creates a project-local `.venv` and does not require conda, mamba, or
Miniforge.

```bash
# 1. Install uv if it is not already available
curl -LsSf https://astral.sh/uv/install.sh | sh

# 2. Clone repository
git clone https://github.com/Lorenzo-Mazza/VersatIL.git
cd VersatIL

# 3. Create the Python environment
PYTHON_VERSION=3.14
uv python install "$PYTHON_VERSION"
uv venv --python "$PYTHON_VERSION"
source .venv/bin/activate
# For Python 3.13, set PYTHON_VERSION=3.13.

# 4. Install dependencies
uv sync --python "$PYTHON_VERSION" --extra gpu
# For CPU-only environments:
# uv sync --python "$PYTHON_VERSION" --extra cpu

# 5. Install pre-commit hooks (formatting + linting on every commit)
pre-commit install
```

With both source options, `uv sync` installs the `dev` dependency group
(pytest, pytest-cov, ruff, pre-commit) by default, so the environment is ready
for development out of the box. Pass `--no-dev` for a runtime-only install.

### Optional ExecuTorch Dependency for Edge Deployment

**ExecuTorch for XNNPACK `.pte` export:** Python 3.13 environments can install
ExecuTorch from PyPI through the optional extra:

```bash
PYTHON_VERSION=3.13
uv sync --python "$PYTHON_VERSION" --extra cpu --extra executorch
# Use --extra gpu instead of --extra cpu when installing the CUDA PyTorch stack.
```

The `executorch` extra is guarded by a Python marker, because the published
ExecuTorch package currently declares `requires-python = ">=3.10,<3.14"`.
Python 3.14 environments need an ExecuTorch package built from source in the
active `versatil` environment:

```bash
cd ..
git clone https://github.com/pytorch/executorch.git
cd executorch
git submodule update --init --recursive

# Build dependencies must be present because --no-build-isolation is used.
pip install "cmake>=3.24,<4.0.0" "packaging>=24.2" pyyaml "setuptools>=77.0.3" wheel zstd certifi ninja

SITE_PACKAGES=$(python - <<'PY'
import site
print(site.getsitepackages()[0])
PY
)
# CUDA and OpenVINO must be disabled explicitly
# because setup.py auto-enables them when nvcc / Linux are detected; the LLM
# kernels are preset defaults this deployment does not need.
CMAKE_PREFIX_PATH="$SITE_PACKAGES" \
CMAKE_BUILD_PARALLEL_LEVEL=8 \
CMAKE_ARGS="-DEXECUTORCH_BUILD_CUDA=OFF -DEXECUTORCH_BUILD_OPENVINO=OFF -DEXECUTORCH_BUILD_KERNELS_LLM=OFF -DEXECUTORCH_BUILD_KERNELS_LLM_AOT=OFF" \
python -m pip install . --no-build-isolation --ignore-requires-python --no-deps -v

cd ../versatil

# Runtime dependencies are skipped by --no-deps; install the AoT set manually.
pip install flatbuffers "ruamel.yaml" sympy tabulate pytorch-tokenizers \
    expecttest hypothesis kgb parameterized

# Now all ExecuTorch-gated tests should pass.
pytest -m requires_executorch -o addopts=""
```

`python -m pip check` can still report a `scikit-learn` metadata conflict in
Python 3.14 environments. The XNNPACK export path works with the built package.

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
1. **[`DatasetSchema`](src/versatil/data/raw/schemas/base.py) (how your raw data is structured)**
   A pluggable class that maps any raw format to a standardized **Zarr** store.

   | Schema                                                | Class | Raw Format              |
   |-------------------------------------------------------|---|-------------------------|
   | [HuggingFace LeRobot](https://huggingface.co/lerobot) | [`LeRobotDatasetSchemaV30`](src/versatil/data/raw/schemas/lerobot.py) | Parquet + MP4/images    |
   | HDF5                                                  | [`Hdf5DatasetSchema`](src/versatil/data/raw/schemas/hdf5.py) | HDF5 archive            |
   | CSV                                                   | [`CsvDatasetSchema`](src/versatil/data/raw/schemas/csv.py) | CSV + raw image folders |
   | Custom                                                | Subclass [`DatasetSchema`](src/versatil/data/raw/schemas/base.py) | Any                     |

2. **Zarr Store Creation**
   Zarr [https://zarr.readthedocs.io/en/stable/]  provides fast, compressed, chunked storage with NumPy-like access.
   - Created **automatically** on first training run if missing — no separate preprocessing script needed.
   - Decouples raw storage from training-optimized layout.
   - Raw keys vs pipeline keys: Raw data formats use their own naming (e.g., LIBERO LeRobot dataset uses `observation.images.image`, LIBERO original HDF5 dataset uses `agentview_image`). During zarr creation, these *raw camera keys* ([`RawCameraKey`](src/versatil/data/constants.py)) are remapped to standardized *pipeline camera keys* ([`Cameras`](src/versatil/data/constants.py)) via `RAW_TO_CAMERA_MAPPING`. After zarr creation, only pipeline keys exist — the rest of the codebase (training, inference, validation) never sees raw format keys. This separation is defined in `src/versatil/data/constants.py`, so adding a new raw data format only takes a new [`RawCameraKey`](src/versatil/data/constants.py) entry and its mapping — the training and inference pipeline stays untouched.

---


### 🏋️‍♂️ During Training
#### 🎯 Task Definition

The **[`TaskSpace`](src/versatil/data/task.py)** selects what subset of the Zarr data to use, allowing multiple tasks from a single dataset without duplication:

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
A robot policy is built from four decoupled components, orchestrated by the [`Policy`](src/versatil/models/policy.py) class:
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

The inference pipeline is transport-agnostic: communication with any environment server (real robot, simulation, or custom) is abstracted behind [`ObservationTransport`](src/versatil/inference/protocol.py) and [`ActionTransport`](src/versatil/inference/protocol.py) Python protocols. Any object satisfying these protocols works — ZMQ, HTTP, etc.

The built-in ZMQ implementation uses our two PyPI packages:
- [**tso-robotics-sockets**](https://pypi.org/project/tso-robotics-sockets/): Generic ZMQ socket client/server with protocol keys (`ServerRoute`, `InferenceRequestKey`, `CompressionType`).
- [**versatil-constants**](https://pypi.org/project/versatil-constants/): Shared domain constants for action/observation message passing (`ActionComponent`, `ActionMetadataField`, `ObsKey`, `GripperType`, `OrientationRepresentation` and dataset/benchmark specific message keys).

Both libraries are server-agnostic — they define the message format, not the server implementation. Any server that speaks the protocol can be integrated by implementing the transport protocols.

The built-in ZMQ transport works for both simulation and real hardware — the dataset format is fully decoupled from the transport layer. For custom setups, implement the [`ObservationTransport`](src/versatil/inference/protocol.py) and [`ActionTransport`](src/versatil/inference/protocol.py) protocols with any transport mechanism.

##### Simulation Servers

We provide ZMQ server wrappers for common robot learning simulators, so VersatIL policies can be rolled out without extra glue code:

| Simulator | Original | ZMQ Server Wrapper |
|---|---|---|
| LIBERO / LIBERO-PRO | [LIBERO](https://github.com/Lifelong-Robot-Learning/LIBERO), [LIBERO-PRO](https://github.com/Zxy-MLlab/LIBERO-PRO/tree/master) | [simulation_libero](https://github.com/nct-tso-robotics/simulation_libero) |
| LIBERO+ | [GitHub](https://github.com/sylvestf/LIBERO-plus) | [simulation_libero_plus](https://github.com/nct-tso-robotics/simulation_libero_plus) |
| MetaWorld | [GitHub](https://github.com/Farama-Foundation/Metaworld) | [simulation_metaworld](https://github.com/nct-tso-robotics/simulation_metaworld) |
| PushT | [Diffusion Policy PushT](https://github.com/real-stanford/diffusion_policy/tree/main/diffusion_policy/env/pusht) | [simulation_pusht](https://github.com/nct-tso-robotics/simulation_pusht) |
| Franka Kitchen | [relay-policy-learning](https://github.com/google-research/relay-policy-learning) | [simulation_kitchen](https://github.com/nct-tso-robotics/simulation_kitchen) |
| BlockPush | [IBC BlockPushing](https://github.com/google-research/ibc/tree/master/environments/block_pushing) | [simulation_block_push](https://github.com/nct-tso-robotics/simulation_block_push) |
| UR3 BlockPush | [VQ-BeT UR3](https://github.com/jayLEE0301/vq_bet_official/tree/main/envs/ur3) | [simulation_ur3_block_push](https://github.com/nct-tso-robotics/simulation_ur3_block_push) |
| Multimodal Ant | [VQ-BeT AntEnv](https://github.com/jayLEE0301/vq_bet_official/tree/main/envs/antenv) | [simulation_multimodal_ant](https://github.com/nct-tso-robotics/simulation_multimodal_ant) |

---

#### 🔍 Explainability
Visual attribution for trained policies: per-camera heatmaps showing which image regions drove the predicted actions.
We currently support Grad-CAM, Grad-CAM++, and Ablation-CAM, computed over every visual module of the policy (encoding-pipeline encoders as well as decoder-owned VLM vision towers), either on offline dataset samples or live during inference.

```bash
python -m versatil.endpoints.explain \
    checkpoint_path=/path/to/training/checkpoint \
    explanation_types='[gradcam]' \
    split=val \
    max_samples=16
```

Heatmap overlays (and optionally raw tensors) are written next to the checkpoint. See the [explainability guide](docs/architecture/explainability.md) for sources, targeting, and output formats.

---


#### 📦 Quantization and Post-Training Compression

**What is post-training quantization?**
Post-training quantization (PTQ) converts trained floating-point model weights and activations to lower-precision integer representations (e.g., INT8). This reduces memory footprint, improves cache utilization, and enables hardware-accelerated integer arithmetic — typically achieving inference speedup on x86 CPUs with minimal accuracy loss. PTQ is done after training. Static quantization uses a small calibration dataset to determine optimal activation ranges per layer; dynamic quantization computes ranges on-the-fly at inference time and needs no calibration.
**What is quantization-aware-training?**
Quantization-aware training inserts fake quantizers between layers of the neural policy to mimic the information loss that the policy will experience at deployment time after PTQ. In this way, the policy learns a mapping that is robust to PTQ-induced information loss, improving downstream performance.
**How VersatIL implements quantization:**
VersatIL's quantization package is built upon PyTorch's native quantization library `torchao`, which supports two types of quantization workflows: eager quantization (mainly used for dynamic quantization of Large Language Models, int8 to int2 support, linear layers only) and PyTorch 2 Export quantization (mainly used for static quantization, int8 only, linear and convolutional layers). Both workflows plug directly into policy training and deployment. See docs/quantization.md for more details about these quantization workflows.

**What is post-training compression?**
The post-training compression (PTC) pipeline that turns a trained policy checkpoint into a deployment artifact for edge or resource-constrained hardware. A PTC run can export a floating-point model, apply pruning, quantize the policy, and save either a Torch Export `.pt2` artifact or an ExecuTorch `.pte` artifact.


**How VersatIL implements PTC:**

The compression pipeline is configurable via Hydra and supports three complementary techniques applied sequentially:

1. **Preparation**: Frozen BatchNorm replacement and Conv+BN weight folding — standard pre-quantization layer fusion that merges batch normalization parameters into convolution weights.

2. **Pruning**: Weight pruning to introduce sparsity before quantization. Supports both unstructured (global L1 magnitude) and structured (per-channel Lp-norm) pruning, composable sequentially, e.g. structured pruning followed by unstructured pruning on the same module.

3. **Quantization workflow**: One workflow is selected per policy:
   - **No quantization**: `quantization: null` exports the floating-point policy.
   - **Eager quantization**: Uses the [`torchao.quantization.quantize_()` API](https://docs.pytorch.org/ao/stable/api_reference/generated/torchao.quantization.quantize_.html#torchao.quantization.quantize_) before export.
   - **PT2E** (PyTorch 2 Export): Exports the policy with `torch.export`, then applies PT2E prepare, optional calibration, and convert.


**Per-module targeting:**

Compression targets configure preparation and pruning globally or per module. Quantization targets live inside the selected workflow under `quantization.targets`, where each [`QuantizationModuleTarget`](src/versatil/quantization/module_target.py) can carry a module-specific quantization config. Target paths must exist in the policy and must not overlap.

**Deployment backends:**

The `deployment_backend` config field selects the artifact format and lowering step for edge deployment. As of now, two backends are supported:
- `TorchInductorBackend`, for running on X86 CPUs.
- `ExecutorchXNNPACKBackend` for running on ARM and X86 mobile CPUs. 

**Compressed inference:**

Compressed models are loaded by [`CompressedPolicyRuntime`](src/versatil/inference/policy_runtime/compressed_runtime.py). Torch Export `.pt2` artifacts run through PyTorch and can be compiled with `torch.compile` when appropriate. ExecuTorch `.pte` artifacts run through the ExecuTorch adapter on CPU.  Currently, compressed models are not fully standalone: they still require a complete VersatIL installation, including its dependencies. Since this is not ideal for edge deployment, self-contained edge-device inference runtime is currently under development.

---

## 🚀 Quick Start

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
VERSATIL_LIBERO_PLUS_LEROBOT_DIR=/path/to/libero_plus_lerobot
VERSATIL_METAWORLD_LEROBOT_DIR=/path/to/metaworld_lerobot
VERSATIL_PUSHT_LEROBOT_DIR=/path/to/pusht_lerobot
VERSATIL_BLOCK_PUSHING_LEROBOT_DIR=/path/to/block_pushing_lerobot_rel
VERSATIL_BLOCK_PUSHING_LEROBOT_ABS_DIR=/path/to/block_pushing_lerobot_abs
VERSATIL_KITCHEN_LEROBOT_DIR=/path/to/kitchen_lerobot
VERSATIL_MULTIMODAL_PEG_TRANSFER_DIR=/path/to/multimodal_peg_transfer
VERSATIL_ANT_LEROBOT_DIR=/path/to/ant_lerobot
VERSATIL_UR3_LEROBOT_DIR=/path/to/ur3_lerobot

# Weights & Biases (optional)
WANDB_PROJECT=versatil
WANDB_ENTITY=your-team
```

These variables are referenced in Hydra configs via OmegaConf resolvers (e.g., `${checkpoint_dir:bowel_retraction}`).

### Available Training Configs

Ready-to-use end-to-end configs are organized by dataset under `src/versatil/hydra_configs/end_to_end_training_runs/`:

| Dataset | Path | Data Link | Notes                                                                   |
|---|---|---|-------------------------------------------------------------------------|
| [Bowel Retraction](https://arxiv.org/abs/2601.21971) | `bowel_retraction/` | [HF Hub](https://huggingface.co/datasets/nct-tso/robotics_bowel_grasping) | Original real-world UR5e surgical robotics dataset release. Language and depth variants included. |
| [LIBERO](https://libero-project.github.io/datasets) (HDF5) | `libero_hdf5/` | [libero-project.github.io](https://libero-project.github.io/datasets) | Original HDF5 format with 128x128 (flipped) images.                     |
| [LIBERO](https://huggingface.co/datasets/lerobot/libero) (LeRobot) | `libero_lerobot/` | [HF Hub](https://huggingface.co/datasets/lerobot/libero) | LeRobot format with OpenVLA filtered demonstrations and 256x256 images. |
| [LIBERO+](https://huggingface.co/datasets/Sylvest/libero_plus_lerobot) | `libero_plus/` | [HF Hub](https://huggingface.co/datasets/Sylvest/libero_plus_lerobot) | Extended LIBERO dataset.                                                |
| [MetaWorld MT50](https://huggingface.co/datasets/lerobot/metaworld_mt50) | `metaworld/` | [HF Hub](https://huggingface.co/datasets/lerobot/metaworld_mt50) | Multi-task benchmark (MT50 variant).                                    |
| PushT | `pusht/` | [HF Hub](https://huggingface.co/datasets/nct-tso/robotics_pusht_lerobot) | LeRobot wrapper of the original PushT benchmark used by Diffusion Policy. |
| Block Pushing | `block_pushing/` | [relative](https://huggingface.co/datasets/nct-tso/robotics_blockpush_relative_lerobot), [absolute](https://huggingface.co/datasets/nct-tso/robotics_blockpush_absolute_lerobot) | LeRobot wrappers of the original IBC/Diffusion Policy BlockPush dataset; relative and absolute action variants. |
| Kitchen | `kitchen/` | [HF Hub](https://huggingface.co/datasets/nct-tso/robotics_kitchen_lerobot) | LeRobot wrapper of the original Franka Kitchen dataset from relay-policy-learning. |
| Multimodal Ant | `ant/` | [HF Hub](https://huggingface.co/datasets/nct-tso/robotics_multimodal_ant_lerobot) | LeRobot wrapper of the original multimodal Ant navigation dataset. |
| UR3 Block Push | `ur3/` | [HF Hub](https://huggingface.co/datasets/nct-tso/robotics_ur3_blockpush_lerobot) | LeRobot wrapper of the original UR3 block-push dataset from VQ-BeT. |
| Multimodal Peg Transfer | `multimodal_peg_transfer/` | Local data | Peg-transfer surgical robot task configs. |
| Synthetic | `synthetic/` | Generated on demand | Lightweight synthetic multimodal benchmark configs. |

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

The [`Policy`](src/versatil/models/policy.py) class orchestrates three stages:

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
It prevents shape mismatches silently propagating. The [`Policy`](src/versatil/models/policy.py) class validates shapes at initialization. If your Decoder expects a **FLAT** feature (1D)
but you feed it **SPATIAL** (3D) (Channel, Height, Width) features from a spatial vision backbone, the code raises a `ValueError` immediately—before training starts.


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
  - [`SpatialRGBEncoder`](src/versatil/models/encoding/encoders/rgb/spatial.py): spatial feature maps — ResNet, EfficientNet, ConvNeXt, MobileNet, EdgeNeXt, Swin, TinyViT, ...
  - [`FlatRGBEncoder`](src/versatil/models/encoding/encoders/rgb/flat.py): token sequences — ViT, DINOv2, DINOv3, CLIP ViT, SigLIP, ...
  - [`DinoV2SigLIPRGBEncoder`](src/versatil/models/encoding/encoders/rgb/dinov2_siglip.py): paired DINOv2+SigLIP patch-token features
  - [`ConditionalCNNEncoder`](src/versatil/models/encoding/encoders/rgb/conditional_cnn.py): ResNet with FiLM conditioning
- **Depth** via [timm](https://github.com/huggingface/pytorch-image-models)
  - [`SpatialDepthEncoder`](src/versatil/models/encoding/encoders/depth/spatial.py): single-channel spatial feature maps
- **Cross-Modal RGBD**
  - [`DFormerEncoder`](src/versatil/models/encoding/encoders/cross_modal/rgbd/dformerv2.py): RGB-D encoder with Geometric Attention ([paper](https://arxiv.org/abs/2504.04701), [pretrained weights](https://huggingface.co/bbynku/DFormerv2), [docs](https://lorenzo-mazza.github.io/VersatIL/architecture/encoding/))
  - [`GeometricRGBDEncoder`](src/versatil/models/encoding/encoders/cross_modal/rgbd/geometric_rgbd.py): Custom lightweight geometric depth encoder
- **Vision Language Model (VLM) encoders** via [HF Transformers](https://github.com/huggingface/transformers):
  - [`VLMEncoder`](src/versatil/models/encoding/encoders/cross_modal/vision_language/vlm_encoder.py): encoder-pipeline module for image-text embedding models, e.g. CLIP and SigLIP.
- **Language encoders** via [HF Transformers](https://github.com/huggingface/transformers): BERT, DistilBERT, MiniLM, EmbeddingGemma, Qwen Embedding, BGE, E5, ALBERT, RoBERTa, DeBERTa, ...
- **Proprioceptive**: [`ProprioceptiveEncoder`](src/versatil/models/encoding/encoders/proprioceptive/base.py) — MLP for robot state

Available encoder backbones are listed in `src/versatil/models/encoding/encoders/constants.py` ([`SpatialBackboneType`](src/versatil/models/encoding/encoders/constants.py), [`FlatBackboneType`](src/versatil/models/encoding/encoders/constants.py), [`DinoV2SigLIPBackboneType`](src/versatil/models/encoding/encoders/constants.py), [`LanguageEncoderType`](src/versatil/models/encoding/encoders/constants.py), [`ImageTextModelType`](src/versatil/models/encoding/encoders/constants.py)).
They can be easily extended by either:
- Adding new Enum values that map to timm or HF Transformers model names.
- Implementing custom encoder classes that subclass [`Encoder`](src/versatil/models/encoding/encoders/unconditional.py) (or [`ConditionalEncoder`](src/versatil/models/encoding/encoders/conditional.py) for conditioned encoders).

### Fusion

- [`ConcatFusion`](src/versatil/models/encoding/fusion/concat.py) - Concatenation
- [`MLPFusion`](src/versatil/models/encoding/fusion/mlp.py) - MLP projection after concat
- [`AttentionFusion`](src/versatil/models/encoding/fusion/attention.py) - Cross-attention

### Algorithms

- [`BehavioralCloning`](src/versatil/models/decoding/algorithm/behavior_cloning.py) - Optimizes likelihood of expert actions via supervised learning
- [`Diffusion`](src/versatil/models/decoding/algorithm/diffusion.py) - Generative modeling via Denoising Score Matching through Diffusion ([paper](https://arxiv.org/abs/2011.13456))
- [`FlowMatching`](src/versatil/models/decoding/algorithm/flow_matching.py) - Flow-Based Generative Modeling via Continuous Normalizing Flows ([paper](https://arxiv.org/abs/2209.03003))
- [`VariationalAlgorithm`](src/versatil/models/decoding/algorithm/variational.py) - Variational Inference wrapper to learn a latent space to use for any base algorithm

### Variational Framework

The [`VariationalAlgorithm`](src/versatil/models/decoding/algorithm/variational.py) wraps any base algorithm with a VAE-style latent space:

- **Posterior Network** q(z|a,s): Encodes actions into latent z during training
- **Prior Network** p(z|s): Samples latent z during inference (no access to actions)

**Posterior Network types:**
- [`VAETransformerEncoder`](src/versatil/models/decoding/latent/posterior/transformer_encoder.py) - Transformer encoder that learns a CLS token to predict latent mean and logvar of a conditional Gaussian posterior
- [`VQPosteriorEncoder`](src/versatil/models/decoding/latent/posterior/vq_encoder.py) - Transformer posterior with residual vector quantization for discrete latent action codes

**Prior Network types:**
- [`GaussianPrior`](src/versatil/models/decoding/latent/prior/gaussian_prior.py) - Fixed Gaussian N(0,I) (standard VAE prior)
- [`PriorTransformerEncoder`](src/versatil/models/decoding/latent/prior/transformer_encoder.py) - Learned conditional gaussian prior using a transformer encoder
- [`DiTPrior`](src/versatil/models/decoding/latent/prior/dit_prior.py) - Multimodal prior trained via diffusion/flow matching
- [`VampPrior`](src/versatil/models/decoding/latent/prior/vamp_prior.py) - Mixture of posteriors ([paper](https://arxiv.org/abs/1705.07120))
- [`UniformCodebookPrior`](src/versatil/models/decoding/latent/prior/uniform_codebook_prior.py) and [`CodebookPrior`](src/versatil/models/decoding/latent/prior/codebook_prior.py) - Fixed or learned priors over VQ codebook indices

Each decoder can customize how it integrates the latent `z` token into its architecture (e.g., prepended token, cross-attention, adaptive normalization).

### Standard Action Decoders

- [`ActionTransformer`](src/versatil/models/decoding/decoders/factory/action_transformer.py) - Bidirectional Transformer Decoder with any configurable positional encoding, normalization, and activation layers.
- [`ACT`](src/versatil/models/decoding/decoders/factory/act.py) - Action Chunking Transformer ([paper](https://arxiv.org/abs/2304.13705))
- [`LACT`](src/versatil/models/decoding/decoders/factory/lact.py) - Latent Action Transformer ([paper](https://arxiv.org/abs/2605.22493))
- [`PhaseACT`](src/versatil/models/decoding/decoders/factory/phase_act.py) - Phase-aware ACT with surgical phase prediction ([paper](https://arxiv.org/abs/2601.21971))
- [`GPTActionTransformer`](src/versatil/models/decoding/decoders/factory/gpt_action_transformer.py) - Autoregressive GPT-style decoder with tokenized actions
- [`ConditionalActionUNet`](src/versatil/models/decoding/decoders/factory/conditional_action_unet.py) - U-Net for Diffusion Policy ([paper](https://arxiv.org/abs/2303.04137))
- [`DiTBlockActionTransformer`](src/versatil/models/decoding/decoders/factory/dit_block_action_transformer.py) - DiT-Block Action Transformer (from [paper](https://arxiv.org/html/2410.10088v1))
- [`DiffusionActionTransformer`](src/versatil/models/decoding/decoders/factory/diffusion_action_transformer.py) - Diffusion Action Transformer supporting two different architectures:
    - With cross-attention to encoder tokens, using an architecture inspired by PixArt ([paper](https://arxiv.org/abs/2310.00426))
    - With a dual-attention stream, using the MultiModal DiT architecture from SD3   ([paper](https://arxiv.org/abs/2403.03206))
- [`MoDE-ACT`](src/versatil/models/decoding/decoders/factory/mode_act.py) - Mixture Density Network Transformer with K Gaussian expert heads
- [`MoEDecoder`](src/versatil/models/decoding/decoders/moe.py) - Mixture of Experts wrapper applicable on top of any decoder

### VLA Decoders

These decoders use a vision-language model directly inside the
action-generation sequence.

- [`AutoregressiveVLADecoder`](src/versatil/models/decoding/decoders/factory/autoregressive_vla.py) - VLM-backed autoregressive action-token decoder used by the `openvla` and `pi0_fast` presets.
- [`OpenVLAOFTDecoder`](src/versatil/models/decoding/decoders/factory/openvla_oft.py) - VLM-backed continuous action-chunk decoder inspired by [OpenVLA-OFT](https://openvla-oft.github.io/).
- [`Pi0Decoder`](src/versatil/models/decoding/decoders/factory/pi0.py) - Interleaved VLM-action joint attention ([Pi0](https://arxiv.org/abs/2410.24164), [Pi0.5](https://arxiv.org/abs/2504.16054)). It runs a configured VLM backbone on image/text observations and pairs VLM layers with expert action layers.
- [`SmolVLADecoder`](src/versatil/models/decoding/decoders/factory/smolvla.py) - Interleaved VLM decoder with alternating joint self-attention and cross-attention over VLM key/value states ([SmolVLA](https://arxiv.org/abs/2506.01844)).

The reusable HF wrappers used by these decoders live in
`src/versatil/models/decoding/generative_language_models/`. The shipped
vision-language model identifiers are available at
`src/versatil/models/decoding/generative_language_models/constants.py`.
HuggingFace-backed language encoders, VLM encoders, and generative VLM
backbones accept an optional `lora_config` for PEFT LoRA fine-tuning.

You can easily extend the available decoders by implementing new classes that subclass [`ActionDecoder`](src/versatil/models/decoding/decoders/base.py).

---

## Configuration System

**The Composition Pattern:**
Instead of massive monolithic config files, we mix and match small, reusable blocks, which are located in `src/versatil/hydra_configs/`.
An end-to-end training config just points to the blocks it wants to use:

```yaml
# src/versatil/hydra_configs/end_to_end_training_runs/bowel_retraction/act.yaml
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
# Run the default local suite: excludes slow, integration, and GPU-only tests
pytest

# Run all tests including integration tests
pytest -m ""

# Run specific test file
pytest tests/models/test_policy.py

# Run tests by marker
pytest -m "unit"                                      # Fast tests with mocked dependencies
pytest -m "integration"                               # Real component integration tests
pytest -m "requires_gpu"                              # GPU-required tests
pytest -m "not slow and not integration and not requires_gpu"  # Explicit default selection
```

---

## 📝 Code Style

- **Docstrings**: Google-style, concise (avoid LLM patterns like numbered lists or excessive words)
- **Type hints**: Required for all function signatures
- **Formatter/Linter**: [Ruff](https://docs.astral.sh/ruff/) (line length 88, lint target pinned to `py313` so annotation imports stay at runtime for OmegaConf)
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
- Verify the NVIDIA driver supports CUDA 13.0 with `nvidia-smi`
- Check `torch.cuda.is_available()` returns `True`

### Data Loading
- Verify Zarr dataset paths and permissions
- Check dataset schema matches your data
- Ensure sufficient disk space for Zarr cache

### Python Compatibility
If torchao crashes on Python 3.14, see the [Known Issues](https://lorenzo-mazza.github.io/VersatIL/known-issues/) docs for active workarounds.
