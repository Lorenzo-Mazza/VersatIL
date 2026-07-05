# VersatIL: Imitation Learning for Any Robot Policy

[![CI](https://github.com/Lorenzo-Mazza/VersatIL/actions/workflows/ci.yml/badge.svg?branch=main)](https://github.com/Lorenzo-Mazza/VersatIL/actions/workflows/ci.yml)
[![codecov](https://codecov.io/gh/Lorenzo-Mazza/VersatIL/branch/main/graph/badge.svg)](https://codecov.io/gh/Lorenzo-Mazza/VersatIL)
[![Ruff](https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/astral-sh/ruff/main/assets/badge/v2.json)](https://github.com/astral-sh/ruff)
[![Python 3.13/3.14](https://img.shields.io/badge/python-3.13%20%7C%203.14-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![PyPI](https://img.shields.io/pypi/v/versatil.svg)](https://pypi.org/project/versatil/)
[![Docs](https://img.shields.io/badge/docs-online-blue)](https://lorenzo-mazza.github.io/VersatIL)

![VersatIL Logo](https://raw.githubusercontent.com/Lorenzo-Mazza/VersatIL/main/media/VersatIL_logo.png)

### 🤯 The Paradox of Research Code
Have you ever found yourself wondering: *"How would this Robot Policy perform if I simply swapped that ResNet18 vision encoder for an EfficientNet, or just changed the KL divergence term in the loss function for a Maximum Mean Discrepancy?"*

So you clone the repo to try it out. You wrestle with a `requirements.txt` from 2018 that demands CUDA 9.0 and a version of PyTorch that seemingly only exists on a floppy disk in a basement. You finally get the environment running, only to discover that the loss function implementation is tightly coupled to a string variable named `"dataset_v2_final_final"` deep in the training loop.


Or perhaps you have wandered through State-Of-The-Art codebases, staring blankly at lines like:
`b = d.unsqueeze(-1).view(b, -1, h//16, w//16).permute(0, 3, 1, 2).contiguous()`
...wondering what unholy things are happening to those poor tensors?

### This ends with VersatIL. ⚡

VersatIL is a modular, composable framework for doing offline Imitation Learning (behavioral cloning) built with PyTorch, that decouples its main pillars:
**data**, **algorithm**, **network architecture** and **loss function** into clean, reusable components.

Swap standard behavioral cloning for diffusion or flow matching, replace a CNN backbone with a ViT or a whole Vision-Language Model, or run your policy on a completely new dataset format. Compatible components can be freely swapped with config or command-line changes, no source code rewrites.

Rapid experimentation, tested and reproducible code, and true reusability across projects.

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
    * **[torchao](https://github.com/pytorch/ao)** for post-training quantization (PT2E and quantize_() APIs).
- 💡 **Invent What Matters** For performance-critical components, we wrote a custom `src/versatil/models/layers` package in pure PyTorch. *Note: These are policy-agnostic and reusable in other projects.*
- 🔒 **Explainability & Reproducibility** – Strict interfaces, full type hints, Google-style docstrings, and runtime config validation.
- 🧪 **Testing** – Comprehensive unit and integration tests for every module.


### Paper Reproducibility Instructions

Paper-specific instructions for reproducing reported experiments are collected in
the [Papers Reproducibility Instructions](https://lorenzo-mazza.github.io/VersatIL/papers-reproducibility-instructions/)
docs section. These notes list the datasets, local path setup, environment
variables, and Hydra configs used for each paper.

---

## 🚀 Installation

VersatIL requires Python 3.13 or 3.14 and is available on PyPI:

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

The default PyPI PyTorch wheel runs on both CPU-only and CUDA machines. For
source installs — developing VersatIL, running the test suite, selecting the
dedicated CPU-only or CUDA 13.0 PyTorch wheel sets, or enabling ExecuTorch
`.pte` export — see the
[installation guide](https://lorenzo-mazza.github.io/VersatIL/getting-started/installation/).

---

## 🏁 Quick Start

### Environment Configuration

VersatIL uses a `.env` file to configure machine-specific paths. Copy the example and customize:

```bash
cp .env.example .env
```

```bash
VERSATIL_CHECKPOINT_DIR=/path/to/checkpoints      # Where model checkpoints are saved
VERSATIL_ZARR_DIR=/path/to/zarr                   # Preprocessed Zarr datasets
VERSATIL_CACHE_DIR=/path/to/cache                 # HuggingFace/torch model cache
VERSATIL_LIBERO_LEROBOT_DIR=/path/to/libero       # Raw data, one variable per dataset
```

The full variable list is in the [installation guide](https://lorenzo-mazza.github.io/VersatIL/getting-started/installation/#environment-configuration).

### Available Benchmarks

Ready-to-use end-to-end configs are organized by dataset under `src/versatil/hydra_configs/end_to_end_training_runs/`, and every simulated benchmark ships a ZMQ server wrapper for rolling out trained policies:

| Benchmark | Configs | Data | Sim Server | Notes |
|---|---|---|---|---|
| [Bowel Retraction](https://arxiv.org/abs/2601.21971) | `bowel_retraction/` | [HF Hub](https://huggingface.co/datasets/nct-tso/robotics_bowel_grasping) | — | Real-world UR5e surgical dataset; language and depth variants included. |
| [LIBERO](https://github.com/Lifelong-Robot-Learning/LIBERO) (HDF5) | `libero_hdf5/` | [libero-project.github.io](https://libero-project.github.io/datasets) | [simulation_libero](https://github.com/nct-tso-robotics/simulation_libero) | Original HDF5 format with 128x128 (flipped) images. |
| [LIBERO](https://github.com/Lifelong-Robot-Learning/LIBERO) (LeRobot) | `libero_lerobot/` | [HF Hub](https://huggingface.co/datasets/lerobot/libero) | [simulation_libero](https://github.com/nct-tso-robotics/simulation_libero) | LeRobot format with OpenVLA filtered demonstrations and 256x256 images. |
| [LIBERO+](https://github.com/sylvestf/LIBERO-plus) | `libero_plus/` | [HF Hub](https://huggingface.co/datasets/Sylvest/libero_plus_lerobot) | [simulation_libero_plus](https://github.com/nct-tso-robotics/simulation_libero_plus) | Extended LIBERO dataset. |
| [MetaWorld MT50](https://github.com/Farama-Foundation/Metaworld) | `metaworld/` | [HF Hub](https://huggingface.co/datasets/lerobot/metaworld_mt50) | [simulation_metaworld](https://github.com/nct-tso-robotics/simulation_metaworld) | Multi-task benchmark (MT50 variant). |
| [PushT](https://github.com/real-stanford/diffusion_policy/tree/main/diffusion_policy/env/pusht) | `pusht/` | [HF Hub](https://huggingface.co/datasets/nct-tso/robotics_pusht_lerobot) | [simulation_pusht](https://github.com/nct-tso-robotics/simulation_pusht) | LeRobot wrapper of the original PushT benchmark used by Diffusion Policy. |
| [Block Pushing](https://github.com/google-research/ibc/tree/master/environments/block_pushing) | `block_pushing/` | [relative](https://huggingface.co/datasets/nct-tso/robotics_blockpush_relative_lerobot), [absolute](https://huggingface.co/datasets/nct-tso/robotics_blockpush_absolute_lerobot) | [simulation_block_push](https://github.com/nct-tso-robotics/simulation_block_push) | LeRobot wrappers of the original IBC/Diffusion Policy BlockPush dataset; relative and absolute action variants. |
| [Kitchen](https://github.com/google-research/relay-policy-learning) | `kitchen/` | [HF Hub](https://huggingface.co/datasets/nct-tso/robotics_kitchen_lerobot) | [simulation_kitchen](https://github.com/nct-tso-robotics/simulation_kitchen) | LeRobot wrapper of the original Franka Kitchen dataset from relay-policy-learning. |
| [Multimodal Ant](https://github.com/jayLEE0301/vq_bet_official/tree/main/envs/antenv) | `ant/` | [HF Hub](https://huggingface.co/datasets/nct-tso/robotics_multimodal_ant_lerobot) | [simulation_multimodal_ant](https://github.com/nct-tso-robotics/simulation_multimodal_ant) | LeRobot wrapper of the original multimodal Ant navigation dataset. |
| [UR3 Block Push](https://github.com/jayLEE0301/vq_bet_official/tree/main/envs/ur3) | `ur3/` | [HF Hub](https://huggingface.co/datasets/nct-tso/robotics_ur3_blockpush_lerobot) | [simulation_ur3_block_push](https://github.com/nct-tso-robotics/simulation_ur3_block_push) | LeRobot wrapper of the original UR3 block-push dataset from VQ-BeT. |
| Synthetic | `synthetic/` | Generated on demand | — | Lightweight synthetic multimodal benchmark configs. |

The LIBERO server wrapper also supports [LIBERO-PRO](https://github.com/Zxy-MLlab/LIBERO-PRO/tree/master) evaluation.

Each config is self-contained, just point to your data path and run! 

### Training Your First Model

```bash
# Train ACT on bowel retraction dataset
python -m versatil.endpoints.train --config-name end_to_end_training_runs/bowel_retraction/act

# Train ACT on LIBERO dataset, overriding any parameter from the CLI
python -m versatil.endpoints.train --config-name end_to_end_training_runs/libero_hdf5/act \
    task.dataloader.batch_size=64 training.optimizer.lr=1e-4
```

CLI overrides, checkpointing, resuming, and WandB tracking are covered in the [training guide](https://lorenzo-mazza.github.io/VersatIL/getting-started/training/).

### 🧩 Available Components

#### Encoders

Observations can be encoded with, in principle, any model hosted on [timm](https://github.com/huggingface/pytorch-image-models) or [HF Transformers](https://github.com/huggingface/transformers) — CNNs, ViTs, language models, and VLMs alike:

- **RGB**: any timm vision backbone, producing spatial feature maps (ResNet, ConvNeXt, Swin, ...) or token sequences (ViT, DINOv2/v3, SigLIP, ...).
- **Depth and RGB-D**: single-channel depth backbones and cross-modal RGB-D encoders with geometric attention.
- **Language and VLM**: text embedding models (BERT-family, EmbeddingGemma, Qwen Embedding, ...) and image-text models (CLIP, SigLIP, ...) for language-conditioned policies.
- **Proprioception**: robot state encoders.

Features from multiple cameras and modalities can then be fused before decoding — by concatenation, MLP projection, or cross-attention. Adding a new backbone is usually a one-line Enum entry mapping to a timm or HF Transformers model name; fully custom encoders just subclass a small base interface. The complete encoder catalog is in the [encoding docs](https://lorenzo-mazza.github.io/VersatIL/architecture/encoding/).

#### Algorithms

The same decoder architecture can be trained with behavioral cloning, denoising diffusion, or flow matching, and any of these can be wrapped with a variational latent space learned through configurable priors and posteriors — see the [algorithms docs](https://lorenzo-mazza.github.io/VersatIL/architecture/algorithms/).

#### Action Decoders

VersatIL reproduces the main policy architectures from the offline imitation-learning/behavioral cloning literature:

| Policy | Paper | Available |
|---|---|:---:|
| ACT | [arXiv:2304.13705](https://arxiv.org/abs/2304.13705) | ✅ |
| Diffusion Policy | [arXiv:2303.04137](https://arxiv.org/abs/2303.04137) | ✅ |
| DiT-Block Policy | [arXiv:2410.10088](https://arxiv.org/abs/2410.10088) | ✅ |
| LACT | [arXiv:2605.22493](https://arxiv.org/abs/2605.22493) | ✅ |
| MoE-ACT | [arXiv:2601.21971](https://arxiv.org/abs/2601.21971) | ✅ |
| OpenVLA | [arXiv:2406.09246](https://arxiv.org/abs/2406.09246) | ✅ |
| OpenVLA-OFT | [project page](https://openvla-oft.github.io/) | ✅ |
| π0 | [arXiv:2410.24164](https://arxiv.org/abs/2410.24164) | ✅ |
| π0-FAST | [arXiv:2501.09747](https://arxiv.org/abs/2501.09747) | ✅ |
| π0.5 | [arXiv:2504.16054](https://arxiv.org/abs/2504.16054) | ✅ |
| SmolVLA | [arXiv:2506.01844](https://arxiv.org/abs/2506.01844) | ✅ |


**But it doesn't stop here!** Beyond the literature presets, generic building blocks (bidirectional, autoregressive, and diffusion action transformers, swappable VLM backbones inside the VLA decoders, plus a Mixture-of-Experts wrapper applicable to any decoder), tuning paradigms (custom LoRA, multi-stage training, custom objective functions) and algorithm sweeping (swap algorithms from the same class, or change the diffusion/flow solver, etc.) enable you to design new architectures. You can also easily implement fully custom decoders by just subclassing [`ActionDecoder`](src/versatil/models/decoding/decoders/base.py). The complete decoder catalog is in the [decoders docs](https://lorenzo-mazza.github.io/VersatIL/architecture/decoders/).

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

Together, this means you can create new experiments by overriding only the parameters that change. Forget about writing verbose YAML files for every single run you want to benchmark — an end-to-end experiment is just a list of reusable blocks:

```yaml
# end_to_end_training_runs/bowel_retraction/act.yaml (excerpt)
defaults:
  - /task/dataset_schema: bowel_retraction_v2   # Raw data format
  - /task/dataloader: bowel_retraction           # Batch size, torch workers, augmentation
  - /task/action_space: deltas_cf_pos_gripper_phase
  - /task/observation_space: stereo_rgb
  - /policy/encoding_pipeline: stereo_rgb        # Encoder + fusion config
  - /policy/decoder: act_default                 # Action decoder architecture
  - /policy/algorithm: bc_with_vae_gaussian      # Learning algorithm
  - /policy/loss: regression_gripper_kl          # Loss composition
```

Every block is validated against a typed dataclass at startup — see the [configuration guide](https://lorenzo-mazza.github.io/VersatIL/getting-started/configuration/) for the full system.

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
   [Zarr](https://zarr.readthedocs.io/en/stable/) provides fast, compressed, chunked storage with NumPy-like access.
   - Created **automatically** on first training run if missing. No separate preprocessing script needed.
   - Decouples raw storage from training-optimized layout.
   - Format-specific keys (e.g. `observation.images.image`, `agentview_image`) are remapped to standardized pipeline keys during ingestion, so training and inference never see raw format naming — see the [data docs](https://lorenzo-mazza.github.io/VersatIL/architecture/data/).

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

Actions can be **precomputed** (stored in Zarr) or computed **on-the-fly** during batching (e.g., deltas from consecutive states).

---



#### 🧠 Policy Composition
A robot policy is built from four decoupled components, orchestrated by the [`Policy`](src/versatil/models/policy.py) class:
1.  👁️ **Encoding Pipeline:** A pipeline of multi-modal encoders that extract features from raw observations plus an optional fusion module that combines the features
into a unified representation.
2.  🧮 **Algorithm:** The learning paradigm that defines how to train the policy. This can be:
- Standard Behavioral Cloning (supervised learning of actions given observations)
- Generative approaches such as Diffusion and Flow Matching.
- Variational approaches that add a learned latent variable to any base algorithm. The latent variable can be learned through different kinds of prior-posterior schemes, which will determine the nature of the latent space.
3. 🕹️ **Action Decoder:** The neural architecture that decodes the features into robot actions. 
We provide a set of factory decoders which comprise some of the SOTA architectures of recent years in the field of offline IL, from the Action Chunking Transformer (ACT), to the Diffusion Policy, until more recent Vision-Language-Action models. More information on the available decoders can be found below.
4.  📉 **Loss Module:** A composable loss module that defines the objective function to optimize during training. This can be a simple regression loss (MSE) or a more complex loss that combines multiple terms (e.g. action regression + KL divergence for variational algorithms).

---


#### ⚡ Training Engine
Powered by **PyTorch Lightning**:
* Automatic handling of training and checkpointing.
* **WandB Integration:** Tracks in real-time metrics and visualizations.
* **Callbacks:** Pluggable training-loop hooks, from EMA weights to latent-space visualizations.

---
### 🚀 Post-Training

#### 🔌 Inference

The inference pipeline is transport-agnostic: communication with any environment server (real robot or simulation) is abstracted behind [`ObservationTransport`](src/versatil/inference/protocol.py) and [`ActionTransport`](src/versatil/inference/protocol.py) Python protocols. Any object satisfying these protocols works (ZMQ, HTTP, etc).

The built-in ZMQ implementation uses our two PyPI packages:
- [**tso-robotics-sockets**](https://pypi.org/project/tso-robotics-sockets/): Generic ZMQ socket client/server.
- [**versatil-constants**](https://pypi.org/project/versatil-constants/): Shared constants defining the observation/action message format.

Both libraries are server-agnostic, i.e. they define the message format but not the server implementation. For custom setups, you just need to implement the transport protocols, [`ObservationTransport`](src/versatil/inference/protocol.py) and [`ActionTransport`](src/versatil/inference/protocol.py), and couple it with your custom server.

We also provide ZMQ server wrappers for common robot learning simulators, so VersatIL policies can be rolled out without extra glue code — each benchmark in the [Available Benchmarks](#available-benchmarks) table links its server wrapper in the Sim Server column.

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

Heatmap overlays (and optionally raw tensors) are written next to the checkpoint. See the [explainability guide](https://lorenzo-mazza.github.io/VersatIL/architecture/explainability/) for sources, targeting, and output formats.

---


#### 📦 Quantization and Post-Training Compression

VersatIL's quantization package is built upon PyTorch's native quantization library `torchao` and supports quantization-aware training as well as eager and PT2E post-training quantization, targeting the whole policy or individual modules. The post-training compression pipeline turns a trained policy checkpoint into a deployment artifact for edge or resource-constrained hardware: a PTC run can export a floating-point model, apply pruning, quantize the policy, and save either a Torch Export `.pt2` artifact (x86 CPUs) or an ExecuTorch `.pte` artifact (ARM and x86 mobile CPUs). Background and full configuration are covered in the [quantization guide](https://lorenzo-mazza.github.io/VersatIL/architecture/quantization/) and the [compression guide](https://lorenzo-mazza.github.io/VersatIL/architecture/post_training_compression/).

---

## 🤝 Contributing

Contributions are welcome! Install from source, then verify your setup:

```bash
# Default local suite: excludes slow, integration, GPU-only, and ExecuTorch-dependent tests
pytest

# Format and lint (pre-commit hooks run this on every commit)
ruff format src/ tests/ && ruff check src/ tests/
```

Development setup, testing conventions, code style rules, and PR guidelines are in [CONTRIBUTING.md](https://github.com/Lorenzo-Mazza/VersatIL/blob/main/CONTRIBUTING.md).

---

## 🐛 Troubleshooting

Common issues (CUDA, data loading, Python 3.14 quirks) are collected in the [training guide](https://lorenzo-mazza.github.io/VersatIL/getting-started/training/#troubleshooting) and the [Known Issues](https://lorenzo-mazza.github.io/VersatIL/known-issues/) page.

---

## 📖 Citation

If you use VersatIL in your research, please cite the repository:

```bibtex
@misc{mazza2026versatil,
    author = {Mazza, Lorenzo and Rodriguez, Ariel and Speidel, Stefanie},
    title = {VersatIL: A Modular Imitation Learning Framework for Robot Policies},
    howpublished = {\url{https://github.com/Lorenzo-Mazza/VersatIL}},
    year = {2026}
}
```

If you use the MoE-ACT policy, please also cite:

```bibtex
@article{mazza2026moe,
  title={MoE-ACT: Improving Surgical Imitation Learning Policies through Supervised Mixture-of-Experts},
  author={Mazza, Lorenzo and Rodriguez, Ariel and Younis, Rayan and Lelis, Martin and Hellig, Ortrun and Li, Chenpan and Bodenstedt, Sebastian and Wagner, Martin and Speidel, Stefanie},
  journal={arXiv preprint arXiv:2601.21971},
  year={2026}
}
```

If you use the LACT policy, please also cite:

```bibtex
@article{mazza2026understanding,
  title={Understanding Multimodal Failure in Action-Chunking Behavioral Cloning},
  author={Mazza, Lorenzo and Datres, Massimiliano and Rodriguez, Ariel and Bodenstedt, Sebastian and Kutyniok, Gitta and Speidel, Stefanie},
  journal={arXiv preprint arXiv:2605.22493},
  year={2026}
}
```

---

## 📄 License

VersatIL is released under the [MIT License](LICENSE).

---

## 🙌 Contributors

Thanks to all [contributors](https://github.com/Lorenzo-Mazza/VersatIL/blob/main/contributors.md) who make VersatIL possible! Want to join them? Start with the [contributing guide](https://github.com/Lorenzo-Mazza/VersatIL/blob/main/CONTRIBUTING.md).
