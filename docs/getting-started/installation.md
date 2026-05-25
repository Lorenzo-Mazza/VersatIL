# Installation

## Prerequisites

| Requirement | Minimum Version | Notes |
|-------------|----------------|-------|
| Python | 3.14 | Reference environment and CI version. `pyproject.toml` allows 3.13+. |
| CUDA driver | Supports CUDA 13.0 runtime | Required for the pinned `cu130` PyTorch wheels |
| Git | Latest | Credentials for private repositories if applicable |

## Setup

### 1. Install Conda/Mamba

Install [Miniforge](https://github.com/conda-forge/miniforge) to get `conda` and `mamba`. Mamba is recommended over conda for significantly faster dependency resolution.

### 2. Clone and Create Environment

```bash
git clone https://gitlab.com/nct_tso_public/versatil.git
cd versatil

# Create environment (use mamba for faster installation)
mamba env create -f environment.yml
mamba activate versatil
```

The `environment.yml` creates a minimal conda environment with Python 3.14 and uv.

### 3. Install Dependencies

VersatIL uses [uv](https://github.com/astral-sh/uv) for fast, reproducible dependency management. All dependencies are declared in `pyproject.toml`.

```bash
UV_PROJECT_ENVIRONMENT=$CONDA_PREFIX uv sync
```

This installs all packages into the active conda environment, including:

- **PyTorch 2.12.0** with CUDA 13.0 wheels from the PyTorch index
- **Hydra + OmegaConf** for configuration
- **Lightning 2.6.1** for training
- **timm**, **transformers**, **diffusers** for model backbones
- **albumentations** for image augmentation
- **wandb** for experiment tracking

### 4. Install Pre-commit Hooks

```bash
pre-commit install
```

This enables automatic Ruff formatting and linting on every `git commit`.

## Environment Configuration

VersatIL uses a `.env` file for machine-specific paths. These variables are resolved at runtime by OmegaConf custom resolvers (e.g., `${checkpoint_dir:bowel_retraction}` in YAML configs).

Copy the example file and edit it:

```bash
cp .env.example .env
```

### Required Variables

```bash
# Where model checkpoints are saved
VERSATIL_CHECKPOINT_DIR=/path/to/checkpoints

# Preprocessed Zarr datasets
VERSATIL_ZARR_DIR=/path/to/zarr

# HuggingFace/torch model cache (downloads from timm, transformers, etc.)
VERSATIL_CACHE_DIR=/path/to/cache
```

### Dataset Path Variables

Set only the variables for datasets you use:

```bash
# Pretrained models directory
VERSATIL_PRETRAINED_DIR=/path/to/pretrained_models

# Raw data paths (one per dataset)
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
```

### WandB Variables (Optional)

```bash
WANDB_PROJECT=versatil
WANDB_ENTITY=your-team
```

!!! tip
    If `VERSATIL_CACHE_DIR` is not set, it defaults to `~/.cache/versatil`. If `VERSATIL_CHECKPOINT_DIR` or `VERSATIL_ZARR_DIR` are not set, they default to the current working directory.

## Verifying the Installation

Activate the environment and run the default local test selection. This excludes
slow, integration, and GPU-only tests via `pyproject.toml`:

```bash
mamba activate versatil
pytest
```

To verify CUDA availability:

```bash
python -c "import torch; print(torch.cuda.is_available())"
```
