# Installation

## Prerequisites

| Requirement | Minimum Version | Notes |
|-------------|----------------|-------|
| Python | 3.13 or 3.14 | Supported by `pyproject.toml` (`requires-python = ">=3.13,<3.15"`). |
| CUDA driver | Supports CUDA 13.0 runtime | Required only when installing the `gpu` extra |
| Git | Latest | Credentials for private repositories if applicable |

## Setup

### Option A: Install from PyPI

Create a Python 3.13/3.14 environment with your preferred manager and install
the package:

```bash
# With uv
uv venv --python 3.14
source .venv/bin/activate
uv pip install versatil --prerelease=allow

# Or with Micromamba (after initializing its shell hook)
micromamba create -n versatil -c conda-forge python=3.14 pip
micromamba activate versatil
python -m pip install versatil
```

The `--prerelease=allow` flag is required with uv: Python 3.13/3.14 support
in `hydra-core` and `omegaconf` is currently published as pre-releases, which
plain `pip` accepts automatically but uv rejects for transitive dependencies,
silently resolving an old versatil version instead.

The default PyPI PyTorch wheel runs on both CPU-only and CUDA machines. The
dedicated CPU-only or CUDA 13.0 wheel sets are selected through the
`--extra cpu` / `--extra gpu` flags of the source installs below.

### Option B: Source Install into a Micromamba Environment

Use a source install when you want to develop VersatIL itself or run the test
suite.

#### 1. Install Micromamba

Install the standalone [Micromamba](https://mamba.readthedocs.io/en/latest/installation/micromamba-installation.html)
executable using its official installer:

```bash
"${SHELL}" <(curl -L micro.mamba.pm/install.sh)
```

If the installer did not initialize Bash, initialize its shell hook once:

```bash
micromamba shell init -s bash -r "$HOME/.local/share/micromamba"
source ~/.bashrc
```

On systems with a small home quota, replace the root prefix with a project or
scratch path.

#### 2. Clone and Create Environment

```bash
git clone https://github.com/Lorenzo-Mazza/VersatIL.git
cd VersatIL

micromamba env create -f environment.yml
micromamba activate versatil
```

The `environment.yml` creates a minimal environment with a supported
Python version and uv. To force Python 3.13 instead of the default solver
choice, create the environment manually:

```bash
micromamba create -n versatil -c conda-forge python=3.13 pip
micromamba activate versatil
python -m pip install uv
```

#### 3. Install Dependencies

VersatIL uses [uv](https://github.com/astral-sh/uv) for fast, reproducible dependency management. All dependencies are declared in `pyproject.toml`.

```bash
PYTHON_VERSION=3.14
environment_prefix="$(python -c 'import sys; print(sys.prefix)')"
UV_PROJECT_ENVIRONMENT="$environment_prefix" uv sync --python "$PYTHON_VERSION" --extra gpu
# For CPU-only environments:
# UV_PROJECT_ENVIRONMENT="$environment_prefix" uv sync --python "$PYTHON_VERSION" --extra cpu
# For Python 3.13, set PYTHON_VERSION=3.13.
```

This installs all packages into the active Micromamba environment.

### Option C: Source Install with uv

Use this path when you want a project-local `.venv` without Micromamba.

```bash
# Install uv if it is not already available
curl -LsSf https://astral.sh/uv/install.sh | sh

git clone https://github.com/Lorenzo-Mazza/VersatIL.git
cd VersatIL

PYTHON_VERSION=3.14
uv python install "$PYTHON_VERSION"
uv venv --python "$PYTHON_VERSION"
source .venv/bin/activate
# For Python 3.13, set PYTHON_VERSION=3.13.

uv sync --python "$PYTHON_VERSION" --extra gpu
# For CPU-only environments:
# uv sync --python "$PYTHON_VERSION" --extra cpu
```

Both source setup paths install:

- **PyTorch 2.13**, **TorchVision 0.28** and **TorchAO 0.18** from the selected
  PyTorch wheel extra (`gpu` or `cpu`)
- **Hydra + OmegaConf** for configuration
- **Lightning 2.6.1** for training
- **timm**, **transformers**, **diffusers** for model backbones
- **albumentations** for image augmentation
- **wandb** for experiment tracking
- **Dev tooling** (pytest, pytest-cov, ruff, pre-commit) from the `dev`
  dependency group, which `uv sync` includes by default — pass `--no-dev` for a
  runtime-only install

### Optional ExecuTorch Dependency

ExecuTorch 1.4.1 provides wheels for Python 3.13 and 3.14. Install it through
the optional extra:

```bash
PYTHON_VERSION=3.14
uv sync --frozen --python "$PYTHON_VERSION" --extra cpu --extra executorch
# Use --extra gpu instead of --extra cpu when installing the CUDA PyTorch stack.
```

This installs the Python export tools and the packaged runtime for CPU/XNNPACK
inference on both supported Python versions.
The lockfile selects the following versions together:

| Package | Version |
| --- | --- |
| PyTorch | 2.13.0 |
| TorchVision | 0.28.0 |
| TorchAO | 0.18.0 |
| ExecuTorch | 1.4.1 |

Check the installed dependencies and run the ExecuTorch tests with:

```bash
uv pip check
python -m pytest -m "requires_executorch and not slow and not requires_gpu"
```

### ExecuTorch CUDA Runtime

The `gpu` extra installs CUDA-enabled PyTorch. The `executorch` extra installs
ExecuTorch's Python export tools and packaged CPU runtime.

CUDA deployment uses AOTInductor to compile the exported model and a native
ExecuTorch runtime built with `-DEXECUTORCH_BUILD_CUDA=ON`. Follow the
[CUDA backend build instructions for ExecuTorch 1.4.1](https://github.com/pytorch/executorch/blob/v1.4.1/docs/source/backends/cuda/cuda-overview.md)
with a matching `v1.4.1` source checkout. Build that runtime separately from the
standard Python wheel installation. Its build environment requires the CUDA
toolkit, including `nvcc`, and a C++ compiler. PyTorch wheels supply the CUDA
libraries used during execution.

VersatIL's ExecuTorch deployment adapter targets XNNPACK on CPU. CUDA integration
requires a deployment adapter for export and a CUDA-enabled runtime for execution.

### Install Pre-commit Hooks

```bash
pre-commit install
```

Ruff then formats and lints your changes on every `git commit`.

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
slow, integration, GPU-only, and ExecuTorch-dependent tests via `pyproject.toml`:

```bash
micromamba activate versatil
pytest
```

To verify CUDA availability:

```bash
python -c "import torch; print(torch.cuda.is_available())"
```
