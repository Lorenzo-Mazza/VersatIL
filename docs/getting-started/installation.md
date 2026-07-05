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

# Or with mamba/conda
mamba create -n versatil python=3.14 pip
mamba activate versatil
pip install versatil
```

The `--prerelease=allow` flag is required with uv: Python 3.13/3.14 support
in `hydra-core` and `omegaconf` is currently published as pre-releases, which
plain `pip` accepts automatically but uv rejects for transitive dependencies,
silently resolving an old versatil version instead.

The default PyPI PyTorch wheel runs on both CPU-only and CUDA machines. The
dedicated CPU-only or CUDA 13.0 wheel sets are selected through the
`--extra cpu` / `--extra gpu` flags of the source installs below.

### Option B: Source Install into a Miniforge/Mamba Environment

Use a source install when you want to develop VersatIL itself or run the test
suite.

#### 1. Install Conda/Mamba

Install [Miniforge](https://github.com/conda-forge/miniforge) to get `conda` and `mamba`. Mamba is recommended over conda for significantly faster dependency resolution.

#### 2. Clone and Create Environment

```bash
git clone https://github.com/Lorenzo-Mazza/VersatIL.git
cd VersatIL

# Create environment (use mamba for faster installation)
mamba env create -f environment.yml
mamba activate versatil
```

The `environment.yml` creates a minimal conda environment with a supported
Python version and uv. To force Python 3.13 instead of the default solver
choice, create the environment manually:

```bash
mamba create -n versatil python=3.13 pip
mamba activate versatil
python -m pip install uv
```

#### 3. Install Dependencies

VersatIL uses [uv](https://github.com/astral-sh/uv) for fast, reproducible dependency management. All dependencies are declared in `pyproject.toml`.

```bash
PYTHON_VERSION=3.14
UV_PROJECT_ENVIRONMENT=$CONDA_PREFIX uv sync --python "$PYTHON_VERSION" --extra gpu
# For CPU-only environments:
# UV_PROJECT_ENVIRONMENT=$CONDA_PREFIX uv sync --python "$PYTHON_VERSION" --extra cpu
# For Python 3.13, set PYTHON_VERSION=3.13.
```

This installs all packages into the active conda environment.

### Option C: Source Install with uv

Use this path when you want a project-local `.venv` without conda, mamba, or
Miniforge.

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

- **PyTorch 2.12.0** from the selected PyTorch wheel extra (`gpu` or `cpu`)
- **Hydra + OmegaConf** for configuration
- **Lightning 2.6.1** for training
- **timm**, **transformers**, **diffusers** for model backbones
- **albumentations** for image augmentation
- **wandb** for experiment tracking
- **Dev tooling** (pytest, pytest-cov, ruff, pre-commit) from the `dev`
  dependency group, which `uv sync` includes by default — pass `--no-dev` for a
  runtime-only install

### Optional ExecuTorch Dependency

Python 3.13 environments can install ExecuTorch from PyPI through the optional
extra:

```bash
PYTHON_VERSION=3.13
uv sync --python "$PYTHON_VERSION" --extra cpu --extra executorch
# Use --extra gpu instead of --extra cpu when installing the CUDA PyTorch stack.
```

The `executorch` extra is ignored on Python 3.14 by package markers because the
published ExecuTorch wheel currently declares `requires-python = ">=3.10,<3.14"`.
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
mamba activate versatil
pytest
```

To verify CUDA availability:

```bash
python -c "import torch; print(torch.cuda.is_available())"
```
