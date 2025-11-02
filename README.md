## Surg-IL
This repository provides a framework for training Imitation Learning policies on surgical tasks.
### Prerequisites
- Python 3.11+
- CUDA 12.4+ for GPU acceleration (required for training)
### Installation
1. Clone the repository:
```bash
git clone https://gitlab.com/nct_tso_public/surg-il.git
```
2. Store your git credentials to be able to access private repositories:
```bash
git config --global credential.helper store
git ls-remote https://gitlab.com/nct_tso_public/imitation-learning-toolkit.git
```
3.Create and activate a Conda or Mamba environment using the provided environment.yml. Then install dependencies with `uv`:
   ```bash
   conda env create -f environment.yml
   conda activate surg-il
   UV_PROJECT_ENVIRONMENT=$CONDA_PREFIX uv sync
    ```
Or with Mamba (recommended for faster installation):
```bash
   mamba create -f environment.yml
   mamba activate surg-il
   UV_PROJECT_ENVIRONMENT=$CONDA_PREFIX uv sync
```
This will create a conda/mamba environment with all necessary packages, installed by uv.

#### Troubleshooting

- CUDA Issues: Ensure your CUDA version matches (12.4). Verify with nvidia-smi.
- SLURM Errors: Check `export NCCL_P2P_DISABLE=1` is set. This disables NCCL P2P which can cause issues in our cluster.
- Data Loading: If Zarr creation fails, verify dataset paths and permissions.