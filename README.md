## Needle-Threading
This repository provides a framework for training Imitation Learning policies on the Needle Threading task in robotic manipulation. It supports multiple policy architectures, flexible configurations, and distributed training using SLURM with PyTorch Distributed Data Parallel (DDP).
Key Features

- Supported Policies: Diffusion Policy, Action Chunking with Transformers (ACT), and Flow Matching Policy.
- Distributed Training: Seamless integration with SLURM and environment variables for multi-node setups.
- Data Pipeline: Includes data augmentation, replay buffer creation (in Zarr format), and efficient PyTorch DataLoaders.

### Prerequisites

- Python 3.11+
- CUDA 12.4+ for GPU acceleration (required for training)
- SLURM cluster for distributed training (optional)
- Dataset: Prepare your Needle Threading dataset in the format expected by the data pipeline (e.g., episode directories with CSV files). Place them in a folder specified via config.dataset_folder.

### Installation
1. Clone the repository:
```bash
git clone https://gitlab.com/nct_tso_public/needlethreading_il.git
```
2. Create and activate a Conda or Mamba environment using the provided environment.yml::
   ```bash
   conda env create -f environment.yml
   conda activate needle-driving
    ```
Or with Mamba (recommended for faster installation):
```bash
   mamba create -f environment.yml
   mamba activate needle-driving
```

### Usage
#### Preparing Data

Ensure your dataset is organized in episode directories (e.g., under data/).
The data pipeline in src/dataset/dataloader.py will automatically create Zarr replay buffers for training and validation splits during the first run.
#### Single-Node Training
Launch a training script directly:
```bash
python diffusion_endpoint.py
```

You can also train the policies using a custom configuration file that overrides the default parameters of the models. The custom config file must be a `json` with `policy_name` included as one of the attributes. An example config file for each policy is located at `src/endpoints/default_config/`. To launch the training script:
```bash
python start_training.py --custom_config_path="/PATH/TO/DEFAULT/CONFIG.json"
```

Replace with other policy scripts as needed (e.g., act_endpoint.py or flow_matching_endpoint.py).
#### Distributed Training
A SLURM batch script (e.g., run_distributed.sh) is provided. It sets up the necessary environment variables for 
distributed training. To submit a job, run:
```bash
sbatch run_distributed.sh
```
#### Configuration
All hyperparameters and settings (e.g., image dimensions, camera names, prediction horizons, batch sizes) are defined in src/config.py via the PolicyConfig class. 
Modify this file to customize:
- Policy-specific parameters (e.g., diffusion steps, transformer layers).
- Data options (e.g., augmentation flags, normalization type).
- Training setup (e.g., epochs, learning rate, device).

#### Troubleshooting

- CUDA Issues: Ensure your CUDA version matches (12.4). Verify with nvidia-smi.
- SLURM Errors: Check `export NCCL_P2P_DISABLE=1` is set. This disables NCCL P2P which can cause issues in our cluster.
- Data Loading: If Zarr creation fails, verify dataset paths and permissions.