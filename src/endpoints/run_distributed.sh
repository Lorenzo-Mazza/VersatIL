#!/bin/bash
#SBATCH --job-name=distributed_diffusionpolicy
#SBATCH --gres=gpu:rtxa5000:2
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=2
#SBATCH --cpus-per-task=16
#SBATCH --time=72:00:00
#SBATCH --output=distributed_%j.log
#SBATCH --mem=32G               # total memory per node (4 GB per cpu-core is default)

# Debug info
echo "SLURM_JOB_ID: $SLURM_JOB_ID"
echo "SLURM_JOB_NODELIST: $SLURM_JOB_NODELIST"
echo "CUDA_VISIBLE_DEVICES: $CUDA_VISIBLE_DEVICES"
echo "Ntasks per node:= "  $SLURM_NTASKS_PER_NODE
echo "Number of nodes:= " $SLURM_JOB_NUM_NODES
echo "GPUS on node:= " $SLURM_GPUS_ON_NODE
echo "SLURM PROC ID:= " $SLURM_PROCID

# ******************* These are read internally it seems ***********************************
# ******** Master port, address and world size MUST be passed as variables for DDP to work
export MASTER_PORT=$(expr 10000 + $(echo -n $SLURM_JOBID | tail -c 4))
export WORLD_SIZE=$(($SLURM_NNODES * $SLURM_NTASKS_PER_NODE))
echo "MASTER_PORT="$MASTER_PORT
echo "WORLD_SIZE="$WORLD_SIZE

master_addr=$(scontrol show hostnames "$SLURM_JOB_NODELIST" | head -n 1)
export MASTER_ADDR=$master_addr
echo "MASTER_ADDR="$MASTER_ADDR
# ******************************************************************************************
export NCCL_P2P_DISABLE=1        # forces NCCL to fall back to shared memory. P2P is broken on the cluster drivers apparently
# Enable distributed training via environment variable
export CONFIG_DISTRIBUTED_TRAINING=true

echo "Run started at:- "
date

# Run the training script
srun --label -u python start_training.py --custom_config_path="/PATH/TO/CUSTOM/CONFIG.json"
