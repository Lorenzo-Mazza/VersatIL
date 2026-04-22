#!/bin/bash
#SBATCH --job-name=libero_sweep
#SBATCH --gpus=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=400G
#SBATCH --time=72:00:00
#SBATCH --output=/mnt/cluster/workspaces/mazzalore/eval_logs/sweep_%j.log
#SBATCH --error=/mnt/cluster/workspaces/mazzalore/eval_logs/sweep_%j.err

set -eo pipefail

CKPT_DIR="${1:?ckpt_dir required}"
CKPT_NAME="${2:?ckpt_name required}"
TAG="${3:?tag required}"

CKPT_DIR="$(echo -n "${CKPT_DIR}" | tr -d '[:space:]')"
CKPT_NAME="$(echo -n "${CKPT_NAME}" | tr -d '[:space:]')"
TAG="$(echo -n "${TAG}" | tr -d '[:space:]')"

mkdir -p /mnt/cluster/workspaces/mazzalore/eval_logs

bash /mnt/cluster/workspaces/mazzalore/surg-il/scripts/sweep_suites_parallel.sh \
    "${CKPT_DIR}" "${CKPT_NAME}" "${TAG}"
