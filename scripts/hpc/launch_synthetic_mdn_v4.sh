#!/bin/bash
# MDN relaunch under v4 code (scale-fix + load_balance).
# Sweep: num_mixture_components ∈ {K, 2K} per task.
# Fixed: entropy_weight=0.05, load_balance_weight=0.2,
#        gmm_init_strategy=kmeans_plus_plus, action_sample_size=1000.
#
# 6 tasks × 2 head counts = 12 runs, sequential on 1 GPU.
# Expected wall time: ~9h (circle/cond_circle ~30min, others ~45min).
#
# Prerequisite: HPC repo must be pulled past commit 7dd784e ("Add load balancing for MDN").
#
# Usage: bash scripts/hpc/launch_synthetic_mdn_v4.sh
set -eo pipefail
cd "$(dirname "$0")/../.."

WORKSPACE=/data/horse/ws/loma592g-imitation_learning
LOGDIR=${WORKSPACE}/logs
mkdir -p "${LOGDIR}"

sbatch \
    --account=p_dl_surgery \
    --partition=capella \
    --nodes=1 --ntasks=1 --cpus-per-task=8 \
    --gres=gpu:1 --time=7-00:00:00 \
    --exclude=c110 \
    --chdir="${WORKSPACE}/code/versatil" \
    --job-name="syn_mdn_v4" \
    --output="${LOGDIR}/syn_mdn_v4_%j.out" \
    --error="${LOGDIR}/syn_mdn_v4_%j.err" \
    --wrap="
export PATH=/data/horse/ws/loma592g-environments/versatil/bin:\$PATH
cd ${WORKSPACE}/code/versatil

for TASK in circle conditional_circle sequential radial_k8 radial_k16 corridor_k8; do
  python -m versatil.endpoints.train --multirun \
    --config-name sweeps/synthetic/mdn/\${TASK}
done
"

echo "Submitted syn_mdn_v4 on 1 GPU — 12 runs sequential (~9h total)."
