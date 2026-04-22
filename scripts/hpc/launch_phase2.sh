#!/bin/bash
# Phase 2 confirmation sweep: 420 runs, 7 sbatch jobs × 60 runs.
# Tasks: circle (K=2, 2k epochs, latent=1), sequential (K=4, 4k, latent=16),
#        corridor_k8 (4k, latent=16), corridor_k16 (4k, latent=16), radial_k16 (4k, latent=16).
# Seeds: 42, 43, 44.
# Protocol: running mean over last 5 rollout checkpoints, 200 rollouts/ckpt
# (requires num_rollouts=200 in synthetic.py:144, already edited).
# Mirrors task_bundle/synthetic/*.yaml defaults (latent_dim + num_epochs) via
# explicit CLI overrides since +task_bundle/synthetic=X is not resolvable from
# CLI relative to an e2e config.

set -euo pipefail
CODE=/data/horse/ws/loma592g-imitation_learning/code/versatil
LOG_DIR=/data/horse/ws/loma592g-imitation_learning/logs/phase2
mkdir -p "$LOG_DIR"

SBATCH_HEADER='#!/bin/bash
#SBATCH --account=p_dl_surgery
#SBATCH --partition=capella
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --mem=48G
#SBATCH --time=7-00:00:00
#SBATCH --exclude=c110
#SBATCH --chdir='"$CODE"'
#SBATCH --output='"$LOG_DIR"'/%x_%j.out

export PATH=/data/horse/ws/loma592g-environments/versatil/bin:$PATH
export WANDB_INIT_TIMEOUT=1800
export WANDB__SERVICE_WAIT=300
cd '"$CODE"'

task_base() {
  case "$1" in
    circle)       echo "task/dataset_schema=synthetic/circle training.num_epochs=2000" ;;
    sequential)   echo "task/dataset_schema=synthetic/sequential training.num_epochs=4000" ;;
    corridor_k8)  echo "task/dataset_schema=synthetic/corridor_navigation task.dataset_schema.num_modes=8 training.num_epochs=4000" ;;
    corridor_k16) echo "task/dataset_schema=synthetic/corridor_navigation task.dataset_schema.num_modes=16 training.num_epochs=4000" ;;
    radial_k16)   echo "task/dataset_schema=synthetic/radial task.dataset_schema.num_modes=16 training.num_epochs=4000" ;;
  esac
}

task_latent() {
  case "$1" in
    circle) echo "1" ;;
    *)      echo "16" ;;
  esac
}

latent_ov() {
  local ld=$(task_latent "$1")
  echo "policy.algorithm.posterior_encoder.latent_dimension=$ld policy.algorithm.prior.latent_dimension=$ld"
}

TASKS="circle sequential corridor_k8 corridor_k16 radial_k16"
SEEDS="42 43 44"
TRAIN="python -m versatil.endpoints.train"
'

sbatch --job-name=p2_kl <<EOF
$SBATCH_HEADER
for ARCH in AT LACT; do
  case \$ARCH in
    AT)   CFG=end_to_end_training_runs/synthetic/action_transformer_kl_gaussian ;;
    LACT) CFG=end_to_end_training_runs/synthetic/lact_kl_gaussian ;;
  esac
  for SEED in \$SEEDS; do
    for W in 0.001 0.01; do
      for TASK in \$TASKS; do
        \$TRAIN --config-name \$CFG \$(task_base \$TASK) \$(latent_ov \$TASK) experiment.seed=\$SEED \\
          policy.loss.loss_modules.kl_divergence.weight=\$W \\
          experiment.name=phase2/kl_\${ARCH}/\${TASK}_w\${W}_s\${SEED}
      done
    done
  done
done
EOF

sbatch --job-name=p2_mmd <<EOF
$SBATCH_HEADER
for ARCH in AT LACT; do
  case \$ARCH in
    AT)   CFG=end_to_end_training_runs/synthetic/action_transformer_mmd_gaussian ;;
    LACT) CFG=end_to_end_training_runs/synthetic/lact_mmd_gaussian ;;
  esac
  for SEED in \$SEEDS; do
    for W in 1 10; do
      for TASK in \$TASKS; do
        \$TRAIN --config-name \$CFG \$(task_base \$TASK) \$(latent_ov \$TASK) experiment.seed=\$SEED \\
          policy.loss.loss_modules.maximum_mean_discrepancy.weight=\$W \\
          policy.loss.loss_modules.maximum_mean_discrepancy.kernel_type=imq \\
          policy.loss.loss_modules.maximum_mean_discrepancy.use_median_heuristic=true \\
          experiment.name=phase2/mmd_\${ARCH}/\${TASK}_w\${W}_s\${SEED}
      done
    done
  done
done
EOF

sbatch --job-name=p2_ot <<EOF
$SBATCH_HEADER
for ARCH in AT LACT; do
  case \$ARCH in
    AT)   CFG=end_to_end_training_runs/synthetic/action_transformer_ot_gaussian ;;
    LACT) CFG=end_to_end_training_runs/synthetic/lact_ot_gaussian ;;
  esac
  for SEED in \$SEEDS; do
    for W in 0.01 0.1; do
      for TASK in \$TASKS; do
        \$TRAIN --config-name \$CFG \$(task_base \$TASK) \$(latent_ov \$TASK) experiment.seed=\$SEED \\
          policy.loss.loss_modules.latent_ot.weight=\$W \\
          policy.loss.loss_modules.latent_ot.blur_fraction=0.1 \\
          policy.loss.loss_modules.latent_ot.reach_multiplier=null \\
          experiment.name=phase2/ot_\${ARCH}/\${TASK}_w\${W}_s\${SEED}
      done
    done
  done
done
EOF

sbatch --job-name=p2_ditfm_at <<EOF
$SBATCH_HEADER
CFG=end_to_end_training_runs/synthetic/action_transformer_dit_prior
for SEED in \$SEEDS; do
  for W in 0.1 1.0; do
    for NFE in 10 100; do
      for TASK in \$TASKS; do
        \$TRAIN --config-name \$CFG \$(task_base \$TASK) \$(latent_ov \$TASK) experiment.seed=\$SEED \\
          policy.loss.loss_modules.denoising_prior.weight=\$W \\
          policy.algorithm.prior.num_inference_steps=\$NFE \\
          experiment.name=phase2/ditfm_AT/\${TASK}_w\${W}_nfe\${NFE}_s\${SEED}
      done
    done
  done
done
EOF

sbatch --job-name=p2_ditfm_lact <<EOF
$SBATCH_HEADER
CFG=end_to_end_training_runs/synthetic/lact_dit_prior
for SEED in \$SEEDS; do
  for W in 0.1 1.0; do
    for NFE in 10 100; do
      for TASK in \$TASKS; do
        \$TRAIN --config-name \$CFG \$(task_base \$TASK) \$(latent_ov \$TASK) experiment.seed=\$SEED \\
          policy.loss.loss_modules.denoising_prior.weight=\$W \\
          policy.algorithm.prior.num_inference_steps=\$NFE \\
          experiment.name=phase2/ditfm_LACT/\${TASK}_w\${W}_nfe\${NFE}_s\${SEED}
      done
    done
  done
done
EOF

sbatch --job-name=p2_ditdiff_at <<EOF
$SBATCH_HEADER
CFG=end_to_end_training_runs/synthetic/action_transformer_dit_prior
for SEED in \$SEEDS; do
  for W in 0.1 1.0; do
    for NFE in 10 100; do
      for TASK in \$TASKS; do
        \$TRAIN --config-name \$CFG \$(task_base \$TASK) \$(latent_ov \$TASK) experiment.seed=\$SEED \\
          policy.loss.loss_modules.denoising_prior.weight=\$W \\
          policy.algorithm.prior.num_inference_steps=\$NFE \\
          'policy.algorithm.prior.algorithm_type=\${denoising_algorithm:DIFFUSION}' \\
          experiment.name=phase2/ditdiff_AT/\${TASK}_w\${W}_nfe\${NFE}_s\${SEED}
      done
    done
  done
done
EOF

sbatch --job-name=p2_mmdit <<EOF
$SBATCH_HEADER
for METHOD in flow diff; do
  case \$METHOD in
    flow) CFG=end_to_end_training_runs/synthetic/mmdit_flow_matching ;;
    diff) CFG=end_to_end_training_runs/synthetic/mmdit_diffusion ;;
  esac
  for SEED in \$SEEDS; do
    for NFE in 10 100; do
      for TASK in \$TASKS; do
        \$TRAIN --config-name \$CFG \$(task_base \$TASK) experiment.seed=\$SEED \\
          policy.algorithm.num_inference_steps=\$NFE \\
          experiment.name=phase2/mmdit_\${METHOD}/\${TASK}_nfe\${NFE}_s\${SEED}
      done
    done
  done
done
EOF

echo "Submitted 7 Phase 2 sbatch jobs: p2_kl, p2_mmd, p2_ot, p2_ditfm_at, p2_ditfm_lact, p2_ditdiff_at, p2_mmdit"
echo "Total: 420 runs (60 per sbatch)."
