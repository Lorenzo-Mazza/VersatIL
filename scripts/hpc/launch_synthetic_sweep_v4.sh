#!/bin/bash
# Synthetic sweep v4: LACT comparison + AT/LACT variational grids + stage-2 learned priors.
# Each sbatch job runs one Hydra multirun sweep defined in hydra_configs/sweeps/synthetic/.
#
# Usage: bash scripts/hpc/launch_synthetic_sweep_v4.sh
set -eo pipefail
cd "$(dirname "$0")/../.."

WORKSPACE=/data/horse/ws/loma592g-imitation_learning
LOGDIR=${WORKSPACE}/logs
mkdir -p "${LOGDIR}"

SBATCH_COMMON=(
    --account=p_dl_surgery
    --partition=capella
    --nodes=1 --ntasks=1 --cpus-per-task=8
    --gres=gpu:1 --time=7-00:00:00
    --exclude=c110
    --chdir="${WORKSPACE}/code/versatil"
)
ENV_SETUP="export PATH=/data/horse/ws/loma592g-environments/versatil/bin:\$PATH; cd ${WORKSPACE}/code/versatil"

submit_sweep() {
    local name="$1"; local sweep="$2"; local runs="$3"
    sbatch "${SBATCH_COMMON[@]}" \
      --job-name="${name}" \
      --output="${LOGDIR}/${name}_v4_%j.out" \
      --error="${LOGDIR}/${name}_v4_%j.err" \
      --wrap="${ENV_SETUP}
python -m versatil.endpoints.train --multirun --config-name ${sweep}"
    echo "  ${name}: ${runs} runs via ${sweep}"
}

echo "Stage 1 — AT vs LACT × variational objectives:"
submit_sweep syn_at_kl       sweeps/synthetic/at_kl_gaussian         30
submit_sweep syn_lact_kl     sweeps/synthetic/lact_kl_gaussian       30
submit_sweep syn_at_mmd      sweeps/synthetic/at_mmd_gaussian       120
submit_sweep syn_lact_mmd    sweeps/synthetic/lact_mmd_gaussian     120
submit_sweep syn_at_ot       sweeps/synthetic/at_ot_gaussian        162
submit_sweep syn_lact_ot     sweeps/synthetic/lact_ot_gaussian      162
submit_sweep syn_at_fmdit    sweeps/synthetic/at_dit_prior            6
submit_sweep syn_lact_fmdit  sweeps/synthetic/lact_dit_prior          6
submit_sweep syn_fm100       sweeps/synthetic/mmdit_flow_matching     6

echo "Stage 1 — MDN per-task sweeps (best-case k-means++ with 100 action samples):"
submit_sweep syn_mdn_circle     sweeps/synthetic/mdn/circle              6
submit_sweep syn_mdn_condcircle sweeps/synthetic/mdn/conditional_circle  6
submit_sweep syn_mdn_seq        sweeps/synthetic/mdn/sequential          6
submit_sweep syn_mdn_radk8      sweeps/synthetic/mdn/radial_k8           6
submit_sweep syn_mdn_radk16     sweeps/synthetic/mdn/radial_k16          6
submit_sweep syn_mdn_corr       sweeps/synthetic/mdn/corridor_k8         6

echo "Stage 2 — learned Gaussian prior × {KL w=0.01, MMD w=10, OT b0.5/r1/w1}:"
submit_sweep syn_at_lp_kl    sweeps/synthetic/at_learned_prior_kl_w0.01    6
submit_sweep syn_lact_lp_kl  sweeps/synthetic/lact_learned_prior_kl_w0.01  6
submit_sweep syn_at_lp_mmd   sweeps/synthetic/at_learned_prior_mmd_w10     6
submit_sweep syn_lact_lp_mmd sweeps/synthetic/lact_learned_prior_mmd_w10   6
submit_sweep syn_at_lp_ot    sweeps/synthetic/at_learned_prior_ot          6
submit_sweep syn_lact_lp_ot  sweeps/synthetic/lact_learned_prior_ot        6

echo
echo "Submitted 21 sbatch jobs. Totals per stage:"
echo "  Stage 1 variational: 30+30+120+120+162+162+6+6+6 = 642 runs"
echo "  Stage 1 MDN:         6*6                         =  36 runs"
echo "  Stage 2 learned:     6*6                         =  36 runs"
echo "  Total:                                             714 runs"
