#!/bin/bash
# Launch all 4 libero_plus suites in parallel against one checkpoint.
# Each suite runs its own server + client on a distinct port.
#
# Usage: bash scripts/sweep_suites_parallel.sh <ckpt_dir> <ckpt_name> <tag>

set -eo pipefail

CKPT_DIR="${1:?ckpt_dir required}"
CKPT_NAME="${2:?ckpt_name required}"
TAG="${3:?tag required}"

RUNNER=/mnt/cluster/workspaces/mazzalore/surg-il/scripts/run_libero_local.sh
LOGDIR=/tmp/libero_local_eval
mkdir -p "${LOGDIR}"

SUITES=(libero_plus_spatial libero_plus_object libero_plus_goal libero_plus_10)

# Derive a unique 4-port block per invocation so multiple sweeps on the
# same node (e.g. one per GPU) don't collide. Uses this script's PID as
# the offset, clamped into a high range with headroom for the 4 ports.
BASE_PORT=$((30000 + ($$ % 2500) * 4))
PORTS=($BASE_PORT $((BASE_PORT + 1)) $((BASE_PORT + 2)) $((BASE_PORT + 3)))
echo "[sweep] using port block: ${PORTS[*]}"

pids=()
for i in "${!SUITES[@]}"; do
    suite="${SUITES[$i]}"
    port="${PORTS[$i]}"
    echo "[sweep] launching ${suite} on port ${port}  (tag=${TAG})"
    bash "${RUNNER}" "${CKPT_DIR}" "${CKPT_NAME}" "${suite}" "${TAG}" "${port}" \
        > "${LOGDIR}/${TAG}_${suite}_runner.log" 2>&1 &
    pids+=($!)
done

echo "[sweep] launched ${#pids[@]} suites, pids: ${pids[*]}"
echo "[sweep] waiting for all to finish..."
fail=0
for pid in "${pids[@]}"; do
    if ! wait "${pid}"; then
        echo "[sweep] runner pid ${pid} failed"
        fail=1
    fi
done
echo "[sweep] done (fail=${fail})"
exit ${fail}
