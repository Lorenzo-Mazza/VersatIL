#!/bin/bash
# Local (non-SLURM) libero-plus evaluation. Starts the sim server in
# libero_plus env and the versatil client in versatil-313 env against one GPU.
#
# Usage: bash scripts/run_libero_local.sh <checkpoint_path> <checkpoint_name> <task_suite> <tag> [port]

set -eo pipefail

CHECKPOINT_PATH="${1:?checkpoint_path required}"
CHECKPOINT_NAME="${2:?checkpoint_name required}"
TASK_SUITE="${3:?task_suite required}"
TAG="${4:?tag required}"
PORT="${5:-28765}"

WORKSPACE=/mnt/cluster/workspaces/mazzalore
LIBERO_PLUS_REPO=${WORKSPACE}/libero-plus
VERSATIL_REPO=${WORKSPACE}/surg-il
MAMBA_BIN=/mnt/cluster/environments/mazzalore/miniforge3/bin/mamba
LOGDIR=/tmp/libero_local_eval
RESULTS_DIR=${WORKSPACE}/eval/libero_plus_local/${TAG}/${TASK_SUITE}

mkdir -p "${LOGDIR}" "${RESULTS_DIR}"
SERVER_LOG=${LOGDIR}/${TAG}_${TASK_SUITE}_server.log
CLIENT_LOG=${LOGDIR}/${TAG}_${TASK_SUITE}_client.log

cleanup() {
    if [[ -n "${SERVER_PID:-}" ]] && kill -0 "${SERVER_PID}" 2>/dev/null; then
        kill "${SERVER_PID}" 2>/dev/null || true
        wait "${SERVER_PID}" 2>/dev/null || true
    fi
}
trap cleanup EXIT

echo "[runner] starting LIBERO-Plus server on port ${PORT}  (log=${SERVER_LOG})"
(
    export PYTHONUNBUFFERED=1
    export MUJOCO_GL=egl
    cd "${LIBERO_PLUS_REPO}"
    "${MAMBA_BIN}" run -n libero_plus python -m versatil_inference.run_evaluation \
        --task_suite_name "${TASK_SUITE}" \
        --ip_address 127.0.0.1 \
        --port "${PORT}" \
        --output_folder "${RESULTS_DIR}" \
        --run_id_note "${TAG}" \
        --max_parallel_envs 10
) > "${SERVER_LOG}" 2>&1 &
SERVER_PID=$!
echo "[runner] server pid=${SERVER_PID}"

echo "[runner] waiting for server to bind 127.0.0.1:${PORT}"
for _ in $(seq 1 120); do
    if (exec 3<>/dev/tcp/127.0.0.1/${PORT}) 2>/dev/null; then
        exec 3<&- ; exec 3>&-
        echo "[runner] server is up"; break
    fi
    if ! kill -0 "${SERVER_PID}" 2>/dev/null; then
        echo "[runner] server died before binding; see ${SERVER_LOG}" >&2
        tail -60 "${SERVER_LOG}" >&2
        exit 1
    fi
    sleep 5
done

echo "[runner] starting VersatIL client (log=${CLIENT_LOG})"
cd "${VERSATIL_REPO}"
set -a; source .env 2>/dev/null || true; set +a
"${MAMBA_BIN}" run -n versatil-313 python -m versatil.endpoints.test \
    --checkpoint_path "${CHECKPOINT_PATH}" \
    --checkpoint_name "${CHECKPOINT_NAME}" \
    --model_server_address 127.0.0.1 \
    --model_server_port "${PORT}" \
    --temporal_aggregation \
    --max_steps 1800000 \
    > "${CLIENT_LOG}" 2>&1

echo "[runner] client finished; waiting for server"
wait "${SERVER_PID}" || true
echo "[runner] done. logs: ${SERVER_LOG}  ${CLIENT_LOG}"
