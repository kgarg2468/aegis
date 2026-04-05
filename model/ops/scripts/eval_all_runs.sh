#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
cd "${PROJECT_ROOT}"

RUNS_ROOT="${RUNS_ROOT:-runs}"
USE_DOCKER_RAW="${USE_DOCKER:-false}"
DOCKER_IMAGE="${DOCKER_IMAGE:-shield-train}"
DOCKER_PLATFORM="${DOCKER_PLATFORM:-}"
SHM_SIZE="${SHM_SIZE:-2g}"
CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-5,6,7}"
EVAL_SEEDS_PER_SCENARIO="${EVAL_SEEDS_PER_SCENARIO:-2}"
EVAL_SEED_START="${EVAL_SEED_START:-1001}"

use_docker="$(echo "${USE_DOCKER_RAW}" | tr '[:upper:]' '[:lower:]')"

if [[ ! -d "${RUNS_ROOT}" ]]; then
  echo "Runs root not found: ${RUNS_ROOT}" >&2
  exit 1
fi

eligible_runs=()
while IFS= read -r run_dir; do
  if [[ -d "${run_dir}/train/final" ]]; then
    eligible_runs+=("$(basename "${run_dir}")")
  fi
done < <(find "${RUNS_ROOT}" -mindepth 1 -maxdepth 1 -type d -name 'run*' | sort)

if (( ${#eligible_runs[@]} == 0 )); then
  echo "No eligible runs found under ${RUNS_ROOT} (expected ${RUNS_ROOT}/runNNN/train/final)." >&2
  exit 1
fi

echo "Found ${#eligible_runs[@]} eligible runs: ${eligible_runs[*]}"

for run_id in "${eligible_runs[@]}"; do
  checkpoint_path="${RUNS_ROOT}/${run_id}/train/final"
  echo "=== Evaluating ${run_id} (${checkpoint_path}) ==="

  if [[ "${use_docker}" == "true" || "${use_docker}" == "1" || "${use_docker}" == "yes" ]]; then
    docker_args=(run --rm)
    if [[ -n "${DOCKER_PLATFORM}" ]]; then
      docker_args+=(--platform "${DOCKER_PLATFORM}")
    fi
    docker_args+=(
      --shm-size "${SHM_SIZE}"
      -e MODE=eval
      -e RUNS_ROOT="${RUNS_ROOT}"
      -e RUN_ID="${run_id}"
      -e CHECKPOINT_PATH="${checkpoint_path}"
      -e EVAL_SEEDS_PER_SCENARIO="${EVAL_SEEDS_PER_SCENARIO}"
      -e EVAL_SEED_START="${EVAL_SEED_START}"
      -v "${PWD}:/app"
    )
    docker "${docker_args[@]}" "${DOCKER_IMAGE}"
  else
    CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES}" uv run --extra train python -m backend.app.rl.eval \
      --runs-root "${RUNS_ROOT}" \
      --run-id "${run_id}" \
      --checkpoint-path "${checkpoint_path}" \
      --seeds-per-scenario "${EVAL_SEEDS_PER_SCENARIO}" \
      --seed-start "${EVAL_SEED_START}"
  fi

  latest_eval="$(ls -1t "${RUNS_ROOT}/${run_id}/eval"/eval_cli_*.json 2>/dev/null | head -n1 || true)"
  if [[ -n "${latest_eval}" ]]; then
    echo "Latest eval for ${run_id}: ${latest_eval}"
  fi
  echo

done
