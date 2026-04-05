#!/usr/bin/env bash
set -euo pipefail

MODE="${MODE:-train}"
STAGE="${STAGE:-smoke}"
RUNS_ROOT="${RUNS_ROOT:-runs}"
RUN_ID="${RUN_ID:-}"
CHECKPOINT_PATH="${CHECKPOINT_PATH:-}"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-5,6,7}"

if [[ "$MODE" == "train" ]]; then
  if [[ -n "$RUN_ID" ]]; then
    uv run --extra train python -m backend.app.rl.train --stage "$STAGE" --runs-root "$RUNS_ROOT" --run-id "$RUN_ID"
  else
    uv run --extra train python -m backend.app.rl.train --stage "$STAGE" --runs-root "$RUNS_ROOT"
  fi
elif [[ "$MODE" == "eval" ]]; then
  if [[ -n "$RUN_ID" && -n "$CHECKPOINT_PATH" ]]; then
    uv run --extra train python -m backend.app.rl.eval --runs-root "$RUNS_ROOT" --run-id "$RUN_ID" --checkpoint-path "$CHECKPOINT_PATH"
  elif [[ -n "$RUN_ID" ]]; then
    uv run --extra train python -m backend.app.rl.eval --runs-root "$RUNS_ROOT" --run-id "$RUN_ID"
  elif [[ -n "$CHECKPOINT_PATH" ]]; then
    uv run --extra train python -m backend.app.rl.eval --runs-root "$RUNS_ROOT" --checkpoint-path "$CHECKPOINT_PATH"
  else
    uv run --extra train python -m backend.app.rl.eval --runs-root "$RUNS_ROOT"
  fi
else
  echo "Unsupported MODE=$MODE (expected train or eval)" >&2
  exit 1
fi
