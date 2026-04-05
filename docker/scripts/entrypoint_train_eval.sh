#!/usr/bin/env bash
set -euo pipefail

MODE="${MODE:-train}"
STAGE="${STAGE:-smoke}"
RUNS_ROOT="${RUNS_ROOT:-runs}"
RUN_ID="${RUN_ID:-}"
CHECKPOINT_PATH="${CHECKPOINT_PATH:-}"
TRAIN_CONFIG_OVERRIDES_PATH="${TRAIN_CONFIG_OVERRIDES_PATH:-}"
TRAIN_TIMESTEPS="${TRAIN_TIMESTEPS:-}"
EVAL_SEEDS_PER_SCENARIO="${EVAL_SEEDS_PER_SCENARIO:-2}"
EVAL_SEED_START="${EVAL_SEED_START:-1001}"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-5,6,7}"

if [[ "$MODE" == "train" ]]; then
  TRAIN_ARGS=(uv run --extra train python -m backend.app.rl.train --stage "$STAGE" --runs-root "$RUNS_ROOT")
  if [[ -n "$RUN_ID" ]]; then
    TRAIN_ARGS+=(--run-id "$RUN_ID")
  fi
  if [[ -n "$TRAIN_CONFIG_OVERRIDES_PATH" ]]; then
    TRAIN_ARGS+=(--config-overrides-path "$TRAIN_CONFIG_OVERRIDES_PATH")
  fi
  if [[ -n "$TRAIN_TIMESTEPS" ]]; then
    TRAIN_ARGS+=(--train-timesteps "$TRAIN_TIMESTEPS")
  fi

  "${TRAIN_ARGS[@]}"
elif [[ "$MODE" == "eval" ]]; then
  EVAL_ARGS=(uv run --extra train python -m backend.app.rl.eval --runs-root "$RUNS_ROOT" --seeds-per-scenario "$EVAL_SEEDS_PER_SCENARIO" --seed-start "$EVAL_SEED_START")
  if [[ -n "$RUN_ID" && -n "$CHECKPOINT_PATH" ]]; then
    EVAL_ARGS+=(--run-id "$RUN_ID" --checkpoint-path "$CHECKPOINT_PATH")
  elif [[ -n "$RUN_ID" ]]; then
    EVAL_ARGS+=(--run-id "$RUN_ID")
  elif [[ -n "$CHECKPOINT_PATH" ]]; then
    EVAL_ARGS+=(--checkpoint-path "$CHECKPOINT_PATH")
  fi
  "${EVAL_ARGS[@]}"
else
  echo "Unsupported MODE=$MODE (expected train or eval)" >&2
  exit 1
fi
