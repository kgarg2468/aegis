SHELL := /bin/zsh

CUDA_VISIBLE_DEVICES ?= 5,6,7
RUNS_ROOT ?= runs
RUN_ID ?=
CHECKPOINT ?=
BUNDLE ?=
NUM_RUNS ?= 10
DOCKER_IMAGE ?= shield-train
USE_DOCKER ?= true
AUTO_TUNE ?= true
START_RUN_ID ?=
TRAIN_TIMESTEPS ?=
EVAL_SEEDS_PER_SCENARIO ?=2
EVAL_SEED_START ?=1001
SWEEP_SPEC ?=
DOCKER_PLATFORM ?=
SHM_SIZE ?=2g
FRONTEND_ROOT ?= ../frontend

.PHONY: sync sync-train test train-smoke train-sanity train eval eval-all package-replays validate-replay autopilot demo-episodes verify-demo-episodes

sync:
	uv sync --extra dev

sync-train:
	uv sync --extra dev --extra train

test:
	uv run pytest -q

train-smoke:
	CUDA_VISIBLE_DEVICES=$(CUDA_VISIBLE_DEVICES) uv run --extra train python -m backend.app.rl.train --stage smoke --runs-root $(RUNS_ROOT) $(if $(RUN_ID),--run-id $(RUN_ID),) $(if $(TRAIN_TIMESTEPS),--train-timesteps $(TRAIN_TIMESTEPS),)

train-sanity:
	CUDA_VISIBLE_DEVICES=$(CUDA_VISIBLE_DEVICES) uv run --extra train python -m backend.app.rl.train --stage sanity --runs-root $(RUNS_ROOT) $(if $(RUN_ID),--run-id $(RUN_ID),) $(if $(TRAIN_TIMESTEPS),--train-timesteps $(TRAIN_TIMESTEPS),)

train:
	CUDA_VISIBLE_DEVICES=$(CUDA_VISIBLE_DEVICES) uv run --extra train python -m backend.app.rl.train --stage full --runs-root $(RUNS_ROOT) $(if $(RUN_ID),--run-id $(RUN_ID),) $(if $(TRAIN_TIMESTEPS),--train-timesteps $(TRAIN_TIMESTEPS),)

eval:
	CUDA_VISIBLE_DEVICES=$(CUDA_VISIBLE_DEVICES) uv run --extra train python -m backend.app.rl.eval --runs-root $(RUNS_ROOT) --seeds-per-scenario $(EVAL_SEEDS_PER_SCENARIO) --seed-start $(EVAL_SEED_START) $(if $(RUN_ID),--run-id $(RUN_ID),) $(if $(CHECKPOINT),--checkpoint-path $(CHECKPOINT),)

eval-all:
	RUNS_ROOT=$(RUNS_ROOT) USE_DOCKER=$(USE_DOCKER) DOCKER_IMAGE=$(DOCKER_IMAGE) DOCKER_PLATFORM=$(DOCKER_PLATFORM) SHM_SIZE=$(SHM_SIZE) CUDA_VISIBLE_DEVICES=$(CUDA_VISIBLE_DEVICES) EVAL_SEEDS_PER_SCENARIO=$(EVAL_SEEDS_PER_SCENARIO) EVAL_SEED_START=$(EVAL_SEED_START) ./ops/scripts/eval_all_runs.sh

package-replays:
	CUDA_VISIBLE_DEVICES=$(CUDA_VISIBLE_DEVICES) uv run --extra train python -m backend.app.rl.eval --runs-root $(RUNS_ROOT) --seeds-per-scenario $(EVAL_SEEDS_PER_SCENARIO) --seed-start $(EVAL_SEED_START) $(if $(RUN_ID),--run-id $(RUN_ID),) $(if $(CHECKPOINT),--checkpoint-path $(CHECKPOINT),)

validate-replay:
	uv run python ops/scripts/validate_replay.py $(BUNDLE)

autopilot:
	CUDA_VISIBLE_DEVICES=$(CUDA_VISIBLE_DEVICES) uv run --extra train python -m backend.app.rl.autopilot --num-runs $(NUM_RUNS) --runs-root $(RUNS_ROOT) --stage full --docker-image $(DOCKER_IMAGE) --seeds-per-scenario $(EVAL_SEEDS_PER_SCENARIO) --seed-start $(EVAL_SEED_START) $(if $(filter false,$(USE_DOCKER)),--no-use-docker,--use-docker) $(if $(filter false,$(AUTO_TUNE)),--no-auto-tune,--auto-tune) $(if $(START_RUN_ID),--start-run-id $(START_RUN_ID),) $(if $(TRAIN_TIMESTEPS),--train-timesteps $(TRAIN_TIMESTEPS),) $(if $(SWEEP_SPEC),--sweep-spec-path $(SWEEP_SPEC),)

demo-episodes:
	uv run python ops/scripts/build_episode_runs.py --runs-root $(RUNS_ROOT) --frontend-root $(FRONTEND_ROOT) --clean

verify-demo-episodes:
	uv run python ops/scripts/build_episode_runs.py --runs-root $(RUNS_ROOT) --frontend-root $(FRONTEND_ROOT) --verify-only
