SHELL := /bin/zsh

CUDA_VISIBLE_DEVICES ?= 5,6,7
RUNS_ROOT ?= runs
RUN_ID ?=
CHECKPOINT ?=
BUNDLE ?=

.PHONY: sync sync-train test train-smoke train-sanity train eval package-replays validate-replay

sync:
	uv sync --extra dev

sync-train:
	uv sync --extra dev --extra train

test:
	uv run pytest -q

train-smoke:
	CUDA_VISIBLE_DEVICES=$(CUDA_VISIBLE_DEVICES) uv run --extra train python -m backend.app.rl.train --stage smoke --runs-root $(RUNS_ROOT) $(if $(RUN_ID),--run-id $(RUN_ID),)

train-sanity:
	CUDA_VISIBLE_DEVICES=$(CUDA_VISIBLE_DEVICES) uv run --extra train python -m backend.app.rl.train --stage sanity --runs-root $(RUNS_ROOT) $(if $(RUN_ID),--run-id $(RUN_ID),)

train:
	CUDA_VISIBLE_DEVICES=$(CUDA_VISIBLE_DEVICES) uv run --extra train python -m backend.app.rl.train --stage full --runs-root $(RUNS_ROOT) $(if $(RUN_ID),--run-id $(RUN_ID),)

eval:
	CUDA_VISIBLE_DEVICES=$(CUDA_VISIBLE_DEVICES) uv run --extra train python -m backend.app.rl.eval --runs-root $(RUNS_ROOT) $(if $(RUN_ID),--run-id $(RUN_ID),) $(if $(CHECKPOINT),--checkpoint-path $(CHECKPOINT),)

package-replays:
	CUDA_VISIBLE_DEVICES=$(CUDA_VISIBLE_DEVICES) uv run --extra train python -m backend.app.rl.eval --runs-root $(RUNS_ROOT) $(if $(RUN_ID),--run-id $(RUN_ID),) $(if $(CHECKPOINT),--checkpoint-path $(CHECKPOINT),)

validate-replay:
	uv run python ops/scripts/validate_replay.py $(BUNDLE)
