# shield

AEGIS PPO blue-team cyber defense pipeline.

## What is included
- `CyberDefenseEnv` Gymnasium environment with 28 canonical contract nodes.
- Scripted red-team scenarios (5) and blue action space (`MultiDiscrete([6, 28])`).
- PPO model/training scaffolding (RLlib + Torch) with staged run options.
- Eval harness with baselines and KPI computation.
- Replay packaging pipeline that emits:
  - `manifest.json`
  - `events.jsonl`
  - `topology_snapshots.json`
  - `metrics.json`
- Replay validator: `ops/scripts/validate_replay.py`.
- FastAPI service endpoints for replay listing/bundle access and websocket streams.

## Quick start

```bash
uv sync --extra dev
uv run pytest -q
```

## Train / Eval

```bash
# Smoke stage
make train-smoke RUNS_ROOT=runs RUN_ID=run001

# Full training stage
make train RUNS_ROOT=runs RUN_ID=run001

# Eval + replay packaging (set seeds_per_scenario for tighter CI bounds)
make eval RUNS_ROOT=runs RUN_ID=run001 CHECKPOINT=/path/to/checkpoint EVAL_SEEDS_PER_SCENARIO=4 EVAL_SEED_START=2001

# Eval all runs that have train/final checkpoints
make eval-all RUNS_ROOT=runs USE_DOCKER=false EVAL_SEEDS_PER_SCENARIO=2 EVAL_SEED_START=1001

# Docker-backed eval-all (useful on Apple Silicon or when local train deps are unavailable)
make eval-all RUNS_ROOT=runs USE_DOCKER=true DOCKER_PLATFORM=linux/amd64 SHM_SIZE=2g EVAL_SEEDS_PER_SCENARIO=2 EVAL_SEED_START=1001

# Validate a replay bundle
make validate-replay BUNDLE=runs/run001/replays/replay_hero_01

# Queue N full runs with eval + CI/variance tracking + conservative auto-tuning
make autopilot RUNS_ROOT=runs NUM_RUNS=10 START_RUN_ID=run006

# Queue a fixed sweep matrix (per-run overrides / timesteps / eval seeds)
make autopilot RUNS_ROOT=runs START_RUN_ID=run006 NUM_RUNS=12 SWEEP_SPEC=ops/sweeps/fast_recovery_v1.json
```

Autopilot artifacts are written to `runs/autopilot/session_<timestamp>.json` and `.csv`.

## Demo episodes

Deterministic episode runs for UI/demo playback are stored under:

- `runs/ep001`
- `runs/ep002`
- `runs/ep003`
- `runs/ep004`
- `runs/ep005`

Build and verify:

```bash
make demo-episodes RUNS_ROOT=runs FRONTEND_ROOT=../frontend
make verify-demo-episodes RUNS_ROOT=runs FRONTEND_ROOT=../frontend
```

Provenance for each episode is documented in `DEMO_PROVENANCE.md`.

## Live proof (judge demo)

Use these commands live to prove reproducibility and integrity:

```bash
# 1) Rebuild deterministic episode artifacts
make demo-episodes RUNS_ROOT=runs FRONTEND_ROOT=../frontend

# 2) Verify checksums + replay contract + frontend parity hashes
make verify-demo-episodes RUNS_ROOT=runs FRONTEND_ROOT=../frontend

# 3) Run one fresh backend eval command on a trained run
make eval RUNS_ROOT=runs RUN_ID=run010 CHECKPOINT=runs/run010/train/final EVAL_SEEDS_PER_SCENARIO=1 EVAL_SEED_START=3001
```

## Docker-first run

```bash
docker build -f docker/Dockerfile.train -t shield-train .

docker run --rm \
  -e MODE=train \
  -e STAGE=smoke \
  -e RUNS_ROOT=runs \
  -e RUN_ID=run001 \
  -e TRAIN_CONFIG_OVERRIDES_PATH=runs/run001/train/config_overrides.json \
  -e CUDA_VISIBLE_DEVICES=5,6,7 \
  -v "$PWD:/app" \
  shield-train
```

## API

```bash
uv run uvicorn backend.app.main:app --host 0.0.0.0 --port 8000
```

Endpoints:
- `GET /health`
- `GET /replay/list`
- `GET /replay/{replay_id}/bundle`
- `WS /stream/replay/{replay_id}`
- `WS /stream/live/{session_id}`

## Notes
- Contract-first node registry is fixed to 28 nodes.
- Hero/backup replay aliases are generated as:
  - `replay_hero_01`
  - `replay_alt_02`
  - `replay_alt_03`
  - `replay_alt_04`
- Run folder convention is `runs/runNNN`.
