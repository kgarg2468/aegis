# Frontend (Sandbox Live + Replay)

## Local Setup (with Cluster Tunnel)

Create `frontend/.env.local`:

```bash
NEXT_PUBLIC_API_BASE_URL=http://127.0.0.1:8000
```

Keep your SSH tunnel active so local `8000` maps to cluster backend `8000`.

```bash
bash ops/scripts/cluster_tunnel.expect
```

## Run Frontend

```bash
npm run dev
```

If `3000` is occupied by tunnel forwarding, run:

```bash
npm run dev -- --port 3001
```

Then open:

- `http://localhost:3000` (default), or
- `http://localhost:3001` (when running on alternate port)

## Contract-First Replay Dev

Generate a deterministic mock replay bundle that follows the AEGIS integration contract:

```bash
npm run generate:mock-replay
```

This writes:

- `frontend/mock/replays/mock_replay_01/manifest.json`
- `frontend/mock/replays/mock_replay_01/events.jsonl`
- `frontend/mock/replays/mock_replay_01/topology_snapshots.json`
- `frontend/mock/replays/mock_replay_01/metrics.json`

Run contract-focused frontend tests:

```bash
npm run test:contract
```

## Demo-Focused Frontend Notes

- The UI is demo-oriented and scenario-first:
  - `Command Grid` for live topology + tactical log
  - `Episode Vault` for scenario/campaign selection
  - `Mission Brief` for plain-language explanation
- Replay ingestion is contract-ready through the canonical `WSMessage` event model (`type + data` routing).
- The adapter layer in `src/lib/replayAdapter.ts` unifies mock + local replay loading and keeps metadata needed for future model runs.
- To plug in future model-backed streams, keep emitting contract-compatible events and map backend run metadata to:
  - `scenario_id`
  - `attack_profile_id`
  - `campaign_stage`
  - `source` (`model_stream`)

## Sandbox Live Contract Notes

- Frontend sends `POST /sandbox/runs` with `{ "episode_spec": ... }`.
- `GET /sandbox/catalog` now includes readiness:
  - `execution_mode`
  - `live_run_enabled`
  - `live_block_reason`
- `Run Live` is disabled when `live_run_enabled=false`.
- Backend validation errors from FastAPI `detail` are shown directly in the UI.
