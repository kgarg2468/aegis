from __future__ import annotations

import asyncio
import json
from pathlib import Path

from fastapi import FastAPI, HTTPException, WebSocket, WebSocketDisconnect

from backend.app.env.cyber_defense_env import CyberDefenseEnv
from backend.app.rl.eval import RuleBasedBaseline

app = FastAPI(title="Shield Backend", version="0.1.0")


def _runs_root() -> Path:
    return Path("runs")


def _iter_run_dirs() -> list[Path]:
    root = _runs_root()
    if not root.exists():
        return []
    run_dirs = [
        p
        for p in root.iterdir()
        if p.is_dir() and (p.name.startswith("run") or p.name.startswith("ep"))
    ]
    run_dirs.sort(key=lambda p: p.name)
    return run_dirs


def _find_replay_bundle(replay_id: str) -> Path | None:
    for run_dir in reversed(_iter_run_dirs()):
        candidate = run_dir / "replays" / replay_id
        if candidate.exists() and candidate.is_dir():
            return candidate
    return None


@app.get("/health")
def health() -> dict:
    return {"status": "ok"}


@app.get("/replay/list")
def replay_list() -> dict:
    replays: list[dict] = []
    for run_dir in _iter_run_dirs():
        replay_root = run_dir / "replays"
        if not replay_root.exists():
            continue
        for replay_dir in sorted(p for p in replay_root.iterdir() if p.is_dir()):
            manifest_path = replay_dir / "manifest.json"
            manifest = {}
            if manifest_path.exists():
                manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            replays.append(
                {
                    "run_id": run_dir.name,
                    "replay_id": replay_dir.name,
                    "scenario_id": manifest.get("scenario_id"),
                    "outcome": manifest.get("outcome"),
                }
            )
    return {"replays": replays}


@app.get("/replay/{replay_id}/bundle")
def replay_bundle(replay_id: str) -> dict:
    bundle = _find_replay_bundle(replay_id)
    if bundle is None:
        raise HTTPException(status_code=404, detail=f"Replay not found: {replay_id}")

    manifest = json.loads((bundle / "manifest.json").read_text(encoding="utf-8"))
    topology = json.loads((bundle / "topology_snapshots.json").read_text(encoding="utf-8"))
    metrics = json.loads((bundle / "metrics.json").read_text(encoding="utf-8"))

    return {
        "manifest": manifest,
        "topology": topology,
        "metrics": metrics,
    }


@app.websocket("/stream/replay/{replay_id}")
async def stream_replay(websocket: WebSocket, replay_id: str):
    await websocket.accept()
    bundle = _find_replay_bundle(replay_id)
    if bundle is None:
        await websocket.send_json({"type": "error", "message": f"Replay not found: {replay_id}"})
        await websocket.close(code=4404)
        return

    events_path = bundle / "events.jsonl"
    try:
        for line in events_path.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            await websocket.send_json(json.loads(line))
            await asyncio.sleep(0.02)
        await websocket.send_json({"type": "terminal", "data": {"completed": True, "replay_id": replay_id}})
    except WebSocketDisconnect:
        return


@app.websocket("/stream/live/{session_id}")
async def stream_live(websocket: WebSocket, session_id: str):
    await websocket.accept()
    env = CyberDefenseEnv({"max_steps": 120, "max_nodes": 28})
    baseline = RuleBasedBaseline()

    obs, info = env.reset(seed=42)
    await websocket.send_json({"type": "topology_init", "data": info["topology"]})

    try:
        terminated = False
        truncated = False
        while not (terminated or truncated):
            action = baseline.select_action(obs)
            obs, _, terminated, truncated, step_info = env.step(action)
            for event in step_info["events"]:
                await websocket.send_json(event)
            await asyncio.sleep(0.05)

        await websocket.send_json(
            {
                "type": "terminal",
                "data": {
                    "completed": True,
                    "session_id": session_id,
                    "steps": env.step_count,
                },
            }
        )
    except WebSocketDisconnect:
        return
