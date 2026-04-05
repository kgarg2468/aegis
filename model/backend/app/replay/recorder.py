from __future__ import annotations

import json
import time
from pathlib import Path

from backend.app.env.network_state import NetworkState
from backend.app.env.registry import SCENARIO_DISPLAY_NAMES


class ReplayRecorder:
    def __init__(self, replay_id: str, scenario_id: str, seed: int, checkpoint_id: str):
        self.replay_id = replay_id
        self.scenario_id = scenario_id
        self.seed = seed
        self.checkpoint_id = checkpoint_id

        self.events: list[dict] = []
        self.metrics_timeseries: list[dict] = []
        self.snapshots_by_step: dict[int, dict] = {}

        self.manifest = {
            "replay_id": replay_id,
            "scenario_id": scenario_id,
            "scenario_display_name": SCENARIO_DISPLAY_NAMES.get(scenario_id, scenario_id),
            "seed": seed,
            "checkpoint_id": checkpoint_id,
            "duration_steps": 0,
            "total_events": 0,
            "outcome": "timeout",
            "kpis": {
                "damage_score": 0.0,
                "containment_time_steps": 0,
                "hvts_compromised": 0,
                "data_exfiltrated": False,
            },
        }

        self.topology_initial: dict | None = None

    def record_topology_init(self, topology_init_data: dict, state: NetworkState) -> None:
        self.topology_initial = {
            "nodes": topology_init_data["nodes"],
            "edges": topology_init_data["edges"],
            "zones": topology_init_data["zones"],
        }
        self.events.append({"type": "topology_init", "data": topology_init_data})
        self.snapshots_by_step[0] = state.state_snapshot()

    def record_step(self, step: int, ordered_events: list[dict], state: NetworkState, metrics_tick: dict) -> None:
        for event in ordered_events:
            if "data" in event and isinstance(event["data"], dict):
                if event["data"].get("step") is None:
                    event["data"]["step"] = step
                if event["data"].get("ts_ms") in {None, 0}:
                    event["data"]["ts_ms"] = int(time.time() * 1000)
            self.events.append(event)

        self.metrics_timeseries.append(
            {
                **metrics_tick,
                "explainability": [
                    event["data"]
                    for event in ordered_events
                    if event.get("type") == "explainability"
                ],
            }
        )

        if step % 10 == 0:
            self.snapshots_by_step[step] = state.state_snapshot()

        self.manifest["duration_steps"] = step

    def finalize(self, outcome: str, summary: dict, kpis: dict) -> None:
        self.manifest["outcome"] = outcome
        self.manifest["kpis"] = {
            "damage_score": round(float(kpis.get("damage_score", 0.0)), 4),
            "containment_time_steps": int(kpis.get("containment_time_steps", 0)),
            "hvts_compromised": int(summary.get("hvts_compromised", 0)),
            "data_exfiltrated": bool(summary.get("data_exfiltrated", False)),
        }

        end_event = {
            "type": "episode_end",
            "data": {
                "outcome": outcome,
                "final_step": self.manifest["duration_steps"],
                "summary": summary,
            },
        }
        self.events.append(end_event)
        self.manifest["total_events"] = len(self.events)

    def save(self, replay_root: str | Path) -> Path:
        replay_root_path = Path(replay_root)
        bundle_dir = replay_root_path / self.replay_id
        bundle_dir.mkdir(parents=True, exist_ok=True)

        events_path = bundle_dir / "events.jsonl"
        with events_path.open("w", encoding="utf-8") as f:
            for event in self.events:
                f.write(json.dumps(event) + "\n")

        topology_payload = {
            "initial": self.topology_initial or {"nodes": [], "edges": [], "zones": []},
            "snapshots": {str(step): payload for step, payload in sorted(self.snapshots_by_step.items()) if step != 0},
        }
        (bundle_dir / "topology_snapshots.json").write_text(json.dumps(topology_payload, indent=2), encoding="utf-8")
        (bundle_dir / "metrics.json").write_text(json.dumps(self.metrics_timeseries, indent=2), encoding="utf-8")

        manifest = {
            **self.manifest,
            "files": {
                "events": "events.jsonl",
                "topology": "topology_snapshots.json",
                "metrics": "metrics.json",
            },
        }
        (bundle_dir / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")

        return bundle_dir
