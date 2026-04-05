import json
from pathlib import Path

from ops.scripts.validate_replay import validate_replay


def _write_json(path: Path, payload: dict) -> None:
    path.write_text(json.dumps(payload, indent=2))


def _write_jsonl(path: Path, rows: list[dict]) -> None:
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row) + "\n")


def test_replay_validator_accepts_minimal_valid_bundle(tmp_path: Path):
    bundle = tmp_path / "replay_hero_01"
    bundle.mkdir(parents=True)

    manifest = {
        "replay_id": "replay_hero_01",
        "scenario_id": "faculty_phish",
        "seed": 1003,
        "checkpoint_id": "ckpt_blue_main_0001",
        "duration_steps": 1,
        "files": {
            "events": "events.jsonl",
            "topology": "topology_snapshots.json",
            "metrics": "metrics.json",
        },
    }

    events = [
        {
            "type": "topology_init",
            "data": {
                "nodes": [],
                "edges": [],
                "zones": [],
                "scenario_id": "faculty_phish",
                "total_steps": 1,
                "seed": 1003,
            },
        },
        {
            "type": "action_event",
            "data": {
                "event_id": "evt_000001",
                "ts_ms": 1712412345678,
                "step": 1,
                "actor": "BLUE",
                "action_type": "monitor_host",
                "source_host": "auth_server",
                "target_host": "auth_server",
                "target_service": "ldap",
                "outcome": "success",
                "mitre_tactic": "Defense Evasion",
                "confidence": 0.91,
                "description": "Monitor auth server for suspicious authentication behavior",
                "severity": "low",
                "risk_score": 0.2,
            },
        },
        {
            "type": "state_delta",
            "data": {
                "ts_ms": 1712412345680,
                "step": 1,
                "node_changes": [
                    {
                        "node_id": "auth_server",
                        "visual_state": "monitored",
                        "overlay": "monitored",
                        "compromise_level": 0,
                        "defense_state": "monitored",
                    }
                ],
                "edge_changes": [],
            },
        },
        {
            "type": "explainability",
            "data": {
                "ts_ms": 1712412345695,
                "step": 1,
                "action": "monitor_host",
                "target_host": "auth_server",
                "confidence": 0.82,
                "reason_features": [{"name": "critical_asset_risk", "value": 0.88}],
                "expected_effect": "increase detection coverage",
            },
        },
        {
            "type": "metrics_tick",
            "data": {
                "step": 1,
                "attack_pressure": 0.1,
                "containment_pressure": 0.3,
                "service_availability": 1.0,
                "open_incidents": 0,
                "contained_incidents": 0,
                "red_actions_total": 0,
                "blue_actions_total": 1,
                "blue_reward_cumulative": 0.1,
                "red_score_cumulative": 0.0,
                "detection_latency_mean": 0.0,
                "hot_targets": [],
                "alert_classification": {},
            },
        },
        {
            "type": "episode_end",
            "data": {
                "outcome": "contained",
                "final_step": 1,
                "summary": {
                    "total_red_actions": 0,
                    "total_blue_actions": 1,
                    "nodes_compromised": 0,
                    "nodes_isolated": 0,
                    "nodes_patched": 0,
                    "hvts_compromised": 0,
                    "data_exfiltrated": False,
                    "final_service_availability": 1.0,
                    "blue_reward_total": 0.1,
                },
            },
        },
    ]

    _write_json(bundle / "manifest.json", manifest)
    _write_jsonl(bundle / "events.jsonl", events)
    _write_json(bundle / "topology_snapshots.json", {"initial": {"nodes": [], "edges": [], "zones": []}, "snapshots": {}})
    _write_json(bundle / "metrics.json", [{"step": 1, "attack_pressure": 0.1}])

    errors = validate_replay(str(bundle))
    assert errors == []


def test_replay_validator_rejects_wrong_first_event(tmp_path: Path):
    bundle = tmp_path / "bad_replay"
    bundle.mkdir(parents=True)

    _write_json(
        bundle / "manifest.json",
        {
            "replay_id": "bad_replay",
            "scenario_id": "faculty_phish",
            "seed": 1003,
            "checkpoint_id": "ckpt",
            "duration_steps": 1,
            "files": {"events": "events.jsonl", "topology": "topology_snapshots.json", "metrics": "metrics.json"},
        },
    )

    _write_jsonl(bundle / "events.jsonl", [{"type": "action_event", "data": {"step": 1}}])
    _write_json(bundle / "topology_snapshots.json", {"initial": {"nodes": [], "edges": [], "zones": []}, "snapshots": {}})
    _write_json(bundle / "metrics.json", [])

    errors = validate_replay(str(bundle))
    assert any("First event must be topology_init" in err for err in errors)
