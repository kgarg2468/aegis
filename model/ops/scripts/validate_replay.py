from __future__ import annotations

import argparse
import json
from pathlib import Path

from backend.app.env.registry import (
    NODE_REGISTRY_IDS,
    VALID_ACTION_TYPES,
    VALID_ACTORS,
    VALID_EDGE_STATES,
    VALID_NODE_STATES,
    VALID_OUTCOMES,
    VALID_SEVERITIES,
)


def load_json(path: str | Path) -> dict:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def load_jsonl(path: str | Path) -> list[dict]:
    rows: list[dict] = []
    for raw in Path(path).read_text(encoding="utf-8").splitlines():
        if raw.strip():
            rows.append(json.loads(raw))
    return rows


def validate_replay(bundle_dir: str) -> list[str]:
    errors: list[str] = []
    bundle = Path(bundle_dir)

    manifest_path = bundle / "manifest.json"
    events_path = bundle / "events.jsonl"
    topology_path = bundle / "topology_snapshots.json"
    metrics_path = bundle / "metrics.json"

    for required in [manifest_path, events_path, topology_path, metrics_path]:
        if not required.exists():
            errors.append(f"Missing required file: {required.name}")

    if errors:
        return errors

    manifest = load_json(manifest_path)
    for field in ["replay_id", "scenario_id", "seed", "checkpoint_id", "duration_steps", "files"]:
        if field not in manifest:
            errors.append(f"manifest missing field: {field}")

    events = load_jsonl(events_path)
    if not events:
        return errors + ["events.jsonl must contain at least one event"]

    if events[0].get("type") != "topology_init":
        errors.append("First event must be topology_init")

    valid_node_ids = set(NODE_REGISTRY_IDS)

    for event in events:
        evt_type = event.get("type")
        data = event.get("data", {})

        if evt_type == "topology_add_node":
            node = data.get("node", {})
            node_id = node.get("id")
            if not node_id or not str(node_id).startswith("decoy_"):
                errors.append("topology_add_node node.id must start with decoy_")
            if node_id:
                valid_node_ids.add(node_id)

        if evt_type == "action_event":
            event_id = data.get("event_id", "unknown")
            for field in ["source_host", "target_host"]:
                value = data.get(field)
                if value and value not in valid_node_ids:
                    errors.append(f"Event {event_id}: unknown {field}={value}")

            actor = data.get("actor")
            if actor not in VALID_ACTORS:
                errors.append(f"Invalid actor: {actor}")

            action_type = data.get("action_type")
            if action_type not in VALID_ACTION_TYPES:
                errors.append(f"Invalid action_type: {action_type}")

            outcome = data.get("outcome")
            if outcome not in VALID_OUTCOMES:
                errors.append(f"Invalid outcome: {outcome}")

            severity = data.get("severity")
            if severity and severity not in VALID_SEVERITIES:
                errors.append(f"Invalid severity: {severity}")

            description = data.get("description", "")
            if len(description) > 120:
                errors.append(f"Event {event_id}: description exceeds 120 chars")

        if evt_type == "state_delta":
            step = data.get("step")
            for node_change in data.get("node_changes", []):
                node_id = node_change.get("node_id")
                if node_id not in valid_node_ids:
                    errors.append(f"StateDelta step {step}: unknown node_id={node_id}")

                visual_state = node_change.get("visual_state")
                if visual_state not in VALID_NODE_STATES:
                    errors.append(f"StateDelta step {step}: invalid visual_state={visual_state}")

                overlay = node_change.get("overlay")
                if overlay not in {None, "monitored"}:
                    errors.append(f"StateDelta step {step}: invalid overlay={overlay}")

            for edge_change in data.get("edge_changes", []):
                edge_id = edge_change.get("edge_id", "")
                if "->" not in edge_id:
                    errors.append(f"StateDelta step {step}: invalid edge_id={edge_id}")
                    continue
                src, tgt = edge_id.split("->", 1)
                if src not in valid_node_ids or tgt not in valid_node_ids:
                    errors.append(f"StateDelta step {step}: invalid edge_id={edge_id}")

                visual_state = edge_change.get("visual_state")
                if visual_state not in VALID_EDGE_STATES:
                    errors.append(f"StateDelta step {step}: invalid edge visual_state={visual_state}")

                direction = edge_change.get("direction")
                if direction not in {"forward", "reverse"}:
                    errors.append(f"StateDelta step {step}: invalid direction={direction}")

    steps = [event.get("data", {}).get("step") for event in events if "step" in event.get("data", {})]
    last_step = -1
    for step in steps:
        if step is None:
            continue
        if int(step) < int(last_step):
            errors.append(f"Step ordering violation: step {step} after step {last_step}")
        last_step = int(step)

    step_counts: dict[int, dict[str, int]] = {}
    for event in events:
        data = event.get("data", {})
        step = data.get("step")
        if step is None:
            continue
        step = int(step)
        step_counts.setdefault(step, {"state_delta": 0, "metrics_tick": 0, "explainability": 0, "blue_action": 0})
        if event["type"] in {"state_delta", "metrics_tick", "explainability"}:
            step_counts[step][event["type"]] += 1
        if event["type"] == "action_event" and event.get("data", {}).get("actor") == "BLUE":
            step_counts[step]["blue_action"] += 1

    for step, counts in step_counts.items():
        if counts["state_delta"] != 1:
            errors.append(f"Step {step}: expected 1 state_delta, got {counts['state_delta']}")
        if counts["metrics_tick"] != 1:
            errors.append(f"Step {step}: expected 1 metrics_tick, got {counts['metrics_tick']}")
        if counts["explainability"] != 1:
            errors.append(f"Step {step}: expected 1 explainability, got {counts['explainability']}")
        if counts["blue_action"] != 1:
            errors.append(f"Step {step}: expected 1 BLUE action_event, got {counts['blue_action']}")

    if events[-1].get("type") != "episode_end":
        errors.append("Last event must be episode_end")

    return errors


def _main() -> int:
    parser = argparse.ArgumentParser(description="Validate replay bundle against integration contract")
    parser.add_argument("bundle_dir", type=str)
    args = parser.parse_args()

    errors = validate_replay(args.bundle_dir)
    if errors:
        print(f"{len(errors)} errors found")
        for error in errors:
            print(f" - {error}")
        return 1

    print("0 errors found")
    return 0


if __name__ == "__main__":
    raise SystemExit(_main())
