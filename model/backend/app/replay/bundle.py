from __future__ import annotations

from pathlib import Path


def select_hero_and_backups(episode_reports: list[dict]) -> dict[str, dict]:
    if len(episode_reports) < 4:
        raise ValueError("Need at least 4 episodes to select hero + 3 backups")

    ranked = sorted(episode_reports, key=lambda item: item.get("composite_score", 0.0), reverse=True)
    hero = ranked[0]

    selected = [hero]
    used_scenarios = {hero.get("scenario_id")}

    for candidate in ranked[1:]:
        if len(selected) >= 4:
            break
        scenario = candidate.get("scenario_id")
        if scenario not in used_scenarios:
            selected.append(candidate)
            used_scenarios.add(scenario)

    for candidate in ranked[1:]:
        if len(selected) >= 4:
            break
        if candidate not in selected:
            selected.append(candidate)

    return {
        "replay_hero_01": selected[0],
        "replay_alt_02": selected[1],
        "replay_alt_03": selected[2],
        "replay_alt_04": selected[3],
    }


def write_selection_manifest(run_replay_dir: str | Path, selection: dict[str, dict]) -> Path:
    run_replay_dir = Path(run_replay_dir)
    manifest_path = run_replay_dir / "selection_manifest.json"

    payload = {
        "hero": "replay_hero_01",
        "backups": ["replay_alt_02", "replay_alt_03", "replay_alt_04"],
        "mapping": {
            label: {
                "source_replay_id": report.get("replay_id"),
                "scenario_id": report.get("scenario_id"),
                "seed": report.get("seed"),
                "composite_score": report.get("composite_score"),
            }
            for label, report in selection.items()
        },
    }
    manifest_path.write_text(__import__("json").dumps(payload, indent=2), encoding="utf-8")
    return manifest_path
