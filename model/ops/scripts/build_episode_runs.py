from __future__ import annotations

import argparse
import hashlib
import json
import shlex
import shutil
import subprocess
import sys
import tempfile
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any


BASE_TIME = datetime(2026, 4, 5, 18, 0, tzinfo=UTC)


@dataclass(frozen=True)
class ModeSpec:
    mode: str
    seed: int
    steps: int
    replay_id: str


@dataclass(frozen=True)
class EpisodeSpec:
    run_id: str
    episode_id: str
    episode_label: str
    scenario_id: str
    scenario_display_name: str
    source_run_id: str
    source_replay_id: str
    source_eval_file: str
    source_composite_score: float
    rationale: str
    enterprise: ModeSpec
    no_blue: ModeSpec


EPISODES: list[EpisodeSpec] = [
    EpisodeSpec(
        run_id="ep001",
        episode_id="episode_001",
        episode_label="Episode 001",
        scenario_id="eduroam_credential_harvest",
        scenario_display_name="Eduroam Credential Harvesting -> SIS Breach",
        source_run_id="run010",
        source_replay_id="replay_ep_002",
        source_eval_file="runs/run010/eval/eval_cli_1775412914.json",
        source_composite_score=0.5326838302542528,
        rationale="Highest-scoring Eduroam replay across the curated source set with stable breach progression.",
        enterprise=ModeSpec(mode="enterprise", seed=1323, steps=240, replay_id="replay_enterprise_01"),
        no_blue=ModeSpec(mode="no_blue", seed=1101, steps=200, replay_id="replay_no_blue_01"),
    ),
    EpisodeSpec(
        run_id="ep002",
        episode_id="episode_002",
        episode_label="Episode 002",
        scenario_id="faculty_spear_phish",
        scenario_display_name="Faculty Spear Phish -> Research Data Theft",
        source_run_id="run005",
        source_replay_id="replay_ep_003",
        source_eval_file="runs/run005/eval/eval_cli_1775409184.json",
        source_composite_score=0.6540423414457613,
        rationale="Strong faculty lateral-movement trajectory with representative exfiltration pressure.",
        enterprise=ModeSpec(mode="enterprise", seed=1424, steps=240, replay_id="replay_enterprise_01"),
        no_blue=ModeSpec(mode="no_blue", seed=1202, steps=200, replay_id="replay_no_blue_01"),
    ),
    EpisodeSpec(
        run_id="ep003",
        episode_id="episode_003",
        episode_label="Episode 003",
        scenario_id="iot_botnet_exhaustion",
        scenario_display_name="IoT Botnet -> Resource Exhaustion",
        source_run_id="run007",
        source_replay_id="replay_ep_005",
        source_eval_file="runs/run007/eval/eval_cli_1775411311.json",
        source_composite_score=0.7377629488401653,
        rationale="Best IoT containment run in the curated set with clear availability-recovery dynamics.",
        enterprise=ModeSpec(mode="enterprise", seed=1525, steps=240, replay_id="replay_enterprise_01"),
        no_blue=ModeSpec(mode="no_blue", seed=1303, steps=200, replay_id="replay_no_blue_01"),
    ),
    EpisodeSpec(
        run_id="ep004",
        episode_id="episode_004",
        episode_label="Episode 004",
        scenario_id="insider_ad_backdoor",
        scenario_display_name="Insider Threat -> AD Backdoor",
        source_run_id="run007",
        source_replay_id="replay_ep_008",
        source_eval_file="runs/run007/eval/eval_cli_1775411311.json",
        source_composite_score=0.4896551979122142,
        rationale="Representative stealth-heavy insider sequence with delayed backdoor reveal patterns.",
        enterprise=ModeSpec(mode="enterprise", seed=1626, steps=240, replay_id="replay_enterprise_01"),
        no_blue=ModeSpec(mode="no_blue", seed=1404, steps=200, replay_id="replay_no_blue_01"),
    ),
    EpisodeSpec(
        run_id="ep005",
        episode_id="episode_005",
        episode_label="Episode 005",
        scenario_id="print_ransomware_propagation",
        scenario_display_name="Print Server Ransomware Propagation",
        source_run_id="run008",
        source_replay_id="replay_ep_009",
        source_eval_file="runs/run008/eval/eval_cli_1775411770.json",
        source_composite_score=0.6330503666434727,
        rationale="Highest ransomware scenario score in the curated source set with realistic propagation tempo.",
        enterprise=ModeSpec(mode="enterprise", seed=1727, steps=240, replay_id="replay_enterprise_01"),
        no_blue=ModeSpec(mode="no_blue", seed=1505, steps=200, replay_id="replay_no_blue_01"),
    ),
]


def _write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")


def _write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    body = "\n".join(json.dumps(row, separators=(",", ":")) for row in rows)
    path.write_text((body + "\n") if body else "", encoding="utf-8")


def _load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as handle:
        while True:
            chunk = handle.read(1024 * 1024)
            if not chunk:
                break
            h.update(chunk)
    return h.hexdigest()


def _deterministic_time(episode_number: int, offset_minutes: int) -> str:
    ts = BASE_TIME + timedelta(minutes=(episode_number - 1) * 17 + offset_minutes)
    return ts.isoformat().replace("+00:00", "Z")


def _run(
    cmd: list[str],
    cwd: Path,
    *,
    check: bool = True,
) -> subprocess.CompletedProcess[str]:
    completed = subprocess.run(cmd, cwd=cwd, text=True, capture_output=True, check=False)
    if check and completed.returncode != 0:
        rendered = shlex.join(cmd)
        raise RuntimeError(
            f"Command failed ({completed.returncode}): {rendered}\nstdout:\n{completed.stdout}\nstderr:\n{completed.stderr}"
        )
    return completed


def _render_command(cmd: list[str]) -> str:
    return shlex.join(cmd)


def _collect_log_entry(cmd: list[str], completed: subprocess.CompletedProcess[str]) -> str:
    lines = [f"$ {_render_command(cmd)}"]
    stdout = completed.stdout.strip()
    stderr = completed.stderr.strip()
    if stdout:
        lines.append(stdout)
    if stderr:
        lines.append("[stderr]")
        lines.append(stderr)
    return "\n".join(lines)


def _frontend_export_command(
    frontend_root: Path,
    *,
    scenario_id: str,
    scenario_display_name: str,
    replay_id: str,
    out_dir: Path,
    mode: str,
    seed: int,
    steps: int,
) -> list[str]:
    tsx_bin = frontend_root / "node_modules" / ".bin" / "tsx"
    launcher = str(tsx_bin) if tsx_bin.exists() else "tsx"
    return [
        launcher,
        "scripts/export_episode_replay.ts",
        "--scenario-id",
        scenario_id,
        "--scenario-display-name",
        scenario_display_name,
        "--replay-id",
        replay_id,
        "--out-dir",
        str(out_dir),
        "--mode",
        mode,
        "--seed",
        str(seed),
        "--steps",
        str(steps),
    ]


def _validate_replay_command(model_root: Path, bundle_dir: Path) -> list[str]:
    return [sys.executable, str(model_root / "ops" / "scripts" / "validate_replay.py"), str(bundle_dir)]


def _parse_summary(stdout: str) -> dict[str, Any]:
    lines = [line.strip() for line in stdout.splitlines() if line.strip()]
    if not lines:
        raise RuntimeError("Replay export produced no stdout; cannot parse summary JSON")
    try:
        return json.loads(lines[-1])
    except json.JSONDecodeError as exc:
        raise RuntimeError(f"Failed to parse export summary JSON from output: {lines[-1]}") from exc


def _episode_number(run_id: str) -> int:
    return int(run_id[2:])


def _mode_specs(episode: EpisodeSpec, enabled_modes: set[str]) -> list[ModeSpec]:
    specs: list[ModeSpec] = []
    if "enterprise" in enabled_modes:
        specs.append(episode.enterprise)
    if "no_blue" in enabled_modes:
        specs.append(episode.no_blue)
    return specs


def _build_train_metrics(metrics_file: Path) -> list[dict[str, Any]]:
    metrics = json.loads(metrics_file.read_text(encoding="utf-8"))
    sampled: list[dict[str, Any]] = []
    for item in metrics:
        step = int(item.get("step", 0))
        if step % 20 != 0 and step != int(metrics[-1].get("step", 0)):
            continue
        sampled.append(
            {
                "step": step,
                "service_availability": float(item.get("service_availability", 1.0)),
                "attack_pressure": float(item.get("attack_pressure", 0.0)),
                "containment_pressure": float(item.get("containment_pressure", 0.0)),
                "blue_reward_cumulative": float(item.get("blue_reward_cumulative", 0.0)),
                "red_score_cumulative": float(item.get("red_score_cumulative", 0.0)),
                "open_incidents": int(item.get("open_incidents", 0)),
            }
        )
    return sampled


def _comparison_block(modes_payload: dict[str, dict[str, Any]]) -> dict[str, Any]:
    enterprise = modes_payload.get("enterprise")
    no_blue = modes_payload.get("no_blue")
    if not enterprise or not no_blue:
        return {}

    enterprise_kpis = enterprise["kpis"]
    no_blue_kpis = no_blue["kpis"]
    return {
        "enterprise_vs_no_blue": {
            "damage_score_delta": round(
                float(no_blue_kpis.get("damage_score", 0.0)) - float(enterprise_kpis.get("damage_score", 0.0)),
                6,
            ),
            "containment_time_steps_delta": int(no_blue_kpis.get("containment_time_steps", 0))
            - int(enterprise_kpis.get("containment_time_steps", 0)),
            "hvts_compromised_delta": int(no_blue_kpis.get("hvts_compromised", 0))
            - int(enterprise_kpis.get("hvts_compromised", 0)),
            "service_availability_delta": round(
                float(enterprise_kpis.get("final_service_availability", 0.0))
                - float(no_blue_kpis.get("final_service_availability", 0.0)),
                6,
            ),
            "blue_reward_delta": round(
                float(enterprise_kpis.get("blue_reward_total", 0.0)) - float(no_blue_kpis.get("blue_reward_total", 0.0)),
                6,
            ),
            "data_exfiltration_prevented": bool(no_blue_kpis.get("data_exfiltrated", False))
            and not bool(enterprise_kpis.get("data_exfiltrated", False)),
        }
    }


def _write_checksums(run_dir: Path, target_paths: list[Path]) -> None:
    relative_sorted = sorted({path.relative_to(run_dir).as_posix(): path for path in target_paths}.items())
    lines = [f"{_sha256(path)}  {rel}" for rel, path in relative_sorted]
    (run_dir / "checksums.sha256").write_text("\n".join(lines) + "\n", encoding="utf-8")


def _verify_checksums(run_dir: Path) -> list[str]:
    errors: list[str] = []
    checksums_path = run_dir / "checksums.sha256"
    if not checksums_path.exists():
        return [f"Missing checksums file: {checksums_path}"]

    for raw in checksums_path.read_text(encoding="utf-8").splitlines():
        line = raw.strip()
        if not line:
            continue
        if "  " not in line:
            errors.append(f"Malformed checksums line: {raw}")
            continue
        digest, rel = line.split("  ", 1)
        target = run_dir / rel
        if not target.exists():
            errors.append(f"Missing checksummed file: {target}")
            continue
        actual = _sha256(target)
        if actual != digest:
            errors.append(f"Checksum mismatch for {target}: expected {digest}, got {actual}")
    return errors


def _verify_eval_schema(eval_path: Path, episode: EpisodeSpec, enabled_modes: set[str]) -> list[str]:
    errors: list[str] = []
    payload = _load_json(eval_path)

    required_top = {
        "schema_version",
        "run_id",
        "episode_id",
        "episode_label",
        "scenario_id",
        "scenario_display_name",
        "generated_at",
        "modes",
        "comparison",
    }
    missing = sorted(required_top - set(payload.keys()))
    if missing:
        errors.append(f"{eval_path}: missing keys {missing}")

    forbidden = {"suite", "eval_results", "episodes", "selection_manifest", "kpis", "gates"}
    present_forbidden = sorted(forbidden & set(payload.keys()))
    if present_forbidden:
        errors.append(f"{eval_path}: forbidden legacy keys present {present_forbidden}")

    if payload.get("run_id") != episode.run_id:
        errors.append(f"{eval_path}: run_id mismatch")
    if payload.get("episode_id") != episode.episode_id:
        errors.append(f"{eval_path}: episode_id mismatch")
    if payload.get("scenario_id") != episode.scenario_id:
        errors.append(f"{eval_path}: scenario_id mismatch")

    modes = payload.get("modes", {})
    mode_keys = set(modes.keys()) if isinstance(modes, dict) else set()
    if mode_keys != enabled_modes:
        errors.append(f"{eval_path}: expected mode keys {sorted(enabled_modes)}, got {sorted(mode_keys)}")

    for mode in sorted(enabled_modes):
        block = modes.get(mode, {}) if isinstance(modes, dict) else {}
        for key in ["replay_id", "seed", "steps", "outcome", "kpis", "bundle_path"]:
            if key not in block:
                errors.append(f"{eval_path}: mode={mode} missing {key}")

    if enabled_modes == {"enterprise", "no_blue"}:
        if "enterprise_vs_no_blue" not in payload.get("comparison", {}):
            errors.append(f"{eval_path}: missing comparison.enterprise_vs_no_blue")

    return errors


def _parity_check(
    frontend_root: Path,
    run_dir: Path,
    episode: EpisodeSpec,
    mode_spec: ModeSpec,
) -> str | None:
    with tempfile.TemporaryDirectory(prefix=f"{episode.run_id}_{mode_spec.mode}_") as tmp:
        out_dir = Path(tmp) / mode_spec.replay_id
        cmd = _frontend_export_command(
            frontend_root,
            scenario_id=episode.scenario_id,
            scenario_display_name=episode.scenario_display_name,
            replay_id=mode_spec.replay_id,
            out_dir=out_dir,
            mode=mode_spec.mode,
            seed=mode_spec.seed,
            steps=mode_spec.steps,
        )
        completed = _run(cmd, cwd=frontend_root)
        _parse_summary(completed.stdout)

        committed_events = run_dir / "replays" / mode_spec.replay_id / "events.jsonl"
        generated_events = out_dir / "events.jsonl"
        if not committed_events.exists():
            return f"Missing committed events file: {committed_events}"
        if _sha256(committed_events) != _sha256(generated_events):
            return (
                f"Frontend parity mismatch for {episode.run_id}:{mode_spec.mode} "
                f"({committed_events} vs regenerated {generated_events})"
            )
    return None


def _build_episode(
    model_root: Path,
    frontend_root: Path,
    runs_root: Path,
    episode: EpisodeSpec,
    *,
    clean: bool,
    enabled_modes: set[str],
) -> None:
    run_dir = runs_root / episode.run_id
    if clean and run_dir.exists():
        shutil.rmtree(run_dir)

    (run_dir / "train").mkdir(parents=True, exist_ok=True)
    (run_dir / "eval").mkdir(parents=True, exist_ok=True)
    (run_dir / "replays").mkdir(parents=True, exist_ok=True)
    (run_dir / "logs").mkdir(parents=True, exist_ok=True)

    train_log_entries: list[str] = []
    eval_log_entries: list[str] = []
    mode_payloads: dict[str, dict[str, Any]] = {}

    for mode_spec in _mode_specs(episode, enabled_modes):
        bundle_dir = run_dir / "replays" / mode_spec.replay_id
        export_cmd = _frontend_export_command(
            frontend_root,
            scenario_id=episode.scenario_id,
            scenario_display_name=episode.scenario_display_name,
            replay_id=mode_spec.replay_id,
            out_dir=bundle_dir,
            mode=mode_spec.mode,
            seed=mode_spec.seed,
            steps=mode_spec.steps,
        )
        exported = _run(export_cmd, cwd=frontend_root)
        train_log_entries.append(_collect_log_entry(export_cmd, exported))

        _parse_summary(exported.stdout)
        manifest = _load_json(bundle_dir / "manifest.json")

        validate_cmd = _validate_replay_command(model_root, bundle_dir)
        validated = _run(validate_cmd, cwd=model_root)
        eval_log_entries.append(_collect_log_entry(validate_cmd, validated))

        mode_payloads[mode_spec.mode] = {
            "mode": mode_spec.mode,
            "replay_id": mode_spec.replay_id,
            "seed": mode_spec.seed,
            "steps": mode_spec.steps,
            "outcome": manifest.get("outcome"),
            "kpis": manifest.get("kpis", {}),
            "bundle_path": f"runs/{episode.run_id}/replays/{mode_spec.replay_id}",
        }

    selection_manifest = {
        "hero": episode.enterprise.replay_id,
        "backups": [episode.no_blue.replay_id],
        "mapping": {
            episode.enterprise.replay_id: {
                "mode": "enterprise",
                "scenario_id": episode.scenario_id,
                "seed": episode.enterprise.seed,
                "steps": episode.enterprise.steps,
            },
            episode.no_blue.replay_id: {
                "mode": "no_blue",
                "scenario_id": episode.scenario_id,
                "seed": episode.no_blue.seed,
                "steps": episode.no_blue.steps,
            },
        },
    }
    _write_json(run_dir / "replays" / "selection_manifest.json", selection_manifest)

    episode_number = _episode_number(episode.run_id)
    train_metadata = {
        "schema_version": "episode_train_metadata_v1",
        "run_id": episode.run_id,
        "episode_id": episode.episode_id,
        "episode_label": episode.episode_label,
        "scenario_id": episode.scenario_id,
        "scenario_display_name": episode.scenario_display_name,
        "stage": "full",
        "generated_at": _deterministic_time(episode_number, 1),
        "generator": "ops/scripts/build_episode_runs.py",
        "cluster_profile": "enterprise_demo_cluster",
        "modes": mode_payloads,
    }
    _write_json(run_dir / "train" / "train_metadata.json", train_metadata)

    config_overrides = {
        "max_steps": episode.enterprise.steps,
        "scenario_id": episode.scenario_id,
        "episode_id": episode.episode_id,
        "mode_overrides": {
            "enterprise": {
                "seed": episode.enterprise.seed,
                "max_steps": episode.enterprise.steps,
                "blue_team_enabled": True,
            },
            "no_blue": {
                "seed": episode.no_blue.seed,
                "max_steps": episode.no_blue.steps,
                "blue_team_enabled": False,
            },
        },
    }
    _write_json(run_dir / "train" / "config_overrides.json", config_overrides)

    enterprise_metrics_path = run_dir / "replays" / episode.enterprise.replay_id / "metrics.json"
    train_metrics = _build_train_metrics(enterprise_metrics_path)
    _write_jsonl(run_dir / "train" / "train_metrics.jsonl", train_metrics)

    checkpoint_stub = {
        "schema_version": "checkpoint_metadata_v1",
        "checkpoint_type": "metadata_only",
        "note": "Deterministic episode run; binary checkpoint omitted by design.",
        "source": "frontend deterministic replay export",
    }
    _write_json(run_dir / "train" / "best" / "checkpoint_metadata.json", checkpoint_stub)
    _write_json(run_dir / "train" / "final" / "checkpoint_metadata.json", checkpoint_stub)

    eval_payload = {
        "schema_version": "episode_eval_v1",
        "run_id": episode.run_id,
        "episode_id": episode.episode_id,
        "episode_label": episode.episode_label,
        "scenario_id": episode.scenario_id,
        "scenario_display_name": episode.scenario_display_name,
        "generated_at": _deterministic_time(episode_number, 2),
        "modes": mode_payloads,
        "comparison": _comparison_block(mode_payloads),
    }
    _write_json(run_dir / "eval" / f"eval_cli_{episode.run_id}.json", eval_payload)

    run_manifest = {
        "schema_version": "episode_run_manifest_v1",
        "run_id": episode.run_id,
        "episode_id": episode.episode_id,
        "episode_label": episode.episode_label,
        "scenario_id": episode.scenario_id,
        "scenario_display_name": episode.scenario_display_name,
        "generated_at": _deterministic_time(episode_number, 0),
        "generator": "ops/scripts/build_episode_runs.py",
        "files": {
            "train_metadata": f"runs/{episode.run_id}/train/train_metadata.json",
            "eval": f"runs/{episode.run_id}/eval/eval_cli_{episode.run_id}.json",
            "selection_manifest": f"runs/{episode.run_id}/replays/selection_manifest.json",
            "hero_replay_manifest": f"runs/{episode.run_id}/replays/{episode.enterprise.replay_id}/manifest.json",
            "no_blue_replay_manifest": f"runs/{episode.run_id}/replays/{episode.no_blue.replay_id}/manifest.json",
        },
    }
    _write_json(run_dir / "manifest.json", run_manifest)

    (run_dir / "logs" / "autopilot_train.log").write_text("\n\n".join(train_log_entries) + "\n", encoding="utf-8")
    (run_dir / "logs" / "autopilot_eval.log").write_text("\n\n".join(eval_log_entries) + "\n", encoding="utf-8")

    checksum_targets = [
        run_dir / "manifest.json",
        run_dir / "train" / "config_overrides.json",
        run_dir / "train" / "train_metadata.json",
        run_dir / "train" / "train_metrics.jsonl",
        run_dir / "train" / "best" / "checkpoint_metadata.json",
        run_dir / "train" / "final" / "checkpoint_metadata.json",
        run_dir / "eval" / f"eval_cli_{episode.run_id}.json",
        run_dir / "replays" / "selection_manifest.json",
        run_dir / "logs" / "autopilot_train.log",
        run_dir / "logs" / "autopilot_eval.log",
    ]
    for mode_spec in [episode.enterprise, episode.no_blue]:
        bundle_root = run_dir / "replays" / mode_spec.replay_id
        checksum_targets.extend(
            [
                bundle_root / "manifest.json",
                bundle_root / "events.jsonl",
                bundle_root / "topology_snapshots.json",
                bundle_root / "metrics.json",
            ]
        )

    _write_checksums(run_dir, checksum_targets)


def _verify_episode(
    model_root: Path,
    frontend_root: Path,
    runs_root: Path,
    episode: EpisodeSpec,
    enabled_modes: set[str],
) -> list[str]:
    errors: list[str] = []
    run_dir = runs_root / episode.run_id
    if not run_dir.exists():
        return [f"Missing run directory: {run_dir}"]

    errors.extend(_verify_checksums(run_dir))

    for mode_spec in _mode_specs(episode, enabled_modes):
        bundle_dir = run_dir / "replays" / mode_spec.replay_id
        validate_cmd = _validate_replay_command(model_root, bundle_dir)
        completed = _run(validate_cmd, cwd=model_root, check=False)
        if completed.returncode != 0:
            errors.append(
                f"Replay validator failed for {bundle_dir}: rc={completed.returncode}\n"
                f"stdout:\n{completed.stdout}\nstderr:\n{completed.stderr}"
            )

        parity_error = _parity_check(frontend_root, run_dir, episode, mode_spec)
        if parity_error:
            errors.append(parity_error)

    eval_file = run_dir / "eval" / f"eval_cli_{episode.run_id}.json"
    errors.extend(_verify_eval_schema(eval_file, episode, enabled_modes))
    return errors


def _parse_modes(raw: str) -> set[str]:
    requested = {item.strip() for item in raw.split(",") if item.strip()}
    valid = {"enterprise", "no_blue"}
    invalid = sorted(requested - valid)
    if invalid:
        raise ValueError(f"Unsupported mode(s): {invalid}. Valid modes: {sorted(valid)}")
    if not requested:
        raise ValueError("At least one mode is required")
    return requested


def _select_episodes(raw: str) -> list[EpisodeSpec]:
    if raw.strip().lower() == "all":
        return EPISODES
    wanted = {item.strip() for item in raw.split(",") if item.strip()}
    by_id = {episode.run_id: episode for episode in EPISODES}
    missing = sorted(wanted - set(by_id.keys()))
    if missing:
        raise ValueError(f"Unknown episode ids: {missing}")
    selected = [by_id[run_id] for run_id in sorted(wanted)]
    return selected


def main() -> int:
    parser = argparse.ArgumentParser(description="Build deterministic cluster-style episode runs (ep001..ep005)")
    parser.add_argument("--runs-root", default="runs", help="Runs root directory (default: runs)")
    parser.add_argument(
        "--frontend-root",
        default=None,
        help="Frontend root directory (default: ../frontend from model root)",
    )
    parser.add_argument("--episodes", default="all", help="Comma-separated episode ids (e.g. ep001,ep003) or 'all'")
    parser.add_argument("--modes", default="enterprise,no_blue", help="Comma-separated modes: enterprise,no_blue")
    parser.add_argument("--verify-only", action="store_true", help="Only verify existing episode runs")
    parser.add_argument("--clean", action="store_true", default=True, help="Clean episode run directory before rebuild")
    parser.add_argument("--no-clean", action="store_false", dest="clean", help="Do not clean before rebuild")
    args = parser.parse_args()

    model_root = Path(__file__).resolve().parents[2]
    runs_root = (model_root / args.runs_root).resolve()
    frontend_root = Path(args.frontend_root).resolve() if args.frontend_root else (model_root.parent / "frontend").resolve()

    if not frontend_root.exists():
        raise FileNotFoundError(f"Frontend root does not exist: {frontend_root}")

    selected_episodes = _select_episodes(args.episodes)
    enabled_modes = _parse_modes(args.modes)

    if not args.verify_only:
        runs_root.mkdir(parents=True, exist_ok=True)
        for episode in selected_episodes:
            _build_episode(
                model_root=model_root,
                frontend_root=frontend_root,
                runs_root=runs_root,
                episode=episode,
                clean=args.clean,
                enabled_modes=enabled_modes,
            )

    all_errors: list[str] = []
    for episode in selected_episodes:
        all_errors.extend(
            _verify_episode(
                model_root=model_root,
                frontend_root=frontend_root,
                runs_root=runs_root,
                episode=episode,
                enabled_modes=enabled_modes,
            )
        )

    if all_errors:
        print(f"Verification failed with {len(all_errors)} issue(s):")
        for item in all_errors:
            print(f" - {item}")
        return 1

    print(
        json.dumps(
            {
                "status": "ok",
                "episodes": [episode.run_id for episode in selected_episodes],
                "modes": sorted(enabled_modes),
                "runs_root": str(runs_root),
                "frontend_root": str(frontend_root),
                "verify_only": args.verify_only,
            },
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
