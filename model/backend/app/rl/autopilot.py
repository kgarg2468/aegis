from __future__ import annotations

import csv
import json
import math
import shlex
import subprocess
import time
from datetime import UTC, datetime
from pathlib import Path
from statistics import mean, stdev
from typing import Any

import typer

from backend.app.replay.run_manager import next_run_id
from backend.app.rl.config import PPO_CONFIG, merge_config
from ops.scripts.validate_replay import validate_replay

app = typer.Typer(add_completion=False)

KPI_THRESHOLDS: dict[str, float] = {
    "damage_reduction_vs_no_defense": 0.25,
    "damage_reduction_vs_rule_based": 0.15,
    "detection_latency_improvement_vs_rule_based": 0.20,
}
KPI_KEYS: tuple[str, ...] = tuple(KPI_THRESHOLDS.keys())
STD_ALERT_THRESHOLD = 0.05

# Two-sided 95% t critical values by degrees of freedom.
_T_CRITICAL_95: dict[int, float] = {
    1: 12.706,
    2: 4.303,
    3: 3.182,
    4: 2.776,
    5: 2.571,
    6: 2.447,
    7: 2.365,
    8: 2.306,
    9: 2.262,
    10: 2.228,
    11: 2.201,
    12: 2.179,
    13: 2.160,
    14: 2.145,
    15: 2.131,
    16: 2.120,
    17: 2.110,
    18: 2.101,
    19: 2.093,
    20: 2.086,
    21: 2.080,
    22: 2.074,
    23: 2.069,
    24: 2.064,
    25: 2.060,
    26: 2.056,
    27: 2.052,
    28: 2.048,
    29: 2.045,
    30: 2.042,
}

DEFAULT_AUTOTUNE_OVERRIDES: dict[str, float | int] = {
    "lr": float(PPO_CONFIG["lr"]),
    "entropy_coeff": float(PPO_CONFIG["entropy_coeff"]),
    "clip_param": float(PPO_CONFIG["clip_param"]),
    "train_batch_size": int(PPO_CONFIG["train_batch_size"]),
}


def _t_critical_95(df: int) -> float:
    if df <= 0:
        return float("nan")
    if df in _T_CRITICAL_95:
        return _T_CRITICAL_95[df]
    # For larger df, normal approximation is close enough for routing decisions.
    return 1.96


def _metric_ci95(values: list[float]) -> tuple[float | None, float | None, float | None]:
    if len(values) < 2:
        return None, None, None

    sigma = float(stdev(values))
    t_value = _t_critical_95(len(values) - 1)
    margin = t_value * (sigma / math.sqrt(len(values)))
    mu = float(mean(values))
    return mu - margin, mu + margin, margin


def compute_kpi_statistics(run_kpis: list[dict[str, float]]) -> dict:
    metrics: dict[str, dict] = {}
    variance_high = False
    all_ci_gates_pass = True
    all_mean_gates_pass = True

    for key, threshold in KPI_THRESHOLDS.items():
        values = [float(item.get(key, 0.0)) for item in run_kpis]
        mu = float(mean(values)) if values else 0.0
        sigma = float(stdev(values)) if len(values) > 1 else 0.0
        ci_low, ci_high, ci_margin = _metric_ci95(values)
        ci_gate_pass = ci_low is not None and ci_low >= threshold
        mean_gate_pass = mu >= threshold

        metric = {
            "threshold": threshold,
            "mean": mu,
            "std": sigma,
            "ci95_lower": ci_low,
            "ci95_upper": ci_high,
            "ci95_margin": ci_margin,
            "passes_ci_gate": ci_gate_pass,
            "passes_mean_gate": mean_gate_pass,
            "std_gt_5pct": sigma > STD_ALERT_THRESHOLD,
        }
        metrics[key] = metric

        variance_high = variance_high or metric["std_gt_5pct"]
        all_ci_gates_pass = all_ci_gates_pass and ci_gate_pass
        all_mean_gates_pass = all_mean_gates_pass and mean_gate_pass

    if not run_kpis:
        all_ci_gates_pass = False
        all_mean_gates_pass = False

    return {
        "num_runs": len(run_kpis),
        "metrics": metrics,
        "variance_high": variance_high,
        "all_ci_gates_pass": all_ci_gates_pass,
        "all_mean_gates_pass": all_mean_gates_pass,
    }


def _clamp(value: float, low: float, high: float) -> float:
    return max(low, min(high, value))


def _round_to_multiple(value: int, multiple: int) -> int:
    return int(round(value / multiple) * multiple)


def _ci_failed(stats: dict, kpi_key: str) -> bool:
    metric = stats.get("metrics", {}).get(kpi_key, {})
    ci95_lower = metric.get("ci95_lower")
    threshold = KPI_THRESHOLDS[kpi_key]
    return ci95_lower is None or float(ci95_lower) < threshold


def suggest_next_overrides(
    stats: dict,
    previous_overrides: dict[str, float | int] | None,
) -> tuple[dict[str, float | int], list[str]]:
    base = dict(DEFAULT_AUTOTUNE_OVERRIDES)
    if previous_overrides:
        base.update(previous_overrides)

    next_overrides: dict[str, float | int] = dict(base)
    reasons: list[str] = []

    if int(stats.get("num_runs", 0)) < 2:
        return next_overrides, ["Not enough runs for CI-based tuning; collecting more data."]

    if bool(stats.get("variance_high", False)):
        current_lr = float(next_overrides["lr"])
        current_batch_size = int(next_overrides["train_batch_size"])
        next_overrides["lr"] = _clamp(current_lr * 0.8, 1e-4, 5e-4)
        proposed_batch = _round_to_multiple(int(current_batch_size * 1.25), 256)
        next_overrides["train_batch_size"] = max(4096, min(16384, proposed_batch))
        reasons.append("High variance observed across runs; decreased lr and increased train_batch_size.")

    damage_fail = _ci_failed(stats, "damage_reduction_vs_no_defense") or _ci_failed(
        stats, "damage_reduction_vs_rule_based"
    )
    latency_fail = _ci_failed(stats, "detection_latency_improvement_vs_rule_based")

    if damage_fail and not latency_fail:
        entropy = float(next_overrides["entropy_coeff"])
        clip = float(next_overrides["clip_param"])
        next_overrides["entropy_coeff"] = _clamp(entropy * 1.2, 0.003, 0.03)
        next_overrides["clip_param"] = _clamp(clip + 0.02, 0.15, 0.30)
        reasons.append("Damage-reduction CI below target; increased exploration and policy update headroom.")
    elif latency_fail:
        entropy = float(next_overrides["entropy_coeff"])
        next_overrides["entropy_coeff"] = _clamp(entropy * 0.9, 0.003, 0.03)
        reasons.append("Detection-latency CI below target; reduced exploration for faster containment.")

    if not reasons:
        reasons.append("No tuning changes: CI and variance signals are stable.")

    return next_overrides, reasons


def _project_root() -> Path:
    return Path(__file__).resolve().parents[3]


def _runs_root_path(project_root: Path, runs_root: str) -> Path:
    path = Path(runs_root)
    if not path.is_absolute():
        path = project_root / path
    return path


def _path_for_container(path: Path, project_root: Path) -> str:
    try:
        relative = path.resolve().relative_to(project_root.resolve())
        return f"/app/{relative.as_posix()}"
    except ValueError:
        return str(path)


def _host_path(candidate: str, project_root: Path) -> Path:
    p = Path(candidate)
    if p.exists():
        return p
    if candidate.startswith("/app/"):
        mapped = project_root / candidate.removeprefix("/app/")
        if mapped.exists():
            return mapped
        return mapped
    if not p.is_absolute():
        mapped = project_root / p
        if mapped.exists():
            return mapped
        return mapped
    return p


def _resolve_path(path_str: str, project_root: Path) -> Path:
    path = Path(path_str)
    if path.is_absolute():
        return path
    return project_root / path


def _load_sweep_spec(path: Path) -> list[dict[str, Any]]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError("Sweep spec must be a JSON object")

    runs = payload.get("runs")
    if not isinstance(runs, list) or not runs:
        raise ValueError("Sweep spec must include a non-empty 'runs' list")

    def _coerce_int(value: Any, *, field: str, minimum: int) -> int:
        if isinstance(value, bool):
            raise ValueError(f"Sweep field '{field}' must be an integer >= {minimum}")
        try:
            out = int(value)
        except (TypeError, ValueError) as exc:
            raise ValueError(f"Sweep field '{field}' must be an integer >= {minimum}") from exc
        if out < minimum:
            raise ValueError(f"Sweep field '{field}' must be >= {minimum}")
        return out

    normalized: list[dict[str, Any]] = []
    for idx, run in enumerate(runs):
        if not isinstance(run, dict):
            raise ValueError(f"Sweep run at index {idx} must be an object")
        overrides = run.get("overrides", {})
        if not isinstance(overrides, dict):
            raise ValueError(f"Sweep run at index {idx} has invalid overrides (expected object)")
        train_timesteps = run.get("train_timesteps")
        if train_timesteps is not None:
            train_timesteps = _coerce_int(train_timesteps, field=f"runs[{idx}].train_timesteps", minimum=1)
        seeds_per_scenario = run.get("seeds_per_scenario")
        if seeds_per_scenario is not None:
            seeds_per_scenario = _coerce_int(
                seeds_per_scenario,
                field=f"runs[{idx}].seeds_per_scenario",
                minimum=1,
            )
        seed_start = run.get("seed_start")
        if seed_start is not None:
            seed_start = _coerce_int(seed_start, field=f"runs[{idx}].seed_start", minimum=0)
        normalized.append(
            {
                "label": str(run.get("label", f"sweep_{idx + 1:03d}")),
                "overrides": overrides,
                "train_timesteps": train_timesteps,
                "seeds_per_scenario": seeds_per_scenario,
                "seed_start": seed_start,
            }
        )
    return normalized


def _command_to_string(cmd: list[str]) -> str:
    return " ".join(shlex.quote(part) for part in cmd)


def _run_with_log(cmd: list[str], cwd: Path, log_path: Path) -> None:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("a", encoding="utf-8") as log_file:
        log_file.write(f"\n[{datetime.now(UTC).isoformat()}] $ {_command_to_string(cmd)}\n")
        log_file.flush()

        process = subprocess.Popen(
            cmd,
            cwd=str(cwd),
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
        )
        assert process.stdout is not None
        for line in process.stdout:
            log_file.write(line)
            log_file.flush()
        return_code = process.wait()
        if return_code != 0:
            raise RuntimeError(f"Command failed with exit code {return_code}: {_command_to_string(cmd)}")


def _build_train_command(
    *,
    use_docker: bool,
    docker_image: str,
    project_root: Path,
    runs_root: str,
    run_id: str,
    stage: str,
    cuda_visible_devices: str,
    config_overrides_path: Path,
    train_timesteps: int | None,
) -> list[str]:
    if use_docker:
        container_overrides = _path_for_container(config_overrides_path, project_root)
        cmd = [
            "docker",
            "run",
            "--rm",
            "--shm-size=64g",
            "--gpus",
            "all",
            "-e",
            "MODE=train",
            "-e",
            f"STAGE={stage}",
            "-e",
            f"RUNS_ROOT={runs_root}",
            "-e",
            f"RUN_ID={run_id}",
            "-e",
            f"CUDA_VISIBLE_DEVICES={cuda_visible_devices}",
            "-e",
            f"TRAIN_CONFIG_OVERRIDES_PATH={container_overrides}",
        ]
        if train_timesteps is not None:
            cmd.extend(["-e", f"TRAIN_TIMESTEPS={int(train_timesteps)}"])
        cmd.extend(
            [
                "-v",
                f"{project_root}:/app",
                docker_image,
            ]
        )
        return cmd

    cmd = [
        "uv",
        "run",
        "--extra",
        "train",
        "python",
        "-m",
        "backend.app.rl.train",
        "--stage",
        stage,
        "--runs-root",
        runs_root,
        "--run-id",
        run_id,
        "--config-overrides-path",
        str(config_overrides_path),
    ]
    if train_timesteps is not None:
        cmd.extend(["--train-timesteps", str(int(train_timesteps))])
    return cmd


def _build_eval_command(
    *,
    use_docker: bool,
    docker_image: str,
    project_root: Path,
    runs_root: str,
    run_id: str,
    checkpoint_path: Path,
    cuda_visible_devices: str,
    seeds_per_scenario: int,
    seed_start: int,
) -> list[str]:
    if use_docker:
        container_checkpoint = _path_for_container(checkpoint_path, project_root)
        return [
            "docker",
            "run",
            "--rm",
            "--shm-size=64g",
            "--gpus",
            "all",
            "-e",
            "MODE=eval",
            "-e",
            f"RUNS_ROOT={runs_root}",
            "-e",
            f"RUN_ID={run_id}",
            "-e",
            f"CHECKPOINT_PATH={container_checkpoint}",
            "-e",
            f"CUDA_VISIBLE_DEVICES={cuda_visible_devices}",
            "-e",
            f"EVAL_SEEDS_PER_SCENARIO={seeds_per_scenario}",
            "-e",
            f"EVAL_SEED_START={seed_start}",
            "-v",
            f"{project_root}:/app",
            docker_image,
        ]

    return [
        "uv",
        "run",
        "--extra",
        "train",
        "python",
        "-m",
        "backend.app.rl.eval",
        "--runs-root",
        runs_root,
        "--run-id",
        run_id,
        "--checkpoint-path",
        str(checkpoint_path),
        "--seeds-per-scenario",
        str(seeds_per_scenario),
        "--seed-start",
        str(seed_start),
    ]


def _latest_eval_file(eval_dir: Path, before: set[Path]) -> Path:
    after = {p for p in eval_dir.glob("*.json") if p.is_file()}
    new_files = sorted(after - before, key=lambda p: p.stat().st_mtime)
    if new_files:
        return new_files[-1]

    if not after:
        raise RuntimeError(f"No eval artifacts found in {eval_dir}")
    return max(after, key=lambda p: p.stat().st_mtime)


def _select_checkpoint_path(train_metadata_path: Path, project_root: Path) -> Path:
    metadata = json.loads(train_metadata_path.read_text(encoding="utf-8"))
    for field in ["final_checkpoint", "best_checkpoint"]:
        candidate = metadata.get(field)
        if not candidate:
            continue
        host_path = _host_path(str(candidate), project_root)
        if host_path.exists():
            return host_path
    raise RuntimeError(f"No usable checkpoint path found in {train_metadata_path}")


def _validate_alias_bundles(replays_dir: Path) -> dict[str, list[str]]:
    results: dict[str, list[str]] = {}
    for replay_id in ["replay_hero_01", "replay_alt_02", "replay_alt_03", "replay_alt_04"]:
        bundle_dir = replays_dir / replay_id
        if not bundle_dir.exists():
            results[replay_id] = [f"Missing bundle: {bundle_dir}"]
            continue
        results[replay_id] = validate_replay(str(bundle_dir))
    return results


def _append_csv_row(csv_path: Path, row: dict) -> None:
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    write_header = not csv_path.exists()
    fields = [
        "timestamp_utc",
        "run_id",
        "damage_reduction_vs_no_defense",
        "damage_reduction_vs_rule_based",
        "detection_latency_improvement_vs_rule_based",
        "ci_pass",
        "variance_high",
    ]

    with csv_path.open("a", newline="", encoding="utf-8") as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=fields)
        if write_header:
            writer.writeheader()
        writer.writerow({key: row.get(key) for key in fields})


def _write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


@app.command()
def main(
    num_runs: int = typer.Option(10, min=1, help="Number of sequential runs to execute"),
    runs_root: str = typer.Option("runs", help="Runs root directory"),
    stage: str = typer.Option("full", help="Training stage (typically full)"),
    use_docker: bool = typer.Option(True, help="Run train/eval through Docker image"),
    docker_image: str = typer.Option("shield-train", help="Docker image name when --use-docker"),
    cuda_visible_devices: str = typer.Option("5,6,7", help="CUDA_VISIBLE_DEVICES value"),
    start_run_id: str | None = typer.Option(None, help="Optional first run id (example: run006)"),
    auto_tune: bool = typer.Option(True, help="Apply conservative config tweaks between runs"),
    train_timesteps: int | None = typer.Option(
        None,
        min=1,
        help="Optional stop.timesteps_total override for each run unless sweep run overrides it",
    ),
    seeds_per_scenario: int = typer.Option(2, min=1, help="Eval seeds per scenario"),
    seed_start: int = typer.Option(1001, min=0, help="Eval seed start"),
    sweep_spec_path: str | None = typer.Option(
        None,
        help="Optional JSON sweep spec with per-run overrides/train_timesteps/eval seed options",
    ),
) -> None:
    project_root = _project_root()
    runs_root_path = _runs_root_path(project_root, runs_root)
    runs_root_path.mkdir(parents=True, exist_ok=True)

    autopilot_dir = runs_root_path / "autopilot"
    session_id = datetime.now(UTC).strftime("%Y%m%d_%H%M%S")
    state_path = autopilot_dir / f"session_{session_id}.json"
    csv_path = autopilot_dir / f"session_{session_id}.csv"

    if start_run_id is None:
        start_run_id = next_run_id(runs_root_path)
    if not (start_run_id.startswith("run") and start_run_id[3:].isdigit()):
        raise ValueError(f"Invalid start run id: {start_run_id}")
    first_run_num = int(start_run_id[3:])

    resolved_sweep_spec_path: Path | None = None
    sweep_runs: list[dict[str, Any]] = []
    if sweep_spec_path:
        resolved_sweep_spec_path = _resolve_path(sweep_spec_path, project_root)
        if not resolved_sweep_spec_path.exists():
            raise FileNotFoundError(f"Sweep spec path not found: {resolved_sweep_spec_path}")
        sweep_runs = _load_sweep_spec(resolved_sweep_spec_path)
        if num_runs > len(sweep_runs):
            raise ValueError(
                f"num_runs={num_runs} exceeds runs in sweep spec ({len(sweep_runs)}): {resolved_sweep_spec_path}"
            )

    run_records: list[dict] = []
    current_overrides: dict[str, float | int] = dict(DEFAULT_AUTOTUNE_OVERRIDES) if auto_tune else {}

    for offset in range(num_runs):
        run_num = first_run_num + offset
        run_id = f"run{run_num:03d}"
        sweep_entry: dict[str, Any] = sweep_runs[offset] if sweep_runs else {}
        sweep_label = str(sweep_entry.get("label", run_id))
        sweep_overrides = sweep_entry.get("overrides", {}) if sweep_entry else {}

        run_overrides: dict[str, Any]
        if auto_tune:
            run_overrides = merge_config(dict(current_overrides), sweep_overrides)
        else:
            run_overrides = merge_config({}, sweep_overrides)

        run_train_timesteps = sweep_entry.get("train_timesteps")
        if run_train_timesteps is None:
            run_train_timesteps = train_timesteps
        run_seeds_per_scenario = int(sweep_entry.get("seeds_per_scenario", seeds_per_scenario))
        run_seed_start = int(sweep_entry.get("seed_start", seed_start))

        run_dir = runs_root_path / run_id
        if run_dir.exists() and any(run_dir.iterdir()):
            raise RuntimeError(f"Run directory already exists and is not empty: {run_dir}")
        train_dir = run_dir / "train"
        eval_dir = run_dir / "eval"
        logs_dir = run_dir / "logs"
        replays_dir = run_dir / "replays"
        for path in [train_dir, eval_dir, logs_dir, replays_dir]:
            path.mkdir(parents=True, exist_ok=True)

        overrides_path = train_dir / "config_overrides.json"
        _write_json(overrides_path, run_overrides)

        print(
            json.dumps(
                {
                    "event": "run_start",
                    "run_id": run_id,
                    "label": sweep_label,
                    "overrides": run_overrides,
                    "train_timesteps": run_train_timesteps,
                    "seeds_per_scenario": run_seeds_per_scenario,
                    "seed_start": run_seed_start,
                },
                indent=2,
            )
        )
        run_started = time.time()

        train_cmd = _build_train_command(
            use_docker=use_docker,
            docker_image=docker_image,
            project_root=project_root,
            runs_root=runs_root,
            run_id=run_id,
            stage=stage,
            cuda_visible_devices=cuda_visible_devices,
            config_overrides_path=overrides_path,
            train_timesteps=run_train_timesteps,
        )
        _run_with_log(train_cmd, cwd=project_root, log_path=logs_dir / "autopilot_train.log")

        metadata_path = train_dir / "train_metadata.json"
        if not metadata_path.exists():
            raise RuntimeError(f"Missing train metadata for {run_id}: {metadata_path}")
        checkpoint_path = _select_checkpoint_path(metadata_path, project_root)

        eval_before = {p for p in eval_dir.glob("*.json") if p.is_file()}
        eval_cmd = _build_eval_command(
            use_docker=use_docker,
            docker_image=docker_image,
            project_root=project_root,
            runs_root=runs_root,
            run_id=run_id,
            checkpoint_path=checkpoint_path,
            cuda_visible_devices=cuda_visible_devices,
            seeds_per_scenario=run_seeds_per_scenario,
            seed_start=run_seed_start,
        )
        _run_with_log(eval_cmd, cwd=project_root, log_path=logs_dir / "autopilot_eval.log")
        eval_payload_path = _latest_eval_file(eval_dir, eval_before)
        eval_payload = json.loads(eval_payload_path.read_text(encoding="utf-8"))

        kpis = {key: float(eval_payload.get("kpis", {}).get(key, 0.0)) for key in KPI_KEYS}
        aggregate = compute_kpi_statistics([record["kpis"] for record in run_records] + [kpis])
        replay_validation = _validate_alias_bundles(replays_dir)

        tuning_reasons: list[str] = []
        if auto_tune:
            tune_seed = {
                key: run_overrides.get(key, current_overrides.get(key, value))
                for key, value in DEFAULT_AUTOTUNE_OVERRIDES.items()
            }
            current_overrides, tuning_reasons = suggest_next_overrides(aggregate, tune_seed)

        row = {
            "timestamp_utc": datetime.now(UTC).isoformat(),
            "run_id": run_id,
            "damage_reduction_vs_no_defense": kpis["damage_reduction_vs_no_defense"],
            "damage_reduction_vs_rule_based": kpis["damage_reduction_vs_rule_based"],
            "detection_latency_improvement_vs_rule_based": kpis["detection_latency_improvement_vs_rule_based"],
            "ci_pass": aggregate["all_ci_gates_pass"],
            "variance_high": aggregate["variance_high"],
        }
        _append_csv_row(csv_path, row)

        run_ended = time.time()
        run_record = {
            "run_id": run_id,
            "label": sweep_label,
            "started_at_unix": run_started,
            "ended_at_unix": run_ended,
            "duration_sec": round(run_ended - run_started, 2),
            "config_overrides_path": str(overrides_path),
            "applied_overrides": run_overrides,
            "sweep_overrides": sweep_overrides,
            "train_timesteps": run_train_timesteps,
            "seeds_per_scenario": run_seeds_per_scenario,
            "seed_start": run_seed_start,
            "train_metadata_path": str(metadata_path),
            "checkpoint_path": str(checkpoint_path),
            "eval_payload_path": str(eval_payload_path),
            "kpis": kpis,
            "kpi_gates": eval_payload.get("gates", {}),
            "aggregate_after_run": aggregate,
            "replay_validation": replay_validation,
            "next_overrides": current_overrides if auto_tune else {},
            "tuning_reasons": tuning_reasons,
        }
        run_records.append(run_record)

        state_payload = {
            "session_id": session_id,
            "runs_root": str(runs_root_path),
            "num_runs_requested": num_runs,
            "auto_tune": auto_tune,
            "train_timesteps_default": train_timesteps,
            "eval_seeds_per_scenario_default": seeds_per_scenario,
            "eval_seed_start_default": seed_start,
            "sweep_spec_path": str(resolved_sweep_spec_path) if resolved_sweep_spec_path else None,
            "completed_runs": len(run_records),
            "current_overrides": current_overrides,
            "aggregate": aggregate,
            "run_records": run_records,
            "kpi_thresholds": KPI_THRESHOLDS,
            "std_alert_threshold": STD_ALERT_THRESHOLD,
        }
        _write_json(state_path, state_payload)

        print(
            json.dumps(
                {
                    "event": "run_complete",
                    "run_id": run_id,
                    "kpis": kpis,
                    "ci_pass": aggregate["all_ci_gates_pass"],
                    "variance_high": aggregate["variance_high"],
                    "state_path": str(state_path),
                },
                indent=2,
            )
        )

    final_payload = json.loads(state_path.read_text(encoding="utf-8"))
    print(
        json.dumps(
            {
                "event": "autopilot_complete",
                "session_id": session_id,
                "completed_runs": final_payload["completed_runs"],
                "state_path": str(state_path),
                "csv_path": str(csv_path),
                "aggregate": final_payload["aggregate"],
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    app()
