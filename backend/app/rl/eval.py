from __future__ import annotations

import json
import random
import shutil
import time
from pathlib import Path
from statistics import mean
from typing import Callable

import numpy as np
import typer

from backend.app.env.cyber_defense_env import CyberDefenseEnv
from backend.app.replay.bundle import select_hero_and_backups, write_selection_manifest
from backend.app.replay.recorder import ReplayRecorder
from backend.app.replay.run_manager import create_run_dirs
from ops.scripts.validate_replay import validate_replay

app = typer.Typer(add_completion=False)

EVAL_SUITE = {
    "suite_id": "heldout_suite_v1",
    "scenarios": [
        {"scenario": "eduroam_harvest", "seeds": [1001, 1002]},
        {"scenario": "faculty_phish", "seeds": [1003, 1004]},
        {"scenario": "iot_botnet", "seeds": [1005, 1006]},
        {"scenario": "insider_threat", "seeds": [1007, 1008]},
        {"scenario": "ransomware_cascade", "seeds": [1009, 1010]},
    ],
}


class NoDefenseBaseline:
    """Baseline A: only monitor random node."""

    def select_action(self, obs: dict) -> list[int]:
        return [0, random.randint(0, 27)]


class RuleBasedBaseline:
    """Baseline B: static threshold-based defender."""

    def select_action(self, obs: dict) -> list[int]:
        node_features = obs["node_features"]
        anomaly_scores = node_features[:, 15]
        most_anomalous = int(np.argmax(anomaly_scores))

        if anomaly_scores[most_anomalous] > 0.6:
            return [2, most_anomalous]
        if anomaly_scores[most_anomalous] > 0.3:
            return [0, most_anomalous]

        vuln_counts = node_features[:, 10]
        most_vulnerable = int(np.argmax(vuln_counts))
        return [1, most_vulnerable]


def compute_kpis(eval_results: dict) -> dict:
    blue_damage = eval_results["blue_mean_damage"]
    no_def_damage = eval_results["no_defense_mean_damage"]
    rule_damage = eval_results["rule_based_mean_damage"]
    blue_latency = eval_results["blue_mean_detection_latency"]
    rule_latency = eval_results["rule_based_mean_detection_latency"]

    return {
        "damage_reduction_vs_no_defense": 1.0 - (blue_damage / max(1e-6, no_def_damage)),
        "damage_reduction_vs_rule_based": 1.0 - (blue_damage / max(1e-6, rule_damage)),
        "detection_latency_improvement_vs_rule_based": 1.0 - (blue_latency / max(1e-6, rule_latency)),
    }


def _register_rllib_components() -> None:
    from ray.rllib.models import ModelCatalog
    from ray.tune.registry import register_env

    from backend.app.rl.model import AegisBlueNet

    # Safe to attempt re-registration in long-lived processes.
    try:
        ModelCatalog.register_custom_model("aegis_blue_net", AegisBlueNet)
    except Exception:
        pass
    try:
        register_env("CyberDefenseEnv", lambda env_cfg: CyberDefenseEnv(env_cfg))
    except Exception:
        pass


def _load_trained_policy(checkpoint_path: str | None) -> Callable[[dict], list[int]]:
    if not checkpoint_path:
        rule = RuleBasedBaseline()
        return rule.select_action

    checkpoint = Path(checkpoint_path)
    if not checkpoint.exists():
        rule = RuleBasedBaseline()
        return rule.select_action

    try:
        import ray
        from ray.rllib.algorithms.algorithm import Algorithm
    except ImportError:
        rule = RuleBasedBaseline()
        return rule.select_action

    ray.init(ignore_reinit_error=True)
    _register_rllib_components()

    algo = Algorithm.from_checkpoint(str(checkpoint.resolve()))

    def select_action(obs: dict) -> list[int]:
        action = algo.compute_single_action(obs, explore=False)
        return [int(action[0]), int(action[1])]

    return select_action


def _evaluate_single_episode(
    policy_fn: Callable[[dict], list[int]],
    scenario_id: str,
    seed: int,
    checkpoint_id: str,
    replay_dir: Path | None,
    replay_id: str | None,
) -> dict:
    env = CyberDefenseEnv({"max_steps": 200, "max_nodes": 28})
    obs, info = env.reset(seed=seed, options={"scenario_id": scenario_id})

    recorder = None
    if replay_dir is not None and replay_id is not None:
        recorder = ReplayRecorder(replay_id=replay_id, scenario_id=scenario_id, seed=seed, checkpoint_id=checkpoint_id)
        recorder.record_topology_init(info["topology"], env.network_state)

    cumulative_reward = 0.0
    damage_trace: list[float] = []

    terminated = False
    truncated = False
    while not (terminated or truncated):
        action = policy_fn(obs)
        obs, reward, terminated, truncated, step_info = env.step(action)
        cumulative_reward += reward

        compromised_criticality = sum(
            host.criticality
            for host in env.network_state.hosts
            if host.compromise_level > 0 and not host.is_decoy and host.zone != "external"
        )
        damage_trace.append(float(compromised_criticality))

        if recorder is not None:
            recorder.record_step(
                step=step_info["step"],
                ordered_events=step_info["events"],
                state=env.network_state.clone(),
                metrics_tick=step_info["metrics"],
            )

    hvts = [h for h in env.network_state.hosts if h.host_type == "hvt"]
    hvts_compromised = sum(h.compromise_level > 0 for h in hvts)
    data_exfiltrated = any(h.compromise_level == 4 for h in hvts)

    if data_exfiltrated:
        outcome = "breached"
    elif hvts_compromised == 0:
        outcome = "contained"
    else:
        outcome = "timeout"

    summary = {
        "total_red_actions": sum(1 for event in (recorder.events if recorder else []) if event["type"] == "action_event" and event["data"].get("actor") == "RED"),
        "total_blue_actions": sum(1 for event in (recorder.events if recorder else []) if event["type"] == "action_event" and event["data"].get("actor") == "BLUE"),
        "nodes_compromised": sum(h.compromise_level > 0 for h in env.network_state.hosts if not h.is_decoy),
        "nodes_isolated": sum(h.defense_state == "isolated" for h in env.network_state.hosts if not h.is_decoy),
        "nodes_patched": sum(h.defense_state == "patched" for h in env.network_state.hosts if not h.is_decoy),
        "hvts_compromised": hvts_compromised,
        "data_exfiltrated": data_exfiltrated,
        "final_service_availability": round(float(env.network_state.service_availability_score), 4),
        "blue_reward_total": round(float(cumulative_reward), 4),
    }

    detection_latency_mean = float(np.mean(env.detection_latencies)) if env.detection_latencies else float(env.max_steps)
    damage_score = float(mean(damage_trace)) if damage_trace else 0.0

    episode_report = {
        "scenario_id": scenario_id,
        "seed": seed,
        "outcome": outcome,
        "damage_score": damage_score,
        "detection_latency_mean": detection_latency_mean,
        "service_availability_final": float(env.network_state.service_availability_score),
        "blue_reward_total": float(cumulative_reward),
        "summary": summary,
        "kpis": {
            "damage_score": damage_score,
            "containment_time_steps": int(next((i + 1 for i, dmg in enumerate(damage_trace) if dmg == 0), len(damage_trace))),
        },
    }

    episode_report["composite_score"] = (
        (1.0 / (1.0 + damage_score)) * 0.5
        + episode_report["service_availability_final"] * 0.3
        + (1.0 / (1.0 + detection_latency_mean)) * 0.2
    )

    if recorder is not None:
        recorder.finalize(outcome=outcome, summary=summary, kpis=episode_report["kpis"])
        bundle_path = recorder.save(replay_dir)
        episode_report["replay_id"] = replay_id
        episode_report["bundle_path"] = str(bundle_path)
        episode_report["validator_errors"] = validate_replay(str(bundle_path))

    return episode_report


def _copy_bundle(src: Path, dst: Path) -> None:
    if dst.exists():
        shutil.rmtree(dst)
    shutil.copytree(src, dst)


@app.command()
def main(
    checkpoint_path: str | None = typer.Option(None, help="Path to trained RLlib checkpoint"),
    runs_root: str = typer.Option("runs", help="Root output directory"),
    run_id: str | None = typer.Option(None, help="Run id to reuse"),
) -> None:
    run_dirs = create_run_dirs(runs_root=runs_root, run_id=run_id)
    run_id_value = str(run_dirs["run_id"])

    replay_dir = run_dirs["replay_dir"]
    eval_dir = run_dirs["eval_dir"]

    checkpoint_id = Path(checkpoint_path).name if checkpoint_path else "rule_based_fallback"
    trained_policy = _load_trained_policy(checkpoint_path)
    no_defense = NoDefenseBaseline()
    rule_based = RuleBasedBaseline()

    blue_reports: list[dict] = []
    no_def_reports: list[dict] = []
    rule_reports: list[dict] = []

    episode_idx = 0
    for scenario_entry in EVAL_SUITE["scenarios"]:
        scenario_id = scenario_entry["scenario"]
        for seed in scenario_entry["seeds"]:
            episode_idx += 1
            replay_id = f"replay_ep_{episode_idx:03d}"
            blue_reports.append(
                _evaluate_single_episode(
                    policy_fn=trained_policy,
                    scenario_id=scenario_id,
                    seed=seed,
                    checkpoint_id=checkpoint_id,
                    replay_dir=replay_dir,
                    replay_id=replay_id,
                )
            )
            no_def_reports.append(
                _evaluate_single_episode(
                    policy_fn=no_defense.select_action,
                    scenario_id=scenario_id,
                    seed=seed,
                    checkpoint_id="baseline_no_defense",
                    replay_dir=None,
                    replay_id=None,
                )
            )
            rule_reports.append(
                _evaluate_single_episode(
                    policy_fn=rule_based.select_action,
                    scenario_id=scenario_id,
                    seed=seed,
                    checkpoint_id="baseline_rule_based",
                    replay_dir=None,
                    replay_id=None,
                )
            )

    eval_results = {
        "blue_mean_damage": float(mean(r["damage_score"] for r in blue_reports)),
        "no_defense_mean_damage": float(mean(r["damage_score"] for r in no_def_reports)),
        "rule_based_mean_damage": float(mean(r["damage_score"] for r in rule_reports)),
        "blue_mean_detection_latency": float(mean(r["detection_latency_mean"] for r in blue_reports)),
        "rule_based_mean_detection_latency": float(mean(r["detection_latency_mean"] for r in rule_reports)),
    }
    kpis = compute_kpis(eval_results)
    gates = {
        "damage_reduction_vs_no_defense": kpis["damage_reduction_vs_no_defense"] >= 0.25,
        "damage_reduction_vs_rule_based": kpis["damage_reduction_vs_rule_based"] >= 0.15,
        "detection_latency_improvement_vs_rule_based": kpis["detection_latency_improvement_vs_rule_based"] >= 0.20,
    }

    valid_blue_reports = [r for r in blue_reports if not r.get("validator_errors")]
    if len(valid_blue_reports) < 4:
        raise RuntimeError("Not enough validator-passing replays to select hero + backups")

    selection = select_hero_and_backups(valid_blue_reports)
    for alias, report in selection.items():
        _copy_bundle(Path(report["bundle_path"]), replay_dir / alias)

    selection_manifest = write_selection_manifest(replay_dir, selection)

    eval_id = f"eval_cli_{int(time.time())}"
    eval_payload = {
        "eval_id": eval_id,
        "run_id": run_id_value,
        "checkpoint_id": checkpoint_id,
        "suite": EVAL_SUITE,
        "eval_results": eval_results,
        "kpis": kpis,
        "gates": gates,
        "selection_manifest": str(selection_manifest),
        "episodes": blue_reports,
    }

    out_path = eval_dir / f"{eval_id}.json"
    out_path.write_text(json.dumps(eval_payload, indent=2), encoding="utf-8")

    print(json.dumps({"run_id": run_id_value, "eval_id": eval_id, "kpis": kpis, "gates": gates}, indent=2))


if __name__ == "__main__":
    app()
