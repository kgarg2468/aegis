import json

import pytest

from backend.app.rl.autopilot import (
    KPI_THRESHOLDS,
    _build_eval_command,
    _build_train_command,
    _load_sweep_spec,
    _select_checkpoint_path,
    compute_kpi_statistics,
    suggest_next_overrides,
)


def test_compute_kpi_statistics_ci_passes_for_strong_consistent_runs():
    runs = [
        {
            "damage_reduction_vs_no_defense": 0.33,
            "damage_reduction_vs_rule_based": 0.20,
            "detection_latency_improvement_vs_rule_based": 0.26,
        },
        {
            "damage_reduction_vs_no_defense": 0.34,
            "damage_reduction_vs_rule_based": 0.21,
            "detection_latency_improvement_vs_rule_based": 0.24,
        },
        {
            "damage_reduction_vs_no_defense": 0.32,
            "damage_reduction_vs_rule_based": 0.19,
            "detection_latency_improvement_vs_rule_based": 0.25,
        },
        {
            "damage_reduction_vs_no_defense": 0.35,
            "damage_reduction_vs_rule_based": 0.22,
            "detection_latency_improvement_vs_rule_based": 0.27,
        },
        {
            "damage_reduction_vs_no_defense": 0.31,
            "damage_reduction_vs_rule_based": 0.18,
            "detection_latency_improvement_vs_rule_based": 0.23,
        },
    ]

    stats = compute_kpi_statistics(runs)

    assert stats["num_runs"] == 5
    assert stats["all_ci_gates_pass"] is True
    assert stats["variance_high"] is False

    for key, threshold in KPI_THRESHOLDS.items():
        metric = stats["metrics"][key]
        assert metric["ci95_lower"] >= threshold
        assert metric["std"] <= 0.05


def test_compute_kpi_statistics_flags_high_variance():
    runs = [
        {
            "damage_reduction_vs_no_defense": 0.40,
            "damage_reduction_vs_rule_based": 0.17,
            "detection_latency_improvement_vs_rule_based": 0.21,
        },
        {
            "damage_reduction_vs_no_defense": 0.10,
            "damage_reduction_vs_rule_based": 0.16,
            "detection_latency_improvement_vs_rule_based": 0.22,
        },
        {
            "damage_reduction_vs_no_defense": 0.36,
            "damage_reduction_vs_rule_based": 0.18,
            "detection_latency_improvement_vs_rule_based": 0.23,
        },
        {
            "damage_reduction_vs_no_defense": 0.12,
            "damage_reduction_vs_rule_based": 0.17,
            "detection_latency_improvement_vs_rule_based": 0.19,
        },
        {
            "damage_reduction_vs_no_defense": 0.38,
            "damage_reduction_vs_rule_based": 0.15,
            "detection_latency_improvement_vs_rule_based": 0.24,
        },
    ]

    stats = compute_kpi_statistics(runs)

    assert stats["variance_high"] is True
    assert stats["metrics"]["damage_reduction_vs_no_defense"]["std"] > 0.05


def test_suggest_next_overrides_reduces_lr_when_variance_is_high():
    runs = [
        {
            "damage_reduction_vs_no_defense": 0.40,
            "damage_reduction_vs_rule_based": 0.17,
            "detection_latency_improvement_vs_rule_based": 0.21,
        },
        {
            "damage_reduction_vs_no_defense": 0.10,
            "damage_reduction_vs_rule_based": 0.16,
            "detection_latency_improvement_vs_rule_based": 0.22,
        },
        {
            "damage_reduction_vs_no_defense": 0.36,
            "damage_reduction_vs_rule_based": 0.18,
            "detection_latency_improvement_vs_rule_based": 0.23,
        },
    ]
    stats = compute_kpi_statistics(runs)
    previous = {"lr": 3e-4, "entropy_coeff": 0.01, "clip_param": 0.2, "train_batch_size": 4096}

    nxt, reasons = suggest_next_overrides(stats, previous)

    assert nxt["lr"] < previous["lr"]
    assert nxt["train_batch_size"] > previous["train_batch_size"]
    assert any("variance" in reason.lower() for reason in reasons)


def test_suggest_next_overrides_keeps_config_until_at_least_two_runs():
    stats = compute_kpi_statistics(
        [
            {
                "damage_reduction_vs_no_defense": 0.31,
                "damage_reduction_vs_rule_based": 0.18,
                "detection_latency_improvement_vs_rule_based": 0.24,
            }
        ]
    )
    previous = {"lr": 3e-4, "entropy_coeff": 0.01, "clip_param": 0.2, "train_batch_size": 4096}

    nxt, reasons = suggest_next_overrides(stats, previous)

    assert nxt == previous
    assert "not enough runs" in reasons[0].lower()


def test_select_checkpoint_prefers_final_checkpoint(tmp_path):
    (tmp_path / "final_ckpt").mkdir()
    (tmp_path / "best_ckpt").mkdir()

    train_metadata = {
        "final_checkpoint": "/app/final_ckpt",
        "best_checkpoint": "/app/best_ckpt",
    }
    metadata_path = tmp_path / "train_metadata.json"
    metadata_path.write_text(json.dumps(train_metadata), encoding="utf-8")

    selected = _select_checkpoint_path(metadata_path, project_root=tmp_path)
    assert selected == (tmp_path / "final_ckpt")


def test_load_sweep_spec_normalizes_and_validates(tmp_path):
    spec = {
        "runs": [
            {
                "label": "r1",
                "overrides": {"lr": 2e-4},
                "train_timesteps": 250000,
                "seeds_per_scenario": 4,
                "seed_start": 2001,
            }
        ]
    }
    path = tmp_path / "spec.json"
    path.write_text(json.dumps(spec), encoding="utf-8")

    runs = _load_sweep_spec(path)
    assert len(runs) == 1
    assert runs[0]["label"] == "r1"
    assert runs[0]["train_timesteps"] == 250000
    assert runs[0]["seeds_per_scenario"] == 4
    assert runs[0]["seed_start"] == 2001


def test_load_sweep_spec_rejects_invalid_seed_count(tmp_path):
    spec = {"runs": [{"label": "bad", "overrides": {}, "seeds_per_scenario": 0}]}
    path = tmp_path / "bad_spec.json"
    path.write_text(json.dumps(spec), encoding="utf-8")

    with pytest.raises(ValueError):
        _load_sweep_spec(path)


def test_build_train_command_includes_train_timesteps_for_local_runs(tmp_path):
    cmd = _build_train_command(
        use_docker=False,
        docker_image="shield-train",
        project_root=tmp_path,
        runs_root="runs",
        run_id="run123",
        stage="full",
        cuda_visible_devices="5,6,7",
        config_overrides_path=tmp_path / "overrides.json",
        train_timesteps=1234,
    )

    assert "--train-timesteps" in cmd
    idx = cmd.index("--train-timesteps")
    assert cmd[idx + 1] == "1234"


def test_build_eval_command_includes_seed_arguments_for_local_runs(tmp_path):
    cmd = _build_eval_command(
        use_docker=False,
        docker_image="shield-train",
        project_root=tmp_path,
        runs_root="runs",
        run_id="run123",
        checkpoint_path=tmp_path / "ckpt",
        cuda_visible_devices="5,6,7",
        seeds_per_scenario=4,
        seed_start=2001,
    )

    assert "--seeds-per-scenario" in cmd
    assert "--seed-start" in cmd
    assert cmd[cmd.index("--seeds-per-scenario") + 1] == "4"
    assert cmd[cmd.index("--seed-start") + 1] == "2001"
