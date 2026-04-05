from backend.app.rl.autopilot import (
    KPI_THRESHOLDS,
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
