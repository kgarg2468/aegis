import math

from backend.app.rl.train import _extract_training_metrics


def test_extract_training_metrics_reads_nested_reward_and_learner_stats():
    result = {
        "env_runners": {
            "episode_reward_mean": 1.25,
        },
        "info": {
            "learner": {
                "default_policy": {
                    "learner_stats": {
                        "policy_loss": 0.11,
                        "vf_loss": 0.22,
                        "entropy": 0.33,
                        "kl": 0.04,
                        "total_loss": 0.55,
                    }
                }
            }
        },
    }

    metrics = _extract_training_metrics(result)

    assert metrics["reward_mean"] == 1.25
    assert metrics["reward_source"] == "env_runners.episode_reward_mean"
    assert metrics["policy_loss"] == 0.11
    assert metrics["vf_loss"] == 0.22
    assert metrics["entropy"] == 0.33
    assert metrics["kl"] == 0.04
    assert metrics["total_loss"] == 0.55


def test_extract_training_metrics_returns_defaults_when_missing():
    metrics = _extract_training_metrics({})

    assert metrics["reward_mean"] == 0.0
    assert metrics["reward_source"] == "missing"
    assert math.isnan(float(metrics["policy_loss"]))
    assert math.isnan(float(metrics["vf_loss"]))
    assert math.isnan(float(metrics["entropy"]))
    assert math.isnan(float(metrics["kl"]))
    assert math.isnan(float(metrics["total_loss"]))
