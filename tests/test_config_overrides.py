from backend.app.rl.config import build_stage_config, merge_config


def test_merge_config_deep_merge_without_mutating_base():
    base = {
        "lr": 3e-4,
        "env_config": {"max_steps": 200, "max_nodes": 28},
    }
    overrides = {
        "lr": 1e-4,
        "env_config": {"max_steps": 150},
    }

    merged = merge_config(base, overrides)

    assert merged["lr"] == 1e-4
    assert merged["env_config"]["max_steps"] == 150
    assert merged["env_config"]["max_nodes"] == 28
    assert base["lr"] == 3e-4
    assert base["env_config"]["max_steps"] == 200


def test_build_stage_config_sets_target_timesteps_with_overrides():
    cfg = build_stage_config("smoke", overrides={"lr": 2e-4, "stop": {"episode_reward_mean": 7.0}})

    assert cfg["lr"] == 2e-4
    assert cfg["stop"]["timesteps_total"] == 5000
    assert cfg["stop"]["episode_reward_mean"] == 7.0


def test_build_stage_config_respects_timesteps_from_stop_override():
    cfg = build_stage_config("smoke", overrides={"stop": {"timesteps_total": 7777}})
    assert cfg["stop"]["timesteps_total"] == 7777


def test_build_stage_config_train_timesteps_override_takes_priority():
    cfg = build_stage_config(
        "full",
        overrides={"stop": {"timesteps_total": 5555}},
        timesteps_override=12345,
    )
    assert cfg["stop"]["timesteps_total"] == 12345
