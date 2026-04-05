from __future__ import annotations

import json
from datetime import UTC, datetime
from numbers import Real
from pathlib import Path

import typer

from backend.app.env.cyber_defense_env import CyberDefenseEnv
from backend.app.replay.run_manager import create_run_dirs
from backend.app.rl.config import build_stage_config

app = typer.Typer(add_completion=False)


def _checkpoint_path(save_result) -> str:
    checkpoint = getattr(save_result, "checkpoint", None)
    if checkpoint is not None:
        path = getattr(checkpoint, "path", None)
        if path is not None:
            return str(path)

    path = getattr(save_result, "path", None)
    if path is not None:
        return str(path)

    return str(save_result)


def _build_algo(config: dict):
    try:
        import ray
        from ray.rllib.algorithms.ppo import PPOConfig
        from ray.rllib.models import ModelCatalog
        from ray.tune.registry import register_env
    except ImportError as exc:  # pragma: no cover - requires train extras
        raise RuntimeError("ray[rllib] is required for training. Install with: uv sync --extra train") from exc

    from backend.app.rl.model import AegisBlueNet

    ray.init(ignore_reinit_error=True)
    ModelCatalog.register_custom_model("aegis_blue_net", AegisBlueNet)
    register_env("CyberDefenseEnv", lambda env_cfg: CyberDefenseEnv(env_cfg))

    ppo_config = (
        PPOConfig()
        .environment("CyberDefenseEnv", env_config=config.get("env_config", {}))
        .framework("torch")
        .training(
            lr=config["lr"],
            gamma=config["gamma"],
            lambda_=config["lambda_"],
            clip_param=config["clip_param"],
            entropy_coeff=config["entropy_coeff"],
            vf_loss_coeff=config["vf_loss_coeff"],
            grad_clip=config["max_grad_norm"],
            train_batch_size=config["train_batch_size"],
            minibatch_size=config.get("minibatch_size", config.get("sgd_minibatch_size", 512)),
            num_epochs=config.get("num_epochs", config["num_sgd_iter"]),
            model=config["model"],
        )
    )

    if hasattr(ppo_config, "api_stack"):
        ppo_config = ppo_config.api_stack(
            enable_rl_module_and_learner=False,
            enable_env_runner_and_connector_v2=False,
        )

    if hasattr(ppo_config, "env_runners"):
        ppo_config = ppo_config.env_runners(
            num_env_runners=config["num_rollout_workers"],
            rollout_fragment_length=config["rollout_fragment_length"],
            batch_mode=config["batch_mode"],
        )
    else:
        ppo_config = ppo_config.rollouts(
            num_rollout_workers=config["num_rollout_workers"],
            rollout_fragment_length=config["rollout_fragment_length"],
            batch_mode=config["batch_mode"],
        )

    ppo_config = ppo_config.resources(
        num_gpus=config["num_gpus"],
        num_gpus_per_worker=config["num_gpus_per_worker"],
    )
    return ppo_config.build()


def _load_config_overrides(config_overrides_path: str | None) -> dict | None:
    if not config_overrides_path:
        return None

    path = Path(config_overrides_path)
    if not path.exists():
        raise FileNotFoundError(f"Config overrides file not found: {path}")

    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError("Config overrides must be a JSON object")
    return payload


def _nested_get(data: dict, path: tuple[str, ...]):
    current = data
    for key in path:
        if not isinstance(current, dict):
            return None
        current = current.get(key)
        if current is None:
            return None
    return current


def _extract_scalar_metric(result: dict, candidates: list[tuple[str, tuple[str, ...]]]) -> tuple[float | None, str | None]:
    for source, path in candidates:
        value = _nested_get(result, path)
        if isinstance(value, Real):
            return float(value), source
    return None, None


def _extract_training_metrics(result: dict) -> dict[str, float | str]:
    reward_candidates = [
        ("episode_reward_mean", ("episode_reward_mean",)),
        ("env_runners.episode_reward_mean", ("env_runners", "episode_reward_mean")),
        ("sampler_results.episode_reward_mean", ("sampler_results", "episode_reward_mean")),
        ("env_runners.episode_return_mean", ("env_runners", "episode_return_mean")),
    ]
    reward_value, reward_source = _extract_scalar_metric(result, reward_candidates)
    if reward_value is None:
        reward_value = 0.0
        reward_source = "missing"

    learner_stat_keys = {
        "policy_loss": [
            ("info.learner.default_policy.learner_stats.policy_loss", ("info", "learner", "default_policy", "learner_stats", "policy_loss")),
        ],
        "vf_loss": [
            ("info.learner.default_policy.learner_stats.vf_loss", ("info", "learner", "default_policy", "learner_stats", "vf_loss")),
        ],
        "entropy": [
            ("info.learner.default_policy.learner_stats.entropy", ("info", "learner", "default_policy", "learner_stats", "entropy")),
        ],
        "kl": [
            ("info.learner.default_policy.learner_stats.kl", ("info", "learner", "default_policy", "learner_stats", "kl")),
        ],
        "total_loss": [
            ("info.learner.default_policy.learner_stats.total_loss", ("info", "learner", "default_policy", "learner_stats", "total_loss")),
        ],
    }

    out: dict[str, float | str] = {
        "reward_mean": reward_value,
        "reward_source": reward_source,
    }
    for name, candidates in learner_stat_keys.items():
        value, source = _extract_scalar_metric(result, candidates)
        out[name] = float(value) if value is not None else float("nan")
        out[f"{name}_source"] = source or "missing"
    return out


@app.command()
def main(
    stage: str = typer.Option("smoke", help="smoke | sanity | full"),
    runs_root: str = typer.Option("runs", help="Root output directory"),
    run_id: str | None = typer.Option(None, help="Reuse an existing run id"),
    config_overrides_path: str | None = typer.Option(
        None,
        help="Optional JSON file with PPO config overrides (deep-merged into default config)",
    ),
    train_timesteps: int | None = typer.Option(
        None,
        min=1,
        help="Optional override for stop.timesteps_total (used for short sweeps)",
    ),
) -> None:
    """Train PPO blue policy and save checkpoints under runs/runNNN/train."""

    run_dirs = create_run_dirs(runs_root=runs_root, run_id=run_id)
    run_id_value = str(run_dirs["run_id"])

    overrides = _load_config_overrides(config_overrides_path)
    cfg = build_stage_config(stage, overrides=overrides, timesteps_override=train_timesteps)
    target_timesteps = int(cfg["stop"]["timesteps_total"])

    algo = _build_algo(cfg)
    train_dir = run_dirs["train_dir"]
    metrics_path = train_dir / "train_metrics.jsonl"

    checkpoint_freq = int(cfg.get("checkpoint_freq", 50))
    iteration = 0
    timesteps_total = 0
    best_reward = float("-inf")
    best_reward_source = "missing"
    best_checkpoint_path = None
    last_metrics: dict[str, float | str] = {}

    while timesteps_total < target_timesteps:
        iteration += 1
        result = algo.train()
        timesteps_total = int(result.get("timesteps_total", timesteps_total))

        iter_metrics = _extract_training_metrics(result)
        reward_mean = float(iter_metrics["reward_mean"])
        reward_source = str(iter_metrics["reward_source"])
        last_metrics = dict(iter_metrics)

        metrics_row = {
            "timestamp_utc": datetime.now(UTC).isoformat(),
            "iteration": iteration,
            "timesteps_total": timesteps_total,
            **iter_metrics,
        }
        with metrics_path.open("a", encoding="utf-8") as metrics_file:
            metrics_file.write(json.dumps(metrics_row) + "\n")

        if reward_mean > best_reward:
            best_reward = reward_mean
            best_reward_source = reward_source
            best_checkpoint_path = _checkpoint_path(algo.save(checkpoint_dir=str(train_dir / "best")))

        if iteration % checkpoint_freq == 0:
            algo.save(checkpoint_dir=str(train_dir / "checkpoints"))

        if reward_mean >= float(cfg["stop"].get("episode_reward_mean", float("inf"))):
            break

    final_checkpoint_path = _checkpoint_path(algo.save(checkpoint_dir=str(train_dir / "final")))

    metadata = {
        "run_id": run_id_value,
        "stage": stage,
        "target_timesteps": target_timesteps,
        "timesteps_total": timesteps_total,
        "best_reward_mean": best_reward,
        "best_reward_source": best_reward_source,
        "best_checkpoint": best_checkpoint_path,
        "final_checkpoint": final_checkpoint_path,
        "metrics_file": str(metrics_path),
        "final_iteration_metrics": last_metrics,
        "config_overrides_path": config_overrides_path,
        "config_overrides": overrides or {},
        "train_timesteps_override": train_timesteps,
    }
    (train_dir / "train_metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")

    print(json.dumps(metadata, indent=2))


if __name__ == "__main__":
    app()
