from __future__ import annotations

PPO_CONFIG: dict = {
    "lr": 3e-4,
    "gamma": 0.99,
    "lambda_": 0.95,
    "clip_param": 0.2,
    "entropy_coeff": 0.01,
    "vf_loss_coeff": 0.5,
    "max_grad_norm": 0.5,
    "train_batch_size": 4096,
    "sgd_minibatch_size": 512,
    "num_sgd_iter": 10,
    "num_rollout_workers": 8,
    "rollout_fragment_length": 200,
    "batch_mode": "complete_episodes",
    "model": {
        "custom_model": "aegis_blue_net",
        "custom_model_config": {
            "node_embed_dim": 64,
            "global_embed_dim": 32,
            "gnn_layers": 2,
            "lstm_hidden": 128,
            "fc_hiddens": [256, 128],
        },
    },
    "num_gpus": 3,
    "num_gpus_per_worker": 0,
    "env": "CyberDefenseEnv",
    "env_config": {
        "max_steps": 200,
        "max_nodes": 28,
        "scenario_weights": [0.2, 0.2, 0.2, 0.2, 0.2],
    },
    "checkpoint_freq": 50,
    "checkpoint_at_end": True,
    "stop": {
        "timesteps_total": 3_000_000,
        "episode_reward_mean": 5.0,
    },
}


def stage_timesteps(stage: str) -> int:
    if stage == "smoke":
        return 5_000
    if stage == "sanity":
        return 50_000
    if stage == "full":
        return 3_000_000
    raise ValueError(f"Unknown stage: {stage}")
