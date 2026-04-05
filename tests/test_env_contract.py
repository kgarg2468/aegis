import numpy as np

from backend.app.env.cyber_defense_env import CyberDefenseEnv
from backend.app.env.registry import NODE_REGISTRY_IDS
from backend.app.env.action_logic import choose_block_connection_edge
from backend.app.env.network_state import EdgeState, NetworkState


def test_node_registry_size_matches_contract():
    assert len(NODE_REGISTRY_IDS) == 28


def test_env_observation_and_action_shapes():
    env = CyberDefenseEnv({"max_steps": 10})
    obs, info = env.reset(seed=7)

    assert set(obs.keys()) == {"node_features", "global_features", "adjacency", "alert_history", "action_mask"}
    assert obs["node_features"].shape == (28, 22)
    assert obs["global_features"].shape == (6,)
    assert obs["adjacency"].shape == (28, 28)
    assert obs["alert_history"].shape == (10, 28)
    assert obs["action_mask"].shape == (6, 28)
    assert np.issubdtype(obs["action_mask"].dtype, np.floating)
    assert env.action_space.nvec.tolist() == [6, 28]
    assert "topology" in info
    assert "valid_action_mask" in info
    assert info["valid_action_mask"].shape == (6, 28)


def test_block_connection_uses_highest_risk_edge_priority():
    state = NetworkState(hosts=[], edges=[
        EdgeState(source="auth_server", target="sis_server", status="active", visual_state="normal", risk_score=0.1),
        EdgeState(source="auth_server", target="finance_server", status="active", visual_state="lateral_movement", risk_score=0.6),
        EdgeState(source="auth_server", target="hr_server", status="active", visual_state="exfiltration", risk_score=0.4),
    ])

    chosen = choose_block_connection_edge(state, "auth_server")
    assert chosen is not None
    assert chosen.source == "auth_server"
    assert chosen.target == "hr_server"
