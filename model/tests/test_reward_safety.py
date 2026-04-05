import numpy as np

from backend.app.env.action_logic import apply_deception
from backend.app.env.network_state import HostState, NetworkState
from backend.app.env.reward import BlueAction, compute_reward
from backend.app.env.topology_generator import generate_chapman_topology


def test_compute_reward_handles_new_decoy_nodes():
    prev_state = generate_chapman_topology(np.random.default_rng(7))
    state = prev_state.clone()
    state.step = 1

    target = state.get_host("auth_server")
    decoy_id = apply_deception(state, target)
    assert decoy_id is not None

    reward = compute_reward(
        state=state,
        action=BlueAction(type="deploy_deception", target="auth_server"),
        prev_state=prev_state,
    )
    assert isinstance(reward, float)


def test_newly_compromised_hosts_handles_nodes_missing_in_prev_state():
    prev_state = NetworkState(hosts=[], edges=[], step=0)
    current_state = NetworkState(
        hosts=[
            HostState(
                host_id="decoy_test",
                zone="campus",
                host_type="infrastructure",
                services=[],
                vulnerabilities=[],
                criticality=0.0,
                compromise_level=2,
                is_decoy=True,
            )
        ],
        edges=[],
        step=1,
    )

    newly = current_state.newly_compromised_hosts(prev_state)
    assert [host.host_id for host in newly] == ["decoy_test"]


def test_reward_weights_override_changes_reward_magnitude():
    prev_state = generate_chapman_topology(np.random.default_rng(11))
    state = prev_state.clone()
    state.step = 1

    action = BlueAction(type="patch_service", target="auth_server")
    reward_low_cost = compute_reward(
        state=state,
        action=action,
        prev_state=prev_state,
        reward_weights={"action_cost_scale": 0.1},
    )
    reward_high_cost = compute_reward(
        state=state,
        action=action,
        prev_state=prev_state,
        reward_weights={"action_cost_scale": 2.0},
    )

    assert reward_low_cost > reward_high_cost
