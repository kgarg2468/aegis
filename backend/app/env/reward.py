from __future__ import annotations

from dataclasses import dataclass

from backend.app.env.network_state import HostState, NetworkState


ACTION_COSTS: dict[str, dict[str, float]] = {
    "monitor_host": {"base_cost": 0.01, "criticality_multiplier": 0.0},
    "patch_service": {"base_cost": 0.05, "criticality_multiplier": 0.1},
    "isolate_host": {"base_cost": 0.15, "criticality_multiplier": 0.3},
    "block_connection": {"base_cost": 0.08, "criticality_multiplier": 0.15},
    "rotate_credentials": {"base_cost": 0.1, "criticality_multiplier": 0.2},
    "deploy_deception": {"base_cost": 0.03, "criticality_multiplier": 0.0},
}


@dataclass
class BlueAction:
    type: str
    target: str


def compute_action_cost(action: str, target: HostState) -> float:
    spec = ACTION_COSTS[action]
    return spec["base_cost"] + spec["criticality_multiplier"] * float(target.criticality)


def _would_have_exfiltrated(state: NetworkState, prev_state: NetworkState, action: BlueAction) -> bool:
    # Simple estimator: if any previously critical host still had active egress and we blocked/isolated it.
    if action.type not in {"isolate_host", "block_connection"}:
        return False
    target = state.get_host(action.target)
    was_hot = prev_state.get_host(target.host_id).compromise_level >= 3
    return was_hot and target.outbound_data_volume > 0.4


def compute_reward(state: NetworkState, action: BlueAction, prev_state: NetworkState) -> float:
    reward = 0.0

    for host in state.hosts:
        prev_host = prev_state.get_host(host.host_id)
        if host.is_compromised_detected and not prev_host.is_compromised_detected:
            if host.compromise_step < 0:
                steps_since_compromise = state.step
            else:
                steps_since_compromise = max(0, state.step - host.compromise_step)
            early_detection_bonus = max(0.0, 1.0 - steps_since_compromise * 0.1)
            reward += 0.3 * early_detection_bonus * host.criticality

    if action.type == "isolate_host":
        target = state.get_host(action.target)
        if target.compromise_level > 0:
            downstream_value = sum(
                neighbor.criticality
                for neighbor in state.iter_neighbors(action.target)
                if neighbor.compromise_level == 0
            )
            reward += 0.5 * downstream_value

    if action.type in {"isolate_host", "block_connection"} and _would_have_exfiltrated(state, prev_state, action):
        reward += 1.0

    reward += 0.02 * state.service_availability_score

    for host in state.newly_compromised_hosts(prev_state):
        reward -= 0.4 * host.criticality

    for host in state.hosts:
        prev_host = prev_state.get_host(host.host_id)
        if host.compromise_level == 4 and prev_host.compromise_level < 4:
            reward -= 2.0 * host.criticality

    action_cost = compute_action_cost(action.type, state.get_host(action.target))
    reward -= action_cost

    if action.type in {"isolate_host", "block_connection"}:
        target = state.get_host(action.target)
        if target.compromise_level == 0 and not target.credential_compromised:
            reward -= 0.2 * target.criticality

    avail_drop = prev_state.service_availability_score - state.service_availability_score
    if avail_drop > 0:
        reward -= 0.3 * avail_drop

    if action.type == "monitor_host" and state.attack_pressure_score > 0.5:
        reward -= 0.05

    return float(reward)


def compute_episode_bonus(trajectory: list[dict]) -> float:
    if not trajectory:
        return 0.0

    final_state: NetworkState = trajectory[-1]["state"]
    bonus = 0.0

    hvt_nodes = [host for host in final_state.hosts if host.host_type == "hvt"]
    if hvt_nodes:
        clean_hvts = sum(1 for host in hvt_nodes if host.compromise_level == 0)
        bonus += 0.5 * (clean_hvts / len(hvt_nodes))

    bonus += 0.3 * final_state.service_availability_score

    total_exfil = sum(host.compromise_level == 4 for host in final_state.hosts)
    bonus -= 1.0 * total_exfil

    return float(bonus)
