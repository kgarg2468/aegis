from __future__ import annotations

import numpy as np

from backend.app.env.network_state import EdgeState, HostState, NetworkState


EDGE_STATE_PRIORITY = {
    "exfiltration": 5,
    "lateral_movement": 4,
    "credential_flow": 3,
    "scanning": 2,
    "normal": 1,
    "blocked": 0,
}


def choose_block_connection_edge(state: NetworkState, target_host: str) -> EdgeState | None:
    candidates = [
        edge
        for edge in state.edges
        if edge.status == "active" and (edge.source == target_host or edge.target == target_host)
    ]
    if not candidates:
        return None

    def rank(edge: EdgeState) -> tuple[float, float, str]:
        return (
            -float(EDGE_STATE_PRIORITY.get(edge.visual_state, 0)),
            -float(edge.risk_score),
            edge.edge_id,
        )

    candidates.sort(key=rank)
    return candidates[0]


def get_valid_action_mask(state: NetworkState, max_nodes: int = 28) -> np.ndarray:
    mask = np.zeros((6, max_nodes), dtype=bool)

    for idx, host in enumerate(state.hosts[:max_nodes]):
        if host.is_decoy:
            # Blue policy does not target decoys directly.
            continue

        if host.defense_state != "isolated":
            mask[0, idx] = True

        if host.vulnerabilities and host.defense_state != "isolated":
            mask[1, idx] = True

        if host.defense_state != "isolated" and host.zone != "external":
            mask[2, idx] = True

        if state.get_active_edges(host.host_id):
            mask[3, idx] = True

        if any(service in host.services for service in ["radius", "ldap", "kerberos", "ssh"]):
            mask[4, idx] = True

        if host.host_type == "hvt" or host.is_compromised_detected:
            mask[5, idx] = True

    return mask


def apply_monitor(host: HostState) -> None:
    if host.defense_state != "isolated":
        host.defense_state = "monitored"


def apply_patch(host: HostState) -> None:
    if host.defense_state != "isolated" and host.vulnerabilities:
        host.vulnerabilities.pop(0)
        host.defense_state = "patched"


def apply_isolate(host: HostState, state: NetworkState) -> None:
    host.defense_state = "isolated"
    for edge in state.edges:
        if edge.source == host.host_id or edge.target == host.host_id:
            edge.status = "blocked"
            edge.visual_state = "blocked"
            edge.risk_score = 0.0


def apply_rotate_credentials(host: HostState) -> None:
    host.credential_compromised = False


def apply_deception(state: NetworkState, host: HostState) -> str | None:
    if host.zone == "external":
        return None
    decoy_id = f"decoy_{host.host_id}"
    if state.maybe_get_host(decoy_id) is not None:
        return None
    decoy = HostState(
        host_id=decoy_id,
        zone=host.zone,
        host_type="infrastructure",
        services=host.services[:2],
        vulnerabilities=[],
        criticality=0.0,
        is_decoy=True,
        defense_state="none",
    )
    state.hosts.append(decoy)
    state._host_by_id[decoy_id] = decoy
    state.edges.append(
        EdgeState(
            source=decoy_id,
            target=host.host_id,
            status="active",
            visual_state="normal",
            risk_score=0.0,
        )
    )
    return decoy_id
