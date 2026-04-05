from __future__ import annotations

from dataclasses import dataclass, field
from typing import Iterable

from backend.app.env.registry import (
    HOST_CRITICALITY,
    HOST_SERVICES,
    NODE_INDEX_BY_ID,
    NODE_REGISTRY,
    NODE_REGISTRY_IDS,
    ZONE_LABELS,
    ZONE_MEMBER_IDS,
)


@dataclass
class HostState:
    host_id: str
    zone: str
    host_type: str
    services: list[str]
    vulnerabilities: list[str]
    criticality: float
    compromise_level: int = 0
    defense_state: str = "none"
    credential_compromised: bool = False
    has_backdoor: bool = False
    is_decoy: bool = False
    traffic_anomaly_score: float = 0.0
    login_failure_count: int = 0
    outbound_data_volume: float = 0.0
    lateral_connection_count: int = 0
    is_compromised_detected: bool = False
    compromise_step: int = -1
    time_since_last_alert: int = 0


@dataclass
class EdgeState:
    source: str
    target: str
    status: str = "active"  # active | blocked
    visual_state: str = "normal"
    risk_score: float = 0.0
    direction: str = "forward"

    @property
    def edge_id(self) -> str:
        return f"{self.source}->{self.target}"


@dataclass
class NetworkState:
    hosts: list[HostState]
    edges: list[EdgeState]
    step: int = 0
    attack_pressure_score: float = 0.0
    service_availability_score: float = 1.0
    blue_action_budget_remaining: float = 1.0

    _host_by_id: dict[str, HostState] = field(default_factory=dict, init=False, repr=False)

    def __post_init__(self) -> None:
        self._host_by_id = {h.host_id: h for h in self.hosts}

    def clone(self) -> "NetworkState":
        copied_hosts = [HostState(**vars(h)) for h in self.hosts]
        copied_edges = [EdgeState(**vars(e)) for e in self.edges]
        clone = NetworkState(
            hosts=copied_hosts,
            edges=copied_edges,
            step=self.step,
            attack_pressure_score=self.attack_pressure_score,
            service_availability_score=self.service_availability_score,
            blue_action_budget_remaining=self.blue_action_budget_remaining,
        )
        return clone

    def get_host(self, host_id: str) -> HostState:
        return self._host_by_id[host_id]

    def maybe_get_host(self, host_id: str) -> HostState | None:
        return self._host_by_id.get(host_id)

    def get_active_edges(self, host_id: str) -> list[EdgeState]:
        return [
            edge
            for edge in self.edges
            if edge.status == "active" and (edge.source == host_id or edge.target == host_id)
        ]

    def iter_neighbors(self, host_id: str) -> Iterable[HostState]:
        for edge in self.get_active_edges(host_id):
            neighbor_id = edge.target if edge.source == host_id else edge.source
            host = self.maybe_get_host(neighbor_id)
            if host is not None:
                yield host

    def set_edge_blocked(self, edge_id: str) -> None:
        for edge in self.edges:
            if edge.edge_id == edge_id:
                edge.status = "blocked"
                edge.visual_state = "blocked"
                edge.risk_score = 0.0

    def count_compromised(self) -> int:
        return sum(1 for host in self.hosts if host.compromise_level > 0)

    def newly_compromised_hosts(self, prev_state: "NetworkState") -> list[HostState]:
        new_hosts: list[HostState] = []
        for host in self.hosts:
            prev = prev_state.maybe_get_host(host.host_id)
            if prev is None:
                if host.compromise_level > 0:
                    new_hosts.append(host)
                continue
            if host.compromise_level > prev.compromise_level:
                new_hosts.append(host)
        return new_hosts

    def all_hvts_compromised(self) -> bool:
        hvts = [h for h in self.hosts if h.host_type == "hvt"]
        return bool(hvts) and all(h.compromise_level >= 4 for h in hvts)

    def state_snapshot(self) -> dict:
        """Full topology snapshot used for replay seek checkpoints."""
        nodes = []
        for node_id in NODE_REGISTRY_IDS:
            host = self.maybe_get_host(node_id)
            if host is None:
                continue
            nodes.append(
                {
                    "id": host.host_id,
                    "zone": host.zone,
                    "type": host.host_type,
                    "visual_state": host_visual_state(host),
                    "overlay": "monitored" if host.defense_state == "monitored" else None,
                    "compromise_level": host.compromise_level,
                    "defense_state": host.defense_state,
                    "is_decoy": host.is_decoy,
                }
            )

        edges = [
            {
                "id": edge.edge_id,
                "source": edge.source,
                "target": edge.target,
                "visual_state": edge.visual_state,
                "status": edge.status,
            }
            for edge in self.edges
        ]
        return {"nodes": nodes, "edges": edges}

    def to_topology_init(self, scenario_id: str, total_steps: int, seed: int) -> dict:
        nodes = []
        for spec in NODE_REGISTRY:
            host = self.get_host(spec.id)
            nodes.append(
                {
                    "id": spec.id,
                    "type": spec.type,
                    "zone": spec.zone,
                    "label": spec.id.upper(),
                    "services": list(host.services),
                    "criticality": float(host.criticality),
                    "visual_state": "neutral",
                    "overlay": None,
                }
            )

        edges = [
            {
                "id": edge.edge_id,
                "source": edge.source,
                "target": edge.target,
                "visual_state": "normal" if edge.status == "active" else "blocked",
            }
            for edge in self.edges
        ]

        zones = [
            {
                "id": f"zone_{zone}",
                "label": ZONE_LABELS[zone],
                "member_ids": members,
            }
            for zone, members in ZONE_MEMBER_IDS.items()
            if zone in ZONE_LABELS
        ]

        return {
            "nodes": nodes,
            "edges": edges,
            "zones": zones,
            "scenario_id": scenario_id,
            "total_steps": int(total_steps),
            "seed": int(seed),
        }

    def node_features_index(self, host_id: str) -> int:
        return NODE_INDEX_BY_ID[host_id]


def host_visual_state(host: HostState) -> str:
    if host.defense_state == "isolated":
        return "isolated"
    if host.compromise_level >= 3:
        return "critical"
    if host.compromise_level >= 1:
        return "compromised"
    if host.defense_state == "patched":
        return "patched"
    if host.defense_state == "monitored":
        return "monitored"
    return "neutral"


def create_empty_network_state() -> NetworkState:
    hosts = [
        HostState(
            host_id=node_id,
            zone=next(spec.zone for spec in NODE_REGISTRY if spec.id == node_id),
            host_type=next(spec.type for spec in NODE_REGISTRY if spec.id == node_id),
            services=list(HOST_SERVICES[node_id]),
            vulnerabilities=[],
            criticality=HOST_CRITICALITY[node_id],
        )
        for node_id in NODE_REGISTRY_IDS
    ]
    return NetworkState(hosts=hosts, edges=[])
