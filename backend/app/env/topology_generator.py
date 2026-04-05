from __future__ import annotations

import numpy as np

from backend.app.env.network_state import EdgeState, HostState, NetworkState
from backend.app.env.registry import HOST_CRITICALITY, HOST_SERVICES, NODE_REGISTRY
from backend.app.env.vulnerability_catalog import vulnerabilities_for_services

CONNECTIVITY_RULES: dict[tuple[str, str], dict[str, float | bool]] = {
    ("external", "perimeter"): {"probability": 1.0, "bidirectional": False},
    ("perimeter", "perimeter"): {"probability": 0.6, "bidirectional": True},
    ("perimeter", "campus"): {"probability": 0.3, "bidirectional": True},
    ("campus", "campus"): {"probability": 0.4, "bidirectional": True},
    ("campus", "admin"): {"probability": 0.15, "bidirectional": False},
    ("campus", "research"): {"probability": 0.2, "bidirectional": True},
    ("admin", "admin"): {"probability": 0.8, "bidirectional": True},
    ("research", "research"): {"probability": 0.7, "bidirectional": True},
    ("admin", "research"): {"probability": 0.3, "bidirectional": True},
}

FORCED_EDGES: list[tuple[str, str]] = [
    ("internet", "vpn_gateway"),
    ("internet", "web_portal"),
    ("internet", "dns_server"),
    ("eduroam_ap_01", "auth_server"),
    ("eduroam_ap_02", "auth_server"),
    ("eduroam_ap_03", "auth_server"),
    ("auth_server", "active_directory"),
    ("active_directory", "sis_server"),
    ("active_directory", "finance_server"),
    ("faculty_device_01", "research_server_01"),
    ("faculty_device_02", "research_server_01"),
    ("faculty_device_03", "research_server_02"),
    ("print_server", "student_device_01"),
    ("print_server", "faculty_device_01"),
    ("print_server", "lab_workstation_01"),
]


def _sample_host_vulnerabilities(host_id: str, rng: np.random.Generator) -> list[str]:
    services = HOST_SERVICES[host_id]
    candidates = vulnerabilities_for_services(services)
    if not candidates:
        return []
    max_sample = min(3, len(candidates))
    sample_size = int(rng.integers(1, max_sample + 1))
    chosen = rng.choice(candidates, size=sample_size, replace=False)
    return [entry["id"] for entry in chosen.tolist()]


def _add_edge(edges: list[EdgeState], seen: set[tuple[str, str]], source: str, target: str) -> None:
    if source == target:
        return
    key = (source, target)
    if key in seen:
        return
    seen.add(key)
    edges.append(EdgeState(source=source, target=target, status="active", visual_state="normal", risk_score=0.0))


def generate_chapman_topology(rng: np.random.Generator) -> NetworkState:
    hosts: list[HostState] = []
    for spec in NODE_REGISTRY:
        hosts.append(
            HostState(
                host_id=spec.id,
                zone=spec.zone,
                host_type=spec.type,
                services=list(HOST_SERVICES[spec.id]),
                vulnerabilities=_sample_host_vulnerabilities(spec.id, rng),
                criticality=HOST_CRITICALITY[spec.id],
                traffic_anomaly_score=0.0 if spec.id == "internet" else float(rng.uniform(0.0, 0.05)),
                outbound_data_volume=float(rng.uniform(0.0, 0.1)),
            )
        )

    edges: list[EdgeState] = []
    seen: set[tuple[str, str]] = set()

    for source, target in FORCED_EDGES:
        _add_edge(edges, seen, source, target)

    for src in hosts:
        for dst in hosts:
            if src.host_id == dst.host_id:
                continue
            rule = CONNECTIVITY_RULES.get((src.zone, dst.zone))
            if rule is None:
                continue
            if rng.random() <= float(rule["probability"]):
                _add_edge(edges, seen, src.host_id, dst.host_id)
                if bool(rule["bidirectional"]):
                    _add_edge(edges, seen, dst.host_id, src.host_id)

    for host in hosts:
        if host.host_id == "print_server":
            continue
        if host.zone in {"campus", "admin"} and rng.random() < 0.6:
            _add_edge(edges, seen, "print_server", host.host_id)

    state = NetworkState(hosts=hosts, edges=edges)
    return state
