from __future__ import annotations

from backend.app.env.network_state import HostState


def compute_detection_probability(host: HostState, blue_monitoring: bool) -> float:
    base_detection = 0.0

    base_detection += float(host.traffic_anomaly_score) * 0.3

    if blue_monitoring:
        base_detection += 0.4

    base_detection += float(host.compromise_level) * 0.1

    if host.outbound_data_volume > 0.7:
        base_detection += 0.3

    if host.credential_compromised and host.compromise_level < 2:
        base_detection *= 0.5

    return max(0.0, min(base_detection, 0.95))
