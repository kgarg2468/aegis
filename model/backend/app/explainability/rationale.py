from __future__ import annotations

import numpy as np


FEATURE_NAMES = [
    "host_type",
    "zone",
    "criticality",
    "num_services",
    "num_vulnerabilities",
    "defense_state",
    "traffic_anomaly_score",
    "login_failure_rate",
    "outbound_data_volume",
    "new_connection_count",
    "is_compromised_detected",
    "time_since_last_alert",
    "neighbor_compromise_count",
]

EFFECT_MAP = {
    "monitor_host": "increase detection coverage",
    "patch_service": "remove exploitation path",
    "isolate_host": "contain lateral spread",
    "block_connection": "sever attack path",
    "rotate_credentials": "invalidate stolen credentials",
    "deploy_deception": "create detection honeypot",
}


def generate_explainability(
    obs: dict,
    action: tuple[int, int],
    action_name: str,
    target_host: str,
    step: int,
) -> dict:
    _, target_idx = action
    node_features = np.asarray(obs["node_features"][target_idx])

    reasons: list[dict[str, float | str]] = []
    if node_features[16] > 0.5:
        reasons.append({"name": "traffic_spike_ratio", "value": round(float(node_features[16]) * 5.0, 2)})
    if node_features[21] > 0:
        reasons.append({"name": "lateral_movement_pattern_match", "value": round(float(node_features[21]), 2)})
    if node_features[8] > 0.7:
        reasons.append({"name": "critical_asset_risk", "value": round(float(node_features[8]), 2)})
    if node_features[18] > 0.3:
        reasons.append({"name": "exfiltration_indicator", "value": round(float(node_features[18]) * 3.0, 2)})

    if not reasons:
        signal_slice = node_features[16:22]
        strongest = int(np.argmax(signal_slice))
        reasons.append(
            {
                "name": FEATURE_NAMES[min(6 + strongest, len(FEATURE_NAMES) - 1)],
                "value": round(float(signal_slice[strongest]), 2),
            }
        )

    confidence = float(np.clip(0.55 + np.max(node_features[16:22]) * 0.4, 0.55, 0.98))

    return {
        "ts_ms": None,
        "step": int(step),
        "action": action_name,
        "target_host": target_host,
        "confidence": round(confidence, 2),
        "reason_features": sorted(reasons, key=lambda item: float(item["value"]), reverse=True)[:3],
        "expected_effect": EFFECT_MAP[action_name],
    }
