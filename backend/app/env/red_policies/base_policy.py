from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from backend.app.env.network_state import NetworkState


@dataclass(frozen=True)
class AttackStage:
    action: str
    targets: list[str]
    mitre: str
    description: str
    severity: str = "medium"


class RedPolicy:
    scenario_id: str = "base"
    kill_chain: list[AttackStage] = []

    def __init__(
        self,
        entry_point_jitter: bool = True,
        step_delay_range: tuple[int, int] = (1, 3),
        exploit_failure_prob: float = 0.15,
        target_shuffle: bool = True,
        rng: np.random.Generator | None = None,
    ) -> None:
        self.entry_point_jitter = entry_point_jitter
        self.step_delay_range = step_delay_range
        self.exploit_failure_prob = exploit_failure_prob
        self.target_shuffle = target_shuffle
        self.rng = rng or np.random.default_rng()

        self._stage_index = 0
        self._delay_steps = 0

    def reset(self, state: NetworkState) -> None:
        self._stage_index = 0
        self._delay_steps = 0

    def _pick_target(self, stage: AttackStage, state: NetworkState) -> str:
        if stage.targets == ["ALL_COMPROMISED"]:
            compromised = [h.host_id for h in state.hosts if h.compromise_level > 0 and not h.is_decoy]
            return compromised[0] if compromised else "auth_server"

        choices = [target for target in stage.targets if state.maybe_get_host(target) is not None]
        if not choices:
            return "auth_server"
        if self.target_shuffle:
            return str(self.rng.choice(choices))
        return choices[0]

    def step(self, state: NetworkState) -> list[dict]:
        if self._stage_index >= len(self.kill_chain):
            return []

        if self._delay_steps > 0:
            self._delay_steps -= 1
            return []

        stage = self.kill_chain[self._stage_index]
        target = self._pick_target(stage, state)

        outcome = "success"
        if stage.action in {"exploit_vulnerability", "lateral_move"} and self.rng.random() < self.exploit_failure_prob:
            outcome = "failure"

        event = {
            "type": "action_event",
            "data": {
                "event_id": "",
                "ts_ms": 0,
                "step": state.step,
                "actor": "RED",
                "action_type": stage.action,
                "source_host": self._source_host_for_stage(stage, target),
                "target_host": target,
                "target_service": None,
                "outcome": outcome,
                "mitre_tactic": stage.mitre,
                "confidence": round(float(self.rng.uniform(0.68, 0.97)), 2),
                "description": stage.description[:120],
                "severity": stage.severity,
                "risk_score": round(float(self.rng.uniform(0.4, 0.95)), 2),
            },
            "signals": self._signals_for_action(stage.action),
        }

        if outcome == "success":
            self._stage_index += 1
        self._delay_steps = int(self.rng.integers(self.step_delay_range[0], self.step_delay_range[1] + 1))

        return [event]

    @staticmethod
    def _source_host_for_stage(stage: AttackStage, target: str) -> str:
        if stage.action == "scan_host":
            return "internet"
        if stage.action in {"lateral_move", "privilege_escalate", "exfiltrate_data"}:
            return target
        return "internet"

    @staticmethod
    def _signals_for_action(action_type: str) -> dict[str, float]:
        if action_type == "scan_host":
            return {"traffic_anomaly": 0.2, "new_connections": 0.2}
        if action_type == "enumerate_service":
            return {"traffic_anomaly": 0.3}
        if action_type == "exploit_vulnerability":
            return {"traffic_anomaly": 0.5, "login_failure_rate": 0.3}
        if action_type == "lateral_move":
            return {"traffic_anomaly": 0.6, "new_connections": 0.7}
        if action_type == "privilege_escalate":
            return {"traffic_anomaly": 0.4}
        if action_type == "exfiltrate_data":
            return {"traffic_anomaly": 0.8, "outbound_data_volume": 0.9}
        return {}
