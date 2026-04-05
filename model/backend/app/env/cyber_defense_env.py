from __future__ import annotations

import time
from typing import Any

import gymnasium as gym
import numpy as np
from gymnasium import spaces

from backend.app.env.action_logic import (
    apply_deception,
    apply_isolate,
    apply_monitor,
    apply_patch,
    apply_rotate_credentials,
    choose_block_connection_edge,
    get_valid_action_mask,
)
from backend.app.env.detection_model import compute_detection_probability
from backend.app.env.network_state import NetworkState, host_visual_state
from backend.app.env.red_policies import sample_scenario
from backend.app.env.registry import BLUE_ACTIONS, NODE_REGISTRY_IDS
from backend.app.env.reward import DEFAULT_REWARD_WEIGHTS, BlueAction, compute_episode_bonus, compute_reward
from backend.app.env.topology_generator import generate_chapman_topology
from backend.app.explainability.rationale import generate_explainability


class CyberDefenseEnv(gym.Env):
    """Single-agent blue-team environment with scripted red-team attacks."""

    metadata = {"render_modes": ["human", "json"]}

    def __init__(self, config: dict | None = None):
        super().__init__()
        self.config = config or {}
        self.max_steps = int(self.config.get("max_steps", 200))
        self.max_nodes = int(self.config.get("max_nodes", 28))

        self.observation_space = spaces.Dict(
            {
                "node_features": spaces.Box(0, 1, (self.max_nodes, 22), dtype=np.float32),
                "global_features": spaces.Box(0, 1, (6,), dtype=np.float32),
                "adjacency": spaces.Box(0, 1, (self.max_nodes, self.max_nodes), dtype=np.float32),
                "alert_history": spaces.Box(0, 1, (10, self.max_nodes), dtype=np.float32),
                "action_mask": spaces.Box(0, 1, (6, self.max_nodes), dtype=np.float32),
            }
        )
        self.action_space = spaces.MultiDiscrete([6, self.max_nodes])
        self.reward_weights = dict(DEFAULT_REWARD_WEIGHTS)
        raw_weights = self.config.get("reward_weights", {})
        if isinstance(raw_weights, dict):
            for key, value in raw_weights.items():
                if key not in self.reward_weights:
                    continue
                try:
                    self.reward_weights[key] = float(value)
                except (TypeError, ValueError):
                    continue

        self.network_state: NetworkState | None = None
        self.red_policy = None
        self.rng = np.random.default_rng()

        self.step_count = 0
        self.alert_history = np.zeros((10, self.max_nodes), dtype=np.float32)

        self.event_counter = 0
        self.det_counter = 0
        self.alert_tactic_counts: dict[str, int] = {}
        self.compromise_first_step: dict[str, int] = {}
        self.detection_latencies: list[int] = []
        self.trajectory: list[dict[str, Any]] = []

    def reset(self, seed: int | None = None, options: dict | None = None):
        super().reset(seed=seed)
        if seed is None:
            seed = int(time.time()) % 1_000_000
        self.rng = np.random.default_rng(seed)

        self.network_state = generate_chapman_topology(self.rng)

        scenario_weights = self.config.get("scenario_weights", [0.2] * 5)
        force_scenario = (options or {}).get("scenario_id") if options else self.config.get("scenario_id")
        self.red_policy = sample_scenario(self.rng, scenario_weights=scenario_weights, force_scenario=force_scenario)
        self.red_policy.reset(self.network_state)

        self.step_count = 0
        self.alert_history = np.zeros((10, self.max_nodes), dtype=np.float32)
        self.event_counter = 0
        self.det_counter = 0
        self.alert_tactic_counts = {}
        self.compromise_first_step = {}
        self.detection_latencies = []
        self.trajectory = []

        obs = self._get_observation()
        info = {
            "topology": self.network_state.to_topology_init(
                scenario_id=self.red_policy.scenario_id,
                total_steps=self.max_steps,
                seed=seed,
            ),
            "scenario_id": self.red_policy.scenario_id,
            "valid_action_mask": self._get_action_mask(),
        }
        return obs, info

    def step(self, action: np.ndarray | list[int] | tuple[int, int]):
        if self.network_state is None:
            raise RuntimeError("Environment not reset")

        action_type, target_idx = int(action[0]), int(action[1])
        target_idx = max(0, min(self.max_nodes - 1, target_idx))
        target_host = NODE_REGISTRY_IDS[target_idx]
        action_name = BLUE_ACTIONS[action_type]

        prev_state = self.network_state.clone()
        self.network_state.step = self.step_count + 1

        blue_event, topology_add = self._execute_blue_action(action_name, target_host)

        red_events = self.red_policy.step(self.network_state)
        if red_events:
            red_event = red_events[0]
            self._finalize_action_event(red_event)
            self._apply_red_action(red_event)
            self.alert_tactic_counts[red_event["data"]["mitre_tactic"]] = (
                self.alert_tactic_counts.get(red_event["data"]["mitre_tactic"], 0) + 1
            )
        else:
            red_event = None

        detection_event = self._update_detection_signals()
        self._refresh_network_scores()

        reward = compute_reward(
            self.network_state,
            BlueAction(type=action_name, target=target_host),
            prev_state,
            reward_weights=self.reward_weights,
        )

        self._update_alert_history()
        state_delta = self._compute_state_delta(prev_state, self.network_state)
        explainability = generate_explainability(
            obs=self._get_observation(),
            action=(action_type, target_idx),
            action_name=action_name,
            target_host=target_host,
            step=self.network_state.step,
        )
        explainability["ts_ms"] = int(time.time() * 1000)
        metrics_tick = self._build_metrics_tick()

        ordered_events: list[dict] = []
        if red_event is not None:
            ordered_events.append(red_event)
        ordered_events.append(blue_event)
        if topology_add is not None:
            ordered_events.append(topology_add)
        ordered_events.append({"type": "state_delta", "data": state_delta})
        if detection_event is not None:
            ordered_events.append(detection_event)
        ordered_events.append({"type": "explainability", "data": explainability})
        ordered_events.append({"type": "metrics_tick", "data": metrics_tick})

        self.step_count += 1
        terminated = self.step_count >= self.max_steps or self.network_state.all_hvts_compromised()
        truncated = False

        if terminated:
            reward += compute_episode_bonus(self.trajectory + [{"state": self.network_state.clone()}])

        self.trajectory.append({"state": self.network_state.clone(), "reward": reward})

        obs = self._get_observation()
        info = {
            "step": self.step_count,
            "events": ordered_events,
            "state_delta": state_delta,
            "metrics": metrics_tick,
            "valid_action_mask": self._get_action_mask(),
            "scenario_id": self.red_policy.scenario_id,
        }
        return obs, float(reward), terminated, truncated, info

    def _execute_blue_action(self, action_name: str, target_host: str) -> tuple[dict, dict | None]:
        assert self.network_state is not None
        host = self.network_state.get_host(target_host)
        outcome = "success"
        description = ""
        topology_add = None

        if action_name == "monitor_host":
            apply_monitor(host)
            description = f"Monitor {target_host} for suspicious behavior"
        elif action_name == "patch_service":
            if host.vulnerabilities and host.defense_state != "isolated":
                apply_patch(host)
                description = f"Patch service on {target_host}"
            else:
                outcome = "failure"
                description = f"Patch skipped on {target_host}; no patchable vuln"
        elif action_name == "isolate_host":
            apply_isolate(host, self.network_state)
            description = f"Isolate {target_host} from network"
        elif action_name == "block_connection":
            edge = choose_block_connection_edge(self.network_state, target_host)
            if edge is None:
                outcome = "failure"
                description = f"No active edge available to block for {target_host}"
            else:
                edge.status = "blocked"
                edge.visual_state = "blocked"
                edge.risk_score = 0.0
                description = f"Block edge {edge.edge_id} to sever attack path"
        elif action_name == "rotate_credentials":
            apply_rotate_credentials(host)
            description = f"Rotate credentials for {target_host}"
        elif action_name == "deploy_deception":
            decoy_id = apply_deception(self.network_state, host)
            if decoy_id is None:
                outcome = "failure"
                description = f"Deception node already exists for {target_host}"
            else:
                description = f"Deploy deception node near {target_host}"
                topology_add = {
                    "type": "topology_add_node",
                    "data": {
                        "node": {
                            "id": decoy_id,
                            "type": "infrastructure",
                            "zone": host.zone,
                            "label": decoy_id.upper(),
                            "services": host.services[:2],
                            "criticality": 0.0,
                            "visual_state": "neutral",
                            "overlay": None,
                            "is_decoy": True,
                        },
                        "edges": [
                            {
                                "id": f"{decoy_id}->{target_host}",
                                "source": decoy_id,
                                "target": target_host,
                                "visual_state": "normal",
                            }
                        ],
                    },
                }
        else:
            outcome = "failure"
            description = f"Unsupported blue action {action_name}"

        event = {
            "type": "action_event",
            "data": {
                "event_id": "",
                "ts_ms": int(time.time() * 1000),
                "step": self.network_state.step,
                "actor": "BLUE",
                "action_type": action_name,
                "source_host": target_host,
                "target_host": target_host,
                "target_service": None,
                "outcome": outcome,
                "mitre_tactic": "Defense Evasion",
                "confidence": round(float(0.7 + self.rng.random() * 0.29), 2),
                "description": description[:120],
                "severity": "low" if outcome == "success" else "info",
                "risk_score": round(float(host.criticality), 2),
            },
        }
        self._finalize_action_event(event)
        return event, topology_add

    def _apply_red_action(self, event: dict) -> None:
        assert self.network_state is not None
        data = event["data"]
        target_id = data.get("target_host")
        host = self.network_state.maybe_get_host(target_id)
        if host is None:
            return

        action_type = data["action_type"]
        signals = event.get("signals", {})
        if data.get("outcome") != "success":
            host.traffic_anomaly_score = min(1.0, host.traffic_anomaly_score + 0.05)
            return

        host.traffic_anomaly_score = min(1.0, host.traffic_anomaly_score + float(signals.get("traffic_anomaly", 0.2)))
        host.login_failure_count += int(round(float(signals.get("login_failure_rate", 0.0)) * 10))
        host.outbound_data_volume = min(1.0, host.outbound_data_volume + float(signals.get("outbound_data_volume", 0.0)))
        host.lateral_connection_count += int(round(float(signals.get("new_connections", 0.0)) * 10))

        if action_type == "scan_host":
            self._mark_related_edges(host.host_id, "scanning", 0.25)
        elif action_type == "enumerate_service":
            self._mark_related_edges(host.host_id, "scanning", 0.35)
        elif action_type == "exploit_vulnerability":
            host.compromise_level = max(1, host.compromise_level)
            if host.compromise_step < 0:
                host.compromise_step = self.network_state.step
                self.compromise_first_step.setdefault(host.host_id, self.network_state.step)
            self._mark_related_edges(host.host_id, "credential_flow", 0.55)
            if action_type == "exploit_vulnerability" and "credential" in data["description"].lower():
                host.credential_compromised = True
        elif action_type == "lateral_move":
            host.compromise_level = max(2, host.compromise_level)
            if host.compromise_step < 0:
                host.compromise_step = self.network_state.step
                self.compromise_first_step.setdefault(host.host_id, self.network_state.step)
            self._mark_related_edges(host.host_id, "lateral_movement", 0.75)
        elif action_type == "privilege_escalate":
            host.compromise_level = max(3, host.compromise_level)
            host.has_backdoor = True
        elif action_type == "exfiltrate_data":
            host.compromise_level = 4
            host.outbound_data_volume = max(host.outbound_data_volume, 0.9)
            self._mark_related_edges(host.host_id, "exfiltration", 0.95)

    def _mark_related_edges(self, host_id: str, visual_state: str, risk_score: float) -> None:
        assert self.network_state is not None
        for edge in self.network_state.edges:
            if edge.status != "active":
                continue
            if edge.source == host_id or edge.target == host_id:
                edge.visual_state = visual_state
                edge.risk_score = max(edge.risk_score, risk_score)

    def _update_detection_signals(self) -> dict | None:
        assert self.network_state is not None

        detection_candidates: list[tuple[float, dict]] = []
        for host in self.network_state.hosts:
            if host.zone == "external":
                continue

            monitoring = host.defense_state == "monitored"
            p_detect = compute_detection_probability(host, monitoring)
            host.time_since_last_alert += 1

            if host.compromise_level > 0 and not host.is_compromised_detected and self.rng.random() < p_detect:
                host.is_compromised_detected = True
                host.time_since_last_alert = 0

                if host.host_id in self.compromise_first_step:
                    self.detection_latencies.append(self.network_state.step - self.compromise_first_step[host.host_id])

                self.det_counter += 1
                detection_candidates.append(
                    (
                        p_detect,
                        {
                            "type": "detection_event",
                            "data": {
                                "event_id": f"det_{self.det_counter:06d}",
                                "ts_ms": int(time.time() * 1000),
                                "step": self.network_state.step,
                                "detector": "BLUE",
                                "target_host": host.host_id,
                                "signal": "traffic_spike",
                                "severity": "high" if host.compromise_level >= 2 else "medium",
                                "detected": True,
                                "mitre_tactic": "Lateral Movement" if host.compromise_level >= 2 else "Reconnaissance",
                            },
                        },
                    )
                )

        if not detection_candidates:
            return None

        detection_candidates.sort(key=lambda x: x[0], reverse=True)
        return detection_candidates[0][1]

    def _refresh_network_scores(self) -> None:
        assert self.network_state is not None
        compromised = sum(1 for h in self.network_state.hosts if h.compromise_level > 0 and not h.is_decoy)
        total_real = sum(1 for h in self.network_state.hosts if not h.is_decoy)
        isolated = sum(1 for h in self.network_state.hosts if h.defense_state == "isolated")

        self.network_state.attack_pressure_score = min(1.0, compromised / max(1, total_real))
        availability_penalty = isolated / max(1, total_real)
        compromised_penalty = sum(h.compromise_level >= 3 for h in self.network_state.hosts) / max(1, total_real)
        self.network_state.service_availability_score = max(0.0, 1.0 - 0.5 * availability_penalty - 0.4 * compromised_penalty)
        self.network_state.blue_action_budget_remaining = max(0.0, 1.0 - (self.step_count / max(1, self.max_steps)))

    def _compute_state_delta(self, prev_state: NetworkState, state: NetworkState) -> dict:
        node_changes = []
        for host in state.hosts:
            prev = prev_state.maybe_get_host(host.host_id)
            visual = host_visual_state(host)
            overlay = "monitored" if host.defense_state == "monitored" else None

            if prev is None:
                changed = True
                prev_visual = None
                prev_overlay = None
                prev_compromise = None
                prev_defense = None
            else:
                prev_visual = host_visual_state(prev)
                prev_overlay = "monitored" if prev.defense_state == "monitored" else None
                prev_compromise = prev.compromise_level
                prev_defense = prev.defense_state
                changed = (
                    visual != prev_visual
                    or overlay != prev_overlay
                    or host.compromise_level != prev.compromise_level
                    or host.defense_state != prev.defense_state
                )

            if changed:
                node_changes.append(
                    {
                        "node_id": host.host_id,
                        "visual_state": visual,
                        "overlay": overlay,
                        "compromise_level": host.compromise_level,
                        "defense_state": host.defense_state,
                    }
                )

        prev_edges = {edge.edge_id: edge for edge in prev_state.edges}
        edge_changes = []
        for edge in state.edges:
            prev_edge = prev_edges.get(edge.edge_id)
            if prev_edge is None or prev_edge.visual_state != edge.visual_state or prev_edge.status != edge.status:
                edge_changes.append(
                    {
                        "edge_id": edge.edge_id,
                        "visual_state": edge.visual_state,
                        "direction": edge.direction,
                    }
                )

        return {
            "ts_ms": int(time.time() * 1000),
            "step": state.step,
            "node_changes": node_changes,
            "edge_changes": edge_changes,
        }

    def _build_metrics_tick(self) -> dict:
        assert self.network_state is not None
        compromised = [host for host in self.network_state.hosts if host.compromise_level > 0 and not host.is_decoy]
        detected = [host for host in compromised if host.is_compromised_detected]
        isolated = [host for host in self.network_state.hosts if host.defense_state == "isolated"]

        total_alerts = sum(self.alert_tactic_counts.values())
        alert_classification = {}
        for tactic, count in self.alert_tactic_counts.items():
            alert_classification[tactic] = {
                "count": count,
                "percentage": round(count / total_alerts, 2) if total_alerts else 0.0,
            }

        hot_targets = sorted(
            [
                {
                    "node_id": host.host_id,
                    "hit_count": int(host.compromise_level + host.lateral_connection_count / 2),
                }
                for host in self.network_state.hosts
                if host.host_id in NODE_REGISTRY_IDS and host.compromise_level > 0
            ],
            key=lambda x: x["hit_count"],
            reverse=True,
        )[:5]

        return {
            "step": self.network_state.step,
            "attack_pressure": round(float(self.network_state.attack_pressure_score), 4),
            "containment_pressure": round(float(len(isolated) / max(1, len(self.network_state.hosts))), 4),
            "service_availability": round(float(self.network_state.service_availability_score), 4),
            "open_incidents": len(compromised),
            "contained_incidents": len([h for h in detected if h.defense_state == "isolated"]),
            "red_actions_total": self.event_counter,
            "blue_actions_total": self.step_count,
            "blue_reward_cumulative": round(float(sum(item["reward"] for item in self.trajectory)), 4),
            "red_score_cumulative": round(float(sum(h.compromise_level for h in compromised)), 4),
            "detection_latency_mean": round(float(np.mean(self.detection_latencies)), 4) if self.detection_latencies else 0.0,
            "hot_targets": hot_targets,
            "alert_classification": alert_classification,
        }

    def _update_alert_history(self) -> None:
        assert self.network_state is not None
        self.alert_history = np.roll(self.alert_history, shift=-1, axis=0)
        row = np.zeros((self.max_nodes,), dtype=np.float32)
        for idx, host_id in enumerate(NODE_REGISTRY_IDS[: self.max_nodes]):
            host = self.network_state.get_host(host_id)
            row[idx] = np.float32(min(1.0, host.traffic_anomaly_score))
        self.alert_history[-1, :] = row

    def _get_observation(self) -> dict[str, np.ndarray]:
        assert self.network_state is not None
        return {
            "node_features": self._encode_node_features(),
            "global_features": self._encode_global_features(),
            "adjacency": self._encode_adjacency(),
            "alert_history": self.alert_history.copy(),
            "action_mask": self._get_action_mask().astype(np.float32),
        }

    def _encode_node_features(self) -> np.ndarray:
        assert self.network_state is not None
        out = np.zeros((self.max_nodes, 22), dtype=np.float32)

        for idx, host_id in enumerate(NODE_REGISTRY_IDS[: self.max_nodes]):
            host = self.network_state.get_host(host_id)
            host_type = {
                "endpoint": [1, 0, 0, 0],
                "infrastructure": [0, 1, 0, 0],
                "hvt": [0, 0, 1, 0],
                "iot": [0, 0, 0, 1],
            }.get(host.host_type, [0, 0, 0, 0])
            zone = {
                "perimeter": [1, 0, 0, 0],
                "campus": [0, 1, 0, 0],
                "admin": [0, 0, 1, 0],
                "research": [0, 0, 0, 1],
            }.get(host.zone, [0, 0, 0, 0])
            defense = {
                "none": [1, 0, 0, 0],
                "monitored": [0, 1, 0, 0],
                "patched": [0, 0, 1, 0],
                "isolated": [0, 0, 0, 1],
            }.get(host.defense_state, [1, 0, 0, 0])
            neighbors_compromised = sum(1 for neighbor in self.network_state.iter_neighbors(host.host_id) if neighbor.compromise_level > 0)

            features = (
                host_type
                + zone
                + [
                    float(host.criticality),
                    min(1.0, len(host.services) / 6.0),
                    min(1.0, len(host.vulnerabilities) / 4.0),
                ]
                + defense
                + [
                    min(1.0, host.traffic_anomaly_score),
                    min(1.0, host.login_failure_count / 10.0),
                    min(1.0, host.outbound_data_volume),
                    min(1.0, host.lateral_connection_count / 10.0),
                    1.0 if host.is_compromised_detected else 0.0,
                    min(1.0, host.time_since_last_alert / max(1, self.max_steps)),
                    min(1.0, neighbors_compromised / 8.0),
                ]
            )
            out[idx, :] = np.array(features, dtype=np.float32)

        return out

    def _encode_global_features(self) -> np.ndarray:
        assert self.network_state is not None
        detected = sum(1 for host in self.network_state.hosts if host.is_compromised_detected)
        isolated = sum(1 for host in self.network_state.hosts if host.defense_state == "isolated")

        return np.array(
            [
                self.step_count / max(1, self.max_steps),
                detected / max(1, self.max_nodes),
                isolated / max(1, self.max_nodes),
                float(self.network_state.attack_pressure_score),
                float(self.network_state.service_availability_score),
                float(self.network_state.blue_action_budget_remaining),
            ],
            dtype=np.float32,
        )

    def _encode_adjacency(self) -> np.ndarray:
        assert self.network_state is not None
        mat = np.zeros((self.max_nodes, self.max_nodes), dtype=np.float32)
        index = {node_id: i for i, node_id in enumerate(NODE_REGISTRY_IDS[: self.max_nodes])}

        for edge in self.network_state.edges:
            if edge.source not in index or edge.target not in index:
                continue
            val = 1.0 if edge.status == "active" else 0.5
            mat[index[edge.source], index[edge.target]] = val
        return mat

    def _get_action_mask(self) -> np.ndarray:
        assert self.network_state is not None
        return get_valid_action_mask(self.network_state, max_nodes=self.max_nodes)

    def _finalize_action_event(self, event: dict) -> None:
        self.event_counter += 1
        event["data"]["event_id"] = f"evt_{self.event_counter:06d}"
        event["data"]["ts_ms"] = int(time.time() * 1000)
        event["data"]["step"] = self.network_state.step if self.network_state is not None else self.step_count
