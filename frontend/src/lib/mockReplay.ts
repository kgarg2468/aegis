import {
  MITRE_TACTICS,
  type ActionEvent,
  type ActionType,
  type DetectionEvent,
  type EdgeVisualState,
  type EpisodeEnd,
  type ExplainabilityRecord,
  type MetricsTick,
  type MitreTactic,
  type NodeVisualState,
  type Severity,
  type StateDelta,
  type TopologyAddNodeData,
  type TopologyEdge,
  type TopologyNode,
  type WSMessage,
  createCanonicalTopologyInit,
} from "./integrationContract";

export interface GeneratedReplay {
  replayId: string;
  scenarioId: string;
  totalSteps: number;
  messages: WSMessage[];
}

export interface GenerateReplayOptions {
  replayId?: string;
  scenarioId?: string;
  totalSteps?: number;
  seed?: number;
  includeBlue?: boolean;
}

type RuntimeNodeState = {
  visual: NodeVisualState;
  overlay: "monitored" | null;
  defenseState: "none" | "monitored" | "isolated" | "patched";
  compromiseLevel: number;
};

const DEFAULT_OPTIONS: Required<GenerateReplayOptions> = {
  replayId: "mock_replay_01",
  scenarioId: "faculty_phish",
  totalSteps: 200,
  seed: 1003,
  includeBlue: true,
};

const ATTACK_PATH = [
  "faculty_device_02",
  "auth_server",
  "active_directory",
  "research_server_01",
  "shared_storage",
  "dns_server",
  "internet",
] as const;

function clamp(value: number, min: number, max: number): number {
  return Math.max(min, Math.min(max, value));
}

function createRng(seed: number): () => number {
  let state = seed >>> 0;
  return () => {
    state = (state * 1664525 + 1013904223) >>> 0;
    return state / 0xffffffff;
  };
}

function pick<T>(rng: () => number, values: readonly T[]): T {
  return values[Math.floor(rng() * values.length) % values.length] as T;
}

function eventId(prefix: "evt" | "det", index: number): string {
  return `${prefix}_${index.toString().padStart(6, "0")}`;
}

function nodeCompromiseLevel(state: NodeVisualState): number {
  if (state === "critical") return 3;
  if (state === "compromised") return 2;
  if (state === "probed") return 1;
  return 0;
}

function edgeDirection(source: string, target: string, edgeId: string): "forward" | "reverse" {
  return edgeId.startsWith(`${source}->${target}`) ? "forward" : "reverse";
}

function defaultMetrics(step: number): MetricsTick {
  const emptyClassification: MetricsTick["alert_classification"] = Object.fromEntries(
    MITRE_TACTICS.map((tactic) => [tactic, { count: 0, percentage: 0 }]),
  );

  return {
    step,
    attack_pressure: 0,
    containment_pressure: 0,
    service_availability: 1,
    open_incidents: 0,
    contained_incidents: 0,
    red_actions_total: 0,
    blue_actions_total: 0,
    blue_reward_cumulative: 0,
    red_score_cumulative: 0,
    detection_latency_mean: 0,
    hot_targets: [],
    alert_classification: emptyClassification,
  };
}

function makeBlueActionType(step: number, rng: () => number): Extract<ActionType, "monitor_host" | "patch_service" | "isolate_host" | "block_connection" | "rotate_credentials" | "deploy_deception"> {
  if (step === 72) return "deploy_deception";
  if (step % 8 === 0) return "isolate_host";
  if (step % 5 === 0) return "patch_service";
  if (step % 6 === 0) return "block_connection";
  if (step % 11 === 0) return "rotate_credentials";
  return rng() > 0.45 ? "monitor_host" : "patch_service";
}

function tacticForAction(actionType: ActionType): MitreTactic {
  switch (actionType) {
    case "scan_host":
      return "Reconnaissance";
    case "enumerate_service":
      return "Initial Access";
    case "exploit_vulnerability":
      return "Execution";
    case "lateral_move":
      return "Lateral Movement";
    case "privilege_escalate":
      return "Credential Access";
    case "exfiltrate_data":
      return "Exfiltration";
    case "monitor_host":
      return "Defense Evasion";
    case "patch_service":
      return "Impact";
    case "isolate_host":
      return "Impact";
    case "block_connection":
      return "Defense Evasion";
    case "rotate_credentials":
      return "Credential Access";
    case "deploy_deception":
      return "Defense Evasion";
    default:
      return "Execution";
  }
}

function severityForAction(actionType: ActionType, success: boolean): Severity {
  if (!success) return "low";
  if (actionType === "exfiltrate_data" || actionType === "privilege_escalate") return "critical";
  if (actionType === "lateral_move" || actionType === "exploit_vulnerability") return "high";
  if (actionType === "scan_host" || actionType === "enumerate_service") return "medium";
  if (actionType === "isolate_host" || actionType === "block_connection") return "medium";
  return "low";
}

function edgeStateForAction(actionType: ActionType): EdgeVisualState {
  switch (actionType) {
    case "scan_host":
    case "enumerate_service":
      return "scanning";
    case "lateral_move":
      return "lateral_movement";
    case "privilege_escalate":
      return "credential_flow";
    case "exfiltrate_data":
      return "exfiltration";
    case "block_connection":
    case "isolate_host":
      return "blocked";
    default:
      return "normal";
  }
}

function appendNodeChange(
  changes: StateDelta["node_changes"],
  nodeId: string,
  current: Map<string, RuntimeNodeState>,
  nextVisual: NodeVisualState,
  nextOverlay: "monitored" | null,
  nextDefense: RuntimeNodeState["defenseState"],
): void {
  const prior = current.get(nodeId) ?? {
    visual: "neutral",
    overlay: null,
    defenseState: "none",
    compromiseLevel: 0,
  };

  const nextCompromise = nodeCompromiseLevel(nextVisual);
  const changed =
    prior.visual !== nextVisual ||
    prior.overlay !== nextOverlay ||
    prior.defenseState !== nextDefense ||
    prior.compromiseLevel !== nextCompromise;

  if (!changed) return;

  current.set(nodeId, {
    visual: nextVisual,
    overlay: nextOverlay,
    defenseState: nextDefense,
    compromiseLevel: nextCompromise,
  });

  changes.push({
    node_id: nodeId,
    visual_state: nextVisual,
    overlay: nextOverlay,
    compromise_level: nextCompromise,
    defense_state: nextDefense,
  });
}

function appendEdgeChange(
  changes: StateDelta["edge_changes"],
  edgeState: Map<string, EdgeVisualState>,
  edgeId: string,
  nextVisual: EdgeVisualState,
  direction: "forward" | "reverse",
): void {
  const prior = edgeState.get(edgeId) ?? "normal";
  if (prior === nextVisual) return;

  edgeState.set(edgeId, nextVisual);
  changes.push({
    edge_id: edgeId,
    visual_state: nextVisual,
    direction,
  });
}

function toDescription(actionType: ActionType, source: string, target: string): string {
  switch (actionType) {
    case "scan_host":
      return `Recon scan from ${source} to ${target}`;
    case "enumerate_service":
      return `Service enumeration on ${target} from ${source}`;
    case "exploit_vulnerability":
      return `Exploit attempt from ${source} into ${target}`;
    case "lateral_move":
      return `Lateral move from ${source} to ${target}`;
    case "privilege_escalate":
      return `Privilege escalation attempt on ${target}`;
    case "exfiltrate_data":
      return `Data exfiltration path opened from ${source} to ${target}`;
    case "monitor_host":
      return `Blue monitoring increased on ${target}`;
    case "patch_service":
      return `Blue patched vulnerable service on ${target}`;
    case "isolate_host":
      return `Blue isolated ${target} from surrounding links`;
    case "block_connection":
      return `Blue blocked route between ${source} and ${target}`;
    case "rotate_credentials":
      return `Blue rotated credentials impacting ${target}`;
    case "deploy_deception":
      return `Blue deployed deception adjacent to ${target}`;
    default:
      return `${actionType} executed on ${target}`;
  }
}

function createDecoyNode(parentNode: string): TopologyNode {
  return {
    id: `decoy_${parentNode}`,
    type: "infrastructure",
    zone: "admin",
    label: `DECOY ${parentNode.replaceAll("_", " ").toUpperCase()}`,
    services: ["ldap", "radius"],
    criticality: 0,
    visual_state: "neutral",
    overlay: null,
    is_decoy: true,
  };
}

export function generateMockReplay(options: GenerateReplayOptions = {}): GeneratedReplay {
  const config = { ...DEFAULT_OPTIONS, ...options };
  const rng = createRng(config.seed);
  const topology = createCanonicalTopologyInit(config.scenarioId, config.totalSteps, config.seed);
  const messages: WSMessage[] = [{ type: "topology_init", data: topology }];

  const nodeState = new Map<string, RuntimeNodeState>();
  for (const node of topology.nodes) {
    nodeState.set(node.id, {
      visual: node.visual_state,
      overlay: node.overlay,
      defenseState: "none",
      compromiseLevel: 0,
    });
  }

  const edgeState = new Map<string, EdgeVisualState>();
  for (const edge of topology.edges) edgeState.set(edge.id, edge.visual_state);

  let eventCounter = 1;
  let detectionCounter = 1;
  let redActionsTotal = 0;
  let blueActionsTotal = 0;
  let blueReward = 0;
  let redScore = 0;
  const hotTargetHits = new Map<string, number>();
  const tacticCounts = new Map<MitreTactic, number>();
  const compromised = new Set<string>();
  const isolated = new Set<string>();
  const patched = new Set<string>();
  const detections: number[] = [];
  let decoyAdded = false;

  const now = Date.now();

  for (let step = 1; step <= config.totalSteps; step += 1) {
    const phase = step / config.totalSteps;
    const pathIndex = Math.min(Math.floor(phase * (ATTACK_PATH.length - 1)), ATTACK_PATH.length - 2);
    const sourceHost = ATTACK_PATH[pathIndex];
    const targetHost = ATTACK_PATH[pathIndex + 1];

    const redActionType: ActionType =
      phase < 0.18
        ? (pick(rng, ["scan_host", "enumerate_service"] as const) as ActionType)
        : phase < 0.42
          ? "exploit_vulnerability"
          : phase < 0.74
            ? "lateral_move"
            : phase < 0.9
              ? "privilege_escalate"
              : "exfiltrate_data";

    const redSuccess = rng() > (redActionType === "exploit_vulnerability" ? 0.22 : 0.12);
    const redOutcome = redSuccess ? "success" : (rng() > 0.5 ? "failure" : "blocked");
    const redRisk = clamp(0.34 + phase * 0.58 + rng() * 0.12, 0, 1);

    const redAction: ActionEvent = {
      event_id: eventId("evt", eventCounter),
      ts_ms: now + step * 1000,
      step,
      actor: "RED",
      action_type: redActionType,
      source_host: sourceHost,
      target_host: targetHost,
      target_service: redActionType === "scan_host" ? "network" : "kerberos",
      outcome: redOutcome,
      mitre_tactic: tacticForAction(redActionType),
      confidence: clamp(0.58 + rng() * 0.35, 0, 1),
      description: toDescription(redActionType, sourceHost, targetHost).slice(0, 118),
      severity: severityForAction(redActionType, redSuccess),
      risk_score: redRisk,
    };
    messages.push({ type: "action_event", data: redAction });
    eventCounter += 1;
    redActionsTotal += 1;
    tacticCounts.set(redAction.mitre_tactic, (tacticCounts.get(redAction.mitre_tactic) ?? 0) + 1);
    hotTargetHits.set(targetHost, (hotTargetHits.get(targetHost) ?? 0) + 1);

    let blueActionType = makeBlueActionType(step, rng);
    if (!config.includeBlue) blueActionType = "monitor_host";

    const blueTarget = redActionType === "exfiltrate_data" ? sourceHost : targetHost;
    const blueSource = sourceHost;
    const blueSuccess = config.includeBlue ? rng() > 0.08 : false;

    const blueAction: ActionEvent = {
      event_id: eventId("evt", eventCounter),
      ts_ms: now + step * 1000 + 125,
      step,
      actor: "BLUE",
      action_type: blueActionType,
      source_host: blueSource,
      target_host: blueTarget,
      target_service: blueActionType === "patch_service" ? "ssh" : undefined,
      outcome: blueSuccess ? "success" : "partial",
      mitre_tactic: tacticForAction(blueActionType),
      confidence: clamp(0.55 + rng() * 0.35, 0, 1),
      description: toDescription(blueActionType, blueSource, blueTarget).slice(0, 118),
      severity: severityForAction(blueActionType, blueSuccess),
      risk_score: clamp(0.18 + (1 - redRisk) * 0.65, 0, 1),
    };
    messages.push({ type: "action_event", data: blueAction });
    eventCounter += 1;
    blueActionsTotal += 1;

    const nodeChanges: StateDelta["node_changes"] = [];
    const edgeChanges: StateDelta["edge_changes"] = [];

    if (redSuccess) {
      if (redActionType === "scan_host" || redActionType === "enumerate_service") {
        appendNodeChange(nodeChanges, targetHost, nodeState, "probed", null, "none");
      } else if (redActionType === "exploit_vulnerability" || redActionType === "lateral_move") {
        compromised.add(targetHost);
        appendNodeChange(nodeChanges, targetHost, nodeState, "compromised", null, "none");
      } else if (redActionType === "privilege_escalate" || redActionType === "exfiltrate_data") {
        compromised.add(targetHost);
        appendNodeChange(nodeChanges, targetHost, nodeState, "critical", null, "none");
      }
    }

    if (blueSuccess && config.includeBlue) {
      if (blueActionType === "monitor_host") {
        const current = nodeState.get(blueTarget);
        appendNodeChange(
          nodeChanges,
          blueTarget,
          nodeState,
          current?.visual ?? "neutral",
          "monitored",
          "monitored",
        );
      }

      if (blueActionType === "patch_service") {
        patched.add(blueTarget);
        compromised.delete(blueTarget);
        appendNodeChange(nodeChanges, blueTarget, nodeState, "patched", null, "patched");
      }

      if (blueActionType === "isolate_host") {
        isolated.add(blueTarget);
        compromised.delete(blueTarget);
        appendNodeChange(nodeChanges, blueTarget, nodeState, "isolated", null, "isolated");
      }
    }

    const activeEdgeId = `${sourceHost}->${targetHost}`;
    appendEdgeChange(
      edgeChanges,
      edgeState,
      activeEdgeId,
      edgeStateForAction(redActionType),
      edgeDirection(sourceHost, targetHost, activeEdgeId),
    );

    if (blueSuccess && (blueActionType === "isolate_host" || blueActionType === "block_connection")) {
      appendEdgeChange(
        edgeChanges,
        edgeState,
        activeEdgeId,
        "blocked",
        edgeDirection(sourceHost, targetHost, activeEdgeId),
      );
    }

    const stateDelta: StateDelta = {
      ts_ms: now + step * 1000 + 220,
      step,
      node_changes: nodeChanges,
      edge_changes: edgeChanges,
    };
    messages.push({ type: "state_delta", data: stateDelta });

    const shouldDetect = redActionType !== "scan_host" && (redSuccess || step % 3 === 0);
    if (shouldDetect && config.includeBlue) {
      const detection: DetectionEvent = {
        event_id: eventId("det", detectionCounter),
        ts_ms: now + step * 1000 + 320,
        step,
        detector: "BLUE",
        target_host: targetHost,
        signal: redActionType === "exfiltrate_data" ? "egress_spike" : "traffic_spike",
        severity: redActionType === "exfiltrate_data" ? "critical" : redActionType === "lateral_move" ? "high" : "medium",
        detected: true,
        mitre_tactic: redAction.mitre_tactic,
      };
      detections.push(clamp(0.7 + rng() * 2.1, 0.4, 3.4));
      messages.push({ type: "detection_event", data: detection });
      detectionCounter += 1;
    }

    if (blueActionType === "deploy_deception" && !decoyAdded && config.includeBlue) {
      decoyAdded = true;
      const decoyNode = createDecoyNode("auth_server");
      const decoyEdge: TopologyEdge = {
        id: `${decoyNode.id}->auth_server`,
        source: decoyNode.id,
        target: "auth_server",
        visual_state: "normal",
      };
      const addEvent: TopologyAddNodeData = {
        node: decoyNode,
        edges: [decoyEdge],
      };
      messages.push({ type: "topology_add_node", data: addEvent });
      nodeState.set(decoyNode.id, {
        visual: "neutral",
        overlay: null,
        defenseState: "none",
        compromiseLevel: 0,
      });
      edgeState.set(decoyEdge.id, "normal");
    }

    const explainability: ExplainabilityRecord = {
      ts_ms: now + step * 1000 + 420,
      step,
      action: blueActionType,
      target_host: blueTarget,
      confidence: clamp(0.62 + rng() * 0.28, 0, 1),
      reason_features: [
        { name: "traffic_spike_ratio", value: clamp(1.1 + rng() * 3.3, 0.8, 4.8) },
        { name: "critical_asset_risk", value: clamp(0.2 + phase * 0.8, 0, 1) },
        { name: "lateral_movement_pattern_match", value: clamp(0.3 + rng() * 0.6, 0, 1) },
      ],
      expected_effect: blueActionType === "isolate_host" ? "contain lateral spread" : "reduce attack surface",
    };
    messages.push({ type: "explainability", data: explainability });

    const compromisedCount = [...nodeState.values()].filter(
      (entry) => entry.visual === "compromised" || entry.visual === "critical",
    ).length;
    const isolatedCount = [...nodeState.values()].filter((entry) => entry.visual === "isolated").length;
    const patchedCount = [...nodeState.values()].filter((entry) => entry.visual === "patched").length;

    redScore = clamp(redScore + redRisk * (redSuccess ? 0.22 : 0.1), 0, 99);
    blueReward = clamp(blueReward + (blueSuccess ? 0.18 : 0.05) + patchedCount * 0.01, -10, 99);

    const attackPressure = clamp((compromisedCount + step * 0.01) / 12, 0, 1);
    const containmentPressure = clamp((isolatedCount + patchedCount + (config.includeBlue ? 0.5 : 0)) / 12, 0, 1);
    const availability = clamp(1 - compromisedCount * 0.032 - attackPressure * 0.08 + patchedCount * 0.012, 0.55, 1);

    const classification = Object.fromEntries(
      MITRE_TACTICS.map((tactic) => {
        const count = tacticCounts.get(tactic) ?? 0;
        const pct = redActionsTotal > 0 ? count / redActionsTotal : 0;
        return [tactic, { count, percentage: Number(pct.toFixed(2)) }];
      }),
    ) as MetricsTick["alert_classification"];

    const topHotTargets = [...hotTargetHits.entries()]
      .map(([node_id, hit_count]) => ({ node_id, hit_count }))
      .sort((a, b) => b.hit_count - a.hit_count)
      .slice(0, 5);

    const metrics: MetricsTick = {
      ...defaultMetrics(step),
      attack_pressure: Number(attackPressure.toFixed(2)),
      containment_pressure: Number(containmentPressure.toFixed(2)),
      service_availability: Number(availability.toFixed(2)),
      open_incidents: Math.max(compromisedCount - isolatedCount, 0),
      contained_incidents: isolatedCount + patchedCount,
      red_actions_total: redActionsTotal,
      blue_actions_total: blueActionsTotal,
      blue_reward_cumulative: Number(blueReward.toFixed(2)),
      red_score_cumulative: Number(redScore.toFixed(2)),
      detection_latency_mean:
        detections.length > 0
          ? Number((detections.reduce((sum, n) => sum + n, 0) / detections.length).toFixed(2))
          : 0,
      hot_targets: topHotTargets,
      alert_classification: classification,
    };
    messages.push({ type: "metrics_tick", data: metrics });
  }

  const summary: EpisodeEnd = {
    outcome: config.includeBlue ? "contained" : "breached",
    final_step: config.totalSteps,
    summary: {
      total_red_actions: redActionsTotal,
      total_blue_actions: blueActionsTotal,
      nodes_compromised: [...nodeState.values()].filter((entry) => entry.visual === "compromised").length,
      nodes_isolated: [...nodeState.values()].filter((entry) => entry.visual === "isolated").length,
      nodes_patched: [...nodeState.values()].filter((entry) => entry.visual === "patched").length,
      hvts_compromised: [...nodeState.entries()].filter(
        ([nodeId, entry]) => nodeId.includes("server") && (entry.visual === "compromised" || entry.visual === "critical"),
      ).length,
      data_exfiltrated: !config.includeBlue,
      final_service_availability: messages
        .filter((msg): msg is { type: "metrics_tick"; data: MetricsTick } => msg.type === "metrics_tick")
        .at(-1)?.data.service_availability ?? 0,
      blue_reward_total: Number(blueReward.toFixed(2)),
    },
  };

  messages.push({ type: "episode_end", data: summary });

  return {
    replayId: config.replayId,
    scenarioId: config.scenarioId,
    totalSteps: config.totalSteps,
    messages,
  };
}
