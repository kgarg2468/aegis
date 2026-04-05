export const NODE_VISUAL_STATES = [
  "neutral",
  "monitored",
  "probed",
  "compromised",
  "critical",
  "isolated",
  "patched",
] as const;

export const EDGE_VISUAL_STATES = [
  "normal",
  "scanning",
  "lateral_movement",
  "exfiltration",
  "credential_flow",
  "blocked",
] as const;

export const OVERLAY_STATES = ["monitored"] as const;

export const ACTORS = ["RED", "BLUE", "ENV"] as const;

export const RED_ACTION_TYPES = [
  "scan_host",
  "enumerate_service",
  "exploit_vulnerability",
  "lateral_move",
  "privilege_escalate",
  "exfiltrate_data",
] as const;

export const BLUE_ACTION_TYPES = [
  "monitor_host",
  "patch_service",
  "isolate_host",
  "block_connection",
  "rotate_credentials",
  "deploy_deception",
] as const;

export const ACTION_TYPES = [...RED_ACTION_TYPES, ...BLUE_ACTION_TYPES] as const;

export const MITRE_TACTICS = [
  "Reconnaissance",
  "Initial Access",
  "Execution",
  "Lateral Movement",
  "Credential Access",
  "Exfiltration",
  "Defense Evasion",
  "Persistence",
  "Collection",
  "Impact",
] as const;

export const OUTCOMES = ["success", "failure", "partial", "blocked"] as const;

export const SEVERITIES = ["critical", "high", "medium", "low", "info"] as const;

export type NodeVisualState = (typeof NODE_VISUAL_STATES)[number];
export type EdgeVisualState = (typeof EDGE_VISUAL_STATES)[number];
export type OverlayState = (typeof OVERLAY_STATES)[number];
export type Actor = (typeof ACTORS)[number];
export type ActionType = (typeof ACTION_TYPES)[number];
export type MitreTactic = (typeof MITRE_TACTICS)[number];
export type Outcome = (typeof OUTCOMES)[number];
export type Severity = (typeof SEVERITIES)[number];

export type NodeType = "external" | "infrastructure" | "endpoint" | "hvt" | "iot";

export type ZoneType = "external" | "perimeter" | "campus" | "admin" | "research";

export type NodeShape = "hexagon" | "round-rectangle" | "ellipse" | "triangle" | "diamond";

export interface NodeRegistryItem {
  id: string;
  type: NodeType;
  zone: ZoneType;
  shape: NodeShape;
  size: number;
}

export interface TopologyNode {
  id: string;
  type: NodeType;
  zone: ZoneType;
  label: string;
  services: string[];
  criticality: number;
  visual_state: NodeVisualState;
  overlay: OverlayState | null;
  is_decoy?: boolean;
}

export interface TopologyEdge {
  id: string;
  source: string;
  target: string;
  visual_state: EdgeVisualState;
}

export interface ZoneDefinition {
  id: string;
  label: string;
  member_ids: string[];
}

export interface TopologyInitData {
  nodes: TopologyNode[];
  edges: TopologyEdge[];
  zones: ZoneDefinition[];
  scenario_id: string;
  total_steps: number;
  seed: number;
}

export interface ActionEvent {
  event_id: string;
  ts_ms: number;
  step: number;
  actor: Actor;
  action_type: ActionType;
  source_host: string;
  target_host: string;
  target_service?: string;
  outcome: Outcome;
  mitre_tactic: MitreTactic;
  confidence: number;
  description: string;
  severity: Severity;
  risk_score: number;
}

export interface StateDeltaNodeChange {
  node_id: string;
  visual_state: NodeVisualState;
  overlay: OverlayState | null;
  compromise_level: number;
  defense_state: "none" | "monitored" | "isolated" | "patched";
}

export interface StateDeltaEdgeChange {
  edge_id: string;
  visual_state: EdgeVisualState;
  direction: "forward" | "reverse";
}

export interface StateDelta {
  ts_ms: number;
  step: number;
  node_changes: StateDeltaNodeChange[];
  edge_changes: StateDeltaEdgeChange[];
}

export interface DetectionEvent {
  event_id: string;
  ts_ms: number;
  step: number;
  detector: "BLUE";
  target_host: string;
  signal: string;
  severity: Severity;
  detected: boolean;
  mitre_tactic: MitreTactic;
}

export interface ExplainabilityFeature {
  name: string;
  value: number;
}

export interface ExplainabilityRecord {
  ts_ms: number;
  step: number;
  action: Extract<ActionType, (typeof BLUE_ACTION_TYPES)[number]>;
  target_host: string;
  confidence: number;
  reason_features: ExplainabilityFeature[];
  expected_effect: string;
}

export interface HotTarget {
  node_id: string;
  hit_count: number;
}

export interface AlertClassificationEntry {
  count: number;
  percentage: number;
}

export type AlertClassification = Partial<Record<MitreTactic, AlertClassificationEntry>>;

export interface MetricsTick {
  step: number;
  attack_pressure: number;
  containment_pressure: number;
  service_availability: number;
  open_incidents: number;
  contained_incidents: number;
  red_actions_total: number;
  blue_actions_total: number;
  blue_reward_cumulative: number;
  red_score_cumulative: number;
  detection_latency_mean: number;
  hot_targets: HotTarget[];
  alert_classification: AlertClassification;
}

export interface TopologyAddNodeData {
  node: TopologyNode;
  edges: TopologyEdge[];
}

export interface EpisodeEnd {
  outcome: "contained" | "breached" | "timeout";
  final_step: number;
  summary: {
    total_red_actions: number;
    total_blue_actions: number;
    nodes_compromised: number;
    nodes_isolated: number;
    nodes_patched: number;
    hvts_compromised: number;
    data_exfiltrated: boolean;
    final_service_availability: number;
    blue_reward_total: number;
  };
}

export type WSMessage =
  | { type: "topology_init"; data: TopologyInitData }
  | { type: "action_event"; data: ActionEvent }
  | { type: "state_delta"; data: StateDelta }
  | { type: "detection_event"; data: DetectionEvent }
  | { type: "explainability"; data: ExplainabilityRecord }
  | { type: "metrics_tick"; data: MetricsTick }
  | { type: "topology_add_node"; data: TopologyAddNodeData }
  | { type: "episode_end"; data: EpisodeEnd };

export const NODE_REGISTRY: NodeRegistryItem[] = [
  { id: "internet", type: "external", zone: "external", shape: "hexagon", size: 40 },
  { id: "vpn_gateway", type: "infrastructure", zone: "perimeter", shape: "round-rectangle", size: 44 },
  { id: "web_portal", type: "infrastructure", zone: "perimeter", shape: "round-rectangle", size: 44 },
  { id: "dns_server", type: "infrastructure", zone: "perimeter", shape: "round-rectangle", size: 44 },
  { id: "eduroam_ap_01", type: "infrastructure", zone: "campus", shape: "round-rectangle", size: 44 },
  { id: "eduroam_ap_02", type: "infrastructure", zone: "campus", shape: "round-rectangle", size: 44 },
  { id: "eduroam_ap_03", type: "infrastructure", zone: "campus", shape: "round-rectangle", size: 44 },
  { id: "student_device_01", type: "endpoint", zone: "campus", shape: "ellipse", size: 36 },
  { id: "student_device_02", type: "endpoint", zone: "campus", shape: "ellipse", size: 36 },
  { id: "student_device_03", type: "endpoint", zone: "campus", shape: "ellipse", size: 36 },
  { id: "student_device_04", type: "endpoint", zone: "campus", shape: "ellipse", size: 36 },
  { id: "student_device_05", type: "endpoint", zone: "campus", shape: "ellipse", size: 36 },
  { id: "faculty_device_01", type: "endpoint", zone: "campus", shape: "ellipse", size: 36 },
  { id: "faculty_device_02", type: "endpoint", zone: "campus", shape: "ellipse", size: 36 },
  { id: "faculty_device_03", type: "endpoint", zone: "campus", shape: "ellipse", size: 36 },
  { id: "print_server", type: "infrastructure", zone: "campus", shape: "round-rectangle", size: 44 },
  { id: "iot_projector_01", type: "iot", zone: "campus", shape: "triangle", size: 32 },
  { id: "lab_workstation_01", type: "endpoint", zone: "campus", shape: "ellipse", size: 36 },
  { id: "lab_workstation_02", type: "endpoint", zone: "campus", shape: "ellipse", size: 36 },
  { id: "auth_server", type: "hvt", zone: "admin", shape: "diamond", size: 52 },
  { id: "active_directory", type: "hvt", zone: "admin", shape: "diamond", size: 52 },
  { id: "sis_server", type: "hvt", zone: "admin", shape: "diamond", size: 52 },
  { id: "finance_server", type: "hvt", zone: "admin", shape: "diamond", size: 52 },
  { id: "hr_server", type: "hvt", zone: "admin", shape: "diamond", size: 52 },
  { id: "research_server_01", type: "infrastructure", zone: "research", shape: "round-rectangle", size: 44 },
  { id: "research_server_02", type: "infrastructure", zone: "research", shape: "round-rectangle", size: 44 },
  { id: "shared_storage", type: "hvt", zone: "research", shape: "diamond", size: 52 },
  { id: "irb_system", type: "hvt", zone: "research", shape: "diamond", size: 52 },
];

export const ZONES: ZoneDefinition[] = [
  {
    id: "zone_perimeter",
    label: "PERIMETER / DMZ",
    member_ids: ["vpn_gateway", "web_portal", "dns_server"],
  },
  {
    id: "zone_campus",
    label: "CAMPUS NETWORK",
    member_ids: [
      "eduroam_ap_01",
      "eduroam_ap_02",
      "eduroam_ap_03",
      "student_device_01",
      "student_device_02",
      "student_device_03",
      "student_device_04",
      "student_device_05",
      "faculty_device_01",
      "faculty_device_02",
      "faculty_device_03",
      "print_server",
      "iot_projector_01",
      "lab_workstation_01",
      "lab_workstation_02",
    ],
  },
  {
    id: "zone_admin",
    label: "ADMIN BACKBONE",
    member_ids: ["auth_server", "active_directory", "sis_server", "finance_server", "hr_server"],
  },
  {
    id: "zone_research",
    label: "RESEARCH SEGMENT",
    member_ids: ["research_server_01", "research_server_02", "shared_storage", "irb_system"],
  },
];

const UNDIRECTED_EDGE_PAIRS: Array<[string, string]> = [
  ["internet", "vpn_gateway"],
  ["internet", "web_portal"],
  ["internet", "dns_server"],
  ["vpn_gateway", "eduroam_ap_01"],
  ["vpn_gateway", "eduroam_ap_02"],
  ["vpn_gateway", "eduroam_ap_03"],
  ["web_portal", "auth_server"],
  ["dns_server", "auth_server"],
  ["dns_server", "research_server_01"],
  ["eduroam_ap_01", "student_device_01"],
  ["eduroam_ap_01", "student_device_02"],
  ["eduroam_ap_01", "faculty_device_01"],
  ["eduroam_ap_01", "print_server"],
  ["eduroam_ap_02", "student_device_03"],
  ["eduroam_ap_02", "student_device_04"],
  ["eduroam_ap_02", "faculty_device_02"],
  ["eduroam_ap_02", "lab_workstation_01"],
  ["eduroam_ap_03", "student_device_05"],
  ["eduroam_ap_03", "faculty_device_03"],
  ["eduroam_ap_03", "iot_projector_01"],
  ["eduroam_ap_03", "lab_workstation_02"],
  ["faculty_device_02", "auth_server"],
  ["auth_server", "active_directory"],
  ["auth_server", "sis_server"],
  ["auth_server", "finance_server"],
  ["auth_server", "hr_server"],
  ["active_directory", "research_server_01"],
  ["active_directory", "research_server_02"],
  ["research_server_01", "shared_storage"],
  ["research_server_02", "shared_storage"],
  ["research_server_01", "irb_system"],
  ["shared_storage", "irb_system"],
];

function toLabel(id: string): string {
  return id.replaceAll("_", " ").toUpperCase();
}

function defaultServices(nodeId: string): string[] {
  if (nodeId.includes("auth") || nodeId.includes("active_directory")) {
    return ["ldap", "kerberos", "radius"];
  }
  if (nodeId.includes("dns")) return ["dns", "dnssec"];
  if (nodeId.includes("web")) return ["https", "waf"];
  if (nodeId.includes("vpn")) return ["vpn", "ipsec"];
  if (nodeId.includes("print")) return ["ipp", "smb"];
  if (nodeId.includes("storage")) return ["nfs", "smb"];
  if (nodeId.includes("research") || nodeId.includes("sis") || nodeId.includes("finance") || nodeId.includes("hr") || nodeId.includes("irb")) {
    return ["https", "ssh"];
  }
  return ["wifi", "dhcp"];
}

function defaultCriticality(type: NodeType): number {
  if (type === "hvt") return 1;
  if (type === "external") return 0;
  if (type === "infrastructure") return 0.72;
  if (type === "iot") return 0.35;
  return 0.5;
}

function toEdge(source: string, target: string): TopologyEdge {
  return {
    id: `${source}->${target}`,
    source,
    target,
    visual_state: "normal",
  };
}

export function buildCanonicalEdges(): TopologyEdge[] {
  const edges: TopologyEdge[] = [];
  for (const [a, b] of UNDIRECTED_EDGE_PAIRS) {
    edges.push(toEdge(a, b));
    edges.push(toEdge(b, a));
  }
  return edges;
}

export function buildCanonicalNodes(): TopologyNode[] {
  return NODE_REGISTRY.map((item) => ({
    id: item.id,
    type: item.type,
    zone: item.zone,
    label: toLabel(item.id),
    services: defaultServices(item.id),
    criticality: defaultCriticality(item.type),
    visual_state: "neutral",
    overlay: null,
  }));
}

export function createCanonicalTopologyInit(
  scenarioId = "faculty_phish",
  totalSteps = 200,
  seed = 1003,
): TopologyInitData {
  return {
    nodes: buildCanonicalNodes(),
    edges: buildCanonicalEdges(),
    zones: ZONES,
    scenario_id: scenarioId,
    total_steps: totalSteps,
    seed,
  };
}

export function isWSMessage(input: unknown): input is WSMessage {
  if (!input || typeof input !== "object") return false;
  const record = input as Record<string, unknown>;
  if (typeof record.type !== "string") return false;
  if (!("data" in record)) return false;
  return [
    "topology_init",
    "action_event",
    "state_delta",
    "detection_event",
    "explainability",
    "metrics_tick",
    "topology_add_node",
    "episode_end",
  ].includes(record.type);
}

export function isNodeIdAllowed(nodeId: string): boolean {
  if (nodeId.startsWith("decoy_")) return true;
  return NODE_REGISTRY.some((node) => node.id === nodeId);
}

export function deriveZoneContainerId(zone: ZoneType): string {
  return `zone_${zone}`;
}

export function normalizeRiskScore(value: number): number {
  if (!Number.isFinite(value)) return 0;
  if (value < 0) return 0;
  if (value > 1) return 1;
  return value;
}
