export type Actor = "RED" | "BLUE" | "ENV";

export interface ActionEvent {
  event_id: string;
  ts_ms: number;
  step: number;
  actor: Actor;
  action_type: string;
  source_host: string;
  target_host: string;
  target_service?: string;
  outcome: string;
  mitre_tactic: string;
  confidence: number;
}

export interface DetectionEvent {
  event_id: string;
  ts_ms: number;
  step: number;
  detector: "BLUE";
  target_host: string;
  signal: string;
  severity: "low" | "medium" | "high";
  detected: boolean;
}

export interface ExplainabilityRecord {
  ts_ms: number;
  step: number;
  action: string;
  target_host: string;
  confidence: number;
  reason_features: Array<{ name: string; value: number }>;
  expected_effect: string;
}
