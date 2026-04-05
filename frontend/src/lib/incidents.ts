import type { ReplayEvent } from "./api";

export type AlertClassification =
  | "Initial Access"
  | "Execution"
  | "Lateral Movement"
  | "Data Exfiltration"
  | "Defense Evasion";

export type RiskLevel = "LOW" | "MEDIUM" | "HIGH" | "CRITICAL";

export type FormattedIncident = {
  step: number;
  actor: ReplayEvent["actor"];
  target: string;
  action: string;
  narrative: string;
  riskLevel: RiskLevel;
  riskScore: number;
};

function clamp(value: number, min: number, max: number): number {
  return Math.max(min, Math.min(max, value));
}

export function humanizeAction(action: string): string {
  return action
    .replaceAll("_", " ")
    .replaceAll("-", " ")
    .trim()
    .replace(/\s+/g, " ")
    .toLowerCase();
}

export function classifyAction(action: string): AlertClassification {
  const value = action.toLowerCase();

  if (
    value.includes("phish") ||
    value.includes("credential") ||
    value.includes("scan") ||
    value.includes("login") ||
    value.includes("exploit")
  ) {
    return "Initial Access";
  }

  if (value.includes("lateral") || value.includes("pivot") || value.includes("movement")) {
    return "Lateral Movement";
  }

  if (value.includes("exfil") || value.includes("dump") || value.includes("steal")) {
    return "Data Exfiltration";
  }

  if (
    value.includes("evade") ||
    value.includes("disable") ||
    value.includes("obfus") ||
    value.includes("persist")
  ) {
    return "Defense Evasion";
  }

  return "Execution";
}

export function scoreRisk(event: ReplayEvent): number {
  const action = event.action.toLowerCase();
  const target = event.target.toLowerCase();
  let score = event.actor === "RED" ? 50 : 30;

  if (action.includes("scan") || action.includes("enumerate") || action.includes("recon")) score += 10;
  if (action.includes("exploit") || action.includes("execute")) score += 16;
  if (action.includes("credential") || action.includes("phish")) score += 20;
  if (action.includes("lateral") || action.includes("pivot")) score += 18;
  if (action.includes("traffic_spike") || action.includes("anomaly")) score += 12;
  if (action.includes("exfil") || action.includes("dump") || action.includes("ransom")) score += 28;
  if (action.includes("isolate") || action.includes("block") || action.includes("quarantine")) score -= 14;
  if (action.includes("step_marker")) score -= 20;

  if (target.includes("engine") || target.includes("identity") || target.includes("vault")) score += 16;
  if (target.includes("db") || target.includes("core") || target.includes("gateway")) score += 12;
  if (target.startsWith("host-0")) score += 6;

  return clamp(score, 0, 100);
}

export function riskLevelFromScore(score: number): RiskLevel {
  if (score <= 29) return "LOW";
  if (score <= 59) return "MEDIUM";
  if (score <= 79) return "HIGH";
  return "CRITICAL";
}

export function buildNarrative(event: ReplayEvent, riskLevel: RiskLevel, riskScore: number): string {
  const action = event.action.toLowerCase();
  const actionLabel = humanizeAction(event.action);
  const actorLabel = event.actor === "RED" ? "Red-team" : "Blue-team";
  const targetLabel = event.target;

  if (action.includes("traffic_spike")) {
    return `${actorLabel} telemetry detected abnormal traffic on ${targetLabel}, indicating possible lateral movement pressure. This event is assessed as ${riskLevel} risk at ${riskScore}%.`;
  }

  if (action.includes("scan")) {
    return `${actorLabel} reconnaissance scanned ${targetLabel} to map exposed services and entry paths. This event is assessed as ${riskLevel} risk at ${riskScore}%.`;
  }

  if (action.includes("enumerate")) {
    return `${actorLabel} service enumeration profiled ${targetLabel} to identify exploitable interfaces. This event is assessed as ${riskLevel} risk at ${riskScore}%.`;
  }

  if (action.includes("isolate")) {
    return `${actorLabel} containment isolated ${targetLabel} from connected systems to limit adversary spread. This event is assessed as ${riskLevel} risk at ${riskScore}%.`;
  }

  if (action.includes("monitor")) {
    return `${actorLabel} monitoring intensified observation on ${targetLabel} after suspicious signals were detected. This event is assessed as ${riskLevel} risk at ${riskScore}%.`;
  }

  if (action.includes("step_marker")) {
    return `The simulator recorded a campaign checkpoint at ${targetLabel} to preserve the current attack-defense state. This point-in-time signal is assessed as ${riskLevel} risk at ${riskScore}%.`;
  }

  return `${actorLabel} executed ${actionLabel} against ${targetLabel} as part of the active engagement sequence. This event is assessed as ${riskLevel} risk at ${riskScore}%.`;
}

export function formatIncident(event: ReplayEvent): FormattedIncident {
  const riskScore = scoreRisk(event);
  const riskLevel = riskLevelFromScore(riskScore);

  return {
    step: event.step,
    actor: event.actor,
    target: event.target,
    action: event.action,
    narrative: buildNarrative(event, riskLevel, riskScore),
    riskLevel,
    riskScore,
  };
}
