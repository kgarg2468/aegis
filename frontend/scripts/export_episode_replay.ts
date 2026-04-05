import { createHash } from "node:crypto";
import { promises as fs } from "node:fs";
import path from "node:path";

import type { MetricsTick } from "../src/lib/integrationContract";
import { generateScenarioReplay } from "../src/lib/scenarioReplay";
import { indexReplayMessages, materializeGraphAtStep } from "../src/lib/replayRuntime";

type Mode = "enterprise" | "no_blue";

function parseArgs(argv: string[]): Record<string, string> {
  const out: Record<string, string> = {};
  for (let i = 0; i < argv.length; i += 1) {
    const token = argv[i];
    if (!token.startsWith("--")) continue;
    const key = token.slice(2);
    const value = argv[i + 1];
    if (!value || value.startsWith("--")) {
      throw new Error(`Missing value for --${key}`);
    }
    out[key] = value;
    i += 1;
  }
  return out;
}

function mustGet(args: Record<string, string>, key: string): string {
  const value = args[key];
  if (!value) {
    throw new Error(`Missing required argument --${key}`);
  }
  return value;
}

async function ensureDir(dir: string): Promise<void> {
  await fs.mkdir(dir, { recursive: true });
}

async function writeJson(filePath: string, payload: unknown): Promise<void> {
  await fs.writeFile(filePath, `${JSON.stringify(payload, null, 2)}\n`, "utf8");
}

async function writeJsonl(filePath: string, rows: unknown[]): Promise<void> {
  const body = rows.map((row) => JSON.stringify(row)).join("\n");
  await fs.writeFile(filePath, `${body}\n`, "utf8");
}

async function sha256(filePath: string): Promise<string> {
  const raw = await fs.readFile(filePath);
  return createHash("sha256").update(raw).digest("hex");
}

function toDamageScore(serviceAvailability: number): number {
  return Number((1 - serviceAvailability).toFixed(4));
}

function toContainmentSteps(metrics: MetricsTick[]): number {
  for (const tick of metrics) {
    if ((tick.open_incidents ?? 0) === 0) {
      return tick.step;
    }
  }
  return metrics.at(-1)?.step ?? 0;
}

function deterministicBaseTsMs(seed: number, mode: Mode): number {
  const modeOffset = mode === "enterprise" ? 0 : 5000;
  return 1_710_000_000_000 + seed * 10_000 + modeOffset;
}

async function main(): Promise<void> {
  const args = parseArgs(process.argv.slice(2));

  const scenarioId = mustGet(args, "scenario-id");
  const scenarioDisplayName = mustGet(args, "scenario-display-name");
  const replayId = mustGet(args, "replay-id");
  const outDir = mustGet(args, "out-dir");

  const mode = mustGet(args, "mode") as Mode;
  if (mode !== "enterprise" && mode !== "no_blue") {
    throw new Error(`Invalid --mode ${mode}`);
  }

  const seed = Number.parseInt(mustGet(args, "seed"), 10);
  const totalSteps = Number.parseInt(mustGet(args, "steps"), 10);
  if (!Number.isFinite(seed)) throw new Error("Invalid --seed");
  if (!Number.isFinite(totalSteps) || totalSteps <= 0) throw new Error("Invalid --steps");

  const includeBlue = mode === "enterprise";
  const generated = generateScenarioReplay({
    replayId,
    scenarioId,
    totalSteps,
    seed,
    includeBlue,
    baseTsMs: deterministicBaseTsMs(seed, mode),
  });

  const timeline = indexReplayMessages(generated.messages);
  await ensureDir(outDir);

  const snapshots: Record<string, { nodes: unknown[]; edges: unknown[] }> = {};
  for (let step = 10; step <= generated.totalSteps; step += 10) {
    const state = materializeGraphAtStep(timeline, step);
    snapshots[String(step)] = {
      nodes: state.nodes,
      edges: state.edges,
    };
  }

  const metrics = generated.messages
    .filter((msg): msg is { type: "metrics_tick"; data: MetricsTick } => msg.type === "metrics_tick")
    .map((msg) => msg.data);

  const finalMetrics = metrics.at(-1);
  const finalEpisode = generated.messages.find((msg) => msg.type === "episode_end");
  const finalOutcome = finalEpisode?.type === "episode_end" ? finalEpisode.data.outcome : "timeout";
  const finalSummary = finalEpisode?.type === "episode_end" ? finalEpisode.data.summary : null;

  const manifest = {
    replay_id: generated.replayId,
    scenario_id: generated.scenarioId,
    scenario_display_name: scenarioDisplayName,
    mode,
    seed,
    checkpoint_id: "episode_rebuild_frontend_deterministic",
    duration_steps: generated.totalSteps,
    total_events: generated.messages.length,
    outcome: finalOutcome,
    kpis: {
      damage_score: toDamageScore(finalMetrics?.service_availability ?? 1),
      containment_time_steps: toContainmentSteps(metrics),
      hvts_compromised: finalSummary?.hvts_compromised ?? 0,
      data_exfiltrated: finalSummary?.data_exfiltrated ?? false,
      final_service_availability: finalSummary?.final_service_availability ?? (finalMetrics?.service_availability ?? 1),
      blue_reward_total: finalSummary?.blue_reward_total ?? (finalMetrics?.blue_reward_cumulative ?? 0),
    },
    files: {
      events: "events.jsonl",
      topology: "topology_snapshots.json",
      metrics: "metrics.json",
    },
  };

  const topologySnapshots = {
    initial: {
      nodes: timeline.topology.nodes,
      edges: timeline.topology.edges,
      zones: timeline.topology.zones,
    },
    snapshots,
  };

  const manifestPath = path.join(outDir, "manifest.json");
  const eventsPath = path.join(outDir, "events.jsonl");
  const topologyPath = path.join(outDir, "topology_snapshots.json");
  const metricsPath = path.join(outDir, "metrics.json");

  await writeJson(manifestPath, manifest);
  await writeJsonl(eventsPath, generated.messages);
  await writeJson(topologyPath, topologySnapshots);
  await writeJson(metricsPath, metrics);

  const summary = {
    replay_id: replayId,
    scenario_id: scenarioId,
    scenario_display_name: scenarioDisplayName,
    mode,
    include_blue: includeBlue,
    seed,
    total_steps: totalSteps,
    total_events: generated.messages.length,
    outcome: manifest.outcome,
    kpis: manifest.kpis,
    files: {
      manifest: manifestPath,
      events: eventsPath,
      topology: topologyPath,
      metrics: metricsPath,
    },
    checksums: {
      manifest_sha256: await sha256(manifestPath),
      events_sha256: await sha256(eventsPath),
      topology_sha256: await sha256(topologyPath),
      metrics_sha256: await sha256(metricsPath),
    },
  };

  process.stdout.write(`${JSON.stringify(summary)}\n`);
}

void main();
