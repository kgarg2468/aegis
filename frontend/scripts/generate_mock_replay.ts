import { promises as fs } from "node:fs";
import path from "node:path";

import { generateMockReplay } from "../src/lib/mockReplay";
import { indexReplayMessages, materializeGraphAtStep } from "../src/lib/replayRuntime";
import type { MetricsTick } from "../src/lib/integrationContract";

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

async function generateMockReplayBundle(): Promise<void> {
  const generated = generateMockReplay({
    replayId: "mock_replay_01",
    scenarioId: "faculty_phish",
    totalSteps: 200,
    seed: 1003,
    includeBlue: true,
  });

  const timeline = indexReplayMessages(generated.messages);
  const root = path.resolve(process.cwd(), "mock/replays", generated.replayId);

  await ensureDir(root);

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

  const finalMetrics = metrics[metrics.length - 1];

  const manifest = {
    replay_id: generated.replayId,
    scenario_id: generated.scenarioId,
    scenario_display_name: "Faculty Spear Phish -> Research Data Theft",
    seed: timeline.topology.seed,
    checkpoint_id: "ckpt_blue_mock_0001",
    duration_steps: generated.totalSteps,
    total_events: generated.messages.length,
    outcome: generated.messages.find((msg) => msg.type === "episode_end")?.data.outcome ?? "contained",
    kpis: {
      damage_score: Number((1 - (finalMetrics?.service_availability ?? 0.8)).toFixed(2)),
      containment_time_steps: 34,
      hvts_compromised: timeline.frames
        .flatMap((frame) => frame.stateDelta?.node_changes ?? [])
        .filter((change) => change.node_id.includes("server") && change.visual_state === "critical").length,
      data_exfiltrated:
        generated.messages.find((msg) => msg.type === "episode_end")?.data.summary.data_exfiltrated ?? false,
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

  await writeJson(path.join(root, "manifest.json"), manifest);
  await writeJsonl(path.join(root, "events.jsonl"), generated.messages);
  await writeJson(path.join(root, "topology_snapshots.json"), topologySnapshots);
  await writeJson(path.join(root, "metrics.json"), metrics);

  console.log(`Mock replay written to ${root}`);
}

void generateMockReplayBundle();
