import { promises as fs, statSync } from "node:fs";
import path from "node:path";

export type LocalReplayListItem = {
  id: string;
  run_id: string;
  replay_id: string;
  scenario_id?: string;
  created_at?: string;
  source: "local_runs";
};

type LocalReplayRef = {
  id: string;
  runId: string;
  replayId: string;
  scenarioId?: string;
  replayDir: string;
  eventsPath?: string;
  metricsPath?: string;
  topologyPath?: string;
  manifestPath?: string;
  createdAt?: string;
};

const DEFAULT_RUNS_CANDIDATES = ["runs", "../runs", "../../runs"];
const DEFAULT_LEGACY_REPLAYS_CANDIDATES = [
  "artifacts/replays",
  "../artifacts/replays",
  "../../artifacts/replays",
];

function safeJoin(...segments: string[]): string {
  return path.resolve(...segments);
}

function buildReplayId(runId: string, replayId: string): string {
  return `${runId}__${replayId}`;
}

function isDirectory(candidate: string): boolean {
  try {
    return Boolean(candidate) && statSync(candidate).isDirectory();
  } catch {
    return false;
  }
}

function uniquePaths(values: string[]): string[] {
  const seen = new Set<string>();
  const result: string[] = [];
  values.forEach((entry) => {
    const normalized = path.resolve(entry);
    if (seen.has(normalized)) return;
    seen.add(normalized);
    result.push(normalized);
  });
  return result;
}

function resolveRunsRoots(): string[] {
  const cwd = process.cwd();
  const configured = process.env.RUNS_DIR ?? process.env.NEXT_RUNS_DIR;
  const configuredEntries = configured
    ? configured
        .split(path.delimiter)
        .map((entry) => entry.trim())
        .filter(Boolean)
        .map((entry) => safeJoin(cwd, entry))
    : [];

  const defaultEntries = DEFAULT_RUNS_CANDIDATES.map((entry) =>
    safeJoin(cwd, entry),
  );

  return uniquePaths([...configuredEntries, ...defaultEntries]).filter(isDirectory);
}

function resolveLegacyReplayRoots(): string[] {
  const cwd = process.cwd();
  return uniquePaths(
    DEFAULT_LEGACY_REPLAYS_CANDIDATES.map((entry) => safeJoin(cwd, entry)),
  ).filter(isDirectory);
}

async function safeReadJson(filePath: string): Promise<unknown | null> {
  try {
    const raw = await fs.readFile(filePath, "utf8");
    return JSON.parse(raw) as unknown;
  } catch {
    return null;
  }
}

async function safeReaddir(directory: string): Promise<string[]> {
  try {
    const entries = await fs.readdir(directory, { withFileTypes: true });
    return entries.filter((entry) => entry.isDirectory()).map((entry) => entry.name);
  } catch {
    return [];
  }
}

async function pathIfExists(candidate: string): Promise<string | undefined> {
  try {
    const stat = await fs.stat(candidate);
    return stat.isFile() ? candidate : undefined;
  } catch {
    return undefined;
  }
}

function toRecord(value: unknown): Record<string, unknown> | null {
  if (!value || typeof value !== "object" || Array.isArray(value)) return null;
  return value as Record<string, unknown>;
}

function toString(value: unknown): string | undefined {
  if (typeof value !== "string") return undefined;
  const trimmed = value.trim();
  return trimmed.length > 0 ? trimmed : undefined;
}

async function discoverRunReplayRefs(): Promise<LocalReplayRef[]> {
  const refs: LocalReplayRef[] = [];
  const roots = resolveRunsRoots();

  for (const runsRoot of roots) {
    const runIds = await safeReaddir(runsRoot);

    for (const runId of runIds) {
      const runDir = path.join(runsRoot, runId);
      const runManifestPath = path.join(runDir, "manifest.json");
      const runManifest = await safeReadJson(runManifestPath);
      const runCreatedAt = toString(toRecord(runManifest)?.created_at);
      const replayRoots = [
        path.join(runDir, "replays"),
        path.join(runDir, "artifacts", "replays"),
      ].filter(isDirectory);

      for (const replayRoot of replayRoots) {
        const replayIds = await safeReaddir(replayRoot);

        for (const replayId of replayIds) {
          const replayDir = path.join(replayRoot, replayId);
          const replayManifestPath = path.join(replayDir, "manifest.json");
          const replayManifest = await safeReadJson(replayManifestPath);
          const replayManifestRecord = toRecord(replayManifest);
          const replayCreatedAt =
            toString(replayManifestRecord?.created_at) ?? runCreatedAt;
          const scenarioId = toString(replayManifestRecord?.scenario_id);

          refs.push({
            id: buildReplayId(runId, replayId),
            runId,
            replayId,
            scenarioId,
            replayDir,
            createdAt: replayCreatedAt,
            eventsPath:
              (await pathIfExists(path.join(replayDir, "events.jsonl"))) ??
              (await pathIfExists(path.join(replayDir, "events.json"))),
            metricsPath: await pathIfExists(path.join(replayDir, "metrics.json")),
            topologyPath: await pathIfExists(
              path.join(replayDir, "topology_snapshots.json"),
            ),
            manifestPath: await pathIfExists(replayManifestPath),
          });
        }
      }
    }
  }

  return refs;
}

async function discoverLegacyReplayRefs(): Promise<LocalReplayRef[]> {
  const refs: LocalReplayRef[] = [];
  const roots = resolveLegacyReplayRoots();

  for (const legacyRoot of roots) {
    const replayIds = await safeReaddir(legacyRoot);

    for (const replayId of replayIds) {
      const replayDir = path.join(legacyRoot, replayId);
      const replayManifestPath = path.join(replayDir, "manifest.json");
      const replayManifest = await safeReadJson(replayManifestPath);
      const scenarioId = toString(toRecord(replayManifest)?.scenario_id);
      refs.push({
        id: buildReplayId("legacy", replayId),
        runId: "legacy",
        replayId,
        scenarioId,
        replayDir,
        eventsPath:
          (await pathIfExists(path.join(replayDir, "events.jsonl"))) ??
          (await pathIfExists(path.join(replayDir, "events.json"))),
        metricsPath: await pathIfExists(path.join(replayDir, "metrics.json")),
        topologyPath: await pathIfExists(path.join(replayDir, "topology_snapshots.json")),
        manifestPath: await pathIfExists(replayManifestPath),
      });
    }
  }

  return refs;
}

async function discoverReplayRefs(): Promise<LocalReplayRef[]> {
  const allRefs = [...(await discoverRunReplayRefs()), ...(await discoverLegacyReplayRefs())];

  const byId = new Map<string, LocalReplayRef>();
  allRefs.forEach((ref) => {
    if (!byId.has(ref.id)) {
      byId.set(ref.id, ref);
    }
  });

  return [...byId.values()].sort((a, b) => {
    const aTs = a.createdAt ? Date.parse(a.createdAt) : Number.NaN;
    const bTs = b.createdAt ? Date.parse(b.createdAt) : Number.NaN;

    if (Number.isFinite(aTs) && Number.isFinite(bTs)) return bTs - aTs;
    if (Number.isFinite(aTs)) return -1;
    if (Number.isFinite(bTs)) return 1;
    return a.id.localeCompare(b.id);
  });
}

function extractRawEvents(payload: unknown): unknown[] {
  if (Array.isArray(payload)) return payload;

  const record = toRecord(payload);
  if (!record) return [];

  if (Array.isArray(record.events)) return record.events;
  if (Array.isArray(record.data)) return record.data;
  if (Array.isArray(record.results)) return record.results;

  const event = toRecord(record.event);
  if (event) return [event];

  return [payload];
}

async function parseEventsFile(filePath: string): Promise<unknown[]> {
  const raw = await fs.readFile(filePath, "utf8");
  const trimmed = raw.trim();
  if (!trimmed) return [];

  if (filePath.endsWith(".jsonl")) {
    return trimmed
      .split("\n")
      .map((line) => line.trim())
      .filter(Boolean)
      .map((line) => {
        try {
          return JSON.parse(line) as unknown;
        } catch {
          return null;
        }
      })
      .filter((entry): entry is unknown => entry !== null);
  }

  const payload = JSON.parse(trimmed) as unknown;
  return extractRawEvents(payload);
}

async function parseJsonFile(filePath: string): Promise<unknown | null> {
  try {
    const raw = await fs.readFile(filePath, "utf8");
    return JSON.parse(raw) as unknown;
  } catch {
    return null;
  }
}

function toProjectRelative(filePath?: string): string | undefined {
  if (!filePath) return undefined;
  return path.relative(process.cwd(), filePath);
}

export async function listLocalReplays(): Promise<LocalReplayListItem[]> {
  const refs = await discoverReplayRefs();
  return refs.map((ref) => ({
    id: ref.id,
    run_id: ref.runId,
    replay_id: ref.replayId,
    scenario_id: ref.scenarioId,
    created_at: ref.createdAt,
    source: "local_runs",
  }));
}

export async function loadLocalReplayBundle(
  replayId: string,
): Promise<Record<string, unknown> | null> {
  const normalizedId = decodeURIComponent(replayId);
  const refs = await discoverReplayRefs();
  const match =
    refs.find((ref) => ref.id === normalizedId) ??
    refs.find((ref) => ref.replayId === normalizedId);

  if (!match) return null;

  const events = match.eventsPath ? await parseEventsFile(match.eventsPath) : [];
  const manifest = match.manifestPath ? await safeReadJson(match.manifestPath) : null;
  const topology = match.topologyPath ? await parseJsonFile(match.topologyPath) : null;

  return {
    id: match.id,
    replay_id: match.replayId,
    run_id: match.runId,
    created_at: match.createdAt,
    source: "local_runs",
    events,
    topology: topology ?? undefined,
    manifest: manifest ?? undefined,
    artifacts: {
      events_jsonl: toProjectRelative(match.eventsPath),
      metrics: toProjectRelative(match.metricsPath),
      topology_snapshots: toProjectRelative(match.topologyPath),
      manifest: toProjectRelative(match.manifestPath),
    },
  };
}
