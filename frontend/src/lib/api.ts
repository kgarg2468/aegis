export type Actor = "RED" | "BLUE";

export type ReplayEvent = {
  step: number;
  actor: Actor;
  action: string;
  target: string;
};

export type ReplayTopology = {
  nodes: Array<{
    id: string;
    zone?: string;
    assetType?: string;
  }>;
  edges: Array<{
    id: string;
    source: string;
    target: string;
  }>;
};

export type ReplayPayload = {
  events: ReplayEvent[];
  topology: ReplayTopology | null;
};

export type ReplayListItem = {
  id: string;
  runId?: string;
  scenarioId?: string;
  createdAt?: string;
  raw: unknown;
};

export type SandboxRunLifecycle =
  | "queued"
  | "running"
  | "completed"
  | "failed"
  | "cancelled";

export type Severity = "low" | "medium" | "high";

export type EpisodeNode = {
  id: string;
  severity: Severity;
  role?: string;
};

export type EpisodeVulnerability = {
  node_id: string;
  vuln_id: string;
  exploitability?: number;
};

export type RedObjective = {
  target_node_id: string;
  objective: string;
  priority?: number;
};

export type EpisodeSpec = {
  name: string;
  seed?: number;
  horizon: number;
  nodes: EpisodeNode[];
  vulnerabilities: EpisodeVulnerability[];
  red_objectives: RedObjective[];
  defender_mode: "aegis";
};

export type SandboxFormNode = {
  id: string;
  role?: string;
  os?: string;
  severity?: Severity;
};

export type SandboxFormVulnerability = {
  id: string;
  node_id: string;
  severity?: Severity;
  exploitability?: number;
};

export type SandboxFormObjective = {
  id: string;
  type: string;
  target: string;
  priority?: number;
};

export type SandboxFormSpec = {
  name: string;
  seed?: number;
  horizon: number;
  nodes: SandboxFormNode[];
  vulnerabilities: SandboxFormVulnerability[];
  red_objectives: SandboxFormObjective[];
  defender_mode: "aegis";
};

export type SandboxRunCreateRequest = {
  episode_spec: EpisodeSpec;
};

export type SandboxRunCreateResponse = {
  run_id: string;
  status: SandboxRunLifecycle;
  stream_url: string;
};

export type SandboxRunStatus = {
  run_id: string;
  status: SandboxRunLifecycle;
  started_at?: string | null;
  ended_at?: string | null;
  kpis?: Record<string, number | string | null> | null;
  error?: {
    code?: string;
    message?: string;
    details?: unknown[];
  } | null;
};

export type SandboxCatalogResponse = {
  vulnerabilities: string[];
  objectives: string[];
  execution_mode: "cluster" | "local";
  live_run_enabled: boolean;
  live_block_reason: string | null;
};

export type LiveStreamMessage =
  | { type: "action"; payload: Record<string, unknown> }
  | { type: "metric"; payload: Record<string, unknown> }
  | { type: "marker"; payload: Record<string, unknown> }
  | { type: string; payload: unknown };

export type BackendHealthStatus = {
  ok: boolean;
  endpoint: "/health" | "/healthz" | null;
  baseUrl: string;
  url: string | null;
  httpStatus: number | null;
  payload: unknown;
  error: string | null;
};

type StreamHandlers = {
  onEvent: (event: ReplayEvent) => void;
  onOpen?: () => void;
  onError?: (error: Event | Error) => void;
  onClose?: (event: CloseEvent) => void;
};

type StreamConnection = {
  close: () => void;
};

type LiveRunStreamHandlers = {
  onAction?: (event: ReplayEvent, payload: Record<string, unknown>) => void;
  onMetric?: (payload: Record<string, unknown>) => void;
  onMarker?: (payload: Record<string, unknown>) => void;
  onUnknownType?: (messageType: string, payload: unknown) => void;
  onOpen?: () => void;
  onError?: (error: Event | Error) => void;
  onClose?: (event: CloseEvent) => void;
};

type EventFetchOptions = {
  replayId?: string;
  signal?: AbortSignal;
};

type EpisodeStitchState = {
  activeEpisodeKey: string | null;
  episodeOffset: number;
  lastEpisodeStep: number;
  lastGlobalStep: number;
};

const DEFAULT_API_BASE_URL = "http://127.0.0.1:8000";
const DEFAULT_REST_EVENTS_PATH = "/events";
const DEFAULT_WS_EVENTS_PATH = "/ws/events";
const DEFAULT_REPLAY_LIST_PATH = "/replay/list";
const DEFAULT_REPLAY_BUNDLE_PATH_TEMPLATE = "/replay/{id}/bundle";
const DEFAULT_WS_REPLAY_PATH_TEMPLATE = "/stream/replay/{id}";
const DEFAULT_WS_LIVE_PATH_TEMPLATE = "/stream/live/{session_id}";
const DEFAULT_WS_LIVE_RUN_PATH_TEMPLATE = "/stream/live/{run_id}";
const DEFAULT_SANDBOX_RUNS_PATH = "/sandbox/runs";
const DEFAULT_SANDBOX_RUN_STATUS_PATH_TEMPLATE = "/sandbox/runs/{run_id}";
const DEFAULT_SANDBOX_RUN_CANCEL_PATH_TEMPLATE = "/sandbox/runs/{run_id}/cancel";
const DEFAULT_SANDBOX_CATALOG_PATH = "/sandbox/catalog";
const LOCAL_REPLAY_LIST_PATH = "/api/local/replay/list";
const LOCAL_REPLAY_BUNDLE_PATH_TEMPLATE = "/api/local/replay/{id}/bundle";

function normalizePath(path: string): string {
  if (!path) return "/";
  return path.startsWith("/") ? path : `/${path}`;
}

function replacePathParam(template: string, paramName: string, value: string): string {
  return template.replace(`{${paramName}}`, encodeURIComponent(value));
}

function asRecord(value: unknown): Record<string, unknown> | null {
  if (!value || typeof value !== "object" || Array.isArray(value)) return null;
  return value as Record<string, unknown>;
}

function toNonEmptyString(value: unknown): string | null {
  if (typeof value !== "string") return null;
  const trimmed = value.trim();
  return trimmed.length > 0 ? trimmed : null;
}

function toStep(value: unknown, fallbackStep: number): number {
  if (typeof value === "number" && Number.isFinite(value) && value >= 0) {
    return Math.floor(value);
  }

  if (typeof value === "string") {
    const parsed = Number.parseInt(value, 10);
    if (Number.isFinite(parsed) && parsed >= 0) {
      return parsed;
    }
  }

  return fallbackStep;
}

function toFiniteNumber(value: unknown): number | null {
  if (typeof value === "number" && Number.isFinite(value)) return value;
  if (typeof value === "string") {
    const parsed = Number.parseFloat(value);
    if (Number.isFinite(parsed)) return parsed;
  }
  return null;
}

function normalizeSeverity(value: unknown): Severity {
  const normalized = toNonEmptyString(value)?.toLowerCase();
  if (normalized === "low" || normalized === "medium" || normalized === "high") {
    return normalized;
  }
  return "medium";
}

export function mapSandboxFormToEpisodeSpec(form: SandboxFormSpec): EpisodeSpec {
  return {
    name: form.name.trim(),
    seed: form.seed,
    horizon: form.horizon,
    nodes: form.nodes.map((node) => ({
      id: node.id.trim(),
      role: toNonEmptyString(node.role) ?? undefined,
      severity: normalizeSeverity(node.severity),
    })),
    vulnerabilities: form.vulnerabilities.map((vulnerability) => {
      const mapped: EpisodeVulnerability = {
        node_id: vulnerability.node_id.trim(),
        vuln_id: vulnerability.id.trim(),
      };
      const exploitability = toFiniteNumber(vulnerability.exploitability);
      if (exploitability !== null && exploitability >= 0 && exploitability <= 1) {
        mapped.exploitability = exploitability;
      }
      return mapped;
    }),
    red_objectives: form.red_objectives.map((objective) => {
      const mapped: RedObjective = {
        target_node_id: objective.target.trim(),
        objective: objective.type.trim(),
      };
      const priority = toFiniteNumber(objective.priority);
      if (priority !== null) {
        mapped.priority = Math.trunc(priority);
      }
      return mapped;
    }),
    defender_mode: "aegis",
  };
}

export function toSandboxRunCreateRequest(episodeSpec: EpisodeSpec): SandboxRunCreateRequest {
  return {
    episode_spec: episodeSpec,
  };
}

function toActor(value: unknown): Actor | null {
  const normalized = toNonEmptyString(value)?.toUpperCase();
  if (!normalized) return null;

  if (
    normalized === "RED" ||
    normalized === "RED-TEAM" ||
    normalized === "REDTEAM" ||
    normalized === "ATTACKER" ||
    normalized === "ADVERSARY" ||
    normalized === "OFFENSE" ||
    normalized === "OFFENSIVE"
  ) {
    return "RED";
  }

  if (
    normalized === "BLUE" ||
    normalized === "BLUE-TEAM" ||
    normalized === "BLUETEAM" ||
    normalized === "DEFENDER" ||
    normalized === "DEFENCE" ||
    normalized === "DEFENSE" ||
    normalized === "SOC"
  ) {
    return "BLUE";
  }

  return null;
}

function inferActor(action: string): Actor {
  const actionLower = action.toLowerCase();

  if (
    actionLower.includes("isolate") ||
    actionLower.includes("monitor") ||
    actionLower.includes("detect") ||
    actionLower.includes("block") ||
    actionLower.includes("defend")
  ) {
    return "BLUE";
  }

  return "RED";
}

function normalizeTargetId(rawTarget: string): string {
  const target = rawTarget.trim();
  if (!target) return target;

  // Normalize host_1 / host-1 / host01 / host_01 => host-01
  const hostMatch = /^host[-_]?(\d{1,2})$/i.exec(target);
  if (hostMatch) {
    const index = Number.parseInt(hostMatch[1], 10);
    if (Number.isFinite(index)) {
      return `host-${String(index).padStart(2, "0")}`;
    }
  }

  return target;
}

function resolveApiBaseUrl(): string {
  return process.env.NEXT_PUBLIC_API_BASE_URL ?? DEFAULT_API_BASE_URL;
}

function resolveUrl(pathOrUrl: string): string {
  if (pathOrUrl.startsWith("http://") || pathOrUrl.startsWith("https://")) {
    return pathOrUrl;
  }

  const base = resolveApiBaseUrl();
  return new URL(normalizePath(pathOrUrl), base).toString();
}

function resolveWsUrl(pathOrUrl: string): string {
  if (pathOrUrl.startsWith("ws://") || pathOrUrl.startsWith("wss://")) {
    return pathOrUrl;
  }

  if (pathOrUrl.startsWith("http://") || pathOrUrl.startsWith("https://")) {
    const parsed = new URL(pathOrUrl);
    parsed.protocol = parsed.protocol === "https:" ? "wss:" : "ws:";
    return parsed.toString();
  }

  const base = new URL(resolveApiBaseUrl());
  const protocol = base.protocol === "https:" ? "wss:" : "ws:";
  return `${protocol}//${base.host}${normalizePath(pathOrUrl)}`;
}

function resolveRestEventsUrl(): string {
  const configured = process.env.NEXT_PUBLIC_EVENTS_REST_PATH ?? DEFAULT_REST_EVENTS_PATH;
  return resolveUrl(configured);
}

function resolveReplayListUrl(): string {
  const configured = process.env.NEXT_PUBLIC_REPLAY_LIST_PATH ?? DEFAULT_REPLAY_LIST_PATH;
  return resolveUrl(configured);
}

function resolveReplayBundleUrl(replayId: string): string {
  const template =
    process.env.NEXT_PUBLIC_REPLAY_BUNDLE_PATH_TEMPLATE ??
    DEFAULT_REPLAY_BUNDLE_PATH_TEMPLATE;
  return resolveUrl(replacePathParam(template, "id", replayId));
}

function resolveReplayStreamUrl(replayId: string): string {
  const template =
    process.env.NEXT_PUBLIC_EVENTS_WS_REPLAY_PATH_TEMPLATE ??
    DEFAULT_WS_REPLAY_PATH_TEMPLATE;
  return resolveWsUrl(replacePathParam(template, "id", replayId));
}

function resolveLiveStreamUrl(sessionId: string): string {
  const template =
    process.env.NEXT_PUBLIC_EVENTS_WS_LIVE_PATH_TEMPLATE ??
    DEFAULT_WS_LIVE_PATH_TEMPLATE;
  return resolveWsUrl(replacePathParam(template, "session_id", sessionId));
}

function resolveLiveRunStreamUrl(runId: string): string {
  const template =
    process.env.NEXT_PUBLIC_SANDBOX_EVENTS_WS_LIVE_PATH_TEMPLATE ??
    process.env.NEXT_PUBLIC_EVENTS_WS_LIVE_RUN_PATH_TEMPLATE ??
    DEFAULT_WS_LIVE_RUN_PATH_TEMPLATE;
  return resolveWsUrl(replacePathParam(template, "run_id", runId));
}

function resolveSandboxRunsUrl(): string {
  const configured = process.env.NEXT_PUBLIC_SANDBOX_RUNS_PATH ?? DEFAULT_SANDBOX_RUNS_PATH;
  return resolveUrl(configured);
}

function resolveSandboxRunStatusUrl(runId: string): string {
  const template =
    process.env.NEXT_PUBLIC_SANDBOX_RUN_STATUS_PATH_TEMPLATE ??
    DEFAULT_SANDBOX_RUN_STATUS_PATH_TEMPLATE;
  return resolveUrl(replacePathParam(template, "run_id", runId));
}

function resolveSandboxRunCancelUrl(runId: string): string {
  const template =
    process.env.NEXT_PUBLIC_SANDBOX_RUN_CANCEL_PATH_TEMPLATE ??
    DEFAULT_SANDBOX_RUN_CANCEL_PATH_TEMPLATE;
  return resolveUrl(replacePathParam(template, "run_id", runId));
}

function resolveSandboxCatalogUrl(): string {
  const configured =
    process.env.NEXT_PUBLIC_SANDBOX_CATALOG_PATH ?? DEFAULT_SANDBOX_CATALOG_PATH;
  return resolveUrl(configured);
}

function resolveDefaultEventsStreamUrl(): string {
  const configured = process.env.NEXT_PUBLIC_EVENTS_WS_PATH ?? DEFAULT_WS_EVENTS_PATH;
  return resolveWsUrl(configured);
}

function resolveLocalReplayListUrl(): string {
  return LOCAL_REPLAY_LIST_PATH;
}

function resolveLocalReplayBundleUrl(replayId: string): string {
  return replacePathParam(LOCAL_REPLAY_BUNDLE_PATH_TEMPLATE, "id", replayId);
}

export function normalizeEvent(
  rawEvent: unknown,
  fallbackStep: number,
): ReplayEvent | null {
  const event = asRecord(rawEvent);
  if (!event) return null;

  const action =
    toNonEmptyString(event.action) ??
    toNonEmptyString(event.action_type) ??
    toNonEmptyString(event.event_type) ??
    toNonEmptyString(event.signal) ??
    "unknown_action";

  const target =
    normalizeTargetId(
      toNonEmptyString(event.target) ??
        toNonEmptyString(event.target_host) ??
        toNonEmptyString(event.host) ??
        toNonEmptyString(event.node_id) ??
        toNonEmptyString(event.target_node) ??
        "unknown_target",
    );

  const actor =
    toActor(event.actor) ??
    toActor(event.detector) ??
    toActor(event.team) ??
    toActor(event.side) ??
    inferActor(action);

  const step = toStep(
    event.step ?? event.seq ?? event.sequence ?? event.index,
    fallbackStep,
  );

  return { step, actor, action, target };
}

function extractRawEvents(payload: unknown): unknown[] {
  if (Array.isArray(payload)) return payload;

  const record = asRecord(payload);
  if (!record) return [];

  if (Array.isArray(record.events)) return record.events;
  if (Array.isArray(record.data)) return record.data;
  if (Array.isArray(record.results)) return record.results;

  const embeddedEvent = asRecord(record.event);
  if (embeddedEvent) return [embeddedEvent];

  return [payload];
}

function extractEpisodeKey(rawEvent: unknown): string | null {
  const event = asRecord(rawEvent);
  if (!event) return null;

  const directEpisode =
    toNonEmptyString(event.episode_id) ??
    toNonEmptyString(event.episodeId) ??
    toNonEmptyString(event.episode) ??
    toNonEmptyString(event.trajectory_id);

  if (directEpisode) return directEpisode;

  const runId = toNonEmptyString(event.run_id) ?? toNonEmptyString(event.runId);
  const replayId =
    toNonEmptyString(event.replay_id) ?? toNonEmptyString(event.replayId);

  if (runId && replayId) return `${runId}:${replayId}`;
  return runId ?? replayId ?? null;
}

function createEpisodeStitchState(): EpisodeStitchState {
  return {
    activeEpisodeKey: null,
    episodeOffset: 0,
    lastEpisodeStep: 0,
    lastGlobalStep: 0,
  };
}

function stitchEventStep(
  event: ReplayEvent,
  rawEvent: unknown,
  state: EpisodeStitchState,
): ReplayEvent {
  const episodeKey = extractEpisodeKey(rawEvent);

  if (state.activeEpisodeKey === null) {
    state.activeEpisodeKey = episodeKey ?? "default";
  }

  const switchedEpisode =
    episodeKey !== null &&
    state.activeEpisodeKey !== null &&
    episodeKey !== state.activeEpisodeKey;

  const impliedBoundary =
    episodeKey === null &&
    state.lastEpisodeStep > 0 &&
    event.step <= state.lastEpisodeStep;

  if (switchedEpisode || impliedBoundary) {
    state.episodeOffset = state.lastGlobalStep;
    state.lastEpisodeStep = 0;
    if (episodeKey) {
      state.activeEpisodeKey = episodeKey;
    }
  }

  let withinEpisodeStep = event.step;
  if (withinEpisodeStep <= 0) {
    withinEpisodeStep = state.lastEpisodeStep + 1;
  }

  if (withinEpisodeStep <= state.lastEpisodeStep) {
    withinEpisodeStep = state.lastEpisodeStep + 1;
  }

  state.lastEpisodeStep = withinEpisodeStep;

  let globalStep = state.episodeOffset + withinEpisodeStep;
  if (globalStep <= state.lastGlobalStep) {
    globalStep = state.lastGlobalStep + 1;
  }

  state.lastGlobalStep = globalStep;

  return {
    ...event,
    step: globalStep,
  };
}

function normalizeEvents(
  rawEvents: unknown[],
  stitchState = createEpisodeStitchState(),
): ReplayEvent[] {
  const normalizedWithRaw = rawEvents
    .map((rawEvent, idx) => {
      const event = normalizeEvent(rawEvent, idx + 1);
      if (!event) return null;
      return { event, rawEvent };
    })
    .filter(
      (entry): entry is { event: ReplayEvent; rawEvent: unknown } => entry !== null,
    );

  return normalizedWithRaw.map(({ event, rawEvent }) =>
    stitchEventStep(event, rawEvent, stitchState),
  );
}

function parseJsonLines(text: string): unknown[] {
  return text
    .split("\n")
    .map((line) => line.trim())
    .filter((line) => line.length > 0)
    .map((line) => {
      try {
        return JSON.parse(line) as unknown;
      } catch {
        return null;
      }
    })
    .filter((value): value is unknown => value !== null);
}

function parseReplayList(payload: unknown): ReplayListItem[] {
  const container = asRecord(payload);
  const rawList: unknown[] = Array.isArray(payload)
    ? payload
    : Array.isArray(container?.replays)
      ? container.replays
      : Array.isArray(container?.data)
        ? container.data
        : Array.isArray(container?.items)
          ? container.items
          : [];

  const parsed: ReplayListItem[] = [];
  rawList.forEach((item, idx) => {
    const record = asRecord(item);
    if (!record) return;

    const id =
      toNonEmptyString(record.id) ??
      toNonEmptyString(record.replay_id) ??
      toNonEmptyString(record.replayId) ??
      toNonEmptyString(record.name) ??
      `replay-${idx + 1}`;

    const runId = toNonEmptyString(record.run_id) ?? toNonEmptyString(record.runId) ?? undefined;
    const scenarioId =
      toNonEmptyString(record.scenario_id) ?? toNonEmptyString(record.scenarioId) ?? undefined;

    const createdAt =
      toNonEmptyString(record.created_at) ?? toNonEmptyString(record.createdAt) ?? undefined;

    parsed.push({ id, runId, scenarioId, createdAt, raw: item });
  });

  return parsed;
}

function extractInlineEvents(bundlePayload: unknown): unknown[] {
  const bundle = asRecord(bundlePayload);
  if (!bundle) return [];

  if (Array.isArray(bundle.events)) return bundle.events;

  const replay = asRecord(bundle.replay);
  if (replay && Array.isArray(replay.events)) return replay.events;

  const data = asRecord(bundle.data);
  if (data && Array.isArray(data.events)) return data.events;

  return [];
}

function extractTopology(bundlePayload: unknown): ReplayTopology | null {
  const bundle = asRecord(bundlePayload);
  if (!bundle) return null;

  const topologyContainer =
    asRecord(bundle.topology) ??
    asRecord(asRecord(bundle.data)?.topology) ??
    asRecord(asRecord(bundle.replay)?.topology);

  if (!topologyContainer) return null;

  const initial = asRecord(topologyContainer.initial) ?? topologyContainer;
  const nodesRaw = Array.isArray(initial.nodes) ? initial.nodes : [];
  const edgesRaw = Array.isArray(initial.edges) ? initial.edges : [];

  const nodes = nodesRaw
    .map((node) => {
      const record = asRecord(node);
      if (!record) return null;
      const id = toNonEmptyString(record.node_id) ?? toNonEmptyString(record.id);
      if (!id) return null;
      return {
        id: normalizeTargetId(id),
        zone: toNonEmptyString(record.zone) ?? undefined,
        assetType: toNonEmptyString(record.asset_type) ?? undefined,
      };
    })
    .filter((node) => node !== null) as ReplayTopology["nodes"];

  const edges = edgesRaw
    .map((edge) => {
      const record = asRecord(edge);
      if (!record) return null;
      const source = toNonEmptyString(record.source);
      const target = toNonEmptyString(record.target);
      if (!source || !target) return null;
      const edgeId =
        toNonEmptyString(record.edge_id) ??
        toNonEmptyString(record.id) ??
        `${normalizeTargetId(source)}=>${normalizeTargetId(target)}`;
      return {
        id: edgeId,
        source: normalizeTargetId(source),
        target: normalizeTargetId(target),
      };
    })
    .filter((edge) => edge !== null) as ReplayTopology["edges"];

  if (nodes.length === 0) return null;
  return { nodes, edges };
}

function extractEventPointer(bundlePayload: unknown): string | null {
  const bundle = asRecord(bundlePayload);
  if (!bundle) return null;

  const directPointer =
    toNonEmptyString(bundle.events_jsonl_url) ??
    toNonEmptyString(bundle.events_url) ??
    toNonEmptyString(bundle.events_path) ??
    toNonEmptyString(bundle.events_jsonl);

  if (directPointer) return directPointer;

  const artifacts = asRecord(bundle.artifacts);
  if (artifacts) {
    const artifactPointer =
      toNonEmptyString(artifacts.events_jsonl_url) ??
      toNonEmptyString(artifacts.events_url) ??
      toNonEmptyString(artifacts.events_path) ??
      toNonEmptyString(artifacts.events_jsonl);

    if (artifactPointer) return artifactPointer;

    const replayArtifacts = asRecord(artifacts.replays);
    if (replayArtifacts) {
      const replayPointer =
        toNonEmptyString(replayArtifacts.events_jsonl_url) ??
        toNonEmptyString(replayArtifacts.events_url) ??
        toNonEmptyString(replayArtifacts.events_path) ??
        toNonEmptyString(replayArtifacts.events_jsonl);

      if (replayPointer) return replayPointer;
    }
  }

  const eventsAsString = toNonEmptyString(bundle.events);
  return eventsAsString ?? null;
}

async function fetchUnknownJson(url: string, signal?: AbortSignal): Promise<unknown> {
  const response = await fetch(url, {
    method: "GET",
    headers: { Accept: "application/json" },
    cache: "no-store",
    signal,
  });

  if (!response.ok) {
    throw new Error(`Failed request (${response.status}) for ${url}`);
  }

  return (await response.json()) as unknown;
}

function parseDetailItem(detail: unknown): string | null {
  const detailRecord = asRecord(detail);
  if (!detailRecord) {
    return toNonEmptyString(detail);
  }

  const msg =
    toNonEmptyString(detailRecord.msg) ??
    toNonEmptyString(detailRecord.message) ??
    null;
  const loc =
    Array.isArray(detailRecord.loc)
      ? detailRecord.loc
          .map((part) => (typeof part === "string" || typeof part === "number" ? String(part) : ""))
          .filter((part) => part.length > 0)
          .join(".")
      : null;

  if (loc && msg) return `${loc}: ${msg}`;
  return msg ?? toNonEmptyString(detailRecord.type);
}

export function extractApiErrorMessage(payload: unknown): string | null {
  const record = asRecord(payload);
  if (!record) return null;

  const direct = toNonEmptyString(record.message);
  if (direct) return direct;

  const detail = record.detail;
  if (Array.isArray(detail)) {
    const entries = detail
      .map((item) => parseDetailItem(item))
      .filter((item): item is string => Boolean(item));
    if (entries.length > 0) return entries.join("; ");
  } else {
    const singleDetail = parseDetailItem(detail);
    if (singleDetail) return singleDetail;
  }

  const errorRecord = asRecord(record.error);
  if (!errorRecord) return toNonEmptyString(record.error);

  return toNonEmptyString(errorRecord.message) ?? toNonEmptyString(errorRecord.code) ?? null;
}

async function readErrorMessage(response: Response, fallback: string): Promise<string> {
  const contentType = response.headers.get("content-type")?.toLowerCase() ?? "";
  if (contentType.includes("application/json")) {
    try {
      const payload = (await response.json()) as unknown;
      const parsed = extractApiErrorMessage(payload);
      if (parsed) return parsed;
    } catch {
      return fallback;
    }
  }
  return fallback;
}

async function probeHealthEndpoint(
  endpoint: "/health" | "/healthz",
  signal?: AbortSignal,
): Promise<BackendHealthStatus> {
  const url = resolveUrl(endpoint);

  try {
    const response = await fetch(url, {
      method: "GET",
      headers: { Accept: "application/json,text/plain" },
      cache: "no-store",
      signal,
    });

    const textBody = await response.text();
    let payload: unknown = textBody;
    if (textBody.trim().startsWith("{") || textBody.trim().startsWith("[")) {
      try {
        payload = JSON.parse(textBody) as unknown;
      } catch {
        payload = textBody;
      }
    }

    return {
      ok: response.ok,
      endpoint,
      baseUrl: resolveApiBaseUrl(),
      url,
      httpStatus: response.status,
      payload,
      error: response.ok ? null : `Health probe failed (${response.status})`,
    };
  } catch (error) {
    return {
      ok: false,
      endpoint,
      baseUrl: resolveApiBaseUrl(),
      url,
      httpStatus: null,
      payload: null,
      error: error instanceof Error ? error.message : "Health probe request failed",
    };
  }
}

function parseSandboxRunCreate(payload: unknown): SandboxRunCreateResponse {
  const record = asRecord(payload);
  if (!record) {
    throw new Error("Invalid sandbox create response");
  }

  const runId = toNonEmptyString(record.run_id);
  const status = toNonEmptyString(record.status) as SandboxRunLifecycle | null;
  const streamUrl = toNonEmptyString(record.stream_url);

  if (!runId || !status || !streamUrl) {
    throw new Error("Sandbox create response missing required fields");
  }

  return { run_id: runId, status, stream_url: streamUrl };
}

function parseSandboxRunStatus(payload: unknown): SandboxRunStatus {
  const record = asRecord(payload);
  if (!record) {
    throw new Error("Invalid sandbox status response");
  }

  const runId = toNonEmptyString(record.run_id);
  const status = toNonEmptyString(record.status) as SandboxRunLifecycle | null;
  if (!runId || !status) {
    throw new Error("Sandbox status response missing required fields");
  }

  const errorRecord = asRecord(record.error);
  const errorString = toNonEmptyString(record.error);
  const kpisRecord = asRecord(record.kpis);

  return {
    run_id: runId,
    status,
    started_at: toNonEmptyString(record.started_at),
    ended_at: toNonEmptyString(record.ended_at),
    kpis: kpisRecord as SandboxRunStatus["kpis"],
    error:
      errorRecord
        ? {
            code: toNonEmptyString(errorRecord.code) ?? undefined,
            message: toNonEmptyString(errorRecord.message) ?? undefined,
            details: Array.isArray(errorRecord.details) ? errorRecord.details : undefined,
          }
        : errorString
          ? {
              message: errorString,
            }
          : null,
  };
}

function parseSandboxCatalog(payload: unknown): SandboxCatalogResponse {
  const record = asRecord(payload);
  if (!record) {
    throw new Error("Invalid sandbox catalog response");
  }

  const vulnerabilities = Array.isArray(record.vulnerabilities)
    ? record.vulnerabilities
        .map((item) => toNonEmptyString(item))
        .filter((item): item is string => Boolean(item))
    : Array.isArray(record.vulnerability_templates)
      ? record.vulnerability_templates
          .map((item) => toNonEmptyString(asRecord(item)?.id))
          .filter((item): item is string => Boolean(item))
      : [];

  const objectives = Array.isArray(record.objectives)
    ? record.objectives
        .map((item) => toNonEmptyString(item))
        .filter((item): item is string => Boolean(item))
    : Array.isArray(record.objective_templates)
      ? record.objective_templates
          .map((item) => toNonEmptyString(asRecord(item)?.type))
          .filter((item): item is string => Boolean(item))
      : [];

  const executionMode = toNonEmptyString(record.execution_mode);
  const liveRunEnabled =
    typeof record.live_run_enabled === "boolean"
      ? record.live_run_enabled
      : executionMode === "cluster";
  const liveBlockReason = toNonEmptyString(record.live_block_reason);

  return {
    vulnerabilities,
    objectives,
    execution_mode: executionMode === "cluster" ? "cluster" : "local",
    live_run_enabled: liveRunEnabled,
    live_block_reason: liveBlockReason,
  };
}

async function fetchEventsFromPointer(
  pointer: string,
  signal?: AbortSignal,
): Promise<ReplayEvent[]> {
  const url = resolveUrl(pointer);
  const response = await fetch(url, {
    method: "GET",
    headers: {
      Accept: "application/json,text/plain",
    },
    cache: "no-store",
    signal,
  });

  if (!response.ok) {
    throw new Error(`Failed to fetch events artifact (${response.status})`);
  }

  const contentType = response.headers.get("content-type")?.toLowerCase() ?? "";
  const bodyText = await response.text();

  if (contentType.includes("application/json")) {
    const payload = JSON.parse(bodyText) as unknown;
    return normalizeEvents(extractRawEvents(payload));
  }

  const trimmed = bodyText.trim();
  if (!trimmed) return [];

  if (trimmed.startsWith("{") || trimmed.startsWith("[")) {
    const parsed = JSON.parse(trimmed) as unknown;
    return normalizeEvents(extractRawEvents(parsed));
  }

  return normalizeEvents(parseJsonLines(trimmed));
}

export async function fetchReplayList(signal?: AbortSignal): Promise<ReplayListItem[]> {
  try {
    const payload = await fetchUnknownJson(resolveReplayListUrl(), signal);
    return parseReplayList(payload);
  } catch {
    const payload = await fetchUnknownJson(resolveLocalReplayListUrl(), signal);
    return parseReplayList(payload);
  }
}

export async function checkBackendHealth(
  signal?: AbortSignal,
): Promise<BackendHealthStatus> {
  const health = await probeHealthEndpoint("/health", signal);
  if (health.ok) return health;

  const healthz = await probeHealthEndpoint("/healthz", signal);
  if (healthz.ok) return healthz;

  return {
    ...healthz,
    error: [health.error, healthz.error].filter(Boolean).join("; ") || "Backend health probe failed",
  };
}

export function pickPreferredReplayId(list: ReplayListItem[]): string | null {
  const envReplayId = process.env.NEXT_PUBLIC_REPLAY_ID;
  if (envReplayId) return envReplayId;
  if (list.length === 0) return null;

  const sorted = [...list].sort((a, b) => {
    const aTs = a.createdAt ? Date.parse(a.createdAt) : Number.NaN;
    const bTs = b.createdAt ? Date.parse(b.createdAt) : Number.NaN;

    if (Number.isFinite(aTs) && Number.isFinite(bTs)) {
      return bTs - aTs;
    }

    if (Number.isFinite(aTs)) return -1;
    if (Number.isFinite(bTs)) return 1;
    return 0;
  });

  return sorted[0]?.id ?? null;
}

export type ScenarioLane = "baseline" | "current" | "enterprise";

export function pickReplayIdForLane(
  list: ReplayListItem[],
  lane: ScenarioLane,
): string | null {
  if (list.length === 0) return null;
  if (lane === "baseline") return null;

  const isEnterprise = (item: ReplayListItem) =>
    (item.scenarioId ?? "").startsWith("scenario_enterprise_");

  if (lane === "enterprise") {
    const enterpriseReplay = list.find(isEnterprise);
    return enterpriseReplay?.id ?? pickPreferredReplayId(list);
  }

  const currentReplay = list.find((item) => !isEnterprise(item));
  return currentReplay?.id ?? pickPreferredReplayId(list);
}

export async function fetchEvents(
  options: EventFetchOptions = {},
): Promise<ReplayEvent[]> {
  const payload = await fetchReplayPayload(options);
  return payload.events;
}

export async function fetchReplayPayload(
  options: EventFetchOptions = {},
): Promise<ReplayPayload> {
  const { replayId, signal } = options;

  if (replayId) {
    try {
      const bundlePayload = await fetchUnknownJson(resolveReplayBundleUrl(replayId), signal);
      const topology = extractTopology(bundlePayload);

      const inlineEvents = extractInlineEvents(bundlePayload);
      if (inlineEvents.length > 0) {
        return {
          events: normalizeEvents(inlineEvents),
          topology,
        };
      }

      const pointer = extractEventPointer(bundlePayload);
      if (pointer) {
        const pointerEvents = await fetchEventsFromPointer(pointer, signal);
        if (pointerEvents.length > 0) {
          return {
            events: pointerEvents,
            topology,
          };
        }
      }
    } catch {
      // Fall through to local runs bundle and then generic events endpoint.
    }

    try {
      const localBundle = await fetchUnknownJson(resolveLocalReplayBundleUrl(replayId), signal);
      const topology = extractTopology(localBundle);

      const inlineEvents = extractInlineEvents(localBundle);
      if (inlineEvents.length > 0) {
        return {
          events: normalizeEvents(inlineEvents),
          topology,
        };
      }

      const pointer = extractEventPointer(localBundle);
      if (pointer) {
        const pointerEvents = await fetchEventsFromPointer(pointer, signal);
        if (pointerEvents.length > 0) {
          return {
            events: pointerEvents,
            topology,
          };
        }
      }
    } catch {
      // Fall back to generic events endpoint.
    }
  }

  const payload = await fetchUnknownJson(resolveRestEventsUrl(), signal);
  return {
    events: normalizeEvents(extractRawEvents(payload)),
    topology: null,
  };
}

export async function fetchSandboxCatalog(signal?: AbortSignal): Promise<SandboxCatalogResponse> {
  const payload = await fetchUnknownJson(resolveSandboxCatalogUrl(), signal);
  return parseSandboxCatalog(payload);
}

export async function createSandboxRun(
  episodeSpec: EpisodeSpec,
  signal?: AbortSignal,
): Promise<SandboxRunCreateResponse> {
  const response = await fetch(resolveSandboxRunsUrl(), {
    method: "POST",
    headers: {
      "Content-Type": "application/json",
      Accept: "application/json",
    },
    body: JSON.stringify(toSandboxRunCreateRequest(episodeSpec)),
    signal,
  });

  if (!response.ok) {
    throw new Error(
      await readErrorMessage(response, `Sandbox run creation failed (${response.status})`),
    );
  }

  const payload = (await response.json()) as unknown;
  return parseSandboxRunCreate(payload);
}

export async function fetchSandboxRunStatus(
  runId: string,
  signal?: AbortSignal,
): Promise<SandboxRunStatus> {
  const response = await fetch(resolveSandboxRunStatusUrl(runId), {
    method: "GET",
    headers: { Accept: "application/json" },
    cache: "no-store",
    signal,
  });

  if (!response.ok) {
    throw new Error(await readErrorMessage(response, `Sandbox status failed (${response.status})`));
  }

  const payload = (await response.json()) as unknown;
  return parseSandboxRunStatus(payload);
}

export async function cancelSandboxRun(
  runId: string,
  signal?: AbortSignal,
): Promise<SandboxRunStatus> {
  const response = await fetch(resolveSandboxRunCancelUrl(runId), {
    method: "POST",
    headers: {
      "Content-Type": "application/json",
      Accept: "application/json",
    },
    body: "{}",
    signal,
  });

  if (!response.ok) {
    throw new Error(
      await readErrorMessage(response, `Sandbox cancel request failed (${response.status})`),
    );
  }

  const payload = (await response.json()) as unknown;
  return parseSandboxRunStatus(payload);
}

function toLiveRunMessage(payload: unknown): LiveStreamMessage | null {
  const message = asRecord(payload);
  if (!message) return null;

  const type = toNonEmptyString(message.type);
  if (!type) return null;

  const rawPayload = message.payload;
  return { type, payload: rawPayload } as LiveStreamMessage;
}

function connectStream(url: string, handlers: StreamHandlers): StreamConnection {
  if (typeof window === "undefined" || typeof WebSocket === "undefined") {
    handlers.onError?.(new Error("WebSocket is not available"));
    return { close: () => undefined };
  }

  const stitchState = createEpisodeStitchState();
  const socket = new WebSocket(url);

  socket.onopen = () => {
    handlers.onOpen?.();
  };

  socket.onmessage = (message) => {
    if (typeof message.data !== "string") return;

    let payload: unknown;
    try {
      payload = JSON.parse(message.data);
    } catch {
      return;
    }

    const rawEvents = extractRawEvents(payload);
    const events = normalizeEvents(rawEvents, stitchState);

    events.forEach((event) => {
      handlers.onEvent(event);
    });
  };

  socket.onerror = (event) => {
    handlers.onError?.(event);
  };

  socket.onclose = (event) => {
    handlers.onClose?.(event);
  };

  return {
    close: () => {
      if (
        socket.readyState === WebSocket.OPEN ||
        socket.readyState === WebSocket.CONNECTING
      ) {
        socket.close();
      }
    },
  };
}

export function connectReplayStream(
  replayId: string,
  handlers: StreamHandlers,
): StreamConnection {
  return connectStream(resolveReplayStreamUrl(replayId), handlers);
}

export function connectLiveStream(
  sessionId: string,
  handlers: StreamHandlers,
): StreamConnection {
  return connectStream(resolveLiveStreamUrl(sessionId), handlers);
}

export function connectLiveRunStream(
  runId: string,
  handlers: LiveRunStreamHandlers,
): StreamConnection {
  if (typeof window === "undefined" || typeof WebSocket === "undefined") {
    handlers.onError?.(new Error("WebSocket is not available"));
    return { close: () => undefined };
  }

  const socket = new WebSocket(resolveLiveRunStreamUrl(runId));
  let fallbackStep = 1;

  socket.onopen = () => {
    handlers.onOpen?.();
  };

  socket.onmessage = (message) => {
    if (typeof message.data !== "string") return;

    let parsedPayload: unknown;
    try {
      parsedPayload = JSON.parse(message.data);
    } catch {
      return;
    }

    const liveMessage = toLiveRunMessage(parsedPayload);
    if (!liveMessage) return;

    const payloadRecord = asRecord(liveMessage.payload);

    if (liveMessage.type === "action" && payloadRecord) {
      const event = normalizeEvent(payloadRecord, fallbackStep);
      fallbackStep += 1;
      if (event) {
        handlers.onAction?.(event, payloadRecord);
      }
      return;
    }

    if (liveMessage.type === "metric" && payloadRecord) {
      handlers.onMetric?.(payloadRecord);
      return;
    }

    if (liveMessage.type === "marker" && payloadRecord) {
      handlers.onMarker?.(payloadRecord);
      return;
    }

    handlers.onUnknownType?.(liveMessage.type, liveMessage.payload);
  };

  socket.onerror = (event) => {
    handlers.onError?.(event);
  };

  socket.onclose = (event) => {
    handlers.onClose?.(event);
  };

  return {
    close: () => {
      if (
        socket.readyState === WebSocket.OPEN ||
        socket.readyState === WebSocket.CONNECTING
      ) {
        socket.close();
      }
    },
  };
}

export function connectEventsStream(handlers: StreamHandlers): StreamConnection {
  const explicitWsUrl = process.env.NEXT_PUBLIC_EVENTS_WS_URL;
  const url = explicitWsUrl
    ? resolveWsUrl(explicitWsUrl)
    : resolveDefaultEventsStreamUrl();

  return connectStream(url, handlers);
}
