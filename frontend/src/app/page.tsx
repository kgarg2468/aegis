"use client";

import { useCallback, useEffect, useMemo, useState } from "react";
import Graph from "../components/Graph";
import type { ActionEvent, EdgeVisualState, ExplainabilityRecord, MetricsTick } from "../lib/integrationContract";
import {
  type LoadedScenarioRun,
  type ScenarioProfileMode,
  type SelectedScenarioRun,
  deriveActiveCampaignStage,
  loadScenarioRun,
} from "../lib/replayAdapter";
import {
  type ReplayTimeline,
  type RuntimeGraphState,
  collectActionsUntilStep,
  materializeGraphAtStep,
  maxReplayStep,
} from "../lib/replayRuntime";
import {
  CAMPAIGN_ALL_5,
  SCENARIO_REGISTRY,
  type ScenarioSelection,
} from "../lib/scenarios";

type TabId = "command" | "episodes" | "mission";
type ReplaySpeed = 0.5 | 1 | 2 | 4;
type LogFilter = "none" | `node:${string}`;

interface RuntimeState {
  timeline: ReplayTimeline;
  graphState: RuntimeGraphState;
  actions: ActionEvent[];
  metrics: MetricsTick | null;
  step: number;
}

interface DisplayConnection {
  neighbor_id: string;
  visual_state: EdgeVisualState;
}

const PROFILE_MODES: Array<{
  id: ScenarioProfileMode;
  label: string;
  subtitle: string;
}> = [
  {
    id: "current_run_enterprise",
    label: "Current Run (Enterprise Level)",
    subtitle: "Defended enterprise profile with identity + trust-path complexity",
  },
  {
    id: "no_blue",
    label: "No Blue Team",
    subtitle: "Red-only baseline for undefended comparison",
  },
];

const TABS: Array<{ id: TabId; label: string; subtitle: string }> = [
  { id: "command", label: "Command Grid", subtitle: "Live topology" },
  { id: "episodes", label: "Episode Vault", subtitle: "Select scenario" },
  { id: "mission", label: "Mission Brief", subtitle: "Plain-language explanation" },
];

const SPEEDS: ReplaySpeed[] = [0.5, 1, 2, 4];

const EDGE_STATE_PRIORITY: Record<EdgeVisualState, number> = {
  exfiltration: 6,
  lateral_movement: 5,
  credential_flow: 4,
  scanning: 3,
  blocked: 2,
  normal: 1,
};

function clamp(value: number, min: number, max: number): number {
  return Math.max(min, Math.min(max, value));
}

function percent(value: number): number {
  return Math.round(clamp(value, 0, 1) * 100);
}

function upper(value: string): string {
  return value.replaceAll("_", " ").toUpperCase();
}

function riskPercent(score: number): number {
  return Math.round(clamp(score, 0, 1) * 100);
}

function severityBadgeClass(severity: ActionEvent["severity"]): string {
  if (severity === "critical") return "border-[#7f8da4] bg-[#304158]/45 text-[#e2ebfb]";
  if (severity === "high") return "border-[#7f8da4] bg-[#253247]/45 text-[#d9e5f7]";
  if (severity === "medium") return "border-[#EF9F27] bg-[#EF9F27]/18 text-[#ffebb8]";
  if (severity === "low") return "border-[#1D9E75] bg-[#1D9E75]/15 text-[#ccffe9]";
  return "border-[#888780] bg-[#888780]/15 text-[#edece8]";
}

function actorBadgeClass(actor: ActionEvent["actor"]): string {
  if (actor === "RED") return "border-[#7f8da4] bg-[#253247]/50 text-[#d9e5f7]";
  if (actor === "BLUE") return "border-[#378ADD] bg-[#378ADD]/20 text-[#d6e9ff]";
  return "border-[#888780] bg-[#888780]/20 text-[#eceae5]";
}

function initialSelection(): SelectedScenarioRun {
  return {
    profile_mode: "current_run_enterprise",
    selection: {
      kind: "scenario",
      scenario_id: "faculty_spear_phish",
    },
    run_id: "run_current_default",
  };
}

function narrativeForMode(mode: ScenarioProfileMode): string {
  if (mode === "current_run_enterprise") {
    return "Current Run (Enterprise Level) shows defended behavior in a higher-complexity trust-path environment.";
  }
  return "No Blue Team removes active defense actions so you can show raw attacker momentum and blast radius.";
}

function modePlainLanguage(mode: ScenarioProfileMode): string {
  if (mode === "current_run_enterprise") {
    return "Defender enabled: AEGIS actively monitors, isolates, and patches during the attack.";
  }
  return "Defender disabled: this is the same attack surface without blue-team intervention.";
}

export default function Home() {
  const [tab, setTab] = useState<TabId>("command");
  const [selectedScenarioRun, setSelectedScenarioRun] = useState<SelectedScenarioRun>(initialSelection);
  const [loadedRun, setLoadedRun] = useState<LoadedScenarioRun | null>(null);
  const [statusLabel, setStatusLabel] = useState("LOADING");
  const [notice, setNotice] = useState<string | null>(null);

  const [runtimeState, setRuntimeState] = useState<RuntimeState | null>(null);
  const [isPlaying, setIsPlaying] = useState(true);
  const [speed, setSpeed] = useState<ReplaySpeed>(1);

  const [selectedNodeId, setSelectedNodeId] = useState<string | null>(null);
  const [highlightedNodeId, setHighlightedNodeId] = useState<string | null>(null);
  const [logFilter, setLogFilter] = useState<LogFilter>("none");
  const [syncDriftMs, setSyncDriftMs] = useState(0);
  const [isGraphFullscreen, setIsGraphFullscreen] = useState(false);
  const [resetViewToken, setResetViewToken] = useState(0);

  useEffect(() => {
    let cancelled = false;

    const load = async () => {
      setStatusLabel("LOADING");
      const loaded = await loadScenarioRun(selectedScenarioRun);
      if (cancelled) return;

      const initialGraph = materializeGraphAtStep(loaded.timeline, 0);
      setLoadedRun(loaded);
      setRuntimeState({
        timeline: loaded.timeline,
        graphState: initialGraph,
        actions: [],
        metrics: null,
        step: 0,
      });
      setStatusLabel(loaded.metadata.status_label);
      setNotice(loaded.notice);
      setIsPlaying(true);
      setSelectedNodeId(null);
      setHighlightedNodeId(null);
      setLogFilter("none");
    };

    void load();

    return () => {
      cancelled = true;
    };
  }, [selectedScenarioRun]);

  useEffect(() => {
    setSelectedNodeId(null);
    setHighlightedNodeId(null);
    setLogFilter("none");
  }, [tab]);

  useEffect(() => {
    if (!isGraphFullscreen) return;
    const priorOverflow = document.body.style.overflow;
    document.body.style.overflow = "hidden";

    const onKeyDown = (event: KeyboardEvent) => {
      if (event.key === "Escape") {
        setIsGraphFullscreen(false);
      }
    };

    window.addEventListener("keydown", onKeyDown);
    return () => {
      window.removeEventListener("keydown", onKeyDown);
      document.body.style.overflow = priorOverflow;
    };
  }, [isGraphFullscreen]);

  useEffect(() => {
    if (tab !== "command" && isGraphFullscreen) {
      setIsGraphFullscreen(false);
    }
  }, [tab, isGraphFullscreen]);

  const maxStep = useMemo(() => {
    if (!runtimeState) return 0;
    return maxReplayStep(runtimeState.timeline);
  }, [runtimeState]);

  const applyStep = useCallback((step: number) => {
    if (!runtimeState) return;
    const bounded = clamp(step, 0, maxReplayStep(runtimeState.timeline));

    const t1 = performance.now();
    const actions = collectActionsUntilStep(runtimeState.timeline, bounded);
    const t2 = performance.now();
    const graphState = materializeGraphAtStep(runtimeState.timeline, bounded);
    const t3 = performance.now();

    let metrics: MetricsTick | null = null;
    for (const frame of runtimeState.timeline.frames) {
      if (frame.step > bounded) break;
      if (frame.metrics) metrics = frame.metrics;
    }

    const drift = Math.max(Math.abs(t1 - t2), Math.abs(t1 - t3), Math.abs(t2 - t3));
    if (drift > 100) {
      console.warn(`SYNC DRIFT at step ${bounded}: ${drift.toFixed(1)}ms`);
    }

    setSyncDriftMs(Number(drift.toFixed(1)));
    setRuntimeState({
      ...runtimeState,
      actions,
      graphState,
      metrics,
      step: bounded,
    });

    if (bounded >= maxReplayStep(runtimeState.timeline)) {
      setIsPlaying(false);
    }
  }, [runtimeState]);

  useEffect(() => {
    if (!runtimeState || !isPlaying) return;
    if (runtimeState.step >= maxStep) return;

    const timer = window.setTimeout(() => {
      applyStep(runtimeState.step + 1);
    }, Math.max(80, Math.floor(700 / speed)));

    return () => {
      window.clearTimeout(timer);
    };
  }, [runtimeState, isPlaying, speed, maxStep, applyStep]);

  const activeScenario = useMemo(() => {
    if (!loadedRun) return null;
    if (loadedRun.metadata.attack_profile_id === "campaign_all_5") {
      return null;
    }

    return SCENARIO_REGISTRY.find((item) => item.id === loadedRun.metadata.attack_profile_id) ?? null;
  }, [loadedRun]);

  const campaignStage = useMemo(() => {
    if (!loadedRun || !runtimeState) return null;
    return deriveActiveCampaignStage(loadedRun.metadata, runtimeState.step);
  }, [loadedRun, runtimeState]);

  const actionLog = useMemo(() => {
    if (!runtimeState) return [];
    return [...runtimeState.actions]
      .sort((a, b) => (b.step - a.step) || (b.ts_ms - a.ts_ms))
      .slice(0, 120);
  }, [runtimeState]);

  const visibleLog = useMemo(() => {
    if (logFilter === "none") return actionLog;
    const nodeId = logFilter.replace("node:", "");
    return actionLog.filter((event) => event.source_host === nodeId || event.target_host === nodeId);
  }, [actionLog, logFilter]);

  const selectedNode = useMemo(() => {
    if (!runtimeState || !selectedNodeId) return null;
    return runtimeState.graphState.nodes.find((node) => node.id === selectedNodeId) ?? null;
  }, [runtimeState, selectedNodeId]);

  const nodeEvents = useMemo(() => {
    if (!selectedNodeId || !runtimeState) return [];
    return runtimeState.actions.filter(
      (event) => event.source_host === selectedNodeId || event.target_host === selectedNodeId,
    );
  }, [runtimeState, selectedNodeId]);

  const latestNodeEvent = nodeEvents[nodeEvents.length - 1] ?? null;

  const latestNodeRedEvent = useMemo<ActionEvent | null>(() => {
    if (!runtimeState || !selectedNodeId) return null;
    let latest: ActionEvent | null = null;
    for (const event of runtimeState.actions) {
      if (event.actor !== "RED") continue;
      if (event.source_host !== selectedNodeId && event.target_host !== selectedNodeId) continue;
      latest = event;
    }
    return latest;
  }, [runtimeState, selectedNodeId]);

  const latestNodeExplainability = useMemo<ExplainabilityRecord | null>(() => {
    if (!runtimeState || !selectedNodeId) return null;
    let latest: ExplainabilityRecord | null = null;
    for (const frame of runtimeState.timeline.frames) {
      if (frame.step > runtimeState.step) break;
      if (frame.explainability?.target_host === selectedNodeId) {
        latest = frame.explainability;
      }
    }
    return latest;
  }, [runtimeState, selectedNodeId]);

  const selectedConnections = useMemo<DisplayConnection[]>(() => {
    if (!runtimeState || !selectedNodeId) return [];
    const byNeighbor = new Map<string, DisplayConnection>();

    for (const edge of runtimeState.graphState.edges) {
      if (edge.source !== selectedNodeId && edge.target !== selectedNodeId) continue;
      const neighbor = edge.source === selectedNodeId ? edge.target : edge.source;
      const existing = byNeighbor.get(neighbor);
      if (!existing || EDGE_STATE_PRIORITY[edge.visual_state] > EDGE_STATE_PRIORITY[existing.visual_state]) {
        byNeighbor.set(neighbor, {
          neighbor_id: neighbor,
          visual_state: edge.visual_state,
        });
      }
    }

    return [...byNeighbor.values()]
      .sort((a, b) => a.neighbor_id.localeCompare(b.neighbor_id))
      .slice(0, 10);
  }, [runtimeState, selectedNodeId]);

  const scenarioHeadline = useMemo(() => {
    if (!loadedRun) return "Loading scenario...";
    if (loadedRun.metadata.attack_profile_id === "campaign_all_5") {
      return `${CAMPAIGN_ALL_5.name}${campaignStage ? ` · Stage ${campaignStage.stage_index + 1}: ${campaignStage.scenario_name}` : ""}`;
    }
    return loadedRun.metadata.scenario_name;
  }, [loadedRun, campaignStage]);

  const activeModeLabel = useMemo(() => {
    return PROFILE_MODES.find((mode) => mode.id === selectedScenarioRun.profile_mode)?.label ?? "Mode";
  }, [selectedScenarioRun.profile_mode]);

  const isNoBlueMode = selectedScenarioRun.profile_mode === "no_blue";

  const nodeAttackNarrative = useMemo(() => {
    const scenarioContext = campaignStage
      ? `Stage ${campaignStage.stage_index + 1} of ${CAMPAIGN_ALL_5.ordered_stages.length}: ${campaignStage.scenario_name}. ${campaignStage.transition_label}.`
      : activeScenario
        ? `${activeScenario.attack_type}. ${activeScenario.expected_flow}`
        : "Attack stream context is loading.";

    if (latestNodeRedEvent) {
      return `${scenarioContext} Latest red action on this node: ${latestNodeRedEvent.description}`;
    }

    if (latestNodeEvent) {
      return `${scenarioContext} Latest node event: ${latestNodeEvent.description}`;
    }

    return `${scenarioContext} No direct activity on this node yet.`;
  }, [campaignStage, activeScenario, latestNodeRedEvent, latestNodeEvent]);

  const canStepBack = Boolean(runtimeState && runtimeState.step > 0);
  const canStepForward = Boolean(runtimeState && runtimeState.step < maxStep);

  const handleNodeSelect = useCallback((nodeId: string | null) => {
    setSelectedNodeId(nodeId);
    setHighlightedNodeId(nodeId);
  }, []);

  const changeProfileMode = (mode: ScenarioProfileMode) => {
    setSelectedScenarioRun((prev) => ({
      ...prev,
      profile_mode: mode,
      run_id: `run_${mode}_${Date.now()}`,
    }));
  };

  const selectScenarioFromVault = (selection: ScenarioSelection) => {
    setSelectedScenarioRun((prev) => ({
      ...prev,
      selection,
      run_id: `run_${prev.profile_mode}_${Date.now()}`,
    }));
    setTab("command");
  };

  return (
    <main className="min-h-screen bg-[var(--surface-primary)] px-4 pb-8 pt-5 text-[var(--text-primary)] md:px-8">
      <div className="mx-auto flex w-full max-w-[1760px] flex-col gap-4">
        <header className="rounded-xl border border-[var(--border)] bg-[var(--surface-card)] px-5 py-4 shadow-[0_20px_56px_rgba(0,0,0,0.32)]">
          <div className="flex flex-wrap items-center justify-between gap-3">
            <div>
              <h1 className="font-mono text-2xl tracking-[0.24em] text-[#f3ead7]">AEGIS</h1>
              <p className="text-xs uppercase tracking-[0.16em] text-[var(--text-secondary)]">
                Scenario-First Command Interface
              </p>
            </div>

            <div className="flex flex-wrap items-center gap-2">
              {TABS.map((item) => (
                <button
                  key={item.id}
                  type="button"
                  onClick={() => setTab(item.id)}
                  className={`rounded-md border px-3 py-2 text-left transition ${
                    tab === item.id
                      ? "border-[#6f809a] bg-[#1a2434] text-[#dde7f8]"
                      : "border-[var(--border)] bg-[#0f131a] text-[var(--text-secondary)]"
                  }`}
                >
                  <div className="font-mono text-[11px] uppercase tracking-[0.14em]">{item.label}</div>
                  <div className="text-[10px] uppercase tracking-[0.1em] opacity-70">{item.subtitle}</div>
                </button>
              ))}
            </div>

            <div className="flex flex-wrap items-center gap-2 text-xs uppercase tracking-[0.12em]">
              <span className="rounded-md border border-[#6f809a] bg-[#1a2434] px-3 py-1 text-[#dde7f8]">
                {statusLabel}
              </span>
              <span className="rounded-md border border-[var(--border)] bg-[#0f131a] px-3 py-1 text-[var(--text-secondary)]">
                Step {runtimeState?.step ?? 0} / {maxStep}
              </span>
            </div>
          </div>

          <div className="mt-4 grid gap-2 md:grid-cols-2">
            {PROFILE_MODES.map((mode) => (
              <button
                key={mode.id}
                type="button"
                onClick={() => changeProfileMode(mode.id)}
                className={`rounded-lg border px-3 py-3 text-left transition ${
                  selectedScenarioRun.profile_mode === mode.id
                    ? "border-[#6f809a] bg-[#1a2434]"
                    : "border-[var(--border)] bg-[#0f131a]"
                }`}
              >
                <div className="font-mono text-[12px] uppercase tracking-[0.12em] text-[#efe8da]">
                  {mode.label}
                </div>
                <div className="mt-1 text-xs text-[var(--text-secondary)]">{mode.subtitle}</div>
              </button>
            ))}
          </div>

          <div className="mt-3 rounded-lg border border-[var(--border)] bg-[#0f131a] px-3 py-2 text-sm text-[#e5e2d8]">
            <span className="font-mono text-[11px] uppercase tracking-[0.14em] text-[var(--text-secondary)]">
              Active Scenario
            </span>
            <div className="mt-1">{scenarioHeadline}</div>
            <div className="mt-1 text-xs text-[#b6c2d7]">
              Mode Meaning: {modePlainLanguage(selectedScenarioRun.profile_mode)}
            </div>
          </div>

          {notice ? (
            <p className="mt-3 rounded-md border border-[#EF9F27]/50 bg-[#EF9F27]/10 px-3 py-2 text-xs text-[#ffe0af]">
              {notice}
            </p>
          ) : null}
        </header>

        {tab === "command" ? (
          <section className="grid gap-4 xl:grid-cols-[260px_minmax(0,1fr)_320px]">
            <aside className="rounded-xl border border-[var(--border)] bg-[var(--surface-card)] p-4">
              <h2 className="font-mono text-xs uppercase tracking-[0.2em] text-[#f3ead7]">Core Telemetry</h2>

              <div className="mt-3 space-y-3">
                <ProgressMetric
                  label="Attack Pressure"
                  value={percent(runtimeState?.metrics?.attack_pressure ?? 0)}
                  color="from-[#253247] via-[#3c4f6c] to-[#8ea2be]"
                />
                <ProgressMetric
                  label="Containment Pressure"
                  value={percent(runtimeState?.metrics?.containment_pressure ?? 0)}
                  color="from-[#134f85] via-[#378ADD] to-[#b7d6ff]"
                />
                <ProgressMetric
                  label="Service Availability"
                  value={percent(runtimeState?.metrics?.service_availability ?? 0)}
                  color="from-[#0f6c50] via-[#1D9E75] to-[#b7ffe4]"
                />
              </div>

              <div className="mt-4 grid grid-cols-2 gap-2">
                <MiniStat label="Open" value={runtimeState?.metrics?.open_incidents ?? 0} />
                <MiniStat label="Contained" value={runtimeState?.metrics?.contained_incidents ?? 0} />
                <MiniStat label="Red Actions" value={runtimeState?.metrics?.red_actions_total ?? 0} />
                <MiniStat label="Blue Actions" value={runtimeState?.metrics?.blue_actions_total ?? 0} />
              </div>

              <div className="mt-4 rounded-lg border border-[var(--border)] bg-[#0e131c] p-3">
                <div className="mb-2 text-[11px] uppercase tracking-[0.14em] text-[var(--text-secondary)]">
                  Hot Targets
                </div>
                <ul className="space-y-1 text-xs text-[var(--text-secondary)]">
                  {(runtimeState?.metrics?.hot_targets ?? []).map((target) => (
                    <li key={target.node_id} className="flex items-center justify-between">
                      <span>{target.node_id}</span>
                      <span>{target.hit_count}</span>
                    </li>
                  ))}
                </ul>
              </div>
            </aside>

            <article className="rounded-xl border border-[var(--border)] bg-[var(--surface-card)] p-4">
              <div className="mb-3 flex flex-wrap items-center justify-between gap-2">
                <div>
                  <h2 className="font-mono text-xs uppercase tracking-[0.2em] text-[#f3ead7]">Threat Topology Matrix</h2>
                  <p className="text-[11px] text-[var(--text-secondary)]">
                    {campaignStage
                      ? `Campaign stage ${campaignStage.stage_index + 1}/${CAMPAIGN_ALL_5.ordered_stages.length}: ${campaignStage.transition_label}`
                      : activeScenario?.expected_flow ?? "Live scenario topology playback"}
                  </p>
                  <div className="mt-1 inline-flex rounded border border-[#6f809a] bg-[#1a2434] px-2 py-0.5 text-[10px] uppercase tracking-[0.12em] text-[#dbe6f7]">
                    Scenario Profile: {activeModeLabel}
                  </div>
                </div>

                <div className="flex flex-wrap items-center gap-2">
                  <button
                    type="button"
                    onClick={() => setIsPlaying((prev) => !prev)}
                    className="rounded-md border border-[#6f809a] bg-[#1a2434] px-3 py-1 text-xs uppercase tracking-[0.12em] text-[#dde7f8]"
                  >
                    {isPlaying ? "Pause" : "Play"}
                  </button>
                  <button
                    type="button"
                    onClick={() => setResetViewToken((prev) => prev + 1)}
                    className="rounded-md border border-[var(--border)] bg-[#0f131a] px-3 py-1 text-xs uppercase tracking-[0.12em] text-[#d2d2cd]"
                  >
                    Reset View
                  </button>
                  <button
                    type="button"
                    onClick={() => setIsGraphFullscreen((prev) => !prev)}
                    className="rounded-md border border-[#6f809a] bg-[#1a2434] px-3 py-1 text-xs uppercase tracking-[0.12em] text-[#dde7f8]"
                  >
                    {isGraphFullscreen ? "Exit Fullscreen" : "Expand"}
                  </button>
                  <button
                    type="button"
                    onClick={() => runtimeState && applyStep(runtimeState.step - 1)}
                    disabled={!canStepBack}
                    className="rounded-md border border-[var(--border)] bg-[#0f131a] px-3 py-1 text-xs uppercase tracking-[0.12em] text-[#d2d2cd] disabled:opacity-35"
                  >
                    Step Back
                  </button>
                  <button
                    type="button"
                    onClick={() => runtimeState && applyStep(runtimeState.step + 1)}
                    disabled={!canStepForward}
                    className="rounded-md border border-[var(--border)] bg-[#0f131a] px-3 py-1 text-xs uppercase tracking-[0.12em] text-[#d2d2cd] disabled:opacity-35"
                  >
                    Step Forward
                  </button>
                </div>
              </div>

              {!isGraphFullscreen ? (
                <>
                  <Graph
                    topology={runtimeState?.timeline.topology ?? null}
                    graphState={runtimeState?.graphState ?? null}
                    selectedNodeId={selectedNodeId}
                    highlightedNodeId={highlightedNodeId}
                    onNodeSelect={handleNodeSelect}
                    resetViewToken={resetViewToken}
                    className="h-[560px] w-full rounded-xl border border-[var(--border)] bg-[#0b0f16]"
                  />

                  <div className="mt-3 grid gap-2 md:grid-cols-[minmax(0,1fr)_220px]">
                    <input
                      type="range"
                      min={0}
                      max={maxStep}
                      value={runtimeState?.step ?? 0}
                      onChange={(event) => {
                        setIsPlaying(false);
                        applyStep(Number(event.target.value));
                      }}
                      className="h-2 w-full cursor-pointer appearance-none rounded-full bg-[#222a36]"
                    />
                    <div className="flex items-center justify-end gap-2">
                      {SPEEDS.map((value) => (
                        <button
                          key={value}
                          type="button"
                          onClick={() => setSpeed(value)}
                          className={`rounded-md border px-2 py-1 text-[11px] uppercase tracking-[0.12em] ${
                            speed === value
                              ? "border-[#6f809a] bg-[#1a2434] text-[#dde7f8]"
                              : "border-[var(--border)] bg-[#0f131a] text-[var(--text-secondary)]"
                          }`}
                        >
                          {value}x
                        </button>
                      ))}
                    </div>
                  </div>
                </>
              ) : null}
            </article>

            <aside className="rounded-xl border border-[var(--border)] bg-[var(--surface-card)] p-4">
              <h2 className="font-mono text-xs uppercase tracking-[0.2em] text-[#f3ead7]">Node Intel</h2>
              {selectedNode ? (
                <>
                  <div className="mt-3 space-y-1 text-xs">
                    <InfoRow label="Node" value={selectedNode.label} />
                    <InfoRow label="Class" value={selectedNode.id} />
                    <InfoRow label="Zone" value={selectedNode.zone} />
                    <InfoRow label="State" value={selectedNode.visual_state} />
                    <InfoRow label="Threat Score" value={String(clamp(nodeEvents.length * 12, 0, 100))} />
                  </div>

                  <div className="mt-3 flex gap-2">
                    <button
                      type="button"
                      onClick={() => setLogFilter(`node:${selectedNode.id}`)}
                      className="rounded border border-[#6f809a] bg-[#1a2434] px-2 py-1 text-[10px] uppercase tracking-[0.12em] text-[#dde7f8]"
                    >
                      Filter Log to Node
                    </button>
                    <button
                      type="button"
                      onClick={() => setLogFilter("none")}
                      className="rounded border border-[var(--border)] bg-[#0f131a] px-2 py-1 text-[10px] uppercase tracking-[0.12em] text-[var(--text-secondary)]"
                    >
                      Clear Filter
                    </button>
                  </div>

                  <NarrativeCard
                    title="Attack In Progress"
                    body={nodeAttackNarrative}
                  />

                  <NarrativeCard
                    title="What Happened"
                    body={latestNodeEvent?.description ?? "No recent event for this node at the current step."}
                  />

                  {!isNoBlueMode ? (
                    <NarrativeCard
                      title="How Aegis Responded"
                      body={
                        latestNodeExplainability
                          ? `${latestNodeExplainability.action} · ${(latestNodeExplainability.confidence * 100).toFixed(0)}% confidence · ${latestNodeExplainability.expected_effect}`
                          : "AEGIS has not taken a targeted blue action for this node yet in the current run."
                      }
                    />
                  ) : null}

                  <div className="mt-3 rounded-lg border border-[var(--border)] bg-[#0e131c] p-3">
                    <div className="mb-2 text-[11px] uppercase tracking-[0.14em] text-[var(--text-secondary)]">
                      Red Team Activity
                    </div>
                    {latestNodeRedEvent ? (
                      <ul className="space-y-1 text-xs text-[#e4e3de]">
                        <li>
                          Action: {upper(latestNodeRedEvent.action_type)}
                        </li>
                        <li>
                          Path: {latestNodeRedEvent.source_host} -&gt; {latestNodeRedEvent.target_host}
                        </li>
                        <li>
                          Tactic: {latestNodeRedEvent.mitre_tactic}
                        </li>
                        <li>
                          Outcome: {upper(latestNodeRedEvent.outcome)}
                        </li>
                        <li>
                          Severity: {upper(latestNodeRedEvent.severity)} · {riskPercent(latestNodeRedEvent.risk_score)}%
                        </li>
                      </ul>
                    ) : (
                      <p className="text-xs leading-relaxed text-[#e4e3de]">
                        No red action has targeted this node yet in the current run.
                      </p>
                    )}
                  </div>

                  <div className="mt-3 rounded-lg border border-[var(--border)] bg-[#0e131c] p-3">
                    <div className="mb-2 text-[11px] uppercase tracking-[0.14em] text-[var(--text-secondary)]">
                      Connections
                    </div>
                    <ul className="space-y-1 text-xs text-[var(--text-secondary)]">
                      {selectedConnections.map((edge) => (
                        <li key={edge.neighbor_id}>
                          ↔ {edge.neighbor_id} [{edge.visual_state}]
                        </li>
                      ))}
                    </ul>
                  </div>
                </>
              ) : (
                <p className="mt-3 text-xs text-[var(--text-secondary)]">
                  Select a node to inspect what happened, current defense rationale, and connection context.
                </p>
              )}
            </aside>
          </section>
        ) : null}

        {tab === "episodes" ? (
          <section className="rounded-xl border border-[var(--border)] bg-[var(--surface-card)] p-5">
            <h2 className="font-mono text-sm uppercase tracking-[0.18em] text-[#f3ead7]">Episode Vault</h2>
            <p className="mt-2 text-sm text-[var(--text-secondary)]">
              Select a predefined attack or stitched campaign. This sets the active run for the Command Grid.
            </p>

            <div className="mt-4 max-h-[560px] space-y-3 overflow-y-auto pr-1">
              {SCENARIO_REGISTRY.map((scenario) => {
                const selected =
                  selectedScenarioRun.selection.kind === "scenario" &&
                  selectedScenarioRun.selection.scenario_id === scenario.id;

                return (
                  <article
                    key={scenario.id}
                    className={`rounded-lg border p-3 ${
                      selected ? "border-[#6f809a] bg-[#1a2434]" : "border-[var(--border)] bg-[#0f131a]"
                    }`}
                  >
                    <div className="flex flex-wrap items-start justify-between gap-3">
                      <div className="min-w-[240px] flex-1">
                        <div className="font-mono text-xs uppercase tracking-[0.12em] text-[#efe8da]">{scenario.name}</div>
                        <p className="mt-1 text-xs text-[var(--text-secondary)]">{scenario.description}</p>
                        <div className="mt-2 text-[11px] text-[var(--text-secondary)]">Attack type: {scenario.attack_type}</div>
                        <div className="text-[11px] text-[var(--text-secondary)]">Severity profile: {scenario.severity_profile}</div>
                        <div className="mt-1 text-[11px] text-[#c4cedd]">Flow: {scenario.expected_flow}</div>
                      </div>
                      <div className="flex items-center gap-2">
                        <span className="rounded border border-[#6f809a] bg-[#1a2434] px-2 py-1 text-[10px] uppercase tracking-[0.12em] text-[#dde7f8]">
                          Scenario
                        </span>
                        <button
                          type="button"
                          onClick={() => selectScenarioFromVault({ kind: "scenario", scenario_id: scenario.id })}
                          className="rounded border border-[#6f809a] bg-[#1a2434] px-3 py-1 text-xs uppercase tracking-[0.12em] text-[#dde7f8]"
                        >
                          Open in Dashboard
                        </button>
                      </div>
                    </div>
                  </article>
                );
              })}

              <article
                className={`rounded-lg border p-3 ${
                  selectedScenarioRun.selection.kind === "campaign"
                    ? "border-[#6f809a] bg-[#1a2434]"
                    : "border-[var(--border)] bg-[#0f131a]"
                }`}
              >
                <div className="flex flex-wrap items-start justify-between gap-3">
                  <div className="min-w-[240px] flex-1">
                    <div className="font-mono text-xs uppercase tracking-[0.12em] text-[#efe8da]">{CAMPAIGN_ALL_5.name}</div>
                    <p className="mt-1 text-xs text-[var(--text-secondary)]">{CAMPAIGN_ALL_5.description}</p>

                    <ul className="mt-2 space-y-1 text-[11px] text-[var(--text-secondary)]">
                      {CAMPAIGN_ALL_5.ordered_stages.map((stage, idx) => (
                        <li key={stage.scenario_id}>
                          {idx + 1}. {upper(stage.scenario_id)} - {stage.transition_label}
                        </li>
                      ))}
                    </ul>
                  </div>
                  <div className="flex items-center gap-2">
                    <span className="rounded border border-[#6f809a] bg-[#1a2434] px-2 py-1 text-[10px] uppercase tracking-[0.12em] text-[#dde7f8]">
                      Campaign
                    </span>
                    <button
                      type="button"
                      onClick={() => selectScenarioFromVault({ kind: "campaign", campaign_id: "campaign_all_5" })}
                      className="rounded border border-[#6f809a] bg-[#1a2434] px-3 py-1 text-xs uppercase tracking-[0.12em] text-[#dde7f8]"
                    >
                      Open in Dashboard
                    </button>
                  </div>
                </div>
              </article>
            </div>
          </section>
        ) : null}

        {tab === "mission" ? (
          <section className="rounded-xl border border-[var(--border)] bg-[var(--surface-card)] p-5">
            <h2 className="font-mono text-sm uppercase tracking-[0.18em] text-[#f3ead7]">Mission Brief</h2>
            <p className="mt-2 text-sm text-[var(--text-secondary)]">
              Plain-language summary of what attack is active, how Aegis is defending, and why it matters.
            </p>
            <div className="mt-2 inline-flex rounded border border-[#6f809a] bg-[#1a2434] px-2 py-0.5 text-[10px] uppercase tracking-[0.12em] text-[#dde7f8]">
              Scenario Profile: {activeModeLabel}
            </div>

            <div className="mt-4 grid gap-4 xl:grid-cols-[1.3fr_1fr]">
              <div className="space-y-3">
                <NarrativeCard
                  title="What Attack Is Happening"
                  body={
                    campaignStage
                      ? `Stage ${campaignStage.stage_index + 1} of ${CAMPAIGN_ALL_5.ordered_stages.length}: ${campaignStage.scenario_name}. ${campaignStage.transition_label}.`
                      : activeScenario
                        ? `${activeScenario.name}. ${activeScenario.attack_type}.`
                        : "Scenario loading..."
                  }
                />

                <NarrativeCard
                  title="How Aegis Is Defending"
                  body={
                    runtimeState?.metrics
                      ? `Aegis is balancing attack pressure (${percent(runtimeState.metrics.attack_pressure)}%) against containment pressure (${percent(runtimeState.metrics.containment_pressure)}%). Current availability is ${percent(runtimeState.metrics.service_availability)}%.`
                      : "Waiting for metrics to initialize."
                  }
                />

                <NarrativeCard
                  title="Why This Matters"
                  body={
                    activeScenario
                      ? `${activeScenario.expected_flow} This view helps judges and operators see not just what changed, but why defender actions were chosen.`
                      : CAMPAIGN_ALL_5.description
                  }
                />

                <NarrativeCard
                  title="How Aegis Works"
                  body={`${narrativeForMode(selectedScenarioRun.profile_mode)} The system consumes canonical events, updates topology + logs in sync, and surfaces blue-team rationale each step.`}
                />
              </div>

              <div className="space-y-3">
                <MiniStat label="Attack Pressure" value={`${percent(runtimeState?.metrics?.attack_pressure ?? 0)}%`} />
                <MiniStat label="Containment" value={`${percent(runtimeState?.metrics?.containment_pressure ?? 0)}%`} />
                <MiniStat label="Availability" value={`${percent(runtimeState?.metrics?.service_availability ?? 0)}%`} />
                <MiniStat label="Open Incidents" value={runtimeState?.metrics?.open_incidents ?? 0} />
                <MiniStat label="Sync Drift" value={`${syncDriftMs}ms`} />
              </div>
            </div>
          </section>
        ) : null}

        {isGraphFullscreen ? (
          <div className="fixed inset-0 z-[95] bg-black/85 p-4">
            <div className="mx-auto flex h-full w-full max-w-[1840px] flex-col rounded-xl border border-[var(--border)] bg-[#0b0f16] p-3 shadow-[0_24px_90px_rgba(0,0,0,0.55)]">
              <div className="mb-3 flex items-center justify-between gap-2">
                <div>
                  <div className="font-mono text-xs uppercase tracking-[0.18em] text-[#f3ead7]">Threat Topology Matrix</div>
                  <div className="text-[11px] text-[var(--text-secondary)]">Fullscreen view · Press Esc to close</div>
                </div>
                <button
                  type="button"
                  onClick={() => setIsGraphFullscreen(false)}
                  className="rounded-md border border-[#6f809a] bg-[#1a2434] px-3 py-1 text-xs uppercase tracking-[0.12em] text-[#dde7f8]"
                >
                  Exit Fullscreen
                </button>
              </div>

              <Graph
                topology={runtimeState?.timeline.topology ?? null}
                graphState={runtimeState?.graphState ?? null}
                selectedNodeId={selectedNodeId}
                highlightedNodeId={highlightedNodeId}
                onNodeSelect={handleNodeSelect}
                resetViewToken={resetViewToken}
                className="h-[calc(100vh-180px)] w-full rounded-xl border border-[var(--border)] bg-[#0b0f16]"
              />

              <div className="mt-3 grid gap-2 md:grid-cols-[minmax(0,1fr)_220px]">
                <input
                  type="range"
                  min={0}
                  max={maxStep}
                  value={runtimeState?.step ?? 0}
                  onChange={(event) => {
                    setIsPlaying(false);
                    applyStep(Number(event.target.value));
                  }}
                  className="h-2 w-full cursor-pointer appearance-none rounded-full bg-[#222a36]"
                />
                <div className="flex items-center justify-end gap-2">
                  {SPEEDS.map((value) => (
                    <button
                      key={value}
                      type="button"
                      onClick={() => setSpeed(value)}
                      className={`rounded-md border px-2 py-1 text-[11px] uppercase tracking-[0.12em] ${
                        speed === value
                          ? "border-[#6f809a] bg-[#1a2434] text-[#dde7f8]"
                          : "border-[var(--border)] bg-[#0f131a] text-[var(--text-secondary)]"
                      }`}
                    >
                      {value}x
                    </button>
                  ))}
                </div>
              </div>
            </div>
          </div>
        ) : null}

        <section className="rounded-xl border border-[var(--border)] bg-[var(--surface-card)] p-4">
          <div className="mb-2 flex flex-wrap items-center justify-between gap-2">
            <h3 className="font-mono text-xs uppercase tracking-[0.2em] text-[#f3ead7]">Live Tactical Log</h3>
            <div className="flex items-center gap-2 text-[11px] uppercase tracking-[0.12em]">
              <span className="text-[var(--text-secondary)]">Filter:</span>
              {logFilter === "none" ? (
                <span className="rounded border border-[var(--border)] px-2 py-0.5 text-[var(--text-secondary)]">none</span>
              ) : (
                <>
                  <span className="rounded border border-[#6f809a] bg-[#1a2434] px-2 py-0.5 text-[#dde7f8]">
                    {logFilter.replace("node:", "node ")}
                  </span>
                  <button
                    type="button"
                    onClick={() => setLogFilter("none")}
                    className="rounded border border-[var(--border)] px-2 py-0.5 text-[var(--text-secondary)]"
                  >
                    clear
                  </button>
                </>
              )}
            </div>
          </div>

          <div className="max-h-[280px] overflow-y-auto rounded-lg border border-[var(--border)] bg-[#0b0f16] p-2">
            {visibleLog.length > 0 ? (
              visibleLog.map((event) => (
                <button
                  key={event.event_id}
                  type="button"
                  onMouseEnter={() => setHighlightedNodeId(event.target_host)}
                  onMouseLeave={() => setHighlightedNodeId(null)}
                  onClick={() => {
                    setSelectedNodeId(event.target_host);
                    setHighlightedNodeId(event.target_host);
                  }}
                  className="mb-2 grid w-full gap-2 rounded-md border border-white/10 bg-black/25 px-2 py-2 text-left text-xs md:grid-cols-[68px_72px_minmax(0,1fr)_140px_120px]"
                >
                  <span className="font-mono text-[#cfd2d8]">#{event.step.toString().padStart(3, "0")}</span>
                  <span className={`rounded border px-2 py-0.5 text-center font-semibold ${actorBadgeClass(event.actor)}`}>
                    {event.actor}
                  </span>
                  <span className="truncate text-[#e8e7e2]">{event.description}</span>
                  <span className="truncate text-[#b8b6ad]">{event.target_host}</span>
                  <span className={`rounded border px-2 py-0.5 text-center ${severityBadgeClass(event.severity)}`}>
                    {event.severity.toUpperCase()} · {riskPercent(event.risk_score)}%
                  </span>
                </button>
              ))
            ) : (
              <p className="p-3 text-xs text-[var(--text-secondary)]">No log events for current filter/step range.</p>
            )}
          </div>
        </section>
      </div>
    </main>
  );
}

function InfoRow({ label, value }: { label: string; value: string }) {
  return (
    <div className="flex items-center justify-between border-b border-white/10 pb-1 text-[11px]">
      <span className="uppercase tracking-[0.12em] text-[var(--text-secondary)]">{label}</span>
      <span className="text-right text-[#ece9df]">{value}</span>
    </div>
  );
}

function NarrativeCard({ title, body }: { title: string; body: string }) {
  return (
    <div className="rounded-lg border border-[var(--border)] bg-[#0e131c] p-3">
      <div className="mb-1 text-[11px] uppercase tracking-[0.14em] text-[var(--text-secondary)]">{title}</div>
      <p className="text-xs leading-relaxed text-[#e4e3de]">{body}</p>
    </div>
  );
}

function MiniStat({ label, value }: { label: string; value: string | number }) {
  return (
    <div className="rounded-md border border-[var(--border)] bg-[#0f131a] px-3 py-2">
      <div className="text-[10px] uppercase tracking-[0.12em] text-[var(--text-secondary)]">{upper(label)}</div>
      <div className="mt-1 font-mono text-sm text-[#ece8df]">{value}</div>
    </div>
  );
}

function ProgressMetric({ label, value, color }: { label: string; value: number; color: string }) {
  return (
    <div>
      <div className="mb-1 flex items-center justify-between text-[11px] uppercase tracking-[0.12em] text-[var(--text-secondary)]">
        <span>{label}</span>
        <span>{value}%</span>
      </div>
      <div className="h-3 overflow-hidden rounded-full border border-[var(--border)] bg-[#090d14]">
        <div className={`h-full bg-gradient-to-r ${color}`} style={{ width: `${Math.max(4, value)}%` }} />
      </div>
    </div>
  );
}
