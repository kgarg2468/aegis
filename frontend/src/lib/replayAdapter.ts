import type { WSMessage } from "./integrationContract";
import { generateScenarioReplay } from "./scenarioReplay";
import type { ReplayTimeline } from "./replayRuntime";
import { indexReplayMessages } from "./replayRuntime";
import {
  CAMPAIGN_ALL_5,
  type AttackProfileId,
  type CampaignStageWindow,
  type ScenarioSelection,
  buildCampaignStageWindows,
  getScenarioById,
} from "./scenarios";

export type ScenarioProfileMode = "no_blue" | "current_run_enterprise";

export interface RunMetadata {
  run_id: string;
  source: "local_bundle" | "generated" | "model_stream";
  scenario_id: string;
  attack_profile_id: AttackProfileId | "campaign_all_5";
  campaign_stage: CampaignStageWindow | null;
  campaign_windows: CampaignStageWindow[];
  profile_mode: ScenarioProfileMode;
  scenario_name: string;
  scenario_description: string;
  status_label: string;
}

export interface SelectedScenarioRun {
  profile_mode: ScenarioProfileMode;
  selection: ScenarioSelection;
  run_id: string;
}

export interface LoadedScenarioRun {
  timeline: ReplayTimeline;
  metadata: RunMetadata;
  notice: string | null;
}

function modeSeed(mode: ScenarioProfileMode): number {
  if (mode === "current_run_enterprise") return 2026;
  if (mode === "no_blue") return 1337;
  return 1003;
}

function modeSteps(mode: ScenarioProfileMode): number {
  if (mode === "current_run_enterprise") return 240;
  return 200;
}

function statusLabelFor(mode: ScenarioProfileMode, source: RunMetadata["source"]): string {
  if (source === "local_bundle") return "LIVE DATA (LOCAL BUNDLE)";
  if (mode === "current_run_enterprise") return "CURRENT RUN (ENTERPRISE LEVEL)";
  if (mode === "no_blue") return "NO BLUE (SIMULATED)";
  return "SCENARIO REPLAY";
}

function scenarioSeed(scenarioId: AttackProfileId): number {
  const map: Record<AttackProfileId, number> = {
    eduroam_credential_harvest: 1101,
    faculty_spear_phish: 1202,
    iot_botnet_exhaustion: 1303,
    insider_ad_backdoor: 1404,
    print_ransomware_propagation: 1505,
  };
  return map[scenarioId];
}

function fromMessages(messages: WSMessage[], metadata: Omit<RunMetadata, "campaign_stage">): LoadedScenarioRun {
  const timeline = indexReplayMessages(messages);
  const campaignStage = deriveCampaignStage(metadata.campaign_windows, 0);

  return {
    timeline,
    metadata: {
      ...metadata,
      campaign_stage: campaignStage,
    },
    notice: null,
  };
}

function deriveCampaignStage(windows: CampaignStageWindow[], step: number): CampaignStageWindow | null {
  if (windows.length === 0) return null;
  return windows.find((window) => step >= window.start_step && step <= window.end_step) ?? windows[0] ?? null;
}

async function loadLocalBundleMessages(): Promise<WSMessage[] | null> {
  try {
    const listResponse = await fetch("/api/local/replay/list", { cache: "no-store" });
    if (!listResponse.ok) return null;
    const listPayload = (await listResponse.json()) as { replays?: Array<{ id: string }> };
    const replayIds = listPayload.replays?.map((entry) => entry.id) ?? [];
    if (replayIds.length === 0) return null;

    for (const replayId of replayIds) {
      const bundleResponse = await fetch(`/api/local/replay/${encodeURIComponent(replayId)}/bundle`, {
        cache: "no-store",
      });
      if (!bundleResponse.ok) continue;

      const bundlePayload = (await bundleResponse.json()) as {
        run_id?: string;
        replay_id?: string;
        events?: unknown[];
      };

      if (!Array.isArray(bundlePayload.events)) continue;
      const parsed = bundlePayload.events.filter((entry): entry is WSMessage => {
        if (!entry || typeof entry !== "object") return false;
        const candidate = entry as Record<string, unknown>;
        return typeof candidate.type === "string" && "data" in candidate;
      });

      if (parsed.length > 0 && parsed[0]?.type === "topology_init") {
        return parsed;
      }
    }

    return null;
  } catch {
    return null;
  }
}

function buildScenarioRun(selection: ScenarioSelection, mode: ScenarioProfileMode): LoadedScenarioRun {
  const includeBlue = mode !== "no_blue";

  if (selection.kind === "campaign") {
    const totalSteps = modeSteps(mode);
    const generated = generateScenarioReplay({
      replayId: `${selection.campaign_id}_${mode}`,
      scenarioId: selection.campaign_id,
      totalSteps,
      seed: modeSeed(mode),
      includeBlue,
    });

    const windows = buildCampaignStageWindows(totalSteps);
    const metadataBase = {
      run_id: generated.replayId,
      source: "generated" as const,
      scenario_id: selection.campaign_id,
      attack_profile_id: selection.campaign_id,
      campaign_windows: windows,
      profile_mode: mode,
      scenario_name: CAMPAIGN_ALL_5.name,
      scenario_description: CAMPAIGN_ALL_5.description,
      status_label: statusLabelFor(mode, "generated"),
    };

    return {
      ...fromMessages(generated.messages, metadataBase),
      notice: null,
    };
  }

  const scenario = getScenarioById(selection.scenario_id);
  const generated = generateScenarioReplay({
    replayId: `${scenario.id}_${mode}`,
    scenarioId: scenario.id,
    totalSteps: modeSteps(mode),
    seed: scenarioSeed(scenario.id) + (mode === "current_run_enterprise" ? 222 : 0),
    includeBlue,
  });

  const metadataBase = {
    run_id: generated.replayId,
    source: "generated" as const,
    scenario_id: scenario.id,
    attack_profile_id: scenario.id,
    campaign_windows: [] as CampaignStageWindow[],
    profile_mode: mode,
    scenario_name: scenario.name,
    scenario_description: scenario.description,
    status_label: statusLabelFor(mode, "generated"),
  };

  return {
    ...fromMessages(generated.messages, metadataBase),
    notice: null,
  };
}

export async function loadScenarioRun(selected: SelectedScenarioRun): Promise<LoadedScenarioRun> {
  if (selected.profile_mode === "current_run_enterprise") {
    const localMessages = await loadLocalBundleMessages();
    if (localMessages) {
      const fallbackScenario =
        selected.selection.kind === "scenario"
          ? getScenarioById(selected.selection.scenario_id)
          : getScenarioById("faculty_spear_phish");

      const local = fromMessages(localMessages, {
        run_id: selected.run_id,
        source: "local_bundle",
        scenario_id: selected.selection.kind === "campaign" ? "campaign_all_5" : selected.selection.scenario_id,
        attack_profile_id: selected.selection.kind === "campaign" ? "campaign_all_5" : selected.selection.scenario_id,
        campaign_windows:
          selected.selection.kind === "campaign"
            ? buildCampaignStageWindows(indexReplayMessages(localMessages).frames.length || 200)
            : [],
        profile_mode: selected.profile_mode,
        scenario_name:
          selected.selection.kind === "campaign" ? CAMPAIGN_ALL_5.name : fallbackScenario.name,
        scenario_description:
          selected.selection.kind === "campaign" ? CAMPAIGN_ALL_5.description : fallbackScenario.description,
        status_label: statusLabelFor(selected.profile_mode, "local_bundle"),
      });

      return local;
    }

    const fallback = buildScenarioRun(selected.selection, selected.profile_mode);
    return {
      ...fallback,
      notice: null,
    };
  }

  return buildScenarioRun(selected.selection, selected.profile_mode);
}

export function deriveActiveCampaignStage(metadata: RunMetadata, step: number): CampaignStageWindow | null {
  return deriveCampaignStage(metadata.campaign_windows, step);
}
