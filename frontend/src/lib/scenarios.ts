export type AttackProfileId =
  | "eduroam_credential_harvest"
  | "faculty_spear_phish"
  | "iot_botnet_exhaustion"
  | "insider_ad_backdoor"
  | "print_ransomware_propagation";

export type SeverityProfile = "stealth" | "balanced" | "aggressive" | "surge";

export interface ScenarioRegistryItem {
  id: AttackProfileId;
  episode_id: string;
  name: string;
  description: string;
  attack_type: string;
  expected_flow: string;
  default_replay_id: string;
  severity_profile: SeverityProfile;
}

export interface CampaignStageDefinition {
  scenario_id: AttackProfileId;
  transition_label: string;
}

export interface CampaignDefinition {
  id: "campaign_all_5";
  name: string;
  description: string;
  ordered_stages: CampaignStageDefinition[];
}

export interface CampaignStageWindow {
  stage_index: number;
  scenario_id: AttackProfileId;
  scenario_name: string;
  start_step: number;
  end_step: number;
  transition_label: string;
}

export const SCENARIO_REGISTRY: ScenarioRegistryItem[] = [
  {
    id: "eduroam_credential_harvest",
    episode_id: "Episode 001",
    name: "Eduroam Credential Harvesting -> SIS Breach",
    description: "Perimeter credential capture pivoting into SIS access.",
    attack_type: "Credential Theft + Admin Pivot",
    expected_flow:
      "Perimeter signals escalate into credential flow and a jump into admin assets.",
    default_replay_id: "replay_eduroam_01",
    severity_profile: "balanced",
  },
  {
    id: "faculty_spear_phish",
    episode_id: "Episode 002",
    name: "Faculty Spear Phish -> Research Data Theft",
    description: "Compromised faculty endpoint laterally reaches research assets.",
    attack_type: "Phishing + Lateral Movement + Exfiltration",
    expected_flow: "Campus endpoint compromise, research pivot, then data egress pressure.",
    default_replay_id: "replay_faculty_01",
    severity_profile: "balanced",
  },
  {
    id: "iot_botnet_exhaustion",
    episode_id: "Episode 003",
    name: "IoT Botnet -> Resource Exhaustion",
    description: "Distributed IoT compromise driving broad campus service degradation.",
    attack_type: "Distributed Compromise / Availability Attack",
    expected_flow: "Wave-like probing and compromise across peripheral nodes.",
    default_replay_id: "replay_iot_01",
    severity_profile: "surge",
  },
  {
    id: "insider_ad_backdoor",
    episode_id: "Episode 004",
    name: "Insider Threat -> AD Backdoor",
    description: "Low-noise internal movement with delayed backdoor reveal.",
    attack_type: "Insider Persistence + Credential Abuse",
    expected_flow: "Subtle admin-zone pressure and credential flow before persistence hardens.",
    default_replay_id: "replay_insider_01",
    severity_profile: "stealth",
  },
  {
    id: "print_ransomware_propagation",
    episode_id: "Episode 005",
    name: "Print Server Ransomware Propagation",
    description: "Rapid lateral spread beginning at print infrastructure.",
    attack_type: "Fast Lateral Propagation / Impact",
    expected_flow: "Quick compromise cascade with broad edge activation.",
    default_replay_id: "replay_print_01",
    severity_profile: "aggressive",
  },
];

export const CAMPAIGN_ALL_5: CampaignDefinition = {
  id: "campaign_all_5",
  name: "All 5 Attack Campaign",
  description: "Stitched sequence of all predefined attacks for demo storytelling.",
  ordered_stages: [
    { scenario_id: "eduroam_credential_harvest", transition_label: "Credential harvest foothold" },
    { scenario_id: "faculty_spear_phish", transition_label: "Phish pivot into research" },
    { scenario_id: "iot_botnet_exhaustion", transition_label: "Distributed campus pressure" },
    { scenario_id: "insider_ad_backdoor", transition_label: "Insider persistence escalation" },
    { scenario_id: "print_ransomware_propagation", transition_label: "Rapid impact propagation" },
  ],
};

export function getScenarioById(id: AttackProfileId): ScenarioRegistryItem {
  const found = SCENARIO_REGISTRY.find((item) => item.id === id);
  if (!found) {
    throw new Error(`Unknown scenario id: ${id}`);
  }
  return found;
}

export function buildCampaignStageWindows(totalSteps: number): CampaignStageWindow[] {
  const stageCount = CAMPAIGN_ALL_5.ordered_stages.length;
  const baseWidth = Math.floor(totalSteps / stageCount);
  const remainder = totalSteps % stageCount;

  const windows: CampaignStageWindow[] = [];
  let cursor = 1;

  CAMPAIGN_ALL_5.ordered_stages.forEach((stage, index) => {
    const extra = index < remainder ? 1 : 0;
    const width = baseWidth + extra;
    const start = cursor;
    const end = index === stageCount - 1 ? totalSteps : cursor + width - 1;
    const scenario = getScenarioById(stage.scenario_id);

    windows.push({
      stage_index: index,
      scenario_id: stage.scenario_id,
      scenario_name: scenario.name,
      start_step: start,
      end_step: end,
      transition_label: stage.transition_label,
    });

    cursor = end + 1;
  });

  return windows;
}

export type ScenarioSelection =
  | { kind: "scenario"; scenario_id: AttackProfileId }
  | { kind: "campaign"; campaign_id: "campaign_all_5" };
