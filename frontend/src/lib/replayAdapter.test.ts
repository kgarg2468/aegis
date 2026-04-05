import assert from "node:assert/strict";
import test from "node:test";

import { deriveActiveCampaignStage, loadScenarioRun } from "./replayAdapter";
import { CAMPAIGN_ALL_5 } from "./scenarios";

test("loadScenarioRun returns scenario metadata in non-current modes", async () => {
  const loaded = await loadScenarioRun({
    profile_mode: "current_run_enterprise",
    selection: { kind: "scenario", scenario_id: "iot_botnet_exhaustion" },
    run_id: "test_run_1",
  });

  assert.equal(loaded.metadata.attack_profile_id, "iot_botnet_exhaustion");
  assert.equal(loaded.metadata.profile_mode, "current_run_enterprise");
  assert.equal(["mock", "local_bundle"].includes(loaded.metadata.source), true);
  assert.equal(loaded.timeline.frames.length > 0, true);
});

test("campaign runs expose stage windows and active stage derivation", async () => {
  const loaded = await loadScenarioRun({
    profile_mode: "no_blue",
    selection: { kind: "campaign", campaign_id: "campaign_all_5" },
    run_id: "test_run_2",
  });

  assert.equal(loaded.metadata.attack_profile_id, "campaign_all_5");
  assert.equal(loaded.metadata.campaign_windows.length, CAMPAIGN_ALL_5.ordered_stages.length);

  const first = deriveActiveCampaignStage(loaded.metadata, 1);
  assert.ok(first);
  assert.equal(first?.stage_index, 0);

  const finalStep = loaded.timeline.frames.at(-1)?.step ?? 0;
  const last = deriveActiveCampaignStage(loaded.metadata, finalStep);
  assert.ok(last);
  assert.equal(last?.stage_index, CAMPAIGN_ALL_5.ordered_stages.length - 1);
});
