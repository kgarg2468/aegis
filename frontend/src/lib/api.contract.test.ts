import test from "node:test";
import assert from "node:assert/strict";

import {
  extractApiErrorMessage,
  mapSandboxFormToEpisodeSpec,
  toSandboxRunCreateRequest,
  type SandboxFormSpec,
} from "./api";

test("mapSandboxFormToEpisodeSpec maps legacy builder keys to API contract keys", () => {
  const form: SandboxFormSpec = {
    name: "Live Sandbox Drill",
    seed: 4242,
    horizon: 120,
    nodes: [
      { id: "workstation-1", role: "endpoint", os: "windows", severity: "high" },
      { id: "db-1", role: "database", os: "linux", severity: "medium" },
    ],
    vulnerabilities: [{ id: "SYNTH-CVE-2026-1001", node_id: "workstation-1", severity: "high" }],
    red_objectives: [{ id: "obj-exfiltrate-db", type: "exfiltrate", target: "db-1" }],
    defender_mode: "aegis",
  };

  const episodeSpec = mapSandboxFormToEpisodeSpec(form);
  assert.deepEqual(episodeSpec.nodes, [
    { id: "workstation-1", role: "endpoint", severity: "high" },
    { id: "db-1", role: "database", severity: "medium" },
  ]);
  assert.deepEqual(episodeSpec.vulnerabilities, [
    { node_id: "workstation-1", vuln_id: "SYNTH-CVE-2026-1001" },
  ]);
  assert.deepEqual(episodeSpec.red_objectives, [
    { target_node_id: "db-1", objective: "exfiltrate" },
  ]);
});

test("toSandboxRunCreateRequest wraps episode spec in episode_spec envelope", () => {
  const form: SandboxFormSpec = {
    name: "quick",
    horizon: 20,
    nodes: [{ id: "host-01", severity: "high" }],
    vulnerabilities: [{ id: "SYNTH-CVE-2026-1001", node_id: "host-01" }],
    red_objectives: [{ id: "obj-1", type: "exfiltrate", target: "host-01" }],
    defender_mode: "aegis",
  };
  const payload = toSandboxRunCreateRequest(mapSandboxFormToEpisodeSpec(form));
  assert.ok(payload.episode_spec);
  assert.equal(payload.episode_spec.name, "quick");
});

test("extractApiErrorMessage parses FastAPI detail array entries", () => {
  const payload = {
    detail: [
      {
        loc: ["body", "episode_spec", "horizon"],
        msg: "Input should be greater than or equal to 10",
      },
      { msg: "another error" },
    ],
  };
  const parsed = extractApiErrorMessage(payload);
  assert.equal(
    parsed,
    "body.episode_spec.horizon: Input should be greater than or equal to 10; another error",
  );
});
