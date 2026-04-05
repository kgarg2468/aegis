import test from "node:test";
import assert from "node:assert/strict";

import { createCanonicalTopologyInit, isNodeIdAllowed } from "./integrationContract";
import { generateMockReplay } from "./mockReplay";

test("canonical topology contains registry nodes and edge id format source->target", () => {
  const topology = createCanonicalTopologyInit();

  assert.ok(topology.nodes.length >= 27);
  assert.ok(topology.edges.length > 0);

  for (const node of topology.nodes) {
    assert.equal(isNodeIdAllowed(node.id), true);
  }

  for (const edge of topology.edges) {
    assert.match(edge.id, /^[a-z0-9_]+->[a-z0-9_]+$/);
    assert.equal(edge.id, `${edge.source}->${edge.target}`);
  }
});

test("mock replay respects event framing contract and topology init order", () => {
  const replay = generateMockReplay({ totalSteps: 40, seed: 7 });

  assert.ok(replay.messages.length > 0);
  assert.equal(replay.messages[0]?.type, "topology_init");
  assert.equal(replay.messages[replay.messages.length - 1]?.type, "episode_end");

  const counters = new Map<number, { state_delta: number; metrics_tick: number; explainability: number; blue_actions: number }>();

  for (const message of replay.messages) {
    if (message.type === "topology_init" || message.type === "episode_end") continue;

    const step = message.data.step;
    const bucket = counters.get(step) ?? {
      state_delta: 0,
      metrics_tick: 0,
      explainability: 0,
      blue_actions: 0,
    };

    if (message.type === "state_delta") bucket.state_delta += 1;
    if (message.type === "metrics_tick") bucket.metrics_tick += 1;
    if (message.type === "explainability") bucket.explainability += 1;
    if (message.type === "action_event" && message.data.actor === "BLUE") bucket.blue_actions += 1;

    counters.set(step, bucket);
  }

  for (const [step, bucket] of counters) {
    assert.equal(bucket.state_delta, 1, `step ${step} should have exactly one state_delta`);
    assert.equal(bucket.metrics_tick, 1, `step ${step} should have exactly one metrics_tick`);
    assert.equal(bucket.explainability, 1, `step ${step} should have exactly one explainability record`);
    assert.equal(bucket.blue_actions, 1, `step ${step} should have exactly one BLUE action`);
  }
});
