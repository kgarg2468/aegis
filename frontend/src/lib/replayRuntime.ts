import type {
  ActionEvent,
  DetectionEvent,
  EpisodeEnd,
  ExplainabilityRecord,
  MetricsTick,
  StateDelta,
  TopologyAddNodeData,
  TopologyEdge,
  TopologyInitData,
  TopologyNode,
  WSMessage,
} from "./integrationContract";

export interface ReplayFrame {
  step: number;
  actions: ActionEvent[];
  stateDelta: StateDelta | null;
  detection: DetectionEvent | null;
  explainability: ExplainabilityRecord | null;
  metrics: MetricsTick | null;
  topologyAdds: TopologyAddNodeData[];
}

export interface ReplayTimeline {
  topology: TopologyInitData;
  frames: ReplayFrame[];
  episodeEnd: EpisodeEnd | null;
}

export type RuntimeNode = TopologyNode;

export interface RuntimeEdge extends TopologyEdge {
  direction?: "forward" | "reverse";
}

export interface RuntimeGraphState {
  nodes: RuntimeNode[];
  edges: RuntimeEdge[];
}

function makeFrame(step: number): ReplayFrame {
  return {
    step,
    actions: [],
    stateDelta: null,
    detection: null,
    explainability: null,
    metrics: null,
    topologyAdds: [],
  };
}

function cloneTopologyNode(node: TopologyNode): RuntimeNode {
  return {
    ...node,
    services: [...node.services],
  };
}

function cloneTopologyEdge(edge: TopologyEdge): RuntimeEdge {
  return {
    ...edge,
  };
}

export function indexReplayMessages(messages: WSMessage[]): ReplayTimeline {
  const topologyInit = messages.find((msg): msg is { type: "topology_init"; data: TopologyInitData } => msg.type === "topology_init");

  if (!topologyInit) {
    throw new Error("Replay missing topology_init event");
  }

  const framesByStep = new Map<number, ReplayFrame>();
  let episodeEnd: EpisodeEnd | null = null;
  let activeStep = 0;

  for (const message of messages) {
    if (message.type === "topology_init") continue;

    if (message.type === "episode_end") {
      episodeEnd = message.data;
      continue;
    }

    let step: number;
    if (message.type === "topology_add_node") {
      step = activeStep;
    } else {
      step = message.data.step;
      activeStep = step;
    }

    if (!Number.isFinite(step) || step < 0) continue;

    const frame = framesByStep.get(step) ?? makeFrame(step);

    if (message.type === "action_event") {
      frame.actions.push(message.data);
    }

    if (message.type === "state_delta") {
      frame.stateDelta = message.data;
    }

    if (message.type === "detection_event") {
      frame.detection = message.data;
    }

    if (message.type === "explainability") {
      frame.explainability = message.data;
    }

    if (message.type === "metrics_tick") {
      frame.metrics = message.data;
    }

    if (message.type === "topology_add_node") {
      frame.topologyAdds.push(message.data);
    }

    framesByStep.set(step, frame);
  }

  const frames = [...framesByStep.values()].sort((a, b) => a.step - b.step);
  return {
    topology: topologyInit.data,
    frames,
    episodeEnd,
  };
}

export function createInitialGraphState(topology: TopologyInitData): RuntimeGraphState {
  return {
    nodes: topology.nodes.map(cloneTopologyNode),
    edges: topology.edges.map(cloneTopologyEdge),
  };
}

function applyTopologyAdds(state: RuntimeGraphState, frame: ReplayFrame): void {
  if (frame.topologyAdds.length === 0) return;

  for (const add of frame.topologyAdds) {
    if (!state.nodes.some((node) => node.id === add.node.id)) {
      state.nodes.push(cloneTopologyNode(add.node));
    }

    for (const edge of add.edges) {
      if (!state.edges.some((existing) => existing.id === edge.id)) {
        state.edges.push(cloneTopologyEdge(edge));
      }
    }
  }
}

function applyStateDelta(state: RuntimeGraphState, frame: ReplayFrame): void {
  if (!frame.stateDelta) return;

  for (const nodeChange of frame.stateDelta.node_changes) {
    const node = state.nodes.find((entry) => entry.id === nodeChange.node_id);
    if (!node) continue;
    node.visual_state = nodeChange.visual_state;
    node.overlay = nodeChange.overlay;
  }

  for (const edgeChange of frame.stateDelta.edge_changes) {
    const edge = state.edges.find((entry) => entry.id === edgeChange.edge_id);
    if (!edge) continue;
    edge.visual_state = edgeChange.visual_state;
    edge.direction = edgeChange.direction;
  }
}

export function materializeGraphAtStep(
  timeline: ReplayTimeline,
  targetStep: number,
): RuntimeGraphState {
  const state = createInitialGraphState(timeline.topology);

  for (const frame of timeline.frames) {
    if (frame.step > targetStep) break;
    applyTopologyAdds(state, frame);
    applyStateDelta(state, frame);
  }

  return state;
}

export function collectActionsUntilStep(timeline: ReplayTimeline, targetStep: number): ActionEvent[] {
  const events: ActionEvent[] = [];
  for (const frame of timeline.frames) {
    if (frame.step > targetStep) break;
    events.push(...frame.actions);
  }
  return events;
}

export function getFrameByStep(timeline: ReplayTimeline, step: number): ReplayFrame | null {
  return timeline.frames.find((frame) => frame.step === step) ?? null;
}

export function maxReplayStep(timeline: ReplayTimeline): number {
  if (timeline.frames.length === 0) return 0;
  return timeline.frames[timeline.frames.length - 1]?.step ?? 0;
}
