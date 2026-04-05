"use client";

import { useEffect, useMemo, useRef } from "react";
import type { Core, CytoscapeOptions, ElementDefinition } from "cytoscape";
import type { TopologyInitData } from "../lib/integrationContract";
import type { RuntimeGraphState } from "../lib/replayRuntime";

type GraphProps = {
  topology: TopologyInitData | null;
  graphState: RuntimeGraphState | null;
  selectedNodeId: string | null;
  highlightedNodeId?: string | null;
  onNodeSelect: (nodeId: string | null) => void;
  resetViewToken?: number;
  className?: string;
};

const NODE_STATE_CLASSES = [
  "state-neutral",
  "state-monitored",
  "state-probed",
  "state-compromised",
  "state-critical",
  "state-isolated",
  "state-patched",
];

const EDGE_STATE_CLASSES = [
  "edge-normal",
  "edge-scanning",
  "edge-lateral_movement",
  "edge-exfiltration",
  "edge-credential_flow",
  "edge-blocked",
];

const EDGE_CONTEXT_CLASSES = ["edge-context-active", "edge-context-muted"];

const ZONE_CLASSES: Record<string, string> = {
  zone_perimeter: "zone-perimeter",
  zone_campus: "zone-campus",
  zone_admin: "zone-admin",
  zone_research: "zone-research",
};

type XYPosition = { x: number; y: number };

function stylesheet(): NonNullable<CytoscapeOptions["style"]> {
  const styles: Array<Record<string, unknown>> = [
    {
      selector: "node",
      style: {
        label: "data(label)",
        color: "#f1eee8",
        "font-size": 10,
        "font-family": "var(--font-mono-custom)",
        "text-wrap": "wrap",
        "text-max-width": "96px",
        "text-valign": "bottom",
        "text-margin-y": 8,
        "border-width": 1,
        "border-color": "#6b7688",
        "background-color": "#2c3340",
      },
    },
    {
      selector: "node:parent",
      style: {
        "background-opacity": 0.08,
        "background-color": "#121923",
        "border-width": 1.2,
        "border-style": "dashed",
        "border-color": "rgba(136,154,178,0.34)",
        label: "data(label)",
        "text-valign": "top",
        "text-halign": "center",
        "font-size": 11,
        color: "#9ca8bb",
        "text-transform": "uppercase",
        "padding-top": "26px",
        "padding-bottom": "26px",
        "padding-left": "28px",
        "padding-right": "28px",
      },
    },
    {
      selector: "node[type='endpoint']",
      style: { shape: "ellipse", width: 36, height: 36 },
    },
    {
      selector: "node[type='infrastructure']",
      style: { shape: "round-rectangle", width: 44, height: 44 },
    },
    {
      selector: "node[type='hvt']",
      style: { shape: "diamond", width: 52, height: 52 },
    },
    {
      selector: "node[type='iot']",
      style: { shape: "triangle", width: 32, height: 32 },
    },
    {
      selector: "node[type='external']",
      style: { shape: "hexagon", width: 40, height: 40 },
    },
    {
      selector: ".zone-perimeter",
      style: { "border-color": "#d0702c" },
    },
    {
      selector: ".zone-campus",
      style: { "border-color": "rgba(255,255,255,0.22)" },
    },
    {
      selector: ".zone-admin",
      style: { "border-color": "#7b8ea8" },
    },
    {
      selector: ".zone-research",
      style: { "border-color": "#6a788f" },
    },
    {
      selector: ".state-neutral",
      style: {
        "background-color": "#2c3340",
        "border-color": "#6b7688",
        "border-width": 1,
        "shadow-opacity": 0,
      },
    },
    {
      selector: ".state-monitored",
      style: {
        "background-color": "#2c3340",
        "border-color": "#1D9E75",
        "border-width": 2,
      },
    },
    {
      selector: ".state-probed",
      style: {
        "background-color": "#EF9F27",
        "border-color": "#BA7517",
        "border-width": 1.5,
      },
    },
    {
      selector: ".state-compromised",
      style: {
        "background-color": "#E24B4A",
        "border-color": "#707c90",
        "border-width": 1.5,
      },
    },
    {
      selector: ".state-critical",
      style: {
        "background-color": "#A32D2D",
        "border-color": "#7a879e",
        "border-width": 2,
        "shadow-color": "rgba(163,45,45,0.6)",
        "shadow-blur": 20,
        "shadow-opacity": 0.95,
      },
    },
    {
      selector: ".state-isolated",
      style: {
        "background-color": "#378ADD",
        "border-color": "#85B7EB",
        "border-width": 2.5,
      },
    },
    {
      selector: ".state-patched",
      style: {
        "background-color": "#2c3340",
        "border-color": "#378ADD",
        "border-style": "dashed",
        "border-width": 1.5,
      },
    },
    {
      selector: ".overlay-monitored",
      style: {
        "shadow-color": "rgba(29,158,117,0.65)",
        "shadow-opacity": 1,
        "shadow-blur": 14,
      },
    },
    {
      selector: ".decoy-node",
      style: { opacity: 0.5, "border-style": "dashed" },
    },
    {
      selector: "edge",
      style: {
        width: 1.15,
        "line-color": "rgba(193,206,226,0.24)",
        "curve-style": "bezier",
        "target-arrow-shape": "none",
        opacity: 0.75,
      },
    },
    {
      selector: ".edge-normal",
      style: {
        width: 1.15,
        "line-style": "solid",
        "line-color": "rgba(193,206,226,0.24)",
      },
    },
    {
      selector: ".edge-scanning",
      style: {
        width: 1.5,
        "line-color": "#EF9F27",
        "line-style": "dashed",
        "line-dash-pattern": [6, 4],
      },
    },
    {
      selector: ".edge-lateral_movement",
      style: {
        width: 2,
        "line-color": "#E24B4A",
        "target-arrow-shape": "triangle",
        "target-arrow-color": "#E24B4A",
      },
    },
    {
      selector: ".edge-exfiltration",
      style: {
        width: 3,
        "line-color": "#A32D2D",
        "target-arrow-shape": "triangle",
        "target-arrow-color": "#A32D2D",
      },
    },
    {
      selector: ".edge-credential_flow",
      style: {
        width: 2,
        "line-color": "#C56C1A",
        "target-arrow-shape": "triangle",
        "target-arrow-color": "#C56C1A",
      },
    },
    {
      selector: ".edge-blocked",
      style: {
        width: 1,
        "line-color": "#378ADD",
        "line-style": "dashed",
        "line-dash-pattern": [4, 6],
      },
    },
    {
      selector: ".edge-context-active",
      style: {
        opacity: 1,
        "line-opacity": 1,
        width: 2.6,
        "z-index": 999,
      },
    },
    {
      selector: ".edge-context-muted",
      style: {
        opacity: 0.16,
        "line-opacity": 0.16,
      },
    },
    {
      selector: ".node-selected",
      style: {
        "overlay-opacity": 0,
        "overlay-padding": 0,
      },
    },
    {
      selector: ".node-highlighted",
      style: {
        "underlay-opacity": 0,
        "underlay-padding": 0,
      },
    },
  ];
  return styles as unknown as NonNullable<CytoscapeOptions["style"]>;
}

function zoneClass(zoneId: string): string {
  return ZONE_CLASSES[zoneId] ?? "";
}

function topDownPositions(): Record<string, XYPosition> {
  return {
    internet: { x: 760, y: 70 },

    vpn_gateway: { x: 620, y: 225 },
    web_portal: { x: 760, y: 225 },
    dns_server: { x: 900, y: 225 },

    eduroam_ap_01: { x: 560, y: 395 },
    eduroam_ap_02: { x: 760, y: 395 },
    eduroam_ap_03: { x: 960, y: 395 },

    student_device_01: { x: 500, y: 520 },
    student_device_02: { x: 560, y: 560 },
    student_device_03: { x: 700, y: 530 },
    student_device_04: { x: 760, y: 565 },
    student_device_05: { x: 860, y: 530 },

    faculty_device_01: { x: 620, y: 470 },
    faculty_device_02: { x: 760, y: 470 },
    faculty_device_03: { x: 900, y: 470 },

    lab_workstation_01: { x: 680, y: 630 },
    lab_workstation_02: { x: 840, y: 630 },
    print_server: { x: 700, y: 670 },
    iot_projector_01: { x: 860, y: 670 },

    auth_server: { x: 540, y: 820 },
    active_directory: { x: 660, y: 790 },
    sis_server: { x: 780, y: 820 },
    finance_server: { x: 620, y: 890 },
    hr_server: { x: 790, y: 890 },

    research_server_01: { x: 940, y: 805 },
    research_server_02: { x: 1050, y: 805 },
    shared_storage: { x: 995, y: 880 },
    irb_system: { x: 1090, y: 920 },
  };
}

function fallbackPositionForZone(zone: string, index: number): XYPosition {
  const spacingX = 64;
  if (zone === "perimeter") return { x: 600 + index * spacingX, y: 225 };
  if (zone === "campus") return { x: 510 + (index % 7) * spacingX, y: 470 + Math.floor(index / 7) * 78 };
  if (zone === "admin") return { x: 520 + index * spacingX, y: 840 };
  if (zone === "research") return { x: 930 + index * spacingX, y: 840 };
  return { x: 760, y: 70 };
}

function applyDeterministicPositions(cy: Core, topology: TopologyInitData): void {
  const explicit = topDownPositions();
  const zoneIndices = new Map<string, number>();

  cy.batch(() => {
    for (const node of topology.nodes) {
      const ele = cy.getElementById(node.id);
      if (!ele.length) continue;

      const existingIndex = zoneIndices.get(node.zone) ?? 0;
      const target = explicit[node.id] ?? fallbackPositionForZone(node.zone, existingIndex);
      zoneIndices.set(node.zone, existingIndex + 1);
      ele.position(target);
    }
  });

  cy.layout({
    name: "preset",
    fit: true,
    padding: 48,
    animate: false,
  }).run();
}

function upsertBaseTopology(cy: Core, topology: TopologyInitData, graphState: RuntimeGraphState): void {
  cy.elements().remove();

  const zoneElements: ElementDefinition[] = topology.zones.map((zone) => ({
    group: "nodes",
    data: { id: zone.id, label: zone.label },
    classes: zoneClass(zone.id),
  }));

  const nodeElements: ElementDefinition[] = graphState.nodes.map((node) => ({
    group: "nodes",
    data: {
      id: node.id,
      label: node.label,
      type: node.type,
      zone: node.zone,
      parent: node.zone === "external" ? undefined : `zone_${node.zone}`,
    },
    classes: [
      `state-${node.visual_state}`,
      node.overlay === "monitored" ? "overlay-monitored" : "",
      node.is_decoy ? "decoy-node" : "",
    ]
      .filter(Boolean)
      .join(" "),
  }));

  const edgeElements: ElementDefinition[] = graphState.edges.map((edge) => ({
    group: "edges",
    data: {
      id: edge.id,
      source: edge.source,
      target: edge.target,
      direction: edge.direction,
    },
    classes: `edge-${edge.visual_state}`,
  }));

  cy.add([...zoneElements, ...nodeElements, ...edgeElements]);
  applyDeterministicPositions(cy, topology);
  cy.resize();
  cy.fit(cy.elements(), 48);
}

function syncGraphState(cy: Core, graphState: RuntimeGraphState): void {
  const nodeIds = new Set(graphState.nodes.map((node) => node.id));
  const edgeIds = new Set(graphState.edges.map((edge) => edge.id));

  cy.batch(() => {
    cy.nodes().forEach((node) => {
      if (node.id().startsWith("zone_")) return;
      if (!nodeIds.has(node.id())) node.remove();
    });

    cy.edges().forEach((edge) => {
      if (!edgeIds.has(edge.id())) edge.remove();
    });

    for (const runtimeNode of graphState.nodes) {
      let node = cy.getElementById(runtimeNode.id);
      if (!node.length) {
        cy.add({
          group: "nodes",
          data: {
            id: runtimeNode.id,
            label: runtimeNode.label,
            type: runtimeNode.type,
            zone: runtimeNode.zone,
            parent: runtimeNode.zone === "external" ? undefined : `zone_${runtimeNode.zone}`,
          },
          classes: runtimeNode.is_decoy ? "decoy-node" : "",
        });
        node = cy.getElementById(runtimeNode.id);

        if (runtimeNode.is_decoy) {
          const parent = cy.getElementById("auth_server");
          if (parent.length) {
            const pos = parent.position();
            node.position({ x: pos.x + 84, y: pos.y - 38 });
          }
        }
      }

      node.removeClass(NODE_STATE_CLASSES.join(" "));
      node.removeClass("overlay-monitored decoy-node");
      node.addClass(`state-${runtimeNode.visual_state}`);
      if (runtimeNode.overlay === "monitored") node.addClass("overlay-monitored");
      if (runtimeNode.is_decoy) node.addClass("decoy-node");
    }

    for (const runtimeEdge of graphState.edges) {
      let edge = cy.getElementById(runtimeEdge.id);
      if (!edge.length) {
        cy.add({
          group: "edges",
          data: {
            id: runtimeEdge.id,
            source: runtimeEdge.source,
            target: runtimeEdge.target,
          },
          classes: `edge-${runtimeEdge.visual_state}`,
        });
        edge = cy.getElementById(runtimeEdge.id);
      }

      edge.removeClass(EDGE_STATE_CLASSES.join(" "));
      edge.addClass(`edge-${runtimeEdge.visual_state}`);
      edge.data("direction", runtimeEdge.direction ?? "forward");
    }
  });
}

export default function Graph({
  topology,
  graphState,
  selectedNodeId,
  highlightedNodeId,
  onNodeSelect,
  resetViewToken = 0,
  className,
}: GraphProps) {
  const containerRef = useRef<HTMLDivElement | null>(null);
  const cyRef = useRef<Core | null>(null);
  const onNodeSelectRef = useRef(onNodeSelect);
  const topologyKeyRef = useRef<string | null>(null);
  const lastCenteredNodeRef = useRef<string | null>(null);

  const shellClass = useMemo(
    () => `${className ?? "h-[620px] w-full rounded-xl border border-[var(--border)] bg-[#0d1018]"} relative overflow-hidden`,
    [className],
  );

  useEffect(() => {
    onNodeSelectRef.current = onNodeSelect;
  }, [onNodeSelect]);

  useEffect(() => {
    let mounted = true;

    const init = async () => {
      if (!containerRef.current || cyRef.current) return;
      const cytoscape = (await import("cytoscape")).default;
      if (!mounted || !containerRef.current) return;

      cyRef.current = cytoscape({
        container: containerRef.current,
        elements: [],
        style: stylesheet(),
        minZoom: 0.5,
        maxZoom: 2.5,
      });

      const cy = cyRef.current;
      cy.on("tap", "node", (evt) => {
        const id = String(evt.target.id());
        if (id.startsWith("zone_")) return;
        onNodeSelectRef.current(id);
      });

      cy.on("tap", (evt) => {
        if (evt.target === cy) onNodeSelectRef.current(null);
      });
    };

    void init();

    return () => {
      mounted = false;
      if (cyRef.current) {
        cyRef.current.destroy();
        cyRef.current = null;
      }
    };
  }, []);

  useEffect(() => {
    if (!containerRef.current) return;
    const observer = new ResizeObserver(() => {
      const cy = cyRef.current;
      if (!cy) return;
      cy.resize();
    });
    observer.observe(containerRef.current);
    return () => observer.disconnect();
  }, []);

  useEffect(() => {
    const cy = cyRef.current;
    if (!cy || cy.nodes().length === 0) return;
    cy.resize();
    cy.fit(cy.elements(), 48);
  }, [resetViewToken]);

  useEffect(() => {
    const cy = cyRef.current;
    if (!cy || !topology || !graphState) return;

    const topologyKey = `${topology.scenario_id}:${topology.seed}:${topology.nodes.length}:${topology.edges.length}`;
    if (topologyKeyRef.current === topologyKey) return;
    topologyKeyRef.current = topologyKey;

    upsertBaseTopology(cy, topology, graphState);
  }, [topology, graphState]);

  useEffect(() => {
    const cy = cyRef.current;
    if (!cy || !graphState) return;
    syncGraphState(cy, graphState);
  }, [graphState]);

  useEffect(() => {
    const cy = cyRef.current;
    if (!cy) return;

    if (!selectedNodeId) {
      lastCenteredNodeRef.current = null;
      return;
    }

    if (lastCenteredNodeRef.current === selectedNodeId) return;
    lastCenteredNodeRef.current = selectedNodeId;

    const selected = cy.getElementById(selectedNodeId);
    if (!selected.length) return;
    cy.animate({
      center: { eles: selected },
    }, { duration: 220 });
  }, [selectedNodeId]);

  useEffect(() => {
    const cy = cyRef.current;
    if (!cy) return;

    const focusNodeId = highlightedNodeId ?? selectedNodeId;

    cy.batch(() => {
      cy.nodes().removeClass("node-selected node-highlighted");
      cy.edges().removeClass(EDGE_CONTEXT_CLASSES.join(" "));

      if (selectedNodeId) {
        const selected = cy.getElementById(selectedNodeId);
        if (selected.length) {
          selected.addClass("node-selected");
        }
      }

      if (highlightedNodeId) {
        const highlighted = cy.getElementById(highlightedNodeId);
        if (highlighted.length) highlighted.addClass("node-highlighted");
      }

      if (focusNodeId) {
        cy.edges().forEach((edge) => {
          const source = String(edge.data("source"));
          const target = String(edge.data("target"));
          if (source === focusNodeId || target === focusNodeId) {
            edge.addClass("edge-context-active");
          } else {
            edge.addClass("edge-context-muted");
          }
        });
      }
    });
  }, [selectedNodeId, highlightedNodeId, graphState]);

  const showLoadingState = !topology || !graphState;
  const showEmptyState = Boolean(topology && graphState && graphState.nodes.length === 0);

  return (
    <div className={shellClass} aria-label="AEGIS threat topology matrix">
      <div ref={containerRef} className="h-full w-full" />
      {showLoadingState ? (
        <div className="pointer-events-none absolute inset-0 flex items-center justify-center bg-black/25 text-xs uppercase tracking-[0.12em] text-[#bba9a9]">
          Loading topology...
        </div>
      ) : null}
      {showEmptyState ? (
        <div className="pointer-events-none absolute inset-0 flex items-center justify-center bg-black/45 px-6 text-center text-xs uppercase tracking-[0.12em] text-[#ffd7d3]">
          Topology unavailable for this run. Check replay source and contract format.
        </div>
      ) : null}
    </div>
  );
}
