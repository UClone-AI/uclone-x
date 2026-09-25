import React, { useState, useEffect, useMemo, useRef } from 'react';
import {
  Brain,
  Search,
  Filter,
  RefreshCw,
  GitFork,
  X,
  Layers,
  Table as TableIcon,
  ZoomIn,
  ZoomOut,
  Maximize2,
} from 'lucide-react';
import { roomDockUrls, useRoomRead, type SeatKnowledge } from '../../lib/roomDock';
import {
  KnowledgeGraphResponse,
  KnowledgeGraphNode,
  KnowledgeGraphEdge,
} from '../../types';
import { Badge } from '../ui/Badge';
import { Button } from '../ui/Button';

interface KnowledgeGraphViewerProps {
  /** The conversation on screen. */
  roomId: string | null;
  /** The seat whose graph is drawn; the dock's seat picker chooses it (#1356). */
  seatId: string | null;
  /** Changes when the conversation moves on, to read again. */
  refreshKey?: unknown;
}

interface SimNode extends KnowledgeGraphNode {
  x: number;
  y: number;
  vx: number;
  vy: number;
}

const TIER_COLORS: Record<string, { bg: string; border: string; glow: string; text: string; badge: string }> = {
  asserted: {
    bg: 'fill-emerald-950/80',
    border: 'stroke-emerald-400',
    glow: 'rgba(52, 211, 153, 0.4)',
    text: 'text-emerald-400',
    badge: 'bg-emerald-500/20 text-emerald-300 border-emerald-500/30',
  },
  induced_enforcing: {
    bg: 'fill-blue-950/80',
    border: 'stroke-blue-400',
    glow: 'rgba(96, 165, 250, 0.4)',
    text: 'text-blue-400',
    badge: 'bg-blue-500/20 text-blue-300 border-blue-500/30',
  },
  induced_candidate: {
    bg: 'fill-amber-950/80',
    border: 'stroke-amber-400',
    glow: 'rgba(251, 191, 36, 0.4)',
    text: 'text-amber-400',
    badge: 'bg-amber-500/20 text-amber-300 border-amber-500/30',
  },
};

const DEFAULT_TIER_STYLE = {
  bg: 'fill-slate-900/90',
  border: 'stroke-slate-400',
  glow: 'rgba(148, 163, 184, 0.3)',
  text: 'text-slate-300',
  badge: 'bg-slate-700/50 text-slate-300 border-slate-600',
};

export const KnowledgeGraphViewer: React.FC<KnowledgeGraphViewerProps> = ({
  roomId,
  seatId,
  refreshKey,
}) => {
  const [reloads, setReloads] = useState(0);
  const read = useRoomRead<SeatKnowledge>(
    roomId && seatId ? roomDockUrls.knowledge(roomId, seatId) : null,
    `${String(refreshKey ?? '')}:${reloads}`,
  );
  const loading = read.loading;
  const error = read.error;
  // Only `ok` carries a graph; the other statuses carry the Core's reason instead (P6).
  const data: KnowledgeGraphResponse | null =
    read.data?.status === 'ok' ? (read.data as KnowledgeGraphResponse) : null;
  const notice = !roomId
    ? 'No conversation is open.'
    : !seatId
      ? 'No agent is seated in this conversation.'
      : // A non-`ok` status, or an `ok` graph with nothing in it: the Core says why.
        (read.data?.reason ?? null);

  // Filters & Controls
  const [tierFilter, setTierFilter] = useState<string>('all');
  const [searchQuery, setSearchQuery] = useState<string>('');
  const [viewMode, setViewMode] = useState<'graph' | 'table'>('graph');

  // Selection & Provenance Popover
  const [selectedNode, setSelectedNode] = useState<KnowledgeGraphNode | null>(null);
  const [selectedEdge, setSelectedEdge] = useState<KnowledgeGraphEdge | null>(null);

  // Pan & Zoom
  const [zoom, setZoom] = useState<number>(1);
  const [pan, setPan] = useState<{ x: number; y: number }>({ x: 0, y: 0 });
  const [isPanning, setIsPanning] = useState<boolean>(false);
  const [startPan, setStartPan] = useState<{ x: number; y: number }>({ x: 0, y: 0 });

  // Dragging Node
  const [draggedNodeId, setDraggedNodeId] = useState<string | null>(null);

  const svgRef = useRef<SVGSVGElement | null>(null);

  // Filtered nodes and edges
  const { filteredNodes, filteredEdges, filteredTriples } = useMemo(() => {
    if (!data) return { filteredNodes: [], filteredEdges: [], filteredTriples: [] };

    const query = searchQuery.trim().toLowerCase();

    // 1. Filter nodes
    const nMap = new Map<string, KnowledgeGraphNode>();
    data.nodes.forEach((n) => {
      const matchesTier = tierFilter === 'all' || n.tier === tierFilter;
      const matchesSearch = !query || n.name.toLowerCase().includes(query) || n.id.toLowerCase().includes(query);
      if (matchesTier && matchesSearch) {
        nMap.set(n.id, n);
      }
    });

    // 2. Filter edges (keep edges whose source and target are both in nMap)
    const eList = data.edges.filter((e) => {
      const hasNodes = nMap.has(e.source) && nMap.has(e.target);
      const matchesTier = tierFilter === 'all' || e.tier === tierFilter;
      return hasNodes && matchesTier;
    });

    // 3. Filter triples
    const tList = data.triples.filter((tr) => {
      const matchesTier = tierFilter === 'all' || tr.tier === tierFilter;
      const matchesSearch =
        !query ||
        tr.subject.toLowerCase().includes(query) ||
        tr.predicate.toLowerCase().includes(query) ||
        tr.object.toLowerCase().includes(query);
      return matchesTier && matchesSearch;
    });

    return {
      filteredNodes: Array.from(nMap.values()),
      filteredEdges: eList,
      filteredTriples: tList,
    };
  }, [data, tierFilter, searchQuery]);

  // 2D Deterministic Spring Layout Simulation
  const [simulationNodes, setSimulationNodes] = useState<SimNode[]>([]);

  useEffect(() => {
    if (filteredNodes.length === 0) {
      setSimulationNodes([]);
      return;
    }

    const width = 600;
    const height = 450;
    const centerX = width / 2;
    const centerY = height / 2;

    // Initialize positions deterministically in a ring
    const nodes: SimNode[] = filteredNodes.map((n, i) => {
      const angle = (2 * Math.PI * i) / filteredNodes.length;
      const radius = Math.min(width, height) * 0.35;
      return {
        ...n,
        x: centerX + radius * Math.cos(angle),
        y: centerY + radius * Math.sin(angle),
        vx: 0,
        vy: 0,
      };
    });

    const nodeIndex = new Map<string, number>();
    nodes.forEach((n, i) => nodeIndex.set(n.id, i));

    // Run 50 iterations of spring / repulsion layout
    const iterations = 50;
    const kRepulsion = 2500;
    const kSpring = 0.05;
    const restLength = 90;
    const damping = 0.85;

    for (let iter = 0; iter < iterations; iter++) {
      // Repulsion between all node pairs
      for (let i = 0; i < nodes.length; i++) {
        for (let j = i + 1; j < nodes.length; j++) {
          const n1 = nodes[i];
          const n2 = nodes[j];
          const dx = n2.x - n1.x;
          const dy = n2.y - n1.y;
          const dist = Math.sqrt(dx * dx + dy * dy) || 1;
          const force = kRepulsion / (dist * dist);
          const fx = (dx / dist) * force;
          const fy = (dy / dist) * force;
          n1.vx -= fx;
          n1.vy -= fy;
          n2.vx += fx;
          n2.vy += fy;
        }
      }

      // Spring attraction along edges
      for (const e of filteredEdges) {
        const i1 = nodeIndex.get(e.source);
        const i2 = nodeIndex.get(e.target);
        if (i1 === undefined || i2 === undefined) continue;
        const n1 = nodes[i1];
        const n2 = nodes[i2];
        const dx = n2.x - n1.x;
        const dy = n2.y - n1.y;
        const dist = Math.sqrt(dx * dx + dy * dy) || 1;
        const displacement = dist - restLength;
        const force = kSpring * displacement;
        const fx = (dx / dist) * force;
        const fy = (dy / dist) * force;
        n1.vx += fx;
        n1.vy += fy;
        n2.vx += fx;
        n2.vy += fy;
      }

      // Gravity towards center
      for (const n of nodes) {
        const dx = centerX - n.x;
        const dy = centerY - n.y;
        n.vx += dx * 0.01;
        n.vy += dy * 0.01;

        n.vx *= damping;
        n.vy *= damping;
        n.x += n.vx;
        n.y += n.vy;
      }
    }

    setSimulationNodes(nodes);
  }, [filteredNodes, filteredEdges]);

  // Fast map from nodeId to SimNode
  const simNodeMap = useMemo(() => {
    const map = new Map<string, SimNode>();
    simulationNodes.forEach((n) => map.set(n.id, n));
    return map;
  }, [simulationNodes]);

  // Pan / Zoom handlers
  const handleMouseDown = (e: React.MouseEvent<SVGSVGElement>) => {
    if (e.target === svgRef.current || (e.target as HTMLElement).tagName === 'svg') {
      setIsPanning(true);
      setStartPan({ x: e.clientX - pan.x, y: e.clientY - pan.y });
    }
  };

  const handleMouseMove = (e: React.MouseEvent<SVGSVGElement>) => {
    if (draggedNodeId) {
      const rect = svgRef.current?.getBoundingClientRect();
      if (rect) {
        const currentSvgX = (e.clientX - rect.left - pan.x) / zoom;
        const currentSvgY = (e.clientY - rect.top - pan.y) / zoom;
        setSimulationNodes((prev) =>
          prev.map((n) => (n.id === draggedNodeId ? { ...n, x: currentSvgX, y: currentSvgY } : n))
        );
      }
      return;
    }

    if (isPanning) {
      setPan({
        x: e.clientX - startPan.x,
        y: e.clientY - startPan.y,
      });
    }
  };

  const handleMouseUp = () => {
    setIsPanning(false);
    setDraggedNodeId(null);
  };

  const handleResetView = () => {
    setZoom(1);
    setPan({ x: 0, y: 0 });
  };

  const handleSelectNode = (node: KnowledgeGraphNode) => {
    setSelectedNode(node);
    setSelectedEdge(null);
  };

  const handleSelectEdge = (edge: KnowledgeGraphEdge) => {
    setSelectedEdge(edge);
    setSelectedNode(null);
  };

  return (
    <div data-testid="knowledge-graph-viewer" className="flex flex-col h-full bg-slate-950 text-slate-200">
      {/* Top Controls Header */}
      <div className="flex flex-wrap items-center justify-between gap-2 p-3 border-b border-slate-800 bg-slate-900/60 shrink-0">
        <div className="flex items-center gap-2">
          <Brain className="w-5 h-5 text-indigo-400 shrink-0" />
          <span className="font-semibold text-sm text-slate-100">Knowledge Graph</span>
          {data && (
            <Badge
              tone="info"
              data-testid="kg-node-count"
              className="text-xs font-mono bg-indigo-500/10 text-indigo-400 border-indigo-500/20"
            >
              {data.summary.total_nodes} nodes · {data.summary.total_edges} edges
            </Badge>
          )}
        </div>

        {/* View Mode Toggle & Refresh */}
        <div className="flex items-center gap-1.5">
          <div className="flex items-center rounded-lg bg-slate-800 p-0.5 border border-slate-700">
            <button
              type="button"
              data-testid="view-graph-btn"
              onClick={() => setViewMode('graph')}
              className={`px-2 py-1 text-xs font-medium rounded-md flex items-center gap-1 transition-colors ${
                viewMode === 'graph' ? 'bg-indigo-600 text-white' : 'text-slate-400 hover:text-slate-200'
              }`}
            >
              <Layers className="w-3.5 h-3.5" />
              Graph
            </button>
            <button
              type="button"
              data-testid="view-table-btn"
              onClick={() => setViewMode('table')}
              className={`px-2 py-1 text-xs font-medium rounded-md flex items-center gap-1 transition-colors ${
                viewMode === 'table' ? 'bg-indigo-600 text-white' : 'text-slate-400 hover:text-slate-200'
              }`}
            >
              <TableIcon className="w-3.5 h-3.5" />
              Triples
            </button>
          </div>

          <Button
            variant="bordered"
            size="icon"
            data-testid="kg-refresh-btn"
            onClick={() => setReloads((n) => n + 1)}
            disabled={loading || !roomId || !seatId}
            title="Refresh Knowledge Graph"
          >
            <RefreshCw className={`w-3.5 h-3.5 ${loading ? 'animate-spin' : ''}`} />
          </Button>
        </div>
      </div>

      {/* Filter / Search Bar */}
      <div className="flex flex-wrap items-center gap-2 p-2.5 bg-slate-900/40 border-b border-slate-800 text-xs shrink-0">
        {/* Tier Filter */}
        <div className="flex items-center gap-1.5 bg-slate-800/80 px-2 py-1 rounded-md border border-slate-700">
          <Filter className="w-3.5 h-3.5 text-indigo-400" />
          <span className="text-slate-400 font-medium">Tier:</span>
          <select
            data-testid="tier-filter-select"
            value={tierFilter}
            onChange={(e) => setTierFilter(e.target.value)}
            className="bg-transparent text-slate-200 outline-none cursor-pointer text-xs font-medium"
          >
            <option value="all" className="bg-slate-900 text-slate-200">
              All Tiers
            </option>
            <option value="asserted" className="bg-slate-900 text-emerald-300">
              Asserted (Axiomatic)
            </option>
            <option value="induced_enforcing" className="bg-slate-900 text-blue-300">
              Induced Enforcing
            </option>
            <option value="induced_candidate" className="bg-slate-900 text-amber-300">
              Induced Candidate
            </option>
          </select>
        </div>

        {/* Search Input */}
        <div className="flex-1 min-w-[140px] relative">
          <Search className="w-3.5 h-3.5 text-slate-500 absolute left-2.5 top-1/2 -translate-y-1/2" />
          <input
            data-testid="kg-search-input"
            type="text"
            value={searchQuery}
            onChange={(e) => setSearchQuery(e.target.value)}
            placeholder="Search entity or predicate..."
            className="w-full bg-slate-800/80 border border-slate-700 rounded-md pl-8 pr-3 py-1 text-xs text-slate-200 placeholder-slate-500 focus:outline-none focus:border-indigo-500"
          />
        </div>
      </div>

      {/* Main Canvas / Table Area */}
      <div className="flex-1 relative overflow-hidden flex flex-col min-h-0">
        {loading && (
          <div className="absolute inset-0 bg-slate-950/70 z-10 flex items-center justify-center">
            <div className="flex items-center gap-2 text-indigo-400 text-xs font-medium">
              <RefreshCw className="w-4 h-4 animate-spin" />
              Loading Knowledge Graph...
            </div>
          </div>
        )}

        {notice && (
          <p data-testid="kg-notice" className="p-4 m-3 text-xs text-slate-300">
            {notice}
          </p>
        )}

        {error && (
          <div className="p-4 m-3 rounded-lg bg-red-950/50 border border-red-800 text-red-300 text-xs">
            <p className="font-semibold">Failed to fetch knowledge graph:</p>
            <p className="mt-1 font-mono text-[11px]">{error}</p>
          </div>
        )}

        {viewMode === 'table' ? (
          /* Triples Table View */
          <div data-testid="kg-triples-table" className="flex-1 overflow-auto p-3">
            <table className="w-full text-left text-xs border-collapse">
              <thead>
                <tr className="border-b border-slate-800 text-slate-400 font-semibold bg-slate-900/50">
                  <th className="p-2">Subject</th>
                  <th className="p-2">Predicate</th>
                  <th className="p-2">Object</th>
                  <th className="p-2">Tier</th>
                  <th className="p-2">Provenance</th>
                </tr>
              </thead>
              <tbody className="divide-y divide-slate-800/60 font-mono text-[11px]">
                {filteredTriples.length === 0 ? (
                  <tr>
                    <td colSpan={5} className="p-4 text-center text-slate-500">
                      No matching triples found.
                    </td>
                  </tr>
                ) : (
                  filteredTriples.map((tr, idx) => (
                    <tr
                      key={idx}
                      onClick={() => {
                        handleSelectNode({
                          id: tr.subject,
                          name: tr.subject,
                          tier: tr.tier,
                          type: 'entity',
                          provenance: tr.provenance,
                        });
                      }}
                      className="hover:bg-slate-800/50 cursor-pointer transition-colors"
                    >
                      <td className="p-2 font-medium text-slate-200">{tr.subject}</td>
                      <td className="p-2 text-indigo-400">{tr.predicate}</td>
                      <td className="p-2 text-slate-300">{tr.object}</td>
                      <td className="p-2">
                        <span
                          className={`px-1.5 py-0.5 rounded text-[10px] uppercase font-semibold border ${
                            TIER_COLORS[tr.tier]?.badge || DEFAULT_TIER_STYLE.badge
                          }`}
                        >
                          {tr.tier}
                        </span>
                      </td>
                      <td className="p-2 text-slate-400">
                        {tr.provenance?.origin || 'axiomatic'} ({tr.provenance?.agent_id || 'default'})
                      </td>
                    </tr>
                  ))
                )}
              </tbody>
            </table>
          </div>
        ) : (
          /* Interactive Force-Directed SVG Canvas */
          <div className="flex-1 w-full h-full relative cursor-grab active:cursor-grabbing select-none overflow-hidden">
            {/* Zoom Controls Overlay */}
            <div className="absolute right-3 top-3 z-10 flex flex-col gap-1 bg-slate-900/80 border border-slate-800 rounded-lg p-1 shadow-lg">
              <Button
                variant="ghost"
                size="icon"
                data-testid="kg-zoom-in"
                onClick={() => setZoom((z) => Math.min(2.5, z + 0.2))}
                title="Zoom In"
              >
                <ZoomIn className="w-3.5 h-3.5" />
              </Button>
              <Button
                variant="ghost"
                size="icon"
                data-testid="kg-zoom-out"
                onClick={() => setZoom((z) => Math.max(0.4, z - 0.2))}
                title="Zoom Out"
              >
                <ZoomOut className="w-3.5 h-3.5" />
              </Button>
              <Button
                variant="ghost"
                size="icon"
                data-testid="kg-zoom-reset"
                onClick={handleResetView}
                title="Reset View"
              >
                <Maximize2 className="w-3.5 h-3.5" />
              </Button>
            </div>

            <svg
              ref={svgRef}
              data-testid="kg-svg-canvas"
              className="w-full h-full"
              onMouseDown={handleMouseDown}
              onMouseMove={handleMouseMove}
              onMouseUp={handleMouseUp}
            >
              <defs>
                {/* Arrow markers for directed edges */}
                <marker
                  id="arrow-asserted"
                  viewBox="0 0 10 10"
                  refX="18"
                  refY="5"
                  markerWidth="6"
                  markerHeight="6"
                  orient="auto-start-reverse"
                >
                  <path d="M 0 0 L 10 5 L 0 10 z" fill="#34d399" />
                </marker>
                <marker
                  id="arrow-enforcing"
                  viewBox="0 0 10 10"
                  refX="18"
                  refY="5"
                  markerWidth="6"
                  markerHeight="6"
                  orient="auto-start-reverse"
                >
                  <path d="M 0 0 L 10 5 L 0 10 z" fill="#60a5fa" />
                </marker>
                <marker
                  id="arrow-candidate"
                  viewBox="0 0 10 10"
                  refX="18"
                  refY="5"
                  markerWidth="6"
                  markerHeight="6"
                  orient="auto-start-reverse"
                >
                  <path d="M 0 0 L 10 5 L 0 10 z" fill="#fbbf24" />
                </marker>
                <marker
                  id="arrow-default"
                  viewBox="0 0 10 10"
                  refX="18"
                  refY="5"
                  markerWidth="6"
                  markerHeight="6"
                  orient="auto-start-reverse"
                >
                  <path d="M 0 0 L 10 5 L 0 10 z" fill="#94a3b8" />
                </marker>
              </defs>

              <g transform={`translate(${pan.x}, ${pan.y}) scale(${zoom})`}>
                {/* 1. Edges */}
                {filteredEdges.map((edge) => {
                  const sourceNode = simNodeMap.get(edge.source);
                  const targetNode = simNodeMap.get(edge.target);
                  if (!sourceNode || !targetNode) return null;

                  const markerId =
                    edge.tier === 'asserted'
                      ? 'url(#arrow-asserted)'
                      : edge.tier === 'induced_enforcing'
                      ? 'url(#arrow-enforcing)'
                      : edge.tier === 'induced_candidate'
                      ? 'url(#arrow-candidate)'
                      : 'url(#arrow-default)';

                  const strokeColor =
                    edge.tier === 'asserted'
                      ? '#34d399'
                      : edge.tier === 'induced_enforcing'
                      ? '#60a5fa'
                      : edge.tier === 'induced_candidate'
                      ? '#fbbf24'
                      : '#64748b';

                  const midX = (sourceNode.x + targetNode.x) / 2;
                  const midY = (sourceNode.y + targetNode.y) / 2;

                  return (
                    <g
                      key={edge.id}
                      data-testid={`edge-${edge.id}`}
                      className="cursor-pointer group"
                      onClick={(e) => {
                        e.stopPropagation();
                        handleSelectEdge(edge);
                      }}
                    >
                      <line
                        x1={sourceNode.x}
                        y1={sourceNode.y}
                        x2={targetNode.x}
                        y2={targetNode.y}
                        stroke={strokeColor}
                        strokeWidth={1.5}
                        strokeDasharray={edge.tier === 'induced_candidate' ? '4 2' : undefined}
                        markerEnd={markerId}
                        className="group-hover:stroke-indigo-400 group-hover:stroke-[2.5px] transition-all"
                      />
                      {/* Edge predicate label */}
                      <text
                        x={midX}
                        y={midY - 4}
                        textAnchor="middle"
                        fill="#cbd5e1"
                        fontSize={9}
                        fontFamily="monospace"
                        className="bg-slate-900 px-1 py-0.5 pointer-events-none drop-shadow-sm select-none"
                      >
                        {edge.predicate}
                      </text>
                    </g>
                  );
                })}

                {/* 2. Nodes */}
                {simulationNodes.map((node) => {
                  const tierStyle = TIER_COLORS[node.tier] || DEFAULT_TIER_STYLE;
                  const isSelected = selectedNode?.id === node.id;
                  const isConcept = node.type === 'concept';

                  return (
                    <g
                      key={node.id}
                      data-testid={`node-${node.id}`}
                      transform={`translate(${node.x}, ${node.y})`}
                      className="cursor-pointer"
                      onClick={(e) => {
                        e.stopPropagation();
                        handleSelectNode(node);
                      }}
                      onMouseDown={(e) => {
                        e.stopPropagation();
                        setDraggedNodeId(node.id);
                      }}
                    >
                      {/* Node Glow Circle */}
                      <circle
                        r={isConcept ? 16 : 14}
                        className={`${tierStyle.bg} ${tierStyle.border} transition-all`}
                        strokeWidth={isSelected ? 3 : 1.5}
                        style={{
                          filter: isSelected ? `drop-shadow(0 0 8px ${tierStyle.glow})` : undefined,
                        }}
                      />

                      {/* Icon Indicator inside Node */}
                      <circle r={3} fill="#f8fafc" className="opacity-80" />

                      {/* Node Label */}
                      <text
                        y={26}
                        textAnchor="middle"
                        fill="#f1f5f9"
                        fontSize={10}
                        fontWeight={600}
                        fontFamily="system-ui, sans-serif"
                        className="pointer-events-none drop-shadow-md select-none"
                      >
                        {node.name}
                      </text>
                      <text
                        y={37}
                        textAnchor="middle"
                        fill="#94a3b8"
                        fontSize={8}
                        fontFamily="monospace"
                        className="pointer-events-none select-none"
                      >
                        {node.tier}
                      </text>
                    </g>
                  );
                })}
              </g>
            </svg>
          </div>
        )}

        {/* Provenance Inspection Popover (P6) */}
        {selectedNode && (
          <div
            data-testid="kg-provenance-popover"
            className="absolute bottom-3 left-3 right-3 max-w-md bg-slate-900/95 border border-indigo-500/40 rounded-xl p-3.5 shadow-2xl backdrop-blur-md z-20"
          >
            <div className="flex items-start justify-between gap-2 border-b border-slate-800 pb-2 mb-2">
              <div className="flex items-center gap-2">
                <Brain className="w-4 h-4 text-indigo-400 shrink-0" />
                <div>
                  <h4 className="text-xs font-bold text-slate-100 flex items-center gap-1.5">
                    {selectedNode.name}
                    <span
                      className={`px-1.5 py-0.2 rounded text-[9px] uppercase font-mono font-semibold border ${
                        TIER_COLORS[selectedNode.tier]?.badge || DEFAULT_TIER_STYLE.badge
                      }`}
                    >
                      {selectedNode.tier}
                    </span>
                  </h4>
                  <span className="text-[10px] text-slate-400 font-mono">Type: {selectedNode.type}</span>
                </div>
              </div>

              <Button
                variant="ghost"
                size="icon"
                data-testid="close-provenance-btn"
                onClick={() => setSelectedNode(null)}
              >
                <X className="w-3.5 h-3.5" />
              </Button>
            </div>

            {/* Provenance Attributes (P6 In-Band Traceability) */}
            <div className="space-y-1.5 text-xs">
              <div className="flex items-center justify-between text-[11px]">
                <span className="text-slate-400">Origin:</span>
                <span className="font-mono text-slate-200">
                  {selectedNode.provenance?.origin || 'axiomatic'}
                </span>
              </div>
              <div className="flex items-center justify-between text-[11px]">
                <span className="text-slate-400">Author Agent:</span>
                <span className="font-mono text-cyan-300">
                  {selectedNode.provenance?.agent_id || 'system'}
                </span>
              </div>
              <div className="flex items-center justify-between text-[11px]">
                <span className="text-slate-400">Confidence:</span>
                <span className="font-mono text-emerald-400">
                  {selectedNode.provenance?.confidence !== undefined
                    ? `${(selectedNode.provenance.confidence * 100).toFixed(0)}%`
                    : '100%'}
                </span>
              </div>
              {selectedNode.provenance?.source_session && (
                <div className="flex items-center justify-between text-[11px]">
                  <span className="text-slate-400">Source Session:</span>
                  <span className="font-mono text-slate-300 truncate max-w-[200px]">
                    {selectedNode.provenance.source_session}
                  </span>
                </div>
              )}
              {selectedNode.provenance?.content_hash && (
                <div className="flex items-center justify-between text-[11px]">
                  <span className="text-slate-400">Content Hash:</span>
                  <span className="font-mono text-slate-500 text-[10px] truncate max-w-[180px]">
                    {selectedNode.provenance.content_hash}
                  </span>
                </div>
              )}
              {selectedNode.provenance?.description && (
                <div className="pt-1 border-t border-slate-800 text-[11px] text-slate-300">
                  <p className="text-slate-400 text-[10px] font-medium">Description:</p>
                  <p className="mt-0.5 italic">{selectedNode.provenance.description}</p>
                </div>
              )}
              {selectedNode.provenance?.rule_expression && (
                <div className="pt-1 border-t border-slate-800 text-[11px]">
                  <p className="text-slate-400 text-[10px] font-medium">Rule Expression:</p>
                  <code className="text-indigo-300 font-mono text-[10px] block mt-0.5 bg-slate-950 p-1 rounded">
                    {selectedNode.provenance.rule_expression}
                  </code>
                </div>
              )}
            </div>
          </div>
        )}

        {/* Provenance for Selected Edge */}
        {selectedEdge && (
          <div
            data-testid="kg-edge-popover"
            className="absolute bottom-3 left-3 right-3 max-w-md bg-slate-900/95 border border-blue-500/40 rounded-xl p-3.5 shadow-2xl backdrop-blur-md z-20"
          >
            <div className="flex items-start justify-between gap-2 border-b border-slate-800 pb-2 mb-2">
              <div className="flex items-center gap-2">
                <GitFork className="w-4 h-4 text-blue-400 shrink-0" />
                <div>
                  <h4 className="text-xs font-bold text-slate-100 flex items-center gap-1.5">
                    {selectedEdge.source} ➔ {selectedEdge.predicate} ➔ {selectedEdge.target}
                  </h4>
                  <span className="text-[10px] text-slate-400 font-mono">
                    Tier: {selectedEdge.tier}
                  </span>
                </div>
              </div>

              <Button
                variant="ghost"
                size="icon"
                data-testid="close-edge-popover-btn"
                onClick={() => setSelectedEdge(null)}
              >
                <X className="w-3.5 h-3.5" />
              </Button>
            </div>

            <div className="space-y-1.5 text-xs">
              <div className="flex items-center justify-between text-[11px]">
                <span className="text-slate-400">Agent:</span>
                <span className="font-mono text-cyan-300">
                  {selectedEdge.provenance?.agent_id || 'system'}
                </span>
              </div>
              <div className="flex items-center justify-between text-[11px]">
                <span className="text-slate-400">Origin:</span>
                <span className="font-mono text-slate-200">
                  {selectedEdge.provenance?.origin || 'axiomatic'}
                </span>
              </div>
              <div className="flex items-center justify-between text-[11px]">
                <span className="text-slate-400">Confidence:</span>
                <span className="font-mono text-emerald-400">
                  {selectedEdge.provenance?.confidence !== undefined
                    ? `${(selectedEdge.provenance.confidence * 100).toFixed(0)}%`
                    : '100%'}
                </span>
              </div>
            </div>
          </div>
        )}
      </div>
    </div>
  );
};
