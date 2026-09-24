import { useMemo, useState } from 'react';
import { GitBranch, Network, RefreshCw, User, Wrench } from 'lucide-react';
import { Badge, type Tone } from './ui/Badge';
import {
  roomDockUrls,
  useRoomRead,
  type RoomTopology,
  type RoomTopologyNode,
} from '../lib/roomDock';

interface TopologyTabProps {
  /** The conversation on screen; the graph is read for it alone (#1355). */
  roomId: string | null;
  /** Changes when the conversation moves on, to read again. */
  refreshKey?: unknown;
}

type SeatNode = Extract<RoomTopologyNode, { kind: 'seat' }>;
type TurnNode = Extract<RoomTopologyNode, { kind: 'turn' }>;
type ToolNode = Extract<RoomTopologyNode, { kind: 'tool' }>;
type SubagentNode = Extract<RoomTopologyNode, { kind: 'subagent' }>;

const TURN_TONE: Record<TurnNode['status'], Tone> = {
  answered: 'success',
  failed: 'danger',
  interrupted: 'neutral',
};

const toolTone = (status: string): Tone =>
  status === 'success' ? 'success' : status === 'running' ? 'warning' : 'danger';

/**
 * The conversation drawn from its edges: who is seated, the turns in the order they were
 * taken, the tools each turn called, and the helper agents a seat started.
 *
 * The order and the nesting come from the Core's edges (`followed_by`, `called`,
 * `spawned`), not from re-sorting the nodes here, so what is drawn is what the Core said.
 * A developer surface (the DAG tab); the everyday view of the same turns is Activity.
 */
export function TopologyTab({ roomId, refreshKey }: TopologyTabProps) {
  const [reloads, setReloads] = useState(0);
  const { data, error, loading } = useRoomRead<RoomTopology>(
    roomId ? roomDockUrls.topology(roomId) : null,
    `${String(refreshKey ?? '')}:${reloads}`,
  );

  const graph = useMemo(() => {
    const nodes = new Map<string, RoomTopologyNode>();
    for (const n of data?.nodes ?? []) nodes.set(n.id, n);
    const out = (source: string, kind: string) =>
      (data?.edges ?? [])
        .filter((e) => e.source === source && e.kind === kind)
        .map((e) => nodes.get(e.target))
        .filter((n): n is RoomTopologyNode => n !== undefined);

    const seats = (data?.nodes ?? []).filter((n): n is SeatNode => n.kind === 'seat');
    const turnNodes = (data?.nodes ?? []).filter((n): n is TurnNode => n.kind === 'turn');
    // The first turn is the one nothing follows from; walk `followed_by` from there.
    const followed = new Set(
      (data?.edges ?? []).filter((e) => e.kind === 'followed_by').map((e) => e.target),
    );
    const ordered: TurnNode[] = [];
    const seen = new Set<string>();
    let cursor: RoomTopologyNode | undefined = turnNodes.find((t) => !followed.has(t.id));
    while (cursor && cursor.kind === 'turn' && !seen.has(cursor.id)) {
      ordered.push(cursor);
      seen.add(cursor.id);
      cursor = out(cursor.id, 'followed_by')[0];
    }
    // A turn the chain did not reach is still a turn; it is listed, not dropped.
    for (const t of turnNodes) if (!seen.has(t.id)) ordered.push(t);

    const seatName = new Map(seats.map((s) => [s.participant_id, s.display_name]));
    return {
      seats,
      turns: ordered,
      toolsOf: (turnId: string) => out(turnId, 'called') as ToolNode[],
      helpersOf: (seatId: string) => out(seatId, 'spawned') as SubagentNode[],
      seatName: (pid: string) => seatName.get(pid) ?? pid,
    };
  }, [data]);

  const reason = !roomId
    ? 'No conversation is open. Open one from the rail to see how it ran.'
    : error
      ? `Couldn't read this conversation's graph: ${error}`
      : !data
        ? 'Reading this conversation…'
        : data.reason;

  return (
    <div className="space-y-4">
      <div className="p-3 bg-slate-900/80 border border-slate-800 rounded-2xl flex flex-wrap items-center justify-between gap-3">
        <div className="flex items-center gap-2.5 min-w-0">
          <div className="p-2 bg-slate-800 rounded-xl border border-slate-700 shrink-0">
            <Network className="w-4 h-4 text-cyan-300" />
          </div>
          <div className="min-w-0">
            <h2 className="text-xs font-bold text-white">How this conversation ran</h2>
            <p className="text-[11px] text-slate-400">
              {data
                ? `${data.summary.seats} seated · ${data.summary.turns} turns · ${data.summary.tool_calls} tool calls listed · ${data.summary.subagents} helpers`
                : 'Seats, their turns in order, the tools each turn called, and the helpers they started'}
            </p>
          </div>
        </div>
        <div className="flex flex-wrap items-center gap-3">
          <button
            type="button"
            onClick={() => setReloads((n) => n + 1)}
            disabled={!roomId || loading}
            className="px-3 py-1.5 rounded-xl bg-slate-800 hover:bg-slate-700 text-xs font-semibold text-slate-200 transition-colors border border-slate-700 inline-flex items-center gap-1.5 disabled:opacity-50"
          >
            <RefreshCw className="w-3.5 h-3.5" />
            Refresh
          </button>
        </div>
      </div>

      {/* What the graph may be missing, in the Core's words (#1366). `history_complete`
          speaks for turn rows only: a turn's tool calls can still be unrecorded or made by a
          helper, so it is never shown as "every tool call" (#1388 N3). */}
      {!reason && data && (
        <div
          data-testid="topology-history"
          className="p-2.5 rounded-xl bg-slate-900/70 border border-slate-800 text-[11px] text-slate-400 space-y-1"
        >
          {(data.history_gaps ?? []).length > 0 ? (
            <div data-testid="topology-history-gaps">
              <span>Turns may be missing from this graph because:</span>
              <ul className="list-disc pl-4">
                {data.history_gaps.map((gap) => (
                  <li key={gap}>{gap}</li>
                ))}
              </ul>
            </div>
          ) : data.history_complete ? (
            <p data-testid="topology-history-complete">No turn is known to be missing from this graph.</p>
          ) : null}
          {/* Apart from the turns: a turn that is shown can still be missing calls (#1388 N3). */}
          {(data.tool_call_gaps ?? []).length > 0 && (
            <div data-testid="topology-tool-call-gaps">
              <span>Tool calls may be missing from the turns shown because:</span>
              <ul className="list-disc pl-4">
                {data.tool_call_gaps.map((gap) => (
                  <li key={gap}>{gap}</li>
                ))}
              </ul>
            </div>
          )}
          <p>
            The tool calls shown are the ones each turn reported; a helper&apos;s own calls are
            not shown.
          </p>
        </div>
      )}

      {reason ? (
        <p data-testid="topology-reason" className="text-xs text-slate-300 text-center py-12">
          {reason}
        </p>
      ) : (
        <>
          {/* Seats */}
          <div className="grid grid-cols-2 gap-2">
            {graph.seats.map((seat) => {
              const helpers = graph.helpersOf(seat.id);
              return (
                <div
                  key={seat.id}
                  data-testid={`topology-seat-${seat.participant_id}`}
                  className="p-2.5 rounded-xl bg-slate-900/70 border border-slate-800 text-xs space-y-1"
                >
                  <div className="flex items-center gap-1.5 flex-wrap">
                    <User className="w-3.5 h-3.5 text-slate-400" />
                    <span className="font-semibold text-slate-100">{seat.display_name}</span>
                    {seat.status === 'left' && <Badge tone="neutral">left</Badge>}
                    {seat.live && <Badge tone="info">running</Badge>}
                  </div>
                  <p className="text-[11px] text-slate-400">
                    {seat.turn_count === 0
                      ? 'No turns listed.'
                      : `${seat.turn_count} ${seat.turn_count === 1 ? 'turn' : 'turns'}`}
                  </p>
                  {helpers.map((h) => (
                    <p key={h.id} className="text-[11px] text-slate-400 flex items-center gap-1">
                      <GitBranch className="w-3 h-3" />
                      Started helper {h.subagent_id}
                    </p>
                  ))}
                </div>
              );
            })}
          </div>

          {/* Turns, in the order they were taken */}
          {graph.turns.length === 0 ? (
            <p className="text-xs text-slate-400 text-center py-6">
              No turns are listed for this conversation.
            </p>
          ) : (
            <ol className="space-y-2 border-l border-slate-800 ml-2 pl-3">
              {graph.turns.map((turn) => {
                const tools = graph.toolsOf(turn.id);
                return (
                  <li
                    key={turn.id}
                    data-testid={`topology-turn-${turn.id}`}
                    className="p-2.5 rounded-xl bg-slate-900/60 border border-slate-800 text-xs space-y-1.5"
                  >
                    <div className="flex items-center gap-2 flex-wrap">
                      <span className="font-mono text-[10px] text-slate-500">#{turn.seq}</span>
                      <span className="font-semibold text-slate-100">
                        {graph.seatName(turn.participant_id)}
                      </span>
                      <Badge tone={TURN_TONE[turn.status]}>{turn.status}</Badge>
                    </div>
                    {!turn.tools_recorded && (
                      <p className="text-[11px] text-slate-400">
                        This turn did not record which tools it used in full.
                      </p>
                    )}
                    {tools.length === 0 ? (
                      turn.tools_recorded && (
                        <p className="text-[11px] text-slate-500">No tool calls listed.</p>
                      )
                    ) : (
                      <ul className="space-y-1">
                        {tools.map((tool) => (
                          <li key={tool.id} className="flex items-center gap-1.5 flex-wrap">
                            <Wrench className="w-3 h-3 text-slate-500" />
                            <span className="font-mono text-slate-200">{tool.tool_name}</span>
                            <Badge tone={toolTone(tool.status)}>{tool.status}</Badge>
                          </li>
                        ))}
                      </ul>
                    )}
                  </li>
                );
              })}
            </ol>
          )}
        </>
      )}
    </div>
  );
}
