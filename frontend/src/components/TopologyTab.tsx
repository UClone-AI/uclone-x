import { useMemo, useState, type ReactNode } from 'react';
import { GitBranch, Network, RefreshCw, User, Wrench } from 'lucide-react';
import { Badge, type Tone } from './ui/Badge';
import { fmt, plural, useCopy, type Plural } from '../i18n';
import {
  roomDockUrls,
  useRoomRead,
  type RoomTopology,
  type RoomTopologyGapCode,
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

/** A status word from the catalog, or the Core's own word for a status this head does not know. */
const statusWord = (table: Record<string, string>, status: string): string =>
  Object.prototype.hasOwnProperty.call(table, status) ? table[status] : status;

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
/**
 * The Core's gap clauses in the reader's language (#1911): each clause's code worded by the
 * catalog, with the count it names. A clause keeps the Core's English when its code is one this
 * head does not know, and every clause does when the codes are missing (a Core older than
 * them) or do not line up one to one with the clauses.
 */
const wordGaps = (
  clauses: readonly string[],
  codes: readonly RoomTopologyGapCode[] | undefined,
  sentences: Readonly<Record<string, string | Plural>>,
): string[] => {
  if (!codes || codes.length !== clauses.length) return [...clauses];
  return clauses.map((clause, i) => {
    const { code, count } = codes[i];
    const worded = Object.prototype.hasOwnProperty.call(sentences, code) ? sentences[code] : undefined;
    if (typeof worded === 'string') return worded;
    if (worded && typeof count === 'number') return plural(worded, count);
    return clause;
  });
};

export function TopologyTab({ roomId, refreshKey }: TopologyTabProps) {
  const t = useCopy().dock.topology;
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

  // The Core's `reason` and gaps are worded from their codes (#1911); the Core's English is
  // shown for a code this head does not know, or from a Core older than the codes.
  const reasons: Readonly<Record<string, string>> = t.reasons;
  const reason: ReactNode = !roomId ? (
    t.noRoom
  ) : error ? (
    <>
      {t.readFailed}{' '}
      <code className="break-words font-mono text-[11px] text-slate-400">{error}</code>
    </>
  ) : !data ? (
    t.loading
  ) : data.reason_code && Object.prototype.hasOwnProperty.call(reasons, data.reason_code) ? (
    reasons[data.reason_code]
  ) : (
    data.reason
  );
  const historyGaps = data ? wordGaps(data.history_gaps ?? [], data.history_gap_codes, t.historyGapReasons) : [];
  const toolCallGaps = data
    ? wordGaps(data.tool_call_gaps ?? [], data.tool_call_gap_codes, t.toolCallGapReasons)
    : [];

  return (
    <div className="space-y-4">
      <div className="p-3 bg-slate-900/80 border border-slate-800 rounded-2xl flex flex-wrap items-center justify-between gap-3">
        <div className="flex items-center gap-2.5 min-w-0">
          <div className="p-2 bg-slate-800 rounded-xl border border-slate-700 shrink-0">
            <Network className="w-4 h-4 text-cyan-300" />
          </div>
          <div className="min-w-0">
            <h2 className="text-xs font-bold text-white">{t.title}</h2>
            <p className="text-[11px] text-slate-400">
              {data
                ? fmt(t.summary, {
                    seats: data.summary.seats,
                    turns: data.summary.turns,
                    toolCalls: data.summary.tool_calls,
                    helpers: data.summary.subagents,
                  })
                : t.subtitle}
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
            {t.refresh}
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
          {historyGaps.length > 0 ? (
            <div data-testid="topology-history-gaps">
              <span>{t.historyGaps}</span>
              <ul className="list-disc pl-4">
                {historyGaps.map((gap) => (
                  <li key={gap}>{gap}</li>
                ))}
              </ul>
            </div>
          ) : data.history_complete ? (
            <p data-testid="topology-history-complete">{t.historyComplete}</p>
          ) : null}
          {/* Apart from the turns: a turn that is shown can still be missing calls (#1388 N3). */}
          {toolCallGaps.length > 0 && (
            <div data-testid="topology-tool-call-gaps">
              <span>{t.toolCallGaps}</span>
              <ul className="list-disc pl-4">
                {toolCallGaps.map((gap) => (
                  <li key={gap}>{gap}</li>
                ))}
              </ul>
            </div>
          )}
          <p>{t.helperCallsNote}</p>
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
                    {seat.status === 'left' && <Badge tone="neutral">{t.left}</Badge>}
                    {seat.live && <Badge tone="info">{t.running}</Badge>}
                  </div>
                  <p className="text-[11px] text-slate-400">
                    {seat.turn_count === 0 ? t.noSeatTurns : plural(t.seatTurns, seat.turn_count)}
                  </p>
                  {helpers.map((h) => (
                    <p key={h.id} className="text-[11px] text-slate-400 flex items-center gap-1">
                      <GitBranch className="w-3 h-3" />
                      {fmt(t.startedHelper, { id: h.subagent_id })}
                    </p>
                  ))}
                </div>
              );
            })}
          </div>

          {/* Turns, in the order they were taken */}
          {graph.turns.length === 0 ? (
            <p className="text-xs text-slate-400 text-center py-6">{t.noTurns}</p>
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
                      <Badge tone={TURN_TONE[turn.status]}>{statusWord(t.turnStatus, turn.status)}</Badge>
                    </div>
                    {!turn.tools_recorded && (
                      <p className="text-[11px] text-slate-400">{t.toolsNotRecorded}</p>
                    )}
                    {tools.length === 0 ? (
                      turn.tools_recorded && (
                        <p className="text-[11px] text-slate-500">{t.noToolCalls}</p>
                      )
                    ) : (
                      <ul className="space-y-1">
                        {tools.map((tool) => (
                          <li key={tool.id} className="flex items-center gap-1.5 flex-wrap">
                            <Wrench className="w-3 h-3 text-slate-500" />
                            <span className="font-mono text-slate-200">{tool.tool_name}</span>
                            <Badge tone={toolTone(tool.status)}>{statusWord(t.toolStatus, tool.status)}</Badge>
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
