import React, { useState, useMemo } from 'react';
import {
  FileEdit,
  FileText,
  Terminal,
  Globe,
  Search,
  Cpu,
  Check,
  Copy,
  ChevronDown,
  ChevronRight,
  AlertTriangle,
  Clock,
  CheckCircle2,
  XCircle,
  AlertOctagon,
  Activity,
  Layers,
  GitBranch,
} from 'lucide-react';
import type { EventEnvelope } from '../../types';
import { Badge } from '../ui/Badge';
import {
  isRoomToolEvent,
  roomDockUrls,
  useRoomRead,
  type RoomToolUse,
  type SeatHistory,
} from '../../lib/roomDock';
import {
  classifyTool,
  parseArgs,
  argsRecord,
  type ActivityCategory,
} from '../../lib/toolLabels';

export type { ActivityCategory };

/**
 * The outcome this panel may claim for a call, in one place because every row asks.
 *
 * It used to read `status === 'error' ? 'error' : 'success'`, which turned every status
 * that was not an error into a green `Pass` -- including a call whose turn was stopped
 * before anything recorded how it ended (#1031). `running` is the same mistake one step
 * earlier (#1051): a call that has not returned has no outcome at all.
 *
 * Only a status that says the call succeeded reads as a pass. A status this panel does
 * not know (`timeout`, say) is a call that did not succeed, and reads as one.
 */
export const activityStatus = (status: string): ActivityItem['status'] => {
  const s = (status || '').toLowerCase();
  if (s === 'success' || s === 'ok') return 'success';
  if (s === 'stopped' || s === 'cancelled' || s === 'interrupted') return 'stopped';
  if (s === 'running') return 'running';
  return 'error';
};

export interface ActivityItem {
  id: string;
  category: ActivityCategory;
  categoryLabel: string;
  title: string;
  tool_name: string;
  /**
   * `stopped`: the turn was cancelled and nothing recorded how the call ended (#1031).
   * `running`: the call is still in flight and has not ended at all yet (#1051).
   */
  status: 'success' | 'error' | 'running' | 'stopped';
  duration_ms?: number;
  timestamp?: string;
  /** The arguments as parsed JSON, or the preview text when it is not whole JSON. */
  arguments?: unknown;
  output?: string;
  truncated?: boolean;
  error?: string | null;
  written_path?: string | null;
  wrote_unnamed?: boolean;
  subagent_id?: string | null;
  raw?: unknown;
}

/** What the timeline says when its read fails and the Core gave no plain reason of its own. */
export const activityReadFailedSentence = (name: string): string =>
  `${name}'s activity could not be read.`;

export interface ActivityTimelineProps {
  /** The conversation on screen; `null` when none is open. */
  roomId: string | null;
  /** The seat the dock describes; `null` when no clone is seated. */
  seatId: string | null;
  /** The seat's name, until its history answers with one. */
  seatName?: string;
  /** The live SSE envelopes; only `room.{roomId}.tool` ones for this seat are read. */
  events?: EventEnvelope[];
  /** Changes when the conversation moves on (a new message), to re-read the history. */
  refreshKey?: unknown;
  /** Front Docs on a file a call wrote. */
  onOpenInDocs?: (path: string) => void;
}

const itemFromUse = (use: RoomToolUse, index: number): ActivityItem => {
  const args = parseArgs(use.arguments_preview);
  const { category, label, summaryTitle } = classifyTool(use.tool_name, argsRecord(args));
  return {
    id: use.tool_call_id || `${use.turn_id}:${index}`,
    category,
    categoryLabel: label,
    title: summaryTitle,
    tool_name: use.tool_name,
    status: activityStatus(use.status),
    duration_ms: typeof use.duration_ms === 'number' ? use.duration_ms : undefined,
    timestamp: use.recorded_at,
    arguments: args,
    output: use.output_preview || undefined,
    truncated: use.truncated,
    error: use.error,
    written_path: use.written_path,
    wrote_unnamed: use.wrote_unnamed,
    subagent_id: use.subagent_id,
    raw: use,
  };
};

type Payload = Record<string, unknown>;
const str = (v: unknown): string | null => (typeof v === 'string' && v ? v : null);

/**
 * The live rows: `TOOL_CALL` opens one, its `TOOL_RESULT` closes it (§3.6).
 *
 * A call and its result pair on `tool_call_id`; a provider that gives none pairs them on
 * turn, name and order within the turn. A turn the history already recorded is skipped,
 * so a call is never listed twice once the re-read lands.
 */
function liveItems(
  events: EventEnvelope[],
  roomId: string,
  seatId: string,
  history: SeatHistory | null,
): ActivityItem[] {
  const recordedCalls = new Set<string>();
  const recordedTurns = new Set<string>();
  for (const turn of history?.turns ?? []) {
    if (turn.turn_id && turn.tools !== null) recordedTurns.add(turn.turn_id);
  }
  for (const use of history?.tool_uses ?? []) {
    if (use.tool_call_id) recordedCalls.add(use.tool_call_id);
  }

  const rows = new Map<string, { call?: Payload; result?: Payload; at?: string }>();
  const ordinals = new Map<string, number>();
  // `events` is newest first; pairing by order needs them oldest first.
  for (const ev of [...events].reverse()) {
    if (!isRoomToolEvent(ev, roomId)) continue;
    const p = (ev.payload ?? {}) as Payload;
    if (p.participant_id !== seatId) continue;
    const turnId = str(p.turn_id) ?? '';
    if (recordedTurns.has(turnId)) continue;
    const callId = str(p.tool_call_id);
    if (callId && recordedCalls.has(callId)) continue;
    const isResult = String(ev.event_type ?? ev.type).toUpperCase() === 'TOOL_RESULT';
    let key = callId;
    if (!key) {
      const base = `${isResult ? 'r' : 'c'}:${turnId}:${String(p.name ?? '')}`;
      const n = ordinals.get(base) ?? 0;
      ordinals.set(base, n + 1);
      key = `${turnId}:${String(p.name ?? '')}:${n}`;
    }
    const row = rows.get(key) ?? {};
    if (isResult) row.result = p;
    else row.call = p;
    row.at = row.at ?? (typeof ev.timestamp === 'string' ? ev.timestamp : undefined);
    rows.set(key, row);
  }

  const items: ActivityItem[] = [];
  for (const [key, { call, result, at }] of rows) {
    const name = str(call?.name) ?? str(result?.name) ?? 'tool';
    const args = parseArgs(str(call?.arguments_preview));
    const { category, label, summaryTitle } = classifyTool(name, argsRecord(args));
    items.push({
      id: key,
      category,
      categoryLabel: label,
      title: summaryTitle,
      tool_name: name,
      status: result ? activityStatus(String(result.status ?? '')) : 'running',
      duration_ms: typeof result?.duration_ms === 'number' ? result.duration_ms : undefined,
      timestamp: at,
      arguments: args,
      output: str(result?.output_preview) ?? undefined,
      truncated: Boolean(result?.truncated),
      error: str(result?.error),
      written_path: str(result?.written_path),
      subagent_id: str(result?.subagent_id),
      raw: { call, result },
    });
  }
  return items.reverse();
}

export const ActivityTimeline: React.FC<ActivityTimelineProps> = ({
  roomId,
  seatId,
  seatName,
  events = [],
  refreshKey,
  onOpenInDocs,
}) => {
  const [searchTerm, setSearchTerm] = useState<string>('');
  const [categoryFilter, setCategoryFilter] = useState<string>('all');
  const [statusFilter, setStatusFilter] = useState<string>('all');
  const [expandedIds, setExpandedIds] = useState<Record<string, boolean>>({});
  const [copiedId, setCopiedId] = useState<string | null>(null);

  // A finished call is saved with its turn, before the event announcing it; re-reading on
  // each result keeps the recorded list current without waiting for the reply.
  const resultsSeen = useMemo(
    () =>
      roomId && seatId
        ? events.filter(
            (ev) =>
              isRoomToolEvent(ev, roomId) &&
              String(ev.event_type ?? ev.type).toUpperCase() === 'TOOL_RESULT' &&
              (ev.payload as Payload | undefined)?.participant_id === seatId,
          ).length
        : 0,
    [events, roomId, seatId],
  );
  const url = roomId && seatId ? roomDockUrls.seatHistory(roomId, seatId) : null;
  const read = useRoomRead<SeatHistory>(url, `${String(refreshKey ?? '')}:${resultsSeen}`);
  const history = read.data;

  const name = history?.display_name || seatName || seatId || 'This clone';

  const activities: ActivityItem[] = useMemo(() => {
    if (!roomId || !seatId) return [];
    const recorded = (history?.tool_uses ?? []).map(itemFromUse).reverse();
    return [...liveItems(events, roomId, seatId, history), ...recorded];
  }, [events, history, roomId, seatId]);

  const unrecorded = (history?.turns ?? []).filter((t) => t.tools === null);
  const recordedTurns = (history?.turns ?? []).length - unrecorded.length;

  // Why the list is empty, in words (P6): each cause is a different sentence. The list is
  // what the seat's saved turns reported, so an empty one is "no tool calls are listed",
  // never "it used no tools" (#1366); the Core's `reason` and `tools_note` say the rest.
  const emptyCause = (): string => {
    if (!roomId) return 'No conversation is open. Open one from the rail to see what its clones did.';
    if (!seatId) return 'No clone is seated in this conversation, so there is no clone to show activity for.';
    // The Core's own plain words where it gave them, else ours; never transport text (#1435).
    if (read.fault) return read.fault.detail ?? activityReadFailedSentence(name);
    if (!history) return `Reading ${name}'s activity…`;
    if (history.reason) return history.reason;
    if (history.turns.length === 0) return `No turns by ${name} are listed in this conversation.`;
    if (recordedTurns === 0) return `None of ${name}'s turns here recorded which tools they used.`;
    return `No tool calls are listed for ${name}'s ${recordedTurns} recorded ${recordedTurns === 1 ? 'turn' : 'turns'} here.`;
  };

  const filteredActivities = useMemo(() => {
    return activities.filter((act) => {
      if (categoryFilter !== 'all' && act.category !== categoryFilter) return false;
      if (statusFilter !== 'all' && act.status !== statusFilter) return false;
      if (searchTerm.trim()) {
        const q = searchTerm.toLowerCase();
        const matchesTitle = act.title.toLowerCase().includes(q);
        const matchesName = act.tool_name.toLowerCase().includes(q);
        const matchesArgs = JSON.stringify(act.arguments ?? '').toLowerCase().includes(q);
        const matchesOutput = (act.output ?? '').toLowerCase().includes(q);
        if (!matchesTitle && !matchesName && !matchesArgs && !matchesOutput) return false;
      }
      return true;
    });
  }, [activities, categoryFilter, statusFilter, searchTerm]);

  const toggleExpand = (id: string) => {
    setExpandedIds((prev) => ({ ...prev, [id]: !prev[id] }));
  };

  const handleCopyJson = (id: string, data: unknown) => {
    navigator.clipboard.writeText(JSON.stringify(data, null, 2));
    setCopiedId(id);
    setTimeout(() => setCopiedId(null), 2000);
  };

  const getCategoryIcon = (category: ActivityCategory) => {
    switch (category) {
      case 'mutation':
        return <FileEdit className="w-4 h-4 text-amber-400" />;
      case 'command':
        return <Terminal className="w-4 h-4 text-cyan-400" />;
      case 'web':
        return <Globe className="w-4 h-4 text-blue-400" />;
      case 'inspection':
        return <Search className="w-4 h-4 text-indigo-400" />;
      default:
        return <Cpu className="w-4 h-4 text-violet-400" />;
    }
  };

  const getCategoryBadgeClass = (category: ActivityCategory) => {
    switch (category) {
      case 'mutation':
        return 'bg-amber-950/80 text-amber-300 border-amber-800/60';
      case 'command':
        return 'bg-cyan-950/80 text-cyan-300 border-cyan-800/60';
      case 'web':
        return 'bg-blue-950/80 text-blue-300 border-blue-800/60';
      case 'inspection':
        return 'bg-indigo-950/80 text-indigo-300 border-indigo-800/60';
      default:
        return 'bg-violet-950/80 text-violet-300 border-violet-800/60';
    }
  };

  const argumentsText = (args: unknown): string =>
    typeof args === 'string' ? args : JSON.stringify(args, null, 2);

  return (
    <div data-testid="activity-timeline" className="flex flex-col h-full space-y-4">
      {/* Header */}
      <div className="p-3 bg-slate-900/80 border border-slate-800 rounded-2xl flex items-center justify-between">
        <div className="flex items-center gap-2.5 min-w-0">
          <div className="p-2 bg-slate-800 rounded-xl border border-slate-700 shrink-0">
            <Activity className="w-4 h-4 text-amber-300" />
          </div>
          <div className="min-w-0">
            <h2 className="text-xs font-bold text-white flex items-center gap-2 flex-wrap">
              Tool calls by {seatId ? name : 'the clone'}
              <Badge tone="neutral" className="text-[10px] font-mono">
                {filteredActivities.length} listed
              </Badge>
            </h2>
            <p className="text-[11px] text-slate-400">
              The tool calls recorded in this conversation, with what went in, what came out, and how long each took
            </p>
          </div>
        </div>
      </div>

      {/* Filter and Search Bar */}
      <div className="flex flex-wrap items-center gap-2 text-xs">
        <div className="relative flex-1 min-w-[160px]">
          <Search className="w-3.5 h-3.5 absolute left-2.5 top-1/2 -translate-y-1/2 text-slate-500" />
          <input
            type="text"
            placeholder="Search actions, files, args..."
            value={searchTerm}
            onChange={(e) => setSearchTerm(e.target.value)}
            className="w-full bg-slate-900 border border-slate-800 text-slate-200 text-xs rounded-xl pl-8 pr-3 py-1.5 focus:outline-none focus:border-amber-500/60 transition-colors"
          />
        </div>

        <div className="flex flex-wrap items-center gap-1 bg-slate-900/80 p-1 rounded-xl border border-slate-800 text-[11px]">
          {(
            [
              { id: 'all', label: 'All' },
              { id: 'mutation', label: 'Mutations' },
              { id: 'command', label: 'Commands' },
              { id: 'web', label: 'Web' },
              { id: 'inspection', label: 'Inspection' },
            ] as const
          ).map((tab) => (
            <button
              key={tab.id}
              type="button"
              onClick={() => setCategoryFilter(tab.id)}
              className={`px-2 py-0.5 rounded-lg font-medium transition-colors ${
                categoryFilter === tab.id
                  ? 'bg-amber-500/20 text-amber-300 font-semibold'
                  : 'text-slate-400 hover:text-slate-200'
              }`}
            >
              {tab.label}
            </button>
          ))}
        </div>

        <div className="flex flex-wrap items-center gap-1 bg-slate-900/80 p-1 rounded-xl border border-slate-800 text-[11px]">
          {(
            [
              { id: 'all', label: 'All Status' },
              { id: 'success', label: 'Passed' },
              { id: 'error', label: 'Errors' },
            ] as const
          ).map((st) => (
            <button
              key={st.id}
              type="button"
              onClick={() => setStatusFilter(st.id)}
              className={`px-2 py-0.5 rounded-lg font-medium transition-colors ${
                statusFilter === st.id
                  ? 'bg-slate-700 text-white font-semibold'
                  : 'text-slate-400 hover:text-slate-200'
              }`}
            >
              {st.label}
            </button>
          ))}
        </div>
      </div>

      {/* What the list covers, and turns it may be missing, in the Core's words (#1366). */}
      {history && (history.tools_note || history.unsaved_note) && (
        <div
          data-testid="activity-scope"
          className="px-1 text-[11px] text-slate-400 space-y-1"
        >
          {history.tools_note && <p data-testid="activity-tools-note">{history.tools_note}</p>}
          {history.unsaved_note && (
            <p data-testid="activity-unsaved-note">{history.unsaved_note}</p>
          )}
        </div>
      )}

      {/* Turns whose tools were not recorded: not the same as "used none" (P6). */}
      {unrecorded.length > 0 && (
        <div
          data-testid="activity-not-recorded"
          className="p-2.5 rounded-xl bg-slate-900/70 border border-slate-800 text-[11px] text-slate-300 space-y-1"
        >
          <p>
            {unrecorded.length} {unrecorded.length === 1 ? 'turn' : 'turns'} did not record which
            tools {unrecorded.length === 1 ? 'it' : 'they'} used in full, so {unrecorded.length === 1 ? 'its' : 'their'} calls
            may not all be listed here.
          </p>
          {Array.from(new Set(unrecorded.map((t) => t.tools_not_recorded_reason).filter(Boolean))).map(
            (why) => (
              <p key={why} className="text-slate-400">
                {why}
              </p>
            ),
          )}
        </div>
      )}

      {/* Action Cards List */}
      <div className="flex-1 overflow-y-auto min-h-0 space-y-2.5 pr-1">
        {filteredActivities.length === 0 ? (
          <div
            data-testid="activity-empty-state"
            className="text-center py-16 text-slate-500 font-sans space-y-2"
          >
            <Layers className="w-8 h-8 mx-auto text-slate-600" />
            <p className="text-xs text-slate-400 font-medium max-w-sm mx-auto">
              {activities.length > 0
                ? `Nothing matches these filters; ${activities.length} ${activities.length === 1 ? 'call is' : 'calls are'} hidden by them.`
                : emptyCause()}
            </p>
          </div>
        ) : (
          filteredActivities.map((act) => {
            const isExpanded = Boolean(expandedIds[act.id]);
            return (
              <div
                key={act.id}
                data-testid={`activity-card-${act.id}`}
                className="border rounded-2xl transition-colors duration-150 border-slate-800/80 bg-slate-900/70 hover:border-slate-700"
              >
                <div
                  role="button"
                  tabIndex={0}
                  aria-expanded={isExpanded}
                  onClick={() => toggleExpand(act.id)}
                  onKeyDown={(e) => {
                    if (e.key === 'Enter' || e.key === ' ') {
                      e.preventDefault();
                      toggleExpand(act.id);
                    }
                  }}
                  className="p-3 flex items-center justify-between gap-2.5 cursor-pointer select-none"
                >
                  <div className="flex items-center gap-2.5 min-w-0">
                    <div className="p-1.5 bg-slate-950 rounded-xl border border-slate-800 shrink-0">
                      {getCategoryIcon(act.category)}
                    </div>
                    <div className="min-w-0">
                      <div className="flex items-center gap-2 flex-wrap">
                        <span
                          className={`text-[10px] px-1.5 py-0.5 rounded-md font-bold uppercase border tracking-wider shrink-0 ${getCategoryBadgeClass(
                            act.category,
                          )}`}
                        >
                          {act.categoryLabel}
                        </span>
                        <span className="text-xs font-semibold text-white truncate font-mono">
                          {act.title}
                        </span>
                      </div>
                      <div className="flex items-center gap-3 text-[10px] text-slate-400 font-mono mt-0.5 flex-wrap">
                        <span className="text-slate-500">{act.tool_name}</span>
                        {act.duration_ms !== undefined && (
                          <span className="flex items-center gap-1 text-slate-400">
                            <Clock className="w-3 h-3 text-slate-500" />
                            {act.duration_ms.toFixed(0)} ms
                          </span>
                        )}
                        {act.timestamp && <span className="text-slate-500">{act.timestamp}</span>}
                      </div>
                    </div>
                  </div>

                  <div className="flex items-center gap-2 shrink-0">
                    {act.status === 'success' ? (
                      <Badge tone="success" className="text-[10px] font-bold">
                        <CheckCircle2 className="w-3 h-3" />
                        <span>Pass</span>
                      </Badge>
                    ) : act.status === 'running' ? (
                      // In flight: no outcome yet, neither a pass nor an error (#1051).
                      <Badge tone="warning" className="text-[10px] font-bold">
                        <Clock className="w-3 h-3" />
                        <span>Running</span>
                      </Badge>
                    ) : act.status === 'stopped' ? (
                      // Same word and icon as the transcript's own mark (#1031).
                      <Badge tone="neutral" className="text-[10px] font-bold">
                        <AlertOctagon className="w-3 h-3" />
                        <span>Stopped</span>
                      </Badge>
                    ) : (
                      <Badge tone="danger" className="text-[10px] font-bold">
                        <XCircle className="w-3 h-3" />
                        <span>Error</span>
                      </Badge>
                    )}
                    <div className="p-1 text-slate-500 hover:text-slate-300">
                      {isExpanded ? <ChevronDown className="w-4 h-4" /> : <ChevronRight className="w-4 h-4" />}
                    </div>
                  </div>
                </div>

                {/* What the call left behind: a file (open it in Docs), or a helper agent. */}
                {(act.written_path || act.wrote_unnamed || act.subagent_id) && (
                  <div className="px-3 pb-2.5 -mt-1 flex flex-wrap items-center gap-2 text-[11px] text-slate-400">
                    {act.written_path && (
                      <>
                        <span className="font-mono truncate max-w-full">Wrote {act.written_path}</span>
                        {onOpenInDocs && (
                          <button
                            type="button"
                            data-testid={`activity-open-doc-${act.id}`}
                            onClick={() => onOpenInDocs(act.written_path as string)}
                            className="inline-flex items-center gap-1 px-2 py-0.5 rounded-lg border border-slate-700 text-slate-200 hover:border-slate-500"
                          >
                            <FileText className="w-3 h-3" />
                            Open in Docs
                          </button>
                        )}
                      </>
                    )}
                    {!act.written_path && act.wrote_unnamed && (
                      <span>May have written to a file without naming it, so nothing can be opened from here.</span>
                    )}
                    {act.subagent_id && (
                      <span className="inline-flex items-center gap-1">
                        <GitBranch className="w-3 h-3" />
                        Started a helper agent ({act.subagent_id})
                      </span>
                    )}
                  </div>
                )}

                {isExpanded && (
                  <div className="px-3.5 pb-3.5 pt-1 border-t border-slate-800/60 space-y-3 font-mono text-xs">
                    {/* Execution time, stated even when absent (P6). */}
                    <div className="flex items-center justify-between gap-2">
                      <div className="space-y-0.5">
                        <div className="text-[10px] text-slate-500 uppercase font-sans">Execution Time</div>
                        <div data-testid={`activity-duration-${act.id}`} className="text-slate-300">
                          {act.duration_ms == null
                            ? act.status === 'running'
                              ? 'still running'
                              : 'not timed'
                            : `${act.duration_ms.toFixed(1)} ms`}
                        </div>
                      </div>
                      {act.raw !== undefined && (
                        <button
                          type="button"
                          data-testid={`activity-copy-trace-${act.id}`}
                          onClick={() => handleCopyJson(`${act.id}:trace`, act.raw)}
                          className="text-[10px] text-slate-400 hover:text-white flex items-center gap-1 font-mono"
                          title="Copy the whole record of this call as JSON"
                        >
                          {copiedId === `${act.id}:trace` ? (
                            <Check className="w-3 h-3 text-emerald-400" />
                          ) : (
                            <Copy className="w-3 h-3" />
                          )}
                          <span>Copy trace</span>
                        </button>
                      )}
                    </div>

                    {act.arguments !== undefined && (
                      <div className="space-y-1">
                        <div className="flex items-center justify-between text-[10px] text-slate-500 uppercase font-sans">
                          <span>Input Parameters</span>
                          <button
                            type="button"
                            onClick={() => handleCopyJson(act.id, act.arguments)}
                            className="text-slate-400 hover:text-white flex items-center gap-1 normal-case font-mono"
                          >
                            {copiedId === act.id ? (
                              <Check className="w-3 h-3 text-emerald-400" />
                            ) : (
                              <Copy className="w-3 h-3" />
                            )}
                            <span>Copy JSON</span>
                          </button>
                        </div>
                        <pre className="p-2.5 rounded-xl bg-slate-950 text-slate-300 text-[11px] overflow-x-auto border border-slate-800/60 max-h-48 whitespace-pre-wrap">
                          {argumentsText(act.arguments)}
                        </pre>
                      </div>
                    )}

                    {act.output !== undefined && (
                      <div className="space-y-1">
                        <div className="text-[10px] text-slate-500 uppercase font-sans">Output Payload</div>
                        <pre className="p-2.5 rounded-xl bg-slate-950 text-slate-300 text-[11px] overflow-x-auto border border-slate-800/60 max-h-56 whitespace-pre-wrap">
                          {act.output}
                        </pre>
                        {act.truncated && (
                          <p className="text-[10px] text-slate-500 font-sans">
                            This output was cut to fit; the clone received all of it.
                          </p>
                        )}
                      </div>
                    )}

                    {act.error && (
                      <div className="p-2.5 rounded-xl bg-rose-950/30 border border-rose-900/50 space-y-1">
                        <div className="flex items-center gap-1.5 text-rose-400 font-semibold text-[11px]">
                          <AlertTriangle className="w-3.5 h-3.5" />
                          <span>Execution Error Trace</span>
                        </div>
                        <pre className="text-rose-300 text-[10px] whitespace-pre-wrap overflow-x-auto">
                          {act.error}
                        </pre>
                      </div>
                    )}
                  </div>
                )}
              </div>
            );
          })
        )}
      </div>
    </div>
  );
};
