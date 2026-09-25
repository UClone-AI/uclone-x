import React, { useState, useEffect } from 'react';
import {
  Gauge,
  Coins,
  RefreshCw,
  TrendingDown,
  Server,
  ShieldCheck,
  Cpu,
  Layers,
  PieChart,
} from 'lucide-react';
import { BudgetData, RoomContext, RoomState, RuntimeSettings } from '../../types';
import { participantLabel } from '../../lib/rooms';
import { Badge } from '../ui/Badge';
import { Button } from '../ui/Button';
import { StatusDot } from '../ui/StatusDot';

/**
 * The words this surface says the open conversation's readings in.
 *
 * Stated once, as constants, because #1059 moved them here from the rail's `RailCopy` and the
 * same sentence must not end up spelled two ways in two files again. #1272 replaced what they
 * describe: the readings are the room's own, and two of them are sentences rather than
 * figures because the quantity they used to state does not exist for a room.
 */
export const RESOURCE_ROOM_COPY = {
  heading: 'This conversation',
  replyBudget: 'Agent replies since your last message',
  replyBudgetNote:
    'The room’s own ceiling on a reply cascade (RoomPolicy.max_agent_turns_per_human_message). ' +
    'It resets whenever you speak.',
  /**
   * Said in words because the figure does not exist, rather than filled from elsewhere (P6).
   *
   * `AgentConfig.max_steps` bounds the model invocations inside one agent run. A room is
   * served by one such run per seat per turn and reports none of their step counts on any
   * route the head can call, so there is no room-level step reading to show.
   */
  noStepCount:
    'No step count: a room reports no per-run agent steps, so this surface states none.',
  /**
   * The other honest absence. `RoomMessage.usage` exists and nothing populates it on the
   * orchestrated path, so a room's turns carry no token count to total. The Core declines to
   * invent the figure; so does this surface.
   */
  noTokenTotal:
    'No token total: a room’s turns carry no token count, so there is nothing to add up. ' +
    'The Token Quota below is the session ledger, which is a different quantity.',
  seatsHeading: 'Turns in context, per seat',
  /**
   * Why there is no single turn number here, rather than a number nobody can source.
   *
   * **No staleness clause any more (#1286).** This sentence used to be followed by a
   * `room-seat-turns-sampled` note saying the figures were "counted up to 4 turns ago",
   * because `contextReadIsDue` sampled the read every four landed turns. #1286 removed
   * that cadence and the counts are now re-read on every landed turn, so the note was no
   * longer describing anything: a surface that exists to refuse substituted values cannot
   * carry a qualifier that is itself false.
   */
  seatsNote:
    'Each seat keeps its own context, so a room has no single turn count. These are the ' +
    'Core’s own per-seat figures from GET /api/rooms/{id}/context, re-read after every ' +
    'landed turn.',
  seatsUnread: 'Not read yet — this conversation’s context has not been asked about.',
  noRoom: 'No conversation is open, so there is nothing here to measure.',
  saturated: 'Saturated',
  saturatedTitle:
    'This seat is long enough that its oldest turns may fall out of the model’s context.',
  fromRecord: 'from saved record',
  fromRecordTitle:
    'Counted from this seat’s saved record, which nothing has spoken in since this server ' +
    'started, so it is behind any turn an earlier run did not write.',
} as const;

export interface ResourceSummaryProps {
  budgetData: BudgetData | null;
  /**
   * The conversation on screen, as `GET /api/rooms/{id}` answers it (#1272).
   *
   * This surface reads the room and never `chatMessages`. The latter is the retired
   * single-agent store (`GET /api/session/history`), which since #1208 describes no
   * conversation the user can see: it followed the rail's clone selection while the centre
   * column showed a room, so every figure derived from it named a different transcript.
   */
  room?: RoomState | null;
  /** Per-seat context fill, as `GET /api/rooms/{id}/context` answers it. */
  roomContext?: RoomContext | null;
  onRefresh: () => void;
  isRefreshing: boolean;
}

export const ResourceSummary: React.FC<ResourceSummaryProps> = ({
  budgetData,
  room = null,
  roomContext = null,
  onRefresh,
  isRefreshing,
}) => {
  const [settings, setSettings] = useState<RuntimeSettings | null>(null);

  useEffect(() => {
    let isMounted = true;
    fetch('/api/settings')
      .then((res) => (res.ok ? res.json() : null))
      .then((data: RuntimeSettings | null) => {
        if (isMounted && data) {
          setSettings(data);
        }
      })
      .catch(() => {
        /* Ignore network errors in test environments */
      });
    return () => {
      isMounted = false;
    };
  }, []);

  const sessionBudget = budgetData?.session_budget || {
    max_tokens: 1_000_000,
    used_input_tokens: 0,
    used_output_tokens: 0,
    total_used_tokens: 0,
    remaining_tokens: 1_000_000,
    budget_used_pct: 0.0,
  };

  const providers = budgetData?.providers || {};
  const compactionHistory = budgetData?.compaction_history || [];
  const totalTokensSaved = compactionHistory.reduce((acc, c) => acc + (c.saved_tokens || 0), 0);

  // The room's own reply-cascade budget. Both halves come off `GET /api/rooms/{id}`:
  // `turn_state.agent_turns_since_human` is what has been spent, and
  // `policy.max_agent_turns_per_human_message` is the ceiling the Core enforces.
  const repliesSpent = room?.turn_state.agent_turns_since_human ?? 0;
  const repliesCeiling = room?.policy.max_agent_turns_per_human_message ?? 0;
  const replyPercentage =
    repliesCeiling > 0 ? Math.min(100, Math.round((repliesSpent / repliesCeiling) * 100)) : 0;

  let stepBarColor = 'bg-emerald-500';
  let stepTextColor = 'text-emerald-400';
  if (replyPercentage > 85) {
    stepBarColor = 'bg-rose-500';
    stepTextColor = 'text-rose-400';
  } else if (replyPercentage > 65) {
    stepBarColor = 'bg-amber-500';
    stepTextColor = 'text-amber-400';
  }

  /** A seat's name as the roster knows it, else its raw id — never a guess. */
  const seatLabel = (participantId: string): string => {
    const participant = room?.participants.find((p) => p.id === participantId);
    return participant ? participantLabel(participant) : participantId;
  };

  // Token percentage
  const tokenPercentage = Math.min(
    100,
    sessionBudget.max_tokens > 0
      ? Math.round((sessionBudget.total_used_tokens / sessionBudget.max_tokens) * 100)
      : 0
  );

  const activeProvider = settings?.llm_provider || 'ollama';
  const activeModel = settings?.llm_model || 'qwen2.5:latest';
  const activeEndpoint = settings?.llm_base_url || 'http://localhost:11434';

  return (
    <div data-testid="resource-summary" className="flex flex-col h-full space-y-4">
      {/* Header Banner */}
      <div className="p-3 bg-slate-900/80 border border-slate-800 rounded-2xl shadow-lg flex items-center justify-between">
        <div className="flex items-center gap-2.5">
          <div className="p-2 bg-gradient-to-tr from-emerald-600 to-teal-600 rounded-xl shadow-emerald-500/20 shadow-md">
            <Gauge className="w-4 h-4 text-white" />
          </div>
          <div>
            <h2 className="text-xs font-bold text-white flex items-center gap-2">
              Resource & Budget Summary
              <Badge
                tone="success"
                className="text-[10px] font-mono bg-emerald-950/80 text-emerald-400 border-emerald-800/60"
              >
                Active Session
              </Badge>
            </h2>
            <p className="text-[11px] text-slate-400">
              Live step meter, token quotas, compaction savings, and connected Ollama status
            </p>
          </div>
        </div>

        <Button
          variant="bordered"
          size="icon"
          data-testid="resource-refresh-btn"
          onClick={onRefresh}
          disabled={isRefreshing}
          className="rounded-xl border-slate-800 bg-slate-900 hover:bg-slate-800"
          title="Refresh resource metrics"
        >
          <RefreshCw className={`w-3.5 h-3.5 ${isRefreshing ? 'animate-spin text-cyan-400' : ''}`} />
        </Button>
      </div>

      {/* Surface Content Scrollable */}
      <div className="flex-1 overflow-y-auto min-h-0 space-y-4 pr-1">
        {/* Surface Section 1: the open conversation's own readings (#1272).
            Every figure here comes off the room the centre column is showing. Where the room
            reports no such quantity the surface says so in a sentence, because a number
            borrowed from another transcript is the plausible-substituted-value P6 forbids --
            which is exactly what this block used to do, reading the retired single-agent
            history while a room was on screen. */}
        <div
          data-testid="room-resource-readings"
          className="p-4 bg-slate-900/80 border border-slate-800 rounded-2xl shadow-md space-y-3"
        >
          <div className="flex items-center justify-between">
            <div className="flex items-center gap-2">
              <Layers className="w-4 h-4 text-emerald-400" />
              <span className="text-xs font-bold text-white uppercase tracking-wider font-mono">
                {RESOURCE_ROOM_COPY.heading}
              </span>
            </div>
            {room && (
              <span
                data-testid="room-reply-budget-readout"
                className={`text-xs font-mono font-bold ${stepTextColor}`}
              >
                {repliesSpent} / {repliesCeiling} ({replyPercentage}%)
              </span>
            )}
          </div>

          {room === null ? (
            <p data-testid="room-readings-unavailable" className="text-[11px] text-slate-400">
              {RESOURCE_ROOM_COPY.noRoom}
            </p>
          ) : (
            <>
              <p className="text-[11px] text-slate-400">{RESOURCE_ROOM_COPY.replyBudget}</p>

              {/* Progress Track */}
              <div className="w-full h-3 bg-slate-950 rounded-full overflow-hidden border border-slate-800/80 p-0.5">
                <div
                  data-testid="step-progress-bar"
                  className={`h-full rounded-full transition-all duration-300 ${stepBarColor}`}
                  style={{ width: `${Math.max(3, replyPercentage)}%` }}
                />
              </div>

              <p className="text-[10px] text-slate-500">{RESOURCE_ROOM_COPY.replyBudgetNote}</p>

              {/* The first of the two absences. Said in words rather than filled from the
                  single-agent store, which is where the "Step 3 / 50" here came from. */}
              <p
                data-testid="room-step-count-absent"
                className="text-[11px] text-slate-400 pt-1 border-t border-slate-800/60"
              >
                {RESOURCE_ROOM_COPY.noStepCount}
              </p>

              {/* Per seat, never totalled: a room's turns live in N separate contexts and any
                  single number over them (a sum, a max) would be one this surface invented. */}
              <div className="pt-1 border-t border-slate-800/60 space-y-1.5">
                <p className="text-[11px] text-slate-400">{RESOURCE_ROOM_COPY.seatsHeading}</p>
                {roomContext === null ? (
                  <p data-testid="room-seat-turns-unread" className="text-[10px] text-slate-500">
                    {RESOURCE_ROOM_COPY.seatsUnread}
                  </p>
                ) : (
                  <div data-testid="room-seat-turns" className="space-y-1">
                    {roomContext.seats.map((seat) => (
                      <div
                        key={seat.participant_id}
                        data-testid={`room-seat-turns-${seat.participant_id}`}
                        className="flex items-center justify-between text-[11px] text-slate-400 font-mono"
                      >
                        <span className="flex items-center gap-1.5">
                          {seatLabel(seat.participant_id)}
                          {!seat.live && (
                            <span
                              data-testid={`room-seat-from-record-${seat.participant_id}`}
                              title={RESOURCE_ROOM_COPY.fromRecordTitle}
                              className="text-[10px] px-1 py-0.5 rounded bg-slate-700/60 text-slate-300 font-sans font-medium"
                            >
                              {RESOURCE_ROOM_COPY.fromRecord}
                            </span>
                          )}
                        </span>
                        <span className="flex items-center gap-1.5">
                          <strong className="text-slate-200">{seat.active_turns}</strong>
                          <span className="text-slate-500">
                            / {roomContext.saturation_threshold}
                          </span>
                          {seat.is_saturated && (
                            <span
                              data-testid={`room-seat-saturated-${seat.participant_id}`}
                              title={RESOURCE_ROOM_COPY.saturatedTitle}
                              className="text-[10px] px-1 py-0.5 rounded bg-amber-500/20 text-amber-300 font-sans font-medium"
                            >
                              {RESOURCE_ROOM_COPY.saturated}
                            </span>
                          )}
                        </span>
                      </div>
                    ))}
                  </div>
                )}
                <p className="text-[10px] text-slate-500">{RESOURCE_ROOM_COPY.seatsNote}</p>
              </div>

              {/* The second absence. `RoomMessage.usage` is declared and nothing fills it on
                  the orchestrated path, so there is no room total to show; a figure here would
                  be one this surface made up. */}
              <p
                data-testid="room-token-total-absent"
                className="flex items-start gap-1.5 text-[11px] text-slate-400 pt-1 border-t border-slate-800/60"
              >
                <Coins className="w-3.5 h-3.5 text-amber-400 shrink-0 mt-0.5" />
                <span>{RESOURCE_ROOM_COPY.noTokenTotal}</span>
              </p>
            </>
          )}
        </div>

        {/* Surface Section 2: Token Quotas. Cost is not ours to manage, so none is shown. */}
        <div className="grid grid-cols-1 sm:grid-cols-2 gap-3">
          {/* Total Used Tokens */}
          <div className="p-3.5 bg-slate-900/80 border border-slate-800 rounded-2xl shadow-sm space-y-1">
            <div className="flex items-center justify-between text-slate-400 text-[11px]">
              <span>Token Quota</span>
              <Coins className="w-3.5 h-3.5 text-cyan-400" />
            </div>
            <p className="text-lg font-bold text-white font-mono">
              {sessionBudget.total_used_tokens.toLocaleString()}
            </p>
            <div className="w-full h-1.5 bg-slate-950 rounded-full overflow-hidden mt-1">
              <div
                className="h-full bg-cyan-500 rounded-full"
                style={{ width: `${Math.max(2, tokenPercentage)}%` }}
              />
            </div>
            <p className="text-[10px] text-cyan-400 font-mono mt-1">
              / {sessionBudget.max_tokens.toLocaleString()} ({tokenPercentage}%)
            </p>
          </div>

          {/* Prompt vs Completion */}
          <div className="p-3.5 bg-slate-900/80 border border-slate-800 rounded-2xl shadow-sm space-y-1">
            <div className="flex items-center justify-between text-slate-400 text-[11px]">
              <span>Prompt / Completion</span>
              <Cpu className="w-3.5 h-3.5 text-violet-400" />
            </div>
            <div className="text-xs font-mono space-y-0.5 pt-1">
              <div className="flex justify-between">
                <span className="text-slate-500">In:</span>
                <span className="text-slate-200 font-semibold">{sessionBudget.used_input_tokens.toLocaleString()}</span>
              </div>
              <div className="flex justify-between">
                <span className="text-slate-500">Out:</span>
                <span className="text-slate-200 font-semibold">{sessionBudget.used_output_tokens.toLocaleString()}</span>
              </div>
            </div>
            <p className="text-[10px] text-slate-500 font-mono pt-1">Accumulated context</p>
          </div>
        </div>

        {/* Surface Section 3: Context Savings from P7 Compaction */}
        <div className="p-4 bg-slate-900/80 border border-slate-800 rounded-2xl shadow-md space-y-3">
          <div className="flex items-center justify-between">
            <div className="flex items-center gap-2">
              <TrendingDown className="w-4 h-4 text-purple-400" />
              <span className="text-xs font-bold text-white uppercase tracking-wider font-mono">
                Dynamic Context Pruning & Savings
              </span>
            </div>
            <Badge
              tone="neutral"
              className="text-[11px] font-mono text-purple-300 bg-purple-950/80 border-purple-800/60"
            >
              {compactionHistory.length} Compactions
            </Badge>
          </div>

          <div className="grid grid-cols-2 gap-3 pt-1">
            <div className="p-3 rounded-xl bg-slate-950/80 border border-slate-800/80">
              <span className="text-[10px] text-slate-500 uppercase font-sans">Pruned Tokens Saved</span>
              <p className="text-base font-bold text-purple-400 font-mono mt-0.5">
                {totalTokensSaved.toLocaleString()} tokens
              </p>
            </div>
            <div className="p-3 rounded-xl bg-slate-950/80 border border-slate-800/80">
              <span className="text-[10px] text-slate-500 uppercase font-sans">Trigger Threshold</span>
              <p className="text-base font-bold text-amber-400 font-mono mt-0.5">
                70% Context Window
              </p>
            </div>
          </div>

          {compactionHistory.length > 0 && (
            <div className="space-y-1.5 pt-1">
              <span className="text-[10px] text-slate-500 uppercase font-sans">Compaction Log</span>
              <div className="space-y-1 max-h-28 overflow-y-auto font-mono text-[11px]">
                {compactionHistory.map((rec, idx) => (
                  <div
                    key={`compaction-${idx}`}
                    data-testid={`resource-compaction-${idx}`}
                    className="p-2 rounded-lg bg-slate-950 border border-slate-800/60 space-y-0.5 text-slate-300"
                  >
                    <div className="flex items-center justify-between">
                      <span className="text-slate-400">{rec.timestamp || `Event #${idx + 1}`}</span>
                      <span className="text-purple-400 font-semibold">+{rec.saved_tokens.toLocaleString()} saved</span>
                    </div>
                    {/* What the retired Budget surface's ledger table showed per event. */}
                    <div className="flex flex-wrap items-center justify-between gap-x-2 text-[10px] text-slate-500">
                      <span>{rec.reason}</span>
                      <span>
                        {rec.original_tokens.toLocaleString()} ➔ {rec.compacted_tokens.toLocaleString()}{' '}
                        (-{rec.compression_ratio_pct.toFixed(1)}%), kept {rec.kept_turns} turns
                      </span>
                    </div>
                  </div>
                ))}
              </div>
            </div>
          )}
        </div>

        {/* Surface Section 4: Connected Ollama Provider Status (0 Mock Leak, G1 / RFC §3.4) */}
        <div className="p-4 bg-slate-900/80 border border-slate-800 rounded-2xl shadow-md space-y-3">
          <div className="flex items-center justify-between">
            <div className="flex items-center gap-2">
              <Server className="w-4 h-4 text-cyan-400" />
              <span className="text-xs font-bold text-white uppercase tracking-wider font-mono">
                Connected Provider & Model
              </span>
            </div>
            <Badge
              tone="success"
              className="flex gap-1.5 font-mono text-emerald-400 bg-emerald-950/80 border-emerald-800/60"
            >
              <StatusDot tone="success" className="animate-pulse" />
              <span>Active</span>
            </Badge>
          </div>

          <div className="p-3 rounded-xl bg-slate-950/90 border border-slate-800 space-y-2 font-mono text-xs">
            <div className="flex items-center justify-between">
              <span className="text-slate-500">Provider:</span>
              <span data-testid="connected-provider-name" className="text-cyan-300 font-semibold uppercase">{activeProvider}</span>
            </div>
            <div className="flex items-center justify-between">
              <span className="text-slate-500">Model:</span>
              <span data-testid="connected-model-name" className="text-white font-semibold">{activeModel}</span>
            </div>
            <div className="flex items-center justify-between">
              <span className="text-slate-500">Base URL:</span>
              <span className="text-slate-400 text-[11px]">{activeEndpoint}</span>
            </div>
            <div className="flex items-center justify-between pt-1 border-t border-slate-800/60">
              <span className="text-slate-500">Execution Mode:</span>
              <span className="text-emerald-400 flex items-center gap-1 text-[11px]">
                <ShieldCheck className="w-3.5 h-3.5" />
                <span>100% Local / Private</span>
              </span>
            </div>
          </div>
        </div>

        {/* Multi-Provider Breakdown (if available) */}
        {Object.keys(providers).length > 0 && (
          <div className="p-4 bg-slate-900/80 border border-slate-800 rounded-2xl shadow-md space-y-3">
            <div className="flex items-center gap-2">
              <PieChart className="w-4 h-4 text-cyan-400" />
              <span className="text-xs font-bold text-white uppercase tracking-wider font-mono">
                Multi-Provider Attribution
              </span>
            </div>
            <div className="space-y-2">
              {Object.entries(providers).map(([pName, pData]) => {
                const totalTokens = (pData.input_tokens || 0) + (pData.output_tokens || 0);
                return (
                  <div
                    key={pName}
                    data-testid={`resource-provider-${pName}`}
                    className="p-2.5 rounded-xl bg-slate-950 border border-slate-800 space-y-1 text-xs font-mono"
                  >
                    <div className="flex items-center justify-between">
                      <span className="text-white font-semibold">{pName}</span>
                      <div className="flex items-center gap-3 text-slate-400 text-[11px]">
                        <span>{totalTokens.toLocaleString()} tokens</span>
                      </div>
                    </div>
                    {/* The split and model list the retired Budget surface showed per provider. */}
                    <div className="flex justify-between text-[10px] text-slate-500">
                      <span>In: {(pData.input_tokens || 0).toLocaleString()}</span>
                      <span>Out: {(pData.output_tokens || 0).toLocaleString()}</span>
                    </div>
                    {(pData.models || []).length > 0 && (
                      <div className="flex flex-wrap gap-1">
                        {pData.models.map((m) => (
                          <span
                            key={m}
                            className="px-1.5 py-0.5 rounded bg-slate-900 border border-slate-800 text-slate-400 text-[9px]"
                          >
                            {m}
                          </span>
                        ))}
                      </div>
                    )}
                  </div>
                );
              })}
            </div>
          </div>
        )}
      </div>
    </div>
  );
};
