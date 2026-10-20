import React, { useState } from 'react';
import { AlertTriangle, Check, Clock, Coins, Copy, FileText, MessageSquare, Wrench } from 'lucide-react';
import type { EventEnvelope, RoomState, RoomTurnRefusal } from '../../types';
import { senderLabel, serviceRefLabel } from '../../lib/rooms';
import { providerFailureRemedy, refusalRemedy, turnFailureSentence } from '../../lib/turnOutcome';
import {
  type TurnDocument,
  type TurnSummary,
  useOpenInDocs,
  useTurnSummary,
} from '../../lib/roomDock';
import { classifyTool, parseArgs, argsRecord } from '../../lib/toolLabels';
import { readDeveloperMode } from '../../lib/developerMode';
import { ModelCalls } from './ModelCalls';
import { fmt, plural, useCopy } from '../../i18n';

export interface TurnDetailProps {
  /** The conversation on screen, or `null` when none is. */
  room: RoomState | null;
  /** The turn `why ›` was pressed on, by `seq`. `null` before any has been. */
  seq: number | null;
  /** Live SSE envelopes for reactive refetching. */
  events?: EventEnvelope[];
  /** Whether developer mode is enabled. When omitted, reads from localStorage. */
  developerMode?: boolean;
  /** Optional pre-loaded or mock summary for tests. */
  summary?: TurnSummary;
}

const Field: React.FC<{ label: string; children: React.ReactNode }> = ({ label, children }) => (
  <div>
    <div className="text-[10px] text-slate-500 uppercase font-sans mb-1">{label}</div>
    <div className="text-slate-300">{children}</div>
  </div>
);

/**
 * The reasons a turn is worth a second look, as `room/turn_summary.py` names them. A turn a
 * provider error ended is `ended_in_error`, never `degraded` or `failover`: the core's error
 * handler records it that way, but nothing answered in the model's place (#2229).
 */
const endedInErrorOf = (error: string | null | undefined, path: string | undefined): boolean =>
  Boolean(error) && path === 'failover';

/**
 * Detail view for one turn: what the clone did (steps & changed files) and how it replied.
 *
 * Follows the reading order of §4.3 in #1491 (turn inspection design):
 * 1. Header: Speaker name, outcome ("Replied", "Couldn't finish", "Stopped"), chips (time, duration, tokens).
 * 2. "What it did": Step rows with classifyTool labels, status, duration. Plain failure on error.
 * 3. "Changed": Modified documents with Open in Docs action, and unnamed writes.
 * 4. Model line: Answered by {served_by}. Collapsed behind disclosure unless notable is non-empty.
 * 5. Error/refusal: Plain words; raw error visible only in developer mode.
 *    A turn a provider error ended names no model as having answered, and with developer
 *    mode off shows none of the error handler's record: its route, its attempts (with their
 *    exception classes) and, in records written before #2229, the clone's id (#2229).
 * 6. Model calls (developer mode only, §4.4.4, #1492): each request as sent and its response.
 *    Not mounted with the mode off, so no trace is requested.
 */
export const TurnDetail: React.FC<TurnDetailProps> = ({
  room,
  seq,
  events = [],
  developerMode,
  summary: overrideSummary,
}) => {
  const openInDocs = useOpenInDocs();
  const copy = useCopy().turnDetail;
  const outcomeCopy = useCopy().conversation.outcome;
  const whyLabel = useCopy().conversation.row.why;
  const devMode =
    developerMode ?? (typeof window !== 'undefined' ? readDeveloperMode() : false);
  const [copiedTurnId, setCopiedTurnId] = useState(false);

  const copyTurnId = async (id: string): Promise<void> => {
    try {
      await navigator.clipboard.writeText(id);
      setCopiedTurnId(true);
      setTimeout(() => setCopiedTurnId(false), 2000);
    } catch {
      /* storage or clipboard write refused */
    }
  };

  if (!room || seq === null) {
    return (
      <div
        data-testid="turn-detail-empty"
        className="text-center py-12 text-slate-500 font-sans space-y-2"
      >
        <MessageSquare className="w-8 h-8 mx-auto text-slate-600" />
        <p>{copy.empty.title}</p>
        <p className="text-[11px] text-slate-600">{fmt(copy.empty.hint, { why: whyLabel })}</p>
      </div>
    );
  }

  const message = room.transcript.find((m) => m.seq === seq);
  const summaryRead = useTurnSummary(
    room.room_id,
    message ? seq : null,
    message?.turn_id,
    events,
  );

  // If the turn is not in transcript, or the summary route returns 404:
  if (!message && !overrideSummary) {
    return (
      <div data-testid="turn-detail-gone" className="py-12 text-center text-slate-500 font-sans">
        <p>{copy.gone.title}</p>
        <p className="mt-1 text-[11px] text-slate-600">{copy.gone.hint}</p>
      </div>
    );
  }

  if (
    summaryRead.fault?.kind === 'status' &&
    (summaryRead.error?.includes('404') || summaryRead.fault.detail?.includes('turn_not_found'))
  ) {
    return (
      <div data-testid="turn-detail-gone" className="py-12 text-center text-slate-500 font-sans">
        <p>{copy.gone.title}</p>
        <p className="mt-1 text-[11px] text-slate-600">{copy.gone.hint}</p>
      </div>
    );
  }

  // Provisional summary if message is in transcript while loading
  const provisionalSummary: TurnSummary | null = message
    ? {
        seq: message.seq,
        turn_id: message.turn_id ?? null,
        sender_id: message.sender_id,
        created_at: message.created_at,
        completed: message.completed,
        error: message.error ?? null,
        refusal: message.refusal ?? null,
        provider_failure: message.provider_failure ?? null,
        provenance: message.provenance ?? null,
        decision: message.decision ?? null,
        rendered_through:
          message.rendered_through != null ? String(message.rendered_through) : null,
        usage: null,
        // As `room/turn_summary.py` reads it: an error-ended turn is not a substitution.
        notable: (endedInErrorOf(message.error, message.provenance?.path)
          ? ['ended_in_error']
          : (message.provenance?.degraded ? ['degraded'] : [])
              .concat(message.provenance?.path === 'failover' ? ['failover'] : [])
              .concat(
                message.provenance?.attempts && message.provenance.attempts.length > 1
                  ? ['retried']
                  : [],
              )
        )
          .concat(
            message.decision && message.decision.selector !== 'sole_agent' ? ['contested'] : [],
          ),
        steps: [],
        documents: [],
        unnamed_writes: 0,
        subagent_steps: [],
        steps_absent_reason: null,
      }
    : null;

  const summary = overrideSummary ?? summaryRead.data ?? provisionalSummary;

  if (!summary) {
    return (
      <div
        data-testid="turn-detail-loading"
        className="py-12 text-center text-slate-500 font-sans"
      >
        <p>{copy.loading}</p>
      </div>
    );
  }

  // 1. Header calculations
  const speakerName = senderLabel(room, summary.sender_id);
  const outcome = !summary.completed
    ? copy.outcome.stopped
    : summary.error || summary.refusal
      ? copy.outcome.couldNotFinish
      : copy.outcome.replied;

  const provenance = summary.provenance;
  const served = serviceRefLabel(provenance?.served_by);
  const requested = serviceRefLabel(provenance?.requested);
  const attempts = provenance?.attempts ?? [];
  const endedInError = summary.notable.includes('ended_in_error');
  // With developer mode off, an error-ended turn's record is the error handler's internals
  // (route, attempts with exception classes, and in older records the clone's id): the
  // error box below says what happened instead (#2229).
  const showRecord = devMode || !endedInError;

  // Duration: only when recorded; no chip if absent, never "0s"
  const durationText = null; // Issue A will provide turn duration

  // Tokens: from usage; never 0 if absent
  const tokenCount = summary.usage?.total_tokens;
  const tokenSource = summary.usage?.count_source;
  const tokensText =
    typeof tokenCount === 'number'
      ? `${plural(copy.tokens, tokenCount, { count: tokenCount.toLocaleString() })}${tokenSource && tokenSource !== 'provider' ? ` (${tokenSource})` : ''}`
      : null;

  const isNotable = summary.notable && summary.notable.length > 0;

  const fullProvenanceFields = (
    <div className="space-y-3 pt-2 font-mono text-xs">
      {/* Notable reasons if non-empty; an error-ended turn's would repeat the model line */}
      {showRecord && isNotable && (
        <div
          data-testid="turn-detail-notable"
          className="p-2 rounded bg-amber-950/20 border border-amber-800/40 text-amber-300 font-sans text-xs space-y-1"
        >
          {summary.notable.map((code) => (
            <p key={code}>{(copy.notable as Partial<Record<string, string>>)[code] ?? code}</p>
          ))}
        </div>
      )}

      {/* Asked for against served */}
      {showRecord && served && requested && requested !== served ? (
        <Field label={copy.record.askedFor}>
          <span data-testid="turn-detail-requested">{requested}</span>
          {provenance?.degraded && !endedInError ? (
            <span className="ml-2 font-sans text-[10px] text-amber-400">
              {copy.record.substituted}
            </span>
          ) : null}
        </Field>
      ) : null}

      {/* Path */}
      {showRecord && provenance ? (
        <Field label={devMode ? copy.record.path : copy.record.route}>
          <span data-testid="turn-detail-path">{provenance.path}</span>
        </Field>
      ) : null}

      {/* Attempts */}
      {showRecord && attempts.length > 0 ? (
        <Field label={fmt(copy.record.attempts, { count: attempts.length })}>
          <ul data-testid="turn-detail-attempts" className="space-y-1">
            {attempts.map((attempt, i) => (
              <li key={i} className="flex flex-wrap items-baseline gap-x-2">
                <span>{serviceRefLabel({ provider: attempt.provider, model: attempt.model })}</span>
                <span className="text-rose-400">{attempt.error_class}</span>
                {attempt.status_code != null ? (
                  <span className="text-slate-500">{attempt.status_code}</span>
                ) : null}
              </li>
            ))}
          </ul>
        </Field>
      ) : null}

      {/* Chosen by */}
      {(devMode || summary.decision?.selector !== 'sole_agent') && (
        <Field label={copy.record.chosenBy}>
          {summary.decision ? (
            <div data-testid="turn-detail-decision" className="space-y-1">
              {devMode ? (
                <p>{summary.decision.selector}</p>
              ) : (
                <p className="font-sans">
                  {summary.decision.selector === 'sole_agent'
                    ? copy.record.onlySeated
                    : copy.record.selected}
                </p>
              )}
              {summary.decision.reasoning ? (
                <p className="font-sans text-slate-400">{summary.decision.reasoning}</p>
              ) : null}
              <p className="font-sans text-[11px] text-slate-500">
                {fmt(devMode ? copy.record.confidenceDev : copy.record.confidence, {
                  value: summary.decision.confidence.toFixed(2),
                })}
              </p>
            </div>
          ) : (
            <span data-testid="turn-detail-no-decision" className="font-sans text-slate-400">
              {copy.record.noDecision}
            </span>
          )}
        </Field>
      )}

      {/* Saw through */}
      <Field label={devMode ? copy.record.sawThrough : copy.record.contextThrough}>
        {summary.rendered_through ? (
          <span data-testid="turn-detail-rendered-through">
            {fmt(copy.record.throughTurn, { n: summary.rendered_through })}
          </span>
        ) : (
          <span data-testid="turn-detail-no-rendered-through" className="font-sans text-slate-400">
            {copy.record.notRecorded}
          </span>
        )}
      </Field>

      {/* In developer mode, expose raw identifiers */}
      {devMode && (
        <div className="pt-2 border-t border-slate-800 text-[10px] text-slate-500 space-y-1">
          {summary.turn_id && (
            <div className="flex items-center justify-between gap-2">
              <span className="font-mono break-all">turn_id: {summary.turn_id}</span>
              <button
                type="button"
                data-testid="turn-copy-id"
                onClick={() => void copyTurnId(summary.turn_id!)}
                className="flex items-center gap-1 text-[10px] text-slate-400 hover:text-slate-200 px-1.5 py-0.5 rounded bg-slate-900 hover:bg-slate-800 border border-slate-800 shrink-0 transition-colors cursor-pointer"
                title={copy.record.copyTurnId}
              >
                {copiedTurnId ? (
                  <>
                    <Check className="w-3 h-3 text-emerald-400" />
                    <span className="text-emerald-400">{copy.record.copied}</span>
                  </>
                ) : (
                  <>
                    <Copy className="w-3 h-3" />
                    <span>{copy.record.copy}</span>
                  </>
                )}
              </button>
            </div>
          )}
          <p>seq: {summary.seq}</p>
        </div>
      )}
    </div>
  );

  return (
    <div data-testid="turn-detail" className="dock-scope @container space-y-4 font-sans text-xs">
      {/* 1. Header */}
      <div className="border-b border-slate-800/80 pb-3 space-y-2">
        <div className="flex items-center justify-between">
          <div className="flex items-center gap-1.5 font-bold text-slate-200">
            <MessageSquare className="w-4 h-4 text-slate-400 shrink-0" />
            <span>{speakerName}</span>
          </div>
          <span
            data-testid="turn-detail-outcome"
            className="text-xs font-medium text-slate-300 bg-slate-800/80 px-2 py-0.5 rounded-full"
          >
            {outcome}
          </span>
        </div>

        {/* Chips: recorded values only */}
        <div className="flex flex-wrap items-center gap-1.5 text-[11px] text-slate-400">
          <span className="flex items-center gap-1 bg-slate-900 px-2 py-0.5 rounded text-slate-400">
            <Clock className="w-3 h-3 text-slate-500" />
            {new Date(summary.created_at).toLocaleTimeString([], {
              hour: '2-digit',
              minute: '2-digit',
              second: '2-digit',
            })}
          </span>

          {durationText ? (
            <span
              data-testid="turn-chip-duration"
              className="bg-slate-900 px-2 py-0.5 rounded text-slate-400"
            >
              {durationText}
            </span>
          ) : null}

          {tokensText ? (
            <span
              data-testid="turn-chip-tokens"
              className="flex items-center gap-1 bg-slate-900 px-2 py-0.5 rounded text-slate-400"
            >
              <Coins className="w-3 h-3 text-slate-500" />
              {tokensText}
            </span>
          ) : null}

          {devMode && summary.turn_id ? (
            <button
              type="button"
              data-testid="turn-chip-copy-id"
              onClick={() => void copyTurnId(summary.turn_id!)}
              className="flex items-center gap-1 bg-slate-900 hover:bg-slate-800 px-2 py-0.5 rounded text-slate-400 hover:text-slate-200 border border-slate-800/80 font-mono transition-colors cursor-pointer"
              title={fmt(copy.record.copyTurnIdChip, { id: summary.turn_id })}
            >
              {copiedTurnId ? (
                <>
                  <Check className="w-3 h-3 text-emerald-400" />
                  <span className="text-emerald-400">{copy.record.copied}</span>
                </>
              ) : (
                <>
                  <Copy className="w-3 h-3 text-slate-500" />
                  <span>{summary.turn_id.slice(0, 8)}…</span>
                </>
              )}
            </button>
          ) : null}
        </div>
      </div>

      {/* 2. Steps — "What it did" */}
      <div data-testid="turn-detail-steps" className="space-y-2">
        <div className="flex items-center gap-1.5 text-xs font-semibold text-slate-300">
          <Wrench className="w-3.5 h-3.5 text-slate-400" />
          <span>{copy.steps.title}</span>
        </div>

        {summary.steps_absent_reason === 'not_recorded' ? (
          <p data-testid="turn-detail-steps-absent" className="text-xs text-slate-400 py-1">
            {copy.steps.notRecorded}
          </p>
        ) : summary.steps_absent_reason === 'not_an_agent_turn' ? (
          <p data-testid="turn-detail-steps-absent" className="text-xs text-slate-400 py-1">
            {copy.steps.notAnAgentTurn}
          </p>
        ) : summary.steps.length === 0 ? (
          <p data-testid="turn-detail-steps-empty" className="text-xs text-slate-400 py-1">
            {copy.steps.none}
          </p>
        ) : (
          <div className="space-y-1.5">
            {summary.steps.map((step, idx) => {
              const args = parseArgs(step.arguments_preview);
              const { label, summaryTitle } = classifyTool(step.tool_name, argsRecord(args));
              const isFailed = step.status === 'error' || Boolean(step.error);

              return (
                <div
                  key={step.tool_call_id || idx}
                  data-testid="turn-step-row"
                  className="p-2 rounded-lg bg-slate-900/60 border border-slate-800/80 space-y-1"
                >
                  <div className="flex items-center justify-between text-xs">
                    <span className="font-medium text-slate-200">
                      {devMode ? `${label} (${step.tool_name})` : summaryTitle || label}
                    </span>
                    <div className="flex items-center gap-2 text-[11px]">
                      {typeof step.duration_ms === 'number' && step.duration_ms > 0 && (
                        <span className="text-slate-500 font-mono">
                          {step.duration_ms >= 1000
                            ? `${(step.duration_ms / 1000).toFixed(1)}s`
                            : `${step.duration_ms}ms`}
                        </span>
                      )}
                      <span
                        className={
                          isFailed
                            ? 'text-rose-400'
                            : step.status === 'running'
                              ? 'text-amber-400'
                              : 'text-emerald-400'
                        }
                      >
                        {(copy.steps.status as Partial<Record<string, string>>)[step.status] ??
                          step.status}
                      </span>
                    </div>
                  </div>

                  {step.subagent_id && (
                    <p className="text-[11px] text-slate-400">
                      {copy.steps.helper}
                    </p>
                  )}

                  {isFailed && (
                    <div className="space-y-1">
                      <p className="text-[11px] text-rose-400">{copy.steps.failed}</p>
                      {devMode && step.error && (
                        <pre className="text-[10px] font-mono text-rose-300 bg-rose-950/40 p-1.5 rounded whitespace-pre-wrap">
                          {step.error}
                        </pre>
                      )}
                    </div>
                  )}

                  {devMode && step.arguments_preview && (
                    // Pretty-printed and wrapped: one long JSON line scrolled sideways
                    // and hid every argument past the first screen width.
                    <pre
                      data-testid="turn-step-args"
                      className="text-[10px] font-mono text-slate-400 bg-slate-950 p-1.5 rounded max-h-64 overflow-y-auto whitespace-pre-wrap break-words"
                    >
                      {typeof args === 'string' ? args : JSON.stringify(args, null, 2)}
                    </pre>
                  )}
                </div>
              );
            })}
          </div>
        )}
      </div>

      {/* 3. Documents — "Changed" */}
      <div data-testid="turn-detail-documents" className="space-y-2">
        <div className="flex items-center gap-1.5 text-xs font-semibold text-slate-300">
          <FileText className="w-3.5 h-3.5 text-slate-400" />
          <span>{copy.changed.title}</span>
        </div>

        {summary.documents.length === 0 && (!summary.unnamed_writes || summary.unnamed_writes === 0) ? (
          <p data-testid="turn-detail-documents-empty" className="text-xs text-slate-400 py-1">
            {copy.changed.none}
          </p>
        ) : (
          <div className="space-y-1.5">
            {summary.documents.map((doc: TurnDocument) => (
              <div
                key={doc.path}
                data-testid="turn-doc-row"
                className="flex items-center justify-between p-2 rounded-lg bg-slate-900/60 border border-slate-800/80 text-xs"
              >
                <div className="flex items-center gap-1.5 min-w-0">
                  <span className="font-mono text-slate-300 truncate">{doc.path}</span>
                  {doc.writes > 1 && (
                    <span className="text-[11px] text-slate-500 shrink-0">
                      {fmt(copy.changed.writes, { count: doc.writes })}
                    </span>
                  )}
                </div>
                {openInDocs && (
                  <button
                    type="button"
                    data-testid={`open-doc-${doc.path}`}
                    onClick={() => openInDocs(doc.path)}
                    className="text-xs text-cyan-400 hover:text-cyan-300 underline shrink-0 ml-2"
                  >
                    {copy.changed.open}
                  </button>
                )}
              </div>
            ))}

            {summary.unnamed_writes > 0 && (
              <p
                data-testid="turn-detail-unnamed-writes"
                className="text-[11px] text-slate-400 py-0.5"
              >
                {plural(copy.changed.unnamed, summary.unnamed_writes)}
              </p>
            )}
          </div>
        )}
      </div>

      {/* 4. Model Line & Notable / Disclosure */}
      <div className="space-y-2 border-t border-slate-800/80 pt-3">
        <div className="text-xs text-slate-300">
          {endedInError ? (
            // The error handler is not a model that answered (#2229).
            <span data-testid="turn-detail-no-answer" className="font-sans text-slate-400">
              {provenance?.requested?.model
                ? fmt(copy.model.noAnswer, { model: provenance.requested.model })
                : copy.model.noAnswerUnnamed}
            </span>
          ) : served ? (
            <span>
              {copy.model.answeredBy} <span data-testid="turn-detail-served">{served}</span>
            </span>
          ) : (
            <span data-testid="turn-detail-no-model" className="font-sans text-slate-400">
              {copy.model.none}
            </span>
          )}
        </div>

        {isNotable ? (
          fullProvenanceFields
        ) : (
          <details data-testid="turn-detail-disclosure" className="group">
            <summary className="cursor-pointer text-xs text-slate-400 hover:text-slate-200 select-none py-1">
              {copy.model.disclosure}
            </summary>
            {fullProvenanceFields}
          </details>
        )}
      </div>

      {/* 5. Error or refusal */}
      {summary.error ? (
        <div data-testid="turn-detail-error" className="p-2.5 rounded-xl bg-rose-950/30 space-y-1">
          <div className="flex items-center gap-1.5 text-rose-400 font-semibold text-[11px]">
            <AlertTriangle className="w-3.5 h-3.5" />
            <span>{copy.error.title}</span>
          </div>
          <p className="text-slate-300 text-xs">
            {turnFailureSentence(
              speakerName,
              summary.refusal as RoomTurnRefusal | null,
              summary.completed,
              summary.provider_failure,
              outcomeCopy,
            )}
          </p>
          {summary.refusal ? (
            <p className="font-sans text-[10px] text-slate-400">
              {refusalRemedy(summary.refusal as RoomTurnRefusal, outcomeCopy)}
            </p>
          ) : summary.provider_failure &&
            providerFailureRemedy(summary.provider_failure.kind, outcomeCopy) ? (
            <p data-testid="turn-detail-remedy" className="font-sans text-[10px] text-slate-400">
              {providerFailureRemedy(summary.provider_failure.kind, outcomeCopy)}
            </p>
          ) : null}
          {devMode && (
            <pre
              data-testid="turn-detail-raw-error"
              className="text-rose-300 text-[10px] whitespace-pre-wrap font-mono mt-1 pt-1 border-t border-rose-900/50"
            >
              {summary.error}
            </pre>
          )}
        </div>
      ) : null}

      {/* 6. Model calls: developer mode only; with it off nothing is mounted and nothing read */}
      {devMode ? <ModelCalls key={`${room.room_id}:${seq}`} roomId={room.room_id} seq={seq} /> : null}
    </div>
  );
};
