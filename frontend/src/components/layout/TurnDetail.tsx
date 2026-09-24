import React from 'react';
import { AlertTriangle, Clock, Coins, FileText, MessageSquare, Wrench } from 'lucide-react';
import type { EventEnvelope, RoomState, RoomTurnRefusal } from '../../types';
import { senderLabel, serviceRefLabel } from '../../lib/rooms';
import { refusalRemedy, turnFailureSentence } from '../../lib/turnOutcome';
import {
  type TurnDocument,
  type TurnSummary,
  useOpenInDocs,
  useTurnSummary,
} from '../../lib/roomDock';
import { classifyTool, parseArgs, argsRecord } from '../../lib/toolLabels';
import { readDeveloperMode } from '../../lib/developerMode';
import { ModelCalls } from './ModelCalls';

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

const NOTABLE_LABELS: Record<string, string> = {
  degraded: 'Substituted: the requested model was not available.',
  failover: 'Served on a failover path after earlier attempts failed.',
  retried: 'Retried: required multiple attempts to complete.',
  contested: 'Contested: multiple candidates were evaluated.',
};

/**
 * Detail view for one turn: what the clone did (steps & changed files) and how it replied.
 *
 * Follows the reading order of §4.3 in #1491 (turn inspection design):
 * 1. Header: Speaker name, outcome ("Replied", "Couldn't finish", "Stopped"), chips (time, duration, tokens).
 * 2. "What it did": Step rows with classifyTool labels, status, duration. Plain failure on error.
 * 3. "Changed": Modified documents with Open in Docs action, and unnamed writes.
 * 4. Model line: Answered by {served_by}. Collapsed behind disclosure unless notable is non-empty.
 * 5. Error/refusal: Plain words; raw error visible only in developer mode.
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
  const devMode =
    developerMode ?? (typeof window !== 'undefined' ? readDeveloperMode() : false);

  if (!room || seq === null) {
    return (
      <div
        data-testid="turn-detail-empty"
        className="text-center py-12 text-slate-500 font-sans space-y-2"
      >
        <MessageSquare className="w-8 h-8 mx-auto text-slate-600" />
        <p>No turn selected.</p>
        <p className="text-[11px] text-slate-600">
          Press <span className="text-slate-400">why ›</span> under any turn to see which model
          served it and how the speaker was chosen.
        </p>
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
        <p>This turn is no longer in the conversation.</p>
        <p className="mt-1 text-[11px] text-slate-600">
          It was removed by a rewind or by clearing the history. Pick another turn.
        </p>
      </div>
    );
  }

  if (
    summaryRead.fault?.kind === 'status' &&
    (summaryRead.error?.includes('404') || summaryRead.fault.detail?.includes('turn_not_found'))
  ) {
    return (
      <div data-testid="turn-detail-gone" className="py-12 text-center text-slate-500 font-sans">
        <p>This turn is no longer in the conversation.</p>
        <p className="mt-1 text-[11px] text-slate-600">
          It was removed by a rewind or by clearing the history. Pick another turn.
        </p>
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
        provenance: message.provenance ?? null,
        decision: message.decision ?? null,
        rendered_through:
          message.rendered_through != null ? String(message.rendered_through) : null,
        usage: null,
        notable: (message.provenance?.degraded ? ['degraded'] : [])
          .concat(message.provenance?.path === 'failover' ? ['failover'] : [])
          .concat(
            message.provenance?.attempts && message.provenance.attempts.length > 1
              ? ['retried']
              : [],
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
        <p>Loading turn record...</p>
      </div>
    );
  }

  // 1. Header calculations
  const speakerName = senderLabel(room, summary.sender_id);
  const outcome = !summary.completed
    ? 'Stopped'
    : summary.error || summary.refusal
      ? "Couldn't finish"
      : 'Replied';

  const provenance = summary.provenance;
  const served = serviceRefLabel(provenance?.served_by);
  const requested = serviceRefLabel(provenance?.requested);
  const attempts = provenance?.attempts ?? [];

  // Duration: only when recorded; no chip if absent, never "0s"
  const durationText = null; // Issue A will provide turn duration

  // Tokens: from usage; never 0 if absent
  const tokenCount = summary.usage?.total_tokens;
  const tokenSource = summary.usage?.count_source;
  const tokensText =
    typeof tokenCount === 'number'
      ? `${tokenCount.toLocaleString()} tokens${tokenSource && tokenSource !== 'provider' ? ` (${tokenSource})` : ''}`
      : null;

  const isNotable = summary.notable && summary.notable.length > 0;

  const fullProvenanceFields = (
    <div className="space-y-3 pt-2 font-mono text-xs">
      {/* Notable reasons if non-empty */}
      {isNotable && (
        <div
          data-testid="turn-detail-notable"
          className="p-2 rounded bg-amber-950/20 border border-amber-800/40 text-amber-300 font-sans text-xs space-y-1"
        >
          {summary.notable.map((code) => (
            <p key={code}>{NOTABLE_LABELS[code] ?? code}</p>
          ))}
        </div>
      )}

      {/* Asked for against served */}
      {served && requested && requested !== served ? (
        <Field label="Asked for">
          <span data-testid="turn-detail-requested">{requested}</span>
          {provenance?.degraded ? (
            <span className="ml-2 font-sans text-[10px] text-amber-400">substituted</span>
          ) : null}
        </Field>
      ) : null}

      {/* Path */}
      {provenance ? (
        <Field label={devMode ? 'Path' : 'Route'}>
          <span data-testid="turn-detail-path">{provenance.path}</span>
        </Field>
      ) : null}

      {/* Attempts */}
      {attempts.length > 0 ? (
        <Field label={`Attempts (${attempts.length})`}>
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
      <Field label="Chosen by">
        {summary.decision ? (
          <div data-testid="turn-detail-decision" className="space-y-1">
            {devMode ? (
              <p>{summary.decision.selector}</p>
            ) : (
              <p className="font-sans">
                {summary.decision.selector === 'sole_agent'
                  ? 'Only seated agent'
                  : 'Selected agent'}
              </p>
            )}
            {summary.decision.reasoning ? (
              <p className="font-sans text-slate-400">{summary.decision.reasoning}</p>
            ) : null}
            <p className="font-sans text-[11px] text-slate-500">
              {devMode
                ? `Selector reported confidence ${summary.decision.confidence.toFixed(2)}.`
                : `Confidence ${summary.decision.confidence.toFixed(2)}.`}
            </p>
          </div>
        ) : (
          <span data-testid="turn-detail-no-decision" className="font-sans text-slate-400">
            Nobody chose this speaker: the turn records no selection.
          </span>
        )}
      </Field>

      {/* Saw through */}
      <Field label={devMode ? 'Saw through' : 'Context through'}>
        {summary.rendered_through ? (
          <span data-testid="turn-detail-rendered-through">turn {summary.rendered_through}</span>
        ) : (
          <span data-testid="turn-detail-no-rendered-through" className="font-sans text-slate-400">
            Not recorded for this turn.
          </span>
        )}
      </Field>

      {/* In developer mode, expose raw identifiers */}
      {devMode && (
        <div className="pt-2 border-t border-slate-800 text-[10px] text-slate-500 space-y-0.5">
          {summary.turn_id && <p>turn_id: {summary.turn_id}</p>}
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
        </div>
      </div>

      {/* 2. Steps — "What it did" */}
      <div data-testid="turn-detail-steps" className="space-y-2">
        <div className="flex items-center gap-1.5 text-xs font-semibold text-slate-300">
          <Wrench className="w-3.5 h-3.5 text-slate-400" />
          <span>What it did</span>
        </div>

        {summary.steps_absent_reason === 'not_recorded' ? (
          <p data-testid="turn-detail-steps-absent" className="text-xs text-slate-400 py-1">
            This turn's steps weren't recorded.
          </p>
        ) : summary.steps_absent_reason === 'not_an_agent_turn' ? (
          <p data-testid="turn-detail-steps-absent" className="text-xs text-slate-400 py-1">
            This turn was not run by an agent.
          </p>
        ) : summary.steps.length === 0 ? (
          <p data-testid="turn-detail-steps-empty" className="text-xs text-slate-400 py-1">
            No tools used.
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
                        {step.status}
                      </span>
                    </div>
                  </div>

                  {step.subagent_id && (
                    <p className="text-[11px] text-slate-400">
                      Handed part of this to a helper. The helper's own steps aren't kept.
                    </p>
                  )}

                  {isFailed && (
                    <div className="space-y-1">
                      <p className="text-[11px] text-rose-400">This step failed.</p>
                      {devMode && step.error && (
                        <pre className="text-[10px] font-mono text-rose-300 bg-rose-950/40 p-1.5 rounded whitespace-pre-wrap">
                          {step.error}
                        </pre>
                      )}
                    </div>
                  )}

                  {devMode && step.arguments_preview && (
                    <pre className="text-[10px] font-mono text-slate-400 bg-slate-950 p-1.5 rounded overflow-x-auto">
                      {step.arguments_preview}
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
          <span>Changed</span>
        </div>

        {summary.documents.length === 0 && (!summary.unnamed_writes || summary.unnamed_writes === 0) ? (
          <p data-testid="turn-detail-documents-empty" className="text-xs text-slate-400 py-1">
            No documents changed.
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
                      ({doc.writes} writes)
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
                    Open
                  </button>
                )}
              </div>
            ))}

            {summary.unnamed_writes > 0 && (
              <p
                data-testid="turn-detail-unnamed-writes"
                className="text-[11px] text-slate-400 py-0.5"
              >
                Also wrote {summary.unnamed_writes} file(s) without a name.
              </p>
            )}
          </div>
        )}
      </div>

      {/* 4. Model Line & Notable / Disclosure */}
      <div className="space-y-2 border-t border-slate-800/80 pt-3">
        <div className="text-xs text-slate-300">
          {served ? (
            <span>
              Answered by <span data-testid="turn-detail-served">{served}</span>
            </span>
          ) : (
            <span data-testid="turn-detail-no-model" className="font-sans text-slate-400">
              This turn reported no model, so none is named here.
            </span>
          )}
        </div>

        {isNotable ? (
          fullProvenanceFields
        ) : (
          <details data-testid="turn-detail-disclosure" className="group">
            <summary className="cursor-pointer text-xs text-slate-400 hover:text-slate-200 select-none py-1">
              How this reply was chosen
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
            <span>This turn did not finish</span>
          </div>
          <p className="text-slate-300 text-xs">
            {turnFailureSentence(
              speakerName,
              summary.refusal as RoomTurnRefusal | null,
              summary.completed,
            )}
          </p>
          {summary.refusal ? (
            <p className="font-sans text-[10px] text-slate-400">
              {refusalRemedy(summary.refusal as RoomTurnRefusal)}
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
