import React, { useState } from 'react';
import { ChevronDown, ChevronRight, Copy, Cpu } from 'lucide-react';
import {
  type RequestBlock,
  type RequestLayer,
  type StepDetail,
  type TraceProblem,
  type TraceRead,
  type TraceStep,
  type TraceToolResult,
  type TurnTraceResponse,
  asText,
  formatDuration,
  fromLog,
  readTrace,
  reasonProblem,
  requestBlocks,
  stepHeadline,
  stepPairJson,
  turnTraceUrls,
} from '../../lib/turnTrace';
import { fmt, useCopy, type Messages } from '../../i18n';

/**
 * Every sentence this panel shows is the catalog's (`dock.modelCalls`, #1903). What the Core
 * sends in English -- a reason's detail, a rebuild error, a tool's output -- is shown beside a
 * sentence in its own `Detail`, never spliced into one, so a Korean sentence stays Korean.
 */
type Words = Messages['dock']['modelCalls'];

const layerLabel = (t: Words, layer: RequestLayer): string =>
  layer === 'identity'
    ? t.layers.identity
    : layer === 'slow_context'
      ? t.layers.slowContext
      : layer === 'turn_context'
        ? t.layers.turnContext
        : '';

/** The catalog's sentence for `code`, when the catalog has one. */
const known = <K extends string>(table: Record<K, string>, code: string | null): string | null =>
  code !== null && Object.prototype.hasOwnProperty.call(table, code) ? table[code as K] : null;

/**
 * Reason codes whose Core text says nothing the catalog's sentence does not, so it is not
 * repeated beside the sentence. Every other code's text names what was missing, and follows.
 */
const SAID_IN_FULL = new Set(['not_recorded', 'before_capture']);

/**
 * Why a half of a step is unavailable, in the reader's language (#1907): the catalog's
 * sentence for the Core's code, with the Core's English reason beside it where it adds
 * something. `sentence` is `null` for a code this head does not know, or a Core older than the
 * code; the reason is then all there is.
 */
const codedReason = <K extends string>(
  table: Record<K, string>,
  code: string | null | undefined,
  reason: string | null,
): { sentence: string | null; detail: string | null } => {
  const sentence = known(table, code ?? null);
  if (sentence === null) return { sentence: null, detail: reason };
  return { sentence, detail: reason && !SAID_IN_FULL.has(code as string) ? reason : null };
};

/** A problem in words: the sentence, and the Core's own text to show beside it, if any. */
interface Worded {
  sentence: string;
  detail?: string | null;
}

/**
 * `problem` in the reader's language. A reason code the catalog knows is its sentence, with a
 * `detail` code after it; one it does not know yet is the Core's message, which is English.
 */
export const wordProblem = (
  t: Words,
  problem: TraceProblem,
  where: { seq: number; step?: number },
): Worded => {
  switch (problem.kind) {
    case 'reason': {
      const sentence = known(t.reasons, problem.code);
      if (sentence === null) return { sentence: problem.message, detail: problem.detail };
      const worded = fmt(sentence, { seq: where.seq, step: where.step ?? '' });
      // A log that could not be read says what kind of failure it was; a kind the catalog
      // knows is a second sentence, one it does not know stays the Core's code.
      const kind = problem.code === 'log_unreadable' ? known(t.logFailures, problem.detail ?? null) : null;
      return kind === null ? { sentence: worded, detail: problem.detail } : { sentence: `${worded} ${kind}` };
    }
    case 'message':
      // The Core's refusal in its own words, which are English: a sentence the reader can
      // read, and the Core's text beside it.
      return { sentence: fmt(t.failure.refused, { status: problem.status }), detail: problem.message };
    case 'unreachable':
      return { sentence: t.failure.unreachable, detail: problem.detail };
    case 'notJson':
      return { sentence: t.failure.notJson };
    case 'http':
      return { sentence: fmt(t.failure.http, { status: problem.status }) };
  }
};

/** The Core's English text beside a sentence: set apart, in the log's own type. */
const Detail: React.FC<{ children: React.ReactNode }> = ({ children }) => (
  <>
    {' '}
    <code data-testid="model-call-core-detail" className="break-words font-mono text-[10px] text-slate-400">
      {children}
    </code>
  </>
);

/**
 * Long text inside the dock: it wraps, and it scrolls inside its own box rather than widening
 * the dock or the page.
 */
const Pre: React.FC<{ testId?: string; tone?: 'plain' | 'error'; children: React.ReactNode }> = ({
  testId,
  tone = 'plain',
  children,
}) => (
  <pre
    data-testid={testId}
    className={`max-h-80 overflow-auto whitespace-pre-wrap break-words rounded bg-slate-950 p-1.5 font-mono text-[10px] ${
      tone === 'error' ? 'text-rose-300' : 'text-slate-300'
    }`}
  >
    {children}
  </pre>
);

const Loading: React.FC<{ text: string }> = ({ text }) => (
  <p className="py-1 text-[11px] text-slate-500">{text}</p>
);

const Failed: React.FC<{ testId: string; worded: Worded }> = ({ testId, worded }) => (
  <p data-testid={testId} className="py-1 text-[11px] text-amber-300">
    {worded.sentence}
    {worded.detail ? <Detail>({worded.detail})</Detail> : null}
  </p>
);

type Tab = 'request' | 'tools' | 'response' | 'raw';

const TABS: Tab[] = ['request', 'tools', 'response', 'raw'];

const RequestBlockView: React.FC<{ block: RequestBlock }> = ({ block }) => {
  const t = useCopy().dock.modelCalls;
  const layer = layerLabel(t, block.layer);
  const toolCalls = block.layer === 'conversation' ? (block.message.tool_calls ?? []) : [];
  return (
    <div
      data-testid="request-block"
      data-role={block.role}
      data-layer={block.layer}
      className={`min-w-0 space-y-1 rounded border p-1.5 ${
        block.layer === 'turn_context'
          ? 'border-cyan-800/60 bg-cyan-950/20'
          : 'border-slate-800 bg-slate-900/60'
      }`}
    >
      <div className="flex flex-wrap items-baseline gap-x-2 font-mono text-[10px] text-slate-400">
        <span className="font-semibold text-slate-200">{block.role}</span>
        {layer ? <span className="text-cyan-400">{layer}</span> : null}
        {block.message.name ? <span>{block.message.name}</span> : null}
        {block.message.tool_call_id ? <span>{block.message.tool_call_id}</span> : null}
      </div>
      {block.content ? (
        <Pre>{block.content}</Pre>
      ) : (
        <p className="text-[10px] text-slate-500">
          {block.layer === 'slow_context' ? t.noSlowContext : t.noMessageText}
        </p>
      )}
      {toolCalls.map((call, i) => (
        <div key={call.id ?? i} className="min-w-0">
          <p className="font-mono text-[10px] text-slate-400">{fmt(t.calls, { name: call.name ?? '' })}</p>
          <Pre>{asText(call.arguments ?? {})}</Pre>
        </div>
      ))}
    </div>
  );
};

const ToolOffered: React.FC<{ tool: { name: string; description?: string; parameters?: unknown } }> = ({
  tool,
}) => {
  const [open, setOpen] = useState(false);
  return (
    <li data-testid="offered-tool" className="min-w-0">
      <button
        type="button"
        onClick={() => setOpen((v) => !v)}
        aria-expanded={open}
        className="flex items-center gap-1 font-mono text-[11px] text-slate-200 hover:text-white"
      >
        {open ? <ChevronDown className="h-3 w-3" /> : <ChevronRight className="h-3 w-3" />}
        {tool.name}
      </button>
      {open ? (
        <div className="ml-4 mt-1 space-y-1">
          {tool.description ? (
            <p className="text-[10px] text-slate-400">{tool.description}</p>
          ) : null}
          <Pre testId="offered-tool-schema">{asText(tool.parameters ?? {})}</Pre>
        </div>
      ) : null}
    </li>
  );
};

/**
 * Why a half of the step is missing: the catalog's sentence for the Core's code, with the
 * Core's reason as its detail; for a code this head does not know, or a Core older than the
 * code, the catalog's `unknown` sentence with the reason beside it (#1911), never the English
 * reason on its own; or the catalog's sentence for no reason at all.
 */
const MissingHalf: React.FC<{
  testId?: string;
  why: { sentence: string | null; detail: string | null };
  unknown: string;
  none: string;
}> = ({ testId, why, unknown, none }) => (
  <p data-testid={testId} className="text-[11px] text-amber-300">
    {why.sentence !== null ? (
      <>
        {why.sentence}
        {why.detail ? <Detail>{why.detail}</Detail> : null}
      </>
    ) : why.detail ? (
      <>
        {unknown}
        <Detail>{why.detail}</Detail>
      </>
    ) : (
      none
    )}
  </p>
);

const ResponseView: React.FC<{ detail: StepDetail }> = ({ detail }) => {
  const t = useCopy().dock.modelCalls;
  const response = detail.response;
  if (!response) {
    return (
      <MissingHalf
        testId="model-call-response-reason"
        why={codedReason(t.responseReasons, detail.response_code, detail.response_reason)}
        unknown={t.missingResponse}
        none={t.noResponse}
      />
    );
  }
  return (
    <div className="space-y-2">
      <div>
        <p className="mb-1 text-[10px] uppercase text-slate-500">{t.response.content}</p>
        {response.content ? (
          <Pre testId="model-call-response-content">{response.content}</Pre>
        ) : (
          <p className="text-[10px] text-slate-500">{t.response.noText}</p>
        )}
      </div>
      {response.thinking ? (
        <div>
          <p className="mb-1 text-[10px] uppercase text-slate-500">{t.response.thinking}</p>
          <Pre testId="model-call-response-thinking">{response.thinking}</Pre>
        </div>
      ) : null}
      {response.tool_calls.length > 0 ? (
        <div>
          <p className="mb-1 text-[10px] uppercase text-slate-500">{t.response.toolCalls}</p>
          <ul className="space-y-1">
            {response.tool_calls.map((call, i) => (
              <li key={call.id ?? i} data-testid="model-call-response-tool-call" className="min-w-0">
                <p className="font-mono text-[11px] text-slate-200">{call.name}</p>
                <Pre>{asText(call.arguments ?? {})}</Pre>
              </li>
            ))}
          </ul>
        </div>
      ) : null}
      {response.error ? (
        <div>
          <p className="mb-1 text-[10px] uppercase text-rose-400">{t.response.error}</p>
          <Pre testId="model-call-response-error" tone="error">
            {asText(response.error)}
          </Pre>
        </div>
      ) : null}
    </div>
  );
};

/**
 * Whether the conversation the step sent is what the session log rebuilds (§5.8, #1848), or
 * why it was not checked. Nothing when the Core says nothing about it.
 *
 * Why it was not checked is the Core's code, worded by the catalog (#1903). The Core's English
 * reason follows as a detail where it names something the sentence does not (which log entry
 * was missing), and after the generic sentence for a code this head does not know or a Core
 * older than the code.
 */
const FromLogLine: React.FC<{ detail: StepDetail }> = ({ detail }) => {
  const t = useCopy().dock.modelCalls.fromLog;
  const check = fromLog(detail);
  if (!check) return null;
  let body: React.ReactNode;
  if (check.status === 'unchecked') {
    const sentence = known(t.reasons, check.code);
    // Why the epoch could not be rebuilt is a code of its own (#1911): a second sentence, with
    // the Core's reason still beside it, since it names the entry that was missing.
    const why = check.code === 'epoch_unreadable' ? known(t.details, check.detailCode) : null;
    const showReason = check.reason !== null && (sentence === null || check.code === 'epoch_unreadable');
    const lead = sentence ?? (check.reason === null ? t.uncheckedNoReason : t.unchecked);
    body = (
      <>
        {why === null ? lead : `${lead} ${why}`}
        {showReason ? <Detail>{check.reason}</Detail> : null}
      </>
    );
  } else {
    body = t[check.status];
  }
  return (
    <p
      data-testid="model-call-from-log"
      data-status={check.status}
      data-code={check.status === 'unchecked' ? (check.code ?? undefined) : undefined}
      className={`text-[11px] ${check.status === 'matches' ? 'text-slate-400' : 'text-amber-300'}`}
    >
      {body}
    </p>
  );
};

const StepTabs: React.FC<{ detail: StepDetail }> = ({ detail }) => {
  const t = useCopy().dock.modelCalls;
  const [tab, setTab] = useState<Tab>('request');
  const [copied, setCopied] = useState<'idle' | 'done' | 'failed'>('idle');

  const copy = async (): Promise<void> => {
    try {
      await navigator.clipboard.writeText(stepPairJson(detail));
      setCopied('done');
    } catch {
      setCopied('failed');
    }
  };

  const request = detail.request;
  return (
    <div data-testid="model-call-detail" className="min-w-0 space-y-2">
      {detail.verified === false ? (
        <p data-testid="model-call-unverified" className="text-[11px] text-amber-300">
          {t.unverified}
        </p>
      ) : null}
      <FromLogLine detail={detail} />
      <div className="flex flex-wrap items-center gap-1" role="tablist">
        {TABS.map((id) => (
          <button
            key={id}
            type="button"
            role="tab"
            aria-selected={tab === id}
            data-testid={`model-call-tab-${id}`}
            onClick={() => setTab(id)}
            className={`rounded px-2 py-0.5 text-[11px] ${
              tab === id ? 'bg-slate-700 text-slate-100' : 'text-slate-400 hover:text-slate-200'
            }`}
          >
            {t.tabs[id]}
          </button>
        ))}
        <button
          type="button"
          data-testid="model-call-copy"
          onClick={() => void copy()}
          className="ml-auto flex items-center gap-1 rounded px-2 py-0.5 text-[11px] text-slate-400 hover:text-slate-200"
        >
          <Copy className="h-3 w-3" />
          {copied === 'done' ? t.copy.done : copied === 'failed' ? t.copy.failed : t.copy.idle}
        </button>
      </div>

      <div role="tabpanel" data-testid={`model-call-panel-${tab}`} className="min-w-0">
        {tab === 'request' ? (
          request ? (
            <div className="space-y-1.5">
              {requestBlocks(request, detail.layers).map((block, i) => (
                <RequestBlockView key={`${block.index}-${block.layer}-${i}`} block={block} />
              ))}
            </div>
          ) : (
            <MissingHalf
              testId="model-call-request-reason"
              why={codedReason(t.requestReasons, detail.request_code, detail.request_reason)}
              unknown={t.missingRequest}
              none={t.noRequest}
            />
          )
        ) : null}
        {tab === 'tools' ? (
          request ? (
            request.tools.length > 0 ? (
              <ul className="space-y-1">
                {request.tools.map((tool) => (
                  <ToolOffered key={tool.name} tool={tool} />
                ))}
              </ul>
            ) : (
              <p className="text-[11px] text-slate-400">{t.noTools}</p>
            )
          ) : (
            <MissingHalf
              why={codedReason(t.requestReasons, detail.request_code, detail.request_reason)}
              unknown={t.missingRequest}
              none={t.noRequest}
            />
          )
        ) : null}
        {tab === 'response' ? <ResponseView detail={detail} /> : null}
        {tab === 'raw' ? <Pre testId="model-call-raw">{JSON.stringify(detail, null, 2)}</Pre> : null}
      </div>
    </div>
  );
};

const ToolResultView: React.FC<{ result: TraceToolResult }> = ({ result }) => {
  const t = useCopy().dock.modelCalls;
  const failed = result.status !== null && result.status !== 'success';
  return (
    <li data-testid="model-call-tool-result" className="min-w-0 space-y-1">
      <div className="flex flex-wrap items-baseline gap-x-2 font-mono text-[10px]">
        <span className="text-slate-200">{result.name}</span>
        <span className={failed ? 'text-rose-400' : 'text-slate-400'}>
          {result.status ?? t.statusNotRecorded}
        </span>
        {result.outcome ? <span className="text-slate-500">{result.outcome}</span> : null}
        {typeof result.duration_ms === 'number' ? (
          <span className="text-slate-500">{formatDuration(result.duration_ms)}</span>
        ) : null}
      </div>
      <Pre tone={failed ? 'error' : 'plain'}>
        {result.output_unavailable
          ? t.outputUnavailable
          : result.output === null || result.output === undefined
            ? t.noOutput
            : asText(result.output)}
      </Pre>
    </li>
  );
};

const StepRow: React.FC<{ roomId: string; seq: number; step: TraceStep }> = ({
  roomId,
  seq,
  step,
}) => {
  const t = useCopy().dock.modelCalls;
  const [open, setOpen] = useState(false);
  const [detail, setDetail] = useState<TraceRead<StepDetail> | null>(null);

  const toggle = (): void => {
    const next = !open;
    setOpen(next);
    if (next && detail === null) {
      setDetail({ status: 'loading' });
      void readTrace<StepDetail>(turnTraceUrls.step(roomId, seq, step.step)).then(setDetail);
    }
  };

  const toolNames = (step.response?.tool_calls ?? [])
    .map((call) => call.name)
    .filter((name): name is string => typeof name === 'string' && name !== '');

  const body = detail?.status === 'ok' ? detail.data : null;
  // A Core that answers the step with a reason instead of a step says why in `reason`.
  const stepReason = reasonProblem(body?.reason);
  const requestWhy = codedReason(t.requestReasons, step.request_code, step.request_reason);
  const responseWhy = codedReason(t.responseReasons, step.response_code, step.response_reason);
  const where = { seq, step: step.step };

  return (
    <li
      data-testid="model-call-row"
      data-step={step.step}
      className="min-w-0 rounded-lg border border-slate-800/80 bg-slate-900/60 p-2"
    >
      <button
        type="button"
        data-testid={`model-call-toggle-${step.step}`}
        aria-expanded={open}
        onClick={toggle}
        className="flex w-full min-w-0 items-start gap-1 text-left"
      >
        {open ? (
          <ChevronDown className="mt-0.5 h-3 w-3 shrink-0 text-slate-400" />
        ) : (
          <ChevronRight className="mt-0.5 h-3 w-3 shrink-0 text-slate-400" />
        )}
        <span className="min-w-0 flex-1 space-y-0.5">
          <span className="flex flex-wrap items-baseline gap-x-2">
            <span data-testid="model-call-headline" className="break-words font-mono text-[11px] text-slate-200">
              {stepHeadline(step, t.headline)}
            </span>
            {step.verified === true ? (
              <span data-testid="model-call-verified" className="text-[10px] text-emerald-400">
                {t.verified}
              </span>
            ) : step.verified === false ? (
              <span data-testid="model-call-verified" className="text-[10px] text-amber-400">
                {t.notVerified}
              </span>
            ) : null}
            {step.response_status === 'error' ? (
              <span className="text-[10px] text-rose-400">{t.callFailed}</span>
            ) : null}
          </span>
          {toolNames.length > 0 ? (
            <span data-testid="model-call-tool-names" className="block break-words font-mono text-[10px] text-slate-400">
              → {toolNames.join(', ')}
            </span>
          ) : null}
        </span>
      </button>

      {step.request_status === 'unavailable' ? (
        <p data-testid="model-call-row-request-reason" className="ml-4 mt-1 text-[11px] text-amber-300">
          {requestWhy.sentence !== null ? (
            <>
              {requestWhy.sentence}
              {requestWhy.detail ? <Detail>{requestWhy.detail}</Detail> : null}
            </>
          ) : step.request_reason ? (
            <>
              {t.rowRequest}
              <Detail>{step.request_reason}</Detail>
            </>
          ) : (
            t.rowRequestNoReason
          )}
        </p>
      ) : null}
      {step.response_status === 'unavailable' ? (
        <p data-testid="model-call-row-response-reason" className="ml-4 mt-1 text-[11px] text-amber-300">
          {responseWhy.sentence !== null ? (
            <>
              {responseWhy.sentence}
              {responseWhy.detail ? <Detail>{responseWhy.detail}</Detail> : null}
            </>
          ) : step.response_reason ? (
            <>
              {t.rowResponse}
              <Detail>{step.response_reason}</Detail>
            </>
          ) : (
            t.rowResponseNoReason
          )}
        </p>
      ) : null}

      {open ? (
        <div className="mt-2 min-w-0 space-y-2 border-t border-slate-800 pt-2">
          {detail === null || detail.status === 'loading' ? <Loading text={t.loadingStep} /> : null}
          {detail?.status === 'failed' ? (
            <Failed testId="model-call-detail-failed" worded={wordProblem(t, detail.problem, where)} />
          ) : null}
          {stepReason ? (
            <Failed testId="model-call-detail-failed" worded={wordProblem(t, stepReason, where)} />
          ) : null}
          {body && !stepReason ? <StepTabs detail={body} /> : null}
          {step.tool_results.length > 0 ? (
            <div className="space-y-1">
              <p className="text-[10px] uppercase text-slate-500">{t.toolResults}</p>
              <ul className="space-y-1.5">
                {step.tool_results.map((result, i) => (
                  <ToolResultView key={result.tool_call_id || i} result={result} />
                ))}
              </ul>
            </div>
          ) : null}
        </div>
      ) : null}
    </li>
  );
};

/**
 * The developer-mode "Model calls" section at the bottom of the Turn surface (§4.4.4, #1492).
 *
 * It exists only in developer mode -- the parent does not mount it otherwise, so no trace is
 * ever requested with the mode off. Mounted, it still reads nothing until it is expanded,
 * because the trace reads the whole session log. It reads once: collapsing and expanding again
 * shows what it already has.
 */
export const ModelCalls: React.FC<{ roomId: string; seq: number }> = ({ roomId, seq }) => {
  const t = useCopy().dock.modelCalls;
  const [open, setOpen] = useState(false);
  const [read, setRead] = useState<TraceRead<TurnTraceResponse> | null>(null);

  const toggle = (): void => {
    const next = !open;
    setOpen(next);
    if (next && read === null) {
      setRead({ status: 'loading' });
      void readTrace<TurnTraceResponse>(turnTraceUrls.trace(roomId, seq)).then(setRead);
    }
  };

  const body = read?.status === 'ok' ? read.data : null;
  const trace = body?.trace ?? null;

  return (
    <div data-testid="model-calls" className="min-w-0 space-y-2 border-t border-slate-800/80 pt-3">
      <button
        type="button"
        data-testid="model-calls-toggle"
        aria-expanded={open}
        onClick={toggle}
        className="flex items-center gap-1.5 text-xs font-semibold text-slate-300 hover:text-slate-100"
      >
        {open ? <ChevronDown className="h-3.5 w-3.5" /> : <ChevronRight className="h-3.5 w-3.5" />}
        <Cpu className="h-3.5 w-3.5 text-slate-400" />
        <span>{t.title}</span>
      </button>

      {open ? (
        <div className="min-w-0 space-y-2">
          {read === null || read.status === 'loading' ? <Loading text={t.loadingTurn} /> : null}
          {read?.status === 'failed' ? (
            <Failed testId="model-calls-failed" worded={wordProblem(t, read.problem, { seq })} />
          ) : null}
          {body && !trace ? (
            <Failed
              testId="model-calls-reason"
              worded={(() => {
                const reason = reasonProblem(body.reason);
                return reason ? wordProblem(t, reason, { seq }) : { sentence: t.noTrace };
              })()}
            />
          ) : null}
          {trace ? (
            <>
              {trace.rolled_back ? (
                <p data-testid="model-calls-rolled-back" className="text-[11px] text-amber-300">
                  {t.rolledBack}
                </p>
              ) : null}
              {trace.subagents.length > 0 ? (
                <p className="text-[11px] text-slate-400">
                  {fmt(t.helpers, { names: trace.subagents.join(', ') })}
                </p>
              ) : null}
              {/* The Core sets `subagents_reason` for one case only (`SUBAGENT_UNREADABLE`), so
                  its presence is the sentence, in the reader's language (#1903). */}
              {trace.subagents_reason ? (
                <p data-testid="model-calls-subagents-reason" className="text-[11px] text-amber-300">
                  {t.helpersUnreadable}
                </p>
              ) : null}
              {trace.steps.length === 0 ? (
                <p data-testid="model-calls-no-steps" className="text-[11px] text-slate-400">
                  {t.noSteps}
                </p>
              ) : (
                <ul className="space-y-1.5">
                  {trace.steps.map((step) => (
                    <StepRow key={step.step} roomId={roomId} seq={seq} step={step} />
                  ))}
                </ul>
              )}
            </>
          ) : null}
        </div>
      ) : null}
    </div>
  );
};
