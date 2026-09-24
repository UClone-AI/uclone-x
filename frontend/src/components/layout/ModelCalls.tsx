import React, { useState } from 'react';
import { ChevronDown, ChevronRight, Copy, Cpu } from 'lucide-react';
import {
  type RequestBlock,
  type RequestLayer,
  type StepDetail,
  type TraceRead,
  type TraceStep,
  type TraceToolResult,
  type TurnTraceResponse,
  asText,
  formatDuration,
  readTrace,
  reasonLine,
  requestBlocks,
  stepHeadline,
  stepPairJson,
  turnTraceUrls,
} from '../../lib/turnTrace';

/** Shown on a step whose rebuilt request could not be checked against what was sent (§4.4.4). */
export const UNVERIFIED_NOTE =
  'Rebuilt from the saved record. A credential-like string was redacted, so this may differ from what was sent.';

const LAYER_LABELS: Record<RequestLayer, string> = {
  identity: 'identity',
  slow_context: 'slow context',
  conversation: '',
  turn_context: 'turn context',
};

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

const Loading: React.FC<{ what: string }> = ({ what }) => (
  <p className="py-1 text-[11px] text-slate-500">Reading {what} from the session log...</p>
);

const Failed: React.FC<{ testId: string; message: string }> = ({ testId, message }) => (
  <p data-testid={testId} className="py-1 text-[11px] text-amber-300">
    {message}
  </p>
);

type Tab = 'request' | 'tools' | 'response' | 'raw';

const TABS: Array<{ id: Tab; label: string }> = [
  { id: 'request', label: 'Request' },
  { id: 'tools', label: 'Tools offered' },
  { id: 'response', label: 'Response' },
  { id: 'raw', label: 'Raw' },
];

const RequestBlockView: React.FC<{ block: RequestBlock }> = ({ block }) => {
  const layer = LAYER_LABELS[block.layer];
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
          {block.layer === 'slow_context'
            ? 'This request carried no slow context in its system message.'
            : 'This message carried no text.'}
        </p>
      )}
      {toolCalls.map((call, i) => (
        <div key={call.id ?? i} className="min-w-0">
          <p className="font-mono text-[10px] text-slate-400">calls {call.name}</p>
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

const ResponseView: React.FC<{ detail: StepDetail }> = ({ detail }) => {
  const response = detail.response;
  if (!response) {
    return (
      <p data-testid="model-call-response-reason" className="text-[11px] text-amber-300">
        {detail.response_reason ?? 'The Core gave no response and no reason for its absence.'}
      </p>
    );
  }
  return (
    <div className="space-y-2">
      <div>
        <p className="mb-1 text-[10px] uppercase text-slate-500">Content</p>
        {response.content ? (
          <Pre testId="model-call-response-content">{response.content}</Pre>
        ) : (
          <p className="text-[10px] text-slate-500">The response carried no text.</p>
        )}
      </div>
      {response.thinking ? (
        <div>
          <p className="mb-1 text-[10px] uppercase text-slate-500">Thinking</p>
          <Pre testId="model-call-response-thinking">{response.thinking}</Pre>
        </div>
      ) : null}
      {response.tool_calls.length > 0 ? (
        <div>
          <p className="mb-1 text-[10px] uppercase text-slate-500">Tool calls</p>
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
          <p className="mb-1 text-[10px] uppercase text-rose-400">Error</p>
          <Pre testId="model-call-response-error" tone="error">
            {asText(response.error)}
          </Pre>
        </div>
      ) : null}
    </div>
  );
};

const StepTabs: React.FC<{ detail: StepDetail }> = ({ detail }) => {
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
          {UNVERIFIED_NOTE}
        </p>
      ) : null}
      <div className="flex flex-wrap items-center gap-1" role="tablist">
        {TABS.map(({ id, label }) => (
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
            {label}
          </button>
        ))}
        <button
          type="button"
          data-testid="model-call-copy"
          onClick={() => void copy()}
          className="ml-auto flex items-center gap-1 rounded px-2 py-0.5 text-[11px] text-slate-400 hover:text-slate-200"
        >
          <Copy className="h-3 w-3" />
          {copied === 'done' ? 'Copied' : copied === 'failed' ? 'Copy refused' : 'Copy as JSON'}
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
            <p data-testid="model-call-request-reason" className="text-[11px] text-amber-300">
              {detail.request_reason ?? 'The Core gave no request and no reason for its absence.'}
            </p>
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
              <p className="text-[11px] text-slate-400">This request offered the model no tools.</p>
            )
          ) : (
            <p className="text-[11px] text-amber-300">
              {detail.request_reason ?? 'The Core gave no request and no reason for its absence.'}
            </p>
          )
        ) : null}
        {tab === 'response' ? <ResponseView detail={detail} /> : null}
        {tab === 'raw' ? <Pre testId="model-call-raw">{JSON.stringify(detail, null, 2)}</Pre> : null}
      </div>
    </div>
  );
};

const ToolResultView: React.FC<{ result: TraceToolResult }> = ({ result }) => {
  const failed = result.status !== null && result.status !== 'success';
  return (
    <li data-testid="model-call-tool-result" className="min-w-0 space-y-1">
      <div className="flex flex-wrap items-baseline gap-x-2 font-mono text-[10px]">
        <span className="text-slate-200">{result.name}</span>
        <span className={failed ? 'text-rose-400' : 'text-slate-400'}>
          {result.status ?? 'status not recorded'}
        </span>
        {result.outcome ? <span className="text-slate-500">{result.outcome}</span> : null}
        {typeof result.duration_ms === 'number' ? (
          <span className="text-slate-500">{formatDuration(result.duration_ms)}</span>
        ) : null}
      </div>
      <Pre tone={failed ? 'error' : 'plain'}>
        {result.output === null || result.output === undefined
          ? 'The log holds no output for this call.'
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
  // A Core that answers the step with a reason instead of a step says why in `reason.message`.
  const stepReason = reasonLine(body?.reason);

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
              {stepHeadline(step)}
            </span>
            {step.verified === true ? (
              <span data-testid="model-call-verified" className="text-[10px] text-emerald-400">
                verified
              </span>
            ) : step.verified === false ? (
              <span data-testid="model-call-verified" className="text-[10px] text-amber-400">
                not verified
              </span>
            ) : null}
            {step.response_status === 'error' ? (
              <span className="text-[10px] text-rose-400">call failed</span>
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
          Request: {step.request_reason ?? 'unavailable, and the Core gave no reason.'}
        </p>
      ) : null}
      {step.response_status === 'unavailable' ? (
        <p data-testid="model-call-row-response-reason" className="ml-4 mt-1 text-[11px] text-amber-300">
          Response: {step.response_reason ?? 'unavailable, and the Core gave no reason.'}
        </p>
      ) : null}

      {open ? (
        <div className="mt-2 min-w-0 space-y-2 border-t border-slate-800 pt-2">
          {detail === null || detail.status === 'loading' ? <Loading what="this call" /> : null}
          {detail?.status === 'failed' ? (
            <Failed testId="model-call-detail-failed" message={detail.message} />
          ) : null}
          {stepReason ? <Failed testId="model-call-detail-failed" message={stepReason} /> : null}
          {body && !stepReason ? <StepTabs detail={body} /> : null}
          {step.tool_results.length > 0 ? (
            <div className="space-y-1">
              <p className="text-[10px] uppercase text-slate-500">Tool results, in full</p>
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
        <span>Model calls</span>
      </button>

      {open ? (
        <div className="min-w-0 space-y-2">
          {read === null || read.status === 'loading' ? <Loading what="the turn's model calls" /> : null}
          {read?.status === 'failed' ? (
            <Failed testId="model-calls-failed" message={read.message} />
          ) : null}
          {body && !trace ? (
            <Failed
              testId="model-calls-reason"
              message={reasonLine(body.reason) ?? 'The Core returned no trace and no reason for it.'}
            />
          ) : null}
          {trace ? (
            <>
              {trace.rolled_back ? (
                <p data-testid="model-calls-rolled-back" className="text-[11px] text-amber-300">
                  This turn was rolled back: the conversation kept nothing of it. Its calls are
                  shown as the log recorded them.
                </p>
              ) : null}
              {trace.subagents.length > 0 ? (
                <p className="text-[11px] text-slate-400">
                  Helpers ran ({trace.subagents.join(', ')}). Their own model calls are not kept.
                </p>
              ) : null}
              {trace.subagents_reason ? (
                <p data-testid="model-calls-subagents-reason" className="text-[11px] text-amber-300">
                  {trace.subagents_reason}
                </p>
              ) : null}
              {trace.steps.length === 0 ? (
                <p data-testid="model-calls-no-steps" className="text-[11px] text-slate-400">
                  The session log lists no model calls for this turn.
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
