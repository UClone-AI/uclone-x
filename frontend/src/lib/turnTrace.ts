/**
 * The developer reader of one room turn: its model calls, each request as sent and its
 * response (turn inspection design §4.4, #1492).
 *
 * The Core half is `src/uclone_x/agent/turn_trace.py` and the two routes in
 * `src/uclone_x/ui/room_dock.py` (#1490). The shapes below mirror their answers.
 *
 * **Reasons are rendered, never interpreted.** A trace the Core cannot give comes back as
 * `trace: null` with a `reason: {code, message}`, and the head prints `message`. The `code` is
 * for tests; no branch here reads it, so a code the Core adds later needs no change on this
 * side. The same holds for a step read that fails: whatever message the Core gave is shown.
 *
 * Read only in developer mode, and only on an explicit expand: the trace reads the whole
 * session log on every request (§6).
 */

const enc = encodeURIComponent;

export const turnTraceUrls = {
  trace: (roomId: string, seq: number): string =>
    `/api/rooms/${enc(roomId)}/turns/${seq}/trace`,
  step: (roomId: string, seq: number, step: number): string =>
    `/api/rooms/${enc(roomId)}/turns/${seq}/trace/steps/${step}`,
};

/** Why the Core could not give a trace. `message` is what the head prints. */
export interface TraceReason {
  code: string;
  message: string;
  /**
   * A stable name for the failure underneath, where the Core gives one (`log_unreadable`
   * names `unknown_event_type`, `malformed_log`, `not_text`, `read_failed` or `unexpected`).
   */
  detail?: string | null;
}

/**
 * A reason as the head prints it: the Core's message, and its `detail` after it when there is
 * one. Neither `code` nor `detail` decides anything here; `detail` is only shown.
 */
export const reasonLine = (reason: TraceReason | null | undefined): string | null => {
  if (!reason || typeof reason.message !== 'string' || reason.message.trim() === '') return null;
  return reason.detail ? `${reason.message} (${reason.detail})` : reason.message;
};

/** One tool call's result in full, as the session log holds it (not the room's preview). */
export interface TraceToolResult {
  tool_call_id: string;
  name: string;
  status: string | null;
  outcome: string | null;
  /** The whole output as logged; on a failed call this is the raw error. */
  output: unknown;
  duration_ms: number | null;
  at: string | null;
}

/** A `MODEL_RESPONSE` event's fields. */
export interface ModelResponseRecord {
  turn_index: number | null;
  step: number | null;
  started_at: string | null;
  ended_at: string | null;
  streamed: boolean;
  content: string | null;
  thinking: string | null;
  tool_calls: Array<{ id?: string; name?: string; arguments?: unknown; [key: string]: unknown }>;
  finish_reason: string | null;
  model_name: string | null;
  /** `TokenUsage` as JSON: `input_tokens`, `output_tokens`, `total_tokens`, `count_source`. */
  usage: Record<string, unknown> | null;
  error: Record<string, unknown> | null;
}

export interface TraceStep {
  step: number;
  request_status: 'ok' | 'unavailable';
  request_reason: string | null;
  /** Whether the rebuilt request matched the recorded digest; `null` when unavailable. */
  verified: boolean | null;
  message_count: number | null;
  model: string | null;
  temperature: number | null;
  max_tokens: number | null;
  response_status: 'ok' | 'error' | 'unavailable';
  response_reason: string | null;
  response: ModelResponseRecord | null;
  tool_results: TraceToolResult[];
}

export interface TurnTrace {
  session_id: string;
  turn_index: number;
  started_at: string | null;
  ended_at: string | null;
  steps: TraceStep[];
  nudges: Array<Record<string, unknown>>;
  rolled_back: boolean;
  dropped_tool_steps: Array<Record<string, unknown>>;
  turn_end: Record<string, unknown> | null;
  subagents: string[];
  /** Set when a tool result names a helper but the Core could not read it. */
  subagents_reason?: string | null;
}

/** `GET …/turns/{seq}/trace`: a trace, or `trace: null` and the reason there is none. */
export interface TurnTraceResponse {
  room_id: string;
  seq: number;
  turn_id: string | null;
  participant_id: string;
  session_id: string;
  trace: TurnTrace | null;
  reason?: TraceReason | null;
}

/** An `LLMRequest` as JSON. */
export interface TraceMessage {
  role: string;
  content: string | null;
  name?: string | null;
  tool_call_id?: string | null;
  tool_calls?: Array<{ id?: string; name?: string; arguments?: unknown }>;
  [key: string]: unknown;
}

export interface TraceToolDefinition {
  name: string;
  description?: string;
  parameters?: unknown;
}

export interface TraceRequest {
  model: string | null;
  messages: TraceMessage[];
  tools: TraceToolDefinition[];
  temperature?: number;
  max_tokens?: number | null;
  [key: string]: unknown;
}

/** The request's layers (llm-request-layering.md §5), separately, as the request held them. */
export interface RequestLayers {
  identity: string;
  slow_context: string;
  turn_context: string;
  tools_count: number;
  /** Whether the request sent a system message at all. */
  system_message: boolean;
}

/** `GET …/trace/steps/{step}`. */
export interface StepDetail {
  step: number;
  request: TraceRequest | null;
  request_reason: string | null;
  verified: boolean | null;
  layers: RequestLayers | null;
  response: ModelResponseRecord | null;
  response_reason: string | null;
  /** Present when the Core answers with a reason instead of a step. */
  reason?: TraceReason | null;
}

/** A read's outcome: the body, or the Core's own message for why there is none. */
export type TraceRead<T> =
  | { status: 'loading' }
  | { status: 'ok'; data: T }
  | { status: 'failed'; message: string };

/**
 * The message a failed answer carries: `detail` as a string, or `detail.message`, which is the
 * shape the trace routes refuse with (`{code, message}`). Anything else is the status line.
 */
export const failureMessage = (status: number, body: unknown): string => {
  if (typeof body === 'object' && body !== null && 'detail' in body) {
    const detail = (body as { detail: unknown }).detail;
    if (typeof detail === 'string' && detail.trim() !== '') return detail;
    if (typeof detail === 'object' && detail !== null && 'message' in detail) {
      const message = (detail as { message: unknown }).message;
      if (typeof message === 'string' && message.trim() !== '') return message;
    }
  }
  return `The runtime answered HTTP ${status} with no explanation.`;
};

/** One `GET` of a trace route. Never throws: a failure is a message to print. */
export async function readTrace<T>(url: string): Promise<TraceRead<T>> {
  let res: Response;
  try {
    res = await fetch(url);
  } catch (err) {
    return {
      status: 'failed',
      message: `The runtime did not answer: ${err instanceof Error ? err.message : String(err)}`,
    };
  }
  let body: unknown = null;
  try {
    body = await res.json();
  } catch {
    if (res.ok) {
      return { status: 'failed', message: 'The runtime answered with something that is not JSON.' };
    }
  }
  if (!res.ok) return { status: 'failed', message: failureMessage(res.status, body) };
  return { status: 'ok', data: body as T };
}

/** Milliseconds between two ISO stamps, or `null` when either is missing or unreadable. */
export const durationMs = (startedAt: string | null, endedAt: string | null): number | null => {
  if (!startedAt || !endedAt) return null;
  const ms = Date.parse(endedAt) - Date.parse(startedAt);
  return Number.isFinite(ms) && ms >= 0 ? ms : null;
};

export const formatDuration = (ms: number): string =>
  ms >= 1000 ? `${(ms / 1000).toFixed(1)}s` : `${Math.round(ms)}ms`;

const count = (usage: Record<string, unknown> | null, key: string): number | null => {
  const value = usage?.[key];
  return typeof value === 'number' ? value : null;
};

/**
 * A step's headline: `step n · model · 1,234 → 210 tokens · 3.7s · finish_reason`.
 *
 * A part nothing recorded is left out rather than drawn as a zero.
 */
export const stepHeadline = (step: TraceStep): string => {
  const response = step.response;
  const parts = [`step ${step.step}`];
  const model = response?.model_name ?? step.model;
  if (model) parts.push(model);
  const input = count(response?.usage ?? null, 'input_tokens');
  const output = count(response?.usage ?? null, 'output_tokens');
  if (input !== null && output !== null) {
    parts.push(`${input.toLocaleString()} → ${output.toLocaleString()} tokens`);
  }
  const ms = response ? durationMs(response.started_at, response.ended_at) : null;
  if (ms !== null) parts.push(formatDuration(ms));
  if (response?.finish_reason) parts.push(response.finish_reason);
  return parts.join(' · ');
};

/** Which layer of the request a block is (llm-request-layering.md §5). */
export type RequestLayer = 'identity' | 'slow_context' | 'conversation' | 'turn_context';

export interface RequestBlock {
  /** The message's index in `request.messages`. */
  index: number;
  role: string;
  layer: RequestLayer;
  content: string | null;
  message: TraceMessage;
}

/**
 * The request's messages as blocks, with the layers named.
 *
 * The system message is identity then slow context (`compose_system_message`), so with the
 * layers it is shown as those two blocks. The turn context is the request's tail
 * (`place_turn_context`): either the last user message whole, or appended to it after a blank
 * line, in which case that message is split in two. Without layers every message is shown as
 * sent, as conversation.
 */
export const requestBlocks = (
  request: TraceRequest,
  layers: RequestLayers | null,
): RequestBlock[] => {
  const blocks: RequestBlock[] = [];
  const messages = request.messages ?? [];
  const turnContext = layers?.turn_context ?? '';
  const last = messages.length - 1;
  messages.forEach((message, index) => {
    const role = message.role;
    if (layers && layers.system_message && index === 0 && role === 'system') {
      blocks.push({ index, role, layer: 'identity', content: layers.identity, message });
      blocks.push({ index, role, layer: 'slow_context', content: layers.slow_context, message });
      return;
    }
    if (turnContext && index === last && role === 'user' && typeof message.content === 'string') {
      if (message.content === turnContext) {
        blocks.push({ index, role, layer: 'turn_context', content: message.content, message });
        return;
      }
      const tail = `\n\n${turnContext}`;
      if (message.content.endsWith(tail)) {
        const head = message.content.slice(0, -tail.length);
        blocks.push({ index, role, layer: 'conversation', content: head, message });
        blocks.push({ index, role, layer: 'turn_context', content: turnContext, message });
        return;
      }
    }
    blocks.push({ index, role, layer: 'conversation', content: message.content, message });
  });
  return blocks;
};

/** What Copy as JSON puts on the clipboard: the step's request and response, together. */
export const stepPairJson = (detail: StepDetail): string =>
  JSON.stringify(
    {
      step: detail.step,
      verified: detail.verified,
      request: detail.request,
      request_reason: detail.request_reason,
      response: detail.response,
      response_reason: detail.response_reason,
    },
    null,
    2,
  );

/** A logged value as text: strings as they are, anything else as indented JSON. */
export const asText = (value: unknown): string =>
  typeof value === 'string' ? value : JSON.stringify(value, null, 2) ?? String(value);
