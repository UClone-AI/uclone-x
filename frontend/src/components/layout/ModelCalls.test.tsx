import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest';
import { fireEvent, render, screen, waitFor, within } from '@testing-library/react';
import { ModelCalls, UNVERIFIED_NOTE } from './ModelCalls';
import type {
  StepDetail,
  TraceStep,
  TurnTrace,
  TurnTraceResponse,
} from '../../lib/turnTrace';

const TRACE_URL = '/api/rooms/room_1/turns/4/trace';
const stepUrl = (n: number): string => `${TRACE_URL}/steps/${n}`;

const IDENTITY = 'You are Scout, a careful research clone.';
const SLOW = '[Workspace]\nThe workspace is /tmp/w.';
const TURN_CONTEXT = '[Turn Context]\nPlan: read the notes.';

const traceStep = (over: Partial<TraceStep> = {}): TraceStep => ({
  step: 1,
  request_status: 'ok',
  request_reason: null,
  verified: true,
  message_count: 2,
  model: 'qwen3:8b',
  temperature: 0.7,
  max_tokens: null,
  response_status: 'ok',
  response_reason: null,
  response: {
    turn_index: 3,
    step: 1,
    started_at: '2026-09-24T10:00:00.000Z',
    ended_at: '2026-09-24T10:00:03.700Z',
    streamed: true,
    content: '',
    thinking: null,
    tool_calls: [{ id: 'call_1', name: 'file_read', arguments: { path: 'notes.md' } }],
    finish_reason: 'tool_calls',
    model_name: 'qwen3:8b',
    usage: { input_tokens: 1234, output_tokens: 210, total_tokens: 1444, count_source: 'provider' },
    error: null,
  },
  tool_results: [],
  ...over,
});

const trace = (steps: TraceStep[], over: Partial<TurnTrace> = {}): TurnTraceResponse => ({
  room_id: 'room_1',
  seq: 4,
  turn_id: 'turn_4',
  participant_id: 'scout',
  session_id: 'sess_scout',
  trace: {
    session_id: 'sess_scout',
    turn_index: 3,
    started_at: '2026-09-24T10:00:00.000Z',
    ended_at: '2026-09-24T10:00:09.000Z',
    steps,
    nudges: [],
    rolled_back: false,
    dropped_tool_steps: [],
    turn_end: {},
    subagents: [],
    ...over,
  },
});

const stepDetail = (over: Partial<StepDetail> = {}): StepDetail => ({
  step: 1,
  request: {
    model: 'qwen3:8b',
    messages: [
      { role: 'system', content: `${IDENTITY}\n\n${SLOW}` },
      { role: 'user', content: `Read my notes.\n\n${TURN_CONTEXT}` },
    ],
    tools: [
      {
        name: 'file_read',
        description: 'Read a file.',
        parameters: { type: 'object', properties: { path: { type: 'string' } } },
      },
    ],
    temperature: 0.7,
    max_tokens: null,
  },
  request_reason: null,
  verified: true,
  layers: {
    identity: IDENTITY,
    slow_context: SLOW,
    turn_context: TURN_CONTEXT,
    tools_count: 1,
    system_message: true,
  },
  response: {
    turn_index: 3,
    step: 1,
    started_at: '2026-09-24T10:00:00.000Z',
    ended_at: '2026-09-24T10:00:03.700Z',
    streamed: true,
    content: 'Let me read them.',
    thinking: 'The user wants the notes.',
    tool_calls: [{ id: 'call_1', name: 'file_read', arguments: { path: 'notes.md' } }],
    finish_reason: 'tool_calls',
    model_name: 'qwen3:8b',
    usage: { input_tokens: 1234, output_tokens: 210 },
    error: null,
  },
  response_reason: null,
  reason: null,
  ...over,
});

let answers: Record<string, { status: number; body: unknown }>;
let fetchMock: ReturnType<typeof vi.fn>;
const requested = (): string[] => fetchMock.mock.calls.map((call) => String(call[0]));

beforeEach(() => {
  answers = {};
  fetchMock = vi.fn(async (url: string) => {
    const answer = answers[url];
    if (!answer) return new Response(JSON.stringify({ detail: `no stub for ${url}` }), { status: 500 });
    return new Response(JSON.stringify(answer.body), { status: answer.status });
  });
  vi.stubGlobal('fetch', fetchMock);
  Object.assign(navigator, {
    clipboard: { writeText: vi.fn().mockImplementation(() => Promise.resolve()) },
  });
});

afterEach(() => {
  vi.unstubAllGlobals();
});

const ok = (body: unknown) => ({ status: 200, body });

const expandSection = async (): Promise<void> => {
  fireEvent.click(screen.getByTestId('model-calls-toggle'));
};

describe('ModelCalls (#1492)', () => {
  // Killed by: frontend/src/components/layout/ModelCalls.tsx :: if (next && read === null) {
  // Becomes: if (next) {
  it('reads nothing until expanded, then reads the trace once however often it is toggled', async () => {
    answers[TRACE_URL] = ok(trace([traceStep()]));
    render(<ModelCalls roomId="room_1" seq={4} />);
    expect(requested()).toEqual([]);

    await expandSection();
    await screen.findByTestId('model-call-row');
    fireEvent.click(screen.getByTestId('model-calls-toggle'));
    fireEvent.click(screen.getByTestId('model-calls-toggle'));
    await screen.findByTestId('model-call-row');
    expect(requested()).toEqual([TRACE_URL]);
  });

  // Killed by: frontend/src/lib/turnTrace.ts :: if (input !== null && output !== null) {
  // Becomes: if (input === null && output !== null) {
  it('draws one row per step with model, tokens, duration, finish reason, verified mark and tool names', async () => {
    answers[TRACE_URL] = ok(
      trace([
        traceStep(),
        traceStep({
          step: 2,
          response: { ...traceStep().response!, step: 2, tool_calls: [], finish_reason: 'stop' },
        }),
      ]),
    );
    render(<ModelCalls roomId="room_1" seq={4} />);
    await expandSection();

    const rows = await screen.findAllByTestId('model-call-row');
    expect(rows).toHaveLength(2);
    const first = within(rows[0]);
    expect(first.getByTestId('model-call-headline')).toHaveTextContent(
      'step 1 · qwen3:8b · 1,234 → 210 tokens · 3.7s · tool_calls',
    );
    expect(first.getByTestId('model-call-verified')).toHaveTextContent('verified');
    expect(first.getByTestId('model-call-tool-names')).toHaveTextContent('file_read');
    expect(within(rows[1]).queryByTestId('model-call-tool-names')).toBeNull();
  });

  // Killed by: frontend/src/components/layout/ModelCalls.tsx :: message={reasonLine(body.reason) ?? 'The Core returned no trace and no reason for it.'}
  // Becomes: message={'The Core returned no trace and no reason for it.'}
  it.each([
    // The codes and messages `room_dock.py` gives as of #1512, and one it does not yet.
    ['not_an_agent_turn', 'This turn was spoken by a person, not an agent.'],
    [
      'turn_not_linked',
      'This turn was recorded before turns were linked to their session (#1489), so its model calls cannot be found.',
    ],
    ['turn_not_saved', "This turn's record was not saved, so its model calls cannot be shown."],
    ['session_not_found', 'No saved session was found for this seat.'],
    [
      'session_unreadable',
      "This seat's saved session could not be read, so the turn cannot be traced.",
    ],
    ['log_missing', 'No session log was found for this seat.'],
    ['log_unreadable', "This seat's session log could not be read, so the turn cannot be traced."],
    [
      'trace_failed',
      "This turn's record could not be read back, so its model calls cannot be shown.",
    ],
    ['a_code_added_later', 'A reason this head has never seen before.'],
  ])('prints the Core’s message for reason %s, whatever the code', async (code, message) => {
    answers[TRACE_URL] = ok({ ...trace([]), trace: null, reason: { code, message } });
    render(<ModelCalls roomId="room_1" seq={4} />);
    await expandSection();
    expect(await screen.findByTestId('model-calls-reason')).toHaveTextContent(message);
    expect(screen.queryByTestId('model-call-row')).toBeNull();
  });

  // Killed by: frontend/src/lib/turnTrace.ts :: return reason.detail ? `${reason.message} (${reason.detail})` : reason.message;
  // Becomes: return reason.message;
  it('shows the stable detail a log_unreadable reason carries after its message', async () => {
    answers[TRACE_URL] = ok({
      ...trace([]),
      trace: null,
      reason: {
        code: 'log_unreadable',
        message: "This seat's session log could not be read, so the turn cannot be traced.",
        detail: 'malformed_log',
      },
    });
    render(<ModelCalls roomId="room_1" seq={4} />);
    await expandSection();
    expect(await screen.findByTestId('model-calls-reason')).toHaveTextContent(
      "This seat's session log could not be read, so the turn cannot be traced. (malformed_log)",
    );
  });

  // Killed by: frontend/src/components/layout/ModelCalls.tsx :: {step.request_status === 'unavailable' ? (
  // Becomes: {step.request_status === 'ok' ? (
  it('states an unavailable half’s reason on its row, and the other steps still render', async () => {
    answers[TRACE_URL] = ok(
      trace([
        traceStep({
          request_status: 'unavailable',
          request_reason: 'request recorded before request capture',
          verified: null,
          model: null,
        }),
        traceStep({
          step: 2,
          response_status: 'unavailable',
          response_reason: 'response recorded from a later version on',
          response: null,
        }),
        traceStep({ step: 3 }),
      ]),
    );
    render(<ModelCalls roomId="room_1" seq={4} />);
    await expandSection();

    const rows = await screen.findAllByTestId('model-call-row');
    expect(rows).toHaveLength(3);
    expect(within(rows[0]).getByTestId('model-call-row-request-reason')).toHaveTextContent(
      'request recorded before request capture',
    );
    expect(within(rows[0]).queryByTestId('model-call-verified')).toBeNull();
    expect(within(rows[1]).getByTestId('model-call-row-response-reason')).toHaveTextContent(
      'response recorded from a later version on',
    );
    expect(within(rows[1]).getByTestId('model-call-headline')).toHaveTextContent('step 2 · qwen3:8b');
    expect(within(rows[2]).getByTestId('model-call-headline')).toHaveTextContent('step 3');
    expect(within(rows[2]).queryByTestId('model-call-row-request-reason')).toBeNull();
  });

  // Killed by: frontend/src/components/layout/ModelCalls.tsx :: if (next && detail === null) {
  // Becomes: if (false) {
  it('fetches a step’s detail on expanding it, and shows the four tabs', async () => {
    answers[TRACE_URL] = ok(trace([traceStep()]));
    answers[stepUrl(1)] = ok(stepDetail());
    render(<ModelCalls roomId="room_1" seq={4} />);
    await expandSection();
    await screen.findByTestId('model-call-row');
    expect(requested()).toEqual([TRACE_URL]);

    fireEvent.click(screen.getByTestId('model-call-toggle-1'));
    await screen.findByTestId('model-call-detail');
    expect(requested()).toEqual([TRACE_URL, stepUrl(1)]);

    // Request: role-labelled blocks, the system message split into its layers, the turn
    // context marked and cut from the user's words.
    const blocks = screen.getAllByTestId('request-block');
    expect(blocks.map((b) => `${b.dataset.role}/${b.dataset.layer}`)).toEqual([
      'system/identity',
      'system/slow_context',
      'user/conversation',
      'user/turn_context',
    ]);
    expect(blocks[0]).toHaveTextContent(IDENTITY);
    expect(blocks[1]).toHaveTextContent('The workspace is /tmp/w.');
    expect(blocks[2]).toHaveTextContent('Read my notes.');
    expect(blocks[2]).not.toHaveTextContent('Plan: read the notes.');
    expect(blocks[3]).toHaveTextContent('Plan: read the notes.');

    fireEvent.click(screen.getByTestId('model-call-tab-tools'));
    const tool = screen.getByTestId('offered-tool');
    expect(tool).toHaveTextContent('file_read');
    expect(screen.queryByTestId('offered-tool-schema')).toBeNull();
    fireEvent.click(within(tool).getByRole('button'));
    expect(screen.getByTestId('offered-tool-schema')).toHaveTextContent('"path"');

    fireEvent.click(screen.getByTestId('model-call-tab-response'));
    expect(screen.getByTestId('model-call-response-content')).toHaveTextContent('Let me read them.');
    expect(screen.getByTestId('model-call-response-thinking')).toHaveTextContent(
      'The user wants the notes.',
    );
    expect(screen.getByTestId('model-call-response-tool-call')).toHaveTextContent('notes.md');

    fireEvent.click(screen.getByTestId('model-call-tab-raw'));
    expect(JSON.parse(screen.getByTestId('model-call-raw').textContent ?? '')).toEqual(stepDetail());

    // Collapsing and expanding again does not read the step again.
    fireEvent.click(screen.getByTestId('model-call-toggle-1'));
    fireEvent.click(screen.getByTestId('model-call-toggle-1'));
    expect(requested()).toEqual([TRACE_URL, stepUrl(1)]);
  });

  // Killed by: frontend/src/components/layout/ModelCalls.tsx :: await navigator.clipboard.writeText(stepPairJson(detail));
  // Becomes: await navigator.clipboard.writeText(String(detail.request));
  it('copies the request and response pair as valid JSON', async () => {
    answers[TRACE_URL] = ok(trace([traceStep()]));
    answers[stepUrl(1)] = ok(stepDetail());
    render(<ModelCalls roomId="room_1" seq={4} />);
    await expandSection();
    fireEvent.click(await screen.findByTestId('model-call-toggle-1'));
    fireEvent.click(await screen.findByTestId('model-call-copy'));

    await waitFor(() => expect(navigator.clipboard.writeText).toHaveBeenCalledTimes(1));
    const copied = vi.mocked(navigator.clipboard.writeText).mock.calls[0][0];
    const pair = JSON.parse(copied);
    expect(pair.request).toEqual(stepDetail().request);
    expect(pair.response).toEqual(stepDetail().response);
    expect(await screen.findByTestId('model-call-copy')).toHaveTextContent('Copied');
  });

  // Killed by: frontend/src/components/layout/ModelCalls.tsx :: {detail.verified === false ? (
  // Becomes: {detail.verified === true ? (
  it('says a rebuilt request may differ from what was sent when it is not verified', async () => {
    answers[TRACE_URL] = ok(trace([traceStep({ verified: false })]));
    answers[stepUrl(1)] = ok(stepDetail({ verified: false }));
    render(<ModelCalls roomId="room_1" seq={4} />);
    await expandSection();
    expect(await screen.findByTestId('model-call-verified')).toHaveTextContent('not verified');
    fireEvent.click(screen.getByTestId('model-call-toggle-1'));
    expect(await screen.findByTestId('model-call-unverified')).toHaveTextContent(UNVERIFIED_NOTE);
  });

  it('shows no unverified wording on a verified step', async () => {
    answers[TRACE_URL] = ok(trace([traceStep()]));
    answers[stepUrl(1)] = ok(stepDetail());
    render(<ModelCalls roomId="room_1" seq={4} />);
    await expandSection();
    fireEvent.click(await screen.findByTestId('model-call-toggle-1'));
    await screen.findByTestId('model-call-detail');
    expect(screen.queryByTestId('model-call-unverified')).toBeNull();
  });

  // Killed by: frontend/src/components/layout/ModelCalls.tsx :: {detail.response_reason ?? 'The Core gave no response and no reason for its absence.'}
  // Becomes: {'The Core gave no response and no reason for its absence.'}
  it('states the step detail’s own reason when its request or response is absent', async () => {
    answers[TRACE_URL] = ok(trace([traceStep({ request_status: 'unavailable', verified: null })]));
    answers[stepUrl(1)] = ok(
      stepDetail({
        request: null,
        layers: null,
        verified: null,
        request_reason: 'request recorded before request capture',
        response: null,
        response_reason: 'response recorded from a later version on',
      }),
    );
    render(<ModelCalls roomId="room_1" seq={4} />);
    await expandSection();
    fireEvent.click(await screen.findByTestId('model-call-toggle-1'));
    expect(await screen.findByTestId('model-call-request-reason')).toHaveTextContent(
      'request recorded before request capture',
    );
    fireEvent.click(screen.getByTestId('model-call-tab-response'));
    expect(screen.getByTestId('model-call-response-reason')).toHaveTextContent(
      'response recorded from a later version on',
    );
  });

  // Killed by: frontend/src/lib/turnTrace.ts :: if (typeof message === 'string' && message.trim() !== '') return message;
  // Becomes: if (false) return message;
  // Killed by: frontend/src/components/layout/ModelCalls.tsx :: const stepReason = reasonLine(body?.reason);
  // Becomes: const stepReason = null;
  it('prints the Core’s message when a step read is refused, in either shape', async () => {
    answers[TRACE_URL] = ok(trace([traceStep(), traceStep({ step: 2 })]));
    answers[stepUrl(1)] = {
      status: 404,
      body: { detail: { code: 'step_not_found', message: 'Step 1 was not found for turn 4.' } },
    };
    // The route's own shape for a turn it cannot read: every field empty, and the reason.
    answers[stepUrl(2)] = ok({
      step: 2,
      request: null,
      request_reason: null,
      verified: null,
      layers: null,
      response: null,
      response_reason: null,
      reason: {
        code: 'log_unreadable',
        message: "This seat's session log could not be read, so the turn cannot be traced.",
        detail: 'not_text',
      },
    });
    render(<ModelCalls roomId="room_1" seq={4} />);
    await expandSection();
    fireEvent.click(await screen.findByTestId('model-call-toggle-1'));
    fireEvent.click(screen.getByTestId('model-call-toggle-2'));
    const failures = await screen.findAllByTestId('model-call-detail-failed');
    expect(failures.map((f) => f.textContent)).toEqual([
      'Step 1 was not found for turn 4.',
      "This seat's session log could not be read, so the turn cannot be traced. (not_text)",
    ]);
  });

  // Killed by: frontend/src/components/layout/ModelCalls.tsx :: : asText(result.output)}
  // Becomes: : asText(result.output).slice(0, 2000)}
  it('shows each tool result in full, not the room’s 2,000-character preview, with its raw error', async () => {
    const long = `${'x'.repeat(2500)}END`;
    answers[TRACE_URL] = ok(
      trace([
        traceStep({
          tool_results: [
            {
              tool_call_id: 'call_1',
              name: 'file_read',
              status: 'success',
              outcome: 'answered',
              output: long,
              duration_ms: 40,
              at: null,
            },
            {
              tool_call_id: 'call_2',
              name: 'run_command',
              status: 'error',
              outcome: null,
              output: 'Traceback (most recent call last):\nPermissionError: /etc/shadow',
              duration_ms: null,
              at: null,
            },
          ],
        }),
      ]),
    );
    answers[stepUrl(1)] = ok(stepDetail());
    render(<ModelCalls roomId="room_1" seq={4} />);
    await expandSection();
    fireEvent.click(await screen.findByTestId('model-call-toggle-1'));
    const results = await screen.findAllByTestId('model-call-tool-result');
    expect(results[0].textContent).toContain(long);
    expect(results[1]).toHaveTextContent('PermissionError: /etc/shadow');
    expect(results[1]).toHaveTextContent('error');
  });

  // Killed by: frontend/src/components/layout/ModelCalls.tsx :: {trace.subagents_reason}
  // Becomes: {null}
  it('states why a helper named in a tool result could not be read', async () => {
    answers[TRACE_URL] = ok(
      trace([traceStep()], {
        subagents_reason: 'A tool result names a helper, but its record could not be read.',
      }),
    );
    render(<ModelCalls roomId="room_1" seq={4} />);
    await expandSection();
    expect(await screen.findByTestId('model-calls-subagents-reason')).toHaveTextContent(
      'A tool result names a helper, but its record could not be read.',
    );
  });

  it('says so when the log lists no model calls for the turn, rather than drawing an empty list', async () => {
    answers[TRACE_URL] = ok(trace([]));
    render(<ModelCalls roomId="room_1" seq={4} />);
    await expandSection();
    expect(await screen.findByTestId('model-calls-no-steps')).toBeInTheDocument();
  });

  it('prints a transport failure instead of an empty section', async () => {
    fetchMock.mockImplementationOnce(async () => {
      throw new TypeError('Failed to fetch');
    });
    render(<ModelCalls roomId="room_1" seq={4} />);
    await expandSection();
    expect(await screen.findByTestId('model-calls-failed')).toHaveTextContent('Failed to fetch');
  });
});
