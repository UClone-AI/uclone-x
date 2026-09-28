import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest';
import { fireEvent, render, screen, waitFor, within } from '@testing-library/react';
import { ModelCalls } from './ModelCalls';
import en from '../../i18n/locales/en/dock.json';
import ko from '../../i18n/locales/ko/dock.json';
import { LocaleProvider } from '../../i18n';
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

  // Killed by: frontend/src/components/layout/ModelCalls.tsx :: return reason ? wordProblem(t, reason, { seq }) : { sentence: t.noTrace };
  // Becomes: return { sentence: t.noTrace };
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
  ])('words reason %s as the Core does, and prints the message of a code it does not know', async (code, message) => {
    answers[TRACE_URL] = ok({ ...trace([]), trace: null, reason: { code, message } });
    render(<ModelCalls roomId="room_1" seq={4} />);
    await expandSection();
    expect(await screen.findByTestId('model-calls-reason')).toHaveTextContent(message);
    expect(screen.queryByTestId('model-call-row')).toBeNull();
  });

  // Killed by: frontend/src/components/layout/ModelCalls.tsx :: {worded.detail ? <Detail>({worded.detail})</Detail> : null}
  // Becomes: {null}
  it('shows the stable detail a log_unreadable reason carries after its message, for a kind it does not know', async () => {
    answers[TRACE_URL] = ok({
      ...trace([]),
      trace: null,
      reason: {
        code: 'log_unreadable',
        message: "This seat's session log could not be read, so the turn cannot be traced.",
        detail: 'a_kind_added_later',
      },
    });
    render(<ModelCalls roomId="room_1" seq={4} />);
    await expandSection();
    expect(await screen.findByTestId('model-calls-reason')).toHaveTextContent(
      "This seat's session log could not be read, so the turn cannot be traced. (a_kind_added_later)",
    );
  });

  // Killed by: frontend/src/components/layout/ModelCalls.tsx :: const kind = problem.code === 'log_unreadable' ? known(t.logFailures, problem.detail ?? null) : null;
  // Becomes: const kind = null;
  it('words the kind of log failure a log_unreadable reason carries, in Korean (#1907)', async () => {
    answers[TRACE_URL] = ok({
      ...trace([]),
      trace: null,
      reason: {
        code: 'log_unreadable',
        message: "This seat's session log could not be read, so the turn cannot be traced.",
        detail: 'malformed_log',
      },
    });
    render(
      <LocaleProvider hints={['ko-KR']}>
        <ModelCalls roomId="room_1" seq={4} />
      </LocaleProvider>,
    );
    await expandSection();
    const shown = await screen.findByTestId('model-calls-reason');
    expect(shown.textContent).toBe(
      `${ko.modelCalls.reasons.log_unreadable} ${ko.modelCalls.logFailures.malformed_log}`,
    );
    expect(within(shown).queryByTestId('model-call-core-detail')).toBeNull();
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
    expect(await screen.findByTestId('model-call-unverified')).toHaveTextContent(en.modelCalls.unverified);
  });

  it.each([
    [true, null, 'matches', 'The conversation this call sent matches the session log.'],
    [false, null, 'differs', 'The conversation this call sent does not match what the session log rebuilds.'],
    [
      null,
      'no context state is recorded for this session',
      'unchecked',
      "Not checked against the session log. The runtime's reason: no context state is recorded for this session",
    ],
  ] as const)(
    'says whether the step’s conversation is what the session log rebuilds (from_log %s), or why it was not checked',
    async (from_log, from_log_reason, status, line) => {
      answers[TRACE_URL] = ok(trace([traceStep()]));
      answers[stepUrl(1)] = ok(stepDetail({ from_log, from_log_reason }));
      render(<ModelCalls roomId="room_1" seq={4} />);
      await expandSection();
      fireEvent.click(await screen.findByTestId('model-call-toggle-1'));
      const shown = await screen.findByTestId('model-call-from-log');
      expect(shown).toHaveAttribute('data-status', status);
      expect(shown.textContent).toBe(line);
    },
  );

  it('shows a differing conversation and an unchecked one on the step, each in its own words', async () => {
    // Killed by: frontend/src/lib/turnTrace.ts ::   if (detail.from_log === false) return { status: 'differs' };
    // Becomes:   if (detail.from_log === null) return { status: 'differs' };
    // Killed by: frontend/src/lib/turnTrace.ts ::   if (code !== null || reason !== null) return { status: 'unchecked', code, detailCode, reason };
    // Becomes:   if (false) return { status: 'unchecked', code, detailCode, reason };
    answers[TRACE_URL] = ok(trace([traceStep()]));
    answers[stepUrl(1)] = ok(stepDetail({ from_log: false, from_log_reason: null }));
    const { unmount } = render(<ModelCalls roomId="room_1" seq={4} />);
    await expandSection();
    fireEvent.click(await screen.findByTestId('model-call-toggle-1'));
    expect(await screen.findByTestId('model-call-from-log')).toHaveAttribute('data-status', 'differs');
    unmount();
    answers[stepUrl(1)] = ok(stepDetail({ from_log: null, from_log_reason: 'no epoch for this request' }));
    render(<ModelCalls roomId="room_1" seq={4} />);
    await expandSection();
    fireEvent.click(await screen.findByTestId('model-call-toggle-1'));
    expect((await screen.findByTestId('model-call-from-log')).textContent).toBe(
      "Not checked against the session log. The runtime's reason: no epoch for this request",
    );
  });

  it('says nothing about the session log when the Core gives no check and no reason', async () => {
    answers[TRACE_URL] = ok(trace([traceStep()]));
    answers[stepUrl(1)] = ok(stepDetail({ from_log: null, from_log_reason: null }));
    render(<ModelCalls roomId="room_1" seq={4} />);
    await expandSection();
    fireEvent.click(await screen.findByTestId('model-call-toggle-1'));
    await screen.findByTestId('model-call-detail');
    expect(screen.queryByTestId('model-call-from-log')).toBeNull();
  });

  it.each([
    ['no_context_state', 'no context state is recorded for this session'],
    ['no_epoch_for_request', 'no epoch of the context state was opened at or before this request'],
  ])('words from_log_code %s from the catalog, without the Core’s English', async (code, reason) => {
    // Killed by: frontend/src/components/layout/ModelCalls.tsx :: const sentence = known(t.reasons, check.code);
    // Becomes: const sentence = null;
    answers[TRACE_URL] = ok(trace([traceStep()]));
    answers[stepUrl(1)] = ok(stepDetail({ from_log: null, from_log_reason: reason, from_log_code: code }));
    render(
      <LocaleProvider hints={['ko-KR']}>
        <ModelCalls roomId="room_1" seq={4} />
      </LocaleProvider>,
    );
    await expandSection();
    fireEvent.click(await screen.findByTestId('model-call-toggle-1'));
    const shown = await screen.findByTestId('model-call-from-log');
    expect(shown).toHaveAttribute('data-code', code);
    expect(shown.textContent).toBe(
      ko.modelCalls.fromLog.reasons[code as keyof typeof ko.modelCalls.fromLog.reasons],
    );
  });

  // Killed by: frontend/src/components/layout/ModelCalls.tsx :: const showReason = check.reason !== null && (sentence === null || check.code === 'epoch_unreadable');
  // Becomes: const showReason = check.reason !== null && sentence === null;
  it('sets the Core’s English apart from the Korean sentence: a named log entry, and an unknown code', async () => {
    answers[TRACE_URL] = ok(trace([traceStep()]));
    answers[stepUrl(1)] = ok(
      stepDetail({ from_log: null, from_log_reason: 'no log entry e7', from_log_code: 'epoch_unreadable' }),
    );
    const { unmount } = render(
      <LocaleProvider hints={['ko-KR']}>
        <ModelCalls roomId="room_1" seq={4} />
      </LocaleProvider>,
    );
    await expandSection();
    fireEvent.click(await screen.findByTestId('model-call-toggle-1'));
    let shown = await screen.findByTestId('model-call-from-log');
    expect(shown.textContent).toBe(`${ko.modelCalls.fromLog.reasons.epoch_unreadable} no log entry e7`);
    expect(within(shown).getByTestId('model-call-core-detail')).toHaveTextContent('no log entry e7');
    unmount();
    // A Core older than the code, or a code this head does not know: the generic sentence.
    answers[stepUrl(1)] = ok(
      stepDetail({ from_log: null, from_log_reason: 'no log entry e7', from_log_code: 'a_code_added_later' }),
    );
    render(
      <LocaleProvider hints={['ko-KR']}>
        <ModelCalls roomId="room_1" seq={4} />
      </LocaleProvider>,
    );
    await expandSection();
    fireEvent.click(await screen.findByTestId('model-call-toggle-1'));
    shown = await screen.findByTestId('model-call-from-log');
    expect(shown.textContent).toBe(`${ko.modelCalls.fromLog.unchecked} no log entry e7`);
    expect(within(shown).getByTestId('model-call-core-detail')).toHaveTextContent('no log entry e7');
  });

  // Killed by: frontend/src/components/layout/ModelCalls.tsx :: const worded = fmt(sentence, { seq: where.seq, step: where.step ?? '' });
  // Becomes: const worded = problem.message;
  it('words the Core’s reasons and failures in Korean, with no English inside a sentence', async () => {
    answers[TRACE_URL] = ok(trace([traceStep(), traceStep({ step: 2 })]));
    answers[stepUrl(1)] = {
      status: 404,
      body: { detail: { code: 'step_not_found', message: 'Step 1 was not found for turn 4.' } },
    };
    answers[stepUrl(2)] = { status: 502, body: {} };
    render(
      <LocaleProvider hints={['ko-KR']}>
        <ModelCalls roomId="room_1" seq={4} />
      </LocaleProvider>,
    );
    await expandSection();
    fireEvent.click(await screen.findByTestId('model-call-toggle-1'));
    fireEvent.click(screen.getByTestId('model-call-toggle-2'));
    const failures = await screen.findAllByTestId('model-call-detail-failed');
    expect(failures.map((f) => f.textContent)).toEqual([
      '4번 턴에서 1단계를 찾지 못했습니다.',
      '런타임이 설명 없이 HTTP 502 응답을 보냈습니다.',
    ]);
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

  // Killed by: frontend/src/components/layout/ModelCalls.tsx :: ) : why.detail ? (
  // Becomes: ) : false ? (
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

  // Killed by: frontend/src/components/layout/ModelCalls.tsx :: const requestWhy = codedReason(t.requestReasons, step.request_code, step.request_reason);
  // Becomes: const requestWhy = { sentence: null, detail: step.request_reason };
  // Killed by: frontend/src/components/layout/ModelCalls.tsx :: const responseWhy = codedReason(t.responseReasons, step.response_code, step.response_reason);
  // Becomes: const responseWhy = { sentence: null, detail: step.response_reason };
  // Killed by: frontend/src/components/layout/ModelCalls.tsx :: return { sentence, detail: reason && !SAID_IN_FULL.has(code as string) ? reason : null };
  // Becomes: return { sentence, detail: reason };
  it('words a row’s unavailable halves from the Core’s codes, in Korean (#1907)', async () => {
    answers[TRACE_URL] = ok(
      trace([
        traceStep({
          request_status: 'unavailable',
          request_reason: 'request 2 extends request 1, last seen 0',
          request_code: 'chain_broken',
          verified: null,
          model: null,
          response_status: 'unavailable',
          response_reason: 'no response is recorded for this step',
          response_code: 'not_recorded',
          response: null,
        }),
      ]),
    );
    render(
      <LocaleProvider hints={['ko-KR']}>
        <ModelCalls roomId="room_1" seq={4} />
      </LocaleProvider>,
    );
    await expandSection();
    // A code that names something the sentence does not: the Core's text follows, set apart.
    const request = await screen.findByTestId('model-call-row-request-reason');
    expect(request.textContent).toBe(
      `${ko.modelCalls.requestReasons.chain_broken} request 2 extends request 1, last seen 0`,
    );
    expect(within(request).getByTestId('model-call-core-detail')).toHaveTextContent('last seen 0');
    // A code whose text says nothing more: the sentence alone, no English.
    expect(screen.getByTestId('model-call-row-response-reason').textContent).toBe(
      ko.modelCalls.responseReasons.not_recorded,
    );
  });

  it('keeps the Core’s reason on a row whose code this head does not know, or a Core without codes', async () => {
    answers[TRACE_URL] = ok(
      trace([
        traceStep({
          request_status: 'unavailable',
          request_reason: 'a reason from a later Core',
          request_code: 'a_code_added_later',
          verified: null,
          model: null,
        }),
      ]),
    );
    render(<ModelCalls roomId="room_1" seq={4} />);
    await expandSection();
    expect((await screen.findByTestId('model-call-row-request-reason')).textContent).toBe(
      `${en.modelCalls.rowRequest} a reason from a later Core`,
    );
  });

  // Killed by: frontend/src/components/layout/ModelCalls.tsx :: why={codedReason(t.responseReasons, detail.response_code, detail.response_reason)}
  // Becomes: why={{ sentence: null, detail: detail.response_reason }}
  it('words a step’s missing request and response from the Core’s codes, in Korean (#1907)', async () => {
    answers[TRACE_URL] = ok(trace([traceStep({ request_status: 'unavailable', verified: null })]));
    answers[stepUrl(1)] = ok(
      stepDetail({
        request: null,
        layers: null,
        verified: null,
        request_reason: 'request recorded before request capture (#1421)',
        request_code: 'before_capture',
        response: null,
        response_reason: 'no response is recorded for this step',
        response_code: 'not_recorded',
      }),
    );
    render(
      <LocaleProvider hints={['ko-KR']}>
        <ModelCalls roomId="room_1" seq={4} />
      </LocaleProvider>,
    );
    await expandSection();
    fireEvent.click(await screen.findByTestId('model-call-toggle-1'));
    expect((await screen.findByTestId('model-call-request-reason')).textContent).toBe(
      ko.modelCalls.requestReasons.before_capture,
    );
    fireEvent.click(screen.getByTestId('model-call-tab-response'));
    expect(screen.getByTestId('model-call-response-reason').textContent).toBe(
      ko.modelCalls.responseReasons.not_recorded,
    );
  });

  // Killed by: frontend/src/lib/turnTrace.ts :: return { kind: 'message', status, message: detail };
  // Becomes: return { kind: 'http', status };
  // Killed by: frontend/src/components/layout/ModelCalls.tsx :: return { sentence: fmt(t.failure.refused, { status: problem.status }), detail: problem.message };
  // Becomes: return { sentence: problem.message };
  it('puts a refusal the Core sent as plain text inside a Korean sentence, its text beside it (#1907)', async () => {
    answers[TRACE_URL] = ok(trace([traceStep()]));
    answers[stepUrl(1)] = { status: 403, body: { detail: 'Turn 4 belongs to another room.' } };
    render(
      <LocaleProvider hints={['ko-KR']}>
        <ModelCalls roomId="room_1" seq={4} />
      </LocaleProvider>,
    );
    await expandSection();
    fireEvent.click(await screen.findByTestId('model-call-toggle-1'));
    const failed = await screen.findByTestId('model-call-detail-failed');
    expect(failed.textContent).toBe(
      `${ko.modelCalls.failure.refused.replace('{status}', '403')} (Turn 4 belongs to another room.)`,
    );
    expect(within(failed).getByTestId('model-call-core-detail')).toHaveTextContent(
      'Turn 4 belongs to another room.',
    );
  });

  // Killed by: frontend/src/components/layout/ModelCalls.tsx :: const why = check.code === 'epoch_unreadable' ? known(t.details, check.detailCode) : null;
  // Becomes: const why = null;
  // Killed by: frontend/src/lib/turnTrace.ts ::       ? detail.from_log_detail_code
  // Becomes:       ? null
  it('words why an epoch could not be rebuilt from its own code, the named entry beside it (#1911)', async () => {
    answers[TRACE_URL] = ok(trace([traceStep()]));
    answers[stepUrl(1)] = ok(
      stepDetail({
        from_log: null,
        from_log_reason: 'no log entry e7',
        from_log_code: 'epoch_unreadable',
        from_log_detail_code: 'log_entry_missing',
      }),
    );
    const { unmount } = render(
      <LocaleProvider hints={['ko-KR']}>
        <ModelCalls roomId="room_1" seq={4} />
      </LocaleProvider>,
    );
    await expandSection();
    fireEvent.click(await screen.findByTestId('model-call-toggle-1'));
    let shown = await screen.findByTestId('model-call-from-log');
    expect(shown.textContent).toBe(
      `${ko.modelCalls.fromLog.reasons.epoch_unreadable} ${ko.modelCalls.fromLog.details.log_entry_missing} no log entry e7`,
    );
    unmount();
    // A detail code this head does not know: the reason's sentence alone, the English beside it.
    answers[stepUrl(1)] = ok(
      stepDetail({
        from_log: null,
        from_log_reason: 'no log entry e7',
        from_log_code: 'epoch_unreadable',
        from_log_detail_code: 'a_code_added_later',
      }),
    );
    render(
      <LocaleProvider hints={['ko-KR']}>
        <ModelCalls roomId="room_1" seq={4} />
      </LocaleProvider>,
    );
    await expandSection();
    fireEvent.click(await screen.findByTestId('model-call-toggle-1'));
    shown = await screen.findByTestId('model-call-from-log');
    expect(shown.textContent).toBe(`${ko.modelCalls.fromLog.reasons.epoch_unreadable} no log entry e7`);
  });

  // Killed by: frontend/src/components/layout/ModelCalls.tsx ::         {unknown}
  // Becomes:         {null}
  it('puts a Korean sentence before the English reason of an older Core, never the English alone (#1911)', async () => {
    answers[TRACE_URL] = ok(trace([traceStep({ request_status: 'unavailable', verified: null })]));
    answers[stepUrl(1)] = ok(
      stepDetail({
        request: null,
        layers: null,
        verified: null,
        request_reason: 'request recorded before request capture (#1421)',
        request_code: null,
        response: null,
        response_reason: 'no response is recorded for this step',
        response_code: null,
      }),
    );
    render(
      <LocaleProvider hints={['ko-KR']}>
        <ModelCalls roomId="room_1" seq={4} />
      </LocaleProvider>,
    );
    await expandSection();
    fireEvent.click(await screen.findByTestId('model-call-toggle-1'));
    const request = await screen.findByTestId('model-call-request-reason');
    expect(request.textContent).toBe(
      `${ko.modelCalls.missingRequest} request recorded before request capture (#1421)`,
    );
    expect(within(request).getByTestId('model-call-core-detail')).toHaveTextContent('before request capture');
    fireEvent.click(screen.getByTestId('model-call-tab-response'));
    const response = screen.getByTestId('model-call-response-reason');
    expect(response.textContent).toBe(`${ko.modelCalls.missingResponse} no response is recorded for this step`);
    expect(within(response).getByTestId('model-call-core-detail')).toHaveTextContent('no response is recorded');
  });

  // Killed by: frontend/src/components/layout/ModelCalls.tsx :: const lead = sentence ?? (check.reason === null ? t.uncheckedNoReason : t.unchecked);
  // Becomes: const lead = sentence ?? t.unchecked;
  it('leaves no dangling "reason:" when an unchecked step comes with no reason (#1907)', async () => {
    answers[TRACE_URL] = ok(trace([traceStep()]));
    answers[stepUrl(1)] = ok(
      stepDetail({ from_log: null, from_log_reason: null, from_log_code: 'a_code_added_later' }),
    );
    render(<ModelCalls roomId="room_1" seq={4} />);
    await expandSection();
    fireEvent.click(await screen.findByTestId('model-call-toggle-1'));
    expect((await screen.findByTestId('model-call-from-log')).textContent).toBe(
      en.modelCalls.fromLog.uncheckedNoReason,
    );
  });

  // Killed by: frontend/src/lib/turnTrace.ts :: return { kind: 'reason', code: typeof code === 'string' ? code : '', message };
  // Becomes: return { kind: 'http', status };
  // Killed by: frontend/src/components/layout/ModelCalls.tsx :: const stepReason = reasonProblem(body?.reason);
  // Becomes: const stepReason = null;
  it('words a refused step read, in either shape', async () => {
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
      `This seat's session log could not be read, so the turn cannot be traced. ${en.modelCalls.logFailures.not_text}`,
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

  // Killed by: frontend/src/components/layout/ModelCalls.tsx :: {trace.subagents_reason ? (
  // Becomes: {false ? (
  it('states why a helper named in a tool result could not be read', async () => {
    answers[TRACE_URL] = ok(
      trace([traceStep()], {
        subagents_reason: 'A tool result names a helper, but its record could not be read.',
      }),
    );
    render(<ModelCalls roomId="room_1" seq={4} />);
    await expandSection();
    expect(await screen.findByTestId('model-calls-subagents-reason')).toHaveTextContent(
      en.modelCalls.helpersUnreadable,
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
    const failed = await screen.findByTestId('model-calls-failed');
    expect(failed).toHaveTextContent(en.modelCalls.failure.unreachable);
    expect(within(failed).getByTestId('model-call-core-detail')).toHaveTextContent('Failed to fetch');
  });
});
