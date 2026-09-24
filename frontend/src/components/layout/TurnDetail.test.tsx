import { describe, expect, it, vi } from 'vitest';
import { fireEvent, render, screen, waitFor } from '@testing-library/react';
import { TurnDetail } from './TurnDetail';
import type { EventEnvelope, RoomState, RoomTranscriptMessage } from '../../types';
import { OpenInDocsContext, type TurnSummary } from '../../lib/roomDock';

const message = (over: Partial<RoomTranscriptMessage> = {}): RoomTranscriptMessage => ({
  seq: 1,
  sender_id: 'scout',
  content: 'an answer',
  kind: 'utterance',
  created_at: '2026-01-01T00:00:00Z',
  completed: true,
  turn_id: 'turn_101',
  ...over,
});

const room = (transcript: RoomTranscriptMessage[]): RoomState => ({
  room_id: 'room_1',
  title: 'A conversation',
  participants: [
    { id: 'user', kind: 'human', display_name: 'Kenny' },
    { id: 'scout', kind: 'agent', display_name: 'Scout' },
  ],
  transcript,
  turn_state: { agent_turns_since_human: 0 },
  policy: {
    max_agent_turns_per_human_message: 4,
    max_span_messages: 40,
    transcript_window: 40,
    hesitation_seconds: 0,
    default_responder_id: '',
  },
});

const defaultSummary = (over: Partial<TurnSummary> = {}): TurnSummary => ({
  seq: 1,
  turn_id: 'turn_101',
  sender_id: 'scout',
  created_at: '2026-01-01T00:00:00Z',
  completed: true,
  error: null,
  refusal: null,
  provenance: {
    path: 'primary',
    requested: { provider: 'ollama', model: 'qwen3:8b' },
    served_by: { provider: 'ollama', model: 'qwen3:8b' },
    degraded: false,
  },
  decision: {
    verdict: 'speak',
    speaker_id: 'scout',
    confidence: 1.0,
    selector: 'sole_agent',
    reasoning: '',
  },
  rendered_through: '0',
  usage: null,
  notable: [],
  steps: [],
  documents: [],
  unnamed_writes: 0,
  subagent_steps: [],
  steps_absent_reason: null,
  ...over,
});

/**
 * The record `why ›` opens, redesigned for general users per #1491 §4.3.
 */
describe('TurnDetail', () => {
  // Killed by: frontend/src/components/layout/TurnDetail.tsx :: const served = serviceRefLabel(provenance?.served_by);
  // Becomes: const served = provenance?.served_by ? `${provenance.served_by.provider}` : null;
  it('names the model the turn was served by, not the one its words claim', () => {
    render(
      <TurnDetail
        seq={1}
        room={room([
          message({
            content: 'I am OpenAI GPT-4.',
            provenance: {
              path: 'primary',
              requested: { provider: 'ollama', model: 'qwen3:8b' },
              served_by: { provider: 'ollama', model: 'qwen3:8b' },
              degraded: false,
            },
          }),
        ])}
      />,
    );

    expect(screen.getByTestId('turn-detail-served')).toHaveTextContent('ollama:qwen3:8b');
  });

  // Killed by: frontend/src/components/layout/TurnDetail.tsx :: This turn reported no model, so none is named here.
  // Becomes: {''}
  it('says a turn reported no model, and why none is named, rather than showing a blank', () => {
    render(
      <TurnDetail
        seq={1}
        room={room([
          message({
            content: 'that did not work',
            provenance: null,
            decision: {
              verdict: 'speak',
              speaker_id: 'scout',
              confidence: 0.5,
              selector: 'SoleAgentSelector',
              reasoning: '',
            },
          }),
        ])}
      />,
    );

    expect(screen.getByTestId('turn-detail-no-model')).toHaveTextContent(
      'This turn reported no model, so none is named here.',
    );
  });

  // Killed by: frontend/src/components/layout/TurnDetail.tsx :: const requested = serviceRefLabel(provenance?.requested);
  // Becomes: const requested = null;
  it('keeps what was asked for reachable on a degraded turn', () => {
    render(
      <TurnDetail
        seq={2}
        room={room([
          message({ seq: 1, content: 'one' }),
          message({
            seq: 2,
            content: 'two',
            provenance: {
              path: 'failover',
              requested: { provider: 'ollama', model: 'hermes3:8b' },
              served_by: { provider: 'ollama', model: 'qwen3:8b' },
              degraded: true,
            },
          }),
        ])}
      />,
    );

    expect(screen.getByTestId('turn-detail-served')).toHaveTextContent('ollama:qwen3:8b');
    expect(screen.getByTestId('turn-detail-requested')).toHaveTextContent('ollama:hermes3:8b');
    expect(screen.getByTestId('turn-detail-path')).toHaveTextContent('failover');
  });

  // Killed by: frontend/src/components/layout/TurnDetail.tsx :: {attempts.length > 0 ? (
  // Becomes: {false ? (
  it('shows the attempts that did not answer, which nothing rendered before', () => {
    render(
      <TurnDetail
        seq={1}
        room={room([
          message({
            provenance: {
              path: 'failover',
              requested: { provider: 'ollama', model: 'hermes3:8b' },
              served_by: { provider: 'ollama', model: 'qwen3:8b' },
              degraded: true,
              attempts: [
                {
                  provider: 'ollama',
                  model: 'hermes3:8b',
                  error_class: 'ConnectionError',
                  status_code: 503,
                },
              ],
            },
          }),
        ])}
      />,
    );

    const attempts = screen.getByTestId('turn-detail-attempts');
    expect(attempts).toHaveTextContent('ollama:hermes3:8b');
    expect(attempts).toHaveTextContent('ConnectionError');
    expect(attempts).toHaveTextContent('503');
  });

  // Killed by: frontend/src/components/layout/TurnDetail.tsx :: Nobody chose this speaker: the turn records no selection.
  // Becomes: {''}
  it('says nobody chose the speaker rather than leaving the field blank', () => {
    render(<TurnDetail seq={1} room={room([message({ decision: null })])} />);

    expect(screen.getByTestId('turn-detail-no-decision')).toHaveTextContent(
      'Nobody chose this speaker: the turn records no selection.',
    );
  });

  // Killed by: frontend/src/components/layout/TurnDetail.tsx :: if (!message && !overrideSummary) {
  // Becomes: if (false) {
  it('says a rewound turn is gone instead of showing whichever turn is still there', () => {
    render(<TurnDetail seq={7} room={room([message({ seq: 1 })])} />);

    expect(screen.getByTestId('turn-detail-gone')).toHaveTextContent(
      'This turn is no longer in the conversation.',
    );
    expect(screen.queryByTestId('turn-detail')).toBeNull();
  });

  it('tells a reader who has picked nothing how to pick something', () => {
    render(<TurnDetail seq={null} room={room([message()])} />);

    expect(screen.getByTestId('turn-detail-empty')).toHaveTextContent('why ›');
  });

  // Acceptance: step rows
  it('renders step rows with human labels, duration, subagent notes, and plain language failures', () => {
    const summary = defaultSummary({
      steps: [
        {
          turn_id: 'turn_101',
          participant_id: 'scout',
          tool_name: 'file_write',
          tool_call_id: 'call_1',
          status: 'success',
          error: null,
          duration_ms: 1250,
          arguments_preview: JSON.stringify({ TargetFile: 'src/main.py' }),
          output_preview: 'Wrote 42 bytes',
          truncated: false,
          written_path: 'src/main.py',
          wrote_unnamed: false,
          subagent_id: null,
          recorded_at: '2026-01-01T00:00:01Z',
          seq: 1,
        },
        {
          turn_id: 'turn_101',
          participant_id: 'scout',
          tool_name: 'subagent_run',
          tool_call_id: 'call_2',
          status: 'success',
          error: null,
          duration_ms: 450,
          arguments_preview: '{}',
          output_preview: 'Done',
          truncated: false,
          written_path: null,
          wrote_unnamed: false,
          subagent_id: 'helper_1',
          recorded_at: '2026-01-01T00:00:02Z',
          seq: 1,
        },
        {
          turn_id: 'turn_101',
          participant_id: 'scout',
          tool_name: 'bash_run',
          tool_call_id: 'call_3',
          status: 'error',
          error: 'Process returned exit code 1: FileNotFoundError',
          duration_ms: 80,
          arguments_preview: JSON.stringify({ CommandLine: 'pytest' }),
          output_preview: 'error',
          truncated: false,
          written_path: null,
          wrote_unnamed: false,
          subagent_id: null,
          recorded_at: '2026-01-01T00:00:03Z',
          seq: 1,
        },
      ],
    });

    render(
      <TurnDetail
        seq={1}
        room={room([message()])}
        summary={summary}
        developerMode={false}
      />,
    );

    const stepRows = screen.getAllByTestId('turn-step-row');
    expect(stepRows).toHaveLength(3);

    // Row 1: File Mutation
    expect(stepRows[0]).toHaveTextContent('File Mutation: main.py');
    expect(stepRows[0]).toHaveTextContent('1.3s');
    expect(stepRows[0]).toHaveTextContent('success');

    // Row 2: Subagent helper
    expect(stepRows[1]).toHaveTextContent(
      "Handed part of this to a helper. The helper's own steps aren't kept.",
    );

    // Row 3: Plain failure text, without raw error dump in U0
    expect(stepRows[2]).toHaveTextContent('Command: pytest');
    expect(stepRows[2]).toHaveTextContent('This step failed.');
    expect(stepRows[2]).not.toHaveTextContent('Process returned exit code 1');
  });

  // Acceptance: document Open calls openInDocs with the path
  it('calls openInDocs when Open is clicked on a document row and shows unnamed writes', () => {
    const onOpen = vi.fn();
    const summary = defaultSummary({
      documents: [
        { path: 'docs/guide.md', writes: 2 },
        { path: 'src/app.py', writes: 1 },
      ],
      unnamed_writes: 3,
    });

    render(
      <OpenInDocsContext.Provider value={onOpen}>
        <TurnDetail seq={1} room={room([message()])} summary={summary} />
      </OpenInDocsContext.Provider>,
    );

    expect(screen.getByText('docs/guide.md')).toBeDefined();
    expect(screen.getByText('(2 writes)')).toBeDefined();

    const openBtn = screen.getByTestId('open-doc-docs/guide.md');
    fireEvent.click(openBtn);
    expect(onOpen).toHaveBeenCalledWith('docs/guide.md');

    expect(screen.getByTestId('turn-detail-unnamed-writes')).toHaveTextContent(
      'Also wrote 3 file(s) without a name.',
    );
  });

  // Acceptance: each absence wording
  it('renders each absence wording correctly', () => {
    // 1) steps_absent_reason: not_recorded
    const { unmount: unmount1 } = render(
      <TurnDetail
        seq={1}
        room={room([message()])}
        summary={defaultSummary({ steps_absent_reason: 'not_recorded', steps: [] })}
      />,
    );
    expect(screen.getByTestId('turn-detail-steps-absent')).toHaveTextContent(
      "This turn's steps weren't recorded.",
    );
    unmount1();

    // 2) steps_absent_reason: not_an_agent_turn
    const { unmount: unmount2 } = render(
      <TurnDetail
        seq={1}
        room={room([message()])}
        summary={defaultSummary({ steps_absent_reason: 'not_an_agent_turn', steps: [] })}
      />,
    );
    expect(screen.getByTestId('turn-detail-steps-absent')).toHaveTextContent(
      'This turn was not run by an agent.',
    );
    unmount2();

    // 3) empty steps with no absent reason
    const { unmount: unmount3 } = render(
      <TurnDetail
        seq={1}
        room={room([message()])}
        summary={defaultSummary({ steps_absent_reason: null, steps: [] })}
      />,
    );
    expect(screen.getByTestId('turn-detail-steps-empty')).toHaveTextContent('No tools used.');
    unmount3();

    // 4) empty documents
    render(
      <TurnDetail
        seq={1}
        room={room([message()])}
        summary={defaultSummary({ documents: [], unnamed_writes: 0 })}
      />,
    );
    expect(screen.getByTestId('turn-detail-documents-empty')).toHaveTextContent(
      'No documents changed.',
    );
  });

  // Acceptance: a one-seat primary turn keeps full provenance collapsed, and a degraded turn shows it expanded
  it('keeps full provenance collapsed behind disclosure on primary turn, and expands when notable', () => {
    // Primary turn (notable is empty)
    const { unmount } = render(
      <TurnDetail
        seq={1}
        room={room([message()])}
        summary={defaultSummary({ notable: [] })}
      />,
    );
    const disclosure = screen.getByTestId('turn-detail-disclosure');
    expect(disclosure).toBeDefined();
    expect(screen.getByText('How this reply was chosen')).toBeDefined();
    expect(screen.queryByTestId('turn-detail-notable')).toBeNull();
    unmount();

    // Degraded turn (notable has 'degraded')
    render(
      <TurnDetail
        seq={1}
        room={room([message()])}
        summary={defaultSummary({
          notable: ['degraded'],
          provenance: {
            path: 'failover',
            requested: { provider: 'anthropic', model: 'claude-3-5-sonnet' },
            served_by: { provider: 'ollama', model: 'qwen3:8b' },
            degraded: true,
          },
        })}
      />,
    );
    expect(screen.queryByTestId('turn-detail-disclosure')).toBeNull();
    const notableBox = screen.getByTestId('turn-detail-notable');
    expect(notableBox).toHaveTextContent('Substituted: the requested model was not available.');
  });

  // Acceptance: no builder vocabulary in U0 (assert on text)
  it('contains no builder vocabulary outside developer mode', () => {
    const summary = defaultSummary({
      notable: ['failover'],
      steps: [
        {
          turn_id: 'turn_101',
          participant_id: 'scout',
          tool_name: 'bash_run',
          tool_call_id: 'call_1',
          status: 'success',
          error: null,
          duration_ms: 100,
          arguments_preview: JSON.stringify({ CommandLine: 'ls -la' }),
          output_preview: 'total 0',
          truncated: false,
          written_path: null,
          wrote_unnamed: false,
          subagent_id: null,
          recorded_at: '2026-01-01T00:00:01Z',
          seq: 1,
        },
      ],
      documents: [{ path: 'README.md', writes: 1 }],
    });

    render(
      <TurnDetail
        seq={1}
        room={room([message()])}
        summary={summary}
        developerMode={false}
      />,
    );

    const containerText = screen.getByTestId('turn-detail').textContent || '';

    // Forbidden vocabulary in U0 per FR-13.9:
    expect(containerText).not.toMatch(/provenance/i);
    expect(containerText).not.toMatch(/selector/i);
    expect(containerText).not.toMatch(/rendered_through/i);
    expect(containerText).not.toMatch(/\bturn_id\b/i);
    expect(containerText).not.toMatch(/\bseq\b/i);
    // Raw tool name bash_run must not be shown outside developer mode:
    expect(containerText).not.toMatch(/bash_run/);
  });

  // Acceptance: raw error hidden with developer mode off
  it('hides raw error when developer mode is off and displays it when on', () => {
    const summary = defaultSummary({
      completed: true,
      error: 'CRITICAL_INTERNAL_RPC_TIMEOUT: 504 Gateway Timeout at server.py:142',
    });

    // Dev mode OFF
    const { unmount } = render(
      <TurnDetail
        seq={1}
        room={room([message()])}
        summary={summary}
        developerMode={false}
      />,
    );

    expect(screen.getByTestId('turn-detail-error')).toBeDefined();
    expect(screen.getByText(/couldn't finish this turn/i)).toBeDefined();
    expect(screen.queryByTestId('turn-detail-raw-error')).toBeNull();
    expect(screen.queryByText(/CRITICAL_INTERNAL_RPC_TIMEOUT/)).toBeNull();
    unmount();

    // Dev mode ON
    render(
      <TurnDetail
        seq={1}
        room={room([message()])}
        summary={summary}
        developerMode={true}
      />,
    );
    expect(screen.getByTestId('turn-detail-raw-error')).toHaveTextContent(
      'CRITICAL_INTERNAL_RPC_TIMEOUT: 504 Gateway Timeout at server.py:142',
    );
  });

  // Acceptance: refetch on matching room.{id}.tool event; assert fetch mock count; do not use timers.
  it('refetches on a matching room.{id}.tool event and on AGENT_REPLY final', async () => {
    let fetchCount = 0;
    const stubSummary = defaultSummary();

    vi.stubGlobal(
      'fetch',
      vi.fn(async () => {
        fetchCount += 1;
        return new Response(JSON.stringify(stubSummary), { status: 200 });
      }),
    );

    const testRoom = room([message({ seq: 1, turn_id: 'turn_101' })]);

    const { rerender } = render(
      <TurnDetail seq={1} room={testRoom} events={[]} />,
    );

    await waitFor(() => {
      expect(fetchCount).toBe(1);
    });

    // 1) Passing an unrelated event should NOT bump fetch count
    const unrelatedEvent: EventEnvelope = {
      id: 'e_unrelated',
      type: 'event',
      timestamp: '2026-01-01T00:00:00Z',
      topic: 'room.room_other.tool',
      event_type: 'TOOL_RESULT',
      payload: { turn_id: 'other_turn' },
    };
    rerender(<TurnDetail seq={1} room={testRoom} events={[unrelatedEvent]} />);
    expect(fetchCount).toBe(1);

    // 2) Passing a matching tool event with same turn_id MUST bump fetch count
    const matchingToolEvent: EventEnvelope = {
      id: 'e_tool_match',
      type: 'event',
      timestamp: '2026-01-01T00:00:00Z',
      topic: 'room.room_1.tool',
      event_type: 'TOOL_RESULT',
      payload: { turn_id: 'turn_101' },
    };
    rerender(
      <TurnDetail
        seq={1}
        room={testRoom}
        events={[unrelatedEvent, matchingToolEvent]}
      />,
    );

    await waitFor(() => {
      expect(fetchCount).toBe(2);
    });

    // 3) Passing matching AGENT_REPLY final for seq 1 MUST bump fetch count
    const finalEvent: EventEnvelope = {
      id: 'e_reply_final',
      type: 'event',
      timestamp: '2026-01-01T00:00:00Z',
      topic: 'room.room_1',
      event_type: 'AGENT_REPLY',
      payload: { status: 'final', seq: 1, turn_id: 'turn_101' },
    };
    rerender(
      <TurnDetail
        seq={1}
        room={testRoom}
        events={[unrelatedEvent, matchingToolEvent, finalEvent]}
      />,
    );

    await waitFor(() => {
      expect(fetchCount).toBe(3);
    });
  });
});
