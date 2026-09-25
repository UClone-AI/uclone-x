import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest';
import { render, screen, fireEvent, waitFor, within } from '@testing-library/react';
import { ActivityTimeline, activityReadFailedSentence } from './ActivityTimeline';
import { expectPlain } from '../../test/plainCopy';
import type { EventEnvelope } from '../../types';
import type { RoomToolUse, SeatHistory, SeatTurn } from '../../lib/roomDock';

const use = (over: Partial<RoomToolUse>): RoomToolUse => ({
  turn_id: 't1',
  participant_id: 'scout',
  tool_name: 'read_file',
  tool_call_id: 'call-1',
  status: 'success',
  error: null,
  duration_ms: 12,
  arguments_preview: '{"path": "notes.md"}',
  output_preview: 'hello',
  truncated: false,
  written_path: null,
  wrote_unnamed: false,
  subagent_id: null,
  recorded_at: '2026-09-22T10:00:00Z',
  seq: 2,
  ...over,
});

const turn = (over: Partial<SeatTurn>): SeatTurn => ({
  seq: 2,
  turn_id: 't1',
  created_at: '2026-09-22T10:00:00Z',
  status: 'answered',
  error: null,
  content_preview: 'done',
  tools: [],
  tools_not_recorded_reason: null,
  ...over,
});

const history = (over: Partial<SeatHistory>): SeatHistory => ({
  room_id: 'room-a',
  participant_id: 'scout',
  display_name: 'Scout',
  session_id: 'sess_room__room-a__scout',
  live: true,
  turns: [],
  tool_uses: [],
  tools_note:
    "This lists the tool calls Scout's saved turns reported. A helper Scout started does its work with tools of its own, which are not listed separately.",
  unsaved_turns: 0,
  unsaved_note: null,
  reason: null,
  ...over,
});

let answers: Record<string, unknown>;
let requested: string[];

beforeEach(() => {
  answers = {};
  requested = [];
  vi.stubGlobal(
    'fetch',
    vi.fn(async (url: string) => {
      requested.push(url);
      const body = answers[url];
      if (body === undefined) {
        return new Response(JSON.stringify({ detail: `no stub for ${url}` }), { status: 404 });
      }
      return new Response(JSON.stringify(body), { status: 200 });
    }),
  );
  Object.assign(navigator, {
    clipboard: { writeText: vi.fn().mockImplementation(() => Promise.resolve()) },
  });
});

afterEach(() => {
  vi.unstubAllGlobals();
});

const HISTORY_A = '/api/rooms/room-a/seats/scout/history';

describe('ActivityTimeline, scoped to a room seat (#1353, #1356)', () => {
  it('reads the seat history of the conversation on screen', async () => {
    answers[HISTORY_A] = history({
      turns: [turn({ tools: [use({})] })],
      tool_uses: [use({})],
    });
    render(<ActivityTimeline roomId="room-a" seatId="scout" events={[]} />);

    await screen.findByTestId('activity-card-call-1');
    expect(requested).toEqual([HISTORY_A]);
  });

  it('re-reads when the conversation changes, and never shows the previous one', async () => {
    answers[HISTORY_A] = history({ tool_uses: [use({ tool_call_id: 'call-a' })] });
    answers['/api/rooms/room-b/seats/scout/history'] = history({
      room_id: 'room-b',
      tool_uses: [use({ tool_call_id: 'call-b' })],
    });
    const { rerender } = render(<ActivityTimeline roomId="room-a" seatId="scout" events={[]} />);
    await screen.findByTestId('activity-card-call-a');

    rerender(<ActivityTimeline roomId="room-b" seatId="scout" events={[]} />);
    await screen.findByTestId('activity-card-call-b');
    expect(screen.queryByTestId('activity-card-call-a')).toBeNull();
    expect(requested).toContain('/api/rooms/room-b/seats/scout/history');
  });

  // The server writes `TOOL_CALL` in upper case; the panel matched `tool_call`, so no live
  // row ever appeared (#1353; the mutation below restores that spelling). This envelope is
  // the literal shape `/api/stream` delivers.
  // Killed by: frontend/src/lib/roomDock.ts :: return kind === 'TOOL_CALL' || kind === 'TOOL_RESULT';
  // Becomes: return kind === 'tool_call' || kind === 'tool_result';
  it('shows one row for a literal upper-case TOOL_CALL envelope on the room tool topic', async () => {
    answers[HISTORY_A] = history({});
    const raw = JSON.parse(
      '{"event_type":"TOOL_CALL","type":"TOOL_CALL","topic":"room.room-a.tool","payload":' +
        '{"room_id":"room-a","participant_id":"scout","session_id":"sess_room__room-a__scout",' +
        '"turn_id":"t9","seq":7,"tool_call_id":"call-live","name":"web_search",' +
        '"arguments_preview":"{\\"query\\": \\"tides\\"}"}}',
    );
    const envelope: EventEnvelope = { id: 'e1', timestamp: '2026-09-22T10:00:00Z', ...raw };
    render(<ActivityTimeline roomId="room-a" seatId="scout" events={[envelope]} />);

    const rows = await screen.findAllByTestId(/^activity-card-/);
    expect(rows).toHaveLength(1);
    expect(screen.getByTestId('activity-card-call-live')).toHaveTextContent('web_search');
  });

  it('ignores tool events from another conversation or another seat', async () => {
    answers[HISTORY_A] = history({});
    const ev = (topic: string, participant: string, id: string): EventEnvelope => ({
      id,
      type: 'TOOL_CALL',
      event_type: 'TOOL_CALL',
      topic,
      timestamp: 'x',
      payload: { participant_id: participant, tool_call_id: id, name: 'read_file', turn_id: 't' },
    });
    render(
      <ActivityTimeline
        roomId="room-a"
        seatId="scout"
        events={[ev('room.room-b.tool', 'scout', 'other-room'), ev('room.room-a.tool', 'critic', 'other-seat')]}
      />,
    );
    await screen.findByTestId('activity-empty-state');
    expect(screen.queryByTestId(/^activity-card-/)).toBeNull();
  });

  it('never reports a stopped, timed-out or failed call as a pass (#1031)', async () => {
    answers[HISTORY_A] = history({
      tool_uses: [
        use({ tool_call_id: 'ok', status: 'success' }),
        use({ tool_call_id: 'bad', status: 'error', error: 'permission denied' }),
        use({ tool_call_id: 'late', status: 'timeout', error: 'took too long' }),
        use({ tool_call_id: 'halt', status: 'stopped' }),
      ],
    });
    render(<ActivityTimeline roomId="room-a" seatId="scout" events={[]} />);
    await screen.findByTestId('activity-card-ok');
    expect(screen.getAllByText('Pass')).toHaveLength(1);
    expect(screen.getAllByText('Error')).toHaveLength(2);
    expect(screen.getAllByText('Stopped')).toHaveLength(1);
  });

  it('says why the list is empty: no turns by the seat are listed', async () => {
    answers[HISTORY_A] = history({ turns: [] });
    render(<ActivityTimeline roomId="room-a" seatId="scout" events={[]} />);
    const empty = await screen.findByTestId('activity-empty-state');
    expect(empty).toHaveTextContent('No turns by Scout are listed in this conversation.');
  });

  // Recorded turns that report no calls are "none listed", never "used no tool" (#1366): a
  // shell, an MCP server or a helper can call tools the turn did not report.
  it('says why the list is empty: the seat’s recorded turns list no tool calls', async () => {
    answers[HISTORY_A] = history({ turns: [turn({}), turn({ seq: 4, turn_id: 't2' })] });
    render(<ActivityTimeline roomId="room-a" seatId="scout" events={[]} />);
    const empty = await screen.findByTestId('activity-empty-state');
    expect(empty).toHaveTextContent("No tool calls are listed for Scout's 2 recorded turns here.");
    expect(empty).not.toHaveTextContent(/without using a tool/);
  });

  // Killed by: frontend/src/components/artifacts/ActivityTimeline.tsx :: {history.tools_note && <p data-testid="activity-tools-note">{history.tools_note}</p>}
  // Becomes: {false && <p data-testid="activity-tools-note">{history.tools_note}</p>}
  it('says what the list covers, and the turns that were not saved, in the Core’s words', async () => {
    const unsaved = '1 turn by Scout started but stopped before its result was saved, so its tool calls are not listed.';
    answers[HISTORY_A] = history({ turns: [turn({})], unsaved_turns: 1, unsaved_note: unsaved });
    render(<ActivityTimeline roomId="room-a" seatId="scout" events={[]} />);
    expect(await screen.findByTestId('activity-tools-note')).toHaveTextContent(
      "This lists the tool calls Scout's saved turns reported.",
    );
    expect(screen.getByTestId('activity-unsaved-note')).toHaveTextContent(unsaved);
  });

  // `tools: null` is "not recorded", which is not "used none" (P6).
  it('states the turns whose tools were not recorded, with the Core’s reason', async () => {
    const why = 'This turn ended before it could report its tools.';
    answers[HISTORY_A] = history({
      turns: [turn({ tools: null, tools_not_recorded_reason: why, status: 'failed' })],
    });
    render(<ActivityTimeline roomId="room-a" seatId="scout" events={[]} />);
    const notice = await screen.findByTestId('activity-not-recorded');
    expect(notice).toHaveTextContent('1 turn did not record which tools it used in full');
    expect(notice).toHaveTextContent('may not all be listed here');
    expect(notice).toHaveTextContent(why);
  });

  it('shows the Core’s reason and makes no request when nothing scopes the read', async () => {
    render(<ActivityTimeline roomId={null} seatId={null} events={[]} />);
    expect(screen.getByTestId('activity-empty-state')).toHaveTextContent(
      'No conversation is open',
    );
    render(<ActivityTimeline roomId="room-a" seatId={null} events={[]} />);
    expect(screen.getAllByTestId('activity-empty-state')[1]).toHaveTextContent(
      'No clone is seated in this conversation, so there is no clone to show activity for.',
    );
    expect(requested).toEqual([]);
  });


  it('opens a written file in Docs from its row', async () => {
    answers[HISTORY_A] = history({
      tool_uses: [use({ tool_call_id: 'w1', tool_name: 'write_file', written_path: 'out/plan.md' })],
    });
    const onOpenInDocs = vi.fn();
    render(
      <ActivityTimeline roomId="room-a" seatId="scout" events={[]} onOpenInDocs={onOpenInDocs} />,
    );
    fireEvent.click(await screen.findByTestId('activity-open-doc-w1'));
    expect(onOpenInDocs).toHaveBeenCalledWith('out/plan.md');
  });

  it('classifies calls and filters by category and search keyword', async () => {
    answers[HISTORY_A] = history({
      tool_uses: [
        use({ tool_call_id: 'm', tool_name: 'write_to_file', arguments_preview: '{"TargetFile": "src/app.py"}' }),
        use({ tool_call_id: 'c', tool_name: 'run_command', arguments_preview: '{"CommandLine": "./ucx test check"}' }),
      ],
    });
    render(<ActivityTimeline roomId="room-a" seatId="scout" events={[]} />);
    await screen.findByText('File Mutation: app.py');
    expect(screen.getByText('Command: ./ucx test check')).toBeDefined();

    fireEvent.click(screen.getByRole('button', { name: 'Commands' }));
    expect(screen.queryByText('File Mutation: app.py')).toBeNull();

    fireEvent.click(screen.getByRole('button', { name: 'All' }));
    fireEvent.change(screen.getByPlaceholderText('Search actions, files, args...'), {
      target: { value: 'check' },
    });
    expect(screen.queryByText('File Mutation: app.py')).toBeNull();
    expect(screen.getByText('Command: ./ucx test check')).toBeDefined();
  });

  it('expands a row to its input, output and error, and copies the whole record', async () => {
    const failed = use({
      tool_call_id: 'x1',
      tool_name: 'run_command',
      arguments_preview: '{"CommandLine": "cat missing.txt"}',
      status: 'error',
      error: 'FileNotFoundError: missing.txt',
      output_preview: 'partial',
      truncated: true,
    });
    answers[HISTORY_A] = history({ tool_uses: [failed] });
    render(<ActivityTimeline roomId="room-a" seatId="scout" events={[]} />);
    fireEvent.click(await screen.findByText('Command: cat missing.txt'));

    expect(screen.getByText('Input Parameters')).toBeDefined();
    expect(screen.getByText(/partial/)).toBeDefined();
    expect(screen.getByText(/cut to fit/)).toBeDefined();
    expect(screen.getByText('Execution Error Trace')).toBeDefined();
    expect(screen.getByTestId('activity-duration-x1')).toHaveTextContent('12.0 ms');

    fireEvent.click(screen.getByTestId('activity-copy-trace-x1'));
    expect(navigator.clipboard.writeText).toHaveBeenCalledWith(JSON.stringify(failed, null, 2));
  });

  it('draws no gradient in its header (ui-authoring)', async () => {
    answers[HISTORY_A] = history({});
    const { container } = render(<ActivityTimeline roomId="room-a" seatId="scout" events={[]} />);
    await waitFor(() => expect(requested).toHaveLength(1));
    expect(container.innerHTML).not.toMatch(/gradient/);
  });
});

/**
 * The registered tool names get the label their behaviour has (#1463). Since #1461 the model
 * is offered `bash_run` and never `run_command`, and the file tools were always `file_write`,
 * `file_edit`, `file_read`, `file_search` and `directory_list`; the classifier matched only
 * other hosts' spellings, so every shell call and every file write read as "Tool Call".
 */
describe('ActivityTimeline labels the tools this runtime registers (#1463)', () => {
  const cardOf = (id: string) => screen.findByTestId(`activity-card-${id}`);

  // Killed by: frontend/src/lib/toolLabels.ts :: name === 'bash_run' ||
  // Becomes: name === 'bash_rux' ||
  it('labels a bash_run call "Command", as it labels run_command', async () => {
    answers[HISTORY_A] = history({
      tool_uses: [
        use({ tool_call_id: 'b1', tool_name: 'bash_run', arguments_preview: '{"command": "ls -la"}' }),
      ],
    });
    render(<ActivityTimeline roomId="room-a" seatId="scout" events={[]} />);

    const card = await cardOf('b1');
    expect(within(card).getByText('Command')).toBeDefined();
    expect(within(card).getByText('Command: ls -la')).toBeDefined();

    fireEvent.click(screen.getByRole('button', { name: 'Commands' }));
    expect(screen.getByTestId('activity-card-b1')).toBeDefined();
  });

  // Killed by: frontend/src/lib/toolLabels.ts :: name === 'file_write' ||
  // Becomes: name === 'file_wrXte' ||
  // Killed by: frontend/src/lib/toolLabels.ts :: name === 'file_edit' ||
  // Becomes: name === 'file_eXit' ||
  it('labels file_write and file_edit calls "File Mutation"', async () => {
    answers[HISTORY_A] = history({
      tool_uses: [
        use({ tool_call_id: 'w1', tool_name: 'file_write', arguments_preview: '{"path": "out/plan.md"}' }),
        use({ tool_call_id: 'e1', tool_name: 'file_edit', arguments_preview: '{"path": "src/app.py"}' }),
      ],
    });
    render(<ActivityTimeline roomId="room-a" seatId="scout" events={[]} />);

    const written = await cardOf('w1');
    expect(within(written).getByText('File Mutation')).toBeDefined();
    expect(within(written).getByText('File Mutation: plan.md')).toBeDefined();
    const edited = await cardOf('e1');
    expect(within(edited).getByText('File Mutation')).toBeDefined();
    expect(within(edited).getByText('File Mutation: app.py')).toBeDefined();
  });

  // Killed by: frontend/src/lib/toolLabels.ts :: name === 'web_fetch' ||
  // Becomes: name === 'web_fetcX' ||
  // Killed by: frontend/src/lib/toolLabels.ts :: name === 'file_read' ||
  // Becomes: name === 'file_reaX' ||
  // Killed by: frontend/src/lib/toolLabels.ts :: name === 'file_search' ||
  // Becomes: name === 'file_searcX' ||
  // Killed by: frontend/src/lib/toolLabels.ts :: name === 'directory_list' ||
  // Becomes: name === 'directory_lisX' ||
  it('labels web_fetch "Web Retrieval" and the read-only file tools "Inspection"', async () => {
    answers[HISTORY_A] = history({
      tool_uses: [
        use({ tool_call_id: 'f1', tool_name: 'web_fetch', arguments_preview: '{"url": "https://example.com/a"}' }),
        use({ tool_call_id: 'r1', tool_name: 'file_read', arguments_preview: '{"path": "docs/notes.md"}' }),
        use({ tool_call_id: 's1', tool_name: 'file_search', arguments_preview: '{"query": "TODO"}' }),
        use({ tool_call_id: 'd1', tool_name: 'directory_list', arguments_preview: '{"path": "src"}' }),
      ],
    });
    render(<ActivityTimeline roomId="room-a" seatId="scout" events={[]} />);

    const fetched = await cardOf('f1');
    expect(within(fetched).getByText('Web Retrieval')).toBeDefined();
    expect(within(fetched).getByText('Web Retrieval: https://example.com/a')).toBeDefined();
    const read = await cardOf('r1');
    expect(within(read).getByText('Inspection')).toBeDefined();
    expect(within(read).getByText('Inspection: notes.md')).toBeDefined();
    const searched = await cardOf('s1');
    expect(within(searched).getByText('Inspection')).toBeDefined();
    const listed = await cardOf('d1');
    expect(within(listed).getByText('Inspection')).toBeDefined();
    expect(within(listed).getByText('Inspection: src')).toBeDefined();
  });
});

/**
 * A failed read, as the browser meets it (#1435). Nothing is cleaned before the timeline
 * sees it: an unreachable runtime is a real `fetch` to a closed port, and a crashed one is a
 * real `Response` carrying the plain-text 500 the server sends.
 */
describe('ActivityTimeline: a failed read shows no transport text (#1435)', () => {
  // Killed by: frontend/src/components/artifacts/ActivityTimeline.tsx :: return read.fault.detail ?? activityReadFailedSentence(name);
  // Becomes: return read.error ?? activityReadFailedSentence(name);
  it('an unreachable runtime reads as a plain sentence', async () => {
    vi.unstubAllGlobals();
    const realFetch = globalThis.fetch;
    // Port 1 on loopback: nothing listens there, so the connection is refused.
    vi.stubGlobal('fetch', (url: string) => realFetch(`http://127.0.0.1:1${url}`));

    render(<ActivityTimeline roomId="room-a" seatId="scout" seatName="Scout" events={[]} />);
    await waitFor(
      () =>
        expect(screen.getByTestId('activity-empty-state')).toHaveTextContent(
          activityReadFailedSentence('Scout'),
        ),
      { timeout: 3000 },
    );
    expectPlain(screen.getByTestId('activity-empty-state').textContent);
  });

  // Killed by: frontend/src/lib/roomDock.ts :: let detail: string | null = null;
  // Becomes: let detail: string | null = message;
  it('a plain-text 500 reads as a plain sentence, not its status line', async () => {
    vi.stubGlobal(
      'fetch',
      async () =>
        new Response('Internal Server Error', {
          status: 500,
          statusText: 'Internal Server Error',
          headers: { 'content-type': 'text/plain; charset=utf-8' },
        }),
    );

    render(<ActivityTimeline roomId="room-a" seatId="scout" seatName="Scout" events={[]} />);
    await waitFor(() =>
      expect(screen.getByTestId('activity-empty-state')).toHaveTextContent(
        activityReadFailedSentence('Scout'),
      ),
    );
    expectPlain(screen.getByTestId('activity-empty-state').textContent);
  });

  // Killed by: frontend/src/components/artifacts/ActivityTimeline.tsx :: return read.fault.detail ?? activityReadFailedSentence(name);
  // Becomes: return activityReadFailedSentence(name);
  it('a refusal in the Core’s own plain words is shown as it gave them', async () => {
    const detail = 'Scout is not seated in this conversation.';
    vi.stubGlobal(
      'fetch',
      async () =>
        new Response(JSON.stringify({ detail }), {
          status: 404,
          headers: { 'content-type': 'application/json' },
        }),
    );

    render(<ActivityTimeline roomId="room-a" seatId="scout" seatName="Scout" events={[]} />);
    await waitFor(() =>
      expect(screen.getByTestId('activity-empty-state')).toHaveTextContent(detail),
    );
    expectPlain(screen.getByTestId('activity-empty-state').textContent);
  });
});
