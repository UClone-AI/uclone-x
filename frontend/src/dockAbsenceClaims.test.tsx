import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest';
import { render, screen, waitFor } from '@testing-library/react';
import { copyLines } from './lib/copyGuard';
import { ABSENCE_CLAIM, DOCK_SOURCES, absenceClaims } from './lib/absenceGuard';
import type {
  RoomArtifacts,
  RoomToolUse,
  RoomTopology,
  SeatHistory,
  SeatKnowledge,
} from './lib/roomDock';
import { DocViewer } from './components/artifacts/DocViewer';
import { ActivityTimeline } from './components/artifacts/ActivityTimeline';
import { RemembersPanel } from './components/artifacts/RemembersPanel';
import { TopologyTab } from './components/TopologyTab';

/**
 * The dock never says that nothing was written or that no tools ran (#1366, #1374).
 *
 * Rev 31 of the room UI design: the room records files saved by name and the tool calls
 * saved turns reported, and a shell, an MCP server or a helper can do either unseen. So
 * every dock surface lists what was recorded and names the gaps it knows about. This reads
 * the dock's sources, and renders each surface in the states where it used to substitute a
 * claim of its own -- an empty list with no `reason` from the Core, a turn with no tool
 * calls listed, a call that may have written without naming a path.
 */

//: Every source under `src`, as text (see `copy.test.ts` for why a raw glob, not `node:fs`).
const SOURCES: Record<string, string> = import.meta.glob('./**/*.{ts,tsx}', {
  query: '?raw',
  import: 'default',
  eager: true,
});

let answers: Record<string, unknown>;

beforeEach(() => {
  answers = {};
  vi.stubGlobal(
    'fetch',
    vi.fn(async (url: string) => {
      const body = answers[url];
      if (body === undefined) {
        return new Response(JSON.stringify({ detail: `no stub for ${url}` }), { status: 404 });
      }
      return new Response(JSON.stringify(body), { status: 200 });
    }),
  );
});

afterEach(() => {
  vi.unstubAllGlobals();
});

const artifacts = (over: Partial<RoomArtifacts>): RoomArtifacts => ({
  room_id: 'room-a',
  artifacts: [],
  total: 0,
  unattributed_writes: 0,
  unattributed_note: null,
  unrecorded_turns: 0,
  unrecorded_note: null,
  unsaved_turns: 0,
  record_gaps: [],
  scope_note: 'This list shows files the clones saved by name.',
  turn_running: false,
  reason: null,
  ...over,
});

const history = (over: Partial<SeatHistory>): SeatHistory => ({
  room_id: 'room-a',
  participant_id: 'scout',
  display_name: 'Scout',
  session_id: 's1',
  live: true,
  turns: [],
  tool_uses: [],
  tools_note: "This lists the tool calls Scout's saved turns reported.",
  unsaved_turns: 0,
  unsaved_note: null,
  reason: null,
  ...over,
});

const use = (over: Partial<RoomToolUse>): RoomToolUse => ({
  turn_id: 't1',
  participant_id: 'scout',
  tool_name: 'run_command',
  tool_call_id: 'c1',
  status: 'success',
  error: null,
  duration_ms: 3,
  arguments_preview: '{"command": "make"}',
  output_preview: '',
  truncated: false,
  written_path: null,
  wrote_unnamed: true,
  subagent_id: null,
  recorded_at: '2026-09-22T10:00:00Z',
  seq: 2,
  ...over,
});

const answered = (tools: RoomToolUse[] | null) => ({
  seq: 2,
  turn_id: 't1',
  created_at: '2026-09-22T10:00:00Z',
  status: 'answered' as const,
  error: null,
  content_preview: 'done',
  tools,
  tools_not_recorded_reason: null,
});

const topology = (over: Partial<RoomTopology>): RoomTopology => ({
  room_id: 'room-a',
  nodes: [
    { id: 'seat:scout', kind: 'seat', participant_id: 'scout', display_name: 'Scout', session_id: 's1', status: 'answered', turn_count: 1, live: true },
    { id: 'seat:critic', kind: 'seat', participant_id: 'critic', display_name: 'Critic', session_id: 's2', status: 'idle', turn_count: 0, live: false },
    { id: 'turn:t1', kind: 'turn', participant_id: 'scout', seq: 2, turn_id: 't1', status: 'answered', tools_recorded: true },
  ],
  edges: [{ id: 'e1', source: 'seat:scout', target: 'turn:t1', kind: 'took_turn' }],
  history_complete: true,
  history_gaps: [],
  tool_call_gaps: [],
  summary: { seats: 2, turns: 1, tool_calls: 0, subagents: 0 },
  reason: null,
  ...over,
});

const knowledge = (over: Partial<SeatKnowledge>): SeatKnowledge =>
  ({
    room_id: 'room-a',
    participant_id: 'scout',
    session_id: 's1',
    status: 'ok',
    reason: null,
    remembers: [],
    triples: [],
    nodes: [],
    edges: [],
    summary: {},
    ...over,
  }) as unknown as SeatKnowledge;

const text = (container: HTMLElement): string =>
  // Option labels are text a reader sees in the closed selector.
  container.textContent ?? '';

describe('the dock makes no categorical absence claim (#1374)', () => {
  it('recognises the claims it forbids, and the negations of them', () => {
    const claims = [
      'No file has been written in this conversation yet.',
      '(No files in this conversation)',
      'Scout has not written any files here.',
      "Scout hasn't written anything.",
      'no file written',
      '0 files written',
      'Nothing was written.',
      'Scout took 2 turns here without using a tool.',
      'No tools called.',
      'No tool was used.',
      'Scout used no tools.',
      "Scout didn't use any tools.",
      'so nobody has done anything here.',
      // #1389: the forms the first list missed.
      'No tool use.',
      'no tool use in this turn',
      'No tool calls were made.',
      '0 tool calls',
      'Zero tool calls in this conversation.',
      'Nothing written.',
      'nothing written yet',
      'Scout never wrote a file.',
      'Scout has never written anything here.',
      'Scout never used a tool.',
      'Scout has written nothing.',
      // Memory: the room cannot know what a clone does not remember either.
      'Scout remembers nothing.',
      'Scout has not remembered anything yet.',
      "Scout hasn't learned anything here.",
      'Nothing remembered.',
      'Nothing is remembered from this conversation.',
      'Scout never remembered a thing.',
      'No memories were saved.',
    ];
    // Every miss at once, not the first: a list this long is widened a phrase at a time.
    expect(claims.filter((claim) => !ABSENCE_CLAIM.test(claim))).toEqual([]);
    const allowed = [
      'No files are listed yet.',
      '(No files listed)',
      'No tool calls listed.',
      'No tool calls are listed for Scout’s 2 recorded turns here.',
      'May have written to a file without naming it, so nothing can be opened from here.',
      'This turn did not record which tools it used in full.',
      'No turns are listed for this conversation.',
      'No tool calls are listed.',
      '0 tool calls listed',
      'No tool use is listed for this turn.',
      'No remembered statements are listed for Scout.',
      'Nothing Scout remembers is listed yet.',
      // A reviewer's two false positives on #1406: a listing, and a true capability statement.
      'No tool calls have been listed.',
      'Scout does not remember across conversations unless memory is on.',
    ];
    expect(allowed.filter((sentence) => ABSENCE_CLAIM.test(sentence))).toEqual([]);
  });

  // Killed by: frontend/src/components/TopologyTab.tsx :: <p className="text-[11px] text-slate-500">No tool calls listed.</p>
  // Becomes: <p className="text-[11px] text-slate-500">No tools called.</p>
  // A form the #1374 pattern let through (#1389):
  // Killed by: frontend/src/components/TopologyTab.tsx :: >No tool calls listed.<
  // Becomes: >No tool calls were made.<
  it('holds no such claim in the copy of any dock source', () => {
    const offences = DOCK_SOURCES.flatMap((path) => {
      const body = SOURCES[path];
      expect(body, `${path} is not in the glob`).toBeTypeOf('string');
      return absenceClaims(path, copyLines(body ?? ''));
    });
    expect(offences).toEqual([]);
  });

  // Killed by: frontend/src/components/artifacts/DocViewer.tsx :: ? listing.reason ?? `No files are listed yet. ${listing.scope_note ?? ''}`.trim()
  // Becomes: ? listing.reason ?? 'No file has been written in this conversation yet.'
  it('Docs, empty and with no reason from the Core, says only what is listed', async () => {
    answers['/api/rooms/room-a/artifacts'] = artifacts({});
    const { container } = render(<DocViewer roomId="room-a" />);
    expect(await screen.findByTestId('doc-reason')).toHaveTextContent('No files are listed yet.');
    expect(screen.getByTestId('doc-reason')).toHaveTextContent('saved by name');
    expect(text(container)).not.toMatch(ABSENCE_CLAIM);
  });

  // Killed by: frontend/src/components/artifacts/ActivityTimeline.tsx :: return `No tool calls are listed for ${name}'s ${recordedTurns} recorded ${recordedTurns === 1 ? 'turn' : 'turns'} here.`;
  // Becomes: return `${name} took ${recordedTurns} turns here without using a tool.`;
  it('Activity, for turns with no tool calls listed, says none are listed and what the list covers', async () => {
    answers['/api/rooms/room-a/seats/scout/history'] = history({ turns: [answered([])] });
    const { container } = render(<ActivityTimeline roomId="room-a" seatId="scout" />);
    expect(await screen.findByTestId('activity-empty-state')).toHaveTextContent(
      'No tool calls are listed',
    );
    expect(screen.getByTestId('activity-tools-note')).toHaveTextContent('saved turns reported');
    expect(text(container)).not.toMatch(ABSENCE_CLAIM);
  });

  // The header's count is of the calls listed, not a total of what the clone did.
  // Killed by: frontend/src/components/artifacts/ActivityTimeline.tsx :: {filteredActivities.length} listed
  // Becomes: {filteredActivities.length} calls
  it('Activity, for a seat with no turns and no reason from the Core, claims nothing', async () => {
    answers['/api/rooms/room-a/seats/scout/history'] = history({});
    const { container } = render(<ActivityTimeline roomId="room-a" seatId="scout" />);
    expect(await screen.findByTestId('activity-empty-state')).toHaveTextContent(
      'No turns by Scout are listed',
    );
    expect(text(container)).toContain('0 listed');
    expect(text(container)).not.toMatch(/\b0 calls?\b|did here/);
    expect(text(container)).not.toMatch(ABSENCE_CLAIM);
  });

  // Killed by: frontend/src/components/artifacts/ActivityTimeline.tsx :: <span>May have written to a file without naming it, so nothing can be opened from here.</span>
  // Becomes: <span>Wrote to a file without naming it, so it can't be opened from here.</span>
  it('Activity says a call *may* have written without naming a path', async () => {
    answers['/api/rooms/room-a/seats/scout/history'] = history({
      turns: [answered([use({})])],
      tool_uses: [use({})],
    });
    render(<ActivityTimeline roomId="room-a" seatId="scout" />);
    expect(await screen.findByText(/May have written to a file without naming it/)).toBeInTheDocument();
    expect(screen.queryByText(/^Wrote to a file without naming it/)).toBeNull();
  });

  it('the DAG, with an idle seat and a turn with no tool calls listed, claims nothing', async () => {
    answers['/api/rooms/room-a/topology'] = topology({});
    const { container } = render(<TopologyTab roomId="room-a" />);
    await screen.findByTestId('topology-turn-turn:t1');
    expect(text(container)).toContain('No tool calls listed.');
    expect(text(container)).not.toMatch(ABSENCE_CLAIM);
    expect(text(container)).not.toMatch(/no turns yet|nobody has taken a turn/i);
  });

  it('the DAG, with no turns at all, says none are listed and names the Core’s gaps', async () => {
    answers['/api/rooms/room-a/topology'] = topology({
      nodes: [topology({}).nodes[1]],
      edges: [],
      history_complete: false,
      history_gaps: ['its history was cleared'],
      summary: { seats: 1, turns: 0, tool_calls: 0, subagents: 0 },
    });
    const { container } = render(<TopologyTab roomId="room-a" />);
    expect(await screen.findByTestId('topology-history-gaps')).toHaveTextContent(
      'its history was cleared',
    );
    expect(text(container)).toContain('No turns are listed for this conversation.');
    expect(text(container)).not.toMatch(ABSENCE_CLAIM);
    expect(text(container)).not.toMatch(/no turns yet|nobody has taken a turn/i);
  });

  it('Remembers, empty with no reason from the Core, says the list is empty', async () => {
    answers['/api/rooms/room-a/knowledge?agent_id=scout'] = knowledge({});
    const { container } = render(
      <RemembersPanel roomId="room-a" seatId="scout" seatName="Scout" />,
    );
    await waitFor(() =>
      expect(screen.getByTestId('remembers-reason')).toHaveTextContent(
        'No remembered statements are listed for Scout.',
      ),
    );
    expect(text(container)).not.toMatch(/has not remembered anything yet/);
    expect(text(container)).not.toMatch(ABSENCE_CLAIM);
  });
});
