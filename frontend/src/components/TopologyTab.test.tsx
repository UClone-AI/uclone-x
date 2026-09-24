import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest';
import { fireEvent, render, screen, waitFor, within } from '@testing-library/react';
import { TopologyTab } from './TopologyTab';
import type { RoomTopology } from '../lib/roomDock';

const TOPOLOGY: RoomTopology = {
  room_id: 'room-a',
  nodes: [
    { id: 'seat:scout', kind: 'seat', participant_id: 'scout', display_name: 'Scout', session_id: 's1', status: 'answered', turn_count: 2, live: true },
    { id: 'seat:critic', kind: 'seat', participant_id: 'critic', display_name: 'Critic', session_id: 's2', status: 'idle', turn_count: 0, live: false },
    { id: 'turn:t1', kind: 'turn', participant_id: 'scout', seq: 2, turn_id: 't1', status: 'answered', tools_recorded: true },
    { id: 'tool:t1:0', kind: 'tool', participant_id: 'scout', tool_name: 'write_file', tool_call_id: 'c1', status: 'success', turn_id: 't1' },
    { id: 'tool:t1:1', kind: 'tool', participant_id: 'scout', tool_name: 'spawn_helper', tool_call_id: 'c2', status: 'error', turn_id: 't1' },
    { id: 'subagent:h1', kind: 'subagent', subagent_id: 'h1', parent_participant_id: 'scout' },
    { id: 'turn:t2', kind: 'turn', participant_id: 'scout', seq: 5, turn_id: 't2', status: 'failed', tools_recorded: false },
  ],
  edges: [
    { id: 'e1', source: 'seat:scout', target: 'turn:t1', kind: 'took_turn' },
    { id: 'e2', source: 'turn:t1', target: 'tool:t1:0', kind: 'called' },
    { id: 'e3', source: 'turn:t1', target: 'tool:t1:1', kind: 'called' },
    { id: 'e4', source: 'seat:scout', target: 'subagent:h1', kind: 'spawned' },
    { id: 'e5', source: 'seat:scout', target: 'turn:t2', kind: 'took_turn' },
    { id: 'e6', source: 'turn:t1', target: 'turn:t2', kind: 'followed_by' },
  ],
  history_complete: true,
  history_gaps: [],
  tool_call_gaps: [],
  summary: { seats: 2, turns: 2, tool_calls: 2, subagents: 1 },
  reason: null,
};

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
});

afterEach(() => {
  vi.unstubAllGlobals();
});

const URL_A = '/api/rooms/room-a/topology';

describe('TopologyTab: the conversation as a graph (#1355)', () => {
  it('reads the topology of the conversation on screen, by room id', async () => {
    answers[URL_A] = TOPOLOGY;
    render(<TopologyTab roomId="room-a" />);
    await screen.findByTestId('topology-seat-scout');
    expect(requested).toEqual([URL_A]);
  });

  it('draws each seat, its turns in order, their tools, and the helpers it started', async () => {
    answers[URL_A] = TOPOLOGY;
    render(<TopologyTab roomId="room-a" />);
    const turns = await screen.findAllByTestId(/^topology-turn-/);
    expect(turns.map((t) => t.getAttribute('data-testid'))).toEqual([
      'topology-turn-turn:t1',
      'topology-turn-turn:t2',
    ]);
    const first = within(turns[0]);
    expect(first.getByText('write_file')).toBeInTheDocument();
    expect(first.getByText('spawn_helper')).toBeInTheDocument();
    expect(within(turns[1]).getByText(/did not record which tools it used in full/)).toBeInTheDocument();
    expect(within(screen.getByTestId('topology-seat-scout')).getByText(/h1/)).toBeInTheDocument();
    // An idle seat lists no turns; the graph cannot know none were taken (#1366).
    expect(within(screen.getByTestId('topology-seat-critic')).getByText('No turns listed.')).toBeInTheDocument();
  });

  it('lists a turn’s calls even when it did not record them in full', async () => {
    answers[URL_A] = {
      ...TOPOLOGY,
      nodes: [
        ...TOPOLOGY.nodes,
        { id: 'tool:t2:0', kind: 'tool', participant_id: 'scout', tool_name: 'read_file', tool_call_id: 'c3', status: 'success', turn_id: 't2' },
      ],
      edges: [...TOPOLOGY.edges, { id: 'e7', source: 'turn:t2', target: 'tool:t2:0', kind: 'called' }],
    };
    render(<TopologyTab roomId="room-a" />);
    const partial = await screen.findByTestId('topology-turn-turn:t2');
    expect(within(partial).getByText('read_file')).toBeInTheDocument();
    expect(within(partial).getByText(/did not record which tools it used in full/)).toBeInTheDocument();
  });

  // Killed by: frontend/src/components/TopologyTab.tsx :: {(data.history_gaps ?? []).length > 0 ? (
  // Becomes: {false ? (
  it('names the gaps in its history the Core knows of, in its words', async () => {
    answers[URL_A] = {
      ...TOPOLOGY,
      history_complete: false,
      history_gaps: ['its history was cleared', '1 turn started but stopped before its result was saved'],
    };
    render(<TopologyTab roomId="room-a" />);
    const gaps = await screen.findByTestId('topology-history-gaps');
    expect(gaps).toHaveTextContent('Turns may be missing from this graph because:');
    expect(gaps).toHaveTextContent('its history was cleared');
    expect(gaps).toHaveTextContent('stopped before its result was saved');
    expect(screen.queryByTestId('topology-history-complete')).toBeNull();
  });

  // `history_complete` speaks for turn rows only (#1388 N3): it must never read as "every tool
  // call is shown", and the summary counts the calls listed, not the calls made.
  it('scopes a complete history to turns, and never claims every tool call is shown', async () => {
    answers[URL_A] = TOPOLOGY;
    const { container } = render(<TopologyTab roomId="room-a" />);
    expect(await screen.findByTestId('topology-history-complete')).toHaveTextContent(
      'No turn is known to be missing from this graph.',
    );
    const history = screen.getByTestId('topology-history');
    expect(history).toHaveTextContent("a helper's own calls are not shown");
    expect(container.textContent).toContain('2 tool calls listed');
    expect(container.textContent).not.toMatch(/all (the )?tool calls|every tool call/i);
  });

  // Killed by: frontend/src/components/TopologyTab.tsx :: {(data.tool_call_gaps ?? []).length > 0 && (
  // Becomes: {false && (
  it('names the tool-call gaps the Core knows of, even when no turn is missing', async () => {
    answers[URL_A] = {
      ...TOPOLOGY,
      tool_call_gaps: ['1 turn(s) ended before reporting their tools'],
    };
    render(<TopologyTab roomId="room-a" />);
    const gaps = await screen.findByTestId('topology-tool-call-gaps');
    expect(gaps).toHaveTextContent('Tool calls may be missing from the turns shown because:');
    expect(gaps).toHaveTextContent('1 turn(s) ended before reporting their tools');
    expect(screen.getByTestId('topology-history-complete')).toHaveTextContent(
      'No turn is known to be missing from this graph.',
    );
  });

  it('shows no tool-call gaps when the Core names none', async () => {
    answers[URL_A] = TOPOLOGY;
    render(<TopologyTab roomId="room-a" />);
    await screen.findByTestId('topology-history');
    expect(screen.queryByTestId('topology-tool-call-gaps')).toBeNull();
  });

  it('shows neither completeness nor gaps when the Core says neither', async () => {
    answers[URL_A] = { ...TOPOLOGY, history_complete: false, history_gaps: [] };
    render(<TopologyTab roomId="room-a" />);
    await screen.findByTestId('topology-history');
    expect(screen.queryByTestId('topology-history-complete')).toBeNull();
    expect(screen.queryByTestId('topology-history-gaps')).toBeNull();
  });

  it('shows the Core’s reason when nobody is seated', async () => {
    answers[URL_A] = {
      room_id: 'room-a',
      nodes: [],
      edges: [],
      history_complete: true,
      history_gaps: [],
      tool_call_gaps: [],
      summary: { seats: 0, turns: 0, tool_calls: 0, subagents: 0 },
      reason: 'No agent is seated in this conversation.',
    };
    render(<TopologyTab roomId="room-a" />);
    expect(await screen.findByTestId('topology-reason')).toHaveTextContent(
      'No agent is seated in this conversation.',
    );
  });

  it('makes no request and says why when no conversation is open', async () => {
    render(<TopologyTab roomId={null} />);
    expect(screen.getByTestId('topology-reason')).toHaveTextContent('No conversation is open');
    await waitFor(() => expect(requested).toEqual([]));
  });

  it('reads again on Refresh, and follows a room switch', async () => {
    answers[URL_A] = TOPOLOGY;
    answers['/api/rooms/room-b/topology'] = { ...TOPOLOGY, room_id: 'room-b' };
    const { rerender } = render(<TopologyTab roomId="room-a" />);
    await screen.findByTestId('topology-seat-scout');
    fireEvent.click(screen.getByRole('button', { name: 'Refresh' }));
    await waitFor(() => expect(requested).toEqual([URL_A, URL_A]));
    rerender(<TopologyTab roomId="room-b" />);
    await waitFor(() => expect(requested).toContain('/api/rooms/room-b/topology'));
  });

  it('neither pulses nor draws gradients (ui-authoring)', async () => {
    answers[URL_A] = TOPOLOGY;
    const { container } = render(<TopologyTab roomId="room-a" />);
    await screen.findByTestId('topology-seat-scout');
    expect(container.innerHTML).not.toMatch(/animate-pulse|gradient/);
  });
});
