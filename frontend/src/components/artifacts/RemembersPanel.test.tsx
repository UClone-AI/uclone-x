import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest';
import { render, screen } from '@testing-library/react';
import { RemembersPanel, readFailedSentence } from './RemembersPanel';
import type { RememberedStatement, SavedFact, SeatKnowledge } from '../../lib/roomDock';
import { ABSENCE_CLAIM } from '../../lib/absenceGuard';

const said = (statement: string, over: Partial<RememberedStatement> = {}): RememberedStatement => ({
  statement,
  subject: 'x',
  relation: 'is_a',
  object: 'y',
  source_session_id: 'sess_room__room-a__scout',
  confidence: 0.9,
  learned: false,
  ...over,
});

const okKnowledge = (remembers: RememberedStatement[], reason: string | null = null): SeatKnowledge =>
  ({
    room_id: 'room-a',
    participant_id: 'scout',
    session_id: 'sess_room__room-a__scout',
    status: 'ok',
    reason,
    remembers,
    // The raw graph the developer surface draws; this panel must not show any of it.
    triples: [{ subject: 'tide', predicate: 'is_a', object: 'rhythm', tier: 'T2' }],
    nodes: [{ id: 'tide', label: 'tide', tier: 'T2', ontology: 'core' }],
    edges: [{ source: 'tide', target: 'rhythm', predicate: 'is_a' }],
    summary: { total_nodes: 2, total_edges: 1 },
  }) as unknown as SeatKnowledge;

const absent = (
  status: 'not_recorded' | 'unreadable' | 'no_ontology',
  reason: string,
): SeatKnowledge => ({
  room_id: 'room-a',
  participant_id: 'scout',
  session_id: 'sess_room__room-a__scout',
  status,
  reason,
  remembers: null,
  saved_facts: null,
  saved_facts_reason: null,
  triples: null,
  nodes: null,
  edges: null,
  summary: null,
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
});

afterEach(() => {
  vi.unstubAllGlobals();
});

const URL_A = '/api/rooms/room-a/knowledge?agent_id=scout';
const FORBIDDEN = /triple|predicate|tier|ontology/i;

describe('RemembersPanel: what the clone remembers (#1357)', () => {
  it('lists each remembered statement plainly, read for the seat in this conversation', async () => {
    answers[URL_A] = okKnowledge([
      said('tide is a rhythm'),
      said('the harbour closes at dusk', { learned: true }),
    ]);
    const { container } = render(<RemembersPanel roomId="room-a" seatId="scout" seatName="Scout" />);

    const items = await screen.findAllByTestId('remembered-statement');
    expect(items.map((i) => i.textContent)).toEqual(
      expect.arrayContaining([expect.stringContaining('tide is a rhythm')]),
    );
    expect(items).toHaveLength(2);
    expect(requested).toEqual([URL_A]);
    expect(container.textContent ?? '').not.toMatch(FORBIDDEN);
  });

  // A missing store must not read as an empty memory. The declaration pins the first case;
  // the second is the same check for the other status the Core can give.
  it('shows the reason when nothing the seat learned has been saved (not_recorded)', async () => {
    // Killed by: frontend/src/components/artifacts/RemembersPanel.tsx :: (data.reason ?? `No remembered statements are listed for ${name}.`)
    // Becomes: (`No remembered statements are listed for ${name}.`)
    const reason = 'Scout has no knowledge record in this conversation.';
    answers[URL_A] = absent('not_recorded', reason);
    const { container } = render(<RemembersPanel roomId="room-a" seatId="scout" seatName="Scout" />);
    expect(await screen.findByTestId('remembers-reason')).toHaveTextContent(reason);
    expect(screen.queryByTestId('remembered-statement')).toBeNull();
    expect(container.textContent ?? '').not.toMatch(FORBIDDEN);
  });

  it('shows the reason when the knowledge record could not be read (unreadable)', async () => {
    // The Core's sentence (room_dock.py). It concerns this conversation's record only: the
    // saved memory facts are another file, still read and listed beside it (#1429, #1434).
    const reason =
      "Scout's knowledge record for this conversation could not be read, so it cannot be " +
      'shown.';
    answers[URL_A] = withSaved(absent('unreadable', reason), [fact('Kenny favourite colour teal')]);
    const { container } = render(<RemembersPanel roomId="room-a" seatId="scout" seatName="Scout" />);
    expect(await screen.findByTestId('remembers-reason')).toHaveTextContent(reason);
    expect(screen.queryByTestId('remembered-statement')).toBeNull();
    expect(screen.getAllByTestId('saved-fact')).toHaveLength(1);
    expect(container.textContent ?? '').not.toMatch(ABSENCE_CLAIM);
    expect(container.textContent ?? '').not.toMatch(FORBIDDEN);
  });

  it('shows the reason when the seat has no knowledge store (no_ontology)', async () => {
    const reason = 'Scout is running without a knowledge store.';
    answers[URL_A] = absent('no_ontology', reason);
    const { container } = render(<RemembersPanel roomId="room-a" seatId="scout" seatName="Scout" />);
    expect(await screen.findByTestId('remembers-reason')).toHaveTextContent(reason);
    expect(screen.queryByTestId('remembered-statement')).toBeNull();
    expect(container.textContent ?? '').not.toMatch(FORBIDDEN);
  });

  it('says why the list is empty, in the Core’s words', async () => {
    const reason = 'Scout has no remembered facts on record in this conversation.';
    answers[URL_A] = okKnowledge([], reason);
    const { container } = render(<RemembersPanel roomId="room-a" seatId="scout" seatName="Scout" />);
    expect(await screen.findByTestId('remembers-reason')).toHaveTextContent(reason);
    expect(container.textContent ?? '').not.toMatch(FORBIDDEN);
  });

  // Without the Core's reason the panel knows only that the list is empty -- not that the
  // clone remembers nothing (#1366).
  it('says only that none are listed when the Core gives no reason', async () => {
    answers[URL_A] = okKnowledge([], null);
    const { container } = render(<RemembersPanel roomId="room-a" seatId="scout" seatName="Scout" />);
    expect(await screen.findByTestId('remembers-reason')).toHaveTextContent(
      'No remembered statements are listed for Scout.',
    );
    expect(container.textContent ?? '').not.toMatch(/has not remembered anything yet/);
  });

  it('never shows builder words in any state it can be in', async () => {
    const { container, rerender } = render(<RemembersPanel roomId={null} seatId={null} />);
    expect(container.textContent ?? '').not.toMatch(FORBIDDEN);
    rerender(<RemembersPanel roomId="room-a" seatId={null} />);
    expect(container.textContent ?? '').not.toMatch(FORBIDDEN);
    rerender(<RemembersPanel roomId="room-a" seatId="scout" />);
    await screen.findByTestId('remembers-reason'); // the 404 below
    expect(container.textContent ?? '').not.toMatch(FORBIDDEN);
    expect(requested).toEqual([URL_A]);
  });

  it('re-reads when the seat changes', async () => {
    answers[URL_A] = okKnowledge([said('tide is a rhythm')]);
    answers['/api/rooms/room-a/knowledge?agent_id=critic'] = okKnowledge([said('ropes fray')]);
    const { rerender } = render(<RemembersPanel roomId="room-a" seatId="scout" />);
    await screen.findByText(/tide is a rhythm/);
    rerender(<RemembersPanel roomId="room-a" seatId="critic" />);
    await screen.findByText(/ropes fray/);
    expect(screen.queryByText(/tide is a rhythm/)).toBeNull();
  });
});

const fact = (statement: string, over: Partial<SavedFact> = {}): SavedFact => ({
  statement,
  subject: 'Kenny',
  relation: 'favourite_colour',
  object: 'teal',
  source_session_id: 'sess_room__room-a__scout',
  saved_here: true,
  confidence: 1,
  ...over,
});

const withSaved = (
  base: SeatKnowledge,
  saved_facts: SavedFact[] | null,
  saved_facts_reason: string | null = null,
): SeatKnowledge => ({ ...base, saved_facts, saved_facts_reason }) as SeatKnowledge;

describe('RemembersPanel: the facts the clone saved to memory (#1401)', () => {
  it('lists the saved facts, and marks the ones saved in this conversation', async () => {
    answers[URL_A] = withSaved(okKnowledge([said('tide is a rhythm')]), [
      fact('Kenny favourite colour teal'),
      fact('the harbour closes at dusk', { saved_here: false }),
    ]);
    const { container } = render(<RemembersPanel roomId="room-a" seatId="scout" seatName="Scout" />);

    const items = await screen.findAllByTestId('saved-fact');
    expect(items.map((i) => i.textContent)).toEqual([
      'Kenny favourite colour tealsaved in this conversation',
      'the harbour closes at dusk',
    ]);
    // The conversation's own record is still listed, as its own group.
    expect(screen.getAllByTestId('remembered-statement')).toHaveLength(1);
    expect(screen.getByTestId('remembers-saved')).toHaveTextContent('Saved to memory');
    expect(container.textContent ?? '').not.toMatch(FORBIDDEN);
  });

  // The seat never ran in a way that left a record here, but its memory is its own: the
  // saved facts are listed even so.
  it('lists the saved facts when this conversation has no knowledge record', async () => {
    const reason = 'Scout has no knowledge record in this conversation.';
    answers[URL_A] = withSaved(absent('not_recorded', reason), [fact('Kenny favourite colour teal')]);
    render(<RemembersPanel roomId="room-a" seatId="scout" seatName="Scout" />);
    expect(await screen.findAllByTestId('saved-fact')).toHaveLength(1);
    expect(screen.getByTestId('remembers-reason')).toHaveTextContent(reason);
  });

  it('shows the Core’s sentence when there are no saved facts to list, or they could not be read', async () => {
    // Killed by: frontend/src/components/artifacts/RemembersPanel.tsx :: data.saved_facts_reason ?? (saved === null ? `${name}'s saved facts are not listed.` : null);
    // Becomes: (saved === null ? `${name}'s saved facts are not listed.` : null);
    const empty = 'No saved facts are listed for Scout.';
    answers[URL_A] = withSaved(okKnowledge([]), [], empty);
    const { unmount } = render(<RemembersPanel roomId="room-a" seatId="scout" seatName="Scout" />);
    expect(await screen.findByTestId('saved-facts-reason')).toHaveTextContent(empty);
    expect(screen.queryByTestId('saved-fact')).toBeNull();
    unmount();

    const unread = "Scout's saved facts could not be read, so they cannot be shown.";
    answers[URL_A] = withSaved(okKnowledge([]), null, unread);
    render(<RemembersPanel roomId="room-a" seatId="scout" seatName="Scout" />);
    expect(await screen.findByTestId('saved-facts-reason')).toHaveTextContent(unread);
  });
});

/**
 * A failed read, as the browser meets it (#1401). Nothing is cleaned before the panel sees
 * it: an unreachable runtime is a real `fetch` to a closed port, rejecting as it does in the
 * page, and a crashed one is a real `Response` carrying the plain-text 500 the server sends.
 */
describe('RemembersPanel: a failed read shows no transport text (#1401)', () => {
  const TECHNICAL =
    /Failed to fetch|fetch failed|TypeError|HTTP \d|\b[45]\d\d\b|Internal Server Error|Traceback|Error:|ECONNREFUSED|\/api\//i;

  // Killed by: frontend/src/lib/roomDock.ts :: fault: err instanceof RoomReadError ? err.fault : { kind: 'unreachable', detail: null },
  // Becomes: fault: err instanceof RoomReadError ? err.fault : { kind: 'unreachable', detail: String(err) },
  it('an unreachable runtime reads as a plain sentence', async () => {
    vi.unstubAllGlobals();
    const realFetch = globalThis.fetch;
    // Port 1 on loopback: nothing listens there, so the connection is refused.
    vi.stubGlobal('fetch', (url: string) => realFetch(`http://127.0.0.1:1${url}`));

    const { container } = render(<RemembersPanel roomId="room-a" seatId="scout" seatName="Scout" />);
    expect(
      await screen.findByTestId('remembers-reason', undefined, { timeout: 5000 }),
    ).toHaveTextContent(readFailedSentence('Scout'));
    expect(container.textContent ?? '').not.toMatch(TECHNICAL);
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

    const { container } = render(<RemembersPanel roomId="room-a" seatId="scout" seatName="Scout" />);
    expect(await screen.findByTestId('remembers-reason')).toHaveTextContent(readFailedSentence('Scout'));
    expect(container.textContent ?? '').not.toMatch(TECHNICAL);
  });

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

    const { container } = render(<RemembersPanel roomId="room-a" seatId="scout" seatName="Scout" />);
    expect(await screen.findByTestId('remembers-reason')).toHaveTextContent(detail);
    expect(container.textContent ?? '').not.toMatch(TECHNICAL);
  });
});
