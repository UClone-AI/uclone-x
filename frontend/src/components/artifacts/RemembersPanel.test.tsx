import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest';
import { fireEvent, render, screen, within } from '@testing-library/react';
import { RemembersPanel, editRefusedSentence, readFailedSentence } from './RemembersPanel';
import type { KnownFact, SeatKnowledge } from '../../lib/roomDock';
import { ABSENCE_CLAIM } from '../../lib/absenceGuard';
import { ko } from '../../i18n/ko';

const fact = (statement: string, over: Partial<KnownFact> = {}): KnownFact => ({
  fact_id: `mem_${statement.length}${statement.charCodeAt(0)}`,
  statement,
  subject: 'Kenny',
  predicate: 'favourite_colour',
  object_value: 'teal',
  origin: 'saved',
  learned_here: true,
  source_turn_id: 'turn-1',
  confidence: 1,
  created_at: '2026-09-26T00:00:00Z',
  ...over,
});

const knowledge = (
  facts: KnownFact[] | null,
  facts_reason: string | null = null,
  status: SeatKnowledge['status'] = 'ok',
  reason: string | null = null,
): SeatKnowledge =>
  ({
    room_id: 'room-a',
    participant_id: 'scout',
    session_id: 'sess_room__room-a__scout',
    status,
    reason,
    facts,
    facts_reason,
    // The raw graph the developer surface draws; this panel must not show any of it.
    triples: [{ subject: 'tide', predicate: 'is_a', object: 'rhythm', tier: 'T2' }],
    nodes: [{ id: 'tide', label: 'tide', tier: 'T2', ontology: 'core' }],
    edges: [{ source: 'tide', target: 'rhythm', predicate: 'is_a' }],
    summary: { total_nodes: 2, total_edges: 1 },
  }) as unknown as SeatKnowledge;

interface Call {
  url: string;
  method: string;
  body: unknown;
}

let answers: Record<string, unknown>;
let edits: Record<string, { status: number; body: unknown }>;
let requested: string[];
let calls: Call[];

beforeEach(() => {
  answers = {};
  edits = {};
  requested = [];
  calls = [];
  vi.stubGlobal(
    'fetch',
    vi.fn(async (url: string, init?: RequestInit) => {
      const method = init?.method ?? 'GET';
      calls.push({ url, method, body: init?.body ? JSON.parse(String(init.body)) : undefined });
      if (method !== 'GET') {
        const edit = edits[`${method} ${url}`] ?? { status: 200, body: { fact: {} } };
        return new Response(JSON.stringify(edit.body), { status: edit.status });
      }
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
const FORBIDDEN = /triple|predicate|tier|ontology|mem_/i;

describe('RemembersPanel: one clone-wide list in two groups (#1638 step 3)', () => {
  it('groups what was learned here before what was learned elsewhere, each with how', async () => {
    // Killed by: frontend/src/components/artifacts/RemembersPanel.tsx :: const here = (facts ?? []).filter((f) => f.learned_here);
    // Becomes: const here = (facts ?? []).filter((f) => !f.learned_here);
    answers[URL_A] = knowledge([
      fact('Kenny works at Hanbit', { learned_here: false, origin: 'told' }),
      fact('Kenny favourite colour teal'),
      fact('Project X uses Postgres 16', { origin: 'corrected' }),
    ]);
    const { container } = render(<RemembersPanel roomId="room-a" seatId="scout" seatName="Scout" />);

    const here = await screen.findByTestId('remembers-here');
    const elsewhere = screen.getByTestId('remembers-elsewhere');
    expect(here).toHaveTextContent('Learned in this conversation');
    expect(elsewhere).toHaveTextContent('From other conversations');
    expect(within(here).getAllByTestId('known-fact').map((i) => i.textContent)).toEqual([
      'Kenny favourite colour tealsaved',
      'Project X uses Postgres 16corrected',
    ]);
    expect(within(elsewhere).getAllByTestId('known-fact').map((i) => i.textContent)).toEqual([
      'Kenny works at Hanbittold',
    ]);
    expect(screen.getByText('What Scout knows')).toBeInTheDocument();
    expect(requested).toEqual([URL_A]);
    expect(container.textContent ?? '').not.toMatch(FORBIDDEN);
  });

  it('lists the facts whatever the per-seat record says, and does not show its reason', async () => {
    const recordReason = 'Scout has no knowledge record in this conversation.';
    answers[URL_A] = knowledge(
      [fact('Kenny favourite colour teal', { learned_here: false })],
      null,
      'not_recorded',
      recordReason,
    );
    render(<RemembersPanel roomId="room-a" seatId="scout" seatName="Scout" />);
    expect(await screen.findAllByTestId('known-fact')).toHaveLength(1);
    expect(screen.queryByText(recordReason)).toBeNull();
    expect(screen.queryByTestId('remembers-here')).toBeNull();
  });

  it('shows the Core’s sentence when there are no facts to list, or they could not be read', async () => {
    // Killed by: frontend/src/components/artifacts/RemembersPanel.tsx :: data.facts_reason ??
    // Becomes: null ??
    const empty = 'No facts are listed for Scout.';
    answers[URL_A] = knowledge([], empty);
    const { container, unmount } = render(
      <RemembersPanel roomId="room-a" seatId="scout" seatName="Scout" />,
    );
    expect(await screen.findByTestId('remembers-reason')).toHaveTextContent(empty);
    expect(screen.queryByTestId('known-fact')).toBeNull();
    expect(container.textContent ?? '').not.toMatch(ABSENCE_CLAIM);
    unmount();

    const unread = 'What Scout knows could not be read, so it cannot be shown.';
    answers[URL_A] = knowledge(null, unread, 'unreadable');
    render(<RemembersPanel roomId="room-a" seatId="scout" seatName="Scout" />);
    expect(await screen.findByTestId('remembers-reason')).toHaveTextContent(unread);
  });

  // Without the Core's reason the panel knows only that the list is empty -- not that the
  // clone knows nothing (#1366).
  it('says only that none are listed when the Core gives no reason', async () => {
    answers[URL_A] = knowledge([], null);
    const { container } = render(<RemembersPanel roomId="room-a" seatId="scout" seatName="Scout" />);
    expect(await screen.findByTestId('remembers-reason')).toHaveTextContent(
      'No facts are listed for Scout.',
    );
    expect(container.textContent ?? '').not.toMatch(ABSENCE_CLAIM);
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
    answers[URL_A] = knowledge([fact('tide is a rhythm')]);
    answers['/api/rooms/room-a/knowledge?agent_id=critic'] = knowledge([fact('ropes fray')]);
    const { rerender } = render(<RemembersPanel roomId="room-a" seatId="scout" />);
    await screen.findByText(/tide is a rhythm/);
    rerender(<RemembersPanel roomId="room-a" seatId="critic" />);
    await screen.findByText(/ropes fray/);
    expect(screen.queryByText(/tide is a rhythm/)).toBeNull();
  });
});

describe('RemembersPanel: Correct and Forget (#1638 step 3)', () => {
  const teal = fact('Kenny favourite colour teal', { fact_id: 'mem_teal' });
  const MEMORY_URL = '/api/agents/scout/memory/mem_teal';

  const openMenu = async () => {
    fireEvent.click(await screen.findByTestId('fact-actions'));
  };

  it('Forget asks first, then retracts and reads the list again', async () => {
    // Killed by: frontend/src/components/artifacts/RemembersPanel.tsx :: const onChanged = () => setEdits((n) => n + 1);
    // Becomes: const onChanged = () => setEdits((n) => n);
    answers[URL_A] = knowledge([teal]);
    render(<RemembersPanel roomId="room-a" seatId="scout" seatName="Scout" />);
    await openMenu();
    fireEvent.click(screen.getByTestId('fact-forget'));

    expect(screen.getByTestId('fact-forget-confirm')).toHaveTextContent(
      'Scout will stop using this. Forget it?',
    );
    expect(calls.filter((c) => c.method !== 'GET')).toEqual([]);

    answers[URL_A] = knowledge([], 'No facts are listed for Scout.');
    fireEvent.click(screen.getByTestId('fact-forget-yes'));

    expect(await screen.findByText('No facts are listed for Scout.')).toBeInTheDocument();
    expect(calls.filter((c) => c.method !== 'GET')).toEqual([
      { url: MEMORY_URL, method: 'DELETE', body: undefined },
    ]);
    expect(requested).toEqual([URL_A, URL_A]);
  });

  it('Correct sends the new value and shows the corrected fact', async () => {
    // Killed by: frontend/src/lib/roomDock.ts :: body: JSON.stringify({ value }),
    // Becomes: body: JSON.stringify({ value: '' }),
    answers[URL_A] = knowledge([teal]);
    render(<RemembersPanel roomId="room-a" seatId="scout" seatName="Scout" />);
    await openMenu();
    fireEvent.click(screen.getByTestId('fact-correct'));
    const input = screen.getByTestId('fact-correct-input');
    expect(input).toHaveValue('teal');
    fireEvent.change(input, { target: { value: 'navy' } });

    answers[URL_A] = knowledge([
      fact('Kenny favourite colour navy', { fact_id: 'mem_navy', origin: 'corrected' }),
    ]);
    fireEvent.click(screen.getByTestId('fact-correct-save'));

    expect(await screen.findByText('Kenny favourite colour navy')).toBeInTheDocument();
    expect(calls.filter((c) => c.method !== 'GET')).toEqual([
      { url: MEMORY_URL, method: 'PATCH', body: { value: 'navy' } },
    ]);
    expect(screen.getByTestId('fact-origin')).toHaveTextContent('corrected');
  });

  it('a blank correction is refused here and sends nothing', async () => {
    answers[URL_A] = knowledge([teal]);
    render(<RemembersPanel roomId="room-a" seatId="scout" seatName="Scout" />);
    await openMenu();
    fireEvent.click(screen.getByTestId('fact-correct'));
    fireEvent.change(screen.getByTestId('fact-correct-input'), { target: { value: '   ' } });
    fireEvent.click(screen.getByTestId('fact-correct-save'));

    expect(screen.getByTestId('fact-edit-refusal')).toHaveTextContent(
      'Write what it should say instead.',
    );
    expect(calls.filter((c) => c.method !== 'GET')).toEqual([]);
  });

  // Each refusal the Core can give, and a runtime that cannot be reached: the panel's own
  // sentence, never the Core's detail, the status, the fact id or the transport's words.
  const PLAIN = /mem_|\/api\/|\b[45]\d\d\b|Error|Failed to fetch|detail|Traceback|KeyError/i;

  it.each([
    [404, "This fact is no longer in Scout's memory. It may have been forgotten or corrected already."],
    [409, "Scout's memory could not be read, so nothing was changed."],
    [500, 'The change could not be made. Try again.'],
  ])('a %i refusal reads as a plain sentence and keeps the fact listed', async (status, said) => {
    // Killed by: frontend/src/components/artifacts/RemembersPanel.tsx :: if (status === 404) return fmt(copy.factGone, { name });
    // Becomes: if (status === 404) return copy.editFailed;
    answers[URL_A] = knowledge([teal]);
    edits[`DELETE ${MEMORY_URL}`] = {
      status,
      body: { detail: 'KeyError: mem_teal at /home/scout/memory.json' },
    };
    const { container } = render(<RemembersPanel roomId="room-a" seatId="scout" seatName="Scout" />);
    await openMenu();
    fireEvent.click(screen.getByTestId('fact-forget'));
    fireEvent.click(screen.getByTestId('fact-forget-yes'));

    expect(await screen.findByTestId('fact-edit-refusal')).toHaveTextContent(said);
    expect(container.textContent ?? '').not.toMatch(PLAIN);
    expect(screen.getAllByTestId('known-fact')).toHaveLength(1);
    expect(requested).toEqual([URL_A]);
  });

  it('an unreachable runtime reads as a plain sentence', async () => {
    // Killed by: frontend/src/lib/roomDock.ts :: return { ok: false, status: null };
    // Becomes: return { ok: true };
    answers[URL_A] = knowledge([teal]);
    const { container } = render(<RemembersPanel roomId="room-a" seatId="scout" seatName="Scout" />);
    await openMenu();
    fireEvent.click(screen.getByTestId('fact-forget'));
    const stubbed = globalThis.fetch;
    vi.stubGlobal('fetch', async () => {
      throw new TypeError('Failed to fetch');
    });
    fireEvent.click(screen.getByTestId('fact-forget-yes'));

    expect(await screen.findByTestId('fact-edit-refusal')).toHaveTextContent(
      'The change could not be made. Try again.',
    );
    expect(container.textContent ?? '').not.toMatch(PLAIN);
    vi.stubGlobal('fetch', stubbed);
  });

  it('says every refusal in Korean when the screen is Korean', () => {
    const copy = ko.dock.remembers;
    for (const status of [400, 404, 409, 500, null]) {
      const said = editRefusedSentence(status, '스카우트', copy);
      expect(said).toMatch(/[가-힣]/);
      expect(said).not.toMatch(/[A-Za-z]/);
    }
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
