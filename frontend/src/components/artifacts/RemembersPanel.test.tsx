import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest';
import { fireEvent, render, screen, within } from '@testing-library/react';
import {
  RemembersPanel,
  editRefusedSentence,
  readFailedSentence,
  workedOutFromSentence,
} from './RemembersPanel';
import type { KnownFact, SeatKnowledge, WorkedOut } from '../../lib/roomDock';
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
  last_seen_at: '2026-09-26T00:00:00Z',
  // Each fact its own relation unless a test puts two under one (#2332).
  relation: `rel_${statement.length}${statement.charCodeAt(0)}`,
  functional: false,
  ask_functional: false,
  quote: null,
  replaced: [],
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
      'Kenny favourite colour tealsavedHistory',
      'Project X uses Postgres 16correctedHistory',
    ]);
    expect(within(elsewhere).getAllByTestId('known-fact').map((i) => i.textContent)).toEqual([
      'Kenny works at HanbittoldHistory',
    ]);
    expect(screen.getByText('What Scout knows')).toBeInTheDocument();
    expect(requested).toEqual([URL_A]);
    expect(container.textContent ?? '').not.toMatch(FORBIDDEN);
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
    answers[URL_A] = knowledge(null, unread);
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

describe('RemembersPanel: what the clone worked out from its facts (#1870)', () => {
  const postgres = fact('Project X uses Postgres 16', { fact_id: 'mem_pg' });
  const hanbit = fact('Kenny works at Hanbit', {
    fact_id: 'mem_hb',
    learned_here: false,
  });
  const withWorkedOut = (
    worked_out: WorkedOut[] | null,
    facts: KnownFact[] | null = [postgres, hanbit],
  ) => ({ ...knowledge(facts), worked_out }) as SeatKnowledge;

  it('lists each worked-out statement after the facts, with the facts it rests on', async () => {
    // Killed by: frontend/src/components/artifacts/RemembersPanel.tsx :: const workedOut: WorkedOut[] = (facts && data.worked_out) ?? [];
    // Becomes: const workedOut: WorkedOut[] = [];
    answers[URL_A] = withWorkedOut([
      { statement: "Project X's database is relational", because: ['mem_pg'] },
      { statement: 'Kenny works in Seoul', because: ['mem_hb', 'mem_pg'] },
    ]);
    const { container } = render(
      <RemembersPanel roomId="room-a" seatId="scout" seatName="Scout" />,
    );

    const group = await screen.findByTestId('remembers-worked-out');
    expect(group).toHaveTextContent('Worked out from these facts');
    const lines = within(group).getAllByTestId('worked-out-line');
    expect(lines.map((l) => l.textContent)).toEqual([
      "Project X's database is relationalworked outFrom: Project X uses Postgres 16",
      'Kenny works in Seoulworked outFrom: Kenny works at Hanbit; Project X uses Postgres 16',
    ]);
    // Corrected or forgotten through the facts it names, so it has no menu of its own.
    expect(within(group).queryByTestId('fact-actions')).toBeNull();
    // After both fact groups.
    const order = Array.from(container.querySelectorAll('section')).map((n) => n.dataset.testid);
    expect(order).toEqual(['remembers-here', 'remembers-elsewhere', 'remembers-worked-out']);
    expect(container.textContent ?? '').not.toMatch(FORBIDDEN);
  });

  it.each([
    ['nothing worked out', [] as WorkedOut[]],
    ['could not be worked out', null],
  ])('draws no group and says nothing more when %s', async (_label, worked) => {
    answers[URL_A] = withWorkedOut(worked);
    const { container } = render(
      <RemembersPanel roomId="room-a" seatId="scout" seatName="Scout" />,
    );
    await screen.findByTestId('remembers-here');
    expect(screen.queryByTestId('remembers-worked-out')).toBeNull();
    expect(screen.queryByTestId('remembers-reason')).toBeNull();
    expect(container.textContent ?? '').not.toMatch(ABSENCE_CLAIM);
  });

  it('names only facts it can say, never an id', () => {
    // Killed by: frontend/src/components/artifacts/RemembersPanel.tsx :: .filter((statement): statement is string => Boolean(statement));
    // Becomes: .map((statement) => statement ?? 'mem_unknown');
    expect(workedOutFromSentence(['mem_gone', 'mem_pg'], [postgres])).toBe(
      'From: Project X uses Postgres 16',
    );
    expect(workedOutFromSentence(['mem_gone'], [postgres])).toBeNull();
  });

  it('says it in Korean when the screen is Korean', () => {
    const copy = ko.dock.remembers;
    const said = workedOutFromSentence(['mem_pg', 'mem_hb'], [postgres, hanbit], copy);
    expect(said).toBe('근거: Project X uses Postgres 16; Kenny works at Hanbit');
    for (const line of [copy.workedOutHeading, copy.workedOut]) {
      expect(line).toMatch(/[가-힣]/);
      expect(line).not.toMatch(/[A-Za-z]/);
    }
    // 합니다체 headings are noun phrases; no line ends in a plain-form verb.
    expect(copy.workedOutHeading).not.toMatch(/다\.?$/);
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
      await screen.findByTestId('remembers-reason', undefined, {
        timeout: 5000,
      }),
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

describe('RemembersPanel: one value at a time (#2332)', () => {
  const ONE_VALUE_URL = '/api/agents/scout/relations/lives_in/functional';
  const busan = fact('Kenny lives in Busan', {
    fact_id: 'mem_busan',
    relation: 'lives_in',
    last_seen_at: '2026-09-01T00:00:00Z',
  });
  const seoul = fact('Kenny lives in Seoul', {
    fact_id: 'mem_seoul',
    relation: 'lives_in',
    origin: 'told',
    last_seen_at: '2026-09-02T00:00:00Z',
    ask_functional: true,
  });

  it('puts the values of one relation together and marks the one said last', async () => {
    // Killed by: frontend/src/components/artifacts/RemembersPanel.tsx :: (b.last_seen_at ?? '') > (a.last_seen_at ?? '') ? b : a,
    // Becomes: (b.last_seen_at ?? '') < (a.last_seen_at ?? '') ? b : a,
    // Killed by: frontend/src/components/artifacts/RemembersPanel.tsx :: const key = `${f.subject.toLowerCase()}\u0000${f.relation}`;
    // Becomes: const key = f.fact_id;
    answers[URL_A] = knowledge([busan, fact('Kenny likes jazz'), seoul]);
    render(<RemembersPanel roomId="room-a" seatId="scout" seatName="Scout" />);

    const group = await screen.findByTestId('relation-group');
    const rows = within(group).getAllByTestId('known-fact');
    expect(rows.map((r) => r.textContent)).toEqual([
      expect.stringContaining('Busan'),
      expect.stringContaining('Seoul'),
    ]);
    expect(within(rows[1]).getByTestId('fact-said-last')).toHaveTextContent('said last');
    expect(within(rows[0]).queryByTestId('fact-said-last')).toBeNull();
    expect(group).toHaveTextContent('Kenny · lives in');
    expect(screen.getAllByTestId('known-fact')).toHaveLength(3);
  });

  it('asks after a person changed a value, and Yes declares it; ignoring saves nothing', async () => {
    // Killed by: frontend/src/components/artifacts/RemembersPanel.tsx :: const outcome = await setRelationFunctional(seatId, first.relation, true);
    // Becomes: const outcome = await setRelationFunctional(seatId, first.relation, false);
    // Killed by: frontend/src/components/artifacts/RemembersPanel.tsx :: const asks = facts.some((f) => f.ask_functional);
    // Becomes: const asks = false;
    answers[URL_A] = knowledge([busan, seoul]);
    render(<RemembersPanel roomId="room-a" seatId="scout" seatName="Scout" />);

    const ask = await screen.findByTestId('relation-ask');
    expect(ask).toHaveTextContent(
      'Can lives in hold only one value at a time? Then a new one will replace the old.',
    );
    expect(ask.textContent).not.toMatch(FORBIDDEN);
    expect(calls.filter((c) => c.method !== 'GET')).toEqual([]);

    answers[URL_A] = knowledge([
      { ...busan, functional: true, ask_functional: false },
      { ...seoul, functional: true, ask_functional: false },
    ]);
    fireEvent.click(screen.getByTestId('relation-ask-yes'));

    await vi.waitFor(() => expect(screen.queryByTestId('relation-ask')).toBeNull());
    expect(calls.filter((c) => c.method !== 'GET')).toEqual([
      { url: ONE_VALUE_URL, method: 'PUT', body: { functional: true } },
    ]);
    expect(screen.getByTestId('relation-one-value')).toBeChecked();
  });

  it('No, several at once is sent and the question does not come back', async () => {
    // Killed by: frontend/src/components/artifacts/RemembersPanel.tsx :: const outcome = await markRelationSeveral(seatId, first.relation);
    // Becomes: const outcome = await setRelationFunctional(seatId, first.relation, true);
    answers[URL_A] = knowledge([busan, seoul]);
    render(<RemembersPanel roomId="room-a" seatId="scout" seatName="Scout" />);
    expect(await screen.findByTestId('relation-ask-no')).toHaveTextContent('No, several at once');

    // The Core keeps the answer, so every read after it no longer asks.
    answers[URL_A] = knowledge([busan, { ...seoul, ask_functional: false }]);
    fireEvent.click(screen.getByTestId('relation-ask-no'));

    await vi.waitFor(() => expect(screen.queryByTestId('relation-ask')).toBeNull());
    expect(calls.filter((c) => c.method !== 'GET')).toEqual([
      { url: '/api/agents/scout/relations/lives_in/several', method: 'POST', body: undefined },
    ]);
    expect(screen.getByTestId('relation-one-value')).not.toBeChecked();
  });

  it('the box in the group header turns it off again', async () => {
    // Killed by: frontend/src/components/artifacts/RemembersPanel.tsx :: const outcome = await setRelationFunctional(seatId, fact.relation, e.target.checked);
    // Becomes: const outcome = await setRelationFunctional(seatId, fact.relation, true);
    const home = { ...seoul, functional: true, ask_functional: false };
    answers[URL_A] = knowledge([home]);
    render(<RemembersPanel roomId="room-a" seatId="scout" seatName="Scout" />);

    const box = await screen.findByTestId('relation-one-value');
    expect(box).toBeChecked();
    fireEvent.click(box);

    await vi.waitFor(() =>
      expect(calls.filter((c) => c.method !== 'GET')).toEqual([
        { url: ONE_VALUE_URL, method: 'PUT', body: { functional: false } },
      ]),
    );
  });

  it('a refused save says so in plain words', async () => {
    edits[`PUT ${ONE_VALUE_URL}`] = { status: 500, body: { detail: 'OSError: disk full' } };
    answers[URL_A] = knowledge([busan, seoul]);
    render(<RemembersPanel roomId="room-a" seatId="scout" seatName="Scout" />);

    fireEvent.click(await screen.findByTestId('relation-ask-yes'));

    const refusal = await screen.findByTestId('relation-ask-refusal');
    expect(refusal).toHaveTextContent('That could not be saved. Try again.');
    expect(refusal.textContent).not.toMatch(/OSError|disk/);
  });

  it('asks in Korean in 합니다체', () => {
    const copy = ko.dock.remembers;
    for (const line of [copy.askOneValue, copy.oneValueOn, copy.oneValueFailed]) {
      expect(line).toMatch(/니까\?|니다\./);
    }
    expect(copy.askOneValue).toContain('{relation}');
  });
});

describe('RemembersPanel: a fact’s history and bringing a value back (#2332)', () => {
  const seoul = fact('Kenny lives in Seoul', {
    fact_id: 'mem_seoul',
    relation: 'lives_in',
    origin: 'told',
    quote: 'I moved to Seoul',
    replaced: [{ fact_id: 'mem_busan', value: 'Busan', said_at: '2026-09-01T00:00:00Z' }],
  });

  it('shows the words, and brings back what it replaced', async () => {
    // Killed by: frontend/src/components/artifacts/RemembersPanel.tsx :: const outcome = await restoreMemoryFact(seatId, old.fact_id);
    // Becomes: const outcome = await restoreMemoryFact(seatId, fact.fact_id);
    answers[URL_A] = knowledge([seoul]);
    render(<RemembersPanel roomId="room-a" seatId="scout" seatName="Scout" />);

    expect(screen.queryByTestId('fact-history')).toBeNull();
    fireEvent.click(await screen.findByTestId('fact-history-toggle'));

    const history = screen.getByTestId('fact-history');
    expect(within(history).getByTestId('fact-history-quote')).toHaveTextContent(
      'Learned from: “I moved to Seoul”',
    );
    expect(within(history).getByTestId('fact-history-replaced')).toHaveTextContent('Replaced Busan');
    expect(history.textContent).not.toMatch(FORBIDDEN);

    answers[URL_A] = knowledge([
      fact('Kenny lives in Busan', { fact_id: 'mem_back', relation: 'lives_in', origin: 'corrected' }),
    ]);
    fireEvent.click(within(history).getByTestId('fact-restore'));

    expect(await screen.findByText('Kenny lives in Busan')).toBeInTheDocument();
    expect(calls.filter((c) => c.method !== 'GET')).toEqual([
      { url: '/api/agents/scout/memory/mem_busan/restore', method: 'POST', body: undefined },
    ]);
  });

  it('reads the list again when the value can no longer come back, so no stale row stays', async () => {
    // Killed by: frontend/src/components/artifacts/RemembersPanel.tsx :: if (!outcome.ok && outcome.status === 409) {
    // Becomes: if (!outcome.ok && outcome.status === 418) {
    edits['POST /api/agents/scout/memory/mem_busan/restore'] = {
      status: 409,
      body: { detail: 'This value can no longer be brought back.' },
    };
    answers[URL_A] = knowledge([seoul]);
    render(<RemembersPanel roomId="room-a" seatId="scout" seatName="Scout" />);
    fireEvent.click(await screen.findByTestId('fact-history-toggle'));

    // Meanwhile another writer replaced Seoul with Daegu.
    answers[URL_A] = knowledge([
      fact('Kenny lives in Daegu', { fact_id: 'mem_daegu', relation: 'lives_in', origin: 'told' }),
    ]);
    fireEvent.click(screen.getByTestId('fact-restore'));

    expect(await screen.findByText('Kenny lives in Daegu')).toBeInTheDocument();
    expect(screen.queryByText('Kenny lives in Seoul')).toBeNull();
    expect(screen.queryByTestId('fact-restore')).toBeNull();
    expect(requested).toEqual([URL_A, URL_A]);
  });

  it('still tells the person after the re-read that their value did not come back', async () => {
    // Killed by: frontend/src/components/artifacts/RemembersPanel.tsx :: onStale(old.value);
    // Becomes: void old;
    // Killed by: frontend/src/components/artifacts/RemembersPanel.tsx :: setStale(null);
    // Becomes: void roomId;
    edits['POST /api/agents/scout/memory/mem_busan/restore'] = {
      status: 409,
      body: { detail: 'OSError mem_busan' },
    };
    answers[URL_A] = knowledge([seoul]);
    answers['/api/rooms/room-a/knowledge?agent_id=guardian'] = knowledge([
      fact('Guardian watches the door'),
    ]);
    const { rerender } = render(<RemembersPanel roomId="room-a" seatId="scout" seatName="Scout" />);
    fireEvent.click(await screen.findByTestId('fact-history-toggle'));

    answers[URL_A] = knowledge([
      fact('Kenny lives in Daegu', { fact_id: 'mem_daegu', relation: 'lives_in', origin: 'told' }),
    ]);
    fireEvent.click(screen.getByTestId('fact-restore'));

    await screen.findByText('Kenny lives in Daegu');
    const notice = screen.getByTestId('remembers-notice');
    expect(notice).toHaveTextContent('Busan could not be brought back: it changed meanwhile.');
    expect(notice.textContent).not.toMatch(/OSError|mem_|409/);

    // Switching to another clone clears the notice so it never speaks for the other clone.
    rerender(<RemembersPanel roomId="room-a" seatId="guardian" seatName="Guardian" />);
    await screen.findByText('Guardian watches the door');
    expect(screen.queryByTestId('remembers-notice')).toBeNull();
  });

  it('says plainly when nothing was replaced, and when a restore is refused', async () => {
    // Killed by: frontend/src/components/artifacts/RemembersPanel.tsx :: if (!outcome.ok) setRefused(true);
    // Becomes: if (!outcome.ok) setRefused(false);
    edits['POST /api/agents/scout/memory/mem_busan/restore'] = {
      status: 409,
      body: { detail: 'Only a value another one replaced can be brought back.' },
    };
    answers[URL_A] = knowledge([seoul, fact('Kenny likes jazz', { fact_id: 'mem_jazz' })]);
    render(<RemembersPanel roomId="room-a" seatId="scout" seatName="Scout" />);

    const [first, second] = await screen.findAllByTestId('fact-history-toggle');
    fireEvent.click(second);
    expect(screen.getByTestId('fact-history-none')).toHaveTextContent('It replaced no earlier value.');
    fireEvent.click(first);
    fireEvent.click(screen.getByTestId('fact-restore'));

    const refusal = await screen.findByTestId('fact-restore-refusal');
    expect(refusal).toHaveTextContent(
      'Scout could not bring that value back. It may have been changed already.',
    );
    expect(refusal.textContent).not.toMatch(/409|mem_/);
  });
});

describe('RemembersPanel: near-matching subjects merged and kept separate (uGraph §6 Q2)', () => {
  const merged = fact('김지현 works at Hanbit', {
    fact_id: 'mem_jh',
    subject: '김지현',
    predicate: 'works_at',
    object_value: 'Hanbit',
    relation: 'works_at',
    merged_from: '지현',
  });

  it('shows the merged subject name and lets a person split it back out', async () => {
    // Killed by: frontend/src/components/artifacts/RemembersPanel.tsx :: const outcome = await unmergeSubject(seatId, source);
    // Becomes: const outcome = await unmergeSubject(seatId, '');
    // Killed by: frontend/src/lib/roomDock.ts :: body: JSON.stringify({ subject }),
    // Becomes: body: JSON.stringify({ subject: '' }),
    // Killed by: frontend/src/i18n/locales/en/dock.json :: "mergedFrom": "Merged with {name}",
    // Becomes: "mergedFrom": "Merged",
    // Killed by: frontend/src/i18n/locales/en/dock.json :: "keepSeparate": "Keep separate"
    // Becomes: "keepSeparate": "Separate"
    answers[URL_A] = knowledge([merged]);
    render(<RemembersPanel roomId="room-a" seatId="scout" seatName="Scout" />);

    const banner = await screen.findByTestId('fact-merged-from');
    expect(banner).toHaveTextContent('Merged with 지현');
    const button = within(banner).getByTestId('fact-unmerge');
    expect(button).toHaveTextContent('Keep separate');

    answers[URL_A] = knowledge([
      fact('김지현 works at Hanbit', {
        fact_id: 'mem_kjh',
        subject: '김지현',
        predicate: 'works_at',
        object_value: 'Hanbit',
        relation: 'works_at',
        merged_from: null,
      }),
      fact('지현 works at Hanbit', {
        fact_id: 'mem_jh_only',
        subject: '지현',
        predicate: 'works_at',
        object_value: 'Hanbit',
        relation: 'works_at',
        merged_from: null,
      }),
    ]);
    fireEvent.click(button);

    await vi.waitFor(() => expect(screen.queryByTestId('fact-merged-from')).toBeNull());
    expect(calls.filter((c) => c.method !== 'GET')).toEqual([
      { url: '/api/agents/scout/subjects/unmerge', method: 'POST', body: { subject: '지현' } },
    ]);
    expect(screen.getAllByTestId('known-fact')).toHaveLength(2);
  });

  it('shows a plain refusal when keeping separate fails, and words Korean copy in Korean', async () => {
    // Killed by: frontend/src/i18n/locales/ko/dock.json :: "mergedFrom": "{name}와(과) 같은 대상으로 합침",
    // Becomes: "mergedFrom": "merged",
    // Killed by: frontend/src/i18n/locales/ko/dock.json :: "keepSeparate": "따로 두기"
    // Becomes: "keepSeparate": "separate"
    edits['POST /api/agents/scout/subjects/unmerge'] = {
      status: 500,
      body: { detail: 'OSError: disk I/O' },
    };
    answers[URL_A] = knowledge([merged]);
    render(<RemembersPanel roomId="room-a" seatId="scout" seatName="Scout" />);

    fireEvent.click(await screen.findByTestId('fact-unmerge'));

    const refusal = await screen.findByTestId('fact-edit-refusal');
    expect(refusal).toHaveTextContent('The change could not be made. Try again.');
    expect(refusal.textContent).not.toMatch(/OSError|disk/);

    const copy = ko.dock.remembers;
    expect(copy.mergedFrom).toContain('{name}');
    for (const line of [copy.mergedFrom.replace('{name}', ''), copy.keepSeparate]) {
      expect(line).toMatch(/[가-힣]/);
      expect(line).not.toMatch(/[A-Za-z]/);
    }
  });
});

