import { afterEach, beforeEach, describe, it, expect, vi } from 'vitest';
import { render, screen, waitFor, fireEvent, within } from '@testing-library/react';
import { StoryView } from './StoryView';
import { LocaleProvider, fmt } from '../../i18n';
import { en } from '../../i18n/en';
import { ko } from '../../i18n/ko';
import { expectPlain } from '../../test/plainCopy';
import type { ProposalView, SceneView, StoryOverview } from '../../lib/artifactLibrary';

const STORY = en.dock.story;

const board = { entries: [], changes: [], continuity: null, findings: [], pending: [] } satisfies Partial<SceneView>;

const proposal = (over: Partial<ProposalView> = {}): ProposalView => ({
  id: 'p-001',
  status: 'pending',
  kind: 'characters',
  entry_id: 'mara',
  entry_name: 'Mara',
  proposed_at: '2026-09-24T10:00:00Z',
  proposed_in: { room_id: 'room-w', title: 'Writing room', exists: true },
  agent_id: 'writer',
  note: 'She loses the map on the train.',
  evidence: [
    { scene_id: 'ch01.s02', scene_title: 'The platform', quote: 'The map slid under the seat.', still_in_scene: true },
    { scene_id: 'ch01.s03', scene_title: 'The tunnel', quote: 'She had no map now.', still_in_scene: false },
  ],
  changes: [{ what: 'has_map', at: 'ch01.s02', before: true, after: false, placed: true }],
  diff: [
    '--- characters/mara.yaml (now)',
    '+++ characters/mara.yaml (if approved)',
    '@@ -3,2 +3,5 @@',
    ' state:',
    '   has_map: true',
    '+progressions:',
  ],
  blocked: null,
  digest: 'd-shown',
  decided_at: null,
  decided_in: null,
  reason: null,
  ...over,
});

const overview = (over: Partial<StoryOverview> = {}): StoryOverview => ({
  story_id: 'night-train',
  title: 'Night Train',
  logline: 'A courier and a map.',
  genre: 'Thriller',
  writer: { room_id: 'room-w', title: 'Writing room', exists: true },
  decide_note: null,
  outline: {
    chapters: [
      {
        id: 'ch01',
        title: 'Departure',
        act: 'Act I',
        functions: [],
        twist: false,
        climax: false,
        scenes: [
          {
            id: 'ch01.s01',
            title: 'The ticket',
            summary: 'Mara buys a ticket.',
            story_time: 2,
            characters: [],
            places: [],
            written: true,
            ...board,
          },
          {
            id: 'ch01.s02',
            title: 'The platform',
            summary: '',
            story_time: 3,
            characters: [],
            places: [],
            written: true,
            ...board,
          },
          {
            id: 'ch01.s03',
            title: 'Years before',
            summary: '',
            story_time: 1,
            characters: [],
            places: [],
            written: false,
            ...board,
          },
        ],
      },
    ],
    story_order: ['ch01.s03', 'ch01.s01', 'ch01.s02'],
    assumptions: ['ch01.s02 is placed after ch01.s01.'],
  },
  outline_note: null,
  board_note: null,
  codex: [
    {
      kind: 'characters',
      label: 'Characters',
      entries: [
        {
          id: 'mara',
          name: 'Mara',
          aliases: ['the courier'],
          profile: 'A courier.',
          state: { has_map: true },
          progressions: [],
          looks: ['red coat'],
        },
      ],
    },
    { kind: 'places', label: 'Places', entries: [] },
  ],
  codex_unreadable: [],
  pending: [proposal()],
  decided: [
    proposal({
      id: 'p-000',
      status: 'applied',
      decided_at: '2026-09-23T10:00:00Z',
      decided_in: 'story_view',
    }),
  ],
  proposals_unreadable: [],
  ...over,
});

interface Call {
  url: string;
  method: string;
  body: unknown;
}
type Answer = { status: number; body: unknown };

const SHOW = 'GET /api/artifacts/library/stories/night-train';
const APPROVE = 'POST /api/artifacts/library/stories/night-train/proposals/p-001/approve';
const REJECT = 'POST /api/artifacts/library/stories/night-train/proposals/p-001/reject';

let answers: Record<string, Answer[] | Answer>;
let calls: Call[];

const answer = (key: string): Answer | undefined => {
  const a = answers[key];
  if (Array.isArray(a)) return a.length > 1 ? a.shift() : a[0];
  return a;
};

beforeEach(() => {
  answers = {};
  calls = [];
  vi.spyOn(console, 'error').mockImplementation(() => {});
  vi.stubGlobal(
    'fetch',
    vi.fn(async (url: string, init?: RequestInit) => {
      const method = init?.method ?? 'GET';
      calls.push({ url, method, body: init?.body ? JSON.parse(String(init.body)) : undefined });
      const found = answer(`${method} ${url}`);
      if (!found) return new Response(JSON.stringify({ detail: `no stub for ${url}` }), { status: 404 });
      const body = typeof found.body === 'string' ? found.body : JSON.stringify(found.body);
      return new Response(body, { status: found.status });
    }),
  );
});

afterEach(() => {
  vi.unstubAllGlobals();
  vi.restoreAllMocks();
});

const ok = (body: unknown): Answer => ({ status: 200, body });

const renderView = () => {
  const onBack = vi.fn();
  render(<StoryView storyId="night-train" onBack={onBack} />);
  return { onBack };
};

const posted = (key: string) => calls.filter((c) => `${c.method} ${c.url}` === key).map((c) => c.body);

describe('StoryView, a story in Files (#1560)', () => {
  it('shows a waiting change as it is now and as it would be, with its quotes and where it came from', async () => {
    answers[SHOW] = ok(overview());
    renderView();

    const card = await screen.findByTestId('story-proposal');
    expect(within(card).getByText('Mara')).toBeInTheDocument();
    expect(within(card).getByText(/Proposed in “Writing room”/)).toBeInTheDocument();
    expect(within(card).getByText('She loses the map on the train.')).toBeInTheDocument();

    const change = within(card).getByTestId('story-change');
    expect(within(change).getByText('has_map')).toBeInTheDocument();
    expect(within(change).getByText('after “The platform”')).toBeInTheDocument();
    expect(within(change).getByTestId('story-change-before')).toHaveTextContent('true');
    expect(within(change).getByTestId('story-change-after')).toHaveTextContent('false');

    const quotes = within(card).getAllByTestId('story-evidence');
    expect(quotes[0]).toHaveTextContent('The map slid under the seat.');
    expect(quotes[0]).not.toHaveTextContent(STORY.quoteGone);
    expect(quotes[1]).toHaveTextContent('She had no map now.');
    expect(quotes[1]).toHaveTextContent(STORY.quoteGone);

    const diff = within(card).getByTestId('story-diff');
    expect(diff).toHaveTextContent('+progressions:');
    expect(diff).toHaveTextContent('--- characters/mara.yaml (now)');

    expect(within(card).getByTestId('story-approve')).toBeEnabled();
    expect(within(card).getByTestId('story-reject')).toBeEnabled();
  });

  it('says a value the entry does not have yet is not set, and when a scene cannot be placed', async () => {
    answers[SHOW] = ok(
      overview({
        pending: [
          proposal({ changes: [{ what: 'mood', at: 'ch09.s01', before: null, after: 'wary', placed: false }] }),
        ],
      }),
    );
    renderView();

    const change = await screen.findByTestId('story-change');
    expect(within(change).getByTestId('story-change-before')).toHaveTextContent(STORY.notSet);
    expect(change).toHaveTextContent(STORY.unplaced);
  });

  it('approves the change as it was shown, says so, and reads the story again', async () => {
    answers[SHOW] = [ok(overview()), ok(overview({ pending: [] }))];
    answers[APPROVE] = ok({
      proposal_id: 'p-001',
      decision: 'applied',
      notes: ['characters/mara.yaml was rewritten with the change, and the comments it had were not kept.'],
    });
    renderView();

    fireEvent.click(await screen.findByTestId('story-approve'));

    const notice = await screen.findByTestId('story-view-notice');
    expect(notice).toHaveTextContent(fmt(STORY.applied, { name: 'Mara' }));
    expect(notice).toHaveTextContent('the comments it had were not kept');
    expect(posted(APPROVE)).toEqual([{ seen_digest: 'd-shown' }]);
    expect(await screen.findByText(STORY.noneWaiting)).toBeInTheDocument();
    expect(posted(REJECT)).toEqual([]);
  });

  it('rejects with the reason given', async () => {
    answers[SHOW] = ok(overview());
    answers[REJECT] = ok({ proposal_id: 'p-001', decision: 'rejected', notes: [] });
    renderView();

    fireEvent.click(await screen.findByTestId('story-reject'));
    fireEvent.change(screen.getByTestId('story-reject-reason'), { target: { value: 'She keeps it.' } });
    fireEvent.click(screen.getByTestId('story-confirm-reject'));

    expect(await screen.findByTestId('story-view-notice')).toHaveTextContent(fmt(STORY.rejected, { name: 'Mara' }));
    expect(posted(REJECT)).toEqual([{ seen_digest: 'd-shown', reason: 'She keeps it.' }]);
    expect(posted(APPROVE)).toEqual([]);
  });

  it("shows the Core's refusal in its own words", async () => {
    const refusal =
      "Proposal 'p-001' changed after it was shown, so nothing was changed. Look at it again before deciding.";
    answers[SHOW] = ok(overview());
    answers[APPROVE] = { status: 400, body: { detail: refusal } };
    renderView();

    fireEvent.click(await screen.findByTestId('story-approve'));

    expect(await screen.findByTestId('story-view-notice')).toHaveTextContent(refusal);
  });

  it('a decision that failed without a reason says so plainly', async () => {
    answers[SHOW] = ok(overview());
    answers[APPROVE] = { status: 502, body: '<html>Bad Gateway</html>' };
    renderView();

    fireEvent.click(await screen.findByTestId('story-approve'));

    const notice = await screen.findByTestId('story-view-notice');
    expect(notice).toHaveTextContent(STORY.decideFailed);
    expectPlain(notice.textContent);
  });

  it('a decision from a window the Core did not open shows its refusal and offers a confirmed window', async () => {
    // Killed by: frontend/src/components/artifacts/StoryView.tsx :: setUnconfirmed(err instanceof CoreFailure && err.status === 403);
    // Becomes: setUnconfirmed(false);
    const refusal = 'Only you can make this decision, and this window could not show that it is you.';
    answers[SHOW] = ok(overview());
    answers[APPROVE] = { status: 403, body: { detail: refusal } };
    answers['POST /api/person/window'] = ok({ opened: true });
    renderView();

    fireEvent.click(await screen.findByTestId('story-approve'));

    const notice = await screen.findByTestId('story-view-notice');
    expect(notice).toHaveTextContent(refusal);
    fireEvent.click(within(notice).getByTestId('story-open-confirmed-window'));
    expect(await screen.findByText(STORY.windowOpened)).toBeInTheDocument();
    expect(calls.filter((c) => c.url === '/api/person/window')).toHaveLength(1);
  });

  it('says plainly when no confirmed window could be opened', async () => {
    // Killed by: frontend/src/components/artifacts/StoryView.tsx :: setNotice([plainFailure(err, t.windowFailed)]);
    // Becomes: setNotice([String(err)]);
    answers[SHOW] = ok(overview());
    answers[APPROVE] = { status: 403, body: { detail: 'Only you can make this decision.' } };
    answers['POST /api/person/window'] = { status: 502, body: '<html>Bad Gateway</html>' };
    renderView();

    fireEvent.click(await screen.findByTestId('story-approve'));
    fireEvent.click(await screen.findByTestId('story-open-confirmed-window'));

    expect(await screen.findByText(STORY.windowFailed)).toBeInTheDocument();
    expectPlain(screen.getByTestId('story-view-notice').textContent);
  });

  it('offers no confirmed window for a refusal that is not about the window', async () => {
    // Killed by: frontend/src/components/artifacts/StoryView.tsx :: setUnconfirmed(err instanceof CoreFailure && err.status === 403);
    // Becomes: setUnconfirmed(err instanceof CoreFailure);
    answers[SHOW] = ok(overview());
    answers[APPROVE] = { status: 400, body: { detail: 'It changed after it was shown.' } };
    renderView();

    fireEvent.click(await screen.findByTestId('story-approve'));

    expect(await screen.findByTestId('story-view-notice')).toHaveTextContent('It changed after it was shown.');
    expect(screen.queryByTestId('story-open-confirmed-window')).toBeNull();
  });

  it('when a decision cannot be saved, says why and offers no button that would fail', async () => {
    const note =
      'No conversation is writing “Night Train”, so a decision cannot be saved. Open the story in a conversation, then decide here.';
    answers[SHOW] = ok(overview({ writer: null, decide_note: note }));
    renderView();

    expect(await screen.findByTestId('story-decide-note')).toHaveTextContent(note);
    expect(screen.getByTestId('story-approve')).toBeDisabled();
    expect(screen.getByTestId('story-reject')).toBeDisabled();
  });

  it('a change that cannot be approved says why, and can still be rejected', async () => {
    const blocked =
      "It cannot be approved: the entry 'mara' changed after this was proposed. Reject it, and ask for the change again if it is still wanted.";
    answers[SHOW] = ok(overview({ pending: [proposal({ blocked })] }));
    renderView();

    expect(await screen.findByTestId('story-blocked')).toHaveTextContent(blocked);
    expect(screen.getByTestId('story-approve')).toBeDisabled();
    expect(screen.getByTestId('story-reject')).toBeEnabled();
  });

  it('shows the outline in reading order, and in the order it happens with what was assumed', async () => {
    answers[SHOW] = ok(overview());
    renderView();

    const outline = await screen.findByTestId('story-outline');
    const ids = () =>
      within(outline)
        .getAllByTestId('story-scene')
        .map((s) => s.getAttribute('data-scene'));
    expect(ids()).toEqual(['ch01.s01', 'ch01.s02', 'ch01.s03']);
    expect(within(outline).getByText(/Act I · Departure/)).toBeInTheDocument();
    expect(within(outline).queryByTestId('story-assumptions')).not.toBeInTheDocument();

    fireEvent.click(within(outline).getByTestId('story-order-story'));
    expect(ids()).toEqual(['ch01.s03', 'ch01.s01', 'ch01.s02']);
    expect(within(outline).getByTestId('story-assumptions')).toHaveTextContent('ch01.s02 is placed after ch01.s01.');
  });

  it('shows the codex by kind, and says which kinds have no entries', async () => {
    answers[SHOW] = ok(overview());
    renderView();

    const groups = await screen.findAllByTestId('story-codex-group');
    expect(groups.map((g) => g.getAttribute('data-kind'))).toEqual(['characters', 'places']);
    expect(within(groups[0]).getByTestId('story-entry')).toHaveTextContent('Mara');
    expect(within(groups[0]).getByTestId('story-entry')).toHaveTextContent('has_map: true');
    expect(groups[1]).toHaveTextContent(fmt(STORY.noEntries, { label: 'places' }));
  });

  it('states every absence: no outline, nothing waiting, nothing decided, files that do not read', async () => {
    answers[SHOW] = ok(
      overview({
        outline: null,
        outline_note: 'This story has no outline yet.',
        pending: [],
        decided: [],
        proposals_unreadable: [{ file: 'proposals/p-009.yaml', reason: 'It is not a proposal.' }],
        codex_unreadable: [{ file: 'codex/characters/old', reason: 'The folder was not read.' }],
      }),
    );
    renderView();

    expect(await screen.findByTestId('story-outline-note')).toHaveTextContent('This story has no outline yet.');
    expect(screen.getByTestId('story-pending')).toHaveTextContent(STORY.noneWaiting);
    expect(screen.getByTestId('story-proposals-unreadable')).toHaveTextContent('It is not a proposal.');
    expect(screen.getByTestId('story-decided')).toHaveTextContent(STORY.noneDecided);
    // The codex reports folders as well as files, and its heading says so (#1595). Proposals
    // are only ever files.
    // Killed by: frontend/src/components/artifacts/StoryView.tsx :: heading={t.codexUnreadable}
    // Becomes: heading={t.unreadable}
    const codex = screen.getByTestId('story-codex-unreadable');
    expect(codex).toHaveTextContent(STORY.codexUnreadable);
    expect(codex).toHaveTextContent('The folder was not read.');
    expect(screen.getByTestId('story-proposals-unreadable')).toHaveTextContent(STORY.unreadable);
  });

  it('lists decided changes with where they were decided', async () => {
    answers[SHOW] = ok(
      overview({
        decided: [
          proposal({ id: 'p-000', status: 'applied', decided_in: 'story_view', decided_at: '2026-09-23T10:00:00Z' }),
          proposal({
            id: 'p-002',
            status: 'rejected',
            decided_in: 'conversation',
            reason: 'She keeps it.',
            proposed_in: { room_id: 'gone', title: null, exists: false },
          }),
        ],
      }),
    );
    renderView();

    const items = await screen.findAllByTestId('story-decided-item');
    expect(items[0]).toHaveTextContent('Applied here');
    expect(items[1]).toHaveTextContent('Rejected in a conversation');
    expect(items[1]).toHaveTextContent('She keeps it.');
    expect(items[1]).toHaveTextContent('Proposed in a conversation that no longer exists');
  });

  it('a story that could not be read says so plainly', async () => {
    answers[SHOW] = { status: 500, body: '<html>oops</html>' };
    renderView();

    const alert = await screen.findByRole('alert');
    expect(alert).toHaveTextContent(STORY.loadFailed);
    expectPlain(alert.querySelector('p')?.textContent);
  });

  it('goes back to the files', async () => {
    answers[SHOW] = ok(overview());
    const { onBack } = renderView();

    fireEvent.click(await screen.findByTestId('story-view-back'));
    await waitFor(() => expect(onBack).toHaveBeenCalled());
  });
});

/** A story whose outline is a board: the codex entries, changes, checks and waiting changes of each scene. */
const boardStory = (): StoryOverview => {
  const base = overview();
  const [chapter] = base.outline!.chapters;
  return overview({
    outline: {
      ...base.outline!,
      chapters: [
        {
          ...chapter,
          functions: ['setup', 'reversal', 'no-such-function'],
          twist: true,
          climax: false,
          scenes: [
            {
              ...chapter.scenes[0],
              entries: [
                { kind: 'characters', id: 'mara', name: 'Mara', how: 'listed', in_codex: true },
                { kind: 'characters', id: 'ivo', name: 'Ivo', how: 'listed', in_codex: false },
              ],
              continuity: 'checked',
            },
            {
              ...chapter.scenes[1],
              entries: [{ kind: 'threads', id: 'the-map', name: 'The map', how: 'planted', in_codex: true }],
              changes: [
                {
                  kind: 'characters',
                  entry_id: 'mara',
                  entry_name: 'Mara',
                  set: { has_map: false },
                  add_looks: ['torn sleeve'],
                  remove_looks: [],
                  note: null,
                },
              ],
              continuity: 'checked',
              findings: [{ quote: 'Her blue coat', note: 'The codex says her coat is red.' }],
              pending: ['p-001'],
            },
            { ...chapter.scenes[2], continuity: 'unread' },
          ],
        },
      ],
    },
    codex: [
      ...base.codex,
      {
        kind: 'threads',
        label: 'Threads',
        entries: [
          { id: 'the-map', name: 'The map', aliases: [], profile: '', state: {}, progressions: [], looks: null },
        ],
      },
    ],
  });
};

describe('StoryView, the story board (§1.1 row 5)', () => {
  const scene = (id: string) => {
    const found = screen.getAllByTestId('story-scene').find((s) => s.getAttribute('data-scene') === id);
    if (!found) throw new Error(`no scene ${id}`);
    return found;
  };

  it("marks each chapter's function and twist, and each scene written or planned", async () => {
    // Killed by: frontend/src/components/artifacts/StoryView.tsx :: ...chapter.functions.map((f) => functions[f]).filter((f): f is string => Boolean(f)),
    // Becomes: ...chapter.functions,
    answers[SHOW] = ok(boardStory());
    renderView();

    const marks = await screen.findByTestId('story-chapter-marks');
    expect(marks).toHaveTextContent(STORY.board.functions.setup);
    expect(marks).toHaveTextContent(STORY.board.functions.reversal);
    expect(marks).toHaveTextContent(STORY.board.twist);
    expect(marks).not.toHaveTextContent(STORY.board.climax);
    // A function the screen has no word for is the arc's code, and is not shown.
    expect(marks).not.toHaveTextContent('no-such-function');
    expect(within(scene('ch01.s01')).getByTestId('story-scene-state')).toHaveTextContent(STORY.board.written);
    expect(within(scene('ch01.s03')).getByTestId('story-scene-state')).toHaveTextContent(STORY.board.planned);
  });

  it('shows the codex entries in a scene, and an entry clicked there is shown in the codex', async () => {
    // Killed by: frontend/src/components/artifacts/StoryView.tsx :: const isChosen = chosen !== null && chosen.kind === group.kind && chosen.id === e.id;
    // Becomes: const isChosen = false;
    answers[SHOW] = ok(boardStory());
    renderView();

    await screen.findByTestId('story-board');
    const chips = within(scene('ch01.s01')).getAllByTestId('story-scene-entry');
    expect(chips.map((c) => c.textContent)).toEqual(['Mara', fmt(STORY.board.notInCodex, { name: 'Ivo' })]);
    // An entry only proposed is named, but there is nothing in the codex to show.
    expect(chips[1].tagName).not.toBe('BUTTON');

    const mapChip = within(scene('ch01.s02')).getByTestId('story-scene-entry');
    expect(mapChip).toHaveTextContent(`The map · ${STORY.board.how.planted}`);
    fireEvent.click(mapChip);
    const chosen = screen
      .getAllByTestId('story-entry')
      .filter((e) => e.getAttribute('data-chosen') === 'true')
      .map((e) => e.getAttribute('data-entry'));
    expect(chosen).toEqual(['the-map']);
  });

  it('shows what the codex records at a scene, what the check found, and the changes waiting on it', async () => {
    answers[SHOW] = ok(boardStory());
    renderView();

    await screen.findByTestId('story-board');
    const platform = scene('ch01.s02');
    expect(within(platform).getByTestId('story-scene-changes')).toHaveTextContent('Mara: has_map: false');
    expect(within(platform).getByTestId('story-scene-changes')).toHaveTextContent(
      fmt(STORY.board.looksAdded, { looks: 'torn sleeve' }),
    );
    const findings = within(platform).getByTestId('story-scene-findings');
    expect(findings).toHaveTextContent('Her blue coat');
    expect(findings).toHaveTextContent('The codex says her coat is red.');
    // A scene with a finding does not also say it was clean.
    expect(within(platform).queryByTestId('story-scene-checked')).toBeNull();
    expect(within(platform).getByTestId('story-scene-waiting')).toHaveTextContent(
      `${fmt(STORY.board.waiting.one, { count: 1 })}: Mara`,
    );

    expect(within(scene('ch01.s01')).getByTestId('story-scene-checked')).toHaveTextContent(STORY.board.checkedClean);
    expect(within(scene('ch01.s03')).getByTestId('story-scene-unread')).toHaveTextContent(STORY.board.unread);
    expect(within(scene('ch01.s03')).queryByTestId('story-scene-waiting')).toBeNull();
  });

  it("says when the story's plan could not be read, in the Core's words", async () => {
    const note = 'The story plan (start.yaml) could not be read, so no scene shows what was checked.';
    answers[SHOW] = ok({ ...boardStory(), board_note: note });
    renderView();

    expect(await screen.findByTestId('story-board-note')).toHaveTextContent(note);
  });

  it('reads Korean when the screen is Korean', async () => {
    answers[SHOW] = ok(boardStory());
    render(
      <LocaleProvider hints={['ko-KR']}>
        <StoryView storyId="night-train" onBack={() => {}} />
      </LocaleProvider>,
    );

    const marks = await screen.findByTestId('story-chapter-marks');
    expect(marks).toHaveTextContent(ko.dock.story.board.twist);
    expect(screen.getByTestId('story-pending')).toHaveTextContent(ko.dock.story.waiting);
    expect(screen.getByTestId('story-view-back')).toHaveTextContent(ko.dock.story.back);
    expect(screen.queryByText(en.dock.story.waiting)).toBeNull();
  });

  const renderKorean = () =>
    render(
      <LocaleProvider hints={['ko-KR']}>
        <StoryView storyId="night-train" onBack={() => {}} />
      </LocaleProvider>,
    );

  it("words the Core's notes from their codes in Korean, not the Core's English", async () => {
    // Killed by: frontend/src/components/artifacts/StoryView.tsx :: {noteText(story.board_code, story.board_note, story, t)}
    // Becomes: {story.board_note}
    // Killed by: frontend/src/components/artifacts/StoryView.tsx :: {noteText(story.decide_code, story.decide_note, story, t)}
    // Becomes: {story.decide_note}
    const english = 'The record of how this story was started could not be read.';
    answers[SHOW] = ok({
      ...boardStory(),
      board_note: english,
      board_code: 'start_unreadable',
      writer: null,
      decide_note: 'No conversation is writing “Night Train”.',
      decide_code: 'no_writer',
    });
    renderKorean();

    const note = await screen.findByTestId('story-board-note');
    expect(note).toHaveTextContent(ko.dock.story.notes.start_unreadable);
    expect(note).not.toHaveTextContent(english);
    const decide = screen.getByTestId('story-decide-note');
    expect(decide).toHaveTextContent(fmt(ko.dock.story.notes.no_writer, { title: 'Night Train' }));
    expectPlain(decide.textContent ?? '');
  });

  it("words the outline's note from its code in Korean", async () => {
    // Killed by: frontend/src/components/artifacts/StoryView.tsx :: {noteText(story.outline_code, story.outline_note, story, t)}
    // Becomes: {story.outline_note}
    answers[SHOW] = ok(overview({ outline: null, outline_note: 'This story has no outline yet.', outline_code: 'no_outline' }));
    renderKorean();

    expect(await screen.findByTestId('story-outline-note')).toHaveTextContent(ko.dock.story.notes.no_outline);
  });

  it('shows the English sentence for a note code this head does not know', async () => {
    const note = 'A newer reason.';
    answers[SHOW] = ok({ ...boardStory(), board_note: note, board_code: 'from_a_newer_core' });
    renderKorean();

    expect(await screen.findByTestId('story-board-note')).toHaveTextContent(note);
  });

  it("words a refused decision from the Core's code in Korean", async () => {
    // Killed by: frontend/src/components/artifacts/StoryView.tsx :: setNotice([refusalText(err, t.decideFailed, t.refusals)]);
    // Becomes: setNotice([plainFailure(err, t.decideFailed)]);
    const english = 'The conversation “Writing room” is answering right now, so the decision was not saved.';
    answers[SHOW] = ok(overview());
    answers[APPROVE] = { status: 409, body: { detail: english, code: 'writer_busy' } };
    renderKorean();

    fireEvent.click(await screen.findByTestId('story-approve'));

    const notice = await screen.findByTestId('story-view-notice');
    expect(notice).toHaveTextContent(ko.dock.story.refusals.writer_busy);
    expect(notice).not.toHaveTextContent(english);
  });
});
