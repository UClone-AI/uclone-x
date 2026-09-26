import { afterEach, beforeEach, describe, it, expect, vi } from 'vitest';
import { render, screen, waitFor, fireEvent, within } from '@testing-library/react';
import { ArtifactLibrary, FILES_COPY } from './ArtifactLibrary';
import { expectPlain } from '../../test/plainCopy';
import type { ArtifactEntry, ArtifactSurvey } from '../../lib/artifactLibrary';

const SCOPE_NOTE =
  "This list shows the files in the workspace's artifacts and stories folders, and files a current conversation saved by name somewhere else.";

const entry = (over: Partial<ArtifactEntry>): ArtifactEntry => ({
  path: 'artifacts/report.md',
  original_path: 'artifacts/report.md',
  name: 'report.md',
  kind: 'document',
  archived: false,
  managed: true,
  size_bytes: 512,
  modified_at: '2026-09-22T10:00:00Z',
  conversations: [],
  story: null,
  ...over,
});

const STORY = entry({
  path: 'stories/night-train',
  original_path: 'stories/night-train',
  name: 'night-train',
  kind: 'story',
  story: {
    story_id: 'night-train',
    title: 'Night Train',
    unreadable_reason: null,
    writer: { room_id: 'room-w', title: 'Writing room', exists: true },
    files: [{ path: 'stories/night-train/chapters/01.md', name: '01.md', kind: 'document', size_bytes: 10 }],
  },
});

const survey = (entries: ArtifactEntry[], record_gaps: string[] = []): ArtifactSurvey => ({
  entries,
  scope_note: SCOPE_NOTE,
  record_gaps,
});

interface Call {
  url: string;
  method: string;
  body: unknown;
}

type Answer = { status: number; body: unknown };

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
      return new Response(JSON.stringify(found.body), { status: found.status });
    }),
  );
});

afterEach(() => {
  vi.unstubAllGlobals();
  vi.restoreAllMocks();
});

const LIST = 'GET /api/artifacts/library';
const ok = (body: unknown): Answer => ({ status: 200, body });

const renderLibrary = (onOpenStory = vi.fn(async () => {})) => {
  const onClose = vi.fn();
  render(<ArtifactLibrary isOpen onClose={onClose} onOpenStory={onOpenStory} />);
  return { onClose, onOpenStory };
};

const rowFor = async (path: string): Promise<HTMLElement> => {
  const rows = await screen.findAllByTestId('files-entry');
  const row = rows.find((r) => r.getAttribute('data-path') === path);
  if (!row) throw new Error(`no row for ${path}`);
  return row;
};

const posts = (suffix: string) => calls.filter((c) => c.method === 'POST' && c.url.endsWith(suffix));

describe('ArtifactLibrary, the Files screen (#1554)', () => {
  it('lists files with the scope note, gaps, and which ones no conversation links', async () => {
    answers[LIST] = ok(
      survey(
        [
          entry({}),
          entry({
            path: 'artifacts/plan.md',
            original_path: 'artifacts/plan.md',
            name: 'plan.md',
            conversations: [{ room_id: 'room-a', title: 'Planning' }],
          }),
        ],
        ['1 conversation could not be read, so the files they saved are not linked to them here.'],
      ),
    );
    renderLibrary();

    const report = await rowFor('artifacts/report.md');
    expect(within(report).getByText(FILES_COPY.notLinked)).toBeInTheDocument();
    expect(within(await rowFor('artifacts/plan.md')).getByText('Planning')).toBeInTheDocument();
    expect(screen.getByTestId('files-scope-note')).toHaveTextContent(SCOPE_NOTE);
    expect(screen.getByTestId('files-record-gaps')).toHaveTextContent('1 conversation could not be read');
  });

  it('says plainly when nothing is listed, without claiming there is nothing', async () => {
    answers[LIST] = ok(survey([]));
    renderLibrary();

    expect(await screen.findByTestId('files-empty')).toHaveTextContent('No files are listed.');
  });

  it('filters by kind and by conversation', async () => {
    answers[LIST] = ok(
      survey([
        entry({}),
        entry({
          path: 'artifacts/cover.png',
          original_path: 'artifacts/cover.png',
          name: 'cover.png',
          kind: 'image',
          conversations: [{ room_id: 'room-a', title: 'Planning' }],
        }),
      ]),
    );
    renderLibrary();
    await rowFor('artifacts/report.md');

    fireEvent.change(screen.getByTestId('files-kind-filter'), { target: { value: 'image' } });
    expect(screen.getAllByTestId('files-entry').map((r) => r.getAttribute('data-path'))).toEqual([
      'artifacts/cover.png',
    ]);

    fireEvent.change(screen.getByTestId('files-kind-filter'), { target: { value: 'all' } });
    fireEvent.change(screen.getByTestId('files-conversation-filter'), { target: { value: 'none' } });
    expect(screen.getAllByTestId('files-entry').map((r) => r.getAttribute('data-path'))).toEqual([
      'artifacts/report.md',
    ]);
  });

  it('opens a file and shows its text', async () => {
    answers[LIST] = ok(survey([entry({ path: 'artifacts/notes.txt', original_path: 'artifacts/notes.txt', name: 'notes.txt' })]));
    answers['GET /api/artifacts/library/file?path=artifacts%2Fnotes.txt'] = ok({
      path: 'artifacts/notes.txt',
      name: 'notes.txt',
      kind: 'document',
      text: 'the whole note',
    });
    renderLibrary();

    fireEvent.click(within(await rowFor('artifacts/notes.txt')).getByTestId('files-open'));

    expect(await screen.findByTestId('files-preview-text')).toHaveTextContent('the whole note');
  });

  // Killed by: frontend/src/components/artifacts/ArtifactLibrary.tsx :: onClick={onAskDelete}
  // Becomes: onClick={() => onDelete(false)}
  it('asks before deleting, and sends nothing until the deletion is confirmed', async () => {
    answers[LIST] = [ok(survey([entry({})])), ok(survey([]))];
    answers['POST /api/artifacts/library/delete'] = ok({ deleted: 'artifacts/report.md' });
    renderLibrary();

    fireEvent.click(within(await rowFor('artifacts/report.md')).getByTestId('files-delete'));

    expect(screen.getByRole('alertdialog')).toHaveTextContent(FILES_COPY.confirmDelete('report.md'));
    expect(posts('/delete')).toEqual([]);

    fireEvent.click(screen.getByTestId('files-confirm-delete'));

    expect(await screen.findByTestId('files-notice')).toHaveTextContent('report.md was deleted.');
    expect(posts('/delete').map((c) => c.body)).toEqual([
      { path: 'artifacts/report.md', confirm: true, release_writer: false },
    ]);
  });

  it('keeping the file sends nothing', async () => {
    answers[LIST] = ok(survey([entry({})]));
    renderLibrary();

    fireEvent.click(within(await rowFor('artifacts/report.md')).getByTestId('files-delete'));
    fireEvent.click(screen.getByText('Keep it'));

    expect(screen.queryByRole('alertdialog')).toBeNull();
    expect(posts('/delete')).toEqual([]);
  });

  // Killed by: frontend/src/components/artifacts/ArtifactLibrary.tsx :: setNotice(note ? `${done} ${note}` : done);
  // Becomes: setNotice(done);
  it.each([
    ['archive', 'files-archive', { path: '.archive/stories/night-train' }, FILES_COPY.archived('night-train')],
    ['delete', 'files-delete', { deleted: 'stories/night-train' }, FILES_COPY.deleted('night-train')],
  ])(
    'a story %sd while some of its conversations could not be updated says so beside the result (#1578)',
    async (action, button, body, done) => {
      const note = 'Some conversations that had this story open could not be updated, so they may still show it as open.';
      answers[LIST] = [ok(survey([STORY])), ok(survey([]))];
      answers[`POST /api/artifacts/library/${action}`] = ok({ ...body, note });
      renderLibrary();

      fireEvent.click(within(await rowFor('stories/night-train')).getByTestId(button));
      if (action === 'delete') fireEvent.click(screen.getByTestId('files-confirm-delete'));

      expect(await screen.findByTestId('files-notice')).toHaveTextContent(`${done} ${note}`);
    },
  );

  // Killed by: frontend/src/components/artifacts/ArtifactLibrary.tsx :: if (inUse && inUse.kind === 'in-use' && isStoryInUse(err)) {
  // Becomes: if (false) {
  it('a story a conversation is writing shows the reason, and going ahead releases the writer', async () => {
    const reason =
      'The conversation “Writing room” is writing this story. Delete that conversation first, or go ahead anyway to stop it writing to this story.';
    answers[LIST] = [ok(survey([STORY])), ok(survey([]))];
    answers['POST /api/artifacts/library/archive'] = [
      { status: 409, body: { detail: reason } },
      ok({ path: '.archive/stories/night-train' }),
    ];
    renderLibrary();

    fireEvent.click(within(await rowFor('stories/night-train')).getByTestId('files-archive'));

    expect(await screen.findByText(reason)).toBeInTheDocument();
    fireEvent.click(screen.getByTestId('files-go-ahead'));

    expect(await screen.findByTestId('files-notice')).toHaveTextContent('night-train was archived.');
    expect(posts('/archive').map((c) => c.body)).toEqual([
      { path: 'stories/night-train', release_writer: false },
      { path: 'stories/night-train', release_writer: true },
    ]);
  });

  // Checked by hand (#1578): with the button label back to `{FILES_COPY.goAhead}` alone, or
  // with `retry: releaseWriter` in `archive` replaced by `retry: false`, this case fails.
  it('after going ahead is refused too, the button says it is trying again', async () => {
    const writing =
      'The conversation “Writing room” is writing this story. Delete that conversation first, or go ahead anyway to stop it writing to this story.';
    const answering =
      'The conversation “Writing room” is answering right now, so it cannot be stopped from writing this story yet. Wait for the answer to finish, then try again.';
    answers[LIST] = [ok(survey([STORY])), ok(survey([]))];
    answers['POST /api/artifacts/library/archive'] = [
      { status: 409, body: { detail: writing } },
      { status: 409, body: { detail: answering } },
      ok({ path: '.archive/stories/night-train' }),
    ];
    renderLibrary();

    fireEvent.click(within(await rowFor('stories/night-train')).getByTestId('files-archive'));
    expect(await screen.findByText(writing)).toBeInTheDocument();
    expect(screen.getByTestId('files-go-ahead')).toHaveTextContent(FILES_COPY.goAhead);

    fireEvent.click(screen.getByTestId('files-go-ahead'));
    expect(await screen.findByText(answering)).toBeInTheDocument();
    expect(screen.getByTestId('files-go-ahead')).toHaveTextContent('Try going ahead again');

    fireEvent.click(screen.getByTestId('files-go-ahead'));
    expect(await screen.findByTestId('files-notice')).toHaveTextContent('night-train was archived.');
    expect(posts('/archive').map((c) => c.body)).toEqual([
      { path: 'stories/night-train', release_writer: false },
      { path: 'stories/night-train', release_writer: true },
      { path: 'stories/night-train', release_writer: true },
    ]);
  });

  // Checked by hand (#1578): with `retry: releaseWriter` in `remove` replaced by
  // `retry: false`, this case fails.
  it('a delete refused again after going ahead also says it is trying again', async () => {
    const writing =
      'The conversation “Writing room” is writing this story. Delete that conversation first, or go ahead anyway to stop it writing to this story.';
    const answering =
      'The conversation “Writing room” is answering right now, so it cannot be stopped from writing this story yet. Wait for the answer to finish, then try again.';
    answers[LIST] = ok(survey([STORY]));
    answers['POST /api/artifacts/library/delete'] = [
      { status: 409, body: { detail: writing } },
      { status: 409, body: { detail: answering } },
    ];
    renderLibrary();

    fireEvent.click(within(await rowFor('stories/night-train')).getByTestId('files-delete'));
    fireEvent.click(screen.getByTestId('files-confirm-delete'));
    expect(await screen.findByText(writing)).toBeInTheDocument();
    expect(screen.getByTestId('files-go-ahead')).toHaveTextContent(FILES_COPY.goAhead);

    fireEvent.click(screen.getByTestId('files-go-ahead'));
    expect(await screen.findByText(answering)).toBeInTheDocument();
    expect(screen.getByTestId('files-go-ahead')).toHaveTextContent('Try going ahead again');
    expect(posts('/delete').map((c) => c.body)).toEqual([
      { path: 'stories/night-train', confirm: true, release_writer: false },
      { path: 'stories/night-train', confirm: true, release_writer: true },
    ]);
  });

  // Checked by hand (#1578): with `notLinked` back to 'Not linked to a conversation that
  // still exists', this case fails.
  it('a file no conversation links says it has no recorded writer, not that the writer is gone', async () => {
    answers[LIST] = ok(survey([entry({})]));
    renderLibrary();

    expect(within(await rowFor('artifacts/report.md')).getByText('No recorded writer')).toBeInTheDocument();
    expect(screen.queryByText(/no longer exists|still exists/)).toBeNull();
  });

  it('restores an archived file', async () => {
    const archived = entry({ path: '.archive/artifacts/report.md', archived: true });
    answers[LIST] = [ok(survey([archived])), ok(survey([entry({})]))];
    answers['POST /api/artifacts/library/restore'] = ok({ path: 'artifacts/report.md' });
    renderLibrary();
    await screen.findByTestId('files-empty');

    fireEvent.click(screen.getByTestId('files-show-archived'));
    fireEvent.click(within(await rowFor('.archive/artifacts/report.md')).getByTestId('files-restore'));

    expect(await screen.findByTestId('files-notice')).toHaveTextContent('report.md was restored.');
    expect(posts('/restore').map((c) => c.body)).toEqual([{ path: '.archive/artifacts/report.md' }]);
  });

  it('opens a story in a new conversation and closes', async () => {
    answers[LIST] = ok(survey([STORY]));
    const { onOpenStory, onClose } = renderLibrary();

    fireEvent.click(within(await rowFor('stories/night-train')).getByTestId('files-open-story'));

    await waitFor(() => expect(onOpenStory).toHaveBeenCalledWith('night-train', 'Night Train'));
    await waitFor(() => expect(onClose).toHaveBeenCalled());
  });

  it("shows a story's view in place of the list, and goes back to the list (#1560)", async () => {
    answers[LIST] = ok(survey([STORY]));
    answers['GET /api/artifacts/library/stories/night-train'] = { status: 404, body: { detail: 'There is no story called night-train any more.' } };
    renderLibrary();

    fireEvent.click(within(await rowFor('stories/night-train')).getByTestId('files-view-story'));

    expect(await screen.findByTestId('story-view')).toBeInTheDocument();
    expect(screen.queryByTestId('files-entry')).not.toBeInTheDocument();
    expect(await screen.findByRole('alert')).toHaveTextContent('There is no story called night-train any more.');

    fireEvent.click(screen.getByTestId('story-view-back'));
    expect(await rowFor('stories/night-train')).toBeInTheDocument();
    expect(screen.queryByTestId('story-view')).not.toBeInTheDocument();
  });

  it('a list that could not be read says so plainly', async () => {
    answers[LIST] = { status: 502, body: '<html>Bad Gateway</html>' };
    renderLibrary();

    const alert = await screen.findByRole('alert');
    expect(alert).toHaveTextContent(FILES_COPY.listFailed);
    expectPlain(alert.querySelector('p')?.textContent);
  });
});
