import { describe, it, expect, vi, afterEach } from 'vitest';
import { render, screen, fireEvent, waitFor } from '@testing-library/react';
import { RichText } from '../RichText';
import { AvatarAuthorContext, AvatarChoiceContext, type AvatarChoice } from '../../lib/avatarChoice';
import { LocaleProvider } from '../../i18n';
import { expectPlain } from '../../test/plainCopy';

const PICTURE = 'artifacts/images/img_1_sess_r.png';
const REPLY = `Here you are.\n\n![Me](/api/artifacts/content?path=${PICTURE})`;

const answer = (status: number, body: unknown = {}) =>
  new Response(JSON.stringify(body), { status, headers: { 'Content-Type': 'application/json' } });

/**
 * `fetch` for the picture routes, answered in turn from `replies`; every other read (the
 * language provider's settings) answers empty. `calls` holds the picture requests only.
 */
const pictureRoutes = (...replies: Response[]) => {
  const calls: [string, RequestInit][] = [];
  const fetchMock = vi.fn(async (url: string, init: RequestInit = {}) => {
    if (!url.startsWith('/api/personas/')) return answer(200, {});
    calls.push([url, init]);
    return replies.shift() ?? answer(200, { status: 'ok' });
  });
  vi.stubGlobal('fetch', fetchMock);
  return calls;
};

const choice = (over: Partial<AvatarChoice> = {}): AvatarChoice => ({
  clones: [
    { name: 'scout', label: 'scout' },
    { name: 'critic', label: 'critic' },
  ],
  onChanged: vi.fn(),
  askFor: vi.fn(),
  ...over,
});

const card = (
  { value = choice(), author = 'scout' as string | null, korean = false, content = REPLY } = {},
) => (
  <LocaleProvider hints={[korean ? 'ko-KR' : 'en-US']}>
    <AvatarChoiceContext.Provider value={value}>
      <AvatarAuthorContext.Provider value={author}>
        <RichText content={content} />
      </AvatarAuthorContext.Provider>
    </AvatarChoiceContext.Provider>
  </LocaleProvider>
);

const renderCard = (options: Parameters<typeof card>[0] = {}) => render(card(options));

afterEach(() => {
  vi.unstubAllGlobals();
});

describe('Use as avatar, under a drawn picture', () => {
  // Killed by: frontend/src/components/RichText.tsx :: {path && <UseAsAvatar path={path} />}
  // Becomes: {false && <UseAsAvatar path={path} />}
  // Killed by: frontend/src/components/avatar/UseAsAvatar.tsx :: onClick={() => void give(author)}
  // Becomes: onClick={() => void give('')}
  it('gives the picture to the clone that drew it, and every picture shown is read again', async () => {
    const calls = pictureRoutes(answer(200, { status: 'ok', previous_path: null }));
    const value = choice();
    renderCard({ value });

    fireEvent.click(screen.getByTestId('use-as-avatar-btn'));

    await waitFor(() => expect(screen.getByTestId('use-as-avatar-done')).toBeInTheDocument());
    const [url, init] = calls[0];
    expect(url).toBe('/api/personas/scout/avatar');
    expect(JSON.parse(init.body as string)).toEqual({ source_path: PICTURE });
    expect(screen.getByTestId('use-as-avatar-done')).toHaveTextContent('scout');
    expect(value.onChanged).toHaveBeenCalledTimes(1);
  });

  // Killed by: frontend/src/components/avatar/UseAsAvatar.tsx :: onClick={() => void give(clone.name)}
  // Becomes: onClick={() => void give(author)}
  it('gives it to another clone chosen from the list', async () => {
    const calls = pictureRoutes();
    renderCard();

    fireEvent.click(screen.getByTestId('use-as-avatar-more'));
    expect(screen.getByTestId('use-as-avatar-list')).toBeInTheDocument();
    fireEvent.click(screen.getByTestId('use-as-avatar-for-critic'));

    await waitFor(() => expect(screen.getByTestId('use-as-avatar-done')).toHaveTextContent('critic'));
    expect(calls[0][0]).toBe('/api/personas/critic/avatar');
    expect(screen.queryByTestId('use-as-avatar-list')).toBeNull();
  });

  // Killed by: frontend/src/components/avatar/UseAsAvatar.tsx :: onClick={() => void undo(outcome.name, outcome.previousPath, outcome.changeId)}
  // Becomes: onClick={() => void undo(outcome.name, null, outcome.changeId)}
  it('undoes by putting back the picture the change replaced', async () => {
    const calls = pictureRoutes(
      answer(200, { status: 'ok', previous_path: '.uclone/personas/scout.prev.png', change_id: 7 }),
      answer(200, { status: 'ok' }),
    );
    const value = choice();
    renderCard({ value });

    fireEvent.click(screen.getByTestId('use-as-avatar-btn'));
    fireEvent.click(await screen.findByTestId('use-as-avatar-undo'));

    await waitFor(() => expect(screen.getByTestId('use-as-avatar-undone')).toBeInTheDocument());
    const [, init] = calls[1];
    expect(init.method).toBe('PUT');
    expect(JSON.parse(init.body as string)).toEqual({ source_path: '.uclone/personas/scout.prev.png', undo_of: 7 });
    expect(value.onChanged).toHaveBeenCalledTimes(2);
  });

  // Killed by: frontend/src/components/avatar/UseAsAvatar.tsx :: {fmt(copy.failure[outcome.failure], { name: labelOf(outcome.name) })}
  // Becomes: {String(outcome.failure)}
  it('says a refusal in plain words, not the runtime`s', async () => {
    pictureRoutes(answer(422, { detail: 'PersonaAvatarError: /Users/x/.uclone/a.png is not an image' }));
    const value = choice();
    renderCard({ value });

    fireEvent.click(screen.getByTestId('use-as-avatar-btn'));

    const failed = await screen.findByTestId('use-as-avatar-failed');
    expectPlain(failed.textContent);
    expect(failed.textContent).not.toMatch(/persona|engine|tool/i);
    // A sentence that names the clone, not a bare code word.
    expect(failed).toHaveTextContent(/scout/);
    expect(value.onChanged).not.toHaveBeenCalled();
  });

  // Killed by: frontend/src/components/avatar/UseAsAvatar.tsx :: if (choice === null || author === null || !canWearPicture(path)) return null;
  // Becomes: if (choice === null || !canWearPicture(path)) return null;
  it('is not offered outside a clone`s message', () => {
    renderCard({ author: null });
    expect(screen.getByTestId('inline-artifact-image')).toBeInTheDocument();
    expect(screen.queryByTestId('use-as-avatar')).toBeNull();
  });

  // Killed by: frontend/src/components/avatar/UseAsAvatar.tsx :: if (choice === null || author === null || !canWearPicture(path)) return null;
  // Becomes: if (choice === null || author === null) return null;
  it('is not offered under an SVG, which the runtime would refuse (#1780)', () => {
    renderCard({ content: '![Logo](/api/artifacts/content?path=artifacts/images/logo.SVG)' });
    expect(screen.getByTestId('inline-artifact-image')).toBeInTheDocument();
    expect(screen.queryByTestId('use-as-avatar')).toBeNull();
  });

  // Killed by: frontend/src/components/avatar/UseAsAvatar.tsx :: {undoStillOffered(outcome.changeId, outcome.name, choice.latestChanges) && (
  // Becomes: {true && (
  it('drops its Undo once a later change is made to the same clone`s picture (#1780)', async () => {
    // Undo puts back what its own change replaced; after a later change that would put
    // back the wrong picture, or reset the newer choice.
    pictureRoutes(
      answer(200, { status: 'ok', previous_path: '.uclone/personas/scout.prev.png', change_id: 1 }),
      answer(200, { status: 'ok', previous_path: '.uclone/personas/scout.prev.png', change_id: 2 }),
    );
    const content = `![A](/api/artifacts/content?path=artifacts/images/a.png)\n\n![B](/api/artifacts/content?path=artifacts/images/b.png)`;
    const { rerender } = renderCard({ content });
    const [first, second] = screen.getAllByTestId('use-as-avatar-btn');

    fireEvent.click(first);
    await waitFor(() => expect(screen.getAllByTestId('use-as-avatar-undo')).toHaveLength(1));
    fireEvent.click(second);

    await waitFor(() => expect(screen.getAllByTestId('use-as-avatar-done')).toHaveLength(2));
    // The listing, read again after the second change, names it as the latest.
    rerender(card({ content, value: choice({ latestChanges: { scout: 2 } }) }));
    // Only the later card still offers an Undo.
    const cards = screen.getAllByTestId('use-as-avatar');
    expect(cards[0].querySelector('[data-testid="use-as-avatar-undo"]')).toBeNull();
    expect(cards[1].querySelector('[data-testid="use-as-avatar-undo"]')).not.toBeNull();
  });

  // Killed by: frontend/src/lib/avatarChoice.ts :: if (!res.ok) return { ok: false, failure: failureOf(res.status, await codeOf(res)) };
  // Becomes: if (!res.ok) return { ok: false, failure: failureOf(res.status, undefined) };
  it('says a path outside the workspace as that, not as a wrong format (#1780)', async () => {
    pictureRoutes(answer(422, { detail: 'That picture is outside the workspace.', code: 'outside_workspace' }));
    renderCard();

    fireEvent.click(screen.getByTestId('use-as-avatar-btn'));

    const failed = await screen.findByTestId('use-as-avatar-failed');
    expectPlain(failed.textContent);
    expect(failed).toHaveTextContent(/outside the workspace/);
    expect(failed.textContent).not.toMatch(/PNG|JPEG|outside_workspace/);
  });

  // Killed by: frontend/src/lib/avatarChoice.ts :: if (!res.ok) return { ok: false, failure: failureOf(res.status, await codeOf(res)) };
  // Becomes: if (!res.ok) return { ok: false, failure: failureOf(res.status, undefined) };
  it('says each reason in Korean too (#1780)', async () => {
    pictureRoutes(answer(422, { detail: 'x', code: 'not_saved' }));
    renderCard({ korean: true });

    fireEvent.click(screen.getByTestId('use-as-avatar-btn'));

    const failed = await screen.findByTestId('use-as-avatar-failed');
    expectPlain(failed.textContent);
    expect(failed).toHaveTextContent(/저장/);
    expect(failed.textContent).toMatch(/니다/);
    expect(failed.textContent).not.toMatch(/PNG|not_saved/);
  });

  it('reads in Korean, in plain words', async () => {
    pictureRoutes(answer(404));
    renderCard({ korean: true });

    expect(screen.getByTestId('use-as-avatar-btn')).toHaveTextContent('아바타로 사용');
    fireEvent.click(screen.getByTestId('use-as-avatar-btn'));
    const failed = await screen.findByTestId('use-as-avatar-failed');
    expectPlain(failed.textContent);
    expect(failed.textContent).toMatch(/니다/);
  });

  // Killed by: frontend/src/components/avatar/UseAsAvatar.tsx :: setOutcome({ kind: 'failed', name, failure });
  // Becomes: void failure;
  it('says in Korean that an Undo came too late, and offers it no more (#1809)', async () => {
    pictureRoutes(
      answer(200, { status: 'ok', previous_path: '.uclone/personas/scout.prev.png', change_id: 5 }),
      answer(409, { detail: 'The picture was changed again after that change.', code: 'stale_change' }),
    );
    renderCard({ korean: true });

    fireEvent.click(screen.getByTestId('use-as-avatar-btn'));
    fireEvent.click(await screen.findByTestId('use-as-avatar-undo'));

    const failed = await screen.findByTestId('use-as-avatar-failed');
    expectPlain(failed.textContent);
    expect(failed.textContent).not.toMatch(/stale_change|409|change_id|undo_of/);
    expect(failed).toHaveTextContent(/scout의 그림이 다시 바뀌어/);
    expect(failed.textContent).toMatch(/니다/);
    expect(screen.queryByTestId('use-as-avatar-undo')).toBeNull();
  });
});
