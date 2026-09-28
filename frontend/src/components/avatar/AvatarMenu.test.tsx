import { describe, it, expect, vi, afterEach } from 'vitest';
import { render, screen, fireEvent, waitFor } from '@testing-library/react';
import { CloneProfile } from '../clones/CloneProfile';
import { AvatarChoiceContext, type AvatarChoice } from '../../lib/avatarChoice';
import { LocaleProvider } from '../../i18n';
import { makePersonaInfo } from '../../test/fixtures';
import { expectPlain } from '../../test/plainCopy';

const answer = (status: number, body: unknown = {}) =>
  new Response(JSON.stringify(body), { status, headers: { 'Content-Type': 'application/json' } });

/** `fetch` for the picture routes, answered in turn; the language provider's read answers empty. */
const pictureRoutes = (...replies: Response[]) => {
  const calls: [string, RequestInit][] = [];
  vi.stubGlobal(
    'fetch',
    vi.fn(async (url: string, init: RequestInit = {}) => {
      if (!url.startsWith('/api/personas/')) return answer(200, {});
      calls.push([url, init]);
      return replies.shift() ?? answer(200, { status: 'ok' });
    }),
  );
  return calls;
};

const choice = (): AvatarChoice => ({
  clones: [{ name: 'reader', label: 'reader' }],
  onChanged: vi.fn(),
  askFor: vi.fn(),
});

const profile = ({
  value = choice() as AvatarChoice | null,
  avatarUrl = '/api/personas/reader/avatar?v=1' as string | null,
  avatarChosen = true,
  korean = false,
} = {}) => (
  <LocaleProvider hints={[korean ? 'ko-KR' : 'en-US']}>
    <AvatarChoiceContext.Provider value={value}>
      <CloneProfile
        cloneId="reader"
        persona={makePersonaInfo({ avatar_url: avatarUrl, avatar_chosen: avatarChosen })}
        onStartConversation={vi.fn()}
      />
    </AvatarChoiceContext.Provider>
  </LocaleProvider>
);

const renderProfile = (options: Parameters<typeof profile>[0] = {}) => render(profile(options));

const openMenu = () => fireEvent.click(screen.getByTestId('clone-profile-avatar'));

afterEach(() => {
  vi.unstubAllGlobals();
});

describe('the avatar menu on a clone`s profile', () => {
  // Killed by: frontend/src/components/avatar/AvatarMenu.tsx :: void run('uploaded', () => uploadAvatar(name, file));
  // Becomes: void 0;
  it('uploads a picture from this computer, and the pictures shown are read again', async () => {
    const calls = pictureRoutes(answer(200, { status: 'ok', previous_path: null }));
    const value = choice();
    renderProfile({ value });

    openMenu();
    expect(screen.getByTestId('clone-avatar-upload')).toHaveTextContent('Upload picture');
    const file = new File([new Uint8Array([137, 80, 78, 71])], 'me.png', { type: 'image/png' });
    fireEvent.change(screen.getByTestId('clone-avatar-file'), { target: { files: [file] } });

    await waitFor(() => expect(screen.getByTestId('clone-avatar-done')).toBeInTheDocument());
    const [url, init] = calls[0];
    expect(url).toBe('/api/personas/reader/avatar');
    expect(init.method).toBe('PUT');
    expect((init.headers as Record<string, string>)['Content-Type']).toBe('image/png');
    expect(value.onChanged).toHaveBeenCalledTimes(1);
  });

  // Killed by: frontend/src/components/avatar/AvatarMenu.tsx :: if (problem !== null) {
  // Becomes: if (false) {
  it('refuses a file it cannot use before sending it, in plain words', () => {
    const calls = pictureRoutes();
    renderProfile();

    openMenu();
    const file = new File(['<svg/>'], 'face.svg', { type: 'image/svg+xml' });
    fireEvent.change(screen.getByTestId('clone-avatar-file'), { target: { files: [file] } });

    expect(calls).toEqual([]);
    const failed = screen.getByTestId('clone-avatar-failed');
    expectPlain(failed.textContent);
    expect(failed).toHaveTextContent(/PNG, JPEG or WebP/);
  });

  // Killed by: frontend/src/components/avatar/AvatarMenu.tsx :: choice.askFor(name, request());
  // Becomes: void name;
  it('asks the clone for pictures without sending anything itself', () => {
    const calls = pictureRoutes();
    const value = choice();
    renderProfile({ value });

    openMenu();
    fireEvent.click(screen.getByTestId('clone-avatar-ask'));

    // The default style goes unnamed: a request with no style is drawn in the house style.
    expect(value.askFor).toHaveBeenCalledWith(
      'reader',
      'Please draw a few profile pictures of yourself and show them to me, so I can choose one.',
    );
    expect(calls).toEqual([]);
    expect(screen.queryByTestId('clone-avatar-menu-list')).toBeNull();
  });

  // Killed by: frontend/src/components/avatar/AvatarMenu.tsx :: useState<AvatarStyle>('default')
  // Becomes: useState<AvatarStyle>('watercolor')
  it('starts with the default style picked', () => {
    pictureRoutes();
    renderProfile();

    openMenu();
    const picked = screen
      .getAllByRole('menuitemradio')
      .filter((chip) => chip.getAttribute('aria-checked') === 'true');
    expect(picked).toEqual([screen.getByTestId('clone-avatar-style-default')]);
    expect(picked[0]).toHaveTextContent('Default (pastel close-up)');
  });

  // Killed by: frontend/src/components/avatar/AvatarMenu.tsx :: onClick={() => setStyle(option)}
  // Becomes: onClick={() => undefined}
  // Killed by: frontend/src/components/avatar/AvatarMenu.tsx :: fmt(copy.askTextStyled, { style: copy.style.name[style] })
  // Becomes: copy.askText
  it('names the picked style in the request it puts in the box', () => {
    pictureRoutes();
    const value = choice();
    renderProfile({ value });

    openMenu();
    fireEvent.click(screen.getByTestId('clone-avatar-style-pixelArt'));
    expect(screen.getByTestId('clone-avatar-style-pixelArt')).toHaveAttribute('aria-checked', 'true');
    expect(screen.getByTestId('clone-avatar-style-default')).toHaveAttribute('aria-checked', 'false');
    fireEvent.click(screen.getByTestId('clone-avatar-ask'));

    expect(value.askFor).toHaveBeenCalledTimes(1);
    const [, request] = vi.mocked(value.askFor).mock.calls[0];
    expect(request).toBe(
      'Please draw a few profile pictures of yourself in the pixel art style and show them to me, so I can choose one.',
    );
    expectPlain(request);
  });

  // Killed by: frontend/src/components/avatar/AvatarMenu.tsx :: onClick={() => setStyle(option)}
  // Becomes: onClick={() => choice.askFor(name, request())}
  it('puts nothing in the box and sends nothing when a style is picked', () => {
    // Picking is only picking: the request goes in when the person asks, and then unsent.
    const calls = pictureRoutes();
    const value = choice();
    renderProfile({ value });

    openMenu();
    fireEvent.click(screen.getByTestId('clone-avatar-style-watercolor'));

    expect(value.askFor).not.toHaveBeenCalled();
    expect(calls).toEqual([]);
    expect(screen.getByTestId('clone-avatar-menu-list')).toBeInTheDocument();
  });

  // Killed by: frontend/src/components/avatar/AvatarMenu.tsx :: void run('reset', () => resetAvatar(name));
  // Becomes: void 0;
  // Killed by: frontend/src/components/avatar/AvatarMenu.tsx :: onClick={() => void undo(outcome.previousPath, outcome.changeId)}
  // Becomes: onClick={() => void undo(null, outcome.changeId)}
  it('resets to the default, and Undo puts the chosen picture back', async () => {
    const calls = pictureRoutes(
      answer(200, { status: 'ok', previous_path: '.uclone/personas/reader.prev.png', change_id: 3 }),
      answer(200, { status: 'ok' }),
    );
    renderProfile();

    openMenu();
    fireEvent.click(screen.getByTestId('clone-avatar-reset'));
    fireEvent.click(await screen.findByTestId('clone-avatar-undo'));

    await waitFor(() => expect(screen.getByTestId('clone-avatar-undone')).toBeInTheDocument());
    expect(calls[0][1].method).toBe('DELETE');
    expect(calls[1][1].method).toBe('PUT');
    // The Undo names the change it takes back, so the runtime can refuse it if another came after.
    expect(JSON.parse(calls[1][1].body as string)).toEqual({
      source_path: '.uclone/personas/reader.prev.png',
      undo_of: 3,
    });
  });

  // Killed by: frontend/src/components/clones/CloneProfile.tsx :: hasChosenPicture={persona?.avatar_chosen === true}
  // Becomes: hasChosenPicture={true}
  it('offers no reset for a clone that already shows the default', () => {
    pictureRoutes();
    renderProfile({ avatarUrl: null, avatarChosen: false });

    openMenu();
    expect(screen.getByTestId('clone-avatar-reset')).toBeDisabled();
  });

  // Killed by: frontend/src/components/clones/CloneProfile.tsx :: hasChosenPicture={persona?.avatar_chosen === true}
  // Becomes: hasChosenPicture={persona?.avatar_url !== null}
  it('offers no reset for a picture the clone ships with (#1780)', () => {
    // Reset would change nothing there and still claim it had.
    pictureRoutes();
    renderProfile({ avatarUrl: '/api/personas/reader/avatar?v=shipped', avatarChosen: false });

    openMenu();
    expect(screen.getByTestId('clone-avatar-reset')).toBeDisabled();
  });

  // Killed by: frontend/src/components/avatar/AvatarMenu.tsx :: {undoStillOffered(outcome.changeId, name, choice.latestChanges) && (
  // Becomes: {true && (
  it('drops its Undo once the listing shows a later change, made anywhere (#1780, #1809)', async () => {
    pictureRoutes(answer(200, { status: 'ok', previous_path: '.uclone/personas/reader.prev.png', change_id: 3 }));
    const { rerender } = renderProfile();

    openMenu();
    fireEvent.click(screen.getByTestId('clone-avatar-reset'));
    await screen.findByTestId('clone-avatar-undo');
    // The listing read again after this change names it as the latest: the Undo stays.
    rerender(profile({ value: { ...choice(), latestChanges: { reader: 3 } } }));
    expect(screen.getByTestId('clone-avatar-undo')).toBeInTheDocument();
    // Another tab, or the clone itself, then gives reader another picture.
    rerender(profile({ value: { ...choice(), latestChanges: { reader: 4 } } }));

    expect(screen.getByTestId('clone-avatar-done')).toBeInTheDocument();
    expect(screen.queryByTestId('clone-avatar-undo')).toBeNull();
  });

  // Killed by: frontend/src/components/avatar/AvatarMenu.tsx :: setOutcome({ kind: 'failed', failure });
  // Becomes: void failure;
  // Killed by: frontend/src/components/avatar/AvatarMenu.tsx :: if (failure === 'staleChange') choice.onChanged();
  // Becomes: if (false) choice.onChanged();
  // Killed by: frontend/src/lib/avatarChoice.ts :: stale_change: 'staleChange',
  // Becomes: stale_change: 'failed',
  it('says plainly that an Undo came too late, drops it, and reads the pictures again (#1809)', async () => {
    // The listing had not caught up yet, so the Undo was still offered; the runtime refused it.
    const calls = pictureRoutes(
      answer(200, { status: 'ok', previous_path: '.uclone/personas/reader.prev.png', change_id: 3 }),
      answer(409, { detail: 'The picture was changed again after that change.', code: 'stale_change' }),
    );
    const value = choice();
    renderProfile({ value });

    openMenu();
    fireEvent.click(screen.getByTestId('clone-avatar-reset'));
    fireEvent.click(await screen.findByTestId('clone-avatar-undo'));

    const failed = await screen.findByTestId('clone-avatar-failed');
    expectPlain(failed.textContent);
    expect(failed.textContent).not.toMatch(/stale_change|409|change_id|undo_of/);
    expect(failed).toHaveTextContent(/reader's picture was changed again/);
    expect(screen.queryByTestId('clone-avatar-undo')).toBeNull();
    expect(calls).toHaveLength(2);
    // Once for the reset, once more so the picture shown is the newer one in force.
    expect(value.onChanged).toHaveBeenCalledTimes(2);
  });

  // Killed by: frontend/src/components/avatar/AvatarMenu.tsx :: {fmt(copy.failure[outcome.failure], { name })}
  // Becomes: {String(outcome.failure)}
  it('says a failed change in plain words, never the runtime`s', async () => {
    pictureRoutes(answer(500, { detail: 'OSError: [Errno 28] /Users/x/.uclone/personas' }));
    renderProfile();

    openMenu();
    fireEvent.click(screen.getByTestId('clone-avatar-reset'));

    const failed = await screen.findByTestId('clone-avatar-failed');
    expectPlain(failed.textContent);
    expect(failed.textContent).not.toMatch(/persona|Errno/i);
    // A sentence that names the clone, not a bare code word.
    expect(failed).toHaveTextContent(/reader/);
  });

  // Killed by: frontend/src/components/avatar/AvatarMenu.tsx :: if (choice === null || !installed) return picture();
  // Becomes: if (!installed) return picture();
  it('is the plain picture where no change can be made', () => {
    renderProfile({ value: null });
    openMenu();
    expect(screen.queryByTestId('clone-avatar-menu')).toBeNull();
  });

  it('reads in Korean', () => {
    pictureRoutes();
    const value = choice();
    renderProfile({ value, korean: true });

    openMenu();
    expect(screen.getByTestId('clone-avatar-upload')).toHaveTextContent('그림 올리기');
    expect(screen.getByTestId('clone-avatar-ask')).toHaveTextContent('reader에게 만들어 달라고 하기');
    expect(screen.getByTestId('clone-avatar-style-default')).toHaveTextContent('기본(파스텔 클로즈업)');
    fireEvent.click(screen.getByTestId('clone-avatar-style-watercolor'));
    fireEvent.click(screen.getByTestId('clone-avatar-ask'));
    expect(vi.mocked(value.askFor).mock.calls[0][1]).toContain('수채화 스타일로');
  });
});
