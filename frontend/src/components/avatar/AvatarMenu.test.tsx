import { describe, it, expect, vi, afterEach } from 'vitest';
import { act, render, screen, fireEvent, waitFor } from '@testing-library/react';
import { CloneProfile } from '../clones/CloneProfile';
import { AvatarChoiceContext, noteAvatarChange, type AvatarChoice } from '../../lib/avatarChoice';
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

const renderProfile = ({
  value = choice() as AvatarChoice | null,
  avatarUrl = '/api/personas/reader/avatar?v=1' as string | null,
  avatarChosen = true,
  korean = false,
} = {}) =>
  render(
    <LocaleProvider hints={[korean ? 'ko-KR' : 'en-US']}>
      <AvatarChoiceContext.Provider value={value}>
        <CloneProfile
          cloneId="reader"
          persona={makePersonaInfo({ avatar_url: avatarUrl, avatar_chosen: avatarChosen })}
          onStartConversation={vi.fn()}
        />
      </AvatarChoiceContext.Provider>
    </LocaleProvider>,
  );

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

  // Killed by: frontend/src/components/avatar/AvatarMenu.tsx :: choice.askFor(name);
  // Becomes: void name;
  it('asks the clone for pictures without sending anything itself', () => {
    const calls = pictureRoutes();
    const value = choice();
    renderProfile({ value });

    openMenu();
    fireEvent.click(screen.getByTestId('clone-avatar-ask'));

    expect(value.askFor).toHaveBeenCalledWith('reader');
    expect(calls).toEqual([]);
    expect(screen.queryByTestId('clone-avatar-menu-list')).toBeNull();
  });

  // Killed by: frontend/src/components/avatar/AvatarMenu.tsx :: void run('reset', () => resetAvatar(name));
  // Becomes: void 0;
  // Killed by: frontend/src/components/avatar/AvatarMenu.tsx :: onClick={() => void undo(outcome.previousPath)}
  // Becomes: onClick={() => void undo(null)}
  it('resets to the default, and Undo puts the chosen picture back', async () => {
    const calls = pictureRoutes(
      answer(200, { status: 'ok', previous_path: '.uclone/personas/reader.prev.png' }),
      answer(200, { status: 'ok' }),
    );
    renderProfile();

    openMenu();
    fireEvent.click(screen.getByTestId('clone-avatar-reset'));
    fireEvent.click(await screen.findByTestId('clone-avatar-undo'));

    await waitFor(() => expect(screen.getByTestId('clone-avatar-undone')).toBeInTheDocument());
    expect(calls[0][1].method).toBe('DELETE');
    expect(calls[1][1].method).toBe('PUT');
    expect(JSON.parse(calls[1][1].body as string)).toEqual({ source_path: '.uclone/personas/reader.prev.png' });
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

  // Killed by: frontend/src/components/avatar/AvatarMenu.tsx :: {latest === outcome.change && (
  // Becomes: {true && (
  it('drops its Undo once the picture is changed again elsewhere (#1780)', async () => {
    pictureRoutes(answer(200, { status: 'ok', previous_path: '.uclone/personas/reader.prev.png' }));
    renderProfile();

    openMenu();
    fireEvent.click(screen.getByTestId('clone-avatar-reset'));
    await screen.findByTestId('clone-avatar-undo');
    // A card under a drawn picture, say, gives reader another one.
    act(() => {
      noteAvatarChange('reader');
    });

    expect(screen.getByTestId('clone-avatar-done')).toBeInTheDocument();
    expect(screen.queryByTestId('clone-avatar-undo')).toBeNull();
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
    renderProfile({ korean: true });

    openMenu();
    expect(screen.getByTestId('clone-avatar-upload')).toHaveTextContent('그림 올리기');
    expect(screen.getByTestId('clone-avatar-ask')).toHaveTextContent('reader에게 만들어 달라고 하기');
  });
});
