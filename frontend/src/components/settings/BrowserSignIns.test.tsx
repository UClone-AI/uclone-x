import { describe, it, expect, vi, afterEach } from 'vitest';
import { fireEvent, render, screen, waitFor, within } from '@testing-library/react';
import type { ReactNode } from 'react';
import { CATALOGS, LocaleProvider } from '../../i18n';
import { expectPlain } from '../../test/plainCopy';
import { SavedSignIns, SignedInSites, type SavedSignIn } from './BrowserSignIns';

const copy = CATALOGS.en.browser;
const PASSWORD = 'hunter2-secret';
const KEY = 'JBSWY3DPEHPK3PXP';

type Answer = { ok: boolean; status: number; json: () => Promise<unknown> };
const answer = (body: unknown, status = 200): Answer => ({
  ok: status >= 200 && status < 300,
  status,
  json: async () => body,
});

interface Call {
  url: string;
  method: string;
  body: unknown;
}

/** Answers each `METHOD url` from `routes`; a route given a function is asked per call. */
const mockFetch = (routes: Record<string, Answer | ((body: unknown) => Answer)>) => {
  const calls: Call[] = [];
  vi.stubGlobal(
    'fetch',
    vi.fn(async (url: string, init?: RequestInit) => {
      const method = init?.method ?? 'GET';
      const body = typeof init?.body === 'string' ? (JSON.parse(init.body) as unknown) : null;
      calls.push({ url, method, body });
      const route = routes[`${method} ${url}`];
      if (route === undefined) return answer({}, 599);
      return typeof route === 'function' ? route(body) : route;
    }),
  );
  return calls;
};

const SHOP: SavedSignIn = { site: 'shop.example.com', username: 'kenny', has_2fa: true };
const MAIL: SavedSignIn = { site: 'mail.example.com', username: 'kenny@example.com', has_2fa: false };
const list = (...rows: SavedSignIn[]) => ({ sign_ins: rows });

const Korean = ({ children }: { children: ReactNode }) => (
  <LocaleProvider hints={['ko-KR']}>{children}</LocaleProvider>
);

const field = (testId: string) => screen.getByTestId(testId) as HTMLInputElement;
const fill = (testId: string, value: string) => fireEvent.change(field(testId), { target: { value } });

afterEach(() => {
  vi.unstubAllGlobals();
});

describe('SavedSignIns', () => {
  it('lists each sign-in by site and username, marks 2FA, and never holds a secret', async () => {
    // Even a list that wrongly carried the secrets would not show them.
    mockFetch({
      'GET /api/browser/sign-ins': answer({ sign_ins: [{ ...SHOP, password: PASSWORD, totp_key: KEY }, MAIL] }),
    });
    const { container } = render(<SavedSignIns />);

    const shop = await screen.findByTestId('sign-in-shop.example.com-kenny');
    expect(shop).toHaveTextContent('shop.example.com');
    expect(shop).toHaveTextContent('kenny');
    expect(within(shop).getByTestId('sign-in-2fa')).toHaveTextContent(copy.signIns.twoFactor);
    const mail = screen.getByTestId('sign-in-mail.example.com-kenny@example.com');
    expect(within(mail).queryByTestId('sign-in-2fa')).toBeNull();
    expect(within(shop).getByRole('button', { name: copy.signIns.edit })).toBeInTheDocument();
    expect(within(shop).getByRole('button', { name: copy.signIns.delete })).toBeInTheDocument();
    expect(screen.getByText(copy.signIns.intro)).toBeInTheDocument();
    expect(container.innerHTML).not.toContain(PASSWORD);
    expect(container.innerHTML).not.toContain(KEY);
  });

  it('says plainly that none are saved, rather than showing an empty list', async () => {
    mockFetch({ 'GET /api/browser/sign-ins': answer(list()) });
    render(<SavedSignIns />);
    expect(await screen.findByTestId('sign-ins-empty')).toHaveTextContent(copy.signIns.none);
  });

  it('adds a sign-in, posts what was typed, then clears the form, password included', async () => {
    const calls = mockFetch({
      'GET /api/browser/sign-ins': answer(list()),
      'POST /api/browser/sign-ins': answer(list(SHOP)),
    });
    const { container } = render(<SavedSignIns />);
    await screen.findByTestId('sign-ins-empty');

    fireEvent.click(screen.getByRole('button', { name: copy.signIns.add }));
    expect(field('sign-in-password').type).toBe('password');
    expect(field('sign-in-password').autocomplete).toBe('new-password');
    expect(field('sign-in-key').autocomplete).toBe('off');
    expect(screen.getByText(copy.signIns.keyHelp)).toBeInTheDocument();
    fill('sign-in-site', ' shop.example.com ');
    fill('sign-in-username', 'kenny');
    fill('sign-in-password', PASSWORD);
    fill('sign-in-key', KEY);
    fireEvent.click(screen.getByTestId('sign-in-save'));

    await screen.findByTestId('sign-in-shop.example.com-kenny');
    expect(calls.filter((c) => c.method === 'POST')).toEqual([
      {
        url: '/api/browser/sign-ins',
        method: 'POST',
        body: {
          site: 'shop.example.com',
          username: 'kenny',
          password: PASSWORD,
          totp_key: KEY,
          clear_totp_key: false,
        },
      },
    ]);
    expect(screen.queryByTestId('sign-in-form')).toBeNull();
    expect(screen.getByTestId('sign-in-saved')).toHaveTextContent(copy.signIns.saved);
    expect(container.innerHTML).not.toContain(PASSWORD);
    expect(container.innerHTML).not.toContain(KEY);

    // Opened again, the form starts empty.
    fireEvent.click(screen.getByRole('button', { name: copy.signIns.add }));
    expect(field('sign-in-site').value).toBe('');
    expect(field('sign-in-username').value).toBe('');
    expect(field('sign-in-password').value).toBe('');
    expect(field('sign-in-key').value).toBe('');
  });

  it('edits with site and username filled and a blank password that keeps the saved one', async () => {
    const calls = mockFetch({
      'GET /api/browser/sign-ins': answer(list(SHOP)),
      'POST /api/browser/sign-ins': answer(list(SHOP)),
    });
    render(<SavedSignIns />);
    const row = await screen.findByTestId('sign-in-shop.example.com-kenny');

    fireEvent.click(within(row).getByRole('button', { name: copy.signIns.edit }));
    expect(field('sign-in-site').value).toBe('shop.example.com');
    expect(field('sign-in-username').value).toBe('kenny');
    expect(field('sign-in-password').value).toBe('');
    expect(field('sign-in-password').placeholder).toBe(copy.signIns.passwordKeep);
    expect(field('sign-in-key').value).toBe('');
    expect(field('sign-in-key').placeholder).toBe(copy.signIns.keyKeep);
    fireEvent.click(screen.getByTestId('sign-in-save'));

    await waitFor(() => expect(screen.queryByTestId('sign-in-form')).toBeNull());
    expect(calls.filter((c) => c.method === 'POST').map((c) => c.body)).toEqual([
      { site: 'shop.example.com', username: 'kenny', password: '', totp_key: '', clear_totp_key: false },
    ]);
  });

  it('removes the 2FA key only when asked, and stops the key field meanwhile', async () => {
    const calls = mockFetch({
      'GET /api/browser/sign-ins': answer(list(SHOP)),
      'POST /api/browser/sign-ins': answer(list({ ...SHOP, has_2fa: false })),
    });
    render(<SavedSignIns />);
    const row = await screen.findByTestId('sign-in-shop.example.com-kenny');
    fireEvent.click(within(row).getByRole('button', { name: copy.signIns.edit }));

    const box = screen.getByRole('checkbox', { name: copy.signIns.clearKey });
    expect(field('sign-in-key').disabled).toBe(false);
    fireEvent.click(box);
    expect(field('sign-in-key').disabled).toBe(true);
    fireEvent.click(screen.getByTestId('sign-in-save'));

    await waitFor(() => expect(screen.queryByTestId('sign-in-form')).toBeNull());
    expect(calls.filter((c) => c.method === 'POST').map((c) => c.body)).toEqual([
      { site: 'shop.example.com', username: 'kenny', password: '', totp_key: '', clear_totp_key: true },
    ]);
  });

  it('offers no key removal for a sign-in without 2FA, nor on Add', async () => {
    mockFetch({ 'GET /api/browser/sign-ins': answer(list(MAIL)) });
    render(<SavedSignIns />);
    const row = await screen.findByTestId('sign-in-mail.example.com-kenny@example.com');
    fireEvent.click(within(row).getByRole('button', { name: copy.signIns.edit }));
    expect(screen.queryByTestId('sign-in-clear-key')).toBeNull();
    expect(field('sign-in-key').placeholder).not.toBe(copy.signIns.keyKeep);
    fireEvent.click(screen.getByTestId('sign-in-cancel'));
    fireEvent.click(screen.getByRole('button', { name: copy.signIns.add }));
    expect(screen.queryByTestId('sign-in-clear-key')).toBeNull();
  });

  it('a new sign-in says the password is needed, not that a blank keeps it', async () => {
    mockFetch({ 'GET /api/browser/sign-ins': answer(list()) });
    render(<SavedSignIns />);
    await screen.findByTestId('sign-ins-empty');
    fireEvent.click(screen.getByRole('button', { name: copy.signIns.add }));
    expect(field('sign-in-password').placeholder).not.toBe(copy.signIns.passwordKeep);
  });

  it('deletes after a confirmation, sending the site and username', async () => {
    const calls = mockFetch({
      'GET /api/browser/sign-ins': answer(list(SHOP, MAIL)),
      'DELETE /api/browser/sign-ins': answer(list(MAIL)),
    });
    render(<SavedSignIns />);
    const row = await screen.findByTestId('sign-in-shop.example.com-kenny');

    fireEvent.click(within(row).getByRole('button', { name: copy.signIns.delete }));
    expect(calls.filter((c) => c.method === 'DELETE')).toEqual([]);
    fireEvent.click(within(row).getByTestId('sign-in-delete-confirm'));

    await waitFor(() => expect(screen.queryByTestId('sign-in-shop.example.com-kenny')).toBeNull());
    expect(calls.filter((c) => c.method === 'DELETE')).toEqual([
      { url: '/api/browser/sign-ins', method: 'DELETE', body: { site: 'shop.example.com', username: 'kenny' } },
    ]);
    expect(screen.getByTestId('sign-in-mail.example.com-kenny@example.com')).toBeInTheDocument();
  });

  const refusals: Array<[number, string, keyof typeof copy.signIns.refused]> = [
    [400, 'site_missing', 'site_missing'],
    [400, 'username_missing', 'username_missing'],
    [400, 'password_missing', 'password_missing'],
    [400, 'setup_key_invalid', 'setup_key_invalid'],
    [503, 'settings_unreadable', 'settings_unreadable'],
    [400, 'something_new', 'other'],
  ];
  it.each(refusals)('a %i %s refusal gets its own sentence, keeps the form, and shows no server text', async (status, code, key) => {
    mockFetch({
      'GET /api/browser/sign-ins': answer(list()),
      'POST /api/browser/sign-ins': answer({ reason_code: code, detail: 'OSError: /Users/x/settings.json' }, status),
    });
    render(<SavedSignIns />);
    await screen.findByTestId('sign-ins-empty');
    fireEvent.click(screen.getByRole('button', { name: copy.signIns.add }));
    fill('sign-in-site', 'shop.example.com');
    fireEvent.click(screen.getByTestId('sign-in-save'));

    const alert = await screen.findByTestId('sign-in-refused');
    expect(alert.textContent).toBe(copy.signIns.refused[key]);
    expectPlain(alert.textContent);
    expect(alert.textContent).not.toContain(code);
    expect(field('sign-in-site').value).toBe('shop.example.com');
  });

  it('gives every refusal a different sentence, each naming its field', () => {
    const r = copy.signIns.refused;
    const sentences = Object.values(r);
    expect(new Set(sentences).size).toBe(sentences.length);
    expect(r.site_missing.toLowerCase()).toContain('site');
    expect(r.username_missing.toLowerCase()).toContain('username');
    expect(r.password_missing.toLowerCase()).toContain('password');
    expect(r.setup_key_invalid).toContain('2FA setup key');
  });

  it('a failed save to an unreachable Core is worded plainly', async () => {
    mockFetch({ 'GET /api/browser/sign-ins': answer(list()) });
    render(<SavedSignIns />);
    await screen.findByTestId('sign-ins-empty');
    vi.stubGlobal(
      'fetch',
      vi.fn(async () => {
        throw new TypeError('Failed to fetch');
      }),
    );
    fireEvent.click(screen.getByRole('button', { name: copy.signIns.add }));
    fireEvent.click(screen.getByTestId('sign-in-save'));
    const alert = await screen.findByTestId('sign-in-refused');
    expect(alert.textContent).toBe(copy.signIns.refused.other);
  });

  it('opens the form on a prefilled site, for the dock’s Save offer', async () => {
    mockFetch({ 'GET /api/browser/sign-ins': answer(list()) });
    render(<SavedSignIns prefill={{ site: 'shop.example.com', key: 1 }} />);
    expect((await screen.findByTestId('sign-in-site') as HTMLInputElement).value).toBe('shop.example.com');
    expect(field('sign-in-username').value).toBe('');
    expect(field('sign-in-password').value).toBe('');
  });

  it('has its sentences in Korean too', async () => {
    mockFetch({ 'GET /api/browser/sign-ins': answer(list(SHOP)) });
    render(<SavedSignIns />, { wrapper: Korean });
    await screen.findByTestId('sign-in-shop.example.com-kenny');
    expect(screen.getByText(CATALOGS.ko.browser.signIns.intro)).toBeInTheDocument();
  });
});

const sessions = (open: boolean, ...sites: Array<{ site: string; last_used: string }>) => ({ open, sites });
const LOCAL = { site: '127.0.0.1', last_used: '2026-10-16T09:00:00+00:00' };
const SHOP_SESSION = { site: 'shop.example.com', last_used: '2026-10-15T18:30:00+00:00' };

describe('SignedInSites', () => {
  it('says no browser is open instead of showing an empty list, and offers no sign-out', async () => {
    mockFetch({ 'GET /api/browser/sessions': answer(sessions(false)) });
    render(<SignedInSites />);
    const line = await screen.findByTestId('signed-in-closed');
    expect(line.textContent).toBe('No browser is open, so there are no signed-in sites to show.');
    expect(screen.queryByTestId('signed-in-none')).toBeNull();
    expect(screen.queryByRole('button', { name: copy.sessions.signOutAll })).toBeNull();
  });

  it('says something different when the browser is open and signed in nowhere', async () => {
    mockFetch({ 'GET /api/browser/sessions': answer(sessions(true)) });
    render(<SignedInSites />);
    const line = await screen.findByTestId('signed-in-none');
    expect(line.textContent).toBe(copy.sessions.none);
    expect(copy.sessions.none).not.toBe(copy.sessions.closed);
    expect(screen.queryByTestId('signed-in-closed')).toBeNull();
  });

  it('lists each site with when it was last used, and signs out of one', async () => {
    const calls = mockFetch({
      'GET /api/browser/sessions': answer(sessions(true, LOCAL, SHOP_SESSION)),
      'POST /api/browser/sessions/sign-out': answer(sessions(true, SHOP_SESSION)),
    });
    render(<SignedInSites />);
    const row = await screen.findByTestId('signed-in-127.0.0.1');
    const when = new Date(LOCAL.last_used).toLocaleString('en', { dateStyle: 'medium', timeStyle: 'short' });
    expect(row).toHaveTextContent('127.0.0.1');
    expect(row).toHaveTextContent(when);
    expect(row.textContent).not.toContain('2026-10-16T09');

    fireEvent.click(within(row).getByRole('button', { name: copy.sessions.signOut }));
    await waitFor(() => expect(screen.queryByTestId('signed-in-127.0.0.1')).toBeNull());
    expect(calls.filter((c) => c.method === 'POST')).toEqual([
      { url: '/api/browser/sessions/sign-out', method: 'POST', body: { site: '127.0.0.1' } },
    ]);
    expect(screen.getByTestId('signed-in-shop.example.com')).toBeInTheDocument();
  });

  it('signs out of everything with an empty body', async () => {
    const calls = mockFetch({
      'GET /api/browser/sessions': answer(sessions(true, LOCAL, SHOP_SESSION)),
      'POST /api/browser/sessions/sign-out': answer(sessions(true)),
    });
    render(<SignedInSites />);
    await screen.findByTestId('signed-in-127.0.0.1');
    fireEvent.click(screen.getByRole('button', { name: copy.sessions.signOutAll }));
    await screen.findByTestId('signed-in-none');
    expect(calls.filter((c) => c.method === 'POST').map((c) => c.body)).toEqual([{}]);
  });

  it('words a failed sign-out plainly', async () => {
    mockFetch({
      'GET /api/browser/sessions': answer(sessions(true, LOCAL)),
      'POST /api/browser/sessions/sign-out': answer({ detail: 'Traceback' }, 500),
    });
    render(<SignedInSites />);
    const row = await screen.findByTestId('signed-in-127.0.0.1');
    fireEvent.click(within(row).getByRole('button', { name: copy.sessions.signOut }));
    const alert = await screen.findByTestId('signed-in-error');
    expect(alert.textContent).toBe(copy.sessions.signOutFailed);
    expectPlain(alert.textContent);
  });

  it('formats the date in Korean for a Korean reader', async () => {
    mockFetch({ 'GET /api/browser/sessions': answer(sessions(true, LOCAL)) });
    render(<SignedInSites />, { wrapper: Korean });
    const row = await screen.findByTestId('signed-in-127.0.0.1');
    const when = new Date(LOCAL.last_used).toLocaleString('ko', { dateStyle: 'medium', timeStyle: 'short' });
    expect(row).toHaveTextContent(when);
  });
});
