import { describe, it, expect, vi, beforeEach, afterEach } from 'vitest';
import { fireEvent, render, screen, waitFor, within } from '@testing-library/react';
import { LinksSection } from './LinksSection';
import { CloneLinkLine } from '../clones/CloneLinkLine';
import { CATALOGS } from '../../i18n';
import { LINK_ERROR_CODES, LINK_STATES, type LinkCard, type LinkState } from '../../lib/linksApi';
import { expectPlain } from '../../test/plainCopy';

type RouteResponse = { ok: boolean; status: number; json: () => Promise<unknown> };
type Handler = (url: string, init?: RequestInit) => RouteResponse | Promise<RouteResponse>;

const jsonResponse = (body: unknown, ok = true, status = 200): RouteResponse => ({
  ok,
  status,
  json: async () => body,
});

const card = (overrides: Partial<LinkCard> = {}): LinkCard => ({
  link_id: 'lnk_1',
  local_agent_id: 'haru',
  remote_username: 'haru_garden',
  remote_display_name: 'Haru',
  page_url: 'https://uclone.test/hompy/haru_garden',
  state: 'online',
  created_at: '2026-09-27T00:00:00+00:00',
  last_connected_at: null,
  ...overrides,
});

/**
 * Routes by "METHOD /path". `lists` answers GET /api/links in turn (the last one repeats);
 * an unrouted request answers 599 and so fails the test that made it.
 */
const mockFetch = (lists: unknown[], routes: Record<string, Handler> = {}) => {
  const calls: Array<{ url: string; method: string; body?: unknown }> = [];
  let reads = 0;
  const fetchMock = vi.fn(async (url: string, init?: RequestInit) => {
    const method = init?.method ?? 'GET';
    const body = init?.body ? JSON.parse(init.body as string) : undefined;
    calls.push({ url, method, body });
    const handler = routes[`${method} ${url}`];
    if (handler) return handler(url, init);
    if (method === 'GET' && url === '/api/links') {
      const answer = lists[Math.min(reads, lists.length - 1)];
      reads += 1;
      return jsonResponse(answer);
    }
    if (method === 'GET' && url === '/api/personas') {
      return jsonResponse({ personas: [{ name: 'clone' }, { name: 'haru' }] });
    }
    return jsonResponse({ detail: `unrouted ${method} ${url}` }, false, 599);
  });
  vi.stubGlobal('fetch', fetchMock);
  return calls;
};

const listOf = (...links: LinkCard[]) => ({ links, unreadable: false });
const sent = (calls: ReturnType<typeof mockFetch>, method: string, url: string) =>
  calls.filter((c) => c.method === method && c.url === url);

const en = CATALOGS.en.links;

describe('LinksSection', () => {
  beforeEach(() => vi.restoreAllMocks());
  afterEach(() => vi.unstubAllGlobals());

  it('says no clone is linked, and offers the connect form', async () => {
    mockFetch([listOf()]);
    render(<LinksSection />);

    expect(await screen.findByTestId('settings-links-empty')).toHaveTextContent(en.empty);
    expect(screen.getByTestId('settings-links-input')).toBeTruthy();
    const options = within(screen.getByTestId('settings-links-clone')).getAllByRole('option');
    await waitFor(() => expect(options.length).toBeGreaterThan(0));
  });

  it('shows the clone pair and a link to the minihompy', async () => {
    mockFetch([listOf(card())]);
    render(<LinksSection />);

    const row = await screen.findByTestId('link-card-lnk_1');
    expect(row).toHaveTextContent('haru');
    const page = within(row).getByTestId('link-card-page');
    expect(page).toHaveTextContent('@haru_garden');
    expect(page.getAttribute('href')).toBe('https://uclone.test/hompy/haru_garden');
    expect(page.getAttribute('rel')).toContain('noopener');
  });

  it('does not link a page address that is not a web page', async () => {
    mockFetch([listOf(card({ page_url: 'javascript:alert(1)' }))]);
    render(<LinksSection />);

    const row = await screen.findByTestId('link-card-lnk_1');
    expect(within(row).queryByTestId('link-card-page')).toBeNull();
    expect(row).toHaveTextContent('@haru_garden');
  });

  // Killed by: frontend/src/components/settings/LinksSection.tsx :: {copy.state[link.state]}
  // Becomes: {copy.state.online}
  it('words a reconnecting link as reconnecting, not as online', async () => {
    mockFetch([listOf(card({ state: 'reconnecting' }))]);
    render(<LinksSection pollMs={60_000} />);

    const line = await screen.findByTestId('link-card-state');
    expect(line.textContent).toBe(en.state.reconnecting);
    expect(line.textContent).not.toBe(en.state.online);
  });

  it.each(LINK_STATES.map((s) => [s]))('words the %s state with its own sentence', async (state) => {
    mockFetch([listOf(card({ state: state as LinkState }))]);
    render(<LinksSection pollMs={60_000} />);

    const line = await screen.findByTestId('link-card-state');
    expect(line.textContent).toBe(en.state[state as LinkState]);
    expectPlain(line.textContent);
  });

  // Killed by: frontend/src/components/settings/LinksSection.tsx :: || state === 'elsewhere') return null;
  // Becomes: ) return null;
  it('offers no switch for a link another UClone-X on this computer is serving', async () => {
    mockFetch([listOf(card({ state: 'elsewhere' }))]);
    render(<LinksSection pollMs={60_000} />);

    const row = await screen.findByTestId('link-card-lnk_1');
    expect(within(row).getByTestId('link-card-state').textContent).toBe(en.state.elsewhere);
    expect(within(row).queryByTestId('link-card-toggle')).toBeNull();
  });

  it('takes an online clone offline, and switches a paused one back on', async () => {
    const calls = mockFetch([listOf(card({ state: 'online' }), card({ link_id: 'lnk_2', state: 'paused' }))], {
      'POST /api/links/lnk_1/enabled': () => jsonResponse(card({ state: 'paused' })),
      'POST /api/links/lnk_2/enabled': () => jsonResponse(card({ link_id: 'lnk_2', state: 'connecting' })),
    });
    render(<LinksSection pollMs={60_000} />);

    const first = await screen.findByTestId('link-card-lnk_1');
    const off = within(first).getByTestId('link-card-toggle');
    expect(off).toHaveTextContent(en.card.goOffline);
    fireEvent.click(off);
    await waitFor(() => expect(sent(calls, 'POST', '/api/links/lnk_1/enabled')).toHaveLength(1));
    expect(sent(calls, 'POST', '/api/links/lnk_1/enabled')[0].body).toEqual({ enabled: false });

    const on = within(screen.getByTestId('link-card-lnk_2')).getByTestId('link-card-toggle');
    expect(on).toHaveTextContent(en.card.goOnline);
    fireEvent.click(on);
    await waitFor(() => expect(sent(calls, 'POST', '/api/links/lnk_2/enabled')).toHaveLength(1));
    expect(sent(calls, 'POST', '/api/links/lnk_2/enabled')[0].body).toEqual({ enabled: true });
  });

  it('offers no switch on an ended link', async () => {
    mockFetch([listOf(card({ state: 'ended' }))]);
    render(<LinksSection />);

    const row = await screen.findByTestId('link-card-lnk_1');
    expect(within(row).queryByTestId('link-card-toggle')).toBeNull();
  });

  it('unlinks only after the confirm step', async () => {
    const calls = mockFetch([listOf(card()), listOf()], {
      'DELETE /api/links/lnk_1': () => jsonResponse({ outcome: 'removed' }),
    });
    render(<LinksSection />);

    fireEvent.click(await screen.findByTestId('link-card-unlink'));
    const confirm = screen.getByTestId('link-unlink-confirm');
    expect(confirm).toHaveTextContent('@haru_garden');
    fireEvent.click(within(confirm).getByRole('button', { name: en.card.cancel }));
    expect(screen.queryByTestId('link-unlink-confirm')).toBeNull();
    expect(sent(calls, 'DELETE', '/api/links/lnk_1')).toHaveLength(0);

    fireEvent.click(screen.getByTestId('link-card-unlink'));
    fireEvent.click(within(screen.getByTestId('link-unlink-confirm')).getByRole('button', { name: en.card.confirm }));
    await waitFor(() => expect(sent(calls, 'DELETE', '/api/links/lnk_1')).toHaveLength(1));
    expect(await screen.findByTestId('settings-links-empty')).toBeTruthy();
  });

  it('says an unlink uClone2 did not confirm is pending, and offers to remove it here only', async () => {
    const calls = mockFetch([listOf(card()), listOf(card({ state: 'unlink_pending' })), listOf()], {
      'DELETE /api/links/lnk_1': () => jsonResponse({ outcome: 'pending' }),
      'DELETE /api/links/lnk_1?local=true': () => jsonResponse({ outcome: 'forgotten' }),
    });
    render(<LinksSection />);

    fireEvent.click(await screen.findByTestId('link-card-unlink'));
    fireEvent.click(within(screen.getByTestId('link-unlink-confirm')).getByRole('button', { name: en.card.confirm }));
    expect(await screen.findByTestId('link-card-pending-note')).toHaveTextContent(en.card.unlinkPending);
    expect(screen.getByTestId('link-card-state')).toHaveTextContent(en.state.unlink_pending);
    // The copy says the link may outlive the local record.
    expect(screen.getByTestId('link-card-forget-help')).toHaveTextContent('may still exist in uClone2');

    fireEvent.click(screen.getByTestId('link-card-forget'));
    const confirm = screen.getByTestId('link-forget-confirm');
    expect(confirm).toHaveTextContent('may still exist in uClone2');
    fireEvent.click(within(confirm).getByRole('button', { name: en.card.confirmForgetButton }));
    await waitFor(() => expect(sent(calls, 'DELETE', '/api/links/lnk_1?local=true')).toHaveLength(1));
    expect(await screen.findByTestId('settings-links-empty')).toBeTruthy();
  });

  it('links from a pasted URL with the chosen clone, then clears the spent code', async () => {
    const calls = mockFetch([listOf(), listOf(card({ local_agent_id: 'clone', state: 'connecting' }))], {
      'POST /api/links/uclone2': () => jsonResponse(card({ local_agent_id: 'clone', state: 'connecting' })),
    });
    render(<LinksSection pollMs={60_000} />);

    await screen.findByTestId('settings-links-empty');
    fireEvent.change(screen.getByTestId('settings-links-input'), {
      target: { value: '  https://uclone.test/link/Ab3dE6gH9j ' },
    });
    await screen.findByRole('option', { name: 'clone' });
    fireEvent.change(screen.getByTestId('settings-links-clone'), { target: { value: 'clone' } });
    fireEvent.click(screen.getByTestId('settings-links-submit'));

    expect(await screen.findByTestId('settings-links-linked')).toHaveTextContent('@haru_garden');
    expect(sent(calls, 'POST', '/api/links/uclone2')[0].body).toEqual({
      connect: 'https://uclone.test/link/Ab3dE6gH9j',
      local_agent_id: 'clone',
    });
    expect((screen.getByTestId('settings-links-input') as HTMLInputElement).value).toBe('');
    expect(await screen.findByTestId('link-card-lnk_1')).toBeTruthy();
  });

  it('sends "the same name" as no clone', async () => {
    const calls = mockFetch([listOf()], {
      'POST /api/links/uclone2': () => jsonResponse(card()),
    });
    render(<LinksSection />);

    await screen.findByTestId('settings-links-empty');
    fireEvent.change(screen.getByTestId('settings-links-input'), { target: { value: 'Ab3dE6gH9j' } });
    fireEvent.click(screen.getByTestId('settings-links-submit'));
    await waitFor(() => expect(sent(calls, 'POST', '/api/links/uclone2')).toHaveLength(1));
    expect(sent(calls, 'POST', '/api/links/uclone2')[0].body).toEqual({ connect: 'Ab3dE6gH9j', local_agent_id: null });
  });

  it('asks for a URL before sending anything', async () => {
    const calls = mockFetch([listOf()]);
    render(<LinksSection />);

    await screen.findByTestId('settings-links-empty');
    fireEvent.click(screen.getByTestId('settings-links-submit'));
    expect(await screen.findByTestId('settings-links-connect-error')).toHaveTextContent(en.connect.empty);
    expect(sent(calls, 'POST', '/api/links/uclone2')).toHaveLength(0);
  });

  it('words a refusal from its code, never from what the server said', async () => {
    mockFetch([listOf()], {
      'POST /api/links/uclone2': () =>
        jsonResponse({ code: 'code_invalid', message: '400 LINKED_CODE_INVALID redis nil', detail: 'x' }, false, 400),
    });
    render(<LinksSection />);

    await screen.findByTestId('settings-links-empty');
    fireEvent.change(screen.getByTestId('settings-links-input'), { target: { value: 'Ab3dE6gH9j' } });
    fireEvent.click(screen.getByTestId('settings-links-submit'));

    const error = await screen.findByTestId('settings-links-connect-error');
    expect(error.textContent).toBe(en.errors.code_invalid);
    expectPlain(error.textContent);
    expect(document.body.textContent).not.toContain('LINKED_');
  });

  it('says an unknown refusal and an unreachable app in plain words', async () => {
    let answer: () => RouteResponse | Promise<RouteResponse> = () => jsonResponse({ detail: 'Traceback' }, false, 500);
    mockFetch([listOf()], { 'POST /api/links/uclone2': () => answer() });
    render(<LinksSection />);

    await screen.findByTestId('settings-links-empty');
    const input = screen.getByTestId('settings-links-input');
    fireEvent.change(input, { target: { value: 'Ab3dE6gH9j' } });
    fireEvent.click(screen.getByTestId('settings-links-submit'));
    const other = await screen.findByTestId('settings-links-connect-error');
    expect(other.textContent).toBe(en.errors.other);

    answer = () => Promise.reject(new TypeError('Failed to fetch'));
    fireEvent.change(input, { target: { value: 'Ab3dE6gH9j' } });
    fireEvent.click(screen.getByTestId('settings-links-submit'));
    await waitFor(() =>
      expect(screen.getByTestId('settings-links-connect-error').textContent).toBe(en.errors.offline),
    );
    expectPlain(screen.getByTestId('settings-links-connect-error').textContent);
  });

  it('says a damaged links file is unreadable, not that nothing is linked', async () => {
    mockFetch([{ links: [], unreadable: true }]);
    render(<LinksSection />);

    expect(await screen.findByTestId('settings-links-unreadable')).toHaveTextContent(en.unreadable);
    expect(screen.queryByTestId('settings-links-empty')).toBeNull();
  });

  it('says a failed read as one, in plain words', async () => {
    mockFetch([{ unexpected: true }]);
    render(<LinksSection />);

    const failure = await screen.findByTestId('settings-links-error');
    expect(failure).toHaveTextContent(en.loadFailed);
    expectPlain(failure.textContent);
  });

  it('reads the list again while a link is connecting', async () => {
    const calls = mockFetch([listOf(card({ state: 'connecting' })), listOf(card({ state: 'online' }))]);
    render(<LinksSection pollMs={10} />);

    await waitFor(() => expect(screen.getByTestId('link-card-state')).toHaveTextContent(en.state.online));
    expect(sent(calls, 'GET', '/api/links').length).toBeGreaterThanOrEqual(2);
  });

  it('has a plain sentence for every state and every refusal, in both languages', () => {
    for (const lang of ['en', 'ko'] as const) {
      const links = CATALOGS[lang].links;
      for (const state of LINK_STATES) expectPlain(links.state[state]);
      for (const code of LINK_ERROR_CODES) expectPlain(links.errors[code]);
    }
  });
});

describe('CloneLinkLine', () => {
  afterEach(() => vi.unstubAllGlobals());

  it('says the clone is active in uClone2 while its link is online', async () => {
    mockFetch([listOf(card({ local_agent_id: 'haru', state: 'online' }))]);
    render(<CloneLinkLine cloneId="haru" />);

    const line = await screen.findByTestId('clone-link-line');
    expect(line).toHaveTextContent('Active in uClone2 · @haru_garden');
    expect(within(line).getByRole('link').getAttribute('href')).toBe('https://uclone.test/hompy/haru_garden');
  });

  // Killed by: frontend/src/components/clones/CloneLinkLine.tsx :: const timer = window.setTimeout(() => reloadRef.current(), pollMs);
  // Becomes: const timer = 0;
  it('takes the line away when the link drops while the page is open', async () => {
    mockFetch([
      listOf(card({ local_agent_id: 'haru', state: 'online' })),
      listOf(card({ local_agent_id: 'haru', state: 'reconnecting' })),
    ]);
    render(<CloneLinkLine cloneId="haru" pollMs={10} />);

    expect(await screen.findByTestId('clone-link-line')).toHaveTextContent('Active in uClone2 · @haru_garden');
    await waitFor(() => expect(screen.queryByTestId('clone-link-line')).toBeNull());
  });

  // Killed by: frontend/src/components/clones/CloneLinkLine.tsx :: return () => window.clearTimeout(timer);
  // Becomes: return () => undefined;
  it('stops reading the list once the page is closed', async () => {
    const calls = mockFetch([listOf(card({ local_agent_id: 'haru', state: 'online' }))]);
    const { unmount } = render(<CloneLinkLine cloneId="haru" pollMs={20} />);

    await screen.findByTestId('clone-link-line');
    unmount();
    const reads = sent(calls, 'GET', '/api/links').length;
    await new Promise((r) => setTimeout(r, 100));
    expect(sent(calls, 'GET', '/api/links')).toHaveLength(reads);
  });

  it.each([
    ['another clone is linked', card({ local_agent_id: 'mina' })],
    ['the link is offline', card({ state: 'paused' })],
    ['the link is reconnecting', card({ state: 'reconnecting' })],
  ])('says nothing when %s', async (_why, link) => {
    const calls = mockFetch([listOf(link)]);
    render(<CloneLinkLine cloneId="haru" />);

    await waitFor(() => expect(sent(calls, 'GET', '/api/links')).toHaveLength(1));
    await new Promise((r) => setTimeout(r, 0));
    expect(screen.queryByTestId('clone-link-line')).toBeNull();
  });
});
