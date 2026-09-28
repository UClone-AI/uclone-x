import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest';
import { confirmThisWindow, confirmWindowsOpenedHere, openConfirmedWindow } from './person';
import { CoreFailure } from './coreFailure';

// A window the Core opened carries its one-time code after `#pair=` (#1589): the page spends it
// and takes it out of the address bar, so neither a reload nor a copied address carries it.

let calls: { url: string; method: string; body: unknown }[];

beforeEach(() => {
  calls = [];
  vi.spyOn(console, 'error').mockImplementation(() => {});
  vi.stubGlobal(
    'fetch',
    vi.fn(async (url: string, init?: RequestInit) => {
      calls.push({ url, method: init?.method ?? 'GET', body: init?.body ? JSON.parse(String(init.body)) : undefined });
      return new Response(JSON.stringify({ confirmed: true }), { status: 200 });
    }),
  );
});

afterEach(() => {
  window.history.replaceState(null, '', '/');
  vi.unstubAllGlobals();
  vi.restoreAllMocks();
});

describe('confirmThisWindow', () => {
  it('spends the code the window was opened with, and removes it from the address', async () => {
    // Killed by: frontend/src/lib/person.ts :: win.history.replaceState(win.history.state, '', `${pathname}${search}`);
    // Becomes: void 0;
    window.history.replaceState(null, '', '/files?x=1#pair=one-time');

    await confirmThisWindow();

    expect(calls).toEqual([{ url: '/api/person/pair', method: 'POST', body: { code: 'one-time' } }]);
    expect(window.location.hash).toBe('');
    expect(`${window.location.pathname}${window.location.search}`).toBe('/files?x=1');
  });

  it('sends nothing from a window opened without a code', async () => {
    // Killed by: frontend/src/lib/person.ts :: if (!hash.startsWith(PAIR_PREFIX)) return;
    // Becomes: if (false) return;
    window.history.replaceState(null, '', '/#section');

    await confirmThisWindow();

    expect(calls).toEqual([]);
    expect(window.location.hash).toBe('#section');
  });
});

describe('openConfirmedWindow', () => {
  it("asks the Core to open a window, and carries the Core's refusal when it cannot", async () => {
    // Killed by: frontend/src/lib/person.ts :: if (!res.ok) throw await failureOf(res);
    // Becomes: if (false) throw await failureOf(res);
    await openConfirmedWindow();
    expect(calls).toEqual([{ url: '/api/person/window', method: 'POST', body: undefined }]);

    vi.stubGlobal(
      'fetch',
      vi.fn(async () => new Response(JSON.stringify({ detail: 'No browser window could be opened.' }), { status: 500 })),
    );
    await expect(openConfirmedWindow()).rejects.toEqual(new CoreFailure(500, 'No browser window could be opened.'));
  });
});

describe('confirmWindowsOpenedHere', () => {
  it('confirms a tab the browser reused, where only the code after # changed', async () => {
    // Killed by: frontend/src/lib/person.ts :: win.addEventListener('hashchange', () => void confirmThisWindow(win));
    // Becomes: void 0;
    const win = new EventTarget() as unknown as Window;
    const location = { hash: '', pathname: '/', search: '' };
    Object.assign(win, {
      location,
      history: { state: null, replaceState: () => (location.hash = '') },
    });

    confirmWindowsOpenedHere(win);
    expect(calls).toEqual([]);
    location.hash = '#pair=reused-tab';
    win.dispatchEvent(new Event('hashchange'));

    await vi.waitFor(() => expect(calls).toEqual([{ url: '/api/person/pair', method: 'POST', body: { code: 'reused-tab' } }]));
    expect(location.hash).toBe('');
  });
});
