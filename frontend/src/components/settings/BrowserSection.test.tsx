import { describe, it, expect, vi, afterEach } from 'vitest';
import { fireEvent, render, screen, waitFor } from '@testing-library/react';
import { BrowserSection, type ExtensionStatus } from './BrowserSection';
import { CATALOGS } from '../../i18n';
import { expectPlain } from '../../test/plainCopy';

const copy = CATALOGS.en.browser;

type Answer = { ok: boolean; status: number; json: () => Promise<unknown> };
const answer = (body: unknown, ok = true, status = 200): Answer => ({ ok, status, json: async () => body });

const status = (overrides: Partial<ExtensionStatus> = {}): ExtensionStatus => ({
  connected: false,
  version: null,
  pairing_code: '8765-' + 'ab'.repeat(16),
  extension_folder: '/Applications/UClone-X/chrome_extension',
  ...overrides,
});

/** GET answers in turn (the last repeats); POST answers `post`. */
const mockFetch = (reads: unknown[], post: () => Answer | Promise<Answer> = () => answer({}, false, 599)) => {
  const calls: Array<{ url: string; method: string }> = [];
  let n = 0;
  vi.stubGlobal(
    'fetch',
    vi.fn(async (url: string, init?: RequestInit) => {
      const method = init?.method ?? 'GET';
      calls.push({ url, method });
      if (method === 'POST' && url === '/api/browser/extension/pairing-code') return post();
      if (method === 'GET' && url === '/api/browser/extension') {
        const body = reads[Math.min(n, reads.length - 1)];
        n += 1;
        return answer(body);
      }
      return answer({}, false, 599);
    }),
  );
  return calls;
};

afterEach(() => {
  vi.unstubAllGlobals();
  vi.useRealTimers();
});

describe('BrowserSection', () => {
  it('shows the steps, the folder and the pairing code while no extension is connected', async () => {
    mockFetch([status()]);
    render(<BrowserSection />);

    const line = await screen.findByTestId('settings-browser-status');
    expect(line.textContent).toBe(copy.notConnected);
    expect(line.getAttribute('data-connected')).toBe('false');
    expect(screen.getByTestId('settings-browser-folder').textContent).toBe(
      '/Applications/UClone-X/chrome_extension',
    );
    expect(screen.getByTestId('settings-browser-code').textContent).toBe('8765-' + 'ab'.repeat(16));
    expect(screen.getByText(copy.stepExtensions)).toBeTruthy();
  });

  it('says it is connected, with the version, and hides the steps', async () => {
    mockFetch([status({ connected: true, version: '0.1.0' })]);
    render(<BrowserSection />);

    const line = await screen.findByTestId('settings-browser-status');
    expect(line.textContent).toBe('Connected. Clones open their tabs in your Chrome (extension 0.1.0).');
    expect(screen.queryByTestId('settings-browser-steps')).toBeNull();
  });

  it('reads again while unpaired and shows the extension connecting', async () => {
    const calls = mockFetch([status(), status({ connected: true, version: null })]);
    render(<BrowserSection pollMs={10} />);

    await waitFor(() =>
      expect(screen.getByTestId('settings-browser-status').getAttribute('data-connected')).toBe('true'),
    );
    expect(screen.getByTestId('settings-browser-status').textContent).toBe(copy.connected);
    const reads = calls.filter((c) => c.method === 'GET').length;
    await new Promise((resolve) => setTimeout(resolve, 50));
    // Connected: no more polling.
    expect(calls.filter((c) => c.method === 'GET').length).toBe(reads);
  });

  it('shows the new pairing code at once', async () => {
    const replaced = status({ pairing_code: '8765-' + 'cd'.repeat(16) });
    const calls = mockFetch([status()], () => answer(replaced));
    render(<BrowserSection pollMs={60_000} />);

    fireEvent.click(await screen.findByTestId('settings-browser-new-code'));

    await waitFor(() =>
      expect(screen.getByTestId('settings-browser-code').textContent).toBe('8765-' + 'cd'.repeat(16)),
    );
    expect(calls.filter((c) => c.method === 'POST')).toHaveLength(1);
  });

  it('words a failed new code plainly, with no status or internals', async () => {
    mockFetch([status()], () => answer({ detail: 'OSError: [Errno 13] /Users/x/settings.json' }, false, 503));
    render(<BrowserSection pollMs={60_000} />);

    fireEvent.click(await screen.findByTestId('settings-browser-new-code'));

    const alert = await screen.findByTestId('settings-browser-new-code-error');
    expect(alert.textContent).toBe(copy.newCodeFailed);
    expectPlain(alert.textContent);
  });

  it('words an unreachable Core plainly and offers to read again', async () => {
    vi.stubGlobal(
      'fetch',
      vi.fn(async () => {
        throw new TypeError('Failed to fetch');
      }),
    );
    vi.spyOn(console, 'error').mockImplementation(() => {});
    render(<BrowserSection />);

    const failure = await screen.findByTestId('settings-browser-error');
    expect(failure.textContent).toContain(copy.loadFailed);
    expect(failure.textContent).not.toContain('Failed to fetch');
  });

  it('has every sentence in Korean too, and all of them plain', () => {
    const ko = CATALOGS.ko.browser;
    expect(Object.keys(ko).sort()).toEqual(Object.keys(copy).sort());
    for (const key of ['notConnected', 'connected', 'newCodeFailed', 'copyFailed', 'loadFailed'] as const) {
      expectPlain(copy[key]);
      expectPlain(ko[key]);
    }
  });
});
