import { describe, it, expect, vi, beforeEach, afterEach } from 'vitest';
import { fireEvent, render, screen, waitFor } from '@testing-library/react';
import { DiagnosticsPanel } from './DiagnosticsPanel';
import { expectPlain } from '../test/plainCopy';
import { SETTINGS_FAILURE } from '../lib/settingsCopy';

/**
 * The panel's job is to ask a question and then not act on its own.
 *
 * The test that matters is the negative one: with failures recorded and a
 * report ready, opening the settings dialog must POST nothing and send
 * nothing. A panel that "helpfully" reported on render would look identical in
 * a screenshot and be a different product.
 */

const consent = (state: string, error: string | null = null) => ({
  state,
  error,
  journal: '/home/u/.uclone/diagnostics/failures.jsonl',
});

const reportBody = {
  state: 'granted',
  available: true,
  error: null,
  unreadable_lines: 0,
  recording_blocked: null,
  count: 3,
  distinct: 2,
  title: '[ucx a1b2c3d4] ValueError',
  body: '## Environment\n\n- UClone-X: 0.1.1',
  issue_url: 'https://github.com/UClone-AI/uclone-x/issues/new?template=bug.yml',
  search_url: 'https://github.com/search?q=a1b2c3d4',
};

const mockFetch = (
  consentState: string,
  report: Record<string, unknown>,
  ok = true,
  status = 200,
) => {
  const calls: Array<{ url: string; method: string }> = [];
  const fetchMock = vi.fn(async (url: string, init?: RequestInit) => {
    calls.push({ url, method: init?.method ?? 'GET' });
    const payload = url.includes('/consent') ? consent(consentState) : report;
    return { ok, status, json: async () => payload } as Response;
  });
  vi.stubGlobal('fetch', fetchMock);
  return calls;
};

beforeEach(() => vi.restoreAllMocks());
afterEach(() => vi.unstubAllGlobals());

describe('DiagnosticsPanel', () => {
  it('offers the choice when nobody has been asked', async () => {
    mockFetch('unasked', { ...reportBody, state: 'unasked', count: 0, distinct: 0 });

    render(<DiagnosticsPanel />);

    expect(await screen.findByRole('button', { name: /record failures locally/i })).toBeTruthy();
    expect(screen.getByText(/on this machine only/i)).toBeTruthy();
  });

  it('sends nothing on render, even with a report ready', async () => {
    const calls = mockFetch('granted', reportBody);

    render(<DiagnosticsPanel />);
    await screen.findByText(/3 failures recorded/i);

    // Reads only. A POST or DELETE here would mean the panel acted without
    // being asked, which is the whole thing this design is built to avoid.
    expect(calls.every((call) => call.method === 'GET')).toBe(true);
  });

  it('links to the pre-filled issue rather than submitting it', async () => {
    mockFetch('granted', reportBody);

    render(<DiagnosticsPanel />);

    const link = await screen.findByRole('link', { name: /report on github/i });
    expect(link.getAttribute('href')).toBe(reportBody.issue_url);
    expect(link.getAttribute('target')).toBe('_blank');
  });

  it('offers the text to copy when the report will not fit in a URL', async () => {
    mockFetch('granted', { ...reportBody, issue_url: null });

    render(<DiagnosticsPanel />);

    expect(await screen.findByText(/too long to pre-fill/i)).toBeTruthy();
    expect(screen.queryByRole('link', { name: /report on github/i })).toBeNull();
  });

  it('records the answer when the user turns recording on', async () => {
    const calls = mockFetch('unasked', { ...reportBody, state: 'unasked', count: 0 });

    render(<DiagnosticsPanel />);
    fireEvent.click(await screen.findByRole('button', { name: /record failures locally/i }));

    await waitFor(() => {
      expect(calls.some((c) => c.method === 'POST' && c.url.includes('/consent'))).toBe(true);
    });
  });

  it('shows the report text only when asked to', async () => {
    mockFetch('granted', reportBody);

    render(<DiagnosticsPanel />);
    await screen.findByText(/3 failures recorded/i);

    expect(screen.queryByLabelText(/report contents/i)).toBeNull();
    fireEvent.click(screen.getByRole('button', { name: /show what would be sent/i }));
    expect(screen.getByLabelText(/report contents/i)).toBeTruthy();
  });

  it('says the journal is unreadable rather than showing nothing to report', async () => {
    // `count: 0` with `available: false` is a broken journal, not a quiet one.
    // Rendering the empty state over it is the substitution the backend was
    // fixed to stop making; the panel must not reintroduce it.
    mockFetch('granted', {
      ...reportBody,
      available: false,
      error: '/home/u/.uclone/diagnostics/failures.jsonl could not be read: PermissionError',
      count: 0,
      distinct: 0,
    });

    render(<DiagnosticsPanel />);

    // Both the heading and the raw error mention it, hence `findAllByText`.
    expect((await screen.findAllByText(/could not be read/i)).length).toBeGreaterThan(0);
    expect(screen.getByText(/PermissionError/)).toBeTruthy();
    expect(screen.queryByRole('link', { name: /report on github/i })).toBeNull();
  });

  it('says the service is unreachable instead of showing the calm empty state', async () => {
    // A 500 or a 403 used to leave `available` undefined, so `=== false` was
    // false and the panel rendered "nothing recorded" over a failing backend —
    // the substitution the backend was fixed to stop making, one layer up.
    mockFetch('granted', reportBody, false, 500);

    render(<DiagnosticsPanel />);

    expect(await screen.findByText(SETTINGS_FAILURE.diagnosticsRead)).toBeTruthy();
    expect(screen.queryByText(/failures recorded/i)).toBeNull();
  });

  it('says so when the choice could not be saved', async () => {
    // The write paths had no `ok` check, so an unwritable diagnostics
    // directory made the POST 500, the following refresh succeeded, and the
    // panel showed "Currently off." after the user had just asked for on —
    // the answer silently not taken. Reachable with no attacker.
    const calls: Array<{ url: string; method: string }> = [];
    vi.stubGlobal(
      'fetch',
      vi.fn(async (url: string, init?: RequestInit) => {
        const method = init?.method ?? 'GET';
        calls.push({ url, method });
        if (method === 'POST') {
          return { ok: false, status: 500, json: async () => ({}) } as Response;
        }
        const payload = url.includes('/consent')
          ? consent('unasked')
          : { ...reportBody, state: 'unasked', count: 0, distinct: 0 };
        return { ok: true, status: 200, json: async () => payload } as Response;
      }),
    );

    render(<DiagnosticsPanel />);
    fireEvent.click(await screen.findByRole('button', { name: /record failures locally/i }));

    expect(await screen.findByText(SETTINGS_FAILURE.diagnosticsChoice)).toBeTruthy();
  });

  it('does not leave a stale failure banner beside a state it no longer describes', async () => {
    // The error was cleared only by the mount effect, so after a failed save
    // and a successful retry the panel showed "Recording on" with "Could not
    // reach the diagnostics service." still beside it, permanently.
    let failNextPost = true;
    vi.stubGlobal(
      'fetch',
      vi.fn(async (url: string, init?: RequestInit) => {
        if ((init?.method ?? 'GET') === 'POST') {
          const failing = failNextPost;
          failNextPost = false;
          return { ok: !failing, status: failing ? 500 : 200, json: async () => ({}) } as Response;
        }
        const payload = url.includes('/consent')
          ? consent('unasked')
          : { ...reportBody, state: 'unasked', count: 0, distinct: 0 };
        return { ok: true, status: 200, json: async () => payload } as Response;
      }),
    );

    render(<DiagnosticsPanel />);
    const button = await screen.findByRole('button', { name: /record failures locally/i });
    fireEvent.click(button);
    expect(await screen.findByTestId('diagnostics-failure')).toBeTruthy();

    fireEvent.click(screen.getByRole('button', { name: /record failures locally/i }));
    await waitFor(() => {
      expect(screen.queryByTestId('diagnostics-failure')).toBeNull();
    });
  });

  it('says so when the preference itself could not be read', async () => {
    mockFetch('unasked', { ...reportBody, state: 'unasked', count: 0, distinct: 0 });
    vi.stubGlobal(
      'fetch',
      vi.fn(async (url: string) => ({
        ok: true,
        status: 200,
        json: async () =>
          url.includes('/consent')
            ? consent('unasked', 'consent.json is not readable JSON')
            : { ...reportBody, state: 'unasked', count: 0, distinct: 0 },
      })) as unknown as typeof fetch,
    );

    render(<DiagnosticsPanel />);

    expect(await screen.findByText(/preference could not be read/i)).toBeTruthy();
    expect(screen.getByText(/not readable JSON/)).toBeTruthy();
  });

  it('says when nothing new is being recorded, even with entries on screen', async () => {
    // The journal is readable and its old entries are intact; what is broken
    // is that nothing more is being kept. Shown with the entries, because a
    // reader given only the entries concludes nothing else has gone wrong.
    mockFetch('granted', {
      ...reportBody,
      recording_blocked: 'failures cannot be recorded: failures.jsonl is not writable',
    });

    render(<DiagnosticsPanel />);

    expect(await screen.findByText(/new failures are not being recorded/i)).toBeTruthy();
    expect(screen.getByText(/is not writable/)).toBeTruthy();
    expect(screen.getByText(/3 failures recorded/i)).toBeTruthy();
  });

  it('asks before deleting the recorded failures', async () => {
    const calls = mockFetch('granted', reportBody);
    const confirmSpy = vi.fn(() => false);
    vi.stubGlobal('confirm', confirmSpy);

    render(<DiagnosticsPanel />);
    await screen.findByText(/3 failures recorded/i);
    fireEvent.click(screen.getByRole('button', { name: /delete/i }));

    expect(confirmSpy).toHaveBeenCalled();
    expect(calls.some((c) => c.method === 'DELETE')).toBe(false);
  });

  describe('failures in plain words (#1436)', () => {
    /** The two answers a reader must never see as text: no answer at all, and a 500 with no JSON. */
    const faults: Array<[string, () => Promise<Response>]> = [
      ['a rejected fetch', () => Promise.reject(new TypeError('Failed to fetch'))],
      [
        'a bodyless 500',
        async () =>
          ({
            ok: false,
            status: 500,
            statusText: 'Internal Server Error',
            json: async () => {
              throw new SyntaxError('Unexpected token \'I\', "Internal S"... is not valid JSON');
            },
          }) as unknown as Response,
      ],
    ];

    /** Answers every request normally except `method`, which meets `fault`. */
    const stubFailing = (method: string, fault: () => Promise<Response>, state = 'unasked') =>
      vi.stubGlobal(
        'fetch',
        vi.fn(async (url: string, init?: RequestInit) => {
          if ((init?.method ?? 'GET') === method) return fault();
          const payload = url.includes('/consent') ? consent(state) : { ...reportBody, state };
          return { ok: true, status: 200, json: async () => payload } as Response;
        }),
      );

    beforeEach(() => {
      vi.spyOn(console, 'error').mockImplementation(() => {});
      vi.stubGlobal('confirm', vi.fn(() => true));
    });

    it.each([
      ...faults,
    ])('says problem reporting could not be loaded after %s', async (_label, fault) => {
      // Killed by: frontend/src/components/DiagnosticsPanel.tsx :: setFetchError(SETTINGS_FAILURE.diagnosticsRead);
      // Becomes: setFetchError(String(err));
      stubFailing('GET', fault);
      render(<DiagnosticsPanel />);

      const failure = await screen.findByTestId('diagnostics-failure');
      expect(failure).toHaveTextContent(SETTINGS_FAILURE.diagnosticsRead);
      expectPlain(failure.textContent);
    });

    it.each([
      ...faults,
    ])('says saving the choice went wrong after %s', async (_label, fault) => {
      // Killed by: frontend/src/components/DiagnosticsPanel.tsx :: setFetchError(SETTINGS_FAILURE.diagnosticsChoice);
      // Becomes: setFetchError(String(err));
      stubFailing('POST', fault);
      render(<DiagnosticsPanel />);
      fireEvent.click(await screen.findByRole('button', { name: /record failures locally/i }));

      const failure = await screen.findByTestId('diagnostics-failure');
      expect(failure).toHaveTextContent(SETTINGS_FAILURE.diagnosticsChoice);
      expectPlain(failure.textContent);
    });

    it.each([
      ...faults,
    ])('says deleting the recorded failures went wrong after %s', async (_label, fault) => {
      // Killed by: frontend/src/components/DiagnosticsPanel.tsx :: setFetchError(SETTINGS_FAILURE.diagnosticsClear);
      // Becomes: setFetchError(String(err));
      stubFailing('DELETE', fault, 'granted');
      render(<DiagnosticsPanel />);
      await screen.findByText(/3 failures recorded/i);
      fireEvent.click(screen.getByRole('button', { name: /delete/i }));

      const failure = await screen.findByTestId('diagnostics-failure');
      expect(failure).toHaveTextContent(SETTINGS_FAILURE.diagnosticsClear);
      expectPlain(failure.textContent);
    });

    /** A 500 carrying the `detail` the Core really sends, which names a path and an exception. */
    const coreRefusal = (detail: string) => async () =>
      ({ ok: false, status: 500, json: async () => ({ detail }) }) as unknown as Response;

    it("keeps the Core's reason off the screen when saving the choice fails", async () => {
      // Killed by: frontend/src/components/DiagnosticsPanel.tsx :: setFetchError(SETTINGS_FAILURE.diagnosticsChoice);
      // Becomes: setFetchError(`${SETTINGS_FAILURE.diagnosticsChoice} ${(err as { coreDetail?: string }).coreDetail ?? ''}`);
      stubFailing(
        'POST',
        coreRefusal(
          "Your choice could not be saved to /Users/u/.uclone/diagnostics/consent.json: [Errno 13] Permission denied: '/Users/u/.uclone/diagnostics/consent.json'",
        ),
      );
      render(<DiagnosticsPanel />);
      fireEvent.click(await screen.findByRole('button', { name: /record failures locally/i }));

      const failure = await screen.findByTestId('diagnostics-failure');
      expect(failure).toHaveTextContent(SETTINGS_FAILURE.diagnosticsChoice);
      expectPlain(failure.textContent);
    });

    it("keeps the Core's reason off the screen when deleting the recorded failures fails", async () => {
      // Killed by: frontend/src/components/DiagnosticsPanel.tsx :: setFetchError(SETTINGS_FAILURE.diagnosticsClear);
      // Becomes: setFetchError(`${SETTINGS_FAILURE.diagnosticsClear} ${(err as { coreDetail?: string }).coreDetail ?? ''}`);
      stubFailing(
        'DELETE',
        coreRefusal(
          "/Users/u/.uclone/diagnostics/failures.jsonl could not be deleted: PermissionError: [Errno 13] Permission denied: '/Users/u/.uclone/diagnostics/failures.jsonl'",
        ),
        'granted',
      );
      render(<DiagnosticsPanel />);
      await screen.findByText(/3 failures recorded/i);
      fireEvent.click(screen.getByRole('button', { name: /delete/i }));

      const failure = await screen.findByTestId('diagnostics-failure');
      expect(failure).toHaveTextContent(SETTINGS_FAILURE.diagnosticsClear);
      expectPlain(failure.textContent);
    });
  });
});
