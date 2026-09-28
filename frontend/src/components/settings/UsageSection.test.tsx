import { describe, it, expect, vi, beforeEach, afterEach } from 'vitest';
import { act, fireEvent, render, screen, waitFor } from '@testing-library/react';
import { UsageSection } from './UsageSection';
import { UsageOffer, USAGE_OFFER_DISMISSED_KEY } from './UsageOffer';
import { UsageBanner } from '../rooms/UsageBanner';
import { USAGE_PRESETS, type UsageLimitsValue, type UsageReport, type UsageWindowStatus } from '../../lib/usage';
import { expectPlain } from '../../test/plainCopy';

type Answer = { ok: boolean; status: number; json: () => Promise<unknown> };
const answer = (body: unknown, ok = true, status = 200): Answer => ({ ok, status, json: async () => body });

const win = (
  window: UsageWindowStatus['window'],
  used: number,
  limit: number | null,
  available_again_at: string | null = null,
): UsageWindowStatus => ({ window, used, limit, available_again_at });

const report = (over: Partial<UsageReport> = {}, limits: UsageLimitsValue = USAGE_PRESETS.none): UsageReport => ({
  checked_at: '2026-09-26T10:00:00Z',
  windows: [
    win('per_10_minutes', 12_000, limits.per_10_minutes),
    win('per_5_hours', 84_000, limits.per_5_hours),
    win('per_week', 400_000, limits.per_week),
  ],
  limits,
  env_overrides: [],
  providers: [],
  ...over,
});

/** Answers GET /api/usage from `reads` in turn (the last repeats) and PUT from `put`. */
const mockFetch = (reads: unknown[], put?: (body: unknown) => Answer) => {
  const calls: Array<{ url: string; method: string; body?: unknown }> = [];
  let i = 0;
  vi.stubGlobal(
    'fetch',
    vi.fn(async (url: string, init?: RequestInit) => {
      const method = init?.method ?? 'GET';
      const body = init?.body ? JSON.parse(init.body as string) : undefined;
      calls.push({ url, method, body });
      if (method === 'GET' && url === '/api/usage') {
        const next = reads[Math.min(i, reads.length - 1)];
        i += 1;
        return next instanceof Error ? Promise.reject(next) : answer(next);
      }
      if (method === 'PUT' && url === '/api/usage/limits' && put) return put(body);
      return answer({ detail: `unrouted ${method} ${url}` }, false, 599);
    }),
  );
  return calls;
};

const usageReads = (calls: ReturnType<typeof mockFetch>) =>
  calls.filter((c) => c.method === 'GET' && c.url === '/api/usage');

beforeEach(() => {
  window.localStorage.clear();
});

afterEach(() => {
  vi.unstubAllGlobals();
});

describe('UsageSection', () => {
  it('shows each window, with a bar only where a limit is set', async () => {
    mockFetch([report({}, { per_10_minutes: null, per_5_hours: 100_000, per_week: null })]);
    render(<UsageSection />);

    const fiveHours = await screen.findByTestId('settings-usage-window-per_5_hours');
    expect(fiveHours).toHaveTextContent('84K of 100K (84%)');
    expect(fiveHours.querySelector('[role="progressbar"]')).toHaveAttribute('aria-valuenow', '84');
    const tenMinutes = screen.getByTestId('settings-usage-window-per_10_minutes');
    expect(tenMinutes).toHaveTextContent('12K used, no limit');
    expect(tenMinutes.querySelector('[role="progressbar"]')).toBeNull();
    // Limits matching no preset open on Custom, with the saved figures in the boxes.
    expect(screen.getByTestId('settings-usage-preset-custom')).toHaveAttribute('aria-checked', 'true');
    expect(screen.getByTestId('settings-usage-input-per_5_hours')).toHaveValue('100000');
  });

  it('says when paid models are paused, for a reached window', async () => {
    mockFetch([
      report({
        windows: [
          win('per_10_minutes', 0, null),
          win('per_5_hours', 510_000, 500_000, '2026-09-26T13:40:00Z'),
          win('per_week', 510_000, null),
        ],
      }),
    ]);
    render(<UsageSection />);

    expect(await screen.findByTestId('settings-usage-paused-per_5_hours')).toHaveTextContent(
      /Paid models are paused until/,
    );
  });

  it('names the variable that overrides a window', async () => {
    mockFetch([report({ env_overrides: ['per_week'] })]);
    render(<UsageSection />);

    expect(await screen.findByTestId('settings-usage-env-per_week')).toHaveTextContent(
      'Set by UCLONE_USAGE_LIMIT_WEEK',
    );
    expect(screen.queryByTestId('settings-usage-env-per_5_hours')).toBeNull();
  });

  it('saves a preset as its three figures, and shows the saved answer', async () => {
    const saved = report({}, USAGE_PRESETS.standard);
    const calls = mockFetch([report()], () => answer(saved));
    render(<UsageSection />);

    fireEvent.click(await screen.findByTestId('settings-usage-preset-standard'));
    expect(screen.queryByTestId('settings-usage-custom')).toBeNull();
    fireEvent.click(screen.getByTestId('settings-usage-save'));

    await screen.findByTestId('settings-usage-saved');
    expect(calls.filter((c) => c.method === 'PUT')).toEqual([
      { url: '/api/usage/limits', method: 'PUT', body: USAGE_PRESETS.standard },
    ]);
    expect(screen.getByTestId('settings-usage-window-per_5_hours')).toHaveTextContent('of 2M');
  });

  it('sends an empty box as no limit, a whole number as a number, and anything else as typed', async () => {
    const calls = mockFetch([report()], () => answer({ detail: 'The weekly limit must be a whole number.' }, false, 400));
    render(<UsageSection />);

    fireEvent.click(await screen.findByTestId('settings-usage-preset-custom'));
    fireEvent.change(screen.getByTestId('settings-usage-input-per_10_minutes'), { target: { value: '5000' } });
    fireEvent.change(screen.getByTestId('settings-usage-input-per_5_hours'), { target: { value: '500,000' } });
    fireEvent.click(screen.getByTestId('settings-usage-save'));

    const failed = await screen.findByTestId('settings-usage-save-failed');
    // "500,000" reaches the Core as typed: saved as "no limit", it would lift the limit.
    expect(calls.find((c) => c.method === 'PUT')?.body).toEqual({
      per_10_minutes: 5000,
      per_5_hours: '500,000',
      per_week: null, // left empty: no limit
    });
    // The Core's refusal is shown in its own words, after what did not happen.
    expect(failed).toHaveTextContent('The limits were not saved. The weekly limit must be a whole number.');
    expect(screen.queryByTestId('settings-usage-saved')).toBeNull();
  });

  it('treats an answer of another shape as a failed read, not as zero usage', async () => {
    mockFetch([{ servers: [] }]);
    render(<UsageSection />);

    const error = await screen.findByTestId('settings-usage-error');
    expectPlain(error.textContent);
    expect(screen.queryByTestId('settings-usage-windows')).toBeNull();
  });

  it('lists the providers used, and links only their consoles', async () => {
    mockFetch([report({ providers: [{ provider: 'anthropic', tokens: 1_200_000, models: [] }] })]);
    render(<UsageSection />);

    expect(await screen.findByTestId('settings-usage-providers')).toHaveTextContent('Anthropic 1.2M');
    const links = screen.getByTestId('settings-usage-money-cap').querySelectorAll('a');
    expect([...links].map((a) => a.textContent)).toEqual(['Anthropic']);
  });

  it('links every console when no provider has been used', async () => {
    mockFetch([report()]);
    render(<UsageSection />);

    await screen.findByText('No paid model has been used in the last 7 days.');
    const links = screen.getByTestId('settings-usage-money-cap').querySelectorAll('a');
    expect([...links].map((a) => a.textContent)).toEqual(['Anthropic', 'OpenAI', 'Google']);
  });
});

describe('UsageOffer', () => {
  it('offers a limit while none is set, and opens Usage when taken', async () => {
    mockFetch([report()]);
    const onOpen = vi.fn();
    render(<UsageOffer onOpen={onOpen} />);

    fireEvent.click(await screen.findByTestId('settings-usage-offer-set'));
    expect(onOpen).toHaveBeenCalledTimes(1);
  });

  it('is not shown once any limit is set', async () => {
    const calls = mockFetch([report({}, { per_10_minutes: null, per_5_hours: null, per_week: 1 })]);
    render(<UsageOffer onOpen={() => {}} />);

    await waitFor(() => expect(usageReads(calls)).toHaveLength(1));
    await act(async () => {});
    expect(screen.queryByTestId('settings-usage-offer')).toBeNull();
  });

  it('stays hidden after "Not now", across a remount', async () => {
    mockFetch([report()]);
    const { unmount } = render(<UsageOffer onOpen={() => {}} />);

    fireEvent.click(await screen.findByTestId('settings-usage-offer-dismiss'));
    expect(screen.queryByTestId('settings-usage-offer')).toBeNull();
    expect(window.localStorage.getItem(USAGE_OFFER_DISMISSED_KEY)).toBe('1');

    unmount();
    render(<UsageOffer onOpen={() => {}} />);
    await act(async () => {});
    expect(screen.queryByTestId('settings-usage-offer')).toBeNull();
  });
});

describe('UsageBanner', () => {
  const limited = { per_10_minutes: null, per_5_hours: 100_000, per_week: null };
  const near = report({ windows: [win('per_5_hours', 85_000, 100_000)] }, limited);
  const reached = report(
    { windows: [win('per_5_hours', 120_000, 100_000, '2026-09-26T13:40:00Z'), win('per_week', 900, 1000)] },
    limited,
  );
  const quiet = report({ windows: [win('per_5_hours', 10_000, 100_000)] }, limited);

  it('shows nothing below 80%', async () => {
    const calls = mockFetch([quiet]);
    render(<UsageBanner refreshKey={0} />);

    await waitFor(() => expect(usageReads(calls)).toHaveLength(1));
    await act(async () => {});
    expect(screen.queryByTestId('room-usage-banner')).toBeNull();
  });

  it('warns at 80%, naming the window', async () => {
    mockFetch([near]);
    render(<UsageBanner refreshKey={0} />);

    const banner = await screen.findByTestId('room-usage-banner');
    expect(banner).toHaveAttribute('data-state', 'near');
    expect(banner).toHaveTextContent("You've used 80% of your 5-hour limit for paid models.");
  });

  it('says paid models are paused when a window is reached, ahead of a near one', async () => {
    mockFetch([reached]);
    render(<UsageBanner refreshKey={0} />);

    const banner = await screen.findByTestId('room-usage-banner');
    expect(banner).toHaveAttribute('data-state', 'reached');
    expect(banner).toHaveTextContent(/Paid models are paused until .+ because your 5-hour limit was reached\./);
    expectPlain(banner.textContent);
  });

  it('re-reads when a turn lands, and shows the new state', async () => {
    const calls = mockFetch([quiet, near]);
    const { rerender } = render(<UsageBanner refreshKey={0} />);
    await waitFor(() => expect(usageReads(calls)).toHaveLength(1));

    rerender(<UsageBanner refreshKey={1} />);

    expect(await screen.findByTestId('room-usage-banner')).toHaveAttribute('data-state', 'near');
    // One read per turn: a re-render with the same key reads nothing more.
    rerender(<UsageBanner refreshKey={1} />);
    await act(async () => {});
    expect(usageReads(calls)).toHaveLength(2);
  });

  it('stays closed for the same state, and returns when the state changes', async () => {
    const calls = mockFetch([near, near, reached]);
    const { rerender } = render(<UsageBanner refreshKey={0} />);

    fireEvent.click(await screen.findByTestId('room-usage-banner-close'));
    expect(screen.queryByTestId('room-usage-banner')).toBeNull();

    rerender(<UsageBanner refreshKey={1} />);
    await waitFor(() => expect(usageReads(calls)).toHaveLength(2));
    await act(async () => {});
    expect(screen.queryByTestId('room-usage-banner')).toBeNull();

    rerender(<UsageBanner refreshKey={2} />);
    expect(await screen.findByTestId('room-usage-banner')).toHaveAttribute('data-state', 'reached');
  });

  it('opens Usage settings from the banner', async () => {
    mockFetch([near]);
    const onOpenSettings = vi.fn();
    render(<UsageBanner refreshKey={0} onOpenSettings={onOpenSettings} />);

    fireEvent.click(await screen.findByTestId('room-usage-banner-open'));
    expect(onOpenSettings).toHaveBeenCalledTimes(1);
  });
});
