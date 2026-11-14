import { afterEach, describe, expect, it, vi } from 'vitest';
import { fireEvent, render, screen, waitFor } from '@testing-library/react';
import { UsageLimitRemedy } from './UsageLimitRemedy';
import { makePersonaInfo } from '../../test/fixtures';

const AGAIN = '2026-10-30T15:40:00Z';

const usage = {
  checked_at: '2026-10-30T15:20:00Z',
  windows: [
    { window: 'per_10_minutes', used: 1200, cached: 0, limit: 1000, available_again_at: AGAIN },
    { window: 'per_5_hours', used: 1200, cached: 0, limit: null, available_again_at: null },
    { window: 'per_week', used: 1200, cached: 0, limit: null, available_again_at: null },
  ],
  limits: { per_10_minutes: 1000, per_5_hours: null, per_week: null },
  env_overrides: [],
  providers: [],
};

const group = (id: string, label: string, model: string) => ({
  connection_id: id,
  label,
  kind: id,
  status: 'connected',
  detail: null,
  models: [{ ref: `${id}/${model}`, id: model, display_name: null, capabilities: ['chat'], context_window: null }],
});

const connection = (id: string, paid: boolean) => ({
  id,
  kind: id,
  label: id,
  base_url: null,
  key_set: paid,
  key_masked: null,
  source: 'settings',
  paid,
  status: 'connected',
  detail: null,
  model_count: 1,
});

const serve = (local: boolean) => {
  const sent: { url: string; method: string; body: unknown }[] = [];
  const fetchMock = vi.fn(async (input: RequestInfo | URL, init?: RequestInit) => {
    const url = String(input);
    const method = init?.method ?? 'GET';
    sent.push({ url, method, body: init?.body ? JSON.parse(String(init.body)) : null });
    const json = (body: unknown) => new Response(JSON.stringify(body), { status: 200 });
    if (url.startsWith('/api/usage')) return json(usage);
    if (url.startsWith('/api/connections'))
      return json({
        connections: [connection('gemini', true), ...(local ? [connection('ollama', false)] : [])],
        kinds: [],
      });
    if (url.startsWith('/api/models'))
      return json({
        groups: [
          group('gemini', 'Google', 'gemini-flash-latest'),
          ...(local ? [group('ollama', 'Ollama', 'qwen3:8b')] : []),
        ],
        defaults: { deep: null, fast: null, image: 'auto' },
        recommended: { deep: null, fast: null },
      });
    if (url.startsWith('/api/clones/composer'))
      return json({ persona: makePersonaInfo({ name: 'composer', model_name: 'gemini/gemini-flash-latest' }) });
    return new Response('{}', { status: 404 });
  });
  vi.stubGlobal('fetch', fetchMock);
  return sent;
};

afterEach(() => vi.unstubAllGlobals());

describe('UsageLimitRemedy: a turn refused at its start by the paid-model limit (#2431)', () => {
  // Killed by: frontend/src/components/rooms/UsageLimitRemedy.tsx :: {until !== null ? fmt(t.againAt, { time: clockTime(until) }) : t.againUnknown}{' '}
  // Becomes: {t.againUnknown}{' '}
  it('says when paid models can be used again, and never asks for a resend or a new conversation', async () => {
    serve(true);
    render(<UsageLimitRemedy cloneId="composer" label="Composer" testId="r" />);

    const when = await screen.findByText(/Paid models can be used again at/);
    expect(when.textContent).toContain('switch Composer to a local model here');
    const all = screen.getByTestId('r').textContent ?? '';
    expect(all).not.toMatch(/send (your|the) message again|new conversation|new one/i);
  });

  // Killed by: frontend/src/components/rooms/UsageLimitRemedy.tsx :: .filter((g) => unpaid.has(g.connection_id) && g.status === 'connected')
  // Becomes: .filter((g) => g.status === 'connected')
  it('offers only local models, and switches this clone to one in place', async () => {
    const sent = serve(true);
    render(<UsageLimitRemedy cloneId="composer" label="Composer" testId="r" />);

    const select = (await screen.findByTestId('r-model')) as HTMLSelectElement;
    expect([...select.options].map((o) => o.value)).toEqual(['ollama/qwen3:8b']);
    fireEvent.click(screen.getByTestId('r-switch'));

    await screen.findByTestId('r-done');
    const put = sent.find((s) => s.method === 'PUT');
    expect(put?.url).toBe('/api/clones/composer');
    expect((put?.body as { model_name: string }).model_name).toBe('ollama/qwen3:8b');
    expect(screen.getByTestId('r-done')).toHaveTextContent('Composer now answers on Ollama · qwen3:8b.');
  });

  it('says plainly when no local model is connected yet', async () => {
    serve(false);
    render(<UsageLimitRemedy cloneId="composer" label="Composer" testId="r" />);

    await waitFor(() => expect(screen.getByTestId('r-no-local')).toBeInTheDocument());
    expect(screen.queryByTestId('r-switch')).toBeNull();
  });
});
