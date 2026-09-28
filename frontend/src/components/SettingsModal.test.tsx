import { describe, it, expect, vi, beforeEach, afterEach } from 'vitest';
import { fireEvent, render, screen, waitFor, within } from '@testing-library/react';
import { SettingsModal } from './SettingsModal';
import type { CatalogResult, RuntimeSettings } from '../types';
import { expectPlain } from '../test/plainCopy';
import { en } from '../i18n/en';
import { fmt, plural } from '../i18n/format';
import { ko } from '../i18n/ko';
import { LocaleProvider } from '../i18n';

const SETTINGS_FAILURE = en.settings.failure;
const PERSONA_EDITOR_COPY = en.personaEditor;

/**
 * Covers the Ollama install/delete controls added to the model management
 * section: loading state, success refresh, error surfacing, and the
 * `window.confirm` cancel path for delete (#1163-adjacent Ollama lifecycle work).
 */

const baseSettings: RuntimeSettings = {
  llm_provider: 'ollama',
  llm_base_url: 'http://localhost:11434',
  llm_model: 'qwen3:8b',
  llm_api_key_set: false,
  llm_api_key_masked: '',
  comfyui_base_url: 'http://127.0.0.1:8188',
  providers_available: ['ollama'],
  available_models: ['qwen3:8b', 'llama3.2:1b'],
};

const consentPayload = {
  state: 'unasked',
  error: null,
  journal: '/home/u/.uclone/diagnostics/failures.jsonl',
};

const reportPayload = {
  state: 'unasked',
  available: true,
  error: null,
  unreadable_lines: 0,
  recording_blocked: null,
  count: 0,
  distinct: 0,
  title: null,
  body: null,
  issue_url: null,
  search_url: null,
};

const personasPayload = { personas: [], available_tools: [], personas_dir: null };

/** Developer mode is a required prop; the cases that are not about it render it off, its default. */
const devModeOff = { developerMode: false, onDeveloperModeChange: () => {} };

type RouteResponse = { ok: boolean; status: number; statusText?: string; json: () => Promise<unknown> };
type Handler = (url: string, init?: RequestInit) => RouteResponse | Promise<RouteResponse>;

const jsonResponse = (body: unknown, ok = true, status = 200): RouteResponse => ({
  ok,
  status,
  json: async () => body,
});

/** Routes the common GETs every render needs, plus whatever overrides the test cares about. */
const mockFetch = (overrides: Record<string, Handler> = {}) => {
  const calls: Array<{ url: string; method: string; body?: unknown }> = [];
  const fetchMock = vi.fn(async (url: string, init?: RequestInit) => {
    const method = init?.method ?? 'GET';
    const body = init?.body ? JSON.parse(init.body as string) : undefined;
    calls.push({ url, method, body });

    for (const [pattern, handler] of Object.entries(overrides)) {
      if (url.includes(pattern)) return handler(url, init);
    }
    if (url.includes('/api/settings/remote-gpu/status')) return jsonResponse({ host: '', connected: false });
    if (url.includes('/api/settings')) return jsonResponse(baseSettings);
    if (url.includes('/api/personas')) return jsonResponse(personasPayload);
    if (url.includes('/api/diagnostics/consent')) return jsonResponse(consentPayload);
    if (url.includes('/api/diagnostics/report')) return jsonResponse(reportPayload);
    return jsonResponse({});
  });
  vi.stubGlobal('fetch', fetchMock);
  return calls;
};

describe('SettingsModal Ollama model management', () => {
  beforeEach(() => vi.restoreAllMocks());
  afterEach(() => vi.unstubAllGlobals());

  it('installs a model and refreshes the list on success', async () => {
    const calls = mockFetch({
      '/api/models/pull': () => jsonResponse({ status: 'ok', model: 'llama3.2:3b' }),
    });

    render(<SettingsModal {...devModeOff} isOpen onClose={() => {}} />);

    const input = await screen.findByLabelText(/model to install/i);
    fireEvent.change(input, { target: { value: 'llama3.2:3b' } });
    fireEvent.click(screen.getByRole('button', { name: /install/i }));

    await waitFor(() => {
      expect(calls.some((c) => c.method === 'POST' && c.url.includes('/api/models/pull'))).toBe(
        true,
      );
    });
    const pullCall = calls.find((c) => c.url.includes('/api/models/pull'));
    expect(pullCall?.body).toEqual({ model: 'llama3.2:3b' });

    expect(await screen.findByText(/installed model/i)).toBeTruthy();
    // Success clears the input and refetches /api/settings to pick up the new model.
    expect((input as HTMLInputElement).value).toBe('');
    expect(calls.filter((c) => c.method === 'GET' && c.url === '/api/settings').length).toBe(
      2,
    );
  });

  it('shows a spinner and disables the input while installing', async () => {
    let resolvePull: (() => void) | undefined;
    mockFetch({
      '/api/models/pull': () =>
        new Promise((resolve) => {
          resolvePull = () => resolve(jsonResponse({ status: 'ok', model: 'llama3.2:3b' }));
        }),
    });

    render(<SettingsModal {...devModeOff} isOpen onClose={() => {}} />);

    const input = await screen.findByLabelText(/model to install/i);
    fireEvent.change(input, { target: { value: 'llama3.2:3b' } });
    fireEvent.click(screen.getByRole('button', { name: /install/i }));

    await waitFor(() => expect((input as HTMLInputElement).disabled).toBe(true));

    resolvePull?.();
    await waitFor(() => expect((input as HTMLInputElement).disabled).toBe(false));
  });

  it('shows the provider-reported error when install fails', async () => {
    mockFetch({
      '/api/models/pull': () =>
        jsonResponse({ detail: 'Ollama provider returned status 404: not found' }, false, 502),
    });

    render(<SettingsModal {...devModeOff} isOpen onClose={() => {}} />);

    const input = await screen.findByLabelText(/model to install/i);
    fireEvent.change(input, { target: { value: 'no-such-model' } });
    fireEvent.click(screen.getByRole('button', { name: /install/i }));

    expect(await screen.findByText(/status 404/i)).toBeTruthy();
  });

  it('asks before deleting, and sends nothing when cancelled', async () => {
    const calls = mockFetch();
    vi.stubGlobal('confirm', vi.fn(() => false));

    render(<SettingsModal {...devModeOff} isOpen onClose={() => {}} />);

    const deleteButton = await screen.findByRole('button', { name: /delete llama3\.2:1b/i });
    fireEvent.click(deleteButton);

    expect(calls.some((c) => c.url.includes('/api/models/delete'))).toBe(false);
  });

  it('deletes a model after confirmation and refreshes the list', async () => {
    const calls = mockFetch({
      '/api/models/delete': () => jsonResponse({ status: 'ok', model: 'llama3.2:1b' }),
    });
    vi.stubGlobal('confirm', vi.fn(() => true));

    render(<SettingsModal {...devModeOff} isOpen onClose={() => {}} />);

    const deleteButton = await screen.findByRole('button', { name: /delete llama3\.2:1b/i });
    fireEvent.click(deleteButton);

    await waitFor(() => {
      expect(calls.some((c) => c.method === 'POST' && c.url.includes('/api/models/delete'))).toBe(
        true,
      );
    });
    const deleteCall = calls.find((c) => c.url.includes('/api/models/delete'));
    expect(deleteCall?.body).toEqual({ model: 'llama3.2:1b' });
    expect(await screen.findByText(/deleted model/i)).toBeTruthy();
  });

  it('shows the provider-reported error when delete fails', async () => {
    mockFetch({
      '/api/models/delete': () =>
        jsonResponse({ detail: 'Ollama provider returned status 404: not found' }, false, 502),
    });
    vi.stubGlobal('confirm', vi.fn(() => true));

    render(<SettingsModal {...devModeOff} isOpen onClose={() => {}} />);

    const deleteButton = await screen.findByRole('button', { name: /delete llama3\.2:1b/i });
    fireEvent.click(deleteButton);

    expect(await screen.findByText(/status 404/i)).toBeTruthy();
  });

  it('does not show model management controls for a non-Ollama provider', async () => {
    mockFetch({
      '/api/settings': () => jsonResponse({ ...baseSettings, llm_provider: 'openai' }),
    });

    render(<SettingsModal {...devModeOff} isOpen onClose={() => {}} />);

    await screen.findByText(/Endpoint Base URL/i);
    expect(screen.queryByTestId('ollama-model-management')).toBeNull();
  });
});

/**
 * The Ollama picker lists what the server on the form answers (#1666). The saved list is empty
 * when the saved address stopped answering (a remote-GPU tunnel that went away), and `[]` is
 * truthy: it used to hide every other source, so the picker vanished with no word why.
 */
describe('SettingsModal local model list (#1666)', () => {
  beforeEach(() => vi.restoreAllMocks());
  afterEach(() => vi.unstubAllGlobals());

  const deadTunnel: RuntimeSettings = {
    ...baseSettings,
    llm_base_url: 'http://127.0.0.1:11435',
    available_models: [],
  };
  const listing = (models: string[], reachable: boolean) => () =>
    jsonResponse({ provider: 'ollama', models, reachable, catalog: null });
  const catalogCalls = (calls: ReturnType<typeof mockFetch>) =>
    calls.filter((c) => c.method === 'POST' && c.url.includes('/api/models/catalog'));

  // Killed by: frontend/src/components/SettingsModal.tsx :: const localNotAnswering = needsLocalListing && localMatches
  // Becomes: const localNotAnswering = false && needsLocalListing && localMatches
  it('says nothing answered at a saved address that is dead, instead of an empty field', async () => {
    const calls = mockFetch({
      '/api/settings': () => jsonResponse(deadTunnel),
      '/api/models/catalog': listing([], false),
    });
    render(<SettingsModal {...devModeOff} isOpen onClose={() => {}} />);

    const note = await screen.findByTestId('settings-local-not-answering');
    expect(note.textContent).toBe(
      fmt(en.settings.llm.ollamaNotAnswering, { endpoint: 'http://127.0.0.1:11435' }),
    );
    expect(catalogCalls(calls)[0]?.body).toEqual({ provider: 'ollama', base_url: 'http://127.0.0.1:11435' });
    expect(screen.queryByTestId('settings-model-select')).toBeNull();
  });

  // Killed by: frontend/src/components/SettingsModal.tsx :: : localMatches && localListing.models.length > 0
  // Becomes: : false && localMatches && localListing.models.length > 0
  it('lists the models at an address typed on the form, before it is saved', async () => {
    const calls = mockFetch({
      '/api/settings': () => jsonResponse(deadTunnel),
      '/api/models/catalog': (_url, init) => {
        const asked = JSON.parse(init?.body as string) as { base_url?: string };
        return asked.base_url === 'http://localhost:11434'
          ? listing(['qwen3:8b', 'gemma3:4b'], true)()
          : listing([], false)();
      },
    });
    render(<SettingsModal {...devModeOff} isOpen onClose={() => {}} />);
    await screen.findByTestId('settings-local-not-answering');

    const endpoint = screen.getByPlaceholderText('http://localhost:11434') as HTMLInputElement;
    fireEvent.change(endpoint, { target: { value: 'http://localhost:11434' } });

    const select = (await screen.findByTestId('settings-model-select', {}, { timeout: 2000 })) as HTMLSelectElement;
    const options = within(select).getAllByRole('option').map((o) => (o as HTMLOptionElement).value);
    expect(options).toEqual(['qwen3:8b', 'gemma3:4b', '__custom__']);
    expect(screen.queryByTestId('settings-local-not-answering')).toBeNull();
    expect(calls.some((c) => c.method === 'POST' && c.url === '/api/settings')).toBe(false);
  });

  // Killed by: frontend/src/components/SettingsModal.tsx :: : testResults?.llm?.models ?? [];
  // Becomes: : [];
  it('shows the models a connection check found when the saved list is empty', async () => {
    mockFetch({
      '/api/settings/test': () =>
        jsonResponse({ results: { llm: { status: 'ok', provider: 'ollama', models: ['hermes3:8b'] } } }),
      '/api/settings': () => jsonResponse(deadTunnel),
      '/api/models/catalog': listing([], false),
    });
    render(<SettingsModal {...devModeOff} isOpen onClose={() => {}} />);
    await screen.findByTestId('settings-local-not-answering');

    fireEvent.click(screen.getByRole('button', { name: /check connection/i }));

    const select = (await screen.findByTestId('settings-model-select')) as HTMLSelectElement;
    expect(within(select).getAllByRole('option').map((o) => (o as HTMLOptionElement).value)).toEqual([
      'hermes3:8b',
      '__custom__',
    ]);
  });

  // Killed by: frontend/src/components/SettingsModal.tsx :: llmProvider === 'ollama' && needsLocalListing && localMatches && localListing.reachable
  // Becomes: false && needsLocalListing && localMatches && localListing.reachable
  it('tells a running Ollama with nothing installed apart from a stopped one', async () => {
    mockFetch({
      '/api/settings': () => jsonResponse({ ...baseSettings, available_models: [] }),
      '/api/models/catalog': listing([], true),
    });
    render(<SettingsModal {...devModeOff} isOpen onClose={() => {}} />);

    const note = await screen.findByTestId('settings-ollama-no-models');
    expect(note.textContent).toBe(en.settings.llm.ollamaNoModels);
    expect(screen.queryByTestId('settings-local-not-answering')).toBeNull();
  });

  // Killed by: frontend/src/components/SettingsModal.tsx :: disabled={installing || !pullModelInput.trim() || !onSavedServer}
  // Becomes: disabled={installing || !pullModelInput.trim()}
  it('installs and removes only at the saved address, which is where those requests go', async () => {
    mockFetch({ '/api/models/catalog': listing(['other:1b'], true) });
    render(<SettingsModal {...devModeOff} isOpen onClose={() => {}} />);
    await screen.findByRole('button', { name: /delete llama3\.2:1b/i });

    const endpoint = screen.getByPlaceholderText('http://localhost:11434') as HTMLInputElement;
    fireEvent.change(endpoint, { target: { value: 'http://gpu-box:11434' } });
    fireEvent.change(screen.getByLabelText(/model to install/i), { target: { value: 'llama3.2:3b' } });

    expect(screen.getByTestId('settings-models-save-first').textContent).toBe(en.settings.models.saveToManage);
    expect(screen.queryByRole('button', { name: /delete llama3\.2:1b/i })).toBeNull();
    expect((screen.getByRole('button', { name: /^install$/i }) as HTMLButtonElement).disabled).toBe(true);
  });

  // Killed by: frontend/src/components/SettingsModal.tsx :: savedModels.length === 0 &&
  // Becomes: true &&
  it('does not ask again when the saved address already listed its models', async () => {
    const calls = mockFetch();
    render(<SettingsModal {...devModeOff} isOpen onClose={() => {}} />);
    await screen.findByTestId('settings-model-select');
    await new Promise((r) => setTimeout(r, 50));
    expect(catalogCalls(calls)).toEqual([]);
  });

  // Killed by: frontend/src/components/SettingsModal.tsx :: sameEndpoint(currentSettings.llm_base_url ?? '', formEndpoint);
  // Becomes: (currentSettings.llm_base_url ?? '').trim() === formEndpoint;
  it('treats localhost and 127.0.0.1 on the saved port as the saved server', async () => {
    const calls = mockFetch({
      '/api/settings': () => jsonResponse({ ...baseSettings, llm_base_url: 'http://127.0.0.1:11434' }),
      '/api/models/catalog': listing([], false),
    });
    render(<SettingsModal {...devModeOff} isOpen onClose={() => {}} />);
    await screen.findByRole('button', { name: /delete llama3\.2:1b/i });

    const endpoint = screen.getByPlaceholderText('http://localhost:11434') as HTMLInputElement;
    fireEvent.change(endpoint, { target: { value: 'http://localhost:11434/' } });
    fireEvent.change(screen.getByLabelText(/model to install/i), { target: { value: 'llama3.2:3b' } });
    await new Promise((r) => setTimeout(r, 600));

    expect(screen.queryByTestId('settings-models-save-first')).toBeNull();
    expect((screen.getByRole('button', { name: /^install$/i }) as HTMLButtonElement).disabled).toBe(false);
    expect(screen.getByRole('button', { name: /delete llama3\.2:1b/i })).toBeTruthy();
    const select = screen.getByTestId('settings-model-select') as HTMLSelectElement;
    expect(within(select).getAllByRole('option').map((o) => (o as HTMLOptionElement).value)).toEqual([
      'qwen3:8b',
      'llama3.2:1b',
      '__custom__',
    ]);
    // The saved list answered; the typed spelling is not asked about as a new server.
    expect(catalogCalls(calls)).toEqual([]);
    expect(screen.getByTestId('settings-llm-connected').textContent).toBe(
      plural(en.settings.llm.localConnected, 2),
    );
  });

  // Killed by: frontend/src/components/SettingsModal.tsx :: {localConnectedCount !== null &&
  // Becomes: {false && localConnectedCount !== null &&
  it('says the local server is connected, and how many models it has, before saving', async () => {
    mockFetch({
      '/api/settings': () => jsonResponse(deadTunnel),
      '/api/models/catalog': (_url, init) => {
        const asked = JSON.parse(init?.body as string) as { base_url?: string };
        return asked.base_url === 'http://localhost:11434'
          ? listing(['qwen3:8b', 'gemma3:4b', 'phi4:14b'], true)()
          : listing([], false)();
      },
    });
    render(<SettingsModal {...devModeOff} isOpen onClose={() => {}} />);
    await screen.findByTestId('settings-local-not-answering');
    expect(screen.queryByTestId('settings-llm-connected')).toBeNull();

    const endpoint = screen.getByPlaceholderText('http://localhost:11434') as HTMLInputElement;
    fireEvent.change(endpoint, { target: { value: 'http://localhost:11434' } });

    const line = await screen.findByTestId('settings-llm-connected', {}, { timeout: 2000 });
    expect(line.textContent).toBe(plural(en.settings.llm.localConnected, 3));
    expect(screen.queryByTestId('settings-local-not-answering')).toBeNull();
  });

  // Killed by: frontend/src/components/SettingsModal.tsx :: : needsLocalListing && localMatches && localListing.reachable
  // Becomes: : needsLocalListing && localMatches
  it('does not say connected when the local server did not answer', async () => {
    mockFetch({
      '/api/settings': () => jsonResponse(deadTunnel),
      '/api/models/catalog': listing([], false),
    });
    render(<SettingsModal {...devModeOff} isOpen onClose={() => {}} />);
    await screen.findByTestId('settings-local-not-answering');
    expect(screen.queryByTestId('settings-llm-connected')).toBeNull();
  });

  it('says it in Korean too', async () => {
    mockFetch({
      '/api/settings': () => jsonResponse(deadTunnel),
      '/api/models/catalog': listing([], false),
    });
    render(
      <LocaleProvider hints={["ko-KR"]}>
        <SettingsModal {...devModeOff} isOpen onClose={() => {}} />
      </LocaleProvider>,
    );
    const note = await screen.findByTestId('settings-local-not-answering');
    expect(note.textContent).toBe(fmt(ko.settings.llm.ollamaNotAnswering, { endpoint: 'http://127.0.0.1:11435' }));
  });
});

/**
 * Escape closing the modal used to be the modal's own `window.addEventListener` (#1036).
 * It now goes through the shared owner in `lib/escapePrecedence.ts`, so what is worth
 * pinning here is that this surface still claims the `'dialog'` layer while open, and
 * gives it up once closed.
 */
describe('SettingsModal Escape (#1036)', () => {
  beforeEach(() => {
    vi.stubGlobal(
      'fetch',
      vi.fn(async (url: string) => {
        if (url.includes('/api/personas')) {
          return { ok: true, json: async () => ({ personas: [], personas_dir: '' }) } as Response;
        }
        if (url.includes('/api/diagnostics/consent')) {
          return {
            ok: true,
            json: async () => ({ state: 'undecided', error: null, journal: '' }),
          } as Response;
        }
        return { ok: true, json: async () => ({}) } as Response;
      }),
    );
  });
  afterEach(() => vi.unstubAllGlobals());

  it('closes on Escape while open', () => {
    const onClose = vi.fn();
    render(<SettingsModal {...devModeOff} isOpen onClose={onClose} />);

    fireEvent.keyDown(window, { key: 'Escape' });

    expect(onClose).toHaveBeenCalledTimes(1);
  });

  it('does not respond to Escape once closed', () => {
    const onClose = vi.fn();
    render(<SettingsModal {...devModeOff} isOpen={false} onClose={onClose} />);

    // Killed by: frontend/src/components/SettingsModal.tsx :: useEscapeOwner('dialog', isOpen, onClose);
    // Becomes: useEscapeOwner('dialog', true, onClose);
    fireEvent.keyDown(window, { key: 'Escape' });

    expect(onClose).not.toHaveBeenCalled();
  });

  it('does not render dialog content while closed', () => {
    render(<SettingsModal {...devModeOff} isOpen={false} onClose={vi.fn()} />);

    expect(screen.queryByText('Settings')).toBeNull();
  });
});

/**
 * Cancelling a model install (#1233).
 *
 * An Ollama pull is minutes to tens of minutes of multi-gigabyte transfer, and
 * before this the browser had no way to stop waiting for one: the `fetch` carried
 * no signal, so closing Settings or navigating away left it running with nobody to
 * receive it. These tests drive the abort themselves rather than waiting on a
 * clock — the fetch mock here holds its promise open until the test's own
 * `signal.addEventListener('abort', ...)` fires, so a component that never wires a
 * signal leaves the promise unresolved and the assertion fails within the
 * `waitFor` budget instead of passing because the request happened to be quick.
 */
describe('SettingsModal install cancellation (#1233)', () => {
  /** A fetch whose `/api/models/pull` never settles until the caller's signal aborts. */
  const mockHeldPull = () => {
    const seen: { signal?: AbortSignal | null } = {};
    const fetchMock = vi.fn(async (url: string, init?: RequestInit) => {
      if (url.includes('/api/models/pull')) {
        seen.signal = init?.signal;
        return new Promise<RouteResponse>((_resolve, reject) => {
          init?.signal?.addEventListener('abort', () => {
            // What a real `fetch` rejects with. Detected by `name`, since jsdom's
            // DOMException is not the one a bundled `instanceof` would match.
            const err = new Error('The operation was aborted.');
            err.name = 'AbortError';
            reject(err);
          });
        });
      }
      if (url.includes('/api/settings/remote-gpu/status')) return jsonResponse({ host: '', connected: false });
      if (url.includes('/api/settings')) return jsonResponse(baseSettings);
      if (url.includes('/api/personas')) return jsonResponse(personasPayload);
      if (url.includes('/api/diagnostics/consent')) return jsonResponse(consentPayload);
      if (url.includes('/api/diagnostics/report')) return jsonResponse(reportPayload);
      return jsonResponse({});
    });
    vi.stubGlobal('fetch', fetchMock);
    return seen;
  };

  const startAnInstall = async () => {
    const input = await screen.findByLabelText(/model to install/i);
    fireEvent.change(input, { target: { value: 'qwen3:8b' } });
    fireEvent.click(screen.getByRole('button', { name: /^install$/i }));
    return input as HTMLInputElement;
  };

  beforeEach(() => vi.restoreAllMocks());
  afterEach(() => vi.unstubAllGlobals());

  it('gives the install request a signal that the Cancel button aborts', async () => {
    // Killed by: frontend/src/components/SettingsModal.tsx :: pullAbortRef.current = controller;
    // Becomes: pullAbortRef.current = null;
    const seen = mockHeldPull();

    render(<SettingsModal {...devModeOff} isOpen onClose={() => {}} />);
    await startAnInstall();

    await waitFor(() => expect(seen.signal).toBeTruthy());
    expect(seen.signal?.aborted).toBe(false);

    fireEvent.click(await screen.findByRole('button', { name: /cancel install/i }));

    await waitFor(() => expect(seen.signal?.aborted).toBe(true));
  });

  it('reports a cancelled install as stopped waiting, not as a failure', async () => {
    // Killed by: frontend/src/components/SettingsModal.tsx :: name === 'AbortError';
    // Becomes: name === 'NotTheNameFetchUses';
    mockHeldPull();

    render(<SettingsModal {...devModeOff} isOpen onClose={() => {}} />);
    const input = await startAnInstall();

    fireEvent.click(await screen.findByRole('button', { name: /cancel install/i }));

    // The user chose this. Ollama keeps its partial blobs and may well finish, so
    // "Failed to install" would report a defeat that did not happen.
    expect(await screen.findByText(/stopped waiting/i)).toBeTruthy();
    expect(screen.queryByText(/failed to install/i)).toBeNull();
    // Back to a state the user can act from, rather than stuck spinning.
    await waitFor(() => expect(input.disabled).toBe(false));
  });

  it('aborts the in-flight install when the modal is closed', async () => {
    // `isOpen={false}` returns null without unmounting, so closing Settings is not
    // an unmount and an unmount-only cleanup would miss it entirely — which is the
    // most likely way for this to be left running.
    //
    // Killed by: frontend/src/components/SettingsModal.tsx :: if (!isOpen) abortModelRequests();
    // Becomes: if (isOpen) return;
    const seen = mockHeldPull();

    const { rerender } = render(<SettingsModal {...devModeOff} isOpen onClose={() => {}} />);
    await startAnInstall();
    await waitFor(() => expect(seen.signal).toBeTruthy());

    rerender(<SettingsModal {...devModeOff} isOpen={false} onClose={() => {}} />);

    await waitFor(() => expect(seen.signal?.aborted).toBe(true));
  });

  it('says so when the server joined this install to one already running', async () => {
    // Killed by: frontend/src/components/SettingsModal.tsx :: text: fmt(data.joined ?
    // Becomes: text: fmt(false ?
    mockFetch({
      '/api/models/pull': () =>
        jsonResponse({ status: 'ok', model: 'qwen3:8b', joined: true }),
    });

    render(<SettingsModal {...devModeOff} isOpen onClose={() => {}} />);
    await startAnInstall();

    // Two clicks cost one download; a plain "Installed" would hide that and teach
    // the user that clicking again is how you make it go faster.
    expect(await screen.findByText(/already running/i)).toBeTruthy();
  });
});

/**
 * vLLM as a selectable provider (#1304).
 *
 * The one provider on this panel with no default endpoint and no default model: a vLLM
 * server is launched per model on a port chosen at the command line, so anything this
 * modal pre-filled would be a guess at somebody else's `vllm serve` arguments. What the
 * card must therefore do is the opposite of the Ollama card — clear, not fill.
 */
describe('SettingsModal vLLM provider (#1304)', () => {
  const openAiSettings: RuntimeSettings = {
    ...baseSettings,
    llm_provider: 'openai',
    llm_base_url: 'https://api.openai.com/v1',
    llm_model: 'gpt-4o',
    providers_available: ['ollama', 'vllm', 'openai', 'anthropic', 'gemini'],
    available_models: [],
  };

  beforeEach(() => vi.restoreAllMocks());
  afterEach(() => vi.unstubAllGlobals());

  // A cloud provider's endpoint sits behind Advanced, open here because the fixture sets one,
  // and its model is the free "Model name" field (#1631).
  const cloudFields = async () => ({
    endpoint: (await screen.findByPlaceholderText(en.settings.catalog.endpointPlaceholder)) as HTMLInputElement,
    model: screen.getByRole('textbox', { name: en.settings.catalog.modelName }) as HTMLInputElement,
  });
  // The vLLM fields are other elements than the cloud ones, so they are read after the switch.
  const vllmFields = () => ({
    endpoint: screen.getByPlaceholderText(/vllm serve/) as HTMLInputElement,
    model: screen.getByPlaceholderText(/qwen3:8b, hermes3:8b/i) as HTMLInputElement,
  });

  const selectVllm = async () => {
    mockFetch({ '/api/settings': () => jsonResponse(openAiSettings) });
    render(<SettingsModal {...devModeOff} isOpen onClose={() => {}} />);
    const { endpoint, model } = await cloudFields();
    expect(endpoint.value).toBe('https://api.openai.com/v1');
    expect(model.value).toBe('gpt-4o');

    fireEvent.click(screen.getByRole('button', { name: /vLLM/ }));
    return vllmFields();
  };

  // Killed by: frontend/src/components/SettingsModal.tsx :: setLlmBaseUrl('');
  // Becomes: setLlmBaseUrl(llmBaseUrl);
  it('clears an endpoint carried over from another provider instead of keeping it', async () => {
    const { endpoint } = await selectVllm();

    // Empty, and empty specifically: `https://api.openai.com/v1` left in this field is the
    // one wrong value worse than a blank one, because it is saved as configuration and
    // then sends the operator's prompts to OpenAI from a panel that says vLLM.
    expect(endpoint.value).toBe('');
  });

  // Killed by: frontend/src/components/SettingsModal.tsx :: setLlmModel('');
  // Becomes: setLlmModel(llmModel);
  it('clears a model carried over from another provider instead of keeping it', async () => {
    const { model } = await selectVllm();

    // A vLLM server answers any name but its own `--model` with a 404 about a model the
    // operator never chose, so `gpt-4o` surviving the switch is worse than a blank field.
    expect(model.value).toBe('');
  });

  // The denylist this replaced named the values the panel *pre-fills*, which is not the set of
  // values the panel can *hold*: `ui/app.py`'s model list offers `gpt-4o-mini`, `o1-mini` and
  // `o3-mini`, and an Ollama endpoint on any port but 11434 is still a legal endpoint. Each of
  // those survived the switch and was then saved as `VLLM_MODEL` / `VLLM_BASE_URL`, which came
  // back from the server as "The model gpt-4o-mini does not exist" — a sentence about a model
  // the operator did not choose on a panel that said vLLM.
  // Killed by: frontend/src/components/SettingsModal.tsx :: if (llmProvider !== 'vllm') {
  // Becomes: if (llmBaseUrl.includes('api.openai.com')) {
  it('clears values a denylist of the pre-filled ones would have kept', async () => {
    mockFetch({
      '/api/settings': () =>
        jsonResponse({
          ...openAiSettings,
          llm_base_url: 'http://10.0.0.7:11500/v1',
          llm_model: 'gpt-4o-mini',
        }),
    });
    render(<SettingsModal {...devModeOff} isOpen onClose={() => {}} />);
    const cloud = await cloudFields();
    expect(cloud.endpoint.value).toBe('http://10.0.0.7:11500/v1');
    expect(cloud.model.value).toBe('gpt-4o-mini');

    fireEvent.click(screen.getByRole('button', { name: /vLLM/ }));

    const { endpoint, model } = vllmFields();
    expect(endpoint.value).toBe('');
    expect(model.value).toBe('');
  });

  // The other half of "clear what another provider left": clearing is what a *change of
  // provider* means, so re-selecting the card already selected must not wipe the two fields
  // the operator just typed — they are the only source of them.
  // Killed by: frontend/src/components/SettingsModal.tsx :: if (llmProvider !== 'vllm') {
  // Becomes: if (true) {
  it('keeps what the operator typed when vLLM is re-selected', async () => {
    mockFetch({
      '/api/settings': () =>
        jsonResponse({
          ...openAiSettings,
          llm_provider: 'vllm',
          llm_base_url: 'http://gpu.internal:8000/v1',
          llm_model: 'qwen2.5-coder-32b-instruct',
        }),
    });
    render(<SettingsModal {...devModeOff} isOpen onClose={() => {}} />);
    // vLLM is already the provider, so the endpoint field carries vLLM's own placeholder.
    const endpoint = (await screen.findByPlaceholderText(/vllm serve/)) as HTMLInputElement;
    const model = screen.getByPlaceholderText(/qwen3:8b, hermes3:8b/i) as HTMLInputElement;

    fireEvent.click(screen.getByRole('button', { name: /vLLM/ }));

    expect(endpoint.value).toBe('http://gpu.internal:8000/v1');
    expect(model.value).toBe('qwen2.5-coder-32b-instruct');
  });

  // Killed by: frontend/src/components/SettingsModal.tsx :: {llmProvider === 'vllm' && (
  // Becomes: {false && (
  it('says the API key is optional, because `vllm serve --api-key` is', async () => {
    await selectVllm();

    expect(screen.getByTestId('vllm-key-optional').textContent).toMatch(/optional/i);
    expect(screen.getByTestId('vllm-key-optional').textContent).toMatch(/--api-key/);
  });

  // Killed by: frontend/src/components/SettingsModal.tsx :: { id: 'vllm', label: 'vLLM' },
  // Becomes: { id: 'vllm', label: 'Self-hosted' },
  it('offers a card labelled vLLM, the name the operator is looking for', async () => {
    mockFetch({ '/api/settings': () => jsonResponse(openAiSettings) });
    render(<SettingsModal {...devModeOff} isOpen onClose={() => {}} />);

    await screen.findByText(/Endpoint Base URL/i);
    expect(screen.getByRole('button', { name: /vLLM/ })).toBeTruthy();
  });

  // The vLLM branch was placed first in `handleProviderSelect`'s chain; this is the
  // neighbour it must not have changed on the way in.
  // Killed by: frontend/src/components/SettingsModal.tsx :: } else if (newProvider === 'ollama') {
  // Becomes: } else if (false) {
  it('still fills in localhost:11434 when Ollama is chosen', async () => {
    mockFetch({ '/api/settings': () => jsonResponse(openAiSettings) });
    render(<SettingsModal {...devModeOff} isOpen onClose={() => {}} />);
    await cloudFields();

    fireEvent.click(screen.getByRole('button', { name: /Ollama/ }));

    const endpoint = screen.getByPlaceholderText('http://localhost:11434') as HTMLInputElement;
    expect(endpoint.value).toBe('http://localhost:11434');
  });
});

describe('SettingsModal developer mode (owner ruling 2026-09-22)', () => {
  beforeEach(() => vi.restoreAllMocks());
  afterEach(() => vi.unstubAllGlobals());

  it('shows the switch off when developer mode is off, and asks to turn it on', async () => {
    // Killed by: frontend/src/components/SettingsModal.tsx :: onClick={() => onDeveloperModeChange(!developerMode)}
    // Becomes: onClick={() => onDeveloperModeChange(developerMode)}
    const calls = mockFetch();
    const onChange = vi.fn();
    render(<SettingsModal isOpen onClose={() => {}} developerMode={false} onDeveloperModeChange={onChange} />);

    const toggle = screen.getByRole('switch', { name: /developer mode/i });
    expect(toggle).toHaveAttribute('aria-checked', 'false');
    fireEvent.click(toggle);
    expect(onChange).toHaveBeenCalledWith(true);
    // A head preference: switching it writes nothing to the runtime.
    await waitFor(() => expect(calls.some((c) => c.url.includes('/api/settings'))).toBe(true));
    expect(calls.filter((c) => c.method !== 'GET')).toEqual([]);
  });

  it('shows the switch on when developer mode is on, and asks to turn it off', () => {
    // Killed by: frontend/src/components/SettingsModal.tsx :: aria-checked={developerMode}
    // Becomes: aria-checked={false}
    mockFetch();
    const onChange = vi.fn();
    render(<SettingsModal isOpen onClose={() => {}} developerMode onDeveloperModeChange={onChange} />);

    const toggle = screen.getByRole('switch', { name: /developer mode/i });
    expect(toggle).toHaveAttribute('aria-checked', 'true');
    fireEvent.click(toggle);
    expect(onChange).toHaveBeenCalledWith(false);
  });
});

/** One registered skill, the shape `GET /api/skills` answers with. */
const skillsPayload = {
  skills: [
    {
      name: 'csv_summariser',
      description: 'Summarises a CSV file into a short table.',
      version: '1.0.0',
      author: 'local',
      origin: 'human',
      status: 'active',
      isolation_level: 'sandbox',
      content_sha256: 'abc123',
      scripts: ['summarise.py'],
      tags: ['data'],
      approved_by: 'owner',
      approved_at: '2026-09-20T10:00:00Z',
      audit_report: {
        skill_name: 'csv_summariser',
        is_safe: true,
        recommendation: 'approve',
        risk_score: 0,
        detected_risks: [],
        auditor_version: '1',
        content_sha256: 'abc123',
      },
    },
  ],
  summary: { total_skills: 1, active_count: 1, pending_count: 0, quarantined_count: 0 },
};

const acpPayload = {
  transport: 'stdio',
  sdk_version_specified: '0.12.1',
  serving: false,
  presence: {
    shell_module_present: false,
    sdk_installed: false,
    transport: 'stdio',
    sdk_version_specified: '0.12.1',
    reason: 'No ACP shell is installed in this build.',
  },
  methods: [],
  mcp_descriptors: [],
  mcp_loader_warning: '',
  counts: {
    agent: { total: 0, implemented: 0, not_implemented: 0, not_implementable: 0, out_of_scope: 0 },
    client: { total: 0, implemented: 0, not_implemented: 0, not_implementable: 0, out_of_scope: 0 },
  },
};

const evalMetrics = {
  total_suites: 1,
  total_probes: 4,
  passed_probes: 4,
  failed_probes: 0,
  pass_rate: 1,
};
const evalsPayload = { status: 'ok', error: null, suites: [], scorecard: {}, metrics: evalMetrics };

describe('SettingsModal Skills section (#1358)', () => {
  beforeEach(() => vi.restoreAllMocks());
  afterEach(() => vi.unstubAllGlobals());

  it('lists a registered skill read from /api/skills', async () => {
    // Killed by: frontend/src/components/SettingsModal.tsx :: <SkillsSection />
    // Becomes:
    mockFetch({ '/api/skills': () => jsonResponse(skillsPayload) });
    render(<SettingsModal {...devModeOff} isOpen onClose={() => {}} />);

    const section = await screen.findByTestId('settings-skills');
    expect(section).toHaveTextContent('Skills');
    await waitFor(() => expect(section).toHaveTextContent('csv_summariser'));
  });

  it("states why the skills could not be read in the runtime's own words, not as a status line", async () => {
    // Killed by: frontend/src/lib/useApiRead.ts :: if (!res.ok) {
    // Becomes: if (false) {
    vi.spyOn(console, 'error').mockImplementation(() => {});
    mockFetch({
      '/api/skills': () => jsonResponse({ detail: 'The skill folder could not be opened.' }, false, 500),
    });
    render(<SettingsModal {...devModeOff} isOpen onClose={() => {}} />);

    const failure = await screen.findByTestId('settings-skills-error');
    expect(failure).toHaveTextContent('Your skills could not be loaded');
    expect(failure).toHaveTextContent('The skill folder could not be opened.');
    expect(failure).not.toHaveTextContent('HTTP');
    expect(failure).not.toHaveTextContent('500');
  });

  it("says in plain words that the runtime could not be reached, never the browser's transport message", async () => {
    // Killed by: frontend/src/i18n/locales/en/skills.json :: UClone-X could not be reached. If
    // Becomes: Failed to fetch. If
    vi.spyOn(console, 'error').mockImplementation(() => {});
    mockFetch({ '/api/skills': () => Promise.reject(new TypeError('Failed to fetch')) });
    render(<SettingsModal {...devModeOff} isOpen onClose={() => {}} />);

    const failure = await screen.findByTestId('settings-skills-error');
    expect(failure).toHaveTextContent('Your skills could not be loaded');
    expect(failure).toHaveTextContent('UClone-X could not be reached');
    expect(failure).not.toHaveTextContent('Failed to fetch');
    expect(failure).not.toHaveTextContent('TypeError');
  });

  it('says in plain words that the runtime gave no reason, never its status line, when a 500 has no JSON body', async () => {
    // Killed by: frontend/src/components/settings/SkillsSection.tsx :: fault.detail ?? copy.plainCause[fault.kind]
    // Becomes: fault.detail ?? fault.kind
    vi.spyOn(console, 'error').mockImplementation(() => {});
    mockFetch({
      '/api/skills': () => ({
        ok: false,
        status: 500,
        statusText: 'Internal Server Error',
        json: async () => {
          throw new SyntaxError('Unexpected token \'I\', "Internal S"... is not valid JSON');
        },
      }),
    });
    render(<SettingsModal {...devModeOff} isOpen onClose={() => {}} />);

    const failure = await screen.findByTestId('settings-skills-error');
    expect(failure).toHaveTextContent('Your skills could not be loaded');
    expect(failure).toHaveTextContent('UClone-X answered with an error but gave no reason');
    for (const raw of ['500', 'HTTP', 'Internal Server Error', 'Unexpected token', 'JSON']) {
      expect(failure).not.toHaveTextContent(raw);
    }
  });

  it('says it is loading the skills while the read is out, not nothing and not an empty list', async () => {
    // Killed by: frontend/src/components/settings/SkillsSection.tsx :: {fault === null && data === null && (
    // Becomes: {false && (
    mockFetch({ '/api/skills': () => new Promise<RouteResponse>(() => {}) });
    render(<SettingsModal {...devModeOff} isOpen onClose={() => {}} />);

    const section = await screen.findByTestId('settings-skills');
    expect(within(section).getByRole('status')).toHaveTextContent('Loading your skills');
    expect(screen.queryByTestId('settings-skills-empty')).toBeNull();
    expect(screen.queryByTestId('settings-skills-error')).toBeNull();
  });

  it('offers to try again after a failed read, and shows the skills once it succeeds', async () => {
    // Killed by: frontend/src/components/settings/SkillsSection.tsx :: onRetry={reload}
    // Becomes: onRetry={() => {}}
    vi.spyOn(console, 'error').mockImplementation(() => {});
    let reads = 0;
    mockFetch({
      '/api/skills': () => {
        reads += 1;
        return reads === 1 ? jsonResponse({ detail: 'boom' }, false, 500) : jsonResponse(skillsPayload);
      },
    });
    render(<SettingsModal {...devModeOff} isOpen onClose={() => {}} />);

    const failure = await screen.findByTestId('settings-skills-error');
    fireEvent.click(within(failure).getByRole('button', { name: 'Try again' }));

    const section = screen.getByTestId('settings-skills');
    await waitFor(() => expect(section).toHaveTextContent('csv_summariser'));
    expect(screen.queryByTestId('settings-skills-error')).toBeNull();
  });

  it("describes the skills in plain words, with none of the checker's internal terms", async () => {
    // Killed by: frontend/src/i18n/locales/en/skills.json :: "hint": "Passed the safety check"
    // Becomes: "hint": "AST Verification Passed"
    mockFetch({ '/api/skills': () => jsonResponse(skillsPayload) });
    render(<SettingsModal {...devModeOff} isOpen onClose={() => {}} />);

    const section = await screen.findByTestId('settings-skills');
    await waitFor(() => expect(section).toHaveTextContent('csv_summariser'));
    const text = section.textContent ?? '';
    const shown = Array.from(section.querySelectorAll('option')).map((o) => o.textContent ?? '');
    for (const term of [
      // Letters only on either side: textContent runs a count into the next word ("1AST").
      /(?<![A-Za-z])AST(?![A-Za-z])/,
      /quarantin/i,
      /governed/i,
      /audit/i,
      /verdict/i,
      /taint/i,
      /registry/i,
      /synthesi[sz]ed/i,
      /SHA-256/,
      /(?<![A-Za-z])LLM(?![A-Za-z])/,
    ]) {
      expect(text, String(term)).not.toMatch(term);
      for (const option of shown) expect(option, String(term)).not.toMatch(term);
    }
  });

  it('says why a skill changed after its approval is not used (#1720)', async () => {
    // Killed by: frontend/src/components/SkillsTab.tsx :: <strong>{notLoadedCopy.label}</strong> {notLoaded}
    // Becomes: <strong>{notLoadedCopy.label}</strong>
    const reason =
      'This skill was changed after it was approved, so it is not used. To use it, check what ' +
      'changed, then approve it again in a terminal window: ucx skill approve csv_summariser';
    const [skill] = skillsPayload.skills;
    mockFetch({
      '/api/skills': () =>
        jsonResponse({
          skills: [{ ...skill, status: 'quarantined', not_loaded_reason: reason }],
          summary: { total_skills: 1, active_count: 0, pending_count: 0, quarantined_count: 1 },
        }),
    });
    render(<SettingsModal {...devModeOff} isOpen onClose={() => {}} />);

    const shown = await screen.findByTestId('skill-not-loaded-reason');
    expect(shown).toHaveTextContent(`Why it is not used: ${reason}`);
    expect(screen.getByTestId('settings-skills')).toHaveTextContent('Blocked');
  });

  /** A skill the Core refused, with the reason as a code (#1777). */
  const refusedSkill = (code: string, reason: string, name = 'csv_summariser') => ({
    ...skillsPayload.skills[0],
    name,
    status: 'quarantined',
    not_loaded_reason: reason,
    not_loaded_code: code,
    not_loaded_params: { name },
  });
  const refusedPayload = (...skills: ReturnType<typeof refusedSkill>[]) => ({
    skills,
    summary: {
      total_skills: skills.length,
      active_count: 0,
      pending_count: 0,
      quarantined_count: skills.length,
    },
  });
  const englishChanged = fmt(en.skills.notLoaded.codes.changed_after_approval, {
    name: 'csv_summariser',
  });

  it('words the reason a skill is not used in Korean when the screens are in Korean (#1777)', async () => {
    // Killed by: frontend/src/components/SkillsTab.tsx :: if (!code || !Object.prototype.hasOwnProperty.call(t.codes, code)) return fallback;
    // Becomes: return fallback;
    window.localStorage.clear();
    mockFetch({
      '/api/settings': () => jsonResponse({ ...baseSettings, ui_language: 'ko' }),
      '/api/skills': () =>
        jsonResponse(refusedPayload(refusedSkill('changed_after_approval', englishChanged))),
    });
    render(
      <LocaleProvider hints={['ko-KR']}>
        <SettingsModal {...devModeOff} isOpen onClose={() => {}} />
      </LocaleProvider>,
    );

    const korean = fmt(ko.skills.notLoaded.codes.changed_after_approval, { name: 'csv_summariser' });
    const shown = await screen.findByTestId('skill-not-loaded-reason');
    await waitFor(() => expect(shown.textContent).toBe(`${ko.skills.notLoaded.label} ${korean}`));
    expect(shown.textContent).not.toContain(englishChanged);
  });

  it('words the reason from the English catalog, and keeps the sent words for an unknown code (#1777)', async () => {
    // Killed by: frontend/src/components/SkillsTab.tsx :: if (!code || !Object.prototype.hasOwnProperty.call(t.codes, code)) return fallback;
    // Becomes: if (!code) return fallback;
    const unknown = 'This skill is not used for a reason this screen does not know yet.';
    mockFetch({
      '/api/skills': () =>
        jsonResponse(
          refusedPayload(
            refusedSkill('renamed_later', unknown, 'alpha_skill'),
            refusedSkill('failed_safety_check', 'sent words', 'beta_skill'),
          ),
        ),
    });
    render(<SettingsModal {...devModeOff} isOpen onClose={() => {}} />);

    const shown = await screen.findByTestId('skill-not-loaded-reason');
    expect(shown.textContent).toBe(`${en.skills.notLoaded.label} ${unknown}`);
    fireEvent.click(within(screen.getByTestId('settings-skills')).getByText('beta_skill'));
    const safety = `${en.skills.notLoaded.label} ${en.skills.notLoaded.codes.failed_safety_check}`;
    await waitFor(() =>
      expect(screen.getByTestId('skill-not-loaded-reason').textContent).toBe(safety),
    );
    expectPlain(screen.getByTestId('skill-not-loaded-reason').textContent);
  });

  it('asks in one line for skills approved before approvals were recorded to be approved again (#1777)', async () => {
    // Killed by: frontend/src/components/settings/SkillsSection.tsx :: .filter((skill) => skill.not_loaded_code === 'approved_before_pins')
    // Becomes: .filter((skill) => skill.not_loaded_code === 'approved_before_pins_')
    const before = (name: string) =>
      refusedSkill(
        'approved_before_pins',
        fmt(en.skills.notLoaded.codes.approved_before_pins, { name }),
        name,
      );
    mockFetch({
      '/api/skills': () =>
        jsonResponse(
          refusedPayload(
            before('alpha_skill'),
            refusedSkill('changed_after_approval', englishChanged),
            before('beta_skill'),
          ),
        ),
    });
    render(<SettingsModal {...devModeOff} isOpen onClose={() => {}} />);

    const notice = await screen.findByTestId('settings-skills-reapprove');
    expect(notice.textContent).toBe(
      plural(en.skills.reapprove, 2, { names: 'alpha_skill, beta_skill' }),
    );
    expect(notice).toHaveTextContent('ucx skill approve');
    expect(notice).not.toHaveTextContent('csv_summariser');
    expectPlain(notice.textContent);
  });

  it('shows no re-approval notice when every skill was approved with a record (#1777)', async () => {
    // Killed by: frontend/src/components/settings/SkillsSection.tsx :: {fault === null && reapprove.length > 0 && (
    // Becomes: {fault === null && (
    mockFetch({ '/api/skills': () => jsonResponse(skillsPayload) });
    render(<SettingsModal {...devModeOff} isOpen onClose={() => {}} />);

    await waitFor(() =>
      expect(screen.getByTestId('settings-skills')).toHaveTextContent('csv_summariser'),
    );
    expect(screen.queryByTestId('settings-skills-reapprove')).toBeNull();
  });

  /** Every sentence of the tab's English copy, for checking none of it shows on a Korean screen. */
  const englishTabCopy = (): string[] => {
    const out: string[] = [];
    const walk = (value: unknown) => {
      if (typeof value === 'string') out.push(value.replace(/\s*\(\{count\}\)$/, ''));
      else if (value && typeof value === 'object') Object.values(value).forEach(walk);
    };
    walk(en.skills.tab);
    return out;
  };

  it("words the whole Skills tab in Korean when the screens are in Korean, with none of it left in English (#1782)", async () => {
    // Killed by: frontend/src/components/SkillsTab.tsx :: <h3 className="text-sm font-semibold text-white">{copy.checkHeading}</h3>
    // Becomes: <h3 className="text-sm font-semibold text-white">Safety check</h3>
    window.localStorage.clear();
    mockFetch({
      '/api/settings': () => jsonResponse({ ...baseSettings, ui_language: 'ko' }),
      '/api/skills': () => jsonResponse(skillsPayload),
    });
    render(
      <LocaleProvider hints={['ko-KR']}>
        <SettingsModal {...devModeOff} isOpen onClose={() => {}} />
      </LocaleProvider>,
    );

    const section = await screen.findByTestId('settings-skills');
    await waitFor(() => expect(section).toHaveTextContent(ko.skills.tab.checkHeading));
    const tab = ko.skills.tab;
    for (const korean of [
      tab.cards.active.hint,
      tab.status.active,
      tab.recommendation.approve,
      tab.noFindings,
      fmt(tab.listHeading, { count: 1 }),
    ]) {
      expect(section).toHaveTextContent(korean);
    }
    expect(within(section).getByRole('button', { name: tab.refresh })).toBeInTheDocument();
    expect(within(section).getByPlaceholderText(tab.search)).toBeInTheDocument();
    const options = Array.from(section.querySelectorAll('option')).map((o) => o.textContent ?? '');
    expect(options).toContain(tab.anyStatus);
    expect(options).toContain(tab.origin.human);

    const shown = [section.textContent ?? '', ...options, tab.search].join('\n');
    for (const english of englishTabCopy()) {
      expect(shown, english).not.toContain(english);
    }
    // Plain words in Korean too: none of the checker's own terms.
    expect(shown).not.toMatch(/(?<![A-Za-z])AST(?![A-Za-z])|quarantin|audit|verdict|synthesi[sz]ed|SHA-256/i);
  });

  it('counts the skills the search leaves in the list heading, in words (#1782)', async () => {
    // Killed by: frontend/src/components/SkillsTab.tsx :: {fmt(copy.listHeading, { count: filteredSkills.length })}
    // Becomes: {copy.listHeading}
    mockFetch({ '/api/skills': () => jsonResponse(skillsPayload) });
    render(<SettingsModal {...devModeOff} isOpen onClose={() => {}} />);

    const section = await screen.findByTestId('settings-skills');
    await waitFor(() => expect(section).toHaveTextContent('Skills (1)'));
    fireEvent.change(within(section).getByPlaceholderText(en.skills.tab.search), {
      target: { value: 'no such skill' },
    });
    await waitFor(() => expect(section).toHaveTextContent('Skills (0)'));
    expect(section.textContent).not.toContain('{count}');
  });

  it('says in words that no skill is registered', async () => {
    // Killed by: frontend/src/components/settings/SkillsSection.tsx :: skills.length === 0
    // Becomes: skills.length === -1
    mockFetch({
      '/api/skills': () =>
        jsonResponse({
          skills: [],
          summary: { total_skills: 0, active_count: 0, pending_count: 0, quarantined_count: 0 },
        }),
    });
    render(<SettingsModal {...devModeOff} isOpen onClose={() => {}} />);

    expect(await screen.findByTestId('settings-skills-empty')).toHaveTextContent(
      'No skills are registered',
    );
  });

  const noStorePayload = {
    skills: [],
    store_missing: true,
    summary: { total_skills: 0, active_count: 0, pending_count: 0, quarantined_count: 0 },
  };

  it('says no skill folder was found, and where to look, rather than that none is registered (#1721)', async () => {
    // Killed by: frontend/src/components/settings/SkillsSection.tsx :: {data.store_missing ? copy.noStore : copy.empty}
    // Becomes: {copy.empty}
    mockFetch({ '/api/skills': () => jsonResponse(noStorePayload) });
    render(<SettingsModal {...devModeOff} isOpen onClose={() => {}} />);

    const empty = await screen.findByTestId('settings-skills-empty');
    expect(empty.textContent).toBe(en.skills.noStore);
    expect(empty).toHaveTextContent('ucx skill --help');
    expect(empty).not.toHaveTextContent(en.skills.empty);
    expectPlain(empty.textContent);
    for (const internal of ['git', 'store_missing', 'rev-parse', '/']) {
      expect(empty.textContent, internal).not.toContain(internal);
    }
  });

  it('says in Korean that no skill folder was found when the screens are in Korean (#1721)', async () => {
    // Killed by: frontend/src/i18n/locales/ko/skills.json :: "noStore": "UClone-X를 시작한 프로젝트에
    // Becomes: "noStore": "No skills are loaded. UClone-X를 시작한 프로젝트에
    window.localStorage.clear();
    mockFetch({
      '/api/settings': () => jsonResponse({ ...baseSettings, ui_language: 'ko' }),
      '/api/skills': () => jsonResponse(noStorePayload),
    });
    render(
      <LocaleProvider hints={['ko-KR']}>
        <SettingsModal {...devModeOff} isOpen onClose={() => {}} />
      </LocaleProvider>,
    );

    const empty = await screen.findByTestId('settings-skills-empty');
    await waitFor(() => expect(empty.textContent).toBe(ko.skills.noStore));
    expect(empty.textContent).not.toMatch(/No skills|loaded/);
  });
});

describe('SettingsModal Diagnostics (#1358)', () => {
  beforeEach(() => vi.restoreAllMocks());
  afterEach(() => vi.unstubAllGlobals());

  const diagnosticsRoutes = {
    '/api/acp/status': () => jsonResponse(acpPayload),
    '/api/evaluations/latest': () => jsonResponse(evalsPayload),
  };

  it('shows the ACP report and the Evals scorecard while developer mode is on', async () => {
    // Killed by: frontend/src/components/SettingsModal.tsx :: {developerMode && <DiagnosticsSection />}
    // Becomes:
    mockFetch(diagnosticsRoutes);
    render(<SettingsModal isOpen onClose={() => {}} developerMode onDeveloperModeChange={() => {}} />);

    const section = await screen.findByTestId('settings-diagnostics');
    expect(section).toHaveTextContent('Diagnostics');
    await waitFor(() => expect(screen.getByTestId('acp-panel')).toBeInTheDocument());
    expect(screen.getByTestId('acp-presence')).toHaveTextContent('No ACP shell is installed in this build.');
    await waitFor(() => expect(screen.getAllByTestId('eval-metric-card')).toHaveLength(4));
  });

  it('offers neither, and reads neither, while developer mode is off', async () => {
    // Killed by: frontend/src/components/SettingsModal.tsx :: {developerMode && <DiagnosticsSection />}
    // Becomes: {<DiagnosticsSection />}
    const calls = mockFetch(diagnosticsRoutes);
    render(<SettingsModal {...devModeOff} isOpen onClose={() => {}} />);

    // The modal rendered and read its data, so what is missing is missing from a live modal.
    await screen.findByTestId('settings-skills');
    await waitFor(() => expect(calls.some((c) => c.url.includes('/api/skills'))).toBe(true));
    expect(screen.queryByTestId('settings-diagnostics')).toBeNull();
    expect(screen.queryByTestId('acp-panel')).toBeNull();
    expect(screen.queryByTestId('acp-unavailable')).toBeNull();
    expect(screen.queryAllByTestId('eval-metric-card')).toHaveLength(0);
    expect(calls.filter((c) => c.url.includes('/api/acp/status'))).toEqual([]);
    expect(calls.filter((c) => c.url.includes('/api/evaluations/latest'))).toEqual([]);
  });

  it("keeps the Evals read failure's cause in words (#1344)", async () => {
    // Killed by: frontend/src/components/EvaluationsTab.tsx :: evaluationsData?.status === 'error'
    // Becomes: evaluationsData?.status === 'never'
    const readError = 'Could not read evaluation reports from /srv/evals/reports: NotADirectoryError';
    mockFetch({
      ...diagnosticsRoutes,
      '/api/evaluations/latest': () =>
        jsonResponse({ status: 'error', error: readError, suites: [], scorecard: {}, metrics: evalMetrics }),
    });
    render(<SettingsModal isOpen onClose={() => {}} developerMode onDeveloperModeChange={() => {}} />);

    const alert = await screen.findByTestId('eval-read-failure');
    expect(alert).toHaveTextContent('Evaluation results could not be read');
    expect(alert).toHaveTextContent(readError);
    expect(screen.queryAllByTestId('eval-metric-card')).toHaveLength(0);
  });

  it('states why the ACP report could not be read, rather than drawing it as unread', async () => {
    // Killed by: frontend/src/components/settings/DiagnosticsSection.tsx :: {acp.error !== null ? (
    // Becomes: {false ? (
    mockFetch({
      ...diagnosticsRoutes,
      '/api/acp/status': () => jsonResponse({ detail: 'no' }, false, 503),
    });
    render(<SettingsModal isOpen onClose={() => {}} developerMode onDeveloperModeChange={() => {}} />);

    const failure = await screen.findByTestId('diagnostics-acp-error');
    expect(failure).toHaveTextContent('ACP status could not be read');
    expect(failure).toHaveTextContent('HTTP 503');
  });

  it('states why the Evals results could not be read, rather than a scorecard of zeros', async () => {
    // Killed by: frontend/src/components/settings/DiagnosticsSection.tsx :: {evals.error !== null ? (
    // Becomes: {false ? (
    vi.spyOn(console, 'error').mockImplementation(() => {});
    mockFetch({
      ...diagnosticsRoutes,
      '/api/evaluations/latest': () => jsonResponse({ detail: 'no' }, false, 502),
    });
    render(<SettingsModal isOpen onClose={() => {}} developerMode onDeveloperModeChange={() => {}} />);

    const failure = await screen.findByTestId('diagnostics-evals-error');
    expect(failure).toHaveTextContent('Evaluation results could not be read');
    expect(failure).toHaveTextContent('HTTP 502');
    expect(screen.queryAllByTestId('eval-metric-card')).toHaveLength(0);
  });

  it('says each read is loading while it is out, never "not read" and never zeros', async () => {
    // Killed by: frontend/src/components/settings/DiagnosticsSection.tsx :: ) : acp.data === null ? (
    // Becomes: ) : false ? (
    // Killed by: frontend/src/components/settings/DiagnosticsSection.tsx :: ) : evals.data === null ? (
    // Becomes: ) : false ? (
    const hold = () => new Promise<RouteResponse>(() => {});
    mockFetch({ '/api/acp/status': hold, '/api/evaluations/latest': hold });
    render(<SettingsModal isOpen onClose={() => {}} developerMode onDeveloperModeChange={() => {}} />);

    expect(await screen.findByTestId('diagnostics-acp-loading')).toHaveTextContent('Reading ACP status');
    expect(screen.getByTestId('diagnostics-evals-loading')).toHaveTextContent('Reading evaluation results');
    expect(screen.queryByTestId('acp-unavailable')).toBeNull();
    expect(screen.queryAllByTestId('eval-metric-card')).toHaveLength(0);
  });

  it('retries a failed ACP read from its error, and shows the report once it succeeds', async () => {
    // Killed by: frontend/src/components/settings/DiagnosticsSection.tsx :: onRetry={acp.reload}
    // Becomes: onRetry={() => {}}
    vi.spyOn(console, 'error').mockImplementation(() => {});
    let reads = 0;
    mockFetch({
      ...diagnosticsRoutes,
      '/api/acp/status': () => {
        reads += 1;
        return reads === 1 ? jsonResponse({ detail: 'no' }, false, 503) : jsonResponse(acpPayload);
      },
    });
    render(<SettingsModal isOpen onClose={() => {}} developerMode onDeveloperModeChange={() => {}} />);

    const failure = await screen.findByTestId('diagnostics-acp-error');
    fireEvent.click(within(failure).getByRole('button', { name: 'Try again' }));
    await waitFor(() => expect(screen.getByTestId('acp-panel')).toBeInTheDocument());
    expect(screen.queryByTestId('diagnostics-acp-error')).toBeNull();
  });

  it('retries a failed Evals read from its error, and shows the scorecard once it succeeds', async () => {
    // Killed by: frontend/src/components/settings/DiagnosticsSection.tsx :: onRetry={evals.reload}
    // Becomes: onRetry={() => {}}
    vi.spyOn(console, 'error').mockImplementation(() => {});
    let reads = 0;
    mockFetch({
      ...diagnosticsRoutes,
      '/api/evaluations/latest': () => {
        reads += 1;
        return reads === 1 ? jsonResponse({ detail: 'no' }, false, 502) : jsonResponse(evalsPayload);
      },
    });
    render(<SettingsModal isOpen onClose={() => {}} developerMode onDeveloperModeChange={() => {}} />);

    const failure = await screen.findByTestId('diagnostics-evals-error');
    fireEvent.click(within(failure).getByRole('button', { name: 'Try again' }));
    await waitFor(() => expect(screen.getAllByTestId('eval-metric-card')).toHaveLength(4));
    expect(screen.queryByTestId('diagnostics-evals-error')).toBeNull();
  });
});

describe('SettingsModal read-only folders', () => {
  beforeEach(() => vi.restoreAllMocks());
  afterEach(() => vi.unstubAllGlobals());

  const withRoots: RuntimeSettings = {
    ...baseSettings,
    workspace_dir: '/home/u/.uclone/workspace',
    read_roots: ['/data/papers', '~/gone'],
    read_roots_missing: ['~/gone'],
  };

  const saveCall = (calls: ReturnType<typeof mockFetch>) =>
    calls.find((c) => c.method === 'POST' && c.url.endsWith('/api/settings'));

  it('shows the workspace folder and each read-only folder, marking the missing one', async () => {
    mockFetch({ '/api/settings': () => jsonResponse(withRoots) });
    render(<SettingsModal {...devModeOff} isOpen onClose={() => {}} />);

    expect((await screen.findByTestId('settings-workspace-dir')).textContent).toBe(
      '/home/u/.uclone/workspace',
    );
    const list = await screen.findByRole('list', { name: 'Read-only folders' });
    const rows = Array.from(list.querySelectorAll('li'));
    expect(rows.map((r) => r.textContent)).toEqual(['/data/papers', '~/gonenot usable']);
  });

  it('shows the folders set by UCLONE_READ_ROOTS and the entries it ignored', async () => {
    mockFetch({
      '/api/settings': () =>
        jsonResponse({
          ...withRoots,
          read_roots_env: ['/srv/shared'],
          read_roots_env_ignored: ["'rel' is not a full path; start it with / or ~ (from UCLONE_READ_ROOTS)"],
        }),
    });
    render(<SettingsModal {...devModeOff} isOpen onClose={() => {}} />);

    const envList = await screen.findByRole('list', { name: 'Read-only folders from the environment' });
    expect(Array.from(envList.querySelectorAll('li')).map((r) => r.textContent)).toEqual(['/srv/shared']);
    expect(screen.getByTestId('settings-read-roots-env-ignored').textContent).toContain("'rel' is not a full path");
  });

  it('says there are no other folders instead of showing an empty list', async () => {
    mockFetch({ '/api/settings': () => jsonResponse({ ...withRoots, read_roots: [], read_roots_missing: [] }) });
    render(<SettingsModal {...devModeOff} isOpen onClose={() => {}} />);

    expect(await screen.findByTestId('settings-read-roots-empty')).toBeTruthy();
    expect(screen.queryByRole('list', { name: 'Read-only folders' })).toBeNull();
  });

  it('sends the edited list as read_roots on save', async () => {
    const calls = mockFetch({
      '/api/settings': (_url, init) =>
        init?.method === 'POST'
          ? jsonResponse({ ...withRoots, read_roots: ['/data/papers', '~/notes'], read_roots_missing: [] })
          : jsonResponse(withRoots),
    });
    render(<SettingsModal {...devModeOff} isOpen onClose={() => {}} />);

    fireEvent.click(await screen.findByRole('button', { name: 'Remove ~/gone' }));
    fireEvent.change(screen.getByLabelText(/folder to add/i), { target: { value: '  ~/notes ' } });
    fireEvent.click(screen.getByRole('button', { name: /add folder/i }));
    fireEvent.click(screen.getByRole('button', { name: /save & apply/i }));

    await waitFor(() => expect(saveCall(calls)).toBeTruthy());
    expect((saveCall(calls)?.body as { read_roots?: string[] }).read_roots).toEqual([
      '/data/papers',
      '~/notes',
    ]);
  });

  it('leaves read_roots out of the save when the list was not touched', async () => {
    const calls = mockFetch({ '/api/settings': () => jsonResponse(withRoots) });
    render(<SettingsModal {...devModeOff} isOpen onClose={() => {}} />);

    await screen.findByRole('list', { name: 'Read-only folders' });
    fireEvent.click(screen.getByRole('button', { name: /save & apply/i }));

    await waitFor(() => expect(saveCall(calls)).toBeTruthy());
    expect(saveCall(calls)?.body).not.toHaveProperty('read_roots');
  });

  it("shows the server's reason when it refuses a folder", async () => {
    const detail = "read_roots: '/nope' is not an existing folder";
    mockFetch({
      '/api/settings': (_url, init) =>
        init?.method === 'POST' ? jsonResponse({ detail }, false, 400) : jsonResponse(withRoots),
    });
    render(<SettingsModal {...devModeOff} isOpen onClose={() => {}} />);

    fireEvent.change(await screen.findByLabelText(/folder to add/i), { target: { value: '/nope' } });
    fireEvent.click(screen.getByRole('button', { name: /add folder/i }));
    fireEvent.click(screen.getByRole('button', { name: /save & apply/i }));

    // Said as a sentence: the Core's words, with the full stop it left off (#1436).
    expect(await screen.findByText(`${detail}.`)).toBeTruthy();
  });
});

describe('SettingsModal external tool servers', () => {
  beforeEach(() => vi.restoreAllMocks());
  afterEach(() => vi.unstubAllGlobals());

  it('shows the tool server section, read from /api/mcp/servers', async () => {
    // Killed by: frontend/src/components/SettingsModal.tsx :: <McpServersSection />
    // Becomes:
    mockFetch({ '/api/mcp/servers': () => jsonResponse({ config_path: '/c/mcp.json', servers: [] }) });
    render(<SettingsModal {...devModeOff} isOpen onClose={() => {}} />);

    const section = await screen.findByTestId('settings-mcp');
    expect(section).toHaveTextContent('External tools (MCP)');
    expect(await screen.findByTestId('settings-mcp-empty')).toHaveTextContent(
      'No tool servers are connected yet.',
    );
  });
});

describe('SettingsModal failures in plain words (#1436)', () => {
  beforeEach(() => {
    vi.restoreAllMocks();
    vi.spyOn(console, 'error').mockImplementation(() => {});
    vi.stubGlobal('confirm', vi.fn(() => true));
  });
  afterEach(() => vi.unstubAllGlobals());

  /** The two answers a reader must never see as text: no answer at all, and a 500 with no JSON. */
  const faults: Array<[string, Handler]> = [
    ['a rejected fetch', () => Promise.reject(new TypeError('Failed to fetch'))],
    [
      'a bodyless 500',
      () => ({
        ok: false,
        status: 500,
        statusText: 'Internal Server Error',
        json: async () => {
          throw new SyntaxError('Unexpected token \'I\', "Internal S"... is not valid JSON');
        },
      }),
    ],
  ];

  /** Only the named request fails, and only for `method`; every other request answers as usual. */
  const failOnly = (path: string, method: string, fault: Handler): Handler => (url, init) => {
    if ((init?.method ?? 'GET') === method) return fault(url, init);
    if (path === '/api/settings') return jsonResponse(baseSettings);
    return jsonResponse({});
  };

  it.each([
    ...faults,
  ])('says the settings could not be loaded after %s', async (_label, fault) => {
    // Killed by: frontend/src/components/SettingsModal.tsx :: plainFailure(err, copyRef.current.settings.failure.load)
    // Becomes: String(err)
    mockFetch({ '/api/settings': failOnly('/api/settings', 'GET', fault) });
    render(<SettingsModal {...devModeOff} isOpen onClose={() => {}} />);

    const feedback = await screen.findByTestId('settings-feedback');
    expect(feedback).toHaveTextContent(SETTINGS_FAILURE.load);
    expectPlain(feedback.textContent);
  });

  it.each([
    ...faults,
  ])('says the connection could not be tested after %s', async (_label, fault) => {
    // Killed by: frontend/src/components/SettingsModal.tsx :: plainFailure(err, t.failure.test)
    // Becomes: String(err)
    mockFetch({ '/api/settings/test': fault, '/api/settings': () => jsonResponse(baseSettings) });
    render(<SettingsModal {...devModeOff} isOpen onClose={() => {}} />);

    await screen.findByLabelText(/model to install/i);
    fireEvent.click(screen.getByRole('button', { name: /check connection/i }));

    const feedback = await screen.findByTestId('settings-feedback');
    expect(feedback).toHaveTextContent(SETTINGS_FAILURE.test);
    expectPlain(feedback.textContent);
  });

  it.each([
    ...faults,
  ])('says saving went wrong after %s', async (_label, fault) => {
    // Killed by: frontend/src/components/SettingsModal.tsx :: plainFailure(err, t.failure.save)
    // Becomes: String(err)
    mockFetch({ '/api/settings': failOnly('/api/settings', 'POST', fault) });
    render(<SettingsModal {...devModeOff} isOpen onClose={() => {}} />);

    await screen.findByLabelText(/model to install/i);
    fireEvent.click(screen.getByRole('button', { name: /save & apply/i }));

    const feedback = await screen.findByTestId('settings-feedback');
    expect(feedback).toHaveTextContent(SETTINGS_FAILURE.save);
    expectPlain(feedback.textContent);
  });

  it.each([
    ...faults,
  ])('says installing went wrong after %s', async (_label, fault) => {
    // Killed by: frontend/src/components/SettingsModal.tsx :: plainFailure(err, t.failure.install)
    // Becomes: String(err)
    mockFetch({ '/api/models/pull': fault });
    render(<SettingsModal {...devModeOff} isOpen onClose={() => {}} />);

    fireEvent.change(await screen.findByLabelText(/model to install/i), { target: { value: 'llama3.2:3b' } });
    fireEvent.click(screen.getByRole('button', { name: /install/i }));

    const feedback = await screen.findByTestId('settings-feedback');
    expect(feedback).toHaveTextContent(SETTINGS_FAILURE.install);
    expectPlain(feedback.textContent);
  });

  it.each([
    ...faults,
  ])('says deleting went wrong after %s', async (_label, fault) => {
    // Killed by: frontend/src/components/SettingsModal.tsx :: plainFailure(err, t.failure.remove)
    // Becomes: String(err)
    mockFetch({ '/api/models/delete': fault });
    render(<SettingsModal {...devModeOff} isOpen onClose={() => {}} />);

    fireEvent.click(await screen.findByRole('button', { name: /delete llama3\.2:1b/i }));

    const feedback = await screen.findByTestId('settings-feedback');
    expect(feedback).toHaveTextContent(SETTINGS_FAILURE.remove);
    expectPlain(feedback.textContent);
  });

  it.each([
    ...faults,
  ])('says the clones could not be loaded after %s', async (_label, fault) => {
    // Killed by: frontend/src/components/SettingsModal.tsx :: setPersonaLoadError(coreReason(err) ?? '');
    // Becomes: setPersonaLoadError(String(err));
    mockFetch({ '/api/personas': fault });
    render(<SettingsModal {...devModeOff} isOpen onClose={() => {}} />);

    const alert = await screen.findByText(/could not load clones/i);
    expect(alert).toHaveTextContent(PERSONA_EDITOR_COPY.loadFailed);
    expectPlain(alert.textContent);
  });

  it("shows the Core's own reason when it gives one, in place of the fixed sentence", async () => {
    // Killed by: frontend/src/lib/coreFailure.ts :: return coreReason(err) ?? asSentence(fallback);
    // Becomes: return asSentence(fallback);
    mockFetch({
      '/api/settings': failOnly('/api/settings', 'GET', () =>
        jsonResponse({ detail: 'The settings file is not readable' }, false, 500),
      ),
    });
    render(<SettingsModal {...devModeOff} isOpen onClose={() => {}} />);

    const feedback = await screen.findByTestId('settings-feedback');
    expect(feedback).toHaveTextContent('The settings file is not readable.');
    expect(feedback).not.toHaveTextContent(SETTINGS_FAILURE.load);
  });

  it("keeps the Core's reason for the clones when it gives one", async () => {
    // Killed by: frontend/src/lib/personasApi.ts :: if (!res.ok) throw await failureOf(res);
    // Becomes: if (!res.ok) throw new Error(`HTTP ${res.status}`);
    mockFetch({ '/api/personas': () => jsonResponse({ detail: 'The clones folder is missing.' }, false, 500) });
    render(<SettingsModal {...devModeOff} isOpen onClose={() => {}} />);

    const alert = await screen.findByText(/could not load clones/i);
    expect(alert).toHaveTextContent(fmt(PERSONA_EDITOR_COPY.loadFailedBecause, { reason: 'The clones folder is missing.' }));
  });
});

describe('SettingsModal API key onboarding', () => {
  const geminiSettings: RuntimeSettings = {
    ...baseSettings,
    llm_provider: 'gemini',
    llm_model: 'gemini-1.5-pro',
  };

  it('renders a direct console link when a public provider is selected', async () => {
    mockFetch({
      '/api/settings': () => jsonResponse(geminiSettings),
    });
    render(<SettingsModal {...devModeOff} isOpen onClose={() => {}} />);

    const link = (await screen.findByTestId('provider-key-console-link')) as HTMLAnchorElement;
    expect(link).toBeTruthy();
    expect(link.href).toBe('https://aistudio.google.com/app/apikey');
    expect(link.textContent).toContain('Google Gemini');
  });

  it('shows a format warning when a mismatched key is entered', async () => {
    mockFetch({
      '/api/settings': () => jsonResponse(geminiSettings),
    });
    render(<SettingsModal {...devModeOff} isOpen onClose={() => {}} />);

    await screen.findByTestId('provider-key-console-link');
    const input = screen.getByPlaceholderText(/Enter API key/i);
    fireEvent.change(input, { target: { value: 'sk-proj-invalid1234567890' } });

    const warning = await screen.findByTestId('api-key-warning');
    expect(warning.textContent).toContain('OpenAI');
  });

  it('shows a valid format badge when a correct key is entered', async () => {
    mockFetch({
      '/api/settings': () => jsonResponse(geminiSettings),
    });
    render(<SettingsModal {...devModeOff} isOpen onClose={() => {}} />);

    await screen.findByTestId('provider-key-console-link');
    const input = screen.getByPlaceholderText(/Enter API key/i);
    fireEvent.change(input, { target: { value: 'AIzaSy' + 'A'.repeat(33) } });

    const validBadge = await screen.findByTestId('api-key-valid');
    expect(validBadge.textContent).toContain('This looks like a valid Google Gemini key.');
  });

  // Killed by: frontend/src/components/SettingsModal.tsx :: setLlmModel(() => '');
  // Becomes: setLlmModel((m) => m);
  it('clears localhost:11434 and the Ollama model when switching to Gemini, and writes no model of its own', async () => {
    // The listing never answers here, so what shows is the form before Google has spoken.
    mockFetch({
      '/api/models/catalog': () => new Promise<never>(() => {}),
      '/api/settings': () =>
        jsonResponse({
          ...baseSettings,
          llm_provider: 'ollama',
          llm_base_url: 'http://localhost:11434',
          llm_model: 'qwen3:8b',
        }),
    });
    render(<SettingsModal {...devModeOff} isOpen onClose={() => {}} />);

    const endpoint = (await screen.findByPlaceholderText('http://localhost:11434')) as HTMLInputElement;
    expect(endpoint.value).toBe('http://localhost:11434');

    fireEvent.click(screen.getByRole('button', { name: /Gemini/ }));

    // No model id is remembered in source: the field is empty until Google's list says
    // what exists, and the reason is said in words.
    expect((screen.getByRole('textbox', { name: 'Model name' }) as HTMLInputElement).value).toBe('');
    expect(screen.getByTestId('settings-model-catalog-detail').textContent).toBe(
      fmt(en.settings.catalog.loading, { vendor: 'Google' }),
    );
    expect(screen.queryByTestId('curated-models-pills')).toBeNull();
    // The endpoint is now an override behind Advanced, closed and empty.
    const advanced = screen.getByRole('button', { name: en.settings.catalog.advanced });
    expect(advanced).toHaveAttribute('aria-expanded', 'false');
    fireEvent.click(advanced);
    expect((screen.getByPlaceholderText(en.settings.catalog.endpointPlaceholder) as HTMLInputElement).value).toBe('');
  });

  it('does not display previous provider configured key badge when switching to Gemini', async () => {
    mockFetch({
      '/api/settings': () =>
        jsonResponse({
          ...baseSettings,
          llm_provider: 'openai',
          llm_base_url: 'https://api.openai.com/v1',
          llm_model: 'gpt-4o',
          llm_api_key_set: true,
          llm_api_key_masked: 'sk-...1234',
        }),
    });
    render(<SettingsModal {...devModeOff} isOpen onClose={() => {}} />);

    await screen.findByText('Saved (sk-...1234)');

    fireEvent.click(screen.getByRole('button', { name: /Gemini/ }));

    expect(screen.queryByText('Saved (sk-...1234)')).toBeNull();
    expect(screen.getByPlaceholderText(/AIzaSy/i)).toBeTruthy();
  });

  it('switches visible sections when tabs are clicked', async () => {
    mockFetch({
      '/api/settings': () => jsonResponse(geminiSettings),
    });
    render(<SettingsModal {...devModeOff} isOpen onClose={() => {}} />);

    await screen.findByTestId('settings-tabs-bar');
    expect(screen.getByTestId('settings-tab-llm')).toBeTruthy();

    fireEvent.click(screen.getByTestId('settings-tab-llm'));
    expect(screen.getByRole('button', { name: /Gemini/ })).toBeTruthy();
  });

  it('opens on the tab a link asks for, each time it opens', async () => {
    mockFetch({
      '/api/settings': () => jsonResponse(geminiSettings),
    });
    const props = { ...devModeOff, onClose: () => {}, initialTab: 'usage' as const };
    const { rerender } = render(<SettingsModal {...props} isOpen />);

    expect(await screen.findByTestId('settings-usage')).toBeTruthy();
    expect(screen.queryByRole('button', { name: /Gemini/ })).toBeNull();

    // The user moves away, closes, and follows the link again: Usage, not where they left it.
    fireEvent.click(screen.getByTestId('settings-tab-all'));
    rerender(<SettingsModal {...props} isOpen={false} />);
    rerender(<SettingsModal {...props} isOpen />);
    await waitFor(() => expect(screen.queryByRole('button', { name: /Gemini/ })).toBeNull());
    expect(screen.getByTestId('settings-usage')).toBeTruthy();
  });

  it('prevents tabs bar and header from shrinking and hides scrollbars (#1644)', async () => {
    // Killed by: frontend/src/components/SettingsModal.tsx :: scrollbar-hide shrink-0
    // Becomes: scrollbar-none
    mockFetch({
      '/api/settings': () => jsonResponse(geminiSettings),
    });
    render(<SettingsModal {...devModeOff} isOpen onClose={() => {}} />);

    const header = await screen.findByTestId('settings-header');
    expect(header.className).toContain('shrink-0');

    const tabsBar = await screen.findByTestId('settings-tabs-bar');
    expect(tabsBar.className).toContain('shrink-0');
    expect(tabsBar.className).toContain('scrollbar-hide');

    const allTab = screen.getByTestId('settings-tab-all');
    expect(allTab.className).toContain('shrink-0');
  });
});



describe('SettingsModal cloud model list (#1631)', () => {
  beforeEach(() => vi.restoreAllMocks());
  afterEach(() => vi.unstubAllGlobals());

  const entry = (id: string, display_name: string | null, chat_capable = true) => ({
    id,
    display_name,
    context_window: null,
    max_output_tokens: null,
    chat_capable,
    created_at: null,
  });

  // Ids of no real provider: the picker shows whatever the listing says, and nothing else.
  const liveCatalog: CatalogResult = {
    provider: 'gemini',
    status: 'live',
    entries: [entry('model-alpha', 'Alpha'), entry('model-beta', 'Beta'), entry('embed-one', null, false)],
    recommended: 'model-beta',
    fetched_at: '2026-09-25T00:00:00Z',
    detail: null,
  };

  const geminiSettings = (over: Partial<RuntimeSettings> = {}): RuntimeSettings => ({
    ...baseSettings,
    llm_provider: 'gemini',
    llm_base_url: '',
    llm_model: '',
    llm_api_key_set: true,
    llm_api_key_masked: 'AIza...wxyz',
    providers_available: ['ollama', 'gemini'],
    available_models: ['model-alpha', 'model-beta'],
    catalog: liveCatalog,
    ...over,
  });

  const keyRejected: CatalogResult = {
    provider: 'gemini',
    status: 'key_rejected',
    entries: [],
    recommended: null,
    fetched_at: null,
    // The Core's English sentence, with the internals a raw error would carry.
    detail: 'ProviderAuthError: HTTP 401 PERMISSION_DENIED from /v1beta/models',
  };

  const saveBody = (calls: Array<{ url: string; method: string; body?: unknown }>) =>
    calls.find((c) => c.url === '/api/settings' && c.method === 'POST')?.body as { llm_model?: string } | undefined;

  // Killed by: frontend/src/components/SettingsModal.tsx :: setLlmModel(recommendedModel);
  // Becomes: setLlmModel(llmModelRef.current);
  it("preselects the provider's recommended model, and saves it", async () => {
    const calls = mockFetch({ '/api/settings': () => jsonResponse(geminiSettings()) });
    render(<SettingsModal {...devModeOff} isOpen onClose={() => {}} />);

    const select = (await screen.findByTestId('settings-model-select')) as HTMLSelectElement;
    await waitFor(() => expect(select.value).toBe('model-beta'));
    const labels = Array.from(select.options).map((o) => o.textContent);
    expect(labels).toEqual([
      'Alpha (model-alpha)',
      `Beta (model-beta) ${en.settings.catalog.recommended}`,
      en.settings.catalog.otherModel,
    ]);
    // Typing stays possible, but is not in the way of a listed choice.
    expect(screen.queryByRole('textbox', { name: 'Model name' })).toBeNull();

    fireEvent.click(screen.getByRole('button', { name: en.settings.footer.save }));
    await waitFor(() => expect(saveBody(calls)?.llm_model).toBe('model-beta'));
  });

  // Killed by: frontend/src/components/SettingsModal.tsx :: if (!recommendedModel || llmModelRef.current.trim() !== '') return;
  // Becomes: if (!recommendedModel) return;
  it('keeps a model the user already chose over the recommendation', async () => {
    const calls = mockFetch({ '/api/settings': () => jsonResponse(geminiSettings({ llm_model: 'model-alpha' })) });
    render(<SettingsModal {...devModeOff} isOpen onClose={() => {}} />);

    await screen.findByTestId('settings-model-select');
    // Read from what is saved, not from the field at first paint: the recommendation would
    // land a render later, after a check of the field had already passed.
    fireEvent.click(screen.getByRole('button', { name: en.settings.footer.save }));
    await waitFor(() => expect(saveBody(calls)?.llm_model).toBeDefined());
    expect(saveBody(calls)?.llm_model).toBe('model-alpha');
    expect((screen.getByTestId('settings-model-select') as HTMLSelectElement).value).toBe('model-alpha');
  });

  // Killed by: frontend/src/components/SettingsModal.tsx :: return fmt(t.catalog.keyRejected, { vendor });
  // Becomes: return catalog.detail ?? fmt(t.catalog.keyRejected, { vendor });
  it('says in plain words that the key was refused, with nothing to pick and typing still open', async () => {
    mockFetch({
      '/api/settings': () =>
        jsonResponse(geminiSettings({ catalog: keyRejected, available_models: [], llm_model: 'my-model' })),
    });
    render(<SettingsModal {...devModeOff} isOpen onClose={() => {}} />);

    const detail = await screen.findByTestId('settings-model-catalog-detail');
    expect(detail.textContent).toBe(fmt(en.settings.catalog.keyRejected, { vendor: 'Google' }));
    expectPlain(detail.textContent);
    expect(detail.textContent).not.toContain('401');
    expect(screen.queryByTestId('settings-model-select')).toBeNull();
    const typed = screen.getByRole('textbox', { name: 'Model name' }) as HTMLInputElement;
    expect(typed.value).toBe('my-model');
    // Nothing listed means nothing to compare against, so no "not in the list" note.
    expect(screen.queryByTestId('settings-model-unlisted')).toBeNull();
  });

  // Killed by: frontend/src/components/SettingsModal.tsx :: catalogIsLive && typedModel !== '' && !catalog.entries.some((e) => e.id === typedModel);
  // Becomes: false;
  it('lets a model missing from the list be typed, notes it softly, and saves it as typed', async () => {
    const calls = mockFetch({ '/api/settings': () => jsonResponse(geminiSettings()) });
    render(<SettingsModal {...devModeOff} isOpen onClose={() => {}} />);

    const select = (await screen.findByTestId('settings-model-select')) as HTMLSelectElement;
    await waitFor(() => expect(select.value).toBe('model-beta'));
    expect(screen.queryByTestId('settings-model-unlisted')).toBeNull();

    fireEvent.change(select, { target: { value: '__custom__' } });
    const typed = screen.getByRole('textbox', { name: 'Model name' });
    fireEvent.change(typed, { target: { value: 'model-gamma' } });

    const note = screen.getByTestId('settings-model-unlisted');
    expect(note.textContent).toBe(fmt(en.settings.catalog.notInList, { vendor: 'Google' }));
    expectPlain(note.textContent);

    fireEvent.click(screen.getByRole('button', { name: en.settings.footer.save }));
    await waitFor(() => expect(saveBody(calls)?.llm_model).toBe('model-gamma'));
  });

  // Killed by: frontend/src/components/SettingsModal.tsx :: const res = await fetch('/api/models?refresh=1');
  // Becomes: const res = await fetch('/api/models');
  it('asks the provider again when the list is refreshed, and shows what it now lists', async () => {
    const refreshed: CatalogResult = {
      ...liveCatalog,
      entries: [...liveCatalog.entries, entry('model-delta', 'Delta')],
    };
    const calls = mockFetch({
      '/api/settings': () => jsonResponse(geminiSettings()),
      '/api/models?refresh=1': () =>
        jsonResponse({
          provider: 'gemini',
          models: ['model-alpha', 'model-beta', 'model-delta'],
          current_model: 'model-beta',
          catalog: refreshed,
        }),
    });
    render(<SettingsModal {...devModeOff} isOpen onClose={() => {}} />);

    await screen.findByTestId('settings-model-select');
    fireEvent.click(screen.getByRole('button', { name: en.settings.catalog.refresh }));

    // Within the chat-model picker: the fast-model picker lists the same models.
    await waitFor(() =>
      expect(
        within(screen.getByTestId('settings-model-select')).getByRole('option', { name: 'Delta (model-delta)' }),
      ).toBeTruthy(),
    );
    expect(calls.some((c) => c.url === '/api/models?refresh=1')).toBe(true);
  });

  // Killed by: frontend/src/components/SettingsModal.tsx :: {endpointOpen && <div className="mt-2">{endpointField}</div>}
  // Becomes: {<div className="mt-2">{endpointField}</div>}
  it('keeps the endpoint override behind Advanced until it is asked for', async () => {
    mockFetch({ '/api/settings': () => jsonResponse(geminiSettings()) });
    render(<SettingsModal {...devModeOff} isOpen onClose={() => {}} />);

    const advanced = await screen.findByRole('button', { name: en.settings.catalog.advanced });
    expect(advanced).toHaveAttribute('aria-expanded', 'false');
    expect(screen.queryByPlaceholderText(en.settings.catalog.endpointPlaceholder)).toBeNull();

    fireEvent.click(advanced);
    expect(advanced).toHaveAttribute('aria-expanded', 'true');
    expect(screen.getByPlaceholderText(en.settings.catalog.endpointPlaceholder)).toBeTruthy();
  });

  // Killed by: frontend/src/components/SettingsModal.tsx :: setEndpointOpen(Boolean(data.llm_base_url));
  // Becomes: setEndpointOpen(false);
  it('opens Advanced by itself when a proxy is already set, so it is never hidden', async () => {
    mockFetch({
      '/api/settings': () => jsonResponse(geminiSettings({ llm_base_url: 'https://proxy.example/v1beta' })),
    });
    render(<SettingsModal {...devModeOff} isOpen onClose={() => {}} />);

    const endpoint = (await screen.findByPlaceholderText(
      en.settings.catalog.endpointPlaceholder,
    )) as HTMLInputElement;
    expect(endpoint.value).toBe('https://proxy.example/v1beta');
  });

  // Killed by: frontend/src/components/SettingsModal.tsx :: {t.catalog.advanced}
  // Becomes: {'Advanced — custom endpoint / proxy'}
  it('shows the model list in Korean, with no English sentence left', async () => {
    window.localStorage.clear();
    mockFetch({
      '/api/settings': () =>
        jsonResponse(geminiSettings({ catalog: keyRejected, available_models: [], ui_language: 'ko' })),
    });
    render(
      <LocaleProvider hints={['ko-KR']}>
        <SettingsModal {...devModeOff} isOpen onClose={() => {}} />
      </LocaleProvider>,
    );

    const detail = await screen.findByTestId('settings-model-catalog-detail');
    expect(detail.textContent).toBe(fmt(ko.settings.catalog.keyRejected, { vendor: 'Google' }));
    expect(screen.getByRole('button', { name: ko.settings.catalog.advanced })).toBeTruthy();
    expect(screen.getByRole('textbox', { name: ko.settings.catalog.modelName })).toHaveAttribute(
      'placeholder',
      ko.settings.catalog.typePlaceholder,
    );
    expect(screen.queryByText(en.settings.catalog.advanced)).toBeNull();
    expect(screen.queryByText(fmt(en.settings.catalog.keyRejected, { vendor: 'Google' }))).toBeNull();
  });

  const noKey: CatalogResult = {
    provider: 'gemini',
    status: 'no_key',
    entries: [],
    recommended: null,
    fetched_at: null,
    detail: null,
  };

  const ollamaSaved = (): RuntimeSettings => ({
    ...baseSettings,
    llm_provider: 'ollama',
    llm_base_url: 'http://localhost:11434',
    llm_model: 'qwen3:8b',
    catalog: null,
  });
  const previewCalls = (calls: Array<{ url: string; method: string; body?: unknown }>) =>
    calls.filter((c) => c.url === '/api/models/catalog' && c.method === 'POST');

  // Killed by: frontend/src/components/SettingsModal.tsx :: const needsPreview = Boolean(providerMeta) && currentSettings !== null && !usesSavedCatalog && typedKeyUsable;
  // Becomes: const needsPreview = false;
  it('lists the models of a provider picked but not yet saved (#1657)', async () => {
    const calls = mockFetch({
      '/api/settings': () => jsonResponse(ollamaSaved()),
      '/api/models/catalog': () => jsonResponse({ provider: 'gemini', models: [], catalog: liveCatalog }),
    });
    render(<SettingsModal {...devModeOff} isOpen onClose={() => {}} />);
    await screen.findByPlaceholderText('http://localhost:11434');

    fireEvent.click(screen.getByRole('button', { name: /Gemini/ }));

    const select = (await screen.findByTestId('settings-model-select')) as HTMLSelectElement;
    await waitFor(() => expect(select.value).toBe('model-beta'));
    expect(Array.from(select.options).map((o) => o.value)).toContain('model-alpha');
    // Asked for Gemini, with no key of the form's and not the Ollama endpoint.
    expect(previewCalls(calls).map((c) => c.body)).toEqual([{ provider: 'gemini', refresh: false }]);
  });

  // Killed by: frontend/src/components/SettingsModal.tsx :: api_key: asked.key || undefined,
  // Becomes: api_key: undefined,
  // Killed by: frontend/src/components/SettingsModal.tsx :: {cloudConnectedCount !== null &&
  // Becomes: {false && cloudConnectedCount !== null &&
  it('says a cloud provider is connected once a typed key lists its models', async () => {
    const key = 'AIzaSy' + 'C'.repeat(33);
    mockFetch({
      '/api/settings': () => jsonResponse(ollamaSaved()),
      '/api/models/catalog': (_url, init) => {
        const body = JSON.parse(init?.body as string) as { api_key?: string };
        return jsonResponse({ provider: 'gemini', models: [], catalog: body.api_key ? liveCatalog : noKey });
      },
    });
    render(<SettingsModal {...devModeOff} isOpen onClose={() => {}} />);
    await screen.findByPlaceholderText('http://localhost:11434');
    fireEvent.click(screen.getByRole('button', { name: /Gemini/ }));
    await screen.findByTestId('settings-model-catalog-detail', {}, { timeout: 3000 });
    expect(screen.queryByTestId('settings-llm-connected')).toBeNull();
    fireEvent.change(screen.getByPlaceholderText(/Enter API key/i), { target: { value: key } });

    const line = await screen.findByTestId('settings-llm-connected', {}, { timeout: 3000 });
    // Two chat models; the embedding model is not one to pick.
    expect(line.textContent).toBe('Connected · 2 models available');
  });

  // Killed by: frontend/src/i18n/locales/en/settings.json :: "one": "Connected · {count} model available",
  // Becomes: "one": "Connected · {count} models available",
  it('says a held key is connected, in the singular for one model', async () => {
    mockFetch({
      '/api/settings': () =>
        jsonResponse(geminiSettings({ catalog: { ...liveCatalog, entries: [entry('model-alpha', 'Alpha')] } })),
    });
    render(<SettingsModal {...devModeOff} isOpen onClose={() => {}} />);
    const line = await screen.findByTestId('settings-llm-connected', {}, { timeout: 3000 });
    expect(line.textContent).toBe('Connected · 1 model available');
  });

  // Killed by: frontend/src/components/SettingsModal.tsx :: catalogIsLive && listedModels.length > 0 ? listedModels.length : null
  // Becomes: catalog !== null ? listedModels.length : null
  it('does not say connected when the listing failed, and says why in plain words', async () => {
    const key = 'AIzaSy' + 'D'.repeat(33);
    mockFetch({
      '/api/settings': () => jsonResponse(ollamaSaved()),
      '/api/models/catalog': () => jsonResponse({ provider: 'gemini', models: [], catalog: keyRejected }),
    });
    render(<SettingsModal {...devModeOff} isOpen onClose={() => {}} />);
    await screen.findByPlaceholderText('http://localhost:11434');
    fireEvent.click(screen.getByRole('button', { name: /Gemini/ }));
    fireEvent.change(screen.getByPlaceholderText(/Enter API key/i), { target: { value: key } });

    await waitFor(
      () =>
        expect(screen.getByTestId('settings-model-catalog-detail').textContent).toBe(
          fmt(en.settings.catalog.keyRejected, { vendor: 'Google' }),
        ),
      { timeout: 3000 },
    );
    expectPlain(screen.getByTestId('settings-model-catalog-detail').textContent ?? '');
    expect(screen.queryByTestId('settings-llm-connected')).toBeNull();
  });

  it('does not say connected on a saved failure', async () => {
    mockFetch({
      '/api/settings': () =>
        jsonResponse(geminiSettings({ available_models: [], catalog: { ...keyRejected, status: 'unreachable' } })),
    });
    render(<SettingsModal {...devModeOff} isOpen onClose={() => {}} />);
    const detail = await screen.findByTestId('settings-model-catalog-detail', {}, { timeout: 3000 });
    expect(detail.textContent).toBe(fmt(en.settings.catalog.unreachable, { vendor: 'Google' }));
    expect(screen.queryByTestId('settings-llm-connected')).toBeNull();
  });

  // Killed by: frontend/src/components/SettingsModal.tsx :: catalogIsLive && listedModels.length > 0 ? listedModels.length : null
  // Becomes: catalogIsLive ? listedModels.length : null
  it('does not say connected on a listing with nothing to chat with', async () => {
    mockFetch({
      '/api/settings': () =>
        jsonResponse(geminiSettings({ catalog: { ...liveCatalog, entries: [entry('embed-one', null, false)] } })),
    });
    render(<SettingsModal {...devModeOff} isOpen onClose={() => {}} />);
    await screen.findByTestId('settings-model-catalog-detail', {}, { timeout: 3000 });
    expect(screen.queryByTestId('settings-llm-connected')).toBeNull();
  });

  it('says neither connected nor failed before a key is entered', async () => {
    mockFetch({
      '/api/settings': () => jsonResponse(ollamaSaved()),
      '/api/models/catalog': () => jsonResponse({ provider: 'gemini', models: [], catalog: noKey }),
    });
    render(<SettingsModal {...devModeOff} isOpen onClose={() => {}} />);
    await screen.findByPlaceholderText('http://localhost:11434');
    fireEvent.click(screen.getByRole('button', { name: /Gemini/ }));
    await waitFor(
      () =>
        expect(screen.getByTestId('settings-model-catalog-detail').textContent).toBe(
          fmt(en.settings.catalog.noKey, { vendor: 'Google' }),
        ),
      { timeout: 3000 },
    );
    expect(screen.queryByTestId('settings-llm-connected')).toBeNull();
  });

  it('asks with a key typed into the form, in the body and never the URL', async () => {
    const key = 'AIzaSy' + 'B'.repeat(33);
    const calls = mockFetch({
      '/api/settings': () => jsonResponse(ollamaSaved()),
      '/api/models/catalog': (_url, init) => {
        const body = JSON.parse(init?.body as string) as { api_key?: string };
        return jsonResponse({ provider: 'gemini', models: [], catalog: body.api_key ? liveCatalog : keyRejected });
      },
    });
    render(<SettingsModal {...devModeOff} isOpen onClose={() => {}} />);
    await screen.findByPlaceholderText('http://localhost:11434');
    fireEvent.click(screen.getByRole('button', { name: /Gemini/ }));
    fireEvent.change(screen.getByPlaceholderText(/Enter API key/i), { target: { value: key } });

    const select = (await screen.findByTestId('settings-model-select', {}, { timeout: 3000 })) as HTMLSelectElement;
    await waitFor(() => expect(select.value).toBe('model-beta'));
    const withKey = previewCalls(calls).filter((c) => (c.body as { api_key?: string }).api_key === key);
    expect(withKey.length).toBe(1);
    expect(calls.every((c) => !c.url.includes(key))).toBe(true);
  });

  // Killed by: frontend/src/components/SettingsModal.tsx :: sameEndpoint(formEndpoint, currentSettings.llm_base_url ?? '');
  // Becomes: true;
  it('asks again when the saved provider is pointed at another endpoint (#1663)', async () => {
    const calls = mockFetch({
      '/api/settings': () => jsonResponse(geminiSettings()),
      '/api/models/catalog': () => jsonResponse({ provider: 'gemini', models: [], catalog: keyRejected }),
    });
    render(<SettingsModal {...devModeOff} isOpen onClose={() => {}} />);
    fireEvent.click(await screen.findByRole('button', { name: en.settings.catalog.advanced }));
    fireEvent.change(screen.getByPlaceholderText(en.settings.catalog.endpointPlaceholder), {
      target: { value: 'https://proxy.example/v1beta' },
    });

    await waitFor(
      () =>
        expect(screen.getByTestId('settings-model-catalog-detail').textContent).toBe(
          fmt(en.settings.catalog.keyRejected, { vendor: 'Google' }),
        ),
      { timeout: 3000 },
    );
    expect(previewCalls(calls).map((c) => c.body)).toEqual([
      { provider: 'gemini', base_url: 'https://proxy.example/v1beta', refresh: false },
    ]);
  });

  // Killed by: frontend/src/components/SettingsModal.tsx :: sameEndpoint(formEndpoint, currentSettings.llm_base_url ?? '');
  // Becomes: formEndpoint === (currentSettings.llm_base_url ?? '').trim();
  it('keeps the saved list when the saved endpoint is respelled as another loopback name', async () => {
    const calls = mockFetch({
      '/api/settings': () => jsonResponse(geminiSettings({ llm_base_url: 'http://127.0.0.1:8080/v1beta' })),
      '/api/models/catalog': () => jsonResponse({ provider: 'gemini', models: [], catalog: keyRejected }),
    });
    render(<SettingsModal {...devModeOff} isOpen onClose={() => {}} />);
    const endpoint = (await screen.findByPlaceholderText(
      en.settings.catalog.endpointPlaceholder,
    )) as HTMLInputElement;
    fireEvent.change(endpoint, { target: { value: 'http://localhost:8080/v1beta/' } });
    await new Promise((r) => setTimeout(r, 700));

    expect(previewCalls(calls)).toEqual([]);
    const select = screen.getByTestId('settings-model-select') as HTMLSelectElement;
    expect(within(select).getAllByRole('option').map((o) => (o as HTMLOptionElement).value)).toContain('model-beta');
  });

  // Killed by: frontend/src/components/SettingsModal.tsx :: if (!typedKeyUsable) return fmt(t.catalog.keyMalformed, { vendor });
  // Becomes: if (!typedKeyUsable) return fmt(t.catalog.noKey, { vendor });
  it('says a typed key is not in the right shape, without asking with it', async () => {
    const calls = mockFetch({ '/api/settings': () => jsonResponse(ollamaSaved()) });
    render(<SettingsModal {...devModeOff} isOpen onClose={() => {}} />);
    await screen.findByPlaceholderText('http://localhost:11434');
    fireEvent.click(screen.getByRole('button', { name: /Gemini/ }));
    fireEvent.change(screen.getByPlaceholderText(/Enter API key/i), { target: { value: 'not-a-key' } });

    await waitFor(() =>
      expect(screen.getByTestId('settings-model-catalog-detail').textContent).toBe(
        fmt(en.settings.catalog.keyMalformed, { vendor: 'Google' }),
      ),
    );
    expect(previewCalls(calls).every((c) => !(c.body as { api_key?: string }).api_key)).toBe(true);
  });

  // Killed by: frontend/src/components/SettingsModal.tsx :: if (!asked) {
  // Becomes: if (false) {
  it("says so when refreshing a picked provider's list fails", async () => {
    let answered = 0;
    mockFetch({
      '/api/settings': () => jsonResponse(ollamaSaved()),
      '/api/models/catalog': () =>
        answered++ === 0
          ? jsonResponse({ provider: 'gemini', models: [], catalog: liveCatalog })
          : jsonResponse({ detail: 'boom' }, false, 500),
    });
    render(<SettingsModal {...devModeOff} isOpen onClose={() => {}} />);
    await screen.findByPlaceholderText('http://localhost:11434');
    fireEvent.click(screen.getByRole('button', { name: /Gemini/ }));
    fireEvent.click(await screen.findByRole('button', { name: en.settings.catalog.refresh }));

    expect(await screen.findByText(en.settings.catalog.refreshFailed)).toBeTruthy();
  });

  it("says Google refused the key when the picked provider's listing is refused", async () => {
    mockFetch({
      '/api/settings': () => jsonResponse(ollamaSaved()),
      '/api/models/catalog': () => jsonResponse({ provider: 'gemini', models: [], catalog: keyRejected }),
    });
    render(<SettingsModal {...devModeOff} isOpen onClose={() => {}} />);
    await screen.findByPlaceholderText('http://localhost:11434');
    fireEvent.click(screen.getByRole('button', { name: /Gemini/ }));

    await waitFor(() =>
      expect(screen.getByTestId('settings-model-catalog-detail').textContent).toBe(
        fmt(en.settings.catalog.keyRejected, { vendor: 'Google' }),
      ),
    );
  });
});

describe('SettingsModal Remote GPU Worker', () => {
  beforeEach(() => vi.restoreAllMocks());
  afterEach(() => vi.unstubAllGlobals());

  it('renders remote GPU card and connects successfully', async () => {
    const mockTunnel = {
      host: 'dell',
      connected: true,
      mappings: [
        { service_name: 'comfyui', remote_port: 8188, local_port: 8188 },
        { service_name: 'ollama', remote_port: 11434, local_port: 11434 },
      ],
      gpu: {
        name: 'NVIDIA GeForce RTX 5070 Ti',
        total_mb: 16303,
        used_mb: 1024,
        driver: '610.88',
      },
    };

    const calls = mockFetch({
      '/api/settings/remote-gpu/connect': () =>
        jsonResponse({
          status: 'ok',
          connected: true,
          tunnel: mockTunnel,
          applied_changes: {
            comfyui_base_url: 'http://127.0.0.1:8188',
          },
        }),
    });

    render(<SettingsModal isOpen={true} onClose={() => {}} {...devModeOff} />);

    const card = await screen.findByTestId('settings-remote-gpu-card');
    expect(card).toBeTruthy();

    const hostInput = screen.getByTestId('remote-gpu-host-input') as HTMLInputElement;
    expect(hostInput.value).toBe('dell');

    const connectBtn = screen.getByTestId('remote-gpu-connect-button');
    fireEvent.click(connectBtn);

    await waitFor(() => {
      expect(
        calls.some((c) => c.method === 'POST' && c.url.includes('/api/settings/remote-gpu/connect')),
      ).toBe(true);
    });

    await waitFor(() => {
      expect(card.innerHTML).toContain('NVIDIA GeForce RTX 5070 Ti');
    });
    expect(screen.getByTestId('remote-gpu-disconnect-button')).toBeTruthy();
  });

  it('keeps the LLM local on connect unless the box is ticked', async () => {
    const tunnel = { host: 'dell', connected: true, mappings: [] };
    const calls = mockFetch({
      '/api/settings/remote-gpu/connect': () =>
        jsonResponse({ status: 'ok', connected: true, tunnel, applied_changes: {} }),
    });
    render(<SettingsModal isOpen={true} onClose={() => {}} {...devModeOff} />);
    const box = (await screen.findByTestId('remote-gpu-sync-llm')) as HTMLInputElement;
    expect(box.checked).toBe(false);
    fireEvent.click(screen.getByTestId('remote-gpu-connect-button'));
    await waitFor(() => expect(calls.some((c) => c.url.includes('/remote-gpu/connect'))).toBe(true));
    const first = calls.find((c) => c.url.includes('/remote-gpu/connect'));
    expect((first?.body as { sync_llm: boolean }).sync_llm).toBe(false);
  });

  it('asks for the LLM too when the box is ticked, and says when the model is missing', async () => {
    const tunnel = { host: 'dell', connected: true, mappings: [] };
    const calls = mockFetch({
      '/api/settings/remote-gpu/connect': () =>
        jsonResponse({
          status: 'ok',
          connected: true,
          tunnel,
          applied_changes: {},
          llm_skipped: 'model_not_on_remote',
        }),
    });
    render(<SettingsModal isOpen={true} onClose={() => {}} {...devModeOff} />);
    fireEvent.click(await screen.findByTestId('remote-gpu-sync-llm'));
    fireEvent.click(screen.getByTestId('remote-gpu-connect-button'));
    await waitFor(() => expect(calls.some((c) => c.url.includes('/remote-gpu/connect'))).toBe(true));
    const first = calls.find((c) => c.url.includes('/remote-gpu/connect'));
    expect((first?.body as { sync_llm: boolean }).sync_llm).toBe(true);
    expect(await screen.findByText(new RegExp(en.settings.remoteGpu.llmModelMissing))).toBeTruthy();
  });

  it('shows the addresses a dead tunnel put back, not the saved tunnel ones', async () => {
    // The status route first: '/api/settings' also matches its URL.
    mockFetch({
      '/api/settings/remote-gpu/status': () =>
        jsonResponse({
          host: '',
          connected: false,
          restored_settings: { llm_base_url: 'http://127.0.0.1:11434' },
        }),
      '/api/settings': () =>
        jsonResponse({ ...baseSettings, llm_base_url: 'http://127.0.0.1:11435' }),
    });
    render(<SettingsModal isOpen={true} onClose={() => {}} {...devModeOff} />);
    await waitFor(() =>
      expect(screen.getAllByDisplayValue('http://127.0.0.1:11434').length).toBeGreaterThan(0),
    );
  });

  it('clears the tunnel address when the address it replaced was empty', async () => {
    mockFetch({
      '/api/settings/remote-gpu/status': () =>
        jsonResponse({ host: 'dell', connected: false, restored_settings: { llm_base_url: '' } }),
      '/api/settings': () =>
        jsonResponse({ ...baseSettings, llm_base_url: 'http://127.0.0.1:11435' }),
    });
    render(<SettingsModal isOpen={true} onClose={() => {}} {...devModeOff} />);
    // The host is set in the same step that applies the restore, so once it shows the
    // restore has run.
    await screen.findByDisplayValue('dell');
    expect(screen.queryAllByDisplayValue('http://127.0.0.1:11435')).toHaveLength(0);
  });

  it('disconnects remote GPU tunnel and restores state', async () => {
    const calls = mockFetch({
      '/api/settings/remote-gpu/status': () =>
        jsonResponse({
          host: 'dell',
          connected: true,
          gpu: { name: 'NVIDIA GeForce RTX 5070 Ti', total_mb: 16303, used_mb: 1024, driver: '610.88' },
        }),
      '/api/settings/remote-gpu/disconnect': () =>
        jsonResponse({
          status: 'ok',
          connected: false,
          restored_settings: {
            comfyui_base_url: 'http://127.0.0.1:8188',
          },
        }),
    });

    render(<SettingsModal isOpen={true} onClose={() => {}} {...devModeOff} />);

    const disconnectBtn = await screen.findByTestId('remote-gpu-disconnect-button');
    fireEvent.click(disconnectBtn);

    await waitFor(() => {
      expect(
        calls.some((c) => c.method === 'POST' && c.url.includes('/api/settings/remote-gpu/disconnect')),
      ).toBe(true);
    });

    expect(await screen.findByTestId('remote-gpu-connect-button')).toBeTruthy();
  });
});

/** The usage offer in "All" reads its own report; a save in Settings → Usage must still reach it. */
describe('SettingsModal usage offer', () => {
  beforeEach(() => {
    vi.restoreAllMocks();
    window.localStorage.clear();
  });
  afterEach(() => vi.unstubAllGlobals());

  const usageReport = (limits: Record<string, number | null>) => ({
    checked_at: '2026-09-26T10:00:00Z',
    windows: Object.entries(limits).map(([window, limit]) => ({ window, used: 0, limit, available_again_at: null })),
    limits,
    env_overrides: [],
    providers: [],
  });
  const noLimits = { per_10_minutes: null, per_5_hours: null, per_week: null };
  const light = { per_10_minutes: 150_000, per_5_hours: 500_000, per_week: 3_000_000 };

  it('hides the offer as soon as a limit is saved, without reopening Settings', async () => {
    const calls = mockFetch({
      '/api/settings': (url) =>
        url.includes('remote-gpu')
          ? jsonResponse({ host: '', connected: false })
          : jsonResponse({ ...baseSettings, llm_provider: 'openai' }),
      '/api/usage/limits': () => jsonResponse(usageReport(light)),
      '/api/usage': () => jsonResponse(usageReport(noLimits)),
    });

    render(<SettingsModal {...devModeOff} isOpen onClose={() => {}} />);

    expect(await screen.findByTestId('settings-usage-offer')).toBeTruthy();
    fireEvent.click(await screen.findByTestId('settings-usage-preset-light'));
    fireEvent.click(screen.getByTestId('settings-usage-save'));

    await screen.findByTestId('settings-usage-saved');
    expect(calls.some((c) => c.method === 'PUT' && c.url === '/api/usage/limits')).toBe(true);
    expect(screen.queryByTestId('settings-usage-offer')).toBeNull();
  });
});

describe('SettingsModal keys per provider, chat and fast models', () => {
  beforeEach(() => vi.restoreAllMocks());
  afterEach(() => vi.unstubAllGlobals());

  const entry = (id: string, display_name: string | null) => ({
    id,
    display_name,
    context_window: null,
    max_output_tokens: null,
    chat_capable: true,
    created_at: null,
  });

  const catalog: CatalogResult = {
    provider: 'gemini',
    status: 'live',
    entries: [entry('model-alpha', 'Alpha'), entry('model-beta', 'Beta')],
    recommended: 'model-beta',
    fetched_at: '2026-09-26T00:00:00Z',
    detail: null,
  };

  const keyed = (over: Partial<RuntimeSettings> = {}): RuntimeSettings => ({
    ...baseSettings,
    llm_provider: 'gemini',
    llm_base_url: '',
    llm_model: 'model-beta',
    llm_model_fast: '',
    llm_api_key_set: true,
    llm_api_key_masked: 'AQ....wxyz',
    providers_available: ['ollama', 'vllm', 'openai', 'anthropic', 'gemini'],
    available_models: ['model-alpha', 'model-beta'],
    catalog,
    providers: [
      { id: 'openai', key_set: true, key_masked: 'sk-...1234', key_source: 'settings', key_env_var: null },
      { id: 'anthropic', key_set: false, key_masked: '', key_source: '', key_env_var: null },
      { id: 'gemini', key_set: true, key_masked: 'AQ....wxyz', key_source: 'settings', key_env_var: null },
      { id: 'vllm', key_set: false, key_masked: '', key_source: '', key_env_var: null },
    ],
    ...over,
  });

  type Call = { url: string; method: string; body?: unknown };
  const saveBody = (calls: Call[]) =>
    calls.find((c) => c.url === '/api/settings' && c.method === 'POST')?.body as Record<string, unknown> | undefined;

  // Killed by: frontend/src/components/SettingsModal.tsx :: const keyRow = currentSettings?.providers?.find((row) => row.id === llmProvider);
  // Becomes: const keyRow = currentSettings?.providers?.find((row) => row.id === currentSettings?.llm_provider);
  it("shows each provider's own saved key, and switching provider never asks for it again", async () => {
    mockFetch({ '/api/settings': () => jsonResponse(keyed()) });
    render(<SettingsModal {...devModeOff} isOpen onClose={() => {}} />);

    expect((await screen.findByTestId('api-key-state')).textContent).toBe(
      fmt(en.settings.apiKey.configured, { masked: 'AQ....wxyz' }),
    );

    fireEvent.click(screen.getByRole('button', { name: /OpenAI/ }));

    expect(screen.getByTestId('api-key-state').textContent).toBe(
      fmt(en.settings.apiKey.configured, { masked: 'sk-...1234' }),
    );
    expect(screen.getByPlaceholderText(en.settings.apiKey.keepPlaceholder)).toBeTruthy();

    fireEvent.click(screen.getByRole('button', { name: /Anthropic/ }));
    expect(screen.queryByTestId('api-key-state')).toBeNull();
    expect(screen.queryByTestId('api-key-remove')).toBeNull();
  });

  it('says which environment variable overrides the key, and offers no removal for it', async () => {
    mockFetch({
      '/api/settings': () =>
        jsonResponse(
          keyed({
            providers: [
              {
                id: 'gemini',
                key_set: true,
                key_masked: 'AQ....9999',
                key_source: 'env',
                key_env_var: 'GEMINI_API_KEY',
              },
            ],
          }),
        ),
    });
    render(<SettingsModal {...devModeOff} isOpen onClose={() => {}} />);

    expect((await screen.findByTestId('api-key-state')).textContent).toBe(
      fmt(en.settings.apiKey.fromEnv, { variable: 'GEMINI_API_KEY' }),
    );
    expect(screen.queryByTestId('api-key-remove')).toBeNull();
  });

  it("removes one provider's key and leaves the others", async () => {
    vi.spyOn(window, 'confirm').mockReturnValue(true);
    const after = keyed();
    after.providers = after.providers!.map((row) =>
      row.id === 'openai' ? { ...row, key_set: false, key_masked: '', key_source: '' } : row,
    );
    const calls = mockFetch({
      // Listed first: '/api/settings' would match this URL too.
      '/api/settings/api-keys/': () => jsonResponse(after),
      '/api/settings': () => jsonResponse(keyed()),
    });
    render(<SettingsModal {...devModeOff} isOpen onClose={() => {}} />);
    await screen.findByTestId('api-key-state');
    fireEvent.click(screen.getByRole('button', { name: /OpenAI/ }));

    fireEvent.click(screen.getByTestId('api-key-remove'));

    await waitFor(() => expect(screen.queryByTestId('api-key-state')).toBeNull());
    expect(calls.some((c) => c.method === 'DELETE' && c.url === '/api/settings/api-keys/openai')).toBe(true);
    expect(screen.getByText(fmt(en.settings.apiKey.removed, { name: 'OpenAI' }))).toBeTruthy();
    fireEvent.click(screen.getByRole('button', { name: /Gemini/ }));
    expect(screen.getByTestId('api-key-state').textContent).toBe(
      fmt(en.settings.apiKey.configured, { masked: 'AQ....wxyz' }),
    );
  });

  // Killed by: frontend/src/components/SettingsModal.tsx :: text: code === 'key_in_use' ? t.apiKey.inUse : t.apiKey.unknownProvider,
  // Becomes: text: plainFailure(new CoreFailure(res.status, detail), t.apiKey.removeFailed),
  it('shows a refused removal in Korean rather than the English sentence from the Core', async () => {
    vi.spyOn(window, 'confirm').mockReturnValue(true);
    mockFetch({
      '/api/settings/api-keys/': () =>
        jsonResponse(
          {
            detail:
              'This key is in use. Switch to another provider first, or paste a new key to replace it.',
            code: 'key_in_use',
          },
          false,
          400,
        ),
      '/api/settings': () => jsonResponse(keyed()),
    });
    render(
      <LocaleProvider hints={['ko-KR']}>
        <SettingsModal {...devModeOff} isOpen onClose={() => {}} />
      </LocaleProvider>,
    );
    await screen.findByTestId('api-key-state');

    fireEvent.click(screen.getByTestId('api-key-remove'));

    const shown = await screen.findByText(ko.settings.apiKey.inUse);
    expect(shown.textContent).toBe('사용 중인 키입니다. 먼저 다른 Provider로 바꾸거나, 새 키를 붙여 넣어 교체하십시오.');
    expectPlain(shown.textContent ?? '');
    expect(screen.queryByText(/This key is in use/)).toBeNull();
    expect(screen.getByTestId('api-key-state')).toBeTruthy();
  });

  // Killed by: frontend/src/components/SettingsModal.tsx :: {(currentSettings?.env_overrides ?? []).length > 0 && (
  // Becomes: {false && (
  it('says in Korean which environment variable overrides the choice', async () => {
    mockFetch({
      '/api/settings': () =>
        jsonResponse(
          keyed({
            llm_provider_source: 'env',
            llm_provider_env_var: 'LLM_PROVIDER',
            llm_model_source: 'env',
            llm_model_env_var: 'GEMINI_MODEL',
            env_overrides: [
              { field: 'llm_provider', env_var: 'LLM_PROVIDER', message: 'The environment variable LLM_PROVIDER is set.' },
              { field: 'llm_model', env_var: 'GEMINI_MODEL', message: 'The environment variable GEMINI_MODEL is set.' },
            ],
          }),
        ),
    });
    render(
      <LocaleProvider hints={['ko-KR']}>
        <SettingsModal {...devModeOff} isOpen onClose={() => {}} />
      </LocaleProvider>,
    );

    const notices = await screen.findByTestId('settings-env-overrides');

    expect(within(notices).getByText(fmt(ko.settings.llm.envOverride.provider, { variable: 'LLM_PROVIDER' }))).toBeTruthy();
    expect(within(notices).getByText(fmt(ko.settings.llm.envOverride.model, { variable: 'GEMINI_MODEL' }))).toBeTruthy();
    expect(notices.textContent).toContain('환경 변수 LLM_PROVIDER에 값이 설정되어 있어 여기서 고른 Provider보다 우선합니다.');
    expect(notices.textContent).not.toContain('The environment variable');
  });

  it('shows no override notice when the saved choice is the one in effect', async () => {
    mockFetch({ '/api/settings': () => jsonResponse(keyed({ env_overrides: [] })) });
    render(<SettingsModal {...devModeOff} isOpen onClose={() => {}} />);
    await screen.findByTestId('api-key-state');

    expect(screen.queryByTestId('settings-env-overrides')).toBeNull();
  });

  // Killed by: frontend/src/components/SettingsModal.tsx :: llm_api_key_provider: typedSaveKey ? llmProvider : undefined,
  // Becomes: llm_api_key_provider: undefined,
  it('files a typed key under the provider it was typed for', async () => {
    const calls = mockFetch({ '/api/settings': () => jsonResponse(keyed()) });
    render(<SettingsModal {...devModeOff} isOpen onClose={() => {}} />);
    await screen.findByTestId('api-key-state');
    fireEvent.click(screen.getByRole('button', { name: /OpenAI/ }));

    fireEvent.change(screen.getByPlaceholderText(en.settings.apiKey.keepPlaceholder), {
      target: { value: 'sk-proj-abcdefghijklmnopqrstuvwxyz0123456789' },
    });
    fireEvent.click(screen.getByRole('button', { name: en.settings.footer.save }));

    await waitFor(() => expect(saveBody(calls)?.llm_api_key_provider).toBe('openai'));
    expect(saveBody(calls)?.llm_api_key).toBe('sk-proj-abcdefghijklmnopqrstuvwxyz0123456789');
  });

  it('saves no key provider when no key was typed', async () => {
    const calls = mockFetch({ '/api/settings': () => jsonResponse(keyed()) });
    render(<SettingsModal {...devModeOff} isOpen onClose={() => {}} />);
    await screen.findByTestId('api-key-state');

    fireEvent.click(screen.getByRole('button', { name: en.settings.footer.save }));

    await waitFor(() => expect(saveBody(calls)).toBeDefined());
    expect(saveBody(calls)?.llm_api_key).toBeUndefined();
    expect(saveBody(calls)?.llm_api_key_provider).toBeUndefined();
  });

  // Killed by: frontend/src/components/SettingsModal.tsx :: llm_model_fast: llmModelFast.trim(),
  // Becomes: llm_model_fast: undefined,
  it('offers a fast model that defaults to the chat model, and saves the choice', async () => {
    const calls = mockFetch({ '/api/settings': () => jsonResponse(keyed()) });
    render(<SettingsModal {...devModeOff} isOpen onClose={() => {}} />);

    const fast = (await screen.findByTestId('settings-fast-model-select')) as HTMLSelectElement;
    expect(fast.value).toBe('');
    expect(fast.options[0].textContent).toBe(en.settings.llm.fastSameAsDeep);
    expect(screen.getByText(en.settings.llm.model)).toBeTruthy();

    fireEvent.change(fast, { target: { value: 'model-alpha' } });
    fireEvent.click(screen.getByRole('button', { name: en.settings.footer.save }));

    await waitFor(() => expect(saveBody(calls)?.llm_model_fast).toBe('model-alpha'));
    expect(saveBody(calls)?.llm_model).toBe('model-beta');
  });

  it('sends an empty fast model so a save can go back to following the chat model', async () => {
    const calls = mockFetch({ '/api/settings': () => jsonResponse(keyed({ llm_model_fast: 'model-alpha' })) });
    render(<SettingsModal {...devModeOff} isOpen onClose={() => {}} />);

    const fast = (await screen.findByTestId('settings-fast-model-select')) as HTMLSelectElement;
    await waitFor(() => expect(fast.value).toBe('model-alpha'));
    fireEvent.change(fast, { target: { value: '' } });
    fireEvent.click(screen.getByRole('button', { name: en.settings.footer.save }));

    await waitFor(() => expect(saveBody(calls)?.llm_model_fast).toBe(''));
  });

  it('shows the key state and both models in Korean', async () => {
    window.localStorage.clear();
    mockFetch({
      '/api/settings': () =>
        jsonResponse(
          keyed({
            ui_language: 'ko',
            providers: [
              {
                id: 'gemini',
                key_set: true,
                key_masked: 'AQ....9999',
                key_source: 'env',
                key_env_var: 'GEMINI_API_KEY',
              },
            ],
          }),
        ),
    });
    render(
      <LocaleProvider hints={['ko-KR']}>
        <SettingsModal {...devModeOff} isOpen onClose={() => {}} />
      </LocaleProvider>,
    );

    expect((await screen.findByTestId('api-key-state')).textContent).toBe(
      '환경 변수 GEMINI_API_KEY 적용 중',
    );
    expect(screen.getByText('대화 모델')).toBeTruthy();
    const fast = screen.getByTestId('settings-fast-model-select') as HTMLSelectElement;
    expect(fast.options[0].textContent).toBe('대화 모델과 같음');
    expect(screen.getByText(ko.settings.llm.fastModel)).toBeTruthy();
  });
});

describe('SettingsModal provider and model follow-ups (#1657, #1672)', () => {
  beforeEach(() => vi.restoreAllMocks());
  afterEach(() => vi.unstubAllGlobals());

  const entry = (id: string, display_name: string | null) => ({
    id,
    display_name,
    context_window: null,
    max_output_tokens: null,
    chat_capable: true,
    created_at: null,
  });
  // Ids of no real provider.
  const liveCatalog: CatalogResult = {
    provider: 'gemini',
    status: 'live',
    entries: [entry('model-alpha', 'Alpha'), entry('model-beta', 'Beta')],
    recommended: 'model-beta',
    fetched_at: '2026-09-27T00:00:00Z',
    detail: null,
  };
  const geminiSaved = (llm_model: string): RuntimeSettings => ({
    ...baseSettings,
    llm_provider: 'gemini',
    llm_base_url: '',
    llm_model,
    llm_api_key_set: true,
    llm_api_key_masked: 'AIza...wxyz',
    providers_available: ['ollama', 'gemini'],
    available_models: ['model-alpha', 'model-beta'],
    catalog: liveCatalog,
  });
  const vllmSaved: RuntimeSettings = {
    ...baseSettings,
    llm_provider: 'vllm',
    llm_base_url: 'http://box:8000/v1',
    llm_model: 'served-model',
    providers_available: ['ollama', 'vllm'],
    available_models: [],
    catalog: null,
  };
  const saveBody = (calls: Array<{ url: string; method: string; body?: unknown }>) =>
    calls.find((c) => c.url === '/api/settings' && c.method === 'POST')?.body as { llm_model?: string } | undefined;

  // Killed by: frontend/src/components/SettingsModal.tsx :: if (remembered) {
  // Becomes: if (false) {
  it('gives the saved model back when the user looks at another provider and returns', async () => {
    const calls = mockFetch({
      '/api/settings': () => jsonResponse(geminiSaved('model-alpha')),
      '/api/models/catalog': () =>
        jsonResponse({ provider: 'ollama', models: ['qwen3:8b'], reachable: true, catalog: null }),
    });
    render(<SettingsModal {...devModeOff} isOpen onClose={() => {}} />);
    const select = (await screen.findByTestId('settings-model-select')) as HTMLSelectElement;
    expect(select.value).toBe('model-alpha');

    fireEvent.click(screen.getByRole('button', { name: /Ollama/ }));
    await screen.findByPlaceholderText('http://localhost:11434');
    fireEvent.click(screen.getByRole('button', { name: /Gemini/ }));

    const back = (await screen.findByTestId('settings-model-select')) as HTMLSelectElement;
    await waitFor(() => expect(back.value).toBe('model-alpha'));
    fireEvent.click(screen.getByRole('button', { name: en.settings.footer.save }));
    await waitFor(() => expect(saveBody(calls)?.llm_model).toBe('model-alpha'));
  });

  // Killed by: frontend/src/components/SettingsModal.tsx :: typedModel === (currentSettings?.llm_model ?? '').trim() &&
  // Becomes: false &&
  it('says the saved model is no longer offered, and switches only when asked', async () => {
    const calls = mockFetch({ '/api/settings': () => jsonResponse(geminiSaved('model-old')) });
    render(<SettingsModal {...devModeOff} isOpen onClose={() => {}} />);

    const notice = await screen.findByTestId('settings-model-retired');
    const sentence = fmt(en.settings.catalog.retired, {
      model: 'model-old',
      vendor: 'Google',
      recommended: 'model-beta',
    });
    expect(notice.textContent).toBe(`${sentence}${en.settings.catalog.switchModel}`);
    expectPlain(sentence);
    expect(sentence).not.toMatch(/404|not_found|catalog/i);
    expect(screen.queryByTestId('settings-model-unlisted')).toBeNull();
    // Not switched by itself: the saved choice is still what the form holds.
    expect((screen.getByRole('textbox', { name: en.settings.catalog.modelName }) as HTMLInputElement).value).toBe(
      'model-old',
    );

    fireEvent.click(screen.getByRole('button', { name: en.settings.catalog.switchModel }));

    expect((screen.getByTestId('settings-model-select') as HTMLSelectElement).value).toBe('model-beta');
    expect(screen.queryByTestId('settings-model-retired')).toBeNull();
    fireEvent.click(screen.getByRole('button', { name: en.settings.footer.save }));
    await waitFor(() => expect(saveBody(calls)?.llm_model).toBe('model-beta'));
  });

  // Killed by: frontend/src/components/SettingsModal.tsx :: typedModel === (currentSettings?.llm_model ?? '').trim() &&
  // Becomes: false &&
  it('says the saved model is no longer offered in Korean too', async () => {
    mockFetch({ '/api/settings': () => jsonResponse(geminiSaved('model-old')) });
    render(
      <LocaleProvider hints={['ko-KR']}>
        <SettingsModal {...devModeOff} isOpen onClose={() => {}} />
      </LocaleProvider>,
    );
    const notice = await screen.findByTestId('settings-model-retired');
    expect(notice.textContent).toBe(
      `${fmt(ko.settings.catalog.retired, { model: 'model-old', vendor: 'Google', recommended: 'model-beta' })}${ko.settings.catalog.switchModel}`,
    );
  });

  // Killed by: frontend/src/components/SettingsModal.tsx :: const localKeyRefused = needsLocalListing && localMatches && localListing.keyRefused === true && detectedModels.length === 0;
  // Becomes: const localKeyRefused = false;
  it('says the vLLM server refused the key, not that no list came back', async () => {
    mockFetch({
      '/api/settings': () => jsonResponse(vllmSaved),
      '/api/models/catalog': () =>
        jsonResponse({ provider: 'vllm', models: [], reachable: true, key_refused: true, catalog: null }),
    });
    render(<SettingsModal {...devModeOff} isOpen onClose={() => {}} />);

    const note = await screen.findByTestId('settings-local-key-refused');
    expect(note.textContent).toBe(fmt(en.settings.llm.vllmKeyRefused, { endpoint: 'http://box:8000/v1' }));
    expect(note.textContent).not.toMatch(/401|403|HTTP|Unauthorized/);
    expectPlain(en.settings.llm.vllmKeyRefused.replace('{endpoint}', 'the address'));
    expect(screen.queryByTestId('settings-local-not-answering')).toBeNull();
  });

  // Killed by: frontend/src/components/SettingsModal.tsx :: const localKeyRefused = needsLocalListing && localMatches && localListing.keyRefused === true && detectedModels.length === 0;
  // Becomes: const localKeyRefused = false;
  it('says the vLLM server refused the key in Korean too', async () => {
    mockFetch({
      '/api/settings': () => jsonResponse(vllmSaved),
      '/api/models/catalog': () =>
        jsonResponse({ provider: 'vllm', models: [], reachable: true, key_refused: true, catalog: null }),
    });
    render(
      <LocaleProvider hints={['ko-KR']}>
        <SettingsModal {...devModeOff} isOpen onClose={() => {}} />
      </LocaleProvider>,
    );
    const note = await screen.findByTestId('settings-local-key-refused');
    expect(note.textContent).toBe(fmt(ko.settings.llm.vllmKeyRefused, { endpoint: 'http://box:8000/v1' }));
  });

  // Killed by: frontend/src/components/SettingsModal.tsx :: : testResults.llm.key_refused
  // Becomes: : false
  it('words a refused key on Check connection instead of showing the status code', async () => {
    mockFetch({
      '/api/settings/test': () =>
        jsonResponse({
          results: {
            llm: { status: 'error', provider: 'vllm', error: 'vLLM HTTP 401', key_refused: true },
          },
        }),
      '/api/settings': () => jsonResponse(vllmSaved),
      '/api/models/catalog': () =>
        jsonResponse({ provider: 'vllm', models: [], reachable: true, key_refused: true, catalog: null }),
    });
    render(<SettingsModal {...devModeOff} isOpen onClose={() => {}} />);
    await screen.findByTestId('settings-local-key-refused');

    fireEvent.click(screen.getByRole('button', { name: /check connection/i }));

    expect(await screen.findByText(en.settings.connectivity.keyRefused)).toBeTruthy();
    expectPlain(en.settings.connectivity.keyRefused);
    expect(screen.queryByText('vLLM HTTP 401')).toBeNull();
  });
});
