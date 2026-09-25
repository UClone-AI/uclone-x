import { describe, it, expect, vi, beforeEach, afterEach } from 'vitest';
import { fireEvent, render, screen, waitFor, within } from '@testing-library/react';
import { SettingsModal } from './SettingsModal';
import type { RuntimeSettings } from '../types';
import { expectPlain } from '../test/plainCopy';
import { SETTINGS_FAILURE } from '../lib/settingsCopy';
import { PERSONA_EDITOR_COPY } from '../lib/personaCopy';

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
    expect(calls.filter((c) => c.method === 'GET' && c.url.includes('/api/settings')).length).toBe(
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
    // Killed by: frontend/src/components/SettingsModal.tsx :: text: data.joined
    // Becomes: text: false
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

  const selectVllm = async () => {
    mockFetch({ '/api/settings': () => jsonResponse(openAiSettings) });
    render(<SettingsModal {...devModeOff} isOpen onClose={() => {}} />);
    const endpoint = (await screen.findByPlaceholderText(
      'https://api.openai.com/v1',
    )) as HTMLInputElement;
    const model = screen.getByPlaceholderText(/qwen3:8b, hermes3:8b, gpt-4o/i) as HTMLInputElement;
    expect(endpoint.value).toBe('https://api.openai.com/v1');
    expect(model.value).toBe('gpt-4o');

    fireEvent.click(screen.getByRole('button', { name: /vLLM/ }));
    return { endpoint, model };
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
    const endpoint = (await screen.findByPlaceholderText(
      'https://api.openai.com/v1',
    )) as HTMLInputElement;
    const model = screen.getByPlaceholderText(/qwen3:8b, hermes3:8b, gpt-4o/i) as HTMLInputElement;
    expect(endpoint.value).toBe('http://10.0.0.7:11500/v1');
    expect(model.value).toBe('gpt-4o-mini');

    fireEvent.click(screen.getByRole('button', { name: /vLLM/ }));

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
    const model = screen.getByPlaceholderText(/qwen3:8b, hermes3:8b, gpt-4o/i) as HTMLInputElement;

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

  // Killed by: frontend/src/components/SettingsModal.tsx :: { id: 'vllm', label: 'vLLM', desc: 'Self-hosted GPU server' },
  // Becomes: { id: 'vllm', label: 'Self-hosted', desc: 'Self-hosted GPU server' },
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
    const endpoint = (await screen.findByPlaceholderText(
      'https://api.openai.com/v1',
    )) as HTMLInputElement;

    fireEvent.click(screen.getByRole('button', { name: /Ollama/ }));

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
    // Killed by: frontend/src/components/settings/SkillsSection.tsx :: UClone-X could not be reached. If
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
    // Killed by: frontend/src/components/settings/SkillsSection.tsx :: fault.detail ?? PLAIN_CAUSE[fault.kind]
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
    // Killed by: frontend/src/components/SkillsTab.tsx :: Passed the safety check
    // Becomes: AST Verification Passed
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
    // Killed by: frontend/src/components/SettingsModal.tsx :: plainFailure(err, SETTINGS_FAILURE.load)
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
    // Killed by: frontend/src/components/SettingsModal.tsx :: plainFailure(err, SETTINGS_FAILURE.test)
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
    // Killed by: frontend/src/components/SettingsModal.tsx :: plainFailure(err, SETTINGS_FAILURE.save)
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
    // Killed by: frontend/src/components/SettingsModal.tsx :: plainFailure(err, SETTINGS_FAILURE.install)
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
    // Killed by: frontend/src/components/SettingsModal.tsx :: plainFailure(err, SETTINGS_FAILURE.remove)
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
    expect(alert).toHaveTextContent(PERSONA_EDITOR_COPY.loadFailed(''));
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
    expect(alert).toHaveTextContent(PERSONA_EDITOR_COPY.loadFailed('The clones folder is missing.'));
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
    expect(validBadge.textContent).toContain('올바른 Google Gemini 키 형식입니다.');
  });

  it('clears localhost:11434 when switching from Ollama to Gemini and offers curated model pills', async () => {
    mockFetch({
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

    expect(endpoint.value).toBe('');
    const pills = await screen.findByTestId('curated-models-pills');
    expect(pills.textContent).toContain('gemini-1.5-pro');
    expect(pills.textContent).toContain('gemini-1.5-flash');
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

    await screen.findByText('Configured: sk-...1234');

    fireEvent.click(screen.getByRole('button', { name: /Gemini/ }));

    expect(screen.queryByText('Configured: sk-...1234')).toBeNull();
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
});


