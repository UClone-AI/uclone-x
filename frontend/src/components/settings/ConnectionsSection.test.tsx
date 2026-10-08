import { afterEach, describe, expect, it, vi } from 'vitest';
import { fireEvent, render, screen, waitFor, within } from '@testing-library/react';
import { ConnectionsSection } from './ConnectionsSection';
import { LocaleProvider } from '../../i18n';
import { en } from '../../i18n/en';
import { ko } from '../../i18n/ko';
import { fmt } from '../../i18n/format';
import { expectPlain } from '../../test/plainCopy';
import type { Connection, ConnectionKind } from '../../lib/modelGateway';

const T = en.gateway;

const row = (over: Partial<Connection>): Connection => ({
  id: 'x',
  kind: 'ollama',
  label: 'X',
  base_url: null,
  key_set: false,
  key_masked: null,
  source: 'settings',
  env_var: null,
  paid: false,
  status: 'connected',
  detail: null,
  model_count: null,
  ...over,
});

const ROWS: Connection[] = [
  row({ id: 'gemini', kind: 'gemini', label: 'Google Gemini', key_set: true, key_masked: 'AIza…f00d', paid: true, model_count: 12 }),
  row({ id: 'ollama', kind: 'ollama', label: 'Ollama', base_url: 'http://127.0.0.1:11434', model_count: 1 }),
  row({ id: 'gpu-box', kind: 'vllm', label: 'GPU box', base_url: 'http://10.0.0.5:8000', status: 'unreachable', detail: 'Connection refused' }),
  row({ id: 'openai', kind: 'openai', label: 'OpenAI', paid: true, status: 'key_rejected', detail: '401 Unauthorized' }),
  row({ id: 'anthropic', kind: 'anthropic', label: 'Anthropic', source: 'env', env_var: 'ANTHROPIC_API_KEY', paid: true, status: 'no_key' }),
];

const KINDS: ConnectionKind[] = [
  { kind: 'gemini', label: 'Google Gemini', needs_key: true, needs_base_url: false, default_base_url: null, key_url: 'https://aistudio.google.com/apikey', capabilities: ['chat'] },
  { kind: 'ollama', label: 'Ollama', needs_key: false, needs_base_url: true, default_base_url: 'http://127.0.0.1:11434', key_url: null, capabilities: ['chat'] },
];

type Answer = { ok: boolean; status: number; json: () => Promise<unknown> };
const json = (body: unknown, status = 200): Answer => ({ ok: status < 400, status, json: async () => body });

const serve = (routes: Record<string, (init: RequestInit) => Answer | Promise<Answer>>) => {
  const calls: { url: string; method: string; body?: unknown }[] = [];
  vi.stubGlobal(
    'fetch',
    vi.fn(async (url: string, init: RequestInit = {}) => {
      const method = init.method ?? 'GET';
      calls.push({ url, method, body: init.body ? JSON.parse(String(init.body)) : undefined });
      const route = routes[`${method} ${url}`];
      if (route) return route(init);
      if (method === 'GET' && url === '/api/connections') return json({ connections: ROWS, kinds: KINDS });
      return json({}, 404);
    }),
  );
  return calls;
};

afterEach(() => vi.unstubAllGlobals());

describe('ConnectionsSection', () => {
  // Killed by: frontend/src/components/settings/gatewayWords.ts :: return count === null ? copy.status.connected : plural(copy.status.connectedCount, count);
  // Becomes: return copy.status.connected;
  it('shows one row per connection with its status in words', async () => {
    serve({});
    render(<ConnectionsSection />);

    expect(await screen.findByTestId('connection-status-gemini')).toHaveTextContent('Connected · 12 models');
    expect(screen.getByTestId('connection-status-ollama')).toHaveTextContent('Connected · 1 model');
    expect(screen.getByTestId('connection-status-gpu-box')).toHaveTextContent(T.status.unreachable);
    expect(screen.getByTestId('connection-status-openai')).toHaveTextContent(T.status.key_rejected);
    expect(screen.getByTestId('connection-status-anthropic')).toHaveTextContent(T.status.no_key);
    // The kind's own name, and never the Core's English detail or a status code.
    expect(screen.getByTestId('connection-row-gemini')).toHaveTextContent('Google Gemini');
    const section = screen.getByTestId('settings-connections');
    expect(section).not.toHaveTextContent('Connection refused');
    expect(section).not.toHaveTextContent('401');
  });

  it('says there is no connection yet instead of drawing an empty list', async () => {
    serve({ 'GET /api/connections': () => json({ connections: [], kinds: KINDS }) });
    render(<ConnectionsSection />);
    expect(await screen.findByTestId('settings-connections-empty')).toHaveTextContent(T.connections.empty);
  });

  it('shows a connection set by an environment variable read-only, naming the variable', async () => {
    serve({});
    render(<ConnectionsSection />);

    const env = await screen.findByTestId('connection-env-anthropic');
    expect(env).toHaveTextContent(fmt(T.connections.fromEnv, { variable: 'ANTHROPIC_API_KEY' }));
    expect(screen.queryByTestId('connection-remove-anthropic')).toBeNull();
    expect(screen.queryByTestId('connection-edit-anthropic')).toBeNull();
    // It can still be checked: checking changes nothing.
    expect(screen.getByTestId('connection-check-anthropic')).toBeTruthy();
  });

  it('checks a connection and shows what the check found', async () => {
    const onChanged = vi.fn();
    const calls = serve({
      'POST /api/connections/gpu-box/check': () =>
        json(row({ id: 'gpu-box', kind: 'vllm', label: 'GPU box', base_url: 'http://10.0.0.5:8000', model_count: 3 })),
    });
    render(<ConnectionsSection onChanged={onChanged} />);

    fireEvent.click(await screen.findByTestId('connection-check-gpu-box'));

    await waitFor(() => expect(screen.getByTestId('connection-status-gpu-box')).toHaveTextContent('Connected · 3 models'));
    expect(calls.some((c) => c.method === 'POST' && c.url === '/api/connections/gpu-box/check')).toBe(true);
    expect(onChanged).toHaveBeenCalled();
  });

  // Killed by: frontend/src/components/settings/ConnectionsSection.tsx :: const dependents = await fetchDependents(row.id);
  // Becomes: const dependents = { clones: [], defaults: [] };
  it('lists the clones and defaults that use a connection before removing it, and removes only on confirm', async () => {
    const onChanged = vi.fn();
    const calls = serve({
      'GET /api/connections/gemini/dependents': () =>
        json({ clones: [{ id: 'agt_1', name: 'Writer', slots: ['model_name', 'image_model'] }], defaults: ['deep'] }),
      'DELETE /api/connections/gemini': () => json({ removed: 'gemini' }),
    });
    render(<ConnectionsSection onChanged={onChanged} />);

    fireEvent.click(await screen.findByTestId('connection-remove-gemini'));

    const confirm = await screen.findByTestId('connection-remove-confirm-gemini');
    expect(within(confirm).getByTestId('connection-remove-clones-gemini')).toHaveTextContent(
      'Writer (conversation model, picture model)',
    );
    expect(within(confirm).getByTestId('connection-remove-defaults-gemini')).toHaveTextContent(T.defaults.deep);
    expect(calls.some((c) => c.method === 'DELETE')).toBe(false);

    fireEvent.click(within(confirm).getByTestId('connection-remove-yes-gemini'));

    await waitFor(() => expect(screen.queryByTestId('connection-row-gemini')).toBeNull());
    expect(calls.filter((c) => c.method === 'DELETE').map((c) => c.url)).toEqual(['/api/connections/gemini']);
    expect(screen.getByTestId('settings-connections-notice')).toHaveTextContent(
      fmt(T.remove.removed, { name: 'Google Gemini' }),
    );
    expect(onChanged).toHaveBeenCalled();
  });

  it('keeps the connection when Keep it is pressed, sending no removal', async () => {
    const calls = serve({
      'GET /api/connections/ollama/dependents': () => json({ clones: [], defaults: [] }),
    });
    render(<ConnectionsSection />);

    fireEvent.click(await screen.findByTestId('connection-remove-ollama'));
    const confirm = await screen.findByTestId('connection-remove-confirm-ollama');
    expect(confirm).toHaveTextContent(T.remove.unused);
    fireEvent.click(within(confirm).getByRole('button', { name: T.remove.keep }));

    expect(screen.queryByTestId('connection-remove-confirm-ollama')).toBeNull();
    expect(screen.getByTestId('connection-row-ollama')).toBeTruthy();
    expect(calls.some((c) => c.method === 'DELETE')).toBe(false);
  });

  // Killed by: frontend/src/components/settings/ConnectionsSection.tsx :: setNotice({ tone: 'error', text: t.remove.readFailed });
  // Becomes: setRemoval({ id: row.id, stage: 'asking', dependents: { clones: [], defaults: [] } });
  it('does not remove a connection when it cannot find out what uses it', async () => {
    vi.spyOn(console, 'error').mockImplementation(() => {});
    const calls = serve({
      'GET /api/connections/ollama/dependents': () => Promise.reject(new TypeError('Failed to fetch')),
    });
    render(<ConnectionsSection />);

    fireEvent.click(await screen.findByTestId('connection-remove-ollama'));

    const notice = await screen.findByTestId('settings-connections-notice');
    expect(notice).toHaveTextContent(T.remove.readFailed);
    expectPlain(notice.textContent);
    expect(screen.queryByTestId('connection-remove-confirm-ollama')).toBeNull();
    expect(calls.some((c) => c.method === 'DELETE')).toBe(false);
  });

  // Killed by: frontend/src/components/settings/ConnectionsSection.tsx :: {kind.needs_base_url && (
  // Becomes: {true && (
  it('asks a cloud kind for its key only, never an address', async () => {
    const calls = serve({
      'POST /api/connections': () =>
        json(row({ id: 'gemini-2', kind: 'gemini', label: 'Work Gemini', key_set: true, key_masked: 'AIza…beef', model_count: 9 })),
    });
    render(<ConnectionsSection />);

    fireEvent.click(await screen.findByTestId('connection-add-open'));
    fireEvent.change(screen.getByTestId('connection-add-kind'), { target: { value: 'gemini' } });

    expect(screen.queryByTestId('connection-add-address')).toBeNull();
    expect(screen.getByTestId('connection-add-key-link')).toHaveAttribute('href', 'https://aistudio.google.com/apikey');
    expect(screen.getByTestId('connection-add-submit')).toBeDisabled();

    fireEvent.change(screen.getByLabelText(T.add.name), { target: { value: 'Work Gemini' } });
    fireEvent.change(screen.getByTestId('connection-add-key'), { target: { value: '  "AIza-secret"  ' } });
    fireEvent.click(screen.getByTestId('connection-add-submit'));

    await screen.findByTestId('connection-row-gemini-2');
    const post = calls.find((c) => c.method === 'POST' && c.url === '/api/connections');
    // The key travels in the body, cleaned of the quotes a paste brings; never in the URL.
    expect(post?.body).toEqual({ kind: 'gemini', label: 'Work Gemini', key: 'AIza-secret' });
    expect(screen.getByTestId('settings-connections-notice')).toHaveTextContent('Work Gemini was added: Connected · 9 models.');
  });

  // Killed by: frontend/src/components/settings/ConnectionsSection.tsx :: {kind.needs_key && (
  // Becomes: {true && (
  it('asks a local kind for its address only, starting on its usual one, never a key', async () => {
    const calls = serve({
      'POST /api/connections': () =>
        json(row({ id: 'ollama-2', kind: 'ollama', label: 'Ollama', base_url: 'http://192.168.0.7:11434', status: 'unreachable' })),
    });
    render(<ConnectionsSection />);

    fireEvent.click(await screen.findByTestId('connection-add-open'));
    fireEvent.change(screen.getByTestId('connection-add-kind'), { target: { value: 'ollama' } });

    expect(screen.queryByTestId('connection-add-key')).toBeNull();
    const address = screen.getByTestId('connection-add-address');
    expect(address).toHaveValue('http://127.0.0.1:11434');
    fireEvent.change(address, { target: { value: 'http://192.168.0.7:11434' } });
    fireEvent.click(screen.getByTestId('connection-add-submit'));

    await screen.findByTestId('connection-row-ollama-2');
    expect(calls.find((c) => c.method === 'POST')?.body).toEqual({ kind: 'ollama', base_url: 'http://192.168.0.7:11434' });
    // Added but not answering: said so, in words.
    expect(screen.getByTestId('connection-status-ollama-2')).toHaveTextContent(T.status.unreachable);
  });

  it("shows a refused add in the Core's own plain words, or a fixed sentence when it gives none", async () => {
    vi.spyOn(console, 'error').mockImplementation(() => {});
    serve({ 'POST /api/connections': () => ({ ok: false, status: 500, json: async () => { throw new SyntaxError('bad'); } }) });
    render(<ConnectionsSection />);

    fireEvent.click(await screen.findByTestId('connection-add-open'));
    fireEvent.change(screen.getByTestId('connection-add-kind'), { target: { value: 'ollama' } });
    fireEvent.click(screen.getByTestId('connection-add-submit'));

    const error = await screen.findByTestId('connection-add-error');
    expect(error).toHaveTextContent(T.add.failed);
    expectPlain(error.textContent);
  });

  it('replaces a saved key with Change, sending only the key', async () => {
    const calls = serve({
      'PATCH /api/connections/openai': () =>
        json(row({ id: 'openai', kind: 'openai', label: 'OpenAI', paid: true, key_set: true, key_masked: 'sk-…9999', model_count: 40 })),
    });
    render(<ConnectionsSection />);

    expect(await screen.findByTestId('connection-row-openai')).toHaveTextContent(T.connections.keyRejectedHint);
    fireEvent.click(screen.getByTestId('connection-edit-openai'));
    fireEvent.change(screen.getByLabelText(T.connections.newKey), { target: { value: 'sk-new' } });
    fireEvent.click(screen.getByTestId('connection-edit-save-openai'));

    await waitFor(() => expect(screen.getByTestId('connection-status-openai')).toHaveTextContent('Connected · 40 models'));
    expect(calls.find((c) => c.method === 'PATCH')?.body).toEqual({ key: 'sk-new' });
  });

  it('reads in Korean, with the Core detail kept off the screen', async () => {
    serve({});
    render(
      <LocaleProvider hints={['ko-KR']}>
        <ConnectionsSection />
      </LocaleProvider>,
    );

    expect(await screen.findByTestId('connection-status-gpu-box')).toHaveTextContent(ko.gateway.status.unreachable);
    expect(screen.getByTestId('connection-status-gemini')).toHaveTextContent('연결됨 · 모델 12개');
    expect(screen.getByTestId('connection-env-anthropic')).toHaveTextContent(
      fmt(ko.gateway.connections.fromEnv, { variable: 'ANTHROPIC_API_KEY' }),
    );
    expect(screen.getByText(ko.gateway.connections.title)).toBeTruthy();
  });

  // Killed by: frontend/src/components/settings/ConnectionsSection.tsx :: const unsupported = row.status === 'unsupported';
  // Becomes: const unsupported = false;
  it('lists a connection of a kind this version does not know, saying so, removable but not changeable', async () => {
    serve({
      'GET /api/connections': () =>
        json({
          connections: [row({ id: 'future', kind: 'martian-llm', label: 'future', status: 'unsupported', detail: 'raw detail' })],
          kinds: KINDS,
        }),
    });
    render(<ConnectionsSection />);

    expect(await screen.findByTestId('connection-status-future')).toHaveTextContent(T.status.unsupported);
    expect(screen.getByTestId('connection-hint-future')).toHaveTextContent(T.connections.unsupportedHint);
    expect(screen.getByTestId('connection-remove-future')).toBeTruthy();
    expect(screen.queryByTestId('connection-edit-future')).toBeNull();
    expect(screen.getByTestId('settings-connections')).not.toHaveTextContent('raw detail');
  });

  // Killed by: frontend/src/components/settings/ConnectionsSection.tsx :: {row.key_env_var && !fromEnv && (
  // Becomes: {false && (
  it('says when the key of a saved row comes from an environment variable', async () => {
    serve({
      'GET /api/connections': () =>
        json({
          connections: [row({ id: 'gemini', kind: 'gemini', label: 'Google Gemini', key_set: true, key_masked: 'AQ....f00d', key_env_var: 'GEMINI_API_KEY' })],
          kinds: KINDS,
        }),
    });
    render(<ConnectionsSection />);

    expect(await screen.findByTestId('connection-key-env-gemini')).toHaveTextContent(
      fmt(T.connections.keyFromEnv, { variable: 'GEMINI_API_KEY' }),
    );
    // The row is still a saved one: it can be changed and removed.
    expect(screen.getByTestId('connection-edit-gemini')).toBeTruthy();
  });

  // Killed by: frontend/src/components/settings/ConnectionsSection.tsx :: }, [load, version]);
  // Becomes: }, [load]);
  it('reads the connections again when something else on the screen changed them', async () => {
    const calls = serve({});
    const { rerender } = render(<ConnectionsSection version={0} />);
    await screen.findByTestId('connection-row-gemini');
    const reads = () => calls.filter((c) => c.method === 'GET' && c.url === '/api/connections').length;
    expect(reads()).toBe(1);
    rerender(<ConnectionsSection version={1} />);
    await waitFor(() => expect(reads()).toBe(2));
  });
});

// #2167 part 3: install and remove models per Ollama connection, with progress and failures
// in plain words built from the answer's status, never the Core's English detail.
describe('ConnectionsSection Ollama models', () => {
  const O = T.ollama;
  const INSTALLED = '/api/models/installed?connection_id=ollama';
  const installedRoute = (models: { id: string; chat: boolean }[]) => () => json({ connection_id: 'ollama', models });
  const openPanel = async () => {
    fireEvent.click(await screen.findByTestId('ollama-models-open-ollama'));
    return screen.findByTestId('ollama-models-ollama');
  };
  const typeAndInstall = (model: string) => {
    fireEvent.change(screen.getByTestId('ollama-install-input-ollama'), { target: { value: model } });
    fireEvent.click(screen.getByTestId('ollama-install-ollama'));
  };

  // Killed by: frontend/src/components/settings/ConnectionsSection.tsx ::                 {row.kind === 'ollama' && (row.status === 'connected' || row.status === 'unchecked') && (
  // Becomes:                 {(row.status === 'connected' || row.status === 'unchecked') && (
  // Killed by: frontend/src/components/settings/OllamaModels.tsx ::                   {!m.chat && <div className="text-slate-500">{t.notChat}</div>}
  // Becomes:                   {false && <div className="text-slate-500">{t.notChat}</div>}
  it('offers the controls on a reachable Ollama only, and lists what it has, embedders said to be one', async () => {
    const calls = serve({
      [`GET ${INSTALLED}`]: installedRoute([
        { id: 'qwen3:14b', chat: true },
        { id: 'bge-m3:latest', chat: false },
      ]),
    });
    render(<ConnectionsSection />);
    await openPanel();

    expect(screen.queryByTestId('ollama-models-open-gemini')).toBeNull();
    expect(screen.queryByTestId('ollama-models-open-gpu-box')).toBeNull();
    const chat = await screen.findByTestId('ollama-model-ollama-qwen3:14b');
    expect(chat).not.toHaveTextContent(O.notChat);
    expect(screen.getByTestId('ollama-model-ollama-bge-m3:latest')).toHaveTextContent(O.notChat);
    expect(calls.some((c) => c.url === INSTALLED)).toBe(true);
  });

  // Killed by: frontend/src/lib/modelGateway.ts ::     await fetch('/api/models/pull', { ...jsonInit('POST', { model, connection_id: id }), signal }),
  // Becomes:     await fetch('/api/models/pull', { ...jsonInit('POST', { model }), signal }),
  // Killed by: frontend/src/components/settings/ConnectionsSection.tsx ::       replaceRow(recounted);
  // Becomes:       void recounted;
  it('installs on this connection, says it is waiting while it runs, then counts the new model', async () => {
    let finish: (a: Answer) => void = () => {};
    let installed = [{ id: 'qwen3:14b', chat: true }];
    const onChanged = vi.fn();
    const calls = serve({
      [`GET ${INSTALLED}`]: () => installedRoute(installed)(),
      'POST /api/models/pull': () => new Promise<Answer>((resolve) => (finish = resolve)),
      'POST /api/connections/ollama/check': () => json({ ...ROWS[1], model_count: 2 }),
    });
    render(<ConnectionsSection onChanged={onChanged} />);
    await openPanel();
    await screen.findByTestId('ollama-model-ollama-qwen3:14b');
    typeAndInstall('gemma3:4b');

    expect(await screen.findByTestId('ollama-installing-ollama')).toHaveTextContent(
      fmt(O.installing, { model: 'gemma3:4b', name: 'Ollama' }),
    );
    expect(screen.getByTestId('ollama-install-cancel-ollama')).toBeInTheDocument();
    const pull = calls.find((c) => c.url === '/api/models/pull');
    expect(pull?.body).toEqual({ model: 'gemma3:4b', connection_id: 'ollama' });

    installed = [...installed, { id: 'gemma3:4b', chat: true }];
    finish(json({ status: 'ok', model: 'gemma3:4b', joined: false }));
    expect(await screen.findByTestId('ollama-models-notice-ollama')).toHaveTextContent(
      fmt(O.installed, { model: 'gemma3:4b', name: 'Ollama' }),
    );
    expect(await screen.findByTestId('ollama-model-ollama-gemma3:4b')).toBeInTheDocument();
    await waitFor(() => expect(screen.getByTestId('connection-status-ollama')).toHaveTextContent('Connected · 2 models'));
    expect(onChanged).toHaveBeenCalled();
    expect(screen.queryByTestId('ollama-installing-ollama')).toBeNull();
  });

  // Killed by: frontend/src/components/settings/OllamaModels.tsx ::         const silent = err instanceof CoreFailure && err.status === 504;
  // Becomes:         const silent = false && err instanceof CoreFailure;
  it('says a refused install and a silent one apart, in its own words', async () => {
    const core = 'Could not pull nope:1b from Ollama. Stopped waiting; run: ucx llm pull nope:1b';
    let status = 502;
    serve({
      [`GET ${INSTALLED}`]: installedRoute([]),
      'POST /api/models/pull': () => json({ detail: core }, status),
    });
    render(<ConnectionsSection />);
    await openPanel();
    expect(await screen.findByTestId('ollama-models-none-ollama')).toHaveTextContent(fmt(O.none, { name: 'Ollama' }));

    typeAndInstall('nope:1b');
    const notice = await screen.findByTestId('ollama-models-notice-ollama');
    expect(notice).toHaveTextContent(fmt(O.installFailed, { model: 'nope:1b', name: 'Ollama' }));
    expect(notice).not.toHaveTextContent('ucx');

    status = 504;
    typeAndInstall('nope:1b');
    await waitFor(() =>
      expect(screen.getByTestId('ollama-models-notice-ollama')).toHaveTextContent(
        fmt(O.installSilent, { model: 'nope:1b', name: 'Ollama' }),
      ),
    );
    expect(screen.getByTestId('ollama-models-notice-ollama')).not.toHaveTextContent('ucx');
  });

  // Killed by: frontend/src/components/settings/OllamaModels.tsx ::       if (isAbort(err)) {
  // Becomes:       if (false) {
  it('stops waiting on Cancel and says the download may go on, not that it failed', async () => {
    serve({
      [`GET ${INSTALLED}`]: installedRoute([]),
      'POST /api/models/pull': (init) =>
        new Promise<Answer>((_resolve, reject) => {
          init.signal?.addEventListener('abort', () => reject(new DOMException('aborted', 'AbortError')));
        }),
    });
    render(<ConnectionsSection />);
    await openPanel();
    await screen.findByTestId('ollama-models-none-ollama');
    typeAndInstall('qwen3:8b');
    fireEvent.click(await screen.findByTestId('ollama-install-cancel-ollama'));

    const notice = await screen.findByTestId('ollama-models-notice-ollama');
    expect(notice).toHaveTextContent(fmt(O.installStopped, { model: 'qwen3:8b' }));
    expect(notice).toHaveAttribute('role', 'status');
  });

  // Killed by: frontend/src/lib/modelGateway.ts ::   await answer<unknown>(await fetch('/api/models/delete', jsonInit('POST', { model, connection_id: id })));
  // Becomes:   await answer<unknown>(await fetch('/api/models/delete', jsonInit('POST', { model })));
  it('removes a model from this connection only after the confirm step', async () => {
    const calls = serve({
      [`GET ${INSTALLED}`]: installedRoute([{ id: 'qwen3:14b', chat: true }]),
      'POST /api/models/delete': () => json({ status: 'ok', model: 'qwen3:14b' }),
      'POST /api/connections/ollama/check': () => json({ ...ROWS[1], model_count: 0 }),
    });
    render(<ConnectionsSection />);
    await openPanel();
    fireEvent.click(await screen.findByTestId('ollama-model-remove-ollama-qwen3:14b'));
    expect(screen.getByTestId('ollama-model-confirm-ollama-qwen3:14b')).toHaveTextContent(
      fmt(O.removeConfirm, { model: 'qwen3:14b', name: 'Ollama' }),
    );
    expect(calls.some((c) => c.url === '/api/models/delete')).toBe(false);

    fireEvent.click(screen.getByTestId('ollama-model-remove-yes-ollama-qwen3:14b'));
    expect(await screen.findByTestId('ollama-models-notice-ollama')).toHaveTextContent(
      fmt(O.removed, { model: 'qwen3:14b', name: 'Ollama' }),
    );
    expect(calls.find((c) => c.url === '/api/models/delete')?.body).toEqual({ model: 'qwen3:14b', connection_id: 'ollama' });
    expect(screen.queryByTestId('ollama-model-ollama-qwen3:14b')).toBeNull();
  });

  // Killed by: frontend/src/components/settings/OllamaModels.tsx ::           <span>{fmt(t.loadFailed, { name })}</span>
  // Becomes:           <span>{t.retry}</span>
  it('says why the installed list is missing, in Korean too', async () => {
    serve({ [`GET ${INSTALLED}`]: () => json({ detail: "Couldn't get an answer from Ollama.", code: 'unreachable' }, 502) });
    render(
      <LocaleProvider hints={['ko-KR']}>
        <ConnectionsSection />
      </LocaleProvider>,
    );
    fireEvent.click(await screen.findByTestId('ollama-models-open-ollama'));
    const error = await screen.findByTestId('ollama-models-error-ollama');
    expect(error).toHaveTextContent(fmt(ko.gateway.ollama.loadFailed, { name: 'Ollama' }));
    expect(error).not.toHaveTextContent("Couldn't");
    expectPlain(error.textContent);
  });
});
