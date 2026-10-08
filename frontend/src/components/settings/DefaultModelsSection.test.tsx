import { afterEach, describe, expect, it, vi } from 'vitest';
import { fireEvent, render, screen, waitFor } from '@testing-library/react';
import { DefaultModelsSection } from './DefaultModelsSection';
import { LocaleProvider } from '../../i18n';
import { en } from '../../i18n/en';
import { ko } from '../../i18n/ko';
import { expectPlain } from '../../test/plainCopy';
import type { ModelSet } from '../../lib/modelGateway';

const T = en.gateway;

const entry = (connection: string, id: string, display_name: string, capabilities = ['chat']) => ({
  ref: `${connection}/${id}`,
  id,
  display_name,
  capabilities,
  context_window: null,
});

const chatSet = (defaults: Partial<ModelSet['defaults']> = {}): ModelSet => ({
  groups: [
    {
      connection_id: 'gemini', label: 'Google Gemini', kind: 'gemini', status: 'connected', detail: null,
      models: [entry('gemini', 'gemini-3.8-pro', 'Gemini 3.8 Pro'), entry('gemini', 'gemini-3.8-flash', 'Gemini 3.8 Flash')],
    },
    {
      connection_id: 'gpu-box', label: 'GPU box', kind: 'vllm', status: 'unreachable', detail: 'Connection refused', models: [],
    },
  ],
  defaults: { deep: null, fast: null, image: 'auto', ...defaults },
  recommended: { deep: 'gemini/gemini-3.8-pro', fast: 'gemini/gemini-3.8-flash' },
});

const imageSet: ModelSet = {
  groups: [
    {
      connection_id: 'gemini', label: 'Google Gemini', kind: 'gemini', status: 'connected', detail: null,
      models: [entry('gemini', 'gemini-2.5-flash-image', 'Gemini 2.5 Flash Image', ['image_create'])],
    },
  ],
  defaults: { deep: null, fast: null, image: 'auto' },
  recommended: { deep: null, fast: null },
};

type Answer = { ok: boolean; status: number; json: () => Promise<unknown> };
const json = (body: unknown, status = 200): Answer => ({ ok: status < 400, status, json: async () => body });

/** `GET /api/media/status`: what draws the next picture. Unanswered unless a test sets it. */
let mediaStatus: () => Answer = () => json({}, 404);

const serve = (chat: () => ModelSet, settings: (body: unknown) => Answer = () => json({})) => {
  const calls: { url: string; method: string; body?: unknown }[] = [];
  vi.stubGlobal(
    'fetch',
    vi.fn(async (url: string, init: RequestInit = {}) => {
      const method = init.method ?? 'GET';
      const body = init.body ? JSON.parse(String(init.body)) : undefined;
      calls.push({ url, method, body });
      if (url.startsWith('/api/models?capability=chat')) return json(chat());
      if (url.startsWith('/api/models?capability=image')) return json(imageSet);
      if (method === 'POST' && url === '/api/settings') return settings(body);
      if (url === '/api/media/status') return mediaStatus();
      if (method === 'POST' && url === '/api/connections') return json({ id: 'comfyui', kind: 'comfyui' });
      return json({}, 404);
    }),
  );
  return calls;
};

const posts = (calls: ReturnType<typeof serve>) => calls.filter((c) => c.method === 'POST');

afterEach(() => {
  vi.unstubAllGlobals();
  mediaStatus = () => json({}, 404);
});

const resolved = (create: { label: string | null; where: string } | null, reason_code: string | null = null) =>
  () => json({ resolved: { create, reason_code, refusal: null } });

describe('DefaultModelsSection', () => {
  it('offers the model set grouped by connection, and says why a connection offers nothing', async () => {
    serve(() => chatSet());
    render(<DefaultModelsSection />);

    const deep = await screen.findByTestId('settings-default-deep-select');
    const groups = Array.from(deep.querySelectorAll('optgroup')).map((g) => g.getAttribute('label'));
    expect(groups).toEqual(['Google Gemini']);
    // No remembered list fills in for the unreachable one: it is a line saying so.
    expect(screen.getByTestId('settings-default-deep-select-silent')).toHaveTextContent(`GPU box: ${T.status.unreachable}`);
    expect(screen.getByTestId('settings-default-models')).not.toHaveTextContent('Connection refused');
  });

  // Killed by: frontend/src/components/settings/DefaultModelsSection.tsx :: const offerRecommended = value === '' && slot === 'deep' && recommended !== null;
  // Becomes: const offerRecommended = false;
  it('shows the recommended model and never saves it unasked', async () => {
    const calls = serve(() => chatSet());
    render(<DefaultModelsSection />);

    const deep = await screen.findByTestId('settings-default-deep-select');
    expect(deep).toHaveValue('');
    expect(screen.getByRole('option', { name: 'Gemini 3.8 Pro (recommended)' })).toBeInTheDocument();
    expect(screen.getByTestId('settings-default-deep-recommended')).toHaveTextContent('Recommended: Gemini 3.8 Pro.');
    await new Promise((resolve) => setTimeout(resolve, 20));
    expect(posts(calls)).toEqual([]);

    fireEvent.click(screen.getByRole('button', { name: T.defaults.useRecommended }));
    await waitFor(() => expect(posts(calls)).toHaveLength(1));
    expect(posts(calls)[0].body).toEqual({ default_models: { deep: 'gemini/gemini-3.8-pro' } });
  });

  // Killed by: frontend/src/components/settings/DefaultModelsSection.tsx :: await saveDefaultModel(slot, next === '' ? null : next);
  // Becomes: await saveDefaultModel(slot, next);
  it('saves each slot alone as it is chosen, and Same as conversations as null', async () => {
    let saved = chatSet({ deep: 'gemini/gemini-3.8-pro', fast: 'gemini/gemini-3.8-flash' });
    const calls = serve(
      () => saved,
      (body) => {
        const { default_models } = body as { default_models: Partial<ModelSet['defaults']> };
        saved = { ...saved, defaults: { ...saved.defaults, ...default_models } as ModelSet['defaults'] };
        return json({});
      },
    );
    render(<DefaultModelsSection />);

    const fast = await screen.findByTestId('settings-default-fast-select');
    await waitFor(() => expect(fast).toHaveValue('gemini/gemini-3.8-flash'));
    fireEvent.change(fast, { target: { value: '' } });

    await waitFor(() => expect(screen.getByTestId('settings-default-fast-status')).toHaveAttribute('data-state', 'saved'));
    expect(posts(calls).map((c) => c.body)).toEqual([{ default_models: { fast: null } }]);
    expect(fast).toHaveValue('');
    expect(screen.getByRole('option', { name: T.defaults.sameAsDeep })).toBeInTheDocument();
  });

  it('offers Automatic for pictures and saves a picture model as a ref', async () => {
    const calls = serve(() => chatSet());
    render(<DefaultModelsSection />);

    const image = await screen.findByTestId('settings-default-image-select');
    expect(image).toHaveValue('auto');
    fireEvent.change(image, { target: { value: 'gemini/gemini-2.5-flash-image' } });

    await waitFor(() => expect(posts(calls)).toHaveLength(1));
    expect(posts(calls)[0].body).toEqual({ default_models: { image: 'gemini/gemini-2.5-flash-image' } });
  });

  it('marks a saved default its connection no longer lists, keeping it selected', async () => {
    serve(() => chatSet({ deep: 'gpu-box/qwen3-coder' }));
    render(<DefaultModelsSection />);

    const deep = await screen.findByTestId('settings-default-deep-select');
    await waitFor(() => expect(deep).toHaveValue('gpu-box/qwen3-coder'));
    expect(screen.getByRole('option', { name: `gpu-box/qwen3-coder ${T.picker.unavailableSuffix}` })).toBeInTheDocument();
  });

  it('says a refused save in plain words under the field, and puts the saved choice back', async () => {
    vi.spyOn(console, 'error').mockImplementation(() => {});
    serve(
      () => chatSet({ deep: 'gemini/gemini-3.8-pro' }),
      () => ({ ok: false, status: 500, json: async () => { throw new SyntaxError('bad'); } }),
    );
    render(<DefaultModelsSection />);

    const deep = await screen.findByTestId('settings-default-deep-select');
    await waitFor(() => expect(deep).toHaveValue('gemini/gemini-3.8-pro'));
    fireEvent.change(deep, { target: { value: 'gemini/gemini-3.8-flash' } });

    const status = await screen.findByTestId('settings-default-deep-status');
    await waitFor(() => expect(status).toHaveAttribute('data-state', 'error'));
    expect(status).toHaveTextContent(T.defaults.saveFailed);
    expectPlain(status.textContent);
    expect(deep).toHaveValue('gemini/gemini-3.8-pro');
  });

  it('asks for a connection when none exists, rather than drawing empty pickers alone', async () => {
    serve(() => ({ groups: [], defaults: { deep: null, fast: null, image: 'auto' }, recommended: { deep: null, fast: null } }));
    render(<DefaultModelsSection />);
    expect(await screen.findByTestId('settings-default-models-no-connection')).toHaveTextContent(T.defaults.noConnection);
  });

  it('reads the set again when a connection changes', async () => {
    const calls = serve(() => chatSet());
    const { rerender } = render(<DefaultModelsSection version={0} />);
    await screen.findByTestId('settings-default-deep-select');
    const before = calls.filter((c) => c.url.startsWith('/api/models?capability=chat')).length;

    rerender(<DefaultModelsSection version={1} />);
    await waitFor(() =>
      expect(calls.filter((c) => c.url.startsWith('/api/models?capability=chat')).length).toBeGreaterThan(before),
    );
  });

  it('reads in Korean', async () => {
    serve(() => chatSet());
    render(
      <LocaleProvider hints={['ko-KR']}>
        <DefaultModelsSection />
      </LocaleProvider>,
    );
    await screen.findByTestId('settings-default-deep-select');
    expect(screen.getByText(ko.gateway.defaults.title)).toBeTruthy();
    expect(screen.getByTestId('settings-default-deep-select-silent')).toHaveTextContent(
      `GPU box: ${ko.gateway.status.unreachable}`,
    );
    expect(screen.getByRole('option', { name: ko.gateway.defaults.sameAsDeep })).toBeInTheDocument();
  });

  // Killed by: frontend/src/components/settings/DefaultModelsSection.tsx :: ? fmt(g.imageNow, { model: create.label, where })
  // Becomes: ? fmt(g.imageNowUnreported, { where })
  it('says under Pictures which model draws now and where, in plain words', async () => {
    mediaStatus = resolved({ label: 'Illustrious v4', where: 'this_computer' });
    serve(() => chatSet());
    render(<DefaultModelsSection />);

    const now = await screen.findByTestId('settings-default-image-now');
    expect(now).toHaveTextContent('Now drawing with Illustrious v4 · This computer.');
    expectPlain(now.textContent);
  });

  it('says the GPU server draws when it names no model', async () => {
    mediaStatus = resolved({ label: null, where: 'gpu_server' });
    serve(() => chatSet());
    render(<DefaultModelsSection />);

    expect(await screen.findByTestId('settings-default-image-now')).toHaveTextContent(
      'Now drawing on Your GPU server; it does not say which model.',
    );
  });

  // Killed by: frontend/src/components/settings/DefaultModelsSection.tsx :: const notReady = resolved.reason_code === 'chosen_not_ready';
  // Becomes: const notReady = false;
  it('tells nothing-can-draw apart from a chosen model that cannot draw', async () => {
    mediaStatus = resolved(null, 'no_image_model');
    serve(() => chatSet());
    const { unmount } = render(<DefaultModelsSection />);
    const none = await screen.findByTestId('settings-default-image-none');
    expect(none).toHaveTextContent(T.defaults.imageNone);
    expect(none).toHaveTextContent(T.defaults.imageNoneHint);
    unmount();

    mediaStatus = resolved(null, 'chosen_not_ready');
    render(<DefaultModelsSection />);
    const notReady = await screen.findByTestId('settings-default-image-none');
    expect(notReady).toHaveTextContent(T.defaults.imageNotReady);
    expect(notReady).not.toHaveTextContent(T.defaults.imageNone);
  });

  // Killed by: frontend/src/components/settings/DefaultModelsSection.tsx :: setPictureVersion((v) => v + 1);
  // Becomes:
  it('reads what draws again once a picture model is saved', async () => {
    mediaStatus = resolved({ label: 'Illustrious v4', where: 'this_computer' });
    const calls = serve(() => chatSet());
    render(<DefaultModelsSection />);
    await screen.findByTestId('settings-default-image-now');
    const reads = () => calls.filter((c) => c.url === '/api/media/status').length;
    const before = reads();

    fireEvent.change(screen.getByTestId('settings-default-image-select'), {
      target: { value: 'gemini/gemini-2.5-flash-image' },
    });

    await waitFor(() => expect(reads()).toBeGreaterThan(before));
  });

  it('says what draws in Korean', async () => {
    mediaStatus = resolved({ label: 'Illustrious v4', where: 'cloud' });
    serve(() => chatSet());
    render(
      <LocaleProvider hints={['ko-KR']}>
        <DefaultModelsSection />
      </LocaleProvider>,
    );
    expect(await screen.findByTestId('settings-default-image-now')).toHaveTextContent(
      '지금은 Illustrious v4 · 클라우드 · Google에서 그립니다.',
    );
  });

  // Killed by: frontend/src/components/settings/DefaultModelsSection.tsx :: await addConnection({ kind: 'comfyui', base_url: address });
  // Becomes:
  it('offers a ComfyUI found on this computer and adds it only when pressed', async () => {
    mediaStatus = () =>
      json({
        resolved: { create: null, reason_code: 'no_image_model', refusal: null },
        detected_comfyui: 'http://127.0.0.1:8188',
      });
    const calls = serve(() => chatSet());
    const changed = vi.fn();
    render(<DefaultModelsSection onConnectionsChanged={changed} />);

    const offer = await screen.findByTestId('settings-default-image-detected');
    expect(offer).toHaveTextContent(
      'A ComfyUI is running on this computer at http://127.0.0.1:8188. Add it to draw pictures with?',
    );
    // Detected, never taken: nothing is saved until the person says so.
    expect(calls.filter((c) => c.url === '/api/connections')).toEqual([]);

    fireEvent.click(screen.getByTestId('settings-default-image-detected-add'));

    await waitFor(() => expect(changed).toHaveBeenCalled());
    const added = calls.filter((c) => c.method === 'POST' && c.url === '/api/connections');
    expect(added.map((c) => c.body)).toEqual([{ kind: 'comfyui', base_url: 'http://127.0.0.1:8188' }]);
  });

  it('offers nothing when no ComfyUI was found', async () => {
    mediaStatus = resolved(null, 'no_image_model');
    serve(() => chatSet());
    render(<DefaultModelsSection />);
    await screen.findByTestId('settings-default-image-none');
    expect(screen.queryByTestId('settings-default-image-detected')).toBeNull();
  });
});
