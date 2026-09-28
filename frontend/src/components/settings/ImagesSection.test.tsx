import { describe, it, expect, vi, afterEach } from 'vitest';
import { render, screen, fireEvent, waitFor } from '@testing-library/react';
import { ImagesSection, imageModelsOf, type ImageSettings } from './ImagesSection';
import { LocaleProvider } from '../../i18n';
import { expectPlain } from '../../test/plainCopy';

const answer = (status: number, body: unknown = {}) =>
  new Response(JSON.stringify(body), { status, headers: { 'Content-Type': 'application/json' } });

const status = (engine: string, engines: [string, boolean, string][], setting = 'auto') => ({
  ready: engine !== 'none',
  engine,
  setting,
  engines: engines.map(([name, ready, reason_code]) => ({ name, ready, reason_code })),
});

const NOTHING = status('none', [
  ['remote-cuda', false, 'not_configured'],
  ['comfyui-local', false, 'not_running'],
  ['diffusers-sdxl', false, 'missing_dependencies'],
  ['gemini', false, 'no_key'],
]);

/** Answers each route from `routes`; the language provider's settings read answers empty. */
const serve = (routes: Record<string, (init: RequestInit) => Response>) => {
  const calls: [string, RequestInit][] = [];
  vi.stubGlobal(
    'fetch',
    vi.fn(async (url: string, init: RequestInit = {}) => {
      calls.push([url, init]);
      const route = routes[url];
      if (route) return route(init);
      return answer(200, {});
    }),
  );
  return calls;
};

const renderSection = (settings: ImageSettings | null, korean = false) =>
  render(
    <LocaleProvider hints={[korean ? 'ko-KR' : 'en-US']}>
      <ImagesSection settings={settings} />
    </LocaleProvider>,
  );

afterEach(() => {
  vi.unstubAllGlobals();
});

describe('Settings › Images', () => {
  // Killed by: frontend/src/components/settings/ImagesSection.tsx :: {fmt(copy.now.drawing, { engine: copy.engines[status.engine] })}
  // Becomes: {status.engine}
  it('says which way draws pictures now, by name and not by its id', async () => {
    serve({
      '/api/media/status': () =>
        answer(
          200,
          status('comfyui-local', [
            ['remote-cuda', false, 'not_configured'],
            ['comfyui-local', true, 'ready'],
            ['diffusers-sdxl', false, 'no_checkpoint'],
            ['gemini', false, 'chat_provider_not_gemini'],
          ]),
        ),
    });
    renderSection({ image_engine: 'auto', image_model: 'gemini-2.5-flash-image', image_settings_problem: '' });

    const now = await screen.findByTestId('settings-images-now');
    expect(now).toHaveTextContent('ComfyUI on this computer');
    expect(now.textContent).not.toMatch(/comfyui-local/);
    expect(screen.getByTestId('settings-images-engine-diffusers-sdxl')).toHaveTextContent(
      'no model file is downloaded',
    );
  });

  // Killed by: frontend/src/components/settings/ImagesSection.tsx :: <p className="text-[11px] text-slate-400">{copy.now.noneHint}</p>
  // Becomes: <p className="text-[11px] text-slate-400"></p>
  it('says plainly when nothing can draw, and what to do about it', async () => {
    serve({ '/api/media/status': () => answer(200, NOTHING) });
    renderSection({ image_engine: 'auto', image_model: 'gemini-2.5-flash-image', image_settings_problem: '' });

    const none = await screen.findByTestId('settings-images-none');
    expect(none).toHaveTextContent('Nothing can draw pictures right now.');
    expect(none).toHaveTextContent(/Gemini API key/);
    expect(none).toHaveTextContent(/upload a picture/);
    for (const text of [none.textContent, screen.getByTestId('settings-images-status').textContent]) {
      expectPlain(text);
      expect(text).not.toMatch(/engine|persona|\btool\b|_/i);
    }
  });

  // Killed by: frontend/src/components/settings/ImagesSection.tsx :: body: JSON.stringify({ image_engine: choice, image_model: model }),
  // Becomes: body: JSON.stringify({}),
  it('saves the choice and the model, then checks again', async () => {
    const calls = serve({
      '/api/media/status': () => answer(200, NOTHING),
      '/api/settings': (init) =>
        init.method === 'POST'
          ? answer(200, { image_engine: 'local', image_model: 'gemini-2.5-flash-image', image_settings_problem: '' })
          : answer(200, {}),
    });
    renderSection({ image_engine: 'auto', image_model: 'gemini-2.5-flash-image', image_settings_problem: '' });
    await screen.findByTestId('settings-images-none');

    fireEvent.click(screen.getByTestId('settings-images-choice-local'));
    fireEvent.click(screen.getByTestId('settings-images-save'));

    await screen.findByTestId('settings-images-saved');
    const post = calls.find(([url, init]) => url === '/api/settings' && init.method === 'POST');
    expect(JSON.parse(post![1].body as string)).toEqual({
      image_engine: 'local',
      image_model: 'gemini-2.5-flash-image',
    });
    await waitFor(() => expect(calls.filter(([url]) => url === '/api/media/status')).toHaveLength(2));
    // Only this computer: no Gemini model to choose.
    expect(screen.queryByTestId('settings-images-model')).toBeNull();
  });

  // Killed by: frontend/src/components/settings/ImagesSection.tsx :: if ((choice !== 'gemini' && !wantList) || listed !== null) return;
  // Becomes: if (listed !== null) return;
  it('offers Google`s picture models under Gemini, and asks Google nothing on arrival', async () => {
    const calls = serve({
      '/api/media/status': () => answer(200, NOTHING),
      '/api/models/catalog': () =>
        answer(200, {
          provider: 'gemini',
          catalog: {
            entries: [
              { id: 'gemini-2.5-flash' },
              { id: 'gemini-3-pro-image-preview' },
              { id: 'imagen-4.0-generate-001' },
            ],
          },
        }),
    });
    renderSection({ image_engine: 'auto', image_model: 'gemini-2.5-flash-image', image_settings_problem: '' });
    await screen.findByTestId('settings-images-none');
    expect(calls.some(([url]) => url === '/api/models/catalog')).toBe(false);

    fireEvent.click(screen.getByTestId('settings-images-choice-gemini'));

    await waitFor(() =>
      expect(screen.getByRole('option', { name: 'gemini-3-pro-image-preview' })).toBeInTheDocument(),
    );
    expect(screen.getByRole('option', { name: 'gemini-2.5-flash-image (recommended)' })).toBeInTheDocument();
    expect(screen.queryByRole('option', { name: 'gemini-2.5-flash' })).toBeNull();
  });

  // Killed by: frontend/src/components/settings/ImagesSection.tsx :: {copy.now.badSetting}
  // Becomes: {String(settings?.image_settings_problem)}
  it('says a saved choice it cannot read in its own words, not the runtime`s', async () => {
    serve({ '/api/media/status': () => answer(409, { detail: "The picture engine must be auto, local or gemini; 'x'" }) });
    renderSection({
      image_engine: 'x',
      image_model: 'gemini-2.5-flash-image',
      image_settings_problem: "The picture engine must be auto, local or gemini; 'x' is not one of them.",
    });

    const bad = screen.getByTestId('settings-images-bad-setting');
    expect(bad).toHaveTextContent('Choose again and save');
    expect(bad.textContent).not.toMatch(/'x'/);
    expect(screen.queryByTestId('settings-images-error')).toBeNull();
  });

  it('keeps Google`s model ids only when they name a picture model', () => {
    expect(
      imageModelsOf({ catalog: { entries: [{ id: 'gemini-2.5-flash-image' }, { id: 'gemini-2.5-pro' }] } }),
    ).toEqual(['gemini-2.5-flash-image']);
    expect(imageModelsOf({ catalog: null })).toEqual([]);
  });

  it('reads in Korean', async () => {
    serve({ '/api/media/status': () => answer(200, NOTHING) });
    renderSection({ image_engine: 'auto', image_model: 'gemini-2.5-flash-image', image_settings_problem: '' }, true);

    const none = await screen.findByTestId('settings-images-none');
    expectPlain(none.textContent);
    expect(none.textContent).toMatch(/니다/);
  });
});
