import { afterEach, describe, expect, it, vi } from 'vitest';
import { fetchPersonaCatalog, savePersona, synthesizePersonaPrompt } from './personasApi';
import { describeRefusal, draftFromPersona } from './personaDraft';
import type { PersonaEditorCopy } from './personaDraft';
import { makePersonaInfo } from '../test/fixtures';

const answer = (body: unknown, status = 200) =>
  new Response(JSON.stringify(body), { status, headers: { 'Content-Type': 'application/json' } });

/** Only a refusal reads the copy; these cases are answered. */
const COPY = {} as PersonaEditorCopy;

afterEach(() => {
  vi.unstubAllGlobals();
});

describe('the clone catalogue over /api/clones (#1814)', () => {
  // Killed by: frontend/src/lib/personasApi.ts :: personas: data.clones ?? [],
  // Becomes: personas: [],
  it('reads the clones the listing returns', async () => {
    const fetchMock = vi.fn(async () => answer({ clones: [makePersonaInfo({ id: 'agt_1', name: 'scout' })] }));
    vi.stubGlobal('fetch', fetchMock);

    const catalog = await fetchPersonaCatalog();

    expect(fetchMock).toHaveBeenCalledWith('/api/clones');
    expect(catalog.personas.map((p) => p.id)).toEqual(['agt_1']);
  });

  // Killed by: frontend/src/lib/personasApi.ts :: const url = mode === 'create' ? '/api/clones' : `/api/clones/${encodeURIComponent(id || draft.name)}`;
  // Becomes: const url = mode === 'create' ? '/api/clones' : `/api/clones/${encodeURIComponent(draft.name)}`;
  // Killed by: frontend/src/lib/personasApi.ts :: body: JSON.stringify(body),
  // Becomes: body: JSON.stringify(draft),
  it('saves an edit to the clone`s id, and a new handle travels in the body only', async () => {
    const calls: Array<[string, RequestInit]> = [];
    vi.stubGlobal(
      'fetch',
      vi.fn(async (url: string, init: RequestInit) => {
        calls.push([url, init]);
        return answer({ persona: makePersonaInfo({ id: 'agt_1', name: 'pathfinder' }) });
      }),
    );
    // A rename: the handle changes, the id does not.
    const draft = { ...draftFromPersona(makePersonaInfo({ id: 'agt_1', name: 'scout' })), name: 'pathfinder' };

    const result = await savePersona(draft, 'edit', COPY);

    expect(result.ok).toBe(true);
    expect(calls[0][0]).toBe('/api/clones/agt_1');
    expect(calls[0][1].method).toBe('PUT');
    const sent = JSON.parse(String(calls[0][1].body)) as Record<string, unknown>;
    expect(sent.name).toBe('pathfinder');
    expect(sent).not.toHaveProperty('id');
  });

  it('creates at the collection', async () => {
    const calls: string[] = [];
    vi.stubGlobal(
      'fetch',
      vi.fn(async (url: string) => {
        calls.push(url);
        return answer({ persona: makePersonaInfo({ id: 'agt_2', name: 'critic' }) });
      }),
    );

    await savePersona(draftFromPersona(makePersonaInfo({ name: 'critic' })), 'create', COPY);

    expect(calls).toEqual(['/api/clones']);
  });
});

const TEST_COPY: PersonaEditorCopy = {
  ...({} as PersonaEditorCopy),
  unreachable: 'The server could not be reached. Check it is running, then save again.',
  saveFailed: 'The save failed. Check the server log, then save again.',
  fieldsRefused: 'The server refused these fields: {fields}. Correct them and save again.',
};

describe('savePersona plain error copy (#1460)', () => {
  it('returns copy.unreachable without error interpolation when fetch rejects', async () => {
    vi.stubGlobal('fetch', vi.fn().mockRejectedValue(new TypeError('Failed to fetch')));

    const draft = draftFromPersona(makePersonaInfo({ name: 'scout' }));
    const result = await savePersona(draft, 'create', TEST_COPY);

    expect(result).toEqual({
      ok: false,
      message: 'The server could not be reached. Check it is running, then save again.',
    });
    expect(result.ok ? '' : result.message).not.toContain('TypeError');
    expect(result.ok ? '' : result.message).not.toContain('Failed to fetch');
  });

  it('returns copy.saveFailed without HTTP status when response is bodyless error', async () => {
    vi.stubGlobal('fetch', vi.fn().mockResolvedValue(new Response('', { status: 500 })));

    const draft = draftFromPersona(makePersonaInfo({ name: 'scout' }));
    const result = await savePersona(draft, 'create', TEST_COPY);

    expect(result).toEqual({
      ok: false,
      message: 'The save failed. Check the server log, then save again.',
    });
    expect(result.ok ? '' : result.message).not.toContain('500');
    expect(result.ok ? '' : result.message).not.toContain('HTTP');
  });

  it('surfaces the Core detail string when response provides one', async () => {
    vi.stubGlobal(
      'fetch',
      vi.fn().mockResolvedValue(answer({ detail: 'Clone name "scout" is reserved.' }, 400)),
    );

    const draft = draftFromPersona(makePersonaInfo({ name: 'scout' }));
    const result = await savePersona(draft, 'create', TEST_COPY);

    expect(result).toEqual({
      ok: false,
      message: 'Clone name "scout" is reserved.',
    });
  });
});

describe('synthesizePersonaPrompt plain error copy (#1460)', () => {
  it('falls back to plain sentence when response is not ok and detail is absent', async () => {
    vi.stubGlobal('fetch', vi.fn().mockResolvedValue(new Response('', { status: 502 })));

    const result = await synthesizePersonaPrompt({
      name: 'scout',
      role: 'explorer',
      description: 'explores code',
      allowed_tools: [],
    });

    expect(result).toEqual({
      ok: false,
      message: 'The draft could not be generated.',
    });
    expect(result.ok ? '' : result.message).not.toContain('502');
    expect(result.ok ? '' : result.message).not.toContain('HTTP');
  });

  it('surfaces Core detail when synthesize response provides one', async () => {
    vi.stubGlobal(
      'fetch',
      vi.fn().mockResolvedValue(answer({ detail: 'Model is not available.' }, 503)),
    );

    const result = await synthesizePersonaPrompt({
      name: 'scout',
      role: 'explorer',
      description: 'explores code',
      allowed_tools: [],
    });

    expect(result).toEqual({
      ok: false,
      message: 'Model is not available.',
    });
  });
});

describe('describeRefusal (#1460)', () => {
  it('returns plain string detail as is', () => {
    expect(describeRefusal('A clone with this name exists.', 400, TEST_COPY)).toBe(
      'A clone with this name exists.',
    );
  });

  it('returns copy.saveFailed without HTTP status code when detail is missing or not a string/array', () => {
    expect(describeRefusal(null, 500, TEST_COPY)).toBe(
      'The save failed. Check the server log, then save again.',
    );
    expect(describeRefusal(undefined, 404, TEST_COPY)).toBe(
      'The save failed. Check the server log, then save again.',
    );
    expect(describeRefusal({}, 502, TEST_COPY)).toBe(
      'The save failed. Check the server log, then save again.',
    );
  });
});
