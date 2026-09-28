import { describe, it, expect, vi, afterEach } from 'vitest';
import {
  avatarChangeIdsOf,
  avatarUrlsOf,
  personaOfSeat,
  personaPicture,
  pictureOf,
  resetAvatar,
  setAvatarFromPath,
  undoAvatarChange,
  undoStillOffered,
  uploadAvatar,
  uploadProblem,
  MAX_UPLOAD_BYTES,
} from './avatarChoice';
import type { RoomState } from '../types';

const room = (participants: RoomState['participants']): RoomState =>
  ({ room_id: 'r1', title: 't', participants, transcript: [] }) as unknown as RoomState;

const answer = (status: number, body: unknown = {}) =>
  new Response(JSON.stringify(body), { status, headers: { 'Content-Type': 'application/json' } });

afterEach(() => {
  vi.unstubAllGlobals();
});

describe('whose picture a seat shows', () => {
  // Killed by: frontend/src/lib/avatarChoice.ts :: return seat?.persona || participantId;
  // Becomes: return participantId;
  it('reads the clone behind a second seat, not the seat id', () => {
    // A second seat of the same clone has an id of its own; the picture is the clone's.
    const r = room([
      { id: 'scout', kind: 'agent', display_name: 'Scout', persona: 'scout' },
      { id: 'scout-2', kind: 'agent', display_name: 'Scout', persona: 'scout' },
    ]);
    expect(personaOfSeat(r, 'scout-2')).toBe('scout');
    // A seat that names no clone falls back to its id, which is the clone's for the first seat.
    expect(personaOfSeat(room([{ id: 'critic', kind: 'agent', display_name: 'Critic' }]), 'critic')).toBe(
      'critic',
    );
  });
});

describe('where a picture is drawn from', () => {
  // Killed by: frontend/src/lib/avatarChoice.ts :: if (persona.avatar_url !== undefined) urls[persona.name] = persona.avatar_url;
  // Becomes: if (false) urls[persona.name] = persona.avatar_url;
  it('uses the listed address, which changes with the picture', () => {
    const urls = avatarUrlsOf([
      { name: 'scout', avatar_url: '/api/personas/scout/avatar?v=abc' },
      { name: 'critic', avatar_url: null },
      { name: 'old' },
    ]);
    expect(pictureOf('scout', urls)).toBe('/api/personas/scout/avatar?v=abc');
    // A listed clone with no picture draws the default and asks for nothing.
    expect(pictureOf('critic', urls)).toBeUndefined();
    // A runtime too old to list the field, and a clone not listed yet, keep the fixed address.
    expect(pictureOf('old', urls)).toBe('/api/personas/old/avatar');
    expect(pictureOf('newcomer', urls)).toBe('/api/personas/newcomer/avatar');
  });

  // Killed by: frontend/src/lib/avatarChoice.ts :: persona && persona.avatar_url !== undefined ? (persona.avatar_url ?? undefined) : personaAvatarUrl(name);
  // Becomes: personaAvatarUrl(name);
  it('gives a profile the persona`s own address', () => {
    expect(personaPicture('scout', { avatar_url: '/api/personas/scout/avatar?v=2' })).toBe(
      '/api/personas/scout/avatar?v=2',
    );
    expect(personaPicture('scout', { avatar_url: null })).toBeUndefined();
    expect(personaPicture('scout', undefined)).toBe('/api/personas/scout/avatar');
  });
});

describe('changing a picture', () => {
  // Killed by: frontend/src/lib/avatarChoice.ts :: body: JSON.stringify({ source_path: path }),
  // Becomes: body: JSON.stringify({ path }),
  // Killed by: frontend/src/lib/avatarChoice.ts :: if (typeof id === 'number' && Number.isInteger(id)) changeId = id;
  // Becomes: if (false) changeId = id;
  it('gives a drawn picture by its place in the workspace, and keeps what undoes it', async () => {
    const fetchMock = vi.fn(async () =>
      answer(200, { status: 'ok', persona: 'scout', previous_path: '.uclone/personas/scout.prev.png', change_id: 4 }),
    );
    vi.stubGlobal('fetch', fetchMock);

    const result = await setAvatarFromPath('scout', 'artifacts/images/a.png');

    expect(result).toEqual({ ok: true, previousPath: '.uclone/personas/scout.prev.png', changeId: 4 });
    const [url, init] = fetchMock.mock.calls[0] as unknown as [string, RequestInit];
    expect(url).toBe('/api/personas/scout/avatar');
    expect(init.method).toBe('PUT');
    expect(JSON.parse(init.body as string)).toEqual({ source_path: 'artifacts/images/a.png' });
  });

  // Killed by: frontend/src/lib/avatarChoice.ts :: body: JSON.stringify({ source_path: previousPath, undo_of: changeId }),
  // Becomes: body: JSON.stringify({ source_path: previousPath }),
  // Killed by: frontend/src/lib/avatarChoice.ts :: : change(name, { method: 'DELETE' }, `?undo_of=${changeId}`);
  // Becomes: : change(name, { method: 'DELETE' });
  it('undoes by putting back the kept picture, or by resetting, naming the change it takes back', async () => {
    const fetchMock = vi.fn(async () => answer(200, { status: 'ok' }));
    vi.stubGlobal('fetch', fetchMock);

    await undoAvatarChange('scout', '.uclone/personas/scout.prev.png', 6);
    await undoAvatarChange('scout', null, 9);

    const [first, second] = fetchMock.mock.calls as unknown as [string, RequestInit][];
    expect(first[1].method).toBe('PUT');
    expect(JSON.parse(first[1].body as string)).toEqual({ source_path: '.uclone/personas/scout.prev.png', undo_of: 6 });
    expect(second[0]).toBe('/api/personas/scout/avatar?undo_of=9');
    expect(second[1].method).toBe('DELETE');
  });

  // Killed by: frontend/src/lib/avatarChoice.ts :: return known === undefined || known <= changeId;
  // Becomes: return known === undefined || known < changeId;
  it('offers an Undo until the listing shows a later change to that clone (#1809)', () => {
    expect(undoStillOffered(3, 'scout', { scout: 3 })).toBe(true);
    // The listing not read again yet still says the change before this one.
    expect(undoStillOffered(3, 'scout', { scout: 2 })).toBe(true);
    expect(undoStillOffered(3, 'scout', { scout: 4 })).toBe(false);
    expect(undoStillOffered(3, 'scout', { critic: 9 })).toBe(true);
    // A change the runtime gave no id cannot be checked, so it is not offered.
    expect(undoStillOffered(null, 'scout', undefined)).toBe(false);
  });

  // Killed by: frontend/src/lib/avatarChoice.ts :: if (typeof persona.avatar_change_id === 'number') ids[persona.name] = persona.avatar_change_id;
  // Becomes: ids[persona.name] = 0;
  it('reads each clone`s latest change from the listing', () => {
    expect(avatarChangeIdsOf([{ name: 'scout', avatar_change_id: 5 }, { name: 'critic' }])).toEqual({ scout: 5 });
    expect(avatarChangeIdsOf(undefined)).toEqual({});
  });

  // Killed by: frontend/src/lib/avatarChoice.ts :: status === 404 ? 'noClone' : status === 422 || status === 415 ? 'refused' : 'failed';
  // Becomes: 'failed';
  it('names why a change failed in its own terms, never with the server`s text', async () => {
    const statuses = [404, 422, 415, 500];
    let i = 0;
    vi.stubGlobal(
      'fetch',
      vi.fn(async () => answer(statuses[i++], { detail: 'PersonaAvatarError: /abs/path' })),
    );
    const failures = [];
    for (let n = 0; n < statuses.length; n += 1) failures.push(await resetAvatar('scout'));
    expect(failures).toEqual([
      { ok: false, failure: 'noClone' },
      { ok: false, failure: 'refused' },
      { ok: false, failure: 'refused' },
      { ok: false, failure: 'failed' },
    ]);
  });

  // Killed by: frontend/src/lib/avatarChoice.ts :: return FAILURE_OF_CODE[code];
  // Becomes: return 'refused';
  it('tells each refusal apart by the reason the runtime gives (#1780)', async () => {
    const codes = ['no_clone', 'not_an_image', 'too_large', 'not_saved', 'no_workspace', 'outside_workspace', 'no_file', 'stale_change', 'new_reason'];
    let i = 0;
    vi.stubGlobal(
      'fetch',
      vi.fn(async () => answer(i === 0 ? 404 : 422, { detail: 'x', code: codes[i++] })),
    );
    const failures = [];
    for (let n = 0; n < codes.length; n += 1) failures.push((await resetAvatar('scout')) as { failure: string });
    expect(failures.map((f) => f.failure)).toEqual([
      'noClone',
      'refused',
      'tooLarge',
      'notSaved',
      'noWorkspace',
      'outsideWorkspace',
      'noFile',
      'staleChange',
      // A reason this head does not know yet falls back to the status.
      'refused',
    ]);
  });

  // Killed by: frontend/src/lib/avatarChoice.ts :: return { ok: false, failure: 'unreachable' };
  // Becomes: return { ok: false, failure: 'failed' };
  it('tells an unreachable runtime from a refusal', async () => {
    vi.stubGlobal('fetch', vi.fn(async () => Promise.reject(new TypeError('Failed to fetch'))));
    expect(await resetAvatar('scout')).toEqual({ ok: false, failure: 'unreachable' });
  });

  // Killed by: frontend/src/lib/avatarChoice.ts :: headers: { 'Content-Type': uploadTypeOf(file) ?? 'application/octet-stream' },
  // Becomes: headers: {},
  it('uploads a file under its picture type, even when the browser gave none', async () => {
    const fetchMock = vi.fn(async () => answer(200, { status: 'ok' }));
    vi.stubGlobal('fetch', fetchMock);

    await uploadAvatar('scout', new File(['x'], 'me.webp', { type: '' }));

    const [, init] = fetchMock.mock.calls[0] as unknown as [string, RequestInit];
    expect(init.method).toBe('PUT');
    expect((init.headers as Record<string, string>)['Content-Type']).toBe('image/webp');
  });
});

describe('what can be uploaded', () => {
  // Killed by: frontend/src/lib/avatarChoice.ts :: if (file.size > MAX_UPLOAD_BYTES) return 'tooLarge';
  // Becomes: if (false) return 'tooLarge';
  it('takes PNG, JPEG and WebP up to the runtime`s limit, and nothing else', () => {
    expect(uploadProblem({ name: 'a.png', type: 'image/png', size: 10 })).toBeNull();
    expect(uploadProblem({ name: 'a.JPG', type: '', size: 10 })).toBeNull();
    expect(uploadProblem({ name: 'a.gif', type: 'image/gif', size: 10 })).toBe('wrongType');
    expect(uploadProblem({ name: 'a.svg', type: 'image/svg+xml', size: 10 })).toBe('wrongType');
    expect(uploadProblem({ name: 'a.png', type: 'image/png', size: MAX_UPLOAD_BYTES + 1 })).toBe('tooLarge');
  });
});
