import { afterEach, describe, expect, it, vi } from 'vitest';
import { fireEvent, render, screen, waitFor } from '@testing-library/react';
import { UseSystemDefault } from './UseSystemDefault';
import { en } from '../../i18n/en';
import { fmt } from '../../i18n/format';
import type { RoomProviderFailure } from '../../types';

const T = en.gateway.clone;

const FAILURE: RoomProviderFailure = {
  kind: 'model_unavailable',
  message: 'alpha is set to use its own model, box/own-model, and it could not be used.',
  retryable: false,
  clone: 'alpha',
  model_ref: 'box/own-model',
  action: 'use_system_default',
};

const PERSONA = {
  id: 'c-1',
  name: 'alpha',
  role: 'Helper',
  system_prompt: 'You help.',
  model_name: 'box/own-model',
  fast_model: 'mock/quick',
  image_model: 'auto',
  allowed_tools: ['read_file'],
};

type Call = { url: string; method: string; body?: Record<string, unknown> };

const serve = (put: (body: Record<string, unknown>) => { ok: boolean; status: number; body: unknown }) => {
  const calls: Call[] = [];
  vi.stubGlobal(
    'fetch',
    vi.fn(async (url: string, init: RequestInit = {}) => {
      const method = init.method ?? 'GET';
      const body = init.body ? (JSON.parse(String(init.body)) as Record<string, unknown>) : undefined;
      calls.push({ url, method, body });
      if (method === 'GET' && url === '/api/clones/alpha') {
        return { ok: true, status: 200, json: async () => ({ status: 'ok', persona: PERSONA }) };
      }
      if (method === 'PUT' && url === '/api/clones/c-1') {
        const answer = put(body ?? {});
        return { ok: answer.ok, status: answer.status, json: async () => answer.body };
      }
      return { ok: false, status: 404, json: async () => ({ detail: 'not here' }) };
    }),
  );
  return calls;
};

afterEach(() => vi.unstubAllGlobals());

describe('UseSystemDefault', () => {
  // Killed by: frontend/src/lib/personasApi.ts ::   for (const slot of named.length > 0 ? named : (['model_name'] as const)) draft[slot] = null;
  // Becomes:   for (const slot of [] as ('model_name')[]) draft[slot] = null;
  it('clears only the slot naming the failed model, sends every other field back, and says so', async () => {
    const onChanged = vi.fn();
    const calls = serve((body) => ({ ok: true, status: 200, body: { persona: { ...PERSONA, ...body } } }));
    render(<UseSystemDefault failure={FAILURE} label="Alpha" testId="use-default" onChanged={onChanged} />);

    fireEvent.click(screen.getByTestId('use-default'));

    expect(await screen.findByTestId('use-default-done')).toHaveTextContent(fmt(T.usedDefault, { name: 'Alpha' }));
    const put = calls.find((c) => c.method === 'PUT');
    expect(put?.body).toMatchObject({
      name: 'alpha',
      model_name: null,
      fast_model: 'mock/quick',
      image_model: 'auto',
      allowed_tools: ['read_file'],
    });
    expect(put?.body).not.toHaveProperty('id');
    expect(onChanged).toHaveBeenCalledTimes(1);
  });

  // Killed by: frontend/src/components/rooms/UseSystemDefault.tsx ::   if (failure.action !== 'use_system_default' || !clone || !ref) return null;
  // Becomes:   if (!clone || !ref) return null;
  it('is not offered when the Core names no action', () => {
    serve(() => ({ ok: true, status: 200, body: {} }));
    render(<UseSystemDefault failure={{ ...FAILURE, action: null }} label="Alpha" testId="use-default" />);
    expect(screen.queryByTestId('use-default')).toBeNull();
  });

  it('keeps the button and says why when the Core refuses the change', async () => {
    serve(() => ({ ok: false, status: 422, body: { detail: 'There is no connection called box.' } }));
    render(<UseSystemDefault failure={FAILURE} label="Alpha" testId="use-default" />);

    fireEvent.click(screen.getByTestId('use-default'));

    await waitFor(() => expect(screen.getByRole('alert')).toHaveTextContent('There is no connection called box.'));
    expect(screen.getByTestId('use-default')).toBeTruthy();
    expect(screen.queryByTestId('use-default-done')).toBeNull();
  });
});
