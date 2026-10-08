import { afterEach, describe, expect, it, vi } from 'vitest';
import {
  addMcpServer,
  describeDetail,
  importMcpServers,
  reconnectMcpServer,
  removeMcpServer,
  setMcpServerEnabled,
} from './mcpServersApi';

const answer = (body: unknown, status = 200) =>
  new Response(JSON.stringify(body), { status, headers: { 'Content-Type': 'application/json' } });

afterEach(() => {
  vi.unstubAllGlobals();
});

describe('describeDetail (#1460)', () => {
  it('returns plain string detail as is', () => {
    expect(describeDetail('Invalid server configuration', 400)).toBe('Invalid server configuration');
  });

  it('joins FastAPI validation errors', () => {
    const detail = [{ msg: 'field required' }, { msg: 'invalid url' }];
    expect(describeDetail(detail, 422)).toBe('field required; invalid url');
  });

  it('returns plain fallback when detail is absent or empty without status code', () => {
    expect(describeDetail(null, 500)).toBe('The server refused the request.');
    expect(describeDetail(undefined, 404)).toBe('The server refused the request.');
    expect(describeDetail('', 502)).toBe('The server refused the request.');
    expect(describeDetail([], 400)).toBe('The server refused the request.');
  });
});

describe('mcp mutations plain error copy (#1460)', () => {
  it('returns plain copy when network fetch rejects', async () => {
    vi.stubGlobal('fetch', vi.fn().mockRejectedValue(new TypeError('Failed to fetch')));

    const result = await addMcpServer({ name: 'test', transport: 'http', url: 'http://localhost:8000' });
    expect(result).toEqual({ ok: false, message: 'The app could not be reached.' });
  });

  it('returns plain fallback on bodyless error without HTTP status', async () => {
    vi.stubGlobal('fetch', vi.fn().mockResolvedValue(new Response('', { status: 500 })));

    const result = await removeMcpServer('test');
    expect(result).toEqual({ ok: false, message: 'The server refused the request.' });
  });

  it('surfaces Core detail when provided', async () => {
    vi.stubGlobal('fetch', vi.fn().mockResolvedValue(answer({ detail: 'No server named "test"' }, 404)));

    const result = await reconnectMcpServer('test');
    expect(result).toEqual({ ok: false, message: 'No server named "test"' });
  });

  it('handles setMcpServerEnabled and importMcpServers errors plainly', async () => {
    vi.stubGlobal('fetch', vi.fn().mockRejectedValue(new Error('connection reset')));

    const enableResult = await setMcpServerEnabled('test', false);
    expect(enableResult).toEqual({ ok: false, message: 'The app could not be reached.' });

    const importResult = await importMcpServers('{}');
    expect(importResult).toEqual({ ok: false, message: 'The app could not be reached.' });
  });
});
