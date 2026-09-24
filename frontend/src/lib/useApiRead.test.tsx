import { describe, it, expect, vi, afterEach } from 'vitest';
import { act, renderHook, waitFor } from '@testing-library/react';
import { useApiRead } from './useApiRead';

type Answer = { ok: boolean; status: number; json: () => Promise<unknown> };

/** A fetch whose every call waits until the test answers it, in whatever order it likes. */
const heldFetch = () => {
  const pending: Array<(answer: Answer) => void> = [];
  const fetchMock = vi.fn(
    () =>
      new Promise<Answer>((resolve) => {
        pending.push(resolve);
      }),
  );
  vi.stubGlobal('fetch', fetchMock);
  const answer = (index: number, body: unknown, ok = true, status = 200) =>
    act(async () => {
      pending[index]({ ok, status, json: async () => body });
    });
  return { fetchMock, answer };
};

describe('useApiRead', () => {
  afterEach(() => vi.unstubAllGlobals());

  it('keeps the newer read when an older one answers after it', async () => {
    // Killed by: frontend/src/lib/useApiRead.ts :: if (seq !== latest.current) return;
    // Becomes:
    const { fetchMock, answer } = heldFetch();
    const { result } = renderHook(() => useApiRead<{ n: number }>('/api/thing'));
    await waitFor(() => expect(fetchMock).toHaveBeenCalledTimes(1));

    act(() => result.current.reload());
    await waitFor(() => expect(fetchMock).toHaveBeenCalledTimes(2));

    await answer(1, { n: 2 });
    expect(result.current.data).toEqual({ n: 2 });
    expect(result.current.loading).toBe(false);

    await answer(0, { n: 1 });
    expect(result.current.data).toEqual({ n: 2 });
    expect(result.current.loading).toBe(false);
  });

  it("does not let an older read's failure overwrite a newer read's answer", async () => {
    // Killed by: frontend/src/lib/useApiRead.ts :: if (latest.current !== seq) return;
    // Becomes:
    vi.spyOn(console, 'error').mockImplementation(() => {});
    const { fetchMock, answer } = heldFetch();
    const { result } = renderHook(() => useApiRead<{ n: number }>('/api/thing'));
    await waitFor(() => expect(fetchMock).toHaveBeenCalledTimes(1));

    act(() => result.current.reload());
    await waitFor(() => expect(fetchMock).toHaveBeenCalledTimes(2));

    await answer(1, { n: 2 });
    await answer(0, { detail: 'late' }, false, 500);
    expect(result.current.error).toBeNull();
    expect(result.current.data).toEqual({ n: 2 });
  });

  it('stays loading while the newer read is still out, though an older one has answered', async () => {
    // Killed by: frontend/src/lib/useApiRead.ts :: if (seq === latest.current) setLoading(false);
    // Becomes: setLoading(false);
    const { fetchMock, answer } = heldFetch();
    const { result } = renderHook(() => useApiRead<{ n: number }>('/api/thing'));
    await waitFor(() => expect(fetchMock).toHaveBeenCalledTimes(1));

    act(() => result.current.reload());
    await waitFor(() => expect(fetchMock).toHaveBeenCalledTimes(2));

    await answer(0, { n: 1 });
    expect(result.current.loading).toBe(true);
    expect(result.current.data).toBeNull();
  });

  it("keeps a failed answer's detail only when it is a sentence, never a validation dump", async () => {
    // Killed by: frontend/src/lib/useApiRead.ts :: if (typeof detail === 'string' && detail.trim() !== '') return detail;
    // Becomes: return detail as string;
    vi.spyOn(console, 'error').mockImplementation(() => {});
    const { fetchMock, answer } = heldFetch();
    const { result } = renderHook(() => useApiRead<{ n: number }>('/api/thing'));
    await waitFor(() => expect(fetchMock).toHaveBeenCalledTimes(1));

    const pydantic = [{ type: 'missing', loc: ['query', 'x'], msg: 'Field required', input: null }];
    await answer(0, { detail: pydantic }, false, 422);
    expect(result.current.fault).toEqual({ kind: 'status', detail: null });
    expect(result.current.error).toBe('HTTP 422');
  });

  it('tells a read that never got an answer from one whose answer could not be read', async () => {
    // Killed by: frontend/src/lib/useApiRead.ts :: fault: { kind: 'unreadable', detail: null }
    // Becomes: fault: { kind: 'unreachable', detail: null }
    vi.spyOn(console, 'error').mockImplementation(() => {});
    vi.stubGlobal(
      'fetch',
      vi
        .fn()
        .mockRejectedValueOnce(new TypeError('Failed to fetch'))
        .mockResolvedValueOnce({
          ok: true,
          status: 200,
          json: async () => {
            throw new SyntaxError('Unexpected token < in JSON');
          },
        }),
    );
    const { result } = renderHook(() => useApiRead<{ n: number }>('/api/thing'));
    await waitFor(() => expect(result.current.fault).toEqual({ kind: 'unreachable', detail: null }));
    expect(result.current.error).toBe('Failed to fetch');

    act(() => result.current.reload());
    await waitFor(() => expect(result.current.fault).toEqual({ kind: 'unreadable', detail: null }));
  });
});
