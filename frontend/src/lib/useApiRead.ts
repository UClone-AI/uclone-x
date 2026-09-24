import { useCallback, useEffect, useRef, useState } from 'react';

/**
 * What kind of failure a read met, so a surface for someone who does not read the code can
 * say it in its own plain words instead of printing a transport message (#1369).
 *
 * - `unreachable`: the request never got an answer (`fetch` itself rejected).
 * - `status`: the runtime answered, with a failure status.
 * - `unreadable`: the runtime answered with success, but not with JSON this head can read.
 */
export type ReadFaultKind = 'unreachable' | 'status' | 'unreadable';

export interface ReadFault {
  kind: ReadFaultKind;
  /**
   * The runtime's own explanation: the JSON `detail` string of a failed answer, or `null` when
   * it gave none (a plain-text 500, an unreachable runtime, a `detail` that is not a string).
   */
  detail: string | null;
}

/** One `GET` of a JSON route, and what became of it. */
export interface ApiRead<T> {
  /** The last body the route answered with, or `null` before the first answer. */
  data: T | null;
  /**
   * Why the last read failed, in technical words (`HTTP 503`, `Failed to fetch`), or `null`
   * when it did not. For developer surfaces; a surface every user sees reads `fault`.
   *
   * Kept apart from `data` so a failure cannot render as the absence it would otherwise look
   * like -- an empty skill catalogue, an ACP report "not read yet" (P6). A later read that
   * succeeds clears it.
   */
  error: string | null;
  /** The same failure, classified, with the runtime's own `detail` if it gave one. */
  fault: ReadFault | null;
  /** Whether the newest read is still out. An older read answering does not end it. */
  loading: boolean;
  reload: () => void;
}

const messageOf = (err: unknown): string => (err instanceof Error ? err.message : String(err));

/** The `detail` string of a failed answer's JSON body, if it has one. */
export const detailOf = async (res: Response): Promise<string | null> => {
  try {
    const body: unknown = await res.json();
    if (typeof body === 'object' && body !== null && 'detail' in body) {
      const { detail } = body as { detail: unknown };
      if (typeof detail === 'string' && detail.trim() !== '') return detail;
    }
  } catch {
    // A body that is not JSON carries no detail; the status still says what happened.
  }
  return null;
};

type Outcome<T> =
  | { ok: true; body: T }
  | { ok: false; error: string; fault: ReadFault; cause: unknown };

async function readOnce<T>(url: string): Promise<Outcome<T>> {
  let res: Response;
  try {
    res = await fetch(url);
  } catch (err) {
    return { ok: false, error: messageOf(err), fault: { kind: 'unreachable', detail: null }, cause: err };
  }
  if (!res.ok) {
    const detail = await detailOf(res);
    const error = `HTTP ${res.status}`;
    return { ok: false, error, fault: { kind: 'status', detail }, cause: error };
  }
  try {
    return { ok: true, body: (await res.json()) as T };
  } catch (err) {
    return { ok: false, error: messageOf(err), fault: { kind: 'unreadable', detail: null }, cause: err };
  }
}

/**
 * Read `url` once on mount, and again on `reload()`.
 *
 * For the Settings sections that moved out of the dock (#1358): each reads its own route
 * when it is shown rather than `App` reading every route at start-up for a surface that may
 * never open, the way `DiagnosticsPanel` already reads its own.
 *
 * Only the newest read may write (#1369). Reads can overlap -- a retry pressed while the first
 * read is still out -- and a response can arrive after a newer one; without the guard an older
 * answer, or an older failure, would overwrite what the newest read said. Unmounting retires
 * every read still out, so none writes to a section that is no longer shown.
 */
export function useApiRead<T>(url: string): ApiRead<T> {
  const [data, setData] = useState<T | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [fault, setFault] = useState<ReadFault | null>(null);
  const [loading, setLoading] = useState<boolean>(true);
  const latest = useRef(0);

  const reload = useCallback(async () => {
    const seq = ++latest.current;
    setLoading(true);
    try {
      const outcome = await readOnce<T>(url);
      if (outcome.ok) {
        if (seq !== latest.current) return;
        setData(outcome.body);
        setError(null);
        setFault(null);
      } else {
        if (latest.current !== seq) return;
        console.error(`Failed to read ${url}:`, outcome.cause);
        setError(outcome.error);
        setFault(outcome.fault);
      }
    } finally {
      if (seq === latest.current) setLoading(false);
    }
  }, [url]);

  useEffect(() => {
    void reload();
    return () => {
      latest.current += 1;
    };
  }, [reload]);

  return { data, error, fault, loading, reload: () => void reload() };
}
