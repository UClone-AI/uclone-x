/**
 * A failed request's cause, for a surface someone who does not read code is shown (#1436).
 *
 * The same rule `roomFailureReason` applies to the room notices, for the Settings requests
 * that are not `useApiRead` reads: only the Core's own `detail` string is passed through. A
 * browser's transport message ("Failed to fetch", "Load failed"), a status line, a parser's
 * complaint about a body that was not JSON and any other exception's text are replaced by a
 * fixed sentence the caller chooses, rather than shown (P0). The caller logs the raw error, so
 * the cause is kept for whoever reads the console (P6).
 */
/**
 * A non-2xx answer, the Core's own `detail` string if it gave one, and its `code` if it gave
 * one: a stable name a surface may render in the reader's language instead of `detail`.
 */
export class CoreFailure extends Error {
  constructor(
    readonly status: number,
    readonly coreDetail: string | null,
    readonly coreCode: string | null = null,
  ) {
    super(coreDetail ?? `HTTP ${status}`);
    this.name = 'CoreFailure';
  }
}

/** A non-empty string field of a JSON object body, or `null`. */
const textField = (body: unknown, key: string): string | null => {
  if (typeof body !== 'object' || body === null || !(key in body)) return null;
  const value = (body as Record<string, unknown>)[key];
  return typeof value === 'string' && value.trim() !== '' ? value : null;
};

/** The failure a non-2xx answer stands for, with its `detail` and `code` read from the body. */
export async function failureOf(res: Response): Promise<CoreFailure> {
  let body: unknown = null;
  try {
    body = await res.json();
  } catch {
    // A body that is not JSON carries no detail; the status still says what happened.
  }
  return new CoreFailure(res.status, textField(body, 'detail'), textField(body, 'code'));
}

const asSentence = (text: string): string => {
  const trimmed = text.trim();
  return /[.!?…]$/.test(trimmed) ? trimmed : `${trimmed}.`;
};

/** The Core's own `detail` carried by `err`, as a sentence, or `null` when it gave none. */
export function coreReason(err: unknown): string | null {
  return err instanceof CoreFailure && err.coreDetail !== null ? asSentence(err.coreDetail) : null;
}

/**
 * What to show for `err`: the Core's `detail` when `err` carries one, `fallback` otherwise.
 * Always ends a sentence, so copy can follow it.
 */
export function plainFailure(err: unknown, fallback: string): string {
  return coreReason(err) ?? asSentence(fallback);
}
