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
import { detailOf } from './useApiRead';

/** A non-2xx answer, and the Core's own `detail` string if it gave one. */
export class CoreFailure extends Error {
  constructor(
    readonly status: number,
    readonly coreDetail: string | null,
  ) {
    super(coreDetail ?? `HTTP ${status}`);
    this.name = 'CoreFailure';
  }
}

/** The failure a non-2xx answer stands for, with its `detail` read from the body. */
export const failureOf = async (res: Response): Promise<CoreFailure> =>
  new CoreFailure(res.status, await detailOf(res));

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
