/**
 * Confirming that this window is the person's, for the decisions only a person may make (#1589).
 *
 * The local API answers any program on this computer, the model's own shell included, so the
 * story view's Approve and Reject are accepted only from a window the server itself opened. The
 * server opens it at `/#pair=<code>`: the part after `#` is never sent to a server, so no
 * program reads the code from a request. This page takes the code out of the address bar and
 * spends it once (`POST /api/person/pair`); the answer is a cookie script cannot read, which the
 * browser sends with every later decision on its own.
 *
 * A window opened any other way (a bookmark, a typed address) is not confirmed; it asks the
 * server to open one that is (`POST /api/person/window`).
 */
import { failureOf } from './coreFailure';

const PAIR_PREFIX = '#pair=';

/**
 * Spend the pairing code this window was opened with, if it was opened with one.
 *
 * Removes the code from the address bar first, so a reload or a copied address does not carry
 * it. Resolves either way: a window that could not be confirmed is told so when it decides.
 */
export async function confirmThisWindow(win: Window = window): Promise<void> {
  const { hash, pathname, search } = win.location;
  if (!hash.startsWith(PAIR_PREFIX)) return;
  const code = hash.slice(PAIR_PREFIX.length);
  win.history.replaceState(win.history.state, '', `${pathname}${search}`);
  try {
    const res = await fetch('/api/person/pair', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ code }),
    });
    if (!res.ok) console.error('This window could not be confirmed', await failureOf(res));
  } catch (err) {
    console.error('This window could not be confirmed', err);
  }
}

/**
 * Confirm this window now, and again whenever its address gains a code.
 *
 * A browser asked to open the dashboard's address may reuse a tab already showing it; only
 * the part after `#` then changes and the page does not load again, so the code arrives as a
 * `hashchange`.
 */
export function confirmWindowsOpenedHere(win: Window = window): void {
  win.addEventListener('hashchange', () => void confirmThisWindow(win));
  void confirmThisWindow(win);
}

/** Ask the server to open a confirmed window in this computer's browser. */
export async function openConfirmedWindow(): Promise<void> {
  const res = await fetch('/api/person/window', { method: 'POST' });
  if (!res.ok) throw await failureOf(res);
}
