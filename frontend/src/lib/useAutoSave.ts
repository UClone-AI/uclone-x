import { useCallback, useEffect, useRef, useState } from 'react';

/**
 * Saving one Settings field as the person edits it (settings-single-source.md §4.1).
 *
 * A choice from a list is saved when it is made (`commit`). Typing is recorded with `edit`
 * and saved when the field is left (`flush`), or after `delayMs` when one is given. Only
 * the field's own value is sent, so a problem with one field cannot refuse another.
 *
 * A failed save puts the last saved value back on the form and says why under the field:
 * the form never shows a value the app is not using.
 */
export type AutoSaveStatus =
  | { kind: 'idle' }
  | { kind: 'saving' }
  | { kind: 'saved' }
  | { kind: 'error'; message: string };

/** The flush of every auto-saved field in one form, called when the form closes. */
export type FlushRegistry = { current: Set<() => void> };

export interface AutoSaveOptions<T> {
  /** The value the app holds now: what an unchanged edit is compared with, and restored on failure. */
  saved: T;
  /** Sends `value`. Rejects when it was not saved. */
  save: (value: T) => Promise<void>;
  /** Puts a value back on the form. */
  restore: (value: T) => void;
  /** The plain-language sentence for a failed save. */
  describeFailure: (err: unknown) => string;
  /** Saves typing this long after the last keystroke; omitted, typing is saved on `flush` only. */
  delayMs?: number;
  /** Where this field's `flush` is registered, so closing the form saves an edit in progress. */
  flushers?: FlushRegistry;
  /** Whether two values are the same setting. `Object.is` when omitted. */
  same?: (a: T, b: T) => boolean;
}

export interface AutoSave<T> {
  status: AutoSaveStatus;
  /** Records a typed value; it is saved on `flush`, or after `delayMs`. */
  edit: (value: T) => void;
  /** Saves `value` now, unless it is what is already saved. */
  commit: (value: T) => void;
  /** Saves the value recorded by `edit`, if there is one. */
  flush: () => void;
  /** Drops an edit not yet sent: the field was set by something else, such as a connection. */
  cancel: () => void;
}

export function useAutoSave<T>(options: AutoSaveOptions<T>): AutoSave<T> {
  const [status, setStatus] = useState<AutoSaveStatus>({ kind: 'idle' });
  const opts = useRef(options);
  opts.current = options;
  const pending = useRef<{ value: T } | null>(null);
  const timer = useRef<ReturnType<typeof setTimeout> | null>(null);
  // Only the latest save may report: an older one answering late must not say "Saved" over it.
  const seq = useRef(0);
  // The value last known saved. Set on success before the caller's state catches up, so a
  // failure straight after restores what was really saved.
  const lastSaved = useRef<{ value: T } | null>(null);
  const savedNow = (): T => (lastSaved.current ? lastSaved.current.value : opts.current.saved);
  // A new read of the settings is the new truth.
  useEffect(() => {
    lastSaved.current = null;
  }, [options.saved]);

  const clearTimer = () => {
    if (timer.current !== null) {
      clearTimeout(timer.current);
      timer.current = null;
    }
  };

  const commit = useCallback((value: T) => {
    clearTimer();
    pending.current = null;
    const same = opts.current.same ?? Object.is;
    if (same(value, savedNow())) return;
    const mine = ++seq.current;
    setStatus({ kind: 'saving' });
    opts.current.save(value).then(
      () => {
        lastSaved.current = { value };
        if (mine === seq.current) setStatus({ kind: 'saved' });
      },
      (err: unknown) => {
        if (mine !== seq.current) return;
        opts.current.restore(savedNow());
        setStatus({ kind: 'error', message: opts.current.describeFailure(err) });
      },
    );
  }, []);

  const flush = useCallback(() => {
    const edited = pending.current;
    if (edited) commit(edited.value);
  }, [commit]);

  const edit = useCallback(
    (value: T) => {
      pending.current = { value };
      setStatus((prev) => (prev.kind === 'idle' ? prev : { kind: 'idle' }));
      clearTimer();
      const delay = opts.current.delayMs;
      if (delay !== undefined) timer.current = setTimeout(flush, delay);
    },
    [flush],
  );

  const cancel = useCallback(() => {
    clearTimer();
    pending.current = null;
    seq.current += 1;
    setStatus({ kind: 'idle' });
  }, []);

  const registry = options.flushers;
  useEffect(() => {
    if (!registry) return;
    const set = registry.current;
    set.add(flush);
    return () => {
      set.delete(flush);
    };
  }, [registry, flush]);

  useEffect(() => clearTimer, []);

  return { status, edit, commit, flush, cancel };
}
