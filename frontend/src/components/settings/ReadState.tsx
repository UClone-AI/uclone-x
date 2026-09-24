import React from 'react';
import { AlertTriangle, RefreshCw } from 'lucide-react';

/**
 * A Settings section's read, still out (#1369).
 *
 * Said in words, so the time before the answer never looks like the answer: not a heading
 * with nothing under it, not a report "not read", not a scorecard of zeros.
 */
export const ReadLoading: React.FC<{ testId: string; children: React.ReactNode }> = ({
  testId,
  children,
}) => (
  <p role="status" data-testid={testId} className="text-[11px] text-slate-500">
    {children}
  </p>
);

/**
 * A Settings section's read that failed: what failed, its cause, and a way to read again
 * without closing Settings (#1369).
 *
 * The caller chooses the cause's words. Diagnostics passes the read's technical message and
 * it is set in monospace; a section every user sees passes a plain sentence with `plain`.
 *
 * The button stays on screen while the retry is out, disabled and saying so, so the cause of
 * the last failure is not replaced by a blank while the next answer is awaited.
 */
export const ReadFailure: React.FC<{
  testId: string;
  what: string;
  cause: string;
  onRetry: () => void;
  retrying: boolean;
  plain?: boolean;
}> = ({ testId, what, cause, onRetry, retrying, plain = false }) => (
  <div
    role="alert"
    data-testid={testId}
    className="p-3 rounded-xl border border-rose-800/80 bg-rose-950/50 space-y-2"
  >
    <div className="flex items-center gap-2 text-xs text-rose-200">
      <AlertTriangle className="w-4 h-4 text-rose-400 shrink-0" />
      <span>{what}</span>
    </div>
    <p className="text-[11px] text-rose-300/80">
      Reason: <span className={plain ? 'break-words' : 'font-mono break-all'}>{cause}</span>
    </p>
    <button
      type="button"
      onClick={onRetry}
      disabled={retrying}
      className="inline-flex items-center gap-1.5 text-[11px] px-2.5 py-1 rounded-lg bg-slate-800 text-slate-200 hover:bg-slate-700 disabled:opacity-60"
    >
      <RefreshCw className="w-3 h-3" />
      {retrying ? 'Trying again…' : 'Try again'}
    </button>
  </div>
);
