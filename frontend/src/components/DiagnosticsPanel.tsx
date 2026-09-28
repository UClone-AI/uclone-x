import React, { useCallback, useEffect, useRef, useState } from 'react';
import { AlertTriangle, ExternalLink, ShieldCheck, Trash2 } from 'lucide-react';
import { CoreFailure, failureOf } from '../lib/coreFailure';
import { openConfirmedWindow } from '../lib/person';
import { plural, useCopy } from '../i18n';

/**
 * Local failure recording, and the one-click way to report what was recorded.
 *
 * The dashboard is where someone who never opens a terminal meets a failure, so
 * it is where the question has to be asked. Nothing here uploads: the report is
 * rendered by the backend from a local journal, shown in full, and the "report"
 * button opens a pre-filled GitHub issue that the person still submits.
 *
 * The answer is the person's consent, so the Core takes it only from a window it opened
 * itself (#1589, `lib/person.ts`). A window opened any other way is refused, says so, and
 * offers to open a confirmed one, as the story view does.
 */

type ConsentState = 'granted' | 'denied' | 'unasked';

interface ConsentPayload {
  state: ConsentState;
  /** Why the preference could not be read, if it could not. */
  error: string | null;
  journal: string;
}

interface ReportPayload {
  state: ConsentState;
  /** False when the journal exists but could not be read. Not the same as `count: 0`. */
  available: boolean;
  error: string | null;
  count: number;
  distinct: number;
  unreadable_lines: number;
  /** Why new failures are not being kept, if they are not. Not the same as a read failure. */
  recording_blocked: string | null;
  title: string;
  body: string;
  issue_url: string | null;
  search_url: string | null;
}

export const DiagnosticsPanel: React.FC = () => {
  const [state, setState] = useState<ConsentState>('unasked');
  const [journal, setJournal] = useState<string>('');
  const [report, setReport] = useState<ReportPayload | null>(null);
  const [busy, setBusy] = useState(false);
  const [fetchError, setFetchError] = useState<string | null>(null);
  const [consentError, setConsentError] = useState<string | null>(null);
  const [showBody, setShowBody] = useState(false);
  // The Core refused the answer from this window: it was not opened by the Core (#1589).
  const [unconfirmed, setUnconfirmed] = useState(false);
  const [windowNote, setWindowNote] = useState<string | null>(null);
  const { settings, diagnostics } = useCopy();
  const failure = settings.failure;
  const copy = diagnostics.panel;
  // Read through a ref so a language switch does not re-run the mount read and clear its error.
  const failureRef = useRef(failure);
  failureRef.current = failure;

  const refresh = useCallback(async (clearError = false) => {
    try {
      if (clearError) {
        setFetchError(null);
      }
      const consentRes = await fetch('/api/diagnostics/consent');
      if (!consentRes.ok) {
        throw await failureOf(consentRes);
      }
      const consent = (await consentRes.json()) as ConsentPayload;
      setState(consent.state);
      setJournal(consent.journal);
      setConsentError(consent.error ?? null);

      const reportRes = await fetch('/api/diagnostics/report');
      if (!reportRes.ok) {
        throw await failureOf(reportRes);
      }
      setReport((await reportRes.json()) as ReportPayload);
    } catch (err) {
      // Checked, not assumed. Without the `ok` checks a 500 or a 403 left
      // `available` undefined, `=== false` false, and the calm "nothing
      // recorded" state rendered over a backend that was failing — the same
      // substitution the backend was just fixed to stop making, one layer up.
      console.error('Failed to read diagnostics state:', err);
      setFetchError(failureRef.current.diagnosticsRead);
      setReport(null);
    }
  }, []);

  useEffect(() => {
    void refresh(true);
  }, [refresh]);

  const decide = async (collect: boolean) => {
    setBusy(true);
    // Cleared when the user acts, not when a read happens to succeed. The
    // previous version cleared it on mount only, so a failed save left its
    // banner on screen beside "Recording on" after the next attempt worked.
    setFetchError(null);
    setUnconfirmed(false);
    setWindowNote(null);
    try {
      const res = await fetch('/api/diagnostics/consent', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ collect }),
      });
      // Checked, like the reads. Without this an unwritable diagnostics
      // directory made the POST 500, `refresh()` then succeeded and cleared
      // the error, and the panel showed "Currently off." after the user had
      // just asked for it to be on — the answer silently not taken.
      if (!res.ok) {
        throw await failureOf(res);
      }
      await refresh();
    } catch (err) {
      console.error('Failed to record the diagnostics choice:', err);
      const refusedHere = err instanceof CoreFailure && err.status === 403;
      setUnconfirmed(refusedHere);
      setFetchError(refusedHere ? copy.unconfirmed : failure.diagnosticsChoice);
    } finally {
      setBusy(false);
    }
  };

  const openWindow = async () => {
    try {
      await openConfirmedWindow();
      setWindowNote(copy.windowOpened);
    } catch (err) {
      console.error('Diagnostics: no confirmed window opened', err);
      const tooSoon = err instanceof CoreFailure && err.status === 429;
      setWindowNote(tooSoon ? copy.windowTooSoon : copy.windowFailed);
    }
  };

  const clear = async () => {
    if (!window.confirm(copy.confirmDelete)) {
      return;
    }
    setBusy(true);
    setFetchError(null);
    try {
      const res = await fetch('/api/diagnostics/report', { method: 'DELETE' });
      if (!res.ok) {
        throw await failureOf(res);
      }
      await refresh();
    } catch (err) {
      console.error('Failed to delete the recorded failures:', err);
      setFetchError(failure.diagnosticsClear);
    } finally {
      setBusy(false);
    }
  };

  const recorded = report?.count ?? 0;
  const unavailable = report !== null && report.available === false;

  return (
    <section className="space-y-3" aria-labelledby="diagnostics-heading">
      <div className="flex items-center gap-2">
        <ShieldCheck className="w-4 h-4 text-cyan-400" />
        <h3 id="diagnostics-heading" className="text-sm font-semibold text-white">
          {copy.title}
        </h3>
      </div>

      <p className="text-xs text-slate-400 leading-relaxed">
        {copy.introBefore}
        <strong>{copy.introEmphasis}</strong>
        {copy.introAfter}
      </p>

      {state === 'granted' ? (
        <div className="flex flex-wrap items-center gap-2">
          <span className="text-xs px-2 py-1 rounded-lg bg-emerald-950/50 border border-emerald-800/80 text-emerald-200">
            {copy.recordingOn}
          </span>
          <button
            type="button"
            disabled={busy}
            onClick={() => void decide(false)}
            className="text-xs px-2.5 py-1 rounded-lg border border-slate-700 text-slate-300 hover:bg-slate-800 disabled:opacity-50"
          >
            {copy.turnOff}
          </button>
        </div>
      ) : (
        <div className="flex flex-wrap items-center gap-2">
          <button
            type="button"
            disabled={busy}
            onClick={() => void decide(true)}
            className="text-xs px-2.5 py-1 rounded-lg bg-cyan-900/70 border border-cyan-700 text-cyan-100 hover:bg-cyan-800 disabled:opacity-50"
          >
            {copy.recordLocally}
          </button>
          <span className="text-xs text-slate-500">
            {state === 'denied' ? copy.offByChoice : copy.off}
          </span>
        </div>
      )}

      {consentError !== null && (
        <div className="p-3 rounded-xl border border-amber-800/80 bg-amber-950/40 space-y-1">
          <div className="flex items-center gap-2 text-xs text-amber-200">
            <AlertTriangle className="w-4 h-4 text-amber-400 shrink-0" />
            <span>{copy.preferenceUnreadable}</span>
          </div>
          <p className="text-[11px] text-amber-300/80 font-mono break-all">{consentError}</p>
        </div>
      )}

      {journal && (
        <p className="text-[11px] text-slate-500 font-mono break-all">{journal}</p>
      )}

      {fetchError !== null && (
        <div
          role="alert"
          data-testid="diagnostics-failure"
          className="p-3 rounded-xl border border-rose-800/80 bg-rose-950/50 space-y-1"
        >
          <div className="flex items-center gap-2 text-xs text-rose-200">
            <AlertTriangle className="w-4 h-4 text-rose-400 shrink-0" />
            <span>{fetchError}</span>
          </div>
          {unconfirmed && (
            <button
              type="button"
              onClick={() => void openWindow()}
              data-testid="diagnostics-open-confirmed-window"
              className="text-xs px-2.5 py-1 rounded-lg border border-slate-700 text-slate-300 hover:bg-slate-800"
            >
              {copy.openConfirmed}
            </button>
          )}
          {windowNote !== null && <p className="text-[11px] text-slate-300">{windowNote}</p>}
        </div>
      )}

      {report?.recording_blocked != null && (
        <div className="p-3 rounded-xl border border-rose-800/80 bg-rose-950/50 space-y-1">
          <div className="flex items-center gap-2 text-xs text-rose-200">
            <AlertTriangle className="w-4 h-4 text-rose-400 shrink-0" />
            <span>{copy.recordingBlocked}</span>
          </div>
          <p className="text-[11px] text-rose-300/80 font-mono break-all">
            {report.recording_blocked}
          </p>
        </div>
      )}

      {unavailable && report && (
        <div className="p-3 rounded-xl border border-rose-800/80 bg-rose-950/50 space-y-1">
          <div className="flex items-center gap-2 text-xs text-rose-200">
            <AlertTriangle className="w-4 h-4 text-rose-400 shrink-0" />
            <span>{copy.journalUnreadable}</span>
          </div>
          <p className="text-[11px] text-rose-300/80 font-mono break-all">{report.error}</p>
          <p className="text-[11px] text-slate-400">
            {copy.journalUnreadableNote}
          </p>
        </div>
      )}

      {recorded > 0 && report && !unavailable && (
        <div className="p-3 rounded-xl border border-slate-800 bg-slate-950/60 space-y-2">
          <div className="flex items-center gap-2 text-xs text-amber-200">
            <AlertTriangle className="w-4 h-4 text-amber-400 shrink-0" />
            <span>{report.unreadable_lines > 0
                ? plural(copy.recordedWithUnreadable, recorded, {
                    distinct: report.distinct,
                    unreadable: report.unreadable_lines,
                  })
                : plural(copy.recorded, recorded, { distinct: report.distinct })}</span>
          </div>

          <div className="flex flex-wrap items-center gap-2">
            <button
              type="button"
              onClick={() => setShowBody((shown) => !shown)}
              className="text-xs px-2.5 py-1 rounded-lg border border-slate-700 text-slate-300 hover:bg-slate-800"
            >
              {showBody ? copy.hideReport : copy.showReport}
            </button>

            {report.issue_url ? (
              <a
                href={report.issue_url}
                target="_blank"
                rel="noreferrer"
                className="text-xs px-2.5 py-1 rounded-lg bg-cyan-900/70 border border-cyan-700 text-cyan-100 hover:bg-cyan-800 inline-flex items-center gap-1.5"
              >
                {copy.reportOnGithub} <ExternalLink className="w-3 h-3" />
              </a>
            ) : (
              <span className="text-xs text-slate-400">
                {copy.tooLong}
              </span>
            )}

            {report.search_url && (
              <a
                href={report.search_url}
                target="_blank"
                rel="noreferrer"
                className="text-xs text-slate-400 underline hover:text-slate-200"
              >
                {copy.alreadyReported}
              </a>
            )}

            <button
              type="button"
              disabled={busy}
              onClick={() => void clear()}
              className="text-xs px-2.5 py-1 rounded-lg border border-slate-700 text-slate-400 hover:bg-slate-800 inline-flex items-center gap-1.5 disabled:opacity-50"
            >
              <Trash2 className="w-3 h-3" /> {copy.delete}
            </button>
          </div>

          {showBody && (
            <textarea
              readOnly
              value={report.body}
              aria-label={copy.reportContents}
              className="w-full h-48 text-[11px] font-mono bg-slate-950 border border-slate-800 rounded-lg p-2 text-slate-300"
            />
          )}
        </div>
      )}
    </section>
  );
};
