import React, { useEffect, useMemo, useState } from 'react';
import { Gauge } from 'lucide-react';
import { fmt, useCopy } from '../../i18n';
import { detailOf, useApiRead } from '../../lib/useApiRead';
import {
  PROVIDER_CONSOLES,
  USAGE_ENV_VARS,
  USAGE_PRESETS,
  USAGE_PRESET_IDS,
  USAGE_WINDOWS,
  compactTokens,
  percentUsed,
  isUsageReport,
  presetOf,
  type UsageLimitsValue,
  type UsagePresetId,
  type UsageReport,
  type UsageWindowStatus,
} from '../../lib/usage';
import { useAutoSave } from '../../lib/useAutoSave';
import { FieldStatus } from './FieldStatus';
import { ReadFailure, ReadLoading } from './ReadState';

/**
 * Settings → Usage: the user's limits on paid-model tokens (the token-gateway design
 * §4.5.1).
 *
 * Not behind developer mode: the limit is the user's spending decision, set in one place for
 * the whole installation. A bar per window, a preset picker whose Custom choice opens the three
 * numbers, the per-provider totals, and a line saying this counts tokens, not money.
 */
export const UsageSection: React.FC<{
  /** Called with the answer to a save that landed, so a sibling reading the same report can refresh. */
  onSaved?: (report: UsageReport) => void;
}> = ({ onSaved }) => {
  const read = useApiRead<unknown>('/api/usage');
  const { loading, reload } = read;
  const data = read.data === null || !isUsageReport(read.data) ? null : read.data;
  // An answer of another shape is a failed read, said as one (P6), not an empty report.
  const fault =
    read.fault ?? (read.data !== null && data === null ? { kind: 'unreadable' as const, detail: null } : null);
  const allCopy = useCopy();
  const copy = allCopy.usage;
  const [report, setReport] = useState<UsageReport | null>(null);
  const [choice, setChoice] = useState<UsagePresetId | 'custom'>('none');
  const [draft, setDraft] = useState<Record<string, string>>({});

  // A fresh read replaces the form: what is shown is what is saved.
  useEffect(() => {
    if (data === null) return;
    setReport(data);
    setChoice(presetOf(data.limits));
    setDraft(toDraft(data.limits));
  }, [data]);

  // Saved as it is changed (settings-single-source.md §4.1): a preset when it is picked, a
  // custom number when its box is left. A refused save puts the saved limits back.
  const savedLimits = report?.limits ?? null;
  const savedDraft = useMemo(() => (savedLimits === null ? {} : toDraft(savedLimits)), [savedLimits]);
  const autoSave = useAutoSave<Record<string, string>>({
    saved: savedDraft,
    save: async (limits) => {
      const res = await fetch('/api/usage/limits', {
        method: 'PUT',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(fromDraft(limits)),
      });
      if (!res.ok) throw new RefusedSave(await detailOf(res));
      const next = (await res.json()) as UsageReport;
      setReport(next);
      setChoice((current) => (current === 'custom' ? current : presetOf(next.limits)));
      setDraft(toDraft(next.limits));
      onSaved?.(next);
    },
    restore: (limits) => {
      setDraft(limits);
      // A refused custom number keeps the boxes open, so it can be corrected.
      setChoice((current) => (current === 'custom' || savedLimits === null ? current : presetOf(savedLimits)));
    },
    describeFailure: (err) =>
      [copy.saveFailed, err instanceof RefusedSave ? err.detail : null, allCopy.settings.autosave.restored]
        .filter(Boolean)
        .join(' '),
    same: (a, b) => JSON.stringify(fromDraft(a)) === JSON.stringify(fromDraft(b)),
  });

  const pick = (id: UsagePresetId | 'custom') => {
    setChoice(id);
    if (id === 'custom') return;
    const limits = toDraft(USAGE_PRESETS[id]);
    setDraft(limits);
    autoSave.commit(limits);
  };

  const consoles = consoleLinks(report);

  return (
    <div className="space-y-3" data-testid="settings-usage">
      <label className="text-xs font-semibold uppercase tracking-wider text-slate-400 flex items-center gap-1.5">
        <Gauge className="w-3.5 h-3.5 text-cyan-400" />
        {copy.title}
      </label>
      <p className="text-[11px] text-slate-500">{copy.intro}</p>

      {fault !== null && (
        <ReadFailure
          testId="settings-usage-error"
          what={copy.loadFailed}
          cause={fault.detail ?? allCopy.skills.plainCause[fault.kind]}
          plain
          onRetry={reload}
          retrying={loading}
        />
      )}
      {fault === null && report === null && (
        <ReadLoading testId="settings-usage-loading">{copy.loading}</ReadLoading>
      )}

      {report !== null && (
        <>
          <div className="space-y-2" data-testid="settings-usage-windows">
            {USAGE_WINDOWS.map((id) => {
              const status = report.windows.find((w) => w.window === id);
              return status ? (
                <WindowBar key={id} status={status} overridden={report.env_overrides.includes(id)} />
              ) : null;
            })}
          </div>

          <div className="space-y-2">
            <div
              role="radiogroup"
              aria-label={copy.limitLabel}
              className="flex flex-wrap items-center gap-1.5"
              data-testid="settings-usage-presets"
            >
              <span className="text-[11px] text-slate-400 mr-1">{copy.limitLabel}</span>
              {[...USAGE_PRESET_IDS, 'custom' as const].map((id) => (
                <button
                  key={id}
                  type="button"
                  role="radio"
                  aria-checked={choice === id}
                  data-testid={`settings-usage-preset-${id}`}
                  onClick={() => pick(id)}
                  className={`text-[11px] px-2.5 py-1 rounded-lg border ${
                    choice === id
                      ? 'border-cyan-500/60 bg-cyan-600/20 text-cyan-100'
                      : 'border-slate-700 text-slate-300 hover:bg-slate-900'
                  }`}
                >
                  {copy.presets[id]}
                </button>
              ))}
            </div>
            <p className="text-[11px] text-slate-500" data-testid="settings-usage-preset-help">
              {copy.presetHelp[choice]}
            </p>

            {choice === 'custom' && (
              <div className="space-y-1.5" data-testid="settings-usage-custom">
                <div className="grid grid-cols-1 sm:grid-cols-3 gap-2">
                  {USAGE_WINDOWS.map((id) => (
                    <label key={id} className="text-[11px] text-slate-400 space-y-1">
                      <span>{copy.fields[id]}</span>
                      {/* Text, not `type="number"`: a number box reports text it cannot parse
                          ("500,000", "1e") as empty, which would save as no limit. */}
                      <input
                        type="text"
                        inputMode="numeric"
                        value={draft[id] ?? ''}
                        onChange={(e) => setDraft((d) => ({ ...d, [id]: e.target.value }))}
                        onBlur={() => autoSave.commit(draft)}
                        data-testid={`settings-usage-input-${id}`}
                        className="w-full bg-slate-950 border border-slate-800 rounded-lg px-2 py-1.5 text-xs text-slate-200"
                      />
                    </label>
                  ))}
                </div>
                <p className="text-[11px] text-slate-500">{copy.customHelp}</p>
              </div>
            )}

            <FieldStatus status={autoSave.status} testId="settings-usage-save-status" />
          </div>

          <div className="space-y-1 text-[11px] text-slate-400" data-testid="settings-usage-providers">
            {report.providers.length === 0 ? (
              <p>{copy.noProviders}</p>
            ) : (
              <p>
                {copy.providers}{' '}
                {report.providers
                  .map((p) => `${PROVIDER_CONSOLES[p.provider]?.label ?? p.provider} ${compactTokens(p.tokens)}`)
                  .join(' · ')}
              </p>
            )}
            <p className="text-slate-500" data-testid="settings-usage-money-cap">
              {copy.moneyCap}{' '}
              {consoles.map((c, i) => (
                <React.Fragment key={c.url}>
                  {i > 0 && ' · '}
                  <a href={c.url} target="_blank" rel="noreferrer" className="text-cyan-400 hover:underline">
                    {c.label}
                  </a>
                </React.Fragment>
              ))}
            </p>
          </div>
        </>
      )}
    </div>
  );
};

/** One window's usage: a bar against its limit, or the figure alone when it has none. */
const WindowBar: React.FC<{ status: UsageWindowStatus; overridden: boolean }> = ({ status, overridden }) => {
  const copy = useCopy().usage;
  const used = compactTokens(status.used);
  const exact = status.limit === null ? `${status.used}` : `${status.used} / ${status.limit}`;
  const reached = status.limit !== null && status.used >= status.limit;
  const pct = status.limit === null ? null : percentUsed(status.used, status.limit);
  const tone = reached ? 'bg-rose-500' : pct !== null && pct >= 80 ? 'bg-amber-500' : 'bg-cyan-500';
  return (
    <div className="space-y-1" data-testid={`settings-usage-window-${status.window}`}>
      <div className="flex items-baseline justify-between gap-2 text-[11px]">
        <span className="text-slate-300">{copy.windows[status.window]}</span>
        <span className="text-slate-400 tabular-nums" title={exact}>
          {status.limit === null
            ? fmt(copy.noLimit, { used })
            : fmt(copy.ofLimit, { used, limit: compactTokens(status.limit), pct: pct ?? 0 })}
        </span>
      </div>
      {pct !== null && (
        <div
          role="progressbar"
          aria-valuemin={0}
          aria-valuemax={100}
          aria-valuenow={pct}
          aria-label={copy.windows[status.window]}
          className="h-1.5 rounded-full bg-slate-800 overflow-hidden"
        >
          <div className={`h-full ${tone}`} style={{ width: `${pct}%` }} />
        </div>
      )}
      {reached && status.available_again_at && (
        <p className="text-[11px] text-rose-300" data-testid={`settings-usage-paused-${status.window}`}>
          {fmt(copy.pausedUntil, { time: clockTime(status.available_again_at) })}
        </p>
      )}
      {overridden && (
        <p className="text-[11px] text-slate-500" data-testid={`settings-usage-env-${status.window}`}>
          {fmt(copy.envOverride, { variable: USAGE_ENV_VARS[status.window] })}
        </p>
      )}
    </div>
  );
};

/** A save the Core refused, with its reason when it gave one. */
class RefusedSave extends Error {
  constructor(readonly detail: string | null) {
    super(detail ?? 'refused');
  }
}

const clockTime = (iso: string): string =>
  new Date(iso).toLocaleTimeString([], { hour: 'numeric', minute: '2-digit' });

const toDraft = (limits: UsageLimitsValue): Record<string, string> =>
  Object.fromEntries(USAGE_WINDOWS.map((w) => [w, limits[w] === null ? '' : String(limits[w])]));

/** The form's boxes as a limits body: an empty box is no limit; anything else is sent as typed. */
const fromDraft = (draft: Record<string, string>): Record<string, number | string | null> =>
  Object.fromEntries(
    USAGE_WINDOWS.map((w) => {
      const text = (draft[w] ?? '').trim();
      if (text === '') return [w, null];
      const n = Number(text);
      // A value that is not a whole number goes as typed, so the Core's refusal names it.
      return [w, Number.isInteger(n) ? n : text];
    }),
  );

/** One console link per provider used, or all three when none has been (§4.5.1). */
const consoleLinks = (report: UsageReport | null): { label: string; url: string }[] => {
  const used = (report?.providers ?? []).map((p) => PROVIDER_CONSOLES[p.provider]).filter(Boolean);
  return used.length > 0 ? used : Object.values(PROVIDER_CONSOLES);
};
