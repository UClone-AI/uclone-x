import React, { useEffect, useRef, useState } from 'react';
import { X } from 'lucide-react';
import { fmt, useCopy } from '../../i18n';
import { useApiRead } from '../../lib/useApiRead';
import { isUsageReport, nearLimitWindow, type UsageReport, type UsageWindowId } from '../../lib/usage';

/**
 * The room's usage banner (the token-gateway design §4.5.3): one line above the
 * composer when a paid-model window is at 80% of its limit, or when one is reached and paid
 * models are paused. It is read from `GET /api/usage` after each turn and is never written into
 * the conversation. It does not block anything.
 *
 * A closed banner stays closed until a different window, or a different state of the same
 * window, is to be shown.
 */
export const UsageBanner: React.FC<{ refreshKey: number; onOpenSettings?: () => void }> = ({
  refreshKey,
  onOpenSettings,
}) => {
  const copy = useCopy().usage;
  const { data, reload } = useApiRead<unknown>('/api/usage');
  const [closed, setClosed] = useState<string | null>(null);
  const firstRef = useRef(true);
  // `useApiRead` hands back a new `reload` on every render, so it is read through a ref: as an
  // effect dependency it would re-read on every render, not on every turn.
  const reloadRef = useRef(reload);
  reloadRef.current = reload;

  useEffect(() => {
    // The first read is `useApiRead`'s own; each later change of the key is a turn landing.
    if (firstRef.current) {
      firstRef.current = false;
      return;
    }
    reloadRef.current();
  }, [refreshKey]);

  const shown = isUsageReport(data) ? bannerOf(data) : null;
  if (shown === null || closed === shown.key) return null;

  const window = copy.phrases[shown.window];
  const text =
    shown.again !== null
      ? fmt(copy.banner.reached, { window, time: clockTime(shown.again) })
      : fmt(copy.banner.text, { window });

  return (
    <div
      role="status"
      data-testid="room-usage-banner"
      data-state={shown.again !== null ? 'reached' : 'near'}
      className={`mb-2 flex items-center gap-2 rounded-lg border px-3 py-1.5 text-[11px] ${
        shown.again !== null
          ? 'border-rose-800/70 bg-rose-950/40 text-rose-200'
          : 'border-amber-800/70 bg-amber-950/30 text-amber-200'
      }`}
    >
      <span className="flex-1">{text}</span>
      {onOpenSettings ? (
        <button type="button" onClick={onOpenSettings} className="underline hover:no-underline" data-testid="room-usage-banner-open">
          {copy.banner.open}
        </button>
      ) : null}
      <button
        type="button"
        onClick={() => setClosed(shown.key)}
        aria-label={copy.banner.close}
        className="text-slate-400 hover:text-slate-200"
        data-testid="room-usage-banner-close"
      >
        <X className="w-3.5 h-3.5" />
      </button>
    </div>
  );
};

/** What the banner shows: a reached window that lifts last, else the nearest-to-limit one. */
const bannerOf = (
  report: UsageReport,
): { key: string; window: UsageWindowId; again: string | null } | null => {
  const reached = report.windows
    .filter((w) => w.limit !== null && w.used >= w.limit && w.available_again_at !== null)
    .sort((a, b) => Date.parse(b.available_again_at ?? '') - Date.parse(a.available_again_at ?? ''));
  if (reached.length > 0) {
    const w = reached[0];
    return { key: `reached:${w.window}`, window: w.window, again: w.available_again_at };
  }
  const near = nearLimitWindow(report.windows);
  return near === null ? null : { key: `near:${near}`, window: near, again: null };
};

const clockTime = (iso: string): string =>
  new Date(iso).toLocaleTimeString([], { hour: 'numeric', minute: '2-digit' });
