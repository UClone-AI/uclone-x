import React, { useState } from 'react';
import { useCopy } from '../../i18n';
import { useApiRead } from '../../lib/useApiRead';
import { hasNoLimit, isUsageReport, type UsageReport } from '../../lib/usage';

/** The `localStorage` key that remembers "Not now" for this browser. */
export const USAGE_OFFER_DISMISSED_KEY = 'uclone.usageOffer.dismissed';

const readDismissed = (): boolean => {
  try {
    return window.localStorage.getItem(USAGE_OFFER_DISMISSED_KEY) === '1';
  } catch {
    return false;
  }
};

/**
 * The preset offer (the token-gateway design §4.5.2): one line under the model
 * picker while a cloud provider is selected and no usage limit is set. Nothing is chosen for
 * the user; "Set a limit" opens Settings → Usage, "Not now" hides the line for this browser.
 *
 * Mounted only while a cloud provider is selected, so its read happens only then. A failed
 * read shows nothing: the offer is a suggestion, and Settings → Usage says why it cannot read.
 *
 * `saved` is the answer to a save made in Settings → Usage while this is mounted. It is newer
 * than this component's own read, so it wins: a limit just set hides the line at once.
 */
export const UsageOffer: React.FC<{ onOpen: () => void; saved?: UsageReport | null }> = ({ onOpen, saved }) => {
  const copy = useCopy().usage.offer;
  const read = useApiRead<unknown>('/api/usage');
  const data = saved ?? read.data;
  const [dismissed, setDismissed] = useState(readDismissed);

  if (dismissed || !isUsageReport(data) || !hasNoLimit(data.limits)) return null;

  const dismiss = () => {
    setDismissed(true);
    try {
      window.localStorage.setItem(USAGE_OFFER_DISMISSED_KEY, '1');
    } catch {
      // A browser that keeps no site data hides the line until Settings is next opened.
    }
  };

  return (
    <p data-testid="settings-usage-offer" className="text-[11px] text-slate-400 flex flex-wrap items-center gap-x-2">
      <span>{copy.text}</span>
      <button type="button" onClick={onOpen} className="text-cyan-400 hover:underline" data-testid="settings-usage-offer-set">
        {copy.set}
      </button>
      <span aria-hidden="true">·</span>
      <button type="button" onClick={dismiss} className="text-slate-500 hover:underline" data-testid="settings-usage-offer-dismiss">
        {copy.dismiss}
      </button>
    </p>
  );
};
