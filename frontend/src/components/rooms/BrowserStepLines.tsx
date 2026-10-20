import React from 'react';
import { Globe } from 'lucide-react';
import { fmt, useCopy } from '../../i18n';
import { stepSentence, type BrowserStep } from '../../lib/browserLive';

/**
 * A turn's browser steps, one quiet line each (`browser-agent.md` §3.4): what the clone did
 * and to which element, by name. Never what it typed, never the tool's own error -- a step
 * that failed says only that it did not work. Each line fronts the dock's Browser tab.
 */
export const BrowserStepLines: React.FC<{
  steps: readonly BrowserStep[];
  onShow?: () => void;
  testId: string;
  /** The clone is waiting for the person to sign in (§3.5 path B): who, and on which site. */
  asked?: { name: string; site: string } | null;
}> = ({ steps, onShow, testId, asked = null }) => {
  const copy = useCopy().dock.browser;
  const t = copy.steps;
  if (steps.length === 0 && !asked) return null;
  return (
    <ol data-testid={testId} className="mb-1.5 flex flex-col gap-0.5">
      {steps.map((step, index) => (
        <li key={index}>
          <button
            type="button"
            data-testid="browser-step"
            data-ok={step.ok ? 'true' : 'false'}
            onClick={onShow}
            disabled={!onShow}
            title={onShow ? t.show : undefined}
            className="flex items-center gap-1.5 text-left text-[11px] text-slate-400 hover:text-slate-200 disabled:hover:text-slate-400"
          >
            <Globe className="w-3 h-3 shrink-0" aria-hidden="true" />
            <span>{stepSentence(t, step)}</span>
          </button>
        </li>
      ))}
      {asked ? (
        <li data-testid="browser-asked" className="flex items-center gap-1.5 text-[11px] text-slate-300">
          <Globe className="w-3 h-3 shrink-0" aria-hidden="true" />
          <span>{fmt(copy.banner.askedUser, { name: asked.name, site: asked.site })}</span>
          {onShow ? (
            <button type="button" onClick={onShow} className="underline text-slate-400 hover:text-slate-200">
              {t.show}
            </button>
          ) : null}
        </li>
      ) : null}
    </ol>
  );
};
