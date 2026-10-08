import React from 'react';
import { CheckCircle2, Loader2, AlertTriangle } from 'lucide-react';
import { useCopy } from '../../i18n';
import type { AutoSaveStatus } from '../../lib/useAutoSave';

/**
 * The line under an auto-saved field: saving, saved, or why it was not saved.
 *
 * Under the field, not in a banner at the top: the person is looking at the field they
 * just changed, and several fields may be saving at once.
 */
export const FieldStatus: React.FC<{ status: AutoSaveStatus; testId: string }> = ({ status, testId }) => {
  const t = useCopy().settings.autosave;
  if (status.kind === 'idle') return null;
  if (status.kind === 'saving') {
    return (
      <p data-testid={testId} data-state="saving" className="mt-1 flex items-center gap-1.5 text-[11px] text-slate-400">
        <Loader2 className="w-3 h-3 animate-spin shrink-0" />
        {t.saving}
      </p>
    );
  }
  if (status.kind === 'saved') {
    return (
      <p data-testid={testId} data-state="saved" className="mt-1 flex items-center gap-1.5 text-[11px] text-emerald-400/90">
        <CheckCircle2 className="w-3 h-3 shrink-0" />
        {t.saved}
      </p>
    );
  }
  return (
    <p
      data-testid={testId}
      data-state="error"
      role="alert"
      className="mt-1 flex items-start gap-1.5 text-[11px] text-rose-300"
    >
      <AlertTriangle className="w-3 h-3 shrink-0 mt-0.5" />
      <span>{status.message}</span>
    </p>
  );
};
