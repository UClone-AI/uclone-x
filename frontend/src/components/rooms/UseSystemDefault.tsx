import React, { useState } from 'react';
import { Loader2 } from 'lucide-react';
import { fmt, useCopy } from '../../i18n';
import { plainFailure } from '../../lib/coreFailure';
import { clearCloneModel } from '../../lib/personasApi';
import type { RoomProviderFailure } from '../../types';
import { Button } from '../ui/Button';

/**
 * The one action a turn that failed on a clone's own model offers (model-gateway.md §3.6):
 * **Use system default**, which clears that clone's slot. The turn is not re-run: the person
 * sends again once it is done, so nothing reaches the default model without their say.
 *
 * Shown only when the Core names the action, the clone and the model; `onChanged` lets the
 * owner read the clone list again, so an editor opened next does not send the old ref back.
 */
export const UseSystemDefault: React.FC<{
  failure: RoomProviderFailure;
  label: string;
  testId: string;
  onChanged?: () => void;
}> = ({ failure, label, testId, onChanged }) => {
  const t = useCopy().gateway.clone;
  const [state, setState] = useState<'idle' | 'saving' | 'done'>('idle');
  const [error, setError] = useState<string | null>(null);
  const { clone, model_ref: ref } = failure;
  if (failure.action !== 'use_system_default' || !clone || !ref) return null;

  const run = async () => {
    setState('saving');
    setError(null);
    try {
      await clearCloneModel(clone, ref);
      setState('done');
      onChanged?.();
    } catch (err) {
      console.error('Failed to put the clone on the system default:', err);
      setError(plainFailure(err, t.useDefaultFailed));
      setState('idle');
    }
  };

  if (state === 'done') {
    return (
      <p data-testid={`${testId}-done`} role="status" className="mt-1 text-xs text-emerald-400/90">
        {fmt(t.usedDefault, { name: label })}
      </p>
    );
  }
  return (
    <div className="mt-1 space-y-1">
      <Button
        variant="bordered"
        data-testid={testId}
        disabled={state === 'saving'}
        onClick={() => void run()}
        className="text-xs px-2.5 py-1 rounded-lg"
      >
        {state === 'saving' && <Loader2 className="w-3 h-3 animate-spin" />}
        {t.useSystemDefault}
      </Button>
      {error && (
        <p role="alert" className="text-xs text-rose-300">
          {error}
        </p>
      )}
    </div>
  );
};
