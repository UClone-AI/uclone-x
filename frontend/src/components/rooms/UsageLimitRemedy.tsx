import React, { useMemo, useState } from 'react';
import { Loader2 } from 'lucide-react';
import { fmt, useCopy } from '../../i18n';
import { plainFailure } from '../../lib/coreFailure';
import { useModelSet, type ConnectionsResponse } from '../../lib/modelGateway';
import { setCloneModel } from '../../lib/personasApi';
import { isUsageReport, pausedUntil } from '../../lib/usage';
import { useApiRead } from '../../lib/useApiRead';
import { Button } from '../ui/Button';

const clockTime = (iso: string): string =>
  new Date(iso).toLocaleTimeString([], { hour: 'numeric', minute: '2-digit' });

/**
 * What a turn refused at its start by the paid-model limit offers (owner ruling 2026-10-30,
 * #2431): when paid models can be used again, and switching this clone to a local model,
 * here. It never asks the person to send their message again or to start a new conversation:
 * the next message answers on whatever the clone is set to then.
 *
 * The time is the Core's own (`GET /api/usage`, the reached window that lifts last). The local
 * models are the chat models on connections that are not paid (`GET /api/connections`); with
 * none, it says so and where to add one.
 */
export const UsageLimitRemedy: React.FC<{
  cloneId: string;
  label: string;
  testId: string;
  onChanged?: () => void;
}> = ({ cloneId, label, testId, onChanged }) => {
  const t = useCopy().conversation.outcome.usageLimitRemedy;
  const usage = useApiRead<unknown>('/api/usage');
  const connections = useApiRead<ConnectionsResponse>('/api/connections');
  const { set } = useModelSet('chat', true);
  const [chosen, setChosen] = useState<string>('');
  const [state, setState] = useState<'idle' | 'saving' | 'done'>('idle');
  const [error, setError] = useState<string | null>(null);

  const until = isUsageReport(usage.data) ? pausedUntil(usage.data) : null;
  const locals = useMemo(() => {
    const unpaid = new Set(
      (connections.data?.connections ?? []).filter((c) => !c.paid).map((c) => c.id),
    );
    return (set?.groups ?? [])
      .filter((g) => unpaid.has(g.connection_id) && g.status === 'connected')
      .flatMap((g) =>
        g.models.map((m) => ({ ref: m.ref, name: `${g.label} · ${m.display_name || m.id}` })),
      );
  }, [connections.data, set]);
  const pick = chosen || (locals[0]?.ref ?? '');

  const run = async () => {
    setState('saving');
    setError(null);
    try {
      await setCloneModel(cloneId, pick);
      setState('done');
      onChanged?.();
    } catch (err) {
      console.error('Failed to switch the clone to a local model:', err);
      setError(plainFailure(err, fmt(t.failed, { name: label })));
      setState('idle');
    }
  };

  const shown = locals.find((l) => l.ref === pick)?.name ?? pick;
  return (
    <div className="mt-1 space-y-1 text-xs text-slate-400" data-testid={testId}>
      <p data-testid={`${testId}-when`}>
        {until !== null ? fmt(t.againAt, { time: clockTime(until) }) : t.againUnknown}{' '}
        {fmt(t.keepGoing, { name: label })}
      </p>
      {state === 'done' ? (
        <p role="status" data-testid={`${testId}-done`} className="text-emerald-400/90">
          {fmt(t.switched, { name: label, model: shown })}
        </p>
      ) : locals.length > 0 ? (
        <div className="flex flex-wrap items-center gap-2">
          <select
            aria-label={fmt(t.localLabel, { name: label })}
            data-testid={`${testId}-model`}
            value={pick}
            onChange={(e) => setChosen(e.target.value)}
            className="bg-slate-950/90 border border-slate-800 rounded-lg px-2 py-1 text-xs text-slate-200"
          >
            {locals.map((l) => (
              <option key={l.ref} value={l.ref}>
                {l.name}
              </option>
            ))}
          </select>
          <Button
            variant="bordered"
            data-testid={`${testId}-switch`}
            disabled={state === 'saving' || !pick}
            onClick={() => void run()}
            className="text-xs px-2.5 py-1 rounded-lg"
          >
            {state === 'saving' && <Loader2 className="w-3 h-3 animate-spin" />}
            {t.switch}
          </Button>
        </div>
      ) : (
        <p data-testid={`${testId}-no-local`}>{t.noLocal}</p>
      )}
      {error && (
        <p role="alert" className="text-rose-300">
          {error}
        </p>
      )}
    </div>
  );
};
