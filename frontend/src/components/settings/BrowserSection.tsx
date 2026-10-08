import React, { useEffect, useState } from 'react';
import { Check, Copy, Globe, RefreshCw } from 'lucide-react';
import { fmt, useCopy } from '../../i18n';
import { useApiRead } from '../../lib/useApiRead';
import { Button } from '../ui/Button';
import { StatusDot } from '../ui/StatusDot';
import { ReadFailure, ReadLoading } from './ReadState';

/** How often the state is read again while no extension is connected, to show it pairing. */
const UNPAIRED_POLL_MS = 3000;

export interface ExtensionStatus {
  connected: boolean;
  version: string | null;
  pairing_code: string;
  extension_folder: string;
}

const isStatus = (value: unknown): value is ExtensionStatus => {
  if (typeof value !== 'object' || value === null) return false;
  const v = value as Record<string, unknown>;
  return (
    typeof v.connected === 'boolean' &&
    typeof v.pairing_code === 'string' &&
    typeof v.extension_folder === 'string'
  );
};

/**
 * Settings ▸ Browser ▸ Connect your Chrome (`browser-agent.md` §3.6): whether the UClone-X
 * extension is connected, and until it is, how to load it and the pairing code to give it.
 *
 * Every sentence comes from the `browser` catalog. The Core's answers carry state, never a
 * sentence this section shows: a failure is worded here, by what failed.
 */
export const BrowserSection: React.FC<{ pollMs?: number }> = ({ pollMs = UNPAIRED_POLL_MS }) => {
  const allCopy = useCopy();
  const copy = allCopy.browser;
  const read = useApiRead<unknown>('/api/browser/extension');
  const { loading, reload } = read;
  const [fresh, setFresh] = useState<ExtensionStatus | null>(null);
  const [busy, setBusy] = useState(false);
  const [failed, setFailed] = useState(false);
  const status = fresh ?? (read.data !== null && isStatus(read.data) ? read.data : null);
  const fault =
    read.fault ?? (read.data !== null && !isStatus(read.data) ? { kind: 'unreadable' as const, detail: null } : null);

  const connected = status?.connected ?? false;
  useEffect(() => {
    if (status === null || connected) return;
    const timer = window.setTimeout(() => {
      setFresh(null);
      reload();
    }, pollMs);
    return () => window.clearTimeout(timer);
  }, [status, connected, read.data, reload, pollMs]);

  const newCode = async () => {
    setBusy(true);
    setFailed(false);
    try {
      const res = await fetch('/api/browser/extension/pairing-code', { method: 'POST' });
      const body: unknown = res.ok ? await res.json() : null;
      if (isStatus(body)) setFresh(body);
      else setFailed(true);
    } catch {
      setFailed(true);
    } finally {
      setBusy(false);
    }
  };

  return (
    <div className="space-y-3" data-testid="settings-browser">
      <label className="text-xs font-semibold uppercase tracking-wider text-slate-400 flex items-center gap-1.5">
        <Globe className="w-3.5 h-3.5 text-cyan-400" />
        {copy.title}
      </label>
      <p className="text-[11px] text-slate-500">{copy.intro}</p>

      {fault !== null && status === null && (
        <ReadFailure
          testId="settings-browser-error"
          what={copy.loadFailed}
          cause={allCopy.skills.plainCause[fault.kind]}
          plain
          onRetry={reload}
          retrying={loading}
        />
      )}
      {fault === null && status === null && (
        <ReadLoading testId="settings-browser-loading">{copy.loading}</ReadLoading>
      )}

      {status !== null && (
        <>
          <p
            role="status"
            data-testid="settings-browser-status"
            data-connected={status.connected ? 'true' : 'false'}
            className="flex items-center gap-2 text-xs text-slate-200"
          >
            <StatusDot tone={status.connected ? 'success' : 'neutral'} />
            {status.connected
              ? status.version
                ? fmt(copy.connectedVersion, { version: status.version })
                : copy.connected
              : copy.notConnected}
          </p>

          {!status.connected && (
            <div className="space-y-2" data-testid="settings-browser-steps">
              <p className="text-[11px] font-semibold text-slate-400">{copy.stepsTitle}</p>
              <ol className="list-decimal pl-5 space-y-2 text-[11px] text-slate-300">
                <li>{copy.stepExtensions}</li>
                <li className="space-y-1">
                  <span>{copy.stepLoad}</span>
                  <CopyField value={status.extension_folder} testId="settings-browser-folder" />
                </li>
                <li className="space-y-1">
                  <span>{copy.stepPaste}</span>
                  <CopyField value={status.pairing_code} testId="settings-browser-code" />
                </li>
              </ol>
            </div>
          )}

          <div className="space-y-1">
            <Button
              variant="bordered"
              size="sm"
              onClick={() => void newCode()}
              disabled={busy}
              data-testid="settings-browser-new-code"
            >
              <RefreshCw className="w-3 h-3" />
              {busy ? copy.newCodeBusy : copy.newCode}
            </Button>
            <p className="text-[11px] text-slate-500">{copy.newCodeHelp}</p>
            {failed && (
              <p role="alert" className="text-[11px] text-rose-300" data-testid="settings-browser-new-code-error">
                {copy.newCodeFailed}
              </p>
            )}
          </div>
        </>
      )}
    </div>
  );
};

const CopyField: React.FC<{ value: string; testId: string }> = ({ value, testId }) => {
  const copy = useCopy().browser;
  const [state, setState] = useState<'idle' | 'copied' | 'failed'>('idle');

  const onCopy = async () => {
    try {
      await navigator.clipboard.writeText(value);
      setState('copied');
    } catch {
      setState('failed');
    }
  };

  return (
    <div className="space-y-1">
      <div className="flex items-center gap-2">
        <code
          data-testid={testId}
          className="flex-1 min-w-0 break-all rounded-lg bg-slate-950/70 border border-slate-800 px-2 py-1 font-mono text-[11px] text-slate-200 select-all"
        >
          {value}
        </code>
        <Button variant="ghost" size="sm" onClick={() => void onCopy()} data-testid={`${testId}-copy`}>
          {state === 'copied' ? <Check className="w-3 h-3" /> : <Copy className="w-3 h-3" />}
          {state === 'copied' ? copy.copied : copy.copy}
        </Button>
      </div>
      {state === 'failed' && (
        <p role="alert" className="text-[11px] text-rose-300">
          {copy.copyFailed}
        </p>
      )}
    </div>
  );
};
