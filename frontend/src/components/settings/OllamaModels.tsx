import React, { useCallback, useEffect, useRef, useState } from 'react';
import { Download, ExternalLink, Loader2, Trash2 } from 'lucide-react';
import { fmt, useCopy } from '../../i18n';
import { CoreFailure } from '../../lib/coreFailure';
import {
  fetchInstalledModels,
  installModel,
  removeInstalledModel,
  type Connection,
  type InstalledModel,
} from '../../lib/modelGateway';
import { Button } from '../ui/Button';
import { ReadLoading } from './ReadState';

/** Where a person finds a model name to install: Ollama's own catalogue. */
const OLLAMA_LIBRARY_URL = 'https://ollama.com/library';

type Notice = { tone: 'ok' | 'error'; text: string } | null;

const isAbort = (err: unknown): boolean =>
  typeof err === 'object' && err !== null && (err as { name?: unknown }).name === 'AbortError';

/**
 * Install and remove models on one Ollama connection (model-gateway.md §3.7, #2167).
 *
 * Shown under an Ollama connection's row, closed until asked for: what is installed (an
 * embedder is listed and said to be one, since it is not offered for conversations), a
 * remove per model with a confirm step, and an install by name. An install is one request
 * that waits for the whole download, so its progress is the wait itself, said in words with
 * a Cancel that stops the wait and not the download; closing Settings cancels it too.
 *
 * Every outcome is a sentence built here from the answer's status, never the Core's English
 * `detail` (which may name a terminal command). `onChanged` is called after an install or a
 * removal, so the connection's model count and the model pickers read again.
 */
export const OllamaModels: React.FC<{ row: Connection; onChanged: () => void }> = ({ row, onChanged }) => {
  const t = useCopy().gateway.ollama;
  const name = row.label;
  const [open, setOpen] = useState(false);
  const [models, setModels] = useState<InstalledModel[] | null>(null);
  const [loadFailed, setLoadFailed] = useState(false);
  const [draft, setDraft] = useState('');
  const [installing, setInstalling] = useState<string | null>(null);
  const [confirming, setConfirming] = useState<string | null>(null);
  const [removing, setRemoving] = useState<string | null>(null);
  const [notice, setNotice] = useState<Notice>(null);
  const abortRef = useRef<AbortController | null>(null);

  // Closing Settings (unmounting) stops waiting for an install, as the Cancel button does.
  useEffect(() => () => abortRef.current?.abort(), []);

  const load = useCallback(async () => {
    setLoadFailed(false);
    try {
      setModels(await fetchInstalledModels(row.id));
    } catch (err) {
      console.error('Failed to read the installed models:', err);
      setLoadFailed(true);
    }
  }, [row.id]);

  useEffect(() => {
    if (open) void load();
  }, [open, load]);

  const install = async () => {
    const model = draft.trim();
    if (!model || installing) return;
    const controller = new AbortController();
    abortRef.current = controller;
    setInstalling(model);
    setNotice(null);
    try {
      const { joined } = await installModel(row.id, model, controller.signal);
      setDraft('');
      setNotice({ tone: 'ok', text: fmt(joined ? t.installJoined : t.installed, { model, name }) });
      await load();
      onChanged();
    } catch (err) {
      if (isAbort(err)) {
        // Cancelling is not failing: Ollama keeps downloading; only the wait stopped.
        setNotice({ tone: 'ok', text: fmt(t.installStopped, { model }) });
      } else {
        console.error('Failed to install the model:', err);
        const silent = err instanceof CoreFailure && err.status === 504;
        setNotice({ tone: 'error', text: fmt(silent ? t.installSilent : t.installFailed, { model, name }) });
      }
    } finally {
      if (abortRef.current === controller) abortRef.current = null;
      setInstalling(null);
    }
  };

  const remove = async (model: string) => {
    setRemoving(model);
    setNotice(null);
    try {
      await removeInstalledModel(row.id, model);
      setConfirming(null);
      setModels((prev) => (prev ? prev.filter((m) => m.id !== model) : prev));
      setNotice({ tone: 'ok', text: fmt(t.removed, { model, name }) });
      onChanged();
    } catch (err) {
      console.error('Failed to remove the model:', err);
      setNotice({ tone: 'error', text: fmt(t.removeFailed, { model, name }) });
    } finally {
      setRemoving(null);
    }
  };

  if (!open) {
    return (
      <Button
        variant="bordered"
        data-testid={`ollama-models-open-${row.id}`}
        onClick={() => setOpen(true)}
        className="text-[11px] px-2.5 py-1 rounded-lg"
      >
        <Download className="w-3 h-3" />
        {t.manageOpen}
      </Button>
    );
  }

  return (
    <div data-testid={`ollama-models-${row.id}`} className="p-3 rounded-lg border border-slate-800 bg-slate-900/40 space-y-2 text-[11px]">
      <div className="flex items-center justify-between gap-2">
        <p className="text-slate-200 font-medium">{fmt(t.title, { name })}</p>
        <button
          type="button"
          onClick={() => setOpen(false)}
          className="text-slate-400 hover:text-slate-200 underline"
        >
          {t.manageClose}
        </button>
      </div>

      {notice && (
        <p
          data-testid={`ollama-models-notice-${row.id}`}
          role={notice.tone === 'error' ? 'alert' : 'status'}
          className={notice.tone === 'error' ? 'text-rose-300' : 'text-emerald-400/90'}
        >
          {notice.text}
        </p>
      )}

      {models === null && !loadFailed && <ReadLoading testId={`ollama-models-loading-${row.id}`}>{t.loading}</ReadLoading>}
      {loadFailed && (
        <div data-testid={`ollama-models-error-${row.id}`} role="alert" className="flex flex-wrap items-center gap-2 text-rose-300">
          <span>{fmt(t.loadFailed, { name })}</span>
          <button type="button" onClick={() => void load()} className="underline text-slate-300 hover:text-white">
            {t.retry}
          </button>
        </div>
      )}
      {models !== null && models.length === 0 && (
        <p data-testid={`ollama-models-none-${row.id}`} className="text-slate-400">
          {fmt(t.none, { name })}
        </p>
      )}
      {models !== null && models.length > 0 && (
        <ul className="space-y-1 max-h-48 overflow-y-auto" data-testid={`ollama-models-list-${row.id}`}>
          {models.map((m) => (
            <li
              key={m.id}
              data-testid={`ollama-model-${row.id}-${m.id}`}
              className="px-2.5 py-1.5 bg-slate-950/60 border border-slate-800 rounded-lg space-y-1"
            >
              <div className="flex items-center justify-between gap-2">
                <div className="min-w-0">
                  <div className="font-mono text-slate-300 truncate">{m.id}</div>
                  {!m.chat && <div className="text-slate-500">{t.notChat}</div>}
                </div>
                {confirming !== m.id && (
                  <button
                    type="button"
                    data-testid={`ollama-model-remove-${row.id}-${m.id}`}
                    onClick={() => setConfirming(m.id)}
                    disabled={removing !== null}
                    title={fmt(t.remove, { model: m.id })}
                    aria-label={fmt(t.remove, { model: m.id })}
                    className="text-slate-500 hover:text-rose-400 disabled:opacity-50 shrink-0"
                  >
                    <Trash2 className="w-3.5 h-3.5" />
                  </button>
                )}
              </div>
              {confirming === m.id && (
                <div data-testid={`ollama-model-confirm-${row.id}-${m.id}`} className="space-y-1">
                  <p className="text-slate-200">{fmt(t.removeConfirm, { model: m.id, name })}</p>
                  <div className="flex items-center gap-2">
                    <Button
                      variant="bordered"
                      data-testid={`ollama-model-remove-yes-${row.id}-${m.id}`}
                      disabled={removing !== null}
                      onClick={() => void remove(m.id)}
                      className="text-[11px] px-2.5 py-1 rounded-lg text-rose-300 border-rose-900/60 hover:bg-rose-950/40"
                    >
                      {removing === m.id && <Loader2 className="w-3 h-3 animate-spin" />}
                      {t.removeYes}
                    </Button>
                    <Button
                      variant="bordered"
                      disabled={removing !== null}
                      onClick={() => setConfirming(null)}
                      className="text-[11px] px-2.5 py-1 rounded-lg"
                    >
                      {t.removeKeep}
                    </Button>
                  </div>
                </div>
              )}
            </li>
          ))}
        </ul>
      )}

      <div className="space-y-1">
        <div className="flex items-center justify-between gap-2">
          <label htmlFor={`ollama-install-${row.id}`} className="block text-slate-400">
            {t.installLabel}
          </label>
          <a
            href={OLLAMA_LIBRARY_URL}
            target="_blank"
            rel="noopener noreferrer"
            className="inline-flex items-center gap-1 text-cyan-400 hover:text-cyan-300"
          >
            {t.library}
            <ExternalLink className="w-3 h-3" />
          </a>
        </div>
        <div className="flex flex-wrap items-center gap-2">
          <input
            id={`ollama-install-${row.id}`}
            data-testid={`ollama-install-input-${row.id}`}
            type="text"
            value={draft}
            onChange={(e) => setDraft(e.target.value)}
            onKeyDown={(e) => {
              if (e.key === 'Enter') void install();
            }}
            placeholder={t.installPlaceholder}
            disabled={installing !== null}
            className="flex-1 min-w-0 bg-slate-950/90 border border-slate-800 rounded-xl px-3 py-1.5 text-xs text-white placeholder-slate-600 focus:outline-none focus:border-cyan-500/80 font-mono disabled:opacity-50"
          />
          <Button
            variant="bordered"
            data-testid={`ollama-install-${row.id}`}
            disabled={installing !== null || draft.trim() === ''}
            onClick={() => void install()}
            className="text-[11px] px-2.5 py-1 rounded-lg shrink-0"
          >
            {installing ? <Loader2 className="w-3 h-3 animate-spin" /> : <Download className="w-3 h-3" />}
            {t.install}
          </Button>
          {installing && (
            <Button
              variant="bordered"
              data-testid={`ollama-install-cancel-${row.id}`}
              onClick={() => abortRef.current?.abort()}
              className="text-[11px] px-2.5 py-1 rounded-lg shrink-0"
            >
              {t.cancelInstall}
            </Button>
          )}
        </div>
        {installing && (
          <div data-testid={`ollama-installing-${row.id}`} role="status" className="text-slate-400 space-y-0.5">
            <p>{fmt(t.installing, { model: installing, name })}</p>
            <p className="text-slate-500">{t.installWait}</p>
          </div>
        )}
      </div>
    </div>
  );
};
