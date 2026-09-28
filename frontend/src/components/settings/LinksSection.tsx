import React, { useEffect, useState } from 'react';
import { ExternalLink, Link2 } from 'lucide-react';
import { fmt, useCopy } from '../../i18n';
import { useApiRead } from '../../lib/useApiRead';
import {
  connectUclone2,
  forgetLinkLocally,
  isLinkList,
  safePageUrl,
  setLinkOnline,
  unlinkLink,
  type LinkCard,
  type LinkErrorCode,
  type LinkState,
} from '../../lib/linksApi';
import { Button } from '../ui/Button';
import { StatusDot } from '../ui/StatusDot';
import type { Tone } from '../ui/Badge';
import { ReadFailure, ReadLoading } from './ReadState';

/** How often the list is read again while a link is on its way somewhere. */
const TRANSIENT_POLL_MS = 3000;

/**
 * States a card leaves on its own: it is read again until it settles. `elsewhere` ends when the
 * other runtime stops and this dashboard takes the links over.
 */
const TRANSIENT: ReadonlySet<LinkState> = new Set(['connecting', 'reconnecting', 'server_paused', 'elsewhere']);

const TONE: Record<LinkState, Tone> = {
  connecting: 'info',
  online: 'success',
  reconnecting: 'warning',
  paused: 'neutral',
  offline: 'neutral',
  replaced: 'warning',
  ended: 'danger',
  server_paused: 'warning',
  update_required: 'danger',
  protocol_error: 'danger',
  unlink_pending: 'warning',
  elsewhere: 'info',
};

/**
 * Which switch a card offers. A session that is running (or trying to) can be taken offline;
 * one that stopped -- by the user, or for a reason a fresh dial may get past -- can be switched
 * back on. An ended link, a pending unlink and a link another runtime serves have no session here
 * to switch.
 */
const toggleFor = (state: LinkState): 'offline' | 'online' | null => {
  if (state === 'ended' || state === 'unlink_pending' || state === 'elsewhere') return null;
  if (state === 'connecting' || state === 'online' || state === 'reconnecting' || state === 'server_paused') {
    return 'offline';
  }
  return 'online';
};

interface PersonaList {
  personas?: { name: string }[];
}

/**
 * Settings → Connections → uClone2 (`uclone2-link.md` §3.5): one card per link, and a form
 * that links a clone from a pasted connect URL or code.
 *
 * Every sentence comes from the `links` catalog, chosen by a state or a refusal code the Core
 * sends. No protocol detail reaches this section -- no frame, close code or status -- and no
 * token: the Core never sends one. The activity rows of §3.5 are not here yet; nothing is
 * recorded to show until the task executor lands.
 */
export const LinksSection: React.FC<{ pollMs?: number }> = ({ pollMs = TRANSIENT_POLL_MS }) => {
  const allCopy = useCopy();
  const copy = allCopy.links;
  const read = useApiRead<unknown>('/api/links');
  const { loading, reload } = read;
  const list = read.data !== null && isLinkList(read.data) ? read.data : null;
  const fault =
    read.fault ?? (read.data !== null && list === null ? { kind: 'unreadable' as const, detail: null } : null);
  const personas = useApiRead<PersonaList>('/api/personas');
  const names = (personas.data?.personas ?? []).map((p) => p.name);

  const transient = list?.links.some((l) => TRANSIENT.has(l.state)) ?? false;
  useEffect(() => {
    if (!transient) return;
    const timer = window.setTimeout(reload, pollMs);
    return () => window.clearTimeout(timer);
  }, [transient, read.data, reload, pollMs]);

  return (
    <div className="space-y-3" data-testid="settings-links">
      <label className="text-xs font-semibold uppercase tracking-wider text-slate-400 flex items-center gap-1.5">
        <Link2 className="w-3.5 h-3.5 text-cyan-400" />
        {copy.title}
      </label>
      <p className="text-[11px] text-slate-500">{copy.intro}</p>

      {fault !== null && (
        <ReadFailure
          testId="settings-links-error"
          what={copy.loadFailed}
          cause={allCopy.skills.plainCause[fault.kind]}
          plain
          onRetry={reload}
          retrying={loading}
        />
      )}
      {fault === null && list === null && <ReadLoading testId="settings-links-loading">{copy.loading}</ReadLoading>}

      {list !== null && list.unreadable && (
        <p role="alert" className="text-[11px] text-rose-300" data-testid="settings-links-unreadable">
          {copy.unreadable}
        </p>
      )}
      {list !== null && !list.unreadable && list.links.length === 0 && (
        <p className="text-[11px] text-slate-500" data-testid="settings-links-empty">
          {copy.empty}
        </p>
      )}
      {list !== null && list.links.length > 0 && (
        <ul className="space-y-2" data-testid="settings-links-list">
          {list.links.map((link) => (
            <LinkCardRow key={link.link_id} link={link} onChanged={reload} />
          ))}
        </ul>
      )}

      <ConnectForm names={names} onLinked={reload} />
    </div>
  );
};

const LinkCardRow: React.FC<{ link: LinkCard; onChanged: () => void }> = ({ link, onChanged }) => {
  const copy = useCopy().links;
  const [busy, setBusy] = useState(false);
  const [confirming, setConfirming] = useState<'unlink' | 'forget' | null>(null);
  const [error, setError] = useState<LinkErrorCode | null>(null);
  const [pendingNote, setPendingNote] = useState(false);
  const toggle = toggleFor(link.state);
  const pageUrl = safePageUrl(link.page_url);

  const run = async (action: () => Promise<{ ok: true } | { ok: false; code: LinkErrorCode }>) => {
    setBusy(true);
    setError(null);
    try {
      const result = await action();
      if (!result.ok) {
        setError(result.code);
        // A link removed elsewhere (the CLI) is gone: the list is read again to say so.
        if (result.code === 'not_found') onChanged();
        return;
      }
      setConfirming(null);
      onChanged();
    } finally {
      setBusy(false);
    }
  };

  const switchOnline = () => run(() => setLinkOnline(link.link_id, toggle === 'online'));
  const unlink = () =>
    run(async () => {
      const result = await unlinkLink(link.link_id);
      if (result.ok && result.value.outcome === 'pending') setPendingNote(true);
      return result;
    });
  const forget = () => run(() => forgetLinkLocally(link.link_id));

  return (
    <li
      data-testid={`link-card-${link.link_id}`}
      className="p-3 rounded-xl bg-slate-950/70 border border-slate-800/80 space-y-2 text-xs"
    >
      <div className="flex flex-wrap items-center gap-x-2 gap-y-1">
        <span className="font-medium text-slate-100" aria-label={fmt(copy.card.localLabel, { name: link.local_agent_id })}>
          {link.local_agent_id}
        </span>
        <span aria-hidden="true" className="text-slate-500">
          ↔
        </span>
        {pageUrl !== null ? (
          <a
            href={pageUrl}
            target="_blank"
            rel="noopener noreferrer"
            data-testid="link-card-page"
            aria-label={fmt(copy.card.openPage, { username: link.remote_username })}
            className="inline-flex items-center gap-1 text-cyan-300 hover:text-cyan-200"
          >
            @{link.remote_username}
            <ExternalLink className="w-3 h-3" />
          </a>
        ) : (
          <span className="text-slate-300">@{link.remote_username}</span>
        )}
        {link.remote_display_name && link.remote_display_name !== link.remote_username && (
          <span className="text-slate-500">{link.remote_display_name}</span>
        )}
      </div>

      <p className="flex items-center gap-1.5 text-slate-300" data-testid="link-card-state" role="status">
        <StatusDot tone={TONE[link.state]} />
        {copy.state[link.state]}
      </p>

      {link.state === 'unlink_pending' && (
        <p className="text-[11px] text-amber-200/90" data-testid="link-card-forget-help">
          {copy.card.forgetHelp}
        </p>
      )}

      <div className="flex flex-wrap items-center gap-2">
        {toggle !== null && (
          <Button onClick={() => void switchOnline()} disabled={busy} data-testid="link-card-toggle">
            {busy && confirming === null ? copy.card.switching : toggle === 'online' ? copy.card.goOnline : copy.card.goOffline}
          </Button>
        )}
        {link.state === 'unlink_pending' ? (
          <Button onClick={() => setConfirming('forget')} disabled={busy} data-testid="link-card-forget">
            {copy.card.forget}
          </Button>
        ) : (
          <Button onClick={() => setConfirming('unlink')} disabled={busy} data-testid="link-card-unlink">
            {copy.card.unlink}
          </Button>
        )}
      </div>

      {confirming === 'unlink' && (
        <div className="flex flex-wrap items-center gap-2 text-[11px] text-slate-300" data-testid="link-unlink-confirm">
          <span>{fmt(copy.card.confirmUnlink, { username: link.remote_username })}</span>
          <Button onClick={() => void unlink()} disabled={busy}>
            {busy ? copy.card.unlinking : copy.card.confirm}
          </Button>
          <Button onClick={() => setConfirming(null)} disabled={busy}>
            {copy.card.cancel}
          </Button>
        </div>
      )}
      {confirming === 'forget' && (
        <div className="flex flex-wrap items-center gap-2 text-[11px] text-slate-300" data-testid="link-forget-confirm">
          <span>{fmt(copy.card.confirmForget, { username: link.remote_username })}</span>
          <Button onClick={() => void forget()} disabled={busy}>
            {copy.card.confirmForgetButton}
          </Button>
          <Button onClick={() => setConfirming(null)} disabled={busy}>
            {copy.card.cancel}
          </Button>
        </div>
      )}

      {pendingNote && link.state === 'unlink_pending' && (
        <p role="status" className="text-[11px] text-amber-200/90" data-testid="link-card-pending-note">
          {copy.card.unlinkPending}
        </p>
      )}
      {error !== null && (
        <p role="alert" className="text-[11px] text-rose-300" data-testid="link-card-error">
          {copy.errors[error]}
        </p>
      )}
    </li>
  );
};

const ConnectForm: React.FC<{ names: string[]; onLinked: () => void }> = ({ names, onLinked }) => {
  const copy = useCopy().links;
  const [pasted, setPasted] = useState('');
  const [clone, setClone] = useState('');
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState<LinkErrorCode | 'empty' | null>(null);
  const [linked, setLinked] = useState<LinkCard | null>(null);

  const submit = async (event: React.FormEvent) => {
    event.preventDefault();
    setLinked(null);
    if (pasted.trim() === '') {
      setError('empty');
      return;
    }
    setBusy(true);
    setError(null);
    try {
      const result = await connectUclone2(pasted.trim(), clone === '' ? null : clone);
      if (!result.ok) {
        setError(result.code);
        return;
      }
      // The code is single use: clear it so a second press cannot resend a spent one.
      setPasted('');
      setLinked(result.value);
      onLinked();
    } finally {
      setBusy(false);
    }
  };

  return (
    <form
      onSubmit={(e) => void submit(e)}
      className="p-3 rounded-xl border border-slate-800/80 space-y-2"
      data-testid="settings-links-connect"
    >
      <p className="text-[11px] font-semibold text-slate-300">{copy.connect.title}</p>
      <label className="block text-[11px] text-slate-400 space-y-1">
        <span>{copy.connect.label}</span>
        <input
          type="text"
          value={pasted}
          onChange={(e) => {
            setPasted(e.target.value);
            setError(null);
          }}
          placeholder={copy.connect.placeholder}
          autoComplete="off"
          spellCheck={false}
          data-testid="settings-links-input"
          className="w-full bg-slate-950 border border-slate-800 rounded-lg px-2 py-1.5 text-xs text-slate-200"
        />
      </label>
      <p className="text-[11px] text-slate-500">{copy.connect.help}</p>
      <label className="block text-[11px] text-slate-400 space-y-1">
        <span>{copy.connect.cloneLabel}</span>
        <select
          value={clone}
          onChange={(e) => setClone(e.target.value)}
          data-testid="settings-links-clone"
          className="w-full bg-slate-950 border border-slate-800 rounded-lg px-2 py-1.5 text-xs text-slate-200"
        >
          <option value="">{copy.connect.sameName}</option>
          {names.map((name) => (
            <option key={name} value={name}>
              {name}
            </option>
          ))}
        </select>
      </label>
      <div className="flex items-center gap-2">
        <Button variant="solid" type="submit" disabled={busy} data-testid="settings-links-submit">
          {busy ? copy.connect.submitting : copy.connect.submit}
        </Button>
      </div>
      {linked !== null && (
        <p role="status" className="text-[11px] text-emerald-400" data-testid="settings-links-linked">
          {fmt(copy.connect.linked, { local: linked.local_agent_id, username: linked.remote_username })}
        </p>
      )}
      {error !== null && (
        <p role="alert" className="text-[11px] text-rose-300" data-testid="settings-links-connect-error">
          {error === 'empty' ? copy.connect.empty : copy.errors[error]}
        </p>
      )}
    </form>
  );
};
