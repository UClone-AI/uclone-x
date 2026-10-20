import React, { useEffect, useState } from 'react';
import { KeyRound, LogOut } from 'lucide-react';
import { fmt, useCopy, useLocale } from '../../i18n';
import { useApiRead } from '../../lib/useApiRead';
import { Button } from '../ui/Button';
import { ReadFailure, ReadLoading } from './ReadState';

/**
 * Settings ▸ Browser ▸ Saved sign-ins and Signed-in sites (`browser-agent.md` §3.5, path A).
 *
 * The Core's list carries the site, the username and whether a 2FA key is kept, never the
 * password or the key; this head never renders either back, not even from what it was given
 * to send. Every sentence is worded here from the `browser` catalog by the Core's code.
 */

export interface SavedSignIn {
  site: string;
  username: string;
  has_2fa: boolean;
}

/** The dock's Save offer: open the form on `site`. `key` changes for each new offer. */
export interface SignInPrefill {
  site: string;
  key: number;
}

const SIGN_INS = '/api/browser/sign-ins';
const SESSIONS = '/api/browser/sessions';

const isRecord = (value: unknown): value is Record<string, unknown> =>
  typeof value === 'object' && value !== null && !Array.isArray(value);

/** Only the three fields a row shows are kept, so nothing else the answer held can reach the DOM. */
const readSignIns = (body: unknown): SavedSignIn[] | null => {
  if (!isRecord(body) || !Array.isArray(body.sign_ins)) return null;
  return body.sign_ins
    .filter(isRecord)
    .filter((r) => typeof r.site === 'string' && typeof r.username === 'string')
    .map((r) => ({ site: r.site as string, username: r.username as string, has_2fa: r.has_2fa === true }));
};

type Refusal =
  | 'site_missing'
  | 'username_missing'
  | 'password_missing'
  | 'setup_key_invalid'
  | 'settings_unreadable'
  | 'other';

const REFUSALS: ReadonlySet<string> = new Set([
  'site_missing',
  'username_missing',
  'password_missing',
  'setup_key_invalid',
  'settings_unreadable',
]);

const refusalOf = async (res: Response): Promise<Refusal> => {
  try {
    const body: unknown = await res.json();
    if (isRecord(body) && typeof body.reason_code === 'string' && REFUSALS.has(body.reason_code)) {
      return body.reason_code as Refusal;
    }
  } catch {
    // Not JSON: no code to word.
  }
  return 'other';
};

const sendJson = (url: string, method: string, body: unknown): Promise<Response> =>
  fetch(url, { method, headers: { 'Content-Type': 'application/json' }, body: JSON.stringify(body) });

interface Draft {
  site: string;
  username: string;
  password: string;
  key: string;
  /** Edit only: remove the saved 2FA key. A blank `key` alone keeps it. */
  clearKey: boolean;
}

const EMPTY: Draft = { site: '', username: '', password: '', key: '', clearKey: false };

/** `has2fa`: the sign-in being edited keeps a 2FA key; always false on Add. */
interface FormState {
  draft: Draft;
  editing: boolean;
  has2fa: boolean;
}

const inputClass =
  'w-full bg-slate-950 border border-slate-800 rounded-lg px-2 py-1.5 text-xs text-slate-200 disabled:opacity-50';

export const SavedSignIns: React.FC<{ prefill?: SignInPrefill | null }> = ({ prefill = null }) => {
  const allCopy = useCopy();
  const copy = allCopy.browser.signIns;
  const read = useApiRead<unknown>(SIGN_INS);
  const [fresh, setFresh] = useState<SavedSignIn[] | null>(null);
  const [form, setForm] = useState<FormState | null>(null);
  const [busy, setBusy] = useState(false);
  const [refused, setRefused] = useState<Refusal | null>(null);
  const [saved, setSaved] = useState(false);

  const readList = read.data !== null ? readSignIns(read.data) : null;
  const rows = fresh ?? readList;
  const fault =
    read.fault ?? (read.data !== null && readList === null ? { kind: 'unreadable' as const, detail: null } : null);

  const prefillKey = prefill?.key;
  const prefillSite = prefill?.site;
  useEffect(() => {
    if (prefillKey === undefined || prefillSite === undefined) return;
    setForm({ draft: { ...EMPTY, site: prefillSite }, editing: false, has2fa: false });
    setRefused(null);
    setSaved(false);
  }, [prefillKey, prefillSite]);

  const open = (draft: Draft, editing: boolean, has2fa = false) => {
    setForm({ draft, editing, has2fa: editing && has2fa });
    setRefused(null);
    setSaved(false);
  };
  const close = () => {
    setForm(null);
    setRefused(null);
  };
  const change = <K extends keyof Draft>(field: K, value: Draft[K]) =>
    setForm((now) => (now ? { ...now, draft: { ...now.draft, [field]: value } } : now));

  const save = async () => {
    if (!form) return;
    const { draft, has2fa } = form;
    const clear = has2fa && draft.clearKey;
    setBusy(true);
    setRefused(null);
    try {
      const res = await sendJson(SIGN_INS, 'POST', {
        site: draft.site.trim(),
        username: draft.username.trim(),
        password: draft.password,
        totp_key: clear ? '' : draft.key.trim(),
        clear_totp_key: clear,
      });
      const list = res.ok ? readSignIns(await res.json()) : null;
      if (list === null) {
        setRefused(res.ok ? 'other' : await refusalOf(res));
        return;
      }
      setFresh(list);
      setForm(null);
      setSaved(true);
    } catch {
      setRefused('other');
    } finally {
      setBusy(false);
    }
  };

  return (
    <div className="space-y-2" data-testid="settings-sign-ins">
      <p className="text-[11px] font-semibold text-slate-300 flex items-center gap-1.5">
        <KeyRound className="w-3.5 h-3.5 text-slate-400" />
        {copy.title}
      </p>
      <p className="text-[11px] text-slate-500">{copy.intro}</p>

      {fault !== null && rows === null && (
        <ReadFailure
          testId="sign-ins-error"
          what={copy.loadFailed}
          cause={allCopy.skills.plainCause[fault.kind]}
          plain
          onRetry={read.reload}
          retrying={read.loading}
        />
      )}
      {fault === null && rows === null && <ReadLoading testId="sign-ins-loading">{copy.loading}</ReadLoading>}

      {rows !== null && rows.length === 0 && (
        <p className="text-[11px] text-slate-500" data-testid="sign-ins-empty">
          {copy.none}
        </p>
      )}
      {rows !== null && rows.length > 0 && (
        <ul className="space-y-1.5" data-testid="sign-ins-list">
          {rows.map((row) => (
            <SignInRow
              key={`${row.site}\u0000${row.username}`}
              row={row}
              onEdit={() => open({ ...EMPTY, site: row.site, username: row.username }, true, row.has_2fa)}
              onDeleted={(list) => setFresh(list)}
            />
          ))}
        </ul>
      )}

      {saved && form === null && (
        <p role="status" className="text-[11px] text-emerald-400" data-testid="sign-in-saved">
          {copy.saved}
        </p>
      )}

      {rows !== null && form === null && (
        <Button variant="outline" size="sm" onClick={() => open(EMPTY, false)} data-testid="sign-in-add">
          {copy.add}
        </Button>
      )}

      {form !== null && (
        <form
          data-testid="sign-in-form"
          className="p-3 rounded-xl border border-slate-800/80 space-y-2"
          onSubmit={(e) => {
            e.preventDefault();
            void save();
          }}
        >
          <p className="text-[11px] font-semibold text-slate-300">
            {form.editing
              ? fmt(copy.formEdit, { site: form.draft.site, username: form.draft.username })
              : copy.formAdd}
          </p>
          <label className="block text-[11px] text-slate-400 space-y-1">
            <span>{copy.site}</span>
            <input
              type="text"
              data-testid="sign-in-site"
              value={form.draft.site}
              readOnly={form.editing}
              placeholder={copy.sitePlaceholder}
              autoComplete="off"
              spellCheck={false}
              onChange={(e) => change('site', e.target.value)}
              className={inputClass}
            />
          </label>
          <label className="block text-[11px] text-slate-400 space-y-1">
            <span>{copy.username}</span>
            <input
              type="text"
              data-testid="sign-in-username"
              value={form.draft.username}
              readOnly={form.editing}
              autoComplete="off"
              spellCheck={false}
              onChange={(e) => change('username', e.target.value)}
              className={inputClass}
            />
          </label>
          <label className="block text-[11px] text-slate-400 space-y-1">
            <span>{copy.password}</span>
            <input
              type="password"
              data-testid="sign-in-password"
              value={form.draft.password}
              autoComplete="new-password"
              placeholder={form.editing ? copy.passwordKeep : copy.passwordNew}
              onChange={(e) => change('password', e.target.value)}
              className={inputClass}
            />
          </label>
          <label className="block text-[11px] text-slate-400 space-y-1">
            <span>{copy.key}</span>
            <input
              type="text"
              data-testid="sign-in-key"
              value={form.draft.key}
              autoComplete="off"
              spellCheck={false}
              disabled={form.has2fa && form.draft.clearKey}
              placeholder={form.has2fa ? copy.keyKeep : undefined}
              onChange={(e) => change('key', e.target.value)}
              className={`${inputClass} font-mono`}
            />
          </label>
          <p className="text-[11px] text-slate-500">{copy.keyHelp}</p>
          {form.has2fa && (
            <label className="flex items-center gap-2 text-[11px] text-slate-300">
              <input
                type="checkbox"
                data-testid="sign-in-clear-key"
                checked={form.draft.clearKey}
                onChange={(e) => change('clearKey', e.target.checked)}
              />
              <span>{copy.clearKey}</span>
            </label>
          )}
          {refused !== null && (
            <p role="alert" className="text-[11px] text-rose-300" data-testid="sign-in-refused">
              {copy.refused[refused]}
            </p>
          )}
          <div className="flex items-center gap-2">
            <Button type="submit" variant="bordered" size="sm" disabled={busy} data-testid="sign-in-save">
              {busy ? copy.saving : copy.save}
            </Button>
            <Button variant="ghost" size="sm" onClick={close} data-testid="sign-in-cancel">
              {copy.cancel}
            </Button>
          </div>
        </form>
      )}
    </div>
  );
};

const SignInRow: React.FC<{
  row: SavedSignIn;
  onEdit: () => void;
  onDeleted: (list: SavedSignIn[]) => void;
}> = ({ row, onEdit, onDeleted }) => {
  const copy = useCopy().browser.signIns;
  const [confirming, setConfirming] = useState(false);
  const [busy, setBusy] = useState(false);
  const [failed, setFailed] = useState(false);

  const remove = async () => {
    setBusy(true);
    setFailed(false);
    try {
      const res = await sendJson(SIGN_INS, 'DELETE', { site: row.site, username: row.username });
      const list = res.ok ? readSignIns(await res.json()) : null;
      if (list === null) setFailed(true);
      else onDeleted(list);
    } catch {
      setFailed(true);
    } finally {
      setBusy(false);
    }
  };

  return (
    <li
      data-testid={`sign-in-${row.site}-${row.username}`}
      className="px-3 py-2 rounded-xl bg-slate-950/70 border border-slate-800/80 space-y-1.5 text-xs"
    >
      <div className="flex flex-wrap items-center gap-x-2 gap-y-1">
        <span className="font-medium text-slate-100 break-all">{row.site}</span>
        <span className="text-slate-400 break-all">{row.username}</span>
        {row.has_2fa && (
          <span
            data-testid="sign-in-2fa"
            title={copy.twoFactorLabel}
            aria-label={copy.twoFactorLabel}
            className="px-1.5 rounded border border-slate-700 text-[10px] text-slate-400"
          >
            {copy.twoFactor}
          </span>
        )}
        <span className="ml-auto flex items-center gap-1">
          <Button variant="ghost" size="sm" onClick={onEdit}>
            {copy.edit}
          </Button>
          <Button variant="ghost" size="sm" onClick={() => setConfirming(true)} disabled={confirming}>
            {copy.delete}
          </Button>
        </span>
      </div>
      {confirming && (
        <div className="flex flex-wrap items-center gap-2 text-[11px] text-slate-300">
          <span>{fmt(copy.confirmDelete, { site: row.site, username: row.username })}</span>
          <Button
            variant="outline"
            size="sm"
            onClick={() => void remove()}
            disabled={busy}
            data-testid="sign-in-delete-confirm"
          >
            {busy ? copy.deleting : copy.confirmDeleteButton}
          </Button>
          <Button variant="ghost" size="sm" onClick={() => setConfirming(false)} disabled={busy}>
            {copy.cancel}
          </Button>
        </div>
      )}
      {failed && (
        <p role="alert" className="text-[11px] text-rose-300" data-testid="sign-in-delete-error">
          {copy.deleteFailed}
        </p>
      )}
    </li>
  );
};

interface Sessions {
  open: boolean;
  sites: Array<{ site: string; last_used: string }>;
}

const readSessions = (body: unknown): Sessions | null => {
  if (!isRecord(body) || typeof body.open !== 'boolean' || !Array.isArray(body.sites)) return null;
  return {
    open: body.open,
    sites: body.sites
      .filter(isRecord)
      .filter((s) => typeof s.site === 'string')
      .map((s) => ({ site: s.site as string, last_used: typeof s.last_used === 'string' ? s.last_used : '' })),
  };
};

const whenOf = (iso: string, language: string): string => {
  const date = new Date(iso);
  if (!iso || Number.isNaN(date.getTime())) return '';
  return date.toLocaleString(language, { dateStyle: 'medium', timeStyle: 'short' });
};

export const SignedInSites: React.FC = () => {
  const allCopy = useCopy();
  const copy = allCopy.browser.sessions;
  const { language } = useLocale();
  const read = useApiRead<unknown>(SESSIONS);
  const [fresh, setFresh] = useState<Sessions | null>(null);
  const [busy, setBusy] = useState<string | null>(null);
  const [failed, setFailed] = useState(false);

  const readNow = read.data !== null ? readSessions(read.data) : null;
  const now = fresh ?? readNow;
  const fault =
    read.fault ?? (read.data !== null && readNow === null ? { kind: 'unreadable' as const, detail: null } : null);

  /** `site` null signs out of every listed site. */
  const signOut = async (site: string | null) => {
    setBusy(site ?? '*');
    setFailed(false);
    try {
      const res = await sendJson(`${SESSIONS}/sign-out`, 'POST', site === null ? {} : { site });
      const next = res.ok ? readSessions(await res.json()) : null;
      if (next === null) setFailed(true);
      else setFresh(next);
    } catch {
      setFailed(true);
    } finally {
      setBusy(null);
    }
  };

  return (
    <div className="space-y-2" data-testid="settings-signed-in">
      <p className="text-[11px] font-semibold text-slate-300 flex items-center gap-1.5">
        <LogOut className="w-3.5 h-3.5 text-slate-400" />
        {copy.title}
      </p>
      <p className="text-[11px] text-slate-500">{copy.intro}</p>

      {fault !== null && now === null && (
        <ReadFailure
          testId="signed-in-read-error"
          what={copy.loadFailed}
          cause={allCopy.skills.plainCause[fault.kind]}
          plain
          onRetry={read.reload}
          retrying={read.loading}
        />
      )}
      {fault === null && now === null && <ReadLoading testId="signed-in-loading">{copy.loading}</ReadLoading>}

      {now !== null && !now.open && (
        <p className="text-[11px] text-slate-500" data-testid="signed-in-closed">
          {copy.closed}
        </p>
      )}
      {now !== null && now.open && now.sites.length === 0 && (
        <p className="text-[11px] text-slate-500" data-testid="signed-in-none">
          {copy.none}
        </p>
      )}
      {now !== null && now.open && now.sites.length > 0 && (
        <>
          <ul className="space-y-1.5" data-testid="signed-in-list">
            {now.sites.map((s) => {
              const when = whenOf(s.last_used, language);
              return (
                <li
                  key={s.site}
                  data-testid={`signed-in-${s.site}`}
                  className="px-3 py-2 rounded-xl bg-slate-950/70 border border-slate-800/80 flex flex-wrap items-center gap-x-2 gap-y-1 text-xs"
                >
                  <span className="font-medium text-slate-100 break-all">{s.site}</span>
                  {when && <span className="text-[11px] text-slate-500">{fmt(copy.lastUsed, { when })}</span>}
                  <Button
                    variant="ghost"
                    size="sm"
                    className="ml-auto"
                    onClick={() => void signOut(s.site)}
                    disabled={busy !== null}
                  >
                    {busy === s.site ? copy.signingOut : copy.signOut}
                  </Button>
                </li>
              );
            })}
          </ul>
          <Button
            variant="outline"
            size="sm"
            onClick={() => void signOut(null)}
            disabled={busy !== null}
            data-testid="signed-in-sign-out-all"
          >
            {busy === '*' ? copy.signingOut : copy.signOutAll}
          </Button>
        </>
      )}
      {failed && (
        <p role="alert" className="text-[11px] text-rose-300" data-testid="signed-in-error">
          {copy.signOutFailed}
        </p>
      )}
    </div>
  );
};
