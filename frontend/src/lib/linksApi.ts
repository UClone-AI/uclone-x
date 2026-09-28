/**
 * The requests behind Settings → Connections → uClone2 (`uclone2-link.md` §3.6).
 *
 * The Core answers a refusal with `{code, message}`: `code` names the case and this head words
 * it from its own catalog (`links.errors`), so the sentence follows the UI language. `message`
 * is the Core's Korean sentence for the CLI and is never shown here. Nothing the Core sends
 * carries the link token, and nothing here asks for it.
 */

/**
 * A card's state: a `LinkSessionState` value, `unlink_pending` for a pending unlink, or
 * `elsewhere` while another UClone-X on this computer (`ucx link run`) holds the links.
 */
export const LINK_STATES = [
  'connecting',
  'online',
  'reconnecting',
  'paused',
  'offline',
  'replaced',
  'ended',
  'server_paused',
  'update_required',
  'protocol_error',
  'unlink_pending',
  'elsewhere',
] as const;
export type LinkState = (typeof LINK_STATES)[number];

export interface LinkCard {
  link_id: string;
  local_agent_id: string;
  remote_username: string;
  remote_display_name: string;
  page_url: string;
  state: LinkState;
  created_at: string;
  last_connected_at: string | null;
}

export interface LinkList {
  links: LinkCard[];
  /** The links file exists and could not be read: not the same as having no links (P6). */
  unreadable: boolean;
}

/** The refusal codes this head words; anything else is said as `other`. */
export const LINK_ERROR_CODES = [
  'bad_input',
  'server_with_url',
  'code_invalid',
  'cap_reached',
  'rate_limited',
  'disabled',
  'token_invalid',
  'unreachable',
  'server_error',
  'not_found',
  'no_local_clone',
  'no_matching_clone',
  'not_pending',
  'elsewhere',
  'unreadable',
  'not_saved_undone',
  'not_saved_not_undone',
  'not_written',
  'offline',
  'other',
] as const;
export type LinkErrorCode = (typeof LINK_ERROR_CODES)[number];

export type LinkResult<T> = { ok: true; value: T } | { ok: false; code: LinkErrorCode };

const isLinkState = (value: unknown): value is LinkState =>
  typeof value === 'string' && (LINK_STATES as readonly string[]).includes(value);

const errorCode = (value: unknown): LinkErrorCode =>
  typeof value === 'string' && (LINK_ERROR_CODES as readonly string[]).includes(value)
    ? (value as LinkErrorCode)
    : 'other';

/** Whether `/api/links` answered with the shape this head reads. */
export const isLinkList = (value: unknown): value is LinkList => {
  if (typeof value !== 'object' || value === null) return false;
  const { links, unreadable } = value as { links?: unknown; unreadable?: unknown };
  return (
    typeof unreadable === 'boolean' &&
    Array.isArray(links) &&
    links.every(
      (l) =>
        typeof l === 'object' &&
        l !== null &&
        typeof (l as LinkCard).link_id === 'string' &&
        typeof (l as LinkCard).local_agent_id === 'string' &&
        typeof (l as LinkCard).remote_username === 'string' &&
        isLinkState((l as LinkCard).state),
    )
  );
};

/** A minihompy link only if it is a web page: a stored record is not trusted to be one. */
export const safePageUrl = (url: string): string | null => {
  try {
    const parsed = new URL(url);
    return parsed.protocol === 'https:' || parsed.protocol === 'http:' ? parsed.href : null;
  } catch {
    return null;
  }
};

const send = async <T>(url: string, method: string, body?: unknown): Promise<LinkResult<T>> => {
  let res: Response;
  try {
    res = await fetch(url, {
      method,
      headers: body === undefined ? undefined : { 'Content-Type': 'application/json' },
      body: body === undefined ? undefined : JSON.stringify(body),
    });
  } catch {
    return { ok: false, code: 'offline' };
  }
  const data = (await res.json().catch(() => null)) as unknown;
  if (!res.ok) {
    const code = data && typeof data === 'object' ? (data as { code?: unknown }).code : undefined;
    return { ok: false, code: errorCode(code) };
  }
  return { ok: true, value: data as T };
};

const linkPath = (id: string) => `/api/links/${encodeURIComponent(id)}`;

export const connectUclone2 = (connect: string, localAgentId: string | null) =>
  send<LinkCard>('/api/links/uclone2', 'POST', { connect, local_agent_id: localAgentId });

export const setLinkOnline = (id: string, enabled: boolean) =>
  send<LinkCard>(`${linkPath(id)}/enabled`, 'POST', { enabled });

export const unlinkLink = (id: string) =>
  send<{ outcome: 'removed' | 'pending' }>(linkPath(id), 'DELETE');

export const forgetLinkLocally = (id: string) =>
  send<{ outcome: 'forgotten' }>(`${linkPath(id)}?local=true`, 'DELETE');
