import { useEffect, useRef } from 'react';
import { fmt, useCopy } from '../../i18n';
import { useApiRead } from '../../lib/useApiRead';
import { isLinkList, safePageUrl } from '../../lib/linksApi';

/** How often the clone page reads the links again: a modest pace for one status line. */
const CLONE_LINE_POLL_MS = 15000;

/**
 * The clone page's one line about uClone2 (`uclone2-link.md` §3.5): *active in uClone2 ·
 * @username*, while this clone's link is online.
 *
 * It renders nothing otherwise -- while the list is loading, when it could not be read
 * (Settings → Connections says why), and when the clone is not linked or its link is not
 * online. A clone that is offline in uClone2 is not "active" there, so the line would be false.
 *
 * The list is read again every `pollMs` while the page is open, the way Settings → Connections
 * re-reads it: a session can drop at any time, and a line read once would keep saying *active*
 * after it has. The timer is cleared on unmount.
 */
export function CloneLinkLine({ cloneId, pollMs = CLONE_LINE_POLL_MS }: { cloneId: string; pollMs?: number }) {
  const copy = useCopy().links;
  const { data, loading, reload } = useApiRead<unknown>('/api/links');
  // `reload` is a new function on every render; held in a ref so a parent re-rendering the
  // page does not keep pushing the next read back.
  const reloadRef = useRef(reload);
  reloadRef.current = reload;
  useEffect(() => {
    if (loading) return;
    const timer = window.setTimeout(() => reloadRef.current(), pollMs);
    return () => window.clearTimeout(timer);
  }, [loading, data, pollMs]);
  const list = data !== null && isLinkList(data) ? data : null;
  const link = list?.links.find((l) => l.local_agent_id === cloneId && l.state === 'online');
  if (link === undefined) return null;
  const pageUrl = safePageUrl(link.page_url);
  const text = fmt(copy.cloneLine, { username: link.remote_username });
  return (
    <p data-testid="clone-link-line" className="text-[11px] text-emerald-300/90">
      {pageUrl !== null ? (
        <a
          href={pageUrl}
          target="_blank"
          rel="noopener noreferrer"
          aria-label={fmt(copy.cloneLineLabel, { username: link.remote_username })}
          className="hover:text-emerald-200"
        >
          {text}
        </a>
      ) : (
        text
      )}
    </p>
  );
}
