/**
 * When something last happened, in compact words (#1053).
 *
 * "now", "5m", "23h", "1d", "1w", "1mo", "1y" -- compact format to prevent
 * title truncation in the conversation list rail.
 */

const MINUTE = 60_000;
const HOUR = 60 * MINUTE;
const DAY = 24 * HOUR;

/** Local midnight of the day `ms` falls on. */
const startOfDay = (ms: number): number => {
  const d = new Date(ms);
  d.setHours(0, 0, 0, 0);
  return d.getTime();
};

/**
 * `timestamp` relative to `now`, formatted compactly, or `null` when `timestamp` is not a date.
 *
 * Under a day it counts elapsed time ("3h"); from a day on it counts calendar
 * days in local time, so "1d" means the day before today and not "24 to 48 hours
 * ago" -- a conversation from 23:30 read at 00:30 is "1h", and one from 01:00 two
 * days back is "2d" because 47 hours have passed. A stamp in the future (a
 * clock ahead of this one) reads as "now" rather than a negative duration.
 */
export function relativeTimeLabel(timestamp: string, now: number): string | null {
  const then = Date.parse(timestamp);
  if (Number.isNaN(then)) return null;

  const elapsed = now - then;
  if (elapsed < MINUTE) return 'now';
  if (elapsed < HOUR) return `${Math.floor(elapsed / MINUTE)}m`;
  if (elapsed < DAY) return `${Math.floor(elapsed / HOUR)}h`;

  // Rounded, not floored: a day that crosses a daylight-saving change is 23 or 25 hours.
  const days = Math.max(1, Math.round((startOfDay(now) - startOfDay(then)) / DAY));
  if (days < 7) return `${days}d`;
  if (days < 30) return `${Math.floor(days / 7)}w`;
  if (days < 365) return `${Math.floor(days / 30)}mo`;
  return `${Math.floor(days / 365)}y`;
}
