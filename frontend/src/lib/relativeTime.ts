/**
 * When something last happened, in the words a person uses for it (#1053).
 *
 * "5 minutes ago", "yesterday", "last week" -- not an ISO stamp, and not a clock time
 * without a date. The rail's rows are how a person finds yesterday's conversation again,
 * and that is a question asked in these words.
 *
 * English on purpose: every other word in the head is English, and a row reading
 * "Architecture triage · vor 5 Minuten" is two languages in one line.
 */

const MINUTE = 60_000;
const HOUR = 60 * MINUTE;
const DAY = 24 * HOUR;

const phrase = new Intl.RelativeTimeFormat('en', { numeric: 'auto' });

/** Local midnight of the day `ms` falls on. */
const startOfDay = (ms: number): number => {
  const d = new Date(ms);
  d.setHours(0, 0, 0, 0);
  return d.getTime();
};

/**
 * `timestamp` relative to `now`, or `null` when `timestamp` is not a date.
 *
 * Under a day it counts elapsed time ("3 hours ago"); from a day on it counts calendar
 * days in local time, so "yesterday" means the day before today and not "24 to 48 hours
 * ago" -- a conversation from 23:30 read at 00:30 is "1 hour ago", and one from 01:00 two
 * days back is not "yesterday" because 47 hours have passed. A stamp in the future (a
 * clock ahead of this one) reads as "just now" rather than "in 2 minutes".
 */
export function relativeTimeLabel(timestamp: string, now: number): string | null {
  const then = Date.parse(timestamp);
  if (Number.isNaN(then)) return null;

  const elapsed = now - then;
  if (elapsed < MINUTE) return 'just now';
  if (elapsed < HOUR) return phrase.format(-Math.floor(elapsed / MINUTE), 'minute');
  if (elapsed < DAY) return phrase.format(-Math.floor(elapsed / HOUR), 'hour');

  // Rounded, not floored: a day that crosses a daylight-saving change is 23 or 25 hours.
  const days = Math.max(1, Math.round((startOfDay(now) - startOfDay(then)) / DAY));
  if (days < 7) return phrase.format(-days, 'day');
  if (days < 30) return phrase.format(-Math.floor(days / 7), 'week');
  if (days < 365) return phrase.format(-Math.floor(days / 30), 'month');
  return phrase.format(-Math.floor(days / 365), 'year');
}
