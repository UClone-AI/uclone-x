/**
 * When something last happened, in compact words (#1053).
 *
 * "now", "5m", "23h", "1d", "1w", "1mo", "1y" -- compact format to prevent
 * title truncation in the conversation list rail. Korean writes its own unit words the same way.
 *
 * The unit words are the `time` catalog's, and the count is written by `Intl.NumberFormat`
 * in the screen's language. Not `Intl.RelativeTimeFormat`: it has no form without "ago"
 * ("5m ago"), which is the width this label exists to save. Not `Intl.NumberFormat`'s
 * narrow units either: English writes months as "m" there, the same as minutes.
 */
import { CATALOGS } from '../i18n/LocaleProvider';
import { fmt } from '../i18n/format';
import type { Language } from '../i18n/language';

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
export function relativeTimeLabel(timestamp: string, now: number, language: Language = 'en'): string | null {
  const then = Date.parse(timestamp);
  if (Number.isNaN(then)) return null;

  const t = CATALOGS[language].time;
  const count = (template: string, n: number) =>
    fmt(template, { count: new Intl.NumberFormat(language).format(n) });

  const elapsed = now - then;
  if (elapsed < MINUTE) return t.now;
  if (elapsed < HOUR) return count(t.minutes, Math.floor(elapsed / MINUTE));
  if (elapsed < DAY) return count(t.hours, Math.floor(elapsed / HOUR));

  // Rounded, not floored: a day that crosses a daylight-saving change is 23 or 25 hours.
  const days = Math.max(1, Math.round((startOfDay(now) - startOfDay(then)) / DAY));
  if (days < 7) return count(t.days, days);
  if (days < 30) return count(t.weeks, Math.floor(days / 7));
  if (days < 365) return count(t.months, Math.floor(days / 30));
  return count(t.years, Math.floor(days / 365));
}

/**
 * The same moment in full, for the tooltip over the compact label: "Last active 9/19/2026,
 * 12:00:05 PM" in English. The fields are the ones
 * `Date.prototype.toLocaleString` writes, now in the screen's language rather than the
 * browser's. A stamp that is not a date is named as it arrived; `Intl` would throw on it.
 */
export function exactTimeLabel(timestamp: string, language: Language = 'en'): string {
  const then = Date.parse(timestamp);
  const when = Number.isNaN(then)
    ? timestamp
    : new Intl.DateTimeFormat(language, {
        year: 'numeric',
        month: 'numeric',
        day: 'numeric',
        hour: 'numeric',
        minute: '2-digit',
        second: '2-digit',
      }).format(then);
  return fmt(CATALOGS[language].time.lastActive, { when });
}
