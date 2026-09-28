import { describe, it, expect } from 'vitest';
import { exactTimeLabel, relativeTimeLabel } from './relativeTime';

// Local times, so "yesterday" is tested as a calendar fact in whatever zone runs this.
const NOW = new Date(2026, 8, 19, 12, 0, 0).getTime();
const at = (year: number, month: number, day: number, hour: number, minute: number) =>
  new Date(year, month, day, hour, minute).toISOString();
const ago = (ms: number) => new Date(NOW - ms).toISOString();

const MINUTE = 60_000;
const HOUR = 60 * MINUTE;

describe('relativeTimeLabel', () => {
  it('says "now" under a minute, and for a stamp from a clock ahead of this one', () => {
    expect(relativeTimeLabel(ago(10_000), NOW)).toBe('now');
    expect(relativeTimeLabel(ago(-2 * MINUTE), NOW)).toBe('now');
  });

  it('counts minutes, then hours, in compact form', () => {
    expect(relativeTimeLabel(ago(MINUTE), NOW)).toBe('1m');
    expect(relativeTimeLabel(ago(5 * MINUTE + 30_000), NOW)).toBe('5m');
    expect(relativeTimeLabel(ago(HOUR), NOW)).toBe('1h');
    expect(relativeTimeLabel(ago(23 * HOUR), NOW)).toBe('23h');
  });

  it('says "1d" for the calendar day before today, not for 24-48 hours', () => {
    expect(relativeTimeLabel(at(2026, 8, 18, 9, 0), NOW)).toBe('1d');
    // 23:30 the night before, read at 00:30: an hour, not a day.
    const halfPastMidnight = new Date(2026, 8, 19, 0, 30).getTime();
    expect(relativeTimeLabel(at(2026, 8, 18, 23, 30), halfPastMidnight)).toBe('1h');
    // 01:00 two days back, read at 00:30: 47.5 hours, and two calendar days.
    expect(relativeTimeLabel(at(2026, 8, 17, 1, 0), halfPastMidnight)).toBe('2d');
  });

  it('moves to weeks, months and years as the gap grows', () => {
    expect(relativeTimeLabel(at(2026, 8, 13, 12, 0), NOW)).toBe('6d');
    expect(relativeTimeLabel(at(2026, 8, 12, 12, 0), NOW)).toBe('1w');
    expect(relativeTimeLabel(at(2026, 7, 29, 12, 0), NOW)).toBe('3w');
    expect(relativeTimeLabel(at(2026, 6, 1, 12, 0), NOW)).toBe('2mo');
    expect(relativeTimeLabel(at(2025, 8, 1, 12, 0), NOW)).toBe('1y');
  });

  it('returns null for a stamp that is not a date, rather than "NaN years ago"', () => {
    expect(relativeTimeLabel('not a date', NOW)).toBeNull();
    expect(relativeTimeLabel('', NOW)).toBeNull();
  });
});

describe('relativeTimeLabel in Korean', () => {
  // Killed by: frontend/src/lib/relativeTime.ts :: const t = CATALOGS[language].time;
  // Becomes: const t = CATALOGS.en.time;
  it('writes the same compact labels with Korean units', () => {
    expect(relativeTimeLabel(ago(10_000), NOW, 'ko')).toBe('지금');
    expect(relativeTimeLabel(ago(5 * MINUTE + 30_000), NOW, 'ko')).toBe('5분');
    expect(relativeTimeLabel(ago(23 * HOUR), NOW, 'ko')).toBe('23시간');
    expect(relativeTimeLabel(at(2026, 8, 18, 9, 0), NOW, 'ko')).toBe('1일');
    expect(relativeTimeLabel(at(2026, 8, 12, 12, 0), NOW, 'ko')).toBe('1주');
    expect(relativeTimeLabel(at(2026, 6, 1, 12, 0), NOW, 'ko')).toBe('2개월');
    expect(relativeTimeLabel(at(2025, 8, 1, 12, 0), NOW, 'ko')).toBe('1년');
    expect(relativeTimeLabel('not a date', NOW, 'ko')).toBeNull();
  });
});

describe('exactTimeLabel', () => {
  const noon = new Date(2026, 8, 19, 12, 0, 5).toISOString();

  // Killed by: frontend/src/lib/relativeTime.ts :: month: 'numeric',
  // Becomes: month: 'short',
  it('writes the moment as the browser used to, in English', () => {
    expect(exactTimeLabel(noon, 'en')).toBe('Last active 9/19/2026, 12:00:05 PM');
  });

  // Killed by: frontend/src/lib/relativeTime.ts :: : new Intl.DateTimeFormat(language, {
  // Becomes: : new Intl.DateTimeFormat('en', {
  // How the day period is spelled ('오후' or 'PM') depends on the ICU data the runtime
  // ships: Node 24 writes 오후, the Node 22 in the public CI writes PM. Only the date
  // order and the time are this function's own.
  it("writes the moment in Korean's own date order", () => {
    expect(exactTimeLabel(noon, 'ko')).toMatch(/^마지막 활동: 2026\. 9\. 19\. \S+ 12:00:05$/);
  });

  // Killed by: frontend/src/lib/relativeTime.ts :: const when = Number.isNaN(then)
  // Becomes: const when = false
  it('names a stamp that is not a date as it arrived, rather than throwing', () => {
    expect(exactTimeLabel('not a date', 'ko')).toBe('마지막 활동: not a date');
  });
});
