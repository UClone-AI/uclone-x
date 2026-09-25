import { describe, it, expect } from 'vitest';
import { relativeTimeLabel } from './relativeTime';

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
