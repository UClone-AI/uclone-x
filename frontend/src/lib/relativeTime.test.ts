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
  it('says "just now" under a minute, and for a stamp from a clock ahead of this one', () => {
    expect(relativeTimeLabel(ago(10_000), NOW)).toBe('just now');
    expect(relativeTimeLabel(ago(-2 * MINUTE), NOW)).toBe('just now');
  });

  it('counts minutes, then hours, in words', () => {
    expect(relativeTimeLabel(ago(MINUTE), NOW)).toBe('1 minute ago');
    expect(relativeTimeLabel(ago(5 * MINUTE + 30_000), NOW)).toBe('5 minutes ago');
    expect(relativeTimeLabel(ago(HOUR), NOW)).toBe('1 hour ago');
    expect(relativeTimeLabel(ago(23 * HOUR), NOW)).toBe('23 hours ago');
  });

  it('says "yesterday" for the calendar day before today, not for 24-48 hours', () => {
    expect(relativeTimeLabel(at(2026, 8, 18, 9, 0), NOW)).toBe('yesterday');
    // 23:30 the night before, read at 00:30: an hour, not a day.
    const halfPastMidnight = new Date(2026, 8, 19, 0, 30).getTime();
    expect(relativeTimeLabel(at(2026, 8, 18, 23, 30), halfPastMidnight)).toBe('1 hour ago');
    // 01:00 two days back, read at 00:30: 47.5 hours, and two calendar days.
    expect(relativeTimeLabel(at(2026, 8, 17, 1, 0), halfPastMidnight)).toBe('2 days ago');
  });

  it('moves to weeks, months and years as the gap grows', () => {
    expect(relativeTimeLabel(at(2026, 8, 13, 12, 0), NOW)).toBe('6 days ago');
    expect(relativeTimeLabel(at(2026, 8, 12, 12, 0), NOW)).toBe('last week');
    expect(relativeTimeLabel(at(2026, 7, 29, 12, 0), NOW)).toBe('3 weeks ago');
    expect(relativeTimeLabel(at(2026, 6, 1, 12, 0), NOW)).toBe('2 months ago');
    expect(relativeTimeLabel(at(2025, 8, 1, 12, 0), NOW)).toBe('last year');
  });

  it('returns null for a stamp that is not a date, rather than "NaN years ago"', () => {
    expect(relativeTimeLabel('not a date', NOW)).toBeNull();
    expect(relativeTimeLabel('', NOW)).toBeNull();
  });
});
