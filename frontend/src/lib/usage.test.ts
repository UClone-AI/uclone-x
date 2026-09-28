import { describe, it, expect } from 'vitest';
import {
  USAGE_PRESETS,
  compactTokens,
  hasNoLimit,
  isUsageReport,
  nearLimitWindow,
  percentUsed,
  presetOf,
  type UsageWindowStatus,
} from './usage';

const win = (window: UsageWindowStatus['window'], used: number, limit: number | null): UsageWindowStatus => ({
  window,
  used,
  limit,
  available_again_at: null,
});

const report = () => ({
  checked_at: '2026-09-26T10:00:00Z',
  windows: [win('per_10_minutes', 0, null), win('per_5_hours', 0, null), win('per_week', 0, null)],
  limits: { per_10_minutes: null, per_5_hours: null, per_week: null },
  env_overrides: [],
  providers: [],
});

describe('presetOf', () => {
  it('names the preset the limits equal', () => {
    expect(presetOf(USAGE_PRESETS.standard)).toBe('standard');
    expect(presetOf(USAGE_PRESETS.none)).toBe('none');
  });

  it('calls limits matching no preset custom, even when two of three windows match', () => {
    expect(presetOf({ ...USAGE_PRESETS.light, per_week: 3_000_001 })).toBe('custom');
  });
});

describe('hasNoLimit', () => {
  it('is true only when every window is unlimited', () => {
    expect(hasNoLimit(USAGE_PRESETS.none)).toBe(true);
    expect(hasNoLimit({ per_10_minutes: null, per_5_hours: null, per_week: 1 })).toBe(false);
  });
});

describe('compactTokens', () => {
  it.each([
    [840, '840'],
    [1_000, '1K'],
    [84_000, '84K'],
    [150_500, '151K'],
    [1_200_000, '1.2M'],
    [30_000_000, '30M'],
  ])('writes %i as %s', (n, text) => {
    expect(compactTokens(n)).toBe(text);
  });
});

describe('percentUsed', () => {
  it('floors to a whole percent and caps at 100 for an overshoot', () => {
    expect(percentUsed(799, 1000)).toBe(79);
    expect(percentUsed(1500, 1000)).toBe(100);
  });
});

describe('nearLimitWindow', () => {
  it('names nothing below 80%', () => {
    expect(nearLimitWindow([win('per_5_hours', 799, 1000)])).toBeNull();
  });

  it('names a window at exactly 80%', () => {
    expect(nearLimitWindow([win('per_5_hours', 800, 1000)])).toBe('per_5_hours');
  });

  it('picks the highest share, and skips a reached or unlimited window', () => {
    expect(
      nearLimitWindow([
        win('per_10_minutes', 1000, 1000), // reached: the banner's other state
        win('per_5_hours', 850, 1000),
        win('per_week', 9_000, 10_000),
        win('per_week', 5_000_000, null),
      ]),
    ).toBe('per_week');
  });
});

describe('isUsageReport', () => {
  it('accepts the Core answer', () => {
    expect(isUsageReport(report())).toBe(true);
  });

  it.each([
    ['null', null],
    ['another endpoint answer', { servers: [] }],
    ['a fractional limit', { ...report(), limits: { per_10_minutes: 1.5, per_5_hours: null, per_week: null } }],
    ['an unknown window', { ...report(), windows: [win('per_minute' as never, 0, null)] }],
    ['a window without a count', { ...report(), windows: [{ window: 'per_week', limit: null }] }],
  ])('refuses %s', (_name, body) => {
    expect(isUsageReport(body)).toBe(false);
  });
});
