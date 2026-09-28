/**
 * The paid-model usage limits, as the head reads and writes them (`GET /api/usage`,
 * `PUT /api/usage/limits`; the token-gateway design §4.5.5).
 *
 * The Core owns the limits and the count; this module only names the shapes, the presets the
 * picker offers, and how a figure is written for a person.
 */

export type UsageWindowId = 'per_10_minutes' | 'per_5_hours' | 'per_week';

export const USAGE_WINDOWS: readonly UsageWindowId[] = ['per_10_minutes', 'per_5_hours', 'per_week'];

export type UsageLimitsValue = Record<UsageWindowId, number | null>;

export interface UsageWindowStatus {
  window: UsageWindowId;
  used: number;
  limit: number | null;
  available_again_at: string | null;
}

export interface UsageModelTotal {
  model: string | null;
  tokens: number;
}

export interface UsageProviderTotal {
  provider: string;
  tokens: number;
  models: UsageModelTotal[];
}

export interface UsageReport {
  checked_at: string;
  windows: UsageWindowStatus[];
  limits: UsageLimitsValue;
  env_overrides: UsageWindowId[];
  providers: UsageProviderTotal[];
}

export type UsagePresetId = 'light' | 'standard' | 'heavy' | 'none';

/** The presets of design §4.4. Placeholders until the owner rules on the figures (§7 Q1). */
export const USAGE_PRESETS: Readonly<Record<UsagePresetId, UsageLimitsValue>> = {
  light: { per_10_minutes: 150_000, per_5_hours: 500_000, per_week: 3_000_000 },
  standard: { per_10_minutes: 400_000, per_5_hours: 2_000_000, per_week: 10_000_000 },
  heavy: { per_10_minutes: 1_000_000, per_5_hours: 5_000_000, per_week: 30_000_000 },
  none: { per_10_minutes: null, per_5_hours: null, per_week: null },
};

export const USAGE_PRESET_IDS: readonly UsagePresetId[] = ['light', 'standard', 'heavy', 'none'];

/** The preset `limits` equal, or `'custom'` when they match none. */
export const presetOf = (limits: UsageLimitsValue): UsagePresetId | 'custom' =>
  USAGE_PRESET_IDS.find((id) =>
    USAGE_WINDOWS.every((w) => USAGE_PRESETS[id][w] === limits[w]),
  ) ?? 'custom';

/** Whether no window has a limit. */
export const hasNoLimit = (limits: UsageLimitsValue): boolean =>
  USAGE_WINDOWS.every((w) => limits[w] === null);

/** A token figure a person reads: `840`, `84K`, `1.2M`. The exact number goes in a tooltip. */
export const compactTokens = (n: number): string => {
  if (n >= 1_000_000) return `${trim(n / 1_000_000)}M`;
  if (n >= 1_000) return `${trim(n / 1_000)}K`;
  return String(n);
};

const trim = (x: number): string => (x >= 100 ? Math.round(x).toString() : x.toFixed(1).replace(/\.0$/, ''));

/** Share of `limit` used, as a whole percent, capped at 100 for the bar. */
export const percentUsed = (used: number, limit: number): number =>
  Math.min(100, Math.floor((used / limit) * 100));

/** The share at which a window is shown as nearly spent (design §4.5.3). */
export const NEAR_LIMIT_PCT = 80;

/**
 * The window a nearly-spent banner should name: the one with a limit, not yet reached, whose
 * share is highest and at least 80%. `null` when none is.
 */
export const nearLimitWindow = (windows: readonly UsageWindowStatus[]): UsageWindowId | null => {
  let best: { id: UsageWindowId; share: number } | null = null;
  for (const w of windows) {
    if (w.limit === null || w.used >= w.limit) continue;
    const share = w.used / w.limit;
    if (share * 100 >= NEAR_LIMIT_PCT && (best === null || share > best.share)) {
      best = { id: w.window, share };
    }
  }
  return best?.id ?? null;
};

/** The environment variable that overrides each window (`llm/usage/limits.py`). */
export const USAGE_ENV_VARS: Readonly<Record<UsageWindowId, string>> = {
  per_10_minutes: 'UCLONE_USAGE_LIMIT_10_MINUTES',
  per_5_hours: 'UCLONE_USAGE_LIMIT_5_HOURS',
  per_week: 'UCLONE_USAGE_LIMIT_WEEK',
};

/** Where each cloud provider lets the user set a spending limit (design §4.5.1). */
export const PROVIDER_CONSOLES: Readonly<Record<string, { label: string; url: string }>> = {
  anthropic: { label: 'Anthropic', url: 'https://console.anthropic.com/settings/limits' },
  openai: { label: 'OpenAI', url: 'https://platform.openai.com/settings/organization/limits' },
  gemini: { label: 'Google', url: 'https://aistudio.google.com/usage' },
};


const isLimit = (v: unknown): boolean => v === null || (typeof v === 'number' && Number.isInteger(v));

/**
 * Whether `body` is a usage report this head can read. An answer of another shape is treated
 * as unreadable rather than drawn from, so a half-read report never renders as zero usage.
 */
export const isUsageReport = (body: unknown): body is UsageReport => {
  if (typeof body !== 'object' || body === null) return false;
  const r = body as Partial<Record<keyof UsageReport, unknown>>;
  if (!Array.isArray(r.windows) || !Array.isArray(r.providers) || !Array.isArray(r.env_overrides)) {
    return false;
  }
  if (typeof r.limits !== 'object' || r.limits === null) return false;
  const limits = r.limits as Record<string, unknown>;
  if (!USAGE_WINDOWS.every((w) => isLimit(limits[w]))) return false;
  return r.windows.every((w: unknown) => {
    if (typeof w !== 'object' || w === null) return false;
    const s = w as Record<string, unknown>;
    return USAGE_WINDOWS.includes(s.window as UsageWindowId) && typeof s.used === 'number' && isLimit(s.limit);
  });
};
