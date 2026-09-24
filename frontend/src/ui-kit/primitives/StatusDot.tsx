import React from 'react';
import { cx } from '../kit';

/** The five colours a status reads as. */
export type KitTone = 'neutral' | 'success' | 'danger' | 'warning' | 'info';

/**
 * The dot's fill, and nothing else.
 *
 * `success` and `danger` carried a zero-offset box shadow apiece -- a halo -- inherited from
 * the pre-kit copy in `components/ui/StatusDot.tsx`. #1061 asks the workspace to stop glowing,
 * and a glow is not information: the hue already says which of the five a dot is, and every
 * caller in the rail also states it in a word or a `title`. So the tones are flat, and
 * `ui-kit.restraint.test.ts` fails if a halo comes back (its comment spells the class; this
 * one may not, since that check reads this file).
 */
const TONE_DOT: Record<KitTone, string> = {
  neutral: 'bg-slate-500',
  success: 'bg-emerald-400',
  danger: 'bg-rose-400',
  warning: 'bg-amber-400',
  info: 'bg-cyan-400',
};

interface StatusDotProps extends Omit<React.HTMLAttributes<HTMLSpanElement>, 'className'> {
  tone?: KitTone;
}

/** A presence dot, the same shape whatever it means. No `className`: see `cx` in `../kit.ts`. */
export const StatusDot: React.FC<StatusDotProps> = ({ tone = 'neutral', ...rest }) => (
  <span className={cx('inline-block w-1.5 h-1.5 rounded-full shrink-0', TONE_DOT[tone])} {...rest} />
);
