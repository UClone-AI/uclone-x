import React from 'react';

interface FillRingProps {
  /** How full, 0 to 1. Values outside that are clamped rather than drawn past the ring. */
  filled: number;
  /** Drawn in the warning colour instead of the quiet one. */
  warn?: boolean;
  /** Read out in place of the ring, which is decoration to a screen reader. */
  label: string;
  'data-testid'?: string;
}

/**
 * A small ring drawn to a fraction, for a figure whose *proportion* is the point.
 *
 * The one place a number belongs as a shape rather than as digits: how full something is
 * against its own ceiling, read at a glance while doing something else. It carries no
 * digits of its own -- whatever sits beside it says what the fraction is of, because a
 * bare ring is exactly the empty container P6 forbids.
 *
 * Two colours, not a gradient across the range: a ring that shades continuously invites
 * the reader to estimate a value from a hue, which is the one thing colour is worst at.
 * Quiet until the thing it measures is at its ceiling, then amber -- the same pairing
 * `ResourceSummary` already uses for a saturated seat, so one state has one colour across
 * the surface.
 */
export const FillRing: React.FC<FillRingProps> = ({
  filled,
  warn = false,
  label,
  'data-testid': testId,
}) => {
  const fraction = Math.min(1, Math.max(0, Number.isFinite(filled) ? filled : 0));
  // 14px across with a 2px stroke, so r is 6 and the track sits inside the box. The dash
  // pattern is the whole circumference, offset back by the part that should not be drawn.
  const circumference = 2 * Math.PI * 6;

  return (
    <svg
      data-testid={testId}
      data-filled={fraction.toFixed(3)}
      viewBox="0 0 16 16"
      className="w-3.5 h-3.5 shrink-0 -rotate-90"
      role="img"
      aria-label={label}
    >
      <circle cx="8" cy="8" r="6" fill="none" strokeWidth="2" className="stroke-slate-700" />
      <circle
        cx="8"
        cy="8"
        r="6"
        fill="none"
        strokeWidth="2"
        strokeLinecap="round"
        strokeDasharray={circumference}
        strokeDashoffset={circumference * (1 - fraction)}
        className={warn ? 'stroke-amber-400' : 'stroke-slate-400'}
      />
    </svg>
  );
};
