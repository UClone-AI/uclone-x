import React from 'react';
import { cn } from '../../lib/utils';

/**
 * The five colors a status reads as, everywhere one is shown.
 *
 * Fifteen components each picked their own emerald/rose/amber/cyan/slate pairing for the
 * same five meanings before this existed, which is how the same status ended up two
 * different shades of green depending which screen you were on.
 */
export type Tone = 'neutral' | 'success' | 'danger' | 'warning' | 'info';

export const TONE_TEXT: Record<Tone, string> = {
  neutral: 'bg-slate-800/80 border-slate-700/60 text-slate-300',
  success: 'bg-emerald-950/50 border-emerald-800/80 text-emerald-200',
  danger: 'bg-rose-950/40 border-rose-800/60 text-rose-200',
  warning: 'bg-amber-950/40 border-amber-800/60 text-amber-200',
  info: 'bg-cyan-950/40 border-cyan-800/60 text-cyan-200',
};

interface BadgeProps extends React.HTMLAttributes<HTMLSpanElement> {
  tone?: Tone;
}

export const Badge: React.FC<BadgeProps> = ({ tone = 'neutral', className, children, ...rest }) => (
  <span
    className={cn(
      'inline-flex items-center gap-1 px-2 py-0.5 rounded-full border text-[11px] font-medium whitespace-nowrap',
      TONE_TEXT[tone],
      className,
    )}
    {...rest}
  >
    {children}
  </span>
);
