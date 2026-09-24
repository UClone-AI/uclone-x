import React from 'react';
import { cn } from '../../lib/utils';
import type { Tone } from './Badge';

const TONE_DOT: Record<Tone, string> = {
  neutral: 'bg-slate-500',
  success: 'bg-emerald-400 shadow-[0_0_6px_rgba(52,211,153,0.8)]',
  danger: 'bg-rose-400 shadow-[0_0_6px_rgba(248,113,113,0.8)]',
  warning: 'bg-amber-400',
  info: 'bg-cyan-400',
};

interface StatusDotProps extends React.HTMLAttributes<HTMLSpanElement> {
  tone?: Tone;
}

/** A presence dot, the same shape whether it means "connected" or "context saturated". */
export const StatusDot: React.FC<StatusDotProps> = ({ tone = 'neutral', className, ...rest }) => (
  <span className={cn('inline-block w-1.5 h-1.5 rounded-full shrink-0', TONE_DOT[tone], className)} {...rest} />
);
