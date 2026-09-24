import React from 'react';
import { cn } from '../../lib/utils';

/**
 * The button shapes this app actually uses, named for what they're for rather than for
 * how they look — `outline` is the small bordered pill (invite, retry, mention, send),
 * `bordered` is the square icon button with its own background (header actions),
 * `ghost` is the icon button with none (sidebar/dock toggles), `solid` is the one that
 * says "this is on" (the active dock toggle).
 */
export type ButtonVariant = 'ghost' | 'bordered' | 'outline' | 'solid';
export type ButtonSize = 'icon' | 'sm';

const VARIANT_CLASSES: Record<ButtonVariant, string> = {
  ghost: 'text-slate-400 hover:text-white hover:bg-slate-800',
  bordered:
    'bg-slate-800/90 hover:bg-slate-700 border border-slate-700/80 text-slate-300 hover:text-white',
  outline: 'text-slate-300 border border-slate-700 hover:bg-slate-900',
  solid: 'bg-cyan-600/30 hover:bg-cyan-600/50 border border-cyan-500/40 text-cyan-100',
};

const SIZE_CLASSES: Record<ButtonSize, string> = {
  icon: 'p-1.5',
  sm: 'px-2 py-0.5 text-xs',
};

interface ButtonProps extends React.ButtonHTMLAttributes<HTMLButtonElement> {
  variant?: ButtonVariant;
  size?: ButtonSize;
}

export const Button = React.forwardRef<HTMLButtonElement, ButtonProps>(
  ({ variant = 'outline', size = 'sm', className, children, type = 'button', ...rest }, ref) => (
    <button
      ref={ref}
      type={type}
      className={cn(
        'inline-flex items-center justify-center gap-1 rounded-lg font-medium transition-colors disabled:opacity-50',
        VARIANT_CLASSES[variant],
        SIZE_CLASSES[size],
        className,
      )}
      {...rest}
    >
      {children}
    </button>
  ),
);
Button.displayName = 'Button';
