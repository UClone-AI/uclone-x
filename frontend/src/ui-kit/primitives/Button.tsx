import React from 'react';
import { cx } from '../kit';

/**
 * The kit's button: the shapes the rail uses, each named for what it is for.
 *
 * `ghost` is the icon button with no background; `quiet` is the same, dimmer, for the row
 * controls that sit beside a conversation's title; `outline` is the small bordered pill; and
 * `danger` is that pill for the one action that cannot be undone.
 *
 * There is no `className`, unlike the head's `components/ui/Button`: that one resolves a
 * clashing class with `tailwind-merge`, which the kit may not import (`../kit.ts`, `cx`). The
 * rail's call sites that used to override a variant's colour or padding are `quiet` and
 * `compact` here, and render the same classes the override produced.
 */
export type KitButtonVariant = 'ghost' | 'quiet' | 'outline' | 'danger';
export type KitButtonSize = 'sm' | 'icon' | 'compact';

const VARIANT_CLASSES: Record<KitButtonVariant, string> = {
  ghost: 'text-slate-400 hover:text-white hover:bg-slate-800',
  quiet: 'text-slate-500 hover:text-white hover:bg-slate-800',
  outline: 'text-slate-300 border border-slate-700 hover:bg-slate-900',
  danger: 'border border-rose-800 text-rose-200 hover:bg-rose-950/60',
};

const SIZE_CLASSES: Record<KitButtonSize, string> = {
  sm: 'px-2 py-0.5 text-xs',
  icon: 'p-1.5',
  compact: 'p-1',
};

interface KitButtonProps extends Omit<React.ButtonHTMLAttributes<HTMLButtonElement>, 'className'> {
  variant?: KitButtonVariant;
  size?: KitButtonSize;
}

export const Button = React.forwardRef<HTMLButtonElement, KitButtonProps>(
  ({ variant = 'outline', size = 'sm', children, type = 'button', ...rest }, ref) => (
    <button
      ref={ref}
      type={type}
      className={cx(
        'inline-flex items-center justify-center gap-1 rounded-lg font-medium transition-colors disabled:opacity-50',
        VARIANT_CLASSES[variant],
        SIZE_CLASSES[size],
      )}
      {...rest}
    >
      {children}
    </button>
  ),
);
Button.displayName = 'Button';
