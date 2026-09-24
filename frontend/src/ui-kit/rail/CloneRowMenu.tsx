import React from 'react';
import type { KitIcon } from '../kit';
import type { UseKitEscape } from './types';

export interface CloneRowMenuItem {
  /** The item's testid. The row's former buttons' testids, so a caller's selector survives. */
  testId: string;
  label: string;
  icon?: KitIcon;
  onSelect: () => void;
}

interface CloneRowMenuProps {
  /** Names the menu for a screen reader: whose actions these are. */
  label: string;
  items: CloneRowMenuItem[];
  onClose: () => void;
  useEscape: UseKitEscape;
  testId: string;
}

/**
 * A clone row's less frequent actions -- pin, edit, profile -- behind one `⋯`.
 *
 * They were four always-drawn buttons beside the name, and a hidden pin still held its width,
 * so a 240px rail left a clone's name about 70px. An action used a few times a day does not
 * need a button on every row; it needs to be one deliberate step away.
 *
 * Closes on a choice, on Escape (the `'overlay'` layer), and on a press outside it. Focus
 * goes to the first item on open, and the arrow keys move between items, which is what a
 * `role="menu"` promises a screen-reader user.
 */
export const CloneRowMenu: React.FC<CloneRowMenuProps> = ({
  label,
  items,
  onClose,
  useEscape,
  testId,
}) => {
  const ref = React.useRef<HTMLDivElement>(null);
  useEscape('overlay', true, onClose);

  React.useEffect(() => {
    ref.current?.querySelector<HTMLButtonElement>('[role="menuitem"]')?.focus();
    const onPointerDown = (e: MouseEvent) => {
      if (ref.current && !ref.current.contains(e.target as Node)) onClose();
    };
    document.addEventListener('mousedown', onPointerDown);
    return () => document.removeEventListener('mousedown', onPointerDown);
  }, [onClose]);

  const onKeyDown = (e: React.KeyboardEvent) => {
    if (e.key !== 'ArrowDown' && e.key !== 'ArrowUp') return;
    e.preventDefault();
    const buttons = Array.from(
      ref.current?.querySelectorAll<HTMLButtonElement>('[role="menuitem"]') ?? [],
    );
    const at = buttons.indexOf(document.activeElement as HTMLButtonElement);
    const step = e.key === 'ArrowDown' ? 1 : -1;
    buttons[(at + step + buttons.length) % buttons.length]?.focus();
  };

  return (
    <div
      ref={ref}
      role="menu"
      aria-label={label}
      data-testid={testId}
      onKeyDown={onKeyDown}
      onClick={(e) => e.stopPropagation()}
      className="absolute right-0 top-full mt-1 z-40 min-w-40 py-1 rounded-lg border border-slate-700/60 bg-slate-900 shadow-lg"
    >
      {items.map(({ testId: itemTestId, label: itemLabel, icon: Icon, onSelect }) => (
        <button
          key={itemTestId}
          type="button"
          role="menuitem"
          data-testid={itemTestId}
          onClick={() => {
            onSelect();
            onClose();
          }}
          className="w-full flex items-center gap-2 px-3 py-1.5 text-left text-xs text-slate-300 hover:bg-slate-800 hover:text-slate-100 focus:bg-slate-800 focus:outline-none"
        >
          {Icon && <Icon className="w-3.5 h-3.5 shrink-0 text-slate-500" />}
          <span className="truncate">{itemLabel}</span>
        </button>
      ))}
    </div>
  );
};
