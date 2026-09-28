import { useEffect, useRef, useState } from 'react';
import { useEscapeOwner } from '../../lib/escapePrecedence';

/**
 * A small menu's open state: Escape and a click anywhere outside `ref` close it.
 *
 * Escape goes through the shared owner at the transient-overlay layer, so closing this menu
 * never also closes Settings or stops a turn behind it.
 */
export function usePopover<T extends HTMLElement>() {
  const [open, setOpen] = useState(false);
  const ref = useRef<T | null>(null);
  useEscapeOwner('overlay', open, () => setOpen(false));
  useEffect(() => {
    if (!open) return undefined;
    const onDown = (event: MouseEvent) => {
      if (ref.current && event.target instanceof Node && !ref.current.contains(event.target)) {
        setOpen(false);
      }
    };
    document.addEventListener('mousedown', onDown);
    return () => document.removeEventListener('mousedown', onDown);
  }, [open]);
  return { open, setOpen, ref };
}
