import { useEffect, useRef } from 'react';

/**
 * Escape's single owner (#1036).
 *
 * Five surfaces used to listen for Escape independently -- `SettingsModal`, the room's
 * delete-conversation dialog, a running turn's stop shortcut, the room composer's mention
 * menu, and the rail's inline title editor -- and none of them called `stopPropagation()`,
 * so a native Escape keydown reached every one of them that happened to be mounted, not
 * just the one nearest to what the user meant to dismiss. Abandoning a rail rename with
 * Settings open behind it also closed Settings; stopping a turn with the mention menu open
 * also closed the menu.
 *
 * Every consumer registers a layer with `useEscapeOwner` instead of adding its own
 * listener. On Escape, only the highest-precedence *active* layer's handler runs. The
 * order -- recorded in `docs/ui-dashboard-architecture.md` -- is: a focus-trapped dialog
 * outranks a running turn, which outranks a transient overlay, which outranks the clone
 * editor's Studio mode (#1377). Studio mode is last because it is a mode rather than a
 * transient: whatever is open on top of it closes first. The dock itself claims none of
 * them; that is a decision, not an omission (see the doc).
 */
export type EscapeLayer = 'dialog' | 'turn' | 'overlay' | 'studio';

const PRECEDENCE: readonly EscapeLayer[] = ['dialog', 'turn', 'overlay', 'studio'];

interface Registration {
  layer: EscapeLayer;
  onEscape: (event: KeyboardEvent) => void;
}

const registrations = new Map<symbol, Registration>();
let attached = false;

/**
 * Which layer wins when `activeLayers` are the ones currently registered. Exported so the
 * precedence order itself has a unit test that needs no DOM.
 */
export function pickEscapeLayer(activeLayers: readonly EscapeLayer[]): EscapeLayer | null {
  for (const layer of PRECEDENCE) {
    if (activeLayers.includes(layer)) return layer;
  }
  return null;
}

function handleWindowKeyDown(event: KeyboardEvent): void {
  if (event.key !== 'Escape') return;
  const active = Array.from(registrations.values());
  const winningLayer = pickEscapeLayer(active.map((registration) => registration.layer));
  if (winningLayer === null) return;
  // Two registrations at the same layer is not a designed case (e.g. two dialogs at once);
  // when it happens anyway, the most recently registered one wins, since a Map preserves
  // insertion order and that is the one most likely to have been opened on top.
  const winner = active.filter((registration) => registration.layer === winningLayer).pop();
  if (!winner) return;
  event.preventDefault();
  winner.onEscape(event);
}

function attach(): void {
  if (attached) return;
  window.addEventListener('keydown', handleWindowKeyDown);
  attached = true;
}

function detachIfIdle(): void {
  if (attached && registrations.size === 0) {
    window.removeEventListener('keydown', handleWindowKeyDown);
    attached = false;
  }
}

/**
 * Claims `layer` for Escape while `active` is true, and calls `onEscape` when it wins.
 * `onEscape` is read fresh on every call without re-registering, so passing an inline
 * closure that closes over the latest state is safe and does not churn the listener.
 * It receives the key event, for an owner that must leave some Escapes alone.
 */
export function useEscapeOwner(
  layer: EscapeLayer,
  active: boolean,
  onEscape: (event: KeyboardEvent) => void,
): void {
  const onEscapeRef = useRef(onEscape);
  onEscapeRef.current = onEscape;

  useEffect(() => {
    if (!active) return undefined;
    const id = Symbol(layer);
    registrations.set(id, {
      layer,
      onEscape: (event) => onEscapeRef.current(event),
    });
    attach();
    return () => {
      registrations.delete(id);
      detachIfIdle();
    };
  }, [layer, active]);
}
