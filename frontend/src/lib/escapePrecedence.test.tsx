import { describe, expect, it, vi } from 'vitest';
import { fireEvent, render } from '@testing-library/react';
import { pickEscapeLayer, useEscapeOwner } from './escapePrecedence';

describe('pickEscapeLayer', () => {
  it('returns null when nothing is active', () => {
    // Killed by: frontend/src/lib/escapePrecedence.ts :: return null;
    // Becomes: return 'dialog';
    expect(pickEscapeLayer([])).toBeNull();
  });

  it('picks dialog over turn and overlay', () => {
    // Killed by: frontend/src/lib/escapePrecedence.ts :: ['dialog', 'turn', 'overlay', 'studio']
    // Becomes: ['turn', 'dialog', 'overlay', 'studio']
    expect(pickEscapeLayer(['overlay', 'turn', 'dialog'])).toBe('dialog');
  });

  it('picks turn over overlay when no dialog is active', () => {
    expect(pickEscapeLayer(['overlay', 'turn'])).toBe('turn');
  });

  it('picks overlay when it is the only active layer', () => {
    expect(pickEscapeLayer(['overlay'])).toBe('overlay');
  });

  it('ranks Studio mode below every transient layer (#1377)', () => {
    // Studio mode is a mode, not a transient: a dialog, a rename or a menu open on top of it
    // closes first, and only an Escape nothing else claims leaves Studio mode.
    // Killed by: frontend/src/lib/escapePrecedence.ts :: ['dialog', 'turn', 'overlay', 'studio']
    // Becomes: ['dialog', 'turn', 'studio', 'overlay']
    expect(pickEscapeLayer(['studio', 'overlay'])).toBe('overlay');
    expect(pickEscapeLayer(['studio'])).toBe('studio');
  });
});

function Consumer({
  layer,
  active,
  onEscape,
}: {
  layer: 'dialog' | 'turn' | 'overlay';
  active: boolean;
  onEscape: () => void;
}) {
  useEscapeOwner(layer, active, onEscape);
  return null;
}

describe('useEscapeOwner precedence', () => {
  it('lets the dialog layer win over a concurrently active turn and overlay', () => {
    const onDialogEscape = vi.fn();
    const onTurnEscape = vi.fn();
    const onOverlayEscape = vi.fn();

    render(
      <>
        <Consumer layer="dialog" active onEscape={onDialogEscape} />
        <Consumer layer="turn" active onEscape={onTurnEscape} />
        <Consumer layer="overlay" active onEscape={onOverlayEscape} />
      </>,
    );

    fireEvent.keyDown(window, { key: 'Escape' });

    // Killed by: frontend/src/lib/escapePrecedence.ts :: const PRECEDENCE: readonly EscapeLayer[] = ['dialog', 'turn', 'overlay', 'studio'];
    // Becomes: const PRECEDENCE: readonly EscapeLayer[] = ['turn', 'dialog', 'overlay', 'studio'];
    expect(onDialogEscape).toHaveBeenCalledTimes(1);
    expect(onTurnEscape).not.toHaveBeenCalled();
    expect(onOverlayEscape).not.toHaveBeenCalled();
  });

  it('falls through to overlay once no dialog or turn is active', () => {
    const onOverlayEscape = vi.fn();

    render(<Consumer layer="overlay" active onEscape={onOverlayEscape} />);

    fireEvent.keyDown(window, { key: 'Escape' });

    expect(onOverlayEscape).toHaveBeenCalledTimes(1);
  });

  it('ignores a layer registered but not active', () => {
    const onOverlayEscape = vi.fn();

    render(<Consumer layer="overlay" active={false} onEscape={onOverlayEscape} />);

    fireEvent.keyDown(window, { key: 'Escape' });

    // Killed by: frontend/src/lib/escapePrecedence.ts :: if (!active) return undefined;
    // Becomes: if (false) return undefined;
    expect(onOverlayEscape).not.toHaveBeenCalled();
  });

  it('ignores keys other than Escape', () => {
    const onOverlayEscape = vi.fn();

    render(<Consumer layer="overlay" active onEscape={onOverlayEscape} />);

    fireEvent.keyDown(window, { key: 'Enter' });

    expect(onOverlayEscape).not.toHaveBeenCalled();
  });
});
