import { afterEach, describe, expect, it, vi } from 'vitest';
import { DEVELOPER_MODE_KEY, DEVELOPER_SURFACES, readDeveloperMode, shownSurface } from './developerMode';
import type { DockSurface } from '../types';

describe('developer mode preference', () => {
  afterEach(() => {
    vi.restoreAllMocks();
    window.localStorage.removeItem(DEVELOPER_MODE_KEY);
  });

  it('reads as off when storage refuses, rather than failing the page', () => {
    // Killed by: frontend/src/lib/developerMode.ts :: return false;
    // Becomes: throw new Error('storage refused');
    vi.spyOn(Storage.prototype, 'getItem').mockImplementation(() => {
      throw new Error('storage blocked');
    });
    expect(readDeveloperMode()).toBe(false);
  });

  it('reads a stored choice back', () => {
    window.localStorage.setItem(DEVELOPER_MODE_KEY, 'true');
    expect(readDeveloperMode()).toBe(true);
  });

  it('leaves a user surface on screen whether the mode is on or off', () => {
    // Killed by: frontend/src/lib/developerMode.ts :: !developerMode && isDeveloperSurface(active)
    // Becomes: !developerMode
    expect(shownSurface('activity', false)).toBe('activity');
    expect(shownSurface('knowledge_graph', false)).toBe('artifacts');
    expect(shownSurface('knowledge_graph', true)).toBe('knowledge_graph');
  });
});

describe('the dock hosts only Knowledge Graph in the developer drawer (#1358, #2442)', () => {
  it('lists Knowledge Graph as the sole developer surface', () => {
    expect([...DEVELOPER_SURFACES]).toEqual(['knowledge_graph']);
  });

  it('refuses removed and moved surfaces as dock surfaces at compile time', () => {
    // `tsc` (the `npm run build` the bundle is rebuilt with) fails on each line below if the
    // member comes back to `DockSurface`: an unused `@ts-expect-error` is itself an error. So a
    // dock tab naming one cannot be written without the type being widened first.
    // @ts-expect-error -- Skills is a Settings section now.
    const skills: DockSurface = 'skills';
    // @ts-expect-error -- ACP is in Settings' Diagnostics now.
    const acp: DockSurface = 'acp';
    // @ts-expect-error -- Evals is in Settings' Diagnostics now.
    const evals: DockSurface = 'evaluations';
    // @ts-expect-error -- DAG (topology) was removed (#2442).
    const topology: DockSurface = 'topology';
    // @ts-expect-error -- EventBus (ledger) was removed (#2442).
    const ledger: DockSurface = 'ledger';
    // @ts-expect-error -- Ontology was removed (#2442).
    const ontology: DockSurface = 'ontology';
    expect([skills, acp, evals, topology, ledger, ontology]).toHaveLength(6);
  });
});
