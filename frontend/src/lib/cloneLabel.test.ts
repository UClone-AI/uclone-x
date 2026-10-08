import { describe, expect, it } from 'vitest';
import { cloneIdOf, cloneLabel } from './cloneLabel';

const SLEEPY = '잠 꾸러기';

describe('cloneLabel', () => {
  // Killed by: frontend/src/lib/cloneLabel.ts :: if (own) return own;
  // Becomes: if (own && false) return own;
  it('reads the display name for the screen language first', () => {
    const clone = { name: 'sleepy', display_name: { en: 'Sleepyhead', ko: SLEEPY } };
    expect(cloneLabel(clone, 'ko')).toBe(SLEEPY);
    expect(cloneLabel(clone, 'en')).toBe('Sleepyhead');
  });

  // Killed by: frontend/src/lib/cloneLabel.ts :: if (trimmed) return trimmed;
  // Becomes: if (trimmed && false) return trimmed;
  it('falls back to the first non-empty display name in another language', () => {
    expect(cloneLabel({ name: 'sleepy', display_name: { en: '  ', ko: SLEEPY } }, 'fr')).toBe(SLEEPY);
    expect(cloneLabel({ name: 'sleepy', display_name: { ko: SLEEPY } }, 'en')).toBe(SLEEPY);
  });

  it('falls back to the id when there is no display name at all', () => {
    expect(cloneLabel({ name: 'sleepy' }, 'ko')).toBe('sleepy');
    expect(cloneLabel({ name: 'sleepy', display_name: {} }, 'ko')).toBe('sleepy');
    expect(cloneLabel({ name: 'sleepy', display_name: { en: ' ' } }, 'en')).toBe('sleepy');
  });
});

describe('cloneIdOf', () => {
  // Killed by: frontend/src/lib/cloneLabel.ts :: export const cloneIdOf = (clone: { id?: string; name: string }): string => clone.id || clone.name;
  // Becomes: export const cloneIdOf = (clone: { id?: string; name: string }): string => clone.name;
  it('keys a clone by its id, and one defined only in memory by its handle (#1814)', () => {
    expect(cloneIdOf({ id: 'agt_1', name: 'scout' })).toBe('agt_1');
    expect(cloneIdOf({ name: 'memo' })).toBe('memo');
    expect(cloneIdOf({ id: '', name: 'memo' })).toBe('memo');
  });
});
