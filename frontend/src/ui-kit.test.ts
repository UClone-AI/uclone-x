import { describe, it, expect } from 'vitest';
import { importSpecifiers, isAllowedImport, kitBoundaryOffences } from './lib/kitBoundary';

/**
 * `ui-kit/` imports nothing from outside itself but `react` (#1063 D, #1158).
 *
 * The kit is portable by construction or not at all: one import of the store, an API client,
 * a copy module or `lucide-react` and it can no longer be lifted into another head as it
 * stands, and nothing about the rail's rendering would show it. So this reads the sources.
 *
 * It sits outside `ui-kit/` because it must: it imports `vitest`, which the rule refuses. The
 * sources arrive through Vite's raw glob for the reason `copy.test.ts` gives (no `@types/node`).
 *
 * Before this there was no guard. #1161 said in its commit message that the rail "stays
 * props-only"; no test said so, so this is the first check of that property, not a successor.
 */

//: Every script under `ui-kit/`, as text, keyed `./ui-kit/...` like the rule expects.
const KIT_SOURCES: Record<string, string> = import.meta.glob(
  './ui-kit/**/*.{ts,tsx,js,jsx,mjs,cjs}',
  { query: '?raw', import: 'default', eager: true },
);

describe('the ui-kit boundary (#1158)', () => {
  // Killed by: frontend/src/ui-kit/rail/PersonaDetail.tsx :: import type { PersonaDetailCopy, RailPersona } from './types';
  // Becomes: import type { PersonaDetailCopy, RailPersona } from '../../components/layout/types';
  it('holds: nothing under ui-kit/ imports from outside it except react', () => {
    // A glob that matched nothing would pass having read nothing. The kit has eleven modules.
    expect(Object.keys(KIT_SOURCES).length).toBeGreaterThanOrEqual(11);

    expect(kitBoundaryOffences(KIT_SOURCES)).toEqual([]);
  });

  // Killed by: frontend/src/lib/kitBoundary.ts :: if (!spec.startsWith('.')) return false;
  // Becomes: if (!spec.startsWith('.')) return true;
  it('refuses every package but react, whatever form names it', () => {
    const planted = {
      './ui-kit/rail/Planted.tsx': [
        "import React from 'react';",
        "import { Bot } from 'lucide-react';",
        "import type { AgentInfo } from 'zustand';",
        "export { twMerge } from 'tailwind-merge';",
        "import 'some-side-effect';",
        "const lazy = () => import('react-dom');",
        "const cjs = require('clsx');",
      ].join('\n'),
    };

    expect(kitBoundaryOffences(planted)).toEqual([
      './ui-kit/rail/Planted.tsx: lucide-react',
      './ui-kit/rail/Planted.tsx: zustand',
      './ui-kit/rail/Planted.tsx: tailwind-merge',
      './ui-kit/rail/Planted.tsx: some-side-effect',
      './ui-kit/rail/Planted.tsx: react-dom',
      './ui-kit/rail/Planted.tsx: clsx',
    ]);
  });

  // Killed by: frontend/src/lib/kitBoundary.ts :: return resolveRelative(path, spec).startsWith(KIT_ROOT);
  // Becomes: return true;
  it('refuses a relative import that climbs out of the kit, and allows one that stays in', () => {
    const path = './ui-kit/rail/Rail.tsx';

    expect(isAllowedImport(path, './PersonaDetail')).toBe(true);
    expect(isAllowedImport(path, '../primitives/Button')).toBe(true);
    expect(isAllowedImport(path, '../kit')).toBe(true);
    expect(isAllowedImport(path, '../../store')).toBe(false);
    expect(isAllowedImport(path, '../../api')).toBe(false);
    expect(isAllowedImport(path, '../../components/ui/Button')).toBe(false);
    // A sibling whose name starts with the kit's is not the kit.
    expect(isAllowedImport(path, '../../ui-kit-extras/x')).toBe(false);
  });

  // Killed by: frontend/src/lib/kitBoundary.ts :: const UNREADABLE_FORMS: readonly RegExp[] = [/\bimport\s*\(\s*(?!['"])/, /\bimport\.meta\b/];
  // Becomes: const UNREADABLE_FORMS: readonly RegExp[] = [];
  it('refuses an import whose target the source does not spell', () => {
    const offences = kitBoundaryOffences({
      './ui-kit/a.ts': 'const target = "../store"; export const load = () => import(target);',
      './ui-kit/b.ts': "export const all = import.meta.glob('../**/*.ts');",
    });

    expect(offences).toHaveLength(2);
    expect(offences[0]).toMatch(/^\.\/ui-kit\/a\.ts: an import the rule cannot resolve/);
    expect(offences[1]).toMatch(/^\.\/ui-kit\/b\.ts: an import the rule cannot resolve/);
  });

  // Killed by: frontend/src/lib/kitBoundary.ts :: const code = withoutComments(source);
  // Becomes: const code = source;
  it('reads imports, not comments that mention them', () => {
    expect(
      importSpecifiers(
        [
          "// import { useStore } from '../../store';",
          "/* export * from 'lucide-react'; */",
          "import React from 'react';",
        ].join('\n'),
      ),
    ).toEqual(['react']);
  });

  // Killed by: frontend/src/lib/kitBoundary.ts :: .filter(([path]) => path.startsWith(KIT_ROOT))
  // Becomes: .filter(() => true)
  it('reads only the kit, so the rest of the head may import what it likes', () => {
    expect(kitBoundaryOffences({ './store.ts': "import x from 'zustand';" })).toEqual([]);
  });
});
