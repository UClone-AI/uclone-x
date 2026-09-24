import type React from 'react';

/**
 * What the kit needs from an icon: a component that takes a class.
 *
 * Icons are injected, never imported (#1158). A kit that imported `lucide-react` would make
 * lucide's version the kit's dependency, and the two heads that consume the kit do not agree
 * on it -- uclone2 is on 0.378 and this repository is on 1.x. Any lucide icon satisfies this
 * type as it stands, and so does any other component that accepts `className`.
 */
export type KitIcon = React.ComponentType<{ className?: string }>;

/**
 * Join class names, dropping the falsy ones.
 *
 * **It resolves no conflicts.** The head's own `cn` runs `tailwind-merge`, so a caller could
 * pass `p-1` to a button whose size is `p-1.5` and have the second one win. That package is not
 * a dependency the kit may take, so here both would be emitted and the stylesheet, not the
 * caller, would decide. Kit components therefore take no `className` where a variant sets the
 * same property; each look they need is a named variant instead.
 */
export const cx = (...parts: Array<string | false | null | undefined>): string =>
  parts.filter(Boolean).join(' ');
