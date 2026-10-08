/**
 * What a person reads as a clone's name, and what every request names it by.
 *
 * A clone has three names (clone-data-scopes §3.1). Its `id` (`agt_…`) is what a room seats
 * and every request sends; it never changes. Its handle -- `name` here -- is the `@mention`
 * token, lowercase ASCII, and a rename changes it. Its `display_name` is free text per
 * locale, e.g. `{ ko: '...', en: 'Sleepyhead' }`. Every label a user reads comes from
 * `cloneLabel`; every key a request or a seat carries comes from `cloneIdOf`.
 *
 * Label order: the name for the screen's language, then the first non-empty name in any
 * language (a clone named only in Korean is still called that on an English screen, rather
 * than by its handle), then the handle.
 */
export interface Labelled {
  /** The clone's id. Absent from a clone defined only in memory, whose id is its handle. */
  id?: string;
  name: string;
  display_name?: Record<string, string>;
}

/** The key a seat, a room's `agent_ids` and every request name this clone by. */
export const cloneIdOf = (clone: { id?: string; name: string }): string => clone.id || clone.name;

export const cloneLabel = (clone: Labelled, language: string): string => {
  const names = clone.display_name ?? {};
  const own = names[language]?.trim();
  if (own) return own;
  for (const text of Object.values(names)) {
    const trimmed = text?.trim();
    if (trimmed) return trimmed;
  }
  return clone.name;
};
