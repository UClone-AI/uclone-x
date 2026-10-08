/**
 * Where a clone's picture is.
 *
 * Pictures only (#1300). The endpoint answers with the image file installed beside the
 * clone's definition, and 404s when there is none -- which is the ordinary case, not a
 * fault. Nothing here draws a face from a name and nothing generates one: a clone with no
 * picture gets `Avatar`'s single default, so the same clone looks the same on every install
 * and a drawn face can never be mistaken for a likeness.
 *
 * The clone is named by its id (a handle is answered too). It is encoded because it lands in
 * a path segment: a name carrying a space or a slash would otherwise address something else.
 */
export const personaAvatarUrl = (clone: string): string =>
  `/api/clones/${encodeURIComponent(clone)}/avatar`;
