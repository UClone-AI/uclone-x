/**
 * Where a clone's picture is.
 *
 * Pictures only (#1300). The endpoint answers with the image file installed beside the
 * clone's definition, and 404s when there is none -- which is the ordinary case, not a
 * fault. Nothing here draws a face from a name and nothing generates one: a clone with no
 * picture gets `Avatar`'s single default, so the same clone looks the same on every install
 * and a drawn face can never be mistaken for a likeness.
 *
 * The name is encoded because it lands in a path segment: a clone whose name carries a
 * space or a slash would otherwise compose a URL that addresses something else.
 */
export const personaAvatarUrl = (name: string): string =>
  `/api/personas/${encodeURIComponent(name)}/avatar`;
