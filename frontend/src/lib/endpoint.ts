/**
 * Whether two server addresses typed into Settings name the same server.
 *
 * Only for comparing: the address the user typed is still what is saved and sent. Two
 * spellings of one server — `http://localhost:11434` and `http://127.0.0.1:11434/` — are one
 * address here, so a form showing one does not disown the saved state of the other.
 */

// The loopback spellings a local server answers on. The port still has to match.
const LOOPBACK_HOSTS = new Set(['localhost', '127.0.0.1', '[::1]']);

// scheme://authority rest — the authority is host plus an optional port.
const URL_SHAPE = /^([a-z][a-z0-9+.-]*):\/\/([^/?#]*)(.*)$/i;
// A bracketed IPv6 host or a plain one, then an optional :port.
const AUTHORITY_SHAPE = /^(\[[^\]]*\]|[^:]*)(?::(\d*))?$/;

/**
 * The comparable form of an address: trimmed, scheme and host lower-cased, trailing slashes
 * dropped, and every loopback host spelled `localhost`.
 */
export function normalizeEndpoint(endpoint: string): string {
  const trimmed = endpoint.trim().replace(/\/+$/, '');
  const url = URL_SHAPE.exec(trimmed);
  if (!url) return trimmed;
  const [, scheme, authority, rest] = url;
  const parts = AUTHORITY_SHAPE.exec(authority.toLowerCase());
  if (!parts) return `${scheme.toLowerCase()}://${authority.toLowerCase()}${rest}`;
  const [, rawHost, port] = parts;
  const host = LOOPBACK_HOSTS.has(rawHost) ? 'localhost' : rawHost;
  return `${scheme.toLowerCase()}://${host}${port ? `:${port}` : ''}${rest}`;
}

/** True when the two addresses name the same server (see {@link normalizeEndpoint}). */
export function sameEndpoint(a: string, b: string): boolean {
  return normalizeEndpoint(a) === normalizeEndpoint(b);
}
