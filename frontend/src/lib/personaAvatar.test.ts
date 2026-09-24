import { describe, it, expect } from 'vitest';
import { personaAvatarUrl } from './personaAvatar';

/**
 * Where this head looks for a clone's picture (#1300).
 *
 * One line, and it is worth a test for one reason: the clone's name goes into a *path*, and a
 * name is a filename in the workspace, not a vetted identifier. The route on the other end
 * refuses a name no clone carries (`tests/unit/test_persona_avatar.py`), but a name that
 * arrives already split across segments never reaches that guard as one name -- it reaches it
 * as a different route, or as no route at all.
 */
describe('personaAvatarUrl', () => {
  // Killed by: frontend/src/lib/personaAvatar.ts :: encodeURIComponent(name)
  // Becomes: name
  it('spends a clone name as one path segment, whatever characters are in it', () => {
    expect(personaAvatarUrl('surveyor')).toBe('/api/personas/surveyor/avatar');
    expect(personaAvatarUrl('../secrets')).toBe('/api/personas/..%2Fsecrets/avatar');
    expect(personaAvatarUrl('site surveyor')).toBe('/api/personas/site%20surveyor/avatar');
  });
});
