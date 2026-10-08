import { describe, expect, it } from 'vitest';
import { withCloneNames } from './cloneSeats';
import { makePersonaInfo } from '../test/fixtures';
import type { RoomState } from '../types';

const room = (participants: RoomState['participants']): RoomState =>
  ({ room_id: 'r1', title: 't', participants, transcript: [] }) as unknown as RoomState;

describe('withCloneNames', () => {
  // Killed by: frontend/src/lib/cloneSeats.ts :: return { ...seat, display_name: cloneLabel(clone, language), handle: clone.name };
  // Becomes: return { ...seat, handle: clone.name };
  it('names a seat by its clone`s display name today, not the one it sat down with (#1814)', () => {
    const scout = makePersonaInfo({ id: 'agt_1', name: 'scout', display_name: { en: 'Scout', ko: '정찰' } });
    const named = withCloneNames(
      room([
        { id: 'user', kind: 'human', display_name: 'Kenny' },
        { id: 'agt_1', kind: 'agent', display_name: 'old name' },
      ]),
      [scout],
      'ko',
    );

    expect(named.participants[1]).toMatchObject({ id: 'agt_1', display_name: '정찰', handle: 'scout' });
    expect(named.participants[0]).toEqual({ id: 'user', kind: 'human', display_name: 'Kenny' });
  });

  // Killed by: frontend/src/lib/cloneSeats.ts :: const clone = seat.kind === 'agent' ? byId.get(seat.id) : undefined;
  // Becomes: const clone = byId.get(seat.id);
  it('leaves a person, and a clone the listing does not carry, as the Core sent them', () => {
    const state = room([
      { id: 'scout', kind: 'human', display_name: 'A person called scout' },
      { id: 'agt_gone', kind: 'agent', display_name: 'Gone' },
    ]);

    const named = withCloneNames(state, [makePersonaInfo({ name: 'scout' })], 'en');

    expect(named).toBe(state);
  });
});
