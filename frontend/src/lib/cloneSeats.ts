/**
 * A room's clone seats, named the way the clone listing names them.
 *
 * A seat is its clone's id (clone-data-scopes §4 step 3), which is not a name anybody reads
 * or types. The seat's stored `display_name` is what the clone was called when it sat down,
 * and a rename since is not in it. So the head names each seat from the listing: its label
 * (`cloneLabel`, the display name for this screen's language) and its handle, which `@`
 * inserts. A seat whose clone the listing does not carry is left as the Core sent it.
 */
import type { PersonaInfo, RoomState } from '../types';
import { cloneIdOf, cloneLabel } from './cloneLabel';

export const withCloneNames = (
  room: RoomState,
  clones: readonly PersonaInfo[],
  language: string,
): RoomState => {
  const byId = new Map(clones.map((clone) => [cloneIdOf(clone), clone]));
  let named = false;
  const participants = room.participants.map((seat) => {
    const clone = seat.kind === 'agent' ? byId.get(seat.id) : undefined;
    if (!clone) return seat;
    named = true;
    return { ...seat, display_name: cloneLabel(clone, language), handle: clone.name };
  });
  return named ? { ...room, participants } : room;
};
