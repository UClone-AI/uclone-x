import { createContext } from 'react';

/**
 * The conversation a piece of text was said in, if any. A file a clone wrote is in that
 * conversation's workspace, which need not be the server's (clone-data-scopes §3.6), so
 * an artifact address drawn inside one names the room for the server to look in.
 */
export const ArtifactRoomContext = createContext<string | null>(null);

/** An `/api/artifacts/content` address, read in `roomId`'s workspace when there is one. */
export const artifactUrlInRoom = (url: string, roomId: string | null): string =>
  roomId ? `${url}&room_id=${encodeURIComponent(roomId)}` : url;
