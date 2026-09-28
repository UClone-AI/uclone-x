import { createContext, useContext, useSyncExternalStore } from 'react';
import type { RoomState } from '../types';
import { personaAvatarUrl } from './personaAvatar';

/**
 * Choosing a clone's picture from the head: the calls, and what reaches the places that offer them.
 *
 * Three places change a picture -- the card under a drawn image, the profile's avatar menu,
 * and an Undo after either -- and every surface that shows one has to read it again after,
 * or the rail keeps the old face while the profile shows the new one. So the calls live here,
 * and `AvatarChoice.onChanged` is the one refresh they all end in.
 */

/** A clone a picture can be given to, as the chooser lists it. */
export interface AvatarTarget {
  name: string;
  label: string;
}

/** What the app hands every place that can change a picture. */
export interface AvatarChoice {
  /** Every installed clone, in the order the app lists them. */
  clones: readonly AvatarTarget[];
  /** Called after a picture changed, so every surface that shows one reads it again. */
  onChanged: () => void;
  /**
   * Open a conversation with this clone with a request for a picture waiting in the box.
   * It is not sent: the person reads it, changes it if they like, and sends it themselves.
   */
  askFor: (name: string) => void;
}

export const AvatarChoiceContext = createContext<AvatarChoice | null>(null);
export const useAvatarChoice = () => useContext(AvatarChoiceContext);

/**
 * The clone that wrote the message a card is in, so "Use as avatar" gives the picture to it.
 * Provided by the transcript row, because the Markdown renderers are module constants and
 * cannot be handed a prop per message. `null` outside a clone's message.
 */
export const AvatarAuthorContext = createContext<string | null>(null);
export const useAvatarAuthor = () => useContext(AvatarAuthorContext);

/**
 * The clone behind a seat: its persona, not the seat's id.
 *
 * A room seat's id is the participant's, which is the persona's name for the seat a
 * conversation opens with and something else for a second seat of the same clone.
 */
export const personaOfSeat = (room: RoomState, participantId: string): string => {
  const seat = room.participants.find((p) => p.id === participantId);
  return seat?.persona || participantId;
};

/** Each clone's `avatar_url`, by name: `null` for a clone with no picture. */
export type AvatarUrls = Readonly<Record<string, string | null>>;

/**
 * The `avatar_url` of every listed persona that carries the field. A runtime too old to send
 * it lists nothing here, so every clone falls back to the fixed address.
 */
export const avatarUrlsOf = (
  personas: readonly { name: string; avatar_url?: string | null }[] | undefined,
): AvatarUrls => {
  const urls: Record<string, string | null> = {};
  for (const persona of personas ?? []) {
    if (persona.avatar_url !== undefined) urls[persona.name] = persona.avatar_url;
  }
  return urls;
};

/**
 * Where to draw a clone's picture from.
 *
 * The listed `avatar_url` when the clone is listed, because it changes with the picture and
 * so every surface shows a new one together; `undefined` for a listed clone with none, which
 * draws the default without asking for a picture that is not there. A clone not listed (yet)
 * falls back to the fixed address.
 */
export const pictureOf = (name: string, urls: AvatarUrls | undefined): string | undefined => {
  if (urls && Object.prototype.hasOwnProperty.call(urls, name)) return urls[name] ?? undefined;
  return personaAvatarUrl(name);
};

/**
 * `pictureOf` for one persona in hand: its `avatar_url` when the runtime sent the field,
 * the fixed address otherwise.
 */
export const personaPicture = (
  name: string,
  persona: { avatar_url?: string | null } | undefined,
): string | undefined =>
  persona && persona.avatar_url !== undefined ? (persona.avatar_url ?? undefined) : personaAvatarUrl(name);

/**
 * Why a change did not happen, in the head's own words: never the server's text.
 *
 * The runtime's refusal carries a `code` for which reason it was, and each one asks the
 * person for something different: another file, a smaller one, saving the clone first.
 * `refused` is the wrong format, and also any refusal whose code this head does not know.
 */
export type AvatarFailure =
  | 'unreachable'
  | 'noClone'
  | 'refused'
  | 'tooLarge'
  | 'notSaved'
  | 'noWorkspace'
  | 'outsideWorkspace'
  | 'noFile'
  | 'failed';

export type AvatarResult =
  | { ok: true; previousPath: string | null }
  | { ok: false; failure: AvatarFailure };

const FAILURE_OF_CODE: Readonly<Record<string, AvatarFailure>> = {
  no_clone: 'noClone',
  not_an_image: 'refused',
  too_large: 'tooLarge',
  not_saved: 'notSaved',
  no_workspace: 'noWorkspace',
  outside_workspace: 'outsideWorkspace',
  no_file: 'noFile',
};

const failureOf = (status: number, code: unknown): AvatarFailure => {
  if (typeof code === 'string' && Object.prototype.hasOwnProperty.call(FAILURE_OF_CODE, code)) {
    return FAILURE_OF_CODE[code];
  }
  return status === 404 ? 'noClone' : status === 422 || status === 415 ? 'refused' : 'failed';
};

const codeOf = async (res: Response): Promise<unknown> => {
  try {
    const body: unknown = await res.json();
    return typeof body === 'object' && body !== null ? (body as { code?: unknown }).code : undefined;
  } catch {
    return undefined;
  }
};

async function change(name: string, init: RequestInit): Promise<AvatarResult> {
  let res: Response;
  try {
    res = await fetch(personaAvatarUrl(name), init);
  } catch {
    return { ok: false, failure: 'unreachable' };
  }
  if (!res.ok) return { ok: false, failure: failureOf(res.status, await codeOf(res)) };
  let previousPath: string | null = null;
  try {
    const body: unknown = await res.json();
    if (typeof body === 'object' && body !== null) {
      const raw = (body as { previous_path?: unknown }).previous_path;
      if (typeof raw === 'string' && raw !== '') previousPath = raw;
    }
  } catch {
    // The picture changed; an answer without a readable body only loses the Undo target.
  }
  return { ok: true, previousPath };
}

/** Give `name` the picture at `path`, a file in the workspace such as a drawn image. */
export const setAvatarFromPath = (name: string, path: string): Promise<AvatarResult> =>
  change(name, {
    method: 'PUT',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ source_path: path }),
  });

/** Give `name` a picture from this computer. Check it with `uploadProblem` first. */
export const uploadAvatar = (name: string, file: File): Promise<AvatarResult> =>
  change(name, {
    method: 'PUT',
    headers: { 'Content-Type': uploadTypeOf(file) ?? 'application/octet-stream' },
    body: file,
  });

/** Put `name`'s chosen picture aside, so it shows its own again or the default. */
export const resetAvatar = (name: string): Promise<AvatarResult> =>
  change(name, { method: 'DELETE' });

/** Take back a change: put back the picture it replaced, or reset when there was none. */
export const undoAvatarChange = (name: string, previousPath: string | null): Promise<AvatarResult> =>
  previousPath ? setAvatarFromPath(name, previousPath) : resetAvatar(name);

/** The formats the upload offers. GIF is also accepted by the runtime, but not offered. */
export const UPLOAD_TYPES = ['image/png', 'image/jpeg', 'image/webp'] as const;
export const UPLOAD_ACCEPT = '.png,.jpg,.jpeg,.webp,image/png,image/jpeg,image/webp';
/** The runtime's own limit on a clone's picture. */
export const MAX_UPLOAD_BYTES = 10 * 1024 * 1024;

const UPLOAD_EXTENSIONS: Record<string, string> = {
  png: 'image/png',
  jpg: 'image/jpeg',
  jpeg: 'image/jpeg',
  webp: 'image/webp',
};

/** The type to send a file under, or `null` for one the upload does not take. */
export const uploadTypeOf = (file: { name: string; type: string }): string | null => {
  if ((UPLOAD_TYPES as readonly string[]).includes(file.type)) return file.type;
  if (file.type !== '') return null;
  const ext = file.name.split('.').pop()?.toLowerCase() ?? '';
  return UPLOAD_EXTENSIONS[ext] ?? null;
};

/** Why a chosen file cannot be uploaded, before it is sent; `null` when it can. */
export const uploadProblem = (file: {
  name: string;
  type: string;
  size: number;
}): 'wrongType' | 'tooLarge' | null => {
  if (uploadTypeOf(file) === null) return 'wrongType';
  if (file.size > MAX_UPLOAD_BYTES) return 'tooLarge';
  return null;
};

/**
 * The last change made to each clone's picture from this head, as a count.
 *
 * An Undo puts back what one change replaced, so it is only right while that change is
 * still the latest: after a later one it would put back the wrong picture, or reset a newer
 * choice. Each place that offers an Undo notes the count its change left and shows the
 * Undo only while the count is the same.
 */
const changeCounts = new Map<string, number>();
const changeListeners = new Set<() => void>();

/** Note that `name`'s picture changed; returns the count this change left. */
export const noteAvatarChange = (name: string): number => {
  const next = (changeCounts.get(name) ?? 0) + 1;
  changeCounts.set(name, next);
  for (const listener of changeListeners) listener();
  return next;
};

/** The count the latest change to `name`'s picture left; `0` before any. */
export const useLatestAvatarChange = (name: string | null): number =>
  useSyncExternalStore(
    (listener) => {
      changeListeners.add(listener);
      return () => changeListeners.delete(listener);
    },
    () => (name === null ? 0 : (changeCounts.get(name) ?? 0)),
  );

/**
 * Whether a drawn picture at `path` is one a clone can wear. The runtime takes PNG, JPEG,
 * WebP and GIF only, so an SVG is never offered.
 */
export const canWearPicture = (path: string): boolean => !/\.svg$/i.test(path.split(/[?#]/)[0]);
