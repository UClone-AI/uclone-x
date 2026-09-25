/**
 * The Files screen's half of `/api/artifacts/library` (#1554).
 *
 * Every decision is the Core's: what is listed, what may move, whether a story is in use.
 * This module only asks, and turns a non-2xx answer into a `CoreFailure` so the screen can
 * show the Core's own sentence, or a fixed one when there was none (`plainFailure`).
 */
import { CoreFailure, failureOf } from './coreFailure';

export type ArtifactKind = 'document' | 'image' | 'story' | 'file';

export interface ConversationRef {
  room_id: string;
  title: string;
}

export interface StoryWriter {
  room_id: string;
  title: string | null;
  exists: boolean;
}

export interface StoryFileEntry {
  path: string;
  name: string;
  kind: ArtifactKind;
  size_bytes: number;
}

export interface StoryInfo {
  story_id: string;
  title: string | null;
  unreadable_reason: string | null;
  writer: StoryWriter | null;
  files: StoryFileEntry[];
}

export interface ArtifactEntry {
  path: string;
  original_path: string;
  name: string;
  kind: ArtifactKind;
  archived: boolean;
  managed: boolean;
  size_bytes: number | null;
  modified_at: string | null;
  conversations: ConversationRef[];
  story: StoryInfo | null;
}

export interface ArtifactSurvey {
  entries: ArtifactEntry[];
  scope_note: string;
  record_gaps: string[];
}

export interface ArtifactContent {
  path: string;
  name: string;
  kind: ArtifactKind;
  text: string | null;
}

export interface StoryOpened {
  room_id: string;
  story_id: string;
  title: string;
  writable: boolean;
  note: string | null;
}

/** A refusal the Core answers 409 to on archive or delete: a conversation is writing the story. */
export const isStoryInUse = (err: unknown): boolean =>
  err instanceof CoreFailure && err.status === 409;

const BASE = '/api/artifacts/library';

async function readJson<T>(res: Response): Promise<T> {
  if (!res.ok) throw await failureOf(res);
  return (await res.json()) as T;
}

const post = (url: string, body: unknown): Promise<Response> =>
  fetch(url, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(body),
  });

export const artifactLibraryApi = {
  survey: async (): Promise<ArtifactSurvey> => readJson<ArtifactSurvey>(await fetch(BASE)),

  open: async (path: string): Promise<ArtifactContent> =>
    readJson<ArtifactContent>(await fetch(`${BASE}/file?path=${encodeURIComponent(path)}`)),

  archive: async (path: string, releaseWriter = false): Promise<{ path: string }> =>
    readJson(await post(`${BASE}/archive`, { path, release_writer: releaseWriter })),

  restore: async (path: string): Promise<{ path: string }> =>
    readJson(await post(`${BASE}/restore`, { path })),

  remove: async (path: string, releaseWriter = false): Promise<{ deleted: string }> =>
    readJson(await post(`${BASE}/delete`, { path, confirm: true, release_writer: releaseWriter })),

  openStory: async (storyId: string, roomId: string): Promise<StoryOpened> =>
    readJson<StoryOpened>(
      await post(`${BASE}/stories/${encodeURIComponent(storyId)}/open`, { room_id: roomId }),
    ),
};

/** The URL an image entry is drawn from: the workspace's existing content route. */
export const imageUrl = (path: string): string =>
  `/api/artifacts/content?path=${encodeURIComponent(path)}`;
