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

/** A conversation a story view names; `exists` is false once it was deleted (#1560). */
export interface StoryConversation {
  room_id: string;
  title: string | null;
  exists: boolean;
}

export interface SceneView {
  id: string;
  title: string;
  summary: string;
  story_time: string | number | null;
  characters: string[];
  places: string[];
  written: boolean;
}

export interface ChapterView {
  id: string;
  title: string;
  act: string | null;
  scenes: SceneView[];
}

export interface OutlineView {
  chapters: ChapterView[];
  story_order: string[];
  assumptions: string[];
}

export interface Unreadable {
  file: string;
  reason: string;
}

export type JsonValue = string | number | boolean | null | JsonValue[] | { [key: string]: JsonValue };

export interface EntryView {
  id: string;
  name: string;
  aliases: string[];
  profile: string;
  state: Record<string, JsonValue>;
  progressions: Record<string, JsonValue>[];
  looks: string[] | null;
}

export interface CodexGroup {
  kind: string;
  label: string;
  entries: EntryView[];
}

export interface EvidenceView {
  scene_id: string;
  scene_title: string | null;
  quote: string;
  still_in_scene: boolean | null;
}

export interface ChangeView {
  what: string;
  at: string | null;
  before: JsonValue;
  after: JsonValue;
  placed: boolean;
}

export interface ProposalView {
  id: string;
  status: string;
  kind: string;
  entry_id: string;
  entry_name: string | null;
  proposed_at: string;
  proposed_in: StoryConversation;
  agent_id: string | null;
  note: string | null;
  evidence: EvidenceView[];
  changes: ChangeView[];
  diff: string[];
  blocked: string | null;
  digest: string;
  decided_at: string | null;
  decided_in: string | null;
  reason: string | null;
}

/** A story's view (#1560): its outline, its codex, and the changes waiting for a person. */
export interface StoryOverview {
  story_id: string;
  title: string;
  logline: string | null;
  genre: string | null;
  writer: StoryConversation | null;
  decide_note: string | null;
  outline: OutlineView | null;
  outline_note: string | null;
  codex: CodexGroup[];
  codex_unreadable: Unreadable[];
  pending: ProposalView[];
  decided: ProposalView[];
  proposals_unreadable: Unreadable[];
}

export interface ProposalDecided {
  proposal_id: string;
  decision: string;
  notes: string[];
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

const proposalUrl = (storyId: string, proposalId: string, decision: 'approve' | 'reject'): string =>
  `${BASE}/stories/${encodeURIComponent(storyId)}/proposals/${encodeURIComponent(proposalId)}/${decision}`;

/**
 * What an archive did. `note` is a plain sentence about a step after the move that did not
 * complete -- the conversations that had a story open not all being cleared -- or null (#1578).
 */
export interface ArtifactChanged {
  path: string;
  note: string | null;
}

/** What a delete did; `note` as on `ArtifactChanged`. */
export interface ArtifactRemoved {
  deleted: string;
  note: string | null;
}

export const artifactLibraryApi = {
  survey: async (): Promise<ArtifactSurvey> => readJson<ArtifactSurvey>(await fetch(BASE)),

  open: async (path: string): Promise<ArtifactContent> =>
    readJson<ArtifactContent>(await fetch(`${BASE}/file?path=${encodeURIComponent(path)}`)),

  archive: async (path: string, releaseWriter = false): Promise<ArtifactChanged> =>
    readJson(await post(`${BASE}/archive`, { path, release_writer: releaseWriter })),

  restore: async (path: string): Promise<{ path: string }> =>
    readJson(await post(`${BASE}/restore`, { path })),

  remove: async (path: string, releaseWriter = false): Promise<ArtifactRemoved> =>
    readJson(await post(`${BASE}/delete`, { path, confirm: true, release_writer: releaseWriter })),

  openStory: async (storyId: string, roomId: string): Promise<StoryOpened> =>
    readJson<StoryOpened>(
      await post(`${BASE}/stories/${encodeURIComponent(storyId)}/open`, { room_id: roomId }),
    ),
  showStory: async (storyId: string): Promise<StoryOverview> =>
    readJson<StoryOverview>(await fetch(`${BASE}/stories/${encodeURIComponent(storyId)}`)),

  /** Decide a proposal as it was shown: `seenDigest` is the digest the view drew it from. */
  approveProposal: async (storyId: string, proposalId: string, seenDigest: string): Promise<ProposalDecided> =>
    readJson<ProposalDecided>(await post(proposalUrl(storyId, proposalId, 'approve'), { seen_digest: seenDigest })),

  rejectProposal: async (
    storyId: string,
    proposalId: string,
    seenDigest: string,
    reason: string,
  ): Promise<ProposalDecided> =>
    readJson<ProposalDecided>(
      await post(proposalUrl(storyId, proposalId, 'reject'), { seen_digest: seenDigest, reason }),
    ),
};

/** The URL an image entry is drawn from: the workspace's existing content route. */
export const imageUrl = (path: string): string =>
  `/api/artifacts/content?path=${encodeURIComponent(path)}`;
