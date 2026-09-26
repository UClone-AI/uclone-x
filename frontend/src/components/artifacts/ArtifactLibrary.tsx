import React, { useCallback, useEffect, useMemo, useState } from 'react';
import { X, FileText, Image as ImageIcon, BookOpen, File, RefreshCw } from 'lucide-react';
import { Button } from '../ui/Button';
import { MarkdownRenderer } from '../MarkdownRenderer';
import { StoryView } from './StoryView';
import { useEscapeOwner } from '../../lib/escapePrecedence';
import { plainFailure } from '../../lib/coreFailure';
import {
  artifactLibraryApi,
  imageUrl,
  isStoryInUse,
  type ArtifactContent,
  type ArtifactEntry,
  type ArtifactKind,
  type ArtifactSurvey,
} from '../../lib/artifactLibrary';

/**
 * The Files screen (#1554): every file the clones saved in the workspace's artifact folders,
 * across conversations -- including ones since deleted -- apart from the room-scoped Docs dock.
 *
 * The list is the Core's (`ArtifactLibrary.survey`), and so is every refusal: this screen
 * shows the Core's sentence, or one of the fixed sentences below when it gave none. It never
 * says the list is complete; it shows the Core's `scope_note` and each `record_gaps` line.
 */

/** The plain `note` an archive or delete answered with, if any (#1578). */
const noteOf = (result: unknown): string | null =>
  typeof result === 'object' && result !== null && 'note' in result && typeof result.note === 'string'
    ? result.note
    : null;

export const FILES_COPY = {
  title: 'Files',
  loading: 'Loading the files…',
  listFailed: 'The files could not be listed.',
  openFailed: 'This file could not be shown.',
  actionFailed: 'That did not work.',
  noneListed: 'No files are listed.',
  noneMatch: 'No listed file matches these filters.',
  // A file with no link may be a deleted conversation's, or a live one's that went unrecorded
  // (a shell command, say), so this never claims its writer is gone (#1578).
  notLinked: 'No recorded writer',
  unmanaged:
    'Saved outside the artifacts and stories folders, so it can be opened here but not archived or deleted.',
  confirmDelete: (name: string) => `Delete ${name} for good? This cannot be undone.`,
  goAhead: 'Go ahead anyway',
  /** After going ahead was itself refused: the same request again, and it says so (#1578). */
  tryAgain: 'Try going ahead again',
  archived: (name: string) => `${name} was archived. Show archived files to restore it.`,
  restored: (name: string) => `${name} was restored.`,
  deleted: (name: string) => `${name} was deleted.`,
  writer: (title: string | null, exists: boolean) =>
    exists
      ? `Being written in “${title ?? 'a conversation that could not be read'}”`
      : 'Its last writer was a conversation that no longer exists',
  pickFile: 'Choose Open on a file to read it here.',
} as const;

type KindFilter = 'all' | ArtifactKind;
/** `''` is every conversation; `'none'` is files with no recorded writer. */
type ConversationFilter = string;

const KIND_LABELS: Record<KindFilter, string> = {
  all: 'All kinds',
  document: 'Documents',
  image: 'Images',
  story: 'Stories',
  file: 'Other files',
};

const KIND_ICONS: Record<ArtifactKind, React.ComponentType<{ className?: string }>> = {
  document: FileText,
  image: ImageIcon,
  story: BookOpen,
  file: File,
};

const formatSize = (bytes: number | null): string => {
  if (bytes === null) return '';
  if (bytes < 1000) return `${bytes} B`;
  if (bytes < 1_000_000) return `${Math.round(bytes / 1000)} KB`;
  return `${(bytes / 1_000_000).toFixed(1)} MB`;
};

const formatWhen = (iso: string | null): string => (iso ? new Date(iso).toLocaleString() : '');

interface ArtifactLibraryProps {
  isOpen: boolean;
  onClose: () => void;
  /** Make a new conversation with this story open; rejects with the reason it could not. */
  onOpenStory: (storyId: string, title: string) => Promise<void>;
}

type Pending =
  | { kind: 'delete'; entry: ArtifactEntry }
  | {
      kind: 'in-use';
      entry: ArtifactEntry;
      action: 'archive' | 'delete';
      reason: string;
      /** The refused request was already a go-ahead, so offering one again is a retry. */
      retry: boolean;
    };

export const ArtifactLibrary: React.FC<ArtifactLibraryProps> = ({ isOpen, onClose, onOpenStory }) => {
  const [survey, setSurvey] = useState<ArtifactSurvey | null>(null);
  const [loading, setLoading] = useState(false);
  const [listError, setListError] = useState<string | null>(null);
  const [kind, setKind] = useState<KindFilter>('all');
  const [conversation, setConversation] = useState<ConversationFilter>('');
  const [showArchived, setShowArchived] = useState(false);
  const [opened, setOpened] = useState<ArtifactContent | null>(null);
  const [openError, setOpenError] = useState<string | null>(null);
  const [pending, setPending] = useState<Pending | null>(null);
  const [notice, setNotice] = useState<string | null>(null);
  const [busy, setBusy] = useState(false);
  /** The story whose view is on screen in place of the list (#1560), or `null`. */
  const [viewing, setViewing] = useState<string | null>(null);

  useEscapeOwner('dialog', isOpen, onClose);

  const load = useCallback(async () => {
    setLoading(true);
    setListError(null);
    try {
      setSurvey(await artifactLibraryApi.survey());
    } catch (err) {
      console.error('Files: the list could not be read', err);
      setListError(plainFailure(err, FILES_COPY.listFailed));
    } finally {
      setLoading(false);
    }
  }, []);

  useEffect(() => {
    if (!isOpen) return;
    setOpened(null);
    setOpenError(null);
    setPending(null);
    setNotice(null);
    setViewing(null);
    void load();
  }, [isOpen, load]);

  const conversations = useMemo(() => {
    const seen = new Map<string, string>();
    for (const entry of survey?.entries ?? []) {
      for (const ref of entry.conversations) seen.set(ref.room_id, ref.title);
    }
    return [...seen.entries()].sort((a, b) => a[1].localeCompare(b[1]));
  }, [survey]);

  const visible = useMemo(
    () =>
      (survey?.entries ?? []).filter(
        (entry) =>
          entry.archived === showArchived &&
          (kind === 'all' || entry.kind === kind) &&
          (conversation === '' ||
            (conversation === 'none'
              ? entry.conversations.length === 0
              : entry.conversations.some((ref) => ref.room_id === conversation))),
      ),
    [survey, showArchived, kind, conversation],
  );

  const openFile = async (path: string) => {
    setOpenError(null);
    try {
      setOpened(await artifactLibraryApi.open(path));
    } catch (err) {
      console.error('Files: a file could not be opened', err);
      setOpened(null);
      setOpenError(plainFailure(err, FILES_COPY.openFailed));
    }
  };

  const act = async (
    run: () => Promise<unknown>,
    done: string,
    inUse?: Pending,
  ) => {
    setBusy(true);
    setNotice(null);
    try {
      const result = await run();
      setPending(null);
      setOpened(null);
      // The change happened; a step after it that did not is said beside it, not dropped (#1578).
      const note = noteOf(result);
      setNotice(note ? `${done} ${note}` : done);
      await load();
    } catch (err) {
      console.error('Files: an action failed', err);
      const reason = plainFailure(err, FILES_COPY.actionFailed);
      if (inUse && inUse.kind === 'in-use' && isStoryInUse(err)) {
        setPending({ ...inUse, reason });
      } else {
        setPending(null);
        setNotice(reason);
      }
    } finally {
      setBusy(false);
    }
  };

  const archive = (entry: ArtifactEntry, releaseWriter = false) =>
    act(
      () => artifactLibraryApi.archive(entry.path, releaseWriter),
      FILES_COPY.archived(entry.name),
      { kind: 'in-use', entry, action: 'archive', reason: '', retry: releaseWriter },
    );

  const remove = (entry: ArtifactEntry, releaseWriter = false) =>
    act(
      () => artifactLibraryApi.remove(entry.path, releaseWriter),
      FILES_COPY.deleted(entry.name),
      { kind: 'in-use', entry, action: 'delete', reason: '', retry: releaseWriter },
    );

  const restore = (entry: ArtifactEntry) =>
    act(() => artifactLibraryApi.restore(entry.path), FILES_COPY.restored(entry.name));

  const openStory = async (entry: ArtifactEntry) => {
    if (!entry.story) return;
    setBusy(true);
    setNotice(null);
    try {
      await onOpenStory(entry.story.story_id, entry.story.title ?? entry.name);
      onClose();
    } catch (err) {
      console.error('Files: the story could not be opened in a new conversation', err);
      setNotice(plainFailure(err, FILES_COPY.actionFailed));
    } finally {
      setBusy(false);
    }
  };

  if (!isOpen) return null;

  return (
    <div
      className="fixed inset-0 z-50 bg-black/60 flex items-center justify-center p-4"
      data-testid="artifact-library"
    >
      <div
        role="dialog"
        aria-modal="true"
        aria-label={FILES_COPY.title}
        className="w-full max-w-5xl h-[85vh] bg-slate-900 border border-slate-700 rounded-xl flex flex-col overflow-hidden"
      >
        <div className="flex items-center justify-between px-4 py-3 border-b border-slate-800">
          <h2 className="text-sm font-semibold text-slate-100">{FILES_COPY.title}</h2>
          <div className="flex items-center gap-2">
            <Button variant="bordered" size="icon" onClick={() => void load()} disabled={loading} title="Refresh the list">
              <RefreshCw className="w-3.5 h-3.5" />
            </Button>
            <Button variant="ghost" size="icon" onClick={onClose} title="Close" aria-label="Close">
              <X className="w-4 h-4" />
            </Button>
          </div>
        </div>

        {viewing !== null && (
          <StoryView
            storyId={viewing}
            onBack={() => {
              setViewing(null);
              void load();
            }}
          />
        )}
        {viewing === null && (
          <>
          <div className="flex flex-wrap items-center gap-2 px-4 py-2 border-b border-slate-800 text-xs">
            <label className="flex items-center gap-1 text-slate-400">
              Kind
              <select
                data-testid="files-kind-filter"
                className="bg-slate-950 border border-slate-700 rounded px-1 py-0.5 text-slate-200"
                value={kind}
                onChange={(e) => setKind(e.target.value as KindFilter)}
              >
                {(Object.keys(KIND_LABELS) as KindFilter[]).map((k) => (
                  <option key={k} value={k}>
                    {KIND_LABELS[k]}
                  </option>
                ))}
              </select>
            </label>
            <label className="flex items-center gap-1 text-slate-400">
              Conversation
              <select
                data-testid="files-conversation-filter"
                className="bg-slate-950 border border-slate-700 rounded px-1 py-0.5 text-slate-200 max-w-[16rem]"
                value={conversation}
                onChange={(e) => setConversation(e.target.value)}
              >
                <option value="">All conversations</option>
                <option value="none">{FILES_COPY.notLinked}</option>
                {conversations.map(([id, title]) => (
                  <option key={id} value={id}>
                    {title}
                  </option>
                ))}
              </select>
            </label>
            <label className="flex items-center gap-1 text-slate-400">
              <input
                type="checkbox"
                data-testid="files-show-archived"
                checked={showArchived}
                onChange={(e) => setShowArchived(e.target.checked)}
              />
              Show archived files
            </label>
          </div>

          {notice && (
            <p role="status" data-testid="files-notice" className="px-4 py-2 text-xs text-amber-200 border-b border-slate-800">
              {notice}
            </p>
          )}

          <div className="flex-1 min-h-0 grid grid-cols-1 md:grid-cols-2">
            <div className="min-h-0 overflow-y-auto border-r border-slate-800">
              {loading && !survey && <p className="p-4 text-xs text-slate-400">{FILES_COPY.loading}</p>}
              {listError && (
                <div className="p-4 text-xs text-rose-300" role="alert">
                  <p>{listError}</p>
                  <Button variant="outline" className="mt-2" onClick={() => void load()}>
                    Try again
                  </Button>
                </div>
              )}
              {survey && (
                <>
                  <p data-testid="files-scope-note" className="px-4 pt-3 text-[11px] text-slate-400">
                    {survey.scope_note}
                  </p>
                  {survey.record_gaps.length > 0 && (
                    <ul data-testid="files-record-gaps" className="px-4 pt-1 text-[11px] text-amber-300 list-disc list-inside">
                      {survey.record_gaps.map((gap) => (
                        <li key={gap}>{gap}</li>
                      ))}
                    </ul>
                  )}
                  {visible.length === 0 && (
                    <p data-testid="files-empty" className="p-4 text-xs text-slate-400">
                      {survey.entries.length === 0 ? FILES_COPY.noneListed : FILES_COPY.noneMatch}
                    </p>
                  )}
                  <ul className="p-2 space-y-1">
                    {visible.map((entry) => (
                      <EntryRow
                        key={entry.path}
                        entry={entry}
                        busy={busy}
                        pending={pending?.entry.path === entry.path ? pending : null}
                        onOpen={() => void openFile(entry.path)}
                        onOpenInner={(path) => void openFile(path)}
                        onArchive={(release) => void archive(entry, release)}
                        onRestore={() => void restore(entry)}
                        onAskDelete={() => setPending({ kind: 'delete', entry })}
                        onDelete={(release) => void remove(entry, release)}
                        onCancel={() => setPending(null)}
                        onOpenStory={() => void openStory(entry)}
                        onViewStory={() => entry.story && setViewing(entry.story.story_id)}
                      />
                    ))}
                  </ul>
                </>
              )}
            </div>
            <div className="min-h-0 overflow-y-auto p-4" data-testid="files-preview">
              {openError && (
                <p role="alert" className="text-xs text-rose-300">
                  {openError}
                </p>
              )}
              {!openError && !opened && <p className="text-xs text-slate-500">{FILES_COPY.pickFile}</p>}
              {opened && (
                <div>
                  <p className="text-xs font-mono text-slate-400 mb-2">{opened.path}</p>
                  {opened.kind === 'image' ? (
                    <img src={imageUrl(opened.path)} alt={opened.name} className="max-w-full rounded" />
                  ) : opened.path.toLowerCase().endsWith('.md') ? (
                    <MarkdownRenderer content={opened.text ?? ''} />
                  ) : (
                    <pre data-testid="files-preview-text" className="text-xs whitespace-pre-wrap text-slate-200">
                      {opened.text}
                    </pre>
                  )}
                </div>
              )}
            </div>
          </div>
          </>
        )}
      </div>
    </div>
  );
};

interface EntryRowProps {
  entry: ArtifactEntry;
  busy: boolean;
  pending: Pending | null;
  onOpen: () => void;
  onOpenInner: (path: string) => void;
  onArchive: (releaseWriter: boolean) => void;
  onRestore: () => void;
  onAskDelete: () => void;
  onDelete: (releaseWriter: boolean) => void;
  onCancel: () => void;
  onOpenStory: () => void;
  onViewStory: () => void;
}

const EntryRow: React.FC<EntryRowProps> = ({
  entry,
  busy,
  pending,
  onOpen,
  onOpenInner,
  onArchive,
  onRestore,
  onAskDelete,
  onDelete,
  onCancel,
  onOpenStory,
  onViewStory,
}) => {
  const Icon = KIND_ICONS[entry.kind];
  const story = entry.story;
  return (
    <li data-testid="files-entry" data-path={entry.path} className="rounded-lg border border-slate-800 bg-slate-950/60 p-2 text-xs">
      <div className="flex items-start gap-2">
        <Icon className="w-4 h-4 mt-0.5 text-slate-400 shrink-0" />
        <div className="min-w-0 flex-1">
          <p className="font-medium text-slate-100 truncate">{entry.name}</p>
          <p className="font-mono text-[10px] text-slate-500 truncate">{entry.original_path}</p>
          <p className="text-[11px] text-slate-400">
            {[formatSize(entry.size_bytes), formatWhen(entry.modified_at)].filter(Boolean).join(' · ')}
          </p>
          <p className="text-[11px] text-slate-400">
            {entry.conversations.length > 0
              ? entry.conversations.map((c) => c.title).join(', ')
              : FILES_COPY.notLinked}
          </p>
          {story?.writer && (
            <p className="text-[11px] text-slate-400">{FILES_COPY.writer(story.writer.title, story.writer.exists)}</p>
          )}
          {story?.unreadable_reason && <p className="text-[11px] text-amber-300">{story.unreadable_reason}</p>}
          {!entry.managed && <p className="text-[11px] text-slate-500">{FILES_COPY.unmanaged}</p>}
          {story && story.files.length > 0 && (
            <ul className="mt-1 space-y-0.5">
              {story.files.map((file) => (
                <li key={file.path}>
                  <button
                    type="button"
                    className="font-mono text-[10px] text-cyan-300 hover:underline"
                    onClick={() => onOpenInner(file.path)}
                  >
                    {file.name}
                  </button>
                </li>
              ))}
            </ul>
          )}
        </div>
      </div>

      {pending?.kind === 'delete' && (
        <div role="alertdialog" aria-label="Confirm delete" className="mt-2 rounded border border-rose-800 bg-rose-950/40 p-2">
          <p className="text-rose-200">{FILES_COPY.confirmDelete(entry.name)}</p>
          <div className="mt-2 flex gap-2">
            <Button variant="solid" disabled={busy} data-testid="files-confirm-delete" onClick={() => onDelete(false)}>
              Delete for good
            </Button>
            <Button variant="outline" disabled={busy} onClick={onCancel}>
              Keep it
            </Button>
          </div>
        </div>
      )}
      {pending?.kind === 'in-use' && (
        <div role="alert" className="mt-2 rounded border border-amber-800 bg-amber-950/40 p-2">
          <p className="text-amber-200">{pending.reason}</p>
          <div className="mt-2 flex gap-2">
            <Button
              variant="solid"
              disabled={busy}
              data-testid="files-go-ahead"
              onClick={() => (pending.action === 'archive' ? onArchive(true) : onDelete(true))}
            >
              {pending.retry ? FILES_COPY.tryAgain : FILES_COPY.goAhead}
            </Button>
            <Button variant="outline" disabled={busy} onClick={onCancel}>
              Cancel
            </Button>
          </div>
        </div>
      )}

      {!pending && (
        <div className="mt-2 flex flex-wrap gap-1">
          {entry.kind !== 'story' && (
            <Button variant="outline" disabled={busy} data-testid="files-open" onClick={onOpen}>
              Open
            </Button>
          )}
          {story && !entry.archived && story.title !== null && (
            <Button variant="outline" disabled={busy} data-testid="files-view-story" onClick={onViewStory}>
              View story
            </Button>
          )}
          {story && !entry.archived && story.title !== null && (
            <Button variant="outline" disabled={busy} data-testid="files-open-story" onClick={onOpenStory}>
              Open in a new conversation
            </Button>
          )}
          {entry.managed && !entry.archived && (
            <Button variant="outline" disabled={busy} data-testid="files-archive" onClick={() => onArchive(false)}>
              Archive
            </Button>
          )}
          {entry.archived && (
            <Button variant="outline" disabled={busy} data-testid="files-restore" onClick={onRestore}>
              Restore
            </Button>
          )}
          {entry.managed && (
            <Button variant="outline" disabled={busy} data-testid="files-delete" onClick={onAskDelete}>
              Delete
            </Button>
          )}
        </div>
      )}
    </li>
  );
};
