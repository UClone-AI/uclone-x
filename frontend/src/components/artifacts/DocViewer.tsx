import React, { useState, useEffect } from 'react';
import {
  FileText,
  Copy,
  Check,
  Code,
  Eye,
  RefreshCw,
  FolderOpen,
  User,
  Info,
} from 'lucide-react';
import { MarkdownRenderer } from '../MarkdownRenderer';
import { roomDockUrls, useRoomRead, type RoomArtifacts } from '../../lib/roomDock';

interface DocViewerProps {
  /** The conversation on screen; Docs lists only the files it wrote (#1354). */
  roomId: string | null;
  /** A file to front, e.g. from "Open in Docs" on a tool call or an inline image. */
  selectedArtifactPath?: string | null;
  /** The reader picked another file in the selector. */
  onSelectArtifact?: (path: string) => void;
  /** Changes when the conversation moves on, to re-read the list. */
  refreshKey?: unknown;
}

/** What Docs says when the file list's read fails and the Core gave no plain reason. */
export const FILES_READ_FAILED = "This conversation's files could not be listed.";
/** What Docs says when a file's content cannot be read; the route's own text is technical. */
export const FILE_READ_FAILED = 'This file could not be shown.';

const IMAGE_FILE = /\.(png|jpg|jpeg|webp|svg|gif)$/i;

export const DocViewer: React.FC<DocViewerProps> = ({
  roomId,
  selectedArtifactPath,
  onSelectArtifact,
  refreshKey,
}) => {
  const [reloads, setReloads] = useState(0);
  const listRead = useRoomRead<RoomArtifacts>(
    roomId ? roomDockUrls.artifacts(roomId) : null,
    `${String(refreshKey ?? '')}:${reloads}`,
  );
  const { data: listing, loading: fetchingList } = listRead;
  const artifacts = listing?.artifacts ?? [];

  // The file the reader (or an "Open in Docs") picked, remembered with the conversation it
  // was picked in, so switching conversations never carries a file across.
  const [picked, setPicked] = useState<{ roomId: string | null; path: string | null }>({
    roomId,
    path: selectedArtifactPath ?? null,
  });
  useEffect(() => {
    if (selectedArtifactPath) setPicked({ roomId, path: selectedArtifactPath });
    // Only a new request fronts a file; a room switch alone does not re-apply an old one.
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [selectedArtifactPath]);

  const selectedPath =
    (picked.roomId === roomId && picked.path) || artifacts[0]?.path || '';
  const currentArtifact = artifacts.find((a) => a.path === selectedPath);
  const missing = currentArtifact?.exists === false;
  // The disk did not answer for this path (outside the workspace, a failed stat): say why,
  // and do not read it.
  const unchecked = currentArtifact?.exists === null ? currentArtifact.exists_reason : null;

  const [loaded, setLoaded] = useState<{ path: string; content: string; error: string | null }>({
    path: '',
    content: '',
    error: null,
  });
  const [showRaw, setShowRaw] = useState<boolean>(false);
  const [copied, setCopied] = useState<boolean>(false);
  const [imageMeta, setImageMeta] = useState<{
    path: string;
    meta: {
      id?: string;
      prompt?: string;
      negative_prompt?: string;
      seed?: number;
      style?: string;
      aspect_ratio?: string;
      width?: number;
      height?: number;
      engine?: string;
      device?: string;
      duration_seconds?: number;
      created_at?: string;
    } | null;
  }>({ path: '', meta: null });

  useEffect(() => {
    if (!selectedPath || missing || unchecked || !IMAGE_FILE.test(selectedPath)) {
      setImageMeta({ path: '', meta: null });
      return;
    }
    let isMounted = true;
    const companionPath = selectedPath.replace(/\.[^.]+$/, '.json');
    fetch(`/api/artifacts/content?path=${encodeURIComponent(companionPath)}`)
      .then(async (res) => {
        if (!res.ok) return null;
        return (await res.json()) as Record<string, unknown>;
      })
      .then((data) => {
        if (isMounted) {
          setImageMeta({ path: selectedPath, meta: data && typeof data === 'object' ? data : null });
        }
      })
      .catch(() => {
        if (isMounted) setImageMeta({ path: selectedPath, meta: null });
      });
    return () => {
      isMounted = false;
    };
  }, [selectedPath, missing, unchecked]);

  useEffect(() => {
    if (!selectedPath || missing || unchecked || IMAGE_FILE.test(selectedPath)) return;
    let isMounted = true;
    fetch(`/api/artifacts/content?path=${encodeURIComponent(selectedPath)}`)
      .then(async (res) => {
        if (!res.ok) {
          const detail = await res.text();
          throw new Error(`HTTP ${res.status}: ${detail || 'Failed to fetch content'}`);
        }
        return res.text();
      })
      .then((text) => {
        if (isMounted) setLoaded({ path: selectedPath, content: text, error: null });
      })
      .catch((err: Error) => {
        if (isMounted) setLoaded({ path: selectedPath, content: '', error: err.message });
      });
    return () => {
      isMounted = false;
    };
  }, [selectedPath, missing, unchecked]);

  const current = loaded.path === selectedPath ? loaded : null;
  const content = current?.content ?? '';
  const error = current?.error ?? null;
  const loading =
    Boolean(selectedPath) &&
    !missing &&
    !unchecked &&
    !IMAGE_FILE.test(selectedPath) &&
    current === null;

  const selectPath = (path: string) => {
    setPicked({ roomId, path });
    onSelectArtifact?.(path);
  };

  const handleCopy = () => {
    if (!content) return;
    navigator.clipboard.writeText(content);
    setCopied(true);
    setTimeout(() => setCopied(false), 2000);
  };

  // Why nothing is listed, in the Core's words (P6). The list is files saved by name, so
  // an empty one never says that nothing was written (#1366); the fallback, for a Core that
  // sent no reason, says only what the list shows and what it covers.
  const emptyReason = !roomId
    ? 'No conversation is open. Open one from the rail to see the files it wrote.'
    : listRead.fault
      ? // The Core's own plain words where it gave them, else ours; never transport text (#1435).
        (listRead.fault.detail ?? FILES_READ_FAILED)
      : listing && artifacts.length === 0
        ? listing.reason ?? `No files are listed yet. ${listing.scope_note ?? ''}`.trim()
        : null;
  // With files listed, what the list covers and each known reason it may be missing one;
  // the empty list's `reason` already says both.
  const listed = listing !== null && artifacts.length > 0;
  const gaps = listed ? (listing?.record_gaps ?? []) : [];
  // A file opened from elsewhere that this conversation's record does not list.
  const offList = selectedPath && !currentArtifact;

  return (
    <div data-testid="doc-viewer" className="flex flex-col h-full bg-slate-950/40 text-slate-100 min-h-0">
      {/* Top Selector & Action Bar */}
      <div className="flex flex-col gap-2 p-3 bg-slate-900/60 border-b border-slate-800/80 shrink-0">
        <div className="flex items-center justify-between gap-2">
          <div className="flex items-center gap-1.5 min-w-0 flex-1">
            <FileText className="w-4 h-4 text-cyan-400 shrink-0" />
            <select
              data-testid="artifact-selector"
              value={selectedPath}
              onChange={(e) => selectPath(e.target.value)}
              disabled={!roomId}
              className="bg-slate-950 border border-slate-700/70 text-slate-200 text-xs rounded-lg px-2.5 py-1.5 focus:outline-none focus:border-cyan-500 flex-1 min-w-0 truncate font-mono"
            >
              {artifacts.length === 0 && !offList && (
                <option value="">(No files listed)</option>
              )}
              {offList && (
                <option value={selectedPath}>
                  {selectedPath.split('/').pop()} (opened from the conversation)
                </option>
              )}
              {artifacts.map((a) => (
                <option key={a.path} value={a.path}>
                  {a.name} ({a.path})
                </option>
              ))}
            </select>
            <button
              type="button"
              data-testid="refresh-artifacts-btn"
              onClick={() => setReloads((n) => n + 1)}
              disabled={fetchingList || !roomId}
              className="p-1.5 bg-slate-800 hover:bg-slate-700 text-slate-300 rounded-lg transition-colors shrink-0 disabled:opacity-50"
              title="Read this conversation's files again"
            >
              <RefreshCw className={`w-3.5 h-3.5 ${fetchingList ? 'animate-spin text-cyan-400' : ''}`} />
            </button>
          </div>

          <div className="flex items-center gap-1 shrink-0">
            <button
              type="button"
              data-testid="toggle-raw-btn"
              onClick={() => setShowRaw(!showRaw)}
              disabled={!content}
              className={`px-2 py-1 text-xs rounded-lg border font-medium flex items-center gap-1 transition-all ${
                showRaw
                  ? 'bg-amber-600/30 border-amber-500/80 text-amber-300'
                  : 'bg-slate-800/80 border-slate-700/60 text-slate-300 hover:text-white'
              } disabled:opacity-40`}
              title={showRaw ? 'Render markdown' : 'Show raw source'}
            >
              {showRaw ? <Eye className="w-3.5 h-3.5" /> : <Code className="w-3.5 h-3.5" />}
              <span>{showRaw ? 'Rendered' : 'Raw'}</span>
            </button>

            <button
              type="button"
              data-testid="copy-content-btn"
              onClick={handleCopy}
              disabled={!content}
              className="p-1.5 bg-slate-800/80 hover:bg-slate-700 border border-slate-700/60 text-slate-300 hover:text-white rounded-lg transition-colors disabled:opacity-40"
              title="Copy markdown text"
            >
              {copied ? <Check className="w-3.5 h-3.5 text-emerald-400" /> : <Copy className="w-3.5 h-3.5" />}
            </button>
          </div>
        </div>

        {/* Metadata Banner */}
        {selectedPath && (
          <div className="flex flex-wrap items-center justify-between gap-2 text-[11px] text-slate-400 bg-slate-950/80 border border-slate-800/60 px-2.5 py-1.5 rounded-lg">
            <div className="flex items-center gap-1.5 truncate font-mono">
              <FolderOpen className="w-3 h-3 text-slate-500 shrink-0" />
              <span className="truncate text-slate-300">{selectedPath}</span>
            </div>

            <div className="flex items-center gap-3 shrink-0">
              {currentArtifact && (
                <div className="flex items-center gap-1 text-cyan-400/90 font-medium">
                  <User className="w-3 h-3" />
                  <span>
                    {currentArtifact.writers.join(', ')}
                    {currentArtifact.write_count > 1 ? ` · written ${currentArtifact.write_count} times` : ''}
                  </span>
                </div>
              )}
              {currentArtifact && currentArtifact.size_bytes != null && (
                <span>{(currentArtifact.size_bytes / 1024).toFixed(1)} KB</span>
              )}
            </div>
          </div>
        )}
      </div>

      {/* What this list covers and cannot show, in the Core's words (P6, #1366). The known
          gaps are the Core's `record_gaps`, which already count the unnamed writes and the
          unreported turns, so those notes are not repeated beside them. */}
      {listed && listing && (
        <div
          data-testid="doc-list-scope"
          className="px-3 py-2 border-b border-slate-800/80 space-y-1 text-[11px] text-slate-400 shrink-0"
        >
          {listing.scope_note && (
            <p data-testid="doc-scope-note" className="flex gap-1.5">
              <Info className="w-3 h-3 mt-0.5 shrink-0 text-slate-500" />
              <span>{listing.scope_note}</span>
            </p>
          )}
          {gaps.length > 0 && (
            <div data-testid="doc-record-gaps" className="flex gap-1.5">
              <Info className="w-3 h-3 mt-0.5 shrink-0 text-slate-500" />
              <div>
                <span>It may also be missing files because:</span>
                <ul className="list-disc pl-4">
                  {gaps.map((gap) => (
                    <li key={gap}>{gap}</li>
                  ))}
                </ul>
              </div>
            </div>
          )}
          {listing.turn_running && (
            <p data-testid="doc-turn-running" className="flex gap-1.5">
              <Info className="w-3 h-3 mt-0.5 shrink-0 text-slate-500" />
              <span>A turn is still running; the files it saves are listed when it finishes.</span>
            </p>
          )}
        </div>
      )}

      {/* Content Area */}
      <div className="flex-1 overflow-y-auto p-4 min-h-0">
        {emptyReason && !offList ? (
          <div className="flex flex-col items-center justify-center h-64 text-center px-4 text-slate-500">
            <FileText className="w-10 h-10 mb-3 text-slate-600 stroke-[1.5]" />
            <p data-testid="doc-reason" className="text-xs text-slate-300 max-w-sm">
              {emptyReason}
            </p>
          </div>
        ) : missing ? (
          <div
            data-testid="doc-missing-file"
            className="p-4 bg-slate-900/60 border border-slate-800 rounded-xl text-slate-300 text-xs"
          >
            This file was written in this conversation but is no longer on disk, so there is
            nothing to show. It may have been moved or deleted after it was written.
          </div>
        ) : unchecked ? (
          <div
            data-testid="doc-unchecked-file"
            className="p-4 bg-slate-900/60 border border-slate-800 rounded-xl text-slate-300 text-xs"
          >
            {unchecked}
          </div>
        ) : loading ? (
          <div className="flex flex-col items-center justify-center h-48 gap-3 text-slate-400">
            <RefreshCw className="w-6 h-6 animate-spin text-cyan-400" />
            <span className="text-xs">Loading document content...</span>
          </div>
        ) : error ? (
          <div
            data-testid="doc-read-failed"
            className="p-4 bg-rose-950/40 border border-rose-800/60 rounded-xl text-rose-300 text-xs"
          >
            {/* Never the route's text: it is an exception's, and can carry a path (#1435). */}
            {FILE_READ_FAILED}
          </div>
        ) : !selectedPath ? (
          <div className="flex flex-col items-center justify-center h-64 text-center px-4 text-slate-500">
            <FileText className="w-10 h-10 mb-3 text-slate-600 stroke-[1.5]" />
            <p className="text-xs text-slate-400 max-w-sm">
              {fetchingList ? "Reading this conversation's files…" : 'Pick a file above to read it.'}
            </p>
          </div>
        ) : IMAGE_FILE.test(selectedPath) ? (
          <div data-testid="image-artifact-preview" className="flex flex-col items-center justify-center p-4 bg-slate-950/60 rounded-xl border border-slate-800/80">
            <img
              src={`/api/artifacts/content?path=${encodeURIComponent(selectedPath)}`}
              alt={selectedPath}
              className="max-h-[600px] w-auto object-contain rounded-lg shadow-lg border border-slate-800/60"
            />
            <div className="mt-3 flex items-center gap-3 text-xs text-slate-400 font-mono">
              <a
                href={`/api/artifacts/content?path=${encodeURIComponent(selectedPath)}`}
                target="_blank"
                rel="noreferrer"
                className="text-cyan-400 hover:underline"
              >
                Open original in new tab
              </a>
            </div>

            {imageMeta.path === selectedPath && imageMeta.meta && (
              <div
                data-testid="image-metadata-card"
                className="mt-4 w-full max-w-xl bg-slate-900/90 border border-slate-800 rounded-xl p-3.5 text-xs text-slate-300 space-y-2.5 font-sans"
              >
                <div className="flex items-center justify-between pb-1.5 border-b border-slate-800/80">
                  <span className="font-semibold text-cyan-300">
                    Generation Details
                  </span>
                  <div className="flex items-center gap-2 text-[11px] font-mono text-slate-400">
                    {imageMeta.meta.engine && (
                      <span className="px-1.5 py-0.5 rounded bg-cyan-950/70 border border-cyan-800/40 text-cyan-300">
                        {imageMeta.meta.engine}
                      </span>
                    )}
                    {imageMeta.meta.seed != null && <span>Seed: {imageMeta.meta.seed}</span>}
                  </div>
                </div>

                {imageMeta.meta.prompt && (
                  <div>
                    <span className="text-slate-400 block text-[10px] font-mono uppercase tracking-wider mb-1">
                      Prompt
                    </span>
                    <p className="bg-slate-950/80 p-2 rounded-lg border border-slate-800/60 font-mono text-[11px] text-slate-200 select-text whitespace-pre-wrap leading-relaxed">
                      {imageMeta.meta.prompt}
                    </p>
                  </div>
                )}

                <div className="flex flex-wrap items-center gap-x-4 gap-y-1 text-[11px] text-slate-400 font-mono">
                  {imageMeta.meta.style && (
                    <span>Style: <strong className="text-slate-200">{imageMeta.meta.style}</strong></span>
                  )}
                  {imageMeta.meta.aspect_ratio && (
                    <span>Aspect: <strong className="text-slate-200">{imageMeta.meta.aspect_ratio}</strong></span>
                  )}
                  {imageMeta.meta.width != null && imageMeta.meta.height != null && (
                    <span>Size: <strong className="text-slate-200">{imageMeta.meta.width}x{imageMeta.meta.height}</strong></span>
                  )}
                  {imageMeta.meta.duration_seconds != null && (
                    <span>Time: <strong className="text-slate-200">{imageMeta.meta.duration_seconds}s</strong></span>
                  )}
                </div>
              </div>
            )}
          </div>
        ) : showRaw ? (
          <pre data-testid="raw-markdown-pre" className="p-3 bg-slate-950 border border-slate-800/80 rounded-xl font-mono text-xs text-slate-200 overflow-x-auto whitespace-pre-wrap select-text">
            {content}
          </pre>
        ) : (
          <div className="prose prose-invert prose-cyan max-w-none text-xs select-text leading-relaxed">
            <MarkdownRenderer content={content} />
          </div>
        )}
      </div>
    </div>
  );
};

