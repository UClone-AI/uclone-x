import React, { useCallback, useEffect, useRef, useState } from 'react';
import { ArrowLeft, RefreshCw } from 'lucide-react';
import { Button } from '../ui/Button';
import { fmt, plural, useCopy, type Messages } from '../../i18n';
import { CoreFailure, plainFailure } from '../../lib/coreFailure';
import { openConfirmedWindow } from '../../lib/person';
import {
  artifactLibraryApi,
  type ChangeView,
  type EvidenceView,
  type JsonValue,
  type ChapterView,
  type OutlineView,
  type ProposalView,
  type SceneView,
  type StoryConversation,
  type StoryOverview,
} from '../../lib/artifactLibrary';

/**
 * A story's view in Files (#1560): the outline, the codex, and the changes waiting for a person.
 *
 * Everything shown is the Core's (`StoryView.show`), and so is every decision: Approve and
 * Reject send the proposal's digest as it was shown, and the Core applies it only if the
 * proposal and its entry are still what this screen drew. A refusal is the Core's sentence.
 * No story or file tool reaches this path. Any program on this computer can call the same
 * local API, a persona's unconfined shell (Clone's `bash_run`) included, so the Core accepts a
 * decision only from a window it opened itself (#1589, `lib/person.ts`); that is what makes
 * "Applied here" mean a person pressed the button. A window opened any other way is refused,
 * shows the Core's sentence, and offers to open a confirmed one.
 *
 * The outline is a story board (§1.1 row 5 of the novel-writer design): chapters and their
 * scenes, each scene with the codex entries in it, the changes the codex records there, what
 * the check after writing found, and the changes waiting for a decision that rest on it. An
 * entry on the board opens in the codex below. The board shows names, never ids or the
 * reasoner's words, so it needs nothing from developer mode.
 */

type StoryCopy = Messages['dock']['story'];

const show = (value: JsonValue, t: StoryCopy): string => {
  if (value === null || value === undefined) return t.notSet;
  if (Array.isArray(value)) return value.length === 0 ? t.notSet : value.map((v) => show(v, t)).join(', ');
  if (typeof value === 'object') return JSON.stringify(value);
  return String(value);
};

const proposedIn = (c: StoryConversation, t: StoryCopy): string =>
  !c.exists ? t.proposedInGone : c.title === null ? t.proposedInUnreadable : fmt(t.proposedIn, { title: c.title });

const decision = (p: ProposalView, t: StoryCopy): string => {
  const applied = p.status === 'applied';
  if (p.decided_in === 'story_view') return applied ? t.appliedHere : t.rejectedHere;
  if (p.decided_in === 'conversation') return applied ? t.appliedInConversation : t.rejectedInConversation;
  return applied ? t.appliedElsewhere : t.rejectedElsewhere;
};

const kindLabel = (kind: string, t: StoryCopy): string =>
  (t.kinds as Record<string, string>)[kind] ?? kind;

/**
 * A note the Core sent with its code, in the reader's language (`dock.story.notes`); the Core's
 * English sentence only for a code this head does not know.
 */
const noteText = (code: string | null | undefined, sentence: string | null, story: StoryOverview, t: StoryCopy): string | null => {
  const notes: Record<string, string> = t.notes;
  if (code && Object.prototype.hasOwnProperty.call(notes, code)) return fmt(notes[code], { title: story.title });
  return sentence;
};

/** A refusal the Core sent with its code, in the reader's language (`dock.story.refusals`). */
const refusalText = (err: unknown, fallback: string, refusals: Record<string, string>): string => {
  if (err instanceof CoreFailure && err.coreCode !== null && Object.prototype.hasOwnProperty.call(refusals, err.coreCode)) {
    return refusals[err.coreCode];
  }
  return plainFailure(err, fallback);
};

const formatWhen = (iso: string | null): string => (iso ? new Date(iso).toLocaleString() : '');

interface StoryViewProps {
  storyId: string;
  onBack: () => void;
}

/** An entry chosen on the board, shown in the codex. */
interface Chosen {
  kind: string;
  id: string;
}

export const StoryView: React.FC<StoryViewProps> = ({ storyId, onBack }) => {
  const t = useCopy().dock.story;
  const [story, setStory] = useState<StoryOverview | null>(null);
  const [loading, setLoading] = useState(false);
  const [loadError, setLoadError] = useState<string | null>(null);
  const [notice, setNotice] = useState<string[] | null>(null);
  const [busy, setBusy] = useState(false);
  // The Core refused a decision from this window: it was not opened by the Core (#1589).
  const [unconfirmed, setUnconfirmed] = useState(false);
  const [chosen, setChosen] = useState<Chosen | null>(null);

  const load = useCallback(async () => {
    setLoading(true);
    setLoadError(null);
    try {
      setStory(await artifactLibraryApi.showStory(storyId));
    } catch (err) {
      console.error('Story view: the story could not be read', err);
      setLoadError(refusalText(err, t.loadFailed, t.refusals));
    } finally {
      setLoading(false);
    }
  }, [storyId, t.loadFailed, t.refusals]);

  useEffect(() => {
    setStory(null);
    setNotice(null);
    setChosen(null);
    void load();
  }, [load]);

  const decide = async (proposal: ProposalView, reject: string | null) => {
    const name = proposal.entry_name ?? proposal.entry_id;
    setBusy(true);
    setNotice(null);
    setUnconfirmed(false);
    try {
      const done =
        reject === null
          ? await artifactLibraryApi.approveProposal(storyId, proposal.id, proposal.digest)
          : await artifactLibraryApi.rejectProposal(storyId, proposal.id, proposal.digest, reject);
      setNotice([fmt(done.decision === 'applied' ? t.applied : t.rejected, { name }), ...done.notes]);
    } catch (err) {
      console.error('Story view: a decision failed', err);
      setUnconfirmed(err instanceof CoreFailure && err.status === 403);
      setNotice([refusalText(err, t.decideFailed, t.refusals)]);
    } finally {
      setBusy(false);
    }
    await load();
  };

  const openWindow = async () => {
    try {
      await openConfirmedWindow();
      setNotice([t.windowOpened]);
    } catch (err) {
      console.error('Story view: no confirmed window opened', err);
      setNotice([plainFailure(err, t.windowFailed)]);
    }
  };

  return (
    <div className="flex-1 min-h-0 flex flex-col" data-testid="story-view">
      <div className="flex items-center gap-2 px-4 py-2 border-b border-slate-800">
        <Button variant="ghost" onClick={onBack} data-testid="story-view-back">
          <ArrowLeft className="w-3.5 h-3.5" />
          {t.back}
        </Button>
        <div className="min-w-0 flex-1">
          {story && (
            <p className="text-sm font-semibold text-slate-100 truncate" data-testid="story-view-title">
              {story.title}
            </p>
          )}
        </div>
        <Button variant="bordered" size="icon" onClick={() => void load()} disabled={loading} title={t.refresh}>
          <RefreshCw className="w-3.5 h-3.5" />
        </Button>
      </div>

      {notice && (
        <div
          role="status"
          data-testid="story-view-notice"
          className="px-4 py-2 text-xs text-amber-200 border-b border-slate-800"
        >
          {notice.map((line) => (
            <p key={line}>{line}</p>
          ))}
          {unconfirmed && (
            <Button
              variant="bordered"
              className="mt-2"
              onClick={() => void openWindow()}
              data-testid="story-open-confirmed-window"
            >
              {t.openConfirmed}
            </Button>
          )}
        </div>
      )}

      <div className="flex-1 min-h-0 overflow-y-auto p-4 space-y-6 text-xs">
        {loading && !story && <p className="text-slate-400">{t.loading}</p>}
        {loadError && (
          <div role="alert" className="text-rose-300">
            <p>{loadError}</p>
            <Button variant="outline" className="mt-2" onClick={() => void load()}>
              {t.tryAgain}
            </Button>
          </div>
        )}
        {story && (
          <>
            {(story.logline || story.genre) && (
              <p className="text-slate-400">{[story.genre, story.logline].filter(Boolean).join(' · ')}</p>
            )}
            <Waiting story={story} busy={busy} onDecide={(p, reason) => void decide(p, reason)} />
            <Outline story={story} onChoose={setChosen} />
            <Codex story={story} chosen={chosen} />
            <Decided story={story} />
          </>
        )}
      </div>
    </div>
  );
};

const Section: React.FC<{
  title: string;
  testId: string;
  children: React.ReactNode;
}> = ({ title, testId, children }) => (
  <section data-testid={testId}>
    <h3 className="text-[11px] font-semibold uppercase tracking-wide text-slate-400 mb-2">{title}</h3>
    {children}
  </section>
);

const Unreadables: React.FC<{
  items: { file: string; reason: string }[];
  testId: string;
  heading: string;
}> = ({ items, testId, heading }) =>
  items.length === 0 ? null : (
    <div data-testid={testId} className="mt-2 text-amber-300">
      <p>{heading}</p>
      <ul className="list-disc list-inside">
        {items.map((u) => (
          <li key={u.file}>
            <span className="font-mono">{u.file}</span>: {u.reason}
          </li>
        ))}
      </ul>
    </div>
  );

const Waiting: React.FC<{
  story: StoryOverview;
  busy: boolean;
  onDecide: (proposal: ProposalView, reject: string | null) => void;
}> = ({ story, busy, onDecide }) => {
  const t = useCopy().dock.story;
  const sceneTitles = new Map(
    (story.outline?.chapters ?? []).flatMap((ch) => ch.scenes.map((sc) => [sc.id, sc.title] as const)),
  );
  return (
    <Section title={t.waiting} testId="story-pending">
      {story.pending.length > 0 && story.decide_note && (
        <p role="note" data-testid="story-decide-note" className="mb-2 text-amber-200">
          {noteText(story.decide_code, story.decide_note, story, t)}
        </p>
      )}
      {story.pending.length === 0 && <p className="text-slate-400">{t.noneWaiting}</p>}
      <ul className="space-y-3">
        {story.pending.map((p) => (
          <PendingProposal
            key={p.id}
            proposal={p}
            canDecide={!story.decide_note && !busy}
            sceneTitles={sceneTitles}
            onDecide={onDecide}
          />
        ))}
      </ul>
      <Unreadables items={story.proposals_unreadable} testId="story-proposals-unreadable" heading={t.unreadable} />
    </Section>
  );
};

const Provenance: React.FC<{ proposal: ProposalView }> = ({ proposal }) => {
  const t = useCopy().dock.story;
  return (
    <p className="text-[11px] text-slate-400">
      {proposedIn(proposal.proposed_in, t)} · {formatWhen(proposal.proposed_at)}
    </p>
  );
};

const Evidence: React.FC<{ evidence: EvidenceView[] }> = ({ evidence }) => {
  const t = useCopy().dock.story;
  return evidence.length === 0 ? null : (
    <div className="mt-2">
      <p className="text-[11px] text-slate-400">{t.from}</p>
      {evidence.map((e) => (
        <figure key={`${e.scene_id}:${e.quote}`} data-testid="story-evidence" className="mt-1">
          <blockquote className="border-l-2 border-slate-600 pl-2 text-slate-200 italic">{e.quote}</blockquote>
          <figcaption className="text-[11px] text-slate-500">
            {e.scene_title ?? e.scene_id}
            {e.still_in_scene === false && <span className="text-amber-300"> — {t.quoteGone}</span>}
            {e.still_in_scene === null && <span> — {t.quoteUnchecked}</span>}
          </figcaption>
        </figure>
      ))}
    </div>
  );
};

const Changes: React.FC<{ changes: ChangeView[]; sceneTitles: Map<string, string> }> = ({ changes, sceneTitles }) => {
  const t = useCopy().dock.story;
  return changes.length === 0 ? null : (
    <table className="mt-2 w-full text-left" data-testid="story-changes">
      <thead className="text-[11px] text-slate-500">
        <tr>
          <th className="font-normal pr-2">{t.what}</th>
          <th className="font-normal pr-2">{t.when}</th>
          <th className="font-normal pr-2">{t.now}</th>
          <th className="font-normal">{t.ifApproved}</th>
        </tr>
      </thead>
      <tbody>
        {changes.map((c) => (
          <tr key={`${c.what}:${c.at ?? ''}`} data-testid="story-change" className="align-top">
            <td className="pr-2 text-slate-300">{c.what}</td>
            <td className="pr-2 text-slate-400">
              {c.at === null ? t.fromStart : fmt(t.after, { scene: sceneTitles.get(c.at) ?? c.at })}
              {!c.placed && <p className="text-[11px] text-slate-500">{t.unplaced}</p>}
            </td>
            <td className="pr-2 text-slate-400" data-testid="story-change-before">
              {show(c.before, t)}
            </td>
            <td className="text-slate-100" data-testid="story-change-after">
              {show(c.after, t)}
            </td>
          </tr>
        ))}
      </tbody>
    </table>
  );
};

const Diff: React.FC<{ lines: string[] }> = ({ lines }) => {
  const t = useCopy().dock.story;
  return lines.length === 0 ? null : (
    <details className="mt-2">
      <summary className="cursor-pointer text-[11px] text-slate-400">{t.fileChange}</summary>
      <pre data-testid="story-diff" className="mt-1 overflow-x-auto rounded bg-slate-950 p-2 font-mono text-[11px]">
        {lines.map((line, i) => (
          <div
            key={i}
            className={
              line.startsWith('+') && !line.startsWith('+++')
                ? 'text-emerald-300'
                : line.startsWith('-') && !line.startsWith('---')
                  ? 'text-rose-300'
                  : 'text-slate-400'
            }
          >
            {line || ' '}
          </div>
        ))}
      </pre>
    </details>
  );
};

const PendingProposal: React.FC<{
  proposal: ProposalView;
  canDecide: boolean;
  sceneTitles: Map<string, string>;
  onDecide: (proposal: ProposalView, reject: string | null) => void;
}> = ({ proposal, canDecide, sceneTitles, onDecide }) => {
  const t = useCopy().dock.story;
  const [rejecting, setRejecting] = useState(false);
  const [reason, setReason] = useState('');
  return (
    <li
      data-testid="story-proposal"
      data-proposal={proposal.id}
      className="rounded-lg border border-slate-800 bg-slate-950/60 p-3"
    >
      <p className="font-medium text-slate-100">{proposal.entry_name ?? proposal.entry_id}</p>
      <Provenance proposal={proposal} />
      {proposal.note && <p className="mt-1 text-slate-300">{proposal.note}</p>}
      <Changes changes={proposal.changes} sceneTitles={sceneTitles} />
      <Evidence evidence={proposal.evidence} />
      <Diff lines={proposal.diff} />
      {proposal.blocked && (
        <p role="note" data-testid="story-blocked" className="mt-2 text-amber-200">
          {proposal.blocked}
        </p>
      )}
      {!rejecting && (
        <div className="mt-3 flex gap-2">
          <Button
            variant="solid"
            data-testid="story-approve"
            disabled={!canDecide || proposal.blocked !== null}
            onClick={() => onDecide(proposal, null)}
          >
            {t.approve}
          </Button>
          <Button variant="outline" data-testid="story-reject" disabled={!canDecide} onClick={() => setRejecting(true)}>
            {t.reject}
          </Button>
        </div>
      )}
      {rejecting && (
        <div className="mt-3 space-y-2">
          <label className="block text-slate-400">
            {t.rejectReason}
            <input
              data-testid="story-reject-reason"
              className="mt-1 block w-full bg-slate-950 border border-slate-700 rounded px-2 py-1 text-slate-200"
              value={reason}
              onChange={(e) => setReason(e.target.value)}
            />
          </label>
          <div className="flex gap-2">
            <Button
              variant="solid"
              data-testid="story-confirm-reject"
              disabled={!canDecide}
              onClick={() => onDecide(proposal, reason)}
            >
              {t.confirmReject}
            </Button>
            <Button variant="outline" onClick={() => setRejecting(false)}>
              {t.keep}
            </Button>
          </div>
        </div>
      )}
    </li>
  );
};

const Outline: React.FC<{
  story: StoryOverview;
  onChoose: (entry: Chosen) => void;
}> = ({ story, onChoose }) => {
  const t = useCopy().dock.story;
  const [order, setOrder] = useState<'reading' | 'story'>('reading');
  const outline: OutlineView | null = story.outline;
  if (!outline) {
    return (
      <Section title={t.outline} testId="story-outline">
        <p className="text-slate-400" data-testid="story-outline-note">
          {noteText(story.outline_code, story.outline_note, story, t)}
        </p>
      </Section>
    );
  }
  const scenes = new Map(outline.chapters.flatMap((ch) => ch.scenes.map((s) => [s.id, s] as const)));
  const waitingNames = new Map(story.pending.map((p) => [p.id, p.entry_name ?? p.entry_id] as const));
  return (
    <Section title={t.outline} testId="story-outline">
      <div className="mb-2 flex gap-1" role="group" aria-label={t.sceneOrder}>
        {(['reading', 'story'] as const).map((o) => (
          <Button
            key={o}
            variant={order === o ? 'solid' : 'ghost'}
            aria-pressed={order === o}
            data-testid={`story-order-${o}`}
            onClick={() => setOrder(o)}
          >
            {o === 'reading' ? t.readingOrder : t.storyOrder}
          </Button>
        ))}
      </div>
      {story.board_note && (
        <p role="note" data-testid="story-board-note" className="mb-2 text-amber-200">
          {noteText(story.board_code, story.board_note, story, t)}
        </p>
      )}
      {order === 'reading' ? (
        <ol className="space-y-3" data-testid="story-board">
          {outline.chapters.map((ch) => (
            <BoardChapter key={ch.id} chapter={ch} waitingNames={waitingNames} onChoose={onChoose} />
          ))}
        </ol>
      ) : (
        <ol className="space-y-1" data-testid="story-order-list">
          {outline.story_order.map((id) => {
            const s = scenes.get(id);
            return (
              <li key={id} data-testid="story-scene" data-scene={id}>
                <span className="text-slate-100">{s?.title ?? id}</span>
                {!(s?.written ?? false) && <span className="text-slate-500"> ({t.unwritten})</span>}
                {s?.summary && <p className="text-slate-400">{s.summary}</p>}
              </li>
            );
          })}
        </ol>
      )}
      {order === 'story' && outline.assumptions.length > 0 && (
        <ul className="mt-2 text-[11px] text-slate-500 list-disc list-inside" data-testid="story-assumptions">
          {outline.assumptions.map((a) => (
            <li key={a}>{a}</li>
          ))}
        </ul>
      )}
    </Section>
  );
};

const BoardChapter: React.FC<{
  chapter: ChapterView;
  waitingNames: Map<string, string>;
  onChoose: (entry: Chosen) => void;
}> = ({ chapter, waitingNames, onChoose }) => {
  const b = useCopy().dock.story.board;
  const functions = b.functions as Record<string, string>;
  // A function the head has no word for is not shown: its code is the arc's, not the reader's.
  const marks = [
    ...chapter.functions.map((f) => functions[f]).filter((f): f is string => Boolean(f)),
    ...(chapter.twist ? [b.twist] : []),
    ...(chapter.climax ? [b.climax] : []),
  ];
  return (
    <li
      data-testid="story-chapter"
      data-chapter={chapter.id}
      className="rounded-lg border border-slate-800 bg-slate-950/40 p-2"
    >
      <p className="text-slate-200 font-medium">
        {chapter.act ? `${chapter.act} · ` : ''}
        {chapter.title}
      </p>
      {marks.length > 0 && (
        <p className="mt-0.5 flex flex-wrap gap-1" data-testid="story-chapter-marks">
          {marks.map((m) => (
            <span key={m} className="rounded bg-slate-800 px-1.5 py-0.5 text-[10px] text-slate-300">
              {m}
            </span>
          ))}
        </p>
      )}
      <ol className="mt-2 space-y-2">
        {chapter.scenes.map((s) => (
          <BoardScene key={s.id} scene={s} waitingNames={waitingNames} onChoose={onChoose} />
        ))}
      </ol>
    </li>
  );
};

const changeLine = (c: SceneView['changes'][number], b: StoryCopy['board'], t: StoryCopy): string => {
  if (c.note) return c.note;
  const parts = Object.entries(c.set).map(([k, v]) => `${k}: ${show(v, t)}`);
  if (c.add_looks.length > 0) parts.push(fmt(b.looksAdded, { looks: c.add_looks.join(', ') }));
  if (c.remove_looks.length > 0) parts.push(fmt(b.looksRemoved, { looks: c.remove_looks.join(', ') }));
  return parts.join(' · ');
};

const BoardScene: React.FC<{
  scene: SceneView;
  waitingNames: Map<string, string>;
  onChoose: (entry: Chosen) => void;
}> = ({ scene, waitingNames, onChoose }) => {
  const t = useCopy().dock.story;
  const b = t.board;
  const how = b.how as Record<string, string>;
  const waiting = scene.pending.map((id) => waitingNames.get(id)).filter((n): n is string => Boolean(n));
  return (
    <li
      data-testid="story-scene"
      data-scene={scene.id}
      data-written={scene.written}
      className="border-l-2 border-slate-700 pl-2"
    >
      <p>
        <span className="text-slate-100">{scene.title}</span>{' '}
        <span className={scene.written ? 'text-emerald-300' : 'text-slate-500'} data-testid="story-scene-state">
          ({scene.written ? b.written : b.planned})
        </span>
      </p>
      {scene.summary && <p className="text-slate-400">{scene.summary}</p>}
      {scene.entries.length > 0 && (
        <div className="mt-1 flex flex-wrap items-center gap-1" data-testid="story-scene-entries">
          <span className="text-[11px] text-slate-500">{b.inScene}</span>
          {scene.entries.map((e) => {
            const label = how[e.how] ? `${e.name} · ${how[e.how]}` : e.name;
            return e.in_codex ? (
              <button
                key={`${e.kind}:${e.id}`}
                type="button"
                data-testid="story-scene-entry"
                data-entry={e.id}
                title={fmt(b.showInCodex, { name: e.name })}
                onClick={() => onChoose({ kind: e.kind, id: e.id })}
                className="rounded border border-sky-800 bg-sky-950/40 px-1.5 py-0.5 text-sky-200 hover:bg-sky-900/60"
              >
                {label}
              </button>
            ) : (
              <span
                key={`${e.kind}:${e.id}`}
                data-testid="story-scene-entry"
                data-entry={e.id}
                className="rounded border border-slate-700 px-1.5 py-0.5 text-slate-400"
              >
                {fmt(b.notInCodex, { name: e.name })}
              </span>
            );
          })}
        </div>
      )}
      {scene.changes.length > 0 && (
        <div className="mt-1" data-testid="story-scene-changes">
          <p className="text-[11px] text-slate-500">{b.changesHere}</p>
          <ul className="ml-2">
            {scene.changes.map((c, i) => (
              <li key={`${c.kind}:${c.entry_id}:${i}`} className="text-slate-300">
                <button
                  type="button"
                  className="text-sky-200 hover:underline"
                  title={fmt(b.showInCodex, { name: c.entry_name })}
                  onClick={() => onChoose({ kind: c.kind, id: c.entry_id })}
                >
                  {c.entry_name}
                </button>
                : {changeLine(c, b, t)}
              </li>
            ))}
          </ul>
        </div>
      )}
      {scene.findings.length > 0 && (
        <div role="note" className="mt-1 text-amber-200" data-testid="story-scene-findings">
          <p className="text-[11px]">{b.findings}</p>
          {scene.findings.map((f) => (
            <figure key={`${f.quote}:${f.note}`} className="mt-0.5">
              {f.quote && (
                <blockquote className="border-l-2 border-amber-700 pl-2 italic text-amber-100">{f.quote}</blockquote>
              )}
              <figcaption>{f.note}</figcaption>
            </figure>
          ))}
        </div>
      )}
      {scene.findings.length === 0 && scene.continuity === 'checked' && (
        <p className="mt-1 text-[11px] text-slate-500" data-testid="story-scene-checked">
          {b.checkedClean}
        </p>
      )}
      {scene.continuity === 'unread' && (
        <p className="mt-1 text-[11px] text-amber-300" data-testid="story-scene-unread">
          {b.unread}
        </p>
      )}
      {waiting.length > 0 && (
        <p className="mt-1 text-amber-200" data-testid="story-scene-waiting">
          {plural(b.waiting, waiting.length)}: {waiting.join(', ')}
        </p>
      )}
    </li>
  );
};

const Codex: React.FC<{ story: StoryOverview; chosen: Chosen | null }> = ({ story, chosen }) => {
  const t = useCopy().dock.story;
  const chosenRef = useRef<HTMLLIElement | null>(null);
  useEffect(() => {
    // jsdom has no scrollIntoView; a browser does.
    chosenRef.current?.scrollIntoView?.({ block: 'nearest', behavior: 'smooth' });
  }, [chosen]);
  return (
    <Section title={t.codex} testId="story-codex">
      <div className="space-y-3">
        {story.codex.map((group) => {
          const label = kindLabel(group.kind, t);
          return (
            <div key={group.kind} data-testid="story-codex-group" data-kind={group.kind}>
              <p className="text-slate-300 font-medium">{label}</p>
              {group.entries.length === 0 ? (
                <p className="text-slate-500">{fmt(t.noEntries, { label: label.toLowerCase() })}</p>
              ) : (
                <ul className="mt-1 space-y-1">
                  {group.entries.map((e) => {
                    const isChosen = chosen !== null && chosen.kind === group.kind && chosen.id === e.id;
                    return (
                      <li
                        key={e.id}
                        ref={isChosen ? chosenRef : undefined}
                        data-testid="story-entry"
                        data-entry={e.id}
                        data-chosen={isChosen || undefined}
                        aria-current={isChosen || undefined}
                        className={isChosen ? 'rounded ring-1 ring-sky-500 bg-sky-950/30 p-1' : undefined}
                      >
                        <span className="text-slate-100">{e.name}</span>
                        {e.aliases.length > 0 && <span className="text-slate-500"> ({e.aliases.join(', ')})</span>}
                        {e.profile && <p className="text-slate-400">{e.profile}</p>}
                        {Object.keys(e.state).length > 0 && (
                          <p className="text-slate-400">
                            {Object.entries(e.state)
                              .map(([k, v]) => `${k}: ${show(v, t)}`)
                              .join(' · ')}
                          </p>
                        )}
                        {e.looks && e.looks.length > 0 && (
                          <p className="text-slate-500">{fmt(t.looks, { looks: e.looks.join(', ') })}</p>
                        )}
                      </li>
                    );
                  })}
                </ul>
              )}
            </div>
          );
        })}
      </div>
      <Unreadables items={story.codex_unreadable} testId="story-codex-unreadable" heading={t.codexUnreadable} />
    </Section>
  );
};

const Decided: React.FC<{ story: StoryOverview }> = ({ story }) => {
  const t = useCopy().dock.story;
  return (
    <Section title={t.decided} testId="story-decided">
      {story.decided.length === 0 ? (
        <p className="text-slate-400">{t.noneDecided}</p>
      ) : (
        <ul className="space-y-2">
          {story.decided.map((p) => (
            <li key={p.id} data-testid="story-decided-item" data-proposal={p.id}>
              <p className="text-slate-200">
                {p.entry_name ?? p.entry_id} · {decision(p, t)} · {formatWhen(p.decided_at)}
              </p>
              <Provenance proposal={p} />
              {p.reason && <p className="text-slate-400">{p.reason}</p>}
            </li>
          ))}
        </ul>
      )}
    </Section>
  );
};
