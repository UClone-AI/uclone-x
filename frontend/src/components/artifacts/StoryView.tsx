import React, { useCallback, useEffect, useState } from 'react';
import { ArrowLeft, RefreshCw } from 'lucide-react';
import { Button } from '../ui/Button';
import { plainFailure } from '../../lib/coreFailure';
import {
  artifactLibraryApi,
  type ChangeView,
  type EvidenceView,
  type JsonValue,
  type OutlineView,
  type ProposalView,
  type StoryConversation,
  type StoryOverview,
} from '../../lib/artifactLibrary';

/**
 * A story's view in Files (#1560): the outline, the codex, and the changes waiting for a person.
 *
 * Everything shown is the Core's (`StoryView.show`), and so is every decision: Approve and
 * Reject send the proposal's digest as it was shown, and the Core applies it only if the
 * proposal and its entry are still what this screen drew. A refusal is the Core's sentence.
 * No story or file tool reaches this path. A persona with an unconfined shell (Clone's
 * `bash_run`) can call the same local API (#1589), so "Applied here" is not yet proof that
 * a person pressed the button.
 */

export const STORY_COPY = {
  back: 'Back to the files',
  loading: 'Loading the story…',
  loadFailed: 'The story could not be shown.',
  decideFailed: 'The decision could not be saved.',
  waiting: 'Waiting for your decision',
  noneWaiting: 'No change is waiting for a decision: none was proposed, or every one was decided.',
  outline: 'Outline',
  readingOrder: 'In reading order',
  storyOrder: 'In the order it happens',
  codex: 'Codex',
  noEntries: (label: string) => `No ${label.toLowerCase()} are in the codex yet.`,
  unreadable: 'These files could not be read, so they are not shown:',
  // The codex reports folders it does not read as well as files (#1576, #1595).
  codexUnreadable: 'These files or folders could not be read, so they are not shown:',
  decided: 'Decided',
  noneDecided: 'No change has been decided yet.',
  proposedIn: (c: StoryConversation) =>
    c.exists
      ? `Proposed in “${c.title ?? 'a conversation that could not be read'}”`
      : 'Proposed in a conversation that no longer exists',
  from: 'Because the story says',
  quoteGone: 'This passage is no longer in the scene.',
  quoteUnchecked: 'The scene has no text yet, so this passage cannot be checked.',
  notSet: 'not set',
  fromStart: 'from the start',
  after: (scene: string) => `after “${scene}”`,
  unplaced: 'This scene is not in the outline, so “now” is how the entry starts.',
  fileChange: 'The entry file, now and if approved',
  approve: 'Approve',
  reject: 'Reject',
  rejectReason: 'Why, if you want to say (optional)',
  confirmReject: 'Reject this change',
  keep: 'Keep deciding',
  applied: (name: string) => `The change to ${name} was applied.`,
  rejected: (name: string) => `The change to ${name} was rejected.`,
  decision: (p: ProposalView) =>
    `${p.status === 'applied' ? 'Applied' : 'Rejected'}${
      p.decided_in === 'story_view' ? ' here' : p.decided_in === 'conversation' ? ' in a conversation' : ''
    }`,
  unwritten: 'not written yet',
} as const;

const show = (value: JsonValue): string => {
  if (value === null || value === undefined) return STORY_COPY.notSet;
  if (Array.isArray(value)) return value.length === 0 ? STORY_COPY.notSet : value.map(show).join(', ');
  if (typeof value === 'object') return JSON.stringify(value);
  return String(value);
};

const formatWhen = (iso: string | null): string => (iso ? new Date(iso).toLocaleString() : '');

interface StoryViewProps {
  storyId: string;
  onBack: () => void;
}

export const StoryView: React.FC<StoryViewProps> = ({ storyId, onBack }) => {
  const [story, setStory] = useState<StoryOverview | null>(null);
  const [loading, setLoading] = useState(false);
  const [loadError, setLoadError] = useState<string | null>(null);
  const [notice, setNotice] = useState<string[] | null>(null);
  const [busy, setBusy] = useState(false);

  const load = useCallback(async () => {
    setLoading(true);
    setLoadError(null);
    try {
      setStory(await artifactLibraryApi.showStory(storyId));
    } catch (err) {
      console.error('Story view: the story could not be read', err);
      setLoadError(plainFailure(err, STORY_COPY.loadFailed));
    } finally {
      setLoading(false);
    }
  }, [storyId]);

  useEffect(() => {
    setStory(null);
    setNotice(null);
    void load();
  }, [load]);

  const decide = async (proposal: ProposalView, reject: string | null) => {
    const name = proposal.entry_name ?? proposal.entry_id;
    setBusy(true);
    setNotice(null);
    try {
      const done =
        reject === null
          ? await artifactLibraryApi.approveProposal(storyId, proposal.id, proposal.digest)
          : await artifactLibraryApi.rejectProposal(storyId, proposal.id, proposal.digest, reject);
      setNotice([done.decision === 'applied' ? STORY_COPY.applied(name) : STORY_COPY.rejected(name), ...done.notes]);
    } catch (err) {
      console.error('Story view: a decision failed', err);
      setNotice([plainFailure(err, STORY_COPY.decideFailed)]);
    } finally {
      setBusy(false);
    }
    await load();
  };

  return (
    <div className="flex-1 min-h-0 flex flex-col" data-testid="story-view">
      <div className="flex items-center gap-2 px-4 py-2 border-b border-slate-800">
        <Button variant="ghost" onClick={onBack} data-testid="story-view-back">
          <ArrowLeft className="w-3.5 h-3.5" />
          {STORY_COPY.back}
        </Button>
        <div className="min-w-0 flex-1">
          {story && (
            <p className="text-sm font-semibold text-slate-100 truncate" data-testid="story-view-title">
              {story.title}
            </p>
          )}
        </div>
        <Button variant="bordered" size="icon" onClick={() => void load()} disabled={loading} title="Refresh the story">
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
        </div>
      )}

      <div className="flex-1 min-h-0 overflow-y-auto p-4 space-y-6 text-xs">
        {loading && !story && <p className="text-slate-400">{STORY_COPY.loading}</p>}
        {loadError && (
          <div role="alert" className="text-rose-300">
            <p>{loadError}</p>
            <Button variant="outline" className="mt-2" onClick={() => void load()}>
              Try again
            </Button>
          </div>
        )}
        {story && (
          <>
            {(story.logline || story.genre) && (
              <p className="text-slate-400">{[story.genre, story.logline].filter(Boolean).join(' · ')}</p>
            )}
            <Waiting story={story} busy={busy} onDecide={(p, reason) => void decide(p, reason)} />
            <Outline outline={story.outline} note={story.outline_note} />
            <Codex story={story} />
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
  heading?: string;
}> = ({ items, testId, heading = STORY_COPY.unreadable }) =>
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
  const sceneTitles = new Map(
    (story.outline?.chapters ?? []).flatMap((ch) => ch.scenes.map((sc) => [sc.id, sc.title] as const)),
  );
  return (
    <Section title={STORY_COPY.waiting} testId="story-pending">
      {story.pending.length > 0 && story.decide_note && (
        <p role="note" data-testid="story-decide-note" className="mb-2 text-amber-200">
          {story.decide_note}
        </p>
      )}
      {story.pending.length === 0 && <p className="text-slate-400">{STORY_COPY.noneWaiting}</p>}
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
      <Unreadables items={story.proposals_unreadable} testId="story-proposals-unreadable" />
    </Section>
  );
};

const Provenance: React.FC<{ proposal: ProposalView }> = ({ proposal }) => (
  <p className="text-[11px] text-slate-400">
    {STORY_COPY.proposedIn(proposal.proposed_in)} · {formatWhen(proposal.proposed_at)}
  </p>
);

const Evidence: React.FC<{ evidence: EvidenceView[] }> = ({ evidence }) =>
  evidence.length === 0 ? null : (
    <div className="mt-2">
      <p className="text-[11px] text-slate-400">{STORY_COPY.from}</p>
      {evidence.map((e) => (
        <figure key={`${e.scene_id}:${e.quote}`} data-testid="story-evidence" className="mt-1">
          <blockquote className="border-l-2 border-slate-600 pl-2 text-slate-200 italic">{e.quote}</blockquote>
          <figcaption className="text-[11px] text-slate-500">
            {e.scene_title ?? e.scene_id}
            {e.still_in_scene === false && <span className="text-amber-300"> — {STORY_COPY.quoteGone}</span>}
            {e.still_in_scene === null && <span> — {STORY_COPY.quoteUnchecked}</span>}
          </figcaption>
        </figure>
      ))}
    </div>
  );

const Changes: React.FC<{ changes: ChangeView[]; sceneTitles: Map<string, string> }> = ({ changes, sceneTitles }) =>
  changes.length === 0 ? null : (
    <table className="mt-2 w-full text-left" data-testid="story-changes">
      <thead className="text-[11px] text-slate-500">
        <tr>
          <th className="font-normal pr-2">What</th>
          <th className="font-normal pr-2">When</th>
          <th className="font-normal pr-2">Now</th>
          <th className="font-normal">If approved</th>
        </tr>
      </thead>
      <tbody>
        {changes.map((c) => (
          <tr key={`${c.what}:${c.at ?? ''}`} data-testid="story-change" className="align-top">
            <td className="pr-2 text-slate-300">{c.what}</td>
            <td className="pr-2 text-slate-400">
              {c.at === null ? STORY_COPY.fromStart : STORY_COPY.after(sceneTitles.get(c.at) ?? c.at)}
              {!c.placed && <p className="text-[11px] text-slate-500">{STORY_COPY.unplaced}</p>}
            </td>
            <td className="pr-2 text-slate-400" data-testid="story-change-before">
              {show(c.before)}
            </td>
            <td className="text-slate-100" data-testid="story-change-after">
              {show(c.after)}
            </td>
          </tr>
        ))}
      </tbody>
    </table>
  );

const Diff: React.FC<{ lines: string[] }> = ({ lines }) =>
  lines.length === 0 ? null : (
    <details className="mt-2">
      <summary className="cursor-pointer text-[11px] text-slate-400">{STORY_COPY.fileChange}</summary>
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

const PendingProposal: React.FC<{
  proposal: ProposalView;
  canDecide: boolean;
  sceneTitles: Map<string, string>;
  onDecide: (proposal: ProposalView, reject: string | null) => void;
}> = ({ proposal, canDecide, sceneTitles, onDecide }) => {
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
            {STORY_COPY.approve}
          </Button>
          <Button variant="outline" data-testid="story-reject" disabled={!canDecide} onClick={() => setRejecting(true)}>
            {STORY_COPY.reject}
          </Button>
        </div>
      )}
      {rejecting && (
        <div className="mt-3 space-y-2">
          <label className="block text-slate-400">
            {STORY_COPY.rejectReason}
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
              {STORY_COPY.confirmReject}
            </Button>
            <Button variant="outline" onClick={() => setRejecting(false)}>
              {STORY_COPY.keep}
            </Button>
          </div>
        </div>
      )}
    </li>
  );
};

const Outline: React.FC<{
  outline: OutlineView | null;
  note: string | null;
}> = ({ outline, note }) => {
  const [order, setOrder] = useState<'reading' | 'story'>('reading');
  if (!outline) {
    return (
      <Section title={STORY_COPY.outline} testId="story-outline">
        <p className="text-slate-400" data-testid="story-outline-note">
          {note}
        </p>
      </Section>
    );
  }
  const scenes = new Map(outline.chapters.flatMap((ch) => ch.scenes.map((s) => [s.id, s] as const)));
  return (
    <Section title={STORY_COPY.outline} testId="story-outline">
      <div className="mb-2 flex gap-1" role="group" aria-label="Scene order">
        {(['reading', 'story'] as const).map((o) => (
          <Button
            key={o}
            variant={order === o ? 'solid' : 'ghost'}
            aria-pressed={order === o}
            data-testid={`story-order-${o}`}
            onClick={() => setOrder(o)}
          >
            {o === 'reading' ? STORY_COPY.readingOrder : STORY_COPY.storyOrder}
          </Button>
        ))}
      </div>
      {order === 'reading' ? (
        <ol className="space-y-2">
          {outline.chapters.map((ch) => (
            <li key={ch.id}>
              <p className="text-slate-200">
                {ch.act ? `${ch.act} · ` : ''}
                {ch.title}
              </p>
              <ol className="ml-3 mt-1 space-y-1">
                {ch.scenes.map((s) => (
                  <SceneLine key={s.id} id={s.id} title={s.title} summary={s.summary} written={s.written} />
                ))}
              </ol>
            </li>
          ))}
        </ol>
      ) : (
        <ol className="space-y-1" data-testid="story-order-list">
          {outline.story_order.map((id) => {
            const s = scenes.get(id);
            return (
              <SceneLine
                key={id}
                id={id}
                title={s?.title ?? id}
                summary={s?.summary ?? ''}
                written={s?.written ?? false}
              />
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

const SceneLine: React.FC<{
  id: string;
  title: string;
  summary: string;
  written: boolean;
}> = ({ id, title, summary, written }) => (
  <li data-testid="story-scene" data-scene={id}>
    <span className="text-slate-100">{title}</span>
    {!written && <span className="text-slate-500"> ({STORY_COPY.unwritten})</span>}
    {summary && <p className="text-slate-400">{summary}</p>}
  </li>
);

const Codex: React.FC<{ story: StoryOverview }> = ({ story }) => (
  <Section title={STORY_COPY.codex} testId="story-codex">
    <div className="space-y-3">
      {story.codex.map((group) => (
        <div key={group.kind} data-testid="story-codex-group" data-kind={group.kind}>
          <p className="text-slate-300 font-medium">{group.label}</p>
          {group.entries.length === 0 ? (
            <p className="text-slate-500">{STORY_COPY.noEntries(group.label)}</p>
          ) : (
            <ul className="mt-1 space-y-1">
              {group.entries.map((e) => (
                <li key={e.id} data-testid="story-entry" data-entry={e.id}>
                  <span className="text-slate-100">{e.name}</span>
                  {e.aliases.length > 0 && <span className="text-slate-500"> ({e.aliases.join(', ')})</span>}
                  {e.profile && <p className="text-slate-400">{e.profile}</p>}
                  {Object.keys(e.state).length > 0 && (
                    <p className="text-slate-400">
                      {Object.entries(e.state)
                        .map(([k, v]) => `${k}: ${show(v)}`)
                        .join(' · ')}
                    </p>
                  )}
                  {e.looks && e.looks.length > 0 && <p className="text-slate-500">looks: {e.looks.join(', ')}</p>}
                </li>
              ))}
            </ul>
          )}
        </div>
      ))}
    </div>
    <Unreadables
      items={story.codex_unreadable}
      testId="story-codex-unreadable"
      heading={STORY_COPY.codexUnreadable}
    />
  </Section>
);

const Decided: React.FC<{ story: StoryOverview }> = ({ story }) => (
  <Section title={STORY_COPY.decided} testId="story-decided">
    {story.decided.length === 0 ? (
      <p className="text-slate-400">{STORY_COPY.noneDecided}</p>
    ) : (
      <ul className="space-y-2">
        {story.decided.map((p) => (
          <li key={p.id} data-testid="story-decided-item" data-proposal={p.id}>
            <p className="text-slate-200">
              {p.entry_name ?? p.entry_id} · {STORY_COPY.decision(p)} · {formatWhen(p.decided_at)}
            </p>
            <Provenance proposal={p} />
            {p.reason && <p className="text-slate-400">{p.reason}</p>}
          </li>
        ))}
      </ul>
    )}
  </Section>
);
