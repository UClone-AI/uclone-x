import React, { useMemo, useState } from 'react';
import { BookOpen, MoreHorizontal } from 'lucide-react';
import {
  editMemoryFact,
  roomDockUrls,
  useRoomRead,
  type KnownFact,
  type SeatKnowledge,
} from '../../lib/roomDock';
import { en, type Messages } from '../../i18n/en';
import { fmt, useCopy } from '../../i18n';

type Copy = Messages['dock']['remembers'];

interface RemembersPanelProps {
  /** The conversation on screen. */
  roomId: string | null;
  /** The clone whose memory is shown. */
  seatId: string | null;
  /** The clone's name, for the sentences around the list. */
  seatName?: string;
  /** Changes when the conversation moves on, to read again. */
  refreshKey?: unknown;
}

/** What the panel says when a read fails and the Core gave no plain reason of its own. */
export const readFailedSentence = (name: string, copy: Copy = en.dock.remembers): string =>
  fmt(copy.readFailed, { name });

/**
 * Why a Correct or Forget did not happen, in the reader's language. The head's own words
 * for each refusal the Core can give; never the transport's, never an id.
 */
export const editRefusedSentence = (
  status: number | null,
  name: string,
  copy: Copy = en.dock.remembers,
): string => {
  if (status === 400) return copy.valueMissing;
  if (status === 404) return fmt(copy.factGone, { name });
  if (status === 409) return fmt(copy.memoryUnreadable, { name });
  return copy.editFailed;
};

type Editing = { kind: 'menu' } | { kind: 'correct'; value: string } | { kind: 'forget' };

interface FactRowProps {
  fact: KnownFact;
  seatId: string;
  name: string;
  t: Copy;
  onChanged: () => void;
}

/** One fact: its sentence, how it was learned, and the `⋯` that corrects or forgets it. */
const FactRow: React.FC<FactRowProps> = ({ fact, seatId, name, t, onChanged }) => {
  const [editing, setEditing] = useState<Editing | null>(null);
  const [working, setWorking] = useState(false);
  const [refusal, setRefusal] = useState<string | null>(null);

  const close = () => {
    setEditing(null);
    setRefusal(null);
  };

  const apply = async (value: string | null) => {
    if (value !== null && value.trim() === '') {
      setRefusal(t.valueMissing);
      return;
    }
    setWorking(true);
    setRefusal(null);
    const outcome = await editMemoryFact(seatId, fact.fact_id, value);
    setWorking(false);
    if (outcome.ok) {
      setEditing(null);
      onChanged();
    } else {
      setRefusal(editRefusedSentence(outcome.status, name, t));
    }
  };

  return (
    <li
      data-testid="known-fact"
      className="px-3 py-2 rounded-xl bg-slate-900/60 border border-slate-800 text-xs text-slate-200 space-y-2"
    >
      <div className="flex items-start gap-2">
        <span className="flex-1 min-w-0 break-words">{fact.statement}</span>
        <span data-testid="fact-origin" className="shrink-0 text-[10px] text-slate-400">
          {t.origin[fact.origin] ?? fact.origin}
        </span>
        <button
          type="button"
          data-testid="fact-actions"
          aria-label={t.actions}
          title={t.actions}
          aria-expanded={editing !== null}
          onClick={() => (editing ? close() : setEditing({ kind: 'menu' }))}
          className="shrink-0 p-0.5 rounded text-slate-400 hover:text-white hover:bg-slate-800"
        >
          <MoreHorizontal className="w-3.5 h-3.5" />
        </button>
      </div>

      {editing?.kind === 'menu' && (
        <div className="flex gap-2">
          <button
            type="button"
            data-testid="fact-correct"
            onClick={() => setEditing({ kind: 'correct', value: fact.object_value })}
            className="px-2 py-1 rounded-lg bg-slate-800 text-[11px] text-slate-200 hover:bg-slate-700"
          >
            {t.correct}
          </button>
          <button
            type="button"
            data-testid="fact-forget"
            onClick={() => setEditing({ kind: 'forget' })}
            className="px-2 py-1 rounded-lg bg-slate-800 text-[11px] text-rose-300 hover:bg-slate-700"
          >
            {t.forget}
          </button>
        </div>
      )}

      {editing?.kind === 'correct' && (
        <form
          className="flex flex-wrap items-center gap-2"
          onSubmit={(e) => {
            e.preventDefault();
            void apply(editing.value);
          }}
        >
          <input
            data-testid="fact-correct-input"
            aria-label={t.correctLabel}
            placeholder={t.correctLabel}
            value={editing.value}
            disabled={working}
            onChange={(e) => setEditing({ kind: 'correct', value: e.target.value })}
            className="flex-1 min-w-0 px-2 py-1 rounded-lg bg-slate-950 border border-slate-700 text-[11px] text-slate-100"
          />
          <button
            type="submit"
            data-testid="fact-correct-save"
            disabled={working}
            className="px-2 py-1 rounded-lg bg-cyan-700 text-[11px] text-white disabled:opacity-50"
          >
            {working ? t.working : t.save}
          </button>
          <button
            type="button"
            onClick={close}
            disabled={working}
            className="px-2 py-1 rounded-lg bg-slate-800 text-[11px] text-slate-300"
          >
            {t.cancel}
          </button>
        </form>
      )}

      {editing?.kind === 'forget' && (
        <div className="space-y-1.5">
          <p data-testid="fact-forget-confirm" className="text-[11px] text-slate-300">
            {fmt(t.forgetConfirm, { name })}
          </p>
          <div className="flex gap-2">
            <button
              type="button"
              data-testid="fact-forget-yes"
              disabled={working}
              onClick={() => void apply(null)}
              className="px-2 py-1 rounded-lg bg-rose-700 text-[11px] text-white disabled:opacity-50"
            >
              {working ? t.working : t.forget}
            </button>
            <button
              type="button"
              onClick={close}
              disabled={working}
              className="px-2 py-1 rounded-lg bg-slate-800 text-[11px] text-slate-300"
            >
              {t.cancel}
            </button>
          </div>
        </div>
      )}

      {refusal && (
        <p data-testid="fact-edit-refusal" role="alert" className="text-[11px] text-amber-300">
          {refusal}
        </p>
      )}
    </li>
  );
};

/**
 * What a clone knows, as plain sentences (#1357, #1401, #1638 step 3).
 *
 * One clone-wide list from the clone's own memory (`facts`), in two groups: what it learned
 * in the conversation on screen, then what it learned elsewhere. Each fact says how it was
 * learned, and its `⋯` corrects it (a `corrected` fact that supersedes it) or forgets it (a
 * retraction; the audit record stays in the Core). After either, the list is read again.
 *
 * With no facts the panel shows the Core's sentence or its own "none listed" (P6): an empty
 * list is "none listed", never "nothing learned". A failed read shows the Core's own plain
 * `detail` or a fixed sentence, never transport text. The per-seat record's `status` and
 * graph fields are for the developer graph and are not shown here.
 */
export const RemembersPanel: React.FC<RemembersPanelProps> = ({
  roomId,
  seatId,
  seatName,
  refreshKey,
}) => {
  const [edits, setEdits] = useState(0);
  // One key for "the conversation moved on" and "a fact was just changed here".
  const readKey = useMemo(() => ({ refreshKey, edits }), [refreshKey, edits]);
  const url = roomId && seatId ? roomDockUrls.knowledge(roomId, seatId) : null;
  const { data, fault } = useRoomRead<SeatKnowledge>(url, readKey);
  const t = useCopy().dock.remembers;
  const name = seatName || seatId || t.thisClone;

  const loadingSentence = fmt(t.loading, { name });
  const reason = !roomId
    ? t.noRoom
    : !seatId
      ? t.noSeat
      : fault
        ? // The Core's own plain words where it gave them; otherwise a sentence of ours.
          // Never the transport's ("Failed to fetch", a status line).
          (fault.detail ?? readFailedSentence(name, t))
        : !data
          ? loadingSentence
          : null;

  const header = (
    <div className="p-3 bg-slate-900/80 border border-slate-800 rounded-2xl flex items-center gap-2.5">
      <div className="p-2 bg-slate-800 rounded-xl border border-slate-700 shrink-0">
        <BookOpen className="w-4 h-4 text-cyan-300" />
      </div>
      <div className="min-w-0">
        <h2 className="text-xs font-bold text-white">
          {fmt(t.title, { name: seatId ? name : t.theClone })}
        </h2>
      </div>
    </div>
  );

  if (reason || !data || !seatId) {
    return (
      <div data-testid="remembers-panel" className="flex flex-col h-full gap-3 min-h-0">
        {header}
        <p
          // A read still out is not a reason; tests that wait for one wait past it.
          data-testid={reason === loadingSentence ? 'remembers-loading' : 'remembers-reason'}
          className="text-xs text-slate-300 text-center max-w-sm mx-auto py-12"
        >
          {reason}
        </p>
      </div>
    );
  }

  const facts = data.facts ?? null;
  // The Core's reason; without one, only that the list is empty or not listed -- a list the
  // head was not told the cause of is not "knows nothing".
  const note =
    data.facts_reason ??
    (facts === null
      ? fmt(t.notListed, { name })
      : facts.length === 0
        ? fmt(t.noneListed, { name })
        : null);
  const here = (facts ?? []).filter((f) => f.learned_here);
  const elsewhere = (facts ?? []).filter((f) => !f.learned_here);
  const onChanged = () => setEdits((n) => n + 1);

  const group = (testid: string, heading: string, list: KnownFact[]) =>
    list.length > 0 && (
      <section data-testid={testid} className="space-y-1.5">
        <h3 className="text-[11px] font-semibold text-slate-300">{heading}</h3>
        <ul className="space-y-1.5">
          {list.map((f) => (
            <FactRow
              key={f.fact_id}
              fact={f}
              seatId={seatId}
              name={name}
              t={t}
              onChanged={onChanged}
            />
          ))}
        </ul>
      </section>
    );

  return (
    <div data-testid="remembers-panel" className="flex flex-col h-full gap-3 min-h-0">
      {header}
      <div className="flex-1 overflow-y-auto min-h-0 space-y-4 pr-1">
        {group('remembers-here', t.hereHeading, here)}
        {group('remembers-elsewhere', t.elsewhereHeading, elsewhere)}
        {note && (
          <p data-testid="remembers-reason" className="text-[11px] text-slate-400">
            {note}
          </p>
        )}
      </div>
    </div>
  );
};
