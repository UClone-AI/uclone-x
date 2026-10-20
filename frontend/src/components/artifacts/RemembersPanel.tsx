import React, { useEffect, useMemo, useState } from 'react';
import { BookOpen, Lightbulb, MoreHorizontal } from 'lucide-react';
import {
  editMemoryFact,
  markRelationSeveral,
  notifyMemoryChanged,
  onMemoryChanged,
  restoreMemoryFact,
  setRelationFunctional,
  unmergeSubject,
  roomDockUrls,
  useRoomRead,
  type KnownFact,
  type SeatKnowledge,
  type WorkedOut,
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

/**
 * What a worked-out statement rests on, as the facts' own sentences (§3.6). An id the list
 * does not hold is left out rather than shown: an id is not a sentence.
 */
export const workedOutFromSentence = (
  because: readonly string[],
  facts: readonly KnownFact[],
  copy: Copy = en.dock.remembers,
): string | null => {
  const said = because
    .map((id) => facts.find((f) => f.fact_id === id)?.statement)
    .filter((statement): statement is string => Boolean(statement));
  return said.length > 0 ? fmt(copy.workedOutFrom, { facts: said.join(copy.workedOutJoin) }) : null;
};

type Editing = { kind: 'menu' } | { kind: 'correct'; value: string } | { kind: 'forget' };

interface FactRowProps {
  fact: KnownFact;
  seatId: string;
  name: string;
  t: Copy;
  onChanged: () => void;
  /** This fact's value is the one said last of several under its relation (#2328). */
  saidLast?: boolean;
  /** A value could not be brought back because it changed meanwhile (#2344). */
  onStale: (value: string | null) => void;
}

/** One fact: its sentence, how it was learned, and the `⋯` that corrects or forgets it. */
const FactRow: React.FC<FactRowProps> = ({
  fact,
  seatId,
  name,
  t,
  onChanged,
  saidLast,
  onStale,
}) => {
  const [editing, setEditing] = useState<Editing | null>(null);
  const [working, setWorking] = useState(false);
  const [refusal, setRefusal] = useState<string | null>(null);
  const [showHistory, setShowHistory] = useState(false);

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
      onStale(null);
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
        {saidLast && (
          <span data-testid="fact-said-last" className="shrink-0 text-[10px] text-cyan-300">
            {t.saidLast}
          </span>
        )}
        <span data-testid="fact-origin" className="shrink-0 text-[10px] text-slate-400">
          {t.origin[fact.origin] ?? fact.origin}
        </span>
        <button
          type="button"
          data-testid="fact-history-toggle"
          aria-expanded={showHistory}
          onClick={() => setShowHistory((open) => !open)}
          className="shrink-0 text-[10px] text-slate-400 underline hover:text-slate-200"
        >
          {showHistory ? t.historyHide : t.history}
        </button>
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

      {fact.merged_from && (
        <div
          data-testid="fact-merged-from"
          className="flex flex-wrap items-center gap-2 text-[11px] text-slate-400"
        >
          <span className="flex-1 min-w-0">{fmt(t.mergedFrom, { name: fact.merged_from })}</span>
          <button
            type="button"
            data-testid="fact-unmerge"
            disabled={working}
            onClick={async () => {
              const source = fact.merged_from;
              if (!source) return;
              setWorking(true);
              setRefusal(null);
              const outcome = await unmergeSubject(seatId, source);
              setWorking(false);
              if (outcome.ok) {
                onStale(null);
                onChanged();
              } else {
                setRefusal(editRefusedSentence(outcome.status, name, t));
              }
            }}
            className="px-2 py-0.5 rounded-lg bg-slate-800 text-slate-200 hover:bg-slate-700 disabled:opacity-50"
          >
            {working ? t.working : t.keepSeparate}
          </button>
        </div>
      )}

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
          <OneValueToggle fact={fact} seatId={seatId} t={t} testId="fact-one-value" />
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

      {showHistory && (
        <FactHistory fact={fact} seatId={seatId} name={name} t={t} onStale={onStale} />
      )}

      {refusal && (
        <p data-testid="fact-edit-refusal" role="alert" className="text-[11px] text-amber-300">
          {refusal}
        </p>
      )}
    </li>
  );
};

/** The day of an ISO time, as the reader's locale writes it; empty when there is none. */
const day = (iso: string | null): string => (iso ? new Date(iso).toLocaleDateString() : '');

interface FactHistoryProps {
  fact: KnownFact;
  seatId: string;
  name: string;
  t: Copy;
  onStale: (value: string | null) => void;
}

/**
 * Where a fact came from and what it replaced (#2332): the words it was learned from, when
 * it was learned and last heard, and each earlier value it replaced, which a person can
 * bring back. Bringing one back saves it as a correction, so the clone learns over it no more.
 */
const FactHistory: React.FC<FactHistoryProps> = ({ fact, seatId, name, t, onStale }) => {
  const [working, setWorking] = useState<string | null>(null);
  const [refused, setRefused] = useState(false);
  return (
    <div data-testid="fact-history" className="pl-1 space-y-1 text-[11px] text-slate-400">
      {fact.quote && <p data-testid="fact-history-quote">{fmt(t.historyQuote, { quote: fact.quote })}</p>}
      <p>
        {fmt(t.historyLearned, { when: day(fact.created_at) })}
        {fact.last_seen_at && fact.last_seen_at !== fact.created_at && (
          <> · {fmt(t.historyLastSeen, { when: day(fact.last_seen_at) })}</>
        )}
      </p>
      {fact.replaced.length === 0 ? (
        <p data-testid="fact-history-none">{t.historyNothingReplaced}</p>
      ) : (
        <ul className="space-y-1">
          {fact.replaced.map((old) => (
            <li key={old.fact_id} data-testid="fact-history-replaced" className="flex items-center gap-2">
              <span className="flex-1 min-w-0">
                {fmt(t.historyReplaced, { value: old.value, when: day(old.said_at) })}
              </span>
              <button
                type="button"
                data-testid="fact-restore"
                disabled={working !== null}
                onClick={async () => {
                  setWorking(old.fact_id);
                  setRefused(false);
                  const outcome = await restoreMemoryFact(seatId, old.fact_id);
                  setWorking(null);
                  // On success every open Remembers reads again (`restoreMemoryFact`).
                  if (outcome.ok) onStale(null);
                  if (!outcome.ok) setRefused(true);
                  // 409: what replaced it changed meanwhile, so this list is stale. Read it
                  // again, so no row or Bring back that no longer holds stays (#2338).
                  if (!outcome.ok && outcome.status === 409) {
                    // The re-read can take this row away with its notice, so the panel keeps
                    // one of its own (#2344).
                    onStale(old.value);
                    notifyMemoryChanged();
                  }
                }}
                className="px-2 py-0.5 rounded-lg bg-slate-800 text-slate-200 hover:bg-slate-700 disabled:opacity-50"
              >
                {working === old.fact_id ? t.working : t.restore}
              </button>
            </li>
          ))}
        </ul>
      )}
      {refused && (
        <p role="alert" data-testid="fact-restore-refusal" className="text-amber-300">
          {fmt(t.restoreFailed, { name })}
        </p>
      )}
    </div>
  );
};

/** A relation as a person reads it: `lives_in` is "lives in". */
const spoken = (relation: string): string => relation.replace(/_/g, ' ').trim();

interface OneValueToggleProps {
  fact: KnownFact;
  seatId: string;
  t: Copy;
  testId: string;
}

/**
 * Whether the fact's relation holds one value at a time, as a checkbox (#2332). Saved in the
 * clone's rules; on success every open Remembers reads again, which is what moves the box.
 */
const OneValueToggle: React.FC<OneValueToggleProps> = ({ fact, seatId, t, testId }) => {
  const [working, setWorking] = useState(false);
  const [failed, setFailed] = useState(false);
  return (
    <label className="inline-flex items-center gap-1.5 text-[11px] text-slate-300">
      <input
        type="checkbox"
        data-testid={testId}
        checked={fact.functional}
        disabled={working}
        onChange={async (e) => {
          setWorking(true);
          setFailed(false);
          const outcome = await setRelationFunctional(seatId, fact.relation, e.target.checked);
          setWorking(false);
          setFailed(!outcome.ok);
        }}
      />
      {t.oneValue}
      {failed && (
        <span role="alert" data-testid="one-value-refusal" className="text-amber-300">
          {t.oneValueFailed}
        </span>
      )}
    </label>
  );
};

interface RelationGroupProps {
  facts: KnownFact[];
  seatId: string;
  name: string;
  t: Copy;
  onChanged: () => void;
  onStale: (value: string | null) => void;
}

/**
 * The current values of one subject's relation, together (#2328, #2332). The value said last
 * is marked, as recall marks it. The header holds the one-value-at-a-time box, and, when a
 * person's words replaced a value of it and nobody has answered yet, the question; ignoring
 * it saves nothing.
 */
const RelationGroup: React.FC<RelationGroupProps> = ({
  facts,
  seatId,
  name,
  t,
  onChanged,
  onStale,
}) => {
  const [first] = facts;
  // The relation's answers are the same on each of its facts; any one asking is the group's.
  const asks = facts.some((f) => f.ask_functional);
  const [asking, setAsking] = useState(false);
  const [failed, setFailed] = useState(false);
  const latest = facts.reduce((a, b) =>
    (b.last_seen_at ?? '') > (a.last_seen_at ?? '') ? b : a,
  );
  return (
    <li data-testid="relation-group" className="space-y-1.5">
      <div className="flex flex-wrap items-center gap-2 px-1">
        <span className="text-[11px] text-slate-400">
          {first.subject} · {spoken(first.relation)}
        </span>
        <OneValueToggle fact={first} seatId={seatId} t={t} testId="relation-one-value" />
      </div>
      {asks && (
        <div
          data-testid="relation-ask"
          className="flex flex-wrap items-center gap-2 px-3 py-2 rounded-xl bg-cyan-950/40 border border-cyan-900 text-[11px] text-slate-200"
        >
          <span className="flex-1 min-w-0">
            {fmt(t.askOneValue, { relation: spoken(first.relation) })}
          </span>
          <button
            type="button"
            data-testid="relation-ask-yes"
            disabled={asking}
            onClick={async () => {
              setAsking(true);
              setFailed(false);
              const outcome = await setRelationFunctional(seatId, first.relation, true);
              setAsking(false);
              setFailed(!outcome.ok);
            }}
            className="px-2 py-1 rounded-lg bg-cyan-700 text-white disabled:opacity-50"
          >
            {asking ? t.working : t.askYes}
          </button>
          <button
            type="button"
            data-testid="relation-ask-no"
            disabled={asking}
            onClick={async () => {
              setAsking(true);
              setFailed(false);
              const outcome = await markRelationSeveral(seatId, first.relation);
              setAsking(false);
              setFailed(!outcome.ok);
            }}
            className="px-2 py-1 rounded-lg bg-slate-800 text-slate-200 disabled:opacity-50"
          >
            {t.askNo}
          </button>
          {failed && (
            <span role="alert" data-testid="relation-ask-refusal" className="text-amber-300">
              {t.oneValueFailed}
            </span>
          )}
        </div>
      )}
      <ul className="space-y-1.5">
        {facts.map((f) => (
          <FactRow
            key={f.fact_id}
            fact={f}
            seatId={seatId}
            name={name}
            t={t}
            onChanged={onChanged}
            onStale={onStale}
            saidLast={facts.length > 1 && f === latest}
          />
        ))}
      </ul>
    </li>
  );
};

/** `list` in groups of one subject's relation, each group where its first fact was. */
const byRelation = (list: KnownFact[]): KnownFact[][] => {
  const groups = new Map<string, KnownFact[]>();
  for (const f of list) {
    const key = `${f.subject.toLowerCase()}\u0000${f.relation}`;
    const held = groups.get(key);
    if (held) held.push(f);
    else groups.set(key, [f]);
  }
  return [...groups.values()];
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
 * `detail` or a fixed sentence, never transport text. The graph fields describe the
 * clone's rules engine, for the developer graph, and are not shown here.
 *
 * What the clone's rules work out from those facts (`worked_out`, step 6) is a third group:
 * each statement with the facts it rests on, and no `⋯` -- a worked-out line is corrected or
 * forgotten through the facts it names (§3.6). It is computed on every read, so forgetting a
 * fact takes what followed from it away on the next read. With none, or when it could not be
 * worked out (`null`), the group is not drawn and nothing is said beyond `facts_reason`: an
 * empty list is not "nothing follows" (#1870).
 */
export const RemembersPanel: React.FC<RemembersPanelProps> = ({
  roomId,
  seatId,
  seatName,
  refreshKey,
}) => {
  const [edits, setEdits] = useState(0);
  // A value that could not be brought back, said at the panel's level so a re-read that
  // removes its row does not take the notice with it (#2344).
  const [stale, setStale] = useState<string | null>(null);
  useEffect(() => {
    setStale(null);
  }, [roomId, seatId]);
  useEffect(() => {
    return onMemoryChanged(() => {
      setEdits((n) => n + 1);
    });
  }, []);
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
  const workedOut: WorkedOut[] = (facts && data.worked_out) ?? [];

  const group = (testid: string, heading: string, list: KnownFact[]) =>
    list.length > 0 && (
      <section data-testid={testid} className="space-y-1.5">
        <h3 className="text-[11px] font-semibold text-slate-300">{heading}</h3>
        <ul className="space-y-1.5">
          {byRelation(list).map((facts) =>
            facts.length > 1 || facts[0].ask_functional || facts[0].functional ? (
              <RelationGroup
                key={facts[0].fact_id}
                facts={facts}
                seatId={seatId}
                name={name}
                t={t}
                onChanged={onChanged}
                onStale={setStale}
              />
            ) : (
              <FactRow
                key={facts[0].fact_id}
                fact={facts[0]}
                seatId={seatId}
                name={name}
                t={t}
                onChanged={onChanged}
                onStale={setStale}
              />
            ),
          )}
        </ul>
      </section>
    );

  return (
    <div data-testid="remembers-panel" className="flex flex-col h-full gap-3 min-h-0">
      {header}
      <div className="flex-1 overflow-y-auto min-h-0 space-y-4 pr-1">
        {stale !== null && (
          <p data-testid="remembers-notice" role="status" className="text-[11px] text-amber-300">
            {fmt(t.restoreStale, { value: stale })}
          </p>
        )}
        {group('remembers-here', t.hereHeading, here)}
        {group('remembers-elsewhere', t.elsewhereHeading, elsewhere)}
        {workedOut.length > 0 && (
          <section data-testid="remembers-worked-out" className="space-y-1.5">
            <h3 className="text-[11px] font-semibold text-slate-300">{t.workedOutHeading}</h3>
            <ul className="space-y-1.5">
              {workedOut.map((w) => {
                const from = workedOutFromSentence(w.because, facts ?? [], t);
                return (
                  <li
                    key={w.statement}
                    data-testid="worked-out-line"
                    className="px-3 py-2 rounded-xl bg-slate-900/40 border border-dashed border-slate-700 text-xs text-slate-200 space-y-1"
                  >
                    <div className="flex items-start gap-2">
                      <Lightbulb
                        aria-hidden="true"
                        className="w-3.5 h-3.5 shrink-0 text-amber-300"
                      />
                      <span className="flex-1 min-w-0 break-words">{w.statement}</span>
                      <span className="shrink-0 text-[10px] text-slate-400">{t.workedOut}</span>
                    </div>
                    {from && (
                      <p data-testid="worked-out-from" className="pl-5 text-[11px] text-slate-400">
                        {from}
                      </p>
                    )}
                  </li>
                );
              })}
            </ul>
          </section>
        )}
        {note && (
          <p data-testid="remembers-reason" className="text-[11px] text-slate-400">
            {note}
          </p>
        )}
      </div>
    </div>
  );
};
