import React from 'react';
import { BookOpen, Lightbulb } from 'lucide-react';
import {
  roomDockUrls,
  useRoomRead,
  type SeatKnowledge,
} from '../../lib/roomDock';

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
export const readFailedSentence = (name: string): string =>
  `What ${name} remembers could not be read.`;

/**
 * What a clone remembers, as plain sentences (#1357, #1401).
 *
 * Two groups, each in the Core's own words:
 *
 * - **Saved to memory** (`saved_facts`): the facts the clone saved to its own memory. They
 *   belong to the clone, so they are listed whichever conversation is on screen, and never
 *   another clone's.
 * - **Known in this conversation** (`remembers`): what the clone holds in this
 *   conversation's knowledge record.
 *
 * Where a group has nothing to list, the panel shows the Core's sentence or its own "none
 * listed" (P6): an empty list is "none listed", never "nothing learned".
 * A failed read shows the Core's own plain `detail` or a fixed sentence, never transport
 * text. The graph those sentences come from stays a developer surface; nothing here names
 * its parts.
 */
export const RemembersPanel: React.FC<RemembersPanelProps> = ({
  roomId,
  seatId,
  seatName,
  refreshKey,
}) => {
  const url = roomId && seatId ? roomDockUrls.knowledge(roomId, seatId) : null;
  const { data, fault } = useRoomRead<SeatKnowledge>(url, refreshKey);
  const name = seatName || seatId || 'This clone';

  const loadingSentence = `Reading what ${name} remembers…`;
  const reason = !roomId
    ? 'No conversation is open. Open one from the rail to see what its clones remember.'
    : !seatId
      ? 'No clone is seated in this conversation yet, so there is nobody to ask.'
      : fault
        ? // The Core's own plain words where it gave them; otherwise a sentence of ours.
          // Never the transport's ("Failed to fetch", a status line).
          (fault.detail ?? readFailedSentence(name))
        : !data
          ? loadingSentence
          : null;

  const header = (
    <div className="p-3 bg-slate-900/80 border border-slate-800 rounded-2xl flex items-center gap-2.5">
      <div className="p-2 bg-slate-800 rounded-xl border border-slate-700 shrink-0">
        <BookOpen className="w-4 h-4 text-cyan-300" />
      </div>
      <div className="min-w-0">
        <h2 className="text-xs font-bold text-white">What {seatId ? name : 'the clone'} remembers</h2>
      </div>
    </div>
  );

  if (reason || !data) {
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

  const saved = data.saved_facts ?? null;
  const savedNote =
    data.saved_facts_reason ?? (saved === null ? `${name}'s saved facts are not listed.` : null);
  const remembers = data.remembers ?? [];
  // The Core's reason; without one, only that the list is empty -- a list the head was not
  // told the cause of is not "remembers nothing".
  const knownNote =
    remembers.length === 0
      ? (data.reason ?? `No remembered statements are listed for ${name}.`)
      : data.reason;

  return (
    <div data-testid="remembers-panel" className="flex flex-col h-full gap-3 min-h-0">
      {header}
      <div className="flex-1 overflow-y-auto min-h-0 space-y-4 pr-1">
        <section data-testid="remembers-saved" className="space-y-1.5">
          <h3 className="text-[11px] font-semibold text-slate-300">Saved to memory</h3>
          {saved && saved.length > 0 && (
            <ul className="space-y-1.5">
              {saved.map((f, i) => (
                <li
                  key={`${f.statement}:${i}`}
                  data-testid="saved-fact"
                  className="px-3 py-2 rounded-xl bg-slate-900/60 border border-slate-800 text-xs text-slate-200 flex items-start gap-2"
                >
                  <span className="flex-1 min-w-0 break-words">{f.statement}</span>
                  {f.saved_here && (
                    <span className="shrink-0 text-[10px] text-slate-400">saved in this conversation</span>
                  )}
                </li>
              ))}
            </ul>
          )}
          {savedNote && (
            <p data-testid="saved-facts-reason" className="text-[11px] text-slate-400">
              {savedNote}
            </p>
          )}
        </section>

        <section data-testid="remembers-known" className="space-y-1.5">
          <h3 className="text-[11px] font-semibold text-slate-300">Known in this conversation</h3>
          {remembers.length > 0 && (
            <ul className="space-y-1.5">
              {remembers.map((r, i) => (
                <li
                  key={`${r.statement}:${i}`}
                  data-testid="remembered-statement"
                  className="px-3 py-2 rounded-xl bg-slate-900/60 border border-slate-800 text-xs text-slate-200 flex items-start gap-2"
                >
                  <span className="flex-1 min-w-0 break-words">{r.statement}</span>
                  {r.learned && (
                    <span
                      className="shrink-0 inline-flex items-center gap-1 text-[10px] text-slate-400"
                      title="Worked out from other things it knew, rather than told"
                    >
                      <Lightbulb className="w-3 h-3" />
                      worked out
                    </span>
                  )}
                </li>
              ))}
            </ul>
          )}
          {knownNote && (
            <p data-testid="remembers-reason" className="text-[11px] text-slate-400">
              {knownNote}
            </p>
          )}
        </section>
      </div>
    </div>
  );
};
