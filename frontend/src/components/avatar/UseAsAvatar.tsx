import React, { useState } from 'react';
import { ChevronDown, UserRound } from 'lucide-react';
import { fmt, useCopy } from '../../i18n';
import {
  canWearPicture,
  setAvatarFromPath,
  undoAvatarChange,
  undoStillOffered,
  useAvatarAuthor,
  useAvatarChoice,
  type AvatarFailure,
} from '../../lib/avatarChoice';
import { usePopover } from './usePopover';

type Outcome =
  | { kind: 'set'; name: string; previousPath: string | null; changeId: number | null }
  | { kind: 'undone'; name: string }
  | { kind: 'failed'; name: string; failure: AvatarFailure };

/**
 * **Use as avatar ▾** under a picture a clone drew.
 *
 * The button gives the picture to the clone that wrote the message; the chevron lists every
 * clone, for a picture one clone drew for another. It is a direct act -- the picture is set
 * with no model in between -- and it answers on the card: which clone now wears it, with an
 * Undo that puts back what it replaced. A failure is said in the head's own words.
 *
 * Offered only inside a clone's message, and only for a picture in the workspace (`path`)
 * that the runtime would take: never under an SVG. The Undo goes once the listing shows a
 * later change to the same clone's picture, made anywhere, since it would no longer put back
 * what this replaced; one the runtime refuses as too late is said in plain words.
 */
export const UseAsAvatar: React.FC<{ path: string }> = ({ path }) => {
  const choice = useAvatarChoice();
  const author = useAvatarAuthor();
  const copy = useCopy().avatar;
  const menu = usePopover<HTMLSpanElement>();
  const [working, setWorking] = useState(false);
  const [outcome, setOutcome] = useState<Outcome | null>(null);

  if (choice === null || author === null || !canWearPicture(path)) return null;
  const labelOf = (name: string) => choice.clones.find((c) => c.name === name)?.label ?? name;

  const give = async (name: string) => {
    menu.setOpen(false);
    setWorking(true);
    setOutcome(null);
    const result = await setAvatarFromPath(name, path);
    setWorking(false);
    if (result.ok) {
      setOutcome({ kind: 'set', name, previousPath: result.previousPath, changeId: result.changeId });
      choice.onChanged();
    } else {
      setOutcome({ kind: 'failed', name, failure: result.failure });
    }
  };

  const undo = async (name: string, previousPath: string | null, changeId: number | null) => {
    if (changeId === null) return;
    setWorking(true);
    const result = await undoAvatarChange(name, previousPath, changeId);
    setWorking(false);
    if (result.ok) {
      setOutcome({ kind: 'undone', name });
      choice.onChanged();
    } else {
      const { failure } = result;
      setOutcome({ kind: 'failed', name, failure });
      // Changed elsewhere: read the pictures again, so the one shown is the one in force.
      if (failure === 'staleChange') choice.onChanged();
    }
  };

  return (
    <span
      data-testid="use-as-avatar"
      className="flex flex-wrap items-center justify-end gap-x-3 gap-y-1 border-t border-slate-800 px-3 py-1.5 text-[11px] text-slate-400"
    >
      {working && (
        <span role="status" className="text-slate-500">
          {copy.card.working}
        </span>
      )}
      {!working && outcome?.kind === 'set' && (
        <span role="status" data-testid="use-as-avatar-done" className="inline-flex items-center gap-1.5 text-slate-400">
          {fmt(copy.card.set, { name: labelOf(outcome.name) })}
          {undoStillOffered(outcome.changeId, outcome.name, choice.latestChanges) && (
            <button
              type="button"
              data-testid="use-as-avatar-undo"
              onClick={() => void undo(outcome.name, outcome.previousPath, outcome.changeId)}
              className="text-slate-300 underline hover:text-slate-100"
            >
              {copy.card.undo}
            </button>
          )}
        </span>
      )}
      {!working && outcome?.kind === 'undone' && (
        <span role="status" data-testid="use-as-avatar-undone" className="text-slate-400">
          {fmt(copy.card.undone, { name: labelOf(outcome.name) })}
        </span>
      )}
      {!working && outcome?.kind === 'failed' && (
        <span role="alert" data-testid="use-as-avatar-failed" className="text-rose-300">
          {fmt(copy.failure[outcome.failure], { name: labelOf(outcome.name) })}
        </span>
      )}
      <span ref={menu.ref} className="relative inline-flex shrink-0 items-center">
        <button
          type="button"
          data-testid="use-as-avatar-btn"
          title={fmt(copy.card.useFor, { name: labelOf(author) })}
          disabled={working}
          onClick={() => void give(author)}
          className="inline-flex items-center gap-1 text-slate-300 hover:text-slate-100 disabled:opacity-50"
        >
          <UserRound size={11} />
          {copy.card.use}
        </button>
        <button
          type="button"
          data-testid="use-as-avatar-more"
          aria-label={copy.card.chooseOther}
          aria-haspopup="menu"
          aria-expanded={menu.open}
          disabled={working}
          onClick={() => menu.setOpen(!menu.open)}
          className="ml-0.5 inline-flex items-center rounded px-0.5 text-slate-400 hover:text-slate-100 disabled:opacity-50"
        >
          <ChevronDown size={11} />
        </button>
        {menu.open && (
          <span
            role="menu"
            aria-label={copy.card.listLabel}
            data-testid="use-as-avatar-list"
            className="absolute bottom-full right-0 z-20 mb-1 flex max-h-60 min-w-40 flex-col overflow-y-auto rounded-lg border border-slate-700 bg-slate-900 py-1 shadow-lg"
          >
            {choice.clones.map((clone) => (
              <button
                key={clone.name}
                type="button"
                role="menuitem"
                data-testid={`use-as-avatar-for-${clone.name}`}
                onClick={() => void give(clone.name)}
                className="px-3 py-1 text-left text-[11px] text-slate-200 hover:bg-slate-800"
              >
                {clone.label}
              </button>
            ))}
          </span>
        )}
      </span>
    </span>
  );
};
