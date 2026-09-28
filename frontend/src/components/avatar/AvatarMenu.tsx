import React, { useRef, useState } from 'react';
import { Bot, ImageUp, RotateCcw, Sparkles } from 'lucide-react';
import { Avatar } from '../../ui-kit';
import { fmt, useCopy } from '../../i18n';
import {
  UPLOAD_ACCEPT,
  noteAvatarChange,
  resetAvatar,
  undoAvatarChange,
  uploadAvatar,
  uploadProblem,
  useAvatarChoice,
  useLatestAvatarChange,
  type AvatarFailure,
} from '../../lib/avatarChoice';
import { usePopover } from './usePopover';

type Outcome =
  | { kind: 'uploaded' | 'reset'; previousPath: string | null; change: number }
  | { kind: 'undone' }
  | { kind: 'failed'; failure: AvatarFailure }
  | { kind: 'badFile'; problem: 'wrongType' | 'tooLarge' };

/**
 * A clone's picture on its profile, as the way to change it.
 *
 * Three things, each a plain act: upload a picture from this computer, ask the clone to draw
 * some (a request put in the composer, not sent -- generation stays in the conversation), or
 * go back to the default. An upload or reset answers under the picture with an Undo, which
 * goes once a later change to this clone's picture is made anywhere in the head.
 *
 * Without the app's avatar context, or for a clone with no installed definition to keep a
 * picture beside, it is the plain picture it was.
 */
export const AvatarMenu: React.FC<{
  name: string;
  imageSrc: string | undefined;
  /** Whether the clone has an installed definition, which is where a picture is kept. */
  installed: boolean;
  /**
   * Whether a picture was chosen for the clone here. A shipped picture, or none, leaves
   * nothing to reset.
   */
  hasChosenPicture: boolean;
}> = ({ name, imageSrc, installed, hasChosenPicture }) => {
  const choice = useAvatarChoice();
  const copy = useCopy().avatar;
  const menu = usePopover<HTMLDivElement>();
  const fileRef = useRef<HTMLInputElement | null>(null);
  const [working, setWorking] = useState(false);
  const [outcome, setOutcome] = useState<Outcome | null>(null);
  const latest = useLatestAvatarChange(name);

  const picture = (onClick?: () => void) => (
    <Avatar
      label={name}
      kind="agent"
      agentIcon={Bot}
      size="lg"
      imageSrc={imageSrc}
      data-testid="clone-profile-avatar"
      onClick={onClick}
      interactiveLabel={onClick ? fmt(copy.menu.open, { name }) : undefined}
    />
  );

  if (choice === null || !installed) return picture();

  const run = async (kind: 'uploaded' | 'reset', act: () => ReturnType<typeof resetAvatar>) => {
    setWorking(true);
    setOutcome(null);
    const result = await act();
    setWorking(false);
    if (result.ok) {
      setOutcome({ kind, previousPath: result.previousPath, change: noteAvatarChange(name) });
      choice.onChanged();
    } else {
      setOutcome({ kind: 'failed', failure: result.failure });
    }
  };

  const onFile = (event: React.ChangeEvent<HTMLInputElement>) => {
    const file = event.target.files?.[0];
    // Cleared so choosing the same file again after a refusal still reports a change.
    event.target.value = '';
    if (!file) return;
    const problem = uploadProblem(file);
    if (problem !== null) {
      setOutcome({ kind: 'badFile', problem });
      return;
    }
    void run('uploaded', () => uploadAvatar(name, file));
  };

  const undo = async (previousPath: string | null) => {
    setWorking(true);
    const result = await undoAvatarChange(name, previousPath);
    setWorking(false);
    if (result.ok) {
      noteAvatarChange(name);
      setOutcome({ kind: 'undone' });
      choice.onChanged();
    } else {
      setOutcome({ kind: 'failed', failure: result.failure });
    }
  };

  const item =
    'flex w-full items-center gap-2 px-3 py-1.5 text-left text-xs text-slate-200 hover:bg-slate-800 disabled:opacity-50 disabled:hover:bg-transparent';

  return (
    <div className="flex flex-col items-center gap-1.5" data-testid="clone-avatar-menu">
      <div ref={menu.ref} className="relative">
        {picture(() => menu.setOpen(!menu.open))}
        {menu.open && (
          <div
            role="menu"
            aria-label={fmt(copy.menu.open, { name })}
            data-testid="clone-avatar-menu-list"
            className="absolute left-1/2 top-full z-20 mt-1 w-56 -translate-x-1/2 rounded-lg border border-slate-700 bg-slate-900 py-1 shadow-lg"
          >
            <button
              type="button"
              role="menuitem"
              data-testid="clone-avatar-upload"
              disabled={working}
              className={item}
              onClick={() => {
                menu.setOpen(false);
                fileRef.current?.click();
              }}
            >
              <ImageUp className="h-3.5 w-3.5 text-slate-400" />
              {copy.menu.upload}
            </button>
            <button
              type="button"
              role="menuitem"
              data-testid="clone-avatar-ask"
              className={item}
              onClick={() => {
                menu.setOpen(false);
                choice.askFor(name);
              }}
            >
              <Sparkles className="h-3.5 w-3.5 text-slate-400" />
              {fmt(copy.menu.ask, { name })}
            </button>
            <button
              type="button"
              role="menuitem"
              data-testid="clone-avatar-reset"
              disabled={working || !hasChosenPicture}
              className={item}
              onClick={() => {
                menu.setOpen(false);
                void run('reset', () => resetAvatar(name));
              }}
            >
              <RotateCcw className="h-3.5 w-3.5 text-slate-400" />
              {copy.menu.reset}
            </button>
          </div>
        )}
      </div>
      <input
        ref={fileRef}
        type="file"
        accept={UPLOAD_ACCEPT}
        className="hidden"
        data-testid="clone-avatar-file"
        onChange={onFile}
      />
      {working && (
        <p role="status" className="text-[11px] text-slate-500">
          {copy.card.working}
        </p>
      )}
      {!working && (outcome?.kind === 'uploaded' || outcome?.kind === 'reset') && (
        <p role="status" data-testid="clone-avatar-done" className="text-[11px] text-slate-400">
          {fmt(outcome.kind === 'uploaded' ? copy.menu.uploaded : copy.menu.resetDone, { name })}{' '}
          {latest === outcome.change && (
            <button
              type="button"
              data-testid="clone-avatar-undo"
              onClick={() => void undo(outcome.previousPath)}
              className="text-slate-300 underline hover:text-slate-100"
            >
              {copy.card.undo}
            </button>
          )}
        </p>
      )}
      {!working && outcome?.kind === 'undone' && (
        <p role="status" data-testid="clone-avatar-undone" className="text-[11px] text-slate-400">
          {fmt(copy.card.undone, { name })}
        </p>
      )}
      {!working && outcome?.kind === 'failed' && (
        <p role="alert" data-testid="clone-avatar-failed" className="text-[11px] text-rose-300">
          {fmt(copy.failure[outcome.failure], { name })}
        </p>
      )}
      {!working && outcome?.kind === 'badFile' && (
        <p role="alert" data-testid="clone-avatar-failed" className="text-[11px] text-rose-300">
          {copy.upload[outcome.problem]}
        </p>
      )}
    </div>
  );
};
