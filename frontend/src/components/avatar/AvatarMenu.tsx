import React, { useRef, useState } from 'react';
import { Bot, ImageUp, RotateCcw, Sparkles } from 'lucide-react';
import { Avatar } from '../../ui-kit';
import { fmt, useCopy } from '../../i18n';
import {
  UPLOAD_ACCEPT,
  resetAvatar,
  undoAvatarChange,
  undoStillOffered,
  uploadAvatar,
  uploadProblem,
  useAvatarChoice,
  type AvatarFailure,
} from '../../lib/avatarChoice';
import { usePopover } from './usePopover';

type Outcome =
  | { kind: 'uploaded' | 'reset'; previousPath: string | null; changeId: number | null }
  | { kind: 'undone' }
  | { kind: 'failed'; failure: AvatarFailure }
  | { kind: 'badFile'; problem: 'wrongType' | 'tooLarge' };

/**
 * The drawing styles a person can pick before asking, by the names the avatar skill's presets
 * use. `default` is the house style, "pastel close-up", and the request leaves it unnamed: a
 * request with no style is what the skill draws in that style.
 */
const STYLES = ['default', 'watercolor', 'realistic', 'pixelArt', 'flat'] as const;
type AvatarStyle = (typeof STYLES)[number];

/**
 * A clone's picture on its profile, as the way to change it.
 *
 * Three things, each a plain act: upload a picture from this computer, ask the clone to draw
 * some in a style picked here (a request put in the composer, not sent -- generation stays in
 * the conversation, and the person can still rewrite the style there), or
 * go back to the default. An upload or reset answers under the picture with an Undo, which
 * goes once the listing shows a later change to this clone's picture, made anywhere. The
 * runtime refuses an Undo it reaches too late, and the menu then says so in plain words.
 *
 * Without the app's avatar context, or for a clone with no installed definition to keep a
 * picture beside, it is the plain picture it was.
 */
export const AvatarMenu: React.FC<{
  /** The clone's id: what every request names. */
  name: string;
  /** What a person reads as its name; absent means `name`. */
  label?: string;
  imageSrc: string | undefined;
  /** Whether the clone has an installed definition, which is where a picture is kept. */
  installed: boolean;
  /**
   * Whether a picture was chosen for the clone here. A shipped picture, or none, leaves
   * nothing to reset.
   */
  hasChosenPicture: boolean;
}> = ({ name, label, imageSrc, installed, hasChosenPicture }) => {
  const shown = label ?? name;
  const choice = useAvatarChoice();
  const copy = useCopy().avatar;
  const menu = usePopover<HTMLDivElement>();
  const fileRef = useRef<HTMLInputElement | null>(null);
  const [working, setWorking] = useState(false);
  const [outcome, setOutcome] = useState<Outcome | null>(null);
  const [style, setStyle] = useState<AvatarStyle>('default');

  const picture = (onClick?: () => void) => (
    <Avatar
      label={shown}
      kind="agent"
      agentIcon={Bot}
      size="lg"
      imageSrc={imageSrc}
      data-testid="clone-profile-avatar"
      onClick={onClick}
      interactiveLabel={onClick ? fmt(copy.menu.open, { name: shown }) : undefined}
    />
  );

  if (choice === null || !installed) return picture();

  const run = async (kind: 'uploaded' | 'reset', act: () => ReturnType<typeof resetAvatar>) => {
    setWorking(true);
    setOutcome(null);
    const result = await act();
    setWorking(false);
    if (result.ok) {
      setOutcome({ kind, previousPath: result.previousPath, changeId: result.changeId });
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

  const undo = async (previousPath: string | null, changeId: number | null) => {
    if (changeId === null) return;
    setWorking(true);
    const result = await undoAvatarChange(name, previousPath, changeId);
    setWorking(false);
    if (result.ok) {
      setOutcome({ kind: 'undone' });
      choice.onChanged();
    } else {
      const { failure } = result;
      setOutcome({ kind: 'failed', failure });
      // Changed elsewhere: read the pictures again, so the one shown is the one in force.
      if (failure === 'staleChange') choice.onChanged();
    }
  };

  /** The request put in the composer: it names the picked style, and none for the default. */
  const request = () =>
    style === 'default' ? copy.askText : fmt(copy.askTextStyled, { style: copy.style.name[style] });

  const item =
    'flex w-full items-center gap-2 px-3 py-1.5 text-left text-xs text-slate-200 hover:bg-slate-800 disabled:opacity-50 disabled:hover:bg-transparent';

  return (
    <div className="flex flex-col items-center gap-1.5" data-testid="clone-avatar-menu">
      <div ref={menu.ref} className="relative">
        {picture(() => menu.setOpen(!menu.open))}
        {menu.open && (
          <div
            role="menu"
            aria-label={fmt(copy.menu.open, { name: shown })}
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
            <div
              role="group"
              aria-label={copy.style.group}
              data-testid="clone-avatar-styles"
              className="flex flex-wrap gap-1 px-3 pb-1 pt-1.5"
            >
              {STYLES.map((option) => (
                <button
                  key={option}
                  type="button"
                  role="menuitemradio"
                  aria-checked={style === option}
                  data-testid={`clone-avatar-style-${option}`}
                  onClick={() => setStyle(option)}
                  className={`rounded-full border px-2 py-0.5 text-[11px] ${
                    style === option
                      ? 'border-sky-500 bg-sky-500/15 text-sky-200'
                      : 'border-slate-700 text-slate-300 hover:bg-slate-800'
                  }`}
                >
                  {copy.style.chip[option]}
                </button>
              ))}
            </div>
            <button
              type="button"
              role="menuitem"
              data-testid="clone-avatar-ask"
              className={item}
              onClick={() => {
                menu.setOpen(false);
                choice.askFor(name, request());
              }}
            >
              <Sparkles className="h-3.5 w-3.5 text-slate-400" />
              {fmt(copy.menu.ask, { name: shown })}
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
          {fmt(outcome.kind === 'uploaded' ? copy.menu.uploaded : copy.menu.resetDone, { name: shown })}{' '}
          {undoStillOffered(outcome.changeId, name, choice.latestChanges) && (
            <button
              type="button"
              data-testid="clone-avatar-undo"
              onClick={() => void undo(outcome.previousPath, outcome.changeId)}
              className="text-slate-300 underline hover:text-slate-100"
            >
              {copy.card.undo}
            </button>
          )}
        </p>
      )}
      {!working && outcome?.kind === 'undone' && (
        <p role="status" data-testid="clone-avatar-undone" className="text-[11px] text-slate-400">
          {fmt(copy.card.undone, { name: shown })}
        </p>
      )}
      {!working && outcome?.kind === 'failed' && (
        <p role="alert" data-testid="clone-avatar-failed" className="text-[11px] text-rose-300">
          {fmt(copy.failure[outcome.failure], { name: shown })}
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
