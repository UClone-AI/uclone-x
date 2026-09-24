import React, { useEffect, useRef, useState } from 'react';
import type { KitIcon } from '../kit';
import { Button } from '../primitives/Button';
import type { ConversationTitleEditorCopy, UseKitEscape } from './types';

interface ConversationTitleEditorProps {
  title: string;
  /** Rejects with the Core's refusal, which is shown here in its own words. */
  onSave: (title: string) => Promise<void>;
  onCancel: () => void;
  copy: ConversationTitleEditorCopy;
  saveIcon: KitIcon;
  cancelIcon: KitIcon;
  /** Escape's registry, injected: the kit may not import the head's (#1036, #1158). */
  useEscape: UseKitEscape;
}

/**
 * A conversation's title, edited in place on its rail row.
 *
 * Enter saves and Escape abandons; Save and Cancel are buttons too, because a touch
 * keyboard does not always offer an Enter and never offers an Escape.
 *
 * Nothing is refused here. A blank title is sent to the Core like any other, because the
 * Core's refusal says what a title is *for*, and a second, head-written refusal would say
 * less and could disagree with it.
 */
export const ConversationTitleEditor: React.FC<ConversationTitleEditorProps> = ({
  title,
  onSave,
  onCancel,
  copy,
  saveIcon: SaveIcon,
  cancelIcon: CancelIcon,
  useEscape,
}) => {
  const [draft, setDraft] = useState(title);
  const [refusal, setRefusal] = useState<string | null>(null);
  const [saving, setSaving] = useState(false);
  const inputRef = useRef<HTMLInputElement>(null);

  useEffect(() => {
    inputRef.current?.focus();
    inputRef.current?.select();
  }, []);

  const save = async () => {
    setSaving(true);
    setRefusal(null);
    try {
      await onSave(draft);
    } catch (err) {
      setRefusal(err instanceof Error ? err.message : String(err));
      setSaving(false);
      inputRef.current?.focus();
    }
  };

  // Escape abandons the rename, but only when no higher-precedence layer claims it
  // first (#1036) -- a dialog or a running turn elsewhere must win over this.
  useEscape('overlay', true, onCancel);

  return (
    <form
      data-testid="conversation-title-editor"
      className="px-1 py-1"
      onSubmit={(event) => {
        event.preventDefault();
        void save();
      }}
    >
      <div className="flex items-center gap-1">
        <input
          ref={inputRef}
          aria-label={copy.titleLabel}
          value={draft}
          disabled={saving}
          onChange={(event) => setDraft(event.target.value)}
          onKeyDown={(event) => {
            if (event.key === 'Enter') {
              event.preventDefault();
              void save();
            }
          }}
          className="min-w-0 flex-1 rounded-md bg-slate-900 border border-slate-700 px-1.5 py-1 text-xs text-slate-100 focus:outline-none focus:border-slate-500"
        />
        <Button type="submit" variant="ghost" size="icon" aria-label={copy.save} disabled={saving}>
          <SaveIcon className="w-3.5 h-3.5" />
        </Button>
        <Button variant="ghost" size="icon" aria-label={copy.cancel} onClick={onCancel}>
          <CancelIcon className="w-3.5 h-3.5" />
        </Button>
      </div>
      {refusal ? (
        <p role="alert" className="px-0.5 pt-1 text-[11px] leading-snug text-rose-300">
          {refusal}
        </p>
      ) : null}
    </form>
  );
};
