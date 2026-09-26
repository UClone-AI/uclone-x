import React, { useState, useEffect, useRef } from 'react';
import { Bot, MessageSquarePlus, Pencil, Plus, Save, X, Loader2, Sparkles } from 'lucide-react';
import { PersonaInfo } from '../../types';
import { personaAvatarUrl } from '../../lib/personaAvatar';
import { PersonaDetail } from '../layout/PersonaDetail';
import { Avatar } from '../../ui-kit';
import { PersonaEditor } from '../personas/PersonaEditor';
import {
  draftFromPersona,
  emptyPersonaDraft,
  type PersonaDraft,
  type PersonaEditMode,
  type PersonaSaveResult,
} from '../../lib/personaDraft';
import { PERSONA_EDITOR_COPY } from '../../lib/personaCopy';
import { synthesizePersonaPrompt } from '../../lib/personasApi';
import { useEscapeOwner } from '../../lib/escapePrecedence';

export interface CloneProfileProps {
  /** The clone the rail has selected. Empty when none is. */
  cloneId: string;
  /** Its installed definition, when a file under that name is loaded. */
  persona?: PersonaInfo;
  /** Open a new conversation seating this clone. */
  onStartConversation: (cloneId: string) => void;
  /** Current mode of the clone panel. Defaults to 'view'. */
  mode?: 'view' | 'edit' | 'create';
  /** Switch between view, edit, and create modes. */
  onModeChange?: (mode: 'view' | 'edit' | 'create') => void;
  /** Direct trigger to start editing. */
  onEdit?: (cloneId: string) => void;
  /** Available tools from runtime. */
  availableTools?: readonly string[];
  /** The running clone's tools that it is given only inside a conversation (#1595). */
  toolsNeedingConversation?: readonly string[];
  /** Available models from runtime. */
  availableModels?: readonly string[];
  /** Existing persona names to prevent duplicate names on create. */
  existingNames?: readonly string[];
  /** Whether the workspace directory is writable. */
  canWrite?: boolean;
  /** Handler to persist the persona draft. */
  onSave?: (draft: PersonaDraft, mode: PersonaEditMode) => Promise<PersonaSaveResult>;
  /**
   * Whether the editor has the whole workspace -- the conversation column and the dock -- to
   * itself (Studio mode). The host owns it, because only the host can give the editor that
   * room: the editor stays where it is in the dock and the host hides the conversation beside
   * it. It used to be this component's own state, drawn as a `fixed` panel over the page, and
   * the dock's `backdrop-blur` made the dock that panel's containing block, so the "full
   * screen" was the dock's own box with a 32px margin -- a modal-looking card in a panel.
   */
  studio?: boolean;
  /** Enter or leave Studio mode. Without it the editor offers no Studio mode at all. */
  onStudioChange?: (studio: boolean) => void;
}

const EDITOR_ICONS = {
  save: Save,
  cancel: X,
  spinner: Loader2,
  sparkles: Sparkles,
};

const safeDraftFromPersona = (persona: PersonaInfo): PersonaDraft =>
  draftFromPersona({
    ...persona,
    allowed_tools: Array.isArray(persona.allowed_tools) ? persona.allowed_tools : [],
  });

/**
 * One clone's profile: who it is, and the actions you can perform with it.
 *
 * It supports viewing the inspection card, editing an existing persona, or creating
 * a brand new persona directly in the dock panel without modal interruption.
 */
export const CloneProfile: React.FC<CloneProfileProps> = ({
  cloneId,
  persona,
  onStartConversation,
  mode = 'view',
  onModeChange,
  onEdit,
  availableTools = [],
  toolsNeedingConversation,
  availableModels = [],
  existingNames = [],
  canWrite = true,
  onSave,
  studio = false,
  onStudioChange,
}) => {
  const isCreate = mode === 'create';
  const isEdit = mode === 'edit';

  const [draft, setDraft] = useState<PersonaDraft>(() =>
    isCreate ? emptyPersonaDraft() : persona ? safeDraftFromPersona(persona) : emptyPersonaDraft(),
  );
  const [saving, setSaving] = useState(false);
  const [saveError, setSaveError] = useState<string | null>(null);

  const lastTargetRef = useRef<{
    mode?: 'view' | 'edit' | 'create';
    cloneId?: string;
    personaLoaded?: boolean;
  }>({
    mode,
    cloneId,
    personaLoaded: Boolean(persona),
  });

  useEffect(() => {
    if (mode === 'create') {
      if (lastTargetRef.current.mode !== 'create') {
        lastTargetRef.current = { mode: 'create' };
        setDraft(emptyPersonaDraft());
        setSaveError(null);
      }
    } else if (mode === 'edit') {
      const isNewTarget =
        lastTargetRef.current.mode !== 'edit' ||
        lastTargetRef.current.cloneId !== cloneId;
      const personaJustLoaded =
        !lastTargetRef.current.personaLoaded && Boolean(persona);
      if (isNewTarget || personaJustLoaded) {
        lastTargetRef.current = {
          mode: 'edit',
          cloneId,
          personaLoaded: Boolean(persona),
        };
        setDraft(persona ? safeDraftFromPersona(persona) : emptyPersonaDraft());
        setSaveError(null);
      }
    } else {
      lastTargetRef.current = {
        mode: 'view',
        cloneId,
        personaLoaded: Boolean(persona),
      };
    }
  }, [mode, cloneId, persona]);

  const handleEditClick = () => {
    if (onModeChange) {
      onModeChange('edit');
    } else if (onEdit) {
      onEdit(cloneId);
    }
  };

  const handleEditorCancel = () => {
    setSaveError(null);
    onStudioChange?.(false); // Cancel leaves Studio mode
    onModeChange?.('view');
  };

  // Escape leaves Studio mode and keeps the draft: leaving Studio is not leaving the editor,
  // the same one re-renders in the dock with what was typed (#1377). Its layer ranks below
  // every dialog and overlay. An Escape on a dropdown or one ending an input method's
  // composition belongs to that control, not to this.
  useEscapeOwner('studio', studio && (isCreate || isEdit) && onStudioChange !== undefined, (event) => {
    if (event.target instanceof HTMLSelectElement) return;
    if (event.isComposing) return;
    onStudioChange?.(false);
  });

  const handleSynthesizePrompt = () =>
    synthesizePersonaPrompt({
      name: draft.name,
      role: draft.role,
      description: draft.description,
      allowed_tools: draft.allowed_tools,
    });

  const handleEditorSubmit = async () => {
    if (!onSave) return;
    setSaving(true);
    setSaveError(null);
    const result = await onSave(draft, isCreate ? 'create' : 'edit');
    setSaving(false);
    if (result.ok) {
      onStudioChange?.(false); // and so does a save that landed
      onModeChange?.('view');
    } else {
      setSaveError(result.message);
    }
  };

  // Editor mode: creating a new clone or editing an existing one
  if (isCreate || isEdit) {
    const editor = (
      <PersonaEditor
        draft={draft}
        mode={isCreate ? 'create' : 'edit'}
        existingNames={existingNames}
        availableTools={availableTools}
        availableModels={availableModels}
        isBuiltin={persona?.builtin ?? false}
        saving={saving}
        error={saveError}
        copy={PERSONA_EDITOR_COPY}
        icons={EDITOR_ICONS}
        isStudio={studio}
        onToggleStudio={onStudioChange ? () => onStudioChange(!studio) : undefined}
        onSynthesizePrompt={handleSynthesizePrompt}
        onChange={(next) => {
          setDraft(next);
          setSaveError(null);
        }}
        onSubmit={handleEditorSubmit}
        onCancel={handleEditorCancel}
      />
    );

    // One wrapper in both modes, so toggling Studio mode re-renders the editor rather than
    // remounting it: what has been typed, and which sections are open, survive the toggle.
    // In Studio mode it is centred at a reading width in the space the host has given it.
    return (
      <div
        data-testid="clone-profile-editor"
        data-studio={studio ? 'true' : 'false'}
        className={studio ? 'mx-auto w-full max-w-4xl space-y-3' : 'space-y-3'}
      >
        {editor}
      </div>
    );
  }

  // View mode, but no clone selected
  if (cloneId === '') {
    return (
      <div className="space-y-3">
        <p data-testid="clone-profile-none" className="text-sm text-slate-400 leading-relaxed">
          No clone is picked yet. Choose one under <span className="text-slate-300">Clones</span> in
          the left rail to see what it is set up to do and to start a conversation with it.
        </p>
        {onModeChange && canWrite && (
          <button
            type="button"
            data-testid="clone-profile-create-empty"
            onClick={() => onModeChange('create')}
            className="flex items-center gap-1.5 px-3 py-1.5 rounded-lg border border-slate-700 bg-slate-800 text-xs font-medium text-slate-200 hover:bg-slate-700 hover:text-white transition-colors"
          >
            <Plus className="w-3.5 h-3.5" />
            {PERSONA_EDITOR_COPY.newPersona}
          </button>
        )}
      </div>
    );
  }

  const name = persona?.name ?? cloneId;

  return (
    <div data-testid={`clone-profile-${cloneId}`} className="space-y-4">
      <div className="flex flex-col items-center text-center gap-2">
        <Avatar
          label={name}
          kind="agent"
          agentIcon={Bot}
          size="lg"
          imageSrc={personaAvatarUrl(name)}
          data-testid="clone-profile-avatar"
        />
        {/* The heading is on one line so that a mutation declaration can name it. A needle
            cannot span lines and must occur once in this file; the bare expression below
            appears in the avatar label and the button label too, so the element carries it. */}
        <h2 data-testid="clone-profile-name" className="text-lg font-semibold text-slate-100">{name}</h2>
        {/* The role, not the name again: for a clone the two are different facts, and the
            name is already the heading an inch above. A clone whose file sets no role says
            so, rather than leaving a gap a reader has to interpret (P6). */}
        <p
          data-testid="clone-profile-role"
          className="text-[11px] font-mono uppercase tracking-wide text-slate-500"
        >
          {persona?.role || 'No role set'}
        </p>
      </div>

      <div className="flex items-center gap-2">
        <button
          type="button"
          data-testid="clone-profile-start"
          aria-label={`Start a conversation with ${name}`}
          onClick={() => onStartConversation(cloneId)}
          className="flex-1 flex items-center justify-center gap-2 px-3 py-2 rounded-lg border border-indigo-500/40 bg-indigo-950/40 text-sm font-medium text-indigo-200 hover:bg-indigo-900/50 transition-colors"
        >
          <MessageSquarePlus className="w-4 h-4" />
          Start a conversation
        </button>
        {(onModeChange || onEdit) && persona && (
          <button
            type="button"
            data-testid="clone-profile-edit"
            aria-label={`Edit ${name}`}
            disabled={!canWrite}
            onClick={handleEditClick}
            className="flex items-center justify-center gap-1.5 px-3 py-2 rounded-lg border border-slate-700 bg-slate-800/80 text-sm font-medium text-slate-200 hover:bg-slate-700 hover:text-white transition-colors disabled:opacity-50"
            title={canWrite ? `Edit ${name}` : PERSONA_EDITOR_COPY.noWorkspace}
          >
            <Pencil className="w-4 h-4" />
            Edit
          </button>
        )}
      </div>

      {persona ? (
        <PersonaDetail persona={persona} toolsNeedingConversation={toolsNeedingConversation} />
      ) : (
        // A clone can be running with no file installed under its name -- the rail builds a
        // row from the live instance in that case. Saying so is the point: "this clone has no
        // settings" and "the settings did not load" are the same blank panel otherwise.
        <p data-testid="clone-profile-no-definition" className="text-sm text-slate-400 leading-relaxed">
          Nothing is installed under this name, so there is no description, model or tool list
          to show. A clone gets those from a definition file in the workspace&apos;s personas
          directory.
        </p>
      )}
    </div>
  );
};
