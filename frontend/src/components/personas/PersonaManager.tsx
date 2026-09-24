import React, { useState } from 'react';
import type { PersonaInfo } from '../../types';
import {
  draftFromPersona,
  emptyPersonaDraft,
  type PersonaDraft,
  type PersonaEditMode,
  type PersonaEditorCopy,
  type PersonaSaveResult,
} from '../../lib/personaDraft';
import { PersonaEditor, type IconComponent, type PersonaEditorIcons } from './PersonaEditor';

/**
 * The persona list with its editor (#892): pick one to edit, or start a new one. Props-only
 * like `PersonaEditor`. The catalogue comes in as `personas`, and a save goes out through
 * `onSave`, whose result decides whether the editor closes (saved) or stays open showing
 * the refusal. What the list shows after a save is whatever the parent passes back in;
 * nothing here assumes the save changed it.
 */

export interface PersonaManagerIcons extends PersonaEditorIcons {
  add: IconComponent;
  edit: IconComponent;
}

export interface PersonaManagerProps {
  personas: readonly PersonaInfo[];
  availableTools: readonly string[];
  availableModels: readonly string[];
  /** Where saves are written; null when the runtime has no workspace to write to. */
  personasDir: string | null;
  /** Why the catalogue could not be loaded, or null. */
  loadError: string | null;
  copy: PersonaEditorCopy;
  icons: PersonaManagerIcons;
  onSave: (draft: PersonaDraft, mode: PersonaEditMode) => Promise<PersonaSaveResult>;
}

interface Editing {
  mode: PersonaEditMode;
  draft: PersonaDraft;
  isBuiltin: boolean;
}

export const PersonaManager: React.FC<PersonaManagerProps> = ({
  personas,
  availableTools,
  availableModels,
  personasDir,
  loadError,
  copy,
  icons,
  onSave,
}) => {
  const [editing, setEditing] = useState<Editing | null>(null);
  const [saving, setSaving] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [notice, setNotice] = useState<string | null>(null);

  const open = (next: Editing) => {
    setEditing(next);
    setError(null);
    setNotice(null);
  };

  const submit = async () => {
    if (!editing) return;
    setSaving(true);
    setError(null);
    const result = await onSave(editing.draft, editing.mode);
    setSaving(false);
    if (result.ok) {
      setNotice(copy.saved(result.persona.name, result.liveAgentsUpdated));
      setEditing(null);
    } else {
      setError(result.message);
    }
  };

  const AddIcon = icons.add;
  const EditIcon = icons.edit;
  const canWrite = personasDir !== null;

  if (editing) {
    return (
      <PersonaEditor
        draft={editing.draft}
        mode={editing.mode}
        existingNames={personas.map((p) => p.name)}
        availableTools={availableTools}
        availableModels={availableModels}
        isBuiltin={editing.isBuiltin}
        saving={saving}
        error={error}
        copy={copy}
        icons={icons}
        onChange={(draft) => {
          setEditing({ ...editing, draft });
          setError(null);
        }}
        onSubmit={submit}
        onCancel={() => setEditing(null)}
      />
    );
  }

  return (
    <div data-testid="persona-manager" className="space-y-3">
      <div className="flex items-center justify-between gap-3">
        <p className="text-[11px] text-slate-500 truncate" title={personasDir ?? undefined}>
          {personasDir !== null ? copy.savedTo(personasDir) : copy.noWorkspace}
        </p>
        <button
          type="button"
          disabled={!canWrite || loadError !== null}
          onClick={() => open({ mode: 'create', draft: emptyPersonaDraft(), isBuiltin: false })}
          className="shrink-0 px-3 py-1.5 rounded-xl border border-slate-700 text-xs text-slate-200 hover:bg-slate-800 flex items-center gap-1.5 disabled:opacity-50"
        >
          <AddIcon className="w-3.5 h-3.5" />
          {copy.newPersona}
        </button>
      </div>

      {notice ? (
        <p role="status" className="text-xs text-emerald-300">
          {notice}
        </p>
      ) : null}

      {loadError !== null ? (
        <p role="alert" className="text-xs text-rose-300">
          {copy.loadFailed(loadError)}
        </p>
      ) : personas.length === 0 ? (
        <p className="text-xs text-slate-500">{copy.emptyList}</p>
      ) : (
        <ul className="divide-y divide-slate-800/80 border border-slate-800 rounded-xl">
          {personas.map((persona) => (
            <li
              key={persona.name}
              data-testid={`persona-row-${persona.name}`}
              className="px-3 py-2 flex items-center justify-between gap-3"
            >
              <div className="min-w-0">
                <div className="flex items-center gap-2">
                  <span className="text-xs font-mono text-white truncate">{persona.name}</span>
                  {persona.builtin ? (
                    <span className="text-[10px] text-slate-400 border border-slate-700 rounded px-1">
                      {copy.builtinBadge}
                    </span>
                  ) : persona.overrides_builtin ? (
                    <span className="text-[10px] text-slate-400 border border-slate-700 rounded px-1">
                      {copy.overrideBadge}
                    </span>
                  ) : null}
                </div>
                <div className="text-[11px] text-slate-400 truncate">{persona.role}</div>
              </div>
              <button
                type="button"
                disabled={!canWrite}
                aria-label={`${copy.edit} ${persona.name}`}
                onClick={() =>
                  open({
                    mode: 'edit',
                    draft: draftFromPersona(persona),
                    isBuiltin: persona.builtin ?? false,
                  })
                }
                className="shrink-0 px-2.5 py-1 rounded-lg border border-slate-800 text-[11px] text-slate-300 hover:bg-slate-800 flex items-center gap-1 disabled:opacity-50"
              >
                <EditIcon className="w-3 h-3" />
                {copy.edit}
              </button>
            </li>
          ))}
        </ul>
      )}
    </div>
  );
};
