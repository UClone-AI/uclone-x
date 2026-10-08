import React, { useState } from 'react';
import type { PersonaModelTier } from '../../types';
import type { Messages } from '../../i18n';
import { AUTO_IMAGE, findModel, modelName, type ModelSet } from '../../lib/modelGateway';
import { ModelRefSelect } from '../settings/ModelRefSelect';
import { pickerWords } from '../settings/gatewayWords';
import {
  MODEL_TIERS,
  cloneLabel,
  draftProblem,
  fmt,
  plural,
  withDisplayName,
  type PersonaDraft,
  type PersonaEditMode,
  type PersonaEditorCopy,
  type PromptDraftResult,
} from '../../lib/personaDraft';

/**
 * The form that creates or edits one persona (#892). Controlled and props-only: the draft,
 * the choices, the words and the icons all arrive as props, and every change leaves as a
 * new draft through `onChange`. Nothing here fetches or saves.
 *
 * Choices are offered, not recalled: tools are pills drawn from the runtime's own tool
 * list, the conversation and picture models are chosen from the model set grouped by
 * connection, each starting on the system default (model-gateway.md §3.7), and the tier is a
 * dropdown. Only the name, role and prose are typed.
 */

export type IconComponent = React.ComponentType<{ className?: string }>;

export interface PersonaEditorIcons {
  save: IconComponent;
  cancel: IconComponent;
  spinner: IconComponent;
  sparkles?: IconComponent;
}

export interface PersonaEditorProps {
  draft: PersonaDraft;
  mode: PersonaEditMode;
  /** Names already in the catalogue, so create can say a name is taken before saving. */
  existingNames: readonly string[];
  availableTools: readonly string[];
  baseTools?: readonly string[];
  writeTools?: readonly string[];
  /** The chat model set, `null` until it has been read. */
  chatModels: ModelSet | null;
  /** The picture model set, `null` until it has been read. */
  imageModels: ModelSet | null;
  /** Whether reading either set failed. */
  modelsFailed?: boolean;
  /** Developer mode: the clone's own fast model is offered only then (decision 3). */
  developerMode?: boolean;
  /** The model pickers' words (the `gateway` catalog). */
  modelCopy: Messages['gateway'];
  /** The persona being edited ships with the app; saving writes an override instead. */
  isBuiltin: boolean;
  saving: boolean;
  /** The last refusal, as the server worded it. */
  error: string | null;
  copy: PersonaEditorCopy;
  /** The screen's language: the display name field edits this language's entry only. */
  language?: string;
  icons: PersonaEditorIcons;
  onChange: (draft: PersonaDraft) => void;
  onSubmit: () => void;
  onCancel: () => void;
  /** Drafts instructions from the draft's fields; the editor shows how the draft was made. */
  onSynthesizePrompt?: () => Promise<PromptDraftResult>;
  isStudio?: boolean;
  onToggleStudio?: () => void;
}

const DEFAULT_WRITE_TOOLS: readonly string[] = [
  'file_write',
  'file_edit',
  'bash_run',
  'run_command',
  'install_package',
  'a2a_call',
  'story_library',
  'story_outline',
  'story_manuscript',
  'story_codex',
  'story_start',
  'character_sheet',
];

const inputClass =
  'w-full bg-slate-950/90 border border-slate-800 rounded-xl px-3 py-2 text-xs text-white placeholder-slate-600 focus:outline-none focus:border-cyan-500/80 transition-colors';
const labelClass = 'block text-xs font-medium text-slate-300 mb-1.5';
/** The pickers' spelling of "follow the system default", saved as `null`. */
const SYSTEM_DEFAULT = '';

type CloneSlot = 'model_name' | 'fast_model' | 'image_model';

/**
 * One of the clone's model pickers: the system default first, naming what it is now, then
 * the model set. A saved ref no connection lists is said to be unavailable, with one action,
 * Use system default, which clears the slot (§3.6). Nothing is ever switched for the person.
 */
const CloneModelPicker: React.FC<{
  slot: CloneSlot;
  label: string;
  value: string | null;
  set: ModelSet | null;
  failed: boolean;
  /** The default's own value now, from the set: a ref, `auto`, or `null` for none chosen. */
  defaultNow: string | null;
  allowAuto: boolean;
  hint?: string;
  copy: Messages['gateway'];
  onChange: (value: string | null) => void;
}> = ({ slot, label, value, set, failed, defaultNow, allowAuto, hint, copy, onChange }) => {
  const id = `persona-${slot.replace('_', '-')}`;
  if (set === null) {
    return (
      <div data-testid={`${id}-pending`}>
        <span className={labelClass}>{label}</span>
        <p className="text-[11px] text-slate-500" role={failed ? 'alert' : 'status'}>
          {failed ? copy.clone.loadFailed : copy.clone.loading}
        </p>
      </div>
    );
  }
  const defaultLabel =
    defaultNow === null
      ? copy.clone.systemDefaultNone
      : defaultNow === AUTO_IMAGE
        ? copy.clone.systemDefaultAuto
        : fmt(copy.clone.systemDefault, { model: modelName(set, defaultNow) });
  const leading = [
    { value: SYSTEM_DEFAULT, label: defaultLabel },
    ...(allowAuto ? [{ value: AUTO_IMAGE, label: copy.clone.auto }] : []),
  ];
  const own = value ?? SYSTEM_DEFAULT;
  const unavailable = value !== null && value !== AUTO_IMAGE && findModel(set, value) === null;
  return (
    <div>
      <ModelRefSelect
        id={id}
        testId={id}
        label={label}
        value={own}
        set={set}
        leading={leading}
        words={pickerWords(copy, '')}
        onChange={(next) => onChange(next === SYSTEM_DEFAULT ? null : next)}
      />
      {unavailable && (
        <div
          data-testid={`${id}-unavailable`}
          role="alert"
          className="mt-1.5 flex flex-wrap items-center gap-2 text-[11px] text-amber-300"
        >
          <span>{fmt(copy.clone.unavailable, { model: value })}</span>
          <button
            type="button"
            onClick={() => onChange(null)}
            className="rounded-md border border-amber-500/40 px-2 py-0.5 text-amber-200 hover:bg-amber-500/10 transition-colors"
          >
            {copy.clone.useSystemDefault}
          </button>
        </div>
      )}
      {hint && <p className="text-[10px] text-slate-500 mt-1">{hint}</p>}
    </div>
  );
};

export const PersonaEditor: React.FC<PersonaEditorProps> = ({
  draft,
  mode,
  existingNames,
  availableTools,
  baseTools,
  writeTools,
  chatModels,
  imageModels,
  modelsFailed = false,
  developerMode = false,
  modelCopy,
  isBuiltin,
  saving,
  error,
  copy,
  language = 'en',
  icons,
  onChange,
  onSubmit,
  onCancel,
  onSynthesizePrompt,
  isStudio = false,
  onToggleStudio,
}) => {
  // A new, untouched form is not yet wrong, so its problem is held back until the first
  // change; Save stays disabled either way.
  const [touched, setTouched] = useState<boolean>(false);
  const [isSynthesizing, setIsSynthesizing] = useState<boolean>(false);
  // What the last draft attempt did, and the instructions it replaced so it can be undone.
  const [draftNotice, setDraftNotice] = useState<{ tone: 'info' | 'warn'; text: string } | null>(
    null,
  );
  const [replacedPrompt, setReplacedPrompt] = useState<string | null>(null);
  const [showAdvanced, setShowAdvanced] = useState<boolean>(false);
  const set = <K extends keyof PersonaDraft>(key: K, value: PersonaDraft[K]) => {
    setTouched(true);
    onChange({ ...draft, [key]: value });
  };

  // Tools the persona lists but the runtime no longer offers stay visible, so saving does
  // not drop them unseen; the server will name them if they are refused.
  const toolChoices = [
    ...availableTools,
    ...draft.allowed_tools.filter((tool) => !availableTools.includes(tool)),
  ];
  const baseList = baseTools ?? [];
  const writeList = writeTools && writeTools.length > 0 ? writeTools : DEFAULT_WRITE_TOOLS;
  const toggleTool = (tool: string) => {
    const isAdding = !draft.allowed_tools.includes(tool);
    const nextAllowed = draft.allowed_tools.includes(tool)
      ? draft.allowed_tools.filter((t) => t !== tool)
      : [...draft.allowed_tools, tool];
    if (isAdding && writeList.includes(tool) && !draft.enable_write_tools) {
      setTouched(true);
      onChange({
        ...draft,
        allowed_tools: nextAllowed,
        enable_write_tools: true,
      });
      return;
    }
    set('allowed_tools', nextAllowed);
  };

  const problem = draftProblem(draft, mode, existingNames, copy);
  const SaveIcon = saving ? icons.spinner : icons.save;
  const CancelIcon = icons.cancel;
  const SparklesIcon = icons.sparkles;

  return (
    <form
      data-testid="persona-editor"
      className="space-y-4"
      onSubmit={(e) => {
        e.preventDefault();
        if (!problem && !saving) onSubmit();
      }}
    >
      <div className="flex items-center justify-between gap-2 border-b border-slate-800 pb-3">
        <div>
          <h3 className="text-sm font-semibold text-white">
            {mode === 'create' ? copy.createTitle : fmt(copy.editTitle, { name: cloneLabel(draft, language) })}
          </h3>
          {mode === 'edit' && isBuiltin ? (
            <p data-testid="persona-builtin-note" className="text-[11px] text-slate-400 mt-0.5">
              {copy.builtinEditNote}
            </p>
          ) : null}
        </div>
        {onToggleStudio && (
          <button
            type="button"
            data-testid="toggle-studio-mode"
            onClick={onToggleStudio}
            className="px-2.5 py-1 rounded-lg border border-slate-700 bg-slate-800/80 text-xs text-slate-200 hover:bg-slate-700 hover:text-white transition-colors flex items-center gap-1.5 shrink-0"
            title={isStudio ? 'Collapse to dock' : 'Expand to Studio mode'}
          >
            <span>{isStudio ? '⤡' : '⤢'}</span>
            <span>{isStudio ? 'Dock mode' : 'Studio mode'}</span>
          </button>
        )}
      </div>

      <div className="grid grid-cols-1 sm:grid-cols-2 gap-3.5">
        <div>
          <label className={labelClass} htmlFor="persona-display-name">
            {copy.fields.displayName}
          </label>
          <input
            id="persona-display-name"
            data-testid="persona-display-name"
            className={inputClass}
            value={draft.display_name[language] ?? ''}
            maxLength={64}
            onChange={(e) => {
              setTouched(true);
              onChange(withDisplayName(draft, language, e.target.value));
            }}
          />
          <p className="text-[11px] text-slate-500 mt-1">{copy.fields.displayNameHint}</p>
        </div>
        <div>
          <label className={labelClass} htmlFor="persona-name">
            {copy.fields.name}
          </label>
          <input
            id="persona-name"
            className={`${inputClass} font-mono disabled:opacity-60`}
            value={draft.name}
            disabled={mode === 'edit'}
            onChange={(e) => set('name', e.target.value)}
          />
          <p className="text-[11px] text-slate-500 mt-1">{copy.fields.nameHint}</p>
        </div>
        <div>
          <label className={labelClass} htmlFor="persona-role">
            {copy.fields.role}
          </label>
          <input
            id="persona-role"
            className={inputClass}
            value={draft.role}
            onChange={(e) => set('role', e.target.value)}
          />
        </div>
      </div>

      <div>
        <label className={labelClass} htmlFor="persona-description">
          {copy.fields.description}
        </label>
        <input
          id="persona-description"
          className={inputClass}
          value={draft.description}
          onChange={(e) => set('description', e.target.value)}
        />
      </div>

      <div>
        <div className="flex items-center justify-between mb-1.5">
          <label className={labelClass} htmlFor="persona-prompt">
            {copy.fields.systemPrompt}
          </label>
          {onSynthesizePrompt && (
            <button
              type="button"
              data-testid="persona-synthesize-prompt"
              disabled={isSynthesizing || (!draft.name && !draft.role && !draft.description)}
              onClick={async () => {
                setIsSynthesizing(true);
                setDraftNotice(null);
                setReplacedPrompt(null);
                try {
                  const result = await onSynthesizePrompt();
                  if (!result.ok) {
                    setDraftNotice({ tone: 'warn', text: fmt(copy.fields.draftFailed, { reason: result.message }) });
                    return;
                  }
                  if (draft.system_prompt.trim()) setReplacedPrompt(draft.system_prompt);
                  set('system_prompt', result.prompt);
                  setDraftNotice(
                    result.source === 'llm'
                      ? { tone: 'info', text: fmt(copy.fields.draftedByModel, { model: result.model }) }
                      : { tone: 'warn', text: fmt(copy.fields.draftedFromTemplate, { reason: result.fallbackReason }) },
                  );
                } finally {
                  setIsSynthesizing(false);
                }
              }}
              className="text-[11px] font-medium text-cyan-400 hover:text-cyan-300 disabled:opacity-40 disabled:hover:text-cyan-400 flex items-center gap-1.5 transition-colors px-2 py-0.5 rounded border border-cyan-800/60 bg-cyan-950/40 hover:bg-cyan-900/50"
              title={copy.fields.generateHint}
            >
              {isSynthesizing ? (
                <>
                  <span className="w-3 h-3 border border-cyan-400 border-t-transparent rounded-full animate-spin inline-block" />
                  <span>{copy.fields.generatingPrompt}</span>
                </>
              ) : (
                <>
                  {SparklesIcon ? <SparklesIcon className="w-3 h-3" /> : <span>✨</span>}
                  <span>{copy.fields.generateWithAi}</span>
                </>
              )}
            </button>
          )}
        </div>
        <textarea
          id="persona-prompt"
          rows={isStudio ? 14 : 6}
          className={`${inputClass} font-mono leading-relaxed`}
          value={draft.system_prompt}
          onChange={(e) => {
            // Undo would discard what was typed since the draft, so typing retires it.
            setReplacedPrompt(null);
            set('system_prompt', e.target.value);
          }}
        />
        {draftNotice && (
          <p
            data-testid="persona-draft-notice"
            role="status"
            className={`mt-1.5 text-[11px] ${draftNotice.tone === 'warn' ? 'text-amber-300' : 'text-slate-400'}`}
          >
            {draftNotice.text}
            {replacedPrompt !== null && (
              <button
                type="button"
                data-testid="persona-draft-undo"
                className="ml-2 underline text-slate-300 hover:text-white"
                onClick={() => {
                  set('system_prompt', replacedPrompt);
                  setReplacedPrompt(null);
                  setDraftNotice(null);
                }}
              >
                {copy.fields.undoDraft}
              </button>
            )}
          </p>
        )}
        <label className="mt-1.5 flex items-center gap-2 text-[11px] text-slate-400">
          <input
            type="checkbox"
            checked={draft.append_default_prompt}
            onChange={(e) => set('append_default_prompt', e.target.checked)}
          />
          {copy.fields.appendDefaultPrompt}
        </label>
      </div>

      <fieldset className="space-y-2">
        <div className="flex items-center justify-between gap-2 flex-wrap">
          <div className="flex items-center gap-2">
            <legend className={labelClass}>{copy.fields.tools}</legend>
            {toolChoices.length > 0 && (
              draft.allowed_tools.length === 0 ? (
                <span
                  data-testid="persona-tools-mode-badge"
                  className="px-2 py-0.5 rounded-full text-[10px] font-medium bg-emerald-950/60 border border-emerald-800/60 text-emerald-300 flex items-center gap-1.5"
                >
                  <span className="w-1.5 h-1.5 rounded-full bg-emerald-400 inline-block" />
                  {copy.fields.toolsAllAllowedBadge ?? 'All tools allowed'}
                </span>
              ) : (
                <span
                  data-testid="persona-tools-mode-badge"
                  className="px-2 py-0.5 rounded-full text-[10px] font-medium bg-amber-950/60 border border-amber-800/60 text-amber-300 flex items-center gap-1.5"
                >
                  <span className="w-1.5 h-1.5 rounded-full bg-amber-400 inline-block" />
                  {copy.fields.toolsRestrictedBadge
                    ? fmt(copy.fields.toolsRestrictedBadge, { count: draft.allowed_tools.length, total: toolChoices.length })
                    : `Restricted allowlist (${draft.allowed_tools.length}/${toolChoices.length})`}
                </span>
              )
            )}
          </div>
          {draft.allowed_tools.length > 0 && (
            <button
              type="button"
              data-testid="persona-tools-reset"
              onClick={() => set('allowed_tools', [])}
              className="text-[11px] text-cyan-400 hover:text-cyan-300 transition-colors underline underline-offset-2 font-medium cursor-pointer"
            >
              {copy.fields.toolsResetToAll ?? 'Allow all tools'}
            </button>
          )}
        </div>

        <p className="text-[11px] text-slate-400 leading-normal">
          {copy.fields.toolsHint ??
            'By default, all runtime tools are available. Select tools below only to restrict this clone to a specific allowlist.'}
        </p>

        {toolChoices.length === 0 ? (
          <p className="text-[11px] text-slate-500">{copy.noToolsAvailable}</p>
        ) : (
          <div className="flex flex-wrap gap-1.5" data-testid="persona-tool-pills">
            {toolChoices.map((tool) => {
              const selected = draft.allowed_tools.includes(tool);
              return (
                <button
                  key={tool}
                  type="button"
                  aria-pressed={selected}
                  onClick={() => toggleTool(tool)}
                  className={`px-2 py-1 rounded-lg border font-mono text-[11px] transition-colors ${
                    selected
                      ? 'bg-cyan-950/60 border-cyan-600/80 text-cyan-200'
                      : 'bg-slate-950/60 border-slate-800 text-slate-400 hover:border-slate-700'
                  }`}
                >
                  {tool}
                </button>
              );
            })}
          </div>
        )}
        {toolChoices.length > 0 && draft.allowed_tools.length === 0 ? (
          <p
            data-testid="persona-no-tools-cause"
            className="flex items-center gap-1.5 text-[11px] text-slate-400 bg-slate-900/50 border border-slate-800/80 px-2.5 py-1.5 rounded-lg mt-1.5"
          >
            <span className="text-emerald-400 text-xs">✓</span>
            <span>{copy.fields.toolsAllAllowedNotice ?? copy.noToolsSelected}</span>
          </p>
        ) : draft.allowed_tools.length > 0 ? (
          <div className="space-y-1.5 mt-1.5">
            <p
              data-testid="persona-restricted-tools-cause"
              className="flex items-center gap-1.5 text-[11px] text-amber-300/90 bg-amber-950/30 border border-amber-900/50 px-2.5 py-1.5 rounded-lg"
            >
              <span className="text-amber-400 text-xs">🔒</span>
              <span>
                {copy.fields.toolsRestrictedNotice
                  ? plural(copy.fields.toolsRestrictedNotice, draft.allowed_tools.length)
                  : `Restricted mode: The ${draft.allowed_tools.length} selected tool(s) plus the base tools will be available.`}
              </span>
            </p>
            {draft.allowed_tools.some((t) => writeList.includes(t)) && !draft.enable_write_tools && (
              <div
                data-testid="persona-write-tools-warning"
                className="flex items-center justify-between gap-2 p-2 rounded-lg bg-amber-950/40 border border-amber-800/60 text-amber-200 text-xs"
              >
                <span>⚠️ {copy.fields.writeToolsDisabledWarning ?? 'Some selected tools require file-writing permission to work.'}</span>
                <button
                  type="button"
                  onClick={() => set('enable_write_tools', true)}
                  className="px-2 py-0.5 bg-amber-800/80 hover:bg-amber-700 text-amber-100 rounded text-[11px] font-medium whitespace-nowrap cursor-pointer"
                >
                  {copy.fields.enableWriteToolsAction ?? 'Allow file-writing tools'}
                </button>
              </div>
            )}
          </div>
        ) : null}

        {baseList.length > 0 && (
          <div data-testid="persona-base-tools" className="space-y-1 pt-1">
            <div className="text-[10px] text-slate-400 font-medium font-sans">
              {copy.fields.baseTools ?? 'Base tools (always included)'}
            </div>
            <p className="text-[10px] text-slate-500">
              {copy.fields.baseToolsHint ?? 'These tools are included for every clone and do not need to be saved in the persona.'}
            </p>
            <div className="flex flex-wrap gap-1" data-testid="persona-base-tool-pills">
              {baseList.map((tool) => (
                <span key={tool} className="px-1.5 py-0.5 rounded bg-slate-900 border border-slate-800 text-slate-400 font-mono text-[10px]">
                  {tool}
                </span>
              ))}
            </div>
          </div>
        )}
      </fieldset>

      <div className="grid grid-cols-1 sm:grid-cols-2 gap-3.5">
        <CloneModelPicker
          slot="model_name"
          label={modelCopy.clone.conversation}
          value={draft.model_name}
          set={chatModels}
          failed={modelsFailed}
          defaultNow={chatModels?.defaults.deep ?? null}
          allowAuto={false}
          copy={modelCopy}
          onChange={(next) => set('model_name', next)}
        />
        <CloneModelPicker
          slot="image_model"
          label={modelCopy.clone.picture}
          value={draft.image_model}
          set={imageModels}
          failed={modelsFailed}
          defaultNow={imageModels?.defaults.image ?? AUTO_IMAGE}
          allowAuto
          copy={modelCopy}
          onChange={(next) => set('image_model', next)}
        />
        {developerMode && (
          <CloneModelPicker
            slot="fast_model"
            label={modelCopy.clone.fast}
            value={draft.fast_model}
            set={chatModels}
            failed={modelsFailed}
            defaultNow={chatModels ? (chatModels.defaults.fast ?? chatModels.defaults.deep) : null}
            allowAuto={false}
            hint={modelCopy.clone.fastHint}
            copy={modelCopy}
            onChange={(next) => set('fast_model', next)}
          />
        )}
        <div>
          <label className={labelClass} htmlFor="persona-tier">
            {copy.fields.tier}
          </label>
          <select
            id="persona-tier"
            className={`${inputClass} cursor-pointer`}
            value={draft.model_tier}
            onChange={(e) => set('model_tier', e.target.value as PersonaModelTier)}
          >
            {MODEL_TIERS.map((tier) => (
              <option key={tier} value={tier}>
                {copy.tierLabels[tier]}
              </option>
            ))}
          </select>
        </div>
        <div className="space-y-1.5 sm:col-span-2">
          <label className={labelClass}>{copy.fields.temperature}</label>
          <div className="grid grid-cols-3 gap-1.5" role="group" aria-label={copy.fields.temperature}>
            {[
              {
                key: 'precise',
                label: copy.fields.temperaturePresets?.precise ?? 'Precise (0.2)',
                value: 0.2,
                icon: '🎯',
              },
              {
                key: 'balanced',
                label: copy.fields.temperaturePresets?.balanced ?? 'Balanced (0.7)',
                value: 0.7,
                icon: '⚖️',
              },
              {
                key: 'creative',
                label: copy.fields.temperaturePresets?.creative ?? 'Creative (1.0)',
                value: 1.0,
                icon: '🎨',
              },
            ].map((preset) => {
              const isSelected = Math.abs(draft.temperature - preset.value) < 0.05;
              return (
                <button
                  key={preset.key}
                  type="button"
                  data-testid={`temperature-preset-${preset.key}`}
                  aria-pressed={isSelected}
                  onClick={() => set('temperature', preset.value)}
                  className={`px-2 py-1.5 rounded-xl border text-[11px] font-medium flex items-center justify-center gap-1.5 transition-all ${
                    isSelected
                      ? 'border-cyan-500 bg-cyan-950/60 text-cyan-200 shadow-sm'
                      : 'border-slate-800 bg-slate-950/60 text-slate-400 hover:border-slate-700 hover:text-slate-200'
                  }`}
                >
                  <span>{preset.icon}</span>
                  <span className="truncate">{preset.label}</span>
                </button>
              );
            })}
          </div>

          <div className="pt-1">
            <button
              type="button"
              data-testid="toggle-advanced-settings"
              onClick={() => setShowAdvanced(!showAdvanced)}
              className="text-[11px] text-slate-500 hover:text-slate-300 flex items-center gap-1 transition-colors"
            >
              <span>{showAdvanced ? '▼' : '▶'}</span>
              <span>{copy.fields.advancedSettings ?? 'Advanced settings'}</span>
            </button>

            {showAdvanced && (
              <div className="mt-2 grid grid-cols-1 sm:grid-cols-2 gap-3 p-3 bg-slate-950/40 border border-slate-800/80 rounded-xl">
                <div>
                  <label className={labelClass} htmlFor="persona-temperature">
                    Exact {copy.fields.temperature}
                  </label>
                  <input
                    id="persona-temperature"
                    type="number"
                    min={0}
                    max={2}
                    step={0.1}
                    className={`${inputClass} font-mono`}
                    value={draft.temperature}
                    onChange={(e) => {
                      const value = Number.parseFloat(e.target.value);
                      if (!Number.isNaN(value)) set('temperature', value);
                    }}
                  />
                </div>
                <div>
                  <label className={labelClass} htmlFor="persona-max-tokens">
                    {copy.fields.maxTokens}
                  </label>
                  <input
                    id="persona-max-tokens"
                    type="number"
                    min={1}
                    step={1}
                    className={`${inputClass} font-mono`}
                    value={draft.max_tokens ?? ''}
                    onChange={(e) => {
                      const value = Number.parseInt(e.target.value, 10);
                      set('max_tokens', Number.isNaN(value) ? null : value);
                    }}
                  />
                  <p className="text-[10px] text-slate-500 mt-1">{copy.fields.maxTokensHint}</p>
                </div>
              </div>
            )}
          </div>
        </div>
      </div>

      {/*
        Offered again now that the runtime enforces both (#1167). #1166 hid them because a
        switch that changes nothing says it does; that reason is spent -- `BaseAgent` reads
        each flag on every tool call, and its refusal names these very labels.
      */}
      <fieldset>
        <legend className={labelClass}>{copy.fields.capabilities}</legend>
        <label className="flex items-center gap-2 text-[11px] text-slate-400">
          <input
            type="checkbox"
            checked={draft.enable_write_tools}
            onChange={(e) => set('enable_write_tools', e.target.checked)}
          />
          {copy.fields.enableWriteTools}
        </label>
        <label className="mt-1.5 flex items-center gap-2 text-[11px] text-slate-400">
          <input
            type="checkbox"
            checked={draft.enable_subagent_tools}
            onChange={(e) => set('enable_subagent_tools', e.target.checked)}
          />
          {copy.fields.enableSubagentTools}
        </label>
        <p className="text-[11px] text-slate-500 mt-1">{copy.fields.capabilitiesHint}</p>
      </fieldset>

      <p className="text-[11px] text-slate-500">{copy.fields.modelSettingsHint}</p>

      {error || (problem && touched) ? (
        <p
          role="alert"
          data-testid="persona-editor-problem"
          className="p-3 rounded-xl border border-rose-800/80 bg-rose-950/50 text-rose-200 text-xs"
        >
          {error ?? problem}
        </p>
      ) : null}

      <div className="flex items-center justify-end gap-2.5">
        <button
          type="button"
          onClick={onCancel}
          className="px-3.5 py-1.5 rounded-xl border border-slate-800 text-xs text-slate-400 hover:text-white hover:bg-slate-800 flex items-center gap-1.5"
        >
          <CancelIcon className="w-3.5 h-3.5" />
          {copy.cancel}
        </button>
        <button
          type="submit"
          disabled={problem !== null || saving}
          className="px-4 py-1.5 rounded-xl bg-cyan-700 hover:bg-cyan-600 text-white text-xs font-semibold flex items-center gap-1.5 disabled:opacity-50"
        >
          <SaveIcon className={saving ? 'w-3.5 h-3.5 animate-spin' : 'w-3.5 h-3.5'} />
          {saving ? copy.saving : copy.save}
        </button>
      </div>
    </form>
  );
};
