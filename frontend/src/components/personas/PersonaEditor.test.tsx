import { describe, expect, it, vi } from 'vitest';
import { fireEvent, render, screen } from '@testing-library/react';
import React from 'react';
import { PersonaEditor, type PersonaEditorProps } from './PersonaEditor';
import { emptyPersonaDraft, type PersonaDraft } from '../../lib/personaDraft';
import type { ModelSet } from '../../lib/modelGateway';
import { en } from '../../i18n/en';
import { fmt, plural } from '../../i18n/format';
import editorSource from './PersonaEditor.tsx?raw';
import managerSource from './PersonaManager.tsx?raw';

const PERSONA_EDITOR_COPY = en.personaEditor;
const GATEWAY = en.gateway;

const entry = (connection: string, id: string, display_name: string, capabilities = ['chat']) => ({
  ref: `${connection}/${id}`,
  id,
  display_name,
  capabilities,
  context_window: null,
});

/** Two connections that list chat models and one that cannot (no key). */
const CHAT_SET: ModelSet = {
  groups: [
    {
      connection_id: 'gemini', label: 'Google Gemini', kind: 'gemini', status: 'connected', detail: null,
      models: [entry('gemini', 'gemini-3.8-pro', 'Gemini 3.8 Pro'), entry('gemini', 'gemini-3.8-flash', 'Gemini 3.8 Flash')],
    },
    {
      connection_id: 'ollama', label: 'Ollama', kind: 'ollama', status: 'connected', detail: null,
      models: [entry('ollama', 'qwen3:14b', 'qwen3:14b')],
    },
    { connection_id: 'openai', label: 'OpenAI', kind: 'openai', status: 'no_key', detail: 'No API key is set', models: [] },
  ],
  defaults: { deep: 'gemini/gemini-3.8-pro', fast: 'gemini/gemini-3.8-flash', image: 'auto' },
  recommended: { deep: 'gemini/gemini-3.8-pro', fast: null },
};

const IMAGE_SET: ModelSet = {
  groups: [
    {
      connection_id: 'gemini', label: 'Google Gemini', kind: 'gemini', status: 'connected', detail: null,
      models: [entry('gemini', 'gemini-2.5-flash-image', 'Gemini 2.5 Flash Image', ['image_create'])],
    },
  ],
  defaults: { deep: 'gemini/gemini-3.8-pro', fast: null, image: 'auto' },
  recommended: { deep: null, fast: null },
};

/**
 * The persona editor (#892) is a controlled, props-only form: what it shows is the draft it
 * was given, and what it does is hand a new draft to `onChange`. These tests render it the
 * way its parent does and read what a person would see and press.
 */

const Icon: React.FC<{ className?: string }> = ({ className }) => (
  <span data-icon className={className} />
);
const ICONS = { save: Icon, cancel: Icon, spinner: Icon };

const valid = (over: Partial<PersonaDraft> = {}): PersonaDraft => ({
  ...emptyPersonaDraft(),
  name: 'surveyor',
  role: 'Site Surveyor',
  system_prompt: 'You survey.',
  ...over,
});

const renderEditor = (over: Partial<PersonaEditorProps> = {}) => {
  const props: PersonaEditorProps = {
    draft: valid(),
    mode: 'create',
    existingNames: [],
    availableTools: ['map_area', 'dig_site'],
    chatModels: CHAT_SET,
    imageModels: IMAGE_SET,
    modelCopy: GATEWAY,
    isBuiltin: false,
    saving: false,
    error: null,
    copy: PERSONA_EDITOR_COPY,
    icons: ICONS,
    onChange: vi.fn(),
    onSubmit: vi.fn(),
    onCancel: vi.fn(),
    ...over,
  };
  render(<PersonaEditor {...props} />);
  return props;
};

describe('PersonaEditor', () => {
  it('offers the runtime tools as pills and adds the one pressed', () => {
    const props = renderEditor({ draft: valid({ allowed_tools: ['map_area'] }) });

    expect(screen.getByRole('button', { name: 'map_area' })).toHaveAttribute('aria-pressed', 'true');
    expect(screen.getByRole('button', { name: 'dig_site' })).toHaveAttribute('aria-pressed', 'false');

    fireEvent.click(screen.getByRole('button', { name: 'dig_site' }));

    // Killed by: frontend/src/components/personas/PersonaEditor.tsx :: : [...draft.allowed_tools, tool];
    // Becomes: : draft.allowed_tools;
    expect(props.onChange).toHaveBeenCalledWith(
      expect.objectContaining({ allowed_tools: ['map_area', 'dig_site'] }),
    );
  });

  it('says an empty tool selection means every tool, not only the base set', () => {
    // Killed by: frontend/src/components/personas/PersonaEditor.tsx :: <span>{copy.fields.toolsAllAllowedNotice ?? copy.noToolsSelected}</span>
    // Becomes: <span>{copy.noToolsSelected}</span>
    renderEditor({ draft: valid({ allowed_tools: [] }) });

    expect(screen.getByTestId('persona-no-tools-cause')).toHaveTextContent(
      PERSONA_EDITOR_COPY.fields.toolsAllAllowedNotice ?? PERSONA_EDITOR_COPY.noToolsSelected,
    );
    expect(screen.getByTestId('persona-tools-mode-badge')).toHaveTextContent(
      PERSONA_EDITOR_COPY.fields.toolsAllAllowedBadge!,
    );
    expect(screen.queryByTestId('persona-tools-reset')).not.toBeInTheDocument();
    expect(screen.queryByTestId('persona-restricted-tools-cause')).not.toBeInTheDocument();
  });

  it('renders the base tools section when baseTools are provided', () => {
    // Killed by: frontend/src/components/personas/PersonaEditor.tsx :: {baseList.length > 0 && (
    // Becomes: {false && (
    renderEditor({ baseTools: ['record_memory_fact', 'file_read'] });

    const baseSection = screen.getByTestId('persona-base-tools');
    expect(baseSection).toBeInTheDocument();
    expect(screen.getByTestId('persona-base-tool-pills')).toHaveTextContent('record_memory_fact');
    expect(screen.getByTestId('persona-base-tool-pills')).toHaveTextContent('file_read');
  });

  it('shows restricted badge and allows resetting to all tools when tools are selected', () => {
    const props = renderEditor({ draft: valid({ allowed_tools: ['map_area'] }) });

    expect(screen.getByTestId('persona-tools-mode-badge')).toHaveTextContent(
      fmt(PERSONA_EDITOR_COPY.fields.toolsRestrictedBadge, { count: 1, total: 2 }),
    );
    expect(screen.getByTestId('persona-restricted-tools-cause')).toHaveTextContent(
      plural(PERSONA_EDITOR_COPY.fields.toolsRestrictedNotice, 1),
    );
    expect(screen.getByTestId('persona-restricted-tools-cause')).toHaveTextContent(
      'plus the base tools',
    );
    expect(screen.queryByTestId('persona-no-tools-cause')).not.toBeInTheDocument();

    const resetBtn = screen.getByTestId('persona-tools-reset');
    expect(resetBtn).toHaveTextContent(PERSONA_EDITOR_COPY.fields.toolsResetToAll!);
    fireEvent.click(resetBtn);

    expect(props.onChange).toHaveBeenCalledWith(
      expect.objectContaining({ allowed_tools: [] }),
    );
  });

  it('keeps a listed tool the runtime no longer offers visible, so a save does not drop it unseen', () => {
    renderEditor({ draft: valid({ allowed_tools: ['retired_tool'] }) });

    // Killed by: frontend/src/components/personas/PersonaEditor.tsx :: ...draft.allowed_tools.filter((tool) => !availableTools.includes(tool)),
    // Becomes:
    expect(screen.getByRole('button', { name: 'retired_tool' })).toHaveAttribute('aria-pressed', 'true');
  });

  // Killed by: frontend/src/components/personas/PersonaEditor.tsx :: : fmt(copy.clone.systemDefault, { model: modelName(set, defaultNow) });
  // Becomes: : copy.clone.systemDefaultNone;
  it('starts the conversation model on the system default, naming it, and picks from the model set by ref', () => {
    const props = renderEditor();
    const select = screen.getByLabelText(GATEWAY.clone.conversation);

    expect(select).toHaveValue('');
    expect(screen.getByRole('option', { name: 'System default (now: Gemini 3.8 Pro)' })).toBeInTheDocument();
    // Grouped by connection, each model by its own name.
    expect(screen.getAllByRole('group').map((g) => g.getAttribute('label'))).toContain('Ollama');
    fireEvent.change(select, { target: { value: 'ollama/qwen3:14b' } });
    expect(props.onChange).toHaveBeenLastCalledWith(expect.objectContaining({ model_name: 'ollama/qwen3:14b' }));
  });

  it('goes back to following the system default when it is chosen again', () => {
    const props = renderEditor({ draft: valid({ model_name: 'ollama/qwen3:14b' }) });
    const select = screen.getByLabelText(GATEWAY.clone.conversation);

    expect(select).toHaveValue('ollama/qwen3:14b');
    fireEvent.change(select, { target: { value: '' } });
    expect(props.onChange).toHaveBeenLastCalledWith(expect.objectContaining({ model_name: null }));
  });

  it('chooses the picture model from the image set, with automatic offered', () => {
    const props = renderEditor();
    const select = screen.getByLabelText(GATEWAY.clone.picture);

    expect(select).toHaveValue('');
    expect(screen.getByRole('option', { name: GATEWAY.clone.systemDefaultAuto })).toBeInTheDocument();
    fireEvent.change(select, { target: { value: 'gemini/gemini-2.5-flash-image' } });
    expect(props.onChange).toHaveBeenLastCalledWith(
      expect.objectContaining({ image_model: 'gemini/gemini-2.5-flash-image' }),
    );
    fireEvent.change(select, { target: { value: 'auto' } });
    expect(props.onChange).toHaveBeenLastCalledWith(expect.objectContaining({ image_model: 'auto' }));
  });

  // Killed by: frontend/src/components/personas/PersonaEditor.tsx :: const unavailable = value !== null && value !== AUTO_IMAGE && findModel(set, value) === null;
  // Becomes: const unavailable = false;
  it('says a saved model its connection does not list is unavailable, and offers only Use system default', () => {
    const props = renderEditor({ draft: valid({ model_name: 'gpu-box/qwen3-coder' }) });

    const select = screen.getByLabelText(GATEWAY.clone.conversation);
    // Kept selected, never switched for the person.
    expect(select).toHaveValue('gpu-box/qwen3-coder');
    const notice = screen.getByTestId('persona-model-name-unavailable');
    expect(notice).toHaveTextContent(fmt(GATEWAY.clone.unavailable, { model: 'gpu-box/qwen3-coder' }));
    expect(props.onChange).not.toHaveBeenCalled();

    fireEvent.click(screen.getByRole('button', { name: GATEWAY.clone.useSystemDefault }));
    expect(props.onChange).toHaveBeenLastCalledWith(expect.objectContaining({ model_name: null }));
  });

  it('says why a connection offers nothing, instead of listing a model it cannot reach', () => {
    renderEditor();

    const lines = screen.getByTestId('persona-model-name-silent');
    expect(lines).toHaveTextContent(`OpenAI: ${GATEWAY.status.no_key}`);
    // Chosen by status, never the Core's English detail.
    expect(lines).not.toHaveTextContent('No API key is set');
  });

  // Killed by: frontend/src/components/personas/PersonaEditor.tsx :: {developerMode && (
  // Becomes: {true && (
  it('offers the clone`s own fast model only in developer mode', () => {
    renderEditor();
    expect(screen.queryByLabelText(GATEWAY.clone.fast)).toBeNull();
  });

  it('shows the fast model in developer mode, starting on the default fast model', () => {
    const props = renderEditor({ developerMode: true });
    const select = screen.getByLabelText(GATEWAY.clone.fast);

    expect(screen.getByRole('option', { name: 'System default (now: Gemini 3.8 Flash)' })).toBeInTheDocument();
    fireEvent.change(select, { target: { value: 'ollama/qwen3:14b' } });
    expect(props.onChange).toHaveBeenLastCalledWith(expect.objectContaining({ fast_model: 'ollama/qwen3:14b' }));
  });

  it('says the models are being read, and that a failed read keeps the saved choice', () => {
    renderEditor({ chatModels: null, imageModels: null });
    expect(screen.getByTestId('persona-model-name-pending')).toHaveTextContent(GATEWAY.clone.loading);

    renderEditor({ chatModels: null, imageModels: null, modelsFailed: true });
    expect(screen.getAllByText(GATEWAY.clone.loadFailed).length).toBeGreaterThan(0);
  });

  it('holds back the name rule on an untouched new form, and states it once the name is wrong', () => {
    const onChange = vi.fn();
    const Harness = () => {
      const [draft, setDraft] = React.useState<PersonaDraft>(emptyPersonaDraft());
      return (
        <PersonaEditor
          draft={draft}
          mode="create"
          existingNames={[]}
          availableTools={[]}
          chatModels={null}
          imageModels={null}
          modelCopy={GATEWAY}
          isBuiltin={false}
          saving={false}
          error={null}
          copy={PERSONA_EDITOR_COPY}
          icons={ICONS}
          onChange={(next) => {
            onChange(next);
            setDraft(next);
          }}
          onSubmit={vi.fn()}
          onCancel={vi.fn()}
        />
      );
    };
    render(<Harness />);

    expect(screen.queryByRole('alert')).not.toBeInTheDocument();
    expect(screen.getByRole('button', { name: PERSONA_EDITOR_COPY.save })).toBeDisabled();

    fireEvent.change(screen.getByLabelText(PERSONA_EDITOR_COPY.fields.name), {
      target: { value: '../escape' },
    });

    // Killed by: frontend/src/lib/personaDraft.ts :: if (!NAME_RULE.test(draft.name)) return copy.nameRule;
    // Becomes:
    expect(screen.getByRole('alert')).toHaveTextContent(PERSONA_EDITOR_COPY.nameRule);
    expect(screen.getByRole('button', { name: PERSONA_EDITOR_COPY.save })).toBeDisabled();
  });

  it('refuses a name that is already taken before the server has to', () => {
    renderEditor({ existingNames: ['surveyor'], error: null });

    fireEvent.change(screen.getByLabelText(PERSONA_EDITOR_COPY.fields.role), {
      target: { value: 'Changed' },
    });

    expect(screen.getByRole('alert')).toHaveTextContent(fmt(PERSONA_EDITOR_COPY.nameTaken, { name: 'surveyor' }));
  });

  it('locks the name on an edit and says a built-in is saved as the user\'s own copy', () => {
    renderEditor({ mode: 'edit', isBuiltin: true, existingNames: ['surveyor'] });

    expect(screen.getByLabelText(PERSONA_EDITOR_COPY.fields.name)).toBeDisabled();
    expect(screen.getByTestId('persona-builtin-note')).toHaveTextContent(
      PERSONA_EDITOR_COPY.builtinEditNote,
    );
    expect(screen.getByRole('button', { name: PERSONA_EDITOR_COPY.save })).toBeEnabled();
  });

  it("shows the server's refusal as it was worded", () => {
    renderEditor({ error: "persona 'surveyor' declares tool(s) that are not registered: 'x'" });

    expect(screen.getByRole('alert')).toHaveTextContent('declares tool(s) that are not registered');
  });

  it('submits a valid draft and cancels on request', () => {
    const props = renderEditor();

    fireEvent.click(screen.getByRole('button', { name: PERSONA_EDITOR_COPY.save }));
    fireEvent.click(screen.getByRole('button', { name: PERSONA_EDITOR_COPY.cancel }));

    expect(props.onSubmit).toHaveBeenCalledTimes(1);
    expect(props.onCancel).toHaveBeenCalledTimes(1);
  });

  it('does not submit while a save is in flight', () => {
    const props = renderEditor({ saving: true });

    fireEvent.submit(screen.getByTestId('persona-editor'));

    expect(props.onSubmit).not.toHaveBeenCalled();
    expect(screen.getByRole('button', { name: PERSONA_EDITOR_COPY.saving })).toBeDisabled();
  });

  it('updates temperature when semantic style preset buttons are clicked', () => {
    const props = renderEditor();

    fireEvent.click(screen.getByTestId('temperature-preset-precise'));
    expect(props.onChange).toHaveBeenCalledWith(
      expect.objectContaining({ temperature: 0.2 }),
    );

    fireEvent.click(screen.getByTestId('temperature-preset-creative'));
    expect(props.onChange).toHaveBeenCalledWith(
      expect.objectContaining({ temperature: 1.0 }),
    );
  });

  it('fills the instructions with the model draft and says which model wrote it', async () => {
    const onSynthesizePrompt = vi
      .fn()
      .mockResolvedValue({ ok: true, prompt: 'Drafted prompt', source: 'llm', model: 'qwen3:8b' });
    const props = renderEditor({ onSynthesizePrompt, draft: valid({ system_prompt: '' }) });

    fireEvent.click(screen.getByTestId('persona-synthesize-prompt'));

    expect(await screen.findByTestId('persona-draft-notice')).toHaveTextContent(
      'Drafted by qwen3:8b. Read it over before saving.',
    );
    expect(onSynthesizePrompt).toHaveBeenCalledTimes(1);
    expect(props.onChange).toHaveBeenCalledWith(
      expect.objectContaining({ system_prompt: 'Drafted prompt' }),
    );
    // Nothing was replaced, so there is nothing to undo.
    expect(screen.queryByTestId('persona-draft-undo')).not.toBeInTheDocument();
  });

  it('says a template draft was not written by a model, and why', async () => {
    const onSynthesizePrompt = vi.fn().mockResolvedValue({
      ok: true,
      prompt: 'You are Surveyor.',
      source: 'template',
      fallbackReason: 'connection refused',
    });
    renderEditor({ onSynthesizePrompt });

    fireEvent.click(screen.getByTestId('persona-synthesize-prompt'));

    const notice = await screen.findByTestId('persona-draft-notice');
    expect(notice).toHaveTextContent('No model answered (connection refused)');
    expect(notice).toHaveTextContent('basic template');
  });

  it('offers to restore the instructions a draft replaced', async () => {
    const onSynthesizePrompt = vi
      .fn()
      .mockResolvedValue({ ok: true, prompt: 'Drafted prompt', source: 'llm', model: 'qwen3:8b' });
    const props = renderEditor({ onSynthesizePrompt, draft: valid({ system_prompt: 'Mine.' }) });

    fireEvent.click(screen.getByTestId('persona-synthesize-prompt'));
    fireEvent.click(await screen.findByTestId('persona-draft-undo'));

    expect(props.onChange).toHaveBeenLastCalledWith(expect.objectContaining({ system_prompt: 'Mine.' }));
  });

  it('reports a failed draft and leaves the instructions alone', async () => {
    const onSynthesizePrompt = vi.fn().mockResolvedValue({ ok: false, message: 'HTTP 500' });
    const props = renderEditor({ onSynthesizePrompt });

    fireEvent.click(screen.getByTestId('persona-synthesize-prompt'));

    expect(await screen.findByTestId('persona-draft-notice')).toHaveTextContent(
      'Could not draft instructions (HTTP 500). Your text was not changed.',
    );
    expect(props.onChange).not.toHaveBeenCalled();
  });

  // Killed by: frontend/src/components/personas/PersonaEditor.tsx :: onChange(withDisplayName(draft, language, e.target.value));
  // Becomes: onChange(withDisplayName(draft, 'en', e.target.value));
  it('edits the display name for the screen language only, keeping the other languages', () => {
    const props = renderEditor({
      language: 'ko',
      draft: valid({ display_name: { en: 'Sleepyhead' } }),
    });
    const field = screen.getByLabelText(PERSONA_EDITOR_COPY.fields.displayName);
    expect(field).toHaveValue('');

    fireEvent.change(field, { target: { value: '잠 꾸러기' } });

    expect(props.onChange).toHaveBeenLastCalledWith(
      expect.objectContaining({ name: 'surveyor', display_name: { en: 'Sleepyhead', ko: '잠 꾸러기' } }),
    );
  });

  it('drops the screen language entry when its display name is cleared', () => {
    const props = renderEditor({
      language: 'ko',
      draft: valid({ display_name: { en: 'Sleepyhead', ko: '잠 꾸러기' } }),
    });
    const field = screen.getByLabelText(PERSONA_EDITOR_COPY.fields.displayName);
    expect(field).toHaveValue('잠 꾸러기');

    fireEvent.change(field, { target: { value: '' } });

    expect(props.onChange).toHaveBeenLastCalledWith(
      expect.objectContaining({ display_name: { en: 'Sleepyhead' } }),
    );
  });

  it('triggers onToggleStudio when studio mode button is clicked', () => {
    const onToggleStudio = vi.fn();
    renderEditor({ onToggleStudio, isStudio: false });

    const btn = screen.getByTestId('toggle-studio-mode');
    fireEvent.click(btn);
    expect(onToggleStudio).toHaveBeenCalledTimes(1);
  });

  it('automatically enables enable_write_tools when a write tool pill is added', () => {
    // Killed by: frontend/src/components/personas/PersonaEditor.tsx :: enable_write_tools: true,
    // Becomes: enable_write_tools: false,
    const props = renderEditor({
      availableTools: ['map_area', 'file_write'],
      writeTools: ['file_write'],
      draft: valid({ allowed_tools: ['map_area'], enable_write_tools: false }),
    });

    const fileWritePill = screen.getByRole('button', { name: 'file_write' });
    fireEvent.click(fileWritePill);

    expect(props.onChange).toHaveBeenLastCalledWith(
      expect.objectContaining({
        allowed_tools: ['map_area', 'file_write'],
        enable_write_tools: true,
      }),
    );
  });

  it('renders a warning and allows one-click enable when draft has write tools but write flag is off', () => {
    // Killed by: frontend/src/components/personas/PersonaEditor.tsx :: data-testid="persona-write-tools-warning"
    // Becomes: data-testid="persona-write-tools-disabled"
    const props = renderEditor({
      availableTools: ['file_write'],
      writeTools: ['file_write'],
      draft: valid({ allowed_tools: ['file_write'], enable_write_tools: false }),
    });

    const warning = screen.getByTestId('persona-write-tools-warning');
    expect(warning).toBeInTheDocument();

    const enableBtn = screen.getByRole('button', {
      name: PERSONA_EDITOR_COPY.fields.enableWriteToolsAction ?? 'Allow file-writing tools',
    });
    fireEvent.click(enableBtn);

    expect(props.onChange).toHaveBeenLastCalledWith(
      expect.objectContaining({ enable_write_tools: true }),
    );
  });
});

/**
 * Owner decision #1063 D: components are props-only. This reads the two sources rather than
 * trusting a reviewer to, so a later edit that reaches for a request or a store from inside
 * them fails here.
 */
describe('persona components stay props-only', () => {
  it.each([
    ['PersonaEditor.tsx', editorSource],
    ['PersonaManager.tsx', managerSource],
  ])('%s makes no request and imports no request module', (_name, source) => {
    expect(source).not.toMatch(/\bfetch\s*\(/);
    expect(source).not.toMatch(/\b(?:window|globalThis|self)\s*\.\s*fetch\b/);
    expect(source).not.toMatch(/personasApi|personaCopy|zustand|\/store/);
    expect(source).not.toMatch(/from 'lucide-react'/);
  });
});
