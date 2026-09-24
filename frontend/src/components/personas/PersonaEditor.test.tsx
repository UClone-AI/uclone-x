import { describe, expect, it, vi } from 'vitest';
import { fireEvent, render, screen } from '@testing-library/react';
import React from 'react';
import { PersonaEditor, type PersonaEditorProps } from './PersonaEditor';
import { emptyPersonaDraft, type PersonaDraft } from '../../lib/personaDraft';
import { PERSONA_EDITOR_COPY } from '../../lib/personaCopy';
import editorSource from './PersonaEditor.tsx?raw';
import managerSource from './PersonaManager.tsx?raw';

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
    availableModels: ['qwen3:8b', 'hermes3:8b'],
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

    // Killed by: frontend/src/components/personas/PersonaEditor.tsx :: : [...draft.allowed_tools, tool],
    // Becomes: : draft.allowed_tools,
    expect(props.onChange).toHaveBeenCalledWith(
      expect.objectContaining({ allowed_tools: ['map_area', 'dig_site'] }),
    );
  });

  it('says an empty tool selection means every tool, not none', () => {
    renderEditor({ draft: valid({ allowed_tools: [] }) });

    expect(screen.getByTestId('persona-no-tools-cause')).toHaveTextContent(
      PERSONA_EDITOR_COPY.noToolsSelected,
    );
    expect(screen.getByTestId('persona-tools-mode-badge')).toHaveTextContent(
      PERSONA_EDITOR_COPY.fields.toolsAllAllowedBadge!,
    );
    expect(screen.queryByTestId('persona-tools-reset')).not.toBeInTheDocument();
    expect(screen.queryByTestId('persona-restricted-tools-cause')).not.toBeInTheDocument();
  });

  it('shows restricted badge and allows resetting to all tools when tools are selected', () => {
    const props = renderEditor({ draft: valid({ allowed_tools: ['map_area'] }) });

    expect(screen.getByTestId('persona-tools-mode-badge')).toHaveTextContent(
      PERSONA_EDITOR_COPY.fields.toolsRestrictedBadge!(1, 2),
    );
    expect(screen.getByTestId('persona-restricted-tools-cause')).toHaveTextContent(
      PERSONA_EDITOR_COPY.fields.toolsRestrictedNotice!(1),
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

  it('chooses the model from the detected list, and the runtime default clears it', () => {
    const props = renderEditor({ draft: valid({ model_name: 'qwen3:8b' }) });
    const select = screen.getByLabelText(PERSONA_EDITOR_COPY.fields.model);

    expect(select).toHaveValue('qwen3:8b');
    fireEvent.change(select, { target: { value: 'hermes3:8b' } });
    expect(props.onChange).toHaveBeenLastCalledWith(expect.objectContaining({ model_name: 'hermes3:8b' }));
    fireEvent.change(select, { target: { value: '' } });
    expect(props.onChange).toHaveBeenLastCalledWith(expect.objectContaining({ model_name: null }));
  });

  it('shows a model the runtime did not detect as "other", with its tag kept', () => {
    renderEditor({ draft: valid({ model_name: 'mystery:1b' }) });

    expect(screen.getByLabelText(PERSONA_EDITOR_COPY.fields.model)).toHaveValue('__other__');
    expect(screen.getByLabelText(PERSONA_EDITOR_COPY.fields.modelOther)).toHaveValue('mystery:1b');
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
          availableModels={[]}
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

    expect(screen.getByRole('alert')).toHaveTextContent(PERSONA_EDITOR_COPY.nameTaken('surveyor'));
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

  it('triggers onToggleStudio when studio mode button is clicked', () => {
    const onToggleStudio = vi.fn();
    renderEditor({ onToggleStudio, isStudio: false });

    const btn = screen.getByTestId('toggle-studio-mode');
    fireEvent.click(btn);
    expect(onToggleStudio).toHaveBeenCalledTimes(1);
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
