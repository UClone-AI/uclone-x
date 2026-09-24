import { describe, expect, it, vi } from 'vitest';
import { fireEvent, render, screen, waitFor } from '@testing-library/react';
import React from 'react';
import { PersonaManager, type PersonaManagerProps } from './PersonaManager';
import { PERSONA_EDITOR_COPY } from '../../lib/personaCopy';
import type { PersonaSaveResult } from '../../lib/personaDraft';
import { makePersonaInfo } from '../../test/fixtures';

/**
 * The persona list and its editor (#892). Saving is a callback whose result decides what
 * happens next: a save that went through closes the editor and says so, a refused one keeps
 * the editor open with the server's sentence.
 */

const Icon: React.FC<{ className?: string }> = ({ className }) => (
  <span data-icon className={className} />
);
const ICONS = { add: Icon, edit: Icon, save: Icon, cancel: Icon, spinner: Icon };

const renderManager = (over: Partial<PersonaManagerProps> = {}) => {
  const props: PersonaManagerProps = {
    personas: [
      makePersonaInfo({ name: 'guide', role: 'Guide', builtin: true, system_prompt: 'You guide.' }),
      makePersonaInfo({
        name: 'reader',
        role: 'Reader',
        overrides_builtin: true,
        system_prompt: 'You read.',
      }),
    ],
    availableTools: ['read_file', 'web_search'],
    availableModels: ['qwen3:8b'],
    personasDir: '/work/.uclone/personas',
    loadError: null,
    copy: PERSONA_EDITOR_COPY,
    icons: ICONS,
    onSave: vi.fn(
      async (draft): Promise<PersonaSaveResult> => ({
        ok: true,
        persona: makePersonaInfo({ name: draft.name }),
        liveAgentsUpdated: 1,
      }),
    ),
    ...over,
  };
  render(<PersonaManager {...props} />);
  return props;
};

describe('PersonaManager', () => {
  it('lists every persona, marks the built-in and the customized one, and says where saves go', () => {
    renderManager();

    expect(screen.getByTestId('persona-row-guide')).toHaveTextContent(PERSONA_EDITOR_COPY.builtinBadge);
    expect(screen.getByTestId('persona-row-reader')).toHaveTextContent(PERSONA_EDITOR_COPY.overrideBadge);
    expect(screen.getByText(PERSONA_EDITOR_COPY.savedTo('/work/.uclone/personas'))).toBeInTheDocument();
  });

  it('states why the list is empty', () => {
    renderManager({ personas: [] });

    expect(screen.getByText(PERSONA_EDITOR_COPY.emptyList)).toBeInTheDocument();
  });

  it('says a runtime with no workspace cannot save, and offers no button that would fail', () => {
    renderManager({ personasDir: null });

    expect(screen.getByText(PERSONA_EDITOR_COPY.noWorkspace)).toBeInTheDocument();
    // Killed by: frontend/src/components/personas/PersonaManager.tsx :: disabled={!canWrite || loadError !== null}
    // Becomes: disabled={loadError !== null}
    expect(screen.getByRole('button', { name: PERSONA_EDITOR_COPY.newPersona })).toBeDisabled();
  });

  it('creates a persona from a blank form and reports the save', async () => {
    const props = renderManager();

    fireEvent.click(screen.getByRole('button', { name: PERSONA_EDITOR_COPY.newPersona }));
    fireEvent.change(screen.getByLabelText(PERSONA_EDITOR_COPY.fields.name), {
      target: { value: 'surveyor' },
    });
    fireEvent.change(screen.getByLabelText(PERSONA_EDITOR_COPY.fields.role), {
      target: { value: 'Site Surveyor' },
    });
    fireEvent.change(screen.getByLabelText(PERSONA_EDITOR_COPY.fields.systemPrompt), {
      target: { value: 'You survey.' },
    });
    fireEvent.click(screen.getByRole('button', { name: 'read_file' }));
    fireEvent.click(screen.getByRole('button', { name: PERSONA_EDITOR_COPY.save }));

    await waitFor(() =>
      expect(screen.getByRole('status')).toHaveTextContent(PERSONA_EDITOR_COPY.saved('surveyor', 1)),
    );
    expect(props.onSave).toHaveBeenCalledWith(
      expect.objectContaining({
        name: 'surveyor',
        role: 'Site Surveyor',
        system_prompt: 'You survey.',
        allowed_tools: ['read_file'],
      }),
      'create',
    );
    expect(screen.queryByTestId('persona-editor')).not.toBeInTheDocument();
  });

  it('opens an edit on the persona as the catalogue describes it, prompt included', async () => {
    const props = renderManager();

    fireEvent.click(screen.getByRole('button', { name: `${PERSONA_EDITOR_COPY.edit} guide` }));

    expect(screen.getByLabelText(PERSONA_EDITOR_COPY.fields.systemPrompt)).toHaveValue('You guide.');
    expect(screen.getByTestId('persona-builtin-note')).toBeInTheDocument();
    fireEvent.click(screen.getByRole('button', { name: PERSONA_EDITOR_COPY.save }));
    // Killed by: frontend/src/components/personas/PersonaManager.tsx :: draft: draftFromPersona(persona),
    // Becomes: draft: emptyPersonaDraft(),
    await waitFor(() =>
      expect(props.onSave).toHaveBeenCalledWith(
        expect.objectContaining({ name: 'guide', system_prompt: 'You guide.' }),
        'edit',
      ),
    );
  });

  it.each([
    [true, false],
    [false, true],
  ])(
    'offers the write and sub-agent switches at their stored values, and saves them (write %s, sub-agents %s)',
    async (write, subagents) => {
      const props = renderManager({
        personas: [
          makePersonaInfo({
            name: 'builder',
            role: 'Builder',
            system_prompt: 'You build.',
            enable_write_tools: write,
            enable_subagent_tools: subagents,
          }),
        ],
      });

      fireEvent.click(screen.getByRole('button', { name: `${PERSONA_EDITOR_COPY.edit} builder` }));

      // #1167: the runtime enforces both flags now, so the editor offers them again --
      // alongside the default-prompt one -- each showing what the persona actually stores.
      expect(screen.getAllByRole('checkbox')).toHaveLength(3);
      const writeBox = screen.getByRole('checkbox', {
        name: PERSONA_EDITOR_COPY.fields.enableWriteTools,
      });
      const subagentBox = screen.getByRole('checkbox', {
        name: PERSONA_EDITOR_COPY.fields.enableSubagentTools,
      });
      expect((writeBox as HTMLInputElement).checked).toBe(write);
      expect((subagentBox as HTMLInputElement).checked).toBe(subagents);
      fireEvent.click(screen.getByRole('button', { name: PERSONA_EDITOR_COPY.save }));

      // Killed by: frontend/src/lib/personaDraft.ts :: enable_write_tools: persona.enable_write_tools ?? false,
      // Becomes: enable_write_tools: false,
      // Killed by: frontend/src/lib/personaDraft.ts :: enable_subagent_tools: persona.enable_subagent_tools ?? false,
      // Becomes: enable_subagent_tools: false,
      await waitFor(() =>
        expect(props.onSave).toHaveBeenCalledWith(
          expect.objectContaining({
            name: 'builder',
            enable_write_tools: write,
            enable_subagent_tools: subagents,
          }),
          'edit',
        ),
      );
    },
  );

  it('saves the capability switches as the person left them, not as they were loaded', async () => {
    // The switch has to change the payload, not merely render (#1167). #1166's editor also
    // "saved the stored values" -- by never offering the switch -- so a test that only
    // round-trips what it loaded passes against an editor that does nothing.
    const props = renderManager({
      personas: [
        makePersonaInfo({
          name: 'builder',
          role: 'Builder',
          system_prompt: 'You build.',
          enable_write_tools: false,
          enable_subagent_tools: true,
        }),
      ],
    });

    fireEvent.click(screen.getByRole('button', { name: `${PERSONA_EDITOR_COPY.edit} builder` }));
    fireEvent.click(
      screen.getByRole('checkbox', { name: PERSONA_EDITOR_COPY.fields.enableWriteTools }),
    );
    fireEvent.click(
      screen.getByRole('checkbox', { name: PERSONA_EDITOR_COPY.fields.enableSubagentTools }),
    );
    fireEvent.click(screen.getByRole('button', { name: PERSONA_EDITOR_COPY.save }));

    // The copy-paste the two near-identical checkbox blocks invite: the write box writing
    // the subagent flag, so the pressed box moves a field the reader did not touch.
    // Killed by: frontend/src/components/personas/PersonaEditor.tsx :: onChange={(e) => set('enable_write_tools', e.target.checked)}
    // Becomes: onChange={(e) => set('enable_subagent_tools', e.target.checked)}
    await waitFor(() =>
      expect(props.onSave).toHaveBeenCalledWith(
        expect.objectContaining({
          name: 'builder',
          enable_write_tools: true,
          enable_subagent_tools: false,
        }),
        'edit',
      ),
    );
  });

  it('keeps the editor open with the refusal when the server refuses the save', async () => {
    renderManager({
      onSave: vi.fn(async (): Promise<PersonaSaveResult> => ({ ok: false, message: 'Refused: reason.' })),
    });

    fireEvent.click(screen.getByRole('button', { name: `${PERSONA_EDITOR_COPY.edit} reader` }));
    fireEvent.click(screen.getByRole('button', { name: PERSONA_EDITOR_COPY.save }));

    // Killed by: frontend/src/components/personas/PersonaManager.tsx :: setError(result.message);
    // Becomes:
    await waitFor(() => expect(screen.getByRole('alert')).toHaveTextContent('Refused: reason.'));
    expect(screen.getByTestId('persona-editor')).toBeInTheDocument();
  });

  it('states why the catalogue is missing when it could not be loaded', () => {
    renderManager({ loadError: 'The clones folder is missing.' });

    expect(screen.getByRole('alert')).toHaveTextContent(
      PERSONA_EDITOR_COPY.loadFailed('The clones folder is missing.'),
    );
  });
});
