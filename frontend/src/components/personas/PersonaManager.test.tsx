import { describe, expect, it, vi } from 'vitest';
import { fireEvent, render, screen, waitFor } from '@testing-library/react';
import React from 'react';
import { PersonaManager, type PersonaManagerProps } from './PersonaManager';
import { en } from '../../i18n/en';
import { ko } from '../../i18n/ko';

const ko_personaEditor = ko.personaEditor;
import { fmt, plural } from '../../i18n/format';
import type { PersonaSaveResult } from '../../lib/personaDraft';
import { makePersonaInfo } from '../../test/fixtures';

const PERSONA_EDITOR_COPY = en.personaEditor;

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
    chatModels: null,
    imageModels: null,
    modelCopy: en.gateway,
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
    expect(screen.getByText(fmt(PERSONA_EDITOR_COPY.savedTo, { dir: '/work/.uclone/personas' }))).toBeInTheDocument();
  });

  it('marks a clone that could not be loaded, says why, and offers no edit (#2220)', () => {
    const reason = "This clone's model does not say which connection it is on.";
    renderManager({
      personas: [
        makePersonaInfo({ name: 'guide', role: 'Guide', system_prompt: 'You guide.' }),
        makePersonaInfo({ name: 'bad', role: '', status: 'unreadable', reason }),
      ],
    });

    const row = screen.getByTestId('persona-row-bad');
    expect(row).toHaveTextContent(PERSONA_EDITOR_COPY.unreadableBadge);
    // Killed by: frontend/src/components/personas/PersonaManager.tsx :: <p className="text-[11px] text-amber-200/90">{unreadableReason(persona, copy)}</p>
    // Becomes: <p className="text-[11px] text-amber-200/90">{persona.role}</p>
    // With no code (an older Core), the Core's own sentence is shown.
    expect(row).toHaveTextContent(reason);
    // Killed by: frontend/src/components/personas/PersonaManager.tsx :: disabled={!canWrite || unreadable}
    // Becomes: disabled={!canWrite}
    expect(screen.getByRole('button', { name: `${PERSONA_EDITOR_COPY.edit} bad` })).toBeDisabled();
    expect(screen.getByRole('button', { name: `${PERSONA_EDITOR_COPY.edit} guide` })).toBeEnabled();
  });

  // Killed by: frontend/src/lib/personaDraft.ts :: if (code && Object.prototype.hasOwnProperty.call(copy.unreadableReasons, code)) {
  // Becomes: if (false) {
  it('words the reason in the screen language when the Core sends its code (#2241)', () => {
    const english = "This clone's model does not say which connection it is on.";
    const ko = ko_personaEditor;
    renderManager({
      copy: ko,
      personas: [
        makePersonaInfo({ name: 'bad', role: '', status: 'unreadable', reason: english, reason_code: 'bare_model' }),
        makePersonaInfo({ name: 'odd', role: '', status: 'unreadable', reason: 'Old words.', reason_code: 'newer_code' }),
      ],
    });

    const bad = screen.getByTestId('persona-row-bad');
    expect(bad).toHaveTextContent(ko.unreadableReasons.bare_model);
    expect(bad).not.toHaveTextContent(english);
    // A code this head does not know yet falls back to the Core's sentence.
    expect(screen.getByTestId('persona-row-odd')).toHaveTextContent('Old words.');
  });

  // #2282: two clones can claim one handle (both listed unreadable); keyed by handle, React
  // saw one key twice and could drop or swap their rows on the next render.
  // Killed by: frontend/src/components/personas/PersonaManager.tsx :: key={cloneIdOf(persona)}
  // Becomes: key={persona.name}
  it('keys each row by its clone id, so two clones sharing a handle stay two rows', () => {
    const errors = vi.spyOn(console, 'error').mockImplementation(() => {});
    renderManager({
      personas: [
        makePersonaInfo({ id: 'agt_a', name: 'twin', role: '', status: 'unreadable', reason: 'One.' }),
        makePersonaInfo({ id: 'agt_b', name: 'twin', role: '', status: 'unreadable', reason: 'Two.' }),
      ],
    });

    expect(errors.mock.calls.flat().join(' ')).not.toMatch(/same key/);
    errors.mockRestore();
  });

  // #2286: the test id was the handle, so the two rows shared one. It is the clone's id now,
  // and the handle is on `data-clone-handle`, which the end-to-end tests look rows up by.
  // Killed by: frontend/src/components/personas/PersonaManager.tsx :: data-testid={`persona-row-${cloneIdOf(persona)}`}
  // Becomes: data-testid={`persona-row-${persona.name}`}
  it('gives each row its own test id, and names its handle beside it', () => {
    renderManager({
      personas: [
        makePersonaInfo({ id: 'agt_a', name: 'twin', role: '', status: 'unreadable', reason: 'One.' }),
        makePersonaInfo({ id: 'agt_b', name: 'twin', role: '', status: 'unreadable', reason: 'Two.' }),
      ],
    });

    expect(screen.getByTestId('persona-row-agt_a')).toHaveTextContent('One.');
    expect(screen.getByTestId('persona-row-agt_b')).toHaveTextContent('Two.');
    expect(screen.getByTestId('persona-row-agt_a').getAttribute('data-clone-handle')).toBe('twin');
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
      expect(screen.getByRole('status')).toHaveTextContent(plural(PERSONA_EDITOR_COPY.savedWithLive, 1, { name: 'surveyor' })),
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
      // alongside the default-prompt one and "Learn from conversations" (#1840) -- each
      // showing what the persona actually stores.
      expect(screen.getAllByRole('checkbox')).toHaveLength(4);
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

  it.each([
    [undefined, true],
    [true, true],
    [false, false],
  ])(
    'shows "Learn from conversations" as the clone stores it (stored %s, shown %s) and saves what the person leaves',
    async (stored, shown) => {
      // #1840: one switch per clone, on by default. A clone listed without the field -- every
      // clone saved before the switch existed -- reads as on, as the Core loads it.
      const props = renderManager({
        personas: [
          makePersonaInfo({
            name: 'builder',
            role: 'Builder',
            system_prompt: 'You build.',
            ...(stored === undefined ? {} : { learn_from_conversations: stored }),
          }),
        ],
      });

      fireEvent.click(screen.getByRole('button', { name: `${PERSONA_EDITOR_COPY.edit} builder` }));
      const learnBox = screen.getByRole('checkbox', {
        name: PERSONA_EDITOR_COPY.fields.learnFromConversations,
      }) as HTMLInputElement;
      // Killed by: frontend/src/lib/personaDraft.ts :: learn_from_conversations: persona.learn_from_conversations ?? true,
      // Becomes: learn_from_conversations: persona.learn_from_conversations ?? false,
      expect(learnBox.checked).toBe(shown);
      expect(screen.getByText(PERSONA_EDITOR_COPY.fields.learnFromConversationsHint)).toBeInTheDocument();
      // The field's name is the Core's, never the reader's.
      expect(screen.queryByText(/learn_from_conversations/)).not.toBeInTheDocument();

      fireEvent.click(learnBox);
      fireEvent.click(screen.getByRole('button', { name: PERSONA_EDITOR_COPY.save }));

      // Killed by: frontend/src/components/personas/PersonaEditor.tsx :: onChange={(e) => set('learn_from_conversations', e.target.checked)}
      // Becomes: onChange={(e) => set('enable_subagent_tools', e.target.checked)}
      await waitFor(() =>
        expect(props.onSave).toHaveBeenCalledWith(
          expect.objectContaining({ name: 'builder', learn_from_conversations: !shown }),
          'edit',
        ),
      );
    },
  );

  it('creates a clone that learns from conversations unless the person turns it off', () => {
    renderManager();

    fireEvent.click(screen.getByRole('button', { name: PERSONA_EDITOR_COPY.newPersona }));

    // Killed by: frontend/src/lib/personaDraft.ts :: learn_from_conversations: true,
    // Becomes: learn_from_conversations: false,
    const learnBox = screen.getByRole('checkbox', {
      name: PERSONA_EDITOR_COPY.fields.learnFromConversations,
    }) as HTMLInputElement;
    expect(learnBox.checked).toBe(true);
  });

  it('keeps who a persona may call when the editor saves it', async () => {
    // The editor has no field for `a2a_peers` (#1558), so a draft rebuilt from the fields it
    // shows would save the writer's persona without its Artist and the call would be refused.
    const props = renderManager({
      personas: [
        makePersonaInfo({
          name: 'writer',
          role: 'Writer',
          system_prompt: 'You write.',
          a2a_peers: ['artist'],
        }),
      ],
    });

    fireEvent.click(screen.getByRole('button', { name: `${PERSONA_EDITOR_COPY.edit} writer` }));
    fireEvent.click(screen.getByRole('button', { name: PERSONA_EDITOR_COPY.save }));

    // Killed by: frontend/src/lib/personaDraft.ts :: a2a_peers: [...(persona.a2a_peers ?? [])],
    // Becomes: a2a_peers: [],
    await waitFor(() =>
      expect(props.onSave).toHaveBeenCalledWith(
        expect.objectContaining({ name: 'writer', a2a_peers: ['artist'] }),
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
      fmt(PERSONA_EDITOR_COPY.loadFailedBecause, { reason: 'The clones folder is missing.' }),
    );
  });
});
