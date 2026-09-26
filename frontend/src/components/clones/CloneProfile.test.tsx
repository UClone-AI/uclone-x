import React from 'react';
import { describe, it, expect, vi } from 'vitest';
import { render, screen, fireEvent, waitFor } from '@testing-library/react';
import { CloneProfile } from './CloneProfile';
import { makePersonaInfo } from '../../test/fixtures';

/**
 * The dock's Clone surface: who a clone is, and the one thing you can do with it (#1300).
 *
 * Clicking a clone in the rail used to change a highlight and nothing else, and the clone's
 * settings were folded into an expanded rail row 240px wide, where the tool pills wrapped four
 * deep. Both moved here: the rail keeps the choosing, the dock says what was chosen. The
 * configuration fields themselves are `PersonaDetail`'s, tested in
 * `components/layout/PersonaDetail.test.tsx`; what these pin is that this surface reaches them,
 * that it says who it is showing, and that each of its three absences says its own cause.
 */
describe('CloneProfile (#1300)', () => {
  const reader = makePersonaInfo();

  // Killed by: frontend/src/components/clones/CloneProfile.tsx :: <h2 data-testid="clone-profile-name" className="text-lg font-semibold text-slate-100">{name}</h2>
  // Becomes: <h2 data-testid="clone-profile-name" className="text-lg font-semibold text-slate-100"></h2>
  it('names the clone it is showing, above its picture', () => {
    render(<CloneProfile cloneId="reader" persona={reader} onStartConversation={vi.fn()} />);

    expect(screen.getByTestId('clone-profile-name')).toHaveTextContent('reader');
    // The picture is beside the name, never instead of it: the default is the same drawing
    // for every clone that has no image, so a picture alone identifies nothing.
    expect(screen.getByTestId('clone-profile-avatar')).toBeInTheDocument();
  });

  // Killed by: frontend/src/components/clones/CloneProfile.tsx :: {persona?.role || 'No role set'}
  // Becomes: {persona?.role ?? 'No role set'}
  it('says a clone has no role rather than leaving the line blank', () => {
    render(
      <CloneProfile
        cloneId="reader"
        persona={makePersonaInfo({ role: '' })}
        onStartConversation={vi.fn()}
      />,
    );

    expect(screen.getByTestId('clone-profile-role')).toHaveTextContent('No role set');
  });

  // Killed by: frontend/src/components/clones/CloneProfile.tsx :: onStartConversation(cloneId)
  // Becomes: undefined
  it('starts a conversation with the clone on screen, by its id', () => {
    const onStartConversation = vi.fn();
    render(
      <CloneProfile cloneId="reader" persona={reader} onStartConversation={onStartConversation} />,
    );

    fireEvent.click(screen.getByTestId('clone-profile-start'));

    expect(onStartConversation).toHaveBeenCalledWith('reader');
  });

  // Killed by: frontend/src/components/clones/CloneProfile.tsx ::         <PersonaDetail persona={persona} toolsNeedingConversation={toolsNeedingConversation} />
  // Becomes:         <></>
  it('carries the two permissions a clone is trusted with, as sentences', () => {
    // These moved here from the rail row. They are the fields that are safety boundaries
    // rather than cosmetics, and a bare `false` beside a label is not a sentence a reader can
    // act on -- `PersonaDetail` spells both out.
    render(
      <CloneProfile
        cloneId="writer"
        persona={makePersonaInfo({
          name: 'writer',
          enable_write_tools: true,
          enable_subagent_tools: false,
        })}
        onStartConversation={vi.fn()}
      />,
    );

    expect(screen.getByTestId('persona-write-access-writer')).toHaveTextContent('Can write files');
    expect(screen.getByTestId('persona-subagent-access-writer')).toHaveTextContent(
      'Cannot spawn sub-agents',
    );
  });

  // Killed by: frontend/src/components/clones/CloneProfile.tsx ::           Nothing is installed under this name, so there is no description, model or tool list
  // Becomes:           {''}
  it('says why a running clone with no file behind it shows no settings', () => {
    // The rail builds a row from a live instance too, and an instance can outlive the file it
    // was started from. "This clone has no settings" and "the settings did not load" are the
    // same blank panel otherwise (P6).
    render(<CloneProfile cloneId="ephemeral" onStartConversation={vi.fn()} />);

    expect(screen.getByTestId('clone-profile-no-definition')).toHaveTextContent(
      'Nothing is installed under this name',
    );
  });

  // Killed by: frontend/src/components/clones/CloneProfile.tsx :: if (cloneId === '') {
  // Becomes: if (false) {
  it('sends the reader to the rail when no clone is picked at all', () => {
    render(<CloneProfile cloneId="" onStartConversation={vi.fn()} />);

    expect(screen.getByTestId('clone-profile-none')).toHaveTextContent('Clones');
    expect(screen.queryByTestId('clone-profile-start')).not.toBeInTheDocument();
  });

  it('shows an edit button when a persona is loaded and triggers mode change', () => {
    const onModeChange = vi.fn();
    render(
      <CloneProfile
        cloneId="reader"
        persona={reader}
        onStartConversation={vi.fn()}
        onModeChange={onModeChange}
      />,
    );

    const editBtn = screen.getByTestId('clone-profile-edit');
    expect(editBtn).toBeInTheDocument();
    fireEvent.click(editBtn);
    expect(onModeChange).toHaveBeenCalledWith('edit');
  });

  it('disables the edit button when workspace is not writable', () => {
    render(
      <CloneProfile
        cloneId="reader"
        persona={reader}
        onStartConversation={vi.fn()}
        onModeChange={vi.fn()}
        canWrite={false}
      />,
    );

    expect(screen.getByTestId('clone-profile-edit')).toBeDisabled();
  });

  it('renders PersonaEditor when mode is "edit"', () => {
    render(
      <CloneProfile
        cloneId="reader"
        persona={reader}
        onStartConversation={vi.fn()}
        mode="edit"
        onModeChange={vi.fn()}
      />,
    );

    expect(screen.getByTestId('clone-profile-editor')).toBeInTheDocument();
    expect(screen.getByTestId('persona-editor')).toBeInTheDocument();
  });

  it('renders PersonaEditor when mode is "create", even if cloneId is empty', () => {
    render(
      <CloneProfile
        cloneId=""
        onStartConversation={vi.fn()}
        mode="create"
        onModeChange={vi.fn()}
      />,
    );

    expect(screen.getByTestId('clone-profile-editor')).toBeInTheDocument();
    expect(screen.getByTestId('persona-editor')).toBeInTheDocument();
  });

  it('provides a create button when no clone is picked', () => {
    const onModeChange = vi.fn();
    render(
      <CloneProfile
        cloneId=""
        onStartConversation={vi.fn()}
        onModeChange={onModeChange}
      />,
    );

    const createBtn = screen.getByTestId('clone-profile-create-empty');
    expect(createBtn).toBeInTheDocument();
    fireEvent.click(createBtn);
    expect(onModeChange).toHaveBeenCalledWith('create');
  });

  it('preserves edited instructions when background polling re-renders with a new persona reference', () => {
    const { rerender } = render(
      <CloneProfile
        cloneId="reader"
        persona={reader}
        onStartConversation={vi.fn()}
        mode="edit"
        onModeChange={vi.fn()}
      />,
    );

    const promptInput = screen.getByRole('textbox', { name: /^instructions$/i });
    fireEvent.change(promptInput, { target: { value: 'Custom ongoing instruction edit' } });
    expect(promptInput).toHaveValue('Custom ongoing instruction edit');

    // Simulate 15s background metadata polling re-render with a new object reference
    rerender(
      <CloneProfile
        cloneId="reader"
        persona={{ ...reader }}
        onStartConversation={vi.fn()}
        mode="edit"
        onModeChange={vi.fn()}
      />,
    );

    expect(promptInput).toHaveValue('Custom ongoing instruction edit');
  });

  it('preserves edited fields in create mode across background polling re-renders', () => {
    const { rerender } = render(
      <CloneProfile
        cloneId=""
        persona={reader}
        onStartConversation={vi.fn()}
        mode="create"
        onModeChange={vi.fn()}
      />,
    );

    const promptInput = screen.getByRole('textbox', { name: /^instructions$/i });
    fireEvent.change(promptInput, { target: { value: 'New clone brand-new instructions' } });
    expect(promptInput).toHaveValue('New clone brand-new instructions');

    // Simulate 15s background metadata polling re-render
    rerender(
      <CloneProfile
        cloneId=""
        persona={{ ...reader }}
        onStartConversation={vi.fn()}
        mode="create"
        onModeChange={vi.fn()}
      />,
    );

    expect(promptInput).toHaveValue('New clone brand-new instructions');
  });

  it('re-initializes draft when switching to edit a different clone', () => {
    const cloneA = makePersonaInfo({ name: 'clone-a', system_prompt: 'Prompt A' });
    const cloneB = makePersonaInfo({ name: 'clone-b', system_prompt: 'Prompt B' });

    const { rerender } = render(
      <CloneProfile
        cloneId="clone-a"
        persona={cloneA}
        onStartConversation={vi.fn()}
        mode="edit"
        onModeChange={vi.fn()}
      />,
    );

    expect(screen.getByRole('textbox', { name: /^instructions$/i })).toHaveValue('Prompt A');

    // Switch to editing clone B
    rerender(
      <CloneProfile
        cloneId="clone-b"
        persona={cloneB}
        onStartConversation={vi.fn()}
        mode="edit"
        onModeChange={vi.fn()}
      />,
    );

    expect(screen.getByRole('textbox', { name: /^instructions$/i })).toHaveValue('Prompt B');
  });

  describe('Studio mode, which the host owns', () => {
    const renderStudioEditor = (overrides: Partial<React.ComponentProps<typeof CloneProfile>> = {}) => {
      const onStudioChange = vi.fn();
      const onModeChange = vi.fn();
      render(
        <CloneProfile
          cloneId="reader"
          persona={reader}
          onStartConversation={vi.fn()}
          mode="edit"
          onModeChange={onModeChange}
          studio
          onStudioChange={onStudioChange}
          {...overrides}
        />,
      );
      return { onStudioChange, onModeChange };
    };

    it('asks the host for the workspace rather than drawing a fixed layer of its own', () => {
      // Killed by: frontend/src/components/clones/CloneProfile.tsx :: onToggleStudio={onStudioChange ? () => onStudioChange(!studio) : undefined}
      // Becomes: onToggleStudio={undefined}
      const { onStudioChange } = renderStudioEditor({ studio: false });

      fireEvent.click(screen.getByTestId('toggle-studio-mode'));

      expect(onStudioChange).toHaveBeenCalledWith(true);
      expect(document.querySelectorAll('.fixed')).toHaveLength(0);
      expect(screen.queryByTestId('clone-studio-overlay')).toBeNull();
    });

    it('offers no Studio mode when no host can give it the room', () => {
      renderStudioEditor({ studio: false, onStudioChange: undefined });

      expect(screen.queryByTestId('toggle-studio-mode')).toBeNull();
    });

    it('leaves Studio mode on Cancel', () => {
      // Killed by: frontend/src/components/clones/CloneProfile.tsx :: onStudioChange?.(false); // Cancel leaves Studio mode
      // Becomes:
      const { onStudioChange, onModeChange } = renderStudioEditor();

      fireEvent.click(screen.getByRole('button', { name: 'Cancel' }));

      expect(onStudioChange).toHaveBeenCalledWith(false);
      expect(onModeChange).toHaveBeenCalledWith('view');
    });

    it('leaves Studio mode when a save lands', async () => {
      // Killed by: frontend/src/components/clones/CloneProfile.tsx :: onStudioChange?.(false); // and so does a save that landed
      // Becomes:
      const onSave = vi.fn().mockResolvedValue({ ok: true, persona: reader, liveAgentsUpdated: 0 });
      // A prompt, because a draft without one cannot be saved at all.
      const { onStudioChange } = renderStudioEditor({
        onSave,
        persona: makePersonaInfo({ system_prompt: 'You read.' }),
      });

      fireEvent.click(screen.getByRole('button', { name: 'Save' }));

      await waitFor(() => expect(onStudioChange).toHaveBeenCalledWith(false));
    });

    // A host that owns the flag the way `App` does, so leaving Studio mode re-renders the
    // editor for real and a lost draft would show up in the field, not only in a spy.
    const StudioHost: React.FC<{ onModeChange: (mode: 'view' | 'edit' | 'create') => void }> = ({
      onModeChange,
    }) => {
      const [studio, setStudio] = React.useState(true);
      return (
        <CloneProfile
          cloneId="reader"
          persona={reader}
          onStartConversation={vi.fn()}
          mode="edit"
          onModeChange={onModeChange}
          studio={studio}
          onStudioChange={setStudio}
        />
      );
    };

    it('leaves Studio mode on Escape and keeps what was typed (#1377)', () => {
      // Killed by: frontend/src/components/clones/CloneProfile.tsx :: useEscapeOwner('studio', studio && (isCreate || isEdit) && onStudioChange !== undefined, (event) => {
      // Becomes: useEscapeOwner('studio', false, (event) => {
      const onModeChange = vi.fn();
      render(<StudioHost onModeChange={onModeChange} />);
      const role = document.getElementById('persona-role') as HTMLInputElement;
      fireEvent.change(role, { target: { value: 'Night reader' } });

      fireEvent.keyDown(role, { key: 'Escape' });

      expect(screen.getByTestId('clone-profile-editor')).toHaveAttribute('data-studio', 'false');
      // Leaving Studio mode is not leaving the editor: the edit is still there to save.
      expect(document.getElementById('persona-role')).toHaveValue('Night reader');
      expect(onModeChange).not.toHaveBeenCalled();
    });

    it('leaves an Escape pressed on a dropdown to the dropdown (#1377)', () => {
      // Killed by: frontend/src/components/clones/CloneProfile.tsx :: if (event.target instanceof HTMLSelectElement) return;
      // Becomes: if (false) return;
      render(<StudioHost onModeChange={vi.fn()} />);

      fireEvent.keyDown(document.getElementById('persona-model') as HTMLSelectElement, {
        key: 'Escape',
      });

      expect(screen.getByTestId('clone-profile-editor')).toHaveAttribute('data-studio', 'true');
    });

    it('leaves an Escape that cancels a composition to the input method (#1377)', () => {
      // Killed by: frontend/src/components/clones/CloneProfile.tsx :: if (event.isComposing) return;
      // Becomes: if (false) return;
      render(<StudioHost onModeChange={vi.fn()} />);

      fireEvent.keyDown(document.getElementById('persona-role') as HTMLInputElement, {
        key: 'Escape',
        isComposing: true,
      });

      expect(screen.getByTestId('clone-profile-editor')).toHaveAttribute('data-studio', 'true');
    });
  });
});
