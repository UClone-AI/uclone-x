import { describe, expect, it } from 'vitest';
import { render, screen } from '@testing-library/react';
import {
  PersonaDetail,
  NO_TOOLS_CAUSE,
  ONLY_IN_CONVERSATION,
  describeWriteAccess,
  describeSubagentAccess,
} from './PersonaDetail';
import { makePersonaInfo } from '../../test/fixtures';

/**
 * Issue #1056, scope (a): `GET /api/personas` has always returned nine fields per persona,
 * and the head rendered two of them (`name`, `role`) everywhere it showed a persona. This
 * pins that the other seven -- description, model, temperature, max_tokens, allowed_tools,
 * and the two permission flags -- are actually on screen once a detail card is open, and
 * that the two safety-relevant flags read as sentences rather than as `true`/`false`.
 */
describe('PersonaDetail', () => {
  it('renders description, model, temperature and max_tokens', () => {
    render(
      <PersonaDetail
        persona={makePersonaInfo({
          description: 'Reads documents without writing to disk.',
          model_name: 'hermes3:8b',
          temperature: 0.4,
          max_tokens: 4096,
        })}
      />,
    );

    expect(screen.getByText('Reads documents without writing to disk.')).toBeInTheDocument();
    expect(screen.getByText('hermes3:8b')).toBeInTheDocument();
    expect(screen.getByText('0.4')).toBeInTheDocument();
    expect(screen.getByText('4096')).toBeInTheDocument();
  });

  it('lists every allowed tool by name', () => {
    render(
      <PersonaDetail
        persona={makePersonaInfo({ allowed_tools: ['read_file', 'web_search', 'list_directory'] })}
      />,
    );

    const list = screen.getByTestId('persona-tools-reader');
    expect(list).toHaveTextContent('read_file');
    expect(list).toHaveTextContent('web_search');
    expect(list).toHaveTextContent('list_directory');
  });

  it('states "no tools" in words rather than rendering an empty box (P6)', () => {
    // Killed by: frontend/src/ui-kit/rail/PersonaDetail.tsx :: persona.allowed_tools && persona.allowed_tools.length > 0
    // Becomes: true
    render(<PersonaDetail persona={makePersonaInfo({ allowed_tools: [] })} />);

    expect(screen.getByTestId('persona-tools-empty-cause-reader')).toHaveTextContent(NO_TOOLS_CAUSE);
    expect(screen.queryByTestId('persona-tools-reader')).not.toBeInTheDocument();
  });

  it('says in words which tools the clone is given only inside a conversation (#1595)', () => {
    // Killed by: frontend/src/ui-kit/rail/PersonaDetail.tsx :: toolsNeedingConversation.length > 0 ?
    // Becomes: false ?
    render(
      <PersonaDetail
        persona={makePersonaInfo({ allowed_tools: [] })}
        toolsNeedingConversation={['story_outline', 'story_codex']}
      />,
    );

    const note = screen.getByTestId('persona-tools-conversation-only-reader');
    expect(note).toHaveTextContent(`${ONLY_IN_CONVERSATION} story_outline, story_codex`);
    expect(note).not.toHaveTextContent(/room|needs_room|capabilities/);
  });

  it('adds no conversation-only note when the clone has no such tool', () => {
    render(<PersonaDetail persona={makePersonaInfo({ allowed_tools: ['read_file'] })} toolsNeedingConversation={[]} />);
    expect(screen.queryByTestId('persona-tools-conversation-only-reader')).not.toBeInTheDocument();
  });

  it('reads the write-tools flag as a capability sentence, never a bare boolean', () => {
    // Killed by: frontend/src/components/layout/PersonaDetail.tsx :: enabled ? 'Can write files' : 'Cannot write files'
    // Becomes: 'Can write files'
    expect(describeWriteAccess(true)).toBe('Can write files');
    expect(describeWriteAccess(false)).toBe('Cannot write files');

    render(<PersonaDetail persona={makePersonaInfo({ enable_write_tools: true })} />);
    expect(screen.getByTestId('persona-write-access-reader')).toHaveTextContent('Can write files');
    expect(screen.queryByText('true')).not.toBeInTheDocument();
    expect(screen.queryByText('enable_write_tools')).not.toBeInTheDocument();
  });

  it('reads the subagent-tools flag as a capability sentence, never a bare boolean', () => {
    // Killed by: frontend/src/components/layout/PersonaDetail.tsx :: enabled ? 'Can spawn sub-agents' : 'Cannot spawn sub-agents'
    // Becomes: 'Can spawn sub-agents'
    expect(describeSubagentAccess(false)).toBe('Cannot spawn sub-agents');
    expect(describeSubagentAccess(true)).toBe('Can spawn sub-agents');

    render(<PersonaDetail persona={makePersonaInfo({ enable_subagent_tools: false })} />);
    expect(screen.getByTestId('persona-subagent-access-reader')).toHaveTextContent(
      'Cannot spawn sub-agents',
    );
  });

  it('states "not set" for a model or temperature the persona did not specify', () => {
    // Killed by: frontend/src/ui-kit/rail/PersonaDetail.tsx :: persona.model_name || copy.notSet
    // Becomes: persona.model_name
    render(
      <PersonaDetail
        persona={makePersonaInfo({ model_name: undefined, temperature: undefined, max_tokens: undefined })}
      />,
    );

    expect(screen.getAllByText('not set')).toHaveLength(3);
  });
});
