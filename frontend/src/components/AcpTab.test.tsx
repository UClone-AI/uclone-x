import { describe, expect, it, vi } from 'vitest';
import { render, screen } from '@testing-library/react';
import { AcpTab } from './AcpTab';
import { AcpStatusData } from '../types';

const notServing: AcpStatusData = {
  transport: 'stdio',
  sdk_version_specified: '0.12.1',
  serving: false,
  presence: {
    shell_module_present: false,
    sdk_installed: false,
    transport: 'stdio',
    sdk_version_specified: '0.12.1',
    reason:
      'No ACP shell is installed: uclone_x.acp.server does not exist and the ' +
      'agent-client-protocol SDK (0.12.1) is not a dependency. This is #649.',
  },
  methods: [
    {
      name: 'prompt',
      side: 'agent',
      status: 'not_implemented',
      counterpart: 'BaseAgent turn execution',
      note: 'The core mapping.',
      spec_section: null,
    },
    {
      name: 'authenticate',
      side: 'agent',
      status: 'out_of_scope',
      counterpart: null,
      note: 'Out of scope for the local stdio head.',
      spec_section: '3.5',
    },
    {
      name: 'session_update',
      side: 'client',
      status: 'not_implemented',
      counterpart: 'AgentEvent to ACP update translation',
      note: 'Blocked on #566.',
      spec_section: '4',
    },
  ],
  mcp_descriptors: [
    {
      name: 'AcpMcpServer',
      status: 'not_implementable',
      note: 'Carries a serverId and no address.',
    },
  ],
  mcp_loader_warning: 'Client-supplied MCP descriptors must not be parsed by MCPConfigFileLoader.',
  counts: {
    agent: {
      total: 2,
      implemented: 0,
      not_implemented: 1,
      not_implementable: 0,
      out_of_scope: 1,
    },
    client: {
      total: 1,
      implemented: 0,
      not_implemented: 1,
      not_implementable: 0,
      out_of_scope: 0,
    },
  },
};

describe('AcpTab', () => {
  it('states that nothing is serving ACP, and why', () => {
    render(<AcpTab acpStatus={notServing} onRefresh={vi.fn()} isLoading={false} />);
    expect(screen.getByTestId('acp-serving-state').textContent).toBe('Not serving ACP');
    expect(screen.getByText(/uclone_x.acp.server does not exist/)).toBeDefined();
    expect(screen.getByText(/#649/)).toBeDefined();
  });

  it('does not present absence as an empty session list', () => {
    // P6: "no shell is installed" and "a shell is running that nobody has connected to" are
    // different facts, and a bare empty list renders them identically. This is the assertion
    // that pins the difference.
    render(<AcpTab acpStatus={notServing} onRefresh={vi.fn()} isLoading={false} />);
    expect(screen.getByTestId('acp-sessions-unavailable')).toBeDefined();
    expect(screen.queryByTestId('acp-sessions-live')).toBeNull();
  });

  it('reports the serving case differently once a shell is present', () => {
    const serving: AcpStatusData = {
      ...notServing,
      serving: true,
      presence: {
        ...notServing.presence,
        shell_module_present: true,
        sdk_installed: true,
        reason: 'An ACP shell module and the agent-client-protocol SDK are both present.',
      },
    };
    render(<AcpTab acpStatus={serving} onRefresh={vi.fn()} isLoading={false} />);
    expect(screen.getByTestId('acp-serving-state').textContent).toBe('Serving ACP');
    expect(screen.getByTestId('acp-sessions-live')).toBeDefined();
    expect(screen.queryByTestId('acp-sessions-unavailable')).toBeNull();
  });

  it('separates the two sides of the protocol', () => {
    render(<AcpTab acpStatus={notServing} onRefresh={vi.fn()} isLoading={false} />);
    expect(screen.getByTestId('acp-method-agent-prompt')).toBeDefined();
    expect(screen.getByTestId('acp-method-client-session_update')).toBeDefined();
    expect(screen.getByTestId('acp-agent-counts').textContent).toBe('0/2 implemented');
    expect(screen.getByTestId('acp-client-counts').textContent).toBe('0/1 implemented');
  });

  it('shows a refused descriptor as refused rather than omitting it', () => {
    render(<AcpTab acpStatus={notServing} onRefresh={vi.fn()} isLoading={false} />);
    const descriptor = screen.getByTestId('acp-descriptor-AcpMcpServer');
    expect(descriptor.textContent).toContain('blocked');
    expect(screen.getByTestId('acp-loader-warning').textContent).toContain('MCPConfigFileLoader');
  });

  it('says the status was never read rather than showing a healthy-looking blank', () => {
    render(<AcpTab acpStatus={null} onRefresh={vi.fn()} isLoading={false} />);
    expect(screen.getByTestId('acp-unavailable')).toBeDefined();
    expect(screen.queryByTestId('acp-panel')).toBeNull();
  });
});
