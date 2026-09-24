import { describe, it, expect, vi, beforeEach, afterEach } from 'vitest';
import { fireEvent, render, screen, waitFor, within } from '@testing-library/react';
import { McpServersSection } from './McpServersSection';
import type { McpServer } from '../../types';
import { expectPlain } from '../../test/plainCopy';

type RouteResponse = { ok: boolean; status: number; json: () => Promise<unknown> };
type Handler = (url: string, init?: RequestInit) => RouteResponse | Promise<RouteResponse>;

const jsonResponse = (body: unknown, ok = true, status = 200): RouteResponse => ({
  ok,
  status,
  json: async () => body,
});

const CONFIG_PATH = '/home/u/.uclone/mcp_servers.json';

const server = (overrides: Partial<McpServer> = {}): McpServer => ({
  name: 'github',
  transport: 'http',
  url: 'https://api.example.com/mcp',
  command: null,
  args: [],
  env_keys: [],
  header_keys: [],
  enabled: true,
  status: 'connected',
  error: null,
  tools: [],
  ...overrides,
});

/**
 * Routes every request by method and path. `list` answers GET /api/mcp/servers; `routes` is
 * keyed "METHOD /path" and matched exactly on the path, so a wrong URL answers 599 and fails.
 */
const mockFetch = (list: McpServer[], routes: Record<string, Handler> = {}) => {
  const calls: Array<{ url: string; method: string; body?: unknown }> = [];
  const fetchMock = vi.fn(async (url: string, init?: RequestInit) => {
    const method = init?.method ?? 'GET';
    const body = init?.body ? JSON.parse(init.body as string) : undefined;
    calls.push({ url, method, body });
    const handler = routes[`${method} ${url}`];
    if (handler) return handler(url, init);
    if (method === 'GET' && url === '/api/mcp/servers') {
      return jsonResponse({ config_path: CONFIG_PATH, servers: list });
    }
    return jsonResponse({ detail: `unrouted ${method} ${url}` }, false, 599);
  });
  vi.stubGlobal('fetch', fetchMock);
  return calls;
};

const posts = (calls: ReturnType<typeof mockFetch>, url: string, method = 'POST') =>
  calls.filter((c) => c.method === method && c.url === url);

/** Every value a user could see: rendered text plus the value of each form control. */
const everythingOnScreen = (container: HTMLElement) =>
  [
    container.textContent ?? '',
    ...Array.from(container.querySelectorAll('input, textarea')).map(
      (el) => (el as HTMLInputElement).value,
    ),
  ].join('\n');

describe('McpServersSection', () => {
  beforeEach(() => vi.restoreAllMocks());
  afterEach(() => vi.unstubAllGlobals());

  it('says in words that no server is connected and how to add one', async () => {
    mockFetch([]);
    render(<McpServersSection />);

    const empty = await screen.findByTestId('settings-mcp-empty');
    expect(empty).toHaveTextContent('No tool servers are connected yet.');
    expect(empty).toHaveTextContent('Add a server');
    expect(empty).toHaveTextContent('Import from JSON');
    expect(screen.queryByRole('list', { name: 'Tool servers' })).toBeNull();
    expect(screen.getByTestId('settings-mcp-config-path')).toHaveTextContent(`Saved in ${CONFIG_PATH}`);
  });

  it("states why the list could not be read, in the Core's own words, rather than showing it empty", async () => {
    // Killed by: frontend/src/lib/mcpServersApi.ts :: if (!res.ok) throw await failureOf(res);
    // Becomes: if (!res.ok) throw new Error(`HTTP ${res.status}`);
    vi.spyOn(console, 'error').mockImplementation(() => {});
    vi.stubGlobal(
      'fetch',
      vi.fn(async () => jsonResponse({ detail: 'The tool server list is not readable' }, false, 500)),
    );
    render(<McpServersSection />);

    const failure = await screen.findByTestId('settings-mcp-error');
    expect(failure).toHaveTextContent('could not be read');
    expect(failure).toHaveTextContent('The tool server list is not readable.');
    expect(screen.queryByTestId('settings-mcp-empty')).toBeNull();
  });

  it.each<[string, () => Promise<RouteResponse>]>([
    ['a rejected fetch', () => Promise.reject(new TypeError('Failed to fetch'))],
    [
      'a bodyless 500',
      async () => ({
        ok: false,
        status: 500,
        json: async () => {
          throw new SyntaxError('Unexpected token \'I\', "Internal S"... is not valid JSON');
        },
      }),
    ],
  ])('says in plain words that the list could not be read after %s (#1436)', async (_label, fault) => {
    // Killed by: frontend/src/components/settings/McpServersSection.tsx :: setLoadError({ reason: coreReason(err) });
    // Becomes: setLoadError({ reason: String(err) });
    vi.spyOn(console, 'error').mockImplementation(() => {});
    vi.stubGlobal('fetch', vi.fn(fault));
    render(<McpServersSection />);

    const failure = await screen.findByTestId('settings-mcp-error');
    expect(failure).toHaveTextContent('The list of tool servers could not be read.');
    expectPlain(failure.textContent);
    expect(screen.getByRole('button', { name: 'Try again' })).toBeTruthy();
  });

  it('shows a connected server with its tool count, kind, address and tools', async () => {
    mockFetch([
      server({
        tools: [
          { name: 'search_issues', description: 'Find issues by text' },
          { name: 'get_repo', description: 'Read a repository' },
        ],
      }),
    ]);
    render(<McpServersSection />);

    const row = await screen.findByTestId('mcp-server-github');
    expect(row).toHaveTextContent('Connected · 2 tools');
    expect(row).toHaveTextContent('Remote');
    expect(row).toHaveTextContent('https://api.example.com/mcp');
    const tools = within(row).getByRole('list', { name: 'Tools from github' });
    expect(Array.from(tools.querySelectorAll('li')).map((li) => li.textContent)).toEqual([
      'search_issues — Find issues by text',
      'get_repo — Read a repository',
    ]);
  });

  it("shows an error row with the server's error text verbatim", async () => {
    mockFetch([
      server({
        name: 'files',
        transport: 'stdio',
        url: null,
        command: 'npx',
        args: ['-y', '@mcp/server-files', '/Users/me/My Documents'],
        status: 'error',
        error: 'spawn npx ENOENT',
      }),
    ]);
    render(<McpServersSection />);

    const row = await screen.findByTestId('mcp-server-files');
    expect(row).toHaveTextContent('Error');
    expect(row).toHaveTextContent('On this computer');
    expect(row).toHaveTextContent('npx -y @mcp/server-files "/Users/me/My Documents"');
    expect(within(row).getByTestId('mcp-server-error')).toHaveTextContent('spawn npx ENOENT');
    expect(row).not.toHaveTextContent('Connected');
  });

  it('posts a remote server with its headers, then shows it without the header value', async () => {
    const secret = 'Bearer s3cr3t-token';
    const calls = mockFetch([], {
      'POST /api/mcp/servers': () =>
        jsonResponse(
          server({ name: 'linear', url: 'https://mcp.linear.app/sse', header_keys: ['Authorization'], status: 'error', error: '401 Unauthorized' }),
          true,
          201,
        ),
    });
    const { container } = render(<McpServersSection />);

    fireEvent.click(await screen.findByRole('button', { name: /add a server/i }));
    fireEvent.change(screen.getByLabelText('Name'), { target: { value: 'linear' } });
    fireEvent.change(screen.getByLabelText(/server address/i), { target: { value: ' https://mcp.linear.app/sse ' } });
    fireEvent.click(screen.getByRole('button', { name: /add header/i }));
    fireEvent.change(screen.getByLabelText('Headers name 1'), { target: { value: 'Authorization' } });
    const valueInput = screen.getByLabelText('Headers value 1');
    expect(valueInput).toHaveAttribute('type', 'password');
    fireEvent.change(valueInput, { target: { value: secret } });
    fireEvent.click(screen.getByRole('button', { name: 'Add server' }));

    const row = await screen.findByTestId('mcp-server-linear');
    expect(posts(calls, '/api/mcp/servers').map((c) => c.body)).toEqual([
      {
        name: 'linear',
        transport: 'http',
        url: 'https://mcp.linear.app/sse',
        headers: { Authorization: secret },
      },
    ]);
    // Saved even though it did not connect, and the row says why.
    expect(within(row).getByTestId('mcp-server-error')).toHaveTextContent('401 Unauthorized');
    expect(row).toHaveTextContent('Authorization');
    expect(everythingOnScreen(container)).not.toContain('s3cr3t');
  });

  it('posts a command server with its arguments and environment, and warns before it runs', async () => {
    const calls = mockFetch([], {
      'POST /api/mcp/servers': () =>
        jsonResponse(
          server({
            name: 'files',
            transport: 'stdio',
            url: null,
            command: 'npx',
            args: ['-y', 'server-files'],
            env_keys: ['API_KEY'],
          }),
          true,
          201,
        ),
    });
    const { container } = render(<McpServersSection />);

    fireEvent.click(await screen.findByRole('button', { name: /add a server/i }));
    expect(screen.queryByTestId('mcp-command-warning')).toBeNull();
    fireEvent.click(screen.getByLabelText('On this computer (command)'));
    expect(screen.getByTestId('mcp-command-warning')).toHaveTextContent(
      'This runs a program on this computer with your permissions. Only add servers you trust.',
    );
    fireEvent.change(screen.getByLabelText('Name'), { target: { value: 'files' } });
    fireEvent.change(screen.getByLabelText('Command'), { target: { value: 'npx' } });
    fireEvent.change(screen.getByLabelText(/arguments/i), { target: { value: '-y\n server-files \n\n' } });
    fireEvent.click(screen.getByRole('button', { name: /add variable/i }));
    fireEvent.change(screen.getByLabelText('Environment variables name 1'), { target: { value: 'API_KEY' } });
    const valueInput = screen.getByLabelText('Environment variables value 1');
    expect(valueInput).toHaveAttribute('type', 'password');
    fireEvent.change(valueInput, { target: { value: 'env-secret-42' } });
    fireEvent.click(screen.getByRole('button', { name: 'Add server' }));

    const row = await screen.findByTestId('mcp-server-files');
    expect(posts(calls, '/api/mcp/servers').map((c) => c.body)).toEqual([
      { name: 'files', transport: 'stdio', command: 'npx', args: ['-y', 'server-files'], env: { API_KEY: 'env-secret-42' } },
    ]);
    expect(row).toHaveTextContent('Connected · 0 tools');
    expect(everythingOnScreen(container)).not.toContain('env-secret-42');
  });

  it('refuses a name the server would refuse, without sending it', async () => {
    const calls = mockFetch([]);
    render(<McpServersSection />);

    fireEvent.click(await screen.findByRole('button', { name: /add a server/i }));
    fireEvent.change(screen.getByLabelText('Name'), { target: { value: 'my server' } });
    fireEvent.change(screen.getByLabelText(/server address/i), { target: { value: 'https://x' } });
    fireEvent.click(screen.getByRole('button', { name: 'Add server' }));

    expect(await screen.findByTestId('mcp-add-error')).toHaveTextContent('The name is not usable.');
    expect(posts(calls, '/api/mcp/servers')).toEqual([]);
  });

  it("shows the server's reason when it refuses a duplicate name", async () => {
    const detail = "A server named 'github' already exists.";
    mockFetch([server()], {
      'POST /api/mcp/servers': () => jsonResponse({ detail }, false, 409),
    });
    render(<McpServersSection />);

    fireEvent.click(await screen.findByRole('button', { name: /add a server/i }));
    fireEvent.change(screen.getByLabelText('Name'), { target: { value: 'github' } });
    fireEvent.change(screen.getByLabelText(/server address/i), { target: { value: 'https://x' } });
    fireEvent.click(screen.getByRole('button', { name: 'Add server' }));

    expect(await screen.findByTestId('mcp-add-error')).toHaveTextContent(detail);
    expect(screen.getByTestId('mcp-add-form')).toBeInTheDocument();
  });

  it('removes a server only after the confirm step, by its encoded name', async () => {
    const calls = mockFetch([server({ name: 'my_srv-1' })], {
      'DELETE /api/mcp/servers/my_srv-1': () => jsonResponse({ status: 'ok' }),
    });
    render(<McpServersSection />);

    fireEvent.click(await screen.findByRole('button', { name: 'Remove my_srv-1' }));
    expect(posts(calls, '/api/mcp/servers/my_srv-1', 'DELETE')).toEqual([]);
    const confirm = screen.getByTestId('mcp-remove-confirm');
    fireEvent.click(within(confirm).getByRole('button', { name: 'Remove' }));

    await waitFor(() => expect(screen.queryByTestId('mcp-server-my_srv-1')).toBeNull());
    expect(posts(calls, '/api/mcp/servers/my_srv-1', 'DELETE')).toHaveLength(1);
    expect(screen.getByTestId('settings-mcp-empty')).toBeInTheDocument();
  });

  it('keeps the server when the confirm step is declined', async () => {
    const calls = mockFetch([server()]);
    render(<McpServersSection />);

    fireEvent.click(await screen.findByRole('button', { name: 'Remove github' }));
    fireEvent.click(screen.getByRole('button', { name: 'Keep' }));

    expect(screen.queryByTestId('mcp-remove-confirm')).toBeNull();
    expect(screen.getByTestId('mcp-server-github')).toBeInTheDocument();
    expect(calls.filter((c) => c.method === 'DELETE')).toEqual([]);
  });

  it('turns a server off and shows it disabled', async () => {
    const calls = mockFetch([server()], {
      'POST /api/mcp/servers/github/enabled': () =>
        jsonResponse(server({ enabled: false, status: 'disabled' })),
    });
    render(<McpServersSection />);

    const toggle = await screen.findByRole('switch', { name: 'Use github' });
    expect(toggle).toHaveAttribute('aria-checked', 'true');
    fireEvent.click(toggle);

    await waitFor(() => expect(screen.getByTestId('mcp-server-github')).toHaveTextContent('Disabled'));
    expect(posts(calls, '/api/mcp/servers/github/enabled').map((c) => c.body)).toEqual([{ enabled: false }]);
    expect(screen.getByRole('switch', { name: 'Use github' })).toHaveAttribute('aria-checked', 'false');
  });

  it('reconnects a server and shows its new state', async () => {
    const calls = mockFetch([server({ status: 'error', error: 'timed out' })], {
      'POST /api/mcp/servers/github/reconnect': () =>
        jsonResponse(server({ tools: [{ name: 'a', description: '' }] })),
    });
    render(<McpServersSection />);

    fireEvent.click(await screen.findByRole('button', { name: 'Reconnect github' }));

    const row = screen.getByTestId('mcp-server-github');
    // Exact text: a substring check would also accept "1 tools".
    expect(await within(row).findByText('Connected · 1 tool')).toBeInTheDocument();
    expect(posts(calls, '/api/mcp/servers/github/reconnect')).toHaveLength(1);
    expect(screen.queryByTestId('mcp-server-error')).toBeNull();
  });

  it('imports a pasted snippet and names each skipped entry with its reason', async () => {
    const snippet = '{"mcpServers": {"fs": {"command": "npx"}, "bad name": {}}}';
    const calls = mockFetch([], {
      'POST /api/mcp/servers/import': () =>
        jsonResponse({
          added: [server({ name: 'fs', transport: 'stdio', url: null, command: 'npx' })],
          skipped: [{ name: 'bad name', reason: 'names may use only letters, digits, - and _' }],
        }),
    });
    render(<McpServersSection />);

    fireEvent.click(await screen.findByRole('button', { name: /import from json/i }));
    fireEvent.change(screen.getByLabelText(/paste a server list/i), { target: { value: snippet } });
    fireEvent.click(screen.getByRole('button', { name: 'Import' }));

    const outcome = await screen.findByTestId('mcp-import-outcome');
    expect(outcome).toHaveTextContent('Added: fs');
    const skipped = within(outcome).getByRole('list', { name: 'Skipped servers' });
    expect(Array.from(skipped.querySelectorAll('li')).map((li) => li.textContent)).toEqual([
      'Skipped bad name: names may use only letters, digits, - and _',
    ]);
    expect(posts(calls, '/api/mcp/servers/import').map((c) => c.body)).toEqual([{ json: snippet }]);
    expect(screen.getByTestId('mcp-server-fs')).toBeInTheDocument();
  });

  it("shows the server's reason when the pasted JSON cannot be read", async () => {
    const detail = 'Expecting value: line 1 column 1 (char 0)';
    mockFetch([], {
      'POST /api/mcp/servers/import': () => jsonResponse({ detail }, false, 400),
    });
    render(<McpServersSection />);

    fireEvent.click(await screen.findByRole('button', { name: /import from json/i }));
    fireEvent.change(screen.getByLabelText(/paste a server list/i), { target: { value: 'nope' } });
    fireEvent.click(screen.getByRole('button', { name: 'Import' }));

    expect(await screen.findByTestId('mcp-import-error')).toHaveTextContent(detail);
  });

  it('shows a connecting server as in progress, then connected once a later read says so', async () => {
    let reads = 0;
    mockFetch([], {
      'GET /api/mcp/servers': () => {
        reads += 1;
        const status = reads === 1 ? 'connecting' : 'connected';
        const tools = reads === 1 ? [] : [{ name: 'ping', description: 'Check it answers' }];
        return jsonResponse({ config_path: CONFIG_PATH, servers: [server({ status, tools })] });
      },
    });
    render(<McpServersSection pollMs={10} />);

    const row = await screen.findByTestId('mcp-server-github');
    expect(within(row).getByText('Connecting…')).toBeInTheDocument();
    expect(within(row).queryByTestId('mcp-server-error')).toBeNull();
    expect(within(row).queryByText(/show tools/i)).toBeNull();

    expect(await within(row).findByText('Connected · 1 tool')).toBeInTheDocument();
    expect(within(row).queryByText('Connecting…')).toBeNull();
  });

  it('encodes a name in the request path', async () => {
    // Names the server accepts never need encoding; a legacy entry with a space still must not
    // turn into a different path.
    const calls = mockFetch([server({ name: 'odd name' })], {
      'POST /api/mcp/servers/odd%20name/reconnect': () => jsonResponse(server({ name: 'odd name' })),
    });
    render(<McpServersSection />);

    fireEvent.click(await screen.findByRole('button', { name: 'Reconnect odd name' }));
    await waitFor(() => expect(posts(calls, '/api/mcp/servers/odd%20name/reconnect')).toHaveLength(1));
  });
});
