/**
 * The requests behind Settings' external tool servers section. Every call goes to the same
 * origin as the other settings calls, with plain `fetch`, as `personasApi.ts` does.
 *
 * Each mutation resolves to `{ ok: false, message }` rather than throwing, so a refusal is
 * shown in the server's own words (its `detail`) next to the control that caused it.
 */
import type { McpImportResult, McpServer, McpServerDraft, McpServerList } from '../types';
import { failureOf } from './coreFailure';

export type McpResult<T> = { ok: true; value: T } | { ok: false; message: string };

/** 1-32 letters, digits, `-` or `_` -- the server's own rule, checked here first. */
export const MCP_NAME_PATTERN = /^[A-Za-z0-9_-]{1,32}$/;

/** The server's `detail`, in words: a string as is, FastAPI's validation list joined. */
export const describeDetail = (detail: unknown, status: number): string => {
  if (typeof detail === 'string' && detail.trim()) return detail;
  if (Array.isArray(detail)) {
    const parts = detail
      .map((d) => (d && typeof d === 'object' && 'msg' in d ? String((d as { msg: unknown }).msg) : ''))
      .filter(Boolean);
    if (parts.length) return parts.join('; ');
  }
  return `The server refused the request (HTTP ${status}).`;
};

const send = async <T>(url: string, method: string, body?: unknown): Promise<McpResult<T>> => {
  let res: Response;
  try {
    res = await fetch(url, {
      method,
      headers: body === undefined ? undefined : { 'Content-Type': 'application/json' },
      body: body === undefined ? undefined : JSON.stringify(body),
    });
  } catch (err) {
    return { ok: false, message: `The app could not be reached: ${String(err)}` };
  }
  const data = (await res.json().catch(() => ({}))) as unknown;
  if (!res.ok) {
    const detail = data && typeof data === 'object' ? (data as { detail?: unknown }).detail : undefined;
    return { ok: false, message: describeDetail(detail, res.status) };
  }
  return { ok: true, value: data as T };
};

const serverPath = (name: string) => `/api/mcp/servers/${encodeURIComponent(name)}`;

export const fetchMcpServers = async (): Promise<McpServerList> => {
  const res = await fetch('/api/mcp/servers');
  if (!res.ok) throw await failureOf(res);
  const data = (await res.json()) as Partial<McpServerList>;
  return { config_path: data.config_path ?? '', servers: data.servers ?? [] };
};

export const addMcpServer = (draft: McpServerDraft) =>
  send<McpServer>('/api/mcp/servers', 'POST', draft);

export const removeMcpServer = (name: string) =>
  send<{ status: string }>(serverPath(name), 'DELETE');

export const reconnectMcpServer = (name: string) =>
  send<McpServer>(`${serverPath(name)}/reconnect`, 'POST');

export const setMcpServerEnabled = (name: string, enabled: boolean) =>
  send<McpServer>(`${serverPath(name)}/enabled`, 'POST', { enabled });

export const importMcpServers = (json: string) =>
  send<McpImportResult>('/api/mcp/servers/import', 'POST', { json });
