/**
 * Shared test fixtures.
 *
 * These build *complete* objects and take a `Partial` override, rather than casting a
 * two-field literal with `as AgentInfo`. The cast compiles until the interface grows a
 * required field, at which point every test that used it is silently exercising a shape
 * the app never sees — `tsc` rejected exactly that when these tests were first written.
 */
import type { AgentInfo, ChatMessage, PersonaInfo, SessionSummary } from '../types';

export const makeAgentInfo = (over: Partial<AgentInfo> = {}): AgentInfo => ({
  id: 'agent-1',
  label: 'agent-1',
  role: 'orchestrator',
  // `AgentState.IDLE`, as `/api/agents` sends it -- `list_agents` puts `ag.state.value` in
  // this field, and that enum (`src/uclone_x/agent/models.py`) has no `healthy`. The default
  // used to be exactly that: a value the endpoint has never sent, which is the shape this
  // file's own docstring exists to prevent.
  status: 'IDLE',
  tier: 'champion',
  isolation_level: 'workspace',
  capabilities: [],
  uptime_s: 0,
  current_task: '',
  parent_id: null,
  subagents: [],
  ...over,
});

/**
 * A complete `/api/personas` entry (`src/uclone_x/ui/app.py` `list_personas`), all nine
 * fields the backend returns, so a test overriding one field never accidentally exercises
 * a shape the endpoint does not send.
 */
export const makePersonaInfo = (over: Partial<PersonaInfo> = {}): PersonaInfo => ({
  name: 'reader',
  role: 'Research Reader',
  description: 'Reads and summarizes documents without modifying the workspace.',
  allowed_tools: ['read_file', 'web_search'],
  model_name: 'qwen2.5:7b',
  temperature: 0.7,
  max_tokens: 2048,
  enable_write_tools: false,
  enable_subagent_tools: false,
  ...over,
});

export const makeChatMessage = (over: Partial<ChatMessage> = {}): ChatMessage => ({
  id: 'm1',
  sender: 'agent',
  content: '',
  timestamp: '00:00:00',
  ...over,
});

export const makeSessionSummary = (over: Partial<SessionSummary> = {}): SessionSummary => ({
  session_id: 'sess_1',
  ...over,
});
