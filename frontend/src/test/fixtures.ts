/**
 * Shared test fixtures.
 *
 * These build *complete* objects and take a `Partial` override, rather than casting a
 * two-field literal with `as PersonaInfo`. The cast compiles until the interface grows a
 * required field, at which point every test that used it is silently exercising a shape
 * the app never sees — `tsc` rejected exactly that when these tests were first written.
 */
import type { ChatMessage, CloneChoice, PersonaInfo, SessionSummary } from '../types';

export const makeCloneChoice = (over: Partial<CloneChoice> = {}): CloneChoice => ({
  id: 'agent-1',
  label: 'agent-1',
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
