/**
 * The surfaces the right-hand workspace dock can show.
 *
 * `chat` is deliberately absent. It was a peer of these in the old `TabType`, which made the
 * conversation one of seven mutually exclusive destinations — so reading the DAG meant leaving
 * the conversation. The conversation is now the centre column and is always present; these are
 * the runtime internals that have no ACP counterpart and therefore stay ours to render
 * (`docs/acp-protocol-spec.md` §2, §4).
 */
export type DockSurface =
  /**
   * One clone's profile: who it is, what it may do, and the control that starts a
   * conversation with it. Reached by picking a clone in the rail (#1300).
   */
  | 'clone'
  /**
   * One turn's provenance: which model served it, which selector chose the speaker and on
   * what confidence, and what the turn had seen when it started. Reached by `why ›` on
   * the turn itself (#1307 follow-up).
   *
   * A surface rather than a disclosure under the row: the dock is where the workspace
   * already keeps what is being looked at, and the row had
   * space for three short lines of a record that has more than three. The transcript is not
   * pushed around to read it, and the detail stays on screen while the reader scrolls the
   * conversation it is about.
   */
  | 'turn'
  | 'artifacts'
  /** What the seated clone remembers, as sentences (#1357). */
  | 'remembers'
  | 'knowledge_graph'
  | 'activity'
  | 'resource'
  | 'topology'
  | 'ledger'
  | 'ontology';
// No 'skills', 'acp' or 'evaluations' (#1358): Skills is a Settings section, and ACP and Evals
// are Settings' Diagnostics section. None is about the conversation on screen, so none is a
// dock surface, and a dock tab naming one is a type error.

export interface ArtifactItem {
  name: string;
  path: string;
  type: string;
  size_bytes?: number;
  author_agent?: string;
  created_at?: string;
  modified_at?: string;
  description?: string;
}

export interface ArtifactsResponse {
  session_id: string;
  artifacts: ArtifactItem[];
  total: number;
}


export type AcpMethodStatus =
  | 'not_implemented'
  | 'implemented'
  | 'not_implementable'
  | 'out_of_scope';

export interface AcpMethod {
  name: string;
  side: 'agent' | 'client';
  status: AcpMethodStatus;
  counterpart: string | null;
  note: string;
  spec_section: string | null;
}

export interface AcpMcpDescriptor {
  name: string;
  status: AcpMethodStatus;
  note: string;
}

export interface AcpSideCounts {
  total: number;
  implemented: number;
  not_implemented: number;
  not_implementable: number;
  out_of_scope: number;
}

export interface AcpPresence {
  shell_module_present: boolean;
  sdk_installed: boolean;
  transport: string;
  sdk_version_specified: string;
  /** Why ACP is or is not being served. Rendered verbatim: an absence must state its cause. */
  reason: string;
}

export interface AcpStatusData {
  transport: string;
  sdk_version_specified: string;
  presence: AcpPresence;
  serving: boolean;
  methods: AcpMethod[];
  mcp_descriptors: AcpMcpDescriptor[];
  mcp_loader_warning: string;
  counts: { agent: AcpSideCounts; client: AcpSideCounts };
}

export interface AgentInfo {
  id: string;
  label: string;
  role: string;
  status: string;
  tier: string;
  isolation_level: string;
  capabilities: string[];
  uptime_s: number;
  current_task: string;
  parent_id: string | null;
  subagents: string[];
  max_steps?: number;
  /** @deprecated alias for max_steps */
  max_turns?: number;
}

export interface PersonaInfo {
  name: string;
  role: string;
  description: string;
  allowed_tools: string[];
  model_name?: string;
  temperature?: number;
  max_tokens?: number;
  enable_write_tools?: boolean;
  enable_subagent_tools?: boolean;
  /** The persona's own prompt, without the appended default prompt (#892). */
  system_prompt?: string;
  /** Whether the runtime's default prompt is appended after `system_prompt`. */
  append_default_prompt?: boolean;
  model_tier?: PersonaModelTier;
  /** Loaded from the package's own directory; an edit writes a workspace override. */
  builtin?: boolean;
  /** A workspace file that replaces a built-in persona of the same name. */
  overrides_builtin?: boolean;
}

export type PersonaModelTier = 'inherit' | 'fast' | 'pro' | 'flash_lite' | 'custom';

/** `GET /api/personas`: the catalogue, plus what an editor needs to offer choices. */
export interface PersonaCatalog {
  personas: PersonaInfo[];
  /** Every tool name a persona may list; the server refuses any other. */
  available_tools: string[];
  /** Where a save is written, or null when the runtime has no workspace. */
  personas_dir: string | null;
}

export interface TopologyNode {
  id: string;
  label: string;
  role: string;
  tier?: string;
  status?: string;
  state?: string;
  isolation_level: string;
  capabilities?: string[];
  allowed_tools?: string[];
  current_task?: string;
  current_turn?: string;
  turn_index?: number;
  max_steps?: number;
  /** @deprecated alias for max_steps */
  max_turns?: number;
  depth?: number;
  uptime_s?: number;
  position: { x: number; y: number };
}

export interface TopologyEdge {
  id: string;
  source: string;
  target: string;
  type: string;
  label: string;
  animated: boolean;
}

export interface TopologyData {
  nodes: TopologyNode[];
  edges: TopologyEdge[];
}

export interface ToolCall {
  id?: string;
  name?: string;
  tool_name?: string;
  arguments: Record<string, unknown>;
  output?: unknown;
  status?: 'success' | 'error' | string;
  error?: string | null;
  duration_ms?: number;
  latency_ms?: number;
  provenance?: {
    component: string;
    producer: string;
    content_hash: string;
    degraded: boolean;
  };
}

export type ToolCallTrace = ToolCall;

export interface ToolExecution {
  tool_call_id?: string;
  tool_name: string;
  arguments: Record<string, unknown>;
  output?: unknown;
  /** `stopped`: the turn was cancelled and nothing recorded how this call ended (#1031). */
  status: 'success' | 'error' | 'stopped' | string;
  error?: string | null;
  /** Absent when nothing timed the call -- a `0` here would be a figure nobody measured. */
  duration_ms?: number;
  provenance?: {
    component: string;
    producer: string;
    content_hash: string;
    degraded: boolean;
  };
}

export interface DebugInfo {
  active_invariants?: string[];
  /** Input tokens the turn booked; `null` when it booked none (#939). */
  prompt_tokens_used?: number | null;
  system_prompt_excerpt?: string;
  [key: string]: unknown;
}

export interface DurabilityInfo {
  persisted: boolean;
  error?: string | null;
  error_type?: string | null;
  stale_conflict?: boolean;
}

export interface MessageProvenance {
  component?: string;
  producer?: string;
  content_hash?: string;
  degraded?: boolean;
  path?: string;
  served_by?: string;
  persona?: string;
}

export interface SessionSummary {
  session_id: string;
  agent_id?: string;
  created_at?: string;
  updated_at?: string;
  turn_counter?: number;
  revision?: number;
  message_count?: number;
  is_saturated?: boolean;
  active_turns?: number;
}

export interface ChatMessage {
  id: string;
  sender: 'user' | 'agent';
  /**
   * `failure`: a turn that failed, saved as such rather than as a reply (#969).
   * `cancelled`: a turn Stop ended before it finished, saved as its own record (#1031) --
   * not a failure and not a reply, and never re-sent to the model.
   */
  role?: 'user' | 'assistant' | 'system' | 'failure' | 'cancelled';
  content: string;
  timestamp: string;
  latencyMs?: number;
  agentId?: string;
  model?: string;
  turnCount?: number;
  runSteps?: number;
  stepBudgetMax?: number;
  stepsRemaining?: number;
  tokensUsed?: number;
  /** Whose count `tokensUsed` is. Absent means nothing says it was counted (#939). */
  tokenCountSource?: 'provider' | 'estimate';
  toolCalls?: ToolCall[];
  toolExecutions?: ToolExecution[];
  debugInfo?: DebugInfo;
  provenance?: MessageProvenance;
  durability?: DurabilityInfo;
  isStopped?: boolean;
  persona?: string;
  isStreaming?: boolean;
  streamingStatus?: string;
  statusDetail?: string;
  /** A record of what a compaction discarded -- not a turn the agent took (#872). */
  compactionLedger?: boolean;
  /** On a prompt: the id the page sent the turn with, kept by the server (#1000). */
  clientTurnId?: string;
  /** How the turn ended, from the server's structured status; names the row's chip (#1007). */
  outcome?: ChatTurnOutcome;
  /** The turn's own error, beside the text the row shows (#969). */
  error?: string;
  /** On a failed turn: why a retry would be refused too. Such a row offers no Retry (#969). */
  refusal?: RoomTurnRefusal;
}

/**
 * `ChatTurnOutcome` (`uclone_x.ui.app`): how a chat turn ended (#1007 item 4). `degraded`
 * is the Core's meaning only: an answer from a model other than the one requested.
 */
export type ChatTurnOutcome = 'completed' | 'degraded' | 'failed' | 'interrupted';

export interface EventEnvelope {
  id: string;
  seq?: number;
  type: string;
  priority?: string;
  event_id?: string;
  event_type?: string;
  source?: string;
  sender_id?: string;
  recipient_id?: string;
  target?: string;
  topic?: string;
  payload?: Record<string, unknown> | string | number | boolean | null;
  timestamp: string | number;
  provenance?: {
    component?: string;
    path?: string;
    producer?: string;
    degraded?: boolean;
    served_by?: string;
    persona?: string;
  };
}

export interface OntologyConcept {
  name: string;
  parent_type: string | null;
  tier: 'asserted' | 'induced_enforcing' | 'induced_candidate';
  precedence: number;
  attributes: Record<string, string>;
  required_fields: string[];
  content_hash: string;
  provenance: {
    source: string;
    origin: string;
    immutable: boolean;
  };
}

export interface OntologyRelation {
  id: string;
  source: string;
  predicate: string;
  target: string;
  is_directed: boolean;
  tier: string;
  confidence: number;
  content_hash: string;
}

export interface OntologyData {
  concepts: OntologyConcept[];
  relations: OntologyRelation[];
  summary: {
    total_concepts: number;
    total_relations: number;
    asserted_count: number;
    induced_enforcing_count: number;
    induced_candidate_count: number;
  };
}

export interface SkillAuditReport {
  skill_name: string;
  is_safe: boolean;
  recommendation: 'approve' | 'require_human_review' | 'reject';
  risk_score: number;
  detected_risks: string[];
  auditor_version: string;
  content_sha256: string;
}

export interface SkillManifest {
  name: string;
  description: string;
  version: string;
  author: string;
  origin: 'human' | 'synthesized';
  status: 'active' | 'pending' | 'quarantined' | 'rejected';
  isolation_level: string;
  content_sha256: string;
  scripts: string[];
  tags: string[];
  approved_by: string | null;
  approved_at: string | null;
  rejected_by?: string | null;
  rejected_at?: string | null;
  rejection_reason?: string | null;
  audit_report: SkillAuditReport;
}

export interface SkillsData {
  skills: SkillManifest[];
  summary: {
    total_skills: number;
    active_count: number;
    pending_count: number;
    quarantined_count: number;
  };
}

export interface ProviderBudget {
  provider: string;
  input_tokens: number;
  output_tokens: number;
  models: string[];
}

export interface RoleBudget {
  role: string;
  input_tokens: number;
  output_tokens: number;
}

export interface CompactionRecord {
  id: string;
  timestamp: string;
  reason: string;
  original_tokens: number;
  compacted_tokens: number;
  saved_tokens: number;
  compression_ratio_pct: number;
  kept_turns: number;
}

export interface BudgetData {
  session_budget: {
    max_tokens: number;
    used_input_tokens: number;
    used_output_tokens: number;
    total_used_tokens: number;
    remaining_tokens: number;
    budget_used_pct: number;
  };
  providers: Record<string, ProviderBudget>;
  roles: Record<string, RoleBudget>;
  compaction_history: CompactionRecord[];
}

export interface RuntimeSettings {
  llm_provider: string;
  llm_base_url: string;
  llm_model: string;
  llm_api_key_set: boolean;
  llm_api_key_masked: string;
  comfyui_base_url: string;
  providers_available: string[];
  workspace_dir?: string;
  available_models?: string[];
  /** Folders outside the workspace that clones may read but never write, as entered. */
  read_roots?: string[];
  /** The subset of `read_roots` that no longer exists on disk. */
  read_roots_missing?: string[];
  /** Folders set by UCLONE_READ_ROOTS when the app started; not editable here. */
  read_roots_env?: string[];
  /** UCLONE_READ_ROOTS entries that were ignored, each with the reason. */
  read_roots_env_ignored?: string[];
}

/** One tool an external tool server offers, as it describes itself. */
export interface McpTool {
  name: string;
  description: string;
}

/**
 * An external MCP tool server, as `/api/mcp/servers` reports it.
 *
 * Secret values (header values, environment values) are never returned; only their key
 * names are, in `header_keys` and `env_keys`.
 */
export interface McpServer {
  name: string;
  transport: 'http' | 'stdio';
  url: string | null;
  command: string | null;
  args: string[];
  env_keys: string[];
  header_keys: string[];
  enabled: boolean;
  /** `connecting` while the runtime is still starting its connection (e.g. just after app start). */
  status: 'connected' | 'connecting' | 'error' | 'disabled';
  /** Why the server is not connected, in the server's words, or `null`. */
  error: string | null;
  tools: McpTool[];
}

export interface McpServerList {
  /** The file the server list is saved in. */
  config_path: string;
  servers: McpServer[];
}

export interface McpServerDraft {
  name: string;
  transport: 'http' | 'stdio';
  url?: string;
  command?: string;
  args?: string[];
  env?: Record<string, string>;
  headers?: Record<string, string>;
}

export interface McpImportResult {
  added: McpServer[];
  skipped: { name: string; reason: string }[];
}

export interface ConnectionTestResult {
  status: 'ok' | 'warning' | 'error';
  message?: string;
  provider?: string;
  url?: string;
  models?: string[];
  online?: boolean;
  error?: string;
  stats?: Record<string, unknown>;
}

export interface EvalProbeResult {
  name: string;
  passed: boolean;
  duration_s: number;
  latency_s: number;
  message: string;
  metadata?: Record<string, unknown>;
}

export interface EvalSummary {
  total_probes: number;
  passed_probes: number;
  failed_probes: number;
  pass_rate: number;
  duration_s: number;
  p50_latency_s: number | null;
  p95_latency_s: number | null;
  worst_latency_s: number | null;
}

export interface EvalReport {
  timestamp: string;
  suite: string;
  model: string | null;
  provider: string | null;
  summary: EvalSummary;
  probes: EvalProbeResult[];
  metadata?: Record<string, unknown>;
}

export interface EvalScorecardMetrics {
  total_suites: number;
  total_probes: number;
  passed_probes: number;
  failed_probes: number;
  pass_rate: number;
}

/**
 * Why an evaluations response holds what it holds. An empty scorecard is three facts, and
 * the server says which: `no_backend` (this runtime ships no evaluation suites), `empty`
 * (none has been run), `error` (the reports could not be read; `error` says why).
 */
export type EvaluationsStatus = 'ok' | 'empty' | 'no_backend' | 'error';

export interface EvaluationsData {
  status: EvaluationsStatus;
  error: string | null;
  suites: EvalReport[];
  scorecard: Record<string, EvalReport>;
  metrics: EvalScorecardMetrics;
}

export interface EvaluationHistoryResponse {
  status: EvaluationsStatus;
  error: string | null;
  reports: EvalReport[];
  total: number;
}

export interface KnowledgeGraphProvenance {
  source_session?: string | null;
  originating_sessions?: string[];
  agent_id?: string;
  confidence?: number;
  content_hash?: string;
  first_seen?: string | null;
  last_seen?: string | null;
  description?: string;
  rule_expression?: string;
  origin?: 'axiomatic' | 'derived' | string;
}

export interface KnowledgeGraphNode {
  id: string;
  name: string;
  tier: 'asserted' | 'induced_enforcing' | 'induced_candidate' | string;
  type: 'entity' | 'concept' | string;
  provenance?: KnowledgeGraphProvenance;
}

export interface KnowledgeGraphEdge {
  id: string;
  source: string;
  target: string;
  predicate: string;
  tier: 'asserted' | 'induced_enforcing' | 'induced_candidate' | string;
  provenance?: KnowledgeGraphProvenance;
}

export interface KnowledgeGraphTriple {
  subject: string;
  predicate: string;
  object: string;
  tier: string;
  provenance?: KnowledgeGraphProvenance;
}

export interface KnowledgeGraphSummary {
  total_triples: number;
  total_nodes: number;
  total_edges: number;
  session_id?: string | null;
  agent_id?: string | null;
}

export interface KnowledgeGraphResponse {
  triples: KnowledgeGraphTriple[];
  nodes: KnowledgeGraphNode[];
  edges: KnowledgeGraphEdge[];
  summary: KnowledgeGraphSummary;
}

// ---------------------------------------------------------------------------------
// Conversations that can seat more than one agent (`uclone_x.room`).
//
// The word "room" is the runtime's, and stays out of anything a user reads: on screen
// these are conversations, their participants are "in this conversation", and an agent
// holding the floor is "Writing...".
// ---------------------------------------------------------------------------------

export interface RoomSummary {
  room_id: string;
  title: string;
  agent_ids: string[];
  human_ids: string[];
  message_count: number;
  updated_at: string;
}

export interface RoomParticipant {
  id: string;
  kind: 'agent' | 'human';
  display_name?: string;
  role?: string;
  session_id?: string;
  ontology_namespace?: string;
  /**
   * Other names this participant answers to.
   *
   * `Participant.aliases` (`room/models.py:87`), which `MentionSelector` resolves after
   * ids and before giving up. A head that matched ids alone told the user `@db` would
   * not be answered when the Core would have answered it.
   */
  aliases?: string[];
  persona_summary?: string;
}

/**
 * A provider and, where one applies, the model within it — `core/provenance.py`'s
 * `ServiceRef`.
 */
export interface RoomServiceRef {
  provider: string;
  model?: string | null;
}

/** One failed attempt preceding the value finally returned (`AttemptRecord`). */
export interface RoomAttemptRecord {
  provider: string;
  model?: string | null;
  error_class: string;
  status_code?: number | null;
  span_id?: string | null;
}

/**
 * `Provenance` as `RoomState.model_dump(mode="json")` actually serializes it.
 *
 * **Not `MessageProvenance`.** That interface describes the flattened shape
 * `/api/chat` hands the single-agent surface, where `served_by` is a string. A room row
 * carries the Core object unflattened, so `served_by` and `requested` are objects and
 * rendering one as a React child blank-screened the conversation on the first agent
 * reply. `degraded` is a computed field on the Python model and always present; `path`
 * is `ExecutionPath`'s value, not a free string.
 */
export interface RoomProvenance {
  path: 'primary' | 'retry' | 'failover';
  requested: RoomServiceRef;
  served_by: RoomServiceRef;
  attempts?: RoomAttemptRecord[];
  degraded?: boolean;
}

export interface RoomSpeakerDecision {
  verdict: 'speak' | 'silence' | 'abstain';
  speaker_id?: string | null;
  /** The selector's own report of how sure it was. Never rendered as accuracy. */
  confidence: number;
  selector: string;
  reasoning: string;
  provenance?: RoomProvenance | null;
}

/** `RoomTurnRefusal`: why a failed turn would fail the same way if retried (#969). */
export type RoomTurnRefusal = 'budget_exceeded';

export interface RoomTranscriptMessage {
  seq: number;
  sender_id: string;
  content: string;
  /** `RoomMessageKind` — speech, or a change to who is in the conversation. */
  kind: 'utterance' | 'join' | 'leave';
  created_at: string;
  decision?: RoomSpeakerDecision | null;
  provenance?: RoomProvenance | null;
  error?: string | null;
  /** Set beside `error` when a retry would meet the same refusal (#969). */
  refusal?: RoomTurnRefusal | null;
  completed: boolean;
  /**
   * The last `seq` this utterance's turn actually saw when it started, per
   * `RoomMessage.rendered_through` (#945). On the wire this is always a concrete number --
   * `0` for a message stored before the field existed, never an absent key -- but the type
   * stays optional for fixtures that omit it; `unansweredSilence` reads `undefined` and `0`
   * both as no evidence, permissively.
   */
  rendered_through?: number;
  /**
   * Set when the speaker's own session could not be written after this turn (#1366). The
   * reply stands; what the turn added to that clone's memory of the conversation will not be
   * there after a restart. Kept apart from `error`, which means the turn itself failed.
   */
  persist_error?: string | null;
  /**
   * Set when what the speaker learned in this turn could not be saved (#1367). The reply
   * stands; the clone will not know it after a restart. Apart from `persist_error` because
   * the two records are written, and lost, independently. Its text is the cause, for
   * diagnostics; the conversation says only that it was not saved.
   */
  knowledge_persist_error?: string | null;
  /**
   * Set when the speaker's knowledge record for this conversation could not be read before
   * this turn and was set aside -- kept under another name, never deleted (#1367). Saved
   * memory facts are another file and are not touched. A flag: where the file went and why
   * are in the log.
   */
  knowledge_set_aside?: boolean;
  /**
   * How many distinct facts the turn asked to save to memory, and how many of them were never
   * saved (#1375). The reply may still claim it saved; these say what happened. Counts, not
   * the tool's error: that is written for the model, and stays in the tool history.
   */
  memory_facts_tried?: number;
  memory_facts_unsaved?: number;
  /**
   * The turn that produced this row, per `RoomMessage.turn_id` -- the id every streamed
   * delta of that turn carried. What lets the head retire a turn's live bubble the moment
   * its row is in the record (#1379). `null` on a human's row, a membership row, and a row
   * stored before the field existed.
   */
  turn_id?: string | null;
}

export interface RoomPolicy {
  /** Resets on every human message. Not a conversation-lifetime budget. */
  max_agent_turns_per_human_message: number;
  max_span_messages: number;
  transcript_window: number;
  hesitation_seconds: number;
  default_responder_id: string;
}

export interface RoomTurnState {
  agent_turns_since_human: number;
  last_speaker_id?: string | null;
}

/** One tool call a seat made, as the room recorded it after the turn landed. */
export interface RoomToolUse {
  turn_id: string;
  participant_id: string;
  tool_name: string;
  tool_call_id: string | null;
  /** `ToolResultStatus`: success, error, timeout... */
  status: string;
  error: string | null;
  duration_ms: number;
  arguments_preview: string;
  output_preview: string;
  truncated: boolean;
  /** The file the call wrote, read from the tool's output. */
  written_path: string | null;
  /**
   * The call *may* have written without naming a path: a declared writer that did not both
   * succeed and name one (a shell, one that failed partway), or any call that started a
   * helper. It is a possibility the Core counts, never a record that a write happened.
   */
  wrote_unnamed: boolean;
  subagent_id: string | null;
  recorded_at: string;
  /** The transcript `seq` of the turn that made the call. */
  seq: number | null;
}

export interface RoomWrittenFile {
  path: string;
  participant_id: string;
  tool_name: string;
  turn_id: string;
  tool_call_id?: string | null;
  written_at: string;
}

export interface RoomFileRecord {
  kept_since_creation: boolean;
  unrecorded_turns: number;
  unattributed_writes: number;
  turns_started: number;
  turns_landed: number;
}

export interface RoomState {
  room_id: string;
  title: string;
  participants: RoomParticipant[];
  transcript: RoomTranscriptMessage[];
  turn_state: RoomTurnState;
  policy: RoomPolicy;
  last_decision?: RoomSpeakerDecision | null;
  tool_uses?: RoomToolUse[];
  written_files?: RoomWrittenFile[];
  file_record?: RoomFileRecord;
}

/**
 * One seat's share of `GET /api/rooms/{id}/context`.
 *
 * `live` is not decoration: a seat nobody has spoken in this process answers from its
 * persisted record, which is behind any turn a previous process did not write. The two
 * copies can disagree, and the surface says which one the figure came from (P6).
 */
export interface RoomSeatContext {
  participant_id: string;
  session_id: string;
  active_turns: number;
  is_saturated: boolean;
  /**
   * Tokens this seat has spent, or `null` when this process has no record of it.
   *
   * The two are different facts and are not folded together (P6): a seat that answered in
   * an earlier run has spent plenty and booked none of it here, because the budget manager
   * holds one process's bookings, and rendering that as `0` would report a fresh
   * conversation where there is a full one.
   *
   * Its maximum is `max_context_tokens`, when there is one.
   */
  used_tokens: number | null;
  cumulative_tokens?: number | null;
  /**
   * The context window `used_tokens` counts against, or `null` when it was not measured.
   *
   * Never a typical figure for the model's family. A locally served model's window is
   * whatever the daemon gave it when it loaded it -- on this machine Ollama reported
   * 32,768 for a model whose published window is 131,072 -- so an unmeasured seat draws
   * turns instead and says why. `TokenBudget.max_tokens` is not this: that is a spend
   * ceiling, and a ring against it reads near empty however full the context is.
   */
  max_context_tokens: number | null;
  /**
   * How `max_context_tokens` was arrived at. `loaded` came from the server that is
   * serving the model; `published` is the limit the provider publishes and enforces.
   * The head turns this key into the sentence a reader sees.
   */
  context_window_source: 'loaded' | 'published' | null;
  live: boolean;
}

/**
 * How full the conversation is. `is_saturated` is true when **any** seat is, because the
 * conversation is what stops being able to continue.
 */
export interface RoomContext {
  room_id: string;
  seats: RoomSeatContext[];
  is_saturated: boolean;
  saturation_threshold: number;
}

/**
 * What compacting one seat did, exactly as the Core stated it.
 *
 * Never merged with its siblings into one figure: compaction is reported with the
 * provenance of whichever producer wrote its ledger, and averaging several of those
 * invents an attribution no producer wrote (P6).
 */
export interface RoomSeatCompaction {
  participant_id: string;
  session_id: string;
  reason: string;
  ledger_source: string;
  tokens_before: number;
  tokens_after: number;
  saved_tokens: number;
  compression_ratio_pct: number;
  messages_before: number;
  messages_after: number;
  superseded_ledger_count: number;
  active_turns: number;
  provenance?: RoomProvenance | null;
}

export interface RoomCompaction {
  room_id: string;
  results: RoomSeatCompaction[];
}

/**
 * What the two history routes answer: the room as it now stands, plus the seats whose own
 * session did not reset with it.
 *
 * `participants_not_reset` is on every answer and is usually empty. A seat in it is still
 * answering from turns the transcript no longer holds, which is exactly the divergence the
 * rewind exists to prevent -- so the surface names it rather than reporting a clean cut.
 */
export interface RoomHistoryAnswer extends RoomState {
  participants_not_reset: string[];
}

/**
 * The four `AGENT_REPLY` payloads on a room's topic, discriminated by `status`.
 *
 * The head needs all four: the floor being taken, a token delta, the landed utterance, and
 * a cascade that stopped before anyone spoke. There is no "the room is finished" event --
 * see `applyRoomEvent`.
 */
export interface RoomTurnStarted {
  room_id: string;
  agent_id: string;
  /**
   * The turn a `generating`, `streaming` or `final` event belongs to -- the key a live
   * bubble accumulates and clears against. Absent only on `error`, where nobody spoke.
   */
  turn_id: string;
  status: 'generating';
  detail?: string;
}

export interface RoomTurnStatusUpdate {
  room_id: string;
  agent_id: string;
  turn_id: string;
  status: 'status_update';
  detail: string;
}

export interface RoomTurnDelta {
  room_id: string;
  agent_id: string;
  turn_id: string;
  status: 'streaming';
  delta: string;
}

export interface RoomTurnLanded {
  room_id: string;
  agent_id: string;
  turn_id: string;
  /**
   * The transcript row, on `final` only. The row is not known until the turn is over, so
   * the Core sends none on `generating` or `streaming` (`_publish_turn_start`).
   */
  seq: number;
  status: 'final';
  content: string;
  /** `false` when Stop cancelled the turn; `error` then says so. */
  completed: boolean;
  error?: string;
  refusal?: RoomTurnRefusal;
}

/** `ui/rooms.py`'s `_announce_failure`: no agent and no turn, because nobody spoke. */
export interface RoomCascadeFailed {
  room_id: string;
  status: 'error';
  error: string;
}

export type RoomReplyPayload =
  | RoomTurnStarted
  | RoomTurnStatusUpdate
  | RoomTurnDelta
  | RoomTurnLanded
  | RoomCascadeFailed;

/** `RoomOrchestrator.note_human_activity`: that the human is composing, never what. */
export interface RoomTypingNotice {
  room_id: string;
  sender_id: string;
  status: 'typing';
}

/** `RoomOrchestrator.interrupt`, published by Stop. An interjection publishes nothing. */
export interface RoomInterruptNotice {
  room_id: string;
  reason: string;
}

/**
 * One event on `room.{room_id}`, as `/api/stream` delivers it: the envelope's `event_type`
 * and its `payload`.
 *
 * **Every kind the topic carries, not only the replies.** This type used to be the reply
 * payload alone, so the composing notice and the interrupt the same topic carries were
 * cast to it and folded as though they were landed rows (#929). Discriminated by
 * `event_type`, and a reply further by `status`; `frontend/src/test/room-stream-events.json`
 * holds one of each, generated by the Core's suite.
 */
export type RoomTopicEvent =
  | { event_type: 'AGENT_REPLY'; payload: RoomReplyPayload }
  | { event_type: 'USER_INPUT'; payload: RoomTypingNotice }
  | { event_type: 'INTERRUPT'; payload: RoomInterruptNotice };
