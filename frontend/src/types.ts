import type { UiLanguage } from './i18n/language';

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

/**
 * A clone a conversation can invite: an installed persona, by name.
 *
 * It replaced `AgentInfo`, the row of `GET /api/agents`, which overlaid live-instance state
 * (status, uptime, sub-agents) onto these (2026-09-27: `/api/agents` removed, #1775).
 */
export interface CloneChoice {
  id: string;
  label: string;
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
  /** Personas this one may call through `a2a_call` (#1558); kept through an edit. */
  a2a_peers?: string[];
  /** The persona's own prompt, without the appended default prompt (#892). */
  system_prompt?: string;
  /** Whether the runtime's default prompt is appended after `system_prompt`. */
  append_default_prompt?: boolean;
  model_tier?: PersonaModelTier;
  /** Loaded from the package's own directory; an edit writes a workspace override. */
  builtin?: boolean;
  /** A workspace file that replaces a built-in persona of the same name. */
  overrides_builtin?: boolean;
  /**
   * Where its picture is shown from, with a `?v=` that changes with the picture; `null` for
   * a clone with none. Absent from a runtime too old to say.
   */
  avatar_url?: string | null;
  /** Whether the picture was chosen in this workspace, not shipped with the clone. */
  avatar_chosen?: boolean;
  /** The id of the latest change to that picture, made anywhere; `0` before any. */
  avatar_change_id?: number;
}

/** Which Settings model a persona's turns run on: the deep one (`inherit`) or the fast one. */
export type PersonaModelTier = 'inherit' | 'fast';

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
 * `ChatTurnOutcome` (`uclone_x.ui.app` until #1731): how a chat turn ended (#1007 item 4).
 * `degraded` is the Core's meaning only: an answer from a model other than the one requested.
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

/** One clone a skill is not offered to, and the required tools that clone does not have. */
export interface SkillHiddenFrom {
  persona: string;
  missing_tools: string[];
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
  /** The tools the skill's steps call (#1826). A clone is offered it only when it has them all. */
  requires_tools?: string[];
  /** The clones that are not offered the skill, and the tools each lacks (#1826). */
  hidden_from?: SkillHiddenFrom[];
  /** The skill comes with UClone-X, so Settings cannot revoke it (#1827). */
  shipped?: boolean;
  approved_by: string | null;
  approved_at: string | null;
  rejected_by?: string | null;
  rejected_at?: string | null;
  rejection_reason?: string | null;
  /**
   * Why the skill is not used, in English (#1720). The panel shows it only when it cannot word
   * `not_loaded_code` itself (#1777).
   */
  not_loaded_reason?: string | null;
  /**
   * Which reason it is (`uclone_x/skills/refusals.py`'s `SkillRefusalCode`), worded by the head
   * from the `skills.notLoaded.codes` catalog. Typed `string`, not the closed set: a newer Core
   * may send a code this head does not know, and that falls back to `not_loaded_reason`.
   */
  not_loaded_code?: string | null;
  /** The values `not_loaded_code`'s sentence uses, by placeholder name (`name`). */
  not_loaded_params?: Readonly<Record<string, string | number>> | null;
  audit_report: SkillAuditReport;
}

/**
 * A skill a clone proposed with `propose_skill`, waiting for a person to approve or turn it
 * down in Settings (#1827). It is not used until then.
 */
export interface SkillProposal {
  name: string;
  version: string;
  description: string;
  requires_tools: string[];
  /** The clone that proposed it, and the conversation it was proposed in. */
  agent_id: string;
  session_id: string;
  proposed_at: string;
  /** The skill's full instructions, as they would be installed. */
  instructions: string;
  /** The version in use now, when this proposal would replace one. */
  current_version: string | null;
  /** The changes from the version in use now, as a unified diff; empty for a new skill. */
  diff: string;
  /** The digest of the text above; Approve sends it back so only this text is installed. */
  digest: string;
}

export interface SkillsData {
  skills: SkillManifest[];
  /** Proposals waiting for a person's decision (#1827). */
  proposals?: SkillProposal[];
  /** The skill folder this runtime reads does not exist, so nothing could be loaded (#1721). */
  store_missing?: boolean;
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
    /** `null` when the session has no token ceiling, the default since the usage limits (#1685). */
    max_tokens: number | null;
    used_input_tokens: number;
    used_output_tokens: number;
    total_used_tokens: number;
    remaining_tokens: number | null;
    budget_used_pct: number | null;
  };
  providers: Record<string, ProviderBudget>;
  roles: Record<string, RoleBudget>;
  compaction_history: CompactionRecord[];
}

/** One model a cloud provider's own listing returned (#1631). */
export interface CatalogEntry {
  id: string;
  display_name: string | null;
  /** `null` when the provider did not say. */
  context_window: number | null;
  max_output_tokens: number | null;
  /** False for embedding, speech and image models. */
  chat_capable: boolean;
  created_at: string | null;
}

/**
 * What a cloud provider's model listing said, as the Core read it (#1631).
 *
 * `entries` is empty unless `status` is `live`, and `recommended` is always one of their ids or
 * `null`: the head never recommends a model the provider did not list. For a non-live status,
 * `detail` says why in plain words.
 */
export interface CatalogResult {
  provider: string;
  status: 'live' | 'no_key' | 'key_rejected' | 'unreachable' | 'no_listing';
  entries: CatalogEntry[];
  recommended: string | null;
  fetched_at: string | null;
  detail: string | null;
}

/** `GET /api/models`, optionally with `?refresh=1` to ask the provider again. */
export interface ModelsResponse {
  provider: string;
  models: string[];
  current_model: string;
  catalog?: CatalogResult | null;
}

/** `POST /api/models/catalog`: a picked provider's listing, before it is saved (#1657). */
export interface CatalogPreviewResponse {
  provider: string;
  models: string[];
  /** Local providers only: whether anything answered at the address (#1666). */
  reachable?: boolean;
  /** Local providers only: the server answered, and refused the key (#1672). */
  key_refused?: boolean;
  catalog: CatalogResult | null;
}

export interface RemoteGpuInfo {
  name: string;
  total_mb: number;
  used_mb: number;
  driver: string;
}

export interface RemotePortMapping {
  service_name: string;
  remote_port: number;
  local_port: number;
}

export interface RemoteGpuStatus {
  host: string;
  connected: boolean;
  pid?: number | null;
  mappings?: RemotePortMapping[];
  gpu?: RemoteGpuInfo | null;
  error?: string | null;
  comfyui_autostarted?: boolean;
  ollama_models?: string[];
  restored_settings?: {
    llm_provider?: string;
    llm_base_url?: string;
    comfyui_base_url?: string;
  };
}

/** One provider's key, as `/api/settings` reports it whichever provider is active. */
export interface ProviderKeyState {
  id: string;
  key_set: boolean;
  /** The first and last few characters, never the key. Empty when no key is held. */
  key_masked: string;
  /** Where the key comes from: saved in Settings, or an environment variable that overrides it. */
  key_source: 'settings' | 'env' | '';
  /** The variable carrying it when `key_source` is `'env'`. */
  key_env_var?: string | null;
}

/** A setting an environment variable decides while it stays set; the file still saves. */
export interface EnvOverride {
  field: 'llm_provider' | 'llm_model' | 'llm_base_url';
  env_var: string;
  /** The Core's English sentence; the screen shows its own translation instead. */
  message?: string;
}

export interface RuntimeSettings {
  llm_provider: string;
  llm_base_url: string;
  /** The chat (deep) model: the one a clone's turn runs on when its persona names none. */
  llm_model: string;
  /** The fast model for auxiliary calls; empty means it follows `llm_model`. */
  llm_model_fast?: string | null;
  llm_api_key_set: boolean;
  llm_api_key_masked: string;
  llm_api_key_source?: 'settings' | 'env' | '';
  llm_api_key_env_var?: string | null;
  /** Every provider's key state, so a key never looks lost when another provider is active. */
  providers?: ProviderKeyState[];
  /** Where the provider, model and endpoint in use come from; `'env'` wins over the file. */
  llm_provider_source?: 'settings' | 'env' | '';
  llm_provider_env_var?: string;
  llm_model_source?: 'settings' | 'env' | '';
  llm_model_env_var?: string;
  llm_base_url_source?: 'settings' | 'env' | '';
  llm_base_url_env_var?: string;
  /** Each setting an environment variable decides instead of the saved choice. */
  env_overrides?: EnvOverride[];
  /**
   * A Settings save found the settings file unreadable and kept it aside, unchanged, before
   * writing a new one (#1860). The keys saved in it are not the ones shown here.
   */
  settings_set_aside?: boolean;
  comfyui_base_url: string;
  /** What draws pictures: `auto`, `local` or `gemini`, as saved. */
  image_engine?: string;
  /** The Gemini picture model, as saved; the runtime's default when none is. */
  image_model?: string;
  /** Why the saved picture choice cannot be used, or `''` when it can. */
  image_settings_problem?: string;
  providers_available: string[];
  workspace_dir?: string;
  available_models?: string[];
  /** The cloud provider's model listing; `null` for Ollama, vLLM and mock. */
  catalog?: CatalogResult | null;
  /** Folders outside the workspace that clones may read but never write, as entered. */
  read_roots?: string[];
  /** The subset of `read_roots` that no longer exists on disk. */
  read_roots_missing?: string[];
  /** Folders set by UCLONE_READ_ROOTS when the app started; not editable here. */
  read_roots_env?: string[];
  /** UCLONE_READ_ROOTS entries that were ignored, each with the reason. */
  read_roots_env_ignored?: string[];
  /**
   * The screens' language as chosen, `'system'` included.
   * `LocaleProvider` owns it; the Settings form neither shows nor sends it.
   */
  ui_language?: UiLanguage;
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
  /** vLLM only: the server answered 401 or 403 (#1672). */
  key_refused?: boolean;
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
  /**
   * Other names this participant answers to.
   *
   * `Participant.aliases` (`room/models.py:87`), which `MentionSelector` resolves after
   * ids and before giving up. A head that matched ids alone told the user `@db` would
   * not be answered when the Core would have answered it.
   */
  aliases?: string[];
  persona_summary?: string;
  /** The clone this seat is (`Participant.persona`); the id when the Core sends none. */
  persona?: string;
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
export type RoomTurnRefusal =
  | 'budget_exceeded'
  | 'usage_limit'
  | 'model_without_tools'
  | 'model_unavailable'
  | 'provider_auth';

/** `ProviderFailureKind`: what went wrong on a hosted provider's side (#1630). */
export type ProviderFailureKind =
  | 'model_unavailable'
  | 'provider_auth'
  | 'provider_quota'
  | 'provider_unreachable'
  | 'provider_outage'
  | 'provider_error';

/**
 * `ProviderFailure`: a hosted provider's failure as the Core tells it (#1630). `message` is
 * written to be shown as is -- it names the provider and says whose side the problem is on,
 * with no status code or response body -- unlike the row's `error`.
 */
export interface RoomProviderFailure {
  kind: ProviderFailureKind;
  message: string;
  retryable: boolean;
  /** The provider's display name ("Google"); absent on a row stored before it was carried. */
  provider?: string | null;
}

export interface RoomTranscriptMessage {
  seq: number;
  sender_id: string;
  content: string;
  /**
   * `RoomMessageKind` — speech, a change to who is in the conversation, or a note the
   * application wrote for the reader (the `/loop` help and status, #1641).
   */
  kind: 'utterance' | 'join' | 'leave' | 'note';
  /**
   * Which notice a `note` row is (`room/notices.py`'s `NoticeCode`), worded by the head in the
   * reader's language from the `notices` catalog. Absent on every other row and on a note
   * stored before codes existed; `content` is then what shows, and it is always the English
   * fallback. Typed `string`, not the closed set: a newer Core may send a code this head does
   * not know, and that falls back to `content` too.
   */
  code?: string | null;
  /** The values `code`'s sentence uses, by placeholder name; `interval_seconds` is raw seconds. */
  params?: Readonly<Record<string, string | number>> | null;
  created_at: string;
  decision?: RoomSpeakerDecision | null;
  provenance?: RoomProvenance | null;
  error?: string | null;
  /** Set beside `error` when a retry would meet the same refusal (#969). */
  refusal?: RoomTurnRefusal | null;
  /** Set beside `error` when the turn failed at a hosted provider (#1630). */
  provider_failure?: RoomProviderFailure | null;
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
   * Set when the speaker's own saved record of this conversation could not be read by this
   * version -- typically one a newer version wrote -- and was set aside when this turn was
   * saved: kept under another name, never written over (#1844). The speaker carried on
   * without it. A flag: where the file went and why are in the log.
   */
  session_set_aside?: boolean;
  /**
   * How many distinct facts the turn asked to save to memory, and how many of them were never
   * saved (#1375). The reply may still claim it saved; these say what happened. Counts, not
   * the tool's error: that is written for the model, and stays in the tool history.
   */
  memory_facts_tried?: number;
  memory_facts_unsaved?: number;
  /**
   * The ids of the facts the speaker saved to its memory from this turn (#1404). Written after
   * the reply landed, when the clone has learned from the turn, so a row first arrives without
   * it. Empty, or absent on a row stored before the field existed, when nothing was saved.
   */
  knowledge_learned?: string[];
  /**
   * Set when the speaker could not learn from this turn (#1404): a plain sentence, never the
   * cause. The reply stands. The conversation shows its own translated line, not this text.
   */
  knowledge_extract_error?: string | null;
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
  /** Absent while it holds the Core's default (#1898): the room API leaves such keys out. */
  max_span_tokens?: number;
  transcript_window: number;
  hesitation_seconds: number;
  default_responder_id: string;
  /** Absent while false (#1898). */
  autonomous?: boolean;
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

export interface RoomActiveTurn {
  in_flight: boolean;
  agent_id: string | null;
  turn_id: string | null;
  status?: string;
  detail?: string | null;
  accumulated_text?: string;
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
  /**
   * The terminal or protocol head that keeps this room (`run`, `loop`, `acp`, `a2a`), absent
   * for a room the app keeps. A head's room has that head as its one writer (#1885): the
   * server refuses every change from here, so the conversation shows it read-only.
   */
  head?: string | null;
  active_turn?: RoomActiveTurn | null;
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
  provider_failure?: RoomProviderFailure;
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
