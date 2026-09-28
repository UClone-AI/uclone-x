import { DiagnosticsPanel } from './DiagnosticsPanel';
import React, { useState, useEffect, useCallback, useRef } from 'react';
import {
  X,
  Settings,
  Server,
  Gauge,
  Key,
  Cpu,
  Image as ImageIcon,
  CheckCircle2,
  AlertTriangle,
  XCircle,
  Loader2,
  RefreshCw,
  ChevronRight,
  Eye,
  EyeOff,
  Radio,
  Plus,
  Pencil,
  Save,
  Users,
  Download,
  Trash2,
  Wrench,
  FolderOpen,
  ExternalLink,
  Globe,
} from 'lucide-react';
import {
  getProviderMeta,
  validateKeyFormat,
  describeKeyProblem,
  sanitizeApiKey,
  isCloudProvider,
} from '../lib/providerRegistry';
import { sameEndpoint } from '../lib/endpoint';
import {
  RuntimeSettings,
  ConnectionTestResult,
  PersonaCatalog,
  type CatalogResult,
  type ModelsResponse,
  type CatalogPreviewResponse,
  type RemoteGpuStatus,
} from '../types';
import { PersonaManager } from './personas/PersonaManager';
import { SkillsSection } from './settings/SkillsSection';
import { McpServersSection } from './settings/McpServersSection';
import { DiagnosticsSection } from './settings/DiagnosticsSection';
import { UsageSection } from './settings/UsageSection';
import { ImagesSection } from './settings/ImagesSection';
import { UsageOffer } from './settings/UsageOffer';
import type { UsageReport } from '../lib/usage';
import { fetchPersonaCatalog, savePersona } from '../lib/personasApi';
import type { PersonaDraft, PersonaEditMode, PersonaSaveResult } from '../lib/personaDraft';
import { useEscapeOwner } from '../lib/escapePrecedence';
import { CoreFailure, coreReason, failureOf, plainFailure } from '../lib/coreFailure';
import { LANGUAGE_AUTONYMS, UI_LANGUAGES, fmt, plural, useCopy, useLocale, type UiLanguage } from '../i18n';

const PERSONA_ICONS = { add: Plus, edit: Pencil, save: Save, cancel: X, spinner: Loader2 };
import { Badge } from './ui/Badge';
import { Button } from './ui/Button';

export type SettingsTabId = 'all' | 'llm' | 'usage' | 'tools' | 'folders' | 'clones' | 'skills' | 'diagnostics';

interface SettingsTabItem {
  id: SettingsTabId;
  icon: React.ComponentType<{ className?: string }>;
}

/** The tabs in order. Each label is `settings.tabs[id]` in the catalog. */
const SETTINGS_TABS: SettingsTabItem[] = [
  { id: 'all', icon: Settings },
  { id: 'llm', icon: Server },
  { id: 'usage', icon: Gauge },
  { id: 'tools', icon: ImageIcon },
  { id: 'folders', icon: FolderOpen },
  { id: 'clones', icon: Users },
  { id: 'skills', icon: Wrench },
  { id: 'diagnostics', icon: CheckCircle2 },
];

/** The provider cards in order. Each description is `settings.llm.providers[id]`. */
const PROVIDER_CARDS = [
  { id: 'ollama', label: 'Ollama' },
  { id: 'vllm', label: 'vLLM' },
  { id: 'openai', label: 'OpenAI' },
  { id: 'anthropic', label: 'Anthropic' },
  { id: 'gemini', label: 'Gemini' },
] as const;

/**
 * Whether a rejected `fetch` was cancelled rather than broken.
 *
 * Checked by `name`, not by `instanceof DOMException`: the runtime that rejects an
 * aborted `fetch` is not always the one this module was compiled against, and
 * `name === 'AbortError'` is the part every one of them agrees on. Getting this
 * wrong reports the user's own cancellation back to them as a failure.
 */
/** A refused request's stable `code` and the Core's `detail`, each `null` when absent. */
const readRefusal = async (res: Response): Promise<{ code: string | null; detail: string | null }> => {
  try {
    const body: unknown = await res.json();
    if (typeof body !== 'object' || body === null) return { code: null, detail: null };
    const { code, detail } = body as { code?: unknown; detail?: unknown };
    return {
      code: typeof code === 'string' ? code : null,
      detail: typeof detail === 'string' && detail.trim() !== '' ? detail : null,
    };
  } catch {
    return { code: null, detail: null };
  }
};

const isAbortError = (err: unknown): boolean =>
  typeof err === 'object' && err !== null && (err as { name?: unknown }).name === 'AbortError';

interface SettingsModalProps {
  isOpen: boolean;
  onClose: () => void;
  onSettingsSaved?: (settings: RuntimeSettings) => void;
  /**
   * Developer mode: whether the workspace dock offers its developer instruments, and whether
   * this modal shows its Diagnostics section (ACP and Evals, #1358). Off by default (owner
   * ruling, 2026-09-22).
   *
   * Held by the head, not the runtime: it is how this screen is laid out, and a second head
   * would not need it (ui-authoring §3). So it is not part of `RuntimeSettings` and not sent
   * with Save -- the switch applies at once.
   */
  developerMode: boolean;
  onDeveloperModeChange: (on: boolean) => void;
  /** The tab to show when the modal opens, e.g. `'usage'` from the room's usage banner. */
  initialTab?: SettingsTabId;
}

export const SettingsModal: React.FC<SettingsModalProps> = ({
  isOpen,
  onClose,
  onSettingsSaved,
  developerMode,
  onDeveloperModeChange,
  initialTab,
}) => {
  const copy = useCopy();
  const t = copy.settings;
  const locale = useLocale();
  // Read through a ref where a callback must stay stable: a language switch must not re-run
  // the load effect, which would refetch the form and throw away what the user has typed.
  const copyRef = useRef(copy);
  copyRef.current = copy;
  const [loading, setLoading] = useState<boolean>(false);
  const [saving, setSaving] = useState<boolean>(false);
  const [testing, setTesting] = useState<boolean>(false);
  const [currentSettings, setCurrentSettings] = useState<RuntimeSettings | null>(null);
  const [activeTab, setActiveTab] = useState<SettingsTabId>('all');

  // Form Fields
  const [llmProvider, setLlmProvider] = useState<string>('ollama');
  const [llmBaseUrl, setLlmBaseUrl] = useState<string>('');
  const [llmModel, setLlmModel] = useState<string>('');
  const [isCustomModel, setIsCustomModel] = useState<boolean>(false);
  // The fast model for auxiliary calls. Empty means "same as the chat model", and is sent
  // empty so a save can clear an earlier choice.
  const [llmModelFast, setLlmModelFast] = useState<string>('');
  const [isCustomFastModel, setIsCustomFastModel] = useState<boolean>(false);
  // A cloud provider's endpoint is an override, so it sits behind a disclosure (#1631). It
  // opens by itself only when an override is already set, so a proxy is never hidden.
  const [endpointOpen, setEndpointOpen] = useState<boolean>(false);
  const [refreshingCatalog, setRefreshingCatalog] = useState<boolean>(false);
  const [previewCatalog, setPreviewCatalog] = useState<{
    provider: string;
    key: string;
    endpoint: string;
    catalog: CatalogResult | null;
  } | null>(null);
  // A local server's installed models, read at the address on the form (#1666).
  const [localListing, setLocalListing] = useState<{
    provider: string;
    endpoint: string;
    models: string[];
    reachable: boolean;
    keyRefused?: boolean;
  } | null>(null);
  // The model last chosen for each provider while the modal is open, so a look at another
  // provider and back does not lose the saved model (#1657). Refilled from each settings read.
  const modelsByProvider = useRef<Record<string, { model: string; fast: string }>>({});
  const [llmApiKey, setLlmApiKey] = useState<string>('');
  const [showApiKey, setShowApiKey] = useState<boolean>(false);
  const [removingKey, setRemovingKey] = useState<boolean>(false);
  const [comfyuiBaseUrl, setComfyuiBaseUrl] = useState<string>('http://127.0.0.1:8188');
  const [remoteHost, setRemoteHost] = useState<string>('dell');
  // Off unless ticked: a worker connected for images keeps the LLM where it was.
  const [remoteSyncLlm, setRemoteSyncLlm] = useState<boolean>(false);
  const [remoteGpuStatus, setRemoteGpuStatus] = useState<RemoteGpuStatus | null>(null);
  const [connectingRemote, setConnectingRemote] = useState<boolean>(false);
  const [disconnectingRemote, setDisconnectingRemote] = useState<boolean>(false);

  // Extra folders clones may read (never write). Edited locally and sent only when the
  // list differs from what the server returned: the server re-checks every entry on
  // each update and refuses the whole save when one has gone missing, so resending an
  // untouched list would let a deleted folder block an unrelated model change.
  const [readRoots, setReadRoots] = useState<string[]>([]);
  const [newReadRoot, setNewReadRoot] = useState<string>('');

  // Ollama model lifecycle (install / delete)
  const [pullModelInput, setPullModelInput] = useState<string>('');
  const [installing, setInstalling] = useState<boolean>(false);
  const [deletingModel, setDeletingModel] = useState<string | null>(null);

  // An install is the longest request this app makes — multi-gigabyte weights, up to
  // the route's 900s ceiling — and until #1233 nothing could stop waiting for it. The
  // modal closed, the request stayed open, and its `setState` landed on a surface
  // nobody was looking at. These hold the controller for whichever model request is
  // in flight, so the user can cancel and so closing the modal cancels for them.
  const pullAbortRef = useRef<AbortController | null>(null);
  const deleteAbortRef = useRef<AbortController | null>(null);

  const abortModelRequests = useCallback(() => {
    pullAbortRef.current?.abort();
    deleteAbortRef.current?.abort();
  }, []);

  // `isOpen` false renders null without unmounting, so closing the modal is not an
  // unmount and would otherwise leave the request running. Both are covered.
  useEffect(() => {
    if (!isOpen) abortModelRequests();
  }, [isOpen, abortModelRequests]);
  useEffect(() => abortModelRequests, [abortModelRequests]);

  // Diagnostic Test Results
  const [testResults, setTestResults] = useState<{
    llm?: ConnectionTestResult;
    comfyui?: ConnectionTestResult;
  } | null>(null);
  const [feedbackMessage, setFeedbackMessage] = useState<{
    type: 'success' | 'error';
    text: string;
  } | null>(null);

  const fetchCurrentSettings = useCallback(async () => {
    setLoading(true);
    setFeedbackMessage(null);
    try {
      const res = await fetch('/api/settings');
      if (!res.ok) throw await failureOf(res);
      const data: RuntimeSettings = await res.json();
      setCurrentSettings(data);
      setLlmProvider(data.llm_provider || 'ollama');
      setLlmBaseUrl(data.llm_base_url || '');
      setEndpointOpen(Boolean(data.llm_base_url));
      setLlmModel(data.llm_model || '');
      setLlmModelFast(data.llm_model_fast || '');
      modelsByProvider.current = {};
      setIsCustomFastModel(false);
      if (data.available_models && data.available_models.length > 0) {
        setIsCustomModel(!data.available_models.includes(data.llm_model || ''));
      }
      setComfyuiBaseUrl(data.comfyui_base_url || 'http://127.0.0.1:8188');
      setReadRoots(data.read_roots ?? []);
      setNewReadRoot('');
      setLlmApiKey('');

      try {
        const gpuRes = await fetch('/api/settings/remote-gpu/status');
        if (gpuRes.ok) {
          const gpuData: RemoteGpuStatus = await gpuRes.json();
          setRemoteGpuStatus(gpuData);
          if (gpuData.host) {
            setRemoteHost(gpuData.host);
          }
          // A tunnel that died put back the addresses it had replaced; show those, not the
          // tunnel's, which the settings read above still carried.
          // An empty original is still an original: skipping it would leave the dead
          // tunnel address on the form for the next Save to write back.
          const restored = gpuData.restored_settings;
          if (restored?.comfyui_base_url !== undefined) setComfyuiBaseUrl(restored.comfyui_base_url);
          if (restored?.llm_base_url !== undefined) setLlmBaseUrl(restored.llm_base_url);
          if (restored?.llm_provider !== undefined) setLlmProvider(restored.llm_provider);
        }
      } catch {
        // Non-blocking status check
      }
    } catch (err) {
      console.error('Failed to load settings:', err);
      setFeedbackMessage({ type: 'error', text: plainFailure(err, copyRef.current.settings.failure.load) });
    } finally {
      setLoading(false);
    }
  }, []);

  useEffect(() => {
    if (isOpen) {
      fetchCurrentSettings();
      setTestResults(null);
    }
  }, [isOpen, fetchCurrentSettings]);

  useEffect(() => {
    if (isOpen && initialTab) setActiveTab(initialTab);
  }, [isOpen, initialTab]);

  // The answer to the last limits save, handed to the offer so it hides without a re-read.
  // Dropped on close: the next opening reads afresh, and a save from elsewhere must not lose.
  const [savedUsage, setSavedUsage] = useState<UsageReport | null>(null);
  useEffect(() => {
    if (!isOpen) setSavedUsage(null);
  }, [isOpen]);

  // The persona editor's data (#892). Fetched here, not in the components, which are
  // props-only; loaded with the rest of the modal so the list is current when it opens.
  const [personaCatalog, setPersonaCatalog] = useState<PersonaCatalog | null>(null);
  const [personaLoadError, setPersonaLoadError] = useState<string | null>(null);
  const loadPersonas = useCallback(async () => {
    try {
      setPersonaCatalog(await fetchPersonaCatalog());
      setPersonaLoadError(null);
    } catch (err) {
      console.error('Failed to load the clones:', err);
      setPersonaLoadError(coreReason(err) ?? '');
    }
  }, []);
  useEffect(() => {
    if (isOpen) loadPersonas();
  }, [isOpen, loadPersonas]);

  const handleSavePersona = async (
    draft: PersonaDraft,
    mode: PersonaEditMode,
  ): Promise<PersonaSaveResult> => {
    const result = await savePersona(draft, mode, copy.personaEditor);
    if (result.ok) {
      await loadPersonas();
      // `onSettingsSaved` is how the app re-reads its metadata, the agent list included;
      // reusing it keeps the rail current without widening this modal's contract.
      if (currentSettings) onSettingsSaved?.(currentSettings);
    }
    return result;
  };

  // Escape closes the modal, but only when no higher-precedence layer claims it first (#1036).
  useEscapeOwner('dialog', isOpen, onClose);

  const handleProviderSelect = (newProvider: string) => {
    const switching = newProvider !== llmProvider;
    if (switching) modelsByProvider.current[llmProvider] = { model: llmModel, fast: llmModelFast };
    setLlmProvider(newProvider);
    setTestResults(null);
    setIsCustomModel(false);
    // Another provider's fast model means nothing to this one; empty follows the chat model.
    if (newProvider !== llmProvider) {
      setLlmModelFast('');
      setIsCustomFastModel(false);
    }
    if (newProvider === 'vllm') {
      if (llmProvider !== 'vllm') {
        setLlmBaseUrl('');
        setLlmModel('');
      }
    } else if (newProvider === 'ollama') {
      if (!llmBaseUrl || llmBaseUrl.includes('api.openai.com') || llmBaseUrl.includes('api.anthropic.com') || llmBaseUrl.includes('googleapis.com')) {
        setLlmBaseUrl('http://localhost:11434');
      }
      // A cloud provider's model means nothing to Ollama. Asked of the provider, not of the
      // model's name, so no cloud model id has to be written here to recognise one (#1631).
      if (!llmModel || isCloudProvider(llmProvider)) {
        setLlmModel('qwen3:8b');
      }
    } else if (isCloudProvider(newProvider)) {
      // Drop an endpoint that belongs to another provider; keep a custom proxy.
      const foreignEndpoints: Record<string, string[]> = {
        openai: ['11434', 'googleapis.com', 'api.anthropic.com'],
        anthropic: ['11434', 'googleapis.com', 'api.openai.com'],
        gemini: ['11434', 'api.openai.com', 'api.anthropic.com'],
      };
      const nextBaseUrl = foreignEndpoints[newProvider].some((host) => llmBaseUrl.includes(host))
        ? ''
        : llmBaseUrl;
      setLlmBaseUrl(nextBaseUrl);
      setEndpointOpen(nextBaseUrl.trim() !== '');
      // No model is written here: which models exist is the provider's listing's to say, and
      // the picker preselects its recommendation once it has one (#1631).
      if (llmProvider !== newProvider) {
        setLlmModel(() => '');
      }
    } else if (newProvider === 'mock') {
      setLlmModel('mock-llm');
    }
    // Back on a provider already looked at: its model comes back with it (#1657).
    const remembered = switching ? modelsByProvider.current[newProvider] : undefined;
    if (remembered) {
      setLlmModel(remembered.model);
      setLlmModelFast(remembered.fast);
    }
  };

  const handleTestConnection = async () => {
    setTesting(true);
    setFeedbackMessage(null);
    try {
      const payload = {
        target: 'all',
        llm_provider: llmProvider,
        llm_base_url: llmBaseUrl.trim() || undefined,
        llm_api_key: sanitizeApiKey(llmApiKey) || undefined,
        comfyui_base_url: comfyuiBaseUrl.trim() || undefined,
      };
      const res = await fetch('/api/settings/test', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(payload),
      });
      if (!res.ok) throw await failureOf(res);
      const data = await res.json();
      setTestResults(data.results || {});
    } catch (err) {
      console.error('Failed to test the connection:', err);
      setFeedbackMessage({ type: 'error', text: plainFailure(err, t.failure.test) });
    } finally {
      setTesting(false);
    }
  };

  const savedReadRoots = currentSettings?.read_roots ?? [];
  const readRootsChanged =
    readRoots.length !== savedReadRoots.length ||
    readRoots.some((root, i) => root !== savedReadRoots[i]);
  const missingReadRoots = new Set(currentSettings?.read_roots_missing ?? []);
  const envReadRoots = currentSettings?.read_roots_env ?? [];
  const envReadRootsIgnored = currentSettings?.read_roots_env_ignored ?? [];

  const handleAddReadRoot = () => {
    const root = newReadRoot.trim();
    if (!root) return;
    if (!readRoots.includes(root)) setReadRoots([...readRoots, root]);
    setNewReadRoot('');
  };

  const handleSaveSettings = async () => {
    setSaving(true);
    setFeedbackMessage(null);
    const typedSaveKey = sanitizeApiKey(llmApiKey);
    try {
      const payload = {
        llm_provider: llmProvider,
        llm_base_url: llmBaseUrl.trim(),
        llm_model: llmModel.trim(),
        llm_model_fast: llmModelFast.trim(),
        llm_api_key: typedSaveKey || undefined,
        // A key is filed under the provider it was typed for, never another one.
        llm_api_key_provider: typedSaveKey ? llmProvider : undefined,
        comfyui_base_url: comfyuiBaseUrl.trim(),
        read_roots: readRootsChanged ? readRoots : undefined,
      };
      const res = await fetch('/api/settings', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(payload),
      });
      if (!res.ok) throw await failureOf(res);
      const data: RuntimeSettings = await res.json();
      setCurrentSettings(data);
      if (data.read_roots) setReadRoots(data.read_roots);
      setLlmApiKey('');
      setFeedbackMessage({
        type: 'success',
        text: t.feedback.saved,
      });
      if (onSettingsSaved) {
        onSettingsSaved(data);
      }
    } catch (err) {
      console.error('Failed to save settings:', err);
      setFeedbackMessage({ type: 'error', text: plainFailure(err, t.failure.save) });
    } finally {
      setSaving(false);
    }
  };

  /** Removes one provider's saved key; the others, and the active choice, are untouched. */
  const handleRemoveKey = async (provider: string, name: string) => {
    if (!window.confirm(fmt(t.apiKey.confirmRemove, { name }))) return;
    setRemovingKey(true);
    setFeedbackMessage(null);
    try {
      const res = await fetch(`/api/settings/api-keys/${encodeURIComponent(provider)}`, { method: 'DELETE' });
      if (!res.ok) {
        // A refusal carries a stable `code`; say it in the reader's language, not the Core's.
        const { code, detail } = await readRefusal(res);
        if (code === 'key_in_use' || code === 'unknown_provider') {
          setFeedbackMessage({
            type: 'error',
            text: code === 'key_in_use' ? t.apiKey.inUse : t.apiKey.unknownProvider,
          });
          return;
        }
        throw new CoreFailure(res.status, detail);
      }
      const data: RuntimeSettings = await res.json();
      setCurrentSettings(data);
      setFeedbackMessage({ type: 'success', text: fmt(t.apiKey.removed, { name }) });
    } catch (err) {
      console.error('Failed to remove the key:', err);
      setFeedbackMessage({ type: 'error', text: plainFailure(err, t.apiKey.removeFailed) });
    } finally {
      setRemovingKey(false);
    }
  };

  const handleConnectRemoteGpu = async () => {
    const trimmed = remoteHost.trim();
    if (!trimmed) return;
    setConnectingRemote(true);
    setFeedbackMessage(null);
    try {
      const res = await fetch('/api/settings/remote-gpu/connect', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          host: trimmed,
          apply_settings: true,
          auto_start_comfyui: true,
          sync_llm: remoteSyncLlm,
        }),
      });
      const data = await res.json();
      if (data.status === 'ok' && data.connected) {
        setRemoteGpuStatus(data.tunnel);
        if (data.applied_changes?.comfyui_base_url) {
          setComfyuiBaseUrl(data.applied_changes.comfyui_base_url);
        }
        if (data.applied_changes?.llm_base_url) {
          setLlmBaseUrl(data.applied_changes.llm_base_url);
        }
        if (data.applied_changes?.llm_provider) {
          setLlmProvider(data.applied_changes.llm_provider);
        }
        await refreshAvailableModels();
        const connectedText = `${t.remoteGpu.statusConnected}: ${data.tunnel.host}${data.tunnel.gpu ? ` (${data.tunnel.gpu.name})` : ''}`;
        setFeedbackMessage({
          type: 'success',
          text:
            data.llm_skipped === 'model_not_on_remote'
              ? `${connectedText}. ${t.remoteGpu.llmModelMissing}`
              : connectedText,
        });
      } else {
        setFeedbackMessage({
          type: 'error',
          text: data.error || 'Failed to connect to remote GPU',
        });
      }
    } catch (err) {
      setFeedbackMessage({
        type: 'error',
        text: err instanceof Error ? err.message : 'Failed to connect to remote GPU',
      });
    } finally {
      setConnectingRemote(false);
    }
  };

  const handleDisconnectRemoteGpu = async () => {
    setDisconnectingRemote(true);
    setFeedbackMessage(null);
    try {
      const res = await fetch('/api/settings/remote-gpu/disconnect', {
        method: 'POST',
      });
      const data = await res.json();
      if (data.status === 'ok') {
        setRemoteGpuStatus(data.tunnel || null);
        if (data.restored_settings?.comfyui_base_url !== undefined) {
          setComfyuiBaseUrl(data.restored_settings.comfyui_base_url);
        }
        if (data.restored_settings?.llm_base_url !== undefined) {
          setLlmBaseUrl(data.restored_settings.llm_base_url);
        }
        if (data.restored_settings?.llm_provider !== undefined) {
          setLlmProvider(data.restored_settings.llm_provider);
        }
        await refreshAvailableModels();
        setFeedbackMessage({
          type: 'success',
          text: t.remoteGpu.statusDisconnected,
        });
      }
    } catch (err) {
      setFeedbackMessage({
        type: 'error',
        text: err instanceof Error ? err.message : 'Failed to disconnect remote GPU',
      });
    } finally {
      setDisconnectingRemote(false);
    }
  };

  // Re-reads /api/settings and pushes it through the same onSettingsSaved
  // wiring the persona editor uses (#892), so App.tsx's model list stays
  // current without a page reload. Kept separate from fetchCurrentSettings,
  // which resets the whole form and would blank out the just-shown
  // install/delete feedbackMessage.
  const refreshAvailableModels = useCallback(async () => {
    try {
      const res = await fetch('/api/settings');
      if (res.ok) {
        const data: RuntimeSettings = await res.json();
        setCurrentSettings(data);
        onSettingsSaved?.(data);
      }
    } catch (err) {
      console.error('Failed to refresh the model list:', err);
    }
  }, [onSettingsSaved]);

  const handlePullModel = async () => {
    const model = pullModelInput.trim();
    if (!model) return;
    setInstalling(true);
    setFeedbackMessage(null);
    const controller = new AbortController();
    pullAbortRef.current = controller;
    try {
      const res = await fetch('/api/models/pull', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ model }),
        signal: controller.signal,
      });
      if (res.ok) {
        const data = await res.json().catch(() => ({}));
        setPullModelInput('');
        setFeedbackMessage({
          type: 'success',
          // `joined` is the server saying this request did not start a second
          // download — an install of this model was already running and this one
          // waited on it. Saying so is the only way the user learns that their
          // second click cost nothing (#1233).
          text: fmt(data.joined ? t.feedback.installJoined : t.feedback.installed, { model }),
        });
        await refreshAvailableModels();
      } else {
        throw await failureOf(res);
      }
    } catch (err) {
      if (isAbortError(err)) {
        // Cancelling is not failing. Ollama keeps whatever it has fetched and the
        // request simply stopped being waited on, so saying "Failed to install"
        // here would report a defeat the user chose and the daemon never had.
        setFeedbackMessage({
          type: 'success',
          text: fmt(t.feedback.installStopped, { model }),
        });
      } else {
        console.error('Failed to install the model:', err);
        setFeedbackMessage({ type: 'error', text: plainFailure(err, t.failure.install) });
      }
    } finally {
      if (pullAbortRef.current === controller) pullAbortRef.current = null;
      setInstalling(false);
    }
  };

  const handleDeleteModel = async (model: string) => {
    if (!window.confirm(fmt(t.feedback.confirmDelete, { model }))) return;
    setDeletingModel(model);
    setFeedbackMessage(null);
    const controller = new AbortController();
    deleteAbortRef.current = controller;
    try {
      const res = await fetch('/api/models/delete', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ model }),
        signal: controller.signal,
      });
      if (res.ok) {
        setFeedbackMessage({ type: 'success', text: fmt(t.feedback.deleted, { model }) });
        await refreshAvailableModels();
      } else {
        throw await failureOf(res);
      }
    } catch (err) {
      if (isAbortError(err)) {
        setFeedbackMessage({
          type: 'success',
          text: fmt(t.feedback.deleteStopped, { model }),
        });
      } else {
        console.error('Failed to delete the model:', err);
        setFeedbackMessage({ type: 'error', text: plainFailure(err, t.failure.remove) });
      }
    } finally {
      if (deleteAbortRef.current === controller) deleteAbortRef.current = null;
      setDeletingModel(null);
    }
  };

  const typedKey = sanitizeApiKey(llmApiKey);
  const formEndpoint = llmBaseUrl.trim();
  // The saved list describes the saved provider at the saved address, and only that: an
  // empty one, or one for another address, is not this form's answer (#1666). "Another
  // address" is another server: `localhost` and `127.0.0.1` on one port are the same one.
  const onSavedServer =
    currentSettings?.llm_provider === llmProvider &&
    sameEndpoint(currentSettings.llm_base_url ?? '', formEndpoint);
  const savedModels = onSavedServer ? currentSettings?.available_models ?? [] : [];
  const listsLocally = llmProvider === 'ollama' || llmProvider === 'vllm';
  // vLLM has no default address, so an empty field is not asked about.
  const needsLocalListing =
    listsLocally &&
    currentSettings !== null &&
    savedModels.length === 0 &&
    (llmProvider === 'ollama' || formEndpoint !== '');
  const localMatches =
    localListing !== null &&
    localListing.provider === llmProvider &&
    localListing.endpoint === formEndpoint;
  const detectedModels =
    savedModels.length > 0
      ? savedModels
      : localMatches && localListing.models.length > 0
        ? localListing.models
        : testResults?.llm?.models ?? [];
  const localNotAnswering = needsLocalListing && localMatches && !localListing.reachable && detectedModels.length === 0;
  // Running, and it turned the key away: a different fix from a stopped server (#1672).
  const localKeyRefused = needsLocalListing && localMatches && localListing.keyRefused === true && detectedModels.length === 0;
  const ollamaEmpty =
    llmProvider === 'ollama' && needsLocalListing && localMatches && localListing.reachable && detectedModels.length === 0;
  // The positive half of the same signal: the local server at the form's address answered,
  // either as the saved list (saved server, models listed) or as this form's own listing.
  const localConnectedCount = !listsLocally
    ? null
    : savedModels.length > 0
      ? savedModels.length
      : needsLocalListing && localMatches && localListing.reachable
        ? localListing.models.length
        : null;

  const localSeq = useRef(0);
  useEffect(() => {
    if (!needsLocalListing) return;
    const seq = ++localSeq.current;
    const asked = { provider: llmProvider, endpoint: formEndpoint };
    // An address being typed is asked about once typing pauses, not per keystroke.
    const timer = setTimeout(async () => {
      try {
        const res = await fetch('/api/models/catalog', {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({ provider: asked.provider, base_url: asked.endpoint || undefined }),
        });
        if (!res.ok) throw await failureOf(res);
        const data: CatalogPreviewResponse = await res.json();
        if (seq === localSeq.current) {
          setLocalListing({
            ...asked,
            models: data.models ?? [],
            reachable: data.reachable ?? false,
            keyRefused: data.key_refused === true,
          });
        }
      } catch (err) {
        console.error('Failed to list the local models:', err);
        if (seq === localSeq.current) setLocalListing({ ...asked, models: [], reachable: false });
      }
    }, onSavedServer ? 0 : 500);
    return () => clearTimeout(timer);
  }, [needsLocalListing, llmProvider, formEndpoint, onSavedServer]);

  // The cloud provider's own model list (#1631). `/api/settings` carries the *saved*
  // provider's. A provider picked on this form, or a key typed into it, is asked about before
  // saving (#1657): a beginner who picks Gemini sees Gemini's models, not an empty field.
  const providerMeta = getProviderMeta(llmProvider);
  // The saved list answers only for the saved provider, key and endpoint: an edited endpoint
  // is asked about like a new provider, and the server sends it no held key (#1663).
  const usesSavedCatalog =
    currentSettings?.llm_provider === llmProvider &&
    typedKey === '' &&
    sameEndpoint(formEndpoint, currentSettings.llm_base_url ?? '');
  const typedKeyUsable = typedKey === '' || validateKeyFormat(llmProvider, typedKey).valid;
  const needsPreview = Boolean(providerMeta) && currentSettings !== null && !usesSavedCatalog && typedKeyUsable;
  const previewMatches =
    previewCatalog !== null &&
    previewCatalog.provider === llmProvider &&
    previewCatalog.key === typedKey &&
    previewCatalog.endpoint === formEndpoint;
  const catalog: CatalogResult | null = !providerMeta
    ? null
    : usesSavedCatalog
      ? currentSettings?.catalog ?? null
      : previewMatches
        ? previewCatalog.catalog
        : null;
  const previewPending = needsPreview && !previewMatches;
  const catalogIsLive = catalog?.status === 'live';
  const listedModels = catalogIsLive ? catalog.entries.filter((e) => e.chat_capable) : [];
  const listedIds = listedModels.map((e) => e.id);
  const typedModel = llmModel.trim();
  // A cloud provider is connected only on a real listing: `live` with models to pick. A cached
  // failure is another status, and an empty or missing catalogue says nothing either way.
  const cloudConnectedCount = catalogIsLive && listedModels.length > 0 ? listedModels.length : null;
  const typedModelUnlisted =
    catalogIsLive && typedModel !== '' && !catalog.entries.some((e) => e.id === typedModel);
  // The saved model, missing from a live listing of the saved provider and key: it has been
  // retired, and the listing's pick is offered in its place, never switched to (§3.4, #1657).
  const retiredReplacement =
    typedModelUnlisted &&
    usesSavedCatalog &&
    typedModel === (currentSettings?.llm_model ?? '').trim() &&
    catalog.recommended &&
    catalog.recommended !== typedModel
      ? catalog.recommended
      : null;

  // An empty model field takes the provider's recommendation, and only then: a model the user
  // chose, or typed, is theirs. Keyed on the catalogue alone, so clearing the field to retype
  // a name does not refill it under the user's cursor.
  const recommendedModel = catalogIsLive ? catalog.recommended : null;
  const llmModelRef = useRef(llmModel);
  llmModelRef.current = llmModel;
  useEffect(() => {
    if (!recommendedModel || llmModelRef.current.trim() !== '') return;
    setLlmModel(recommendedModel);
    setIsCustomModel(false);
  }, [catalog, recommendedModel]);

  const previewSeq = useRef(0);
  /** Resolves false when the listing could not be asked for at all. */
  const loadPreview = useCallback(
    async (refresh: boolean): Promise<boolean> => {
      const seq = ++previewSeq.current;
      const asked = { provider: llmProvider, key: typedKey, endpoint: formEndpoint };
      try {
        // The key goes in the body, never the URL.
        const res = await fetch('/api/models/catalog', {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({
            provider: asked.provider,
            api_key: asked.key || undefined,
            base_url: asked.endpoint || undefined,
            refresh,
          }),
        });
        if (!res.ok) throw await failureOf(res);
        const data: CatalogPreviewResponse = await res.json();
        if (seq === previewSeq.current) setPreviewCatalog({ ...asked, catalog: data.catalog ?? null });
        return true;
      } catch (err) {
        console.error('Failed to load the model list:', err);
        if (seq === previewSeq.current) setPreviewCatalog({ ...asked, catalog: null });
        return false;
      }
    },
    [llmProvider, typedKey, formEndpoint],
  );

  useEffect(() => {
    if (!needsPreview) return;
    // A key or an endpoint being typed is asked about once typing pauses, not per keystroke.
    const typing = typedKey !== '' || formEndpoint !== '';
    const timer = setTimeout(() => void loadPreview(false), typing ? 500 : 0);
    return () => clearTimeout(timer);
  }, [needsPreview, loadPreview, typedKey, formEndpoint]);

  const handleRefreshCatalog = async () => {
    setRefreshingCatalog(true);
    setFeedbackMessage(null);
    if (!usesSavedCatalog) {
      const asked = await loadPreview(true);
      setRefreshingCatalog(false);
      if (!asked) {
        setFeedbackMessage({ type: 'error', text: copyRef.current.settings.catalog.refreshFailed });
      }
      return;
    }
    try {
      const res = await fetch('/api/models?refresh=1');
      if (!res.ok) throw await failureOf(res);
      const data: ModelsResponse = await res.json();
      setCurrentSettings((prev) =>
        prev && prev.llm_provider === data.provider
          ? { ...prev, catalog: data.catalog ?? null, available_models: data.models }
          : prev,
      );
    } catch (err) {
      console.error('Failed to refresh the model list:', err);
      setFeedbackMessage({
        type: 'error',
        text: plainFailure(err, copyRef.current.settings.catalog.refreshFailed),
      });
    } finally {
      setRefreshingCatalog(false);
    }
  };

  /**
   * The sentence for a cloud picker with nothing to pick from. Chosen by `status`, never the
   * Core's `detail`: that is English, and would put English on a Korean screen.
   */
  const catalogMessage = (vendor: string): string => {
    if (!catalog) {
      if (previewPending) return fmt(t.catalog.loading, { vendor });
      if (!typedKeyUsable) return fmt(t.catalog.keyMalformed, { vendor });
      return fmt(t.catalog.unavailable, { vendor });
    }
    switch (catalog.status) {
      case 'live':
        return fmt(t.catalog.emptyList, { vendor });
      case 'no_key':
        return fmt(t.catalog.noKey, { vendor });
      case 'key_rejected':
        return fmt(t.catalog.keyRejected, { vendor });
      case 'unreachable':
        return fmt(t.catalog.unreachable, { vendor });
      case 'no_listing':
        return fmt(t.catalog.noListing, { vendor });
      default:
        return fmt(t.catalog.unavailable, { vendor });
    }
  };

  const handleLanguageChange = (next: UiLanguage) => {
    setFeedbackMessage(null);
    locale.setChoice(next).catch((err) => {
      console.error('Failed to save the language:', err);
      setFeedbackMessage({ type: 'error', text: t.language.saveFailed });
    });
  };

  // One element for both kinds of provider, so "is it connected" has one answer on the page.
  const connectedLine = (text: string) => (
    <p
      data-testid="settings-llm-connected"
      className="mt-1.5 flex items-center gap-1.5 text-[11px] text-emerald-400/90"
    >
      <CheckCircle2 className="w-3 h-3 shrink-0" />
      {text}
    </p>
  );

  // Where the endpoint field goes depends on the provider: in the main block where it is
  // required (Ollama, vLLM), behind the Advanced disclosure where it is an override.
  const endpointField = (
    <div>
      <label className="block text-xs font-medium text-slate-300 mb-1.5 flex items-center justify-between">
        <span>{t.llm.endpoint}</span>
        <span className="text-[10px] text-slate-500 font-mono">
          {llmProvider === 'ollama'
            ? 'http://localhost:11434'
            : llmProvider === 'vllm'
            ? t.llm.endpointRequired
            : isCloudProvider(llmProvider)
            ? t.llm.endpointOptional
            : t.llm.endpointOther}
        </span>
      </label>
      <input
        type="text"
        value={llmBaseUrl}
        onChange={(e) => setLlmBaseUrl(e.target.value)}
        placeholder={
          llmProvider === 'ollama'
            ? 'http://localhost:11434'
            : llmProvider === 'vllm'
            ? t.llm.endpointVllmPlaceholder
            : providerMeta
            ? t.catalog.endpointPlaceholder
            : t.llm.endpointPlaceholder
        }
        className="w-full bg-slate-950/90 border border-slate-800 rounded-xl px-3 py-2 text-xs text-white placeholder-slate-600 focus:outline-none focus:border-cyan-500/80 font-mono transition-colors"
      />
      {isCloudProvider(llmProvider) && (
        <p className="text-[10px] text-slate-500 mt-1">
          {fmt(t.llm.endpointCloudHint, { name: providerMeta?.displayName ?? '' })}
        </p>
      )}
    </div>
  );

  if (!isOpen) return null;

  return (
    <div
      className="fixed inset-0 z-50 flex items-center justify-center p-4 bg-black/70 backdrop-blur-sm animate-in fade-in duration-150 select-none"
      role="dialog"
      aria-modal="true"
    >
      <div className="bg-slate-900 border border-slate-800 rounded-2xl shadow-2xl w-full max-w-2xl max-h-[90vh] flex flex-col overflow-hidden">
        {/* Modal Header */}
        {/* At phone width the header must shrink, not overflow (#1416): the text column is
            `min-w-0` so the `truncate`d workspace path cannot set its minimum width, the title
            row wraps its badge, and the close button is `shrink-0` so it is never pushed out. */}
        <div
          data-testid="settings-header"
          className="px-6 py-4 border-b border-slate-800 flex items-center justify-between gap-3 bg-slate-950/60 shrink-0"
        >
          <div className="flex items-center gap-2.5 min-w-0 flex-1">
            <div className="p-2 rounded-xl bg-cyan-950/80 text-cyan-400 border border-cyan-800/80 shrink-0">
              <Settings className="w-5 h-5" />
            </div>
            <div className="min-w-0">
              <h2 className="text-base font-bold text-white tracking-tight flex flex-wrap items-center gap-x-2 gap-y-1">
                {t.header.title}
                <Badge tone="info" className="text-[10px] uppercase">
                  {t.header.hotReload}
                </Badge>
              </h2>
              <p className="text-xs text-slate-400">
                {t.header.subtitle}
                {currentSettings?.workspace_dir && (
                  <span className="block text-[11px] font-mono text-cyan-400/90 mt-0.5 truncate">
                    {fmt(t.header.workspace, { dir: currentSettings.workspace_dir })}
                  </span>
                )}
              </p>
            </div>
          </div>
          <Button variant="ghost" size="icon" onClick={onClose} title={t.header.close} className="shrink-0 w-11 h-11 -mr-2">
            <X className="w-5 h-5" />
          </Button>
        </div>

        {/* Navigation Tabs Bar */}
        <div
          data-testid="settings-tabs-bar"
          className="px-6 py-2 bg-slate-950/40 border-b border-slate-800/80 flex items-center gap-1.5 overflow-x-auto scrollbar-hide shrink-0"
        >
          {SETTINGS_TABS.map((tab) => {
            const Icon = tab.icon;
            const isActive = activeTab === tab.id;
            return (
              <button
                key={tab.id}
                type="button"
                onClick={() => setActiveTab(tab.id)}
                data-testid={`settings-tab-${tab.id}`}
                className={`flex items-center gap-1.5 px-3 py-1.5 rounded-lg text-xs font-medium whitespace-nowrap shrink-0 transition-colors ${
                  isActive
                    ? 'bg-cyan-500/10 text-cyan-400 border border-cyan-500/30 font-semibold'
                    : 'text-slate-400 hover:text-slate-200 hover:bg-slate-900 border border-transparent'
                }`}
              >
                <Icon className="w-3.5 h-3.5 shrink-0" />
                <span>{t.tabs[tab.id]}</span>
              </button>
            );
          })}
        </div>

        {/* Modal Body */}
        <div className="flex-1 overflow-y-auto p-6 space-y-6 text-sm text-slate-200">
          {feedbackMessage && (
            <div
              data-testid="settings-feedback"
              className={`p-3.5 rounded-xl border flex items-start gap-2.5 text-xs animate-in slide-in-from-top-1 duration-150 ${
                feedbackMessage.type === 'success'
                  ? 'bg-emerald-950/50 border-emerald-800/80 text-emerald-200'
                  : 'bg-rose-950/50 border-rose-800/80 text-rose-200'
              }`}
            >
              {feedbackMessage.type === 'success' ? (
                <CheckCircle2 className="w-4 h-4 text-emerald-400 shrink-0 mt-0.5" />
              ) : (
                <AlertTriangle className="w-4 h-4 text-rose-400 shrink-0 mt-0.5" />
              )}
              <div className="flex-1">{feedbackMessage.text}</div>
            </div>
          )}

          {/* The language control, first so it can be found by someone who cannot read the rest. */}
          {activeTab === 'all' && (
            <div className="space-y-2" data-testid="settings-language">
              <label
                htmlFor="settings-language-select"
                className="text-xs font-semibold uppercase tracking-wider text-slate-400 flex items-center gap-1.5"
              >
                <Globe className="w-3.5 h-3.5 text-cyan-400" />
                {t.language.title}
              </label>
              <select
                id="settings-language-select"
                data-testid="settings-language-select"
                value={locale.choice}
                onChange={(e) => handleLanguageChange(e.target.value as UiLanguage)}
                className="w-full bg-slate-950/90 border border-slate-800 rounded-xl px-3 py-2 text-xs text-white focus:outline-none focus:border-cyan-500/80 transition-colors cursor-pointer"
              >
                {UI_LANGUAGES.map((choice) => (
                  <option key={choice} value={choice} className="bg-slate-950 text-slate-200">
                    {choice === 'system'
                      ? fmt(t.language.system, { resolved: LANGUAGE_AUTONYMS[locale.systemLanguage] })
                      : LANGUAGE_AUTONYMS[choice]}
                  </option>
                ))}
              </select>
              <p className="text-[11px] text-slate-500">{t.language.hint}</p>
            </div>
          )}

          {activeTab === 'all' && <hr className="border-slate-800/80" />}

          {(activeTab === 'all' || activeTab === 'diagnostics') && <DiagnosticsPanel />}

          {/* Section 1: LLM Provider */}
          {(activeTab === 'all' || activeTab === 'llm') && (
          <div className="space-y-4">
            <div className="flex items-center justify-between">
              <label className="text-xs font-semibold uppercase tracking-wider text-slate-400 flex items-center gap-1.5">
                <Server className="w-3.5 h-3.5 text-cyan-400" />
                {t.llm.title}
              </label>
              {currentSettings && (
                <span className="text-[11px] font-mono text-slate-400">
                  {t.llm.active} <span className="text-cyan-400 font-semibold">{currentSettings.llm_provider}</span>
                </span>
              )}
            </div>

            {(currentSettings?.env_overrides ?? []).length > 0 && (
              <div data-testid="settings-env-overrides" className="space-y-1">
                {(currentSettings?.env_overrides ?? []).map((o) => (
                  <p
                    key={o.field}
                    className="text-[11px] text-amber-300 bg-amber-500/10 border border-amber-500/30 rounded-lg px-2.5 py-1.5"
                  >
                    {fmt(
                      o.field === 'llm_provider'
                        ? t.llm.envOverride.provider
                        : o.field === 'llm_model'
                          ? t.llm.envOverride.model
                          : t.llm.envOverride.baseUrl,
                      { variable: o.env_var },
                    )}
                  </p>
                ))}
              </div>
            )}

            {/* Provider Cards */}
            <div className="grid grid-cols-2 sm:grid-cols-3 md:grid-cols-5 gap-2">
              {PROVIDER_CARDS.map((p) => {
                const isSelected = llmProvider === p.id;
                return (
                  <button
                    key={p.id}
                    type="button"
                    onClick={() => handleProviderSelect(p.id)}
                    className={`p-2.5 rounded-xl border text-left transition-all flex flex-col justify-between ${
                      isSelected
                        ? 'bg-cyan-950/60 border-cyan-500/80 shadow-cyan-950/50 shadow-sm ring-1 ring-cyan-500/50'
                        : 'bg-slate-950/60 border-slate-800 hover:border-slate-700 hover:bg-slate-800/40'
                    }`}
                  >
                    <div className="flex items-center justify-between w-full">
                      <span className={`text-xs font-bold ${isSelected ? 'text-white' : 'text-slate-300'}`}>
                        {p.label}
                      </span>
                      {isSelected && <Radio className="w-3 h-3 text-cyan-400 fill-cyan-400" />}
                    </div>
                    <span className="text-[10px] text-slate-500 mt-1">{t.llm.providers[p.id]}</span>
                  </button>
                );
              })}
            </div>

            {/* Provider Configuration Fields */}
            {(() => {
              // A div, not a label: a label forwards its clicks to the first control inside it,
              // which would make clicking the words "Chat model" refresh the list.
              const modelHeader = (
                <div className="block text-xs font-medium text-slate-300 mb-1.5 flex items-center justify-between">
                  <span>{t.llm.model}</span>
                  <span className="flex items-center gap-2">
                    {catalog && (
                      <button
                        type="button"
                        onClick={handleRefreshCatalog}
                        disabled={refreshingCatalog}
                        className="inline-flex items-center gap-1 text-[11px] font-normal text-slate-400 hover:text-slate-200 disabled:opacity-50 transition-colors"
                      >
                        {refreshingCatalog ? (
                          <Loader2 className="w-3 h-3 animate-spin" />
                        ) : (
                          <RefreshCw className="w-3 h-3" />
                        )}
                        {t.catalog.refresh}
                      </button>
                    )}
                    <Cpu className="w-3.5 h-3.5 text-slate-500" />
                  </span>
                </div>
              );

              if (providerMeta) {
                // A cloud provider: the list is the provider's own (#1631). Typing a name is
                // always possible, and never refused.
                const vendor = providerMeta.vendorName;
                const showTyping =
                  listedModels.length === 0 || isCustomModel || !listedIds.includes(llmModel);
                return (
                  <div className="space-y-3 pt-2">
                    <div>
                      {modelHeader}
                      <div className="space-y-1.5">
                        {listedModels.length > 0 ? (
                          <select
                            data-testid="settings-model-select"
                            aria-label={t.catalog.modelPicker}
                            value={
                              isCustomModel ? '__custom__' : listedIds.includes(llmModel) ? llmModel : '__custom__'
                            }
                            onChange={(e) => {
                              if (e.target.value === '__custom__') {
                                setIsCustomModel(true);
                              } else {
                                setIsCustomModel(false);
                                setLlmModel(e.target.value);
                              }
                            }}
                            className="w-full bg-slate-950/90 border border-slate-800 rounded-xl px-3 py-2 text-xs text-white focus:outline-none focus:border-cyan-500/80 font-mono transition-colors cursor-pointer"
                          >
                            {listedModels.map((entry) => {
                              const name =
                                entry.display_name && entry.display_name !== entry.id
                                  ? `${entry.display_name} (${entry.id})`
                                  : entry.id;
                              const isRecommended = entry.id === catalog?.recommended;
                              return (
                                <option key={entry.id} value={entry.id} className="bg-slate-950 text-slate-200">
                                  {isRecommended ? `${name} ${t.catalog.recommended}` : name}
                                </option>
                              );
                            })}
                            <option value="__custom__" className="bg-slate-950 text-slate-400">
                              {t.catalog.otherModel}
                            </option>
                          </select>
                        ) : (
                          <p data-testid="settings-model-catalog-detail" className="text-[11px] text-slate-400">
                            {catalogMessage(vendor)}
                          </p>
                        )}
                        {showTyping && (
                          <input
                            type="text"
                            aria-label={t.catalog.modelName}
                            value={llmModel}
                            onChange={(e) => setLlmModel(e.target.value)}
                            placeholder={t.catalog.typePlaceholder}
                            className="w-full bg-slate-950/90 border border-slate-800 rounded-xl px-3 py-2 text-xs text-white placeholder-slate-600 focus:outline-none focus:border-cyan-500/80 font-mono transition-colors"
                          />
                        )}
                        {cloudConnectedCount !== null &&
                          connectedLine(plural(t.llm.cloudConnected, cloudConnectedCount))}
                        {retiredReplacement !== null ? (
                          <div
                            data-testid="settings-model-retired"
                            className="flex flex-wrap items-center gap-2 text-[11px] text-amber-400/90"
                          >
                            <span>
                              {fmt(t.catalog.retired, { model: typedModel, vendor, recommended: retiredReplacement })}
                            </span>
                            <button
                              type="button"
                              onClick={() => {
                                setLlmModel(retiredReplacement);
                                setIsCustomModel(false);
                              }}
                              className="rounded-md border border-amber-500/40 px-2 py-0.5 text-amber-200 hover:bg-amber-500/10 transition-colors"
                            >
                              {t.catalog.switchModel}
                            </button>
                          </div>
                        ) : (
                          typedModelUnlisted && (
                            <p data-testid="settings-model-unlisted" className="text-[11px] text-amber-400/90">
                              {fmt(t.catalog.notInList, { vendor })}
                            </p>
                          )
                        )}
                      </div>
                    </div>
                  </div>
                );
              }

              return (
                <div className="grid grid-cols-1 sm:grid-cols-2 gap-3.5 pt-2">
                  {endpointField}

                  <div>
                    {modelHeader}
                    {detectedModels.length > 0 ? (
                      <div className="space-y-1.5">
                        <select
                          data-testid="settings-model-select"
                          value={isCustomModel ? '__custom__' : (detectedModels.includes(llmModel) ? llmModel : '__custom__')}
                          onChange={(e) => {
                            if (e.target.value === '__custom__') {
                              setIsCustomModel(true);
                            } else {
                              setIsCustomModel(false);
                              setLlmModel(e.target.value);
                            }
                          }}
                          className="w-full bg-slate-950/90 border border-slate-800 rounded-xl px-3 py-2 text-xs text-white focus:outline-none focus:border-cyan-500/80 font-mono transition-colors cursor-pointer"
                        >
                          {detectedModels.map((m) => (
                            <option key={m} value={m} className="bg-slate-950 text-slate-200">
                              {m}
                            </option>
                          ))}
                          <option value="__custom__" className="bg-slate-950 text-slate-400">
                            {t.llm.customModelOption}
                          </option>
                        </select>
                        {(isCustomModel || !detectedModels.includes(llmModel)) && (
                          <input
                            type="text"
                            value={llmModel}
                            onChange={(e) => setLlmModel(e.target.value)}
                            placeholder={t.llm.customModelPlaceholder}
                            className="w-full bg-slate-950/90 border border-slate-800 rounded-xl px-3 py-2 text-xs text-white placeholder-slate-600 focus:outline-none focus:border-cyan-500/80 font-mono transition-colors"
                          />
                        )}
                      </div>
                    ) : (
                      <div>
                        <input
                          type="text"
                          value={llmModel}
                          onChange={(e) => setLlmModel(e.target.value)}
                          placeholder={t.llm.modelPlaceholder}
                          className="w-full bg-slate-950/90 border border-slate-800 rounded-xl px-3 py-2 text-xs text-white placeholder-slate-600 focus:outline-none focus:border-cyan-500/80 font-mono transition-colors"
                        />
                      </div>
                    )}
                    {localConnectedCount !== null &&
                      connectedLine(plural(t.llm.localConnected, localConnectedCount))}
                    {localNotAnswering && (
                      <p data-testid="settings-local-not-answering" className="mt-1.5 text-[11px] text-amber-400/90">
                        {fmt(llmProvider === 'ollama' ? t.llm.ollamaNotAnswering : t.llm.vllmNotAnswering, {
                          endpoint: formEndpoint || 'http://localhost:11434',
                        })}
                      </p>
                    )}
                    {localKeyRefused && (
                      <p data-testid="settings-local-key-refused" className="mt-1.5 text-[11px] text-amber-400/90">
                        {fmt(t.llm.vllmKeyRefused, { endpoint: formEndpoint })}
                      </p>
                    )}
                    {ollamaEmpty && (
                      <p data-testid="settings-ollama-no-models" className="mt-1.5 text-[11px] text-slate-400">
                        {t.llm.ollamaNoModels}
                      </p>
                    )}
                  </div>
                </div>
              );
            })()}

            {/* The fast model: auxiliary calls such as choosing who speaks next. Empty follows
                the chat model, and a clone that names its own model keeps it. */}
            {llmProvider !== 'mock' && (() => {
              const fastOptions = providerMeta
                ? listedModels.map((entry) => ({
                    id: entry.id,
                    label:
                      entry.display_name && entry.display_name !== entry.id
                        ? `${entry.display_name} (${entry.id})`
                        : entry.id,
                  }))
                : detectedModels.map((m) => ({ id: m, label: m }));
              const fastIds = fastOptions.map((o) => o.id);
              const fastTyped = llmModelFast.trim();
              const fastUnlisted = fastTyped !== '' && !fastIds.includes(fastTyped);
              const showFastTyping = fastOptions.length === 0 || isCustomFastModel || fastUnlisted;
              return (
                <div className="pt-1" data-testid="settings-fast-model">
                  <div className="block text-xs font-medium text-slate-300 mb-1.5">{t.llm.fastModel}</div>
                  <div className="space-y-1.5">
                    {fastOptions.length > 0 && (
                      <select
                        data-testid="settings-fast-model-select"
                        aria-label={t.llm.fastModel}
                        value={isCustomFastModel || fastUnlisted ? '__custom__' : fastTyped}
                        onChange={(e) => {
                          if (e.target.value === '__custom__') {
                            setIsCustomFastModel(true);
                          } else {
                            setIsCustomFastModel(false);
                            setLlmModelFast(e.target.value);
                          }
                        }}
                        className="w-full bg-slate-950/90 border border-slate-800 rounded-xl px-3 py-2 text-xs text-white focus:outline-none focus:border-cyan-500/80 font-mono transition-colors cursor-pointer"
                      >
                        <option value="" className="bg-slate-950 text-slate-400">
                          {t.llm.fastSameAsDeep}
                        </option>
                        {fastOptions.map((o) => (
                          <option key={o.id} value={o.id} className="bg-slate-950 text-slate-200">
                            {o.label}
                          </option>
                        ))}
                        <option value="__custom__" className="bg-slate-950 text-slate-400">
                          {providerMeta ? t.catalog.otherModel : t.llm.customModelOption}
                        </option>
                      </select>
                    )}
                    {showFastTyping && (
                      <input
                        type="text"
                        aria-label={t.llm.fastModelName}
                        value={llmModelFast}
                        onChange={(e) => setLlmModelFast(e.target.value)}
                        placeholder={t.llm.fastSameAsDeep}
                        className="w-full bg-slate-950/90 border border-slate-800 rounded-xl px-3 py-2 text-xs text-white placeholder-slate-600 focus:outline-none focus:border-cyan-500/80 font-mono transition-colors"
                      />
                    )}
                  </div>
                  <p className="text-[10px] text-slate-500 mt-1">{t.llm.fastHint}</p>
                </div>
              );
            })()}

            {isCloudProvider(llmProvider) && <UsageOffer onOpen={() => setActiveTab('usage')} saved={savedUsage} />}

            {/* Ollama Model Management (install / delete) */}
            {llmProvider === 'ollama' && (
              <div className="pt-1 space-y-2" data-testid="ollama-model-management">
                <label className="block text-xs font-medium text-slate-300 mb-1.5">
                  {t.models.title}
                </label>
                <div className="flex items-center gap-2">
                  <input
                    type="text"
                    value={pullModelInput}
                    onChange={(e) => setPullModelInput(e.target.value)}
                    onKeyDown={(e) => {
                      if (e.key === 'Enter' && !installing && onSavedServer) handlePullModel();
                    }}
                    placeholder={t.models.installPlaceholder}
                    disabled={installing}
                    aria-label={t.models.installLabel}
                    className="flex-1 bg-slate-950/90 border border-slate-800 rounded-xl px-3 py-2 text-xs text-white placeholder-slate-600 focus:outline-none focus:border-cyan-500/80 font-mono transition-colors disabled:opacity-50"
                  />
                  <Button
                    variant="bordered"
                    onClick={handlePullModel}
                    disabled={installing || !pullModelInput.trim() || !onSavedServer}
                    className="px-3 py-2 rounded-xl bg-slate-800/80 border-slate-700/80 text-slate-200 hover:text-slate-200 shrink-0"
                  >
                    {installing ? (
                      <Loader2 className="w-3.5 h-3.5 animate-spin text-cyan-400" />
                    ) : (
                      <Download className="w-3.5 h-3.5 text-cyan-400" />
                    )}
                    {t.models.install}
                  </Button>
                  {installing && (
                    <Button
                      variant="bordered"
                      onClick={() => pullAbortRef.current?.abort()}
                      title={t.models.cancelInstallTitle}
                      className="px-3 py-2 rounded-xl bg-slate-800/80 border-slate-700/80 text-slate-300 hover:text-slate-200 shrink-0"
                    >
                      {/* Not bare "Cancel" (`settings.models.cancelInstall`): the modal
                          footer already has one, and it closes Settings. */}
                      {t.models.cancelInstall}
                    </Button>
                  )}
                </div>
                {installing && (
                  <p className="text-[11px] text-slate-500">
                    {t.models.installWait}
                  </p>
                )}
                {!onSavedServer && (
                  <p data-testid="settings-models-save-first" className="text-[11px] text-slate-500">
                    {t.models.saveToManage}
                  </p>
                )}
                {savedModels.length > 0 && (
                  <ul className="space-y-1 max-h-32 overflow-y-auto">
                    {savedModels.map((m) => (
                      <li
                        key={m}
                        className="flex items-center justify-between px-2.5 py-1.5 bg-slate-950/60 border border-slate-800 rounded-lg text-[11px] font-mono text-slate-300"
                      >
                        <span className="truncate">{m}</span>
                        <button
                          type="button"
                          onClick={() => handleDeleteModel(m)}
                          disabled={deletingModel !== null}
                          title={fmt(t.models.delete, { model: m })}
                          aria-label={fmt(t.models.delete, { model: m })}
                          className="text-slate-500 hover:text-rose-400 disabled:opacity-50 shrink-0 ml-2"
                        >
                          {deletingModel === m ? (
                            <Loader2 className="w-3.5 h-3.5 animate-spin" />
                          ) : (
                            <Trash2 className="w-3.5 h-3.5" />
                          )}
                        </button>
                      </li>
                    ))}
                  </ul>
                )}
              </div>
            )}

            {/* API Key (if cloud provider or vLLM) */}
            {llmProvider !== 'mock' && llmProvider !== 'ollama' && (() => {
              const keyValidation = providerMeta && llmApiKey ? validateKeyFormat(llmProvider, llmApiKey) : null;
              const keyWarning =
                providerMeta && keyValidation ? describeKeyProblem(keyValidation, providerMeta, t.apiKey) : null;
              // Each provider's key is reported on its own (`providers`), so switching provider
              // shows that provider's key and never asks again for one already saved. A server
              // that predates the list reports the active provider's key only.
              const keyRow = currentSettings?.providers?.find((row) => row.id === llmProvider);
              const isKeyConfiguredForThisProvider = keyRow
                ? keyRow.key_set
                : currentSettings?.llm_provider === llmProvider && Boolean(currentSettings?.llm_api_key_set);
              const maskedKey = keyRow ? keyRow.key_masked : currentSettings?.llm_api_key_masked ?? '';
              const keyFromEnv = keyRow?.key_source === 'env';
              const canRemoveKey = Boolean(keyRow?.key_set) && keyRow?.key_source === 'settings';
              const providerName =
                providerMeta?.displayName ?? PROVIDER_CARDS.find((c) => c.id === llmProvider)?.label ?? llmProvider;
              return (
                <div className="pt-1">
                  <label className="block text-xs font-medium text-slate-300 mb-1.5 flex items-center justify-between">
                    <span className="flex items-center gap-1.5">
                      <Key className="w-3.5 h-3.5 text-amber-400" />
                      {t.apiKey.title}
                      {llmProvider === 'vllm' && (
                        <span className="text-[10px] font-normal text-slate-500" data-testid="vllm-key-optional">
                          {t.apiKey.vllmOptional}
                        </span>
                      )}
                    </span>
                    <div className="flex items-center gap-2">
                      {providerMeta && (
                        <a
                          href={providerMeta.keyConsoleUrl}
                          target="_blank"
                          rel="noopener noreferrer"
                          className="inline-flex items-center gap-1 text-[11px] text-cyan-400 hover:text-cyan-300 transition-colors font-sans"
                          data-testid="provider-key-console-link"
                          title={fmt(t.apiKey.consoleTitle, { name: providerMeta.displayName })}
                        >
                          <span>{fmt(t.apiKey.consoleLink, { name: providerMeta.displayName })}</span>
                          <ExternalLink className="w-3 h-3" />
                        </a>
                      )}
                      {isKeyConfiguredForThisProvider && (
                        <span
                          data-testid="api-key-state"
                          className={`text-[11px] font-mono px-2 py-0.5 rounded border ${
                            keyFromEnv
                              ? 'text-amber-300 bg-amber-950/50 border-amber-800/60'
                              : 'text-emerald-400 bg-emerald-950/60 border-emerald-800/60'
                          }`}
                        >
                          {keyFromEnv
                            ? fmt(t.apiKey.fromEnv, { variable: keyRow?.key_env_var ?? '' })
                            : fmt(t.apiKey.configured, { masked: maskedKey })}
                        </span>
                      )}
                      {canRemoveKey && (
                        <button
                          type="button"
                          onClick={() => handleRemoveKey(llmProvider, providerName)}
                          disabled={removingKey}
                          title={fmt(t.apiKey.removeTitle, { name: providerName })}
                          data-testid="api-key-remove"
                          className="inline-flex items-center gap-1 text-[11px] text-slate-400 hover:text-rose-400 disabled:opacity-50 transition-colors font-sans"
                        >
                          {removingKey ? <Loader2 className="w-3 h-3 animate-spin" /> : <Trash2 className="w-3 h-3" />}
                          {t.apiKey.remove}
                        </button>
                      )}
                    </div>
                  </label>
                  <div className="relative">
                    <input
                      type={showApiKey ? 'text' : 'password'}
                      value={llmApiKey}
                      onChange={(e) => setLlmApiKey(e.target.value)}
                      onBlur={() => setLlmApiKey(sanitizeApiKey(llmApiKey))}
                      placeholder={
                        isKeyConfiguredForThisProvider
                          ? t.apiKey.keepPlaceholder
                          : fmt(t.apiKey.placeholder, { example: providerMeta ? providerMeta.placeholder : 'sk-...' })
                      }
                      className="w-full bg-slate-950/90 border border-slate-800 rounded-xl px-3 py-2 pr-10 text-xs text-white placeholder-slate-600 focus:outline-none focus:border-cyan-500/80 font-mono transition-colors"
                    />
                    <button
                      type="button"
                      onClick={() => setShowApiKey(!showApiKey)}
                      className="absolute right-2.5 top-1/2 -translate-y-1/2 text-slate-500 hover:text-slate-300"
                      title={showApiKey ? t.apiKey.hide : t.apiKey.show}
                    >
                      {showApiKey ? <EyeOff className="w-4 h-4" /> : <Eye className="w-4 h-4" />}
                    </button>
                  </div>
                  {keyWarning && (
                    <div className="mt-1.5 flex items-center gap-1.5 text-[11px] text-amber-400" data-testid="api-key-warning">
                      <AlertTriangle className="w-3.5 h-3.5 shrink-0" />
                      <span>{keyWarning}</span>
                    </div>
                  )}
                  {keyValidation && keyValidation.valid && llmApiKey.trim().length > 0 && (
                    <div className="mt-1.5 flex items-center gap-1.5 text-[11px] text-emerald-400" data-testid="api-key-valid">
                      <CheckCircle2 className="w-3.5 h-3.5 shrink-0" />
                      <span>{fmt(t.apiKey.valid, { name: providerMeta?.displayName ?? '' })}</span>
                    </div>
                  )}
                  {providerMeta && (
                    <div className="mt-1 text-[10px] text-slate-400 font-sans leading-normal">
                      💡 {t.providers.costTips[providerMeta.id]}
                    </div>
                  )}
                </div>
              );
            })()}

            {/* Endpoint override for a cloud provider: optional, so it is one deliberate action
                away rather than an empty field that reads as required (#1631). */}
            {providerMeta && (
              <div className="pt-1" data-testid="settings-endpoint-advanced">
                <button
                  type="button"
                  aria-expanded={endpointOpen}
                  onClick={() => setEndpointOpen((open) => !open)}
                  className="inline-flex items-center gap-1 text-[11px] text-slate-400 hover:text-slate-200 transition-colors"
                >
                  <ChevronRight className={`w-3.5 h-3.5 transition-transform ${endpointOpen ? 'rotate-90' : ''}`} />
                  {t.catalog.advanced}
                </button>
                {endpointOpen && <div className="mt-2">{endpointField}</div>}
              </div>
            )}
          </div>
          )}

          {activeTab === 'all' && <hr className="border-slate-800/80" />}

          {/* Paid model usage limits (llm-token-gateway.md §4.5.1): not behind developer mode */}
          {(activeTab === 'all' || activeTab === 'usage') && <UsageSection onSaved={setSavedUsage} />}

          {activeTab === 'all' && <hr className="border-slate-800/80" />}

          {/* Section 2: Remote GPU & ComfyUI Endpoint */}
          {(activeTab === 'all' || activeTab === 'tools') && (
            <div className="space-y-4">
              {/* What draws a picture, and what can now; the ways below are what it chooses from */}
              <ImagesSection settings={currentSettings} />
              {/* Remote GPU Worker (SSH Auto-Tunnel) Card */}
              <div
                data-testid="settings-remote-gpu-card"
                className="p-4 rounded-xl bg-slate-950/70 border border-slate-800/80 space-y-3"
              >
                <div className="flex items-center justify-between">
                  <label className="text-xs font-semibold uppercase tracking-wider text-slate-400 flex items-center gap-1.5">
                    <Cpu className="w-3.5 h-3.5 text-cyan-400" />
                    {t.remoteGpu.title}
                  </label>
                  <span className="text-[10px] text-slate-500 font-mono">
                    {t.remoteGpu.usedBy}
                  </span>
                </div>

                <p className="text-[11px] text-slate-400 leading-relaxed">
                  {t.remoteGpu.hint}
                </p>

                <div className="flex items-center gap-2">
                  <input
                    type="text"
                    value={remoteHost}
                    onChange={(e) => setRemoteHost(e.target.value)}
                    placeholder={t.remoteGpu.hostPlaceholder}
                    disabled={remoteGpuStatus?.connected || connectingRemote}
                    data-testid="remote-gpu-host-input"
                    aria-label={t.remoteGpu.hostPlaceholder}
                    className="flex-1 bg-slate-900/90 border border-slate-800 rounded-xl px-3 py-2 text-xs text-white placeholder-slate-600 focus:outline-none focus:border-cyan-500/80 font-mono transition-colors disabled:opacity-60"
                  />
                  {remoteGpuStatus?.connected ? (
                    <Button
                      variant="bordered"
                      onClick={handleDisconnectRemoteGpu}
                      disabled={disconnectingRemote}
                      data-testid="remote-gpu-disconnect-button"
                      className="text-xs shrink-0 text-rose-300 hover:text-rose-200 border-rose-800/60 hover:bg-rose-950/50 px-3 py-2 rounded-xl"
                    >
                      {disconnectingRemote ? (
                        <>
                          <Loader2 className="w-3.5 h-3.5 animate-spin mr-1.5" />
                          {t.remoteGpu.disconnecting}
                        </>
                      ) : (
                        t.remoteGpu.disconnect
                      )}
                    </Button>
                  ) : (
                    <Button
                      variant="bordered"
                      onClick={handleConnectRemoteGpu}
                      disabled={connectingRemote || !remoteHost.trim()}
                      data-testid="remote-gpu-connect-button"
                      className="text-xs shrink-0 text-cyan-300 hover:text-cyan-200 border-cyan-800/60 hover:bg-cyan-950/50 px-3 py-2 rounded-xl"
                    >
                      {connectingRemote ? (
                        <>
                          <Loader2 className="w-3.5 h-3.5 animate-spin mr-1.5" />
                          {t.remoteGpu.connecting}
                        </>
                      ) : (
                        t.remoteGpu.connect
                      )}
                    </Button>
                  )}
                </div>

                <label className="flex items-center gap-2 text-[11px] text-slate-300">
                  <input
                    type="checkbox"
                    checked={remoteSyncLlm}
                    onChange={(e) => setRemoteSyncLlm(e.target.checked)}
                    disabled={remoteGpuStatus?.connected || connectingRemote}
                    data-testid="remote-gpu-sync-llm"
                    className="accent-cyan-500"
                  />
                  {t.remoteGpu.useForLlm}
                </label>

                {/* Status indicator and GPU info */}
                {remoteGpuStatus && (
                  <div className="flex flex-wrap items-center gap-2 pt-1" data-testid="remote-gpu-status-info">
                    <Badge
                      tone={remoteGpuStatus.connected ? 'success' : 'neutral'}
                      className="text-[10px]"
                    >
                      <Radio className={`w-2.5 h-2.5 mr-1 ${remoteGpuStatus.connected ? 'text-emerald-400 animate-pulse' : 'text-slate-500'}`} />
                      {remoteGpuStatus.connected ? t.remoteGpu.statusConnected : t.remoteGpu.statusDisconnected}
                    </Badge>
                    {remoteGpuStatus.connected && remoteGpuStatus.gpu && (
                      <span className="text-[11px] text-slate-300 font-mono">
                        {fmt(t.remoteGpu.gpuDetected, {
                          name: remoteGpuStatus.gpu.name,
                          totalMb: String(remoteGpuStatus.gpu.total_mb),
                        })}
                      </span>
                    )}
                    {remoteGpuStatus.connected && remoteGpuStatus.mappings && remoteGpuStatus.mappings.length > 0 && (
                      <span className="text-[11px] text-emerald-400/90 font-mono">
                        {fmt(t.remoteGpu.portsForwarded, {
                          ports: remoteGpuStatus.mappings.map((m) => `${m.service_name} -> :${m.local_port}`).join(', '),
                        })}
                      </span>
                    )}
                    {remoteGpuStatus.connected && remoteGpuStatus.comfyui_autostarted && (
                      <Badge tone="warning" className="text-[10px]">
                        {t.remoteGpu.autoStarted}
                      </Badge>
                    )}
                  </div>
                )}
              </div>

              {/* ComfyUI manual endpoint input */}
              <div className="space-y-3 pt-1">
                <div className="flex items-center justify-between">
                  <label className="text-xs font-semibold uppercase tracking-wider text-slate-400 flex items-center gap-1.5">
                    <ImageIcon className="w-3.5 h-3.5 text-indigo-400" />
                    {t.comfyui.title}
                  </label>
                  <span className="text-[10px] text-slate-500 font-mono">
                    {t.comfyui.usedBy}
                  </span>
                </div>

                <div>
                  <input
                    type="text"
                    value={comfyuiBaseUrl}
                    onChange={(e) => setComfyuiBaseUrl(e.target.value)}
                    placeholder="http://127.0.0.1:8188"
                    className="w-full bg-slate-950/90 border border-slate-800 rounded-xl px-3 py-2 text-xs text-white placeholder-slate-600 focus:outline-none focus:border-cyan-500/80 font-mono transition-colors"
                  />
                  <p className="text-[11px] text-slate-500 mt-1">
                    {t.comfyui.hint}
                  </p>
                </div>
              </div>
            </div>
          )}

          {activeTab === 'all' && <hr className="border-slate-800/80" />}

          {/* Section 3: Folders clones can read */}
          {(activeTab === 'all' || activeTab === 'folders') && (
            <div className="space-y-3" data-testid="settings-read-roots">
              <label className="text-xs font-semibold uppercase tracking-wider text-slate-400 flex items-center gap-1.5">
                <FolderOpen className="w-3.5 h-3.5 text-cyan-400" />
                {t.folders.title}
              </label>

              <div>
                <span className="block text-xs font-medium text-slate-300 mb-1.5">{t.folders.workspace}</span>
                <div
                  className="px-3 py-2 bg-slate-950/60 border border-slate-800 rounded-xl text-xs font-mono text-slate-300 truncate"
                  data-testid="settings-workspace-dir"
                >
                  {currentSettings?.workspace_dir || t.folders.workspaceUnknown}
                </div>
                <p className="text-[11px] text-slate-500 mt-1">
                  {t.folders.workspaceHint}
                </p>
              </div>

              <div className="space-y-2">
                <span className="block text-xs font-medium text-slate-300">{t.folders.others}</span>
                <p className="text-[11px] text-slate-500">
                  {t.folders.othersHint}
                </p>
                {readRoots.length > 0 ? (
                  <ul className="space-y-1" aria-label={t.folders.listLabel}>
                    {readRoots.map((root) => (
                      <li
                        key={root}
                        className="flex items-center justify-between px-2.5 py-1.5 bg-slate-950/60 border border-slate-800 rounded-lg text-[11px] font-mono text-slate-300"
                      >
                        <span className="truncate">{root}</span>
                        <span className="flex items-center gap-2 shrink-0 ml-2">
                          {missingReadRoots.has(root) && (
                            <span className="font-sans text-amber-400" title={t.folders.notUsableTitle}>
                              {t.folders.notUsable}
                            </span>
                          )}
                          <button
                            type="button"
                            onClick={() => setReadRoots(readRoots.filter((r) => r !== root))}
                            title={fmt(t.folders.remove, { root })}
                            aria-label={fmt(t.folders.remove, { root })}
                            className="text-slate-500 hover:text-rose-400"
                          >
                            <Trash2 className="w-3.5 h-3.5" />
                          </button>
                        </span>
                      </li>
                    ))}
                  </ul>
                ) : (
                  <p className="text-[11px] text-slate-400" data-testid="settings-read-roots-empty">
                    {t.folders.empty}
                  </p>
                )}
                {envReadRoots.length > 0 && (
                  <div data-testid="settings-read-roots-env">
                    <p className="text-[11px] text-slate-500">
                      {t.folders.envIntro}
                    </p>
                    <ul className="space-y-1 mt-1" aria-label={t.folders.envListLabel}>
                      {envReadRoots.map((root) => (
                        <li
                          key={root}
                          className="px-2.5 py-1.5 bg-slate-950/40 border border-slate-800 rounded-lg text-[11px] font-mono text-slate-400 truncate"
                        >
                          {root}
                        </li>
                      ))}
                    </ul>
                  </div>
                )}
                {envReadRootsIgnored.map((reason) => (
                  <p key={reason} className="text-[11px] text-amber-400" data-testid="settings-read-roots-env-ignored">
                    {fmt(t.folders.ignored, { reason })}
                  </p>
                ))}
                <div className="flex items-center gap-2">
                  <input
                    type="text"
                    value={newReadRoot}
                    onChange={(e) => setNewReadRoot(e.target.value)}
                    onKeyDown={(e) => {
                      if (e.key === 'Enter') handleAddReadRoot();
                    }}
                    placeholder={t.folders.addPlaceholder}
                    aria-label={t.folders.addLabel}
                    className="flex-1 bg-slate-950/90 border border-slate-800 rounded-xl px-3 py-2 text-xs text-white placeholder-slate-600 focus:outline-none focus:border-cyan-500/80 font-mono transition-colors"
                  />
                  <Button
                    variant="bordered"
                    onClick={handleAddReadRoot}
                    disabled={!newReadRoot.trim()}
                    className="px-3 py-2 rounded-xl bg-slate-800/80 border-slate-700/80 text-slate-200 hover:text-slate-200 shrink-0"
                  >
                    <Plus className="w-3.5 h-3.5 text-cyan-400" />
                    {t.folders.add}
                  </Button>
                </div>
                {readRootsChanged && (
                  <p className="text-[11px] text-slate-500" data-testid="settings-read-roots-unsaved">
                    {t.folders.unsaved}
                  </p>
                )}
              </div>
            </div>
          )}

          {activeTab === 'all' && <hr className="border-slate-800/80" />}

          {/* Section 4: Agents (persona files) */}
          {(activeTab === 'all' || activeTab === 'clones') && (
            <div className="space-y-3" data-testid="settings-personas">
              <label className="text-xs font-semibold uppercase tracking-wider text-slate-400 flex items-center gap-1.5">
                <Users className="w-3.5 h-3.5 text-cyan-400" />
                {copy.personaEditor.sectionTitle}
              </label>
              {personaCatalog || personaLoadError !== null ? (
                <PersonaManager
                  personas={personaCatalog?.personas ?? []}
                  availableTools={personaCatalog?.available_tools ?? []}
                  availableModels={detectedModels}
                  personasDir={personaCatalog ? personaCatalog.personas_dir : null}
                  loadError={personaLoadError}
                  copy={copy.personaEditor}
                  icons={PERSONA_ICONS}
                  onSave={handleSavePersona}
                />
              ) : null}
            </div>
          )}

          {activeTab === 'all' && <hr className="border-slate-800/80" />}

          {/* Section 5: Skills & MCP Tools */}
          {(activeTab === 'all' || activeTab === 'skills') && (
            <div className="space-y-6">
              <SkillsSection />
              <hr className="border-slate-800/80" />
              <McpServersSection />
            </div>
          )}

          {activeTab === 'all' && <hr className="border-slate-800/80" />}

          {/* Section 6: Developer mode & Diagnostics */}
          {(activeTab === 'all' || activeTab === 'diagnostics') && (
            <div className="space-y-6">
              <div className="space-y-2" data-testid="settings-developer-mode">
                <div className="flex items-center justify-between gap-4">
                  <label
                    htmlFor="developer-mode-switch"
                    className="text-xs font-semibold uppercase tracking-wider text-slate-400 flex items-center gap-1.5"
                  >
                    <Wrench className="w-3.5 h-3.5 text-cyan-400" />
                    {t.developerMode.title}
                  </label>
                  <button
                    id="developer-mode-switch"
                    type="button"
                    role="switch"
                    aria-checked={developerMode}
                    data-testid="developer-mode-switch"
                    onClick={() => onDeveloperModeChange(!developerMode)}
                    className={`relative inline-flex h-5 w-9 shrink-0 items-center rounded-full border transition-colors ${
                      developerMode ? 'bg-cyan-700 border-cyan-600' : 'bg-slate-800 border-slate-700'
                    }`}
                  >
                    <span
                      className={`inline-block h-3.5 w-3.5 rounded-full bg-slate-100 transition-transform ${
                        developerMode ? 'translate-x-4' : 'translate-x-0.5'
                      }`}
                    />
                  </button>
                </div>
                <p className="text-[11px] text-slate-500">
                  {t.developerMode.hint}
                </p>
              </div>

              {developerMode && <DiagnosticsSection />}
            </div>
          )}

          {/* Connectivity Test Diagnostics Card */}
          {testResults && (
            <div className="p-3.5 bg-slate-950/80 border border-slate-800 rounded-xl space-y-2.5 text-xs font-mono">
              <span className="text-[11px] font-sans font-bold text-slate-400 uppercase tracking-wider">
                {t.connectivity.title}
              </span>

              {testResults.llm && (
                <div className="flex items-start gap-2">
                  {testResults.llm.status === 'ok' ? (
                    <CheckCircle2 className="w-4 h-4 text-emerald-400 shrink-0 mt-0.5" />
                  ) : testResults.llm.status === 'warning' ? (
                    <AlertTriangle className="w-4 h-4 text-amber-400 shrink-0 mt-0.5" />
                  ) : (
                    <XCircle className="w-4 h-4 text-rose-400 shrink-0 mt-0.5" />
                  )}
                  <div className="flex-1">
                    <span className="font-bold text-slate-300">{fmt(t.connectivity.llm, { provider: testResults.llm.provider ?? '' })}</span>
                    <span
                      className={
                        testResults.llm.status === 'ok'
                          ? 'text-emerald-400'
                          : testResults.llm.status === 'warning'
                          ? 'text-amber-400'
                          : 'text-rose-400'
                      }
                    >
                      {testResults.llm.message ||
                        (testResults.llm.status === 'ok'
                          ? t.connectivity.reachable
                          : testResults.llm.key_refused
                            ? t.connectivity.keyRefused
                            : testResults.llm.error)}
                    </span>
                    {testResults.llm.models && testResults.llm.models.length > 0 && (
                      <p className="text-[10px] text-slate-500 mt-0.5">
                        {fmt(t.connectivity.models, { list: testResults.llm.models.slice(0, 4).join(', ') })}
                        {testResults.llm.models.length > 4 ? fmt(t.connectivity.more, { count: testResults.llm.models.length - 4 }) : ''}
                      </p>
                    )}
                  </div>
                </div>
              )}

              {testResults.comfyui && (
                <div className="flex items-start gap-2">
                  {testResults.comfyui.online ? (
                    <CheckCircle2 className="w-4 h-4 text-emerald-400 shrink-0 mt-0.5" />
                  ) : (
                    <XCircle className="w-4 h-4 text-rose-400 shrink-0 mt-0.5" />
                  )}
                  <div className="flex-1">
                    <span className="font-bold text-slate-300">{t.connectivity.comfyui}</span>
                    <span className={testResults.comfyui.online ? 'text-emerald-400' : 'text-rose-400'}>
                      {testResults.comfyui.online ? t.connectivity.comfyuiOnline : testResults.comfyui.error || t.connectivity.comfyuiUnreachable}
                    </span>
                  </div>
                </div>
              )}
            </div>
          )}
        </div>

        {/* Modal Footer */}
        <div className="px-6 py-4 border-t border-slate-800 bg-slate-950/60 flex items-center justify-between">
          <Button
            variant="bordered"
            onClick={handleTestConnection}
            disabled={testing || saving || loading}
            className="px-3.5 py-1.5 rounded-xl bg-slate-800/80 border-slate-700/80 text-slate-200 hover:text-slate-200"
          >
            {testing ? (
              <Loader2 className="w-3.5 h-3.5 animate-spin text-cyan-400" />
            ) : (
              <RefreshCw className="w-3.5 h-3.5 text-cyan-400" />
            )}
            {t.footer.check}
          </Button>

          <div className="flex items-center gap-2.5">
            <Button
              onClick={onClose}
              className="px-3.5 py-1.5 rounded-xl border-slate-800 text-slate-400 hover:text-white hover:bg-slate-800"
            >
              {t.footer.cancel}
            </Button>
            <button
              type="button"
              onClick={handleSaveSettings}
              disabled={saving || loading}
              className="px-4 py-1.5 rounded-xl bg-gradient-to-r from-cyan-600 to-blue-600 hover:from-cyan-500 hover:to-blue-500 text-white text-xs font-semibold shadow-md shadow-cyan-900/30 flex items-center gap-2 transition-all disabled:opacity-50"
            >
              {saving ? <Loader2 className="w-3.5 h-3.5 animate-spin" /> : null}
              {t.footer.save}
            </button>
          </div>
        </div>
      </div>
    </div>
  );
};
