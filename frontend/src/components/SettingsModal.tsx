import { DiagnosticsPanel } from './DiagnosticsPanel';
import React, { useState, useEffect, useCallback, useRef } from 'react';
import {
  X,
  Settings,
  Server,
  Key,
  Cpu,
  Image as ImageIcon,
  CheckCircle2,
  AlertTriangle,
  XCircle,
  Loader2,
  RefreshCw,
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
} from 'lucide-react';
import { RuntimeSettings, ConnectionTestResult, PersonaCatalog } from '../types';
import { PersonaManager } from './personas/PersonaManager';
import { SkillsSection } from './settings/SkillsSection';
import { McpServersSection } from './settings/McpServersSection';
import { DiagnosticsSection } from './settings/DiagnosticsSection';
import { fetchPersonaCatalog, savePersona } from '../lib/personasApi';
import { PERSONA_EDITOR_COPY } from '../lib/personaCopy';
import type { PersonaDraft, PersonaEditMode, PersonaSaveResult } from '../lib/personaDraft';
import { useEscapeOwner } from '../lib/escapePrecedence';
import { coreReason, failureOf, plainFailure } from '../lib/coreFailure';
import { SETTINGS_FAILURE } from '../lib/settingsCopy';

const PERSONA_ICONS = { add: Plus, edit: Pencil, save: Save, cancel: X, spinner: Loader2 };
import { Badge } from './ui/Badge';
import { Button } from './ui/Button';

/**
 * Whether a rejected `fetch` was cancelled rather than broken.
 *
 * Checked by `name`, not by `instanceof DOMException`: the runtime that rejects an
 * aborted `fetch` is not always the one this module was compiled against, and
 * `name === 'AbortError'` is the part every one of them agrees on. Getting this
 * wrong reports the user's own cancellation back to them as a failure.
 */
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
}

export const SettingsModal: React.FC<SettingsModalProps> = ({
  isOpen,
  onClose,
  onSettingsSaved,
  developerMode,
  onDeveloperModeChange,
}) => {
  const [loading, setLoading] = useState<boolean>(false);
  const [saving, setSaving] = useState<boolean>(false);
  const [testing, setTesting] = useState<boolean>(false);
  const [currentSettings, setCurrentSettings] = useState<RuntimeSettings | null>(null);

  // Form Fields
  const [llmProvider, setLlmProvider] = useState<string>('ollama');
  const [llmBaseUrl, setLlmBaseUrl] = useState<string>('');
  const [llmModel, setLlmModel] = useState<string>('');
  const [isCustomModel, setIsCustomModel] = useState<boolean>(false);
  const [llmApiKey, setLlmApiKey] = useState<string>('');
  const [showApiKey, setShowApiKey] = useState<boolean>(false);
  const [comfyuiBaseUrl, setComfyuiBaseUrl] = useState<string>('http://127.0.0.1:8188');

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
      setLlmModel(data.llm_model || '');
      if (data.available_models && data.available_models.length > 0) {
        setIsCustomModel(!data.available_models.includes(data.llm_model || ''));
      }
      setComfyuiBaseUrl(data.comfyui_base_url || 'http://127.0.0.1:8188');
      setReadRoots(data.read_roots ?? []);
      setNewReadRoot('');
      setLlmApiKey('');
    } catch (err) {
      console.error('Failed to load settings:', err);
      setFeedbackMessage({ type: 'error', text: plainFailure(err, SETTINGS_FAILURE.load) });
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
    const result = await savePersona(draft, mode, PERSONA_EDITOR_COPY);
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
    setLlmProvider(newProvider);
    setTestResults(null);
    if (newProvider === 'vllm') {
      // Deliberately no defaults. A vLLM server is launched per model on a port chosen at
      // the command line, so there is no endpoint or model this panel could pre-fill that
      // is not a guess at somebody else's `vllm serve` arguments — and a guess that is
      // wrong is saved as configuration and then reported as a refused connection or a 404
      // about a model the operator never chose. What another provider left behind is
      // cleared, because `https://api.openai.com/v1` pointing at a local deployment is the
      // one wrong value worse than an empty field.
      //
      // Cleared on any change of provider, not by a list of values: the denylist this
      // replaced named what the panel pre-fills, and the panel can hold more than that —
      // `gpt-4o-mini`, `o1-mini` and `o3-mini` are all in `ui/app.py`'s model list, and an
      // Ollama endpoint on a port other than 11434 is still an Ollama endpoint. Each
      // survived the switch and was saved as `VLLM_MODEL` / `VLLM_BASE_URL`. Re-selecting
      // vLLM changes no provider, so it keeps what the operator typed — this panel is the
      // only source of those two values.
      if (llmProvider !== 'vllm') {
        setLlmBaseUrl('');
        setLlmModel('');
      }
    } else if (newProvider === 'ollama') {
      if (!llmBaseUrl || llmBaseUrl.includes('api.openai.com') || llmBaseUrl.includes('api.anthropic.com')) {
        setLlmBaseUrl('http://localhost:11434');
      }
      if (!llmModel || llmModel === 'gpt-4o' || llmModel.startsWith('claude')) {
        setLlmModel('qwen3:8b');
      }
    } else if (newProvider === 'openai') {
      if (!llmBaseUrl || llmBaseUrl.includes('11434')) {
        setLlmBaseUrl('https://api.openai.com/v1');
      }
      if (!llmModel || llmModel.includes('qwen') || llmModel.startsWith('claude')) {
        setLlmModel('gpt-4o');
      }
    } else if (newProvider === 'anthropic') {
      if (!llmBaseUrl || llmBaseUrl.includes('11434')) {
        setLlmBaseUrl('https://api.anthropic.com');
      }
      if (!llmModel || llmModel.includes('qwen') || llmModel === 'gpt-4o') {
        setLlmModel('claude-3-5-sonnet-20241022');
      }
    } else if (newProvider === 'gemini') {
      if (!llmModel || llmModel.includes('qwen')) {
        setLlmModel('gemini-1.5-pro');
      }
    } else if (newProvider === 'mock') {
      setLlmModel('mock-llm');
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
        llm_api_key: llmApiKey.trim() || undefined,
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
      setFeedbackMessage({ type: 'error', text: plainFailure(err, SETTINGS_FAILURE.test) });
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
    try {
      const payload = {
        llm_provider: llmProvider,
        llm_base_url: llmBaseUrl.trim(),
        llm_model: llmModel.trim(),
        llm_api_key: llmApiKey.trim() || undefined,
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
        text: 'Settings successfully applied! Live agents and tools hot-reloaded.',
      });
      if (onSettingsSaved) {
        onSettingsSaved(data);
      }
    } catch (err) {
      console.error('Failed to save settings:', err);
      setFeedbackMessage({ type: 'error', text: plainFailure(err, SETTINGS_FAILURE.save) });
    } finally {
      setSaving(false);
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
          text: data.joined
            ? `Installed model "${model}" (an install was already running; this joined it).`
            : `Installed model "${model}".`,
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
          text: `Stopped waiting for "${model}". Ollama may still be downloading it.`,
        });
      } else {
        console.error('Failed to install the model:', err);
        setFeedbackMessage({ type: 'error', text: plainFailure(err, SETTINGS_FAILURE.install) });
      }
    } finally {
      if (pullAbortRef.current === controller) pullAbortRef.current = null;
      setInstalling(false);
    }
  };

  const handleDeleteModel = async (model: string) => {
    if (!window.confirm(`Delete model "${model}"? This cannot be undone.`)) return;
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
        setFeedbackMessage({ type: 'success', text: `Deleted model "${model}".` });
        await refreshAvailableModels();
      } else {
        throw await failureOf(res);
      }
    } catch (err) {
      if (isAbortError(err)) {
        setFeedbackMessage({
          type: 'success',
          text: `Stopped waiting for the removal of "${model}".`,
        });
      } else {
        console.error('Failed to delete the model:', err);
        setFeedbackMessage({ type: 'error', text: plainFailure(err, SETTINGS_FAILURE.remove) });
      }
    } finally {
      if (deleteAbortRef.current === controller) deleteAbortRef.current = null;
      setDeletingModel(null);
    }
  };

  const detectedModels = currentSettings?.available_models || testResults?.llm?.models || [];

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
          className="px-6 py-4 border-b border-slate-800 flex items-center justify-between gap-3 bg-slate-950/60"
        >
          <div className="flex items-center gap-2.5 min-w-0 flex-1">
            <div className="p-2 rounded-xl bg-cyan-950/80 text-cyan-400 border border-cyan-800/80 shrink-0">
              <Settings className="w-5 h-5" />
            </div>
            <div className="min-w-0">
              <h2 className="text-base font-bold text-white tracking-tight flex flex-wrap items-center gap-x-2 gap-y-1">
                Runtime Settings & Endpoints
                <Badge tone="info" className="text-[10px] uppercase">
                  Hot-Reload
                </Badge>
              </h2>
              <p className="text-xs text-slate-400">
                Configure live LLM model providers and endpoints without restart.
                {currentSettings?.workspace_dir && (
                  <span className="block text-[11px] font-mono text-cyan-400/90 mt-0.5 truncate">
                    📁 Workspace: {currentSettings.workspace_dir}
                  </span>
                )}
              </p>
            </div>
          </div>
          <Button variant="ghost" size="icon" onClick={onClose} title="Close" className="shrink-0 w-11 h-11 -mr-2">
            <X className="w-5 h-5" />
          </Button>
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

          <DiagnosticsPanel />

          {/* Section 1: LLM Provider */}
          <div className="space-y-4">
            <div className="flex items-center justify-between">
              <label className="text-xs font-semibold uppercase tracking-wider text-slate-400 flex items-center gap-1.5">
                <Server className="w-3.5 h-3.5 text-cyan-400" />
                LLM Provider
              </label>
              {currentSettings && (
                <span className="text-[11px] font-mono text-slate-400">
                  Active: <span className="text-cyan-400 font-semibold">{currentSettings.llm_provider}</span>
                </span>
              )}
            </div>

            {/* Provider Cards */}
            <div className="grid grid-cols-2 sm:grid-cols-3 md:grid-cols-5 gap-2">
              {[
                { id: 'ollama', label: 'Ollama', desc: 'Local Private (Default)' },
                { id: 'vllm', label: 'vLLM', desc: 'Self-hosted GPU server' },
                { id: 'openai', label: 'OpenAI', desc: 'GPT-4o & API' },
                { id: 'anthropic', label: 'Anthropic', desc: 'Claude 3.5' },
                { id: 'gemini', label: 'Gemini', desc: 'Google Flash' },
              ].map((p) => {
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
                    <span className="text-[10px] text-slate-500 mt-1">{p.desc}</span>
                  </button>
                );
              })}
            </div>

            {/* Provider Configuration Fields */}
            <div className="grid grid-cols-1 sm:grid-cols-2 gap-3.5 pt-2">
              <div>
                <label className="block text-xs font-medium text-slate-300 mb-1.5 flex items-center justify-between">
                  <span>Endpoint Base URL</span>
                  <span className="text-[10px] text-slate-500 font-mono">
                    {llmProvider === 'ollama'
                      ? 'http://localhost:11434'
                      : llmProvider === 'vllm'
                      ? 'required — no default'
                      : 'e.g. proxy / host'}
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
                      ? 'http://<host>:8000/v1 — where you ran `vllm serve`'
                      : llmProvider === 'openai'
                      ? 'https://api.openai.com/v1'
                      : 'Endpoint URL'
                  }
                  className="w-full bg-slate-950/90 border border-slate-800 rounded-xl px-3 py-2 text-xs text-white placeholder-slate-600 focus:outline-none focus:border-cyan-500/80 font-mono transition-colors"
                />
              </div>

              <div>
                <label className="block text-xs font-medium text-slate-300 mb-1.5 flex items-center justify-between">
                  <span>Default Model</span>
                  <Cpu className="w-3.5 h-3.5 text-slate-500" />
                </label>
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
                        Custom / Other model...
                      </option>
                    </select>
                    {(isCustomModel || !detectedModels.includes(llmModel)) && (
                      <input
                        type="text"
                        value={llmModel}
                        onChange={(e) => setLlmModel(e.target.value)}
                        placeholder="Enter custom model tag..."
                        className="w-full bg-slate-950/90 border border-slate-800 rounded-xl px-3 py-2 text-xs text-white placeholder-slate-600 focus:outline-none focus:border-cyan-500/80 font-mono transition-colors"
                      />
                    )}
                  </div>
                ) : (
                  <input
                    type="text"
                    value={llmModel}
                    onChange={(e) => setLlmModel(e.target.value)}
                    placeholder="e.g. qwen3:8b, hermes3:8b, gpt-4o"
                    className="w-full bg-slate-950/90 border border-slate-800 rounded-xl px-3 py-2 text-xs text-white placeholder-slate-600 focus:outline-none focus:border-cyan-500/80 font-mono transition-colors"
                  />
                )}
              </div>
            </div>

            {/* Ollama Model Management (install / delete) */}
            {llmProvider === 'ollama' && (
              <div className="pt-1 space-y-2" data-testid="ollama-model-management">
                <label className="block text-xs font-medium text-slate-300 mb-1.5">
                  Manage Local Models
                </label>
                <div className="flex items-center gap-2">
                  <input
                    type="text"
                    value={pullModelInput}
                    onChange={(e) => setPullModelInput(e.target.value)}
                    onKeyDown={(e) => {
                      if (e.key === 'Enter' && !installing) handlePullModel();
                    }}
                    placeholder="e.g. llama3.2:1b"
                    disabled={installing}
                    aria-label="Model to install"
                    className="flex-1 bg-slate-950/90 border border-slate-800 rounded-xl px-3 py-2 text-xs text-white placeholder-slate-600 focus:outline-none focus:border-cyan-500/80 font-mono transition-colors disabled:opacity-50"
                  />
                  <Button
                    variant="bordered"
                    onClick={handlePullModel}
                    disabled={installing || !pullModelInput.trim()}
                    className="px-3 py-2 rounded-xl bg-slate-800/80 border-slate-700/80 text-slate-200 hover:text-slate-200 shrink-0"
                  >
                    {installing ? (
                      <Loader2 className="w-3.5 h-3.5 animate-spin text-cyan-400" />
                    ) : (
                      <Download className="w-3.5 h-3.5 text-cyan-400" />
                    )}
                    Install
                  </Button>
                  {installing && (
                    <Button
                      variant="bordered"
                      onClick={() => pullAbortRef.current?.abort()}
                      title="Stop waiting for this install"
                      className="px-3 py-2 rounded-xl bg-slate-800/80 border-slate-700/80 text-slate-300 hover:text-slate-200 shrink-0"
                    >
                      {/* Not bare "Cancel": the modal footer already has one, and it
                          closes Settings. Two identically labelled buttons on one
                          screen make the user work out which is which — P0. */}
                      Cancel install
                    </Button>
                  )}
                </div>
                {installing && (
                  <p className="text-[11px] text-slate-500">
                    Large models take a while. Cancelling — or closing this window — stops
                    the wait, not the download. Reopen Settings to see whether it arrived.
                  </p>
                )}
                {detectedModels.length > 0 && (
                  <ul className="space-y-1 max-h-32 overflow-y-auto">
                    {detectedModels.map((m) => (
                      <li
                        key={m}
                        className="flex items-center justify-between px-2.5 py-1.5 bg-slate-950/60 border border-slate-800 rounded-lg text-[11px] font-mono text-slate-300"
                      >
                        <span className="truncate">{m}</span>
                        <button
                          type="button"
                          onClick={() => handleDeleteModel(m)}
                          disabled={deletingModel !== null}
                          title={`Delete ${m}`}
                          aria-label={`Delete ${m}`}
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

            {/* API Key (if cloud provider) */}
            {llmProvider !== 'mock' && (
              <div className="pt-1">
                <label className="block text-xs font-medium text-slate-300 mb-1.5 flex items-center justify-between">
                  <span className="flex items-center gap-1.5">
                    <Key className="w-3.5 h-3.5 text-amber-400" />
                    API Key
                    {llmProvider === 'vllm' && (
                      <span className="text-[10px] font-normal text-slate-500" data-testid="vllm-key-optional">
                        optional — only if you started the server with --api-key
                      </span>
                    )}
                  </span>
                  {currentSettings?.llm_api_key_set && (
                    <span className="text-[11px] text-emerald-400 font-mono bg-emerald-950/60 px-2 py-0.5 rounded border border-emerald-800/60">
                      Configured: {currentSettings.llm_api_key_masked}
                    </span>
                  )}
                </label>
                <div className="relative">
                  <input
                    type={showApiKey ? 'text' : 'password'}
                    value={llmApiKey}
                    onChange={(e) => setLlmApiKey(e.target.value)}
                    placeholder={
                      currentSettings?.llm_api_key_set
                        ? 'Leave blank to keep current key, or enter new key'
                        : 'Enter API key (e.g. sk-...)'
                    }
                    className="w-full bg-slate-950/90 border border-slate-800 rounded-xl px-3 py-2 pr-10 text-xs text-white placeholder-slate-600 focus:outline-none focus:border-cyan-500/80 font-mono transition-colors"
                  />
                  <button
                    type="button"
                    onClick={() => setShowApiKey(!showApiKey)}
                    className="absolute right-2.5 top-1/2 -translate-y-1/2 text-slate-500 hover:text-slate-300"
                    title={showApiKey ? 'Hide key' : 'Show key'}
                  >
                    {showApiKey ? <EyeOff className="w-4 h-4" /> : <Eye className="w-4 h-4" />}
                  </button>
                </div>
              </div>
            )}
          </div>

          <hr className="border-slate-800/80" />

          {/* Section 2: ComfyUI Endpoint */}
          <div className="space-y-3">
            <div className="flex items-center justify-between">
              <label className="text-xs font-semibold uppercase tracking-wider text-slate-400 flex items-center gap-1.5">
                <ImageIcon className="w-3.5 h-3.5 text-indigo-400" />
                ComfyUI Image Generation Endpoint
              </label>
              <span className="text-[10px] text-slate-500 font-mono">
                Used by &apos;generate_image&apos; tool
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
                Points to local or LAN GPU workstation running ComfyUI with standard txt2img workflows.
              </p>
            </div>
          </div>

          <hr className="border-slate-800/80" />

          {/* Section 3: Folders clones can read */}
          <div className="space-y-3" data-testid="settings-read-roots">
            <label className="text-xs font-semibold uppercase tracking-wider text-slate-400 flex items-center gap-1.5">
              <FolderOpen className="w-3.5 h-3.5 text-cyan-400" />
              Folders Clones Can Read
            </label>

            <div>
              <span className="block text-xs font-medium text-slate-300 mb-1.5">Workspace folder</span>
              <div
                className="px-3 py-2 bg-slate-950/60 border border-slate-800 rounded-xl text-xs font-mono text-slate-300 truncate"
                data-testid="settings-workspace-dir"
              >
                {currentSettings?.workspace_dir || 'Not reported by the server'}
              </div>
              <p className="text-[11px] text-slate-500 mt-1">
                Clones read and write their files here.
              </p>
            </div>

            <div className="space-y-2">
              <span className="block text-xs font-medium text-slate-300">Other folders (read only)</span>
              <p className="text-[11px] text-slate-500">
                Clones' file tools can open files in these folders but not change or delete
                them. A clone that can run shell commands is not limited by this list.
              </p>
              {readRoots.length > 0 ? (
                <ul className="space-y-1" aria-label="Read-only folders">
                  {readRoots.map((root) => (
                    <li
                      key={root}
                      className="flex items-center justify-between px-2.5 py-1.5 bg-slate-950/60 border border-slate-800 rounded-lg text-[11px] font-mono text-slate-300"
                    >
                      <span className="truncate">{root}</span>
                      <span className="flex items-center gap-2 shrink-0 ml-2">
                        {missingReadRoots.has(root) && (
                          <span className="font-sans text-amber-400" title="Clones cannot read this folder">
                            not usable
                          </span>
                        )}
                        <button
                          type="button"
                          onClick={() => setReadRoots(readRoots.filter((r) => r !== root))}
                          title={`Remove ${root}`}
                          aria-label={`Remove ${root}`}
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
                  No other folders added here.
                </p>
              )}
              {envReadRoots.length > 0 && (
                <div data-testid="settings-read-roots-env">
                  <p className="text-[11px] text-slate-500">
                    Also readable, set by UCLONE_READ_ROOTS when the app started:
                  </p>
                  <ul className="space-y-1 mt-1" aria-label="Read-only folders from the environment">
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
                  Ignored: {reason}
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
                  placeholder="e.g. ~/Documents/notes"
                  aria-label="Folder to add"
                  className="flex-1 bg-slate-950/90 border border-slate-800 rounded-xl px-3 py-2 text-xs text-white placeholder-slate-600 focus:outline-none focus:border-cyan-500/80 font-mono transition-colors"
                />
                <Button
                  variant="bordered"
                  onClick={handleAddReadRoot}
                  disabled={!newReadRoot.trim()}
                  className="px-3 py-2 rounded-xl bg-slate-800/80 border-slate-700/80 text-slate-200 hover:text-slate-200 shrink-0"
                >
                  <Plus className="w-3.5 h-3.5 text-cyan-400" />
                  Add folder
                </Button>
              </div>
              {readRootsChanged && (
                <p className="text-[11px] text-slate-500" data-testid="settings-read-roots-unsaved">
                  Folder changes take effect when you save.
                </p>
              )}
            </div>
          </div>

          <hr className="border-slate-800/80" />

          {/* Section 4: Agents (persona files) */}
          <div className="space-y-3" data-testid="settings-personas">
            <label className="text-xs font-semibold uppercase tracking-wider text-slate-400 flex items-center gap-1.5">
              <Users className="w-3.5 h-3.5 text-cyan-400" />
              {PERSONA_EDITOR_COPY.sectionTitle}
            </label>
            {personaCatalog || personaLoadError !== null ? (
              <PersonaManager
                personas={personaCatalog?.personas ?? []}
                availableTools={personaCatalog?.available_tools ?? []}
                availableModels={detectedModels}
                personasDir={personaCatalog ? personaCatalog.personas_dir : null}
                loadError={personaLoadError}
                copy={PERSONA_EDITOR_COPY}
                icons={PERSONA_ICONS}
                onSave={handleSavePersona}
              />
            ) : null}
          </div>

          <hr className="border-slate-800/80" />

          {/* Section 4: Skills -- the installation's skill catalogue, beside its clones (#1358) */}
          <SkillsSection />

          <hr className="border-slate-800/80" />

          {/* External tool servers (MCP) -- tools clones can use, beside the skills they have */}
          <McpServersSection />

          <hr className="border-slate-800/80" />

          {/* Section 5: Developer mode -- a head preference, applied at once, not saved to the runtime */}
          <div className="space-y-2" data-testid="settings-developer-mode">
            <div className="flex items-center justify-between gap-4">
              <label
                htmlFor="developer-mode-switch"
                className="text-xs font-semibold uppercase tracking-wider text-slate-400 flex items-center gap-1.5"
              >
                <Wrench className="w-3.5 h-3.5 text-cyan-400" />
                Developer mode
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
              Adds a drawer of developer tools to the workspace dock (Knowledge Graph, DAG,
              EventBus and Ontology) and a Diagnostics section below (ACP and Evals). Takes effect
              at once and is kept in this browser.
            </p>
          </div>

          {/* Section 6: Diagnostics -- the build's ACP report and evaluation scorecard, in
              developer mode only, directly under the switch that reveals it (#1358) */}
          {developerMode && <DiagnosticsSection />}

          {/* Connectivity Test Diagnostics Card */}
          {testResults && (
            <div className="p-3.5 bg-slate-950/80 border border-slate-800 rounded-xl space-y-2.5 text-xs font-mono">
              <span className="text-[11px] font-sans font-bold text-slate-400 uppercase tracking-wider">
                Connectivity Test Results
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
                    <span className="font-bold text-slate-300">LLM ({testResults.llm.provider}): </span>
                    <span
                      className={
                        testResults.llm.status === 'ok'
                          ? 'text-emerald-400'
                          : testResults.llm.status === 'warning'
                          ? 'text-amber-400'
                          : 'text-rose-400'
                      }
                    >
                      {testResults.llm.message || (testResults.llm.status === 'ok' ? 'Endpoint Reachable' : testResults.llm.error)}
                    </span>
                    {testResults.llm.models && testResults.llm.models.length > 0 && (
                      <p className="text-[10px] text-slate-500 mt-0.5">
                        Available models: {testResults.llm.models.slice(0, 4).join(', ')}
                        {testResults.llm.models.length > 4 ? ` (+${testResults.llm.models.length - 4} more)` : ''}
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
                    <span className="font-bold text-slate-300">ComfyUI: </span>
                    <span className={testResults.comfyui.online ? 'text-emerald-400' : 'text-rose-400'}>
                      {testResults.comfyui.online ? 'Server Online & Ready' : testResults.comfyui.error || 'Unreachable at configured URL'}
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
            Check Connection
          </Button>

          <div className="flex items-center gap-2.5">
            <Button
              onClick={onClose}
              className="px-3.5 py-1.5 rounded-xl border-slate-800 text-slate-400 hover:text-white hover:bg-slate-800"
            >
              Cancel
            </Button>
            <button
              type="button"
              onClick={handleSaveSettings}
              disabled={saving || loading}
              className="px-4 py-1.5 rounded-xl bg-gradient-to-r from-cyan-600 to-blue-600 hover:from-cyan-500 hover:to-blue-500 text-white text-xs font-semibold shadow-md shadow-cyan-900/30 flex items-center gap-2 transition-all disabled:opacity-50"
            >
              {saving ? <Loader2 className="w-3.5 h-3.5 animate-spin" /> : null}
              Save & Apply Settings
            </button>
          </div>
        </div>
      </div>
    </div>
  );
};
