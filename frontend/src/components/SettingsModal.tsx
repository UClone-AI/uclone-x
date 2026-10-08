import { DiagnosticsPanel } from './DiagnosticsPanel';
import { FieldStatus } from './settings/FieldStatus';
import React, { useState, useEffect, useCallback, useRef } from 'react';
import {
  X,
  Settings,
  Server,
  Gauge,
  Cpu,
  Image as ImageIcon,
  CheckCircle2,
  AlertTriangle,
  Loader2,
  Radio,
  Plus,
  Pencil,
  Save,
  Users,
  Trash2,
  Wrench,
  FolderOpen,
  Globe,
  Link2,
  AlertCircle,
} from 'lucide-react';
import { RuntimeSettings, PersonaCatalog, type RemoteGpuStatus } from '../types';
import { ConnectionsSection } from './settings/ConnectionsSection';
import { DefaultModelsSection } from './settings/DefaultModelsSection';
import { useModelSet } from '../lib/modelGateway';
import { PersonaManager } from './personas/PersonaManager';
import { SkillsSection } from './settings/SkillsSection';
import { McpServersSection } from './settings/McpServersSection';
import { DiagnosticsSection } from './settings/DiagnosticsSection';
import { UsageSection } from './settings/UsageSection';
import { LinksSection } from './settings/LinksSection';
import { BrowserSection } from './settings/BrowserSection';
import { UsageOffer } from './settings/UsageOffer';
import type { UsageReport } from '../lib/usage';
import { fetchPersonaCatalog, savePersona } from '../lib/personasApi';
import type { PersonaDraft, PersonaEditMode, PersonaSaveResult } from '../lib/personaDraft';
import { useEscapeOwner } from '../lib/escapePrecedence';
import { useAutoSave } from '../lib/useAutoSave';
import { coreReason, failureOf, plainFailure } from '../lib/coreFailure';
import { LANGUAGE_AUTONYMS, UI_LANGUAGES, fmt, useCopy, useLocale, type UiLanguage } from '../i18n';

const PERSONA_ICONS = { add: Plus, edit: Pencil, save: Save, cancel: X, spinner: Loader2 };
import { Badge } from './ui/Badge';
import { Button } from './ui/Button';

export type SettingsTabId =
  | 'all'
  | 'llm'
  | 'usage'
  | 'tools'
  | 'folders'
  | 'clones'
  | 'skills'
  | 'links'
  | 'browser'
  | 'diagnostics';

interface SettingsTabItem {
  id: SettingsTabId;
  icon: React.ComponentType<{ className?: string }>;
}

/** The tabs in order. Each label is `settings.tabs[id]` in the catalog. */
const SETTINGS_TABS: SettingsTabItem[] = [
  { id: 'all', icon: Settings },
  { id: 'llm', icon: Server },
  { id: 'usage', icon: Gauge },
  { id: 'tools', icon: Cpu },
  { id: 'folders', icon: FolderOpen },
  { id: 'clones', icon: Users },
  { id: 'skills', icon: Wrench },
  { id: 'links', icon: Link2 },
  { id: 'browser', icon: Globe },
  { id: 'diagnostics', icon: CheckCircle2 },
];

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
   * would not need it (ui-authoring §3). So it is not part of `RuntimeSettings` and is not
   * sent to the server -- the switch applies at once.
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
  const [currentSettings, setCurrentSettings] = useState<RuntimeSettings | null>(null);
  const [activeTab, setActiveTab] = useState<SettingsTabId>('all');
  // Bumped whenever a model connection is added, checked, changed or removed, so every model
  // picker on the screen reads the model set again (model-gateway.md §3.7).
  const [gatewayVersion, setGatewayVersion] = useState<number>(0);
  const bumpGateway = useCallback(() => setGatewayVersion((v) => v + 1), []);
  // Raised when the GPU computer's connect or disconnect changed the connections itself.
  const [connectionsVersion, setConnectionsVersion] = useState<number>(0);
  const connectionsChangedHere = useCallback(() => {
    setConnectionsVersion((v) => v + 1);
    bumpGateway();
  }, [bumpGateway]);
  // Empty until the person names a computer: there is no host every install shares.
  const [remoteHost, setRemoteHost] = useState<string>('');
  const [remoteHostPresets, setRemoteHostPresets] = useState<string[]>([]);
  // Off unless ticked: a worker connected for images keeps the LLM where it was.
  const [remoteSyncLlm, setRemoteSyncLlm] = useState<boolean>(false);
  const [remoteGpuStatus, setRemoteGpuStatus] = useState<RemoteGpuStatus | null>(null);
  const [connectingRemote, setConnectingRemote] = useState<boolean>(false);
  const [disconnectingRemote, setDisconnectingRemote] = useState<boolean>(false);
  const [remoteConnectError, setRemoteConnectError] = useState<{ message: string; rawError?: string } | null>(null);

  // Extra folders clones may read (never write). Saved on each add and remove, alone: the
  // server re-checks every entry and refuses the list when one has gone missing, so it is
  // never sent with another field that such a refusal would take down with it.
  const [readRoots, setReadRoots] = useState<string[]>([]);
  const [newReadRoot, setNewReadRoot] = useState<string>('');
  const [feedbackMessage, setFeedbackMessage] = useState<{
    type: 'success' | 'error';
    text: string;
  } | null>(null);


  // Settings are saved as they are changed (settings-single-source.md §4.1): each field sends
  // only its own key, so a refusal of one never takes another down with it.
  const onSavedRef = useRef(onSettingsSaved);
  onSavedRef.current = onSettingsSaved;
  const postSettings = useCallback(async (changes: Record<string, unknown>): Promise<RuntimeSettings> => {
    const res = await fetch('/api/settings', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(changes),
    });
    if (!res.ok) throw await failureOf(res);
    const data: RuntimeSettings = await res.json();
    setCurrentSettings(data);
    onSavedRef.current?.(data);
    return data;
  }, []);
  const describeSaveFailure = useCallback(
    (err: unknown) =>
      `${plainFailure(err, copyRef.current.settings.failure.save)} ${copyRef.current.settings.autosave.restored}`,
    [],
  );
  // Every field's pending edit, saved when the modal closes.
  const flushers = useRef(new Set<() => void>());

  const readRootsSave = useAutoSave<string[]>({
    saved: currentSettings?.read_roots ?? [],
    save: async (roots) => {
      const data = await postSettings({ read_roots: roots });
      if (data.read_roots) setReadRoots(data.read_roots);
    },
    restore: setReadRoots,
    describeFailure: describeSaveFailure,
    same: (a, b) => a.length === b.length && a.every((root, i) => root === b[i]),
  });

  const handleClose = useCallback(() => {
    flushers.current.forEach((flush) => flush());
    onClose();
  }, [onClose]);

  const fetchCurrentSettings = useCallback(async () => {
    setFeedbackMessage(null);
    try {
      const res = await fetch('/api/settings');
      if (!res.ok) throw await failureOf(res);
      const data: RuntimeSettings = await res.json();
      setCurrentSettings(data);
      setReadRoots(data.read_roots ?? []);
      setNewReadRoot('');

      try {
        const gpuRes = await fetch('/api/settings/remote-gpu/status');
        if (gpuRes.ok) {
          const gpuData: RemoteGpuStatus = await gpuRes.json();
          setRemoteGpuStatus(gpuData);
          if (gpuData.host) {
            setRemoteHost(gpuData.host);
          }
          // A tunnel that died removed the connections it had added; read them again.
          if (gpuData.restored_settings) connectionsChangedHere();
        }
      } catch {
        // Non-blocking status check
      }

      try {
        const hostsRes = await fetch('/api/settings/remote-gpu/hosts');
        if (hostsRes.ok) {
          const hostsData = await hostsRes.json();
          if (Array.isArray(hostsData.hosts)) {
            setRemoteHostPresets(hostsData.hosts);
          }
        }
      } catch {
        // Non-blocking hosts check
      }
    } catch (err) {
      console.error('Failed to load settings:', err);
      setFeedbackMessage({ type: 'error', text: plainFailure(err, copyRef.current.settings.failure.load) });
    }
  }, [connectionsChangedHere]);

  useEffect(() => {
    if (isOpen) {
      fetchCurrentSettings();
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
  useEscapeOwner('dialog', isOpen, handleClose);


  const missingReadRoots = new Set(currentSettings?.read_roots_missing ?? []);
  const envReadRoots = currentSettings?.read_roots_env ?? [];
  const envReadRootsIgnored = currentSettings?.read_roots_env_ignored ?? [];

  const handleAddReadRoot = () => {
    const root = newReadRoot.trim();
    if (!root) return;
    setNewReadRoot('');
    if (readRoots.includes(root)) return;
    const next = [...readRoots, root];
    setReadRoots(next);
    readRootsSave.commit(next);
  };

  const handleRemoveReadRoot = (root: string) => {
    const next = readRoots.filter((r) => r !== root);
    setReadRoots(next);
    readRootsSave.commit(next);
  };

  const handleConnectRemoteGpu = async (targetHost?: string, syncLlmOverride?: boolean) => {
    const hostToUse = (typeof targetHost === 'string' ? targetHost : remoteHost).trim();
    if (!hostToUse) return;
    const syncLlmToUse = syncLlmOverride !== undefined ? syncLlmOverride : remoteSyncLlm;
    setConnectingRemote(true);
    setRemoteConnectError(null);
    setFeedbackMessage(null);
    try {
      const res = await fetch('/api/settings/remote-gpu/connect', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          host: hostToUse,
          apply_settings: true,
          auto_start_comfyui: true,
          sync_llm: syncLlmToUse,
        }),
      });
      const data = await res.json();
      if (data.status === 'ok' && data.connected) {
        setRemoteConnectError(null);
        setRemoteGpuStatus({
          ...data.tunnel,
          llm_on_remote: data.llm_on_remote,
          images_on_remote: data.images_on_remote,
        });
        if (targetHost) setRemoteHost(hostToUse);
        if (syncLlmOverride !== undefined) setRemoteSyncLlm(syncLlmToUse);
        setRemoteHostPresets((prev) => [hostToUse, ...prev.filter((h) => h !== hostToUse)]);
        await refreshAvailableModels();
        connectionsChangedHere(); // what the connect added
        const connectedText = `${t.remoteGpu.statusConnected}: ${data.tunnel.host}${data.tunnel.gpu ? ` (${data.tunnel.gpu.name})` : ''}`;
        setFeedbackMessage({
          type: 'success',
          text:
            data.llm_skipped === 'model_not_on_remote'
              ? `${connectedText}. ${t.remoteGpu.llmModelMissing}`
              : connectedText,
        });
      } else {
        const plainMsg = fmt(t.remoteGpu.connectFailedHost, { host: hostToUse });
        setRemoteConnectError({
          message: plainMsg,
          rawError: data.error || undefined,
        });
        setFeedbackMessage({
          type: 'error',
          text: plainMsg,
        });
      }
    } catch (err) {
      console.error('Failed to connect to the GPU computer:', err);
      const plainMsg = fmt(t.remoteGpu.connectFailedHost, { host: hostToUse });
      setRemoteConnectError({
        message: plainMsg,
        rawError: err instanceof Error ? err.message : String(err),
      });
      setFeedbackMessage({ type: 'error', text: plainMsg });
    } finally {
      setConnectingRemote(false);
    }
  };

  const handleDisconnectRemoteGpu = async () => {
    setDisconnectingRemote(true);
    setRemoteConnectError(null);
    setFeedbackMessage(null);
    try {
      const res = await fetch('/api/settings/remote-gpu/disconnect', {
        method: 'POST',
      });
      const data = await res.json();
      if (data.status === 'ok') {
        setRemoteGpuStatus(data.tunnel || null);
        await refreshAvailableModels();
        connectionsChangedHere(); // what the disconnect removed
        setFeedbackMessage({
          type: 'success',
          text: t.remoteGpu.statusDisconnected,
        });
      } else {
        setFeedbackMessage({ type: 'error', text: t.remoteGpu.disconnectFailed });
      }
    } catch (err) {
      console.error('Failed to disconnect from the GPU computer:', err);
      setFeedbackMessage({ type: 'error', text: t.remoteGpu.disconnectFailed });
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
  const handleLanguageChange = (next: UiLanguage) => {
    setFeedbackMessage(null);
    locale.setChoice(next).catch((err) => {
      console.error('Failed to save the language:', err);
      setFeedbackMessage({ type: 'error', text: t.language.saveFailed });
    });
  };

  // The clone editor's pickers choose from the model set (model-gateway.md §3.7), read while
  // the clones are on screen and again after a connection changes.
  const clonesShown = isOpen && (activeTab === 'all' || activeTab === 'clones');
  const chatModels = useModelSet('chat', clonesShown, gatewayVersion);
  const imageModels = useModelSet('image', clonesShown, gatewayVersion);

  const remoteGpuCard = (
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

      <div className="space-y-1.5">
        <div className="flex items-center gap-2">
          <input
            type="text"
            value={remoteHost}
            onChange={(e) => {
              setRemoteHost(e.target.value);
              setRemoteConnectError(null);
            }}
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
              onClick={() => handleConnectRemoteGpu()}
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

        {remoteConnectError && (
          <div
            data-testid="remote-gpu-connect-error"
            className="p-3 rounded-xl bg-rose-950/30 border border-rose-900/50 text-xs space-y-1.5"
          >
            <div className="flex items-center gap-1.5 text-rose-300 font-medium">
              <AlertCircle className="w-3.5 h-3.5 text-rose-400 shrink-0" />
              <span>{remoteConnectError.message}</span>
            </div>
            {remoteConnectError.rawError && (
              <details className="text-[11px] text-slate-400" data-testid="remote-gpu-raw-error-details">
                <summary className="cursor-pointer hover:text-slate-300 select-none text-[10px] font-mono text-slate-500">
                  {t.remoteGpu.errorDetails || 'Details'}
                </summary>
                <pre className="mt-1.5 p-2 rounded bg-slate-950 border border-slate-800 text-[10px] font-mono text-rose-300/90 whitespace-pre-wrap break-all max-h-32 overflow-y-auto">
                  {remoteConnectError.rawError}
                </pre>
              </details>
            )}
          </div>
        )}

        {remoteHostPresets.length > 0 && (
          <div className="flex items-center gap-1.5 flex-wrap" data-testid="remote-gpu-presets">
            <span className="text-[10px] text-slate-500 font-mono">{t.remoteGpu.presets}</span>
            {remoteHostPresets.map((preset) => (
              <button
                key={preset}
                type="button"
                disabled={remoteGpuStatus?.connected || connectingRemote}
                onClick={() => {
                  setRemoteHost(preset);
                  setRemoteConnectError(null);
                }}
                data-testid={`remote-gpu-preset-${preset}`}
                className="px-2 py-0.5 rounded-md text-[10px] font-mono bg-slate-900 hover:bg-slate-800 text-cyan-300 border border-slate-800 hover:border-cyan-500/50 transition-colors disabled:opacity-50 cursor-pointer"
              >
                {preset}
              </button>
            ))}
          </div>
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

      {/* Remote Routing Status */}
      {remoteGpuStatus?.connected && (remoteGpuStatus.llm_on_remote || remoteGpuStatus.images_on_remote) && (
        <div
          data-testid="remote-gpu-routing-status"
          className="p-3 rounded-xl bg-slate-900/80 border border-slate-800 text-xs space-y-1.5"
        >
          <div className="flex flex-wrap items-center gap-2 text-slate-200">
            {remoteGpuStatus.llm_on_remote && (
              <span
                data-testid="remote-gpu-routing-llm"
                className="inline-flex items-center gap-1.5 px-2.5 py-1 rounded-lg bg-cyan-950/80 border border-cyan-800/60 text-cyan-300 font-medium"
              >
                <Cpu className="w-3.5 h-3.5 text-cyan-400" />
                {fmt(t.remoteGpu.routingLlm, { host: remoteGpuStatus.host || remoteHost || 'remote' })}
              </span>
            )}
            {remoteGpuStatus.images_on_remote && (
              <span
                data-testid="remote-gpu-routing-images"
                className="inline-flex items-center gap-1.5 px-2.5 py-1 rounded-lg bg-indigo-950/80 border border-indigo-800/60 text-indigo-300 font-medium"
              >
                <ImageIcon className="w-3.5 h-3.5 text-indigo-400" />
                {fmt(t.remoteGpu.routingImages, { host: remoteGpuStatus.host || remoteHost || 'remote' })}
              </span>
            )}
          </div>
          <p className="text-[10px] text-slate-400 leading-normal">
            {t.remoteGpu.routingNotice}
          </p>
        </div>
      )}

      {/* Status indicator and GPU info */}
      {remoteGpuStatus && (
        <div className="flex flex-wrap items-center gap-2 pt-1" data-testid="remote-gpu-status-info">
          <Badge
            tone={remoteGpuStatus.connected ? 'success' : 'neutral'}
            className="text-[10px]"
          >
            <Radio className={`w-2.5 h-2.5 mr-1 ${remoteGpuStatus.connected ? 'text-emerald-400' : 'text-slate-500'}`} />
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
              {fmt(t.remoteGpu.servicesReached, {
                services: remoteGpuStatus.mappings
                  .map((m) => fmt(t.remoteGpu.serviceAt, { service: m.service_name, port: String(m.local_port) }))
                  .join(', '),
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
              <h2 className="text-base font-bold text-white tracking-tight">
                {t.header.title}
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
          <Button variant="ghost" size="icon" onClick={handleClose} title={t.header.close} className="shrink-0 w-11 h-11 -mr-2">
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

          {/* A Settings save kept an unreadable settings file aside (#1860). On every tab: the
              person otherwise finds their API keys gone and nothing saying why (P6). No path
              and no cause -- the page's own words, in the person's language. */}
          {currentSettings?.settings_set_aside && (
            <div
              data-testid="settings-set-aside"
              role="status"
              className="p-3.5 rounded-xl border flex items-start gap-2.5 text-xs bg-amber-950/40 border-amber-800/70 text-amber-200"
            >
              <AlertTriangle className="w-4 h-4 text-amber-400 shrink-0 mt-0.5" />
              <div className="flex-1">{t.setAside.notice}</div>
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

          {/* Section 1: Models -- every connection, and the default models chosen from them
              (model-gateway.md §3.7). */}
          {(activeTab === 'all' || activeTab === 'llm') && (
            <div className="space-y-6" data-testid="settings-models">
              <ConnectionsSection
                onChanged={bumpGateway}
                version={connectionsVersion}
                paidOffer={<UsageOffer onOpen={() => setActiveTab('usage')} saved={savedUsage} />}
              />
              <DefaultModelsSection
                version={gatewayVersion}
                flushers={flushers}
                onConnectionsChanged={connectionsChangedHere}
              />
            </div>
          )}

          {activeTab === 'all' && <hr className="border-slate-800/80" />}

          {/* Paid model usage limits (llm-token-gateway.md §4.5.1): not behind developer mode */}
          {(activeTab === 'all' || activeTab === 'usage') && <UsageSection onSaved={setSavedUsage} />}

          {activeTab === 'all' && <hr className="border-slate-800/80" />}

          {/* Section 2: the GPU computer. Its ComfyUI and Ollama become connections under
              Models, where the picture model is chosen (model-gateway.md §3.5). */}
          {(activeTab === 'all' || activeTab === 'tools') && (
            <div className="space-y-4">{remoteGpuCard}</div>
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
                            onClick={() => handleRemoveReadRoot(root)}
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
                <FieldStatus status={readRootsSave.status} testId="settings-read-roots-status" />
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
                  baseTools={personaCatalog?.base_tools ?? []}
                  writeTools={personaCatalog?.write_tools ?? []}
                  chatModels={chatModels.set}
                  imageModels={imageModels.set}
                  modelsFailed={chatModels.failed !== null || imageModels.failed !== null}
                  developerMode={developerMode}
                  modelCopy={copy.gateway}
                  personasDir={personaCatalog ? personaCatalog.personas_dir : null}
                  loadError={personaLoadError}
                  copy={copy.personaEditor}
                  language={locale.language}
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

          {/* Section 5b: Connections -- uClone2 links (uclone2-link.md §3.5) */}
          {(activeTab === 'all' || activeTab === 'links') && <LinksSection />}

          {activeTab === 'all' && <hr className="border-slate-800/80" />}

          {/* Section 5c: Browser -- connect your Chrome (browser-agent.md §3.6) */}
          {(activeTab === 'all' || activeTab === 'browser') && <BrowserSection />}

          {activeTab === 'all' && <hr className="border-slate-800/80" />}

          {/* Section 6: Developer mode & Diagnostics */}
          {(activeTab === 'all' || activeTab === 'diagnostics') && (
            <div className="space-y-6">
              {/* Problem reporting is for everyone; the rest of Diagnostics waits for developer
                  mode. One place for both, so Diagnostics is found once. */}
              <DiagnosticsPanel />
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

        </div>

        {/* Modal Footer */}
        <div className="px-6 py-4 border-t border-slate-800 bg-slate-950/60 flex items-center justify-end">
          {/* Nothing to save here: each field saves as it is changed. Closing saves a field
              still being typed in. */}
          <Button
            variant="bordered"
            onClick={handleClose}
            data-testid="settings-close"
            className="px-4 py-1.5 rounded-xl bg-slate-800/80 border-slate-700/80 text-slate-200 hover:text-white"
          >
            {t.footer.close}
          </Button>
        </div>
      </div>
    </div>
  );
};