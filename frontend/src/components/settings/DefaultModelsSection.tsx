import React, { useEffect, useState } from 'react';
import { Cpu } from 'lucide-react';
import { fmt, useCopy } from '../../i18n';
import { plainFailure } from '../../lib/coreFailure';
import {
  AUTO_IMAGE,
  addConnection,
  modelName,
  saveDefaultModel,
  useModelSet,
  type DefaultSlot,
  type ModelSet,
} from '../../lib/modelGateway';
import { useApiRead } from '../../lib/useApiRead';
import { useAutoSave, type FlushRegistry } from '../../lib/useAutoSave';
import { FieldStatus } from './FieldStatus';
import { ModelRefSelect } from './ModelRefSelect';
import { ReadLoading } from './ReadState';
import { pickerWords } from './gatewayWords';

/**
 * One default-model picker, saved alone as it is chosen (settings-single-source.md §4.1).
 *
 * `value` is the picker's own spelling: `''` is the slot's empty choice ("Not chosen yet",
 * "Same as conversations"), sent as `null`.
 */
const DefaultPicker: React.FC<{
  slot: DefaultSlot;
  set: ModelSet;
  saved: string;
  label: string;
  hint: string;
  leading: { value: string; label: string }[];
  recommended: string | null;
  flushers?: FlushRegistry;
  onSaved: () => void;
}> = ({ slot, set, saved, label, hint, leading, recommended, flushers, onSaved }) => {
  const t = useCopy();
  const g = t.gateway;
  const [value, setValue] = useState(saved);
  useEffect(() => setValue(saved), [saved]);
  const autoSave = useAutoSave<string>({
    saved,
    save: async (next) => {
      await saveDefaultModel(slot, next === '' ? null : next);
      onSaved();
    },
    restore: setValue,
    describeFailure: (err) => `${plainFailure(err, g.defaults.saveFailed)} ${t.settings.autosave.restored}`,
    flushers,
  });
  const choose = (next: string) => {
    setValue(next);
    autoSave.commit(next);
  };
  // Shown, never saved unasked: the slot stays empty until the person picks or presses Use it.
  const offerRecommended = value === '' && slot === 'deep' && recommended !== null;
  return (
    <div data-testid={`settings-default-${slot}`}>
      <ModelRefSelect
        id={`settings-default-${slot}-select`}
        testId={`settings-default-${slot}-select`}
        label={label}
        value={value}
        set={set}
        leading={leading}
        recommended={recommended}
        words={pickerWords(g, g.defaults.recommendedSuffix)}
        onChange={choose}
      />
      {offerRecommended && (
        <p data-testid={`settings-default-${slot}-recommended`} className="mt-1 flex items-center gap-2 text-[11px] text-slate-400">
          <span>{fmt(g.defaults.recommendedHint, { model: modelName(set, recommended) })}</span>
          <button
            type="button"
            onClick={() => choose(recommended)}
            className="rounded-md border border-slate-700 px-2 py-0.5 text-slate-200 hover:bg-slate-800"
          >
            {g.defaults.useRecommended}
          </button>
        </p>
      )}
      <FieldStatus status={autoSave.status} testId={`settings-default-${slot}-status`} />
      <p className="text-[10px] text-slate-500 mt-1">{hint}</p>
    </div>
  );
};

/** Where `GET /api/media/status` says the next picture is drawn (design §3.1). */
const WHERE = ['this_computer', 'gpu_server', 'cloud'] as const;
type Where = (typeof WHERE)[number];

/** The part of `GET /api/media/status` the Pictures line reads. */
interface ResolvedPicture {
  create: { label: string | null; where: string } | null;
  reason_code: string | null;
}

/** A ComfyUI answering on this computer while none is connected, or `null`. */
const detectedOf = (body: unknown): string | null => {
  const found = (body as { detected_comfyui?: unknown } | null)?.detected_comfyui;
  return typeof found === 'string' && found !== '' ? found : null;
};

/**
 * A ComfyUI found running on this computer, offered and never taken (detect, never seize:
 * agent-assisted-installation rung 1). Pressing Add saves it as a `comfyui` connection.
 */
const DetectedComfy: React.FC<{ address: string; onAdded: () => void }> = ({ address, onAdded }) => {
  const g = useCopy().gateway.defaults;
  const [state, setState] = useState<'idle' | 'adding' | 'failed'>('idle');
  const add = async () => {
    setState('adding');
    try {
      await addConnection({ kind: 'comfyui', base_url: address });
      onAdded();
    } catch {
      setState('failed');
    }
  };
  return (
    <div data-testid="settings-default-image-detected" className="flex flex-wrap items-center gap-2 text-[11px] text-slate-300">
      <span>{fmt(g.comfyFound, { address })}</span>
      <button
        type="button"
        onClick={() => void add()}
        disabled={state === 'adding'}
        data-testid="settings-default-image-detected-add"
        className="rounded-md border border-slate-700 px-2 py-0.5 text-slate-200 hover:bg-slate-800 disabled:opacity-60"
      >
        {g.comfyAdd}
      </button>
      {state === 'failed' && <span role="alert" className="text-rose-300">{g.comfyAddFailed}</span>}
    </div>
  );
};

const resolvedOf = (body: unknown): ResolvedPicture | null => {
  const resolved = (body as { resolved?: unknown } | null)?.resolved;
  if (typeof resolved !== 'object' || resolved === null || !('create' in resolved)) return null;
  return resolved as ResolvedPicture;
};

/**
 * What draws the next picture under the Pictures default, read from the Core
 * (`GET /api/media/status`, model-gateway.md §3.5): the model and where it runs, or, when
 * nothing can draw, why and what to do. Each state says itself; none is an empty line.
 */
const PictureNow: React.FC<{ version: number; onConnectionAdded: () => void }> = ({
  version,
  onConnectionAdded,
}) => {
  const g = useCopy().gateway.defaults;
  const status = useApiRead<unknown>('/api/media/status');
  // `reload` is a new function on every render, so it is read through a ref: only a new
  // `version` (a save, a connection change) asks again; the first read is the hook's own.
  const reload = React.useRef(status.reload);
  reload.current = status.reload;
  const first = React.useRef(true);
  useEffect(() => {
    if (first.current) {
      first.current = false;
      return;
    }
    reload.current();
  }, [version]);
  if (status.fault !== null) {
    return (
      <p data-testid="settings-default-image-now-failed" className="text-[11px] text-slate-500">
        {g.imageStatusFailed}
      </p>
    );
  }
  const resolved = resolvedOf(status.data);
  if (resolved === null) return null;
  const create = resolved.create;
  const detected = detectedOf(status.data);
  const offer =
    detected !== null ? (
      <DetectedComfy
        address={detected}
        onAdded={() => {
          onConnectionAdded();
          reload.current();
        }}
      />
    ) : null;
  if (create !== null && (WHERE as readonly string[]).includes(create.where)) {
    const where = g.where[create.where as Where];
    return (
      <>
        <p data-testid="settings-default-image-now" className="text-[11px] text-slate-300">
          {create.label
            ? fmt(g.imageNow, { model: create.label, where })
            : fmt(g.imageNowUnreported, { where })}
        </p>
        {offer}
      </>
    );
  }
  const notReady = resolved.reason_code === 'chosen_not_ready';
  return (
    <div role="status" data-testid="settings-default-image-none" className="space-y-0.5">
      <p className="text-[11px] text-amber-200">{notReady ? g.imageNotReady : g.imageNone}</p>
      {!notReady && <p className="text-[11px] text-slate-400">{g.imageNoneHint}</p>}
      {offer}
    </div>
  );
};

/**
 * Settings › Default models (model-gateway.md §3.7): Conversations (deep), Quick tasks (fast)
 * and Pictures (image), each chosen from the model set grouped by connection. What every clone
 * uses unless it names its own.
 *
 * `version` changes when a connection is added, checked or removed, so the set is read again.
 */
export const DefaultModelsSection: React.FC<{
  version?: number;
  flushers?: FlushRegistry;
  /** A connection was added from here (a ComfyUI found on this computer). */
  onConnectionsChanged?: () => void;
}> = ({ version = 0, flushers, onConnectionsChanged }) => {
  const g = useCopy().gateway;
  // Bumped when the Pictures default is saved, so the line under it is read again.
  const [pictureVersion, setPictureVersion] = useState(0);
  const chat = useModelSet('chat', true, version);
  const image = useModelSet('image', true, version);
  const failed = chat.failed !== null || image.failed !== null;
  const reload = () => {
    chat.reload();
    image.reload();
  };
  const noGroups = chat.set !== null && chat.set.groups.length === 0;

  return (
    <div className="space-y-4" data-testid="settings-default-models">
      <h3 className="text-xs font-semibold uppercase tracking-wider text-slate-400 flex items-center gap-1.5">
        <Cpu className="w-3.5 h-3.5 text-cyan-400" />
        {g.defaults.title}
      </h3>
      <p className="text-[11px] text-slate-500">{g.defaults.intro}</p>

      {failed && (
        <div data-testid="settings-default-models-error" role="alert" className="flex items-center gap-2 text-[11px] text-rose-300">
          <span>{g.defaults.loadFailed}</span>
          <button type="button" onClick={reload} className="underline text-slate-300 hover:text-white">
            {g.connections.retry}
          </button>
        </div>
      )}
      {!failed && (chat.set === null || image.set === null) && (
        <ReadLoading testId="settings-default-models-loading">{g.defaults.loading}</ReadLoading>
      )}
      {noGroups && (
        <p data-testid="settings-default-models-no-connection" className="text-[11px] text-slate-400">
          {g.defaults.noConnection}
        </p>
      )}

      {chat.set !== null && (
        <>
          <DefaultPicker
            slot="deep"
            set={chat.set}
            saved={chat.set.defaults.deep ?? ''}
            label={g.defaults.deep}
            hint={g.defaults.deepHint}
            leading={[{ value: '', label: g.defaults.notChosen }]}
            recommended={chat.set.recommended.deep}
            flushers={flushers}
            onSaved={() => chat.reload()}
          />
          <DefaultPicker
            slot="fast"
            set={chat.set}
            saved={chat.set.defaults.fast ?? ''}
            label={g.defaults.fast}
            hint={g.defaults.fastHint}
            leading={[{ value: '', label: g.defaults.sameAsDeep }]}
            recommended={chat.set.recommended.fast}
            flushers={flushers}
            onSaved={() => chat.reload()}
          />
        </>
      )}
      {image.set !== null && (
        <DefaultPicker
          slot="image"
          set={image.set}
          saved={image.set.defaults.image || AUTO_IMAGE}
          label={g.defaults.image}
          hint={g.defaults.imageHint}
          leading={[{ value: AUTO_IMAGE, label: g.defaults.auto }]}
          recommended={null}
          flushers={flushers}
          onSaved={() => {
            image.reload();
            setPictureVersion((v) => v + 1);
          }}
        />
      )}
      {/* Outside the picker's condition: it stays mounted while the set is read again, so a
          save must ask it to read what draws again. */}
      <PictureNow
        version={version + pictureVersion}
        onConnectionAdded={() => {
          onConnectionsChanged?.();
          image.reload();
        }}
      />
    </div>
  );
};
