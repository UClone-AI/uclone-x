import React, { useEffect, useState } from 'react';
import { ImageIcon } from 'lucide-react';
import { fmt, useCopy } from '../../i18n';
import { useApiRead } from '../../lib/useApiRead';
import { Button } from '../ui/Button';
import { ReadFailure, ReadLoading } from './ReadState';

/** The `image_engine` setting's values, in the order they are offered. */
export const IMAGE_CHOICES = ['auto', 'local', 'gemini'] as const;
export type ImageChoice = (typeof IMAGE_CHOICES)[number];

/** The picture model the runtime uses when none is saved. */
export const DEFAULT_IMAGE_MODEL = 'gemini-2.5-flash-image';

/** The engines the status names, each with a line of copy. */
const ENGINE_NAMES = ['remote-cuda', 'comfyui-local', 'diffusers-sdxl', 'gemini'] as const;
type EngineName = (typeof ENGINE_NAMES)[number];
const isEngineName = (value: string): value is EngineName =>
  (ENGINE_NAMES as readonly string[]).includes(value);

const REASONS = [
  'ready',
  'not_configured',
  'unreachable',
  'not_running',
  'missing_dependencies',
  'no_checkpoint',
  'disabled_by_setting',
  'no_key',
  'chat_provider_not_gemini',
] as const;
type Reason = (typeof REASONS)[number] | 'unknown';
/** A reason code the copy has a line for; a code added later reads as "Not available". */
export const reasonOf = (code: string): Reason =>
  (REASONS as readonly string[]).includes(code) ? (code as Reason) : 'unknown';

interface MediaStatus {
  ready: boolean;
  engine: string;
  setting: string;
  engines: { name: string; ready: boolean; reason_code: string }[];
}

const isMediaStatus = (value: unknown): value is MediaStatus =>
  typeof value === 'object' &&
  value !== null &&
  typeof (value as MediaStatus).engine === 'string' &&
  Array.isArray((value as MediaStatus).engines);

/** The part of `GET /api/settings` this section reads. */
export interface ImageSettings {
  image_engine?: unknown;
  image_model?: unknown;
  image_settings_problem?: unknown;
}

const choiceOf = (value: unknown): ImageChoice =>
  (IMAGE_CHOICES as readonly unknown[]).includes(value) ? (value as ImageChoice) : 'auto';

/**
 * Gemini's picture models, from a catalog answer: ids that start with `gemini` and name an
 * image model. Anything else -- the chat models, a listing that failed -- offers none.
 */
export const imageModelsOf = (body: unknown): string[] => {
  const entries = (body as { catalog?: { entries?: unknown } } | null)?.catalog?.entries;
  if (!Array.isArray(entries)) return [];
  const ids: string[] = [];
  for (const entry of entries) {
    const id = (entry as { id?: unknown } | null)?.id;
    if (typeof id === 'string' && id.startsWith('gemini') && id.includes('image') && !ids.includes(id)) {
      ids.push(id);
    }
  }
  return ids;
};

/**
 * Settings › Images: what draws a picture when a clone is asked for one.
 *
 * Three choices -- whatever is ready, only the person's own machines, or Google Gemini -- and,
 * where Gemini may draw, which of its picture models. Under them, in plain words, what draws
 * now and why each other way cannot; and when nothing can, what to do about it. The runtime's
 * own messages and engine ids are never shown: every line is this screen's copy.
 */
export const ImagesSection: React.FC<{
  /** The settings as Settings last read them; `null` until it has. */
  settings: ImageSettings | null;
}> = ({ settings: read }) => {
  const allCopy = useCopy();
  const copy = allCopy.images;
  /** What a save of this section answered, which is newer than the read it was given. */
  const [answered, setAnswered] = useState<ImageSettings | null>(null);
  useEffect(() => setAnswered(null), [read]);
  const settings = answered ?? read;
  const statusRead = useApiRead<unknown>('/api/media/status');
  const [choice, setChoice] = useState<ImageChoice>('auto');
  const [model, setModel] = useState<string>(DEFAULT_IMAGE_MODEL);
  const [savedModel, setSavedModel] = useState<string>(DEFAULT_IMAGE_MODEL);
  const [listed, setListed] = useState<string[] | null>(null);
  const [listFailed, setListFailed] = useState(false);
  const [saving, setSaving] = useState(false);
  const [saveState, setSaveState] = useState<'saved' | 'failed' | null>(null);

  const badSetting =
    settings !== null &&
    typeof settings.image_settings_problem === 'string' &&
    settings.image_settings_problem !== '';

  // A fresh read replaces the form: what is shown is what is saved.
  useEffect(() => {
    if (settings === null) return;
    setChoice(choiceOf(settings.image_engine));
    const saved =
      typeof settings.image_model === 'string' && settings.image_model !== ''
        ? settings.image_model
        : DEFAULT_IMAGE_MODEL;
    setModel(saved);
    setSavedModel(saved);
  }, [settings]);

  // Google's list, read once: when Gemini is chosen, or when the list is opened under
  // `auto`. Not on arrival, so opening Settings does not ask Google for anything.
  const offersModel = choice !== 'local';
  const [wantList, setWantList] = useState(false);
  useEffect(() => {
    if ((choice !== 'gemini' && !wantList) || listed !== null) return;
    let live = true;
    void (async () => {
      try {
        const res = await fetch('/api/models/catalog', {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({ provider: 'gemini' }),
        });
        const ids = res.ok ? imageModelsOf(await res.json()) : [];
        if (!live) return;
        setListed(ids);
        setListFailed(ids.length === 0);
      } catch {
        if (!live) return;
        setListed([]);
        setListFailed(true);
      }
    })();
    return () => {
      live = false;
    };
  }, [choice, wantList, listed]);

  const modelOptions = [DEFAULT_IMAGE_MODEL, savedModel, ...(listed ?? [])].filter(
    (id, index, all) => all.indexOf(id) === index,
  );

  const save = async () => {
    setSaving(true);
    setSaveState(null);
    try {
      const res = await fetch('/api/settings', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ image_engine: choice, image_model: model }),
      });
      if (!res.ok) {
        setSaveState('failed');
        return;
      }
      setSaveState('saved');
      try {
        setAnswered((await res.json()) as ImageSettings);
      } catch {
        // Saved; an answer without a readable body leaves the form as chosen.
      }
      statusRead.reload();
    } catch {
      setSaveState('failed');
    } finally {
      setSaving(false);
    }
  };

  const status = statusRead.data !== null && isMediaStatus(statusRead.data) ? statusRead.data : null;
  // A stored choice the runtime refuses is said as such, not as a failed check.
  const statusFault = badSetting ? null : statusRead.fault;

  return (
    <div className="space-y-4" data-testid="settings-images">
      <label className="text-xs font-semibold uppercase tracking-wider text-slate-400 flex items-center gap-1.5">
        <ImageIcon className="w-3.5 h-3.5 text-cyan-400" />
        {copy.title}
      </label>
      <p className="text-[11px] text-slate-500">{copy.intro}</p>

      <fieldset className="space-y-2" data-testid="settings-images-choice">
        <legend className="text-[11px] text-slate-400 mb-1">{copy.choice.label}</legend>
        {IMAGE_CHOICES.map((id) => (
          <label key={id} className="flex items-start gap-2 text-xs text-slate-200 cursor-pointer">
            <input
              type="radio"
              name="image-engine"
              value={id}
              checked={choice === id}
              onChange={() => {
                setChoice(id);
                setSaveState(null);
              }}
              data-testid={`settings-images-choice-${id}`}
              className="mt-0.5"
            />
            <span>
              {copy.choice[id]}
              <span className="block text-[11px] text-slate-500">{copy.choice[`${id}Hint`]}</span>
            </span>
          </label>
        ))}
      </fieldset>

      {offersModel && (
        <div className="space-y-1">
          <label htmlFor="settings-images-model" className="text-[11px] text-slate-400">
            {copy.model.label}
          </label>
          <select
            id="settings-images-model"
            data-testid="settings-images-model"
            value={model}
            onFocus={() => setWantList(true)}
            onChange={(event) => {
              setModel(event.target.value);
              setSaveState(null);
            }}
            className="block w-full rounded-lg border border-slate-700 bg-slate-900 px-2 py-1.5 text-xs text-slate-200"
          >
            {modelOptions.map((id) => (
              <option key={id} value={id}>
                {id === DEFAULT_IMAGE_MODEL ? fmt(copy.model.default, { model: id }) : id}
              </option>
            ))}
          </select>
          {listFailed && (
            <p className="text-[11px] text-slate-500" data-testid="settings-images-model-list-failed">
              {copy.model.listFailed}
            </p>
          )}
        </div>
      )}

      <div className="flex items-center gap-2">
        <Button
          variant="solid"
          onClick={() => void save()}
          disabled={saving || settings === null}
          data-testid="settings-images-save"
        >
          {saving ? copy.saving : copy.save}
        </Button>
        {saveState === 'saved' && (
          <span role="status" className="text-[11px] text-emerald-400" data-testid="settings-images-saved">
            {copy.saved}
          </span>
        )}
        {saveState === 'failed' && (
          <span role="alert" className="text-[11px] text-rose-300" data-testid="settings-images-save-failed">
            {copy.saveFailed}
          </span>
        )}
      </div>

      {badSetting && (
        <p role="alert" className="text-[11px] text-rose-300" data-testid="settings-images-bad-setting">
          {copy.now.badSetting}
        </p>
      )}
      {statusFault !== null && (
        <ReadFailure
          testId="settings-images-error"
          what={copy.loadFailed}
          cause={allCopy.skills.plainCause[statusFault.kind]}
          plain
          onRetry={statusRead.reload}
          retrying={statusRead.loading}
        />
      )}
      {!badSetting && statusFault === null && status === null && (
        <ReadLoading testId="settings-images-loading">{copy.loading}</ReadLoading>
      )}
      {!badSetting && status !== null && (
        <div className="space-y-2" data-testid="settings-images-status">
          {isEngineName(status.engine) ? (
            <p className="text-xs text-slate-200" data-testid="settings-images-now">
              {fmt(copy.now.drawing, { engine: copy.engines[status.engine] })}
            </p>
          ) : (
            <div role="status" data-testid="settings-images-none" className="space-y-1">
              <p className="text-xs text-amber-200">{copy.now.none}</p>
              <p className="text-[11px] text-slate-400">{copy.now.noneHint}</p>
            </div>
          )}
          <p className="text-[11px] text-slate-400">{copy.enginesLabel}</p>
          <ul className="space-y-1">
            {status.engines.filter((e) => isEngineName(e.name)).map((e) => (
              <li
                key={e.name}
                data-testid={`settings-images-engine-${e.name}`}
                className="flex justify-between gap-3 text-[11px]"
              >
                <span className="text-slate-300">{copy.engines[e.name as EngineName]}</span>
                <span className={e.ready ? 'text-emerald-400' : 'text-slate-500'}>
                  {copy.reasons[reasonOf(e.reason_code)]}
                </span>
              </li>
            ))}
          </ul>
        </div>
      )}
    </div>
  );
};
