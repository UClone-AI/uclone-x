import React from 'react';
import { findModel, pickableGroups, silentGroups, type ModelGroup, type ModelSet } from '../../lib/modelGateway';

/** The words a model picker shows. Passed in, so the picker carries none of its own. */
export interface ModelRefSelectWords {
  /** Appended to the recommended model's name. */
  recommendedSuffix: string;
  /** Appended to a saved ref no connection lists now. */
  unavailableSuffix: string;
  /** A group with no model to offer, as one line: its name and why. */
  groupLine: (group: ModelGroup) => string;
}

export interface ModelRefSelectProps {
  id: string;
  label: string;
  testId: string;
  /** The ref chosen, or a leading option's value; `''` is the first leading option. */
  value: string;
  /** The model set, grouped by connection. */
  set: ModelSet;
  /** Choices above the model set: a system default, "Same as conversations", "Automatic". */
  leading: { value: string; label: string }[];
  /** A ref to mark as recommended; shown, never chosen by the picker itself. */
  recommended?: string | null;
  words: ModelRefSelectWords;
  disabled?: boolean;
  onChange: (value: string) => void;
}

/**
 * One model picker over the model set (model-gateway.md §3.7): the leading choices, then each
 * connection's models under its name. A connection that cannot list -- no key, not reachable
 * -- is said in a line under the picker, never filled with a remembered list. The line is
 * chosen by the group's `status`, never its `detail`, which is the Core's English and would put
 * English on a Korean screen (the rule #1631 set for the catalog). A saved ref no
 * connection lists stays selected and is marked unavailable, so the screen never shows a
 * choice the runtime is not using.
 */
export const ModelRefSelect: React.FC<ModelRefSelectProps> = ({
  id,
  label,
  testId,
  value,
  set,
  leading,
  recommended = null,
  words,
  disabled = false,
  onChange,
}) => {
  const isLeading = leading.some((option) => option.value === value);
  const unlisted = !isLeading && value !== '' && findModel(set, value) === null;
  const silent = silentGroups(set);
  return (
    <div className="space-y-1">
      <label htmlFor={id} className="block text-xs font-medium text-slate-300">
        {label}
      </label>
      <select
        id={id}
        data-testid={testId}
        value={value}
        disabled={disabled}
        onChange={(e) => onChange(e.target.value)}
        className="w-full bg-slate-950/90 border border-slate-800 rounded-xl px-3 py-2 text-xs text-white focus:outline-none focus:border-cyan-500/80 transition-colors cursor-pointer disabled:opacity-60"
      >
        {leading.map((option) => (
          <option key={`lead-${option.value}`} value={option.value} className="bg-slate-950 text-slate-300">
            {option.label}
          </option>
        ))}
        {unlisted && (
          <option value={value} className="bg-slate-950 text-amber-300">
            {`${value} ${words.unavailableSuffix}`}
          </option>
        )}
        {pickableGroups(set).map((group) => (
          <optgroup key={group.connection_id} label={group.label}>
            {group.models.map((model) => {
              const name = model.display_name || model.id;
              return (
                <option key={model.ref} value={model.ref} className="bg-slate-950 text-slate-200">
                  {model.ref === recommended ? `${name} ${words.recommendedSuffix}` : name}
                </option>
              );
            })}
          </optgroup>
        ))}
      </select>
      {silent.length > 0 && (
        <ul className="space-y-0.5" data-testid={`${testId}-silent`}>
          {silent.map((group) => (
            <li key={group.connection_id} className="text-[11px] text-slate-500">
              {words.groupLine(group)}
            </li>
          ))}
        </ul>
      )}
    </div>
  );
};
