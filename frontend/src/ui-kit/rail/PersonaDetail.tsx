import React from 'react';
import type { PersonaDetailCopy, RailPersona } from './types';

interface PersonaDetailProps {
  persona: RailPersona;
  copy: PersonaDetailCopy;
}

/**
 * Read-only detail for one persona (#1056, scope (a)): every field the persona listing returns,
 * including the two that are safety boundaries rather than cosmetics -- whether the persona may
 * write files and whether it may spawn sub-agents.
 *
 * Inspection only: nothing here can change a persona. Label above value, `text-[10px]
 * uppercase font-sans` section headers, as the head's retired `ToolDetail` drawer laid them out.
 */
export const PersonaDetail: React.FC<PersonaDetailProps> = ({ persona, copy }) => {
  return (
    <div
      data-testid={`persona-detail-${persona.name}`}
      className="mt-1 mb-1.5 p-2.5 rounded-xl bg-slate-900/70 border border-slate-800/60 space-y-2 text-xs"
    >
      {persona.description ? (
        <p className="text-slate-300 leading-relaxed">{persona.description}</p>
      ) : null}

      <div className="grid grid-cols-2 gap-x-3 gap-y-1.5 font-mono">
        <div>
          <div className="text-[10px] text-slate-500 uppercase font-sans">{copy.model}</div>
          <div className="text-slate-300 truncate" title={persona.model_name}>
            {persona.model_name || copy.notSet}
          </div>
        </div>
        <div>
          <div className="text-[10px] text-slate-500 uppercase font-sans">{copy.temperature}</div>
          <div className="text-slate-300">
            {persona.temperature ?? copy.notSet}
          </div>
        </div>
        <div>
          <div className="text-[10px] text-slate-500 uppercase font-sans">{copy.maxTokens}</div>
          <div className="text-slate-300">{persona.max_tokens ?? copy.notSet}</div>
        </div>
      </div>

      <div>
        <div className="text-[10px] text-slate-500 uppercase font-sans mb-1">{copy.tools}</div>
        {persona.allowed_tools && persona.allowed_tools.length > 0 ? (
          <div className="flex flex-wrap gap-1" data-testid={`persona-tools-${persona.name}`}>
            {persona.allowed_tools.map((tool) => (
              <span
                key={tool}
                className="px-1.5 py-0.5 rounded bg-slate-800/80 text-slate-300 text-[10px] font-mono"
              >
                {tool}
              </span>
            ))}
          </div>
        ) : (
          <p
            data-testid={`persona-tools-empty-cause-${persona.name}`}
            className="text-slate-500 text-[11px] leading-relaxed"
          >
            {copy.noTools}
          </p>
        )}
      </div>

      <div className="space-y-1 pt-0.5">
        <div
          data-testid={`persona-write-access-${persona.name}`}
          className={`text-[11px] ${persona.enable_write_tools ? 'text-amber-300' : 'text-slate-500'}`}
        >
          {copy.writeAccess(persona.enable_write_tools)}
        </div>
        <div
          data-testid={`persona-subagent-access-${persona.name}`}
          className={`text-[11px] ${persona.enable_subagent_tools ? 'text-amber-300' : 'text-slate-500'}`}
        >
          {copy.subagentAccess(persona.enable_subagent_tools)}
        </div>
      </div>
    </div>
  );
};
