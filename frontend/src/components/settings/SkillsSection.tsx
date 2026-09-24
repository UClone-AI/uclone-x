import React from 'react';
import { ShieldCheck } from 'lucide-react';
import { SkillsTab } from '../SkillsTab';
import { useApiRead, type ReadFaultKind } from '../../lib/useApiRead';
import type { SkillsData } from '../../types';
import { ReadFailure, ReadLoading } from './ReadState';

/**
 * Why the skills could not be loaded, in our own plain words, for a failure the runtime did
 * not explain itself. Never the browser's transport message ("Failed to fetch") or a status
 * line ("HTTP 500"): this section is open to every user, not only to someone who reads the
 * code (#1369).
 */
const PLAIN_CAUSE: Record<ReadFaultKind, string> = {
  unreachable: 'UClone-X could not be reached. If it has stopped, start it again, then try again.',
  status: 'UClone-X answered with an error but gave no reason.',
  unreadable: "UClone-X's answer could not be read.",
};

/**
 * The skill catalogue, as a section of Settings (#1358; owner ruling 2026-09-22).
 *
 * It was a dock surface, and the dock is about the conversation on screen. Which skills are
 * registered, approved or quarantined is configuration of the installation, so it sits beside
 * the clone definitions in Settings. Not behind developer mode: installing and approving a
 * skill is a setting a user changes, not an instrument -- which is why its copy is plain
 * words for someone who does not read the code (#1369).
 *
 * Four states, each said in words (#1369): the read is out, the read failed (with its cause
 * and a way to try again), no skill is registered, or the catalogue. The cause is the
 * runtime's own `detail` when it gave one, otherwise a fixed sentence from `PLAIN_CAUSE`.
 *
 * Carries `.dock-scope` so `SkillsTab`'s grids answer to the modal's width rather than the
 * window's, as they did in the dock (ui-authoring §3).
 */
export const SkillsSection: React.FC = () => {
  const { data, fault, loading, reload } = useApiRead<SkillsData>('/api/skills');
  const skills = data?.skills ?? [];

  return (
    <div className="space-y-3" data-testid="settings-skills">
      <label className="text-xs font-semibold uppercase tracking-wider text-slate-400 flex items-center gap-1.5">
        <ShieldCheck className="w-3.5 h-3.5 text-cyan-400" />
        Skills
      </label>
      <p className="text-[11px] text-slate-500">
        Skills are add-ons that give your clones new abilities. Each one is listed here with the
        result of its safety check.
      </p>
      {fault !== null && (
        <ReadFailure
          testId="settings-skills-error"
          what="Your skills could not be loaded."
          cause={fault.detail ?? PLAIN_CAUSE[fault.kind]}
          plain
          onRetry={reload}
          retrying={loading}
        />
      )}
      {fault === null && data === null && (
        <ReadLoading testId="settings-skills-loading">Loading your skills…</ReadLoading>
      )}
      {fault === null && data !== null && skills.length === 0 && (
        <p data-testid="settings-skills-empty" className="text-[11px] text-slate-500">
          No skills are registered with this installation.
        </p>
      )}
      {fault === null && skills.length > 0 && (
        <div className="dock-scope">
          <SkillsTab skillsData={data} onRefresh={reload} isLoading={loading} />
        </div>
      )}
    </div>
  );
};
