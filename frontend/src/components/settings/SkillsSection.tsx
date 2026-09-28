import React from 'react';
import { ShieldCheck } from 'lucide-react';
import { SkillsTab } from '../SkillsTab';
import { plural, useCopy } from '../../i18n';
import { useApiRead } from '../../lib/useApiRead';
import type { SkillsData } from '../../types';
import { ReadFailure, ReadLoading } from './ReadState';
import { SkillDecisionNotice, SkillProposalList, useSkillDecisions } from './SkillProposals';

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
 * and a way to try again), no skill is registered (or, when the runtime found no skill folder
 * at all, that none could be loaded and where to look, #1721), or the catalogue. The cause is the
 * runtime's own `detail` when it gave one, otherwise a fixed sentence from `skills.plainCause`.
 * Above the catalogue, one line names the skills that must be approved again (#1777), and
 * the skills clones proposed wait for a person's decision (#1827). An approved skill that did
 * not ship with UClone-X can be revoked from its safety check.
 *
 * Carries `.dock-scope` so `SkillsTab`'s grids answer to the modal's width rather than the
 * window's, as they did in the dock (ui-authoring §3).
 */
export const SkillsSection: React.FC = () => {
  const { data, fault, loading, reload } = useApiRead<SkillsData>('/api/skills');
  const skills = data?.skills ?? [];
  const copy = useCopy().skills;
  const proposals = data?.proposals ?? [];
  const decisions = useSkillDecisions(reload);
  // Skills approved before approvals were pinned (#1776) are not used until they are approved
  // again; one line says so and what to run (#1777).
  const reapprove = skills
    .filter((skill) => skill.not_loaded_code === 'approved_before_pins')
    .map((skill) => skill.name);

  return (
    <div className="space-y-3" data-testid="settings-skills">
      <label className="text-xs font-semibold uppercase tracking-wider text-slate-400 flex items-center gap-1.5">
        <ShieldCheck className="w-3.5 h-3.5 text-cyan-400" />
        {copy.title}
      </label>
      <p className="text-[11px] text-slate-500">
        {copy.intro}
      </p>
      {fault !== null && (
        <ReadFailure
          testId="settings-skills-error"
          what={copy.loadFailed}
          cause={fault.detail ?? copy.plainCause[fault.kind]}
          plain
          onRetry={reload}
          retrying={loading}
        />
      )}
      {fault === null && data === null && (
        <ReadLoading testId="settings-skills-loading">{copy.loading}</ReadLoading>
      )}
      {fault === null && data !== null && skills.length === 0 && (
        <p data-testid="settings-skills-empty" className="text-[11px] text-slate-500">
          {data.store_missing ? copy.noStore : copy.empty}
        </p>
      )}
      {fault === null && reapprove.length > 0 && (
        <p data-testid="settings-skills-reapprove" className="text-[11px] text-amber-300">
          {plural(copy.reapprove, reapprove.length, { names: reapprove.join(', ') })}
        </p>
      )}
      <SkillDecisionNotice decisions={decisions} />
      {fault === null && <SkillProposalList proposals={proposals} decisions={decisions} />}
      {fault === null && skills.length > 0 && (
        <div className="dock-scope">
          <SkillsTab
            skillsData={data}
            onRefresh={reload}
            isLoading={loading}
            onRevoke={(name) => void decisions.revoke(name)}
            deciding={decisions.busy}
          />
        </div>
      )}
    </div>
  );
};
