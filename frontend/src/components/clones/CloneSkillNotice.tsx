import { fmt, useCopy } from '../../i18n';
import { useApiRead } from '../../lib/useApiRead';
import type { SkillsData } from '../../types';

/** The skills `cloneId` is not offered, each with the tools it needs that the clone lacks. */
export const skillsHiddenFrom = (
  data: SkillsData | null,
  cloneId: string,
): { skill: string; tools: string[] }[] =>
  (data?.skills ?? []).flatMap((skill) => {
    const entry = (skill.hidden_from ?? []).find((h) => h.persona === cloneId);
    return entry ? [{ skill: skill.name, tools: entry.missing_tools }] : [];
  });

/**
 * Says which skills this clone is not offered because it lacks a tool they need (#1826).
 *
 * The runtime leaves those skills out of the clone's skill list, so the clone never mentions
 * them; this is where a person looking at the clone learns why. It renders nothing while the
 * skill list is still loading, when it could not be read (Settings > Skills says why), and
 * when the clone is offered every skill -- in each case there is nothing true to say here.
 */
export function CloneSkillNotice({ cloneId }: { cloneId: string }) {
  const t = useCopy().skills.requiresTools;
  const { data } = useApiRead<SkillsData>('/api/skills');
  const hidden = skillsHiddenFrom(data, cloneId);
  if (hidden.length === 0) return null;
  return (
    <div
      data-testid="clone-skill-notice"
      className="p-3 rounded-lg bg-slate-900/60 border border-slate-700 text-slate-300 text-xs space-y-1.5"
    >
      <strong className="text-slate-200">{t.cloneHeading}</strong>
      <ul className="space-y-0.5">
        {hidden.map((h) => (
          <li key={h.skill}>{fmt(t.cloneRow, { skill: h.skill, tools: h.tools.join(', ') })}</li>
        ))}
      </ul>
      <p className="text-slate-400">{t.cloneRemedy}</p>
    </div>
  );
}
