import { useState, useMemo } from 'react';
import {
  ShieldCheck,
  ShieldAlert,
  ShieldX,
  Search,
  Hash,
  FileCode,
  CheckCircle2,
  AlertTriangle,
  Lock,
  Layers,
} from 'lucide-react';
import { SkillsData, SkillManifest } from '../types';
import { fmt, useCopy, type Messages } from '../i18n';
import { placeholders } from '../i18n/format';
import { Button } from './ui/Button';

type NotLoadedCopy = Messages['skills']['notLoaded'];

/**
 * Why a skill is not used, in the reader's language (#1777).
 *
 * The Core sends a `not_loaded_code` and its params (`uclone_x/skills/refusals.py`), and the
 * sentence is written here from the `skills` catalog, the way the conversation notices are
 * (`lib/notices.ts`). The English `not_loaded_reason` shows only when there is no code, when
 * the code is one this head does not know (a newer Core), or when the params lack a value the
 * sentence needs; a reason is never left blank or half-filled.
 */
export const notLoadedText = (skill: SkillManifest, t: NotLoadedCopy): string | null => {
  const code = skill.not_loaded_code;
  const fallback = skill.not_loaded_reason ?? null;
  if (!code || !Object.prototype.hasOwnProperty.call(t.codes, code)) return fallback;
  const template = t.codes[code as keyof NotLoadedCopy['codes']];
  const values = skill.not_loaded_params ?? {};
  if (!placeholders(template).every((name) => Object.prototype.hasOwnProperty.call(values, name))) {
    return fallback;
  }
  return fmt(template, values);
};

/**
 * Plain words for the checker's own labels (#1369), from `skills.tab` in the reader's language
 * (#1782).
 *
 * The catalogue is shown to every user in Settings, including one who has never read the code,
 * so the runtime's terms -- quarantine, AST, verdict, synthesized -- are translated rather than
 * printed. A value the catalog does not know is printed as the runtime sent it: an
 * unrecognised state is shown, never guessed at.
 */
const labelOf = (table: Record<string, string>, value: string): string => table[value] ?? value;

/**
 * Which clones are not offered this skill, and the tools each lacks (#1826).
 *
 * The runtime leaves such a skill out of that clone's skill list and refuses to load it
 * there, so without this the skill would vanish from that clone with no word of why.
 * Renders nothing when every clone can use the skill.
 */
export function SkillHiddenFromNotice({ skill }: { skill: SkillManifest }) {
  const t = useCopy().skills.requiresTools;
  const hidden = skill.hidden_from ?? [];
  if (hidden.length === 0) return null;
  return (
    <div
      data-testid="skill-hidden-from"
      className="p-3 rounded-lg bg-slate-900/60 border border-slate-700 text-slate-300 text-xs space-y-1.5"
    >
      <strong className="text-slate-200">{t.heading}</strong>
      <ul className="space-y-0.5">
        {hidden.map((h) => (
          <li key={h.persona}>
            {fmt(t.row, { clone: h.persona, tools: h.missing_tools.join(', ') })}
          </li>
        ))}
      </ul>
      <p className="text-slate-400">{t.remedy}</p>
    </div>
  );
}

interface SkillsTabProps {
  skillsData: SkillsData | null;
  onRefresh: () => void;
  isLoading: boolean;
  /**
   * Revoke an approved skill (#1827). Offered for an active skill that did not ship with
   * UClone-X; without it the tab offers no revoke at all.
   */
  onRevoke?: (name: string) => void;
  /** A decision is in flight, so the revoke button waits. */
  deciding?: boolean;
}

export function SkillsTab({
  skillsData,
  onRefresh,
  isLoading,
  onRevoke,
  deciding = false,
}: SkillsTabProps) {
  const [searchTerm, setSearchTerm] = useState<string>('');
  const [statusFilter, setStatusFilter] = useState<string>('ALL');
  const [originFilter, setOriginFilter] = useState<string>('ALL');
  const [selectedSkillName, setSelectedSkillName] = useState<string | null>(null);

  const skills = skillsData?.skills || [];
  const summary = skillsData?.summary || {
    total_skills: 0,
    active_count: 0,
    pending_count: 0,
    quarantined_count: 0,
  };

  const filteredSkills = useMemo(() => {
    return skills.filter((s) => {
      const matchesStatus =
        statusFilter === 'ALL' || s.status.toLowerCase() === statusFilter.toLowerCase();
      const matchesOrigin =
        originFilter === 'ALL' || s.origin.toLowerCase() === originFilter.toLowerCase();
      const term = searchTerm.toLowerCase();
      const matchesSearch =
        !term ||
        s.name.toLowerCase().includes(term) ||
        s.description.toLowerCase().includes(term) ||
        s.tags.some((t) => t.toLowerCase().includes(term));

      return matchesStatus && matchesOrigin && matchesSearch;
    });
  }, [skills, statusFilter, originFilter, searchTerm]);

  const activeSkill: SkillManifest | null =
    skills.find((s) => s.name === selectedSkillName) || skills[0] || null;
  const skillsCopy = useCopy().skills;
  const copy = skillsCopy.tab;
  const notLoadedCopy = skillsCopy.notLoaded;
  const notLoaded = activeSkill ? notLoadedText(activeSkill, notLoadedCopy) : null;

  return (
    <div className="space-y-6">
      {/* How many skills are in each state */}
      <div className="grid grid-cols-1 md:grid-cols-4 gap-4">
        <div className="p-4 bg-slate-900/80 border border-slate-800 rounded-2xl shadow-lg">
          <div className="flex items-center justify-between text-slate-400 text-xs">
            <span>{copy.cards.total.label}</span>
            <Layers className="w-4 h-4 text-cyan-400" />
          </div>
          <p className="mt-2 text-2xl font-bold text-white">{summary.total_skills}</p>
          <p className="text-[11px] text-slate-400 mt-0.5">{copy.cards.total.hint}</p>
        </div>

        <div className="p-4 bg-slate-900/80 border border-slate-800 rounded-2xl shadow-lg">
          <div className="flex items-center justify-between text-slate-400 text-xs">
            <span>{copy.cards.active.label}</span>
            <ShieldCheck className="w-4 h-4 text-emerald-400" />
          </div>
          <p className="mt-2 text-2xl font-bold text-emerald-400">{summary.active_count}</p>
          <p className="text-[11px] text-slate-400 mt-0.5">{copy.cards.active.hint}</p>
        </div>

        <div className="p-4 bg-slate-900/80 border border-slate-800 rounded-2xl shadow-lg">
          <div className="flex items-center justify-between text-slate-400 text-xs">
            <span>{copy.cards.pending.label}</span>
            <ShieldAlert className="w-4 h-4 text-amber-400" />
          </div>
          {/* A clone's proposals wait for review too, though the catalogue does not list them (#1827). */}
          <p data-testid="skills-waiting-count" className="mt-2 text-2xl font-bold text-amber-400">
            {summary.pending_count + (skillsData?.proposals?.length ?? 0)}
          </p>
          <p className="text-[11px] text-slate-400 mt-0.5">{copy.cards.pending.hint}</p>
        </div>

        <div className="p-4 bg-slate-900/80 border border-slate-800 rounded-2xl shadow-lg">
          <div className="flex items-center justify-between text-slate-400 text-xs">
            <span>{copy.cards.blocked.label}</span>
            <ShieldX className="w-4 h-4 text-rose-400" />
          </div>
          <p className="mt-2 text-2xl font-bold text-rose-400">{summary.quarantined_count}</p>
          <p className="text-[11px] text-slate-400 mt-0.5">{copy.cards.blocked.hint}</p>
        </div>
      </div>

      {/* Filter & Search Bar */}
      <div className="p-4 bg-slate-900/80 border border-slate-800 rounded-2xl shadow-lg flex flex-wrap gap-3">
        <div className="relative grow basis-48">
          <Search className="w-4 h-4 absolute left-3 top-2.5 text-slate-500" />
          <input
            type="text"
            placeholder={copy.search}
            value={searchTerm}
            onChange={(e) => setSearchTerm(e.target.value)}
            className="w-full pl-9 pr-4 py-2 bg-slate-950 border border-slate-800 rounded-xl text-xs text-slate-200 placeholder-slate-500 focus:outline-none focus:border-cyan-500 shadow-inner"
          />
        </div>

        <div className="flex flex-wrap items-center gap-2">
          <select
            value={statusFilter}
            onChange={(e) => setStatusFilter(e.target.value)}
            className="bg-slate-950 border border-slate-800 text-xs text-slate-300 rounded-xl px-3 py-2 focus:outline-none focus:border-cyan-500 font-mono shadow-inner"
          >
            <option value="ALL">{copy.anyStatus}</option>
            <option value="active">{copy.status.active}</option>
            <option value="pending">{copy.status.pending}</option>
            <option value="quarantined">{copy.status.quarantined}</option>
          </select>

          <select
            value={originFilter}
            onChange={(e) => setOriginFilter(e.target.value)}
            className="bg-slate-950 border border-slate-800 text-xs text-slate-300 rounded-xl px-3 py-2 focus:outline-none focus:border-cyan-500 font-mono shadow-inner"
          >
            <option value="ALL">{copy.anyone}</option>
            <option value="human">{copy.origin.human}</option>
            <option value="synthesized">{copy.origin.synthesized}</option>
          </select>

          <button
            onClick={onRefresh}
            disabled={isLoading}
            className="px-3 py-2 bg-slate-800 hover:bg-slate-700 text-slate-200 text-xs rounded-xl font-medium transition-colors border border-slate-700 whitespace-nowrap"
          >
            {isLoading ? copy.refreshing : copy.refresh}
          </button>
        </div>
      </div>

      {/* The list, and the chosen skill's safety check */}
      <div className="grid grid-cols-1 lg:grid-cols-3 gap-6">
        {/* Skills Listing */}
        <div className="p-5 bg-slate-900/80 border border-slate-800 rounded-2xl shadow-lg space-y-3">
          <div className="flex items-center justify-between pb-3 border-b border-slate-800">
            <h3 className="text-sm font-semibold text-white flex items-center gap-2">
              <FileCode className="w-4 h-4 text-cyan-400" />
              {fmt(copy.listHeading, { count: filteredSkills.length })}
            </h3>
          </div>

          <div className="space-y-3 max-h-[520px] overflow-y-auto pr-1">
            {filteredSkills.map((skill) => {
              const isSelected = activeSkill?.name === skill.name;
              const isActive = skill.status === 'active';
              const isPending = skill.status === 'pending';

              return (
                <div
                  key={skill.name}
                  onClick={() => setSelectedSkillName(skill.name)}
                  className={`p-3.5 rounded-xl border cursor-pointer transition-all ${
                    isSelected
                      ? 'bg-cyan-950/40 border-cyan-500/80 shadow-md shadow-cyan-950/30'
                      : 'bg-slate-950/70 border-slate-800/80 hover:border-slate-700'
                  }`}
                >
                  <div className="flex flex-wrap items-center justify-between gap-1.5">
                    <span className="font-mono text-xs font-bold text-white break-all">
                      {skill.name}
                    </span>
                    <span
                      className={`text-[9px] font-bold px-2 py-0.5 rounded whitespace-nowrap ${
                        isActive
                          ? 'bg-emerald-950 text-emerald-300 border border-emerald-800/60'
                          : isPending
                          ? 'bg-amber-950 text-amber-300 border border-amber-800/60'
                          : 'bg-rose-950 text-rose-300 border border-rose-800/60'
                      }`}
                    >
                      {labelOf(copy.status, skill.status)}
                    </span>
                  </div>

                  <p className="text-[11px] text-slate-400 mt-1 line-clamp-2">
                    {skill.description}
                  </p>

                  <div className="mt-2.5 flex flex-wrap items-center justify-between gap-1 text-[10px] text-slate-500">
                    <span className="flex items-center gap-1">
                      <Lock className="w-3 h-3 text-slate-500" />
                      {labelOf(copy.isolation, skill.isolation_level)}
                    </span>
                    <span className="px-1.5 py-0.2 rounded bg-slate-900 text-slate-300">
                      v{skill.version} • {labelOf(copy.origin, skill.origin)}
                    </span>
                  </div>
                </div>
              );
            })}
          </div>
        </div>

        {/* The chosen skill's safety check */}
        <div className="lg:col-span-2 p-5 bg-slate-900/80 border border-slate-800 rounded-2xl shadow-lg space-y-4">
          <div className="flex items-center justify-between pb-3 border-b border-slate-800">
            <div className="flex items-center gap-2">
              <ShieldCheck className="w-4 h-4 text-emerald-400" />
              <h3 className="text-sm font-semibold text-white">{copy.checkHeading}</h3>
            </div>
          </div>

          {activeSkill ? (
            <div className="space-y-4 text-xs">
              {/* Main Skill Summary Card */}
              <div className="p-4 bg-slate-950/80 border border-slate-800 rounded-xl space-y-2">
                <div className="flex items-center justify-between">
                  <div>
                    <h4 className="font-mono text-base font-bold text-cyan-300">
                      {activeSkill.name}
                    </h4>
                    <p className="text-slate-400 text-xs mt-0.5">{activeSkill.description}</p>
                  </div>
                  <div className="text-right text-[11px]">
                    <span className="text-slate-400">{copy.author} </span>
                    <span className="text-slate-200">{activeSkill.author}</span>
                  </div>
                </div>

                <div className="pt-2 border-t border-slate-800/80 flex flex-wrap items-center justify-between gap-2 text-[11px] text-slate-400">
                  <span className="flex items-center gap-1" title={activeSkill.content_sha256}>
                    <Hash className="w-3 h-3 text-slate-500" />
                    {copy.fingerprint}{' '}
                    <span className="font-mono">{activeSkill.content_sha256.slice(0, 12)}…</span>
                  </span>
                  <span>
                    <strong className="text-slate-200">
                      {labelOf(copy.isolation, activeSkill.isolation_level)}
                    </strong>
                  </span>
                </div>
              </div>

              {/* The check's result, and how risky it judged the skill */}
              <div className="p-4 rounded-xl bg-slate-950/90 border border-slate-800 space-y-3">
                <div className="flex items-center justify-between">
                  <span className="text-xs font-semibold text-slate-200">{copy.result}</span>
                  <span
                    className={`font-bold text-xs px-2.5 py-1 rounded ${
                      activeSkill.audit_report.recommendation === 'approve'
                        ? 'bg-emerald-950 text-emerald-300 border border-emerald-800'
                        : activeSkill.audit_report.recommendation === 'require_human_review'
                        ? 'bg-amber-950 text-amber-300 border border-amber-800'
                        : 'bg-rose-950 text-rose-300 border border-rose-800'
                    }`}
                  >
                    {labelOf(copy.recommendation, activeSkill.audit_report.recommendation)}
                  </span>
                </div>

                {/* Risk, as a bar */}
                <div>
                  <div className="flex justify-between text-[11px] text-slate-400 mb-1">
                    <span>{copy.risk}</span>
                    <span className="font-bold text-slate-200">
                      {(activeSkill.audit_report.risk_score * 100).toFixed(0)}%
                    </span>
                  </div>
                  <div className="w-full h-2.5 bg-slate-900 rounded-full overflow-hidden border border-slate-800">
                    <div
                      className={`h-full transition-all ${
                        activeSkill.audit_report.risk_score > 0.7
                          ? 'bg-rose-500'
                          : activeSkill.audit_report.risk_score > 0.2
                          ? 'bg-amber-500'
                          : 'bg-emerald-500'
                      }`}
                      style={{
                        width: `${Math.min(100, activeSkill.audit_report.risk_score * 100)}%`,
                      }}
                    />
                  </div>
                </div>

                {/* What the check found */}
                <div>
                  <span className="text-[11px] font-semibold text-slate-300">
                    {copy.findings}
                  </span>
                  {activeSkill.audit_report.detected_risks.length === 0 ? (
                    <div className="mt-1.5 p-2.5 rounded-lg bg-emerald-950/30 border border-emerald-800/40 text-emerald-300 flex items-center gap-2 text-[11px]">
                      <CheckCircle2 className="w-4 h-4 text-emerald-400" />
                      <span>{copy.noFindings}</span>
                    </div>
                  ) : (
                    <div className="mt-1.5 space-y-1.5">
                      {activeSkill.audit_report.detected_risks.map((risk) => (
                        <div
                          key={risk}
                          className="p-2 rounded-lg bg-rose-950/30 border border-rose-800/50 text-rose-300 flex items-center gap-2 text-[11px] font-mono"
                        >
                          <AlertTriangle className="w-3.5 h-3.5 text-rose-400" />
                          <span>{risk}</span>
                        </div>
                      ))}
                    </div>
                  )}
                </div>

                <SkillHiddenFromNotice skill={activeSkill} />

                {notLoaded && (
                  <div
                    data-testid="skill-not-loaded-reason"
                    className="p-3 rounded-lg bg-rose-950/40 border border-rose-800/60 text-rose-200 text-xs"
                  >
                    <strong>{notLoadedCopy.label}</strong> {notLoaded}
                  </div>
                )}

                {onRevoke && activeSkill.status === 'active' && !activeSkill.shipped && (
                  <div className="pt-2 border-t border-slate-800/80 flex flex-wrap items-center justify-between gap-2">
                    <span className="text-[11px] text-slate-400">{skillsCopy.revoke.hint}</span>
                    <Button
                      variant="bordered"
                      disabled={deciding}
                      onClick={() => onRevoke(activeSkill.name)}
                      data-testid="skill-revoke"
                    >
                      {skillsCopy.revoke.button}
                    </Button>
                  </div>
                )}

                {activeSkill.rejection_reason && (
                  <div className="p-3 rounded-lg bg-rose-950/40 border border-rose-800/60 text-rose-200 text-xs">
                    <strong>{copy.turnedDown}</strong> {activeSkill.rejection_reason}
                  </div>
                )}
              </div>
            </div>
          ) : (
            <div className="p-10 text-center text-slate-500 text-xs">
              {copy.choose}
            </div>
          )}
        </div>
      </div>
    </div>
  );
}
