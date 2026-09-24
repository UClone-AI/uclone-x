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

/**
 * Plain words for the checker's own labels (#1369).
 *
 * The catalogue is shown to every user in Settings, including one who has never read the code,
 * so the runtime's terms -- quarantine, AST, verdict, synthesized -- are translated here rather
 * than printed. A value this table does not know is printed as the runtime sent it: an
 * unrecognised state is shown, never guessed at.
 */
const STATUS_LABEL: Record<string, string> = {
  active: 'Ready to use',
  pending: 'Waiting for review',
  quarantined: 'Blocked',
  rejected: 'Turned down',
};

const ORIGIN_LABEL: Record<string, string> = {
  human: 'Written by a person',
  synthesized: 'Written by the assistant',
};

const ISOLATION_LABEL: Record<string, string> = {
  none: 'Runs without a sandbox',
  workspace: 'Runs in your workspace folder',
  container: 'Runs in a container',
  wasm: 'Runs in a sandbox',
};

const RECOMMENDATION_LABEL: Record<string, string> = {
  approve: 'Safe to use',
  require_human_review: 'Needs your review',
  reject: 'Not safe to use',
};

const labelOf = (table: Record<string, string>, value: string): string => table[value] ?? value;

interface SkillsTabProps {
  skillsData: SkillsData | null;
  onRefresh: () => void;
  isLoading: boolean;
}

export function SkillsTab({ skillsData, onRefresh, isLoading }: SkillsTabProps) {
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

  return (
    <div className="space-y-6">
      {/* How many skills are in each state */}
      <div className="grid grid-cols-1 md:grid-cols-4 gap-4">
        <div className="p-4 bg-slate-900/80 border border-slate-800 rounded-2xl shadow-lg">
          <div className="flex items-center justify-between text-slate-400 text-xs">
            <span>Skills</span>
            <Layers className="w-4 h-4 text-cyan-400" />
          </div>
          <p className="mt-2 text-2xl font-bold text-white">{summary.total_skills}</p>
          <p className="text-[11px] text-slate-400 mt-0.5">Installed</p>
        </div>

        <div className="p-4 bg-slate-900/80 border border-slate-800 rounded-2xl shadow-lg">
          <div className="flex items-center justify-between text-slate-400 text-xs">
            <span>Ready to use</span>
            <ShieldCheck className="w-4 h-4 text-emerald-400" />
          </div>
          <p className="mt-2 text-2xl font-bold text-emerald-400">{summary.active_count}</p>
          <p className="text-[11px] text-slate-400 mt-0.5">Passed the safety check</p>
        </div>

        <div className="p-4 bg-slate-900/80 border border-slate-800 rounded-2xl shadow-lg">
          <div className="flex items-center justify-between text-slate-400 text-xs">
            <span>Waiting for review</span>
            <ShieldAlert className="w-4 h-4 text-amber-400" />
          </div>
          <p className="mt-2 text-2xl font-bold text-amber-400">{summary.pending_count}</p>
          <p className="text-[11px] text-slate-400 mt-0.5">Needs a person to approve it</p>
        </div>

        <div className="p-4 bg-slate-900/80 border border-slate-800 rounded-2xl shadow-lg">
          <div className="flex items-center justify-between text-slate-400 text-xs">
            <span>Blocked</span>
            <ShieldX className="w-4 h-4 text-rose-400" />
          </div>
          <p className="mt-2 text-2xl font-bold text-rose-400">{summary.quarantined_count}</p>
          <p className="text-[11px] text-slate-400 mt-0.5">Failed the safety check</p>
        </div>
      </div>

      {/* Filter & Search Bar */}
      <div className="p-4 bg-slate-900/80 border border-slate-800 rounded-2xl shadow-lg flex flex-wrap gap-3">
        <div className="relative grow basis-48">
          <Search className="w-4 h-4 absolute left-3 top-2.5 text-slate-500" />
          <input
            type="text"
            placeholder="Search skills by name, description, or tags..."
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
            <option value="ALL">Any status</option>
            <option value="active">{STATUS_LABEL.active}</option>
            <option value="pending">{STATUS_LABEL.pending}</option>
            <option value="quarantined">{STATUS_LABEL.quarantined}</option>
          </select>

          <select
            value={originFilter}
            onChange={(e) => setOriginFilter(e.target.value)}
            className="bg-slate-950 border border-slate-800 text-xs text-slate-300 rounded-xl px-3 py-2 focus:outline-none focus:border-cyan-500 font-mono shadow-inner"
          >
            <option value="ALL">Anyone</option>
            <option value="human">{ORIGIN_LABEL.human}</option>
            <option value="synthesized">{ORIGIN_LABEL.synthesized}</option>
          </select>

          <button
            onClick={onRefresh}
            disabled={isLoading}
            className="px-3 py-2 bg-slate-800 hover:bg-slate-700 text-slate-200 text-xs rounded-xl font-medium transition-colors border border-slate-700 whitespace-nowrap"
          >
            {isLoading ? 'Refreshing…' : 'Refresh'}
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
              Skills ({filteredSkills.length})
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
                      {labelOf(STATUS_LABEL, skill.status)}
                    </span>
                  </div>

                  <p className="text-[11px] text-slate-400 mt-1 line-clamp-2">
                    {skill.description}
                  </p>

                  <div className="mt-2.5 flex flex-wrap items-center justify-between gap-1 text-[10px] text-slate-500">
                    <span className="flex items-center gap-1">
                      <Lock className="w-3 h-3 text-slate-500" />
                      {labelOf(ISOLATION_LABEL, skill.isolation_level)}
                    </span>
                    <span className="px-1.5 py-0.2 rounded bg-slate-900 text-slate-300">
                      v{skill.version} • {labelOf(ORIGIN_LABEL, skill.origin)}
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
              <h3 className="text-sm font-semibold text-white">Safety check</h3>
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
                    <span className="text-slate-400">Author: </span>
                    <span className="text-slate-200">{activeSkill.author}</span>
                  </div>
                </div>

                <div className="pt-2 border-t border-slate-800/80 flex flex-wrap items-center justify-between gap-2 text-[11px] text-slate-400">
                  <span className="flex items-center gap-1" title={activeSkill.content_sha256}>
                    <Hash className="w-3 h-3 text-slate-500" />
                    Fingerprint{' '}
                    <span className="font-mono">{activeSkill.content_sha256.slice(0, 12)}…</span>
                  </span>
                  <span>
                    <strong className="text-slate-200">
                      {labelOf(ISOLATION_LABEL, activeSkill.isolation_level)}
                    </strong>
                  </span>
                </div>
              </div>

              {/* The check's result, and how risky it judged the skill */}
              <div className="p-4 rounded-xl bg-slate-950/90 border border-slate-800 space-y-3">
                <div className="flex items-center justify-between">
                  <span className="text-xs font-semibold text-slate-200">Result</span>
                  <span
                    className={`font-bold text-xs px-2.5 py-1 rounded ${
                      activeSkill.audit_report.recommendation === 'approve'
                        ? 'bg-emerald-950 text-emerald-300 border border-emerald-800'
                        : activeSkill.audit_report.recommendation === 'require_human_review'
                        ? 'bg-amber-950 text-amber-300 border border-amber-800'
                        : 'bg-rose-950 text-rose-300 border border-rose-800'
                    }`}
                  >
                    {labelOf(RECOMMENDATION_LABEL, activeSkill.audit_report.recommendation)}
                  </span>
                </div>

                {/* Risk, as a bar */}
                <div>
                  <div className="flex justify-between text-[11px] text-slate-400 mb-1">
                    <span>Risk found in its code</span>
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
                    What the check found
                  </span>
                  {activeSkill.audit_report.detected_risks.length === 0 ? (
                    <div className="mt-1.5 p-2.5 rounded-lg bg-emerald-950/30 border border-emerald-800/40 text-emerald-300 flex items-center gap-2 text-[11px]">
                      <CheckCircle2 className="w-4 h-4 text-emerald-400" />
                      <span>Nothing unsafe was found in this skill's code.</span>
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

                {activeSkill.rejection_reason && (
                  <div className="p-3 rounded-lg bg-rose-950/40 border border-rose-800/60 text-rose-200 text-xs">
                    <strong>Why it was turned down:</strong> {activeSkill.rejection_reason}
                  </div>
                )}
              </div>
            </div>
          ) : (
            <div className="p-10 text-center text-slate-500 text-xs">
              Choose a skill to see its safety check.
            </div>
          )}
        </div>
      </div>
    </div>
  );
}
