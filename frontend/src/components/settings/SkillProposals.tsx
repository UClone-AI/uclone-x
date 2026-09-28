import React, { useState } from 'react';
import { Check, X } from 'lucide-react';
import { Button } from '../ui/Button';
import { fmt, useCopy } from '../../i18n';
import { CoreFailure, failureOf, plainFailure } from '../../lib/coreFailure';
import { openConfirmedWindow } from '../../lib/person';
import type { SkillProposal } from '../../types';

/**
 * A person's decisions about skills, in Settings (#1827).
 *
 * A clone proposes a skill with `propose_skill`; it is not used until a person approves it
 * here. Approve and Turn down act on one proposal's version; Revoke stops an approved skill
 * that did not ship with UClone-X. The Core accepts each only from a window it opened itself
 * (#1589), so a refusal for that reason offers to open one, as the story view does. Approve
 * sends the digest of the text shown, and the Core installs that text or nothing; when the
 * proposal changed since, the panel says so and shows it again. Any other refusal is the
 * Core's own plain sentence, or a fixed one when it gave none.
 */

async function postDecision(name: string, action: string, body: object): Promise<void> {
  const res = await fetch(`/api/skills/${encodeURIComponent(name)}/${action}`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(body),
  });
  if (!res.ok) throw await failureOf(res);
}

export interface SkillDecisions {
  busy: boolean;
  notice: string | null;
  unconfirmed: boolean;
  approve: (proposal: SkillProposal) => Promise<void>;
  turnDown: (proposal: SkillProposal) => Promise<void>;
  revoke: (name: string) => Promise<void>;
  openWindow: () => Promise<void>;
}

/** The decision requests, and what the last one said. `onChanged` rereads the catalogue. */
export function useSkillDecisions(onChanged: () => void): SkillDecisions {
  const copy = useCopy().skills;
  const [busy, setBusy] = useState(false);
  const [notice, setNotice] = useState<string | null>(null);
  const [unconfirmed, setUnconfirmed] = useState(false);

  const decide = async (run: () => Promise<void>, done: string, failed: string) => {
    setBusy(true);
    setNotice(null);
    setUnconfirmed(false);
    try {
      await run();
      setNotice(done);
    } catch (err) {
      console.error('Skills: a decision failed', err);
      const notThisWindow = err instanceof CoreFailure && err.status === 403;
      // 412: the proposal is no longer the text shown, so nothing was approved (#1827).
      const changedSinceShown = err instanceof CoreFailure && err.status === 412;
      setUnconfirmed(notThisWindow);
      setNotice(
        notThisWindow
          ? copy.confirmWindow.unconfirmed
          : changedSinceShown
          ? copy.proposals.changed
          : plainFailure(err, failed),
      );
    } finally {
      setBusy(false);
    }
    onChanged();
  };

  return {
    busy,
    notice,
    unconfirmed,
    approve: (p) =>
      decide(
        () => postDecision(p.name, 'approve', { version: p.version, seen_digest: p.digest }),
        fmt(copy.proposals.approved, { name: p.name }),
        copy.proposals.approveFailed,
      ),
    turnDown: (p) =>
      decide(
        () => postDecision(p.name, 'reject', { version: p.version }),
        fmt(copy.proposals.turnedDown, { name: p.name }),
        copy.proposals.turnDownFailed,
      ),
    revoke: (name) =>
      decide(
        () => postDecision(name, 'revoke', {}),
        fmt(copy.revoke.done, { name }),
        copy.revoke.failed,
      ),
    openWindow: async () => {
      try {
        await openConfirmedWindow();
        setNotice(copy.confirmWindow.opened);
        setUnconfirmed(false);
      } catch (err) {
        console.error('Skills: no confirmed window opened', err);
        setNotice(plainFailure(err, copy.confirmWindow.failed));
      }
    },
  };
}

/** What the last decision said, with the offer to open a confirmed window when that was why. */
export const SkillDecisionNotice: React.FC<{ decisions: SkillDecisions }> = ({ decisions }) => {
  const copy = useCopy().skills.confirmWindow;
  if (decisions.notice === null) return null;
  return (
    <div
      role="status"
      data-testid="skill-decision-notice"
      className="p-3 rounded-lg bg-slate-900/60 border border-slate-700 text-xs text-amber-200 space-y-2"
    >
      <p>{decisions.notice}</p>
      {decisions.unconfirmed && (
        <Button
          variant="bordered"
          onClick={() => void decisions.openWindow()}
          data-testid="skill-open-confirmed-window"
        >
          {copy.open}
        </Button>
      )}
    </div>
  );
};

/** A unified diff, each added or removed line tinted. */
const DiffView: React.FC<{ diff: string }> = ({ diff }) => (
  <pre
    data-testid="skill-proposal-diff"
    className="max-h-64 overflow-auto p-2 rounded bg-slate-950 border border-slate-800 text-[11px] leading-snug whitespace-pre-wrap break-words"
  >
    {diff.split('\n').map((line, i) => (
      <div
        key={i}
        className={
          line.startsWith('+') && !line.startsWith('+++')
            ? 'text-emerald-300'
            : line.startsWith('-') && !line.startsWith('---')
            ? 'text-rose-300'
            : 'text-slate-400'
        }
      >
        {line || ' '}
      </div>
    ))}
  </pre>
);

/** The proposals waiting for a decision, each with what it would install. */
export const SkillProposalList: React.FC<{
  proposals: SkillProposal[];
  decisions: SkillDecisions;
}> = ({ proposals, decisions }) => {
  const copy = useCopy().skills.proposals;
  if (proposals.length === 0) return null;
  return (
    <div data-testid="skill-proposals" className="space-y-2">
      <p className="text-xs font-semibold text-slate-200">{copy.heading}</p>
      <p className="text-[11px] text-slate-500">{copy.intro}</p>
      {proposals.map((p) => (
        <div
          key={`${p.name}@${p.version}`}
          data-testid="skill-proposal"
          className="p-3 rounded-lg bg-slate-900/80 border border-slate-800 space-y-2 text-xs"
        >
          <div className="flex flex-wrap items-baseline justify-between gap-2">
            <span className="font-mono font-bold text-white break-all">{p.name}</span>
            <span className="text-[11px] text-slate-400">
              {fmt(copy.version, { version: p.version })} · {fmt(copy.by, { clone: p.agent_id })}
            </span>
          </div>
          <p className="text-slate-300">{p.description}</p>
          <p className="text-[11px] text-slate-400">
            {p.requires_tools.length > 0
              ? fmt(copy.tools, { tools: p.requires_tools.join(', ') })
              : copy.noTools}
          </p>
          {p.current_version !== null && p.diff !== '' ? (
            <>
              <p className="text-[11px] text-slate-400">
                {fmt(copy.replaces, { version: p.current_version })}
              </p>
              <DiffView diff={p.diff} />
            </>
          ) : (
            <>
              <p className="text-[11px] text-slate-400">{copy.newSkill}</p>
              <pre
                data-testid="skill-proposal-instructions"
                className="max-h-64 overflow-auto p-2 rounded bg-slate-950 border border-slate-800 text-[11px] leading-snug text-slate-300 whitespace-pre-wrap break-words"
              >
                {p.instructions}
              </pre>
            </>
          )}
          <div className="flex gap-2">
            <Button
              variant="bordered"
              disabled={decisions.busy}
              onClick={() => void decisions.approve(p)}
              data-testid="skill-proposal-approve"
            >
              <Check className="w-3.5 h-3.5" />
              {copy.approve}
            </Button>
            <Button
              variant="ghost"
              disabled={decisions.busy}
              onClick={() => void decisions.turnDown(p)}
              data-testid="skill-proposal-turn-down"
            >
              <X className="w-3.5 h-3.5" />
              {copy.turnDown}
            </Button>
          </div>
        </div>
      ))}
    </div>
  );
};
