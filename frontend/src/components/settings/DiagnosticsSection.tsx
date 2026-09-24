import React from 'react';
import { Stethoscope } from 'lucide-react';
import { AcpTab } from '../AcpTab';
import { EvaluationsTab } from '../EvaluationsTab';
import { useApiRead } from '../../lib/useApiRead';
import type { AcpStatusData, EvaluationsData } from '../../types';
import { ReadFailure, ReadLoading } from './ReadState';

/**
 * Diagnostics about the build: its Agent Client Protocol report and its latest evaluation
 * scorecard (#1358; owner ruling 2026-09-22).
 *
 * They were dock surfaces, and neither is about the conversation on screen: one is a protocol
 * conformance table, the other a scorecard of the build. They are instruments (ui-authoring
 * §2 step 4), so `SettingsModal` mounts this only while developer mode is on, directly under
 * the switch that reveals it -- one deliberate action, and nothing is read while it is off.
 *
 * Not folded into `DiagnosticsPanel`: that is problem reporting, a consent a first-time user
 * is asked for whatever the mode, and it stays where it is.
 *
 * Each read is in one of three states, and each is drawn as itself (#1369): out (said in words,
 * never the panel's own "not read" block or a scorecard of zeros), failed (the cause, and a way
 * to read again), or answered (the panel).
 *
 * Carries `.dock-scope` so the panels' grids answer to the modal's width, not the window's,
 * as they did in the dock (ui-authoring §3).
 */
export const DiagnosticsSection: React.FC = () => {
  const acp = useApiRead<AcpStatusData>('/api/acp/status');
  const evals = useApiRead<EvaluationsData>('/api/evaluations/latest');

  return (
    <div className="space-y-3" data-testid="settings-diagnostics">
      <label className="text-xs font-semibold uppercase tracking-wider text-slate-400 flex items-center gap-1.5">
        <Stethoscope className="w-3.5 h-3.5 text-cyan-400" />
        Diagnostics
      </label>
      <p className="text-[11px] text-slate-500">
        What this build answers of the Agent Client Protocol, and its latest evaluation run.
      </p>
      <div className="dock-scope space-y-6">
        {acp.error !== null ? (
          <ReadFailure
            testId="diagnostics-acp-error"
            what="ACP status could not be read from the runtime."
            cause={acp.error}
            onRetry={acp.reload}
            retrying={acp.loading}
          />
        ) : acp.data === null ? (
          <ReadLoading testId="diagnostics-acp-loading">Reading ACP status…</ReadLoading>
        ) : (
          <AcpTab acpStatus={acp.data} onRefresh={acp.reload} isLoading={acp.loading} />
        )}
        {evals.error !== null ? (
          <ReadFailure
            testId="diagnostics-evals-error"
            what="Evaluation results could not be read from the runtime."
            cause={evals.error}
            onRetry={evals.reload}
            retrying={evals.loading}
          />
        ) : evals.data === null ? (
          <ReadLoading testId="diagnostics-evals-loading">Reading evaluation results…</ReadLoading>
        ) : (
          <EvaluationsTab
            evaluationsData={evals.data}
            onRefresh={evals.reload}
            isLoading={evals.loading}
          />
        )}
      </div>
    </div>
  );
};
