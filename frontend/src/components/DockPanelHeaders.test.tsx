import { describe, expect, it, vi } from 'vitest';
import { render, screen } from '@testing-library/react';
import { SkillsTab } from './SkillsTab';
import { OntologyTab } from './OntologyTab';
import { LedgerTab } from './LedgerTab';
import { TopologyTab } from './TopologyTab';
import { EvaluationsTab } from './EvaluationsTab';
import { ActivityTimeline } from './artifacts/ActivityTimeline';

/**
 * The dock panels' header and filter rows wrap rather than run past the dock (#1029).
 *
 * Every one of these panels renders in the dock (`ArtifactsDock`), whose width is the user's
 * (360-900px) and has nothing to do with the viewport. The rows used to be single-line flex
 * rows that switched to a column only at the *viewport's* `sm:` breakpoint, so in a 520px
 * dock in a 1280px window Skills measured 546px and Evaluations 575px in a 519px surface:
 * the surface scrolled sideways and the last controls sat past its edge. A row that may
 * wrap costs nothing where it fits, and moves its last controls to a second line where it
 * does not -- which is the behaviour `.column-scope`'s header row already has (#1015).
 *
 * jsdom does no layout, so these cases pin the rows' wrapping; that the wrapped rows
 * actually fit is measured at 1280 and 400 in `tests/e2e/test_workspace_layout_e2e.py`.
 */

/** The nearest ancestor of `el` that is a flex row -- the row `el` sits in. */
function rowOf(el: HTMLElement): HTMLElement {
  let node = el.parentElement;
  while (node !== null && !node.classList.contains('flex') && !node.classList.contains('inline-flex')) {
    node = node.parentElement;
  }
  if (node === null) throw new Error('the control is not inside a flex row');
  return node;
}

function expectWraps(row: HTMLElement): void {
  expect(row.classList.contains('flex-wrap'), row.className).toBe(true);
  // A row that becomes a column below a *viewport* breakpoint is exactly what did not work
  // in the dock: the dock is narrow while the viewport is wide.
  expect(row.className).not.toMatch(/\bflex-col\b/);
}

describe('dock panel header rows wrap (#1029)', () => {
  // Killed by: frontend/src/components/SkillsTab.tsx :: shadow-lg flex flex-wrap gap-3
  // Becomes: shadow-lg flex flex-col sm:flex-row gap-3
  it('Skills: the search bar wraps its filters under the search', () => {
    render(<SkillsTab skillsData={null} onRefresh={vi.fn()} isLoading={false} />);
    const search = screen.getByPlaceholderText(/Search skills/);
    expectWraps(rowOf(search.parentElement as HTMLElement));
  });

  // Killed by: frontend/src/components/SkillsTab.tsx :: <div className="flex flex-wrap items-center gap-2">
  // Becomes: <div className="flex items-center gap-2">
  it('Skills: the filter selects wrap among themselves', () => {
    render(<SkillsTab skillsData={null} onRefresh={vi.fn()} isLoading={false} />);
    expectWraps(rowOf(screen.getByRole('option', { name: 'Any status' }).closest('select') as HTMLElement));
  });

  // Killed by: frontend/src/components/SkillsTab.tsx :: relative grow basis-48
  // Becomes: relative flex-1
  it('Skills: the search field keeps a usable width once the row wraps', () => {
    render(<SkillsTab skillsData={null} onRefresh={vi.fn()} isLoading={false} />);
    const field = screen.getByPlaceholderText(/Search skills/).parentElement as HTMLElement;
    // `flex-1` is `flex: 1 1 0%`: in a wrapping row a zero basis lets the search shrink to
    // nothing beside the selects instead of taking a line of its own.
    expect(field.className).toMatch(/\bbasis-48\b/);
    expect(field.className).toMatch(/\bgrow\b/);
  });

  // Killed by: frontend/src/components/OntologyTab.tsx :: shadow-lg flex flex-wrap gap-3
  // Becomes: shadow-lg flex flex-col sm:flex-row gap-3
  it('Ontology: the search bar wraps its filter under the search', () => {
    render(<OntologyTab ontology={null} onRefresh={vi.fn()} isLoading={false} />);
    const search = screen.getByPlaceholderText(/Search concepts/);
    expectWraps(rowOf(search.parentElement as HTMLElement));
  });

  // Killed by: frontend/src/components/LedgerTab.tsx :: shadow-lg flex flex-wrap gap-3
  // Becomes: shadow-lg flex flex-col sm:flex-row gap-3
  it('Ledger: the search bar wraps its filters under the search', () => {
    render(
      <LedgerTab
        events={[]}
        isPaused={false}
        onTogglePause={vi.fn()}
        onClearEvents={vi.fn()}
        lastHeartbeat="--"
      />,
    );
    const search = screen.getByPlaceholderText(/Search topic/);
    expectWraps(rowOf(search.parentElement as HTMLElement));
  });

  // Killed by: frontend/src/components/LedgerTab.tsx :: <div className="flex flex-wrap items-center gap-3">
  // Becomes: <div className="flex items-center gap-3">
  it('Ledger: the heartbeat, Pause and Clear wrap', () => {
    render(
      <LedgerTab
        events={[]}
        isPaused={false}
        onTogglePause={vi.fn()}
        onClearEvents={vi.fn()}
        lastHeartbeat="--"
      />,
    );
    expectWraps(rowOf(screen.getByText(/Last Heartbeat/).closest('div') as HTMLElement));
  });

  // Killed by: frontend/src/components/TopologyTab.tsx :: <div className="flex flex-wrap items-center gap-3">
  // Becomes: <div className="flex items-center gap-3">
  it('Topology: the Refresh control wraps', () => {
    render(<TopologyTab roomId={null} />);
    expectWraps(rowOf(screen.getByRole('button', { name: 'Refresh' })));
  });

  // Killed by: frontend/src/components/EvaluationsTab.tsx :: flex flex-wrap items-center justify-between gap-4
  // Becomes: flex flex-col sm:flex-row items-start sm:items-center justify-between gap-4
  it('Evaluations: the title and the view controls wrap', () => {
    render(<EvaluationsTab evaluationsData={null} onRefresh={vi.fn()} isLoading={false} />);
    const heading = screen.getByRole('heading', { name: /Quality & Evaluation Dashboard/ });
    // The heading's own row is the title-and-badge line; the header is the row that holds it.
    expectWraps(rowOf(rowOf(heading)));
  });

  // Killed by: frontend/src/components/EvaluationsTab.tsx :: <div className="flex flex-wrap items-center gap-2">
  // Becomes: <div className="flex items-center gap-2">
  it('Evaluations: the view switcher and its neighbours wrap', () => {
    render(<EvaluationsTab evaluationsData={null} onRefresh={vi.fn()} isLoading={false} />);
    const switcher = rowOf(screen.getByRole('button', { name: 'Overview' }));
    expectWraps(rowOf(switcher));
  });

  // Killed by: frontend/src/components/EvaluationsTab.tsx :: inline-flex flex-wrap bg-slate-950/80
  // Becomes: inline-flex bg-slate-950/80
  it('Evaluations: the four view buttons wrap within the switcher', () => {
    render(<EvaluationsTab evaluationsData={null} onRefresh={vi.fn()} isLoading={false} />);
    expectWraps(rowOf(screen.getByRole('button', { name: 'Overview' })));
  });

  // The two filter groups are one identical line each, so neither can be named by a needle
  // that occurs once; their mutations are recorded in the #1029 PR instead.
  it('Activity: the category and status filters wrap within their groups', () => {
    render(<ActivityTimeline roomId={null} seatId={null} events={[]} />);
    expectWraps(rowOf(screen.getByRole('button', { name: 'Mutations' })));
    expectWraps(rowOf(screen.getByRole('button', { name: 'All Status' })));
  });
});
