import { describe, expect, it, vi } from 'vitest';
import { fireEvent, render, screen } from '@testing-library/react';
import { EvaluationsTab } from './EvaluationsTab';
import type { EvaluationsData } from '../types';

/**
 * The scorecard's four metric cards, at the layer jsdom can see.
 *
 * The cards are laid out by `lg:grid-cols-4` -- a *viewport* breakpoint inside a dock whose
 * width the reader drags -- so in the default 520px dock on a wide screen all four went
 * across it, about 110px each, and each held 133-155px of reading. `overflow-hidden` then
 * cut the excess off, which is the part that made it a defect rather than a cramped panel:
 * "axiomatic defense" was drawn as "axiomatic def" and read as a complete phrase.
 *
 * The fix is `.dock-scope`'s new container query (`index.css`), and it is the fix these two
 * cases cannot check: jsdom resolves no stylesheet and evaluates no container query. What
 * they pin is the card's own half of it -- that a card too narrow for its words neither
 * hides them nor keeps them on one line. `tests/e2e/test_dock_metric_cards_e2e.py` measures
 * the layout in a browser at two dock widths.
 */
describe('the Evaluations scorecard cards', () => {
  const renderTab = () =>
    render(<EvaluationsTab evaluationsData={null} onRefresh={vi.fn()} isLoading={false} />);

  // Killed by: frontend/src/components/EvaluationsTab.tsx :: 'p-4 bg-slate-900/80 border border-slate-800 rounded-2xl shadow-lg group hover:border-slate-700 transition-all';
  // Becomes: 'relative overflow-hidden p-4 bg-slate-900/80 border border-slate-800 rounded-2xl shadow-lg group hover:border-slate-700 transition-all';
  it('lets a card overflow its own words rather than cutting them off', () => {
    renderTab();
    const cards = screen.getAllByTestId('eval-metric-card');

    expect(cards).toHaveLength(4);
    for (const card of cards) {
      // Not `toHaveClass`, which would pass on a card that clipped by some other route; the
      // question is whether anything on this card hides what does not fit.
      expect(card.className, card.className).not.toMatch(/\boverflow-hidden\b/);
    }
  });

  // Killed by: frontend/src/components/EvaluationsTab.tsx :: const METRIC_VALUE_ROW = 'mt-2 flex flex-wrap items-baseline gap-x-2';
  // Becomes: const METRIC_VALUE_ROW = 'mt-2 flex items-baseline gap-2';
  it('puts a reading`s unit under the figure where it will not fit beside it', () => {
    renderTab();

    for (const card of screen.getAllByTestId('eval-metric-card')) {
      const value = card.querySelector('.items-baseline');
      expect(value, card.textContent).not.toBeNull();
      expect(value?.className, value?.className).toMatch(/\bflex-wrap\b/);
    }
  });
});

/**
 * An empty scorecard is three different facts, and the server names which (P6): no suites
 * are installed, none has been run, or the reports could not be read. Before this, a read
 * failure was returned as the same empty body as "never ran", so the tab drew the empty
 * scorecard -- "100%", "Zero Regressions" -- over a directory it had failed to read.
 */
describe('the Evaluations tab states why its scorecard is empty', () => {
  const METRICS = {
    total_suites: 0,
    total_probes: 0,
    passed_probes: 0,
    failed_probes: 0,
    pass_rate: 0,
  };
  const READ_ERROR =
    'Could not read evaluation reports from /srv/evals/reports: NotADirectoryError: [Errno 20] Not a directory';
  const renderWith = (data: EvaluationsData) =>
    render(<EvaluationsTab evaluationsData={data} onRefresh={vi.fn()} isLoading={false} />);

  // Killed by: frontend/src/components/EvaluationsTab.tsx :: evaluationsData?.status === 'error'
  // Becomes: evaluationsData?.status === 'never'
  it('shows a read failure in words, in place of the scorecard', () => {
    renderWith({ status: 'error', error: READ_ERROR, suites: [], scorecard: {}, metrics: METRICS });

    const alert = screen.getByTestId('eval-read-failure');
    expect(alert).toHaveTextContent('Evaluation results could not be read');
    expect(alert).toHaveTextContent(READ_ERROR);
    // The failure is not drawn as an empty scorecard beside it.
    expect(screen.queryAllByTestId('eval-metric-card')).toHaveLength(0);
    expect(screen.queryByText('No Evaluation Reports Found')).toBeNull();
  });

  it('shows "never ran" as the empty scorecard, with no failure', () => {
    renderWith({ status: 'empty', error: null, suites: [], scorecard: {}, metrics: METRICS });

    expect(screen.queryByTestId('eval-read-failure')).toBeNull();
    expect(screen.getAllByTestId('eval-metric-card')).toHaveLength(4);
    expect(screen.getByText('No Evaluation Reports Found')).toBeInTheDocument();
  });

  // Killed by: frontend/src/components/EvaluationsTab.tsx :: const noBackend = evaluationsData?.status === 'no_backend';
  // Becomes: const noBackend = false;
  it('says the suites are not installed rather than telling the reader to run them', () => {
    renderWith({ status: 'no_backend', error: null, suites: [], scorecard: {}, metrics: METRICS });

    expect(screen.queryByTestId('eval-read-failure')).toBeNull();
    expect(screen.getByText('No Evaluation Suites Installed')).toBeInTheDocument();
    expect(screen.queryByText('./ucx eval run')).toBeNull();
  });

  // Killed by: frontend/src/components/EvaluationsTab.tsx :: {historyFailure !== null && !isHistoryLoading ? (
  // Becomes: {false ? (
  it('shows a history read failure in words, not as "no historical reports"', async () => {
    const fetchMock = vi.fn().mockResolvedValue({
      ok: true,
      json: async () => ({ status: 'error', error: READ_ERROR, reports: [], total: 0 }),
    });
    vi.stubGlobal('fetch', fetchMock);
    try {
      renderWith({ status: 'empty', error: null, suites: [], scorecard: {}, metrics: METRICS });
      fireEvent.click(screen.getByRole('button', { name: 'History' }));

      const cell = await screen.findByTestId('eval-history-read-failure');
      expect(cell).toHaveTextContent(READ_ERROR);
      expect(screen.queryByText('No historical evaluation reports found.')).toBeNull();
    } finally {
      vi.unstubAllGlobals();
    }
  });
});
