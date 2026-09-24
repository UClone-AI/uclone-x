import { describe, expect, it, vi } from 'vitest';
import { render, screen } from '@testing-library/react';
import { LedgerTab } from './LedgerTab';
import type { EventEnvelope } from '../types';

/**
 * The ledger offers the priorities the stream can carry, and no others (#1027).
 *
 * `src/uclone_x/ui/app.py` stamps the SSE envelope's `priority` by envelope type: `"P1"` on
 * `SYSTEM_CONNECTED`, `"P2"` on `AGENT_EVENT`, `"P3"` on `HEARTBEAT`. It reads
 * `AgentEvent.priority` for none of them, and `EventPriority.CRITICAL` -- the int `0` -- is
 * never serialised, so `"P0"` is a value no event can have. The filter offered `P0 -
 * Critical` anyway: choosing it emptied the ledger and said only that nothing matched the
 * criteria, which is the absence-without-a-cause P6 forbids.
 */

function renderLedger(events: EventEnvelope[] = []) {
  render(
    <LedgerTab
      events={events}
      isPaused={false}
      onTogglePause={vi.fn()}
      onClearEvents={vi.fn()}
      lastHeartbeat="--"
    />,
  );
}

/** The priority filter, found by the option every ledger has. */
function priorityFilter(): HTMLSelectElement {
  const select = screen.getByRole('option', { name: 'All Priorities' }).closest('select');
  if (select === null) throw new Error('the priority options are not inside a select');
  return select;
}

describe('LedgerTab priority filter (#1027)', () => {
  // Killed by: frontend/src/components/LedgerTab.tsx :: <option value="ALL">All Priorities</option>
  // Becomes: <option value="ALL">All Priorities</option><option value="P0">P0 - Critical</option>
  it('offers only the priorities the Core sends', () => {
    renderLedger();

    expect([...priorityFilter().options].map((option) => option.value)).toEqual([
      'ALL',
      'P1',
      'P2',
      'P3',
    ]);
  });

  // Killed by: frontend/src/components/LedgerTab.tsx :: {ev.priority}
  // Becomes: {null}
  it('still shows an event its own priority', () => {
    renderLedger([
      { id: 'e1', type: 'AGENT_EVENT', priority: 'P2', topic: 'engine', timestamp: 1 },
    ]);

    expect(screen.getByText('P2')).toBeInTheDocument();
  });
});
