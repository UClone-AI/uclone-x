import { describe, expect, it, vi, beforeEach } from 'vitest';
import { render, screen, fireEvent, waitFor } from '@testing-library/react';
import { KnowledgeGraphViewer } from './KnowledgeGraphViewer';
import { KnowledgeGraphResponse } from '../../types';

const mockGraphData: KnowledgeGraphResponse = {
  triples: [
    {
      subject: 'SessionManager',
      predicate: 'manages',
      object: 'DialogueSession',
      tier: 'asserted',
      provenance: {
        agent_id: 'champion',
        confidence: 1.0,
        origin: 'axiomatic',
        source_session: 'sess_1',
      },
    },
    {
      subject: 'DialogueSession',
      predicate: 'records',
      object: 'TurnObservation',
      tier: 'induced_enforcing',
      provenance: {
        agent_id: 'scout',
        confidence: 0.95,
        origin: 'derived',
        source_session: 'sess_1',
      },
    },
    {
      subject: 'TurnObservation',
      predicate: 'triggers',
      object: 'HypothesisCandidate',
      tier: 'induced_candidate',
      provenance: {
        agent_id: 'critic',
        confidence: 0.72,
        origin: 'derived',
        source_session: 'sess_1',
        description: 'Candidate hypothesis inferred from context',
      },
    },
  ],
  nodes: [
    {
      id: 'SessionManager',
      name: 'SessionManager',
      tier: 'asserted',
      type: 'entity',
      provenance: {
        agent_id: 'champion',
        confidence: 1.0,
        origin: 'axiomatic',
      },
    },
    {
      id: 'DialogueSession',
      name: 'DialogueSession',
      tier: 'asserted',
      type: 'entity',
      provenance: {
        agent_id: 'champion',
        confidence: 1.0,
        origin: 'axiomatic',
      },
    },
    {
      id: 'TurnObservation',
      name: 'TurnObservation',
      tier: 'induced_enforcing',
      type: 'concept',
      provenance: {
        agent_id: 'scout',
        confidence: 0.95,
        origin: 'derived',
      },
    },
    {
      id: 'HypothesisCandidate',
      name: 'HypothesisCandidate',
      tier: 'induced_candidate',
      type: 'concept',
      provenance: {
        agent_id: 'critic',
        confidence: 0.72,
        origin: 'derived',
        description: 'Candidate hypothesis inferred from context',
      },
    },
  ],
  edges: [
    {
      id: 'edge_1',
      source: 'SessionManager',
      target: 'DialogueSession',
      predicate: 'manages',
      tier: 'asserted',
      provenance: {
        agent_id: 'champion',
        confidence: 1.0,
      },
    },
    {
      id: 'edge_2',
      source: 'DialogueSession',
      target: 'TurnObservation',
      predicate: 'records',
      tier: 'induced_enforcing',
      provenance: {
        agent_id: 'scout',
        confidence: 0.95,
      },
    },
    {
      id: 'edge_3',
      source: 'TurnObservation',
      target: 'HypothesisCandidate',
      predicate: 'triggers',
      tier: 'induced_candidate',
      provenance: {
        agent_id: 'critic',
        confidence: 0.72,
      },
    },
  ],
  summary: {
    total_triples: 3,
    total_nodes: 4,
    total_edges: 3,
    session_id: 'sess_1',
    agent_id: 'all',
  },
};

describe('KnowledgeGraphViewer', () => {
  beforeEach(() => {
    vi.clearAllMocks();
    vi.spyOn(globalThis, 'fetch').mockImplementation(async (url: RequestInfo | URL) => {
      const urlStr = String(url);
      if (urlStr.includes('/api/rooms/room-a/knowledge')) {
        return new Response(
          JSON.stringify({
            ...mockGraphData,
            room_id: 'room-a',
            participant_id: 'scout',
            status: 'ok',
            reason: null,
            remembers: [],
          }),
          { status: 200 },
        );
      }
      return { ok: false, status: 404 } as Response;
    });
  });

  it('renders graph header, node count badge, and SVG canvas', async () => {
    render(<KnowledgeGraphViewer roomId="room-a" seatId="scout" />);

    await waitFor(() => {
      expect(screen.getByTestId('kg-node-count')).toBeDefined();
      expect(screen.getByTestId('kg-node-count').textContent).toContain('4 nodes');
      expect(screen.getByTestId('node-SessionManager')).toBeDefined();
    });

    expect(screen.getByTestId('kg-svg-canvas')).toBeDefined();
    expect(screen.getByTestId('node-DialogueSession')).toBeDefined();
    expect(screen.getByTestId('node-TurnObservation')).toBeDefined();
    expect(screen.getByTestId('node-HypothesisCandidate')).toBeDefined();
  });

  it('filters nodes by tier selection', async () => {
    render(<KnowledgeGraphViewer roomId="room-a" seatId="scout" />);

    await waitFor(() => {
      expect(screen.getByTestId('node-SessionManager')).toBeDefined();
    });

    const tierSelect = screen.getByTestId('tier-filter-select');
    fireEvent.change(tierSelect, { target: { value: 'induced_candidate' } });

    await waitFor(() => {
      expect(screen.queryByTestId('node-SessionManager')).toBeNull();
      expect(screen.getByTestId('node-HypothesisCandidate')).toBeDefined();
    });
  });

  it('filters nodes by search query input', async () => {
    render(<KnowledgeGraphViewer roomId="room-a" seatId="scout" />);

    await waitFor(() => {
      expect(screen.getByTestId('node-SessionManager')).toBeDefined();
    });

    const searchInput = screen.getByTestId('kg-search-input');
    fireEvent.change(searchInput, { target: { value: 'Hypothesis' } });

    await waitFor(() => {
      expect(screen.queryByTestId('node-SessionManager')).toBeNull();
      expect(screen.getByTestId('node-HypothesisCandidate')).toBeDefined();
    });
  });

  it('switches between Graph and Triples Table views', async () => {
    render(<KnowledgeGraphViewer roomId="room-a" seatId="scout" />);

    await waitFor(() => {
      expect(screen.getByTestId('kg-svg-canvas')).toBeDefined();
    });

    const tableBtn = screen.getByTestId('view-table-btn');
    fireEvent.click(tableBtn);

    await waitFor(() => {
      expect(screen.getByTestId('kg-triples-table')).toBeDefined();
      expect(screen.getByText('manages')).toBeDefined();
      expect(screen.getByText('records')).toBeDefined();
      expect(screen.getByText('triggers')).toBeDefined();
    });

    const graphBtn = screen.getByTestId('view-graph-btn');
    fireEvent.click(graphBtn);

    await waitFor(() => {
      expect(screen.getByTestId('kg-svg-canvas')).toBeDefined();
    });
  });

  it('opens provenance inspection popover on node click (P6 in-band traceability)', async () => {
    render(<KnowledgeGraphViewer roomId="room-a" seatId="scout" />);

    await waitFor(() => {
      expect(screen.getByTestId('node-HypothesisCandidate')).toBeDefined();
    });

    // Click on node
    fireEvent.click(screen.getByTestId('node-HypothesisCandidate'));

    await waitFor(() => {
      expect(screen.getByTestId('kg-provenance-popover')).toBeDefined();
    });
    expect(screen.getAllByText('critic').length).toBeGreaterThan(0);
    expect(screen.getByText('72%')).toBeDefined();
    expect(screen.getByText('Candidate hypothesis inferred from context')).toBeDefined();

    // Close popover
    fireEvent.click(screen.getByTestId('close-provenance-btn'));
    expect(screen.queryByTestId('kg-provenance-popover')).toBeNull();
  });

  it('opens provenance inspection popover on edge click', async () => {
    render(<KnowledgeGraphViewer roomId="room-a" seatId="scout" />);

    await waitFor(() => {
      expect(screen.getByTestId('edge-edge_1')).toBeDefined();
    });

    fireEvent.click(screen.getByTestId('edge-edge_1'));

    await waitFor(() => {
      expect(screen.getByTestId('kg-edge-popover')).toBeDefined();
      expect(screen.getByText(/SessionManager ➔ manages ➔ DialogueSession/)).toBeDefined();
    });
  });

  it('reads the seat graph for the conversation on screen, by room and seat', async () => {
    render(<KnowledgeGraphViewer roomId="room-a" seatId="scout" />);
    await waitFor(() => {
      expect(screen.getByTestId('kg-node-count')).toBeDefined();
    });
    expect(globalThis.fetch).toHaveBeenCalledWith('/api/rooms/room-a/knowledge?agent_id=scout');
    expect(globalThis.fetch).not.toHaveBeenCalledWith(expect.stringContaining('session_id'));
  });

  it.each(['not_recorded', 'unreadable', 'no_ontology'] as const)(
    'shows the Core’s reason instead of an empty graph (%s)',
    async (status) => {
      vi.spyOn(globalThis, 'fetch').mockImplementation(
        async () =>
          new Response(
            JSON.stringify({
              room_id: 'room-a',
              participant_id: 'scout',
              session_id: 's',
              status,
              reason: `Scout cannot be read (${status}).`,
              remembers: null,
              triples: null,
              nodes: null,
              edges: null,
              summary: null,
            }),
            { status: 200 },
          ),
      );
      render(<KnowledgeGraphViewer roomId="room-a" seatId="scout" />);
      expect(await screen.findByTestId('kg-notice')).toHaveTextContent(
        `Scout cannot be read (${status}).`,
      );
      expect(screen.queryByTestId('kg-node-count')).toBeNull();
    },
  );

  // An `ok` graph with nothing in it is "nothing listed", and the Core says why: the head
  // cannot tell an empty record from one it was not told about (#1366).
  // Killed by: frontend/src/components/artifacts/KnowledgeGraphViewer.tsx :: (read.data?.reason ?? null);
  // Becomes: (read.data && read.data.status !== 'ok' ? read.data.reason : null);
  it('shows the Core’s reason for an empty graph it could read', async () => {
    const reason = 'Scout has no remembered facts on record in this conversation.';
    vi.spyOn(globalThis, 'fetch').mockImplementation(
      async () =>
        new Response(
          JSON.stringify({
            room_id: 'room-a',
            participant_id: 'scout',
            session_id: 's',
            status: 'ok',
            reason,
            remembers: [],
            triples: [],
            nodes: [],
            edges: [],
            summary: { total_triples: 0, total_nodes: 0, total_edges: 0 },
          }),
          { status: 200 },
        ),
    );
    render(<KnowledgeGraphViewer roomId="room-a" seatId="scout" />);
    expect(await screen.findByTestId('kg-notice')).toHaveTextContent(reason);
  });
});
