import { describe, expect, it, vi, beforeEach } from 'vitest';
import { render, screen, fireEvent, waitFor } from '@testing-library/react';
import { ResourceSummary, RESOURCE_ROOM_COPY } from './ResourceSummary';
import { BudgetData, RoomContext, RoomState } from '../../types';

const mockBudgetData: BudgetData = {
  session_budget: {
    max_tokens: 1_000_000,
    used_input_tokens: 15_200,
    used_output_tokens: 3_800,
    total_used_tokens: 19_000,
    remaining_tokens: 981_000,
    budget_used_pct: 1.9,
  },
  providers: {
    ollama: {
      provider: 'ollama',
      input_tokens: 15_200,
      output_tokens: 3_800,
      models: ['qwen2.5:latest'],
    },
  },
  roles: {},
  compaction_history: [
    {
      id: 'cmp-1',
      timestamp: '2026-09-13 15:30:00',
      reason: '70% threshold reached',
      original_tokens: 14_000,
      compacted_tokens: 9_800,
      saved_tokens: 4_200,
      compression_ratio_pct: 30.0,
      kept_turns: 5,
    },
  ],
};

/**
 * A room the way `GET /api/rooms/{id}` answers it, with the two fields this surface reads
 * spelled out per test. Not a `Partial<RoomState>` cast: the point of #1272 is that these
 * readings come off the room's real shape, so the fixture is that shape.
 */
const roomWith = (over: {
  turn_state?: Partial<RoomState['turn_state']>;
  policy?: Partial<RoomState['policy']>;
  participants?: RoomState['participants'];
} = {}): RoomState => ({
  room_id: 'room-1',
  title: 'Design review',
  participants: over.participants ?? [
    { id: 'architect', kind: 'agent', display_name: 'Architect' },
    { id: 'reviewer', kind: 'agent', display_name: 'Reviewer' },
  ],
  transcript: [],
  turn_state: { agent_turns_since_human: 0, ...over.turn_state },
  policy: {
    max_agent_turns_per_human_message: 6,
    max_span_messages: 40,
    transcript_window: 30,
    hesitation_seconds: 2,
    default_responder_id: 'architect',
    ...over.policy,
  },
});

/** `GET /api/rooms/{id}/context`: one entry per seat, never a room-wide total. */
const contextWith = (seats: RoomContext['seats']): RoomContext => ({
  room_id: 'room-1',
  seats,
  is_saturated: seats.some((s) => s.is_saturated),
  saturation_threshold: 20,
});

describe('ResourceSummary', () => {
  beforeEach(() => {
    vi.clearAllMocks();
    vi.spyOn(globalThis, 'fetch').mockImplementation(async (url: RequestInfo | URL) => {
      const urlStr = String(url);
      if (urlStr.includes('/api/settings')) {
        return {
          ok: true,
          json: async () => ({
            llm_provider: 'ollama',
            llm_model: 'qwen2.5:latest',
            llm_base_url: 'http://127.0.0.1:11434',
            llm_api_key_set: false,
            llm_api_key_masked: '',
            comfyui_base_url: 'http://127.0.0.1:8188',
            providers_available: ['ollama'],
          }),
        } as Response;
      }
      return { ok: false, status: 404 } as Response;
    });
  });

  it('meters the open room reply cascade against the ceiling the room states', () => {
    render(
      <ResourceSummary
        budgetData={mockBudgetData}
        room={roomWith({ turn_state: { agent_turns_since_human: 8 }, policy: { max_agent_turns_per_human_message: 25 } })}
        onRefresh={vi.fn()}
        isRefreshing={false}
      />
    );

    expect(screen.getByText(RESOURCE_ROOM_COPY.heading)).toBeDefined();
    expect(screen.getByTestId('room-reply-budget-readout')).toHaveTextContent('8 / 25 (32%)');

    const progressBar = screen.getByTestId('step-progress-bar');
    expect(progressBar.style.width).toBe('32%');
  });

  it('says there is nothing to measure when no conversation is open', () => {
    render(<ResourceSummary budgetData={mockBudgetData} onRefresh={vi.fn()} isRefreshing={false} />);

    expect(screen.getByTestId('room-readings-unavailable')).toHaveTextContent(
      RESOURCE_ROOM_COPY.noRoom,
    );
    expect(screen.queryByTestId('room-reply-budget-readout')).not.toBeInTheDocument();
    expect(screen.queryByTestId('step-progress-bar')).not.toBeInTheDocument();
  });

  it('renders token quotas and the prompt/completion breakdown, and no cost', () => {
    const { container } = render(
      <ResourceSummary
        budgetData={mockBudgetData}
        onRefresh={vi.fn()}
        isRefreshing={false}
      />
    );

    expect(screen.getByText('19,000')).toBeDefined();
    expect(screen.getByText('15,200')).toBeDefined();
    expect(screen.getByText('3,800')).toBeDefined();
    // UClone-X counts tokens only (#1392): no spend, no ceiling, no dollar figure anywhere.
    expect(screen.queryByText('Cost Ceiling')).toBeNull();
    expect(container.textContent).not.toContain('$');
  });

  it('displays dynamic context pruning and token savings', () => {
    render(
      <ResourceSummary
        budgetData={mockBudgetData}
        onRefresh={vi.fn()}
        isRefreshing={false}
      />
    );

    expect(screen.getByText('Dynamic Context Pruning & Savings')).toBeDefined();
    expect(screen.getByText('1 Compactions')).toBeDefined();
    expect(screen.getByText('4,200 tokens')).toBeDefined();
    expect(screen.getByText('+4,200 saved')).toBeDefined();
  });

  // What the retired Budget surface's ledger table showed per event, now on each log row.
  // Killed by: frontend/src/components/artifacts/ResourceSummary.tsx :: <span>{rec.reason}</span>
  // Becomes: <span />
  it('shows each compaction event with its reason, token counts and kept turns', () => {
    render(<ResourceSummary budgetData={mockBudgetData} onRefresh={vi.fn()} isRefreshing={false} />);

    const row = screen.getByTestId('resource-compaction-0');
    expect(row).toHaveTextContent('70% threshold reached');
    expect(row).toHaveTextContent('14,000 ➔ 9,800 (-30.0%), kept 5 turns');
  });

  // The per-provider split and model list the retired Budget surface showed.
  it('shows each provider with its input/output split and the models it served', () => {
    render(<ResourceSummary budgetData={mockBudgetData} onRefresh={vi.fn()} isRefreshing={false} />);

    const row = screen.getByTestId('resource-provider-ollama');
    expect(row).toHaveTextContent('In: 15,200');
    expect(row).toHaveTextContent('Out: 3,800');
    expect(row).toHaveTextContent('qwen2.5:latest');
  });

  it('displays connected local Ollama provider status and verifies 0 mock leakage', async () => {
    render(
      <ResourceSummary
        budgetData={mockBudgetData}
        onRefresh={vi.fn()}
        isRefreshing={false}
      />
    );

    await waitFor(() => {
      expect(screen.getByTestId('connected-model-name').textContent).toContain('qwen2.5:latest');
      expect(screen.getByTestId('connected-provider-name').textContent?.toLowerCase()).toContain('ollama');
      expect(screen.getByText('100% Local / Private')).toBeDefined();
    });

    // Invariant: absolutely zero mock LLM connector or fake benchmark leak
    expect(screen.queryByText(/Mock LLM Connector/i)).toBeNull();
  });

  it('triggers onRefresh callback when refresh button is clicked', () => {
    const onRefreshMock = vi.fn();
    render(
      <ResourceSummary
        budgetData={mockBudgetData}
        onRefresh={onRefreshMock}
        isRefreshing={false}
      />
    );

    const refreshBtn = screen.getByTestId('resource-refresh-btn');
    fireEvent.click(refreshBtn);
    expect(onRefreshMock).toHaveBeenCalled();
  });
});

/**
 * The readings #1059 moved off U0's default screen, and #1272 re-sourced.
 *
 * The rail stated its own contract -- `ui-kit/rail/ConversationList.tsx`, "The rail's one job:
 * list conversations" -- and then ended in a step-budget meter, a turn counter and a token
 * total. ui-authoring §2 step 4 calls token accounting an instrument, and instruments are U1's
 * and U3's: one deliberate action away, never in the default state. So they are here, on the
 * dock's Resource surface.
 *
 * #1272 is about *what they count*. They were derived from `chatMessages`, which is
 * `GET /api/session/history` for whichever clone the rail has selected -- and since #1208
 * retired the single-agent surface the centre column is unconditionally a room, so those
 * figures named a transcript the user could not see. Picking a clone moved them; opening a
 * room did not. Two of the four have no room-level equivalent at all, and the surface says so
 * in words rather than showing a number from the other store (P6).
 *
 * These pin the *arrival*. `WorkspaceSidebar.test.tsx` pins the departure, and pins it against
 * this file so that a departure with no arrival cannot pass as a fix.
 */
describe('ResourceSummary reads the room on screen, not the retired history (#1272)', () => {
  const renderResource = (over: Partial<React.ComponentProps<typeof ResourceSummary>> = {}) =>
    render(
      <ResourceSummary
        budgetData={mockBudgetData}
        room={roomWith({ turn_state: { agent_turns_since_human: 3 } })}
        roomContext={contextWith([
          { participant_id: 'architect', session_id: 's-a', active_turns: 7, is_saturated: false, used_tokens: null, max_context_tokens: null, context_window_source: null, live: true },
        ])}
        onRefresh={() => {}}
        isRefreshing={false}
        {...over}
      />,
    );

  // The defect itself, in one case. The room and the retired store disagree on every reading,
  // and the surface must be reporting the room. There is no `chatMessages`-shaped prop left to
  // pass, which is the structural half of the fix -- so the contradiction is staged the only
  // way it still can be: the session ledger beside it holds 19,000 tokens and the per-seat
  // figures hold 7 turns, and the conversation block states neither a token total nor a step
  // count while its two live figures are the room's own.
  // Killed by: frontend/src/components/artifacts/ResourceSummary.tsx :: const repliesSpent = room?.turn_state.agent_turns_since_human ?? 0;
  // Becomes: const repliesSpent = 0;
  it('meters the reply cascade from the room, not from any other transcript', () => {
    renderResource({
      room: roomWith({
        turn_state: { agent_turns_since_human: 4 },
        policy: { max_agent_turns_per_human_message: 8 },
      }),
    });

    expect(screen.getByTestId('room-reply-budget-readout')).toHaveTextContent('4 / 8 (50%)');
    // The session ledger is a different quantity and must not have leaked into this block.
    expect(screen.getByTestId('room-resource-readings').textContent).not.toContain('19,000');
  });

  // A room has no token total to show: `RoomMessage.usage` is declared and nothing fills it on
  // the orchestrated path, so a room's turns carry no token count to total, and no HTTP route
  // serves one. P6: say the absence, do not substitute the plausible neighbour.
  // Killed by: frontend/src/components/artifacts/ResourceSummary.tsx :: <span>{RESOURCE_ROOM_COPY.noTokenTotal}</span>
  // Becomes: <span>{sessionBudget.total_used_tokens.toLocaleString()}</span>
  it('states the absent token total in words, and names the session ledger as a different thing', () => {
    renderResource();

    const absent = screen.getByTestId('room-token-total-absent');
    expect(absent).toHaveTextContent('No token total');
    expect(absent).toHaveTextContent('a different quantity');
    // The old readout is gone, not renamed: a number here would be the substituted value.
    expect(screen.queryByTestId('token-total-readout')).not.toBeInTheDocument();
    expect(screen.queryByTestId('token-total-estimated')).not.toBeInTheDocument();
  });

  // The same for steps. `AgentConfig.max_steps` bounds one agent run; a room is served by one
  // run per seat per turn and reports none of their step counts on any route the head calls.
  // Killed by: frontend/src/components/artifacts/ResourceSummary.tsx :: {RESOURCE_ROOM_COPY.noStepCount}
  // Becomes: {`Step ${repliesSpent} / ${repliesCeiling}`}
  it('states the absent step count in words rather than relabelling the reply budget', () => {
    renderResource();

    expect(screen.getByTestId('room-step-count-absent')).toHaveTextContent('No step count');
    expect(screen.queryByTestId('step-budget-readout')).not.toBeInTheDocument();
    // `Step Budget` was the old label on a figure that is not this one. Reusing the word on
    // the reply budget would re-state the quantity the room does not have.
    expect(screen.queryByText(/Step Budget/)).not.toBeInTheDocument();
    // `Turn Budget` is likewise not this surface's phrase -- `test_workspace_layout_e2e.py`
    // asserts it appears nowhere on the page.
    expect(screen.queryByText(/Turn Budget/)).not.toBeInTheDocument();
  });

  // Turns exist per seat and only per seat. Each seat carries its own context, so a sum or a
  // max over them would be a room-level figure no producer wrote.
  //
  // This case also carries the absence of `room-seat-turns-sampled` (#1286). That note said
  // the rows were "counted up to 4 turns ago", which was true while `contextReadIsDue`
  // sampled the read every four landed turns and is false now the head reads on every one.
  // The absence is asserted here rather than in a case of its own because there is nothing
  // left in the component to mutate for it -- what pins the behaviour the note described is
  // `App.test.tsx`'s per-turn read case, which can restore the gate and watch the reads stop.
  // Killed by: frontend/src/components/artifacts/ResourceSummary.tsx :: {roomContext.seats.map((seat) => (
  // Becomes: {roomContext.seats.slice(0, 1).map((seat) => (
  it('reports turns once per seat, with the threshold the Core sent, and never totals them', () => {
    renderResource({
      roomContext: contextWith([
        { participant_id: 'architect', session_id: 's-a', active_turns: 22, is_saturated: true, used_tokens: null, max_context_tokens: null, context_window_source: null, live: true },
        { participant_id: 'reviewer', session_id: 's-r', active_turns: 3, is_saturated: false, used_tokens: null, max_context_tokens: null, context_window_source: null, live: false },
      ]),
    });

    const architect = screen.getByTestId('room-seat-turns-architect');
    expect(architect).toHaveTextContent('Architect');
    expect(architect).toHaveTextContent('22');
    expect(architect).toHaveTextContent('/ 20');
    expect(screen.getByTestId('room-seat-saturated-architect')).toHaveTextContent('Saturated');

    const reviewer = screen.getByTestId('room-seat-turns-reviewer');
    expect(reviewer).toHaveTextContent('3');
    expect(screen.queryByTestId('room-seat-saturated-reviewer')).not.toBeInTheDocument();

    // No single turn number anywhere: 25 is the sum and 22 the max, and neither is a figure
    // the Core reports for a room.
    expect(screen.queryByTestId('conversation-turns-readout')).not.toBeInTheDocument();
    expect(screen.getByTestId('room-seat-turns').textContent).not.toContain('25');

    // Gone with the staleness it described (#1286): no note anywhere claims an age for
    // these rows, and none should, because they are now read on every landed turn.
    expect(screen.queryByTestId('room-seat-turns-sampled')).not.toBeInTheDocument();
    expect(screen.queryByText(/Counted up to 4 turns ago/)).not.toBeInTheDocument();
  });

  // #939's qualifier, re-pointed at the thing that is actually uncertain here. A seat nobody
  // has spoken in this process answers from its persisted record, which is behind any turn an
  // earlier process did not write -- so the figure says where it came from.
  // Killed by: frontend/src/components/artifacts/ResourceSummary.tsx :: {!seat.live && (
  // Becomes: {false && (
  it('says which seat figures came from a saved record rather than a live session', () => {
    renderResource({
      roomContext: contextWith([
        { participant_id: 'architect', session_id: 's-a', active_turns: 7, is_saturated: false, used_tokens: null, max_context_tokens: null, context_window_source: null, live: true },
        { participant_id: 'reviewer', session_id: 's-r', active_turns: 2, is_saturated: false, used_tokens: null, max_context_tokens: null, context_window_source: null, live: false },
      ]),
    });

    expect(screen.getByTestId('room-seat-from-record-reviewer')).toHaveTextContent(
      RESOURCE_ROOM_COPY.fromRecord,
    );
    expect(screen.queryByTestId('room-seat-from-record-architect')).not.toBeInTheDocument();
  });

  // An unread context is not a zero-turn conversation, and the surface must not draw one as
  // the other.
  // Killed by: frontend/src/components/artifacts/ResourceSummary.tsx :: {roomContext === null ? (
  // Becomes: {false ? (
  it('distinguishes an unread context from an empty one', () => {
    renderResource({ roomContext: null });

    expect(screen.getByTestId('room-seat-turns-unread')).toHaveTextContent('Not read yet');
    expect(screen.queryByTestId('room-seat-turns')).not.toBeInTheDocument();
    // The reply budget still reads, because it comes off the room and not off the context.
    expect(screen.getByTestId('room-reply-budget-readout')).toHaveTextContent('3 / 6');
  });
});
