import { describe, it, expect, vi } from 'vitest';
import { render, screen, fireEvent, within } from '@testing-library/react';
import { RAIL_WIDTH_CLASS, RAIL_WIDTH_PX, WorkspaceSidebar } from './WorkspaceSidebar';
import { makeAgentInfo, makePersonaInfo } from '../../test/fixtures';
import { ResourceSummary, RESOURCE_ROOM_COPY } from '../artifacts/ResourceSummary';
import type { RoomContext, RoomState, RoomSummary } from '../../types';

const agents = [makeAgentInfo({ id: 'agent-1' })];

const renderSidebar = (over: Partial<React.ComponentProps<typeof WorkspaceSidebar>> = {}) =>
  render(
    <WorkspaceSidebar
      agents={agents}
      selectedAgent="agent-1"
      onSelectAgent={() => {}}
      rooms={[]}
      unreadableRoomIds={[]}
      currentRoomId={null}
      onSelectRoom={() => {}}
      onNewRoom={() => {}}
      onRenameRoom={async () => {}}
      onDeleteRoom={async () => {}}
      modelConfigured
      overlay={false}
      {...over}
    />,
  );

/**
 * The rail gave these readings up to the dock (#1059) -- it did not lose them.
 *
 * The rail states its own contract one component up (`ui-kit/rail/ConversationList.tsx`: "The
 * rail's one job: list conversations") and then ended in a step-budget meter, a turn counter
 * and a token total. ui-authoring §2 puts instruments one deliberate action away rather than
 * in U0's default state, so they went to the dock's Resource surface.
 *
 * **Both halves are asserted in one test on purpose.** An absence assertion alone is not a
 * guard: it passes just as well on the day someone deletes the feature outright, and it would
 * have passed on this fixture before the move for any readout the fixture never lit. So the
 * same numbers are rendered into `ResourceSummary` in the same case, and the test fails
 * whether the readings come back to the rail or vanish from the dock.
 */
describe('WorkspaceSidebar readouts moved to the dock (#1059)', () => {
  // The kill is on the destination -- see the note in the test. #1272 replaced what the
  // destination shows (the room's readings, not the retired single-agent store's), so the
  // declaration moved with it onto a sentence the arrival half reads.
  // Killed by: frontend/src/components/artifacts/ResourceSummary.tsx :: {RESOURCE_ROOM_COPY.noTokenTotal}
  // Becomes: {RESOURCE_ROOM_COPY.heading}
  it('draws no step budget, turn or token reading, and the dock draws all three', () => {
    // 1. Not on U0's default screen. Every one of these lit the rail before #1059, and the
    //    rail is given a conversation long and expensive enough to light all of them.
    const conversation: RoomSummary = {
      room_id: 'room-1',
      title: 'A long conversation',
      agent_ids: ['agent-1'],
      human_ids: ['user'],
      message_count: 40,
      updated_at: '2026-09-17T00:00:00Z',
    };
    const { unmount } = renderSidebar({ rooms: [conversation], currentRoomId: 'room-1' });
    // The dock's own testids, not the retired ones #1272 deleted: an absence assertion on a
    // testid that exists nowhere passes on any tree at all and guards nothing.
    for (const testid of [
      'room-reply-budget-readout',
      'room-seat-turns',
      'room-step-count-absent',
      'room-token-total-absent',
    ]) {
      expect(screen.queryByTestId(testid)).not.toBeInTheDocument();
    }
    expect(screen.queryByText('Step Budget')).not.toBeInTheDocument();
    expect(screen.queryByText('Saturated')).not.toBeInTheDocument();
    unmount();

    // 2. Reachable, one keystroke away. Without this half the case above would also pass on
    //    a change that deleted the readings instead of moving them.
    const room: RoomState = {
      room_id: 'room-1',
      title: 'A long conversation',
      participants: [{ id: 'agent-1', kind: 'agent', display_name: 'Agent One' }],
      transcript: [],
      turn_state: { agent_turns_since_human: 3 },
      policy: {
        max_agent_turns_per_human_message: 50,
        max_span_messages: 40,
        transcript_window: 30,
        hesitation_seconds: 2,
        default_responder_id: 'agent-1',
      },
    };
    const roomContext: RoomContext = {
      room_id: 'room-1',
      seats: [
        { participant_id: 'agent-1', session_id: 's-1', active_turns: 40, is_saturated: true, used_tokens: null, max_context_tokens: null, context_window_source: null, live: true },
      ],
      is_saturated: true,
      saturation_threshold: 20,
    };
    render(
      <ResourceSummary
        budgetData={null}
        room={room}
        roomContext={roomContext}
        onRefresh={() => {}}
        isRefreshing={false}
      />,
    );
    expect(screen.getByTestId('room-reply-budget-readout')).toHaveTextContent('3 / 50');
    expect(screen.getByTestId('room-seat-turns-agent-1')).toHaveTextContent('40');
    expect(screen.getByTestId('room-seat-saturated-agent-1')).toHaveTextContent('Saturated');
    expect(screen.getByTestId('room-step-count-absent')).toHaveTextContent(
      RESOURCE_ROOM_COPY.noStepCount,
    );
    expect(screen.getByTestId('room-token-total-absent')).toHaveTextContent(
      RESOURCE_ROOM_COPY.noTokenTotal,
    );
  });
});

describe('WorkspaceSidebar clones section', () => {
  it('labels the section as Clones, never Agents or Active Swarm (#869, R3)', () => {
    renderSidebar({ agents: [makeAgentInfo({ id: 'champion', label: 'Champion', role: 'Coach' })] });
    expect(screen.getByText('Clones')).toBeInTheDocument();
    // `Agents` and `Personas` were two words this head used for one thing a user sees; the
    // product's word is `Clones` (§3.2.2 R3). The types keep their names -- this pins what
    // is read, not what is typed.
    expect(screen.queryByText('Agents')).not.toBeInTheDocument();
    expect(screen.queryByText('Active Swarm')).not.toBeInTheDocument();
    expect(screen.queryByText(/nodes/i)).not.toBeInTheDocument();
  });

  it('omits robotic count when only 1 agent is present and shows untruncated role', () => {
    renderSidebar({ agents: [makeAgentInfo({ id: 'champion', label: 'Champion', role: 'Coach' })] });
    expect(screen.getByText('Champion')).toBeInTheDocument();
    expect(screen.getByText('Coach')).toBeInTheDocument();
    expect(screen.queryByText(/\d+\s+active/i)).not.toBeInTheDocument();
  });

  it('shows active count when multiple agents are collaborating', () => {
    renderSidebar({
      agents: [
        makeAgentInfo({ id: 'champion', label: 'Champion', role: 'Coach' }),
        makeAgentInfo({ id: 'researcher', label: 'Researcher', role: 'Specialist' }),
      ],
    });
    expect(screen.getByText('2 active')).toBeInTheDocument();
    expect(screen.getByText('Champion')).toBeInTheDocument();
    expect(screen.getByText('Researcher')).toBeInTheDocument();
    expect(screen.getByText('Specialist')).toBeInTheDocument();
  });

  // Killed by: frontend/src/ui-kit/rail/Rail.tsx :: {copy.activeClones(clones.filter((clone) => clone.live).length)}
  // Becomes: {copy.activeClones(clones.length)}
  it('counts the clones it lists, and only the ones with an instance running', () => {
    // The badge read `agents` while the rows came from `clones`. Three installed personas
    // with one running instance is the ordinary case, and it printed "1 active" over three
    // rows -- or, below two live agents, nothing at all. The number and the list now come
    // from one expression.
    renderSidebar({
      personas: [
        makePersonaInfo({ name: 'scout' }),
        makePersonaInfo({ name: 'critic' }),
        makePersonaInfo({ name: 'scribe' }),
      ],
      agents: [makeAgentInfo({ id: 'scout' })],
    });

    expect(screen.getAllByTestId(/^persona-item-/)).toHaveLength(3);
    expect(screen.getByText('1 active')).toBeInTheDocument();
    expect(screen.queryByText('3 active')).not.toBeInTheDocument();
  });

  // Killed by: frontend/src/ui-kit/rail/Rail.tsx :: clones.length > 1 && (
  // Becomes: agents.length > 1 && (
  it('shows the count beside the clones it lists, not beside the live instances', () => {
    renderSidebar({
      personas: [makePersonaInfo({ name: 'scout' }), makePersonaInfo({ name: 'critic' })],
      agents: [],
    });

    // Two rows, so the count is worth printing; zero of them are running, and "0 active"
    // over two rows is a true and useful thing to read.
    expect(screen.getByText('0 active')).toBeInTheDocument();
  });

  it('renders dynamic personas and triggers onSelectAgent when clicked', () => {
    const onSelectAgent = vi.fn();
    const personas = [
      { name: 'story_writer', role: 'Fiction Novelist', description: 'Co-author', allowed_tools: ['write_to_file'] },
    ];
    renderSidebar({ personas, onSelectAgent });
    const item = screen.getByTestId('persona-item-story_writer');
    expect(item).toBeInTheDocument();
    fireEvent.click(item);
    expect(onSelectAgent).toHaveBeenCalledWith('story_writer');
  });

  it('triggers onInspectAgent when the avatar is clicked', () => {
    const onSelectAgent = vi.fn();
    const onInspectAgent = vi.fn();
    const personas = [
      { name: 'story_writer', role: 'Fiction Novelist', description: 'Co-author', allowed_tools: ['write_to_file'] },
    ];
    renderSidebar({ personas, onSelectAgent, onInspectAgent });
    const avatar = screen.getByTestId('clone-avatar-story_writer');
    expect(avatar).toBeInTheDocument();
    fireEvent.click(avatar);
    expect(onInspectAgent).toHaveBeenCalledWith('story_writer');
    expect(onSelectAgent).not.toHaveBeenCalled();
  });

  it('triggers onEditAgent when the edit settings button is clicked', () => {
    const onSelectAgent = vi.fn();
    const onInspectAgent = vi.fn();
    const onEditAgent = vi.fn();
    const personas = [
      { name: 'story_writer', role: 'Fiction Novelist', description: 'Co-author', allowed_tools: ['write_to_file'] },
    ];
    renderSidebar({ personas, onSelectAgent, onInspectAgent, onEditAgent });
    fireEvent.click(screen.getByTestId('clone-menu-story_writer'));
    const editBtn = screen.getByTestId('persona-edit-story_writer');
    expect(editBtn).toBeInTheDocument();
    fireEvent.click(editBtn);
    expect(onEditAgent).toHaveBeenCalledWith('story_writer');
    expect(onInspectAgent).not.toHaveBeenCalled();
    expect(onSelectAgent).not.toHaveBeenCalled();
  });
});

/**
 * Clone row interactions:
 * - Avatar click -> inspect profile (mode: 'view' in dock)
 * - Card body click -> select agent / open recent conversation
 * - Settings button click -> direct edit (mode: 'edit' in dock)
 */
describe('WorkspaceSidebar clone row interactions', () => {
  const scout = makePersonaInfo({ name: 'scout', role: 'Research' });
  const critic = makePersonaInfo({ name: 'critic', role: 'Review' });

  it('triggers onSelectAgent when clone body is clicked', () => {
    const onSelectAgent = vi.fn();
    renderSidebar({ personas: [scout, critic], onSelectAgent });

    fireEvent.click(screen.getByTestId('persona-item-scout'));
    expect(onSelectAgent).toHaveBeenCalledWith('scout');
  });

  it('triggers onInspectAgent when avatar is clicked, without triggering onSelectAgent', () => {
    const onSelectAgent = vi.fn();
    const onInspectAgent = vi.fn();
    renderSidebar({ personas: [scout, critic], onSelectAgent, onInspectAgent });

    fireEvent.click(screen.getByTestId('clone-avatar-scout'));
    expect(onInspectAgent).toHaveBeenCalledWith('scout');
    expect(onSelectAgent).not.toHaveBeenCalled();
  });

  it('triggers onEditAgent when settings button is clicked, without triggering onSelectAgent or onInspectAgent', () => {
    const onSelectAgent = vi.fn();
    const onInspectAgent = vi.fn();
    const onEditAgent = vi.fn();
    renderSidebar({ personas: [scout, critic], onSelectAgent, onInspectAgent, onEditAgent });

    fireEvent.click(screen.getByTestId('clone-menu-scout'));
    fireEvent.click(screen.getByTestId('persona-edit-scout'));
    expect(onEditAgent).toHaveBeenCalledWith('scout');
    expect(onInspectAgent).not.toHaveBeenCalled();
    expect(onSelectAgent).not.toHaveBeenCalled();
  });

  it('draws no inline accordion chevron or expansion list in the clone row', () => {
    renderSidebar({ personas: [scout, critic] });

    expect(screen.queryByTestId('persona-detail-toggle-scout')).not.toBeInTheDocument();
    expect(screen.queryByTestId('clone-expansion-scout')).not.toBeInTheDocument();
  });
});

/**
 * What a clone's row says about the instance behind it.
 *
 * The row compared `status` against `'busy'`, a value `/api/agents` has never sent: it sends
 * `AgentState` (`src/uclone_x/agent/models.py`), whose eight values are `IDLE`, `INGESTING`,
 * `REASONING`, `CALLING_TOOL`, `AWAITING_INPUT`, `EMITTING_RESPONSE`, `ERROR` and
 * `TERMINATED`. So the amber branch was unreachable and a clone mid-tool-call, in `ERROR` or
 * stopped drew the same emerald dot as an idle one. These pin the states the rail must tell
 * apart, and that it tells them apart in a word, not only in a hue.
 */
describe('WorkspaceSidebar clone liveness (#1061)', () => {
  const withStatus = (status: string) =>
    renderSidebar({
      personas: [],
      agents: [makeAgentInfo({ id: 'scout', label: 'Scout', status })],
    });

  // Killed by: frontend/src/ui-kit/rail/Rail.tsx :: CALLING_TOOL: 'busy',
  // Becomes: CALLING_TOOL: 'idle',
  it('says a clone is working, in a word beside its name', () => {
    withStatus('CALLING_TOOL');

    expect(screen.getByTestId('clone-liveness-word-scout')).toHaveTextContent('working');
    expect(screen.getByTestId('clone-liveness-scout')).toHaveAttribute(
      'title',
      'Working on something right now.',
    );
  });

  // Killed by: frontend/src/ui-kit/rail/Rail.tsx :: ERROR: 'error',
  // Becomes: ERROR: 'idle',
  it('does not draw a clone whose last run failed as a healthy one', () => {
    withStatus('ERROR');

    expect(screen.getByTestId('clone-liveness-word-scout')).toHaveTextContent('error');
    expect(screen.getByTestId('clone-liveness-scout')).toHaveAttribute(
      'title',
      'Its last run ended in an error.',
    );
  });

  // Killed by: frontend/src/ui-kit/rail/Rail.tsx :: TERMINATED: 'terminated',
  // Becomes: TERMINATED: 'idle',
  it('says a stopped clone is stopped', () => {
    withStatus('TERMINATED');

    expect(screen.getByTestId('clone-liveness-word-scout')).toHaveTextContent('stopped');
  });

  // Killed by: frontend/src/ui-kit/rail/Rail.tsx :: (LIVENESS_BY_STATE[status] ?? 'unknown')
  // Becomes: (LIVENESS_BY_STATE[status] ?? 'idle')
  it('reports a state it has not been taught as unknown, never as healthy', () => {
    // A state the runtime grows later. Folding it onto idle would put "running, nothing in
    // progress" on screen as a claim the rail cannot support (P6), so it is named instead --
    // and the raw value goes in the tooltip, because "unknown" alone is not actionable.
    withStatus('COMPACTING_CONTEXT');

    expect(screen.getByTestId('clone-liveness-word-scout')).toHaveTextContent('unknown');
    expect(screen.getByTestId('clone-liveness-scout').getAttribute('title')).toContain(
      'COMPACTING_CONTEXT',
    );
  });

  // Killed by: frontend/src/ui-kit/rail/Rail.tsx :: livenessOf(clone.live?.status)
  // Becomes: livenessOf('IDLE')
  it('says nothing extra about the two ordinary states, running and not', () => {
    // A rail that labels every row says nothing by labelling any. Idle and offline carry the
    // dot and the tooltip and no word; the four above carry all three.
    withStatus('IDLE');
    expect(screen.queryByTestId('clone-liveness-word-scout')).not.toBeInTheDocument();
    expect(screen.getByTestId('clone-liveness-scout')).toHaveAttribute(
      'title',
      'Running, with nothing in progress.',
    );

    // A persona with no instance behind it at all: the ordinary state of one nobody has
    // messaged yet, and not a fault.
    renderSidebar({ personas: [makePersonaInfo({ name: 'idler' })], agents: [] });
    expect(screen.queryByTestId('clone-liveness-word-idler')).not.toBeInTheDocument();
    expect(screen.getByTestId('clone-liveness-idler')).toHaveAttribute(
      'title',
      'Not running. It starts when you message it.',
    );
  });

  // Killed by: frontend/src/ui-kit/rail/Rail.tsx :: tone={TONE_BY_LIVENESS[liveness]}
  // Becomes: tone="success"
  it('draws busy and idle in different colours as well as different words', () => {
    // Belt and braces, in both directions: the colour is not the only carrier (the words
    // above), and the words are not the only carrier either.
    withStatus('REASONING');
    expect(screen.getByTestId('clone-liveness-scout').className).toContain('bg-amber-400');

    renderSidebar({
      personas: [],
      agents: [makeAgentInfo({ id: 'calm', label: 'Calm', status: 'IDLE' })],
    });
    expect(screen.getByTestId('clone-liveness-calm').className).toContain('bg-emerald-400');
  });
});


describe('WorkspaceSidebar with no agents and no personas (#1060)', () => {
  // Killed by: frontend/src/ui-kit/rail/Rail.tsx :: {copy.conversations.emptyCause(modelConfigured, clones.length)}
  // Becomes: <span>General Assistant</span>
  it('names no clone, because the runtime has offered none', () => {
    renderSidebar({ agents: [], personas: [], rooms: [] });

    expect(screen.queryByText('General Assistant')).not.toBeInTheDocument();
    expect(screen.getByTestId('agents-empty-cause')).toHaveTextContent(
      'No clones are set up yet',
    );
  });

  // Killed by: frontend/src/ui-kit/rail/Rail.tsx :: {copy.conversations.emptyCause(modelConfigured, clones.length)}
  // Becomes: {'No agents configured.'}
  it.each([
    ['a model is configured', true as boolean | null],
    ['the runtime has not said whether one is', null as boolean | null],
    ['no model is configured', false as boolean | null],
  ])('states the same cause and remedy as the list above it when %s', (_case, modelConfigured) => {
    renderSidebar({ agents: [], personas: [], rooms: [], modelConfigured });

    // The property, not the wording: whatever the two regions say about this one fact,
    // they say the same thing. Both read `emptyCause` in `lib/emptyStates.ts`, so a
    // change to either sentence moves both or fails here.
    const listCause = screen.getByTestId('conversations-empty-cause').textContent;
    expect(listCause).toBeTruthy();
    expect(screen.getByTestId('agents-empty-cause').textContent).toBe(listCause);
  });
});

describe('WorkspaceSidebar conversation list', () => {
  it('offers the list and its New control on a fresh install, with no conversations', () => {
    // The list used to be gated on `rooms.length > 0` while the only control that creates
    // the first conversation lived inside it, so the feature could not be reached at all.
    const onNewRoom = vi.fn();
    renderSidebar({ rooms: [], onNewRoom });

    expect(screen.getByTestId('sessions-list')).toBeVisible();
    expect(screen.getByTestId('conversation-list')).toBeVisible();
    fireEvent.click(screen.getByTestId('new-conversation-button'));

    expect(onNewRoom).toHaveBeenCalledTimes(1);
  });

  it('offers one New control, and it starts a conversation', () => {
    // There were two -- a rail "New chat" that started a one-agent session and the list's
    // "New" that started a conversation -- side by side, and a user could not tell them
    // apart. Every new conversation is a conversation (D1), so one control is the whole
    // of the choice.
    const onNewRoom = vi.fn();
    renderSidebar({ onNewRoom });

    const news = screen.getAllByRole('button', { name: /new/i });
    expect(news).toHaveLength(1);
    expect(screen.queryByTestId('new-session-button')).toBeNull();
    fireEvent.click(news[0]);

    expect(onNewRoom).toHaveBeenCalledTimes(1);
  });

  // Killed by: frontend/src/ui-kit/rail/Rail.tsx :: agentCount={clones.length}
  // Becomes: agentCount={agents.length}
  it('does not say no clones are set up directly above the clones it lists (#1088)', () => {
    // The list's empty cause was counted from `agents` -- the *running* instances -- while
    // the section below it lists `clones`. An install with personas nobody has messaged yet
    // has three of the second and none of the first, so the rail read "No clones are set up
    // yet" an inch above three clones.
    renderSidebar({
      personas: [makePersonaInfo({ name: 'scout' }), makePersonaInfo({ name: 'critic' })],
      agents: [],
      rooms: [],
    });

    expect(screen.getByTestId('persona-item-scout')).toBeInTheDocument();
    expect(screen.getByTestId('conversations-empty-cause')).toHaveTextContent(
      'No conversations yet',
    );
    expect(screen.queryByText(/No clones are set up yet/)).not.toBeInTheDocument();
  });

  it('lists a conversation that seats several agents', () => {
    renderSidebar({
      rooms: [
        {
          room_id: 'r1',
          title: 'Architecture triage',
          agent_ids: ['scout', 'critic'],
          human_ids: ['user'],
          message_count: 2,
          updated_at: '2026-09-14T00:00:00Z',
        },
      ],
    });

    expect(screen.getByTestId('conversation-r1')).toHaveTextContent('Architecture triage');
  });
});

/**
 * Tailwind's width scale, for the handful of classes the rail could plausibly carry. Stated
 * here rather than imported because the point is to compare the rail's *class* with the
 * *number* the dock reserves, and a shared constant for both would compare nothing.
 */
const TAILWIND_WIDTH_PX: Record<string, number> = {
  'w-52': 208,
  'w-56': 224,
  'w-60': 240,
  'w-64': 256,
  'w-72': 288,
};

describe('the rail width the dock reserves', () => {
  it('is the width the rail actually renders', () => {
    // `ArtifactsDock` subtracts RAIL_WIDTH_PX from the window to decide whether the workspace
    // row can seat it. If the number and the class ever part, the dock reserves the wrong
    // amount and mis-decides the overlay at exactly the widths #1030 is about -- silently,
    // since jsdom resolves no stylesheet and nothing else compares the two.
    // Killed by: frontend/src/ui-kit/rail/Rail.tsx :: export const RAIL_WIDTH_PX = 240;
    // Becomes: export const RAIL_WIDTH_PX = 200;
    renderSidebar();

    expect(screen.getByTestId('chat-sidebar').className).toContain(RAIL_WIDTH_CLASS);
    expect(TAILWIND_WIDTH_PX[RAIL_WIDTH_CLASS]).toBe(RAIL_WIDTH_PX);
  });
});

describe('the rail over the conversation (#1062)', () => {
  // jsdom lays nothing out, so these read the classes that decide the geometry; the geometry
  // itself is pinned in the browser by tests/e2e/test_rail_responsive_e2e.py.
  it('is drawn over the workspace, out of the row, when told to overlay', () => {
    // Killed by: frontend/src/ui-kit/rail/Rail.tsx :: overlay ? 'absolute inset-y-0 left-0 z-30 bg-slate-950 shadow-2xl' : 'relative bg-slate-950/80'
    // Becomes: 'relative bg-slate-950/80'
    renderSidebar({ overlay: true });

    const rail = screen.getByTestId('chat-sidebar');
    expect(rail.className.split(/\s+/)).toEqual(expect.arrayContaining(['absolute', 'left-0', 'z-30']));
    expect(rail).toHaveAttribute('data-overlay', 'true');
  });

  it('takes its share of the row, and is not positioned over it, when not', () => {
    renderSidebar({ overlay: false });

    const classes = screen.getByTestId('chat-sidebar').className.split(/\s+/);
    expect(classes).not.toContain('absolute');
    expect(classes).toContain('relative');
    expect(classes).toContain(RAIL_WIDTH_CLASS);
    expect(screen.getByTestId('chat-sidebar')).toHaveAttribute('data-overlay', 'false');
  });

  it('keeps the width the dock reserves in both positions', () => {
    renderSidebar({ overlay: true });
    expect(screen.getByTestId('chat-sidebar').className).toContain(RAIL_WIDTH_CLASS);
  });
});

describe('WorkspaceSidebar clone pinning and hybrid MRU sorting', () => {
  it('triggers onTogglePinClone when the pin button is clicked', () => {
    const onTogglePinClone = vi.fn();
    const personas = [makePersonaInfo({ name: 'scout' })];
    renderSidebar({ personas, onTogglePinClone });

    fireEvent.click(screen.getByTestId('clone-menu-scout'));
    const pinBtn = screen.getByTestId('clone-pin-scout');
    expect(pinBtn).toBeInTheDocument();
    fireEvent.click(pinBtn);
    expect(onTogglePinClone).toHaveBeenCalledWith('scout');
  });

  it('renders pinned clones ahead of unpinned clones', () => {
    const personas = [
      makePersonaInfo({ name: 'alpha' }),
      makePersonaInfo({ name: 'beta' }),
      makePersonaInfo({ name: 'gamma' }),
    ];
    // beta is pinned
    renderSidebar({ personas, pinnedCloneIds: ['beta'] });

    const items = screen.getAllByTestId(/^persona-item-/);
    expect(items.map((el) => el.getAttribute('data-testid'))).toEqual([
      'persona-item-beta',
      'persona-item-alpha',
      'persona-item-gamma',
    ]);
  });

  it('orders unpinned clones by recent room activity, then alphabetical for inactive', () => {
    const personas = [
      makePersonaInfo({ name: 'alpha' }),
      makePersonaInfo({ name: 'beta' }),
      makePersonaInfo({ name: 'charlie' }),
    ];
    const rooms: RoomSummary[] = [
      {
        room_id: 'room-1',
        title: 'Older chat',
        agent_ids: ['alpha'],
        human_ids: ['user'],
        message_count: 5,
        updated_at: '2026-09-20T10:00:00Z',
      },
      {
        room_id: 'room-2',
        title: 'Newer chat',
        agent_ids: ['charlie'],
        human_ids: ['user'],
        message_count: 5,
        updated_at: '2026-09-21T10:00:00Z',
      },
    ];
    renderSidebar({ personas, rooms });

    const items = screen.getAllByTestId(/^persona-item-/);
    expect(items.map((el) => el.getAttribute('data-testid'))).toEqual([
      'persona-item-charlie',
      'persona-item-alpha',
      'persona-item-beta',
    ]);
  });
});

describe('WorkspaceSidebar uclone2-style group chats and clone smart accordion sessions', () => {
  const scout = makePersonaInfo({ name: 'scout', role: 'Coach' });

  it('excludes 1:1 sessions from top Group Chats and isolates them strictly under the clone', () => {
    const rooms: RoomSummary[] = [
      {
        room_id: 'r-group',
        title: 'Multi-Agent Room',
        agent_ids: ['scout', 'critic'],
        human_ids: ['user'],
        message_count: 5,
        updated_at: '2026-09-22T10:00:00Z',
      },
      {
        room_id: 'r-solo',
        title: 'Solo Scout Thread',
        agent_ids: ['scout'],
        human_ids: ['user'],
        message_count: 2,
        updated_at: '2026-09-22T09:00:00Z',
      },
    ];

    renderSidebar({ personas: [scout], rooms });

    // Multi-agent room is present in top list
    expect(
      within(screen.getByTestId('conversation-list')).getByTestId('conversation-r-group'),
    ).toHaveTextContent('Multi-Agent Room');
    // Solo 1:1 room is excluded from top list
    expect(
      within(screen.getByTestId('conversation-list')).queryByTestId('conversation-r-solo'),
    ).toBeNull();
    // But Solo 1:1 room is present under clone
    expect(
      within(screen.getByTestId('clone-expansion-scout')).getByTestId('conversation-r-solo'),
    ).toHaveTextContent('Solo Scout Thread');
  });

  it('draws no chevron for a clone with 0 sessions and clicking starts first 1:1 room', () => {
    const onSelectAgent = vi.fn();
    const onNewRoom = vi.fn();
    renderSidebar({ personas: [scout], rooms: [], onSelectAgent, onNewRoom });

    expect(screen.queryByTestId('clone-chevron-scout')).not.toBeInTheDocument();
    fireEvent.click(screen.getByTestId('persona-item-scout'));

    expect(onSelectAgent).toHaveBeenCalledWith('scout');
    expect(onNewRoom).toHaveBeenCalledWith('scout');
  });

  it('draws chevron for a clone with 1 session, clicking row opens that 1:1 room and toggles accordion', () => {
    const onSelectAgent = vi.fn();
    const onSelectRoom = vi.fn();
    const rooms: RoomSummary[] = [
      {
        room_id: 'r-solo',
        title: 'Solo Scout Thread',
        agent_ids: ['scout'],
        human_ids: ['user'],
        message_count: 2,
        updated_at: '2026-09-22T09:00:00Z',
      },
    ];

    renderSidebar({ personas: [scout], rooms, onSelectAgent, onSelectRoom });

    expect(screen.getByTestId('clone-chevron-scout')).toBeInTheDocument();
    // Initially expanded
    expect(screen.getByTestId('clone-expansion-scout')).toBeInTheDocument();

    // Clicking row selects session and toggles accordion (collapses)
    fireEvent.click(screen.getByTestId('persona-item-scout'));

    expect(onSelectAgent).toHaveBeenCalledWith('scout');
    expect(onSelectRoom).toHaveBeenCalledWith('r-solo');
    expect(screen.queryByTestId('clone-expansion-scout')).not.toBeInTheDocument();

    // Clicking chevron expands again
    fireEvent.click(screen.getByTestId('clone-chevron-scout'));
    expect(screen.getByTestId('clone-expansion-scout')).toBeInTheDocument();
  });

  it('draws chevron for 2+ sessions, toggles accordion and opens newest on row click, and allows selecting nested sessions', () => {
    const onSelectAgent = vi.fn();
    const onSelectRoom = vi.fn();
    const rooms: RoomSummary[] = [
      {
        room_id: 'r-older',
        title: 'Older Thread',
        agent_ids: ['scout'],
        human_ids: ['user'],
        message_count: 2,
        updated_at: '2026-09-22T08:00:00Z',
      },
      {
        room_id: 'r-newer',
        title: 'Newer Thread',
        agent_ids: ['scout'],
        human_ids: ['user'],
        message_count: 5,
        updated_at: '2026-09-22T10:00:00Z',
      },
    ];

    renderSidebar({ personas: [scout], rooms, currentRoomId: null, onSelectAgent, onSelectRoom });

    // Chevron present
    const chevron = screen.getByTestId('clone-chevron-scout');
    expect(chevron).toBeInTheDocument();

    // Nested list is initially expanded
    expect(screen.getByTestId('clone-expansion-scout')).toBeInTheDocument();
    expect(screen.getByTestId('conversation-r-newer')).toHaveTextContent('Newer Thread');
    expect(screen.getByTestId('conversation-r-older')).toHaveTextContent('Older Thread');

    // Clicking chevron collapses accordion
    fireEvent.click(chevron);
    expect(screen.queryByTestId('clone-expansion-scout')).not.toBeInTheDocument();

    // Clicking row re-expands accordion and selects newest session
    fireEvent.click(screen.getByTestId('persona-item-scout'));
    expect(onSelectAgent).toHaveBeenCalledWith('scout');
    expect(onSelectRoom).toHaveBeenCalledWith('r-newer');
    expect(screen.getByTestId('clone-expansion-scout')).toBeInTheDocument();

    // Clicking older nested session selects it
    fireEvent.click(screen.getByTestId('conversation-r-older'));
    expect(onSelectRoom).toHaveBeenCalledWith('r-older');
  });

  it('triggers onNewRoom with clone id when clicking clone new thread button', () => {
    const onSelectAgent = vi.fn();
    const onNewRoom = vi.fn();
    renderSidebar({ personas: [scout], onSelectAgent, onNewRoom });

    const newThreadBtn = screen.getByTestId('clone-new-thread-scout');
    expect(newThreadBtn).toBeInTheDocument();
    fireEvent.click(newThreadBtn);

    expect(onSelectAgent).toHaveBeenCalledWith('scout');
    expect(onNewRoom).toHaveBeenCalledWith('scout');
  });

  it('starts group room when clicking + in top Group Chats header', () => {
    const onNewRoom = vi.fn();
    renderSidebar({ personas: [scout], onNewRoom });

    const newGroupBtn = screen.getByTestId('new-conversation-button');
    expect(newGroupBtn).toBeInTheDocument();
    fireEvent.click(newGroupBtn);

    expect(onNewRoom).toHaveBeenCalledWith(undefined, true);
  });

  it('keeps room in Group Chats even when seating 1 agent if included in groupRoomIds', () => {
    const rooms: RoomSummary[] = [
      {
        room_id: 'r-group-solo',
        title: 'Group Room Waiting for More Clones',
        agent_ids: ['scout'],
        human_ids: ['user'],
        message_count: 1,
        updated_at: '2026-09-22T10:00:00Z',
      },
    ];

    renderSidebar({ personas: [scout], rooms, groupRoomIds: ['r-group-solo'] });

    // Present in top Group Chats list
    expect(
      within(screen.getByTestId('conversation-list')).getByTestId('conversation-r-group-solo'),
    ).toHaveTextContent('Group Room Waiting for More Clones');

    // Excluded from clone 1:1 session expansion
    expect(screen.queryByTestId('clone-expansion-scout')).toBeNull();
  });
});


/**
 * The clone row keeps its width for the name: new-thread and `⋯` float over the name's end
 * only while the row is hovered or focused, and pin, edit and profile sit behind `⋯`. Four
 * buttons beside the name left it about 70px of a 240px rail, and a pin hidden by opacity
 * alone still held its 22px.
 */
describe('WorkspaceSidebar clone row actions menu', () => {
  const scout = makePersonaInfo({ name: 'scout', role: 'Research' });
  const critic = makePersonaInfo({ name: 'critic', role: 'Review' });
  const handlers = () => ({
    onInspectAgent: vi.fn(),
    onEditAgent: vi.fn(),
    onTogglePinClone: vi.fn(),
  });

  // Killed by: frontend/src/ui-kit/rail/Rail.tsx :: absolute right-0 inset-y-0 my-auto h-fit
  // Becomes: relative right-0 inset-y-0 my-auto h-fit
  it('takes the row actions out of the flow, so they hold none of the name’s width at rest', () => {
    renderSidebar({ personas: [scout], ...handlers() });

    const actions = screen.getByTestId('clone-actions-scout');
    const classes = actions.className.split(/\s+/);
    expect(classes).toContain('absolute');
    expect(classes).toContain('opacity-0');
    expect(classes).toContain('group-focus-within:opacity-100');
    expect(within(actions).getByTestId('clone-new-thread-scout')).toBeInTheDocument();
    expect(within(actions).getByTestId('clone-menu-scout')).toBeInTheDocument();
  });

  it('draws pin, edit and profile only once the menu is opened', () => {
    renderSidebar({ personas: [scout], ...handlers() });

    expect(screen.queryByTestId('clone-pin-scout')).toBeNull();
    expect(screen.queryByTestId('persona-edit-scout')).toBeNull();
    expect(screen.getByTestId('clone-menu-scout')).toHaveAttribute('aria-expanded', 'false');

    fireEvent.click(screen.getByTestId('clone-menu-scout'));

    const menu = screen.getByRole('menu', { name: 'More actions for scout' });
    expect(within(menu).getAllByRole('menuitem').map((el) => el.textContent)).toEqual([
      'Pin clone to top',
      'Edit settings in dock',
      'View profile in dock',
    ]);
    expect(screen.getByTestId('clone-menu-scout')).toHaveAttribute('aria-expanded', 'true');
  });

  it('opens the profile from the menu, and closes the menu on a choice', () => {
    const h = handlers();
    renderSidebar({ personas: [scout], ...h });

    fireEvent.click(screen.getByTestId('clone-menu-scout'));
    fireEvent.click(screen.getByTestId('clone-profile-item-scout'));

    expect(h.onInspectAgent).toHaveBeenCalledWith('scout');
    expect(screen.queryByRole('menu')).toBeNull();
  });

  it('offers Unpin for a clone that is already pinned', () => {
    renderSidebar({ personas: [scout], pinnedCloneIds: ['scout'], ...handlers() });

    fireEvent.click(screen.getByTestId('clone-menu-scout'));
    expect(screen.getByTestId('clone-pin-scout')).toHaveTextContent('Unpin clone');
  });

  it('opens the same menu on a right-click of the row', () => {
    renderSidebar({ personas: [scout], ...handlers() });

    fireEvent.contextMenu(screen.getByTestId('persona-item-scout'));
    expect(screen.getByRole('menu', { name: 'More actions for scout' })).toBeInTheDocument();
  });

  it('closes the menu on a press outside it and on Escape', () => {
    renderSidebar({ personas: [scout, critic], ...handlers() });

    fireEvent.click(screen.getByTestId('clone-menu-scout'));
    fireEvent.mouseDown(document.body);
    expect(screen.queryByRole('menu')).toBeNull();

    fireEvent.click(screen.getByTestId('clone-menu-scout'));
    fireEvent.keyDown(window, { key: 'Escape' });
    expect(screen.queryByRole('menu')).toBeNull();
  });

  it('keeps one menu open at a time', () => {
    renderSidebar({ personas: [scout, critic], ...handlers() });

    fireEvent.click(screen.getByTestId('clone-menu-scout'));
    fireEvent.click(screen.getByTestId('clone-menu-critic'));
    expect(screen.getAllByRole('menu')).toHaveLength(1);
    expect(screen.getByRole('menu', { name: 'More actions for critic' })).toBeInTheDocument();
  });

  it('draws no `⋯` when the head passes no action to put in it', () => {
    renderSidebar({ personas: [scout], onInspectAgent: undefined });

    expect(screen.queryByTestId('clone-menu-scout')).toBeNull();
    expect(screen.getByTestId('clone-new-thread-scout')).toBeInTheDocument();
  });

  it('heads the pinned clones once and rules them off from the rest', () => {
    renderSidebar({
      personas: [scout, critic, makePersonaInfo({ name: 'alpha' })],
      pinnedCloneIds: ['critic', 'scout'],
    });

    const heading = screen.getByTestId('clones-pinned-heading');
    expect(heading).toHaveTextContent('Pinned');
    expect(screen.getAllByTestId('clones-pinned-heading')).toHaveLength(1);
    expect(screen.getAllByTestId('clones-pinned-separator')).toHaveLength(1);
    // The heading sits over the first pinned row only, the rule between the two tiers.
    const rows = screen
      .getAllByTestId(/^persona-item-|^clones-pinned-/)
      .map((el) => el.getAttribute('data-testid'));
    expect(rows).toEqual([
      'clones-pinned-heading',
      'persona-item-critic',
      'persona-item-scout',
      'clones-pinned-separator',
      'persona-item-alpha',
    ]);
  });

  it('draws no pinned heading or rule when nothing is pinned', () => {
    renderSidebar({ personas: [scout, critic] });

    expect(screen.queryByTestId('clones-pinned-heading')).toBeNull();
    expect(screen.queryByTestId('clones-pinned-separator')).toBeNull();
  });
});
