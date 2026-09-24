import { describe, it, expect, vi } from 'vitest';
import { render, screen, fireEvent } from '@testing-library/react';
import { Rail } from './ui-kit';
import type { KitIcon, RailCopy, RailIcons, RailProps } from './ui-kit';
import { RAIL_COPY, RAIL_ICONS } from './components/layout/WorkspaceSidebar';

/**
 * The kit rail draws the words and glyphs it is given, and no others (#1158).
 *
 * The boundary test (`ui-kit.test.ts`) proves the kit *imports* no copy and no icon set. This
 * proves the other half: that it holds none inline either. It mounts the kit's `Rail` with every
 * string replaced by a marker naming its own key, and every icon by a stub naming its slot, then
 * checks that nothing this head says -- `RAIL_COPY`, the words the head binds -- reaches the
 * screen. A string left written into the kit's markup would show up here as an English word
 * between the markers.
 */

type Marked<T> = { [K in keyof T]: T[K] };

/** `copy` with every string replaced by `«path»` and every function by one naming its args. */
const mark = <T extends object>(copy: T, path: string): Marked<T> =>
  Object.fromEntries(
    Object.entries(copy).map(([key, value]) => {
      const at = path ? `${path}.${key}` : key;
      if (typeof value === 'string') return [key, `«${at}»`];
      if (typeof value === 'function')
        return [key, (...args: unknown[]) => `«${at}(${args.map(String).join(',')})»`];
      return [key, mark(value as object, at)];
    }),
  ) as Marked<T>;

const MARKED_COPY: RailCopy = mark(RAIL_COPY, '');

const stubIcon = (slot: string): KitIcon => {
  const Stub: KitIcon = ({ className }) => <svg data-icon={slot} className={className} />;
  return Stub;
};

const STUB_ICONS = Object.fromEntries(
  Object.keys(RAIL_ICONS).map((slot) => [slot, stubIcon(slot)]),
) as unknown as RailIcons;

/** Every plain string the head binds, nested included. */
const headStrings = (copy: object): string[] =>
  Object.values(copy).flatMap((value) =>
    typeof value === 'string' ? [value] : typeof value === 'object' ? headStrings(value) : [],
  );

const baseProps: RailProps = {
  // This test is about words and glyphs, so Escape goes unclaimed: a head that keeps no
  // registry passes a hook that does nothing, and the kit still renders.
  useEscape: () => {},
  // `AgentState` values, because that is what `/api/agents` sends: `list_agents` puts
  // `ag.state.value` in this field. A fixture inventing `idle`/`busy` exercises a shape the
  // endpoint never produces, and the row's own mapping then goes untested.
  agents: [
    { id: 'ag_1', label: 'ag one', role: 'r1', status: 'IDLE' },
    { id: 'ag_2', label: 'ag two', role: 'r2', status: 'CALLING_TOOL' },
  ],
  personas: [
    {
      name: 'pers_a',
      role: 'r1',
      description: '',
      allowed_tools: [],
      enable_write_tools: true,
      enable_subagent_tools: false,
    },
  ],
  selectedAgent: 'pers_a',
  onSelectAgent: () => {},
  onInspectAgent: () => {},
  onClose: () => {},
  overlay: false,
  unreadableRoomIds: [],
  rooms: [
    {
      room_id: 'room_a',
      title: '',
      agent_ids: ['ag_1', 'ag_2'],
      message_count: 1,
      updated_at: '2026-01-01T00:00:00Z',
    },
  ],
  currentRoomId: 'room_a',
  onSelectRoom: () => {},
  onNewRoom: () => {},
  onRenameRoom: async () => {},
  onDeleteRoom: async () => {},
  modelConfigured: true,
  copy: MARKED_COPY,
  icons: STUB_ICONS,
};

/** Everything a user can read in `root`: its text, and the titles and labels on its elements. */
const readable = (root: HTMLElement): string =>
  [
    root.textContent ?? '',
    ...Array.from(root.querySelectorAll('[title], [aria-label]')).flatMap((el) => [
      el.getAttribute('title') ?? '',
      el.getAttribute('aria-label') ?? '',
    ]),
  ].join('\n');

describe('the kit rail holds no words of its own (#1158)', () => {
  // Killed by: frontend/src/ui-kit/rail/Rail.tsx :: <span>{copy.clones}</span>
  // Becomes: <span>Clones</span>
  it('draws only the copy it is passed, in every region at once', () => {
    render(<Rail {...baseProps} />);
    // Open the delete dialog, so its words are on screen too.
    fireEvent.click(screen.getByTestId('delete-conversation-room_a'));

    const text = readable(document.body);
    // The markers themselves spell key names ('«agents»'), so they are read past, not into.
    const unmarked = text.replace(/«[^»]*»/g, ' ');
    const leaked = headStrings(RAIL_COPY).filter((phrase) => unmarked.includes(phrase));
    expect(leaked).toEqual([]);

    // And the markers did arrive, from each nested copy object.
    expect(text).toContain('«clones»');
    expect(text).toContain('«conversations.heading»');
    expect(text).toContain('«conversations.deleteDialog.body(1,ag_1, ag_2)»');
  });

  // Killed by: frontend/src/ui-kit/rail/Rail.tsx :: {copy.conversations.emptyCause(modelConfigured, clones.length)}
  // Becomes: {'No agents configured.'}
  it('states an empty Clones section with the cause function it is passed', () => {
    render(<Rail {...baseProps} agents={[]} personas={[]} rooms={[]} />);

    expect(screen.getByTestId('agents-empty-cause')).toHaveTextContent(
      '«conversations.emptyCause(true,0)»',
    );
  });

  // Killed by: frontend/src/ui-kit/rail/ConversationList.tsx :: agentIcon={icons.agent}
  // Becomes: agentIcon={icons.heading}
  it('draws each glyph from the slot it is passed', () => {
    render(<Rail {...baseProps} />);

    const row = screen.getByTestId('conversation-room_a');
    expect(row.querySelectorAll('svg[data-icon="agent"]')).toHaveLength(2);
    const drawn = new Set(
      Array.from(document.body.querySelectorAll('svg[data-icon]')).map((el) =>
        el.getAttribute('data-icon'),
      ),
    );
    // Everything but the editor's two, the expanded chevron, and the `⋯` menu's glyphs, which
    // this state does not draw: pin, edit and profile are drawn only once the menu is open.
    // `budget` left the list with the meter it was the glyph for (#1059), and left `RailIcons`
    // with it -- so a head cannot pass a coin here for nothing to draw.
    expect([...drawn].sort()).toEqual(
      ['agent', 'clones', 'collapseRail', 'delete', 'heading', 'more', 'newConversation', 'newThread', 'rename'].sort(),
    );
  });

  it('triggers onInspectAgent when the avatar is clicked', () => {
    const onInspectAgent = vi.fn();
    render(<Rail {...baseProps} onInspectAgent={onInspectAgent} />);

    fireEvent.click(screen.getByTestId('clone-avatar-pers_a'));
    expect(onInspectAgent).toHaveBeenCalledWith('pers_a');
  });

  it('triggers onEditAgent when the edit settings button is clicked', () => {
    const onEditAgent = vi.fn();
    render(<Rail {...baseProps} onEditAgent={onEditAgent} />);

    fireEvent.click(screen.getByTestId('clone-menu-pers_a'));
    fireEvent.click(screen.getByTestId('persona-edit-pers_a'));
    expect(onEditAgent).toHaveBeenCalledWith('pers_a');
  });

  it('renders clone cards matching uclone2 size specifications and 2-line layout', () => {
    render(<Rail {...baseProps} />);

    const item = screen.getByTestId('persona-item-pers_a');
    expect(item).toHaveClass('py-2');
    expect(item).toHaveClass('px-2.5');
    expect(item).toHaveClass('rounded-lg');

    const avatar = screen.getByTestId('clone-avatar-pers_a');
    expect(avatar).toHaveClass('w-10');
    expect(avatar).toHaveClass('h-10');

    const livenessDot = screen.getByTestId('clone-liveness-pers_a');
    expect(livenessDot.parentElement).toHaveClass('flex');
    expect(livenessDot.parentElement).toHaveClass('items-center');
    expect(livenessDot.parentElement).toHaveClass('justify-center');
    expect(livenessDot.parentElement).toHaveClass('rounded-full');

    const label = item.querySelector('.text-sm.font-medium.text-slate-200');
    expect(label).toBeInTheDocument();
    expect(label).toHaveTextContent('pers_a');

    const roleContainer = item.querySelector('.text-\\[12px\\]');
    expect(roleContainer).toBeInTheDocument();
    expect(roleContainer).toHaveClass('font-mono');
    expect(roleContainer).toHaveClass('text-slate-500');

    // Preserves untruncated child span so exact matchers work
    expect(screen.getByText('r1')).toBeInTheDocument();
  });
});
