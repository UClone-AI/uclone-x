/**
 * The UI kit: components a head renders by passing data, words and glyphs in (#1063 D, #1158).
 *
 * Portable by construction. Nothing under `ui-kit/` imports from outside it except `react`,
 * so the kit reads no store, calls no API, bakes in no copy and depends on no icon library.
 * `frontend/src/ui-kit.test.ts` fails the build of any file that breaks that.
 */
export type { KitIcon } from './kit';
export { Avatar } from './primitives/Avatar';
export type { AvatarProps, AvatarShape, AvatarSize } from './primitives/Avatar';
export { Button } from './primitives/Button';
export { FillRing } from './primitives/FillRing';
export { StatusDot } from './primitives/StatusDot';
export { Rail, RAIL_WIDTH_CLASS, RAIL_WIDTH_PX } from './rail/Rail';
export type { RailProps } from './rail/Rail';
export { ConversationList } from './rail/ConversationList';
export type { ConversationListProps } from './rail/ConversationList';
export { PersonaDetail } from './rail/PersonaDetail';
export type {
  CloneConversationsCopy,
  CloneLiveness,
  CloneLivenessCopy,
  ConversationListCopy,
  ConversationListIcons,
  ConversationTitleEditorCopy,
  DeleteConversationCopy,
  EmptyCause,
  KitEscapeLayer,
  PersonaDetailCopy,
  RailAgent,
  RailCopy,
  RailIcons,
  RailPersona,
  RailRoom,
  UseKitEscape,
} from './rail/types';
