import React from 'react';
import { cx, type KitIcon } from '../kit';

/** First-and-last initials, or the first two characters of a one-word label. */
const initials = (label: string): string => {
  const words = label.trim().split(/[\s._-]+/).filter(Boolean);
  if (words.length === 0) return '?';
  if (words.length === 1) return words[0].slice(0, 2).toUpperCase();
  return (words[0][0] + words[words.length - 1][0]).toUpperCase();
};

export type AvatarSize = '2xs' | 'xs' | 'sm' | 'base' | 'md' | 'lg' | 'xl';
export type AvatarShape = 'circle' | 'square';

export interface AvatarProps {
  /** The name shown on screen for this participant. */
  label: string;
  kind: 'agent' | 'human';
  /** The glyph an agent is drawn with. Injected: the kit imports no icon set. */
  agentIcon: KitIcon;
  /**
   * Where this participant's picture is, when one is installed.
   *
   * A URL, not a file and not a generator seed: the kit fetches nothing and draws nothing
   * from the name. Left out, or answered with anything but an image, the default below is
   * what shows.
   */
  imageSrc?: string;
  size?: AvatarSize;
  shape?: AvatarShape;
  onClick?: (e: React.MouseEvent<HTMLElement>) => void;
  interactiveLabel?: string;
  /** Added to the avatar's own classes, never merged with them (`cx`): use it for a ring. */
  className?: string;
  'data-testid'?: string;
  id?: string;
  role?: string;
}

const DIMENSIONS: Record<AvatarSize, string> = {
  '2xs': 'w-5 h-5 text-[9px]',
  xs: 'w-8 h-8 text-xs',
  sm: 'w-10 h-10 text-sm',
  base: 'w-8 h-8 text-xs',
  md: 'w-16 h-16 text-lg',
  lg: 'w-24 h-24 text-2xl',
  xl: 'w-36 h-36 text-4xl',
};

const GLYPHS: Record<AvatarSize, string> = {
  '2xs': 'w-3 h-3',
  xs: 'w-4 h-4',
  sm: 'w-5 h-5',
  base: 'w-4 h-4',
  md: 'w-8 h-8',
  lg: 'w-10 h-10',
  xl: 'w-14 h-14',
};

const SHAPES: Record<AvatarShape, string> = {
  circle: 'rounded-full',
  square: 'rounded-2xl',
};

/**
 * A participant's visual identity: a picture when one is installed, otherwise one default.
 *
 * Pictures only. Nothing here draws a face from the name, so a clone with no picture gets
 * the same default as every other clone with no picture -- which is the point. A generated
 * face looks like identity, and reading one as identity is a mistake the reader cannot
 * detect: two installs of the same clone would wear different faces, and a rename would
 * change one. The default is deliberately not a likeness, and the name is rendered beside
 * the avatar everywhere it appears, so nothing is ever named by a picture alone.
 *
 * An agent has no face to abbreviate, and two agents whose ids share a first letter would
 * render the same initials -- the glyph is a kind marker, not a name.
 */
export const Avatar: React.FC<AvatarProps> = ({
  label,
  kind,
  agentIcon: AgentIcon,
  imageSrc,
  size = '2xs',
  shape = 'circle',
  onClick,
  interactiveLabel,
  className,
  ...rest
}) => {
  // A picture that does not load falls back to the default rather than to the browser's
  // broken-image mark. Keyed on the source, so a row that is reused for another participant
  // tries that one's picture instead of inheriting this one's failure.
  const [failedSrc, setFailedSrc] = React.useState<string | null>(null);
  const showImage = imageSrc !== undefined && imageSrc !== failedSrc;

  const content = showImage ? (
    // Decorative: the label is on this element's title and is rendered beside it, so an
    // alt would have a screen reader say the same name twice.
    <img
      src={imageSrc}
      alt=""
      className="w-full h-full object-cover"
      onError={() => {
        setFailedSrc(imageSrc);
      }}
    />
  ) : kind === 'agent' ? (
    <AgentIcon className={GLYPHS[size]} />
  ) : (
    initials(label)
  );

  const sharedClasses = cx(
    'inline-flex items-center justify-center shrink-0 overflow-hidden border font-semibold select-none',
    SHAPES[shape],
    kind === 'agent'
      ? 'bg-cyan-950/70 border-cyan-800/70 text-cyan-200'
      : 'bg-slate-800 border-slate-700 text-slate-300',
    DIMENSIONS[size],
    className,
  );

  if (onClick) {
    return (
      <button
        type="button"
        aria-label={interactiveLabel || label}
        title={interactiveLabel || label}
        onClick={onClick}
        className={sharedClasses}
        {...rest}
      >
        {content}
      </button>
    );
  }

  return (
    <span
      title={label}
      className={sharedClasses}
      {...rest}
    >
      {content}
    </span>
  );
};
