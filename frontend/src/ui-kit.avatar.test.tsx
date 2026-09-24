import { describe, it, expect, vi } from 'vitest';
import { render, screen, fireEvent } from '@testing-library/react';
import { Avatar } from './ui-kit';
import type { KitIcon } from './ui-kit';

/**
 * A clone's avatar is a picture or it is the default, and never a face drawn from its name.
 *
 * The owner's decision on 2026-09-20: images only, one default when there is no image, and no
 * generator service. The reason is that a generated face reads as identity -- a reader who has
 * seen `scout` twice expects the same face both times -- while it is actually a hash of a
 * string. Two installs would disagree, a rename would change it, and the service's next
 * version would change all of them at once. The tests below pin the three states the component
 * can be in; the scan at the bottom pins that no generator comes back in through a URL.
 */

const Glyph: KitIcon = ({ className }) => <svg data-testid="default-avatar" className={className} />;

describe('Avatar (#1300)', () => {
  // Killed by: frontend/src/ui-kit/primitives/Avatar.tsx :: src={imageSrc}
  // Becomes: src=""
  it('draws the picture it is pointed at', () => {
    render(
      <Avatar
        label="surveyor"
        kind="agent"
        agentIcon={Glyph}
        imageSrc="/api/personas/surveyor/avatar"
        data-testid="avatar"
      />,
    );

    const picture = screen.getByTestId('avatar').querySelector('img');
    expect(picture).toHaveAttribute('src', '/api/personas/surveyor/avatar');
    // Decorative: the name is on the wrapper's title and is rendered beside it, so a screen
    // reader that read an alt too would say the clone's name twice.
    expect(picture).toHaveAttribute('alt', '');
    expect(screen.queryByTestId('default-avatar')).not.toBeInTheDocument();
  });

  // Killed by: frontend/src/ui-kit/primitives/Avatar.tsx ::             setFailedSrc(imageSrc);
  // Becomes:             setFailedSrc(null);
  it('falls back to the default when that picture does not load', () => {
    // The route answers 404 for most clones, and a head that asked for the picture anyway
    // would otherwise leave the browser's broken-image mark on the row -- a wrong-looking
    // screen that says nothing about why.
    render(
      <Avatar
        label="surveyor"
        kind="agent"
        agentIcon={Glyph}
        imageSrc="/api/personas/surveyor/avatar"
        data-testid="avatar"
      />,
    );

    fireEvent.error(screen.getByTestId('avatar').querySelector('img') as HTMLImageElement);

    expect(screen.getByTestId('avatar').querySelector('img')).toBeNull();
    expect(screen.getByTestId('default-avatar')).toBeInTheDocument();
  });

  /*
   * No mutation declared. Dropping the `imageSrc !== undefined` guard does turn this case
   * red -- an `<img>` with no source appears -- but it also turns `the kit rail holds no
   * words of its own > draws each glyph from the slot it is passed` red, because that suite
   * passes the rail no picture source at all and every row would then draw one of these
   * instead of its glyph. One defect, two observers: a declaration naming this case would
   * be a claim about the neighbour that is not true.
   */
  it('asks for no picture at all when it is pointed at none, and draws one default for everyone', () => {
    // Both halves matter. An `<img>` with no source is a request for the page itself in some
    // browsers, and the second assertion is the whole of the owner's decision: two clones with
    // no picture look alike, so nothing on this screen is identified by an image.
    render(<Avatar label="surveyor" kind="agent" agentIcon={Glyph} data-testid="a" />);
    render(<Avatar label="courier" kind="agent" agentIcon={Glyph} data-testid="b" />);

    const surveyor = screen.getByTestId('a');
    const courier = screen.getByTestId('b');
    expect(surveyor.querySelector('img')).toBeNull();
    expect(courier.querySelector('img')).toBeNull();
    expect(surveyor.innerHTML).toEqual(courier.innerHTML);
  });

  it('applies the standard size scale and glyph dimensions', () => {
    const cases: Array<{
      size: '2xs' | 'xs' | 'sm' | 'base' | 'md' | 'lg' | 'xl';
      expectedDimensions: string;
      expectedGlyph: string;
    }> = [
      { size: '2xs', expectedDimensions: 'w-5 h-5 text-[9px]', expectedGlyph: 'w-3 h-3' },
      { size: 'xs', expectedDimensions: 'w-8 h-8 text-xs', expectedGlyph: 'w-4 h-4' },
      { size: 'sm', expectedDimensions: 'w-10 h-10 text-sm', expectedGlyph: 'w-5 h-5' },
      { size: 'base', expectedDimensions: 'w-8 h-8 text-xs', expectedGlyph: 'w-4 h-4' },
      { size: 'md', expectedDimensions: 'w-16 h-16 text-lg', expectedGlyph: 'w-8 h-8' },
      { size: 'lg', expectedDimensions: 'w-24 h-24 text-2xl', expectedGlyph: 'w-10 h-10' },
      { size: 'xl', expectedDimensions: 'w-36 h-36 text-4xl', expectedGlyph: 'w-14 h-14' },
    ];

    for (const { size, expectedDimensions, expectedGlyph } of cases) {
      const { unmount } = render(
        <Avatar
          label="bot"
          kind="agent"
          agentIcon={Glyph}
          size={size}
          data-testid={`avatar-${size}`}
        />,
      );
      const el = screen.getByTestId(`avatar-${size}`);
      for (const cls of expectedDimensions.split(' ')) {
        expect(el).toHaveClass(cls);
      }
      const glyph = el.querySelector('[data-testid="default-avatar"]');
      expect(glyph).toBeInTheDocument();
      for (const cls of expectedGlyph.split(' ')) {
        expect(glyph).toHaveClass(cls);
      }
      unmount();
    }
  });

  it('defaults to 2xs size when size is omitted', () => {
    render(<Avatar label="bot" kind="agent" agentIcon={Glyph} data-testid="default-size" />);
    const el = screen.getByTestId('default-size');
    expect(el).toHaveClass('w-5');
    expect(el).toHaveClass('h-5');
    expect(el).toHaveClass('text-[9px]');
    const glyph = el.querySelector('[data-testid="default-avatar"]');
    expect(glyph).toHaveClass('w-3');
    expect(glyph).toHaveClass('h-3');
  });

  it('supports circle and square shapes', () => {
    // Default is circle
    const { rerender } = render(
      <Avatar label="bot" kind="agent" agentIcon={Glyph} data-testid="shape-avatar" />,
    );
    expect(screen.getByTestId('shape-avatar')).toHaveClass('rounded-full');
    expect(screen.getByTestId('shape-avatar')).not.toHaveClass('rounded-2xl');

    // Explicit circle
    rerender(
      <Avatar label="bot" kind="agent" agentIcon={Glyph} shape="circle" data-testid="shape-avatar" />,
    );
    expect(screen.getByTestId('shape-avatar')).toHaveClass('rounded-full');

    // Explicit square
    rerender(
      <Avatar label="bot" kind="agent" agentIcon={Glyph} shape="square" data-testid="shape-avatar" />,
    );
    expect(screen.getByTestId('shape-avatar')).toHaveClass('rounded-2xl');
    expect(screen.getByTestId('shape-avatar')).not.toHaveClass('rounded-full');
  });

  it('renders as an accessible button when onClick is provided', () => {
    const handleClick = vi.fn();
    render(
      <Avatar
        label="bot"
        kind="agent"
        agentIcon={Glyph}
        onClick={handleClick}
        interactiveLabel="Visit Bot Profile"
        data-testid="interactive-avatar"
      />,
    );

    const btn = screen.getByRole('button', { name: 'Visit Bot Profile' });
    expect(btn).toBeInTheDocument();
    expect(btn).toHaveAttribute('type', 'button');
    expect(btn).toHaveAttribute('title', 'Visit Bot Profile');

    fireEvent.click(btn);
    expect(handleClick).toHaveBeenCalledTimes(1);
  });

  it('falls back to label for aria-label and title when interactiveLabel is not provided', () => {
    const handleClick = vi.fn();
    render(
      <Avatar
        label="bot"
        kind="agent"
        agentIcon={Glyph}
        onClick={handleClick}
        data-testid="interactive-avatar"
      />,
    );

    const btn = screen.getByRole('button', { name: 'bot' });
    expect(btn).toBeInTheDocument();
    expect(btn).toHaveAttribute('title', 'bot');
  });

  it('renders as non-button span when onClick is not provided', () => {
    render(<Avatar label="bot" kind="agent" agentIcon={Glyph} data-testid="static-avatar" />);
    const el = screen.getByTestId('static-avatar');
    expect(el.tagName.toLowerCase()).toBe('span');
    expect(el).toHaveAttribute('title', 'bot');
    expect(screen.queryByRole('button')).not.toBeInTheDocument();
  });

  it('renders initials for humans and falls back to initials when human image fails', () => {
    const { rerender } = render(
      <Avatar label="Jane Doe" kind="human" agentIcon={Glyph} data-testid="human-avatar" />,
    );
    expect(screen.getByTestId('human-avatar')).toHaveTextContent('JD');

    // Single-word label uses first two characters
    rerender(
      <Avatar label="Alice" kind="human" agentIcon={Glyph} data-testid="human-avatar" />,
    );
    expect(screen.getByTestId('human-avatar')).toHaveTextContent('AL');

    // Broken image fallback to initials
    rerender(
      <Avatar
        label="Jane Doe"
        kind="human"
        agentIcon={Glyph}
        imageSrc="/api/users/jane/avatar"
        data-testid="human-avatar"
      />,
    );
    const img = screen.getByTestId('human-avatar').querySelector('img');
    expect(img).toBeInTheDocument();
    fireEvent.error(img as HTMLImageElement);
    expect(screen.getByTestId('human-avatar').querySelector('img')).toBeNull();
    expect(screen.getByTestId('human-avatar')).toHaveTextContent('JD');
  });
});

/** Every script the head and the kit ship, as text. `?raw` needs no `@types/node`. */
const SOURCES: Record<string, string> = import.meta.glob('./**/*.{ts,tsx}', {
  query: '?raw',
  import: 'default',
  eager: true,
});

/**
 * The spellings of an avatar drawn from a name rather than read from a file.
 *
 * All four are hosted generators that take the name in the URL. They are named individually
 * rather than matched by shape because the shape -- a remote URL with the clone's name in it
 * -- is also what a legitimate CDN would look like, and a check that cannot tell them apart
 * would be turned off the first time it was wrong.
 */
const GENERATORS = [
  { pattern: /dicebear/i, why: 'a DiceBear avatar generated from a name' },
  { pattern: /ui-avatars\.com/i, why: 'a ui-avatars.com avatar generated from a name' },
  { pattern: /boringavatars/i, why: 'a Boring Avatars avatar generated from a name' },
  { pattern: /gravatar\.com/i, why: 'a Gravatar, which is a hash of an email address' },
];

const generatorOffences = (sources: Record<string, string>): string[] =>
  Object.entries(sources).flatMap(([path, text]) =>
    GENERATORS.filter(({ pattern }) => pattern.test(text)).map(({ why }) => `${path}: ${why}`),
  );

describe('avatars are files this workspace holds, never a service (#1300)', () => {
  /*
   * No mutation declared, and the reason is the test's own shape: it asserts an absence, so
   * the only edit that turns it red is the edit it exists to forbid -- adding a generator's
   * URL to the tree. There is no line to break, because the line was never written. The
   * neighbour below is what proves the check is not vacuous, and it does carry one.
   */
  it('holds: nothing under src/ composes an avatar URL from a clone name', () => {
    // A glob that matched nothing would pass having read nothing.
    expect(Object.keys(SOURCES).length).toBeGreaterThanOrEqual(50);
    expect(generatorOffences(SOURCES)).toEqual([]);
  });

  /*
   * Nor here, and for a second reason: the matcher this case exercises lives in this file,
   * so a declaration naming it would quote the needle a second time and match twice. What
   * stands in for one is the assertion itself -- exact equality on a planted pair, one of
   * which is the real `personaAvatar.ts` -- which cannot pass while the check is blind.
   */
  it('would notice the spelling the sibling head uses', () => {
    const planted = {
      './lib/plantedAvatar.ts':
        "export const url = (n: string) => `https://api.dicebear.com/7.x/bottts/svg?seed=${n}`;",
      './lib/personaAvatar.ts':
        'export const personaAvatarUrl = (name: string): string =>\n' +
        '  `/api/personas/${encodeURIComponent(name)}/avatar`;',
    };

    expect(generatorOffences(planted)).toEqual([
      './lib/plantedAvatar.ts: a DiceBear avatar generated from a name',
    ]);
  });
});
