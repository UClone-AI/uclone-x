import { describe, it, expect, vi, afterEach } from 'vitest';
import { render, screen, fireEvent } from '@testing-library/react';
import {
  preprocessLaTeX,
  findYouTubeVideo,
  stripYouTubeLinks,
  findArtifactImage,
  artifactImageSrc,
  RichText,
} from './RichText';
import { OpenInDocsContext } from '../lib/roomDock';

describe('preprocessLaTeX', () => {
  it('converts inline \\( ... \\) to $ ... $', () => {
    expect(preprocessLaTeX('Setup: \\(F = 4D\\) works.')).toBe('Setup: $F = 4D$ works.');
  });

  it('converts display \\[ ... \\] to an isolated $$ block', () => {
    expect(preprocessLaTeX('So:\\[2D = 20\\]done')).toBe('So:\n\n$$\n2D = 20\n$$\n\ndone');
  });

  it('preserves legacy $ ... $ math when it does not start with a digit', () => {
    expect(preprocessLaTeX('Setup: $F = 4D$')).toBe('Setup: $F = 4D$');
  });

  it('escapes currency so prices render literally, not as math', () => {
    expect(preprocessLaTeX('It costs $5 for one and $10 for two.')).toBe(
      'It costs \\$5 for one and \\$10 for two.',
    );
  });

  it('handles currency alongside inline math', () => {
    expect(preprocessLaTeX('Pay $5, and \\(x^2 = 4\\).')).toBe('Pay \\$5, and $x^2 = 4$.');
  });

  it('leaves $ inside code spans untouched', () => {
    expect(preprocessLaTeX('Run `echo $PATH` now.')).toBe('Run `echo $PATH` now.');
  });

  it('leaves $ inside fenced code blocks untouched', () => {
    const input = 'Text $5\n```\nprice = $10\n```\nmore $5';
    expect(preprocessLaTeX(input)).toBe('Text \\$5\n```\nprice = $10\n```\nmore \\$5');
  });

  it('is a no-op for text with no math or currency (fast path)', () => {
    const plain = 'Just a normal sentence with no math.';
    expect(preprocessLaTeX(plain)).toBe(plain);
  });

  it('handles empty or null string gracefully', () => {
    expect(preprocessLaTeX('')).toBe('');
  });
});

describe('YouTube link detection and stripping', () => {
  it('detects standard youtube watch url', () => {
    const result = findYouTubeVideo('Check this video: https://www.youtube.com/watch?v=dQw4w9WgXcQ');
    expect(result).not.toBeNull();
    expect(result?.videoId).toBe('dQw4w9WgXcQ');
  });

  it('detects youtu.be short url', () => {
    const result = findYouTubeVideo('Watch at https://youtu.be/dQw4w9WgXcQ');
    expect(result).not.toBeNull();
    expect(result?.videoId).toBe('dQw4w9WgXcQ');
  });

  it('strips youtube links from text', () => {
    const text = 'Here is the link https://www.youtube.com/watch?v=dQw4w9WgXcQ to check out';
    expect(stripYouTubeLinks(text)).toBe('Here is the link to check out');
  });
});

describe('RichText component', () => {
  it('renders markdown prose with both children and content props', () => {
    const { container: c1 } = render(<RichText>Hello **World**</RichText>);
    expect(c1.querySelector('strong')?.textContent).toBe('World');

    const { container: c2 } = render(<RichText content="Another **test**" />);
    expect(c2.querySelector('strong')?.textContent).toBe('test');
  });

  it('renders inline and display math via KaTeX', () => {
    const { container } = render(
      <RichText>{'The formula is \\(E = mc^2\\) and display:\n\\[\\sum_{i=1}^n i\\]'}</RichText>,
    );
    // KaTeX outputs elements with class "katex"
    const katexElements = container.querySelectorAll('.katex');
    expect(katexElements.length).toBeGreaterThan(0);
  });

  it('renders currency literally without KaTeX error', () => {
    render(<RichText>{'The price is $50 and $99 for two items.'}</RichText>);
    expect(screen.getByText(/The price is \$50 and \$99 for two items\./)).toBeInTheDocument();
  });

  it('renders code blocks with language badge and copy functionality', () => {
    const codeMarkdown = '```python\nprint("hello world")\n```';
    const { container } = render(<RichText>{codeMarkdown}</RichText>);

    expect(screen.getByText('python')).toBeInTheDocument();
    expect(screen.getByText('Copy')).toBeInTheDocument();
    expect(container.querySelector('code')?.textContent).toContain('print("hello world")');

    // Test copy button interaction
    const copyButton = screen.getByTitle('Copy code');
    const writeTextSpy = vi.fn();
    Object.assign(navigator, {
      clipboard: {
        writeText: writeTextSpy,
      },
    });

    fireEvent.click(copyButton);
    expect(writeTextSpy).toHaveBeenCalled();
    expect(screen.getByText('Copied!')).toBeInTheDocument();
  });

  it('renders YouTube preview cards for YouTube links', () => {
    const { container } = render(<RichText>{'https://www.youtube.com/watch?v=dQw4w9WgXcQ'}</RichText>);
    expect(screen.getAllByText('YouTube Video').length).toBeGreaterThan(0);
    expect(screen.getByText('Click to play inline')).toBeInTheDocument();

    // Clicking card opens iframe
    fireEvent.click(screen.getByText('Click to play inline'));
    expect(container.querySelector('iframe')).toBeInTheDocument();
    expect(container.querySelector('iframe')?.src).toContain('dQw4w9WgXcQ');
  });
});

/**
 * The reply an artist turn actually produced on 2026-09-21, verbatim from the room document
 * (`room_74d41ac2f9c9.json`, seq 17). A link, not an `![]()`: the picture existed on disk and
 * the conversation showed a line of blue text (#1208's loss). Substituting the real string is
 * the point -- a hand-written `![](...)` would have passed against the old renderer too.
 */
const ARTIST_REPLY =
  "Here's your scene.\n\nImage generated: [Artistic 16:9]  \n" +
  '🔗 [View Image](/api/artifacts/content?path=artifacts/images/img_3008971970_sess_r.png)  \n\n' +
  'Need adjustments?';

describe('generated images in a reply (#1208)', () => {
  // Killed by: frontend/src/components/RichText.tsx :: const artifact = findArtifactImage(href);
  // Becomes: const artifact = null;
  it('draws the picture a reply links to, not a line of blue text', () => {
    render(<RichText content={ARTIST_REPLY} />);

    const img = screen.getByTestId('inline-artifact-image');
    expect(img).toHaveAttribute(
      'src',
      '/api/artifacts/content?path=artifacts%2Fimages%2Fimg_3008971970_sess_r.png',
    );
    // The prose around it survives, and the file stays reachable: kept, under its own name.
    expect(screen.getByText(/Need adjustments/)).toBeInTheDocument();
    const download = screen.getByTestId('artifact-image-download');
    expect(download).toHaveAttribute(
      'href',
      '/api/artifacts/content?path=artifacts%2Fimages%2Fimg_3008971970_sess_r.png',
    );
    expect(download).toHaveAttribute('download', 'img_3008971970_sess_r.png');
  });

  // Killed by: frontend/src/components/RichText.tsx :: if (!ARTIFACT_CONTENT_RE.test(href.slice(0, q))) return null;
  // Becomes: if (false) return null;
  it('leaves a link to somewhere else a link the reader chooses to follow', () => {
    render(<RichText content={'[a picture](https://example.com/pic.png?path=x.png)'} />);

    expect(screen.queryByTestId('inline-artifact-image')).toBeNull();
    expect(screen.getByRole('link', { name: 'a picture' })).toHaveAttribute(
      'href',
      'https://example.com/pic.png?path=x.png',
    );
  });

  // Killed by: frontend/src/components/RichText.tsx :: if (!path || !ARTIFACT_IMAGE_SUFFIX_RE.test(path)) return null;
  // Becomes: if (!path) return null;
  it('leaves a link to a non-image artifact alone', () => {
    expect(findArtifactImage('/api/artifacts/content?path=docs/plan.md')).toBeNull();
    expect(findArtifactImage('/api/artifacts/content?path=a/b.PNG')).toEqual({
      url: '/api/artifacts/content?path=a%2Fb.PNG',
      name: 'b.PNG',
    });
  });

  // Killed by: frontend/src/components/RichText.tsx :: return { url: `/api/artifacts/content?path=${encodeURIComponent(path)}`, name: path.split('/').pop() || path };
  // Becomes: return { url: href, name: path.split('/').pop() || path };
  it('builds image card url strictly from parsed path ignoring host', () => {
    expect(
      findArtifactImage('https://evil-host.com/api/artifacts/content?path=artifacts/images/pic.png'),
    ).toEqual({
      url: '/api/artifacts/content?path=artifacts%2Fimages%2Fpic.png',
      name: 'pic.png',
    });
  });

  // Killed by: frontend/src/components/RichText.tsx :: return `/api/artifacts/content?path=${encodeURIComponent(src)}`;
  // Becomes: return src;
  it('addresses a bare artifact path in an image so the browser can load it', () => {
    render(<RichText content={'![a knight](artifacts/images/img_1.png)'} />);

    expect(screen.getByAltText('a knight')).toHaveAttribute(
      'src',
      '/api/artifacts/content?path=artifacts%2Fimages%2Fimg_1.png',
    );
  });

  it('leaves an address the browser can already load exactly as written', () => {
    expect(artifactImageSrc('https://example.com/a.png')).toBe('https://example.com/a.png');
    expect(artifactImageSrc('/api/artifacts/content?path=a.png')).toBe(
      '/api/artifacts/content?path=a.png',
    );
    expect(artifactImageSrc('data:image/png;base64,AAAA')).toBe('data:image/png;base64,AAAA');
  });

  it('draws the picture when wrapped in <image>...</image> tags from local models', () => {
    const OLLAMA_IMAGE_REPLY =
      '<image>\n' +
      '![Bikini Girl Art](/api/artifacts/content?path=artifacts/images/img_2614005732_sess_r.png)\n' +
      '</image>  \n' +
      '16:9 artistic-style bikini girl illustration created with Danbooru tags (1girl, bikini, solo, artistic). ' +
      'The image is saved as `artifacts/images/img_2614005732_sess_r.png` for your reference. ' +
      'Would you like to adjust any aspects of the composition or style?';

    render(<RichText content={OLLAMA_IMAGE_REPLY} />);

    const img = screen.getByTestId('inline-artifact-image');
    expect(img).toHaveAttribute(
      'src',
      '/api/artifacts/content?path=artifacts%2Fimages%2Fimg_2614005732_sess_r.png',
    );
    expect(img).toHaveAttribute('alt', 'Bikini Girl Art');

    // <image> and </image> tags must not be rendered as literal text
    expect(screen.queryByText(/<image>/)).toBeNull();
    expect(screen.queryByText(/<\/image>/)).toBeNull();

    // Prose around it and code spans must survive intact
    expect(screen.getByText(/16:9 artistic-style bikini girl illustration/)).toBeInTheDocument();
    expect(screen.getByText('artifacts/images/img_2614005732_sess_r.png')).toBeInTheDocument();
  });

  it('normalizes <image>path</image> and self-closing tags', () => {
    const { container: c1 } = render(
      <RichText content="<image>/api/artifacts/content?path=artifacts/images/img_2.png</image>" />,
    );
    expect(c1.querySelector('img')).toHaveAttribute(
      'src',
      '/api/artifacts/content?path=artifacts%2Fimages%2Fimg_2.png',
    );

    const { container: c2 } = render(
      <RichText content="<image src=&quot;/api/artifacts/content?path=artifacts/images/img_3.png&quot; />" />,
    );
    expect(c2.querySelector('img')).toHaveAttribute(
      'src',
      '/api/artifacts/content?path=artifacts%2Fimages%2Fimg_3.png',
    );
  });

  it('preserves <image> tags inside inline code and fenced code blocks', () => {
    const codeExample =
      'Here is an example: `<image>test</image>`\n\n```html\n<image>inside block</image>\n```';
    render(<RichText content={codeExample} />);

    expect(screen.getByText('<image>test</image>')).toBeInTheDocument();
    expect(screen.getByText('<image>inside block</image>')).toBeInTheDocument();
  });
});

const TABLE = '| column one | path |\n| --- | --- |\n| alpha | /srv/x |\n';

/**
 * jsdom lays nothing out, so every box measures zero and `TableScroll` would read every table
 * as fitting. The two widths are what the component reads, so they are what is substituted
 * (R1): the test is about which of them produces a region, not about the engine's layout.
 */
const measuring = (scrollWidth: number, clientWidth: number): void => {
  vi.spyOn(Element.prototype, 'scrollWidth', 'get').mockReturnValue(scrollWidth);
  vi.spyOn(Element.prototype, 'clientWidth', 'get').mockReturnValue(clientWidth);
};

describe('RichText tables (#1015, #1018)', () => {
  afterEach(() => {
    vi.restoreAllMocks();
  });

  // Killed by: frontend/src/components/RichText.tsx :: el.scrollWidth > el.clientWidth + 1
  // Becomes: false
  it('wraps a table wider than its bubble in a named region the keyboard can reach', () => {
    measuring(400, 200);

    render(<RichText content={TABLE} />);

    const region = screen.getByRole('region', { name: /table/i });
    expect(region).toHaveAttribute('tabindex', '0');
    expect(region).toContainElement(screen.getByRole('table'));
  });

  // Killed by: frontend/src/components/RichText.tsx :: el.scrollWidth > el.clientWidth + 1
  // Becomes: true
  it('makes a table that fits neither a landmark nor a Tab stop', () => {
    measuring(200, 200);

    render(<RichText content={TABLE} />);

    // The table is rendered and wrapped; what it does not carry is a region a screen reader
    // announces and a stop a keyboard walks into, with nothing behind either (#1018).
    const wrapper = screen.getByTestId('table-scroll');
    expect(wrapper).toContainElement(screen.getByRole('table'));
    expect(screen.queryByRole('region')).toBeNull();
    expect(wrapper).not.toHaveAttribute('tabindex');
    expect(wrapper).not.toHaveAttribute('aria-label');
  });
});

describe('opening a generated image in Docs (#1354)', () => {
  // Killed by: frontend/src/components/ImageCard.tsx :: {openInDocs && path && (
  // Becomes: {false && openInDocs && path && (
  it('fronts Docs on the file the picture is, from the picture itself', () => {
    const openInDocs = vi.fn();
    render(
      <OpenInDocsContext.Provider value={openInDocs}>
        <RichText content={ARTIST_REPLY} />
      </OpenInDocsContext.Provider>,
    );
    fireEvent.click(screen.getByTestId('artifact-image-more'));
    fireEvent.click(screen.getByTestId('artifact-open-docs-btn'));
    expect(openInDocs).toHaveBeenCalledWith('artifacts/images/img_3008971970_sess_r.png');
    expect(screen.queryByTestId('artifact-image-menu')).toBeNull();
  });

  it('offers no Docs button where there is no dock to open', () => {
    render(<RichText content={ARTIST_REPLY} />);
    fireEvent.click(screen.getByTestId('artifact-image-more'));
    expect(screen.queryByTestId('artifact-open-docs-btn')).toBeNull();
    // The browser tab is still offered: the menu is never empty.
    expect(screen.getByTestId('artifact-open-tab')).toHaveAttribute('target', '_blank');
  });
});

/**
 * A re-render keeps the elements it already drew.
 *
 * The `table`, `img` and `a` overrides were written inline in the JSX, so every render gave
 * react-markdown new functions -- new component types, to React -- and it replaced every link,
 * table and image in every message. The conversation re-renders on each keystroke, so typing
 * rebuilt all of them each time. The className changes between the two renders because the
 * memo would otherwise skip the second one, and the test could not tell the two apart.
 */
describe('RichText re-render', () => {
  const REPLY = [
    'See [the docs](https://example.com/docs).',
    '',
    '| a | b |',
    '| - | - |',
    '| 1 | 2 |',
  ].join('\n');

  it('keeps its links and tables when drawn again with the same text', () => {
    const { container, rerender } = render(<RichText content={REPLY} className="one" />);
    const link = screen.getByRole('link', { name: 'the docs' });
    const table = container.querySelector('table');
    expect(table).not.toBeNull();

    rerender(<RichText content={REPLY} className="two" />);

    expect(container.firstElementChild).toHaveClass('two');
    expect(link.isConnected).toBe(true);
    expect(table?.isConnected).toBe(true);
  });
});

/**
 * The list the image tool hands a reply for a batch, as it wrote it on 2026-09-27 (its
 * `markdown_gallery`, prompts shortened). A clone passes it on as written.
 */
const BATCH = [1, 2, 3]
  .map((n) => `${n}. ![fiona #${n}](/api/artifacts/content?path=artifacts/images/img_9ed3de_${n}.png)`)
  .join('\n');

describe('a set of drawn pictures', () => {
  // Killed by: frontend/src/components/RichText.tsx :: if (images.length < 2) {
  // Becomes: if (true) {
  it('is one gallery card, not a column of cards', () => {
    render(<RichText content={`Here are three.\n\n${BATCH}\n\nPick one.`} />);

    expect(screen.getAllByTestId('artifact-image-gallery')).toHaveLength(1);
    expect(screen.getAllByTestId('artifact-image-thumb')).toHaveLength(3);
    // One large picture, the first, and the prose on both sides kept.
    expect(screen.getByTestId('inline-artifact-image')).toHaveAttribute('alt', 'fiona #1');
    expect(screen.getByText('Here are three.')).toBeInTheDocument();
    expect(screen.getByText('Pick one.')).toBeInTheDocument();
    expect(screen.queryByRole('list')).toBeNull();
  });

  it('groups pictures written one paragraph each, too', () => {
    const paragraphs = [1, 2]
      .map((n) => `![p${n}](artifacts/images/img_aa_${n}.png)`)
      .join('\n\n');
    render(<RichText content={paragraphs} />);
    expect(screen.getAllByTestId('artifact-image-thumb')).toHaveLength(2);
  });

  // Killed by: frontend/src/components/RichText.tsx :: if (!inner) return null;
  // Becomes: if (!inner) continue;
  it('leaves a list with prose in it a list', () => {
    render(<RichText content={`${BATCH}\n4. and a sentence`} />);
    expect(screen.queryByTestId('artifact-image-gallery')).toBeNull();
    expect(screen.getByRole('list')).toBeInTheDocument();
    expect(screen.getAllByTestId('inline-artifact-image')).toHaveLength(3);
  });

  it('shows the chosen picture large and acts on that one', () => {
    render(<RichText content={BATCH} />);
    fireEvent.click(screen.getAllByTestId('artifact-image-thumb')[1]);

    expect(screen.getByTestId('inline-artifact-image')).toHaveAttribute('alt', 'fiona #2');
    expect(screen.getAllByTestId('artifact-image-thumb')[1]).toHaveAttribute('aria-pressed', 'true');
    const visible = screen.getAllByTestId('artifact-image-download').filter((el) => !el.closest('[hidden]'));
    expect(visible).toHaveLength(1);
    expect(visible[0]).toHaveAttribute('download', 'img_9ed3de_2.png');
  });
});

describe('looking at a picture larger', () => {
  it('opens from the picture and closes on Escape', () => {
    render(<RichText content={ARTIST_REPLY} />);
    expect(screen.queryByTestId('image-lightbox')).toBeNull();

    fireEvent.click(screen.getByTestId('artifact-image-zoom'));
    expect(screen.getByRole('dialog')).toBeInTheDocument();
    expect(screen.getByTestId('image-lightbox-image')).toHaveAttribute(
      'src',
      '/api/artifacts/content?path=artifacts%2Fimages%2Fimg_3008971970_sess_r.png',
    );
    // One picture: nothing to step to.
    expect(screen.queryByTestId('image-lightbox-next')).toBeNull();

    fireEvent.keyDown(document, { key: 'Escape' });
    expect(screen.queryByTestId('image-lightbox')).toBeNull();
  });

  // Killed by: frontend/src/components/ImageCard.tsx :: else if (event.key === 'ArrowRight') onIndex((index + 1) % images.length);
  // Becomes: else if (event.key === 'ArrowRight') onIndex(index);
  it('walks a set with the arrow keys, and the card follows', () => {
    render(<RichText content={BATCH} />);
    fireEvent.click(screen.getByTestId('artifact-image-zoom'));
    expect(screen.getByTestId('image-lightbox-position')).toHaveTextContent('1 / 3');

    fireEvent.keyDown(window, { key: 'ArrowRight' });
    expect(screen.getByTestId('image-lightbox-image')).toHaveAttribute('alt', 'fiona #2');
    fireEvent.keyDown(window, { key: 'ArrowLeft' });
    fireEvent.keyDown(window, { key: 'ArrowLeft' });
    expect(screen.getByTestId('image-lightbox-position')).toHaveTextContent('3 / 3');

    fireEvent.click(screen.getByTestId('image-lightbox-close'));
    expect(screen.getByTestId('inline-artifact-image')).toHaveAttribute('alt', 'fiona #3');
  });

  it('closes on a click beside the picture, not on the picture', () => {
    render(<RichText content={ARTIST_REPLY} />);
    fireEvent.click(screen.getByTestId('artifact-image-zoom'));
    fireEvent.click(screen.getByTestId('image-lightbox-image'));
    expect(screen.getByTestId('image-lightbox')).toBeInTheDocument();
    fireEvent.click(screen.getByTestId('image-lightbox'));
    expect(screen.queryByTestId('image-lightbox')).toBeNull();
  });
});

describe('the larger view, for keyboard and screen-reader users', () => {
  it('names the picture on the control that opens it', () => {
    render(<RichText content={BATCH} />);
    expect(screen.getByRole('button', { name: /fiona #1/ })).toBe(screen.getByTestId('artifact-image-zoom'));
  });

  // Killed by: frontend/src/components/ImageCard.tsx :: if (target && (target.isContentEditable || ['INPUT', 'TEXTAREA', 'SELECT'].includes(target.tagName))) return;
  // Becomes: if (false) return;
  it('leaves an arrow typed into a field to the field', () => {
    render(
      <>
        <textarea data-testid="field" />
        <RichText content={BATCH} />
      </>,
    );
    fireEvent.click(screen.getByTestId('artifact-image-zoom'));
    fireEvent.keyDown(screen.getByTestId('field'), { key: 'ArrowRight' });
    fireEvent.keyDown(window, { key: 'ArrowRight', altKey: true });
    expect(screen.getByTestId('image-lightbox-position')).toHaveTextContent('1 / 3');
  });

  // Killed by: frontend/src/components/ImageCard.tsx :: event.preventDefault();
  // Becomes: return;
  it('keeps Tab inside the viewer', () => {
    render(<RichText content={BATCH} />);
    fireEvent.click(screen.getByTestId('artifact-image-zoom'));
    const close = screen.getByTestId('image-lightbox-close');
    expect(close).toHaveFocus();
    const stops = Array.from(screen.getByRole('dialog').querySelectorAll('a[href], button'));
    // From the last stop, Tab comes round to the first rather than leaving for the page.
    (stops[stops.length - 1] as HTMLElement).focus();
    fireEvent.keyDown(window, { key: 'Tab' });
    expect(stops[0]).toHaveFocus();
  });
});
