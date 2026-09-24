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
      '/api/artifacts/content?path=artifacts/images/img_3008971970_sess_r.png',
    );
    // The prose around it survives, and the file stays reachable at full size.
    expect(screen.getByText(/Need adjustments/)).toBeInTheDocument();
    expect(screen.getByRole('link', { name: /Open/ })).toHaveAttribute(
      'href',
      '/api/artifacts/content?path=artifacts/images/img_3008971970_sess_r.png',
    );
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
      url: '/api/artifacts/content?path=a/b.PNG',
      name: 'b.PNG',
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
      '/api/artifacts/content?path=artifacts/images/img_2614005732_sess_r.png',
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
      '/api/artifacts/content?path=artifacts/images/img_2.png',
    );

    const { container: c2 } = render(
      <RichText content="<image src=&quot;/api/artifacts/content?path=artifacts/images/img_3.png&quot; />" />,
    );
    expect(c2.querySelector('img')).toHaveAttribute(
      'src',
      '/api/artifacts/content?path=artifacts/images/img_3.png',
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
  // Killed by: frontend/src/components/RichText.tsx :: {openInDocs && path && (
  // Becomes: {false && openInDocs && path && (
  it('fronts Docs on the file the picture is, from the picture itself', () => {
    const openInDocs = vi.fn();
    render(
      <OpenInDocsContext.Provider value={openInDocs}>
        <RichText content={ARTIST_REPLY} />
      </OpenInDocsContext.Provider>,
    );
    fireEvent.click(screen.getByTestId('artifact-open-docs-btn'));
    expect(openInDocs).toHaveBeenCalledWith('artifacts/images/img_3008971970_sess_r.png');
  });

  it('offers no Docs button where there is no dock to open', () => {
    render(<RichText content={ARTIST_REPLY} />);
    expect(screen.queryByTestId('artifact-open-docs-btn')).toBeNull();
  });
});
