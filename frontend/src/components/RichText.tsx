import React, { useEffect, useRef, useState } from 'react';
import ReactMarkdown from 'react-markdown';
import remarkGfm from 'remark-gfm';
import remarkMath from 'remark-math';
import rehypeKatex from 'rehype-katex';
import 'katex/dist/katex.min.css';
import { Play, ExternalLink, Copy, Check, FileText } from 'lucide-react';
import { useOpenInDocs } from '../lib/roomDock';
import { cn } from '../lib/utils';

// Single source of truth for rendering user/agent text (chat messages, logs, tool traces).
// Markdown via react-markdown + remark-gfm (so bare URLs autolink, tables, checklists work),
// with YouTube links upgraded to an inline preview card.
//
// Math: KaTeX via remark-math + rehype-katex. Agents are steered to emit inline math as \( … \)
// and display math as $$ … $$, following the convention. remark-math only understands $ / $$
// delimiters, so preprocessLaTeX() normalizes the LaTeX-style \( \) / \[ \] delimiters into
// $ / $$ and escapes stray currency "$" so prices don't get parsed as math. Code spans/blocks
// are left untouched so a literal "$" in code never turns into an equation.

const CODE_SEGMENT_RE = /(```[\s\S]*?```|`[^`\n]*`)/g;

// Convert LaTeX-style math delimiters to the $/$$ that remark-math parses, and neutralize currency.
// Runs only on the non-code segments of the text.
const normalizeMathSegment = (segment: string): string =>
  segment
    // Currency first, on the raw text: a "$" directly before a digit is a price, not a delimiter.
    // (Agents emit \( … \) for math, so a bare "$12" in prose is currency — escape it to render literally.)
    .replace(/\$(?=\d)/g, '\\$')
    // Display math: \[ … \]  ->  $$ … $$ isolated on its own lines, so remark-math parses it as a
    // centered display block (a $$…$$ pair inline within a paragraph is only rendered as inline math).
    .replace(/\\\[([\s\S]*?)\\\]/g, (_m, body) => `\n\n$$\n${body.trim()}\n$$\n\n`)
    // Inline math: \( … \)  ->  $ … $
    .replace(/\\\(([\s\S]*?)\\\)/g, (_m, body) => `$${body}$`);

export const preprocessLaTeX = (text: string): string => {
  if (!text) return '';
  // Fast path: nothing that could be math.
  if (!text.includes('$') && !text.includes('\\(') && !text.includes('\\[')) return text;
  // Split on code spans/blocks so their contents are preserved verbatim.
  return text
    .split(CODE_SEGMENT_RE)
    .map((part) => (part.startsWith('`') ? part : normalizeMathSegment(part)))
    .join('');
};

// Normalize <image>...</image> tags and self-closing image tags emitted by models into markdown.
// Runs only on the non-code segments so code blocks displaying <image> remain verbatim.
const normalizeImageSegment = (segment: string): string =>
  segment
    .replace(/<image(?:\s+[^>]*)?>([\s\S]*?)<\/image>/gi, (_match, body) => {
      const trimmed = body.trim();
      if (!trimmed) return '';
      if (trimmed.startsWith('!') || trimmed.startsWith('[')) {
        return `\n\n${trimmed}\n\n`;
      }
      return `\n\n![Image](${trimmed})\n\n`;
    })
    .replace(/<(?:image|img)\s+[^>]*?(?:src|href)=["']([^"']+)["'][^>]*\/?>/gi, (_match, src) => {
      return `\n\n![Image](${src.trim()})\n\n`;
    })
    .replace(/<\/?image(?:\s+[^>]*)?>/gi, '');

export const preprocessImageTags = (text: string): string => {
  if (!text) return '';
  // Fast path: nothing that could be an image tag.
  if (!text.includes('<image') && !text.includes('</image>') && !text.includes('<img')) return text;
  return text
    .split(CODE_SEGMENT_RE)
    .map((part) => (part.startsWith('`') ? part : normalizeImageSegment(part)))
    .join('');
};

const getTitleText = (children: unknown): string => {
  if (typeof children === 'string') return children;
  if (Array.isArray(children)) return children.map((c) => getTitleText(c)).join('');
  if (children && typeof children === 'object' && 'props' in children) {
    return getTitleText((children as { props?: { children?: unknown } }).props?.children);
  }
  return '';
};

const YOUTUBE_RE = /(?:youtube\.com\/watch\?v=|youtu\.be\/|youtube\.com\/embed\/)([a-zA-Z0-9_-]{11})/;

/**
 * Find the first YouTube video in a block of text.
 */
export const findYouTubeVideo = (text?: string | null): { videoId: string; url: string } | null => {
  if (!text) return null;
  const match = text.match(YOUTUBE_RE);
  if (!match || !match[1]) return null;
  return { videoId: match[1], url: match[0].startsWith('http') ? match[0] : `https://www.youtube.com/watch?v=${match[1]}` };
};

/**
 * Remove YouTube URLs from text that is being shown alongside a video preview.
 */
export const stripYouTubeLinks = (text?: string | null): string => {
  if (!text) return '';
  return text
    .split(/\s+/)
    .filter((token) => !YOUTUBE_RE.test(token))
    .join(' ')
    .trim();
};

/**
 * A generated image, drawn where the reply mentions it (#1208's loss).
 *
 * The image tools write a file and the reply names it, and what a model writes is a *link*:
 * `[View Image](/api/artifacts/content?path=artifacts/images/img_3008971970_sess_r.png)` is
 * the exact text an artist turn produced on 2026-09-21, and it rendered as one line of blue
 * text over a 600KB picture nobody saw. `MediaCard` used to draw that picture from the
 * `generate_image` tool result, and lost its only caller when `PlaygroundTab` was retired
 * (#1208) -- the same removal `MessageBody` records for markdown. A room row cannot bring it
 * back: `RoomMessage` carries no tool results by design, so the link in the prose is the only
 * evidence the head gets.
 *
 * Upgrading it here rather than steering the prompt to emit `![]()`: the surface must not
 * depend on a local model getting markdown syntax right, which is the same reason attribution
 * never comes from the text.
 *
 * Only our own artifact endpoint is upgraded. An arbitrary URL from model output stays a link
 * the reader chooses to follow -- inlining one would fetch a third party on the model's say-so.
 */
const ARTIFACT_IMAGE_SUFFIX_RE = /\.(png|jpe?g|webp|gif|svg)$/i;
const ARTIFACT_CONTENT_RE = /(^|\/)api\/artifacts\/content$/;

export const findArtifactImage = (href?: string | null): { url: string; name: string } | null => {
  if (!href) return null;
  const q = href.indexOf('?');
  if (q < 0) return null;
  if (!ARTIFACT_CONTENT_RE.test(href.slice(0, q))) return null;
  const path = new URLSearchParams(href.slice(q + 1)).get('path');
  if (!path || !ARTIFACT_IMAGE_SUFFIX_RE.test(path)) return null;
  return { url: href, name: path.split('/').pop() || path };
};

/**
 * A repo-relative artifact path in an `![]()` is not a URL a browser can load.
 *
 * `artifacts/images/img_1.png` resolves against the page and 404s. The endpoint that serves
 * it is the one `findArtifactImage` reads, so the same address is built here rather than the
 * broken src being passed through to a broken-image icon. Anything already addressable --
 * absolute, rooted, `data:`, `blob:` -- is left exactly as written.
 */
export const artifactImageSrc = (src?: string | null): string => {
  if (!src) return '';
  if (/^([a-z][a-z0-9+.-]*:|\/\/|\/)/i.test(src)) return src;
  if (!ARTIFACT_IMAGE_SUFFIX_RE.test(src)) return src;
  return `/api/artifacts/content?path=${encodeURIComponent(src)}`;
};

export const ArtifactImageCard: React.FC<{ url: string; name: string; alt?: string }> = ({
  url,
  name,
  alt,
}) => {
  const openInDocs = useOpenInDocs();
  const q = url.indexOf('?');
  const path = q < 0 ? null : new URLSearchParams(url.slice(q + 1)).get('path');
  return (
  <span className="my-3 block max-w-xl overflow-hidden rounded-xl border border-slate-800 bg-slate-950">
    <img
      src={url}
      alt={alt || name}
      data-testid="inline-artifact-image"
      loading="lazy"
      className="block max-h-[480px] w-full object-contain"
    />
    <span className="flex items-center justify-between gap-3 border-t border-slate-800 px-3 py-2 text-[11px] text-slate-400">
      <span className="truncate font-mono">{name}</span>
      <a
        href={url}
        target="_blank"
        rel="noreferrer"
        className="inline-flex shrink-0 items-center gap-1 text-slate-300 hover:text-slate-100"
      >
        Open <ExternalLink size={11} />
      </a>
      {openInDocs && path && (
        <button
          type="button"
          data-testid="artifact-open-docs-btn"
          onClick={() => openInDocs(path)}
          className="inline-flex shrink-0 items-center gap-1 text-slate-300 hover:text-slate-100"
        >
          Open in Docs <FileText size={11} />
        </button>
      )}
    </span>
  </span>
  );
};

interface YoutubePreviewCardProps {
  url: string;
  videoId: string;
  title: string;
}

export const YoutubePreviewCard: React.FC<YoutubePreviewCardProps> = ({ url, videoId, title }) => {
  const [isPlaying, setIsPlaying] = useState(false);
  const displayTitle = title && !title.startsWith('http') ? title : 'YouTube Video';

  if (isPlaying) {
    return (
      <span className="block my-4 max-w-xl overflow-hidden rounded-xl border border-indigo-500/30 bg-slate-950 shadow-2xl shadow-indigo-500/5">
        <span className="block relative aspect-video w-full">
          <iframe
            className="absolute inset-0 h-full w-full"
            src={`https://www.youtube.com/embed/${videoId}?autoplay=1`}
            title={displayTitle}
            allow="accelerometer; autoplay; clipboard-write; encrypted-media; gyroscope; picture-in-picture"
            allowFullScreen
          />
        </span>
        <span className="p-3 bg-slate-900 border-t border-slate-800 flex justify-between items-center gap-4 text-xs font-semibold text-slate-200">
          <span className="truncate">{displayTitle}</span>
          <a
            href={url}
            target="_blank"
            rel="noreferrer"
            className="text-indigo-400 hover:text-indigo-300 font-medium inline-flex items-center gap-1 shrink-0"
          >
            Open <ExternalLink size={12} />
          </a>
        </span>
      </span>
    );
  }

  return (
    <span
      onClick={() => setIsPlaying(true)}
      className="my-4 max-w-xl group overflow-hidden rounded-xl border border-slate-800/80 bg-slate-900 hover:border-indigo-500/30 shadow-lg hover:shadow-indigo-500/5 transition-all duration-300 cursor-pointer flex flex-col sm:flex-row h-auto sm:h-32 text-left block"
    >
      <span className="relative w-full sm:w-48 aspect-video sm:aspect-auto h-auto sm:h-full overflow-hidden bg-slate-950 shrink-0 block">
        <img
          src={`https://img.youtube.com/vi/${videoId}/mqdefault.jpg`}
          alt={displayTitle}
          className="w-full h-full object-cover group-hover:scale-105 transition-transform duration-500"
        />
        <span className="absolute inset-0 flex items-center justify-center bg-slate-950/20 group-hover:bg-slate-950/40 transition-colors duration-300">
          <span className="w-12 h-12 rounded-full bg-indigo-600/90 group-hover:bg-indigo-500 group-hover:scale-110 flex items-center justify-center text-white shadow-lg transition-all duration-300 border border-indigo-400/25">
            <Play size={20} className="fill-white translate-x-0.5" />
          </span>
        </span>
      </span>
      <span className="p-4 flex flex-col justify-between flex-1 min-w-0">
        <span className="space-y-1 block">
          <span className="text-[13px] uppercase tracking-wider font-semibold text-indigo-400 block">YouTube Video</span>
          <span className="text-sm font-semibold text-slate-100 line-clamp-2 group-hover:text-indigo-300 transition-colors block">
            {displayTitle}
          </span>
        </span>
        <span className="flex items-center justify-between text-[14px] text-slate-400 mt-2">
          <span>Click to play inline</span>
          <a
            href={url}
            target="_blank"
            rel="noreferrer"
            onClick={(e) => e.stopPropagation()}
            className="text-indigo-400 hover:text-indigo-300 font-medium inline-flex items-center gap-0.5"
          >
            Open on YouTube <ExternalLink size={10} />
          </a>
        </span>
      </span>
    </span>
  );
};

/**
 * A table wider than its bubble scrolls in its own box (#1010).
 *
 * While it overflows it is a labelled region in the Tab order, so a keyboard reaches it and
 * scrolls it with the arrow keys without relying on the engine to make a scroller focusable.
 * While more of the table lies past its right edge, that edge fades out
 * (`.table-scroll-more` in index.css), which is the visible cue that it scrolls: overlay
 * scrollbars show nothing until it is already being scrolled (#1015).
 *
 * A table that *fits* is none of those things (#1018, PR #1017 review note N4). It used to be
 * one anyway: every table was a Tab stop with nothing to scroll and a landmark named
 * "Scrollable table", so a page with three tables announced three identically named regions
 * and a keyboard walked into each. The observer that already drives the fade decides this
 * too, so overflow is measured once and read twice rather than by a second mechanism.
 */
export const TableScroll: React.FC<{ children: React.ReactNode }> = ({ children }) => {
  const ref = useRef<HTMLDivElement>(null);
  const [more, setMore] = useState(false);
  const [overflows, setOverflows] = useState(false);

  useEffect(() => {
    const el = ref.current;
    if (!el) return;
    const update = () => {
      setOverflows(el.scrollWidth > el.clientWidth + 1);
      setMore(el.scrollLeft + el.clientWidth < el.scrollWidth - 1);
    };
    update();
    el.addEventListener('scroll', update, { passive: true });
    const observer = typeof ResizeObserver === 'undefined' ? null : new ResizeObserver(update);
    observer?.observe(el);
    if (el.firstElementChild) observer?.observe(el.firstElementChild);
    return () => {
      el.removeEventListener('scroll', update);
      observer?.disconnect();
    };
  }, []);

  // A region and a Tab stop only while there is something to scroll.
  const scrollable: React.HTMLAttributes<HTMLDivElement> = overflows
    ? { role: 'region', 'aria-label': 'Scrollable table', tabIndex: 0 }
    : {};

  return (
    <div
      ref={ref}
      data-testid="table-scroll"
      {...scrollable}
      className={cn(
        'overflow-x-auto rounded-sm focus:outline-none focus-visible:ring-1 focus-visible:ring-cyan-500/60',
        more && 'table-scroll-more',
      )}
    >
      {children}
    </div>
  );
};

export const PreBlock: React.FC<React.ComponentPropsWithoutRef<'pre'>> = ({ children, ...props }) => {
  const [copied, setCopied] = useState(false);
  const preRef = React.useRef<HTMLPreElement>(null);

  let lang = 'code';
  let codeString = '';
  if (React.isValidElement(children) && children.props) {
    const childProps = children.props as { className?: string; children?: unknown };
    if (typeof childProps.className === 'string') {
      const match = /language-(\w+)/.exec(childProps.className);
      if (match) {
        lang = match[1];
      }
    }
    if (typeof childProps.children === 'string') {
      codeString = childProps.children;
    }
  }

  const handleCopy = () => {
    const textToCopy = codeString || preRef.current?.innerText || '';
    if (textToCopy) {
      navigator.clipboard.writeText(textToCopy);
      setCopied(true);
      setTimeout(() => setCopied(false), 2000);
    }
  };

  return (
    <div className="my-3 rounded-xl overflow-hidden border border-slate-800 bg-slate-950 font-mono text-xs shadow-lg not-prose">
      <div className="flex items-center justify-between px-3.5 py-1.5 bg-slate-900/90 border-b border-slate-800/80 text-[11px] text-slate-400">
        <span className="font-semibold text-cyan-400 lowercase">{lang}</span>
        <button
          type="button"
          onClick={handleCopy}
          className="flex items-center gap-1 hover:text-white transition-colors text-[10px]"
          title="Copy code"
        >
          {copied ? (
            <>
              <Check className="w-3 h-3 text-emerald-400" />
              <span className="text-emerald-400">Copied!</span>
            </>
          ) : (
            <>
              <Copy className="w-3 h-3" />
              <span>Copy</span>
            </>
          )}
        </button>
      </div>
      <pre ref={preRef} className="p-3.5 overflow-x-auto text-slate-200 leading-relaxed font-mono" {...props}>
        {children}
      </pre>
    </div>
  );
};

export interface RichTextProps {
  children?: string;
  content?: string;
  className?: string;
}

export const RichText: React.FC<RichTextProps> = ({ children, content, className }) => {
  const rawText = children ?? content ?? '';
  const preprocessed = preprocessLaTeX(preprocessImageTags(rawText));
  return (
    <div
      className={cn(
        // `overflow-wrap: anywhere`: a token with no break opportunity (a path, a URL, a hash)
        // wraps inside its bubble instead of running past it (#1007). Not in tables: there it
        // lets one long cell shrink every other column until ordinary words break mid-word, so
        // a table keeps whole words and scrolls sideways in its own wrapper instead (#1010).
        'prose prose-slate prose-invert max-w-none [overflow-wrap:anywhere] [&_table]:[overflow-wrap:normal]',
        'prose-p:leading-relaxed prose-pre:bg-slate-950 prose-pre:text-slate-100',
        className,
      )}
    >
      <ReactMarkdown
        remarkPlugins={[remarkGfm, remarkMath]}
        rehypePlugins={[[rehypeKatex, { throwOnError: false }]]}
        components={{
          pre: PreBlock,
          table: ({ node, ...props }) => (
            <TableScroll>
              <table {...props} />
            </TableScroll>
          ),
          img: ({ node, src, alt, ...props }) => {
            const resolvedSrc = artifactImageSrc(src);
            const artifact = findArtifactImage(resolvedSrc);
            if (artifact) {
              return <ArtifactImageCard url={artifact.url} name={artifact.name} alt={alt} />;
            }
            return <img src={resolvedSrc} alt={alt} {...props} />;
          },
          a: ({ node, href, children: linkChildren, ...props }) => {
            const match = href?.match(YOUTUBE_RE);
            if (match && match[1]) {
              return <YoutubePreviewCard url={href || ''} videoId={match[1]} title={getTitleText(linkChildren)} />;
            }
            const artifact = findArtifactImage(href);
            if (artifact) {
              return <ArtifactImageCard url={artifact.url} name={artifact.name} />;
            }
            return (
              <a href={href} target="_blank" rel="noopener noreferrer" className="text-indigo-400 hover:underline" {...props}>
                {linkChildren}
              </a>
            );
          },
        }}
      >
        {preprocessed}
      </ReactMarkdown>
    </div>
  );
};

export const MarkdownRenderer = RichText;
export default RichText;
