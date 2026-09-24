import React, { useState } from 'react';
import { ExternalLink, Globe, ShieldCheck, Copy, Check, ListFilter, Code } from 'lucide-react';
import { Button } from '../ui/Button';

export interface SearchResultItem {
  title: string;
  url: string;
  snippet: string;
  domain?: string;
}

export interface SearchResultCardProps {
  result: SearchResultItem;
  index: number;
}

/**
 * Extract clean hostname/domain from a candidate URL.
 */
export function extractDomain(rawUrl: string): string {
  try {
    const parsed = new URL(rawUrl);
    return parsed.hostname.replace(/^www\./, '');
  } catch {
    // Fallback if URL is relative or malformed
    const match = rawUrl.match(/^(?:https?:\/\/)?([^/]+)/i);
    return match ? match[1].replace(/^www\./, '') : rawUrl;
  }
}

/**
 * Robustly parses raw tool output into structured SearchResultItem list.
 */
export function extractSearchResults(raw: unknown): SearchResultItem[] {
  if (!raw) return [];

  let data: unknown = raw;
  if (typeof raw === 'string') {
    const trimmed = raw.trim();
    if (trimmed.startsWith('[') || trimmed.startsWith('{')) {
      try {
        data = JSON.parse(trimmed);
      } catch {
        // Not valid JSON, continue with string extraction
      }
    }
  }

  // Case 1: Array of objects
  if (Array.isArray(data)) {
    const items: SearchResultItem[] = [];
    for (const item of data) {
      if (typeof item === 'object' && item !== null) {
        const obj = item as Record<string, unknown>;
        const url = String(obj.url || obj.href || obj.link || '');
        const title = String(obj.title || obj.heading || obj.name || url || 'Search Result');
        const snippet = String(obj.snippet || obj.body || obj.description || obj.abstract || '');
        if (url || title) {
          items.push({
            title,
            url,
            snippet,
            domain: url ? extractDomain(url) : undefined,
          });
        }
      }
    }
    if (items.length > 0) return items;
  }

  // Case 2: Object with results array
  if (typeof data === 'object' && data !== null) {
    const obj = data as Record<string, unknown>;
    const candidateList = obj.results || obj.items || obj.data;
    if (Array.isArray(candidateList)) {
      return extractSearchResults(candidateList);
    }
  }

  // Case 3: Markdown link list extraction fallback
  if (typeof raw === 'string') {
    const linkRegex = /\[([^\]]+)\]\((https?:\/\/[^\s)]+)\)(?:\s*[-:]\s*([^\n\r]+))?/g;
    const items: SearchResultItem[] = [];
    let match: RegExpExecArray | null;
    while ((match = linkRegex.exec(raw)) !== null) {
      const title = match[1].trim();
      const url = match[2].trim();
      const snippet = (match[3] || '').trim();
      items.push({
        title,
        url,
        snippet,
        domain: extractDomain(url),
      });
    }
    if (items.length > 0) return items;
  }

  return [];
}

/**
 * Individual rich search result card with verified domain badge, snippet, and copy link action.
 */
export const SearchResultCard: React.FC<SearchResultCardProps> = ({ result, index }) => {
  const [copied, setCopied] = useState(false);
  const domain = result.domain || extractDomain(result.url);
  const isSecure = result.url.startsWith('https://');

  const handleCopy = (e: React.MouseEvent) => {
    e.stopPropagation();
    navigator.clipboard.writeText(result.url);
    setCopied(true);
    setTimeout(() => setCopied(false), 2000);
  };

  return (
    <div
      data-testid={`search-result-card-${index}`}
      className="group relative flex flex-col justify-between p-3 rounded-xl bg-slate-900/80 hover:bg-slate-800/80 border border-slate-800/80 hover:border-cyan-500/50 transition-all duration-200 shadow-sm"
    >
      <div className="space-y-1.5">
        {/* Domain & Protocol Header */}
        <div className="flex items-center justify-between gap-2 text-[11px] font-mono text-slate-400">
          <div className="flex items-center gap-1.5 truncate">
            <Globe className="w-3.5 h-3.5 text-cyan-400 shrink-0" />
            <span className="truncate text-slate-300 font-semibold">{domain}</span>
            {isSecure && (
              <span
                title="Verified HTTPS connection"
                className="inline-flex items-center text-emerald-400"
              >
                <ShieldCheck className="w-3 h-3" />
              </span>
            )}
          </div>

          <div className="flex items-center gap-1 shrink-0">
            <Button
              variant="ghost"
              size="icon"
              onClick={handleCopy}
              title="Copy URL"
              data-testid={`copy-link-btn-${index}`}
              className="p-1 rounded text-slate-400 hover:text-cyan-300 hover:bg-slate-700/60"
            >
              {copied ? (
                <Check className="w-3 h-3 text-emerald-400" />
              ) : (
                <Copy className="w-3 h-3" />
              )}
            </Button>
            <a
              href={result.url}
              target="_blank"
              rel="noopener noreferrer"
              title="Open link in new tab"
              className="p-1 rounded text-slate-400 hover:text-cyan-300 hover:bg-slate-700/60 transition-colors"
            >
              <ExternalLink className="w-3 h-3" />
            </a>
          </div>
        </div>

        {/* Title */}
        <a
          href={result.url}
          target="_blank"
          rel="noopener noreferrer"
          className="block text-xs sm:text-sm font-medium text-cyan-200 group-hover:text-cyan-100 group-hover:underline line-clamp-2 leading-snug"
        >
          {result.title}
        </a>

        {/* Snippet */}
        {result.snippet && (
          <p className="text-[11px] leading-relaxed text-slate-400 line-clamp-3 font-sans">
            {result.snippet}
          </p>
        )}
      </div>
    </div>
  );
};

export interface SearchResultsViewProps {
  results: SearchResultItem[];
  rawOutput?: unknown;
}

/**
 * Grid container for multiple search result cards with raw JSON toggle.
 */
export const SearchResultsView: React.FC<SearchResultsViewProps> = ({ results, rawOutput }) => {
  const [viewMode, setViewMode] = useState<'cards' | 'raw'>('cards');

  if (!results || results.length === 0) {
    return null;
  }

  return (
    <div data-testid="search-results-view" className="space-y-2.5">
      {/* Search Results Toolbar */}
      <div className="flex items-center justify-between text-xs text-slate-400 pb-1 border-b border-slate-800/60">
        <div className="flex items-center gap-1.5 font-medium text-slate-300 font-sans">
          <Globe className="w-3.5 h-3.5 text-cyan-400" />
          <span>Web Research Findings</span>
          <span className="text-[10px] font-mono px-1.5 py-0.2 rounded bg-cyan-950 text-cyan-300 border border-cyan-800/60">
            {results.length} sources
          </span>
        </div>

        {rawOutput !== undefined && (
          <div className="flex items-center gap-1 text-[11px] font-sans">
            <button
              type="button"
              data-testid="toggle-view-cards"
              onClick={() => setViewMode('cards')}
              className={`px-2 py-0.5 rounded flex items-center gap-1 transition-colors ${
                viewMode === 'cards'
                  ? 'bg-slate-800 text-cyan-300 font-semibold'
                  : 'text-slate-400 hover:text-slate-200'
              }`}
            >
              <ListFilter className="w-3 h-3" />
              <span>Cards</span>
            </button>
            <button
              type="button"
              data-testid="toggle-view-raw"
              onClick={() => setViewMode('raw')}
              className={`px-2 py-0.5 rounded flex items-center gap-1 transition-colors ${
                viewMode === 'raw'
                  ? 'bg-slate-800 text-cyan-300 font-semibold'
                  : 'text-slate-400 hover:text-slate-200'
              }`}
            >
              <Code className="w-3 h-3" />
              <span>Raw</span>
            </button>
          </div>
        )}
      </div>

      {viewMode === 'cards' ? (
        <div className="grid grid-cols-1 sm:grid-cols-2 gap-2.5">
          {results.map((item, idx) => (
            <SearchResultCard key={`${item.url}-${idx}`} result={item} index={idx} />
          ))}
        </div>
      ) : (
        <pre
          data-testid="search-raw-output"
          className="p-2.5 rounded-lg bg-slate-900/90 text-slate-200 text-[11px] overflow-x-auto max-h-72 whitespace-pre-wrap border border-slate-800/60 leading-relaxed font-mono"
        >
          {typeof rawOutput === 'string' ? rawOutput : JSON.stringify(rawOutput, null, 2)}
        </pre>
      )}
    </div>
  );
};
