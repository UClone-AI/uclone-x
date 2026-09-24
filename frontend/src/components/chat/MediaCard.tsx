import React, { useState } from 'react';
import { Image as ImageIcon, ExternalLink, Download, Copy, Check, Eye } from 'lucide-react';
import { Badge } from '../ui/Badge';
import { Button } from '../ui/Button';
import { useOpenInDocs } from '../../lib/roomDock';

export interface GeneratedImageMetadata {
  status?: string;
  path?: string;
  relative_url?: string;
  prompt?: string;
  style?: string;
  aspect_ratio?: string;
  width?: number;
  height?: number;
  seed?: number;
  recipe_hash?: string;
  engine?: string;
  device?: string;
  duration_seconds?: number;
  bytes_written?: number;
}

export function extractImageMetadata(output: unknown): GeneratedImageMetadata | null {
  if (!output || typeof output !== 'object') {
    return null;
  }
  const obj = output as Record<string, unknown>;
  const path = typeof obj.path === 'string' ? obj.path : undefined;
  const relUrl = typeof obj.relative_url === 'string' ? obj.relative_url : undefined;
  if (!path && !relUrl) {
    return null;
  }

  return {
    status: typeof obj.status === 'string' ? obj.status : undefined,
    path,
    relative_url: relUrl || (path ? `/api/artifacts/content?path=${encodeURIComponent(path)}` : undefined),
    prompt: typeof obj.prompt === 'string' ? obj.prompt : undefined,
    style: typeof obj.style === 'string' ? obj.style : undefined,
    aspect_ratio: typeof obj.aspect_ratio === 'string' ? obj.aspect_ratio : undefined,
    width: typeof obj.width === 'number' ? obj.width : undefined,
    height: typeof obj.height === 'number' ? obj.height : undefined,
    seed: typeof obj.seed === 'number' ? obj.seed : undefined,
    recipe_hash: typeof obj.recipe_hash === 'string' ? obj.recipe_hash : undefined,
    engine: typeof obj.engine === 'string' ? obj.engine : undefined,
    device: typeof obj.device === 'string' ? obj.device : undefined,
    duration_seconds: typeof obj.duration_seconds === 'number' ? obj.duration_seconds : undefined,
    bytes_written: typeof obj.bytes_written === 'number' ? obj.bytes_written : undefined,
  };
}

interface MediaCardProps {
  metadata: GeneratedImageMetadata;
  onOpenInDock?: (path: string) => void;
}

export const MediaCard: React.FC<MediaCardProps> = ({ metadata, onOpenInDock }) => {
  // A card drawn inside the room page fronts the dock's Docs surface on its file (#1354).
  const fromPage = useOpenInDocs();
  const openInDock = onOpenInDock ?? fromPage;
  const [copied, setCopied] = useState<boolean>(false);
  const imageUrl = metadata.relative_url || (metadata.path ? `/api/artifacts/content?path=${encodeURIComponent(metadata.path)}` : '');

  const handleCopyPath = () => {
    if (metadata.path) {
      navigator.clipboard.writeText(metadata.path);
      setCopied(true);
      setTimeout(() => setCopied(false), 2000);
    }
  };

  return (
    <div
      data-testid="media-card"
      className="my-3 rounded-2xl bg-slate-950/80 border border-slate-800/80 overflow-hidden shadow-lg font-sans transition-all hover:border-cyan-800/60"
    >
      {/* Header Info */}
      <div className="flex items-center justify-between px-3.5 py-2 bg-slate-900/90 border-b border-slate-800/70 text-xs">
        <div className="flex items-center gap-2">
          <span className="flex items-center gap-1.5 font-semibold text-cyan-300">
            <ImageIcon className="w-3.5 h-3.5 text-cyan-400" />
            <span>Generated Media</span>
          </span>
          {metadata.engine && (
            <Badge
              tone="info"
              data-testid="media-engine-badge"
              className="text-[10px] font-mono text-cyan-300 bg-cyan-950/60 border-cyan-800/50"
            >
              {metadata.engine}
            </Badge>
          )}
          {metadata.device && (
            <span className="hidden sm:inline-block text-[10px] text-slate-400 font-mono">
              ({metadata.device})
            </span>
          )}
        </div>

        <div className="flex items-center gap-2 font-mono text-[11px] text-slate-400">
          {metadata.duration_seconds !== undefined && (
            <span>{metadata.duration_seconds.toFixed(1)}s</span>
          )}
        </div>
      </div>

      {/* Image Preview Canvas */}
      <div className="relative group bg-slate-950 flex items-center justify-center p-2 sm:p-4 overflow-hidden min-h-[220px]">
        {imageUrl ? (
          <img
            src={imageUrl}
            alt={metadata.prompt || 'Generated AI media'}
            data-testid="generated-image-preview"
            className="rounded-xl max-h-[480px] w-auto object-contain shadow-md border border-slate-800/60 transition-transform duration-200 group-hover:scale-[1.01]"
            loading="lazy"
          />
        ) : (
          <div className="flex flex-col items-center justify-center py-10 text-slate-500">
            <ImageIcon className="w-10 h-10 mb-2 opacity-40" />
            <span className="text-xs">No image source URL</span>
          </div>
        )}

        {/* Floating Quick Action Overlay */}
        {imageUrl && (
          <div className="absolute top-4 right-4 flex items-center gap-1.5 opacity-0 group-hover:opacity-100 transition-opacity bg-slate-900/90 p-1.5 rounded-xl border border-slate-700/60 shadow-lg backdrop-blur-sm">
            {openInDock && metadata.path && (
              <Button
                variant="ghost"
                size="icon"
                data-testid="media-open-dock-btn"
                onClick={() => openInDock(metadata.path!)}
                className="text-slate-300 hover:text-cyan-300"
                title="View side-by-side in Artifacts Dock"
              >
                <Eye className="w-3.5 h-3.5" />
              </Button>
            )}
            <a
              href={imageUrl}
              target="_blank"
              rel="noreferrer"
              data-testid="media-open-tab-btn"
              className="p-1.5 text-slate-300 hover:text-cyan-300 rounded-lg hover:bg-slate-800 transition-colors"
              title="Open full size image in new tab"
            >
              <ExternalLink className="w-3.5 h-3.5" />
            </a>
            <a
              href={imageUrl}
              download={metadata.path?.split('/').pop() || 'generated_image.png'}
              data-testid="media-download-btn"
              className="p-1.5 text-slate-300 hover:text-emerald-300 rounded-lg hover:bg-slate-800 transition-colors"
              title="Download image"
            >
              <Download className="w-3.5 h-3.5" />
            </a>
          </div>
        )}
      </div>

      {/* Provenance Metadata Bar */}
      <div className="px-3.5 py-2.5 bg-slate-900/70 border-t border-slate-800/70 space-y-1.5 text-xs text-slate-300">
        {metadata.prompt && (
          <div className="text-slate-200 line-clamp-2 leading-relaxed text-[11px]">
            <span className="text-slate-400 font-medium mr-1.5">Prompt:</span>
            <span>{metadata.prompt}</span>
          </div>
        )}

        <div className="flex flex-wrap items-center justify-between gap-2 pt-1 font-mono text-[10px] text-slate-400 border-t border-slate-800/40">
          <div className="flex items-center gap-3 flex-wrap">
            {metadata.seed !== undefined && (
              <span data-testid="media-seed">
                seed: <span className="text-cyan-300">{metadata.seed}</span>
              </span>
            )}
            {metadata.aspect_ratio && (
              <span>ratio: <span className="text-slate-300">{metadata.aspect_ratio}</span></span>
            )}
            {metadata.width && metadata.height && (
              <span>dim: {metadata.width}×{metadata.height}</span>
            )}
            {metadata.recipe_hash && (
              <span>hash: {metadata.recipe_hash}</span>
            )}
          </div>

          {metadata.path && (
            <div className="flex items-center gap-1.5">
              <span className="truncate max-w-[200px]">{metadata.path}</span>
              <Button
                variant="ghost"
                size="icon"
                onClick={handleCopyPath}
                className="p-1 text-slate-400 hover:text-cyan-300"
                title="Copy artifact path"
              >
                {copied ? <Check className="w-3 h-3 text-emerald-400" /> : <Copy className="w-3 h-3" />}
              </Button>
            </div>
          )}
        </div>
      </div>
    </div>
  );
};
