import React, { useEffect, useRef, useState } from 'react';
import { createPortal } from 'react-dom';
import { ChevronLeft, ChevronRight, Download, ExternalLink, FileText, MoreHorizontal, X } from 'lucide-react';
import { useOpenInDocs } from '../lib/roomDock';
import { useEscapeOwner } from '../lib/escapePrecedence';
import { UseAsAvatar } from './avatar/UseAsAvatar';
import { usePopover } from './avatar/usePopover';
import { fmt, useCopy } from '../i18n';
import { cn } from '../lib/utils';

/**
 * A picture a clone drew, as a card in the conversation (#1208, #1354, #1300).
 *
 * The file name is not shown. A drawn picture's name is `img_<id>_<n>.png`, which tells a reader
 * nothing, and it took a row of its own. It is still where a reader who wants it looks: the
 * download link's tooltip, and the name the downloaded file is saved under.
 *
 * What a reader does with a drawn picture, most often first: looks at it larger (the picture
 * itself is the control), keeps it (download), gives it to a clone. Opening it in Docs or in a
 * browser tab is rarer, so those two sit behind a menu rather than beside the others.
 */
export interface ArtifactImage {
  url: string;
  name: string;
  alt?: string;
}

/** The workspace path the artifact endpoint serves, or null for an address that names none. */
const pathOf = (url: string): string | null => {
  const q = url.indexOf('?');
  return q < 0 ? null : new URLSearchParams(url.slice(q + 1)).get('path');
};

const ICON_BUTTON =
  'inline-flex shrink-0 items-center justify-center rounded p-1 text-slate-400 hover:bg-slate-800 hover:text-slate-100';

/** The one row of actions under a picture. */
const ImageActions: React.FC<{ image: ArtifactImage }> = ({ image }) => {
  const copy = useCopy().imageCard;
  const openInDocs = useOpenInDocs();
  const menu = usePopover<HTMLSpanElement>();
  const path = pathOf(image.url);
  return (
    <span className="flex flex-wrap items-center justify-end gap-x-2 gap-y-1 border-t border-slate-800 px-2 py-1 text-[11px] text-slate-400">
      {/* Renders nothing outside a clone's message, or for a picture no clone could wear. */}
      {path && <UseAsAvatar path={path} />}
      <a
        href={image.url}
        download={image.name}
        data-testid="artifact-image-download"
        aria-label={copy.download}
        title={fmt(copy.downloadName, { name: image.name })}
        className={ICON_BUTTON}
      >
        <Download size={13} />
      </a>
      <span ref={menu.ref} className="relative inline-flex">
        <button
          type="button"
          data-testid="artifact-image-more"
          aria-label={copy.more}
          title={copy.more}
          aria-haspopup="menu"
          aria-expanded={menu.open}
          onClick={() => menu.setOpen(!menu.open)}
          className={ICON_BUTTON}
        >
          <MoreHorizontal size={13} />
        </button>
        {menu.open && (
          <span
            role="menu"
            data-testid="artifact-image-menu"
            className="absolute bottom-full right-0 z-20 mb-1 flex min-w-44 flex-col rounded-lg border border-slate-700 bg-slate-900 py-1 shadow-lg"
          >
            {openInDocs && path && (
              <button
                type="button"
                role="menuitem"
                data-testid="artifact-open-docs-btn"
                onClick={() => {
                  menu.setOpen(false);
                  openInDocs(path);
                }}
                className="inline-flex items-center gap-2 px-3 py-1.5 text-left text-[11px] text-slate-200 hover:bg-slate-800"
              >
                <FileText size={12} /> {copy.openInDocs}
              </button>
            )}
            <a
              role="menuitem"
              href={image.url}
              target="_blank"
              rel="noreferrer"
              data-testid="artifact-open-tab"
              onClick={() => menu.setOpen(false)}
              className="inline-flex items-center gap-2 px-3 py-1.5 text-[11px] text-slate-200 no-underline hover:bg-slate-800"
            >
              <ExternalLink size={12} /> {copy.openInTab}
            </a>
          </span>
        )}
      </span>
    </span>
  );
};

/** The picture at full size over the page; ←/→ walk the set it came from. */
export const ImageLightbox: React.FC<{
  images: ArtifactImage[];
  index: number;
  onIndex: (index: number) => void;
  onClose: () => void;
}> = ({ images, index, onIndex, onClose }) => {
  const copy = useCopy().imageCard;
  const closeRef = useRef<HTMLButtonElement | null>(null);
  const dialogRef = useRef<HTMLDivElement | null>(null);
  const image = images[index];
  const many = images.length > 1;
  useEscapeOwner('dialog', true, onClose);

  useEffect(() => {
    const opener = document.activeElement instanceof HTMLElement ? document.activeElement : null;
    closeRef.current?.focus();
    return () => opener?.focus();
  }, []);

  useEffect(() => {
    const onKey = (event: KeyboardEvent) => {
      // Tab stays inside the viewer: it is modal, and the page behind it is not reachable.
      if (event.key === 'Tab') {
        const stops = Array.from(dialogRef.current?.querySelectorAll<HTMLElement>('a[href], button') ?? []);
        if (stops.length === 0) return;
        const at = stops.indexOf(document.activeElement as HTMLElement);
        const next = event.shiftKey ? (at <= 0 ? stops.length - 1 : at - 1) : (at + 1) % stops.length;
        event.preventDefault();
        stops[next].focus();
        return;
      }
      // Alt+← is the browser's Back, and an arrow typed into a field moves its caret.
      if (!many || event.altKey || event.ctrlKey || event.metaKey || event.shiftKey) return;
      const target = event.target as HTMLElement | null;
      if (target && (target.isContentEditable || ['INPUT', 'TEXTAREA', 'SELECT'].includes(target.tagName))) return;
      if (event.key === 'ArrowLeft') onIndex((index - 1 + images.length) % images.length);
      else if (event.key === 'ArrowRight') onIndex((index + 1) % images.length);
    };
    window.addEventListener('keydown', onKey);
    return () => window.removeEventListener('keydown', onKey);
  }, [many, index, images.length, onIndex]);

  const step = (by: number) => (event: React.MouseEvent) => {
    event.stopPropagation();
    onIndex((index + by + images.length) % images.length);
  };

  return createPortal(
    <div
      ref={dialogRef}
      role="dialog"
      aria-modal="true"
      aria-label={copy.viewer}
      data-testid="image-lightbox"
      onClick={onClose}
      className="fixed inset-0 z-50 flex items-center justify-center bg-black/85 p-4"
    >
      <div className="absolute right-3 top-3 flex items-center gap-1" onClick={(e) => e.stopPropagation()}>
        {many && (
          <span data-testid="image-lightbox-position" className="mr-2 text-xs text-slate-300">
            {fmt(copy.position, { n: index + 1, count: images.length })}
          </span>
        )}
        <a
          href={image.url}
          download={image.name}
          aria-label={copy.download}
          title={fmt(copy.downloadName, { name: image.name })}
          className="rounded p-2 text-slate-300 hover:bg-white/10 hover:text-white"
        >
          <Download size={18} />
        </a>
        <button
          ref={closeRef}
          type="button"
          data-testid="image-lightbox-close"
          aria-label={copy.close}
          onClick={onClose}
          className="rounded p-2 text-slate-300 hover:bg-white/10 hover:text-white"
        >
          <X size={18} />
        </button>
      </div>
      {many && (
        <button
          type="button"
          data-testid="image-lightbox-prev"
          aria-label={copy.previous}
          onClick={step(-1)}
          className="absolute left-3 top-1/2 -translate-y-1/2 rounded-full p-2 text-slate-300 hover:bg-white/10 hover:text-white"
        >
          <ChevronLeft size={24} />
        </button>
      )}
      <img
        src={image.url}
        alt={image.alt || image.name}
        data-testid="image-lightbox-image"
        onClick={(e) => e.stopPropagation()}
        className="max-h-[90vh] max-w-[90vw] object-contain"
      />
      {many && (
        <button
          type="button"
          data-testid="image-lightbox-next"
          aria-label={copy.next}
          onClick={step(1)}
          className="absolute right-3 top-1/2 -translate-y-1/2 rounded-full p-2 text-slate-300 hover:bg-white/10 hover:text-white"
        >
          <ChevronRight size={24} />
        </button>
      )}
    </div>,
    document.body,
  );
};

/** The picture itself, as the control that opens it larger. */
const ZoomablePicture: React.FC<{ image: ArtifactImage; onZoom: () => void }> = ({ image, onZoom }) => {
  const copy = useCopy().imageCard;
  return (
    <button
      type="button"
      data-testid="artifact-image-zoom"
      aria-label={fmt(copy.zoom, { name: image.alt || image.name })}
      onClick={onZoom}
      className="block w-full cursor-zoom-in"
    >
      <img
        src={image.url}
        alt={image.alt || image.name}
        data-testid="inline-artifact-image"
        loading="lazy"
        className="block max-h-[480px] w-full object-contain"
      />
    </button>
  );
};

// `not-prose`: the message's typography gives every <img> 2em of margin, which opened a band
// above the picture and stretched each thumbnail's box.
const CARD = 'not-prose my-3 block max-w-xl overflow-hidden rounded-xl border border-slate-800 bg-slate-950';

export const ArtifactImageCard: React.FC<ArtifactImage> = (image) => {
  const [zoomed, setZoomed] = useState(false);
  return (
    <span className={CARD}>
      <ZoomablePicture image={image} onZoom={() => setZoomed(true)} />
      <ImageActions image={image} />
      {zoomed && <ImageLightbox images={[image]} index={0} onIndex={() => {}} onClose={() => setZoomed(false)} />}
    </span>
  );
};

/**
 * A set drawn together -- the image tool draws several candidates for one request -- as one
 * card: the chosen picture large, the set as thumbnails, and one row of actions for the chosen
 * one. Five cards in a column made a reader scroll to compare the pictures they were choosing
 * between, and repeated the same actions five times over.
 *
 * Every picture keeps its own action row, hidden unless chosen, so what was done to one (the
 * avatar it became, and its Undo) is still there after looking at another.
 */
export const ArtifactImageGallery: React.FC<{ images: ArtifactImage[] }> = ({ images }) => {
  const copy = useCopy().imageCard;
  const [chosen, setChosen] = useState(0);
  const [zoomed, setZoomed] = useState(false);
  return (
    <span className={CARD} data-testid="artifact-image-gallery" role="group" aria-label={fmt(copy.gallery, { count: images.length })}>
      <ZoomablePicture image={images[chosen]} onZoom={() => setZoomed(true)} />
      <span className="flex gap-1.5 overflow-x-auto border-t border-slate-800 p-2">
        {images.map((image, i) => (
          <button
            key={`${i}:${image.url}`}
            type="button"
            data-testid="artifact-image-thumb"
            aria-label={fmt(copy.pick, { n: i + 1, count: images.length })}
            aria-pressed={i === chosen}
            onClick={() => setChosen(i)}
            className={cn(
              'shrink-0 overflow-hidden rounded-md border-2',
              i === chosen ? 'border-slate-300' : 'border-transparent opacity-60 hover:opacity-100',
            )}
          >
            <img src={image.url} alt="" loading="lazy" className="block h-14 w-14 object-cover" />
          </button>
        ))}
      </span>
      {images.map((image, i) => (
        <span key={`${i}:${image.url}`} hidden={i !== chosen} className={i === chosen ? 'block' : undefined}>
          <ImageActions image={image} />
        </span>
      ))}
      {zoomed && (
        <ImageLightbox images={images} index={chosen} onIndex={setChosen} onClose={() => setZoomed(false)} />
      )}
    </span>
  );
};
