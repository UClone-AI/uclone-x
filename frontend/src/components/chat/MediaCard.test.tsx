import { describe, it, expect, vi } from 'vitest';
import { render, screen, fireEvent } from '@testing-library/react';
import { MediaCard, extractImageMetadata } from './MediaCard';
import { OpenInDocsContext } from '../../lib/roomDock';

describe('extractImageMetadata', () => {
  it('returns null for empty or non-object values', () => {
    expect(extractImageMetadata(null)).toBeNull();
    expect(extractImageMetadata('plain text')).toBeNull();
    expect(extractImageMetadata({})).toBeNull();
  });

  it('correctly extracts structured metadata', () => {
    const raw = {
      status: 'success',
      path: 'artifacts/images/img_123.png',
      relative_url: '/api/artifacts/content?path=artifacts/images/img_123.png',
      prompt: 'a scenic mountain lake',
      seed: 98765,
      engine: 'diffusers-sdxl',
      device: 'Apple M3 Max',
      duration_seconds: 2.4,
      width: 1024,
      height: 1024,
      recipe_hash: 'abc12345',
    };

    const meta = extractImageMetadata(raw);
    expect(meta).not.toBeNull();
    expect(meta?.path).toBe('artifacts/images/img_123.png');
    expect(meta?.seed).toBe(98765);
    expect(meta?.engine).toBe('diffusers-sdxl');
    expect(meta?.width).toBe(1024);
  });
});

describe('MediaCard component', () => {
  const sampleMeta = {
    path: 'artifacts/images/scenic.png',
    relative_url: '/api/artifacts/content?path=artifacts/images/scenic.png',
    prompt: 'a picturesque sunrise over misty hills',
    seed: 424242,
    engine: 'diffusers-sdxl',
    device: 'Apple M3',
    duration_seconds: 2.1,
    width: 1024,
    height: 1024,
    aspect_ratio: '1:1',
    recipe_hash: 'hash9988',
  };

  it('renders image card with engine badge, seed, and preview image', () => {
    render(<MediaCard metadata={sampleMeta} />);

    expect(screen.getByTestId('media-card')).toBeInTheDocument();
    expect(screen.getByTestId('media-engine-badge')).toHaveTextContent('diffusers-sdxl');
    expect(screen.getByTestId('media-seed')).toHaveTextContent('424242');
    expect(screen.getByText(/picturesque sunrise/)).toBeInTheDocument();

    const img = screen.getByTestId('generated-image-preview');
    expect(img).toHaveAttribute('src', '/api/artifacts/content?path=artifacts/images/scenic.png');
  });

  it('invokes onOpenInDock when dock button is clicked', () => {
    const onOpen = vi.fn();
    render(<MediaCard metadata={sampleMeta} onOpenInDock={onOpen} />);

    const dockBtn = screen.getByTestId('media-open-dock-btn');
    fireEvent.click(dockBtn);
    expect(onOpen).toHaveBeenCalledWith('artifacts/images/scenic.png');
  });
});

describe('MediaCard: open in Docs through the dock (#1354)', () => {
  // Killed by: frontend/src/components/chat/MediaCard.tsx :: const openInDock = onOpenInDock ?? fromPage;
  // Becomes: const openInDock = onOpenInDock;
  it('fronts Docs on its file when the page provides the dock and no handler is passed', () => {
    const openInDocs = vi.fn();
    const meta = extractImageMetadata({
      status: 'success',
      path: 'artifacts/images/img_9.png',
      relative_url: '/api/artifacts/content?path=artifacts/images/img_9.png',
    });
    render(
      <OpenInDocsContext.Provider value={openInDocs}>
        <MediaCard metadata={meta!} />
      </OpenInDocsContext.Provider>,
    );
    fireEvent.click(screen.getByTestId('media-open-dock-btn'));
    expect(openInDocs).toHaveBeenCalledWith('artifacts/images/img_9.png');
  });
});
