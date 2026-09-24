import { describe, expect, it, vi, beforeEach } from 'vitest';
import { render, screen, fireEvent } from '@testing-library/react';
import {
  SearchResultCard,
  SearchResultsView,
  extractDomain,
  extractSearchResults,
} from './SearchResultCard';

describe('SearchResultCard & Utilities', () => {
  describe('extractDomain', () => {
    it('extracts clean hostname without www', () => {
      expect(extractDomain('https://www.soundonsound.com/reviews/gear')).toBe('soundonsound.com');
      expect(extractDomain('http://sweetwater.com/store/item')).toBe('sweetwater.com');
      expect(extractDomain('https://sub.domain.org/path?q=1')).toBe('sub.domain.org');
    });

    it('falls back gracefully on malformed inputs', () => {
      expect(extractDomain('custom-domain-only')).toBe('custom-domain-only');
    });
  });

  describe('extractSearchResults', () => {
    it('parses array of result objects', () => {
      const raw = [
        {
          title: 'Audio Interface Comparison',
          url: 'https://example.com/audio',
          snippet: 'Comparison of popular TRS and XLR interfaces.',
        },
      ];
      const parsed = extractSearchResults(raw);
      expect(parsed).toHaveLength(1);
      expect(parsed[0].title).toBe('Audio Interface Comparison');
      expect(parsed[0].domain).toBe('example.com');
      expect(parsed[0].snippet).toBe('Comparison of popular TRS and XLR interfaces.');
    });

    it('parses JSON string output', () => {
      const jsonStr = JSON.stringify([
        {
          title: 'Studio Monitor Guide',
          url: 'https://studio.com/monitors',
          snippet: 'Best nearfield monitors under $500.',
        },
      ]);
      const parsed = extractSearchResults(jsonStr);
      expect(parsed).toHaveLength(1);
      expect(parsed[0].title).toBe('Studio Monitor Guide');
      expect(parsed[0].domain).toBe('studio.com');
    });

    it('extracts markdown link lists', () => {
      const md = `
1. [Soundcraft EPM8](https://soundcraft.com/epm8) - Compact analog mixer with 8 mono inputs.
2. [Focusrite Scarlett](https://focusrite.com/scarlett) - 2-in 2-out USB-C audio interface.
      `;
      const parsed = extractSearchResults(md);
      expect(parsed).toHaveLength(2);
      expect(parsed[0].title).toBe('Soundcraft EPM8');
      expect(parsed[0].domain).toBe('soundcraft.com');
      expect(parsed[1].title).toBe('Focusrite Scarlett');
      expect(parsed[1].domain).toBe('focusrite.com');
    });
  });

  describe('SearchResultCard Component', () => {
    const mockResult = {
      title: 'Universal Audio Apollo Solo',
      url: 'https://uaudio.com/apollo-solo',
      snippet: 'Desktop 2x4 Thunderbolt 3 audio interface with real-time UAD processing.',
      domain: 'uaudio.com',
    };

    beforeEach(() => {
      vi.clearAllMocks();
      Object.assign(navigator, {
        clipboard: {
          writeText: vi.fn().mockResolvedValue(undefined),
        },
      });
    });

    it('renders title, domain, snippet and security badge', () => {
      render(<SearchResultCard result={mockResult} index={0} />);
      expect(screen.getByTestId('search-result-card-0')).toBeDefined();
      expect(screen.getByText('Universal Audio Apollo Solo')).toBeDefined();
      expect(screen.getByText('uaudio.com')).toBeDefined();
      expect(
        screen.getByText('Desktop 2x4 Thunderbolt 3 audio interface with real-time UAD processing.')
      ).toBeDefined();
    });

    it('copies link on copy button click', () => {
      render(<SearchResultCard result={mockResult} index={0} />);
      const copyBtn = screen.getByTestId('copy-link-btn-0');
      fireEvent.click(copyBtn);
      expect(navigator.clipboard.writeText).toHaveBeenCalledWith('https://uaudio.com/apollo-solo');
    });
  });

  describe('SearchResultsView Component', () => {
    const mockResults = [
      {
        title: 'Result 1',
        url: 'https://site1.com/1',
        snippet: 'Snippet 1',
      },
      {
        title: 'Result 2',
        url: 'https://site2.com/2',
        snippet: 'Snippet 2',
      },
    ];

    it('renders grid of cards and handles raw view toggle', () => {
      render(<SearchResultsView results={mockResults} rawOutput={mockResults} />);
      expect(screen.getByTestId('search-results-view')).toBeDefined();
      expect(screen.getByText('2 sources')).toBeDefined();
      expect(screen.getByTestId('search-result-card-0')).toBeDefined();
      expect(screen.getByTestId('search-result-card-1')).toBeDefined();

      // Switch to raw
      const rawBtn = screen.getByTestId('toggle-view-raw');
      fireEvent.click(rawBtn);
      expect(screen.getByTestId('search-raw-output')).toBeDefined();

      // Switch back to cards
      const cardsBtn = screen.getByTestId('toggle-view-cards');
      fireEvent.click(cardsBtn);
      expect(screen.getByTestId('search-result-card-0')).toBeDefined();
    });
  });
});
