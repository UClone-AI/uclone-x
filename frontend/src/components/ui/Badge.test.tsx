import { describe, it, expect } from 'vitest';
import { render, screen } from '@testing-library/react';
import { Badge } from './Badge';

describe('Badge', () => {
  it('renders its children with a tone class', () => {
    render(<Badge tone="success" data-testid="badge">served</Badge>);
    const badge = screen.getByTestId('badge');
    expect(badge).toHaveTextContent('served');
    expect(badge.className).toContain('emerald');
  });

  it('defaults to the neutral tone', () => {
    render(<Badge data-testid="badge">plain</Badge>);
    expect(screen.getByTestId('badge').className).toContain('slate');
  });
});
