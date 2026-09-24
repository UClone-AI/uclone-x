import { describe, it, expect } from 'vitest';
import { render, screen } from '@testing-library/react';
import { StatusDot } from './StatusDot';

describe('StatusDot', () => {
  it('carries the tone in its class list', () => {
    render(<StatusDot tone="danger" data-testid="dot" />);
    expect(screen.getByTestId('dot').className).toContain('rose');
  });

  it('defaults to neutral', () => {
    render(<StatusDot data-testid="dot" />);
    expect(screen.getByTestId('dot').className).toContain('slate');
  });
});
