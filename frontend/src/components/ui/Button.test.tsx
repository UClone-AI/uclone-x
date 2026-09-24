import { describe, it, expect, vi } from 'vitest';
import { render, screen, fireEvent } from '@testing-library/react';
import { Button } from './Button';

describe('Button', () => {
  it('defaults to type="button" so it never submits a form by accident', () => {
    render(<Button data-testid="btn">go</Button>);
    expect(screen.getByTestId('btn')).toHaveAttribute('type', 'button');
  });

  it('fires onClick', () => {
    const onClick = vi.fn();
    render(
      <Button data-testid="btn" onClick={onClick}>
        go
      </Button>,
    );
    fireEvent.click(screen.getByTestId('btn'));
    expect(onClick).toHaveBeenCalledTimes(1);
  });

  it('disables like any other button when disabled', () => {
    render(
      <Button data-testid="btn" disabled>
        go
      </Button>,
    );
    expect(screen.getByTestId('btn')).toBeDisabled();
  });
});
