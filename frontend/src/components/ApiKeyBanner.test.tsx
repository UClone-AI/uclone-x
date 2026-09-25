import { describe, it, expect, vi } from 'vitest';
import { render, screen, fireEvent } from '@testing-library/react';
import { ApiKeyBanner } from './ApiKeyBanner';

describe('ApiKeyBanner', () => {
  it('renders nothing when key is already configured', () => {
    const { container } = render(
      <ApiKeyBanner provider="gemini" hasKey={true} onOpenSettings={() => {}} />
    );
    expect(container.firstChild).toBeNull();
  });

  it('renders nothing for self-hosted or mock providers', () => {
    const { container } = render(
      <ApiKeyBanner provider="ollama" hasKey={false} onOpenSettings={() => {}} />
    );
    expect(container.firstChild).toBeNull();
  });

  it('renders banner with link and settings button for missing Gemini key', () => {
    const onOpen = vi.fn();
    render(
      <ApiKeyBanner provider="gemini" hasKey={false} onOpenSettings={onOpen} />
    );

    const banner = screen.getByTestId('api-key-banner');
    expect(banner).toBeTruthy();
    expect(banner.textContent).toContain('Google Gemini');

    const link = screen.getByTestId('api-key-banner-link') as HTMLAnchorElement;
    expect(link.href).toBe('https://aistudio.google.com/app/apikey');

    const settingsBtn = screen.getByTestId('api-key-banner-settings');
    fireEvent.click(settingsBtn);
    expect(onOpen).toHaveBeenCalledTimes(1);
  });
});
