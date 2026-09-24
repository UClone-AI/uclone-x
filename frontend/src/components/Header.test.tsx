import { describe, it, expect, vi } from 'vitest';
import { render, screen, fireEvent } from '@testing-library/react';
import { Header } from './Header';

describe('Header', () => {
  it('renders the UClone logo and title', () => {
    render(
      <Header
        isConnected={true}
        healthStatus="healthy"
        onRefresh={vi.fn()}
        isRefreshing={false}
      />,
    );

    const logo = screen.getByTestId('header-logo');
    expect(logo).toBeInTheDocument();
    expect(logo).toHaveAttribute('src', '/uclone_logo_circle.png');
    expect(logo).toHaveAttribute('alt', 'UClone Logo');

    const title = screen.getByTestId('header-app-title');
    expect(title).toBeInTheDocument();
    expect(title).toHaveTextContent('UClone-X');
  });

  it('triggers onToggleSidebar when sidebar toggle button is clicked', () => {
    const onToggleSidebar = vi.fn();
    render(
      <Header
        isConnected={true}
        healthStatus="healthy"
        onRefresh={vi.fn()}
        isRefreshing={false}
        isSidebarOpen={true}
        onToggleSidebar={onToggleSidebar}
      />,
    );

    const toggleButton = screen.getByTestId('toggle-sidebar');
    fireEvent.click(toggleButton);
    expect(onToggleSidebar).toHaveBeenCalledTimes(1);
  });
});
