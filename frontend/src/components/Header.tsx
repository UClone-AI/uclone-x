import React from 'react';
import {
  RefreshCw,
  PanelLeftClose,
  PanelLeftOpen,
  PanelRightClose,
  PanelRightOpen,
  Settings,
  FolderOpen,
} from 'lucide-react';
import { Button } from './ui/Button';
import { StatusDot } from './ui/StatusDot';

interface HeaderProps {
  isConnected: boolean;
  healthStatus: string;
  gitCommit?: string;
  onRefresh: () => void;
  isRefreshing: boolean;
  isSidebarOpen?: boolean;
  onToggleSidebar?: () => void;
  isDockOpen?: boolean;
  onToggleDock?: () => void;
  onOpenSettings?: () => void;
  /** Opens the Files screen: every file the clones saved, across conversations (#1554). */
  onOpenFiles?: () => void;
}

/**
 * The application header.
 *
 * It no longer carries a tab bar. The seven tabs it used to host made the conversation one
 * destination among seven, so every internal view was reached by leaving the conversation;
 * six of them now live in the workspace dock and the seventh is the centre column itself.
 */
export const Header: React.FC<HeaderProps> = ({
  isConnected,
  healthStatus,
  gitCommit,
  onRefresh,
  isRefreshing,
  isSidebarOpen = true,
  onToggleSidebar,
  isDockOpen = false,
  onToggleDock,
  onOpenSettings,
  onOpenFiles,
}) => {
  return (
    <header className="border-b border-slate-800/80 bg-slate-900/90 backdrop-blur-md px-3.5 py-2 flex items-center justify-between gap-2 shrink-0 select-none z-30">
      {/* Left section: Sidebar toggle & Logo / Breadcrumb */}
      <div className="flex items-center gap-2">
        {onToggleSidebar && (
          <Button
            variant="ghost"
            size="icon"
            data-testid="toggle-sidebar"
            onClick={onToggleSidebar}
            title={isSidebarOpen ? 'Collapse sidebar' : 'Expand sidebar'}
          >
            {isSidebarOpen ? <PanelLeftClose className="w-4 h-4" /> : <PanelLeftOpen className="w-4 h-4" />}
          </Button>
        )}

        <div className="flex items-center gap-2">
          <img
            src="/uclone_logo_circle.png"
            alt="UClone Logo"
            data-testid="header-logo"
            className="w-6 h-6 object-contain drop-shadow-sm select-none"
          />
          <span
            data-testid="header-app-title"
            title="UClone-X Agentic Runtime Platform"
            className="text-sm font-bold tracking-tight text-slate-100 hidden sm:inline cursor-default select-none"
          >
            UClone-X
          </span>
          <span
            data-testid="header-version-badge"
            title={`Current Release Version: v${__APP_VERSION__}`}
            className="text-[10px] font-mono font-medium bg-slate-800/80 text-slate-400 border border-slate-700/60 px-1 py-0.2 rounded hidden md:inline cursor-help"
          >
            v{__APP_VERSION__}{gitCommit ? ` · ${gitCommit}` : ''}
          </span>
        </div>
      </div>

      {/* System Status Indicators & Dock Toggle */}
      <div className="flex items-center gap-2">
        <div className="flex items-center gap-1.5 px-2 py-1 rounded-lg bg-slate-950/80 border border-slate-800 text-[11px] font-mono hidden md:flex">
          <StatusDot tone={isConnected ? 'success' : 'danger'} />
          <span className="text-slate-300 font-medium">{healthStatus}</span>
        </div>

        <Button
          variant="bordered"
          size="icon"
          onClick={onRefresh}
          disabled={isRefreshing}
          title="Refresh runtime state"
        >
          <RefreshCw className={`w-3.5 h-3.5 ${isRefreshing ? 'animate-spin text-cyan-400' : ''}`} />
        </Button>

        {onOpenFiles && (
          <Button
            variant="bordered"
            data-testid="open-artifact-library"
            onClick={onOpenFiles}
            aria-label="Files"
            title="Files the clones saved, across every conversation"
          >
            <FolderOpen className="w-3.5 h-3.5" />
            <span className="hidden lg:inline">Files</span>
          </Button>
        )}

        {onOpenSettings && (
          <Button
            variant="bordered"
            size="icon"
            data-testid="open-settings"
            onClick={onOpenSettings}
            title="Configure Runtime Settings & Endpoints"
          >
            <Settings className="w-3.5 h-3.5 text-cyan-400" />
          </Button>
        )}

        {onToggleDock && (
          <Button
            variant={isDockOpen ? 'solid' : 'bordered'}
            data-testid="toggle-dock"
            onClick={onToggleDock}
            aria-label={isDockOpen ? 'Collapse workspace dock' : 'Open workspace dock'}
            className={isDockOpen ? 'font-semibold' : 'text-slate-400 hover:text-slate-200'}
            title={
              isDockOpen
                ? 'Collapse workspace dock'
                : 'Open workspace dock (Docs, Knowledge Graph, Activity, Resource)'
            }
          >
            {isDockOpen ? <PanelRightClose className="w-3.5 h-3.5" /> : <PanelRightOpen className="w-3.5 h-3.5" />}
            <span className="hidden lg:inline">Workspace</span>
          </Button>
        )}
      </div>
    </header>
  );
};
