import React from 'react';
import { AlertTriangle, ExternalLink, Settings } from 'lucide-react';
import { getProviderMeta } from '../lib/providerRegistry';

export interface ApiKeyBannerProps {
  provider: string;
  hasKey: boolean;
  onOpenSettings: () => void;
}

export const ApiKeyBanner: React.FC<ApiKeyBannerProps> = ({
  provider,
  hasKey,
  onOpenSettings,
}) => {
  if (hasKey) return null;

  const meta = getProviderMeta(provider);
  if (!meta) return null;

  return (
    <div
      data-testid="api-key-banner"
      role="alert"
      className="bg-amber-950/60 border-b border-amber-800/80 px-4 py-2.5 flex items-center justify-between gap-3 text-xs text-amber-200 animate-in fade-in duration-150"
    >
      <div className="flex items-center gap-2 min-w-0">
        <AlertTriangle className="w-4 h-4 text-amber-400 shrink-0" />
        <span className="truncate">
          <strong className="font-semibold text-amber-300">{meta.displayName}</strong> API Key가 설정되지 않았습니다.
        </span>
      </div>

      <div className="flex items-center gap-2 shrink-0">
        <a
          href={meta.keyConsoleUrl}
          target="_blank"
          rel="noopener noreferrer"
          className="inline-flex items-center gap-1 px-2.5 py-1 rounded-lg bg-amber-900/60 hover:bg-amber-900 border border-amber-700/80 text-amber-100 font-medium transition-colors"
          data-testid="api-key-banner-link"
        >
          <span>키 발급받기</span>
          <ExternalLink className="w-3 h-3" />
        </a>
        <button
          type="button"
          onClick={onOpenSettings}
          className="inline-flex items-center gap-1 px-2.5 py-1 rounded-lg bg-slate-900/80 hover:bg-slate-800 border border-slate-700 text-slate-200 font-medium transition-colors"
          data-testid="api-key-banner-settings"
        >
          <Settings className="w-3 h-3 text-slate-400" />
          <span>설정에서 입력</span>
        </button>
      </div>
    </div>
  );
};
