export interface ProviderKeyMetadata {
  id: string;
  displayName: string;
  badgeText?: string;
  keyConsoleUrl: string;
  keyPrefix: string;
  keyPattern: RegExp;
  placeholder: string;
  costTip: string;
  defaultModel: string;
}

export const PROVIDER_REGISTRY: Record<string, ProviderKeyMetadata> = {
  gemini: {
    id: 'gemini',
    displayName: 'Google Gemini',
    badgeText: '무료 티어 제공',
    keyConsoleUrl: 'https://aistudio.google.com/app/apikey',
    keyPrefix: 'AIzaSy',
    keyPattern: /^AIzaSy[A-Za-z0-9_-]{33}$/,
    placeholder: 'AIzaSy...',
    costTip: 'Google AI Studio에서 신용카드 등록 없이 분당 15회 무료(Free Tier) 키를 발급받을 수 있습니다.',
    defaultModel: 'gemini-1.5-pro',
  },
  anthropic: {
    id: 'anthropic',
    displayName: 'Anthropic Claude',
    keyConsoleUrl: 'https://console.anthropic.com/settings/keys',
    keyPrefix: 'sk-ant-',
    keyPattern: /^sk-ant-[A-Za-z0-9_-]{20,}$/,
    placeholder: 'sk-ant-api03-...',
    costTip: 'Anthropic Console의 API Keys 메뉴에서 키를 발급받을 수 있습니다 (Credit 충전 필요).',
    defaultModel: 'claude-3-5-sonnet-20241022',
  },
  openai: {
    id: 'openai',
    displayName: 'OpenAI',
    keyConsoleUrl: 'https://platform.openai.com/api-keys',
    keyPrefix: 'sk-',
    keyPattern: /^sk-(?:proj-)?[A-Za-z0-9_-]{20,}$/,
    placeholder: 'sk-proj-...',
    costTip: 'OpenAI Platform의 API Keys 메뉴에서 비밀 키를 생성할 수 있습니다.',
    defaultModel: 'gpt-4o',
  },
};

/**
 * Clean up an API key by trimming whitespace and stripping surrounding quotes.
 */
export function sanitizeApiKey(key: string): string {
  let cleaned = key.trim();
  if ((cleaned.startsWith('"') && cleaned.endsWith('"')) || (cleaned.startsWith("'") && cleaned.endsWith("'"))) {
    cleaned = cleaned.slice(1, -1).trim();
  }
  return cleaned;
}

/**
 * Retrieve metadata for a public LLM provider, or null if self-hosted / mock.
 */
export function getProviderMeta(providerId: string): ProviderKeyMetadata | null {
  return PROVIDER_REGISTRY[providerId] || null;
}

/**
 * Detect which provider a key likely belongs to based on known prefixes.
 */
export function detectKeyProvider(key: string): string | null {
  const sanitized = sanitizeApiKey(key);
  if (!sanitized) return null;
  if (sanitized.startsWith('AIzaSy')) return 'gemini';
  if (sanitized.startsWith('sk-ant-')) return 'anthropic';
  if (sanitized.startsWith('sk-proj-') || sanitized.startsWith('sk-')) return 'openai';
  return null;
}

export interface KeyValidationResult {
  valid: boolean;
  warning?: string;
  detectedProvider?: string;
}

/**
 * Validate that an API key matches the expected format for a given provider.
 */
export function validateKeyFormat(providerId: string, key: string): KeyValidationResult {
  const sanitized = sanitizeApiKey(key);
  if (!sanitized) {
    return { valid: true };
  }

  const meta = getProviderMeta(providerId);
  if (!meta) {
    return { valid: true };
  }

  const detected = detectKeyProvider(sanitized);
  if (detected && detected !== providerId) {
    const detectedMeta = getProviderMeta(detected);
    const detectedName = detectedMeta ? detectedMeta.displayName : detected;
    return {
      valid: false,
      warning: `${detectedName} 형식의 키가 입력되었습니다. ${meta.displayName} 키가 맞는지 확인해 주세요.`,
      detectedProvider: detected,
    };
  }

  if (!sanitized.startsWith(meta.keyPrefix)) {
    return {
      valid: false,
      warning: `${meta.displayName} 키는 '${meta.keyPrefix}'로 시작해야 합니다.`,
    };
  }

  if (!meta.keyPattern.test(sanitized)) {
    return {
      valid: false,
      warning: `${meta.displayName} 키 형식이 올바르지 않습니다 (길이나 문자를 확인해 주세요).`,
    };
  }

  return { valid: true };
}
