/**
 * The public cloud providers this head knows key formats for.
 *
 * The registry holds no sentences: a provider's cost tip and every key warning are in the
 * catalog (`settings.providers`, `settings.apiKey`), so they follow the language setting.
 * `validateKeyFormat` says what is wrong as a `problem` and Settings words it.
 */
import { fmt } from '../i18n/format';

export type CloudProviderId = 'gemini' | 'anthropic' | 'openai';

export interface ProviderKeyMetadata {
  id: CloudProviderId;
  displayName: string;
  keyConsoleUrl: string;
  keyPrefix: string;
  keyPattern: RegExp;
  placeholder: string;
  /**
   * The company's own name, as a sentence about its model list uses it ("Google's list").
   * `displayName` names the product ("Google Gemini"), which reads wrongly there. A proper
   * noun, so it is the same in every language.
   */
  vendorName: string;
  defaultBaseUrl: string;
}

export const PROVIDER_REGISTRY: Record<CloudProviderId, ProviderKeyMetadata> = {
  gemini: {
    id: 'gemini',
    vendorName: 'Google',
    displayName: 'Google Gemini',
    keyConsoleUrl: 'https://aistudio.google.com/app/apikey',
    keyPrefix: 'AIzaSy',
    keyPattern: /^AIzaSy[A-Za-z0-9_-]{33}$/,
    placeholder: 'AIzaSy...',
    defaultBaseUrl: 'https://generativelanguage.googleapis.com/v1beta',
  },
  anthropic: {
    id: 'anthropic',
    vendorName: 'Anthropic',
    displayName: 'Anthropic Claude',
    keyConsoleUrl: 'https://console.anthropic.com/settings/keys',
    keyPrefix: 'sk-ant-',
    keyPattern: /^sk-ant-[A-Za-z0-9_-]{20,}$/,
    placeholder: 'sk-ant-api03-...',
    defaultBaseUrl: 'https://api.anthropic.com',
  },
  openai: {
    id: 'openai',
    vendorName: 'OpenAI',
    displayName: 'OpenAI',
    keyConsoleUrl: 'https://platform.openai.com/api-keys',
    keyPrefix: 'sk-',
    keyPattern: /^sk-(?:proj-)?[A-Za-z0-9_-]{20,}$/,
    placeholder: 'sk-proj-...',
    defaultBaseUrl: 'https://api.openai.com/v1',
  },
};

/**
 * Whether the provider is a managed public cloud provider requiring an API key.
 */
export function isCloudProvider(providerId: string): providerId is CloudProviderId {
  return providerId in PROVIDER_REGISTRY;
}

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
  return isCloudProvider(providerId) ? PROVIDER_REGISTRY[providerId] : null;
}

/**
 * Detect which provider a key likely belongs to based on known prefixes.
 */
export function detectKeyProvider(key: string): CloudProviderId | null {
  const sanitized = sanitizeApiKey(key);
  if (!sanitized) return null;
  if (sanitized.startsWith('AIzaSy')) return 'gemini';
  if (sanitized.startsWith('sk-ant-')) return 'anthropic';
  if (sanitized.startsWith('sk-proj-') || sanitized.startsWith('sk-')) return 'openai';
  return null;
}

/**
 * What is wrong with a key: it belongs to `detectedProvider`, it lacks the provider's prefix,
 * or it has the prefix but not the length or characters.
 */
export type KeyProblem = 'otherProvider' | 'wrongPrefix' | 'badPattern';

export interface KeyValidationResult {
  valid: boolean;
  problem?: KeyProblem;
  detectedProvider?: CloudProviderId;
}

/** The words for each `KeyProblem`, from the catalog's `settings.apiKey`. */
export interface KeyProblemCopy {
  /** `{detected}` and `{expected}` are providers' display names. */
  otherProvider: string;
  /** `{name}` is the provider's display name, `{prefix}` its key prefix. */
  wrongPrefix: string;
  badPattern: string;
}

/** The sentence Settings shows under a key `validateKeyFormat` found a problem with. */
export function describeKeyProblem(
  result: KeyValidationResult,
  meta: ProviderKeyMetadata,
  copy: KeyProblemCopy,
): string | null {
  switch (result.problem) {
    case 'otherProvider':
      return fmt(copy.otherProvider, {
        detected: result.detectedProvider ? PROVIDER_REGISTRY[result.detectedProvider].displayName : '',
        expected: meta.displayName,
      });
    case 'wrongPrefix':
      return fmt(copy.wrongPrefix, { name: meta.displayName, prefix: meta.keyPrefix });
    case 'badPattern':
      return fmt(copy.badPattern, { name: meta.displayName });
    default:
      return null;
  }
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
    return { valid: false, problem: 'otherProvider', detectedProvider: detected };
  }

  if (!sanitized.startsWith(meta.keyPrefix)) {
    return { valid: false, problem: 'wrongPrefix' };
  }

  if (!meta.keyPattern.test(sanitized)) {
    return { valid: false, problem: 'badPattern' };
  }

  return { valid: true };
}
