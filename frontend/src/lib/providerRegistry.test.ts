import { describe, it, expect } from 'vitest';
import {
  sanitizeApiKey,
  getProviderMeta,
  detectKeyProvider,
  validateKeyFormat,
  isCloudProvider,
  getCuratedModels,
} from './providerRegistry';

describe('providerRegistry', () => {
  describe('sanitizeApiKey', () => {
    it('trims whitespace', () => {
      expect(sanitizeApiKey('  AIzaSy12345  ')).toBe('AIzaSy12345');
    });

    it('strips double and single quotes', () => {
      expect(sanitizeApiKey('"sk-ant-12345"')).toBe('sk-ant-12345');
      expect(sanitizeApiKey("'sk-proj-12345'")).toBe('sk-proj-12345');
    });
  });

  describe('getProviderMeta', () => {
    it('returns metadata for known public providers', () => {
      expect(getProviderMeta('gemini')?.displayName).toBe('Google Gemini');
      expect(getProviderMeta('gemini')?.defaultBaseUrl).toBe('https://generativelanguage.googleapis.com/v1beta');
      expect(getProviderMeta('anthropic')?.displayName).toBe('Anthropic Claude');
      expect(getProviderMeta('openai')?.displayName).toBe('OpenAI');
    });

    it('returns null for local or mock providers', () => {
      expect(getProviderMeta('ollama')).toBeNull();
      expect(getProviderMeta('vllm')).toBeNull();
      expect(getProviderMeta('mock')).toBeNull();
    });
  });

  describe('isCloudProvider', () => {
    it('returns true for public cloud providers', () => {
      expect(isCloudProvider('gemini')).toBe(true);
      expect(isCloudProvider('openai')).toBe(true);
      expect(isCloudProvider('anthropic')).toBe(true);
    });

    it('returns false for local or custom providers', () => {
      expect(isCloudProvider('ollama')).toBe(false);
      expect(isCloudProvider('vllm')).toBe(false);
      expect(isCloudProvider('mock')).toBe(false);
    });
  });

  describe('getCuratedModels', () => {
    it('returns curated models for gemini, openai, and anthropic', () => {
      expect(getCuratedModels('gemini')).toContain('gemini-1.5-pro');
      expect(getCuratedModels('openai')).toContain('gpt-4o');
      expect(getCuratedModels('anthropic')).toContain('claude-3-5-sonnet-20241022');
    });

    it('returns empty array for local providers', () => {
      expect(getCuratedModels('ollama')).toEqual([]);
    });
  });

  describe('detectKeyProvider', () => {
    it('detects Gemini keys by prefix', () => {
      expect(detectKeyProvider('AIzaSyD_EXAMPLE_1234567890123456789012')).toBe('gemini');
    });

    it('detects Anthropic keys by prefix', () => {
      expect(detectKeyProvider('sk-ant-api03-abcdef1234567890123456789012')).toBe('anthropic');
    });

    it('detects OpenAI keys by prefix', () => {
      expect(detectKeyProvider('sk-proj-abcdef1234567890123456789012')).toBe('openai');
      expect(detectKeyProvider('sk-abcdef1234567890123456789012')).toBe('openai');
    });

    it('returns null for unrecognized strings or empty', () => {
      expect(detectKeyProvider('')).toBeNull();
      expect(detectKeyProvider('some-random-key')).toBeNull();
    });
  });

  describe('validateKeyFormat', () => {
    it('considers blank keys valid (optional or untouched)', () => {
      expect(validateKeyFormat('gemini', '').valid).toBe(true);
      expect(validateKeyFormat('gemini', '   ').valid).toBe(true);
    });

    it('validates a correct Gemini key format', () => {
      // 39 chars: AIzaSy + 33 chars
      const key = 'AIzaSy' + 'A'.repeat(33);
      const res = validateKeyFormat('gemini', key);
      expect(res.valid).toBe(true);
    });

    it('warns when an OpenAI key is entered for Gemini', () => {
      const res = validateKeyFormat('gemini', 'sk-proj-12345678901234567890');
      expect(res.valid).toBe(false);
      expect(res.warning).toContain('OpenAI');
    });

    it('warns when a Gemini key is entered for Anthropic', () => {
      const key = 'AIzaSy' + 'A'.repeat(33);
      const res = validateKeyFormat('anthropic', key);
      expect(res.valid).toBe(false);
      expect(res.warning).toContain('Gemini');
    });

    it('validates a correct Anthropic key format', () => {
      const key = 'sk-ant-api03-' + 'x'.repeat(30);
      expect(validateKeyFormat('anthropic', key).valid).toBe(true);
    });

    it('validates a correct OpenAI key format', () => {
      const key = 'sk-proj-' + 'x'.repeat(30);
      expect(validateKeyFormat('openai', key).valid).toBe(true);
    });
  });
});
