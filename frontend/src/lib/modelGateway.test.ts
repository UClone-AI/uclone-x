import { describe, expect, it } from 'vitest';
import { findModel, modelName, parseRef, sanitizeApiKey, statusOf, type ModelSet } from './modelGateway';

const SET: ModelSet = {
  groups: [
    {
      connection_id: 'gpu-box', label: 'GPU box', kind: 'vllm', status: 'connected', detail: null,
      models: [
        { ref: 'gpu-box/meta-llama/Llama-3.3-70B', id: 'meta-llama/Llama-3.3-70B', display_name: 'Llama 3.3 70B', capabilities: ['chat'], context_window: null },
      ],
    },
  ],
  defaults: { deep: null, fast: null, image: 'auto' },
  recommended: { deep: null, fast: null },
};

describe('model refs (model-gateway.md §3.1)', () => {
  // Killed by: frontend/src/lib/modelGateway.ts :: const at = ref.indexOf('/');
  // Becomes: const at = ref.lastIndexOf('/');
  it('splits at the first slash, because a model id may hold one and a connection id may not', () => {
    expect(parseRef('gpu-box/meta-llama/Llama-3.3-70B')).toEqual({ connectionId: 'gpu-box', modelId: 'meta-llama/Llama-3.3-70B' });
    expect(parseRef('ollama/qwen3:14b')).toEqual({ connectionId: 'ollama', modelId: 'qwen3:14b' });
  });

  it('reads a ref without a connection as no ref', () => {
    expect(parseRef('qwen3:14b')).toBeNull();
    expect(parseRef('/qwen3')).toBeNull();
    expect(parseRef('ollama/')).toBeNull();
  });

  it('names a listed ref by its display name, and an unlisted one as written', () => {
    expect(findModel(SET, 'gpu-box/meta-llama/Llama-3.3-70B')?.id).toBe('meta-llama/Llama-3.3-70B');
    expect(modelName(SET, 'gpu-box/meta-llama/Llama-3.3-70B')).toBe('Llama 3.3 70B');
    expect(modelName(SET, 'gemini/gemini-3.8-pro')).toBe('gemini/gemini-3.8-pro');
    expect(findModel(null, 'gpu-box/x')).toBeNull();
  });

  it('reads a status it has no words for as not checked yet', () => {
    expect(statusOf('connected')).toBe('connected');
    expect(statusOf('rate_limited')).toBe('unchecked');
  });

  it('cleans a pasted key of whitespace and surrounding quotes', () => {
    expect(sanitizeApiKey('  AIzaSy12345  ')).toBe('AIzaSy12345');
    expect(sanitizeApiKey('"sk-ant-12345"')).toBe('sk-ant-12345');
    expect(sanitizeApiKey("'sk-proj-12345'")).toBe('sk-proj-12345');
  });
});
