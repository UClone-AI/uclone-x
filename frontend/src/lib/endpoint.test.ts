import { describe, it, expect } from 'vitest';
import { normalizeEndpoint, sameEndpoint } from './endpoint';

describe('sameEndpoint', () => {
  it('treats the loopback spellings of one port as one server', () => {
    // Killed by: frontend/src/lib/endpoint.ts :: const host = LOOPBACK_HOSTS.has(rawHost) ? 'localhost' : rawHost;
    // Becomes: const host = rawHost;
    expect(sameEndpoint('http://localhost:11434', 'http://127.0.0.1:11434')).toBe(true);
    expect(sameEndpoint('http://[::1]:11434', 'http://localhost:11434')).toBe(true);
    expect(sameEndpoint('http://[::1]:8000', 'http://127.0.0.1:8000')).toBe(true);
  });

  it('keeps loopback addresses on different ports apart', () => {
    expect(sameEndpoint('http://localhost:11434', 'http://127.0.0.1:8000')).toBe(false);
    expect(sameEndpoint('http://localhost:11434', 'http://localhost')).toBe(false);
  });

  it('ignores surrounding space, trailing slashes and the case of scheme and host', () => {
    expect(sameEndpoint('  HTTP://LocalHost:11434// ', 'http://localhost:11434')).toBe(true);
    expect(sameEndpoint('http://GPU-Box:8000/v1/', 'http://gpu-box:8000/v1')).toBe(true);
  });

  it('keeps different hosts, schemes and paths apart', () => {
    expect(sameEndpoint('http://gpu-box:8000', 'http://localhost:8000')).toBe(false);
    expect(sameEndpoint('https://localhost:8000', 'http://localhost:8000')).toBe(false);
    expect(sameEndpoint('http://localhost:8000/v1', 'http://localhost:8000/v2')).toBe(false);
  });

  it('compares an empty field with an empty field, and with nothing else', () => {
    expect(sameEndpoint('', '  ')).toBe(true);
    expect(sameEndpoint('', 'http://localhost:11434')).toBe(false);
  });
});

describe('normalizeEndpoint', () => {
  it('leaves the case of a path alone', () => {
    expect(normalizeEndpoint('http://Host:1/API/')).toBe('http://host:1/API');
  });

  it('only trims a string with no scheme', () => {
    expect(normalizeEndpoint(' localhost:11434/ ')).toBe('localhost:11434');
  });
});
