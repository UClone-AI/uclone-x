import { defineConfig } from 'vite';
import react from '@vitejs/plugin-react';
import path from 'path';

// The single frontend version source. `tests/unit/test_smoke.py` keeps this in
// agreement with `pyproject.toml` and `src/uclone_x/__init__.py`, so injecting it
// here is what stops the dashboard badge from drifting -- it was hardcoded
// `v0.1.0` and an E2E test pinned the same literal, so the gate stayed green
// precisely because both were stale.
import pkg from './package.json';

export default defineConfig({
  plugins: [react()],
  define: {
    __APP_VERSION__: JSON.stringify(pkg.version),
  },
  build: {
    outDir: path.resolve(__dirname, '../src/uclone_x/ui_static'),
    emptyOutDir: true,
  },
  test: {
    // jsdom, because the component tests assert on rendered text rather than on a
    // component's props. `globals: false` keeps `describe`/`it`/`expect` explicit
    // imports, so a test file reads the same as any other module.
    environment: 'jsdom',
    globals: false,
    setupFiles: ['./src/test/setup.ts'],
    include: ['src/**/*.test.{ts,tsx}'],
    // A committed `it.only` / `describe.only` must fail the run, not shrink it. vitest
    // defaults this to `!isCI`, and the gate runs locally without `CI` set, so a stray
    // `.only` would have run a fraction of the suite and still exited 0 (#930).
    // `tests/unit/test_frontend_vitest_config.py` pins the value.
    allowOnly: false,
  },
  server: {
    port: 5173,
    proxy: {
      '/api': {
        target: 'http://localhost:5180',
        changeOrigin: true,
      },
    },
  },
});

