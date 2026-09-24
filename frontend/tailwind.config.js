import typography from '@tailwindcss/typography';

/** @type {import('tailwindcss').Config} */
export default {
  content: [
    "./index.html",
    "./src/**/*.{js,ts,jsx,tsx}",
    // Test files are not shipped, so the classes they name must not reach the production
    // stylesheet. Without these exclusions a `querySelector('.bg-rose-500')` in a test
    // keeps that utility alive in the bundle even if no component uses it any more.
    "!./src/**/*.test.{ts,tsx}",
    "!./src/test/**",
  ],
  theme: {
    extend: {},
  },
  plugins: [typography],
};
