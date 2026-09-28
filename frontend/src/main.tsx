import React from 'react';
import ReactDOM from 'react-dom/client';
import App from './App';
import { LocaleProvider } from './i18n';
import './index.css';
import { confirmWindowsOpenedHere } from './lib/person';

// Before anything else reads the address: a window the Core opened carries a one-time code
// after `#pair=`, spent here so the decisions only a person may make are accepted (#1589).
confirmWindowsOpenedHere();

ReactDOM.createRoot(document.getElementById('root')!).render(
  <React.StrictMode>
    <LocaleProvider>
      <App />
    </LocaleProvider>
  </React.StrictMode>
);
