import type { DockSurface } from '../types';

/**
 * Developer mode: whether the dock offers its developer instruments at all, and whether
 * Settings shows its Diagnostics section.
 *
 * Off by default (owner ruling, 2026-09-22). A first-time user needs none of these surfaces,
 * and a developer reaches them with one deliberate action -- the switch in Settings --
 * rather than by having them removed (ui-authoring §2).
 *
 * A head presentation preference, kept in this browser beside the dock's width and the rail's
 * open state: a second head attached to the same Core would not need it (ui-authoring §3).
 */
export const DEVELOPER_MODE_KEY = 'uclone-x.developer-mode';

/**
 * The dock surfaces that are instruments rather than product surfaces, in drawer order.
 *
 * Knowledge Graph is among them: its triple-graph form is for reading the runtime, not for a
 * first-time user's work. ACP and Evals are instruments too, but about the build rather than
 * the conversation, so they are Settings' Diagnostics section (also developer mode only), and
 * Skills is a Settings section (#1358).
 */
export const DEVELOPER_SURFACES: readonly DockSurface[] = [
  'knowledge_graph',
  'topology',
  'ledger',
  'ontology',
];

export const isDeveloperSurface = (surface: DockSurface): boolean =>
  DEVELOPER_SURFACES.includes(surface);

/** The stored choice; anything but an explicit `'true'` -- including no storage -- is off. */
export const readDeveloperMode = (): boolean => {
  try {
    return window.localStorage.getItem(DEVELOPER_MODE_KEY) === 'true';
  } catch {
    /* Storage refused (private mode, a blocked origin): the default, which is off. */
    return false;
  }
};

/** Keep the user's choice for the next load. */
export const storeDeveloperMode = (on: boolean): void => {
  try {
    window.localStorage.setItem(DEVELOPER_MODE_KEY, String(on));
  } catch {
    /* A choice that cannot be kept still applies to this page. */
  }
};

/**
 * The surface the dock actually shows.
 *
 * A developer surface can be the selected one while developer mode is off -- the mode was
 * switched off with it open -- and the dock then shows Docs & Artifacts rather than a
 * surface the user has no tab for, or nothing at all. The selection itself is left alone, so
 * switching the mode back on returns to it.
 */
export const shownSurface = (active: DockSurface, developerMode: boolean): DockSurface =>
  !developerMode && isDeveloperSurface(active) ? 'artifacts' : active;
