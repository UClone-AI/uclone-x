/**
 * Modular Voice Synthesizer Registry.
 *
 * Every played note is synthesized in the browser by one of these voice modules.
 * Dispatches `VoiceRenderContext` events to the matching `VoiceModule` plugin
 * (`drumKitVoiceModule`, `pianoVoiceModule`, `stringsVoiceModule`, `bassVoiceModule`,
 * `windsBrassVoiceModule`, `guitarSynthVoiceModule`, or custom registered modules).
 */

import type { ResolvedInstrumentPatch, VoiceModule, VoiceRenderContext } from './types';
import { bassVoiceModule } from './voices/bassVoice';
import { drumKitVoiceModule } from './voices/drumKitVoice';
import { guitarSynthVoiceModule } from './voices/guitarSynthVoice';
import { pianoVoiceModule } from './voices/pianoVoice';
import { stringsVoiceModule } from './voices/stringsVoice';
import { windsBrassVoiceModule } from './voices/windsBrassVoice';

const BUILTIN_VOICE_MODULES: VoiceModule[] = [
  drumKitVoiceModule,
  pianoVoiceModule,
  stringsVoiceModule,
  bassVoiceModule,
  windsBrassVoiceModule,
  guitarSynthVoiceModule,
];

export class VoiceRegistry {
  private modules: VoiceModule[] = [...BUILTIN_VOICE_MODULES];

  /**
   * Register a custom `VoiceModule` at highest priority (front of the lookup chain).
   */
  public register(module: VoiceModule): void {
    this.modules = [module, ...this.modules.filter((m) => m.id !== module.id)];
  }

  public listModules(): readonly VoiceModule[] {
    return this.modules;
  }

  public resolveModule(patch: ResolvedInstrumentPatch): VoiceModule {
    for (const mod of this.modules) {
      if (mod.supports(patch)) {
        return mod;
      }
    }
    return pianoVoiceModule;
  }

  public triggerVoice(rc: VoiceRenderContext): void {
    const mod = this.resolveModule(rc.patch);
    mod.trigger(rc);
  }
}

export const defaultVoiceRegistry = new VoiceRegistry();
