import { describe, expect, it } from 'vitest';
import { render, screen, within } from '@testing-library/react';
import { ModelRefSelect } from './ModelRefSelect';
import type { ModelEntry, ModelSet } from '../../lib/modelGateway';

const model = (id: string, display_name: string | null): ModelEntry => ({
  ref: `gemini/${id}`,
  id,
  display_name,
  capabilities: ['chat'],
  context_window: null,
});

const set = (models: ModelEntry[]): ModelSet => ({
  defaults: { deep: null, fast: null, image: 'auto' },
  recommended: { deep: null, fast: null },
  groups: [
    {
      connection_id: 'gemini',
      label: 'Google',
      kind: 'gemini',
      status: 'connected',
      detail: null,
      models,
    },
  ],
});

const words = {
  recommendedSuffix: '(recommended)',
  unavailableSuffix: '(unavailable)',
  groupLine: () => '',
};

const shown = (models: ModelEntry[]): string[] => {
  render(
    <ModelRefSelect
      id="m"
      label="Model"
      testId="model"
      value=""
      set={set(models)}
      leading={[{ value: '', label: 'Default' }]}
      words={words}
      onChange={() => {}}
    />,
  );
  const group = screen.getByRole('group', { name: 'Google' });
  return within(group)
    .getAllByRole('option')
    .map((option) => option.textContent ?? '');
};

describe('ModelRefSelect: two models with one name (#2416)', () => {
  // Killed by: frontend/src/lib/modelGateway.ts :: twice ? `${name} (${model.id})` : name
  // Becomes: name
  it('shows the id beside a display name another model in the group shares', () => {
    expect(
      shown([
        model('gemini-3.5-flash', 'Gemini 3.5 Flash'),
        model('gemini-3-pro', 'Gemini Pro'),
        model('gemini-3-pro-latest', 'Gemini Pro'),
      ]),
    ).toEqual(['Gemini 3.5 Flash', 'Gemini Pro (gemini-3-pro)', 'Gemini Pro (gemini-3-pro-latest)']);
  });

  it('shows a name no other model has as it is, and the id when there is no name', () => {
    expect(shown([model('gemini-3.5-flash', 'Gemini 3.5 Flash'), model('gemini-x', null)])).toEqual([
      'Gemini 3.5 Flash',
      'gemini-x',
    ]);
  });
});
