import { fmt, plural, type Messages } from '../../i18n';
import type { ConnectionStatus, ModelGroup } from '../../lib/modelGateway';
import type { ModelRefSelectWords } from './ModelRefSelect';

type GatewayCopy = Messages['gateway'];

/**
 * A connection's status as the line a person reads: "Connected · 12 models", "No key",
 * "Not reachable". Chosen by `status`, never by the Core's `detail`.
 */
export const statusLine = (copy: GatewayCopy, status: ConnectionStatus, count: number | null): string => {
  if (status !== 'connected') return copy.status[status];
  return count === null ? copy.status.connected : plural(copy.status.connectedCount, count);
};

/** The picker's words, for every model picker on the screen. */
export const pickerWords = (copy: GatewayCopy, recommendedSuffix: string): ModelRefSelectWords => ({
  recommendedSuffix,
  unavailableSuffix: copy.picker.unavailableSuffix,
  groupLine: (group: ModelGroup) =>
    fmt(copy.picker.groupLine, {
      name: group.label,
      status: group.status === 'connected' ? copy.picker.noModelsOfKind : copy.status[group.status],
    }),
});
