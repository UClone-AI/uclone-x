import { en, type Messages } from '../i18n/en';
import { fmt } from '../i18n/format';

export type ActivityCategory = 'mutation' | 'command' | 'web' | 'inspection' | 'tool';

export interface ToolClassification {
  category: ActivityCategory;
  label: string;
  summaryTitle: string;
}

export const classifyTool = (
  toolName: string,
  args?: Record<string, unknown>,
  /** The step names, in the reader's language. English unless a surface passes its own. */
  copy: Messages['toolSteps'] = en.toolSteps,
): ToolClassification => {
  const name = (toolName || '').toLowerCase();

  // File Mutations. `file_write` and `file_edit` are the registered names; the others are
  // other hosts' spellings (#1463).
  if (
    name === 'file_write' ||
    name === 'file_edit' ||
    name.includes('write_to_file') ||
    name.includes('replace_file_content') ||
    name.includes('write_file') ||
    name.includes('edit_file') ||
    name.includes('patch_file') ||
    name.includes('delete_file')
  ) {
    const target =
      (args?.TargetFile as string) ||
      (args?.path as string) ||
      (args?.file_path as string) ||
      (args?.target as string) ||
      '';
    const basename = target ? target.split('/').pop() : '';
    return {
      category: 'mutation',
      label: copy.mutation.label,
      summaryTitle: target
        ? fmt(copy.mutation.on, { target: basename || target })
        : fmt(copy.mutation.bare, { tool: toolName }),
    };
  }

  // Command Execution. `bash_run` is the one shell the model is offered since #1461;
  // `run_command` is its unadvertised alias, still resolvable from stored calls (#1463).
  if (
    name === 'bash_run' ||
    name.includes('run_command') ||
    name === 'bash' ||
    name === 'sh' ||
    name.includes('exec')
  ) {
    const cmd =
      (args?.CommandLine as string) ||
      (args?.command as string) ||
      (args?.cmd as string) ||
      '';
    const shortCmd = cmd.length > 50 ? `${cmd.substring(0, 47)}...` : cmd;
    return {
      category: 'command',
      label: copy.command.label,
      summaryTitle: cmd
        ? fmt(copy.command.on, { target: shortCmd })
        : fmt(copy.command.bare, { tool: toolName }),
    };
  }

  // Web Retrieval / Browsing
  if (
    name.includes('search_web') ||
    name.includes('web_search') ||
    name === 'web_fetch' ||
    name.includes('read_url_content') ||
    name.includes('fetch_url') ||
    name.includes('read_browser_page')
  ) {
    const query =
      (args?.query as string) ||
      (args?.Url as string) ||
      (args?.url as string) ||
      '';
    const shortQuery = query.length > 50 ? `${query.substring(0, 47)}...` : query;
    return {
      category: 'web',
      label: copy.web.label,
      summaryTitle: query
        ? fmt(copy.web.on, { target: shortQuery })
        : fmt(copy.web.bare, { tool: toolName }),
    };
  }

  // Code & Directory Inspection
  if (
    name === 'file_read' ||
    name === 'file_search' ||
    name === 'directory_list' ||
    name.includes('read_file') ||
    name.includes('view_file') ||
    name.includes('grep_search') ||
    name.includes('find_by_name') ||
    name.includes('list_dir')
  ) {
    const target =
      (args?.AbsolutePath as string) ||
      (args?.SearchPath as string) ||
      (args?.DirectoryPath as string) ||
      (args?.TargetFile as string) ||
      (args?.path as string) ||
      (args?.query as string) ||
      '';
    const basename = target ? target.split('/').pop() : '';
    return {
      category: 'inspection',
      label: copy.inspection.label,
      summaryTitle: target
        ? fmt(copy.inspection.on, { target: basename || target })
        : fmt(copy.inspection.bare, { tool: toolName }),
    };
  }

  return {
    category: 'tool',
    label: copy.tool.label,
    summaryTitle: fmt(copy.tool.on, { tool: toolName || copy.tool.unknown }),
  };
};

export const parseArgs = (preview: string | undefined | null): unknown => {
  if (!preview) return undefined;
  try {
    return JSON.parse(preview);
  } catch {
    // A preview cut to fit is not whole JSON; the text is still what was sent.
    return preview;
  }
};

export const argsRecord = (args: unknown): Record<string, unknown> | undefined =>
  args && typeof args === 'object' && !Array.isArray(args)
    ? (args as Record<string, unknown>)
    : undefined;
