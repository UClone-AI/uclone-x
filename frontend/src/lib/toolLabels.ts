export type ActivityCategory = 'mutation' | 'command' | 'web' | 'inspection' | 'tool';

export interface ToolClassification {
  category: ActivityCategory;
  label: string;
  summaryTitle: string;
}

export const classifyTool = (
  toolName: string,
  args?: Record<string, unknown>
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
      label: 'File Mutation',
      summaryTitle: target ? `File Mutation: ${basename || target}` : `File Mutation (${toolName})`,
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
      label: 'Command',
      summaryTitle: cmd ? `Command: ${shortCmd}` : `Command Execution (${toolName})`,
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
      label: 'Web Retrieval',
      summaryTitle: query ? `Web Retrieval: ${shortQuery}` : `Web Retrieval (${toolName})`,
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
      label: 'Inspection',
      summaryTitle: target ? `Inspection: ${basename || target}` : `Inspection (${toolName})`,
    };
  }

  return {
    category: 'tool',
    label: 'Tool Call',
    summaryTitle: `Tool Call: ${toolName || 'Unknown Tool'}`,
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
