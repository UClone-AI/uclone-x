/**
 * The claims the dock may not make about what did not happen (#1366, #1374).
 *
 * The room records the files its clones saved *by name* and the tool calls their saved
 * turns reported. A shell, an MCP server or a helper a clone started can write, and can
 * call tools, without the room seeing it, so no dock surface can know that nothing was
 * written or that a clone used no tools. It lists what was recorded and names the gaps it
 * knows about (design doc Rev 31). An empty list is "none listed", never "none happened".
 *
 * `../dockAbsenceClaims.test.tsx` applies this pattern twice: to the copy in
 * every dock source (comments removed, as `copyGuard.ts` does it), and to the text each
 * dock surface renders in its empty and partial states. The pattern lives here, not in the
 * test, so a `Killed by:` declaration can name one of its lines.
 *
 * A pattern rather than an allow-list of empty-state sentences (#1389 weighed both). Most of
 * those sentences carry a seat's name, a count or the Core's own `reason`, so the list would
 * be of templates, which is a pattern again; and an allow-list cannot read the copy in a
 * source file, which is the half of the check that catches a claim before it renders.
 */

/**
 * A categorical claim that nothing was written, no tool was used, nothing is remembered, or
 * nothing happened.
 *
 * Covers the negations as well as the bare forms: "has not written", "hasn't written",
 * "never wrote a file", "has written nothing", "nothing written", "no file written", "no
 * files in this conversation", "without using a tool", "No tools called", "no tool use",
 * "0 tool calls", "remembers nothing", "has not remembered", "nobody has done anything".
 * "No files are listed", "No tool calls listed", "0 tool calls listed" and "No remembered
 * statements are listed" say what the list shows, and do not match.
 */
export const ABSENCE_CLAIM = new RegExp(
  [
    // Nothing was written.
    String.raw`\bno (?:files?|documents?) (?:(?:has|have|had) been |was |were )?(?:written|saved|created)\b`,
    String.raw`\b(?:0|zero|none of the) files? (?:(?:has|have|had) been |was |were )?(?:written|saved)\b`,
    String.raw`\bnothing (?:(?:is|was|has been|had been|were) )?(?:written|saved)\b`,
    String.raw`\b(?:has|have|had|did)(?: not|n't) (?:written|saved|write|save)\b`,
    String.raw`\b(?:wrote|written|saved) (?:no|nothing)\b`,
    String.raw`\bno files? in this conversation\b`,
    // Never did it at all: wrote, saved, used or called a tool, remembered or learned.
    String.raw`\bnever (?:wrote|written|write|saved|save|used?|called?|ran|run|remembered|remember|learned|learnt)\b`,
    // No tool was used.
    String.raw`\bwithout (?:using|calling) (?:a|any) tools?\b`,
    String.raw`\bno tools? (?:(?:was|were|has been|have been) )?(?:called|used|run)\b`,
    String.raw`\b(?:used|called) no tools?\b`,
    String.raw`\b(?:has|have|had|did)(?: not|n't) (?:used?|called?) (?:a|any) tools?\b`,
    // "No tool calls were made", "0 tool calls", "no tool use" -- unless it is what is listed.
    String.raw`\b(?:0|zero|no) tool (?:calls?|uses?|usage)\b(?! (?:(?:is|are|was|were|has been|have been) )?listed)`,
    // Nothing is remembered.
    String.raw`\b(?:remembers?|remembered|knows|learned|learnt) nothing\b`,
    String.raw`\b(?:has|have|had|did|does|do)(?: not|n't) (?:remember(?:ed)?|learn(?:ed|t)?)\b(?! (?:across|between) )`,
    String.raw`\bnothing (?:(?:is|was|has been|had been|were) )?(?:remembered|learned|learnt)\b`,
    String.raw`\bno memor(?:y|ies) (?:(?:is|are|was|were|has been|have been) )?(?:remembered|saved|stored|kept|learned)\b`,
    // Nothing happened.
    String.raw`\bnobody has (?:done|taken)\b`,
  ].join('|'),
  'i',
);

/** Every `<path>:<line>: <text>` among `lines` of `path` that makes an absence claim. */
export const absenceClaims = (path: string, lines: string[]): string[] =>
  lines.flatMap((line, index) =>
    ABSENCE_CLAIM.test(line) ? [`${path}:${index + 1}: ${line.trim()}`] : [],
  );

/**
 * The dock's own sources: the surfaces that read the room-scoped routes, and the types
 * they read them through. A path is relative to `src`, as Vite's glob gives it.
 */
export const DOCK_SOURCES: readonly string[] = [
  './components/artifacts/DocViewer.tsx',
  './components/artifacts/ActivityTimeline.tsx',
  './components/artifacts/RemembersPanel.tsx',
  './components/artifacts/KnowledgeGraphViewer.tsx',
  './components/TopologyTab.tsx',
  './components/layout/ArtifactsDock.tsx',
  './lib/roomDock.ts',
];
