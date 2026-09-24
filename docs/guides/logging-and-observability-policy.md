# Application Logging and Observability Policy

**Document Version**: 1.0.0  
**Effective Date**: 2026-09-04  
**Authoritative Scope**: UClone-X Runtime Agents, CLI Runners, Background Daemons, and Swarm Builders  
**Issue Reference**: #447  
**Core Principle Alignment**: [P6: Fail-Fast & Zero Silent Fallbacks](../principles/core-principles.md)

---

## 1. Executive Summary & Problem Context

UClone-X is designed around rigorous observability and provenance invariants:
1. **OpenTelemetry Standard (`src/uclone_x/telemetry/`)**: Full OTLP tracing, spans, and metrics for distributed runtime evaluation and performance tracking.
2. **In-Band Provenance Envelope (P6)**: Every model response envelope contains execution provenance (`served_by`, `latency_ms`, `attempts`, `degraded`).
3. **Structured Session Store (`~/.uclone/sessions/`)**: Atomic JSON session records cleanly separating core cognitive turns (`core/`) from UI presentation transcripts (`ui/`).

However, practical local development and operations revealed a significant operational gap: **the absence of a unified, persistent application logging policy**. Runtime diagnostics, unhandled warnings, background daemon execution details, and tool failure signals were emitted solely to standard streams (`stdout`/`stderr`). Once a process terminated or terminal closed, developer diagnostic visibility was completely lost.

This document establishes the official application logging policy, operational file hierarchy, rotation mechanics, and format specifications across all UClone-X execution modes.

---

## 2. Core Separation of Concerns: Telemetry vs. Session State vs. Logs

To maintain architectural purity, UClone-X strictly delineates three distinct data planes:

| Data Plane | Purpose | Primary Location | Lifecycle & Storage | Consumers |
| :--- | :--- | :--- | :--- | :--- |
| **Telemetry & Spans** | Distributed tracing, span latencies, performance bottlenecks, parent-child agent calls. | Memory buffer / OTLP Exporter (Langfuse / Datadog) | Ephemeral / Exported via OpenTelemetry standard. | Observability dashboards, QA evaluation pipelines. |
| **Session State** | Conversation memory, conversational turns, tool call arguments, message tree reconstruction. | `~/.uclone/sessions/{core,ui}/sess_<id>.json` | Persistent until deleted by user or session manager. | Agent cognitive loop, UI chat history, frontend renderers. |
| **Application Logs** | Runtime diagnostic events, execution flow, tool exceptions, HTTP request logs, system warnings. | `~/.uclone/logs/ucx.log` | Rotating plain-text / structured JSONL files. | Developers, operators, system troubleshooters (`tail -f`). |

---

## 3. Log Storage Hierarchy & Rotation Standard

### 3.1 Default Directory & File Structure
All local runtime logs must be centralized under the user runtime home directory:
```
~/.uclone/
└── logs/
    ├── ucx.log             # Active application log (current runtime session)
    ├── ucx.log.1           # Rotated historical log archive
    ├── ucx.log.2           # Rotated historical log archive
    ├── ucx.audit.jsonl     # High-fidelity security/tool execution audit log
    └── ui-daemon.log       # Background web server / Vite dev server log
```

### 3.2 Rotation & Retention Policy
- **Maximum File Size**: 50 MB per log file.
- **Backup Count**: 5 historical rotations (`ucx.log.1` through `ucx.log.5`).
- **Disk Ceiling**: Maximum 300 MB total log footprint per local environment.
- **Auto-Pruning**: Rotator discards oldest archives beyond retention limits automatically without blocking event loop operations.

---

## 4. Log Formatting Standards

Logging in UClone-X operates under a dual-output model:

### 4.1 Console Output (Human-Friendly / Developer Interactive)
Formatted for readability with colorized ANSI log levels, concise timestamps, and contextual module paths:
```text
2026-09-04 17:35:01 [INFO] [uclone_x.agent.core] Initializing agent session 'sess_default' (model=qwen3:8b)
2026-09-04 17:35:02 [DEBUG] [uclone_x.tools.web] Dispatching tool 'web_search' with query='Apple stock price'
2026-09-04 17:35:03 [WARN] [uclone_x.tools.web] Upstream endpoint slow (latency=1420ms > threshold=1000ms)
2026-09-04 17:35:04 [INFO] [uclone_x.agent.core] Turn complete (served_by=qwen3:8b, latency=2180ms)
```

### 4.2 File Output (Structured JSONL)
When writing to `~/.uclone/logs/ucx.log`, all entries are serialized as single-line JSON objects to enable seamless ingestion into log aggregators (e.g., Vector, jq, ELK):
```json
{"timestamp":"2026-09-04T17:35:01.120Z","level":"INFO","logger":"uclone_x.agent.core","message":"Initializing agent session","session_id":"sess_default","model":"qwen3:8b","trace_id":"4bf92f3577b34da6a3ce929d0e0e4736"}
{"timestamp":"2026-09-04T17:35:02.341Z","level":"DEBUG","logger":"uclone_x.tools.web","message":"Dispatching tool","tool":"web_search","query":"Apple stock price","trace_id":"4bf92f3577b34da6a3ce929d0e0e4736"}
{"timestamp":"2026-09-04T17:35:03.761Z","level":"WARN","logger":"uclone_x.tools.web","message":"Upstream endpoint slow","latency_ms":1420,"threshold_ms":1000,"trace_id":"4bf92f3577b34da6a3ce929d0e0e4736"}
{"timestamp":"2026-09-04T17:35:04.941Z","level":"INFO","logger":"uclone_x.agent.core","message":"Turn complete","served_by":"qwen3:8b","latency_ms":2180,"trace_id":"4bf92f3577b34da6a3ce929d0e0e4736"}
```

---

## 5. Configuration & CLI Control

Developers and automated harnesses can dynamically configure logging behavior using either CLI flags or environment variables:

### 5.1 Environment Variables
- `UCX_LOG_LEVEL`: Log filtering threshold (`DEBUG`, `INFO`, `WARN`, `ERROR`, `CRITICAL`). Default: `INFO`.
- `UCX_LOG_DIR`: Custom path for log directory (Defaults to `~/.uclone/logs`).
- `UCX_LOG_FORMAT`: Format style (`json` or `text`). Default: `json` for file, `text` for console.
- `UCX_LOG_CONSOLE`: Toggle console log emission (`1`/`true` or `0`/`false`). Default: `true`.

### 5.2 CLI Subcommand Flags
All `./ucx` entrypoints accept universal logging arguments:
```bash
# Run server with verbose debug logging to both console and ucx.log
./ucx serve --log-level debug

# Run evaluation harness suppressing info logs
./ucx eval run --log-level warn

# Tail live application log stream
tail -f ~/.uclone/logs/ucx.log | jq .
```

---

## 6. Principle 6 Compliance: Zero Silent Fallbacks

Per **Core Principle 6 (P6: Fail-Fast & Zero Silent Fallbacks)**:
> *"Logging an exception is NEVER a substitute for propagating it. A system must fail loudly and cleanly rather than continue in an invalid or masked state."*

### Mandatory Rules for Builders & Runtime Developers:
1. **Never Swallow Exceptions in Catch Blocks**:
   ```python
   # ❌ FORBIDDEN (Violates P6)
   try:
       result = execute_critical_tool()
   except Exception as e:
       logger.error(f"Tool failed: {e}")
       return ""  # Silent fallback

   # ✅ REQUIRED (P6 Compliant)
   try:
       result = execute_critical_tool()
   except Exception as e:
       logger.error("Tool execution failed", exc_info=True, extra={"tool": tool_name})
       raise ToolExecutionError(f"Tool {tool_name} failed: {e}") from e
   ```
2. **Tool Execution Diagnostic Visibility**:
   - If a tool returns an empty result or error snippet, the runtime MUST log a `WARN` or `ERROR` entry with complete diagnostic context (input args, exit status, duration) before packaging the response.
3. **No Sensitive Secret Leaks**:
   - Authentication tokens, GitHub private keys, and API credentials must be sanitized before passing to `logger.*`.
