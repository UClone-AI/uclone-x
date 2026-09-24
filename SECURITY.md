# Security Policy

## Reporting a vulnerability

If you believe you have found a security vulnerability in UClone-X, please
report it privately by emailing **kennylim@uclone.net**. Do not open a
public GitHub issue for a suspected vulnerability.

Please include, where possible:

- a description of the vulnerability and its potential impact;
- steps to reproduce, or a proof-of-concept;
- the version or commit hash you tested against.

There is currently no formal SLA for acknowledgement or remediation timelines
— this is a pre-release, single-maintainer project. You should expect a
best-effort response, not a guaranteed turnaround.

## Supported versions

| Version | Supported |
| :--- | :--- |
| `0.2.x` | Pre-release. No security support commitment yet. |
| `0.1.x` | No longer supported. |

UClone-X has not yet reached a `1.0` release. There is no long-term-support
branch and no backport policy for security fixes.

## Current security posture — read before relying on this project

UClone-X maintains an explicit threat model and security boundary analysis in:
👉 [`docs/security-threat-model.md`](docs/security-threat-model.md)

Key security properties of the architecture:

- **Sandbox Default Isolation**: Per Principle 3, the default execution sandbox isolation level is `workspace` (restricting filesystem writes to dedicated directories/worktrees). `none` is available only via explicit user opt-in.
- **Skill Execution & Synthesis**: Dynamic skill synthesis is subject to human-in-the-loop verification and Skill Auditor inspection (Fail-Closed default).
- **A2A Dual-Transport**: In-memory transport operates within the same process boundaries, while external connections adhere to Google A2A protocol authentication and authorization standards.

Do not assume any protection beyond what is explicitly documented in [`docs/security-threat-model.md`](docs/security-threat-model.md). If you are evaluating UClone-X for untrusted execution scenarios, review the threat model in full before proceeding.
