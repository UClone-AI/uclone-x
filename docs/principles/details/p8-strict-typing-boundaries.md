# Principle 8: Strict Typing, Language Boundaries & Core-First Headless Autonomy

## Detailed Specification & Rationale

* **Core Law**: The system enforces strict static typing across clear language boundaries, and guarantees that all runtime logic and session state reside strictly in the headless Core Engine.
* **Strict Rule**: 
  - **Core Runtime**: Implemented in **Python 3.11+** with strict type annotations (`pyright --strict`, `ruff`, and Pydantic v2 models).
  - **Developer GUI**: Implemented in **TypeScript** (React 19 + Vite).
  - Dynamic untyped `dict` / `Any` contracts are forbidden across agent interfaces.
  - **Core-First Authority & Headless Autonomy**: All business logic, session state, conversation histories, tool execution outcomes, and reasoning loops MUST reside exclusively in the Core Engine (`src/uclone_x/`). The Core runtime must be completely autonomous, headless, and fully capable of resuming sessions and executing multi-turn tasks via CLI (`./ucx run`) or programmatic SDK without requiring a UI or browser.
  - **Heads Hold No Primary State**: A head — the Developer UI, the shipped desktop head ([P0](p0-product-purpose.md)), or any other — MUST NEVER hold primary state, manage independent conversation lifecycles, or implement logic that is absent from the Core. The rule is audience-neutral: it binds a product surface exactly as it binds a developer testbench. Only the *audience* characterisation moved to P0; the state authority stated here is unchanged and unweakened.
  - **Automated Quality Gate**: Strict typing (Pyright 0 error), lint/formatting (Ruff), and **unit test coverage >= 70%** are enforced on every build (`./ucx test check`).
  - **Verification Is Local. A Hosted Runner Is Not Verification.** The gate runs on the developer's own machine, through `./ucx test check` and the `pre-commit` hook, and that run is the verification. The published repository carries a GitHub Actions workflow so that a contribution can be checked by someone other than its author, and on axes one machine does not have — three Python versions, Linux, and a from-scratch install of the built distribution. It repeats the local gate; it does not replace it, and **a change is not considered verified because a hosted runner said so**. The development repository has no hosted CI at all.
  - **What this costs, stated rather than implied.** After this rule there is **no mechanical enforcement of anything** in this repository. Branch protection is unavailable while the repository is private on a free plan; `.github/CODEOWNERS` therefore blocks nothing; and the `pre-commit` hook is bypassable with `git commit --no-verify`, which Builders have used and recorded. **The gate is a discipline, not a control.** A Builder that skips it leaves no trace beyond its own commit message, and the project accepts that in exchange for a signal that means something when it fires.
  - All reasoning spans and A2A events must natively emit standard **OpenTelemetry (OTel)** telemetry.

* **Why**: Guarantees compile-time safety, architectural integrity, zero runtime schema bugs, and ensures the Core remains 100% headless, testable, and autonomous.
* **Implementation Reference**: `docs/cli-specification.md`, `docs/ui-dashboard-architecture.md`

