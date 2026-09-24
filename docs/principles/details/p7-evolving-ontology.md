# Principle 7: Evolving Ontology Grounding

## Detailed Specification & Rationale

* **Core Law**: Every agent operates over a structured domain ontology that the agent actively self-constructs, expands through experience, and refines via developer guidance.
* **Strict Rule**: 
  - **Self-Evolution & User/Workspace Isolation**: Agents autonomously extract, structure, and accumulate domain entities, relations, state rules, and proven problem-solving patterns into scoped ontologies isolated per user and per workspace. Cross-workspace contamination is strictly prevented.
  - **Automated Curation & Filtering**: Ingested knowledge must pass a multi-stage curation pipeline:
    1. *Durable Knowledge Gate*: Filters out ephemeral conversation noise and transient outputs, capturing only reusable domain rules.
    2. *Semantic Deduplication & Conflict Resolution*: Merges semantically equivalent assertions and supersedes contradicted historical knowledge.
    3. *Confidence Scoring & Temporal Decay*: Knowledge items track empirical success/failure scores; unreinforced or failing items decay and are automatically pruned to prevent knowledge poisoning.
  - **Human Guidance**: Developers can explicitly define, steer, or augment an agent's ontology via schemas, prompts, or CLI instructions.
  - **Semantic Grounding**: Decisions and A2A collaboration must be validated against the agent's active ontology to prevent semantic drift.
* **Why**: Ontologies eliminate semantic drift, prevent hallucinations, enable deterministic validation of agent states, and provide a self-cleaning, verifiable knowledge substrate for complex multi-agent reasoning.

* **Implementation Reference**: `docs/agent-ontology-architecture.md`
