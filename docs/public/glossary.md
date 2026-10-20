# Glossary: uGraph, the knowledge graph

The one source for the names of the knowledge graph's parts. It covers the clone's memory and
the story example's codex. Each term gives:
- its proper name in English and Korean;
- where it lives in the code;
- what a person sees, in Korean and English;
- what it means;
- the names not to use for it;
- where that is defined.

Every code name below is in this repository. A design document may explain *why* a term is
shaped the way it is; *what* the term means is defined here.

Names marked **pending (owner)** are not decided. The page records what the screens say today,
and no one should rename them until the owner decides.

Paths are relative to the repository root. `src/uclone_x/knowledge/` is the uGraph package;
`examples/story/src/uclone_x_story/` is the story example, the first extension built on it.

## uGraph

**uGraph** (유그래프) is one fact model, and one way of resolving names, shared by every
vessel that keeps knowledge. Today that is a clone's memory and a story's codex. Their records
never mix (see *scope*). The package is `uclone_x.knowledge` (`src/uclone_x/knowledge/__init__.py`).

Three things a reader needs to know about it:

- **What reasoning does.** It derives facts and checks them, in memory, when they are read. The facts it derives are not written to either knowledge store.
  - The reasoner (`OntologyReasoner` in `src/uclone_x/ontology/reasoner.py`) does two things:
    - **derives** new facts to a fixed point (`materialize`, with nine fixed rules in `DERIVATION_RULES` plus safe Horn rules, `parse_horn_rule`);
    - **detects** contradictions (`check_consistency`, three detectors in `DETECTION_RULES`).
  - **The story:** the continuity check builds one in-memory engine per check and keeps nothing (`audit_scene` in `examples/story/src/uclone_x_story/audit.py`).
  - **The clone:** reading its Remembers works out, on every read, what the clone's rules derive from the facts that hold now. Nothing of it is saved (`worked_out_list` in `src/uclone_x/ui/knowledge.py`, called from `src/uclone_x/ui/room_dock.py`; the reasoner's `worked_out`).
- **Where rules come from.** The rules are data. No rule is built into the reasoner except the fixed derivation rules:
  - a story's own rules: `axioms` in its `story.yaml` (`StoryAxiom`, `story_axioms` in `examples/story/src/uclone_x_story/schemas.py`);
  - the defaults when a story has none (`DEFAULT_AXIOMS`, same file);
  - the rules the continuity check adds for itself alone: alive and dead are disjoint, one holder per item, and one gender and one relation per pair (`identity_axioms` in `examples/story/src/uclone_x_story/continuity.py`);
  - **a clone's own rules engine**, one per clone (`clone_ontology` and `ontology_map` in `src/uclone_x/agent/clone_builder.py`). It is loaded from the clone's `ontology.yaml` (`AgentHome.ontology_path`), and its rules carry tiers, from asserted to induced (`get_active_invariants`). Searching `src/uclone_x` for `save_to_yaml` finds two code paths that write that file: the `ucx ontology` commands (`teach`, `review`, `forget`; `src/uclone_x/cli/commands/ontology.py`), and the start-up import of a repository's ontology files (`import_repository_ontologies`). What it saves is rules, not derived facts.
- **What the predicate vocabulary is.** It is **open**. A predicate is a string, not a schema term (`Edge.predicate`).
  - The story's reading uses a fixed set of keys: `status`, `possesses`, `loses`, `gender`, `family`, `condition` (the prompt in `enrich_prompt`, `examples/story/src/uclone_x_story/enrich.py`).
  - A codex entry's `state` may hold any key a person writes.
  - A clone's learning files any short snake_case relation the model names (`src/uclone_x/memory/extractor.py`).
  - A predicate means something to the reasoner only where a rule names it.

## The model

| Term (en / ko) | In the code | What a person sees (ko / en) | Definition | Don't use | Source |
| :--- | :--- | :--- | :--- | :--- | :--- |
| uGraph / 유그래프 | `uclone_x.knowledge` | — (no screen says it) | The shared fact model and resolver above. | clone-kg, clodex | `src/uclone_x/knowledge/__init__.py` |
| KG (clone) / 클론 KG | `CrossSessionMemory` over `SqliteKnowledgeStore`; the file `knowledge.sqlite3` in the clone's folder | 기억 / Remembers | One clone's facts about its person, its workspace, itself, and what it found. | memory.json (the old file) | `src/uclone_x/memory/store.py`, `src/uclone_x/knowledge/sqlite_store.py` |
| KG (story) / 이야기 KG | `CodexYamlStore` over the codex files | 설정집 / Codex (**pending, owner**) | One story's codex read as uGraph edges. | — | `examples/story/src/uclone_x_story/codex_store.py` |
| scope / 그릇 | `Entity.scope`, `Edge.scope`: `clone:<agent id>`, `story:<story id>` | — | The vessel a record belongs to. A store opens one scope and never reads two together. | container, namespace | `src/uclone_x/knowledge/models.py`, `examples/story/src/uclone_x_story/codex_store.py` (`story_scope`) |
| entity / 엔티티 | `Entity` (`id`, `scope`, `kind`, `name`, `aliases`, `summary`) | 항목 / entry | Someone or something facts are about. | node | `src/uclone_x/knowledge/models.py` |
| edge / 간선 | `Edge` (`subject_id`, `predicate`, `object_id` or `value`, …) | — | One fact: a subject, a predicate, and another entity or a value. | triple (except for the reasoner's input) | `src/uclone_x/knowledge/models.py` |
| episode / 에피소드 | `Episode` (`kind`: `turn`, `scene`, `tool`, `manual`) | — | Where a fact was read: a turn, a scene, a tool result, or a hand edit. | — | `src/uclone_x/knowledge/models.py` |
| reserved entity / 예약 엔티티 | `CLONE_RESERVED_ENTITIES`: `PERSON_ENTITY` (`user`), `PROJECT_ENTITY` (`project`), `SELF_ENTITY` (`self`) | — | The three entities every clone has from its first fact. Their id is their name. | — | `src/uclone_x/knowledge/policy.py` |

## Time

| Term (en / ko) | In the code | What a person sees (ko / en) | Definition | Don't use | Source |
| :--- | :--- | :--- | :--- | :--- | :--- |
| valid interval / 유효 구간 | `Edge.valid` (an `Interval` with `axis`, `start`, `end`) | — | When a fact was true in its world. | — | `src/uclone_x/knowledge/models.py` |
| recorded interval / 기록 구간 | `Edge.recorded_at`, `Edge.expired_at` | — | When the vessel believed it. `expired_at` ends the belief; it is not a status. | — | `src/uclone_x/knowledge/models.py` |
| axis / 축 | `Axis`: `wall` (real time, the clone) or `story` (scene place, the story) | — | The time a `valid` interval is measured on. Two axes are never compared. | — | `src/uclone_x/knowledge/models.py`, `ClonePolicy.axis` |
| point / 시점 | `Position` | — | One place on an axis: an ISO-8601 UTC time on `wall`, a scene's place on `story`. | — | `src/uclone_x/knowledge/models.py` |
| view / 보기 | `APPROVED` (the default) and `APPROVED_OR_PROPOSED` (`proposed=True` on `edges_at`, `facts_at`, `holds_at`) | — | What holds at a point: what the vessel believes, or that plus what waits for a decision. In the second, a proposed change ends the value it replaces (`Edge.proposed_valid`). | — | `src/uclone_x/knowledge/store.py` |

## State, confidence, where a fact came from

| Term (en / ko) | In the code | What a person sees (ko / en) | Definition | Don't use | Source |
| :--- | :--- | :--- | :--- | :--- | :--- |
| status / 상태 | `EdgeStatus`: `proposed`, `approved`, `retracted` | 제안, 승인 / proposal, approved | Whether a fact is believed, waits for a person, or was taken back. There is no `expired` status; see *recorded interval*. | expired (as a status) | `src/uclone_x/knowledge/models.py` |
| origin / 출처 종류 | `Edge.origin` (a string); the clone's `FactOrigin`: `told`, `found`, `saved`, `corrected` | — | How a fact entered the vessel: a person said it, a tool result showed it, the clone saved it, a person corrected it. | — | `src/uclone_x/knowledge/models.py`, `src/uclone_x/memory/models.py` |
| confidence / 확신도 | `Edge.confidence`; the clone's `MIN_CONFIDENCE` (kept from 0.5) and `EXTRACTED_CONFIDENCE_CAP` (filed at most 0.8) | — | How sure the reading was, 0 to 1. | — | `src/uclone_x/knowledge/models.py`, `src/uclone_x/memory/extractor.py` |
| provenance / 출처 | `Provenance` | — | Which model or service produced a result, attributed in-band (P6). | — | `src/uclone_x/core/provenance.py` |
| quote / 인용 | `Edge.evidence` (episode id, quote); `ExtractedFact.quote` | — | The words of the source a fact rests on, copied exactly. | excerpt, paraphrase | `src/uclone_x/knowledge/models.py`, `src/uclone_x/knowledge/extraction.py` |
| quote check / 인용 확인 | `quote_found`, `MIN_QUOTE_CHARACTERS` | — | Whether a quote is in its source: matched ignoring case, Unicode form and white space, whole words, at least three letters or digits. | — | `src/uclone_x/knowledge/quotes.py` |

## Resolving names

| Term (en / ko) | In the code | What a person sees (ko / en) | Definition | Don't use | Source |
| :--- | :--- | :--- | :--- | :--- | :--- |
| fold / 접기 | `fold` (`quotes.folded` is the same function) | — | Text as names are compared: NFC, white space collapsed, case-folded. | — | `src/uclone_x/knowledge/fold.py` |
| resolution / 해석 | `EntityResolver.resolve` → `EntityRef` (`kind`: `existing`, `new`, `ambiguous`); `name_key` | — | Which entity of a scope a name means: by its id, name or alias, folded, with a parenthetical note or quotation marks taken off. Every write goes through it. | lookup, matching | `src/uclone_x/knowledge/resolver.py` |
| alias / 별칭 | `Entity.aliases`; a reading's `ExtractedFact.named`, candidate `ReadFact.alias`, confirmed by `alias_prompt` and `confirmed_aliases`; the story's `ProposalChange.aliases` | — | Another name an entity goes by, such as the same name in another script. The story proposes it, and a person approves. | nickname | `src/uclone_x/knowledge/extraction.py`, `examples/story/src/uclone_x_story/schemas.py` |
| ambiguity / 모호함 | `EntityRef.ambiguous_ids`; the drop reason `ambiguous_name` | — | A name two entities share. The fact is dropped; neither entity is guessed. | — | `src/uclone_x/knowledge/resolver.py`, `src/uclone_x/knowledge/extraction.py` |
| merge / 병합 | — (not implemented) | — | Taking two entities as one. Nothing merges entities today. On the clone, a near name such as `Mina` beside `Mina Park` is a new subject. | — | `src/uclone_x/knowledge/policy.py` (`ClonePolicy`) |

## Reading a source

| Term (en / ko) | In the code | What a person sees (ko / en) | Definition | Don't use | Source |
| :--- | :--- | :--- | :--- | :--- | :--- |
| reading / 읽기 | the story's `enrich_scene`; the clone's `KnowledgeExtractor.extract` | — | One model call that reads a scene or a turn and answers with facts, each with its quote. | extraction pass | `examples/story/src/uclone_x_story/enrich.py`, `src/uclone_x/memory/extractor.py` |
| reading the reply / 응답 읽기 | `listed_facts`, `parse_facts`, `reply_object`, `salvage_extraction` | — | The reply's facts read past a small model's mistakes, including a reply cut off by its token limit. | — | `src/uclone_x/knowledge/extraction.py` |
| verify_facts / 사실 확인 | `verify_facts` → `Verified` (`kept`, `dropped`) | — | Each fact held to its source and to the scope's entities. It checks that the quote is in the text, that names resolve to one entity, and that the object is named in the quote, and it keeps each fact once. | — | `src/uclone_x/knowledge/extraction.py` |
| drop reason / 버림 사유 | `Verified.dropped`: `quote_not_in_text`, `not_an_entity`, `ambiguous_name`, `named_other`, `object_not_in_quote`, `duplicate`; the story's reading adds its own, such as `subject_not_named`, `alias_rejected`, `already_pending` | — | Why a fact was not kept, counted by reason; nothing is lost silently. | — | `src/uclone_x/knowledge/extraction.py`, `examples/story/src/uclone_x_story/enrich.py` |

## The story's side

| Term (en / ko) | In the code | What a person sees (ko / en) | Definition | Don't use | Source |
| :--- | :--- | :--- | :--- | :--- | :--- |
| codex / 설정집 | `CodexIndex`; the files under a story's `codex/` | 설정집 / Codex (**pending, owner**: the public name of *codex*, and the English UI name of 설정집) | A story's people, places, items and threads, as YAML files a person can edit. | lore, story bible (until decided) | `examples/story/src/uclone_x_story/context.py`, `frontend/src/i18n/locales/*/dock.json` |
| entry / 항목 | `CodexEntry` (`id`, `name`, `aliases`, `state`, `relations`, `progressions`, …) | 항목 / entry | One codex file: one entity of the story. | — | `examples/story/src/uclone_x_story/schemas.py` |
| progression / 변화 | `Progression` (`at`, `set`, `relations`) | — | A change to an entry, in force once scene `at` has ended. | — | `examples/story/src/uclone_x_story/schemas.py` |
| relations / 관계 | `CodexEntry.relations`, `Progression.relations` | — | What an entry is to another entry, by that entry's id. | `kin_<id>` state keys (the old form) | `examples/story/src/uclone_x_story/schemas.py` |
| proposal / 제안 | `Proposal`, `ProposalChange`; decided by `apply_proposal` or `reject_proposal`, shown by `preview_proposal` | 제안 / proposal (a reply calls them 설정집 보강안) | A change to the codex a reading or the Writer suggests, and a person approves or rejects. | suggestion | `examples/story/src/uclone_x_story/schemas.py`, `examples/story/src/uclone_x_story/proposals.py` |
| continuity / 연속성 확인 | `checked_store`, `findings_from` | — | A deterministic check of a written scene's facts against the codex at that scene, in the approved-and-proposed view. | — | `examples/story/src/uclone_x_story/continuity.py` |
| finding / 지적 | `Finding` (`kind`: `dead_then_acting`, `holder_changed`, `disjoint`, `functional`, `inverse_functional`, `plan_*`) | the plain sentence of what does not hold | One thing the check found, with the quote it rests on. | error | `examples/story/src/uclone_x_story/continuity.py` |

## Stores and rules

| Term (en / ko) | In the code | What a person sees (ko / en) | Definition | Don't use | Source |
| :--- | :--- | :--- | :--- | :--- | :--- |
| knowledge store / 저장소 | `KnowledgeStore` (protocol), with `facts_at` | — | What a reader asks for the facts that hold at a point. | — | `src/uclone_x/knowledge/store.py` |
| SqliteKnowledgeStore | `SqliteKnowledgeStore`, `ClonePolicy` | — | The clone's store: one SQLite file, one record per write, safe for writers in parallel. New facts start `approved`. | — | `src/uclone_x/knowledge/sqlite_store.py`, `src/uclone_x/knowledge/policy.py` |
| CodexYamlStore | `CodexYamlStore` | — | The story's store: the codex YAML files and pending proposals, read as edges. The files stay the source. | — | `examples/story/src/uclone_x_story/codex_store.py` |
| reasoner / 추론기 | `OntologyReasoner` (`materialize`, `check_consistency`) | — | Derives facts in memory and finds contradictions; what it derives is not written to a knowledge store. See *uGraph* above. | — | `src/uclone_x/ontology/reasoner.py` |
| rule / 규칙 | `DERIVATION_RULES`, `DETECTION_RULES`, `parse_horn_rule`; a story's `StoryAxiom`, `DEFAULT_AXIOMS`, `identity_axioms`; a clone's rules engine (`clone_ontology`) | — | What the reasoner applies. Fixed derivation and detection rules, plus the rules a story gives or a clone's `ontology.yaml` holds. | — | `src/uclone_x/ontology/rules.py`, `examples/story/src/uclone_x_story/schemas.py`, `examples/story/src/uclone_x_story/continuity.py`, `src/uclone_x/agent/clone_builder.py` |

## Pending (owner)

These names are not decided. No decision was found in the repository or its issues. The
screens' current wording is recorded so that nothing is renamed before the owner decides.

1. **The public name of *codex*.** It is used today in the code and in the English UI. The
   choices are to keep it, or to use a name such as story bible, lore or setting.
2. **What a person calls the knowledge graph.** Today the clone's screen is 기억 / Remembers,
   and no screen says uGraph. The choices are 기억, 지식 or uGraph.
3. **The English UI name of 설정집.** Today it is "Codex" (`frontend/src/i18n/locales/en/dock.json`).
