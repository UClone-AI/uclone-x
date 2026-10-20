# pyright: reportPrivateUsage=false
"""Bilingual reply notes, stop descriptions, and flow guards for `story_start`."""

from __future__ import annotations

import re
from collections import Counter
from collections.abc import Mapping, Sequence
from typing import Any, cast

import yaml
from pydantic import ValidationError

from uclone_x_story.enrich import growth_line
from uclone_x_story.schemas import Outline
from uclone_x_story.start_state import (
    MAX_CAST_TRIES,
    START_FILE,
    Brief,
    Language,
    Stage,
    StartState,
    StoryStartParams,
    genre_label,
)
from uclone_x_story.work import StoryWork


def _listed_genres(state: StartState) -> dict[str, list[str]]:
    """The genres the genre stop's note lists, in each language's own words
    (`NOTE_LISTS_KEY`)."""
    return {lang: [genre_label(g, lang) for g in state.genre_options] for lang in ("ko", "en")}


_NEXT: Mapping[Stage, str] = {
    "choose_genre": "The request named no genre, so genres are proposed, the recommended "
    "one first. Ask the person which they want, and call story_start 'continue' with "
    "'choice' (1–3), 'genre', or 'go_on=true' for the recommended genre (plus 'must_have' / "
    "'avoid' if they named any).",
    "choose_premise": "Show the person the three premises and ask which one. Call story_start "
    "'continue' with 'choice' (1–3; the recommended number for a bare yes) (or 'changes' "
    "if they reject all three and want new premises).",
    "review_cast": "The cast and places are proposed; show them to the person and wait for "
    "their answer. Call story_start 'continue' with 'go_on=true' (plus 'must_have' / 'avoid' "
    "if they asked to add or drop characters or places). The first chapter's plan waits until "
    "the proposals are decided or the person says to go on anyway.",
    "review_outline": "The outline is saved; ask the person whether it works. If they ask for "
    "a change, call story_start 'continue' with 'changes' (without 'go_on'): the outline is "
    "revised and shown again. When they agree to the outline, call 'continue' with "
    "'go_on=true': the plot is settled, its key events are proposed for the codex, and the "
    "first chapter's scene plan is shown.",
    "review_chapter": "A chapter's scene plan is shown before the chapter is written (after "
    "the chapter before it, if any, was written). Ask the person whether it works. If they "
    "ask for a change, call story_start 'continue' with 'changes' (without 'go_on'): the plan "
    "is revised and shown again. When they agree, call 'continue' with 'go_on=true': that "
    "chapter is written, checked, and the plan of the one after it is shown.",
    "review_scene": "Three RPG choices and the current world state (KG) are shown before "
    "the scene is written. Show the person the world state and the three choices, and ask "
    "which they want. Call story_start 'continue' with 'choice' (1–3; or 'go_on=true' for "
    "choice 1), or pass their custom action or KG update in 'changes' (with or without "
    "'choice'): the chosen scene is written, the KG transitions are updated in chronological "
    "order, and the next scene's choices are shown.",
    "done": "Every chapter of the outline is written. Go on with story_context and "
    "story_manuscript if the person wants more.",
}
_CAST_FAILED_NEXT = (
    "No character could be proposed from the chosen premise, so the flow stopped before "
    "the outline. Tell the person so; when they want to try again, call story_start "
    "'continue' (with 'choice' if they choose another premise)."
)
_CONTINUITY_NEXT = (
    "A chapter is written, and the check after writing found lines that disagree with the "
    "codex, or with the scene's plan or the settled plot. Show the person the notes as they "
    "are, and ask whether to rewrite those scenes. Only if they ask for it, call story_start "
    "'continue' with 'rewrite=true'. Anything else they say goes on as for the plan shown, "
    "if there is one."
)
_WAITING_NEXT = (
    "The first chapter was not planned: proposed cast entries are still waiting for the "
    "person's approval. Ask them to approve or reject them in the story's view, then call "
    "story_start 'continue'; or, if they say to go on as proposed, call 'continue' with "
    "'go_on=true'."
)


#: Why a 'continue' or 'leave' in the turn that made the stop is refused: the stop is the
#: person's to answer, by a message of their own (#2300). Decided by the room turn's id,
#: not by reading any words.
_SAME_TURN = (
    "The story stopped for the person in this same turn, and they have not answered yet, "
    "so nothing was changed. End your reply here: show them what the stop showed and ask "
    "what it asks. Call story_start 'continue' only with their next message."
)


def _next_of(state: StartState) -> str:
    """What the model is told to do next at the state's stop: the stop's own `next`, unless
    the cast is waiting, the cast failed, or the check after writing found something."""
    if state.cast_waiting:
        return _WAITING_NEXT
    if state.cast_failed:
        return _CAST_FAILED_NEXT
    if state.stage in ("review_chapter", "review_scene", "done") and _shown_findings(state):
        return _CONTINUITY_NEXT
    if state.stage == "review_chapter" and state.details_scene_edits_ambiguous:
        return _SCENE_EDIT_AMBIGUOUS_NEXT.format(
            scenes="; ".join(
                f"'{a['title']}' ({a['en']})" for a in state.details_scene_edits_ambiguous
            )
        )
    if state.stage == "review_chapter" and state.details_scene_edits:
        return _SCENE_EDIT_NEXT.format(
            scenes=", ".join(f"{e['id']} '{e['title']}'" for e in state.details_scene_edits)
        )
    return _NEXT[state.stage]


_SCENE_EDIT_AMBIGUOUS_NEXT = (
    "The person's words name a scene title that more than one written scene has ({scenes}), "
    "so no scene was written again and the plan was kept as it was. Ask them which one they "
    "mean, by chapter and scene; then pass their answer to story_start 'continue', word for "
    "word, as 'changes'. Do not write the scene yourself."
)

_SCENE_EDIT_NEXT = (
    "The person's words ask to change scenes already written ({scenes}), not the plan of "
    "the next chapter, so those scenes were written again as they asked and saved, and the "
    "plan was kept as it was. Tell the person, and that the plan still waits for their "
    "answer. Do not write those scenes again yourself."
)


#: Each stop before the chapters, in words: what the person is answering there. A refusal
#: names the stop this way rather than by its stage id (#2270).
_STOP_WORDS: Mapping[str, str] = {
    "choose_genre": "the genres it proposed",
    "choose_premise": "the three premises it offered",
    "review_cast": "the cast and places it proposed",
    "review_outline": "the outline it made",
    "review_chapter": "the chapter's scene plan it showed",
    "review_scene": "the scene's RPG choices it showed",
}


def _waiting_state(work: StoryWork) -> StartState | None:
    found = work.read(START_FILE)
    if found is None:
        return None
    try:
        return StartState.model_validate(yaml.safe_load(found.text))
    except (yaml.YAMLError, ValidationError):
        return None


def start_waiting(work: StoryWork) -> StartState | None:
    """The start of this story while it waits for the person at any stop, a chapter's plan
    included; `None` when it has no start, every chapter is written, the person left it, or
    its record cannot be read (#2300)."""
    state = _waiting_state(work)
    if state is None or state.stage == "done" or state.left_request is not None:
        return None
    return state


_RAW_SCENE_ID_RE = re.compile(r"\bch(\d+)\.s(\d+)\b")


def _human_scene_ids(text: str, lang: Language) -> str:
    """Replace internal `chXX.sYY` IDs with human-readable labels so user replies never leak IDs."""

    def _repl(match: re.Match[str]) -> str:
        ch = int(match.group(1))
        sc = int(match.group(2))
        return f"{ch}장 {sc}번째 장면" if lang == "ko" else f"chapter {ch}, scene {sc}"

    return _RAW_SCENE_ID_RE.sub(_repl, text)


def stop_words(state: StartState) -> dict[str, str]:
    """The stop the start waits at, as the person reads it, in each language (#2300)."""
    if state.stage == "review_chapter":
        number = state.details_chapter or state.chapter_written + 1
        return {"ko": f"{number}장의 장면 계획", "en": f"chapter {number}'s scene plan"}
    if state.stage == "review_scene":
        sid = state.scene_id or ""
        ko_sid = _human_scene_ids(sid, "ko") if sid else "다음 장면"
        en_sid = _human_scene_ids(sid, "en") if sid else "the next scene"
        return {"ko": f"{ko_sid}의 선택지", "en": f"{en_sid}'s choices"}
    words = {
        "choose_genre": ("제안한 장르", "the genres proposed"),
        "choose_premise": ("제안한 세 가지 발상", "the three premises proposed"),
        "review_cast": ("제안한 인물과 장소", "the cast and places proposed"),
        "review_outline": ("개요", "the outline"),
    }
    ko, en = words.get(state.stage, ("지금 단계", "this step"))
    return {"ko": ko, "en": en}


def flow_stop(work: StoryWork) -> StartState | None:
    """The start of this story while it waits for the person at any stop, a chapter's plan
    included; `None` when it has no start, every chapter is written, it was left (`leave`),
    or its record cannot be read.

    Until #2300 this held only before the chapter loop. In a live run at the first
    chapter's plan, qwen3:8b wrote that chapter's first scene by hand; `story_start` then
    could not write the chapter over it ("already has text"), and the start stood still
    for the rest of the conversation."""
    return start_waiting(work)


def by_hand_refusal(state: StartState, *, written: bool = False) -> str:
    """Why an outline or scene written by hand was refused during a start, and what goes on
    (`_next_of`, so a failed cast gets its own way on). It never names `leave`: a model
    that just tried to write by hand would be told how to switch the guard off (#2273
    review); only the Writer prompt names it, for when the person asks.

    `written`: the write was over a scene that has text. The start rewrites such a scene
    itself when the person's words ask for it (#2361), so the refusal names that call."""
    if written:
        return (
            f"Nothing was written: this story's start is waiting for the person's answer to "
            f"{_STOP_WORDS.get(state.stage, 'what it showed')}, and while it waits a scene "
            "already written is rewritten by it, not by hand. Call story_start 'continue' "
            "with the person's words, word for word, as 'changes': it writes the scene they "
            "ask about again and keeps the plan."
        )
    return (
        f"Nothing was written: this story's start is waiting for the person's answer to "
        f"{_STOP_WORDS.get(state.stage, 'what it showed')}, and until every chapter is "
        f"written it makes the outline and the new scenes itself. {_next_of(state)}"
    )


def _waiting_scene_words(pending: Sequence[Mapping[str, str]]) -> str:
    """The words of a scene request still waiting for "which one?", for the plan question
    to read apart from the feedback (#2368): only to tell which written scene the answer
    means, never as a change to this plan, so a person who moved on to the plan does not
    have it revised by an older request."""
    if not pending:
        return ""
    said = "".join(f"- {a['words']}\n" for a in pending)
    return (
        "Earlier, the person asked to change a scene already written, and was asked which "
        "one, since more than one has that title. What they asked then, only to tell which "
        f"written scene their feedback now means (edits_written), not a change to this "
        f"plan:\n{said}\n"
    )


def _scene_labels(outline: Outline, written: Mapping[str, str]) -> dict[str, str]:
    """Each written scene's id and the label the plan prompt lists it by: its title, or,
    for a title more than one written scene has, the title and where it is (#2365)."""
    seen = Counter(written.values())
    labels: dict[str, str] = {}
    for chapter_no, chapter in enumerate(outline.chapters, start=1):
        for scene_no, scene in enumerate(chapter.scenes, start=1):
            if scene.id not in written:
                continue
            title = written[scene.id]
            label = (
                title if seen[title] == 1 else f"{title} (chapter {chapter_no}, scene {scene_no})"
            )
            labels[scene.id] = label
    return labels


def _where(outline: Outline, written: Mapping[str, str], title: str) -> dict[str, str]:
    """`title` and where each written scene that has it is, as the person reads it."""
    places = [
        (chapter_no, scene_no)
        for chapter_no, chapter in enumerate(outline.chapters, start=1)
        for scene_no, scene in enumerate(chapter.scenes, start=1)
        if written.get(scene.id) == title
    ]
    return {
        "title": title,
        "ko": ", ".join(f"{c}장 {s}번째" for c, s in places),
        "en": ", ".join(f"chapter {c}, scene {s}" for c, s in places),
    }


#: Why no character was proposed, as the person is told it.
_CAST_FAILED: Mapping[str, tuple[str, str]] = {
    "unreadable": (
        f"인물 목록을 {MAX_CAST_TRIES}번 만들었지만 모두 형식이 깨져 읽을 수 없었습니다.",
        f"the cast came back {MAX_CAST_TRIES} times in a form that could not be read",
    ),
    "empty": (
        f"{MAX_CAST_TRIES}번 시도했지만 이름이 있는 인물이 한 명도 나오지 않았습니다.",
        f"{MAX_CAST_TRIES} tries gave no character with a name",
    ),
    "taken": (
        "제안한 인물이 모두 이 이야기에 이미 있는 인물과 겹쳤습니다.",
        "every character proposed is already in the story",
    ),
}


def _reply_note(
    state: StartState, done: Sequence[str] = (), outline: Outline | None = None
) -> dict[str, str]:
    """What code tells the person, whatever the model writes: the stage and the question.

    The call that settled the plot says so first, and what it proposed for the codex. At
    the outline stop, `outline` is the outline the person is asked about, shown in full.
    """
    note = _stage_note(state, done, outline)
    if "introduced" in done:
        names = [
            str(k["name"])
            for k in state.introduced
            if k.get("change") == "new_entry" and k.get("chapter") == state.chapter_written
        ]
        if names:
            listed = ", ".join(names)
            note = {
                "ko": f"계획에 나온 새 인물({listed})과 그 가족 관계를 설정집에 제안했습니다. "
                f"이야기 보기에서 승인하시면 설정집에 들어갑니다. {note['ko']}",
                "en": f"The new characters the plan brought in ({listed}) and their family "
                f"ties were proposed for the codex; they join it once you approve them. "
                f"{note['en']}",
            }
    if "plot" not in done:
        return note
    ko, en = _plot_note(state)
    return {"ko": f"{ko} {note['ko']}", "en": f"{en} {note['en']}"}


#: How the person undoes a misread from the said-back line, in one turn (#2342 item 1).
_KEPT_CHANGE_NOW = (
    "지금 바꾸고 싶으시면 말씀해 주세요.",
    "Tell me now if you would like it changed instead.",
)


def _kept_note(kept: Mapping[str, Sequence[str]]) -> tuple[str, str] | None:
    """The line that says back what the person asked for beside taking what a stop showed,
    in Korean and English; `None` when they asked for nothing (#2313). Made from the
    reading's fields, so it names their requests as the model read them."""
    if not any(kept.get(k) for k in ("must_have", "avoid")):
        return None
    must = ", ".join(f"‘{t}’" for t in kept.get("must_have", ()))
    avoid = ", ".join(f"‘{t}’" for t in kept.get("avoid", ()))
    ko = "덧붙이신 요청은 앞으로 쓸 내용에 반영하겠습니다."
    en = "I will keep what you added for the story ahead."
    if must:
        ko += f" 넣을 것: {must}."
        en += f" To include: {must}."
    if avoid:
        ko += f" 넣지 않을 것: {avoid}."
        en += f" To leave out: {avoid}."
    ko += f" {_KEPT_CHANGE_NOW[0]}"
    en += f" {_KEPT_CHANGE_NOW[1]}"
    return ko, en


def _plot_note(state: StartState) -> tuple[str, str]:
    if not state.plot_kg:
        return (
            "줄거리를 확정했습니다. 설정집에 새로 제안할 항목은 없었습니다.",
            "The plot is settled. Nothing new was proposed for the codex.",
        )
    names = ", ".join(k["name"] for k in state.plot_kg)
    count = len(state.plot_kg)
    return (
        f"줄거리를 확정했습니다. 확정한 줄거리에서 설정집에 넣을 항목 {count}개({names})를 "
        "제안했습니다. 이야기 보기에서 승인하시면 설정집에 들어갑니다.",
        f"The plot is settled. {count} entries from it were proposed for the codex "
        f"({names}); they join it once you approve them in the story's view.",
    )


def _outline_lines(outline: Outline | None) -> str:
    """The outline as the person reads it in the reply: each chapter and its scenes."""
    if outline is None:
        return ""
    lines = [
        f"{n}. {chapter.title}"
        + (": " + " · ".join(s.title for s in chapter.scenes) if chapter.scenes else "")
        for n, chapter in enumerate(outline.chapters, start=1)
    ]
    return "\n" + "\n".join(lines) + "\n"


def _stage_note(
    state: StartState, done: Sequence[str] = (), outline: Outline | None = None
) -> dict[str, str]:
    if state.cast_failed is not None:
        why_ko, why_en = _CAST_FAILED.get(state.cast_failed, _CAST_FAILED["empty"])
        return {
            "ko": f"고르신 {state.chosen or 1}번 발상에서 인물을 뽑지 못했습니다. {why_ko} "
            "인물 없이 개요나 첫 장을 만들지는 않았습니다. 다시 해 보라고 하시거나, "
            "다른 발상의 번호를 골라 주세요.",
            "en": f"No character could be drawn from premise {state.chosen or 1}: {why_en}. "
            "Nothing was outlined or written without a cast. Ask me to try again, or choose "
            "another premise by number.",
        }
    if state.stage == "choose_genre":
        ko_options = [genre_label(g, "ko") for g in state.genre_options]
        en_options = [genre_label(g, "en") for g in state.genre_options]
        lines = [
            f"{n}. {g}" + (" (추천)" if n == 1 else "") for n, g in enumerate(ko_options, start=1)
        ]
        lines_en = [
            f"{n}. {g}" + (" (recommended)" if n == 1 else "")
            for n, g in enumerate(en_options, start=1)
        ]
        return {
            "ko": "요청에 장르가 없어 장르부터 제안합니다.\n" + "\n".join(lines) + "\n"
            "‘네’라고 하시면 추천한 1번 장르로 발상을 만들겠습니다. 다른 장르를 원하시면 "
            "번호나 장르 이름을 말씀해 주세요.",
            "en": "The request named no genre, so here are some to choose from:\n"
            + "\n".join(lines_en)
            + "\nSay yes and I will build the premises in the recommended one, number 1; or "
            "name another genre or number.",
        }
    if state.stage == "choose_premise":
        lines = [f"{p.number}. {p.title} — {p.premise}" for p in state.premises]
        genre = f"{genre_label(state.brief.genre, 'ko')} " if state.genre_options else ""
        ko = f"{genre}이야기의 핵심 발상 세 가지입니다.\n" + "\n".join(lines)
        en = "Three premises for the story:\n" + "\n".join(lines)
        if state.premise_feedback:
            ko = f"말씀하신 내용으로 발상을 새로 만들었습니다. {ko}"
            en = f"Here are new premises, made from what you said. {en}"
        ko += (
            f"\n마음에 드는 발상의 번호를 골라 주세요. 추천은 {state.recommended}번입니다. "
            f"‘네’라고만 하셔도 {state.recommended}번으로 진행합니다."
        )
        en += f"\nPlease choose one by number. I recommend {state.recommended}; say yes to take it."
        return {"ko": ko, "en": en}
    if state.stage == "review_cast":
        names = ", ".join(str(c.get("name", "")) for c in state.cast)
        return {
            "ko": f"인물과 장소를 제안했습니다: {names}. 이야기 보기의 파일에서 승인하거나 "
            "고칠 수 있습니다. ‘네’라고 하시면 개요를 만들겠습니다. 첫 장은 제안을 모두 "
            "정하신 뒤에 계획합니다. 다만 개요에서 ‘네’라고만 하시면 정하지 않은 제안도 "
            "제안대로 보고 첫 장을 계획하고 쓰며, 그 제안은 승인하실 때까지 승인되지 않은 채로 "
            "남습니다.",
            "en": f"The cast and places are proposed: {names}. You can approve or change "
            "them in the story's view. Say yes and I will make the outline. The first "
            "chapter waits until every proposal is decided; but if you just say yes at the "
            "outline, it is planned and written with the undecided ones as proposed, and they stay "
            "unapproved until you approve them.",
        }
    if state.stage == "review_outline" and state.cast_waiting:
        count = len(state.cast_waiting)
        return {
            "ko": f"제안한 인물과 장소 중 {count}개가 아직 승인되지 않아 첫 장을 계획하지 "
            "않았습니다. 이야기 보기에서 승인하거나 거절하신 뒤 이어 달라고 말씀해 주세요. "
            "‘네’라고 하시거나 제안대로 진행하라고 하시면 첫 장의 계획을 보여 드리겠습니다.",
            "en": f"{count} proposed cast entries are not approved yet, so the first chapter "
            "is not planned. Approve or reject them in the story's view and ask me to go "
            "on, or tell me to go ahead as proposed.",
        }
    if state.stage == "review_outline":
        ko_shape, en_shape = _arc_note(state)
        if state.revision_failed:
            ko_head = "말씀하신 수정을 반영하지 못해 개요를 그대로 두었습니다."
            en_head = "Your change could not be made, so the outline is as it was."
        elif state.outline_feedback:
            ko_head = "말씀하신 대로 개요를 고쳐 저장했습니다."
            en_head = "The outline is revised as you asked, and saved."
        else:
            ko_head = "개요를 저장했습니다."
            en_head = "The outline is saved."
        shown = _outline_lines(outline)
        ko_not_yet, en_not_yet = _NOT_YET_OUTLINE
        return {
            "ko": f"{ko_head}{ko_shape}{shown}{ko_not_yet} 바꾸고 싶은 점이 있으면 말씀해 "
            "주세요. ‘네’라고 하시면 이 줄거리로 확정하고 첫 장의 장면 계획을 보여 드리겠습니다.",
            "en": f"{en_head}{en_shape}{shown}{en_not_yet} Tell me what to change, or say yes "
            "to settle the plot and see the first chapter's scene plan.",
        }
    if state.stage == "review_chapter" and state.chapter_failed:
        number = state.details_chapter or state.chapter_written + 1
        return {
            "ko": f"{number}장의 초안이 저장 기준을 통과하지 못해 저장하지 못했습니다. 다시 써 "
            "보려면 ‘네’라고 말씀해 주세요.",
            "en": f"Chapter {number}'s draft did not pass the checks, so nothing was saved. "
            "Say yes and I will try again.",
        }
    if state.stage == "review_scene" and state.chapter_failed:
        sid = state.scene_id or "현재 장면"
        return {
            "ko": f"장면 {sid}의 초안이 저장 기준을 통과하지 못해 저장하지 못했습니다. 다시 써 "
            "보려면 ‘네’라고 말씀해 주세요.",
            "en": f"Scene {sid}'s draft did not pass the checks, so nothing was saved. "
            "Say yes and I will try again.",
        }
    wrote = state.stage == "done" or "chapter" in done or "scene" in done or "rewrite" in done
    ko, en = _written_note(state, done) if wrote else ("", "")
    if state.stage == "review_chapter":
        ko_plan, en_plan = _plan_note(state, done)
        ko = f"{ko}\n\n{ko_plan}" if ko else ko_plan
        en = f"{en}\n\n{en_plan}" if en else en_plan
    if state.stage == "review_scene":
        ko_scene, en_scene = _scene_choices_note(state, done)
        ko = f"{ko}\n\n{ko_scene}" if ko else ko_scene
        en = f"{en}\n\n{en_scene}" if en else en_scene
    return {"ko": ko, "en": en}


#: What the outline stop has not made yet, said with every outline it shows (#2360:
#: mind_changer T4's reply showed a plan of chapter 1 the start had not made).
_NOT_YET_OUTLINE = (
    "줄거리는 아직 확정하지 않았고, 첫 장의 장면 계획도 아직 없습니다.",
    "The plot is not settled yet, and there is no scene plan for the first chapter yet.",
)


def _written_note(state: StartState, done: Sequence[str]) -> tuple[str, str]:
    """What the call wrote: the chapter, the single RPG scene, or the scenes rewritten, and
    what the check found."""
    number = state.chapter_written or 1
    count = len(state.chapter_scenes) or len(state.scenes_written)
    ko_label = "첫 장" if number == 1 else f"{number}장"
    en_label = "The first chapter" if number == 1 else f"Chapter {number}"
    if "scene" in done and state.scene_written_last:
        sid = state.scene_written_last
        ko_sid = _human_scene_ids(sid, "ko")
        en_sid = _human_scene_ids(sid, "en")
        ko = f"{ko_sid}을(를) 쓰고 시간순 KG 상태를 업데이트했습니다."
        en = f"Wrote {en_sid} and updated the chronological KG state."
    else:
        ko = f"{ko_label}을 썼습니다. 장면 {count}개를 저장했습니다."
        en = f"{en_label} is written: {count} scene(s) saved."
    first = number == 1
    if first and state.quick and state.genre_options and not state.genre_answered:
        ko = f"요청에 장르가 없어 {genre_label(state.brief.genre, 'ko')}로 정해 썼습니다. " + ko
        en = (
            f"The request named no genre, so it was written as "
            f"{genre_label(state.brief.genre, 'en')}. " + en
        )
    if state.stage == "done" and ("chapter" in done or "scene" in done):
        ko += " 개요의 마지막 장까지 모두 썼습니다."
        en += " Every chapter of the outline is written."
    if state.rewritten and (
        "rewrite" in done
        or (state.stage == "done" and "chapter" not in done and "scene" not in done)
    ):
        shown = [
            s for s in state.rewritten if not state.chapter_scenes or s in state.chapter_scenes
        ]
        ko = f"말씀하신 대로 장면 {len(shown)}개를 다시 써서 저장했습니다."
        en = f"{len(shown)} scene(s) rewritten as you asked, and saved."
    grown = growth_line(len(state.growth_now), scenes=True)
    if grown is not None and ("chapter" in done or "scene" in done or "rewrite" in done):
        ko += f" {grown['ko']}"
        en += f" {grown['en']}"
    if first and state.cast_assumed and "rewrite" not in done:
        ko += (
            " 제안한 인물과 장소를 승인된 것으로 보고 썼습니다. 제안은 아직 이야기 보기에서 "
            "승인을 기다리고 있으니, 확인하고 승인하거나 고쳐 주세요."
        )
        en += (
            " It was written treating the proposed cast and places as approved; the "
            "proposals still wait for your approval in the story's view."
        )
    ko_check, en_check = _continuity_note(state)
    ko_out = ko + ko_check
    en_out = en + en_check
    if "scene" in done and state.scene_written_last:
        shown_sids = [state.scene_written_last]
    elif state.rewritten and (
        "rewrite" in done
        or (state.stage == "done" and "chapter" not in done and "scene" not in done)
    ):
        shown_sids = [
            s for s in state.rewritten if not state.chapter_scenes or s in state.chapter_scenes
        ]
    else:
        shown_sids = list(state.chapter_scenes)
    for sid in shown_sids:
        url = state.scene_illustrations.get(sid)
        if url:
            ko_sid = _human_scene_ids(sid, "ko")
            en_sid = _human_scene_ids(sid, "en")
            ko_out += f"\n![{ko_sid}]({url})"
            en_out += f"\n![{en_sid}]({url})"
    return ko_out, en_out


def _scene_choices_note(state: StartState, done: Sequence[str]) -> tuple[str, str]:
    """The current world state (KG) and the 3 RPG choices for `state.scene_id`."""
    sid = state.scene_id or "ch01.s01"
    ko_sid = _human_scene_ids(sid, "ko")
    en_sid = _human_scene_ids(sid, "en")
    ko_lines: list[str] = [f"다음은 {ko_sid}의 진행 선택지입니다."]
    en_lines: list[str] = [f"Here are the RPG choices for {en_sid}."]
    if state.scene_entering_kg:
        ko_kg = _human_scene_ids(state.scene_entering_kg, "ko")
        en_kg = _human_scene_ids(state.scene_entering_kg, "en")
        ko_lines.append(f"[현재 월드 KG 상태]\n{ko_kg}")
        en_lines.append(f"[Current World State (KG)]\n{en_kg}")
    for c in state.scene_choices:
        num = c.get("number", 1)
        title = c.get("title", "")
        approach = c.get("approach", "")
        summary = c.get("summary", "")
        trans_raw = c.get("transitions")
        trans_list: list[object] = (
            cast("list[object]", trans_raw) if isinstance(trans_raw, list) else []
        )
        delta_parts: list[str] = []
        for tr in trans_list:
            if not isinstance(tr, dict):
                continue
            tr_dict = cast("dict[str, Any]", tr)
            subj = str(tr_dict.get("subject") or tr_dict.get("entity") or "")
            note = str(tr_dict.get("note") or "")
            if subj and note:
                delta_parts.append(f"{subj}: {note}")
            elif note:
                delta_parts.append(note)
        delta_str = f" (KG 변화: {'; '.join(delta_parts)})" if delta_parts else ""
        delta_en = f" (KG delta: {'; '.join(delta_parts)})" if delta_parts else ""
        ko_lines.append(f"{num}. [{approach}] {title} — {summary}{delta_str}")
        en_lines.append(f"{num}. [{approach}] {title} — {summary}{delta_en}")
    ko_lines.append(
        "원하시는 선택지 번호(1~3, ‘네’라고만 하시면 1번)를 고르시거나, 직접 원하는 행동이나 "
        "추가할 설정(KG)을 말씀해 주세요."
    )
    en_lines.append(
        "Choose an option by number (1–3; say yes for 1), or describe a custom action or "
        "KG update in your own words."
    )
    return "\n".join(ko_lines), "\n".join(en_lines)


def _plan_note(state: StartState, done: Sequence[str]) -> tuple[str, str]:
    """The next chapter's plan, shown before it is written, and what the person may say."""
    number = state.details_chapter or state.chapter_written + 1
    if "scene_edit" in done and state.details_scene_edits_ambiguous:
        ko = " ".join(
            f"「{a['title']}」 장면이 여러 개({a['ko']})라 어느 장면인지 알 수 없어 고치지 않았습니다."
            for a in state.details_scene_edits_ambiguous
        )
        en = " ".join(
            f"More than one written scene is called ‘{a['title']}’ ({a['en']}), so none was "
            "changed."
            for a in state.details_scene_edits_ambiguous
        )
        ko += f" 몇 장의 몇 번째 장면인지 말씀해 주세요. {number}장의 장면 계획은 그대로입니다."
        en += f" Tell me which one, by chapter and scene. Chapter {number}'s plan is as it was."
    elif "scene_edit" in done:
        failed = set(state.details_scene_edits_failed)
        saved = [e for e in state.details_scene_edits if e["id"] not in failed]
        unsaved = [e for e in state.details_scene_edits if e["id"] in failed]
        ko_titles = ", ".join(f"「{e['title']}」" for e in saved)
        ko_failed = ", ".join(f"「{e['title']}」" for e in unsaved)
        en_titles = ", ".join(f"‘{e['title']}’" for e in saved)
        en_failed = ", ".join(f"‘{e['title']}’" for e in unsaved)
        ko = (
            (f"말씀하신 대로 이미 쓴 장면({ko_titles})을 다시 써서 저장했습니다. " if saved else "")
            + (f"이미 쓴 장면({ko_failed})은 다시 쓰지 못해 그대로 두었습니다. " if unsaved else "")
            + f"{number}장의 장면 계획은 그대로입니다."
        )
        en = (
            (
                f"The scenes you asked about ({en_titles}) are written again and saved. "
                if saved
                else ""
            )
            + (
                f"The scenes {en_failed} could not be written again and are as they were. "
                if unsaved
                else ""
            )
            + f"Chapter {number}'s plan is as it was."
        )
    elif "details_revised" in done and state.details_revision_failed:
        ko = f"말씀하신 수정을 반영하지 못해 {number}장의 장면 계획을 그대로 두었습니다."
        en = f"Your change could not be made, so chapter {number}'s plan is as it was."
    elif "details_revised" in done:
        ko = f"말씀하신 대로 {number}장의 장면 계획을 고쳤습니다."
        en = f"Chapter {number}'s plan is revised as you asked."
    elif state.details_from_outline:
        ko = f"{number}장의 장면 계획을 새로 짜지 못해 개요에 있는 장면을 그대로 보여 드립니다."
        en = f"Chapter {number}'s plan could not be made, so here are its scenes from the outline."
    else:
        ko = f"다음은 {number}장의 장면 계획입니다."
        en = f"Here is the plan of chapter {number}."
    lines = "\n".join(
        f"{n}. {s.get('title', '')}"
        + (f" — {s['summary']}" if s.get("summary") and s["summary"] != s.get("title") else "")
        for n, s in enumerate(state.chapter_details, start=1)
    )
    ko += f"\n{lines}\n"
    en += f"\n{lines}\n"
    if state.details_new:
        names = ", ".join(str(c["name"]) for c in state.details_new)
        ko += (
            f"이 장에서 새로 나오는 인물은 {names}입니다. 이대로 쓰면 이 인물과 가족 관계를 "
            "설정집에 제안합니다.\n"
        )
        en += (
            f"New in this chapter: {names}. Written as planned, they and their family ties "
            "are proposed for the codex.\n"
        )
    # What the plan's stop has not done yet, said with every plan it shows (#2360: vague
    # T5's reply showed chapter 1's prose while nothing of it was written).
    ko += f"{number}장은 아직 쓰지 않았습니다. "
    en += f"Chapter {number} is not written yet. "
    if not state.quick:
        ko += (
            "바꾸고 싶은 곳이 있으면 말씀해 주세요. ‘네’라고 하시면 이대로 "
            f"{number}장을 쓰겠습니다."
        )
        en += f"Tell me what to change, or say yes and I will write chapter {number} as planned."
        return ko, en
    # Asked for at once: still one chapter per call, and the person is told how to go on.
    ko += (
        f"한 번에 한 장씩 씁니다. ‘계속’이라고 하시면 {number}장을 이어 쓰겠습니다. "
        "바꾸고 싶은 곳을 말씀하시면 고쳐서 바로 쓰겠습니다."
    )
    en += (
        f"One chapter is written per request. Say 'continue' and I will write chapter "
        f"{number}; tell me what to change and I will change it and write it."
    )
    return ko, en


#: A problem the outline check left, as the person reads it (`uclone_x_story.arc`).
_PROBLEM_KO: Mapping[str, str] = {
    "twist_early": "{n}장에서 반전이 너무 일찍 드러납니다",
    "twist_missing": "{n}장에서 반전이 드러나지 않습니다",
    "resolved_early": "{n}장에서 갈등이 너무 일찍 풀립니다",
    "repeats": "{n}장이 앞 장의 사건을 되풀이합니다",
    "no_twist": "반전이 정해지지 않았습니다",
    "missing_chapters": "{n}장부터 개요가 비어 있습니다",
}


#: The same problems in English, for the person and for the model's result (#2371: the
#: Writer was given the codes, and told the person the outline "resolves twist_missing").
_PROBLEM_EN: Mapping[str, str] = {
    "twist_early": "Chapter {n} reveals the twist too early",
    "twist_missing": "Chapter {n} does not reveal the twist",
    "resolved_early": "Chapter {n} resolves the conflict too early",
    "repeats": "Chapter {n} repeats the events of the chapter before",
    "no_twist": "No twist was settled",
    "missing_chapters": "The outline is empty from chapter {n}",
}


def _problem_en(problem: Mapping[str, Any]) -> str:
    return _PROBLEM_EN.get(str(problem.get("kind")), "Chapter {n} has something to fix").format(
        n=problem.get("chapter")
    )


def _plain_check(check: Any) -> Any:
    """`check` (the outline check as recorded) with each problem's code replaced by its
    words, for the tool's result: the record keeps the codes (#2371)."""
    if isinstance(check, dict):
        found = cast("dict[str, Any]", check)
        if "kind" in found and found.get("kind") in _PROBLEM_EN:
            rest = {k: v for k, v in found.items() if k != "kind"}
            return {"problem": _problem_en(found), **rest}
        return {k: _plain_check(v) for k, v in found.items()}
    if isinstance(check, list):
        return [_plain_check(v) for v in cast("list[Any]", check)]
    return check


#: What a result says first when 'go_on' came with a change that was made (#2371).
_GO_ON_NOT_TAKEN = (
    "Nothing was settled and the start did not move on: 'changes' came with 'go_on', so the "
    "change was made instead. Tell the person what was changed and ask again; do not say "
    "anything was settled or approved. "
)
#: What a call that made a change records as done.
_CHANGED = frozenset({"premises", "outline_revised", "details_revised", "scene_edit"})
#: What a call that moved the start on records as done: settled the plot, wrote a
#: chapter, wrote an RPG scene, or rewrote flagged scenes. A story asked for at once revises a plan and writes
#: its chapter in one call (#2372 review).
_MOVED = frozenset({"plot", "chapter", "scene", "rewrite"})


def _not_settled(result: dict[str, Any], params: StoryStartParams) -> dict[str, Any]:
    """`result`, said plainly when 'go_on' came with a change the call made and the start
    did not move on: the change wins (#2355), so nothing was settled or taken, and the
    result says that first. A live
    Writer sent both, the outline was revised, and it told the person the plot was settled
    (#2370)."""
    done = set(result.get("done_now") or ())
    if params.go_on and (params.changes or "").strip() and done & _CHANGED and not done & _MOVED:
        result["go_on_not_taken"] = True
        result["next"] = _GO_ON_NOT_TAKEN + str(result.get("next") or "")
    return result


def _arc_note(state: StartState) -> tuple[str, str]:
    """Where the outline turns, and what its check still found, in a sentence or two."""
    if state.arc is None:
        return "", ""
    twist, climax = state.arc["twist_chapter"], state.arc["climax_chapter"]
    ko = f" 반전은 {twist}장에서 처음 드러나고, 갈등은 {climax}장에서 결판나도록 짰습니다."
    en = f" The twist is revealed in chapter {twist} and the conflict decided in chapter {climax}."
    raw = (state.outline_check or {}).get("final")
    left = [cast(dict[str, Any], p) for p in cast(list[object], raw or []) if isinstance(p, dict)]
    if left:
        said = ", ".join(
            _PROBLEM_KO.get(str(p["kind"]), "{n}장에 고칠 곳이 있습니다").format(n=p["chapter"])
            for p in left
        )
        ko += f" 다만 개요를 점검해 보니 아직 고칠 곳이 있습니다: {said}."
        en += " The outline check still found: " + ", ".join(_problem_en(p) for p in left) + "."
    return ko, en


def _continuity_note(state: StartState) -> tuple[str, str]:
    """What the check after writing found, with each quote, and the offer to rewrite.

    Only the chapter written last is reported: earlier chapters' findings were shown when
    they were written.
    """
    ko = en = ""
    scenes = set(state.chapter_scenes)
    shown = _shown_findings(state)
    checked = [s for s in state.continuity_checked if not scenes or s in scenes]
    unread = [s for s in state.continuity_unread if not scenes or s in scenes]
    if shown:
        lines = "\n".join(f"- {f['note']}" for f in shown)
        count = len(shown)
        ko += (
            f"\n\n쓴 장면을 설정집과 앞 장면, 장면 계획에 대조해 보니 맞지 않는 곳이 {count}곳 "
            f"있습니다.\n{lines}\n"
            "이 장면을 다시 쓸까요? 다시 써 달라고 하시면 그 장면만 고쳐 쓰겠습니다. "
            "이야기에서 일부러 그렇게 쓴 것이라면 그대로 두셔도 됩니다."
        )
        en += (
            f"\n\nChecked against the codex and the scenes' plan, {count} line(s) "
            f"disagree:\n{lines}\n"
            "Shall I rewrite those scenes? Ask me to and I will rewrite only them; if it is "
            "meant that way, leave it."
        )
    elif checked and unread:
        # Some scenes were never read, so "nothing disagreed" holds only for the rest.
        ko += (
            f"\n\n대조한 장면 {len(checked)}개에서는 인물의 생사, 성별, 가족 관계가 설정집과 "
            "어긋나는 곳을 찾지 못했습니다."
        )
        en += f"\n\nIn the {len(checked)} scene(s) checked against the codex, nothing disagreed."
    elif checked:
        ko += (
            "\n\n쓴 장면의 인물 생사, 성별, 가족 관계를 설정집과 대조했고, 어긋나는 곳은 "
            "찾지 못했습니다."
        )
        en += "\n\nThe scenes' characters were checked against the codex; nothing disagreed."
    if unread:
        ko += f" 장면 {len(unread)}개는 설정집과 대조하지 못했습니다."
        en += f" {len(unread)} scene(s) could not be checked."
    plan_unread = [s for s in state.plan_unread if not scenes or s in scenes]
    if not shown and scenes and len(plan_unread) < len(scenes):
        ko += " 장면 계획에 있던 인물과 사건, 반전을 드러낼 때와도 어긋나는 곳을 찾지 못했습니다."
        en += " Nothing disagreed with the scenes' plan or the twist's timing either."
    if plan_unread:
        ko += f" 장면 {len(plan_unread)}개는 장면 계획과 대조하지 못했습니다."
        en += f" {len(plan_unread)} scene(s) could not be checked against their plan."
    return ko, en


#: The headings of a story's style notes, in the story's language.
_STYLE_HEADINGS: Mapping[Language, tuple[str, str, str, str]] = {
    "ko": ("톤", "독자", "분량", "피할 것"),
    "en": ("Tone", "Audience", "Length", "Avoid"),
}


def _style_notes(brief: Brief) -> str:
    tone, audience, length, avoid = _STYLE_HEADINGS[brief.language]
    parts = [f"{tone}: {brief.tone}", f"{audience}: {brief.audience}", f"{length}: {brief.length}"]
    if brief.avoid:
        parts.append(f"{avoid}: " + ", ".join(brief.avoid))
    return "; ".join(parts)


def _shown_findings(state: StartState) -> list[dict[str, str]]:
    """The check's notes on the chapter written last: what the person is shown, and what
    a rewrite they ask for rewrites. Notes on earlier chapters were shown with them."""
    if not state.chapter_scenes:
        return list(state.continuity)
    return [f for f in state.continuity if f.get("scene_id") in state.chapter_scenes]


__all__ = [
    "_CAST_FAILED",
    "_CAST_FAILED_NEXT",
    "_CHANGED",
    "_CONTINUITY_NEXT",
    "_GO_ON_NOT_TAKEN",
    "_KEPT_CHANGE_NOW",
    "_MOVED",
    "_NEXT",
    "_NOT_YET_OUTLINE",
    "_PROBLEM_EN",
    "_PROBLEM_KO",
    "_SAME_TURN",
    "_SCENE_EDIT_AMBIGUOUS_NEXT",
    "_SCENE_EDIT_NEXT",
    "_STOP_WORDS",
    "_STYLE_HEADINGS",
    "_WAITING_NEXT",
    "_arc_note",
    "_continuity_note",
    "_kept_note",
    "_listed_genres",
    "_next_of",
    "_not_settled",
    "_outline_lines",
    "_plain_check",
    "_plan_note",
    "_plot_note",
    "_problem_en",
    "_reply_note",
    "_scene_labels",
    "_shown_findings",
    "_stage_note",
    "_style_notes",
    "_waiting_scene_words",
    "_waiting_state",
    "_where",
    "_written_note",
    "by_hand_refusal",
    "flow_stop",
    "start_waiting",
    "stop_words",
]
