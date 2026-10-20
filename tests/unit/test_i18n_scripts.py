"""`uclone_x.i18n.scripts`: a letter's script from the character itself, the same for every
language (#2301, #2303).

What these pin: a word glued across scripts is found whatever the scripts are ("죄孽",
"Привет" in a Korean text, a Latin word in one); width forms read as their script; and a
text is cut into sentences by the marks' Unicode names, so a run of Chinese with no spaces
is cut as English with spaces is.
"""

from __future__ import annotations

from uclone_x.i18n.scripts import letter_scripts, sentences, words_outside


def test_a_word_with_a_letter_outside_the_scripts_is_found_whatever_the_script() -> None:
    """Killed by: src/uclone_x/i18n/scripts.py :: return _ALIASES.get(words[0], words[0])
    Becomes: return words[0]
    """
    text = "그는 죄孽을 숨겼다. 그녀는 Привет 하고 웃었고, 직觉적으로 eclipsed 노래를 들었다."
    assert words_outside(text, {"HANGUL"}) == ["죄孽을", "Привет", "직觉적으로", "eclipsed"]
    assert letter_scripts("孽") == {"HAN"}
    # Digits, punctuation and Hangul jamo are in every Korean text.
    assert words_outside("3장, ‘ㅋㅋ’ — 1,000원!", {"HANGUL"}) == []


def test_a_width_form_reads_as_its_script() -> None:
    """Killed by: src/uclone_x/i18n/scripts.py :: while words and words[0] in _FORMS:
    Becomes: while False:
    """
    assert letter_scripts("ＳＦ") == {"LATIN"}
    assert words_outside("ＳＦ 이야기", {"HANGUL", "LATIN"}) == []


def test_sentences_are_cut_by_their_marks_in_any_script() -> None:
    """Killed by: src/uclone_x/i18n/scripts.py :: if end == size or line[end].isspace() or _spaceless(line[end - 1]):
    Becomes: if end == size or line[end].isspace():
    """
    assert sentences("他走了。她笑了！你呢？") == ["他走了。", "她笑了！", "你呢？"]
    assert sentences('하린은 웃었다. "정말?" 3.5점이었다…\nShe left. Then') == [
        "하린은 웃었다.",
        '"정말?"',
        "3.5점이었다…",
        "She left.",
        "Then",
    ]
