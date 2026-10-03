from whisperflow.config import DictionaryEntry
from whisperflow.processing.dictionary import DictionaryProcessor


def _processor(*rows: tuple[str, str] | tuple[str, str, bool]) -> DictionaryProcessor:
    entries = []
    for row in rows:
        enabled = True if len(row) == 2 else row[2]
        entries.append(DictionaryEntry(row[0], row[1], enabled))
    return DictionaryProcessor(entries)


def test_exact_replacement_keeps_the_configured_casing():
    text = _processor(("github", "GitHub")).process("I use github daily")
    assert text == "I use GitHub daily"


def test_multiple_replacements():
    text = _processor(("github", "GitHub"), ("javascript", "JavaScript")).process(
        "github and javascript"
    )
    assert text == "GitHub and JavaScript"


def test_longest_phrase_wins():
    text = _processor(
        ("pull", "PULL"),
        ("open pull request", "Open Pull Request"),
        ("pull request", "PR"),
    ).process("open pull request")
    assert text == "Open Pull Request"


def test_shorter_entry_does_not_rewrite_part_of_a_word():
    text = _processor(("git", "Git")).process("github")
    assert text == "github"


def test_punctuation_beside_a_phrase_stays():
    text = _processor(("github", "GitHub")).process("See github.")
    assert text == "See GitHub."


def test_match_is_case_insensitive():
    text = _processor(("github", "GitHub")).process("GitHub")
    assert text == "GitHub"


def test_empty_dictionary_returns_the_transcript():
    assert DictionaryProcessor(()).process("leave this") == "leave this"


def test_disabled_entry_is_skipped():
    text = _processor(("github", "GitHub", False)).process("github")
    assert text == "github"
