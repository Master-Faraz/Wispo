from whisperflow.config import DictionaryEntry
from whisperflow.processing.prompt import DEFAULT_STYLE_ANCHOR, build_initial_prompt


def test_build_prompt_empty():
    prompt = build_initial_prompt(None)
    assert prompt == DEFAULT_STYLE_ANCHOR


def test_build_prompt_with_dictionary():
    entries = [
        DictionaryEntry(phrase="pie torch", replacement="PyTorch", enabled=True),
        DictionaryEntry(phrase="git hub", replacement="GitHub", enabled=True),
        DictionaryEntry(phrase="ignored", replacement="DisabledWord", enabled=False),
    ]
    prompt = build_initial_prompt(entries)
    assert "PyTorch" in prompt
    assert "GitHub" in prompt
    assert "DisabledWord" not in prompt
    assert "Vocabulary: PyTorch, GitHub." in prompt


def test_build_prompt_no_style():
    entries = [DictionaryEntry(phrase="git hub", replacement="GitHub", enabled=True)]
    prompt = build_initial_prompt(entries, include_style=False)
    assert prompt == "Vocabulary: GitHub."
