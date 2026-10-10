from whisperflow.processing.polish import polish_transcript


def test_polish_empty():
    assert polish_transcript("") == ""
    assert polish_transcript("   ") == ""


def test_polish_capitalization():
    assert polish_transcript("hello world") == "Hello world"
    assert polish_transcript("already capitalized.") == "Already capitalized."
    assert polish_transcript('"quoted string"') == '"Quoted string"'


def test_polish_strips_artifacts():
    assert polish_transcript("Thank you [applause] for watching.") == "Thank you for watching."
    assert polish_transcript("(music) let's code") == "Let's code"


def test_polish_no_capitalize_when_disabled():
    assert polish_transcript("hello world", auto_capitalize=False) == "hello world"
