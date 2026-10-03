from whisperflow.hotkey.linux import (
    CTRL_CODES,
    META_CODES,
    chord_from_codes,
    chord_satisfied,
    codes_for,
    format_chord,
)
from whisperflow.hotkey.portal import describe_binding


def test_ctrl_and_super_must_both_be_down():
    groups = (CTRL_CODES, META_CODES)
    assert not chord_satisfied({29}, groups)
    assert not chord_satisfied({125}, groups)
    assert chord_satisfied({29, 125}, groups)
    assert chord_satisfied({97, 126}, groups)


def test_a_saved_shortcut_uses_either_side_of_a_modifier():
    assert format_chord(("ctrl", "super")) == "Ctrl+Super"
    assert format_chord(("alt", "space")) == "Alt+Space"
    assert chord_from_codes({97, 126}) == ("ctrl", "super")
    assert chord_satisfied({100, 57}, (codes_for("alt"), codes_for("space")))


def test_binding_description_comes_from_the_portal():
    text = describe_binding(
        {
            "shortcuts": [
                (
                    "push-to-talk",
                    {"trigger_description": ("s", "Ctrl+Super"), "description": ("s", "Hold")},
                )
            ]
        }
    )
    assert text == "Ctrl+Super"


def test_other_shortcuts_are_ignored():
    assert describe_binding({"shortcuts": [("something-else", {})]}) == ""
