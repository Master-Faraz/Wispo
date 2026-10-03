from whisperflow.hotkey.linux import CTRL_CODES, META_CODES, chord_satisfied
from whisperflow.hotkey.portal import describe_binding


def test_ctrl_and_super_must_both_be_down():
    groups = (CTRL_CODES, META_CODES)
    assert not chord_satisfied({29}, groups)
    assert not chord_satisfied({125}, groups)
    assert chord_satisfied({29, 125}, groups)
    assert chord_satisfied({97, 126}, groups)


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
