from wispo.injection.keymap import KEY_ENTER, KEY_LEFTSHIFT, KEY_SPACE, UnsupportedCharacter, key_events


def _presses(events):
    return [code for code, state in events if state == 1 and code != KEY_LEFTSHIFT]


def test_letters_use_shift_only_while_needed():
    events = key_events("Ab")
    assert events[0] == (KEY_LEFTSHIFT, 1)
    # 'b' is unshifted, so Shift goes up before that key, and the last event
    # is 'b' being released rather than another Shift event.
    shift_release = events.index((KEY_LEFTSHIFT, 0))
    b_press = next(
        i
        for i, (code, state) in enumerate(events)
        if state == 1 and code != KEY_LEFTSHIFT and i > shift_release
    )
    assert shift_release < b_press
    assert events[-1][1] == 0


def test_phrase_from_the_gpu_run_is_typeable():
    text = "So this is the output of the GPU."
    events = key_events(text)
    assert _presses(events)[0] != _presses(events)[1]
    assert events[-1][0] != KEY_LEFTSHIFT or events[-1][1] == 0
    assert len(_presses(events)) == len(text)


def test_space_and_newline():
    events = key_events("a b\n")
    pressed = _presses(events)
    assert KEY_SPACE in pressed
    assert pressed[-1] == KEY_ENTER


def test_unknown_character_is_rejected():
    try:
        key_events("café")
    except UnsupportedCharacter as exc:
        assert exc.char == "é"
    else:
        raise AssertionError("expected UnsupportedCharacter")
