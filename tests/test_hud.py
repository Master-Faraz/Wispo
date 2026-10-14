from wispo.hud import FloatingPillHUD
from wispo.settings import _load_gtk


def test_hud_initializes_and_transitions_states():
    gtk = _load_gtk()
    hud = FloatingPillHUD(gtk)
    assert hud._state == "idle"

    hud.set_audio_level(0.2)
    assert hud._target_level > 0.0

    hud.start_listening()
    assert hud._state == "listening"
    assert hud._target_anim == 1.0

    hud.start_processing()
    assert hud._state == "processing"

    hud.finish(wait_hide=False)
    assert hud._state == "closing"

    hud.close()
