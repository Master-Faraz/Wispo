from whisperflow.audio.sources import choose_source, parse_wpctl_sources
from whisperflow.hotkey.linux import _parse_key_bits

STATUS = """
Audio
 ├─ Devices:
 │      48. Arrow Lake cAVS
 ├─ Sinks:
 │      62. Arrow Lake cAVS Speaker
 ├─ Sources:
 │      63. Arrow Lake cAVS Stereo Microphone   [vol: 1.00]
 │      64. Arrow Lake cAVS Digital Microphone   [vol: 0.84]
 ├─ Filters:
 │     118. bluez_output.example
Video
 ├─ Sources:
 │  *   83. Integrated Camera (V4L2)
"""


def test_audio_sources_ignore_the_camera_section():
    sources = parse_wpctl_sources(STATUS)
    assert sources == [
        (63, "Arrow Lake cAVS Stereo Microphone"),
        (64, "Arrow Lake cAVS Digital Microphone"),
    ]


def test_digital_microphone_is_preferred():
    assert choose_source(parse_wpctl_sources(STATUS)) == (
        64,
        "Arrow Lake cAVS Digital Microphone",
    )


def test_keyboard_bitmap_marks_right_ctrl():
    bits = _parse_key_bits("402000007 ff803078f800d001 feffffdfffcfffff fffffffffffffffe")
    assert bits & (1 << 97)
    assert bits & (1 << 0) == 0
