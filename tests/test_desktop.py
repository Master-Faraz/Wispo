from whisperflow.desktop import desktop_entry_text
from whisperflow.status import ICON_NAMES


def test_desktop_entry_starts_the_background_service():
    text = desktop_entry_text("/home/faraz/.venv/bin/python", "/home/faraz/Documents/Codes/my-wisper")
    assert "Exec=/home/faraz/.venv/bin/python -m whisperflow.service" in text
    assert "Terminal=false" in text
    assert "Name=WhisperFlow" in text
    assert "StartupWMClass=com.whisperflow.WhisperFlow" in text


def test_status_icon_changes_with_the_recording_state():
    assert ICON_NAMES["idle"] != ICON_NAMES["recording"]
    assert ICON_NAMES["recording"] != ICON_NAMES["processing"]
