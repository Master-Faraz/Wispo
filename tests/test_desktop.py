from wispo.desktop import desktop_entry_text
from wispo.status import ICON_NAMES


def test_desktop_entry_starts_the_background_service():
    text = desktop_entry_text("/home/faraz/.venv/bin/python", "/home/faraz/Documents/Codes/my-wisper")
    assert "Exec=/home/faraz/.venv/bin/python -m wispo.service" in text
    assert "Terminal=false" in text
    assert "Name=Wispo" in text
    assert "StartupWMClass=com.wispo.Wispo" in text


def test_desktop_entry_can_use_the_packaged_launcher():
    text = desktop_entry_text(
        "/usr/bin/python3",
        "/opt/wispo",
        command="/usr/bin/wispo",
    )

    assert "Exec=/usr/bin/wispo" in text
    assert "-m wispo.service" not in text


def test_status_icon_changes_with_the_recording_state():
    assert ICON_NAMES["idle"] != ICON_NAMES["recording"]
    assert ICON_NAMES["recording"] != ICON_NAMES["processing"]
