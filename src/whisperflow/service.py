"""Background WhisperFlow.

    uv run python -m whisperflow.service

Hold Ctrl+Super, speak, and release. The finished sentence is typed into
the focused window. The app icon stays in the dock. This machine hides the
top bar, so there is no system tray to put an icon in.

Closing the window leaves WhisperFlow running. Click the dock icon to open
it again. Quit stops it.

Starting this again while it is already running opens the window instead of
a second listener.
"""

from __future__ import annotations

import logging
import subprocess
import sys
import threading
import time
from pathlib import Path

from jeepney import DBusAddress, new_method_call
from jeepney.bus_messages import message_bus
from jeepney.io.blocking import open_dbus_connection

from whisperflow.config import load_config, project_root
from whisperflow.desktop import install_desktop_entry
from whisperflow.injection.portal import PortalError, _checked

_BUS_NAME = "org.whisperflow.Service"
_SERVICE_PATH = "/org/whisperflow/Service"
_SERVICE = "org.whisperflow.Service"


def main(argv: list[str] | None = None) -> int:
    args = list(sys.argv[1:] if argv is None else argv)
    _configure_log()
    if "--quit" in args:
        return 0 if _call("Quit") else 1
    if _already_running():
        _call("OpenSettings")
        return 0
    install_desktop_entry(sys.executable, str(project_root()))
    return _run()


def _run() -> int:
    from whisperflow.desktop import APP_ID
    from whisperflow.settings import SettingsApp, _load_gtk
    from whisperflow.status import StatusIcon

    gtk = _load_gtk()
    from gi.repository import GLib

    GLib.set_prgname(APP_ID)
    stop_event = threading.Event()
    ui: dict = {}

    def request_quit() -> None:
        stop_event.set()

    def open_settings() -> None:
        def present() -> bool:
            window = ui.get("window")
            if window is not None:
                window.present()
                window.unminimize()
            return False

        GLib.idle_add(present)

    status = StatusIcon(on_settings=open_settings, on_quit=request_quit)
    try:
        status.start()
    except RuntimeError as exc:
        logging.error("%s", exc)
        _call("OpenSettings")
        return 0
    except Exception:
        logging.exception("status icon failed")
        status.stop()
        status = None

    application = gtk.Application(application_id=APP_ID)
    settings = SettingsApp(gtk, on_quit=request_quit)

    def on_activate(app) -> None:
        window = settings.build(app)
        ui["window"] = window

        def keep_running(*_args) -> bool:
            # The top bar is hidden on this machine, so the dock icon is the
            # only visible control. Closing the window must not exit.
            window.minimize()
            return True

        window.connect("close-request", keep_running)
        app.hold()
        window.present()

    application.connect("activate", on_activate)
    worker = threading.Thread(
        target=_listen,
        args=(stop_event, status),
        name="whisperflow-listen",
    )
    worker.start()
    try:
        application.run(None)
    finally:
        stop_event.set()
        worker.join(timeout=5)
        if status is not None:
            status.stop()
    return 0


def _listen(stop_event: threading.Event, status) -> None:
    from gi.repository import GLib

    from whisperflow.application import TranscriptionApp
    from whisperflow.cli import _inject_final
    from whisperflow.hotkey.linux import HoldChord, OverviewKey
    from whisperflow.injection.portal import PortalKeyboard
    from whisperflow.processing.dictionary import DictionaryProcessor

    def finish() -> None:
        GLib.idle_add(_quit_gtk)

    config = load_config()
    app = TranscriptionApp(config)
    keyboard = PortalKeyboard()
    dictionary = DictionaryProcessor(config.dictionary)
    overview = OverviewKey()
    hold = None
    recording = False
    try:
        logging.info("opening the input session")
        keyboard.open()
        if "super" in config.hotkey.keys:
            try:
                overview.start()
            except (OSError, subprocess.CalledProcessError):
                logging.exception("could not move the overview key aside")
                overview = None
        else:
            overview = None
        if status is not None:
            status.set_state("processing")
        logging.info("loading the model")
        app.load_model()
        logging.info(
            "model %s on %s (%s)",
            config.asr.model,
            app.device.name,
            app.device.label,
        )
        hold = HoldChord(config.hotkey.keys)
        hold.start()
        if status is not None:
            status.set_state("idle")
        logging.info("ready")
        while not stop_event.is_set():
            if hold.held.is_set():
                if not recording:
                    app.start_recording()
                    recording = True
                    if status is not None:
                        status.set_state("recording")
                app.read_once(force_speech=True)
            elif recording:
                if status is not None:
                    status.set_state("processing")
                text = app.stop_recording()
                recording = False
                _inject_final(keyboard, text, dictionary)
                if status is not None:
                    status.set_state("idle")
            else:
                time.sleep(0.02)
    except PortalError:
        logging.exception("input session failed")
    except KeyboardInterrupt:
        return
    finally:
        if recording:
            app.stop_recording()
        if hold is not None:
            hold.stop()
        app.capture.stop()
        if overview is not None:
            overview.stop()
        keyboard.close()
        finish()


def _quit_gtk() -> None:
    from gi.repository import Gtk

    app = Gtk.Application.get_default()
    if app is not None:
        app.quit()


def _already_running() -> bool:
    conn = open_dbus_connection(bus="SESSION")
    try:
        reply = _checked(conn, message_bus.NameHasOwner(_BUS_NAME), timeout=2)
    except PortalError:
        return False
    finally:
        conn.close()
    return bool(reply[0])


def _call(method: str) -> bool:
    conn = open_dbus_connection(bus="SESSION")
    try:
        address = DBusAddress(_SERVICE_PATH, bus_name=_BUS_NAME, interface=_SERVICE)
        _checked(conn, new_method_call(address, method), timeout=2)
        return True
    except PortalError:
        logging.error("WhisperFlow is not running")
        return False
    finally:
        conn.close()


def _configure_log() -> None:
    folder = Path.home() / ".local" / "state" / "whisperflow"
    folder.mkdir(parents=True, exist_ok=True)
    handlers: list[logging.Handler] = [logging.FileHandler(folder / "service.log")]
    if sys.stderr.isatty():
        handlers.append(logging.StreamHandler())
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s", handlers=handlers)


if __name__ == "__main__":
    raise SystemExit(main())
