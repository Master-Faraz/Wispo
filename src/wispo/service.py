"""Background Wispo.

    uv run python -m wispo.service

Hold Ctrl+Super, speak, and release. The finished sentence is typed into
the focused window. The app icon stays in the dock. This machine hides the
top bar, so there is no system tray to put an icon in.

Closing the window leaves Wispo running. Click the dock icon to open
it again. Quit stops it.

Starting this again while it is already running opens the window instead of
a second listener.
"""

from __future__ import annotations

import logging
import queue
import subprocess
import sys
import threading
import time
from pathlib import Path

from jeepney import DBusAddress, new_method_call
from jeepney.bus_messages import message_bus
from jeepney.io.blocking import open_dbus_connection

from wispo.config import load_config, project_root
from wispo.desktop import install_desktop_entry
from wispo.injection.portal import PortalError, _checked

_BUS_NAME = "org.wispo.Service"
_SERVICE_PATH = "/org/wispo/Service"
_SERVICE = "org.wispo.Service"


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
    from wispo.desktop import APP_ID
    from wispo.settings import SettingsApp, _load_gtk
    from wispo.status import StatusIcon

    gtk = _load_gtk()
    from gi.repository import GLib

    GLib.set_prgname(APP_ID)
    stop_event = threading.Event()
    reload_queue: queue.Queue = queue.Queue()
    ui: dict = {}

    def request_quit() -> None:
        stop_event.set()
        GLib.idle_add(application.quit)

    def open_settings() -> None:
        def present() -> bool:
            window = ui.get("window")
            if window is not None:
                window.set_visible(True)
                window.unminimize()
                window.present()
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

    from wispo.hud import FloatingPillHUD

    application = gtk.Application(application_id=APP_ID)
    settings = SettingsApp(gtk, on_quit=request_quit, on_save_config=reload_queue.put)
    hud = FloatingPillHUD(gtk)

    def on_activate(app) -> None:
        window = ui.get("window")
        if window is not None:
            window.set_visible(True)
            window.unminimize()
            window.present()
            return

        window = settings.build(app)
        ui["window"] = window

        def keep_running(*_args) -> bool:
            window.set_visible(False)
            return True

        window.connect("close-request", keep_running)
        app.hold()
        window.present()

    application.connect("activate", on_activate)
    worker = threading.Thread(
        target=_listen,
        args=(stop_event, status, hud, settings, reload_queue),
        name="wispo-listen",
    )
    worker.start()
    try:
        application.run(None)
    finally:
        stop_event.set()
        worker.join(timeout=5)
        if status is not None:
            status.stop()
        if hud is not None:
            hud.close()
    return 0


def _listen(
    stop_event: threading.Event,
    status,
    hud=None,
    settings=None,
    reload_queue: queue.Queue | None = None,
) -> None:
    from gi.repository import GLib

    from wispo.application import TranscriptionApp
    from wispo.cli import _inject_final
    from wispo.hotkey.linux import HoldChord, OverviewKey
    from wispo.injection.portal import PortalKeyboard
    from wispo.processing.dictionary import DictionaryProcessor

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
        hotkey_keys = {k.lower() for k in config.hotkey.keys}
        if "super" in hotkey_keys or "copilot" in hotkey_keys:
            try:
                overview.start()
            except (OSError, subprocess.CalledProcessError):
                logging.exception("could not move the overview key aside")
                overview = None
        else:
            overview = None
        if status is not None:
            status.set_state("processing")
        logging.info("loading and warming up the model")

        stop_ticker = threading.Event()

        def _progress_ticker():
            stages = [
                (25, 1.2, "Loading Model...", "Reading weights into memory"),
                (45, 2.0, "Compiling GPU Shaders...", "Initializing Level Zero backend"),
                (65, 2.5, "Compiling GPU Shaders...", "JIT-compiling SYCL kernels"),
                (82, 2.0, "Warming Up Pipeline...", "Running dummy inference pass"),
                (94, 1.5, "Priming Audio Pipeline...", "Testing microphone capture"),
            ]
            pct = 10
            if settings is not None:
                settings.set_loading_progress(pct, "Starting Engine...", "Initializing runtime")
            for target_pct, duration, title, subtitle in stages:
                steps = 10
                step_sleep = duration / steps
                increment = (target_pct - pct) / steps
                for _ in range(steps):
                    if stop_ticker.is_set():
                        return
                    pct += increment
                    if settings is not None:
                        settings.set_loading_progress(int(pct), title, subtitle)
                    time.sleep(step_sleep)

        ticker_thread = threading.Thread(target=_progress_ticker, daemon=True, name="startup-progress")
        ticker_thread.start()

        try:
            app.load_model(warmup=True)
        finally:
            stop_ticker.set()
            ticker_thread.join(timeout=0.5)

        if settings is not None:
            settings.set_loading_progress(100, "Engine Ready", f"{app.device.name} • {config.asr.model}")
            time.sleep(0.15)
            settings.set_model_loaded(f"{app.device.name} • {config.asr.model}")

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
            try:
                if reload_queue is not None and not recording and not reload_queue.empty():
                    pending_cfg = None
                    while not reload_queue.empty():
                        try:
                            pending_cfg = reload_queue.get_nowait()
                        except queue.Empty:
                            break
                    if pending_cfg is not None:
                        new_config = pending_cfg

                        # 1. Dictionary update
                        if new_config.dictionary != config.dictionary:
                            dictionary = DictionaryProcessor(new_config.dictionary)
                            logging.info("dictionary updated (%d entries)", len(new_config.dictionary))

                        # 2. VAD update
                        if new_config.vad != config.vad:
                            app.update_vad(
                                rms_threshold=new_config.vad.rms_threshold,
                                noise_multiplier=new_config.vad.noise_multiplier,
                                hangover_ms=new_config.vad.hangover_ms,
                                highpass_hz=new_config.vad.highpass_hz,
                            )
                            logging.info("VAD parameters updated")

                        # 3. Audio device update
                        if new_config.audio.device != config.audio.device:
                            try:
                                app.update_audio_device(new_config.audio.device)
                                logging.info("audio input device updated to %s", new_config.audio.device)
                            except Exception:
                                logging.exception("failed to update audio input device")

                        # 4. Hotkey update
                        new_chord = tuple(k.lower() for k in new_config.hotkey.keys)
                        old_chord = tuple(k.lower() for k in config.hotkey.keys)
                        if new_chord != old_chord:
                            try:
                                if hold is not None:
                                    hold.stop()
                                new_hotkey_keys = set(new_chord)
                                old_hotkey_keys = set(old_chord)
                                needs_overview = "super" in new_hotkey_keys or "copilot" in new_hotkey_keys
                                had_overview = "super" in old_hotkey_keys or "copilot" in old_hotkey_keys
                                if needs_overview and not had_overview:
                                    if overview is None:
                                        overview = OverviewKey()
                                    try:
                                        overview.start()
                                    except Exception:
                                        overview = None
                                elif not needs_overview and had_overview:
                                    if overview is not None:
                                        try:
                                            overview.stop()
                                        except Exception:
                                            pass
                                        overview = None
                                hold = HoldChord(new_config.hotkey.keys)
                                hold.start()
                                logging.info("hotkey updated to %s", new_config.hotkey.keys)
                            except Exception:
                                logging.exception("failed to update hotkey chord")

                        # 5. Model update
                        if (
                            new_config.asr.model != config.asr.model
                            or new_config.asr.model_path != config.asr.model_path
                        ):
                            if status is not None:
                                status.set_state("processing")
                            try:
                                def _report_progress(pct: int, title: str, subtitle: str) -> None:
                                    if settings is not None:
                                        settings.set_loading_progress(pct, title, subtitle)

                                app.reload_model(
                                    model_name=new_config.asr.model,
                                    model_path=new_config.resolve(new_config.asr.model_path),
                                    on_progress=_report_progress,
                                )
                                if settings is not None:
                                    settings.set_model_loaded(f"{app.device.name} • {new_config.asr.model}")
                                logging.info("model reloaded: %s", new_config.asr.model)
                            except Exception as exc:
                                logging.exception("failed to reload model %s", new_config.asr.model)
                                if settings is not None:
                                    settings._idle(
                                        settings._status_label.set_markup,
                                        f"<span font_size='8.5pt' color='#f87171'>● Failed to load model: {exc}</span>",
                                    )
                                    settings.set_model_loaded(f"{app.device.name} • {config.asr.model}")
                            finally:
                                if status is not None:
                                    status.set_state("idle")

                        config = new_config
                if hold.held.is_set():
                    if not recording:
                        logging.info("hotkey down: recording started")
                        app.start_recording()
                        recording = True
                        if status is not None:
                            status.set_state("recording")
                        if hud is not None:
                            hud.start_listening()
                    app.read_once(force_speech=True, transcribe=False)
                    if hud is not None:
                        hud.set_audio_level(app.last_audio_level)
                elif recording:
                    logging.info("hotkey up: stopping recording and finalizing")
                    if status is not None:
                        status.set_state("processing")
                    if hud is not None:
                        hud.start_processing()
                    text = app.stop_recording()
                    recording = False
                    logging.info("transcribed text: %r", text)
                    if status is not None:
                        status.set_state("idle")
                    if hud is not None:
                        hud.finish()
                    _inject_final(keyboard, text, dictionary)
                else:
                    time.sleep(0.02)
            except Exception:
                logging.exception("error during recording cycle; recovering state")
                if recording:
                    try:
                        app.stop_recording()
                    except Exception:
                        pass
                    recording = False
                if status is not None:
                    status.set_state("idle")
                if hud is not None:
                    hud.finish()
                time.sleep(0.2)
    except PortalError:
        logging.exception("input session failed")
    except KeyboardInterrupt:
        return
    finally:
        if recording:
            app.stop_recording()
        if hud is not None:
            hud.finish()
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
        logging.error("Wispo is not running")
        return False
    finally:
        conn.close()


def _configure_log() -> None:
    folder = Path.home() / ".local" / "state" / "wispo"
    folder.mkdir(parents=True, exist_ok=True)
    handlers: list[logging.Handler] = [logging.FileHandler(folder / "service.log")]
    if sys.stderr.isatty():
        handlers.append(logging.StreamHandler())
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s", handlers=handlers)


if __name__ == "__main__":
    raise SystemExit(main())
