"""Settings window. It edits the YAML file and does not listen for the shortcut.

    uv run python -m whisperflow.settings

The transcription command keeps the model it loaded at startup. Restart it
after saving to use a new model or dictionary.
"""

from __future__ import annotations

import sys
import threading
import time

from whisperflow.config import DictionaryEntry, load_config, project_root, save_user_settings
from whisperflow.hotkey.linux import capture_chord, format_chord
from whisperflow.models import CATALOG, DEFAULT_MODEL, download_model, installed_models, model_path


def main(argv: list[str] | None = None) -> int:
    gtk = _load_gtk()
    app = SettingsApp(gtk)
    return app.run(argv)


def _load_gtk():
    # The uv environment does not ship PyGObject. Ubuntu's copy is built for
    # this same Python version, so the settings window uses that.
    try:
        import gi
    except ModuleNotFoundError:
        sys.path.append("/usr/lib/python3/dist-packages")
        import gi
    gi.require_version("Gtk", "4.0")
    from gi.repository import Gtk

    return Gtk


class SettingsApp:
    def __init__(self, gtk, on_quit=None) -> None:
        self.gtk = gtk
        self._on_quit = on_quit
        self._rows = gtk.ListBox()
        self._status = gtk.Label(xalign=0)
        self._model_buttons: dict[str, object] = {}
        self._download_buttons: dict[str, object] = {}
        self._hotkey_keys = ("ctrl", "super")
        self._shortcut_label = None
        self._capture_stop = threading.Event()
        self.window = None

    def run(self, argv: list[str] | None) -> int:
        app = self.gtk.Application(application_id="com.whisperflow.Settings")
        app.connect("activate", self._activate)
        return app.run(None if argv is None else argv)

    def _activate(self, app) -> None:
        window = self.build(app)
        window.present()

    def build(self, app):
        gtk = self.gtk
        config = load_config()
        window = gtk.ApplicationWindow(application=app, title="WhisperFlow")
        window.set_default_size(560, 640)
        self.window = window
        self._hotkey_keys = config.hotkey.keys

        page = gtk.Box(orientation=gtk.Orientation.VERTICAL, spacing=16)
        page.set_margin_top(16)
        page.set_margin_bottom(16)
        page.set_margin_start(16)
        page.set_margin_end(16)

        page.append(self._heading("Model"))
        page.append(self._model_box(config.asr.model))
        page.append(self._heading("Push-to-talk"))
        page.append(self._shortcut_box())
        window.connect("close-request", self._stop_capture)
        page.append(self._heading("Dictionary"))
        page.append(self._dictionary_box(config.dictionary))

        if self._on_quit is None:
            self._status.set_text("Changes apply the next time transcription starts.")
        else:
            self._status.set_text("Closing this window leaves WhisperFlow in the dock.")
        page.append(self._status)

        actions = gtk.Box(spacing=8, halign=gtk.Align.END)
        if self._on_quit is not None:
            quit_button = gtk.Button(label="Quit")
            quit_button.connect("clicked", lambda *_args: self._on_quit())
            actions.append(quit_button)
        save = gtk.Button(label="Save")
        save.add_css_class("suggested-action")
        save.connect("clicked", self._save)
        actions.append(save)
        page.append(actions)

        window.set_child(page)
        return window

    def _heading(self, text: str):
        label = self.gtk.Label(label=text, xalign=0)
        label.add_css_class("title-4")
        return label

    def _model_box(self, selected: str):
        gtk = self.gtk
        box = gtk.Box(orientation=gtk.Orientation.VERTICAL, spacing=4)
        group = None
        on_disk = {name for name, _size in installed_models()}
        for name, _size in CATALOG:
            row = gtk.Box(spacing=8)
            button = gtk.CheckButton(label=self._model_label(name))
            if group is None:
                group = button
            else:
                button.set_group(group)
            present = name in on_disk
            button.set_sensitive(present)
            button.set_active(present and name == selected)
            self._model_buttons[name] = button
            row.append(button)
            if not present:
                download = gtk.Button(label="Download")
                download.connect("clicked", self._start_download, name)
                self._download_buttons[name] = download
                row.append(download)
            box.append(row)
        if not any(button.get_active() for button in self._model_buttons.values()):
            default = self._model_buttons.get(DEFAULT_MODEL)
            if default is not None and default.get_sensitive():
                default.set_active(True)
        return box

    def _model_label(self, name: str) -> str:
        path = model_path(name)
        if path.is_file():
            label = f"{name}    {path.stat().st_size / (1024 * 1024):.0f} MB"
        else:
            from whisperflow.models import catalog_size

            label = f"{name}    {catalog_size(name)}    not downloaded"
        if name == DEFAULT_MODEL:
            label += "    recommended"
        return label

    def _shortcut_box(self):
        gtk = self.gtk
        row = gtk.Box(spacing=8)
        self._shortcut_label = gtk.Label(label=format_chord(self._hotkey_keys), xalign=0, hexpand=True)
        change = gtk.Button(label="Change shortcut")
        change.connect("clicked", self._start_capture)
        row.append(self._shortcut_label)
        row.append(change)
        return row

    def _stop_capture(self, *_args) -> bool:
        self._capture_stop.set()
        return False

    def _start_capture(self, *_args) -> None:
        self._capture_stop.set()
        self._capture_stop = threading.Event()
        stop = self._capture_stop
        self._status.set_text("Press the shortcut, then release.")

        def work() -> None:
            try:
                keys = capture_chord(stop)
            except Exception as exc:
                self._idle(self._shortcut_failed, str(exc))
                return
            self._idle(self._shortcut_done, keys)

        threading.Thread(target=work, name="whisperflow-capture", daemon=True).start()

    def _shortcut_done(self, keys: tuple[str, ...]) -> bool:
        if not keys:
            self._status.set_text("No shortcut captured.")
            return False
        self._hotkey_keys = keys
        if self._shortcut_label is not None:
            self._shortcut_label.set_label(format_chord(keys))
        self._status.set_text(f"Shortcut set to {format_chord(keys)}. Save to keep it.")
        return False

    def _shortcut_failed(self, message: str) -> bool:
        self._status.set_text(message)
        return False

    def _start_download(self, _button, name: str) -> None:
        button = self._download_buttons.get(name)
        if button is not None:
            button.set_sensitive(False)
        self._status.set_text(f"Downloading {name}…")
        last = 0.0

        def progress(written: int, total: int) -> None:
            nonlocal last
            now = time.monotonic()
            if written != total and now - last < 0.3:
                return
            last = now
            self._idle(self._download_progress, name, written, total)

        def work() -> None:
            try:
                download_model(name, progress=progress)
            except Exception as exc:
                self._idle(self._download_failed, name, str(exc))
                return
            self._idle(self._download_done, name)

        threading.Thread(target=work, name=f"whisperflow-download-{name}", daemon=True).start()

    def _download_progress(self, name: str, written: int, total: int) -> bool:
        if total:
            self._status.set_text(f"Downloading {name}… {written * 100 // total}%")
        else:
            self._status.set_text(f"Downloading {name}… {written / (1024 * 1024):.0f} MB")
        return False

    def _download_done(self, name: str) -> bool:
        button = self._model_buttons[name]
        button.set_label(self._model_label(name))
        button.set_sensitive(True)
        button.set_active(True)
        download = self._download_buttons.get(name)
        if download is not None:
            download.set_visible(False)
        self._status.set_text(f"Downloaded {name}. Save, then restart WhisperFlow to use it.")
        return False

    def _download_failed(self, name: str, message: str) -> bool:
        download = self._download_buttons.get(name)
        if download is not None:
            download.set_sensitive(True)
        self._status.set_text(f"Could not download {name}: {message}")
        return False

    def _idle(self, callback, *args) -> None:
        from gi.repository import GLib

        GLib.idle_add(callback, *args)

    def _dictionary_box(self, entries: tuple[DictionaryEntry, ...]):
        gtk = self.gtk
        self._rows.set_selection_mode(gtk.SelectionMode.SINGLE)
        for entry in entries:
            self._append_row(entry.phrase, entry.replacement, entry.enabled)
        scrolled = gtk.ScrolledWindow(vexpand=True, min_content_height=180)
        scrolled.set_child(self._rows)
        add = gtk.Button(label="Add")
        remove = gtk.Button(label="Delete")
        add.connect("clicked", lambda *_: self._append_row("", "", True))
        remove.connect("clicked", self._delete_selected)
        buttons = gtk.Box(spacing=8)
        buttons.append(add)
        buttons.append(remove)
        wrap = gtk.Box(orientation=gtk.Orientation.VERTICAL, spacing=8)
        wrap.append(scrolled)
        wrap.append(buttons)
        return wrap

    def _append_row(self, phrase: str, replacement: str, enabled: bool) -> None:
        gtk = self.gtk
        row = gtk.ListBoxRow()
        box = gtk.Box(spacing=8)
        box.set_margin_top(4)
        box.set_margin_bottom(4)
        box.set_margin_start(4)
        box.set_margin_end(4)
        phrase_entry = gtk.Entry(text=phrase, placeholder_text="Spoken phrase", hexpand=True)
        replacement_entry = gtk.Entry(text=replacement, placeholder_text="Replacement", hexpand=True)
        enabled_button = gtk.CheckButton(active=enabled)
        box.append(enabled_button)
        box.append(phrase_entry)
        box.append(replacement_entry)
        row.phrase_entry = phrase_entry
        row.replacement_entry = replacement_entry
        row.enabled_button = enabled_button
        row.set_child(box)
        self._rows.append(row)

    def _delete_selected(self, *_args) -> None:
        row = self._rows.get_selected_row()
        if row is not None:
            self._rows.remove(row)

    def _selected_model(self) -> str:
        for name, button in self._model_buttons.items():
            if button.get_active():
                return name
        return DEFAULT_MODEL

    def _dictionary_entries(self) -> list[DictionaryEntry]:
        entries: list[DictionaryEntry] = []
        row = self._rows.get_row_at_index(0)
        index = 0
        while row is not None:
            phrase = row.phrase_entry.get_text().strip()
            replacement = row.replacement_entry.get_text()
            if phrase:
                entries.append(DictionaryEntry(phrase, replacement, row.enabled_button.get_active()))
            index += 1
            row = self._rows.get_row_at_index(index)
        return entries

    def _save(self, *_args) -> None:
        model = self._selected_model()
        if not model_path(model).is_file():
            self._status.set_text("Download that model before saving it.")
            return
        path = project_root() / "whisperflow.yaml"
        save_user_settings(path, model, self._dictionary_entries(), self._hotkey_keys)
        self._status.set_text("Saved. Restart WhisperFlow to use these settings.")


if __name__ == "__main__":
    raise SystemExit(main())
