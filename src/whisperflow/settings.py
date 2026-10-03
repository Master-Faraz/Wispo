"""Settings window. It edits the YAML file and does not listen for the shortcut.

    uv run python -m whisperflow.settings

The transcription command keeps the model it loaded at startup. Restart it
after saving to use a new model or dictionary.
"""

from __future__ import annotations

import sys
from pathlib import Path

from whisperflow.config import DictionaryEntry, load_config, project_root, save_user_settings

_MODELS = ("tiny.en", "base.en", "small.en")


def installed_models(root: Path | None = None) -> list[tuple[str, int]]:
    """Models that have a ggml file on disk, with the file size in bytes."""
    root = root or project_root()
    found: list[tuple[str, int]] = []
    for name in _MODELS:
        path = root / "models" / f"ggml-{name}.bin"
        if path.is_file():
            found.append((name, path.stat().st_size))
    return found


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
    def __init__(self, gtk) -> None:
        self.gtk = gtk
        self.app = gtk.Application(application_id="com.whisperflow.Settings")
        self.app.connect("activate", self._activate)
        self._rows = gtk.ListBox()
        self._status = gtk.Label(xalign=0)
        self._model_buttons: dict[str, object] = {}

    def run(self, argv: list[str] | None) -> int:
        return self.app.run(None if argv is None else argv)

    def _activate(self, app) -> None:
        gtk = self.gtk
        config = load_config()
        window = gtk.ApplicationWindow(application=app, title="WhisperFlow")
        window.set_default_size(560, 520)

        page = gtk.Box(orientation=gtk.Orientation.VERTICAL, spacing=16)
        page.set_margin_top(16)
        page.set_margin_bottom(16)
        page.set_margin_start(16)
        page.set_margin_end(16)

        page.append(self._heading("Model"))
        page.append(self._model_box(config.asr.model))
        page.append(self._heading("Push-to-talk"))
        shortcut = gtk.Label(label="Ctrl+Super", xalign=0)
        page.append(shortcut)
        page.append(self._heading("Dictionary"))
        page.append(self._dictionary_box(config.dictionary))

        self._status.set_text("Changes apply the next time transcription starts.")
        page.append(self._status)

        actions = gtk.Box(spacing=8, halign=gtk.Align.END)
        save = gtk.Button(label="Save")
        save.add_css_class("suggested-action")
        save.connect("clicked", self._save)
        actions.append(save)
        page.append(actions)

        window.set_child(page)
        window.present()

    def _heading(self, text: str):
        label = self.gtk.Label(label=text, xalign=0)
        label.add_css_class("title-4")
        return label

    def _model_box(self, selected: str):
        box = self.gtk.Box(orientation=self.gtk.Orientation.VERTICAL, spacing=4)
        group = None
        installed = installed_models()
        if not installed:
            box.append(self.gtk.Label(label="No models found in models/", xalign=0))
            return box
        for name, size in installed:
            label = f"{name}    {size / (1024 * 1024):.0f} MB"
            if name == "base.en":
                label += "    recommended"
            button = self.gtk.CheckButton(label=label)
            if group is None:
                group = button
            else:
                button.set_group(group)
            button.set_active(name == selected)
            self._model_buttons[name] = button
            box.append(button)
        if selected not in self._model_buttons and self._model_buttons:
            next(iter(self._model_buttons.values())).set_active(True)
        return box

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
        return "base.en"

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
        path = project_root() / "whisperflow.yaml"
        save_user_settings(path, self._selected_model(), self._dictionary_entries())
        self._status.set_text("Saved. Restart transcription to use these settings.")


if __name__ == "__main__":
    raise SystemExit(main())
