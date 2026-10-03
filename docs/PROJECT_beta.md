# WhisperFlow

WhisperFlow is a local, offline speech-to-text app for Ubuntu GNOME on Wayland. You hold a keyboard shortcut, speak, and release. The finished sentence is typed into whichever window is focused. Audio and the transcript stay on this machine. Nothing is sent to a cloud API.

The speech engine is whisper.cpp with an Intel SYCL build, so an Intel Arc GPU is used when one is present. The desktop shell around that engine is Python: a global hotkey, a dictionary, a settings window, and keyboard injection through the XDG desktop portal.

This file describes the tree as it is now: how a process starts, which file owns each step, and how audio becomes typed text.

## What you run

All commands are from the repo root, with `uv`.

| Command | What it does |
| --- | --- |
| `uv run python -m whisperflow.service` | The real app. Loads the model, listens for the shortcut, types the finished sentence, and keeps a dock icon. |
| `uv run python -m whisperflow.service --quit` | Asks the running app to stop. |
| `uv run python -m whisperflow.settings` | Settings window alone. It edits `whisperflow.yaml` and does not listen for the shortcut. |
| `uv run python -m whisperflow.transcribe` | Terminal hold-to-talk. Prints partial lines. Does not type into other apps. |
| `uv run python -m whisperflow.transcribe --inject` | Same engine as the terminal tool, but types the finished sentence. |
| `uv run python -m whisperflow.inject "text"` | Types one string, for checking injection without the microphone. |
| `uv run python -m whisperflow.benchmark` | Times the model on a wav file. |
| `uv run pytest -q` | Unit tests. |

Starting the service a second time does not start a second listener. It asks the running process to show the settings window.

The log is `~/.local/state/whisperflow/service.log`. After the model loads it records the model name and the device, for example `model base.en on Intel(R) Arc(TM) Graphics (Intel SYCL)`.

A saved model, dictionary, or shortcut is read the next time the service starts. The running process does not reload `whisperflow.yaml` by itself.

## Machine this was built on

- Ubuntu with GNOME Shell on Wayland
- Intel Arc GPU, whisper.cpp SYCL backend, default model `base.en`
- Python 3.14 from `uv` (`.venv`)
- User in the `input` group, so `/dev/input/event*` is readable
- PipeWire. The app must not record from ALSA `default` (on this machine that path is a silent Bluetooth loopback)
- Top bar hidden by the Hide Top Bar extension, so the visible control is the bottom dock, not a system tray

## Folder structure

```text
my-wisper/
  whisperflow.yaml          settings the app reads at startup
  pyproject.toml            package name whisperflow, dependencies, pytest
  PROJECT.md                this file
  prd.md                    product requirements
  next.md                   stage-2 brief and what is already done
  CONTEXT.md                stage-1 handoff notes
  CURRENT_PROGRESS.md       what was verified on the machine

  src/whisperflow/          the Python package
    service.py              background app: window, hotkey, inject
    settings.py             GTK settings window
    cli.py                  terminal transcribe / inject commands
    application.py          microphone to text, no keyboard, no UI
    config.py               YAML load and save
    models.py               model catalog and Hugging Face downloads
    desktop.py              .desktop file installed for the app grid
    status.py               tray icon and dock badge over D-Bus
    inject.py               type one string and exit
    transcribe.py           entry point for the terminal tool
    benchmark.py            file timing
    audio/                  microphone, buffers, voice detection
    asr/                    device choice, windows, whisper, text merge
    hotkey/                 keyboard chord and GNOME overview key
    injection/              portal keyboard and US keymap
    processing/             dictionary

  native/
    whisperflow_asr.cpp     C API in front of whisper.cpp

  models/                   ggml weight files (downloaded, not source)
  scripts/
    build_sycl.sh           build whisper.cpp and the native library
    download_model.sh       download one ggml file with curl
  tests/                    unit tests, no GPU and no microphone required
  third_party/whisper.cpp   whisper.cpp 1.9.4-dev
```

`application.py` is the speech engine. It does not import the hotkey, the dictionary, the injector, or GTK. The service and the CLI call it. That split is intentional: the desktop pieces must not sit inside the audio loop.

## How the background app starts

`python -m whisperflow.service` runs `service.main`.

1. Open the log file `~/.local/state/whisperflow/service.log`.
2. `--quit` calls `Quit` on the D-Bus name `org.whisperflow.Service` and exits.
3. If that name is already owned, call `OpenSettings` on the running process and exit. One listener only.
4. Write `~/.local/share/applications/com.whisperflow.WhisperFlow.desktop` and remove an older `whisperflow.desktop` if it is there. `desktop.py` does this. The desktop id matches the GTK application id so GNOME can put the icon on the dock.
5. Import GTK. The uv environment does not ship PyGObject, so `settings._load_gtk` adds `/usr/lib/python3/dist-packages` when `gi` is missing. System Python is the same 3.14 series, so the `_gi` module loads.
6. `GLib.set_prgname("com.whisperflow.WhisperFlow")` before the window exists. Wayland uses that string as the app id.
7. `StatusIcon` in `status.py` takes the bus name `org.whisperflow.Service` and registers a StatusNotifierItem. If the name was taken between the check and this call, the new process opens settings on the old one and exits.
8. Build a GTK application with id `com.whisperflow.WhisperFlow` and the settings window from `SettingsApp.build`. Closing the window minimizes it and returns “handled”, so the process stays up. `app.hold()` keeps GTK from quitting when the window is hidden. Quit in the window sets a stop event.
9. A non-daemon thread named `whisperflow-listen` runs `_listen`. GTK stays on the main thread. The worker never touches GTK widgets. It asks the main thread to present the window or to quit through `GLib.idle_add`.
10. `_listen` loads config, opens the portal keyboard, starts the overview-key workaround if the shortcut contains Super, loads the Whisper model, then starts `HoldChord`.

On the way out, the worker stops the microphone, the hotkey thread, the capture device, restores the GNOME overview key, and closes the portal session. Then the main thread quits GTK. A daemon thread is not used for this loop, because a daemon would be killed before the overview key was restored.

## Data flow

This is the path for one hold of the shortcut while the service is running.

```text
/dev/input/event*                         hotkey/linux.py  HoldChord
        |  chord down
        v
service._listen
        |
        +-- status.py          icon / dock badge -> recording
        |
        v
application.start_recording    application.py
        |
        v
audio/sources.py               pick the laptop mic (not ALSA default)
audio/capture.py               16 kHz mono float frames, 30 ms
        |
        v
while the chord is still held:
    application.read_once(force_speech=True)
        audio/buffer.py        preroll ring
        audio/vad.py           bypassed while the key is held
        asr/chunker.py         5 s window, 250 ms overlap, every 1 s
        asr/whisper_cpp.py     partial text (not typed)
        asr/accumulator.py     merge repeated words
        |
        v
chord released
        |
        v
application.stop_recording
        one more whisper call on the whole utterance (up to 30 s)
        |
        v
cli._inject_final
        processing/dictionary.py     phrase replacement, once
        injection/portal.py          type key codes into the focused window
        |
        v
status.py                      back to idle
```

Partials exist so the model stays warm and the final window is not the first thing the GPU sees. They are printed only by the terminal command without `--inject`. The service and `--inject` type the finished sentence and nothing earlier.

If the utterance is longer than 30 seconds, the final Whisper call only sees the tail. The older part stays as the merged partial windows. Whisper’s useful context is 30 seconds.

## Configuration

`config.py` reads `whisperflow.yaml` into a frozen `Config`. The speech code never opens the file itself.

| Section | Used by | Meaning |
| --- | --- | --- |
| `audio` | `audio/capture.py`, `audio/sources.py` | 16000 Hz, mono, 30 ms frames, `device: auto` |
| `asr` | `asr/whisper_cpp.py`, `asr/device.py` | model file, `device: auto` (SYCL if an Arc GPU is listed), threads, `language` field |
| `streaming` | `asr/chunker.py` | window 5000 ms, overlap 250 ms, step 1000 ms |
| `vad` | `audio/vad.py` | energy threshold, hangover, preroll, high-pass. Used when the key is not forcing speech |
| `dictionary` | `processing/dictionary.py` | phrase, replacement, enabled |
| `hotkey.keys` | `hotkey/linux.py` | names that must all be down, default `ctrl` and `super` |

`save_user_settings` rewrites the model, the dictionary, and the hotkey. Other sections stay. PyYAML does not keep comments. The model path it writes is always `models/ggml-<name>.bin`.

`language` in the YAML is stored on the config. The native call in `native/whisperflow_asr.cpp` currently passes the language `"en"` itself.

## Audio

`audio/sources.py` asks PipeWire, through `wpctl`, for the built-in digital microphone and records that node. `audio.device: auto` is that choice. A specific node name can be set instead.

`audio/capture.py` opens the ALSA device `pipewire` with `PIPEWIRE_PROPS` aimed at that node. Frames are float32, 16 kHz, mono. Do not point this at ALSA `default`.

`audio/vad.py` is an energy detector with a noise floor, hangover, and preroll. The terminal tool uses it. The service, while the shortcut is held, calls `read_once(force_speech=True)`, so the detector cannot drop the phrase. Silence at the start and end of the key press is still inside the buffer that the final pass transcribes.

`audio/buffer.py` is the short ring that keeps audio from just before speech starts.

## Speech engine

`asr/device.py` runs `sycl-ls`. `device: auto` selects SYCL when a Level Zero GPU is listed, otherwise CPU. `--device cpu` on the terminal command forces CPU. The same `libwhisperflow_asr.so` is used for both. The GPU flag is an argument to `wf_load`.

`asr/whisper_cpp.py` loads that library with ctypes. Intel OpenMP and Level Zero are not on the default library path. If `LD_LIBRARY_PATH` is missing those directories, this module sets them and re-executes the process once. The dynamic linker only reads that variable at process start, so setting it later does nothing. Do not source oneAPI `setvars.sh` under `set -u`.

`native/whisperflow_asr.cpp` is the only file that includes whisper.cpp headers.

| C function | Role |
| --- | --- |
| `wf_load` | Load the ggml file. `use_gpu=1` and `GGML_SYCL_DEVICE` select the Arc GPU. Flash attention is off. The model stays loaded until `wf_free`. |
| `wf_transcribe` | One buffer in, text out. |
| `wf_system_info` | Build flags string. |
| `wf_free` | Unload. The service does not call this between sentences. |

Current decode flags inside `wf_transcribe`:

- greedy sampling
- `temperature_inc = 0` (no retry at a higher temperature)
- `no_timestamps = true`
- `no_context = true` (this call does not see the previous call’s text)
- `language = "en"`
- `single_segment` is true for a streaming window and false for the finished utterance

`asr/chunker.py` cuts the live audio into overlapping windows. `asr/accumulator.py` merges the partial texts so a word that sat in the overlap is not repeated.

`application.py` owns the session: `load_model`, `start_recording`, `read_once`, `stop_recording`. `stop_recording` returns the finished string and leaves the model loaded.

## Models

Weight files live in `models/`. The settings window lists `models.py` `CATALOG`. A missing file shows a Download button. The file is saved as `models/ggml-<name>.bin` even when the Hugging Face name is different.

| Name | Download |
| --- | --- |
| `tiny.en`, `base.en`, `small.en`, `medium.en`, `large-v3-turbo`, `large-v3-turbo-q5_0` | `ggerganov/whisper.cpp` |
| `distil-small.en` | `distil-whisper/distil-small.en`, file `ggml-distil-small.en.bin` |
| `distil-medium.en` | `distil-whisper/distil-medium.en`, remote file `ggml-medium-32-2.en.bin` |
| `distil-large-v3` | `distil-whisper/distil-large-v3-ggml` |
| `distil-large-v3.5` | `distil-whisper/distil-large-v3.5-ggml`, remote file `ggml-model.bin` |

`base.en` is the default. whisper.cpp reads the layer counts from the ggml header and runs every layer in that file. There is no fallback from a distilled file to `base.en`. If the file is missing or rejected, startup fails.

`scripts/download_model.sh` is the older curl helper for the whisper.cpp repo. The settings window uses `models.download_model` instead.

The vendored whisper.cpp is commit `6e4ab854` (1.9.4-dev). `scripts/build_sycl.sh` builds it and `native/whisperflow_asr.cpp`.

## Hotkey

`hotkey/linux.py` reads the kernel input device. It does not use the `evdev` Python package. The event struct is `llHHi` (time, type, code, value). Value 1 is press, 0 is release, 2 is repeat and is ignored.

`HoldKey` watches one key. The terminal tool without `--inject` uses it (default left Ctrl).

`HoldChord` watches the names in `hotkey.keys`. The chord is active only while every name is down. Left and right Ctrl both count as `ctrl`. The same is true for Shift, Alt, and Super. A letter or Space is a single key code.

GNOME opens the overview when Super is released. Listening to the key does not stop that. If the chord contains `super`, `OverviewKey` sets `org.gnome.mutter overlay-key` to empty for the life of the process and writes the previous value back on exit. A chord without Super does not touch that setting.

`hotkey/portal.py` is the XDG GlobalShortcuts experiment. It is not on the service path. A modifier-only shortcut was stored as key `0x00` and never fired. The working path is the evdev chord above.

Changing the shortcut in the settings window calls `capture_chord`, which reads the next press-and-release from the same devices. Save writes the names into the YAML. The running listener keeps the old chord until the next start.

## Dictionary

`processing/dictionary.py` runs once, on the finished sentence, and only on the inject path. The terminal tool that prints text does not apply it.

Rules: case-insensitive, whole words and phrases, longest phrase wins, punctuation after a phrase still matches, a shorter entry does not rewrite the inside of a longer word (`git` does not change `github`). Disabled entries and blank phrases are skipped.

## Text injection

`injection/portal.py` talks to the XDG RemoteDesktop portal with jeepney. It does not use the clipboard, wtype, ydotool, or libei.

Session setup, once per process:

1. `CreateSession`
2. `SelectDevices` with the keyboard bit and persist mode 2
3. `Start` (the first time, GNOME shows a permission dialog)
4. Save the restore token in `~/.local/state/whisperflow/portal-restore-token` (mode 0600)

Later starts reuse the token. `NotifyKeyboardKeycode` sends evdev key codes. `injection/keymap.py` is a US layout: shift is held across a run of shifted characters. A character that is not on that layout raises `UnsupportedCharacter`.

Portal responses are matched on the request path, not on the well-known sender name. The sender header is a unique bus name.

## Settings window

`settings.py` is configuration only. It does not record audio and it does not own the global hotkey.

The window has three sections:

- **Model.** Radios for the catalog. Download fetches the ggml file on a background thread and reports progress on the GTK thread. A model that is not on disk cannot be selected.
- **Push-to-talk.** Change shortcut, then Save.
- **Dictionary.** Add and delete rows.

Save calls `save_user_settings`. The status line says to restart WhisperFlow. Quit, when the window is inside the service, stops the listener. Closing the window does not.

The standalone command `python -m whisperflow.settings` uses application id `com.whisperflow.Settings`. Inside the service, the same widgets are placed in the `com.whisperflow.WhisperFlow` window so the dock icon and the window are the same app.

## Dock and tray

`status.py` implements the KDE StatusNotifierItem protocol on the session bus, because the Ayatana AppIndicator GIR is not installed. Icon names: idle `audio-input-microphone-symbolic`, recording `media-record-symbolic`, processing `system-run-symbolic`.

On this machine the top bar is hidden, so that tray icon is easy to miss. The dock icon is the window from the service. `com.canonical.Unity.LauncherEntry.Update` sets `urgent` while the app is not idle, which is the recording badge. The app id in that signal is `application://com.whisperflow.WhisperFlow.desktop`.

Left click on the tray icon opens settings. Secondary click quits. The menu has Settings and Quit.

## Terminal commands

`cli.py` is the other front end on the same `TranscriptionApp`.

- Without `--inject`: `HoldKey`, print partials and the final line, no dictionary, no portal.
- With `--inject`: portal keyboard, dictionary, `HoldChord`, overview key when needed, type the final line, do not print the transcript.
- `--device cpu|sycl|auto` overrides the YAML for that process.

`inject.py` opens the portal, waits, and types the argument. `benchmark.py` times a file. `transcribe.py` forwards to the CLI.

## Tests

`tests/` covers the config, the chord logic, the dictionary, the keymap, the desktop file text, model names, and the audio helpers. They do not open the microphone, the GPU, or D-Bus. `uv run pytest -q` is the check.

## What is deliberately not in this tree yet

- Applying a saved model, dictionary, or shortcut without restarting the service
- Temperature retry and a single-segment final pass (the current final call can add extra text on silence, which is why larger models can look worse than `base.en` on a short hold)
- AppImage packaging
- Login autostart
- A local LLM pass over the transcript
- Chunked long-form Distil-Whisper (whisper.cpp does not implement that strategy; a normal hold is one short clip)

## Rules that keep the app working

- Keep the model loaded between holds.
- Do not import the hotkey, dictionary, injector, or UI from `application.py`.
- Do not record from ALSA `default`.
- Do not `pip install evdev`.
- Do not type partial text.
- Do not use the clipboard for a normal transcript.
- Do not grab the whole keyboard. Only the configured chord is watched, and Super’s overview action is moved aside only while that chord uses Super.
- Restore the overview key when the process exits.
