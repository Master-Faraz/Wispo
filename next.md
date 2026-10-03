# WhisperFlow — Stage 2 Implementation Brief

## Purpose

WhisperFlow is a privacy-first, local, low-latency speech transcription application for Linux.

Stage 1 (the transcription engine) is complete and working well.

The next phase is to turn the existing terminal-based prototype into a proper Linux desktop application that can:

1. Run as a background desktop application.
2. Use configurable **push-to-talk** keyboard shortcuts.
3. Transcribe speech using the existing local Whisper engine.
4. Apply an optional local dictionary/replacement layer.
5. Insert the final transcription directly into the currently focused application.
6. Provide a small desktop UI for configuration.
7. Eventually be packaged as an AppImage.
8. Remain lightweight and must not introduce meaningful latency or memory/CPU overhead into the transcription pipeline.

The Local LLM functionality is explicitly **deferred** for now.

## Locked decisions (2026-10-03)

These override earlier suggestions in this brief where they differ.

1. **Keep the current hold-to-talk engine.** Holding the shortcut records and streams windows through Whisper, the same way `python -m whisperflow.transcribe` does today. Releasing it runs the final pass. Only that finished sentence is inserted into the focused application. Partial lines stay inside the engine and are not typed. Do not switch this to "record only, transcribe once on release."
2. **GPU is the default.** Live testing on this machine showed Intel Arc SYCL (`base.en`) fast and accurate, and `--device cpu` slow and inaccurate. The earlier file benchmark (CPU RTF about 0.09, GPU RTF about 0.65 on an 11-second wav) does not describe this hold-to-talk loop and must not override it. Keep `device: auto`, which selects SYCL when the Arc GPU is present. CPU stays available as a fallback, not as the preselected desktop device.
3. **Swallow the push-to-talk chord.** GNOME must not also act on it (Super must not open the overview). Swallow only that chord. Do not grab the entire keyboard.
4. **Dictionary matching.** Case-insensitive whole words and phrases. The longest matching phrase wins. Punctuation beside a phrase still matches. A shorter entry must not rewrite part of a longer word (`git` does not change `github`).
5. **Recording feedback.** A subtle visual state for recording and processing, tied to the app's dock or tray presence. Research the mechanism GNOME on Ubuntu 26.04 actually supports before picking a toolkit API.
6. **No clipboard for normal transcription.** The injection mechanism is still a research decision. Prefer a direct insert. Do not use the clipboard as the normal path.

## Stage 2 progress (user-verified 2026-10-03)

The vertical slice is working on this machine. The user has confirmed: portal text injection, Ctrl+Super hold-to-talk with GPU `base.en`, dictionary replacements, GTK settings window, background launch from Show Apps, dock icon, and close-window-without-quit behavior.

### Completed

| Area | What was built | Key files |
| ---- | -------------- | --------- |
| Text injection | XDG RemoteDesktop portal `NotifyKeyboardKeycode` via jeepney. US evdev keymap. Session persisted with restore token. No clipboard. | `injection/portal.py`, `injection/keymap.py`, `inject.py` |
| Push-to-talk | `HoldChord`: any Ctrl + any Super from `/dev/input/event*`. GNOME overview suppressed via `org.gnome.mutter overlay-key` while the service runs. Streaming hold-to-talk preserved (partials internal; only final sentence injected). | `hotkey/linux.py`, `cli.py`, `service.py` |
| Dictionary | Longest phrase wins, case-insensitive, punctuation-tolerant, whole-word boundaries. Applied once on the finished sentence, inject path only. | `processing/dictionary.py` |
| Settings UI | GTK 4 via system PyGObject (`python3-gi`). Model selection, dictionary add/delete, save to `whisperflow.yaml`. Restart required for model/dictionary changes. | `settings.py` |
| Background service | `uv run python -m whisperflow.service`. Single instance on D-Bus (`org.whisperflow.Service`). Worker thread for ASR/hotkey/inject; GTK on main thread. | `service.py` |
| Desktop integration | `com.whisperflow.WhisperFlow.desktop` with `StartupWMClass`. Dock icon via embedded GTK window. Close minimizes; Quit button stops the service. StatusNotifierItem + LauncherEntry for recording/processing state. | `desktop.py`, `status.py` |

### Technology choices (research outcomes)

| Component | Chosen | Rejected / deferred | Reason |
| --------- | ------ | ------------------- | ------ |
| Text injection | XDG RemoteDesktop portal keyboard | libei (headers not installed), wtype (Mutter lacks `zwp_virtual_keyboard_v1`), ydotool, clipboard | Portal works on Ubuntu 26.04 GNOME Wayland; first run shows a permission dialog; restore token avoids repeat prompts |
| Push-to-talk | Direct `/dev/input/event*` (`HoldChord`) | GlobalShortcuts portal (modifier-only binding stored as `0x00`, never fires), python-evdev | User is in `input` group; chord swallowed without grabbing the whole keyboard |
| Dictionary | Tokenize + longest-phrase replacement | — | Negligible vs ASR; 33 unit tests pass |
| UI | GTK 4 / PyGObject (system package) | PySide6 (heavier; not evaluated live) | Lightweight; `gi` not in uv venv so `/usr/lib/python3/dist-packages` is appended on ImportError |
| Recording feedback | LauncherEntry `urgent` badge + StatusNotifierItem icon states | Ayatana AppIndicator3 GIR (not available) | On this machine `hidetopbar` hides the top bar, so the tray icon is invisible; the dock icon is the visible control |
| Packaging | Deferred | AppImage | Stage 2.7 not started |

### Not done yet

* Shortcut configuration UI (chord is hardcoded to Ctrl+Super).
* AppImage and clean-environment packaging tests.
* Live model/dictionary reload without restarting the service.
* LLM post-processing (explicitly deferred).
* Full acceptance-test matrix across every listed editor/browser (injection verified in focused apps; not every item in section 30 re-tested formally).

### How to run

```bash
uv run python -m whisperflow.service          # start (or open settings if already running)
uv run python -m whisperflow.service --quit     # stop
uv run python -m whisperflow.settings         # settings only (standalone)
uv run python -m whisperflow.inject "text"    # injection smoke test
```

Log file: `~/.local/state/whisperflow/service.log` (includes `model base.en on Intel(R) Arc(TM) Graphics (Intel SYCL)` after load).

---

# 1. Current Project State

## Stage 1 — COMPLETE

The following functionality already exists and must remain stable:

```text
Microphone
    ↓
VAD
    ↓
Audio chunking
    ↓
Whisper.cpp
    ↓
Transcript accumulation
    ↓
Final transcript
```

Current environment:

* OS: Ubuntu 26.04 LTS
* Desktop: Wayland / GNOME
* CPU: Intel Core Ultra 5 225H
* GPU: Intel Arc 130T integrated GPU
* RAM: ~15 GB shared system memory
* Python: 3.14 managed through `uv`
* ASR: `whisper.cpp`
* GPU backend: SYCL
* Native bridge: C ABI + `ctypes`
* Audio: direct ALSA capture
* Current default model: `base.en`

Available models:

```text
tiny.en
base.en   ← current default
small.en
```

The user has manually tested `base.en` and considers its transcription quality very good.

Therefore:

> `base.en` should remain the default model.

Users should still be able to select another installed model from the UI.

---

# 2. Current Performance

Existing benchmark results:

```text
base.en / CPU:
RTF ≈ 0.09

base.en / Intel Arc SYCL:
RTF ≈ 0.65

tiny.en / Intel Arc SYCL:
RTF ≈ 0.54
```

The transcription engine is currently the most computationally expensive part of the application.

Therefore Stage 2 must follow this rule:

> Nothing introduced by the desktop/UI/input/dictionary layers should meaningfully slow down or interfere with ASR inference.

Do not optimize prematurely at the expense of architectural simplicity, but avoid introducing heavyweight dependencies where a lightweight native mechanism is available.

---

# 3. Revised Roadmap

The project roadmap is now:

```text
[x] Stage 1 — Local Transcription Engine

[~] Stage 2 — Desktop Application & System Integration
      [x] Wayland text injection (XDG RemoteDesktop portal)
      [x] Global push-to-talk (Ctrl+Super; not yet configurable in UI)
      [x] Dictionary
      [x] Settings window (model + dictionary; no shortcut UI yet)
      [x] Background operation + dock icon
      [ ] Shortcut configuration UI
      [ ] AppImage

[ ] Stage 3 — Linux Distribution
      [x] .desktop entry (com.whisperflow.WhisperFlow.desktop)
      [x] application icon (dock + launcher)
      [ ] AppImage
      [ ] optional startup integration

[ ] Stage 4 — Hardware & Model Optimization
      ├── SYCL optimization
      ├── quantization
      ├── FP16
      └── RTF improvements

[ ] Stage 5 — Local LLM Processing
      ├── transcript cleanup
      ├── formatting
      └── optional intelligent processing
```

Do not implement Stage 5 now.

---

# 4. Stage 2 Goals

Stage 2 should produce a usable desktop version of WhisperFlow.

The intended user experience is:

```text
WhisperFlow running in background
            │
            │
       Hold shortcut
            │
            ▼
      Start recording
            │
            │ user speaks
            │
       Release shortcut
            │
            ▼
      Stop recording
            │
            ▼
       Whisper ASR
            │
            ▼
        Dictionary
            │
            ▼
      Text injection
            │
            ▼
Currently focused application
```

Example:

````text
User opens VS Code.

User holds Ctrl + Super.

User says:

"create a function called calculate total"

User releases Ctrl + Super.

WhisperFlow produces:

"create a function called calculate total"

and inserts that text directly into the active VS Code editor.

No terminal interaction should be required.

---

# 5. Push-to-Talk Is Required

WhisperFlow should use **push-to-talk**, not toggle recording.

Required behavior:

```text
SHORTCUT DOWN
    ↓
START RECORDING

SHORTCUT HELD
    ↓
CONTINUE RECORDING

SHORTCUT RELEASED
    ↓
STOP RECORDING
    ↓
FINALIZE TRANSCRIPTION
    ↓
PROCESS DICTIONARY
    ↓
INJECT TEXT
````

Do not implement toggle mode as the primary interaction at this stage.

Toggle mode may be considered in the future, but is not part of the current requirement.

---

# 6. Configurable Shortcut

The current prototype uses:

```text
Left Ctrl
```

This is only a testing configuration.

The intended default shortcut is:

```text
Ctrl + Super
```

The user wants configurable shortcuts.

Potential future examples:

```text
Ctrl + Super
Ctrl + Shift
Alt + Space
Super + Shift + V
Caps Lock
```

The shortcut system should support combinations of keys rather than being hardcoded to one key.

The shortcut configuration should be stored in application configuration.

---

# 7. Shortcut Handling Architecture

The UI must NOT directly handle the push-to-talk events.

The UI is configuration only.

Use a separate component:

```text
ShortcutManager
```

Suggested architecture:

```text
/dev/input/event*
        ↓
Linux keyboard event reader
        ↓
ShortcutManager
        ↓
START / STOP recording events
        ↓
Application Controller
```

The current project already has a working direct `/dev/input/event*` implementation using Python's standard library and binary event parsing.

Do not unnecessarily replace it with `python-evdev`.

The current implementation exists because the `evdev` Python package previously introduced unwanted build/system dependencies.

However, do not assume the existing implementation is automatically the final architecture.

Before finalizing it:

* verify multi-key combinations work reliably;
* handle key-down and key-up correctly;
* handle modifier state;
* handle repeated key events;
* handle releases in any order;
* avoid duplicate START/STOP transitions;
* verify behavior when another key is pressed while the shortcut is held;
* investigate whether GNOME also receives the shortcut;
* avoid grabbing the entire keyboard unless there is a demonstrated need.

The implementation should not interfere with normal keyboard operation.

---

# 8. Important Separation: UI vs Shortcut Handling

The desktop UI should never be responsible for:

* listening for the push-to-talk shortcut;
* starting/stopping audio capture;
* injecting text;
* processing keyboard events.

Instead:

```text
                 Desktop UI
                     │
                     │ configuration
                     ▼
             Application Core
                     │
        ┌────────────┼─────────────┐
        │            │             │
        ▼            ▼             ▼
 ShortcutManager   ASR Engine   Dictionary
        │            │             │
        └────────────┴──────┬──────┘
                            ▼
                       TextInjector
```

This keeps the application maintainable and allows the UI to evolve independently.

---

# 9. Text Injection — Critical Requirement

The final transcription must be inserted directly into the currently focused application.

Examples:

* VS Code
* Chrome
* Firefox
* GNOME Terminal
* Ptyxis
* Alacritty
* GTK text fields
* Qt text fields
* ordinary browser text inputs

The application should NOT depend on clipboard manipulation for normal transcription.

The clipboard should remain completely untouched.

Do NOT implement:

```text
transcript
   ↓
clipboard
   ↓
Ctrl+V
```

Do NOT:

* overwrite the user's clipboard;
* save and restore the clipboard;
* depend on clipboard ownership;
* use clipboard as an intermediate transport.

If a user wants the transcript copied to the clipboard, that can be a separate future user-controlled feature.

Normal transcription should directly inject/type the text.

---

# 10. Wayland Research Direction

The application runs on Wayland.

Wayland intentionally restricts arbitrary synthetic input.

The implementation therefore needs to use an appropriate supported mechanism rather than assuming X11-style automation will work.

Investigate the following technologies:

## A. libei / Emulated Input

Investigate `libei` as the primary candidate.

libei is designed for emulated input and integrates with the Linux desktop input stack.

Investigate:

* `libei`
* `libeis`
* `liboeffis`
* XDG Desktop Portal RemoteDesktop APIs
* GNOME/Mutter support
* required authorization/session flow
* whether a persistent session/token can be used
* keyboard input
* Unicode text input
* latency
* failure/reconnection behavior

The conceptual flow is:

```text
WhisperFlow
    ↓
libei / liboeffis
    ↓
XDG Desktop Portal
    ↓
GNOME / Mutter
    ↓
focused application
```

Determine whether this is practical and reliable on the target Ubuntu 26.04 GNOME environment.

---

# 11. Other Input Technologies Must Be Investigated

Before implementation, research and compare:

### `wtype`

Investigate the Wayland virtual keyboard protocol and `wtype`.

Questions:

* Does it work reliably on GNOME/Mutter?
* Is it appropriate for arbitrary Unicode?
* Does it require wlroots-specific protocols?
* Does it work reliably on Ubuntu 26.04 GNOME?
* What are the runtime dependencies?
* Does invoking it for every transcription introduce unnecessary process overhead?

### `ydotool`

Investigate:

* `ydotool`
* `ydotoold`
* `/dev/uinput`

Questions:

* Does it work cleanly on Ubuntu 26.04?
* Does it require elevated permissions or special device permissions?
* How does it handle Unicode?
* Does the daemon create unnecessary complexity?
* Is it more or less reliable than libei?

### Direct `/dev/uinput`

Investigate whether directly creating a virtual keyboard is practical.

Consider:

* permissions;
* Unicode handling;
* keyboard layouts;
* modifier state;
* security;
* reliability;
* packaging;
* AppImage compatibility.

### Wayland text-input protocols

Research `text-input-v3` and related protocols.

Determine whether they are actually appropriate for this application.

Do not assume they are an input-injection mechanism merely because they contain the concept of "commit text".

---

# 12. Python Libraries Must Also Be Investigated

Research available Python libraries/bindings for:

* libei;
* Wayland;
* virtual keyboard input;
* uinput;
* Linux input events.

One candidate to investigate is:

```text
python-libei
```

However, do not automatically adopt it.

Evaluate:

* project maturity;
* release history;
* maintenance;
* API stability;
* dependency footprint;
* performance;
* Python overhead;
* whether it is suitable for a production application;
* whether a small C/C++ shim is safer.

The project already uses:

```text
Python
   ↓
ctypes
   ↓
stable C ABI
   ↓
native library
```

for Whisper.

A similar approach may be appropriate for libei if Python bindings are immature.

---

# 13. Preferred Text Injector Architecture

The Python application should expose an abstract interface:

```python
class TextInjector(ABC):

    @abstractmethod
    def insert(self, text: str) -> None:
        ...
```

The ASR/application layer should only know:

```python
injector.insert(transcript)
```

It must not know:

* Wayland;
* libei;
* portals;
* uinput;
* wtype;
* ydotool;
* keyboard layouts.

Potential implementation:

```text
src/whisperflow/injection/
    base.py
    wayland.py
```

If a native implementation is required:

```text
native/
    whisperflow_input.cpp
```

with a small C ABI.

For example:

```text
wf_input_init()
wf_input_insert_text()
wf_input_shutdown()
```

The exact ABI should be designed based on the chosen technology.

---

# 14. Performance Requirements for Text Injection

Text injection must not become a bottleneck.

Do NOT:

* spawn a new subprocess for every transcript unless benchmarking proves it is negligible;
* start/stop a daemon for every transcription;
* initialize a Wayland connection for every utterance;
* initialize input infrastructure for every utterance;
* involve an LLM;
* touch the clipboard;
* run expensive text processing.

Prefer:

```text
Application startup
    ↓
Initialize input subsystem once
    ↓
Keep session/connection alive
    ↓
transcript
    ↓
inject immediately
```

The input subsystem should be initialized once and reused.

---

# 15. Dictionary System

A local dictionary/replacement system is required.

It must be:

* local;
* deterministic;
* extremely lightweight;
* fast;
* optional;
* independent from Whisper;
* independent from the UI;
* independent from the future LLM.

Pipeline:

```text
Final Whisper transcript
        ↓
DictionaryProcessor
        ↓
TextInjector
```

Example:

```text
Input:
    "open pull request"

Dictionary:
    "open pull request" → "Open Pull Request"

Output:
    "Open Pull Request"
```

Other examples:

```text
"github" → "GitHub"

"javascript" → "JavaScript"

"my email" → "user@example.com"

"company name" → "Example Corporation"
```

---

# 16. Dictionary Must NOT Process Streaming Text

Do not run dictionary replacement continuously against partial ASR hypotheses.

Avoid:

```text
partial transcript
    ↓
dictionary
    ↓
partial transcript
    ↓
dictionary
    ↓
...
```

Instead:

```text
record
  ↓
ASR
  ↓
final transcript
  ↓
dictionary ONCE
  ↓
injection
```

This keeps the dictionary effectively free from a performance perspective.

---

# 17. Dictionary Algorithm

Start simple.

The first implementation can use an efficient deterministic replacement mechanism.

Do not prematurely introduce a complex NLP engine.

For example:

```python
processed_text = dictionary.process(final_text)
```

The dictionary should support:

```text
spoken phrase → replacement text
```

The data model should eventually support:

* exact phrase replacements;
* case sensitivity options;
* enable/disable individual entries;
* import/export;
* ordering/priority;
* potentially regular expressions in a future version.

Do NOT implement regex support unless there is a concrete requirement.

For a small/medium dictionary, a simple optimized lookup/replacement implementation is sufficient.

If future requirements create very large dictionaries, investigate:

* trie;
* Aho-Corasick;
* other multi-pattern matching algorithms.

Do not optimize for a problem that does not yet exist.

---

# 18. Dictionary Interface

Use an abstraction similar to:

```python
class TextProcessor(ABC):

    @abstractmethod
    def process(self, text: str) -> str:
        ...
```

Then:

```text
final transcript
      ↓
DictionaryProcessor
      ↓
future LLM processor
      ↓
TextInjector
```

This means a future LLM can be inserted without modifying the ASR engine.

---

# 19. UI Requirements

The UI should initially remain intentionally small.

The primary configuration areas are:

## Model

```text
Model

○ tiny.en
● base.en
○ small.en
```

Default:

```text
base.en
```

The UI should only show models that actually exist/are installed.

Potentially show useful metadata:

```text
base.en
141 MB
Recommended default
```

But do not make model switching dynamically unload/reload models unless explicitly required.

Because model initialization is expensive, consider:

* loading the selected model at application startup;
* keeping it resident;
* restarting/reloading only when necessary.

Do not unload the model after each recording.

---

# 20. Shortcut Settings

Example UI:

```text
Push-to-talk shortcut

[ Ctrl + Super ]

        [ Change Shortcut ]
```

The UI should provide a way to capture a new key combination.

Important:

The UI only modifies configuration.

It does not become the actual global keyboard listener.

---

# 21. Dictionary UI

Example:

```text
Dictionary

┌──────────────────────┬─────────────────────────┐
│ Spoken phrase        │ Replacement             │
├──────────────────────┼─────────────────────────┤
│ github               │ GitHub                  │
│ javascript           │ JavaScript              │
│ open pull request    │ Open Pull Request       │
└──────────────────────┴─────────────────────────┘

[ + Add ]

[ Edit ] [ Delete ]
```

The UI should provide basic CRUD functionality.

The dictionary should be persisted in application configuration/data.

Do not make the dictionary UI overly complex in Stage 2.

---

# 22. Desktop Application

The application should run as a proper Linux desktop application.

The user should be able to:

* launch WhisperFlow from the Ubuntu application launcher;
* keep it running in the background;
* open its configuration window;
* configure model/shortcut/dictionary;
* close the configuration window without terminating the background service.

A tray/status indicator may be included if practical.

However, tray functionality should not delay the core Stage 2 functionality.

---

# 23. UI Technology

Investigate appropriate lightweight Linux desktop UI frameworks.

Candidate:

```text
PySide6 / Qt
```

Evaluate:

* memory usage;
* startup time;
* AppImage compatibility;
* Wayland support;
* tray support;
* maintainability;
* packaging complexity;
* compatibility with Python 3.14.

Other reasonable alternatives may be researched.

Do not select a UI framework solely based on familiarity.

The final choice should consider the entire application:

```text
Python ASR
+
native C/C++
+
Wayland
+
background process
+
desktop UI
+
AppImage
```

---

# 24. UI Must Not Bottleneck ASR

The UI must not:

* block the ASR worker;
* block audio capture;
* perform synchronous heavy operations on the inference thread;
* reload models unnecessarily;
* process dictionary entries continuously;
* interact with the keyboard listener;
* perform text injection itself.

The architecture should use clear asynchronous/event-driven boundaries where appropriate.

---

# 25. AppImage

AppImage packaging is part of Stage 2/3.

The final application should eventually be launchable as:

```text
WhisperFlow-x86_64.AppImage
```

It should include:

* Python runtime;
* required Python dependencies;
* native WhisperFlow libraries;
* whisper.cpp libraries;
* required resources;
* application icon;
* `.desktop` integration where appropriate.

Important:

Do not bundle system-specific GPU drivers.

The application should use the host's:

* Intel Level Zero runtime;
* SYCL runtime;
* required GPU drivers.

The AppImage should not attempt to bundle the entire Intel graphics stack.

---

# 26. Ubuntu Dock

The application should provide a valid Linux `.desktop` entry so it can appear in the Ubuntu application launcher and be pinned to the dock.

The desktop entry should eventually contain:

```text
Name=WhisperFlow
Exec=...
Icon=...
Type=Application
Categories=Utility;
```

The exact implementation should be compatible with AppImage deployment.

---

# 27. Suggested Project Structure

Do not blindly apply this structure if the existing architecture suggests a better organization, but preserve the separation of concerns.

Suggested:

```text
src/whisperflow/

├── application/
│   ├── controller.py
│   ├── recording.py
│   └── events.py
│
├── asr/
│   ├── engine.py
│   ├── whisper_cpp.py
│   ├── chunker.py
│   └── accumulator.py
│
├── audio/
│   ├── capture.py
│   ├── sources.py
│   ├── vad.py
│   └── buffer.py
│
├── hotkey/
│   ├── linux.py
│   ├── manager.py
│   └── bindings.py
│
├── injection/
│   ├── base.py
│   └── wayland.py
│
├── processing/
│   ├── base.py
│   └── dictionary.py
│
├── ui/
│   ├── app.py
│   ├── window.py
│   ├── settings.py
│   └── dictionary.py
│
└── config/
    └── ...
```

Potential native additions:

```text
native/
    whisperflow_asr.cpp
    whisperflow_input.cpp
```

Do not restructure existing ASR code unnecessarily.

---

# 28. Architectural Rules

These rules are mandatory.

### Rule 1 — ASR stays independent

The ASR engine must never import:

* UI;
* hotkey system;
* text injector;
* dictionary;
* LLM.

### Rule 2 — UI is configuration only

The UI must not become the global keyboard listener.

### Rule 3 — No clipboard

Normal transcription must never touch the user's clipboard.

### Rule 4 — No LLM

LLM processing is explicitly deferred.

### Rule 5 — Model stays loaded

Do not unload Whisper models after every recording.

### Rule 6 — Input subsystem stays initialized

Avoid initializing the Wayland input mechanism for every transcript.

### Rule 7 — Final transcript only

Dictionary processing happens after final ASR output.

### Rule 8 — Linux/Wayland first

Do not introduce X11 dependencies for the primary implementation.

### Rule 9 — Avoid unnecessary Python dependencies

Prefer standard-library Python or small native dependencies when practical.

### Rule 10 — Preserve the working Stage 1 implementation

Do not destabilize the working microphone, VAD, chunking, accumulator, or ASR system while implementing Stage 2.

---

# 29. Recommended Implementation Order

Do not build the complete UI first.

Build a vertical slice first.

## Milestone 2.1 — Text Injection Proof of Concept ✅

**Done (2026-10-03).** XDG RemoteDesktop portal `NotifyKeyboardKeycode`. User-verified in focused applications. Clipboard untouched.

Implement:

```text
existing ASR
    ↓
final transcript
    ↓
text injector
    ↓
focused application
```

Test with:

1. VS Code
2. Firefox
3. Chrome
4. GNOME Terminal
5. Ptyxis
6. GTK text field
7. Qt text field

The implementation must not touch the clipboard.

---

## Milestone 2.2 — Push-to-Talk ✅

**Done (2026-10-03).** `HoldChord` (Ctrl+Super) + `OverviewKey` suppression. User-verified. Shortcut not yet editable from settings.

Connect:

```text
Ctrl + Super DOWN
        ↓
start recording

Ctrl + Super UP
        ↓
stop recording
        ↓
ASR
        ↓
dictionary
        ↓
text injection
```

Verify:

* correct modifier handling;
* correct key release behavior;
* repeated key events;
* releasing keys in either order;
* rapid press/release;
* accidental key presses;
* no keyboard interference.

---

## Milestone 2.3 — Dictionary ✅

**Done (2026-10-03).** `DictionaryProcessor` with unit tests. User-verified (`github` → `GitHub`, etc.).

Add:

```text
final transcript
      ↓
dictionary
      ↓
text injector
```

Create unit tests for:

* exact replacements;
* multiple replacements;
* overlapping phrases;
* punctuation;
* capitalization;
* empty dictionary;
* disabled entries.

Benchmark dictionary processing.

The benchmark should demonstrate that dictionary processing is negligible relative to ASR.

---

## Milestone 2.4 — Desktop UI (partial)

**Done:** model selection, dictionary management (GTK 4 settings window).

**Remaining:** shortcut configuration / capture UI.

The UI should control configuration but not handle global shortcut events.

---

## Milestone 2.5 — Background Application ✅

**Done (2026-10-03).** `python -m whisperflow.service`, single-instance D-Bus, settings embedded in service process, close-to-minimize.

Make the application:

* launchable without a terminal;
* persistent in the background;
* capable of opening/closing the settings window;
* capable of running ASR independently of the UI.

---

## Milestone 2.6 — Desktop Integration (mostly done)

**Done (2026-10-03).** `.desktop` file, dock icon, LauncherEntry recording badge, StatusNotifierItem (tray hidden when top bar is hidden by `hidetopbar`).

**Remaining:** pin-to-dock documentation, optional autostart.

Add:

* application icon;
* `.desktop` file;
* launcher integration;
* optional tray/status indicator.

---

## Milestone 2.7 — AppImage

Only after the application works correctly:

* build AppImage;
* test on the development system;
* test in a clean Ubuntu environment;
* verify native library loading;
* verify microphone access;
* verify Wayland input;
* verify Intel GPU/CPU fallback behavior.

---

# 30. First Acceptance Test

**Status (2026-10-03):** steps 1–13, 15–17 verified by the user. Step 14 (change shortcut from UI) not implemented. Step 11 (every listed app) not formally re-tested as a matrix.

Before considering Stage 2 successful, the following must work:

```text
1. Launch WhisperFlow from Ubuntu application launcher.

2. WhisperFlow remains running in the background.

3. Open VS Code.

4. Hold Ctrl + Super.

5. Speak:
   "create a hello world function"

6. Release Ctrl + Super.

7. WhisperFlow transcribes using base.en.

8. Dictionary processing runs.

9. Final text appears directly in the focused VS Code editor.

10. Clipboard remains unchanged.

11. Repeat the same test in:
    - Firefox
    - Chrome
    - terminal
    - GTK text field

12. Open WhisperFlow settings.

13. Change the model.

14. Change the push-to-talk shortcut.

15. Add/edit/delete a dictionary entry.

16. Close the settings window.

17. WhisperFlow continues running in the background.
```

---

# 31. Performance Acceptance Criteria

Stage 2 must not materially degrade the current ASR performance.

Measure:

```text
ASR inference time
Dictionary processing time
Text injection latency
End-to-end release → text appearance latency
Memory usage
Application startup time
```

The important metric is:

```text
shortcut release
       ↓
final text visible
```

The new components should contribute as little latency as reasonably possible.

Dictionary processing should be effectively negligible.

Text injection should not require launching a heavyweight process per utterance.

---

# 32. Research Requirement Before Implementation

Before committing to the final implementation, perform fresh research on the current Ubuntu 26.04 / Wayland ecosystem.

Specifically investigate:

### Input injection

* libei
* libeis
* liboeffis
* XDG Desktop Portal RemoteDesktop
* GNOME/Mutter support
* Wayland virtual keyboard protocol
* wtype
* ydotool
* uinput
* Python bindings for the above

### Shortcut detection

* `/dev/input/event*`
* python-evdev
* libinput
* other lightweight Linux keyboard event APIs

### Desktop UI

* PySide6 / Qt
* GTK/PyGObject
* other appropriate lightweight Linux desktop UI options

### Packaging

* AppImage
* linuxdeploy
* appimagetool
* PyInstaller
* Nuitka
* other viable approaches

Evaluate the tradeoffs specifically for:

* Ubuntu 26.04;
* GNOME;
* Wayland;
* Python 3.14;
* Intel GPU/SYCL;
* AppImage;
* low memory usage;
* low startup time;
* low runtime overhead;
* long-term maintainability.

---

# 33. Consultant / Researcher Decision Report

Before implementing the final solution, **do not simply follow the technologies suggested in this document**.

Perform your own technical research.

At the end of your research, provide a concise decision report answering:

### 1. Text Injection

Which approach would you personally choose?

Compare:

```text
libei
wtype / Wayland virtual keyboard
ydotool
direct uinput
Python bindings
native C/C++ integration
```

Explain:

* why you chose it;
* why you rejected the alternatives;
* GNOME/Wayland compatibility;
* security/permission requirements;
* Unicode handling;
* expected latency;
* memory overhead;
* reliability;
* packaging implications.

### 2. Shortcut Detection

Compare:

```text
direct /dev/input/event*
python-evdev
libinput
other relevant options
```

Recommend one and explain why.

### 3. Dictionary

Recommend the implementation approach.

Explain why it will not become a performance bottleneck.

Include an appropriate benchmark/test strategy.

### 4. UI

Research the current options and recommend:

```text
PySide6/Qt
GTK/PyGObject
other candidate
```

Explain the memory/startup/AppImage tradeoffs.

### 5. AppImage

Recommend the packaging approach.

Consider the existing:

```text
Python 3.14
whisper.cpp
native C/C++
SYCL
Intel oneAPI/Level Zero
```

environment.

Explicitly identify which system libraries should remain external rather than being bundled.

### 6. Final Recommendation

Provide a final table:

| Component      | Recommended technology | Alternatives considered | Reason |
| -------------- | ---------------------- | ----------------------- | ------ |
| Push-to-talk   | ?                      | ?                       | ?      |
| Text injection | ?                      | ?                       | ?      |
| Dictionary     | ?                      | ?                       | ?      |
| UI             | ?                      | ?                       | ?      |
| Packaging      | ?                      | ?                       | ?      |

Do not blindly accept the recommendations in this document.

Use current documentation and, where useful, actual experiments on the development machine.

If a proposed technology does not work correctly with Ubuntu 26.04 + GNOME + Wayland, reject it rather than forcing the architecture around it.

---

# 34. Important Final Principle

WhisperFlow's primary purpose is **fast local transcription**.

Every Stage 2 decision should be evaluated against:

```text
Does this make transcription slower?
Does this increase memory significantly?
Does this add unnecessary dependencies?
Does this introduce unnecessary processes?
Does this complicate the native/runtime environment?
```

If a feature does not materially improve the current user experience, defer it.

The desired architecture is:

```text
              FAST LOCAL ASR
                    │
                    ▼
              FINAL TRANSCRIPT
                    │
             ┌──────┴──────┐
             ▼             ▼
        Dictionary      (future LLM)
             │
             ▼
       Direct Wayland
       text injection
             │
             ▼
       Focused application
```

Keep this pipeline minimal, deterministic, and local.
