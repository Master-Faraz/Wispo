# Wispo

Local, push-to-talk speech dictation for Linux. Wispo records from a microphone, transcribes speech with a local `whisper.cpp` model, then types the final text into the focused application through the XDG RemoteDesktop portal.

The application is designed for a GNOME/PipeWire-style Linux desktop. It has been developed around Intel oneAPI/SYCL acceleration, and it can select the CPU backend at runtime from the same native build. Treat hardware and desktop compatibility outside that environment as unverified until you test it.

## What it does

- Runs Whisper locally; ordinary transcription audio and text are not sent to a Wispo service.
- Reads a configurable hold-to-talk chord from Linux input devices.
- Captures 16 kHz mono audio through ALSA (including PipeWire's ALSA plugin), detects speech, and produces partial transcripts while the chord is held.
- Re-runs the complete utterance on release, applies the configured replacement dictionary, and injects the final text into the focused application.
- Provides a GTK4 settings window, click-through recording HUD, desktop launcher entry, and D-Bus status item.

Model downloads are the one intentional network operation. The desktop portal may also display a permission prompt before Wispo can inject keystrokes.

## Requirements

This is a Linux desktop application. A working installation needs:

- Python 3.12 or newer and [`uv`](https://docs.astral.sh/uv/).
- GTK4 and PyGObject (`python3-gi`, `gir1.2-gtk-4.0` on Debian/Ubuntu).
- ALSA development/runtime libraries. PipeWire with its ALSA plugin is recommended when selecting a PipeWire source by name.
- Permission to read `/dev/input/event*` for the global hold-to-talk shortcut. On many distributions, add the user to the `input` group, then sign out and back in.
- A desktop portal implementation that supports the RemoteDesktop keyboard permission for text insertion.
- `cmake`, a C/C++ build toolchain, `curl`, and the compatible `whisper.cpp` source tree.

For the current native build, Intel oneAPI is required even when you intend to select the CPU backend at runtime: `scripts/build_sycl.sh` builds the SYCL-enabled shared library with `icpx` and oneMKL. Intel Arc/Level Zero support is required to select `sycl`; otherwise set `asr.device: cpu` or allow `auto` to select CPU.

### Debian/Ubuntu packages

Package names vary across releases. The following is a starting point, not a portability guarantee:

```bash
sudo apt update
sudo apt install -y \
  build-essential cmake pkg-config curl \
  python3-gi gir1.2-gtk-4.0 \
  libasound2-dev libze1 libze-intel-gpu1
```

Install Intel's DPC++ compiler, oneMKL development files, and the Level Zero development files using the oneAPI instructions for your distribution. The build script checks for `icpx` and `MKLConfig.cmake` and reports a missing dependency before it starts compiling.

## Install from a checkout

```bash
git clone https://github.com/Master-Faraz/my-wisper.git wispo
cd wispo
```

### Provide `whisper.cpp`

The repository currently records `third_party/whisper.cpp` as a Git link but does **not** include `.gitmodules`. A fresh clone therefore does not fetch it automatically. Until the repository pins and publishes submodule metadata, provide a compatible checkout manually:

```bash
git clone --depth 1 https://github.com/ggml-org/whisper.cpp.git third_party/whisper.cpp
```

For a production release, pin the tested `whisper.cpp` revision and restore `.gitmodules`; the command above follows the upstream default branch and can change over time.

### Create the Python environment

```bash
uv sync
```

### Build the native bridge

```bash
bash scripts/build_sycl.sh
```

The command builds `whisper-cli` beneath `third_party/whisper.cpp/build-sycl/` and creates the Python-facing library at `native/build/libwispo_asr.so`.

### Download a model

The checked-in configuration selects `small.en`:

```bash
bash scripts/download_model.sh small.en
```

Model files are stored beneath `models/` and are intentionally ignored by Git. The settings application can also download catalogue models.

## Run

Start the desktop service:

```bash
uv run python -m wispo.service
```

On first start, grant the portal's keyboard-input permission. Hold the configured shortcut (the checked-in configuration uses the Copilot key), speak, then release it to insert the completed transcript. Closing the settings window hides it; choose **Quit** in the UI or run the following to stop the service:

```bash
uv run python -m wispo.service --quit
```

Open the settings application without the background dictation service:

```bash
uv run python -m wispo.settings
```

### Command-line tools

The package script requires a subcommand:

```bash
# Print partial and final transcripts in the terminal.
uv run wispo transcribe

# Inject the final phrase through the desktop portal.
uv run wispo transcribe --inject

# Measure one 16 kHz, 16-bit PCM WAV file.
uv run wispo benchmark --wav /path/to/sample.wav
```

Equivalent module entry points are `python -m wispo.transcribe` and `python -m wispo.benchmark`. The benchmark is the appropriate source of local performance numbers; this README intentionally makes no universal latency or accuracy claim.

## Configuration

Wispo reads [`wispo.yaml`](wispo.yaml) from the project root at startup. The settings application writes the same file. The service can apply several settings while idle, but restart it after changes when in doubt—especially after changing device/runtime prerequisites.

```yaml
audio:
  sample_rate: 16000
  channels: 1
  device: auto
  frame_ms: 30

asr:
  backend: whisper_cpp
  model: small.en
  model_path: models/ggml-small.en.bin
  device: auto       # auto, sycl, or cpu
  quantization: fp16
  threads: 4
  language: en
  gpu_device: 0

streaming:
  chunk_ms: 1000
  overlap_ms: 250
  length_ms: 5000

vad:
  rms_threshold: 0.015
  noise_multiplier: 4.0
  hangover_ms: 700
  preroll_ms: 300
  highpass_hz: 80.0

hotkey:
  keys: [copilot]
```

See [the configuration reference](docs/configuration.md) for behavior, valid values, and tuning guidance. Add custom post-processing replacements under `dictionary`; the built-in file contains examples.

## Architecture

The core data flow is:

```text
Linux input events ──> hold-to-talk state
                                 │
ALSA / PipeWire microphone ─> float32 frames ─> VAD + preroll ─> rolling windows
                                                                    │
                                                            whisper.cpp bridge
                                                            (SYCL or CPU)
                                                                    │
                                     final utterance ─> dictionary replacement ─> portal keycodes
                                                                                         │
                                                                                focused application
```

The model remains loaded between utterances. Intermediate windows support terminal partials; when recording ends, Wispo transcribes the full utterance (up to 30 seconds) for the final result. See [Architecture](docs/architecture.md) for module ownership, lifecycle, and concurrency details.

## Development

Read [Contributing](CONTRIBUTING.md) before changing the code. In a headless shell, run the non-GUI tests with:

```bash
UV_CACHE_DIR=/tmp/wispo-uv-cache uv run pytest -q -k 'not hud'
```

The complete suite includes a HUD test that needs an available GTK display server. Run it locally with a display (or `xvfb-run`) rather than treating a headless GTK initialization failure as a product regression.

## License

This project is licensed under the [MIT License](LICENSE).
