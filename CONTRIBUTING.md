# Contributing to Wispo

Contributions are welcome for correctness, accessibility, documentation, portability, and performance. Please keep changes small enough to review and validate them on the environment they affect.

## Development setup

Follow the [README](README.md) through Python environment creation. Work that touches native inference also needs:

- a compatible checkout at `third_party/whisper.cpp`;
- Intel oneAPI DPC++, oneMKL, and Level Zero development/runtime dependencies;
- a successful `bash scripts/build_sycl.sh` run.

Do not commit model weights, native build output, virtual environments, or generated logs. They are local artifacts and are covered by `.gitignore`.

## Useful commands

```bash
# Run the headless-safe test suite. The explicit cache keeps runs from
# depending on a user-writable cache directory.
UV_CACHE_DIR=/tmp/wispo-uv-cache uv run pytest -q -k 'not hud'

# Run one focused test module.
UV_CACHE_DIR=/tmp/wispo-uv-cache uv run pytest -q tests/test_chunker.py

# Rebuild after changing native code or the whisper.cpp checkout.
bash scripts/build_sycl.sh

# Exercise terminal transcription.
uv run wispo transcribe

# Benchmark a known 16 kHz, 16-bit PCM WAV input.
uv run wispo benchmark --wav /path/to/sample.wav
```

`tests/test_hud.py` constructs GTK windows. It needs a graphical display and will fail in a plain headless shell. Use a local desktop session or an X virtual display such as `xvfb-run` when appropriate, then run the complete suite with `UV_CACHE_DIR=/tmp/wispo-uv-cache uv run pytest -q`. Do not treat a display-initialization failure as a product regression without checking the test environment.

## Change-specific checks

| Area changed | Expected validation |
| --- | --- |
| Configuration or pure Python logic | Relevant pytest module(s), plus the complete non-GUI suite |
| Audio/VAD/chunking | Unit tests and a recording using an actual microphone |
| `native/` or `third_party/whisper.cpp` | Rebuild, model load, a CPU transcription, and SYCL transcription when hardware is available |
| Portal injection | A real portal permission/session and insertion into a focused application |
| GTK settings/HUD | Manual interaction on a graphical desktop; check close/hide and quit behavior |
| Documentation | Verify commands, module names, paths, and claims against the source before submitting |

Do not report a benchmark number as a general result without recording the model, model file/quantization, device/backend, driver/runtime, audio fixture, and command used.

## Design constraints

### Preserve the process boundaries

Audio capture, hotkey reads, model inference, and portal insertion are blocking operations. GTK must remain on the GLib main thread; schedule UI changes with `GLib.idle_add` rather than calling GTK from a worker.

The model is deliberately kept resident between utterances. Do not move model loading into the hotkey path unless the user explicitly chooses that trade-off.

### Treat desktop integration as privileged and fallible

The application reads Linux input-device events and requests portal keyboard permission. Both may be unavailable, denied, or revoked. Surface actionable errors and keep the service recoverable; never silently fall back to a broader input-injection mechanism.

### Keep configuration compatible

`wispo.yaml` is validated by `config.py`. Add fields to the relevant dataclass and loader validation, preserve existing values when saving settings, add tests, and document user-facing behavior in `docs/configuration.md`.

### Keep native boundaries narrow

`native/wispo_asr.cpp` is the C ABI boundary. If a `whisper.cpp` API change is needed, contain it there and update the build/docs compatibility notes. Do not expose `whisper.cpp` internals throughout Python modules.

## Pull requests

1. Create a focused branch.
2. Describe the user-visible problem and the chosen solution.
3. Include validation performed and the environment for hardware/desktop testing.
4. Call out changes to dependencies, portal permissions, configuration formats, or model compatibility.
5. Keep commits readable; conventional prefixes such as `fix:`, `feat:`, `docs:`, and `test:` are welcome but not required.

If a change cannot be tested on a required device or desktop, say so plainly in the pull request rather than inferring success.
