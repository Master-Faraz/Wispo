# Configuration reference

In a source checkout, Wispo loads `wispo.yaml` from the project root. The packaged launcher sets `WISPO_DATA_DIR` and uses `~/.local/share/wispo/wispo.yaml` by default, creating that user-writable copy on the first Settings save. Until then, it reads the packaged `/opt/wispo/wispo.yaml` template. It rejects unknown keys in `audio`, `asr`, `streaming`, `vad`, `hotkey`, and each `dictionary` item, so configuration mistakes fail early rather than being silently ignored.

The settings window writes this file with PyYAML. That means comments and original formatting are not preserved after a save. Keep a copy if hand-written comments matter to you.

## Complete example

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
  device: auto
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

dictionary:
  - phrase: github
    replacement: GitHub
    enabled: true
```

## `audio`

| Key | Meaning | Current behavior |
| --- | --- | --- |
| `sample_rate` | Requested capture and processing sample rate. | Whisper's pipeline is designed around 16000. Keep this at `16000`; the native engine measures audio duration using that rate. |
| `channels` | Number of logical channels. | Retained and validated, but capture is currently hard-coded to mono. Use `1`. |
| `device` | Microphone source choice. | `auto` uses `wpctl` to select a PipeWire source; `default` uses ALSA's default device; `pipewire` uses PipeWire's ALSA device; any other string is treated as a PipeWire node name. |
| `frame_ms` | Capture/VAD frame duration. | Positive, practical values are short durations such as `30`. Lower values mean more callbacks; larger values delay speech detection. |

The app requests S16_LE from ALSA and converts frames to floating point internally. A device is not guaranteed to support every requested rate; PipeWire can resample when configured through its ALSA plugin.

## `asr`

| Key | Meaning | Current behavior |
| --- | --- | --- |
| `backend` | Backend label. | Stored in configuration, but current application code always constructs the `whisper_cpp` backend. Keep `whisper_cpp`. |
| `model` | User-facing model name. | Used by the settings UI and when it updates `model_path`. It should match a downloaded model. |
| `model_path` | Model file path, relative to the project root or absolute. | Must point to an existing GGML `.bin` model file at service/model startup. Under the packaged launcher, relative paths resolve beneath `~/.local/share/wispo/`. |
| `device` | Inference selection. | `auto` picks SYCL when `sycl-ls` reports a Level Zero GPU, otherwise CPU (result is cached in `~/.local/state/wispo/device_cache.json` for 7 days to eliminate startup latency). `sycl` requires that GPU; `cpu` forces CPU selection, bypassing GPU memory bus contention entirely. |
| `quantization` | Display label for the selected model. | Currently displayed by terminal tools but not passed as an inference option. The model file itself determines its quantization. |
| `threads` | Whisper inference worker count. | Passed to the native bridge. Start with the default and benchmark before raising it. |
| `language` | Whisper language code. | Passed to the native bridge; `en` is the checked-in value. |
| `gpu_device` | SYCL GPU index. | Passed when SYCL is used. Keep `0` unless `sycl-ls` shows multiple usable devices. |

The settings UI recognizes the model catalogue in `src/wispo/models.py`. The download script can request other upstream `ggml-<name>.bin` paths, but a file downloaded outside the UI may not appear as a selectable catalogue entry.

## `streaming`

| Key | Meaning | Current behavior |
| --- | --- | --- |
| `chunk_ms` | Amount of new speech audio needed before an intermediate Whisper call. | Smaller values can produce partial text sooner but invoke inference more often. Must be positive. |
| `length_ms` | Maximum rolling-window length provided to an intermediate call. | Must be at least `chunk_ms`; the application uses the larger of the two values. |
| `overlap_ms` | Intended overlap description. | Stored and validated but not currently consumed by `SlidingChunker`. Overlap is presently an emergent result of retaining the most recent `length_ms` as each `chunk_ms` arrives. |

Intermediate calls are useful for terminal partials. The desktop service intentionally does not inject intermediate text; it injects only the final transcript after release.

## `vad`

| Key | Meaning | Tuning guidance |
| --- | --- | --- |
| `rms_threshold` | Minimum RMS energy that can count as speech. | Raise it if background noise triggers speech; lower it if a quiet voice is ignored. |
| `noise_multiplier` | Adaptive noise-floor multiplier. | Raise it for noisy rooms; lower it cautiously for quiet speech. |
| `hangover_ms` | How long quiet frames remain part of an active utterance. | Raise it to avoid splitting phrases across pauses; lower it for faster VAD endings. |
| `preroll_ms` | Audio retained before VAD first detects speech. | Helps preserve opening consonants. |
| `highpass_hz` | One-pole high-pass cutoff used only for VAD energy measurement. | Set to `0` to disable it; audio sent to Whisper remains unfiltered. |

The hold-to-talk service forces speech while the chord is held, so changing these values mainly affects the non-forced flow and visual/audio-level behavior. They still matter for direct pipeline use and for post-release boundaries.

## `hotkey`

`keys` is a non-empty list of names that must all be held simultaneously. It also accepts a `+`-separated string when parsed manually, for example `"ctrl+super"`.

Supported names include `shift`, `ctrl`, `alt`, `super`, `copilot`, `f23`, letters, digits, `space`, `enter`, `tab`, `esc`, and `backspace`. Hardware and kernel key mapping must expose the selected key on a readable `/dev/input/event*` device. The actual mapping is defined in `src/wispo/hotkey/linux.py`.

Examples:

```yaml
hotkey:
  keys: [ctrl, super]
```

```yaml
hotkey:
  keys: [copilot]
```

A bare `super` can conflict with GNOME's overview shortcut. Wispo attempts special handling for that exact configuration, but `ctrl+super` or a dedicated key is usually less disruptive.

## `dictionary`

Each item has `phrase`, `replacement`, and optional `enabled` fields. The dictionary runs after final transcription and before portal insertion:

```yaml
dictionary:
  - phrase: tensor flow
    replacement: TensorFlow
    enabled: true
```

It is replacement post-processing, not model training, a prompt, or a decoding-time hotword feature. Disable an entry by setting `enabled: false`; retain the entry if you want to preserve it in the settings UI.

## Applying changes

Saving from the settings UI queues a configuration update. When not recording, the service can update its dictionary, VAD values, microphone source, and shortcut. Changing model name/path triggers a reload. The implementation does not currently reload every ASR or streaming field in place, so restart the desktop service after editing `wispo.yaml` manually or after changing threads, device mode, language, GPU index, or streaming values.
