# WhisperFlow — agent context

This file is the handoff for a new chat. Read it before changing the project. The product spec is `prd.md`. This file is what was actually built, what was measured, and what is still open.

The work so far is **Stage 1 only**: a local transcription engine. There is no LLM, no system-wide text injection, and no packaging.

## Product

Local WhisperFlow is an offline, system-wide voice transcription app for Linux. The user holds a key, speaks, and the final text should eventually be inserted into the focused application. Development is staged. Each stage must stay usable before the next one starts.

```text
Stage 1  Transcription engine          <- current work
Stage 2  System-wide text injection
Stage 3  Optional local LLM on the final transcript only
Stage 4  Hardware / model profiles
Stage 5  Packaging and UX
```

Hard rule from the PRD: ASR, LLM, and text injection stay independent. The ASR engine must not import a hotkey, an LLM, or a text injector.

## This machine

* Ubuntu 26.04 LTS
* Intel Core Ultra 5 225H (14 threads)
* Intel Arc integrated GPU, reported as Arrow Lake-P Arc 130T / "Intel(R) Arc(TM) Graphics"
* About 15 GB RAM
* Wayland session (`wayland-0`), with `DISPLAY=:0`
* User `faraz` is in the `input` group, so `/dev/input/event*` is readable
* Laptop keyboard node: `/dev/input/event2` (`AT Translated Set 2 keyboard`)
* Audio is PipeWire. Built-in mics:
  * `Arrow Lake cAVS Digital Microphone` — this is the one the app uses
  * `Arrow Lake cAVS Stereo Microphone`
* A Bluetooth device, ZEB-DUKE PRO, is the default output. Its loopback was the default capture path and returns silence.

Python is 3.14, managed by `uv`. The virtualenv is `.venv`. Run commands with `uv run`.

## Toolchain that is installed

Do not install the full `intel-oneapi-toolkit`. The compiler package plus oneMKL is enough.

Already installed:

* `build-essential`, `cmake`, `pkg-config`
* `libze1`, `libze-intel-gpu1`, `libze-dev`, `intel-opencl-icd`
* `intel-oneapi-compiler-dpcpp-cpp` 2026.1 (`icpx` / `icx`)
* `intel-oneapi-mkl-devel-2026.1`

`icpx` is not on `PATH` until oneAPI is sourced. `scripts/build_sycl.sh` does that.

Verified with `sycl-ls`:

```text
[level_zero:gpu] Intel(R) Arc(TM) Graphics
[opencl:gpu]     Intel(R) Arc(TM) Graphics
[opencl:cpu]     Intel(R) Core(TM) Ultra 5 225H
```

whisper.cpp's SYCL CMake file requires oneMKL (`MKLConfig.cmake`) on Intel. Configure failed until `intel-oneapi-mkl-devel-2026.1` was installed. Level Zero headers (`libze-dev`) are also installed. oneDNN was turned off (`-DGGML_SYCL_DNN=OFF`) because the full toolkit is not installed.

## What was built

The original `uv` scaffold was `my_wisper`. It was replaced by the `whisperflow` package under `src/whisperflow/`.

### GPU build

* `third_party/whisper.cpp` is a shallow clone of ggml-org/whisper.cpp (1.9.4-dev at the time of the clone).
* `scripts/build_sycl.sh` configures and builds it, then compiles `native/whisperflow_asr.cpp` into `native/build/libwhisperflow_asr.so`.
* CMake flags: `GGML_SYCL=ON`, `GGML_SYCL_DNN=OFF`, `GGML_SYCL_F16=OFF`, shared libraries, `icx` / `icpx`, Release.
* Build directory: `third_party/whisper.cpp/build-sycl/`. Binary: `build-sycl/bin/whisper-cli`.
* The wrapper's RPATH points at the whisper build and `/opt/intel/oneapi/compiler/2026.1/lib`.

Build-script traps already hit:

* `set -u` crashes Intel's `vars.sh` because it unsets `OCL_ICD_FILENAMES` and then reads it. Nounset is disabled only while sourcing.
* Sourcing `setvars.sh` a second time in the same shell exits 3. The script skips it when `ONEAPI_ROOT` is already set, then sources MKL by itself if `MKLROOT` is empty. An environment sourced before MKL was installed does not contain MKL.
* CMake is passed `-DMKL_DIR` pointing at `MKLConfig.cmake`.

### Python engine

Entry points:

```bash
uv run python -m whisperflow.transcribe
uv run python -m whisperflow.benchmark
```

Data flow while Left Ctrl is held:

```text
PipeWire digital mic
  -> 16 kHz mono float32 frames (30 ms)
  -> sliding window (every 1 s, last 5 s)
  -> libwhisperflow_asr.so -> whisper.cpp on the Arc GPU (SYCL)
  -> word-overlap transcript merge
  -> [partial] lines
release Left Ctrl
  -> one more whisper pass on the whole utterance (up to 30 s)
  -> Final:
```

The model is loaded once and kept loaded between holds. `stop()` does not unload it.

Hold-to-talk is wired in `cli.py`. `application.py` does not import the keyboard code. While the key is held, frames are forced through as speech so the energy VAD cannot drop the phrase.

### Why the microphone is not "default"

ALSA `default` and ALSA `pipewire` with no target both recorded peak 0. `wpctl` showed the session default following the Bluetooth sink loopback, not a microphone. `audio/sources.py` parses `wpctl status`, prefers the Digital Microphone, and capture sets `PIPEWIRE_PROPS={"target.object": <node name>}` before opening the `pipewire` ALSA device. A direct read of that mic had a non-zero signal (peak about 0.32).

Config key: `audio.device: auto` in `whisperflow.yaml`.

### Why Python restarts itself

`libggml-cpu.so` needs `libiomp5.so` from the oneAPI compiler directory. A normal terminal does not search that directory, and a parent library's RUNPATH is not used for libraries that dependency loads. Importing `whisperflow.asr.whisper_cpp` adds the compiler, oneMKL, TBB, TCM, and UMF library directories to `LD_LIBRARY_PATH` and re-executes the process once. `LD_LIBRARY_PATH` is only read at process start, so setting it later does not work. Without the Unified Runtime / UMF directories, SYCL reports "No device of requested type available" even though the Arc GPU is present.

### Hold key

Terminals do not report key-up. `hotkey/linux.py` reads `/dev/input/event*` directly (no `evdev` package: that package needs `Python.h`, which is not installed). The default key is **Left Ctrl** (`KEY_LEFTCTRL` = 29).

Right Ctrl was the first default. A test run never printed `Recording...`, so that key was never seen. This laptop's Ctrl key is the left one. Left Ctrl+C still quits the process, so the user must hold Left Ctrl alone while speaking.

### Model

`models/ggml-tiny.en.bin` (about 75 MB), Hugging Face `ggerganov/whisper.cpp`. Quantization in the config is `fp16` (the upstream ggml file, not an integer quant). The PRD's first benchmark candidate was `small.en`. The user chose `tiny.en` so the pipeline could be proved first. Larger and quantized models have not been benchmarked.

## Measurements already taken

`whisper-cli` on `third_party/whisper.cpp/samples/jfk.wav` used device `SYCL0` / `Intel Arc Graphics` and produced:

```text
And so my fellow Americans ask not what your country can do for you, ask what you can do for your country.
```

First GPU encode was about 16.6 s (SYCL kernel warmup). The next run was about 7.6 s total for 11 s of audio.

`uv run python -m whisperflow.benchmark` on the same file, after warmup:

```text
Model load:       0.28 s
Idle RAM:         34 MB
Model RAM:        354 MB
Peak RAM:         666 MB
GPU memory:       shared with system RAM (integrated Arc)
CPU utilization:  100 %
GPU utilization:  6 %
RTF:              0.54
Audio:            11.00 s
Infer:            5941 ms
```

The transcript was correct. The Stage 1 target is RTF under 0.5, so this run is slightly slow. CPU at 100% means a lot of the time is still CPU work. The 6% GPU figure is an estimate from the i915 RC6 idle counter (`metrics.py`), not a vendor GPU profiler. Do not treat 6% as proof that SYCL is unused. The `whisper-cli` log explicitly said `using SYCL0 backend`.

GPU memory is shared system RAM. There is no separate VRAM counter for this iGPU.

## Layout

```text
prd.md                         product spec
CONTEXT.md                     this handoff
whisperflow.yaml               stage 1 settings
pyproject.toml                 package name whisperflow, uv
scripts/build_sycl.sh          SYCL build plus the Python wrapper
native/whisperflow_asr.cpp     stable C API over whisper.cpp
native/build/libwhisperflow_asr.so
third_party/whisper.cpp/       clone and build-sycl/
models/ggml-tiny.en.bin
src/whisperflow/
  cli.py                       hold-to-talk CLI and benchmark CLI
  transcribe.py                python -m whisperflow.transcribe
  benchmark.py                 python -m whisperflow.benchmark
  application.py               mic -> windows -> whisper -> text
  config.py
  metrics.py
  audio/capture.py             ALSA 16 kHz mono float
  audio/sources.py             PipeWire mic selection
  audio/buffer.py              preroll tail
  audio/vad.py                 frame energy VAD
  asr/whisper_cpp.py           ctypes wrapper, Intel library path restart
  asr/device.py                sycl-ls -> SYCL or CPU
  asr/chunker.py               overlapping windows
  asr/accumulator.py           overlap word merge
  hotkey/linux.py              Left Ctrl hold via evdev nodes
tests/                         buffer, VAD, chunker, merge, config, device, mic parser
```

Ignored by git: `.venv`, `third_party/whisper.cpp/build-sycl/`, `native/build/`, `models/*.bin`.

## How to test

```bash
uv run python -m whisperflow.benchmark
uv run python -m whisperflow.transcribe
```

Benchmark does not need the microphone. A good result is the Kennedy sentence above.

For the microphone: hold **Left Ctrl** (lower left of the keyboard), speak for at least a second, release. Expected:

```text
Recording...
[partial] ...

Final:
...
```

Partials wait for about one second of audio (`streaming.chunk_ms`). Release runs the final pass. Ctrl+C quits. Pressing C while Left Ctrl is held also quits, because that is Ctrl+C.

Unit tests, no GPU and no microphone:

```bash
uv run pytest -q
```

14 tests were passing after the mic-selection and keyboard-bitmap tests were added.

## Decisions

* First GPU build uses SYCL, not Vulkan, because the PRD names Intel SYCL.
* CPU remains the fallback when `sycl-ls` shows no Level Zero GPU, or when `--device cpu` is passed. The same `.so` is used. `use_gpu=0` selects CPU.
* `tiny.en` first, then larger models. Not done yet.
* Flash attention is off in the C wrapper. SYCL flash attention was left off for the first GPU path.
* No hotkey library. `/dev/input` is read with `struct`.
* The keyboard watcher is only in the CLI. Stage 2 should keep that split and add a `TextInjector`, not fold injection into the ASR engine.

## Not done

* Stage 1 RTF is 0.54 on `tiny.en`, not yet under 0.5. FP16 SYCL (`GGML_SYCL_F16`) and integer quants were not tried.
* No accuracy set beyond the single JFK sample.
* No system-wide paste into the focused app (Stage 2).
* No LLM (Stage 3). The LLM must run only on the final transcript, never during recording.
* No tray, daemon, or package (Stage 5).
* Live speech through Left Ctrl was not confirmed in a successful transcript during this chat. The mic device was confirmed to carry signal, and the file benchmark was confirmed on the GPU. The user still needs to hold Left Ctrl and speak.

## Do not redo

* Do not switch the package back to `my_wisper`.
* Do not point capture at ALSA `default`. It is silent here.
* Do not `pip install evdev` unless `python3-dev` exists. The current reader does not need it.
* Do not source `setvars.sh` under `set -u`.
* Do not assume `LD_LIBRARY_PATH` can be fixed after the Python process has started.
