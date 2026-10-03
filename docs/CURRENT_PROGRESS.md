# WhisperFlow — Current Progress & Model Handoff

> **Purpose:** This document is the comprehensive handoff and architectural summary of the Local WhisperFlow project. Any AI model or engineer continuing this work should read this file first alongside [`prd.md`](file:///home/faraz/Documents/Codes/my-wisper/prd.md) and [`CONTEXT.md`](file:///home/faraz/Documents/Codes/my-wisper/CONTEXT.md).

---

## 1. Project Overview & Architecture

**Local WhisperFlow** is a privacy-first, completely offline, low-latency, system-wide speech transcription tool for Linux. The user holds a global shortcut, speaks, and upon key release the transcribed text is inserted into the active focused application.

### Stage Progression

```text
[x] Stage 1: Transcription Engine (Local ASR, streaming audio, VAD, benchmarks) -> COMPLETE
[ ] Stage 2: System-Wide Text Injection (Global hotkey, active window text typing/paste) -> NEXT
[ ] Stage 3: Optional Local LLM Post-Processing (Final transcript cleanup only, never streaming)
[ ] Stage 4: Hardware & Model Optimization (Quantization, FP16 SYCL, RTF optimization)
[ ] Stage 5: Packaging & UX (System daemon, tray icon, CLI management)
```

> **Strict Architectural Rule:**  
> Subsystems must remain strictly decoupled. The ASR engine must **never** import hotkeys, text injectors, or LLMs. Each subsystem interacts only through defined interfaces.

---

## 2. Developer Reference Environment

* **OS:** Ubuntu 26.04 LTS (x86_64 Linux)
* **Desktop Session:** Wayland (`wayland-0`, `DISPLAY=:0`)
* **CPU:** Intel Core Ultra 5 225H (14 cores / 14 threads, AVX2, AVX_VNNI)
* **GPU:** Intel Arc 130T integrated GPU (Arrow Lake-P, Level Zero device: `Intel(R) Arc(TM) Graphics`)
* **RAM:** ~15 GB shared system memory
* **Audio Server:** PipeWire (ALSA layer)
  * Default capture hardware: `Arrow Lake cAVS Digital Microphone` (node parsed via `wpctl`)
  * Note: Default ALSA/session capture defaults to a silent Bluetooth loopback (ZEB-DUKE PRO); the app explicitly resolves and targets the hardware digital mic.
* **Input Permissions:** User `faraz` is in the `input` group; `/dev/input/event*` devices are directly readable.
* **Python Runtime:** Python 3.14 managed by `uv` (in `.venv`).

---

## 3. Toolchain & Native Build

* **Installed Packages:** `build-essential`, `cmake`, `pkg-config`, `libze1`, `libze-intel-gpu1`, `libze-dev`, `intel-opencl-icd`, `intel-oneapi-compiler-dpcpp-cpp` (2026.1), `intel-oneapi-mkl-devel-2026.1`.
* **ASR Engine Source:** `third_party/whisper.cpp` (built with SYCL enabled, `GGML_SYCL=ON`, `GGML_SYCL_DNN=OFF`, `GGML_SYCL_F16=OFF`, Intel `icx`/`icpx`, oneMKL).
* **Native C Shim:** [`native/whisperflow_asr.cpp`](file:///home/faraz/Documents/Codes/my-wisper/native/whisperflow_asr.cpp) compiled into [`native/build/libwhisperflow_asr.so`](file:///home/faraz/Documents/Codes/my-wisper/native/build/libwhisperflow_asr.so).
  * Exposes stable C ABI: `wf_load()`, `wf_transcribe()`, `wf_system_info()`, `wf_free()`.
  * Python interacts solely through `ctypes` in [`src/whisperflow/asr/whisper_cpp.py`](file:///home/faraz/Documents/Codes/my-wisper/src/whisperflow/asr/whisper_cpp.py).
* **Environment Re-exec Pattern:** `libggml-cpu.so` requires Intel OpenMP (`libiomp5.so`) and oneAPI Level Zero runtime libraries. Since `LD_LIBRARY_PATH` is only evaluated at process startup, [`src/whisperflow/asr/whisper_cpp.py`](file:///home/faraz/Documents/Codes/my-wisper/src/whisperflow/asr/whisper_cpp.py) dynamically inspects `/opt/intel/oneapi/` and re-execs Python once if runtime paths are absent.

---

## 4. Current Status: Stage 1 (Complete)

All criteria for **Stage 1 (Definition of Done)** in [`prd.md`](file:///home/faraz/Documents/Codes/my-wisper/prd.md#L771-L793) have been built and verified:

1. **Microphone Capture:** [`src/whisperflow/audio/capture.py`](file:///home/faraz/Documents/Codes/my-wisper/src/whisperflow/audio/capture.py) opens ALSA PCM directly (`libasound.so.2`) at 16 kHz mono float32 without third-party audio wrappers.
2. **Mic Detection:** [`src/whisperflow/audio/sources.py`](file:///home/faraz/Documents/Codes/my-wisper/src/whisperflow/audio/sources.py) identifies the real digital mic from `wpctl status` and passes `PIPEWIRE_PROPS={"target.object": ...}` to ALSA.
3. **Voice Activity Detection:** [`src/whisperflow/audio/vad.py`](file:///home/faraz/Documents/Codes/my-wisper/src/whisperflow/audio/vad.py) implements high-pass filtered RMS energy detection with dynamic noise estimation, hangover buffers, and a ring preroll buffer ([`buffer.py`](file:///home/faraz/Documents/Codes/my-wisper/src/whisperflow/audio/buffer.py)).
4. **ASR Models:** Models stored in `models/`:
   * `ggml-tiny.en.bin` (~75 MB)
   * `ggml-base.en.bin` (~141 MB, current default in [`whisperflow.yaml`](file:///home/faraz/Documents/Codes/my-wisper/whisperflow.yaml))
   * `ggml-small.en.bin` (~465 MB)
5. **GPU Acceleration & CPU Fallback:**
   * SYCL GPU backend runs on `Intel Arc 130T`.
   * CPU fallback runs via OpenMP/AVX2/AVX_VNNI.
6. **Streaming & Accumulation:**
   * [`src/whisperflow/asr/chunker.py`](file:///home/faraz/Documents/Codes/my-wisper/src/whisperflow/asr/chunker.py) handles sliding audio windows (1s chunks over 5s context).
   * [`src/whisperflow/asr/accumulator.py`](file:///home/faraz/Documents/Codes/my-wisper/src/whisperflow/asr/accumulator.py) handles overlap word deduplication and separates `committed` vs `partial` text hypotheses.
7. **Hold-to-Talk Terminal CLI:**
   * [`src/whisperflow/hotkey/linux.py`](file:///home/faraz/Documents/Codes/my-wisper/src/whisperflow/hotkey/linux.py) reads `/dev/input/event*` kernel structs directly for key-down/key-up without external X11/evdev package dependencies. Configured to **Left Ctrl** (`KEY_LEFTCTRL` = 29).
   * Runner: `uv run python -m whisperflow.transcribe`
8. **Performance Benchmarks:**
   * Benchmark CLI: `uv run python -m whisperflow.benchmark`
   * **CPU Fallback (`base.en`)**: **RTF = 0.09** (1,001 ms infer for 11.00 s audio, well below the 0.5 RTF threshold).
   * **Intel Arc SYCL (`base.en`)**: **RTF = 0.65** (7,113 ms infer for 11.00 s audio).
   * **Intel Arc SYCL (`tiny.en`)**: **RTF = 0.54** (5,941 ms infer for 11.00 s audio).
9. **Automated Tests:** 14/14 unit tests pass (`uv run pytest`).

---

## 5. Next Stage: Stage 2 (System-Wide Text Injection)

The goal of Stage 2 is to inject the final transcribed speech directly into the currently focused application window (browser, VS Code, terminal, editor).

### Requirements & Roadmap for Stage 2:
1. **Background Service / Global Hotkey Listener:**
   * Currently, `HoldKey` works in the foreground CLI via `/dev/input/event*`.
   * Stage 2 needs a reliable global shortcut listener that runs in the background.
2. **Text Injector Interface:**
   * Define an abstract injector contract:
     ```python
     class TextInjector(ABC):
         @abstractmethod
         def insert(self, text: str) -> None:
             ...
     ```
3. **Wayland-Compatible Text Injection on Ubuntu 26.04:**
   * Wayland restricts arbitrary synthetic keypresses and window focus manipulation.
   * Target approaches:
     - **Clipboard + Simulated Paste:** (e.g. `wl-copy` / Wayland data-control + `wtype` or `ydotool` / `uinput` for Ctrl+V / Shift+Insert).
     - **Direct Virtual Keyboard:** (`/dev/uinput` via Linux kernel virtual input device).
   * Preserve user clipboard state when using clipboard-based paste.
4. **Target Application Verification:**
   * Must verify injection into:
     - VS Code
     - Web browser (Chrome/Firefox text inputs)
     - Terminal (GNOME Terminal / Ptyxis / Alacritty)

---

## 6. Critical Technical Traps (Do NOT Redo)

1. **Do not use default ALSA capture device:** ALSA `default` points to Bluetooth loopback and outputs silence. Keep using `audio.device: auto` via [`sources.py`](file:///home/faraz/Documents/Codes/my-wisper/src/whisperflow/audio/sources.py).
2. **Do not install `evdev` pip package:** It requires `python3-dev` / `Python.h` which is not installed. Direct binary parsing with Python's standard `struct` in [`hotkey/linux.py`](file:///home/faraz/Documents/Codes/my-wisper/src/whisperflow/hotkey/linux.py) works reliably.
3. **Do not unload models between recordings:** Model initialization (`wf_load`) has a high startup latency (especially SYCL kernel warmup). The model must stay loaded in memory (`engine.stop()` must remain a no-op).
4. **Do not couple ASR with UI/Injectors:** Keep [`src/whisperflow/application.py`](file:///home/faraz/Documents/Codes/my-wisper/src/whisperflow/application.py) strictly focused on audio $\rightarrow$ text.
5. **Do not run oneAPI `setvars.sh` under `set -u`:** It crashes due to uninitialized internal variables in Intel's scripts.

---

## 7. Useful Diagnostic Commands

```bash
# Run unit tests
uv run pytest -q

# Run benchmark on GPU (default config)
uv run python -m whisperflow.benchmark

# Run benchmark on CPU fallback
uv run python -m whisperflow.benchmark --device cpu

# Run live hold-to-talk transcription in terminal (Hold Left Ctrl)
uv run python -m whisperflow.transcribe

# Verify Intel SYCL devices
sycl-ls

# Inspect PipeWire audio devices
wpctl status
```
