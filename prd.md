# Local WhisperFlow

## 1. Product Overview

Local WhisperFlow is a privacy-first, offline, system-wide voice transcription application for Linux.

The application allows the user to:

1. Hold a configurable keyboard shortcut.
2. Speak naturally.
3. Continuously transcribe speech in near real time.
4. Release the shortcut.
5. Finalize the transcription.
6. Optionally process the final transcript through a local LLM.
7. Insert the resulting text into the currently focused application at the active cursor.

The application is designed primarily for **low-resource local hardware** and should take advantage of available GPU acceleration when possible.

### Primary target hardware

The developer's reference machine:

* Ubuntu Linux
* Intel Core Ultra 5 225H
* Intel Arc 130T integrated GPU

The Intel Arc GPU should be the preferred ASR acceleration path on supported systems. whisper.cpp provides an Intel SYCL backend and supports Intel Arc and Intel integrated Arc GPUs on Linux.

CPU inference must remain available as a fallback.

---

# 2. Product Goals

## Primary Goals

* Near-real-time speech transcription.
* Very low latency.
* Low CPU and RAM consumption.
* GPU acceleration when available.
* Completely local/offline operation.
* System-wide text insertion.
* Hold-to-talk interaction.
* Optional post-processing with a local LLM.
* Modular architecture allowing ASR and LLM implementations to be replaced independently.
* Support lower-end hardware through smaller/quantized models.

## Non-Goals for V1

* Cloud transcription.
* Cloud LLM APIs.
* Mobile applications.
* Web application.
* Multi-user functionality.
* Complex GUI.
* Conversation history.
* Automatic actions based on spoken commands.

---

# 3. Development Philosophy

Development must be divided into independent stages.

Each stage must produce a working, testable application before the next stage begins.

```text
Stage 1
Transcription Engine
        ↓
Stage 2
System-wide Text Injection
        ↓
Stage 3
LLM Post-processing
        ↓
Stage 4
Hardware / Model Optimization
        ↓
Stage 5
Packaging & UX
```

The application must remain usable after every completed stage.

---

# 4. High-Level Architecture

```text
                         ┌──────────────┐
                         │  Hotkey      │
                         │  Controller  │
                         └──────┬───────┘
                                │
                         key down / up
                                │
                                ▼
┌─────────────┐          ┌──────────────┐
│ Microphone  │─────────►│ Audio Engine │
└─────────────┘          └──────┬───────┘
                                 │
                                 ▼
                          ┌─────────────┐
                          │     VAD     │
                          └──────┬──────┘
                                 │
                           speech chunks
                                 │
                                 ▼
                          ┌─────────────┐
                          │ ASR Engine  │
                          │ whisper.cpp │
                          └──────┬──────┘
                                 │
                          partial results
                                 │
                                 ▼
                       ┌──────────────────┐
                       │ Transcript       │
                       │ Accumulator      │
                       └────────┬─────────┘
                                │
                         key released
                                │
                                ▼
                       ┌──────────────────┐
                       │ Finalize ASR     │
                       └────────┬─────────┘
                                │
                                ▼
                       ┌──────────────────┐
                       │ Optional LLM     │
                       │ Post-processor   │
                       └────────┬─────────┘
                                │
                                ▼
                       ┌──────────────────┐
                       │ Text Injector    │
                       └──────────────────┘
                                │
                                ▼
                         Active application
```

---

# 5. Stage 1 — Transcription Engine

## Objective

Build a standalone local transcription engine that can capture microphone audio and continuously produce text while the user is speaking.

**No LLM.**

**No system-wide text injection.**

**No complicated GUI.**

The only goal is to make transcription fast, reliable, and measurable.

## 5.1 Technology

Primary language:

* Python 3.12+

ASR engine:

* `whisper.cpp`

GPU backend:

* Intel SYCL / Level Zero

CPU fallback:

* whisper.cpp CPU backend

Audio:

* Linux microphone input
* 16 kHz
* mono
* PCM audio

VAD:

* whisper.cpp VAD initially
* Silero VAD may be evaluated later

whisper.cpp already includes a real-time microphone streaming example and sliding-window/VAD mode.

## 5.2 Model Strategy

The application must support configurable models.

Initial benchmark candidates:

```text
tiny
tiny.en
base
base.en
small
small.en
```

Quantized models should be evaluated because whisper.cpp supports integer quantization to reduce memory and potentially improve processing efficiency.

For an English-first low-resource implementation, start benchmarking with:

```text
small.en
```

and smaller variants.

The final default model must be selected based on actual measurements rather than parameter count alone.

## 5.3 Hardware Selection

Default behavior:

```text
Detect hardware
      │
      ├── Intel Arc GPU available
      │       ↓
      │   Use SYCL GPU
      │
      └── GPU unavailable
              ↓
          Use CPU
```

The Intel Core Ultra 5 225H contains an Intel Arc 130T integrated GPU with 7 Xe cores and up to 63 INT8 TOPS.

GPU acceleration must be implemented behind an abstraction so the Python application does not depend directly on Intel-specific logic.

## 5.4 Real-Time Processing

The engine must continuously process audio while the hotkey is held.

Conceptual flow:

```text
Microphone
    ↓
Audio buffer
    ↓
VAD
    ↓
Speech detected
    ↓
Audio window
    ↓
Whisper
    ↓
Partial transcript
    ↓
Transcript accumulator
```

Use a sliding/rolling audio window rather than waiting for the entire recording.

whisper.cpp's existing streaming implementation samples audio continuously and supports sliding-window VAD transcription.

## 5.5 Transcript Accumulation

The engine must distinguish between:

* committed text
* current partial transcription

Example:

```text
Committed:
"Create a new function"

Partial:
"called calculate"
```

When a new ASR result arrives, the engine must merge it without duplicating overlapping words.

The implementation should eventually support:

```text
audio chunk overlap
        ↓
ASR result
        ↓
overlap detection
        ↓
deduplication
        ↓
committed transcript
```

## 5.6 Hotkey State

Stage 1 may initially use a simple development trigger.

The final Stage 1 engine must expose:

```text
start_recording()
stop_recording()
```

without coupling the ASR engine to the keyboard implementation.

---

# 6. Stage 1 Performance Requirements

The transcription engine must be benchmarked.

Measure:

* Model loading time.
* Idle RAM.
* Model RAM.
* Peak RAM.
* CPU utilization.
* GPU utilization.
* GPU memory/shared memory.
* Audio-to-text latency.
* Real-Time Factor (RTF).
* Transcription accuracy.
* Startup time.
* Time from speech ending to final transcript.

## Target

The engine should comfortably maintain:

```text
RTF < 0.5
```

on the reference machine.

The long-term goal is to support lower-end machines with a smaller model/quantization profile.

---

# 7. Stage 1 Deliverable

A command-line application:

```bash
python -m whisperflow.transcribe
```

Expected behavior:

```text
WhisperFlow Transcription Engine

Backend: Intel SYCL
Device: Intel Arc 130T
Model: small.en
Quantization: <selected>

Listening...

[partial] hello
[partial] hello this is
[partial] hello this is a test

Final:
hello this is a test
```

No LLM.

No text injection.

---

# 8. Stage 2 — System-Wide Text Injection

## Objective

Allow the user to hold a global keyboard shortcut while speaking and insert the final transcript into the currently focused application.

## Interaction

```text
HOTKEY DOWN
    ↓
Start recording
    ↓
Continuous transcription
    ↓
HOTKEY UP
    ↓
Stop recording
    ↓
Finalize transcription
    ↓
Insert text
```

## Requirements

* Global keyboard shortcut.
* Configurable shortcut.
* Key-down starts recording.
* Key-up stops recording.
* Text insertion into active application.
* No application-specific integrations initially.

## Text Injection Abstraction

```python
class TextInjector:
    def insert(self, text: str) -> None:
        ...
```

Initial implementation may use clipboard + paste.

Future implementations may support other Linux input mechanisms.

---

# 9. Stage 3 — Optional LLM Processing

## Objective

Add an optional local LLM that processes the **final transcript only**.

The LLM must NOT run continuously during speech.

```text
Speech
  ↓
Streaming ASR
  ↓
Final transcript
  ↓
Optional LLM
  ↓
Final text
  ↓
Text injection
```

## LLM Responsibilities

Potential operations:

* Remove filler words.
* Fix punctuation.
* Correct obvious transcription errors.
* Format text.
* Structure output.
* Convert speech into code-friendly text.
* Produce JSON when explicitly configured.

## LLM Interface

```python
class TextProcessor:
    def process(self, text: str) -> str:
        ...
```

Possible backend:

* Ollama
* llama.cpp
* OpenAI-compatible local inference server

The ASR engine must not depend on a particular LLM backend.

## Configuration

```yaml
llm:
  enabled: false
  backend: ollama
  model: <model-name>
```

---

# 10. Stage 4 — Hardware Optimization

After functional development is complete, benchmark and optimize.

Profiles:

```text
LOW
 └── tiny / tiny.en
 └── CPU
 └── aggressive quantization

BALANCED
 └── small.en
 └── Intel GPU when available
 └── moderate quantization

QUALITY
 └── larger model
 └── GPU
```

The application should eventually automatically select an appropriate configuration.

---

# 11. Stage 5 — Packaging and UX

Only after the core engine is stable.

Features:

* Background daemon.
* System tray indicator.
* Configuration file.
* Model management.
* Hardware detection.
* Startup on login.
* Logging.
* Diagnostics.
* Benchmark command.
* Installation package.

Example:

```bash
whisperflow start
whisperflow stop
whisperflow status
whisperflow benchmark
whisperflow config
```

---

# 12. Proposed Python Project Structure

```text
whisperflow/
│
├── pyproject.toml
├── README.md
├── LICENSE
│
├── src/
│   └── whisperflow/
│       │
│       ├── __init__.py
│       ├── cli.py
│       ├── config.py
│       │
│       ├── audio/
│       │   ├── capture.py
│       │   ├── buffer.py
│       │   └── vad.py
│       │
│       ├── asr/
│       │   ├── base.py
│       │   ├── whisper_cpp.py
│       │   ├── device.py
│       │   ├── chunker.py
│       │   └── accumulator.py
│       │
│       ├── hotkey/
│       │   ├── base.py
│       │   └── linux.py
│       │
│       ├── processing/
│       │   ├── base.py
│       │   └── llm.py
│       │
│       ├── output/
│       │   ├── base.py
│       │   └── desktop.py
│       │
│       └── application.py
│
└── tests/
    ├── test_audio.py
    ├── test_chunker.py
    ├── test_accumulator.py
    └── test_config.py
```

---

# 13. Core Interfaces

Keep components loosely coupled.

```python
class ASREngine:
    def start(self) -> None:
        ...

    def transcribe_chunk(self, audio) -> str:
        ...

    def finalize(self) -> str:
        ...

    def stop(self) -> None:
        ...
```

```python
class AudioCapture:
    def start(self) -> None:
        ...

    def read(self):
        ...

    def stop(self) -> None:
        ...
```

```python
class TextProcessor:
    def process(self, text: str) -> str:
        ...
```

```python
class TextInjector:
    def insert(self, text: str) -> None:
        ...
```

This allows each subsystem to be replaced without rewriting the application.

---

# 14. Configuration

Example:

```yaml
audio:
  sample_rate: 16000
  channels: 1

asr:
  backend: whisper_cpp
  model: small.en
  device: auto
  quantization: auto

streaming:
  chunk_ms: 1000
  overlap_ms: 250

hotkey:
  keys:
    - ctrl
    - space

llm:
  enabled: false
  backend: ollama
  model: ""

output:
  backend: clipboard
```

---

# 15. Testing Strategy

## Unit Tests

Test independently:

* Audio buffer.
* Chunk creation.
* Overlap handling.
* Transcript merging.
* Configuration.
* Hardware detection.

## Integration Tests

Test:

```text
Microphone
    ↓
VAD
    ↓
ASR
    ↓
Accumulator
```

## Hardware Tests

At minimum:

1. Intel Arc GPU.
2. CPU-only machine.
3. Low-memory machine.

## Accuracy Tests

Create a fixed speech test set containing:

* Normal speech.
* Fast speech.
* Technical terminology.
* Programming terminology.
* Numbers.
* Punctuation.
* Short phrases.
* Long phrases.
* Background noise.

Compare models and quantization levels.

---

# 16. Performance Benchmark Command

Implement:

```bash
whisperflow benchmark
```

Output:

```text
WhisperFlow Benchmark
──────────────────────────────

Hardware
CPU: Intel Core Ultra 5 225H
GPU: Intel Arc 130T
RAM: 16 GB

Model: small.en
Backend: whisper.cpp
Device: Intel SYCL

Model load:       1.42 s
Peak RAM:         XXX MB
GPU memory:       XXX MB
CPU utilization:  XX %
GPU utilization:  XX %
RTF:              X.XX
First result:     XXX ms
Final result:     XXX ms
```

The benchmark must be used to select the default model/backend.

---

# 17. Definition of Done — Stage 1

Stage 1 is complete only when:

* [ ] Microphone capture works.
* [ ] 16 kHz mono audio works.
* [ ] VAD works.
* [ ] Whisper model loads successfully.
* [ ] Intel Arc GPU backend works on the reference machine.
* [ ] CPU fallback works.
* [ ] Streaming transcription works.
* [ ] Partial transcripts are produced.
* [ ] Duplicate overlapping text is handled.
* [ ] Final transcript is produced correctly.
* [ ] Model remains loaded between recordings.
* [ ] RAM usage is measured.
* [ ] CPU usage is measured.
* [ ] GPU usage is measured.
* [ ] Latency is measured.
* [ ] Benchmark command works.
* [ ] No LLM dependency exists.

Only after all items are complete should Stage 2 begin.

---

# 18. Definition of Done — Stage 2

* [ ] Global hotkey works.
* [ ] Key-down starts recording.
* [ ] Key-up stops recording.
* [ ] Final transcript is generated.
* [ ] Transcript can be inserted into the active application.
* [ ] Works in at least VS Code, browser text field, and terminal.
* [ ] ASR engine remains independent from input injection.

---

# 19. Definition of Done — Stage 3

* [ ] LLM processing is optional.
* [ ] LLM is never invoked during recording.
* [ ] Final transcript is passed to LLM after key release.
* [ ] LLM output can be inserted into active application.
* [ ] LLM backend is replaceable.
* [ ] LLM can be disabled without changing ASR behavior.

---

# 20. Important Architectural Rule

The most important design constraint is:

> **ASR, LLM, and text injection must be independent subsystems.**

The application should be able to run:

```text
ASR only
```

or:

```text
ASR → Text Injection
```

or:

```text
ASR → LLM → Text Injection
```

without modifying the ASR engine itself.

This makes the project fast to develop, easy to benchmark, and scalable to additional models/backends later.

---

# 21. Initial Development Order

Implement exactly in this order:

```text
1. Project setup
      ↓
2. Microphone capture
      ↓
3. Audio buffering
      ↓
4. VAD
      ↓
5. whisper.cpp integration
      ↓
6. Intel SYCL GPU execution
      ↓
7. CPU fallback
      ↓
8. Streaming/chunked transcription
      ↓
9. Transcript merging
      ↓
10. Benchmarking
      ↓
11. Stage 1 complete
      ↓
12. Global hotkey
      ↓
13. Text injection
      ↓
14. Stage 2 complete
      ↓
15. LLM abstraction
      ↓
16. Local LLM integration
      ↓
17. Stage 3 complete
      ↓
18. Hardware/model auto-selection
      ↓
19. Packaging
```

## First milestone

The first development milestone should be extremely small:

```bash
python -m whisperflow.transcribe
```

Hold nothing. No hotkey yet.

It should simply:

**capture microphone → detect speech → transcribe continuously → print partial text → print final text.**

Once that is reliable, everything else can be layered on top.
