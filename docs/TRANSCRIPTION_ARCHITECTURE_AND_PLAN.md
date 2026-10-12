# wispero — Transcription Architecture, Lifecycle & Optimization Plan

> **Document Version:** 1.0.0  
> **Date:** October 12, 2026  
> **Status:** Architecture Review & Proposed Implementation  
> **Target Scope:** Audio Capture, Whisper Inference Lifecycle, HUD Audio Reactivity, and Push-to-Talk Optimization

---

## 1. Executive Summary

WhisperFlow is a local, privacy-first, ultra-low-latency speech transcription system for Linux (specifically optimized for Intel Core Ultra + Intel Arc integrated GPU via oneAPI/SYCL).

### User-Facing Product Model

The intended user experience is **Pure Push-to-Talk Dictation**:

1. **Hold Hotkey:** The floating liquid-glass pill HUD appears immediately above the dock; audio waveform bars react dynamically to the user's voice. Real-time text preview is intentionally **disabled** to avoid distraction.
2. **Speak:** Speech audio is buffered continuously into memory.
3. **Release Hotkey:** The HUD indicates "Processing" with animated ripple dots; the entire spoken sentence is transcribed in one single, high-accuracy pass, formatted through custom dictionary replacements, and typed instantly into the active window.

---

## 2. Full Transcription Lifecycle (How It Works Today)

Understanding the current execution flow is divided into four distinct phases:

```text
[ Application Launch ]
       │
       ├── 1. GPU SYCL Model Warmup (Dummy pass JIT-compiles kernels)
       ├── 2. Audio Capture Stream Warmup (Pre-opens ALSA/PipeWire mic)
       └── 3. HUD GTK4 Surface Realization (Pre-allocates ARGB surface)
       │
[ Hotkey Pressed (Hold) ]
       │
       ├── HUD opens: "Listening" state
       └── Audio loop runs every frame (~30ms):
             ├── Calculates RMS level ──► drives HUD waveform bars
             └── [CURRENT BEHAVIOR] Pushes audio into SlidingChunker
                   └── Runs Whisper inference every 1s ──► RESULT DISCARDED
       │
[ Hotkey Released (Unhold) ]
       │
       ├── HUD switches to "Processing"
       ├── App drains remaining microphone buffer
       ├── Runs SINGLE-PASS finalization over the entire recording
       ├── Discards all 1s chunk guesses from earlier
       ├── Filters out [BLANK_AUDIO] or silent taps
       ├── Applies Dictionary replacements
       └── Types final text into focused application via Portal
```

---

### Phase A: Application Initialization & Pre-Warming

To eliminate cold-start lag and prevent the first spoken word from being missed, the system executes three critical warmups on startup:

1. **Model GPU Warmup (`WhisperCppEngine.warmup()` in `whisper_cpp.py`):**
   - Intel Arc GPU via oneAPI / SYCL JIT-compiles device kernels on first execution, which normally takes **7 to 15 seconds**.
   - During startup, a dummy silent 8000-sample pass is executed through `wf_transcribe()`. When the user makes their first dictation, inference is instantaneous (~0.15s – 0.3s).
2. **Audio Capture Stream Warmup (`load_model()` in `application.py`):**
   - Linux audio servers (PipeWire / ALSA) often suffer a 50–150ms buffer allocation delay on initial stream creation.
   - On startup, the mic stream is pre-opened (`capture.read(timeout=0.1)`) and closed, ensuring the hardware endpoint (`Arrow Lake cAVS Digital Mic`) is primed and zero syllables are truncated on the first keypress.
3. **GTK4 HUD Surface Realization (`FloatingPillHUD._init_window()` in `hud.py`):**
   - GTK4 windows create surfaces lazily. By calling `self._window.realize()` at initialization, the transparent 32-bit ARGB surface and click-through input regions are pre-registered with Wayland/X11, guaranteeing the HUD pops up on the first hotkey press with zero visual stutter.

---

### Phase B: Hotkey Press & Hold (Speech Capture)

When the global hotkey (e.g., `Super`, `Copilot`, or `Ctrl`) is pressed and held:

1. `service.py` detects `hold.held.is_set()` and calls:
   - `app.start_recording()`
   - `hud.start_listening()`
2. In the listen loop, `app.read_once(force_speech=True)` reads incoming 16 kHz mono float32 PCM frames.
3. **Audio Level Calculation:**
   ```python
   combined = np.concatenate([frame, *extra_frames]) if extra_frames else frame
   if combined.size:
       self.last_audio_level = float(np.sqrt(np.mean(combined ** 2)))
   ```
   This RMS amplitude is passed directly to `hud.set_audio_level()`, which interpolates 23 dynamic waveform bars rendered with Cairo.
4. **Current Background Chunk Inference:**
   - Audio frames are pushed into `SlidingChunker(chunk_ms=1000, length_ms=5000)`.
   - Every 1000ms, `engine.transcribe_chunk(window)` runs Whisper inference in the background.
   - `TranscriptAccumulator.merge()` calculates partial strings.
   - **Crucial Finding:** In `service.py`, this partial text is completely ignored because `service.py` only consumes `app.last_audio_level`.

---

### Phase C: Hotkey Release (Unhold & Finalization)

When the user unholds the key:

1. `service.py` detects key release:
   - Calls `hud.start_processing()` (rippling wave dot animation).
   - Calls `text = app.stop_recording()`.
2. Inside `TranscriptionApp._finish_utterance()` (`application.py`):
   - For utterances between 0.3 seconds and 30 seconds (virtually all normal dictation):
     ```python
     final_phrase = self.engine.finalize(audio)
     if final_phrase:
         self.accumulator.reset()       # <--- WIPES the 1-second chunk guesses
         self.accumulator.merge(final_phrase)
         phrase = self.accumulator.finish()
     ```
   - **Whisper is invoked on the ENTIRE recorded audio buffer in one contiguous call.**
   - Because Whisper hears the full sentence with complete linguistic context, capitalization and punctuation are significantly more accurate than real-time chunk guesses.
   - Any prior chunk hypotheses from Phase B are completely wiped.
3. For rare recordings exceeding 30 seconds (Whisper's maximum context length), the engine falls back to the stitched sliding accumulator output.

---

### Phase D: Post-Processing & Text Injection

1. **Blank Audio Filtering:**
   - If the user briefly taps the key or silence is detected, Whisper may output `[BLANK_AUDIO]`.
   - `_inject_final()` checks:
     ```python
     if not text or text.strip() in ("[BLANK_AUDIO]", "[blank_audio]"):
         return
     ```
     This safely ignores blank input without typing unwanted tags.
2. **Dictionary Replacement:**
   - Custom words and developer vocabulary (e.g., "pie torch" $\rightarrow$ `PyTorch`, "github" $\rightarrow$ `GitHub`) are substituted via regex/token rules in `DictionaryProcessor`.
3. **Portal Keyboard Injection:**
   - Via `PortalKeyboard.insert(text)` (Desktop portal `org.freedesktop.portal.RemoteDesktop`), the string is simulated as real keystrokes into whatever window currently has focus.
4. **HUD Close:**
   - `hud.finish()` runs the shrink and fade-out ease animation.

---

## 3. What Was Already Done (Recent Commits Breakdown)

| Commit        | Summary                                 | Key Technical Additions                                                                                                                                                                                |
| :------------ | :-------------------------------------- | :----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| **`f563280`** | Model warmup & audio finalization logic | • Added `WhisperCppEngine.warmup()` with dummy forward pass to eliminate 7–15s SYCL JIT stall.<br>• Added whole-utterance `finalize(audio)` pass in `_finish_utterance()`.                             |
| **`98d72ed`** | Liquid glass design overhaul            | • Liquid glass app assets, token styling, and desktop launcher icon update.                                                                                                                            |
| **`1a5b89e`** | Performance models & base.en default    | • Updated settings UI components and tuned default model configuration to `base.en`.                                                                                                                   |
| **`10620eb`** | Floating liquid-glass pill HUD overlay  | • Frameless, 32-bit ARGB Cairo HUD window centered above dock.<br>• Real-time reactive waveform bar calculation via `last_audio_level`.<br>• Empty input region so HUD is 100% click-through.          |
| **`5993c94`** | Cold-start & HUD fixes                  | • Microphone capture stream warmup in `load_model()`.<br>• Window realization `self._window.realize()` to guarantee first-press HUD appearance.<br>• `[BLANK_AUDIO]` suppression in `_inject_final()`. |

---

## 4. Current Inefficiency / What We Are Optimizing

### The Identified Issue

Because the product requirement is **pure Push-to-Talk** (no live text display, only audio waveform bars while holding, text pasted upon release):

- **Redundant Computation:** Every 1 second while holding the key, `SlidingChunker` triggers `engine.transcribe_chunk()`.
- **Zero Utility:** The output text is never shown in the HUD and is explicitly erased on key release by `accumulator.reset()`.
- **Resource Burden:** For a typical 8-second utterance, Whisper runs **8 separate times** during speech, and then runs a **9th time** on release.
- **Drawbacks:** Unnecessary GPU power consumption, laptop battery drain, and thermal fan spin.

---

## 5. Implementation Plan: Lean Push-to-Talk Engine

### Objective

Keep 100% of the current behavior, fixes, HUD visuals, and accuracy, while eliminating the unnecessary background chunk inference during speech.

### Preservation Checklist (Must NOT Change)

- [x] **HUD Realization:** Keep `self._window.realize()` in `hud.py` so the HUD always pops up on the first hotkey press.
- [x] **Microphone Stream Warmup:** Keep `self.capture.read(timeout=0.1)` in `load_model()`.
- [x] **Model GPU Warmup:** Keep `engine.warmup()` dummy inference pass on startup.
- [x] **Waveform Animation:** Keep `self.last_audio_level` RMS calculation and `hud.set_audio_level()` calls.
- [x] **Blank Audio Filter:** Keep `[BLANK_AUDIO]` checks.
- [x] **Final Single-Pass Transcription:** Keep `engine.finalize(audio)` when releasing the hotkey.

---

### Step-by-Step Code Adjustments

#### Step 1: Add a Lightweight Audio-Only Mode to `read_once()`

In `src/whisperflow/application.py`:

- In `read_once()`, allow an option (or default when not doing streaming UI) to only read audio, calculate `last_audio_level`, and append to `_utterance` without calling `chunker.push()` or `engine.transcribe_chunk()`.
- Audio accumulation for sentences $\le$ 30s is preserved.

#### Step 2: Fallback for Long Dictation (> 30s)

- If the user holds the hotkey for longer than 30 seconds, audio chunking can cleanly engage to prevent overflowing Whisper's 30-second context window.
- For normal speech (< 30s), zero model inference runs while holding the key.

#### Step 3: Clean up `service.py` Listen Loop

In `src/whisperflow/service.py`:

- The loop continues to call `app.read_once()`, updating `hud.set_audio_level(app.last_audio_level)`.
- On key release, `app.stop_recording()` runs `engine.finalize(audio)` and types the resulting text into the focused window.

---

## 6. Verification & Test Plan

1. **Unit Tests:**
   - Run `uv run pytest` to ensure all 37 test cases continue to pass with zero regressions.
2. **First-Run Verification:**
   - Launch background service `python -m whisperflow.service`.
   - Press hotkey for the very first time.
   - Verify: HUD pill appears instantly above the dock without flicker.
   - Verify: Waveform bars bounce according to voice loudness.
   - Verify: Initial words of the sentence are not cut off.
3. **Performance & GPU Verification:**
   - Verify GPU utilization remains near 0% while holding the hotkey (only audio capture is active).
   - Verify GPU triggers once on key release for ~0.2s to finalize and type the text.
4. **Blank Speech Handling:**
   - Tap hotkey without speaking $\rightarrow$ verify HUD opens and closes smoothly with nothing typed.
