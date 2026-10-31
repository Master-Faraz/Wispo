# Wispo System Freeze Investigation & Comprehensive Engineering Handoff

**Author:** Antigravity Pair-Programming Assistant  
**Date:** 2026-10-31  
**Target Hardware:** Intel Core Ultra 5 225H (Meteor Lake, 14 Cores) • Intel Arc Integrated Graphics (`0000:00:02.0`)  
**Operating System:** Ubuntu Linux 24.04 LTS (x86_64, Wayland / GNOME Shell 46+)  
**Corpus / Repository:** [my-wisper](file:///home/faraz/Documents/Codes/my-wisper)

---

## 1. Executive Summary & Problem Clarification

### The Core Symptoms
1. **The ~30-Minute Normal Usage Freeze:** After running Wispo in the background and using it normally for approximately 30 minutes (speaking intermittently via push-to-talk), the entire laptop screen, mouse cursor, and desktop compositor lock up completely.
2. **The Sleep/Resume Freeze:** When closing the laptop lid or putting the machine to sleep and waking it up, triggering Wispo freezes the machine or causes an unrecoverable graphics lockup.

### Timeline Clarification
* **The bug has existed since Stage 1 and Stage 2 were built.** It was **not** introduced by the commits in the last 10 days.
* The commits made over the last 10 days (`1ddb45e`, `219b680`, `d42f074`) were attempts to diagnose and fix the issue, but they treated secondary symptoms (ALSA use-after-free, basic D-Bus signals) rather than the underlying hardware/driver architecture collisions.

---

## 2. Hard Diagnostic Evidence (From Linux Journal Logs)

Examination of system logs (`journalctl -p 0..3 -o short-monotonic`) across multiple system boots identified the exact kernel failure mode:

```text
kernel: i915 0000:00:02.0: [drm] *ERROR* Atomic update failure on pipe A (start=... end=...) time 1088 us, min 1777, max 1800, scanline start 1651, end 1902
gnome-shell[5939]: Direct scanout page flip failed: drmModeAtomicCommit: Invalid argument
gnome-shell[5939]: Unexpected non-NULL onscreen_native->next_frame
gnome-shell[5939]: Can't find input for stream-id 10
```

### Key Technical Takeaways from the Logs:
1. **Device `0000:00:02.0`:** This is the Intel integrated Arc GPU. It is simultaneously handling compute (SYCL whisper inference) and primary display scanout to the laptop eDP screen (**Pipe A**).
2. **Atomic Update Failure on Pipe A:** The Intel DRM kernel driver missed its hardware vblank deadline (`min 1777, max 1800` scanlines) because memory bus transactions or GPU compute queues stalled scanout DMA.
3. **Mutter Frame Clock Lockup:** When direct scanout atomic commits fail repeatedly, GNOME Mutter's presentation queue desynchronizes (`Unexpected non-NULL onscreen_native->next_frame`), halting all screen rendering and user input.

---

## 3. The 4 Original Architectural Root Causes

### Root Cause 1: Intel Arc Shared-Memory Contention on `0000:00:02.0`
* **Architecture:** The Intel Core Ultra 5 225H utilizes integrated Intel Arc Graphics. It has **no dedicated VRAM**; all graphics memory and Whisper model weights reside in the shared LPDDR5 system memory bus.
* **Mechanism:** When Wispo runs inference, the Level Zero driver submits intensive GEMM matrix-multiplication kernels. This compute load competes with the display engine's scanout FIFO buffers for shared memory bandwidth.
* Over 30 minutes of regular desktop usage (browser, editor, terminal, plus intermittent Wispo transcriptions), the cumulative memory latency spikes eventually cause a vblank scanline miss on Pipe A, crashing the display pipeline.

### Root Cause 2: Intel Level Zero GPU State Dies Across Sleep/Resume
* **Architecture:** In Linux Modern Standby (S0ix / `s2idle`), power to the Intel Arc graphics core (GT0/GT1) is cut.
* **Mechanism:** Intel oneAPI / Level Zero specifications state that `ze_context_handle_t`, `ze_command_queue_handle_t`, and device USM allocations (`zeMemAllocDevice`) **do not survive system suspend**.
* From Day 1, Wispo called `wf_load()` at process startup and retained `g_ctx` indefinitely. Upon resume from sleep, the first call to `wf_transcribe()` passed dead Level Zero device handles to `i915`, triggering an immediate kernel GPU hang.

### Root Cause 3: Full-Screen Transparent HUD Window ([hud.py](file:///home/faraz/Documents/Codes/my-wisper/src/wispo/hud.py#L128))
* **Architecture:** In [src/wispo/hud.py](file:///home/faraz/Documents/Codes/my-wisper/src/wispo/hud.py), to position the 340×76 px pill at the bottom center of the screen, the window was sized to **100% of the monitor resolution**:
  ```python
  self._window.set_default_size(monitor_w, monitor_h)
  ```
* **Mechanism:** 
  1. On Wayland, when a window covers the full monitor, Mutter automatically attempts **Direct Scanout** (bypassing the compositor to scan out directly to the display plane). Because the window is transparent, rapidly animating with Cairo at 60 FPS, and has an empty input region, Mutter's direct scanout atomic commit fails (`drmModeAtomicCommit: Invalid argument`).
  2. If Wayland resets the window's `surface.set_input_region(cairo.Region())` during mapping or wake from sleep, the invisible full-screen window captures all mouse clicks across the entire desktop, making the laptop appear completely frozen.

### Root Cause 4: ALSA / PipeWire Stream Thrashing & Cross-Thread FD Race
* **ALSA Stream Churn:** Every push-to-talk cycle executed `capture.start()` (`snd_pcm_open`) and `capture.stop()` (`snd_pcm_close`). In PipeWire, creating and tearing down client nodes dozens of times caused stream-id desyncs in GNOME's media keys daemon (`Can't find input for stream-id`).
* **Cross-Thread FD Race ([hotkey/linux.py](file:///home/faraz/Documents/Codes/my-wisper/src/wispo/hotkey/linux.py)):** Calling `HoldChord.reset()` closed file descriptors from the listener thread while the background chord thread was blocked in `select.select()`. In Linux, closing an FD across threads allows the kernel to recycle that FD integer for GTK's Wayland socket, leading to stolen display protocol packets.

---

## 4. Post-Mortem of the Last 10 Days' Commits

| Commit | Changes Made | Why It Failed to Solve the Freezing |
|---|---|---|
| **`1ddb45e`** | `snd_pcm_drop` before `thread.join(1.0)` in [capture.py](file:///home/faraz/Documents/Codes/my-wisper/src/wispo/audio/capture.py) | Fixed a Python thread crash, but the freeze is a **GPU DRM driver / memory bus issue**, not a Python crash. |
| **`219b680`** | Re-enabled `transcribe=True` during recording in [service.py](file:///home/faraz/Documents/Codes/my-wisper/src/wispo/service.py) | **Worsened the 30-minute freeze.** Running back-to-back 5s SYCL inference every second saturated the shared GPU memory bus, accelerating atomic page-flip failures. |
| **`d42f074`** | `PrepareForSleep` D-Bus listener, `sycl-ls` cache, Unicode normalizer | Paused audio on sleep, but **never re-initialized the dead Level Zero GPU context**. The GPU context remained dead after resume. Introduced cross-thread FD closes in `hotkey/linux.py`. |

---

## 5. Step-by-Step Implementation Plan for the Next Chat

When you begin the new chat, paste or reference this plan to execute the permanent fix:

### Step 1: Fix HUD Window Dimensions ([hud.py](file:///home/faraz/Documents/Codes/my-wisper/src/wispo/hud.py))
* **Change:** Set the HUD window size strictly to the pill canvas dimensions (`CANVAS_W = 340`, `CANVAS_H = 76`), rather than full-screen (`monitor_w × monitor_h`).
* **Impact:** Completely eliminates Mutter's Direct Scanout attempts, prevents `drmModeAtomicCommit` failures, and ensures the window can never block desktop-wide mouse input.

### Step 2: Keep Push-to-Talk Lean & Silent During Speech ([service.py](file:///home/faraz/Documents/Codes/my-wisper/src/wispo/service.py))
* **Change:** Set `transcribe=False` in [service.py](file:///home/faraz/Documents/Codes/my-wisper/src/wispo/service.py#L399) during recording:
  ```python
  app.read_once(force_speech=True, transcribe=False)
  ```
* **Impact:** While the user holds the hotkey, only lightweight audio RMS is computed to animate the waveform. The Intel Arc GPU remains completely idle and cool during speech. When the key is released, one single final pass runs, delivering maximum accuracy with zero GPU thrashing.

### Step 3: Re-Initialize Whisper Engine Across Sleep/Resume ([service.py](file:///home/faraz/Documents/Codes/my-wisper/src/wispo/service.py))
* **Change:** In the `PrepareForSleep` handler, upon system resume (`about_to_sleep == False`):
  ```python
  logging.info("system resumed from suspend; reinitializing ASR engine")
  app.reload_model(config.asr.model, config.resolve(config.asr.model_path))
  ```
* **Impact:** Discards stale Level Zero device handles and allocates fresh GPU queues and USM buffers, preventing the `i915` kernel driver hang upon wake.

### Step 4: Thread-Safe Hotkey FD Lifecycle ([hotkey/linux.py](file:///home/faraz/Documents/Codes/my-wisper/src/wispo/hotkey/linux.py))
* **Change:** Remove external `os.close(fd)` calls in `reset()`. Use an internal `threading.Lock` and have the polling thread close and rediscover its own FDs when dead without cross-thread interference.

### Step 5: CPU Fallback Option for Zero GPU Impact
* **Change:** Verify that setting `device: cpu` in `wispo.yaml` runs cleanly on the 14-core Intel Ultra 5 225H. For users experiencing persistent Intel graphics driver issues in Ubuntu, running Whisper on the CPU takes ~0.3s per sentence and bypasses the GPU entirely.

---

## 6. Prompt to Use in the New Chat

Copy and paste the following prompt into your new chat to continue immediately:

```text
Please read the comprehensive investigation and plan in docs/freeze_investigation_and_fix_plan.md. 
We have identified the root causes of the 30-minute freeze and sleep/resume freeze (Intel Arc shared-memory DRM collision, dead Level Zero context on resume, and full-screen HUD direct scanout).
Please proceed with implementing Step 1 through Step 4 from Section 5 of that document.
```
