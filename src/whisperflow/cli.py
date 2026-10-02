"""Command line for the Stage 1 engine.

`python -m whisperflow.transcribe` listens until Ctrl+C.
`python -m whisperflow.benchmark` times one file and prints the measurements
the model choice will be based on.
"""

from __future__ import annotations

import argparse
import sys
import time
import wave
from pathlib import Path

import numpy as np
import psutil

from whisperflow.application import Final, Partial, TranscriptionApp
from whisperflow.asr.device import select_device
from whisperflow.asr.whisper_cpp import WhisperCppEngine
from whisperflow.config import load_config
from whisperflow.metrics import (
    cpu_model,
    gpu_busy_percent,
    peak_rss_mb,
    process_rss_mb,
    ram_total_gb,
    sample_gpu,
)


def transcribe_main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Hold a key, speak, and print the transcript.")
    parser.add_argument("--config", type=Path, default=None)
    parser.add_argument("--device", choices=("auto", "sycl", "cpu"), default=None)
    parser.add_argument(
        "--key",
        default="leftctrl",
        help="Key to hold while speaking (leftctrl, rightctrl, rightalt, leftalt, space).",
    )
    args = parser.parse_args(argv)
    config = load_config(args.config)
    if args.device:
        config = _with_device(config, args.device)

    from whisperflow.hotkey.linux import HoldKey

    app = TranscriptionApp(config)
    _print_banner(app)
    print(f"Mic: {app.capture.description}")
    if app.device.use_gpu:
        print("Inference: Intel Arc GPU through whisper.cpp SYCL. The model stays on the GPU.")
    else:
        print("Inference: whisper.cpp CPU.")
    print()
    print("Loading model...")
    started = time.perf_counter()
    app.load_model()
    print(f"Model ready in {time.perf_counter() - started:.2f}s")
    print(app.engine.system_info().strip())
    print()
    hold = HoldKey(args.key)
    hold.start()
    print("Hold the Left Ctrl key and speak. Release it to finish.")
    print("Keep holding for about a second to see a partial line. Press Ctrl+C to quit.")
    print()
    recording = False
    try:
        while True:
            if hold.held.is_set():
                if not recording:
                    app.start_recording()
                    recording = True
                    print()
                    print("Recording...")
                event = app.read_once(force_speech=True)
                if isinstance(event, Partial) and event.text:
                    print(f"[partial] {event.text}", flush=True)
            elif recording:
                text = app.stop_recording()
                recording = False
                _print_final(text)
                print("Hold Left Ctrl to talk again.")
            else:
                time.sleep(0.02)
    except KeyboardInterrupt:
        print()
        if recording:
            _print_final(app.stop_recording())
        return 0
    finally:
        hold.stop()
        app.capture.stop()


def _print_final(text: str) -> None:
    print()
    print("Final:")
    print(text or "(no speech heard)")
    print()


def benchmark_main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Measure model load, memory, and RTF.")
    parser.add_argument("--config", type=Path, default=None)
    parser.add_argument("--device", choices=("auto", "sycl", "cpu"), default=None)
    parser.add_argument(
        "--wav",
        type=Path,
        default=None,
        help="16-bit PCM wav. Defaults to whisper.cpp's jfk sample.",
    )
    args = parser.parse_args(argv)
    config = load_config(args.config)
    if args.device:
        config = _with_device(config, args.device)
    device = select_device(config.asr.device)
    wav_path = args.wav or (
        config.resolve("third_party/whisper.cpp/samples/jfk.wav")
    )
    audio = read_wav(wav_path)

    idle_rss = process_rss_mb()
    engine = WhisperCppEngine(
        model_path=config.resolve(config.asr.model_path),
        use_gpu=device.use_gpu,
        gpu_device=config.asr.gpu_device,
        threads=config.asr.threads,
    )
    load_started = time.perf_counter()
    engine.start()
    load_seconds = time.perf_counter() - load_started
    model_rss = process_rss_mb()

    gpu_before = sample_gpu()
    cpu = psutil.Process()
    cpu.cpu_percent(interval=None)
    infer_started = time.perf_counter()
    text = engine.finalize(audio)
    infer_seconds = time.perf_counter() - infer_started
    cpu_percent = cpu.cpu_percent(interval=None)
    gpu_after = sample_gpu()
    busy = gpu_busy_percent(gpu_before, gpu_after, infer_seconds * 1000.0)
    audio_seconds = audio.size / config.audio.sample_rate
    rtf = infer_seconds / audio_seconds if audio_seconds else 0.0

    print("WhisperFlow Benchmark")
    print("──────────────────────────────")
    print()
    print("Hardware")
    print(f"CPU: {cpu_model()}")
    print(f"GPU: {device.name}")
    print(f"RAM: {ram_total_gb():.0f} GB")
    print()
    print(f"Model: {config.asr.model}")
    print("Backend: whisper.cpp")
    print(f"Device: {device.label}")
    print(f"Quantization: {config.asr.quantization}")
    print()
    print(f"Model load:       {load_seconds:.2f} s")
    print(f"Idle RAM:         {idle_rss:.0f} MB")
    print(f"Model RAM:        {max(0.0, model_rss - idle_rss):.0f} MB")
    print(f"Peak RAM:         {peak_rss_mb():.0f} MB")
    print("GPU memory:       shared with system RAM")
    print(f"CPU utilization:  {cpu_percent:.0f} %")
    if busy is None:
        print("GPU utilization:  unavailable")
    else:
        print(f"GPU utilization:  {busy:.0f} %")
    print(f"RTF:              {rtf:.2f}")
    print(f"Audio:            {audio_seconds:.2f} s")
    print(f"Infer:            {infer_seconds * 1000:.0f} ms")
    print()
    print("Transcript:")
    print(text or "(empty)")
    print()
    print(engine.system_info().strip())
    return 0


def read_wav(path: Path) -> np.ndarray:
    """Load a 16-bit PCM wav and return 16 kHz mono float32."""
    if not path.is_file():
        raise FileNotFoundError(path)
    with wave.open(str(path), "rb") as handle:
        channels = handle.getnchannels()
        sample_width = handle.getsampwidth()
        rate = handle.getframerate()
        frames = handle.readframes(handle.getnframes())
    if sample_width != 2:
        raise ValueError(f"{path} must be 16-bit PCM")
    if rate != 16000:
        raise ValueError(f"{path} must be 16000 Hz, got {rate}")
    audio = np.frombuffer(frames, dtype=np.int16).astype(np.float32) / 32768.0
    if channels == 2:
        audio = audio.reshape(-1, 2).mean(axis=1)
    elif channels != 1:
        raise ValueError(f"{path} must be mono or stereo")
    return np.ascontiguousarray(audio, dtype=np.float32)


def _print_banner(app: TranscriptionApp) -> None:
    print("WhisperFlow Transcription Engine")
    print()
    print(f"Backend: {app.device.label}")
    print(f"Device: {app.device.name}")
    print(f"Model: {app.config.asr.model}")
    print(f"Quantization: {app.config.asr.quantization}")
    print()


def _with_device(config, device: str):
    from dataclasses import replace

    return replace(config, asr=replace(config.asr, device=device))


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="whisperflow")
    parser.add_argument("command", choices=("transcribe", "benchmark"))
    parser.add_argument("rest", nargs=argparse.REMAINDER)
    args = parser.parse_args(argv)
    if args.command == "transcribe":
        return transcribe_main(args.rest)
    return benchmark_main(args.rest)


if __name__ == "__main__":
    sys.exit(main())
