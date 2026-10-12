"""Settings window. Apple-inspired Liquid Glass modern aesthetic matching macOS Sonoma design.

    uv run python -m whisperflow.settings

The transcription command keeps the model it loaded at startup. Restart it
after saving to use a new model or dictionary.
"""

from __future__ import annotations

import html
import math
import os
import queue
import subprocess
import sys
import threading
import time
from pathlib import Path

import numpy as np

from whisperflow.asr.device import select_device
from whisperflow.audio.capture import AudioCapture
from whisperflow.audio.sources import _node_name, parse_wpctl_sources, resolve_mic
from whisperflow.config import DictionaryEntry, load_config, project_root, save_user_settings
from whisperflow.hotkey.linux import capture_chord, format_chord
from whisperflow.models import CATALOG, DEFAULT_MODEL, download_model, installed_models, model_path
from whisperflow.processing.dictionary import DictionaryProcessor

# Helper to escape any text for Pango markup
def pango_esc(text: str) -> str:
    return html.escape(str(text))


def _format_bytes(bytes_count: int) -> str:
    if bytes_count >= 1024 * 1024 * 1024:
        return f"{bytes_count / (1024 ** 3):.2f} GB"
    if bytes_count >= 1024 * 1024:
        return f"{bytes_count / (1024 ** 2):.1f} MB"
    if bytes_count >= 1024:
        return f"{bytes_count / 1024:.0f} KB"
    return f"{bytes_count} B"


# Rich metadata verified from whisper.cpp and distil-whisper official benchmarks
MODEL_METADATA: dict[str, dict] = {
    "tiny.en": {
        "title": "tiny.en",
        "subtitle": "Fastest, good for clean audio.",
        "desc": "An ultra-compact 39M parameter model optimized for minimal memory footprint and fast inference. Best for rapid commands and clean audio on low-power CPUs.",
        "params": "39M",
        "size_str": "75 MB",
        "ram_str": "~420 MB idle / ~720 MB active",
        "lang": "English only",
        "best_for": "Quick commands & clean audio",
        "speed_pct": 0.95,
        "speed_label": "Very Fast",
        "acc_pct": 0.55,
        "acc_label": "Good",
        "tag": "",
    },
    "base.en": {
        "title": "base.en",
        "subtitle": "Balanced speed and accuracy for English.",
        "desc": "The standard balanced 74M parameter model. Provides reliable English dictation with near-zero latency and minimal CPU/GPU overhead.",
        "params": "74M",
        "size_str": "142 MB",
        "ram_str": "~510 MB idle / ~840 MB active",
        "lang": "English only",
        "best_for": "General usage & everyday speech",
        "speed_pct": 0.85,
        "speed_label": "Fast",
        "acc_pct": 0.72,
        "acc_label": "Balanced",
        "tag": "Recommended",
    },
    "small.en": {
        "title": "small.en",
        "subtitle": "Better accuracy, still fast.",
        "desc": "High accuracy 244M parameter model capable of handling diverse regional accents, technical vocabulary, and noisy background environments.",
        "params": "244M",
        "size_str": "466 MB",
        "ram_str": "~900 MB idle / ~1.4 GB active",
        "lang": "English only",
        "best_for": "Accents & technical terms",
        "speed_pct": 0.65,
        "speed_label": "Moderate",
        "acc_pct": 0.86,
        "acc_label": "High",
        "tag": "",
    },
    "medium.en": {
        "title": "medium.en",
        "subtitle": "High accuracy for nuanced context.",
        "desc": "Deep 769M parameter acoustic model with excellent linguistic nuance and low Word Error Rate. Ideal for users with dedicated GPU headroom.",
        "params": "769M",
        "size_str": "1.5 GB",
        "ram_str": "~2.0 GB idle / ~2.6 GB active",
        "lang": "English only",
        "best_for": "Complex vocabulary & noisy audio",
        "speed_pct": 0.40,
        "speed_label": "Slower",
        "acc_pct": 0.92,
        "acc_label": "Very High",
        "tag": "",
    },
    "large-v3-turbo-q5_0": {
        "title": "large-v3-turbo-q5_0",
        "subtitle": "High accuracy, optimized for speed.",
        "desc": "Quantized 5-bit Turbo checkpoint. Delivers state-of-the-art Large-v3 accuracy with 3x faster decoding, low VRAM usage, and 809M parameters.",
        "params": "809M (Q5_0)",
        "size_str": "548 MB",
        "ram_str": "~850 MB idle / ~1.6 GB active",
        "lang": "Multilingual / English",
        "best_for": "Maximum accuracy with fast speed",
        "speed_pct": 0.62,
        "speed_label": "Fast",
        "acc_pct": 0.95,
        "acc_label": "High",
        "tag": "",
    },
    "large-v3-turbo": {
        "title": "large-v3-turbo",
        "subtitle": "Uncompromising accuracy, 4-layer decoder.",
        "desc": "OpenAI's official 4-layer fast decoder model. Uncompromised Large-v3 robustness across technical jargon and acronyms with 8x lower latency.",
        "params": "809M",
        "size_str": "1.5 GB",
        "ram_str": "~1.8 GB idle / ~2.5 GB active",
        "lang": "Multilingual / English",
        "best_for": "Professional transcription",
        "speed_pct": 0.50,
        "speed_label": "Moderate",
        "acc_pct": 0.96,
        "acc_label": "Near-Perfect",
        "tag": "",
    },
    "distil-small.en": {
        "title": "distil-small.en",
        "subtitle": "Fast and lightweight distilled model.",
        "desc": "Knowledge-distilled version of small.en running 6x faster while preserving over 99% accuracy. Ideal for real-time streaming on Intel Arc.",
        "params": "166M",
        "size_str": "321 MB",
        "ram_str": "~620 MB idle / ~1.1 GB active",
        "lang": "English only",
        "best_for": "Real-time streaming & coding",
        "speed_pct": 0.88,
        "speed_label": "Very Fast",
        "acc_pct": 0.83,
        "acc_label": "High",
        "tag": "",
    },
    "distil-medium.en": {
        "title": "distil-medium.en",
        "subtitle": "Good balance of speed, size and accuracy.",
        "desc": "Distilled 394M parameter model offering the sweet spot of speed, compact size, and accuracy. 6.8x faster than standard medium.",
        "params": "394M",
        "size_str": "758 MB",
        "ram_str": "~1.2 GB idle / ~1.7 GB active",
        "lang": "English only",
        "best_for": "General usage & coding",
        "speed_pct": 0.75,
        "speed_label": "Fast",
        "acc_pct": 0.90,
        "acc_label": "High",
        "tag": "",
    },
    "distil-large-v3": {
        "title": "distil-large-v3",
        "subtitle": "Flagship accuracy with 7.5x speedup.",
        "desc": "Distilled flagship model providing Large-v3 robustness with 7.5x faster throughput on modern Intel Arc GPUs and 756M parameters.",
        "params": "756M",
        "size_str": "1.5 GB",
        "ram_str": "~1.8 GB idle / ~2.6 GB active",
        "lang": "English only",
        "best_for": "Heavy prose & technical dictation",
        "speed_pct": 0.58,
        "speed_label": "Moderate",
        "acc_pct": 0.94,
        "acc_label": "Very High",
        "tag": "",
    },
    "distil-large-v3.5": {
        "title": "distil-large-v3.5",
        "subtitle": "Best accuracy, larger size.",
        "desc": "Latest state-of-the-art distilled large checkpoint with enhanced noise robustness, 756M parameters, and specialized technical tokenization.",
        "params": "756M",
        "size_str": "1.45 GB",
        "ram_str": "~1.8 GB idle / ~2.6 GB active",
        "lang": "English only",
        "best_for": "Coding, punctuation & technical dictation",
        "speed_pct": 0.52,
        "speed_label": "Moderate",
        "acc_pct": 0.97,
        "acc_label": "Near-Perfect",
        "tag": "",
    },
}

# SVG Icon Assets for Crisp Retina Rendering
SVG_ICONS = {
    "cube": """<svg xmlns="http://www.w3.org/2000/svg" width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="white" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><path d="m21 16-9 5-9-5V8l9-5 9 5v8Z"/><path d="M3.27 6.96 12 12.01l8.73-5.05"/><path d="M12 22.08V12"/></svg>""",
    "cube_badge": """<svg xmlns="http://www.w3.org/2000/svg" width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="white" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><path d="m21 16-9 5-9-5V8l9-5 9 5v8Z"/><path d="M3.27 6.96 12 12.01l8.73-5.05"/><path d="M12 22.08V12"/></svg>""",
    "microphone": """<svg xmlns="http://www.w3.org/2000/svg" width="17" height="17" viewBox="0 0 24 24" fill="none" stroke="rgba(255,255,255,0.8)" stroke-width="1.8" stroke-linecap="round" stroke-linejoin="round"><path d="M12 2a3 3 0 0 0-3 3v7a3 3 0 0 0 6 0V5a3 3 0 0 0-3-3Z"/><path d="M19 10v2a7 7 0 0 1-14 0v-2"/><line x1="12" x2="12" y1="19" y2="22"/></svg>""",
    "keyboard": """<svg xmlns="http://www.w3.org/2000/svg" width="17" height="17" viewBox="0 0 24 24" fill="none" stroke="rgba(255,255,255,0.8)" stroke-width="1.8" stroke-linecap="round" stroke-linejoin="round"><rect width="20" height="14" x="2" y="5" rx="2.5"/><line x1="6" x2="6.01" y1="9" y2="9" stroke-width="2.5"/><line x1="10" x2="10.01" y1="9" y2="9" stroke-width="2.5"/><line x1="14" x2="14.01" y1="9" y2="9" stroke-width="2.5"/><line x1="18" x2="18.01" y1="9" y2="9" stroke-width="2.5"/><line x1="7" x2="17" y1="14" y2="14"/></svg>""",
    "book": """<svg xmlns="http://www.w3.org/2000/svg" width="17" height="17" viewBox="0 0 24 24" fill="none" stroke="rgba(255,255,255,0.8)" stroke-width="1.8" stroke-linecap="round" stroke-linejoin="round"><path d="M4 19.5v-15A2.5 2.5 0 0 1 6.5 2H20v20H6.5a2.5 2.5 0 0 1-2.5-2.5Z"/><path d="M6 6h10"/><path d="M6 10h10"/></svg>""",
    "cog": """<svg xmlns="http://www.w3.org/2000/svg" width="17" height="17" viewBox="0 0 24 24" fill="none" stroke="rgba(255,255,255,0.8)" stroke-width="1.8" stroke-linecap="round" stroke-linejoin="round"><circle cx="12" cy="12" r="3"/><path d="M19.4 15a1.65 1.65 0 0 0 .33 1.82l.06.06a2 2 0 0 1 0 2.83 2 2 0 0 1-2.83 0l-.06-.06a1.65 1.65 0 0 0-1.82-.33 1.65 1.65 0 0 0-1 1.51V21a2 2 0 0 1-2 2 2 2 0 0 1-2-2v-.09A1.65 1.65 0 0 0 9 19.4a1.65 1.65 0 0 0-1.82.33l-.06.06a2 2 0 0 1-2.83 0 2 2 0 0 1 0-2.83l.06-.06a1.65 1.65 0 0 0 .33-1.82 1.65 1.65 0 0 0-1.51-1H3a2 2 0 0 1-2-2 2 2 0 0 1 2-2h.09A1.65 1.65 0 0 0 4.6 9a1.65 1.65 0 0 0-.33-1.82l-.06-.06a2 2 0 0 1 0-2.83 2 2 0 0 1 2.83 0l.06.06a1.65 1.65 0 0 0 1.82.33H9a1.65 1.65 0 0 0 1-1.51V3a2 2 0 0 1 2-2 2 2 0 0 1 2 2v.09a1.65 1.65 0 0 0 1 1.51 1.65 1.65 0 0 0 1.82-.33l.06-.06a2 2 0 0 1 2.83 0 2 2 0 0 1 0 2.83l-.06.06a1.65 1.65 0 0 0-.33 1.82V9a1.65 1.65 0 0 0 1.51 1H21a2 2 0 0 1 2 2 2 2 0 0 1-2 2h-.09a1.65 1.65 0 0 0-1.51 1z"/></svg>""",
    "info": """<svg xmlns="http://www.w3.org/2000/svg" width="17" height="17" viewBox="0 0 24 24" fill="none" stroke="rgba(255,255,255,0.8)" stroke-width="1.8" stroke-linecap="round" stroke-linejoin="round"><circle cx="12" cy="12" r="10"/><line x1="12" x2="12" y1="16" y2="12"/><line x1="12" x2="12.01" y1="8" y2="8" stroke-width="2.5"/></svg>""",
    "drive": """<svg xmlns="http://www.w3.org/2000/svg" width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="rgba(255,255,255,0.7)" stroke-width="1.8" stroke-linecap="round" stroke-linejoin="round"><rect width="20" height="8" x="2" y="14" rx="2"/><path d="M6 18h.01"/><path d="M10 18h.01"/><rect width="20" height="8" x="2" y="4" rx="2"/><path d="M6 8h.01"/><path d="M10 8h.01"/></svg>""",
    "download": """<svg xmlns="http://www.w3.org/2000/svg" width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="rgba(255,255,255,0.9)" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><path d="M21 15v4a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2v-4"/><polyline points="7 10 12 15 17 10"/><line x1="12" x2="12" y1="15" y2="3"/></svg>""",
    "cancel": """<svg xmlns="http://www.w3.org/2000/svg" width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="#f87171" stroke-width="2.2" stroke-linecap="round" stroke-linejoin="round"><line x1="18" y1="6" x2="6" y2="18"/><line x1="6" y1="6" x2="18" y2="18"/></svg>""",
    "radio_off": """<svg xmlns="http://www.w3.org/2000/svg" width="18" height="18" viewBox="0 0 24 24"><circle cx="12" cy="12" r="9" fill="rgba(255, 255, 255, 0.04)" stroke="rgba(255, 255, 255, 0.22)" stroke-width="1.5"/></svg>""",
    "radio_on": """<svg xmlns="http://www.w3.org/2000/svg" width="18" height="18" viewBox="0 0 24 24"><circle cx="12" cy="12" r="9.5" fill="#8b75ff" stroke="rgba(255, 255, 255, 0.2)" stroke-width="1"/><circle cx="12" cy="12" r="3.6" fill="#ffffff"/></svg>""",
    "checkbox_checked": """<svg xmlns="http://www.w3.org/2000/svg" width="17" height="17" viewBox="0 0 24 24"><rect width="20" height="20" x="2" y="2" rx="5" fill="#8b75ff" stroke="rgba(255, 255, 255, 0.2)" stroke-width="1"/><polyline points="6.5 12 10.5 16 17.5 8.5" fill="none" stroke="#ffffff" stroke-width="2.4" stroke-linecap="round" stroke-linejoin="round"/></svg>""",
    "pencil": """<svg xmlns="http://www.w3.org/2000/svg" width="13" height="13" viewBox="0 0 24 24" fill="none" stroke="rgba(255,255,255,0.85)" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><path d="M17 3a2.85 2.83 0 1 1 4 4L7.5 20.5 2 22l1.5-5.5Z"/><path d="m15 5 4 4"/></svg>""",
    "reticle": """<svg xmlns="http://www.w3.org/2000/svg" width="15" height="15" viewBox="0 0 24 24" fill="none" stroke="rgba(255,255,255,0.85)" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><circle cx="12" cy="12" r="10"/><path d="M12 2v4"/><path d="M12 18v4"/><path d="M2 12h4"/><path d="M18 12h4"/></svg>""",
    "footer_info": """<svg xmlns="http://www.w3.org/2000/svg" width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="rgba(255,255,255,0.65)" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><circle cx="12" cy="12" r="10"/><line x1="12" x2="12" y1="16" y2="12"/><line x1="12" x2="12.01" y1="8" y2="8" stroke-width="2.5"/></svg>""",
    "traffic_close": """<svg xmlns="http://www.w3.org/2000/svg" width="14" height="14" viewBox="0 0 14 14" fill="none"><path d="M4.5 4.5L9.5 9.5M9.5 4.5L4.5 9.5" stroke="#4c0000" stroke-width="1.3" stroke-linecap="round"/></svg>""",
    "traffic_max": """<svg xmlns="http://www.w3.org/2000/svg" width="14" height="14" viewBox="0 0 14 14" fill="none"><path d="M4.5 9.5L9.5 4.5M4.5 4.5H9.5V9.5" stroke="#004d00" stroke-width="1.2" stroke-linecap="round" stroke-linejoin="round"/></svg>""",
    "traffic_min": """<svg xmlns="http://www.w3.org/2000/svg" width="14" height="14" viewBox="0 0 14 14" fill="none"><line x1="3.8" y1="7" x2="10.2" y2="7" stroke="#664400" stroke-width="1.3" stroke-linecap="round"/></svg>""",
    "wave": """<svg xmlns="http://www.w3.org/2000/svg" width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="rgba(255,255,255,0.85)" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><path d="M2 10v4"/><path d="M6 6v12"/><path d="M10 3v18"/><path d="M14 8v8"/><path d="M18 5v14"/><path d="M22 10v4"/></svg>""",
    "sliders": """<svg xmlns="http://www.w3.org/2000/svg" width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="rgba(255,255,255,0.85)" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><line x1="4" x2="4" y1="21" y2="14"/><line x1="4" x2="4" y1="10" y2="3"/><line x1="12" x2="12" y1="21" y2="12"/><line x1="12" x2="12" y1="8" y2="3"/><line x1="20" x2="20" y1="21" y2="16"/><line x1="20" x2="20" y1="12" y2="3"/><line x1="1" x2="7" y1="14" y2="14"/><line x1="9" x2="15" y1="8" y2="8"/><line x1="17" x2="23" y1="16" y2="16"/></svg>""",
    "refresh": """<svg xmlns="http://www.w3.org/2000/svg" width="13" height="13" viewBox="0 0 24 24" fill="none" stroke="rgba(255,255,255,0.8)" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><path d="M21 12a9 9 0 0 0-9-9 9.75 9.75 0 0 0-6.74 2.74L3 8"/><path d="M3 3v5h5"/><path d="M3 12a9 9 0 0 0 9 9 9.75 9.75 0 0 0 6.74-2.74L21 16"/><path d="M16 21h5v-5"/></svg>""",
    "chip": """<svg xmlns="http://www.w3.org/2000/svg" width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="rgba(255,255,255,0.85)" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><rect width="16" height="16" x="4" y="4" rx="2"/><rect width="6" height="6" x="9" y="9" rx="1"/><path d="M9 1v3"/><path d="M15 1v3"/><path d="M9 20v3"/><path d="M15 20v3"/><path d="M20 9h3"/><path d="M20 14h3"/><path d="M1 9h3"/><path d="M1 14h3"/></svg>""",
    "shield_check": """<svg xmlns="http://www.w3.org/2000/svg" width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="rgba(255,255,255,0.85)" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><path d="M12 22s8-4 8-10V5l-8-3-8 3v7c0 6 8 10 8 10z"/><path d="m9 12 2 2 4-4"/></svg>""",
    "zap": """<svg xmlns="http://www.w3.org/2000/svg" width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="rgba(255,255,255,0.85)" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><polygon points="13 2 3 14 12 14 11 22 21 10 12 10 13 2"/></svg>""",
    "trash": """<svg xmlns="http://www.w3.org/2000/svg" width="13" height="13" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><path d="M3 6h18"/><path d="M19 6v14c0 1-1 2-2 2H7c-1 0-2-1-2-2V6"/><path d="M8 6V4c0-1 1-2 2-2h4c1 0 2 1 2 2v2"/><line x1="10" x2="10" y1="11" y2="17"/><line x1="14" x2="14" y1="11" y2="17"/></svg>""",
    "sparkles": """<svg xmlns="http://www.w3.org/2000/svg" width="15" height="15" viewBox="0 0 24 24" fill="none" stroke="rgba(255,255,255,0.85)" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><path d="m12 3-1.9 5.8a2 2 0 0 1-1.3 1.3L3 12l5.8 1.9a2 2 0 0 1 1.3 1.3L12 21l1.9-5.8a2 2 0 0 1 1.3-1.3L21 12l-5.8-1.9a2 2 0 0 1-1.3-1.3Z"/></svg>""",
    "flask": """<svg xmlns="http://www.w3.org/2000/svg" width="15" height="15" viewBox="0 0 24 24" fill="none" stroke="rgba(255,255,255,0.85)" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><path d="M10 2v7.31L4.69 18.5A2 2 0 0 0 6.4 21.5h11.2a2 2 0 0 0 1.71-3L14 9.31V2"/><path d="M8.5 2h7"/><path d="M6.5 16h11"/></svg>""",
}

_MACOS_LIQUID_GLASS_CSS = """
/* =========================================================================
   Apple macOS Liquid Glass Design System
   Specification: docs/SKILL-liquid-glass-ui.md
   ========================================================================= */

:root {
    --surface-canvas: #171922;
    --surface-content: #20232d;
    --surface-elevated: #282b37;
    --text-primary: #f5f5fa;
    --text-secondary: #b8bdcc;
    --text-tertiary: #858da2;
    --accent: #8b75ff;
    --accent-hover: #a091ff;
    --accent-pressed: #7358e8;
    --accent-subtle: rgba(139, 117, 255, 0.14);
    --accent-border: rgba(139, 117, 255, 0.45);
    --border-subtle: rgba(255, 255, 255, 0.08);
    --border-glass: rgba(255, 255, 255, 0.14);
    --glass-tint: rgba(27, 29, 39, 0.72);
    --focus-ring: rgba(157, 137, 255, 0.92);
    --radius-window: 20px;
    --radius-panel: 14px;
    --radius-control: 11px;
    --radius-pill: 999px;
    --space-1: 4px;
    --space-2: 8px;
    --space-3: 12px;
    --space-4: 16px;
    --space-5: 20px;
    --space-6: 24px;
    --space-8: 32px;
}

/* Base Window Container */
window {
    background: transparent;
}

window.macos-window {
    background-color: var(--surface-canvas);
    color: var(--text-primary);
    border-radius: var(--radius-window);
    border: 1px solid var(--border-glass);
    box-shadow: 0 24px 60px rgba(0, 0, 0, 0.6), 0 4px 16px rgba(0, 0, 0, 0.35);
}

/* Maximized / Full Screen */
window.macos-window.window-maximized {
    border-radius: 0;
    border: none;
    box-shadow: none;
}

window.macos-window.window-maximized .sidebar-panel {
    border-top-left-radius: 0;
    border-bottom-left-radius: 0;
}

window.macos-window.window-maximized .footer-bar {
    border-bottom-right-radius: 0;
}

/* Scrolled Windows & Viewports */
scrolledwindow,
scrolledwindow viewport,
viewport {
    background: transparent;
    border: none;
    box-shadow: none;
}

scrollbar,
scrollbar trough,
scrollbar slider {
    background: transparent;
    border: none;
    min-width: 0;
    min-height: 0;
    opacity: 0;
}

/* Sidebar (Functional Control / Glass Layer) */
.sidebar-panel {
    background-color: var(--glass-tint);
    border-right: 1px solid var(--border-subtle);
    padding: 16px 14px 14px 14px;
    border-top-left-radius: var(--radius-window);
    border-bottom-left-radius: var(--radius-window);
    min-width: 215px;
}

/* Apple Traffic Light Controls */
button.traffic-light {
    border-radius: 50%;
    min-width: 12px;
    min-height: 12px;
    padding: 0;
    margin: 0;
    border: 0.5px solid rgba(0, 0, 0, 0.25);
    background-image: none;
    box-shadow: inset 0 1px 1px rgba(255, 255, 255, 0.35);
    transition: all 120ms ease;
}

button.traffic-light image {
    opacity: 0;
    transition: opacity 120ms ease;
}

.traffic-box:hover button.traffic-light image,
button.traffic-light:hover image {
    opacity: 0.85;
}

button.traffic-red,
button.traffic-light.traffic-red {
    background: #ff5f57;
    background-color: #ff5f57;
    background-image: none;
}
button.traffic-red:hover,
button.traffic-light.traffic-red:hover {
    background: #ff6e67;
    background-color: #ff6e67;
}

button.traffic-yellow,
button.traffic-light.traffic-yellow {
    background: #febc2e;
    background-color: #febc2e;
    background-image: none;
}
button.traffic-yellow:hover,
button.traffic-light.traffic-yellow:hover {
    background: #ffc848;
    background-color: #ffc848;
}

button.traffic-green,
button.traffic-light.traffic-green {
    background: #28c840;
    background-color: #28c840;
    background-image: none;
}
button.traffic-green:hover,
button.traffic-light.traffic-green:hover {
    background: #34d74c;
    background-color: #34d74c;
}

/* App Brand Header */
.app-brand-badge {
    background: transparent;
    border-radius: 8px;
    min-width: 32px;
    min-height: 32px;
    border: none;
    box-shadow: none;
    padding: 0;
}
.app-brand-badge.is-fallback {
    background: linear-gradient(135deg, #6366f1 0%, #8b5cf6 100%);
    border: 1px solid rgba(255, 255, 255, 0.22);
    box-shadow: 0 2px 8px rgba(99, 102, 241, 0.35);
}

/* Navigation Items */
.nav-item {
    background: transparent;
    border: 1px solid transparent;
    border-radius: var(--radius-control);
    padding: 8px 10px;
    margin-bottom: 3px;
    transition: all 140ms ease;
}

.nav-item:hover {
    background-color: rgba(255, 255, 255, 0.05);
    border-color: rgba(255, 255, 255, 0.06);
}

.nav-item-active {
    background-color: var(--accent-subtle);
    border: 1px solid var(--accent-border);
}

/* Engine Status Card (Sidebar Bottom) */
.engine-status-card {
    background-color: rgba(255, 255, 255, 0.04);
    border: 1px solid rgba(255, 255, 255, 0.08);
    border-radius: 12px;
    padding: 10px 14px;
    margin-top: 6px;
    margin-bottom: 2px;
    transition: all 180ms ease;
}

.engine-status-card:hover {
    background-color: rgba(255, 255, 255, 0.06);
    border-color: rgba(255, 255, 255, 0.12);
}

.status-dot {
    min-width: 9px;
    min-height: 9px;
    border-radius: 50%;
    transition: all 250ms ease;
}

.status-dot-initializing {
    background-color: #f59e0b;
    box-shadow: 0 0 8px rgba(245, 158, 11, 0.75);
}

.status-dot-ready {
    background-color: #10b981;
    box-shadow: 0 0 10px rgba(16, 185, 129, 0.65), 0 0 3px rgba(16, 185, 129, 0.9);
}

.status-dot-recording {
    background-color: #ef4444;
    box-shadow: 0 0 8px rgba(239, 68, 68, 0.75);
}

.status-dot-processing {
    background-color: #a855f7;
    box-shadow: 0 0 8px rgba(168, 85, 247, 0.75);
}

.sidebar-loading-bar {
    min-height: 4px;
    padding: 0;
    margin-top: 4px;
    margin-bottom: 4px;
}

.sidebar-loading-bar trough {
    background-color: rgba(255, 255, 255, 0.08);
    border: none;
    border-radius: 3px;
    min-height: 4px;
}

.sidebar-loading-bar progress {
    background-color: #8b75ff;
    border-radius: 3px;
    min-height: 4px;
    box-shadow: 0 0 6px rgba(139, 117, 255, 0.5);
}

/* Center Panel (Content Layer) */
.center-panel {
    background-color: var(--surface-canvas);
    padding: 22px 24px 14px 24px;
}

button.btn-learn,
.btn-learn {
    background: rgba(255, 255, 255, 0.06);
    background-color: rgba(255, 255, 255, 0.06);
    background-image: none;
    border: 1px solid var(--border-glass);
    border-radius: var(--radius-control);
    color: var(--text-secondary);
    font-size: 11px;
    font-weight: 600;
    padding: 5px 12px;
    transition: all 140ms ease;
}

button.btn-learn:hover,
.btn-learn:hover {
    background: rgba(255, 255, 255, 0.1);
    background-color: rgba(255, 255, 255, 0.1);
    color: var(--text-primary);
    border-color: rgba(255, 255, 255, 0.22);
}

/* Model Row Items */
.model-card-row {
    background-color: var(--surface-content);
    border: 1px solid var(--border-subtle);
    border-radius: var(--radius-control);
    padding: 9px 13px;
    margin-bottom: 5px;
    box-shadow: 0 1px 3px rgba(0, 0, 0, 0.2);
    transition: all 140ms ease;
}

.model-card-row:hover {
    background-color: var(--surface-elevated);
    border-color: rgba(255, 255, 255, 0.12);
}

.model-card-selected {
    background-color: var(--accent-subtle);
    border: 1.5px solid var(--accent-border);
}

.pill-size {
    background-color: rgba(255, 255, 255, 0.07);
    border: 1px solid rgba(255, 255, 255, 0.1);
    border-radius: 6px;
    padding: 2px 7px;
    font-size: 10px;
    font-weight: 600;
    color: var(--text-secondary);
}

.pill-recommended {
    background-color: rgba(16, 185, 129, 0.14);
    border: 1px solid rgba(52, 211, 153, 0.35);
    border-radius: 6px;
    padding: 2px 7px;
    font-size: 9.5px;
    font-weight: 700;
    color: #6ee7b7;
}

button.btn-download-icon,
.btn-download-icon {
    background: rgba(255, 255, 255, 0.06);
    background-color: rgba(255, 255, 255, 0.06);
    background-image: none;
    border: 1px solid var(--border-subtle);
    border-radius: 8px;
    min-width: 30px;
    min-height: 30px;
    padding: 0;
    transition: all 140ms ease;
}

button.btn-download-icon:hover,
.btn-download-icon:hover {
    background: var(--accent-subtle);
    background-color: var(--accent-subtle);
    border-color: var(--accent-border);
}

button.btn-cancel-icon,
.btn-cancel-icon {
    background: rgba(239, 68, 68, 0.14);
    background-color: rgba(239, 68, 68, 0.14);
    background-image: none;
    border: 1px solid rgba(239, 68, 68, 0.35);
    border-radius: 8px;
    min-width: 30px;
    min-height: 30px;
    padding: 0;
    transition: all 140ms ease;
}

button.btn-cancel-icon:hover,
.btn-cancel-icon:hover {
    background: rgba(239, 68, 68, 0.25);
    background-color: rgba(239, 68, 68, 0.25);
    border-color: #f87171;
}

/* Right Inspector Panel (Contextual Layer) */
.inspector-panel {
    background-color: var(--surface-canvas);
    border-left: 1px solid var(--border-subtle);
    padding: 22px 18px 14px 18px;
    min-width: 320px;
}

.inspector-card {
    background-color: var(--surface-content);
    border: 1px solid var(--border-subtle);
    border-radius: var(--radius-panel);
    padding: 15px 16px;
    margin-bottom: 11px;
    box-shadow: 0 2px 8px rgba(0, 0, 0, 0.2);
}

.inspector-badge {
    background: linear-gradient(135deg, #6366f1 0%, #8b5cf6 100%);
    border: 1px solid rgba(255, 255, 255, 0.2);
    border-radius: 7px;
    min-width: 26px;
    min-height: 26px;
    padding: 0;
}

.mini-pbar {
    min-height: 6px;
    border-radius: var(--radius-pill);
}

.mini-pbar trough {
    background-color: rgba(255, 255, 255, 0.08);
    border: none;
    border-radius: var(--radius-pill);
    min-height: 6px;
}

.mini-pbar progress {
    background-color: var(--accent);
    border-radius: var(--radius-pill);
    min-height: 6px;
}

.kbd-chip {
    background-color: var(--surface-elevated);
    border: 1px solid rgba(255, 255, 255, 0.14);
    border-bottom: 2px solid rgba(255, 255, 255, 0.24);
    border-radius: 7px;
    padding: 4px 10px;
    font-weight: 700;
    font-size: 11px;
    color: var(--text-primary);
}

button.btn-edit-pill,
.btn-edit-pill {
    background: rgba(255, 255, 255, 0.06);
    background-color: rgba(255, 255, 255, 0.06);
    background-image: none;
    border: 1px solid var(--border-subtle);
    border-radius: 7px;
    color: var(--text-secondary);
    font-size: 11px;
    font-weight: 600;
    padding: 3px 8px;
    transition: all 120ms ease;
}

button.btn-edit-pill:hover,
.btn-edit-pill:hover {
    background: rgba(255, 255, 255, 0.1);
    background-color: rgba(255, 255, 255, 0.1);
    color: var(--text-primary);
    border-color: rgba(255, 255, 255, 0.2);
}

/* Footer Bottom Dock */
.footer-bar {
    border-top: 1px solid var(--border-subtle);
    background-color: var(--surface-canvas);
    padding: 12px 22px;
    border-bottom-right-radius: var(--radius-window);
}

button.btn-quit,
.btn-quit {
    background: rgba(255, 255, 255, 0.06);
    background-color: rgba(255, 255, 255, 0.06);
    background-image: none;
    border: 1px solid var(--border-subtle);
    color: var(--text-secondary);
    border-radius: 9px;
    padding: 7px 20px;
    font-weight: 600;
    font-size: 11.5px;
    transition: all 130ms ease;
}

button.btn-quit:hover,
.btn-quit:hover {
    background: rgba(255, 255, 255, 0.1);
    background-color: rgba(255, 255, 255, 0.1);
    color: var(--text-primary);
    border-color: rgba(255, 255, 255, 0.18);
}

button.btn-save-purple,
.btn-save-purple {
    background: #8b75ff;
    background-color: #8b75ff;
    background-image: none;
    border: 1px solid rgba(255, 255, 255, 0.2);
    border-radius: 9px;
    color: #ffffff;
    font-weight: 700;
    font-size: 11.5px;
    padding: 7px 28px;
    box-shadow: 0 2px 10px rgba(139, 117, 255, 0.35);
    transition: all 130ms ease;
}

button.btn-save-purple:hover,
.btn-save-purple:hover {
    background: #a091ff;
    background-color: #a091ff;
    box-shadow: 0 3px 12px rgba(139, 117, 255, 0.5);
}

button.btn-save-purple:active,
.btn-save-purple:active {
    background: #7358e8;
    background-color: #7358e8;
}

/* Inputs / Glass Entry */
.glass-entry {
    background-color: var(--surface-elevated);
    border: 1px solid var(--border-subtle);
    border-radius: 8px;
    color: var(--text-primary);
    padding: 7px 11px;
}

.glass-entry:focus {
    border-color: var(--accent);
    box-shadow: 0 0 0 1px var(--accent);
}

/* Audio Section */
.audio-panel {
    background-color: var(--surface-canvas);
    padding: 22px 26px 16px 26px;
}

.audio-card {
    background-color: var(--surface-content);
    border: 1px solid var(--border-subtle);
    border-radius: var(--radius-panel);
    padding: 16px 18px;
    margin-bottom: 12px;
    box-shadow: 0 2px 8px rgba(0, 0, 0, 0.2);
}

.audio-badge {
    background: linear-gradient(135deg, #6366f1 0%, #8b5cf6 100%);
    border-radius: 8px;
    min-width: 28px;
    min-height: 28px;
    padding: 0;
}

.mic-device-row {
    background-color: var(--surface-content);
    border: 1px solid var(--border-subtle);
    border-radius: var(--radius-control);
    padding: 9px 12px;
    margin-bottom: 5px;
    transition: all 140ms ease;
}

.mic-device-row:hover {
    background-color: var(--surface-elevated);
    border-color: rgba(255, 255, 255, 0.12);
}

.mic-device-selected {
    background-color: var(--accent-subtle);
    border: 1.5px solid var(--accent-border);
}

.vu-meter-bar {
    min-height: 10px;
    border-radius: var(--radius-pill);
}

.vu-meter-bar trough {
    background-color: rgba(255, 255, 255, 0.08);
    border: none;
    border-radius: var(--radius-pill);
    min-height: 10px;
}

.vu-meter-bar progress {
    background: linear-gradient(90deg, #10b981 0%, #3b82f6 60%, #8b5cf6 85%, #ef4444 100%);
    border-radius: var(--radius-pill);
    min-height: 10px;
}

button.btn-test-mic,
.btn-test-mic {
    background: rgba(255, 255, 255, 0.06);
    background-color: rgba(255, 255, 255, 0.06);
    background-image: none;
    border: 1px solid var(--border-subtle);
    border-radius: var(--radius-control);
    color: var(--text-primary);
    font-weight: 600;
    font-size: 11.5px;
    padding: 6px 16px;
    transition: all 140ms ease;
}

button.btn-test-mic:hover,
.btn-test-mic:hover {
    background: var(--accent-subtle);
    background-color: var(--accent-subtle);
    border-color: var(--accent-border);
}

button.btn-test-mic-active,
.btn-test-mic-active {
    background: #ef4444;
    background-color: #ef4444;
    background-image: none;
    border: 1px solid rgba(255, 255, 255, 0.25);
    border-radius: var(--radius-control);
    color: #ffffff;
    font-weight: 700;
    font-size: 11.5px;
    padding: 6px 16px;
}

button.preset-pill,
.preset-pill {
    background: rgba(255, 255, 255, 0.06);
    background-color: rgba(255, 255, 255, 0.06);
    background-image: none;
    border: 1px solid var(--border-subtle);
    border-radius: var(--radius-pill);
    color: var(--text-secondary);
    font-size: 11px;
    font-weight: 600;
    padding: 4px 14px;
    transition: all 130ms ease;
}

button.preset-pill:hover,
.preset-pill:hover {
    background: rgba(255, 255, 255, 0.1);
    background-color: rgba(255, 255, 255, 0.1);
    color: var(--text-primary);
}

button.preset-pill-active,
.preset-pill-active {
    background: var(--accent-subtle);
    background-color: var(--accent-subtle);
    background-image: none;
    border: 1px solid var(--accent-border);
    color: #ffffff;
    font-weight: 700;
}

.badge-speech-detected {
    background-color: rgba(16, 185, 129, 0.16);
    border: 1px solid rgba(52, 211, 153, 0.4);
    border-radius: 6px;
    padding: 3px 8px;
    font-size: 10px;
    font-weight: 700;
    color: #6ee7b7;
}

.badge-speech-idle {
    background-color: rgba(255, 255, 255, 0.06);
    border: 1px solid var(--border-subtle);
    border-radius: 6px;
    padding: 3px 8px;
    font-size: 10px;
    font-weight: 600;
    color: var(--text-tertiary);
}


scale.glass-scale trough {
    background-color: rgba(255, 255, 255, 0.08);
    border: none;
    border-radius: var(--radius-pill);
    min-height: 5px;
}

scale.glass-scale highlight {
    background-color: var(--accent);
    border-radius: var(--radius-pill);
    min-height: 5px;
}

scale.glass-scale slider {
    background-color: #ffffff;
    border-radius: 50%;
    min-width: 15px;
    min-height: 15px;
    margin: -5px;
    border: 1px solid rgba(255, 255, 255, 0.8);
    box-shadow: 0 2px 6px rgba(0, 0, 0, 0.4);
}

/* Push-to-talk Studio */
.kbd-chip-large {
    background-color: var(--surface-elevated);
    border: 1px solid rgba(255, 255, 255, 0.16);
    border-bottom: 3px solid rgba(255, 255, 255, 0.3);
    border-radius: 10px;
    padding: 8px 18px;
    font-weight: 800;
    font-size: 14px;
    color: var(--text-primary);
}

.kbd-chip-modifier {
    background-color: var(--accent-subtle);
    border: 1px solid var(--accent-border);
    border-bottom: 3px solid var(--accent);
    border-radius: 10px;
    padding: 8px 18px;
    font-weight: 800;
    font-size: 14px;
    color: var(--text-primary);
}

button.btn-record-chord,
.btn-record-chord {
    background: #8b75ff;
    background-color: #8b75ff;
    background-image: none;
    border: 1px solid rgba(255, 255, 255, 0.2);
    border-radius: var(--radius-control);
    color: #ffffff;
    font-weight: 700;
    font-size: 12px;
    padding: 7px 20px;
    box-shadow: 0 2px 10px rgba(139, 117, 255, 0.35);
    transition: all 140ms ease;
}

button.btn-record-chord:hover,
.btn-record-chord:hover {
    background: var(--accent-hover);
    background-color: var(--accent-hover);
}

button.btn-record-active,
.btn-record-active {
    background: #ef4444;
    background-color: #ef4444;
    background-image: none;
    border: 1px solid rgba(255, 255, 255, 0.3);
    border-radius: var(--radius-control);
    color: #ffffff;
    font-weight: 700;
    font-size: 12px;
    padding: 7px 20px;
    box-shadow: 0 0 16px rgba(239, 68, 68, 0.45);
}

.preset-card-row {
    background-color: var(--surface-content);
    border: 1px solid var(--border-subtle);
    border-radius: var(--radius-control);
    padding: 10px 14px;
    margin-bottom: 5px;
    transition: all 140ms ease;
}

.preset-card-row:hover {
    background-color: var(--surface-elevated);
    border-color: rgba(255, 255, 255, 0.12);
}

.preset-card-selected {
    background-color: var(--accent-subtle);
    border: 1.5px solid var(--accent-border);
}

/* Dictionary Studio */
.dict-panel {
    background-color: var(--surface-canvas);
    padding: 22px 26px 20px 26px;
}

.dict-card {
    background-color: var(--surface-content);
    border: 1px solid var(--border-subtle);
    border-radius: var(--radius-panel);
    padding: 16px 18px;
    margin-bottom: 12px;
    box-shadow: 0 2px 8px rgba(0, 0, 0, 0.2);
}

.dict-badge {
    background-color: rgba(16, 185, 129, 0.16);
    color: #6ee7b7;
    border: 1px solid rgba(52, 211, 153, 0.35);
    border-radius: 6px;
    padding: 2px 8px;
    font-size: 10px;
    font-weight: 700;
}

.dict-rule-row {
    background-color: var(--surface-content);
    border: 1px solid var(--border-subtle);
    border-radius: var(--radius-control);
    padding: 9px 13px;
    margin-bottom: 5px;
    transition: all 140ms ease;
}

.dict-rule-row:hover {
    background-color: var(--surface-elevated);
    border-color: rgba(255, 255, 255, 0.12);
}

.dict-chip-spoken {
    background-color: var(--surface-elevated);
    border: 1px solid var(--border-subtle);
    border-radius: 7px;
    padding: 4px 11px;
    font-weight: 600;
    color: var(--text-primary);
}

.dict-chip-repl {
    background-color: var(--accent-subtle);
    border: 1px solid var(--accent-border);
    border-radius: 7px;
    padding: 4px 11px;
    font-weight: 700;
    color: #ffffff;
}

button.dict-btn-del,
.dict-btn-del {
    background: rgba(244, 63, 94, 0.12);
    background-color: rgba(244, 63, 94, 0.12);
    background-image: none;
    border: 1px solid rgba(244, 63, 94, 0.25);
    color: #fb7185;
    border-radius: 7px;
    padding: 4px 10px;
    font-size: 11px;
    font-weight: 600;
    transition: all 130ms ease;
}

button.dict-btn-del:hover,
.dict-btn-del:hover {
    background: rgba(244, 63, 94, 0.22);
    background-color: rgba(244, 63, 94, 0.22);
    border-color: rgba(244, 63, 94, 0.45);
    color: #fda4af;
}

.dict-preset-chip {
    background-color: var(--surface-elevated);
    border: 1px solid var(--border-subtle);
    border-radius: 10px;
    padding: 10px 14px;
    color: var(--text-secondary);
    transition: all 140ms ease;
}

.dict-preset-chip:hover {
    border-color: var(--accent-border);
    color: var(--text-primary);
}

.dict-preview-row {
    padding: 4px 6px;
    border-radius: 6px;
}

.dict-preview-row:hover {
    background-color: rgba(255, 255, 255, 0.04);
}

.dict-test-box {
    background-color: var(--surface-canvas);
    border: 1px solid var(--border-subtle);
    border-radius: 10px;
    padding: 12px 14px;
}
"""


def main(argv: list[str] | None = None) -> int:
    gtk = _load_gtk()
    app = SettingsApp(gtk)
    return app.run(argv)


def _load_gtk():
    try:
        import gi
    except ModuleNotFoundError:
        sys.path.append("/usr/lib/python3/dist-packages")
        import gi
    gi.require_version("Gtk", "4.0")
    from gi.repository import Gtk

    return Gtk


class SettingsApp:
    def __init__(self, gtk, on_quit=None, on_save_config=None) -> None:
        self.gtk = gtk
        self._on_quit = on_quit
        self._on_save_config = on_save_config
        self._config = load_config()
        self._device = select_device(self._config.asr.device)
        self._selected_model_name = self._config.asr.model
        self._hotkey_keys = tuple(self._config.hotkey.keys)
        self._dictionary_entries_list = list(self._config.dictionary)

        # UI Widget references
        self.window = None
        self._gtk_app = None
        self._status_label = gtk.Label(xalign=0)
        self._model_row_boxes: dict[str, object] = {}
        self._model_name_labels: dict[str, object] = {}
        self._radio_indicators: dict[str, object] = {}
        self._download_buttons: dict[str, object] = {}
        self._download_cancel_buttons: dict[str, object] = {}
        self._download_pbar: dict[str, object] = {}
        self._download_pbar_labels: dict[str, object] = {}
        self._download_pbar_boxes: dict[str, object] = {}
        self._download_status_labels: dict[str, object] = {}
        self._download_cancel_events: dict[str, threading.Event] = {}

        # Inspector widgets
        self._inspector_model_title = None
        self._inspector_model_desc = None
        self._inspector_params = None
        self._inspector_size = None
        self._inspector_ram = None
        self._inspector_lang = None
        self._inspector_best_for = None
        self._inspector_speed_bar = None
        self._inspector_speed_lbl = None
        self._inspector_acc_bar = None
        self._inspector_acc_lbl = None

        # Push to talk widgets
        self._hotkey_box = None
        self._dict_preview_box = None
        self._capture_stop = threading.Event()

        # Push-to-talk Studio state
        self._is_capturing_shortcut = False
        self._ptt_hotkey_box = None
        self._ptt_record_btn = None
        self._ptt_status_lbl = None
        self._ptt_preset_rows: dict[str, object] = {}
        self._ptt_preset_radio_imgs: dict[str, object] = {}

        # Audio & Mic state
        self._selected_audio_device = self._config.audio.device
        self._vad_rms_threshold = float(self._config.vad.rms_threshold)
        self._vad_noise_mult = float(self._config.vad.noise_multiplier)
        self._vad_hangover_ms = int(self._config.vad.hangover_ms)
        self._vad_highpass_hz = float(self._config.vad.highpass_hz)

        self._is_testing_mic = False
        self._test_capture: AudioCapture | None = None
        self._test_timer_id: int | None = None
        self._meter_level = 0.0
        self._meter_target = 0.0

        # Audio widgets
        self._test_mic_btn = None
        self._test_vu_meter = None
        self._test_db_label = None
        self._test_speech_badge = None
        self._rms_scale = None
        self._rms_val_lbl = None
        self._hangover_scale = None
        self._hangover_val_lbl = None
        self._highpass_scale = None
        self._highpass_val_lbl = None
        self._preset_pills: dict[str, object] = {}
        self._mic_device_rows: dict[str, object] = {}
        self._mic_radio_indicators: dict[str, object] = {}
        self._devices_list_box = None

        # Dictionary Studio state
        self._dict_search_query = ""
        self._dict_rules_vbox = None
        self._dict_badge_lbl = None
        self._dict_count_lbl = None
        self._dict_new_phrase_entry = None
        self._dict_new_repl_entry = None
        self._dict_add_feedback = None
        self._dict_test_input = None
        self._dict_test_output = None

        # Navigation state
        self._main_stack = None
        self._sidebar_nav_rows: dict[str, object] = {}

        # Engine status card state
        self._engine_state = "ready"
        self._engine_title = "Loading Engine..."
        self._engine_sub = "Compiling GPU shaders"
        self._engine_ready_sub = f"{self._device.name} • {self._config.asr.model}"
        self._engine_stack = None
        self._status_dot = None
        self._loading_pbar = None
        self._loading_pct_label = None
        self._loading_title_label = None
        self._loading_sub_label = None
        self._ready_title_label = None
        self._ready_sub_label = None

    def _make_svg(self, name: str):
        from gi.repository import Gdk, GLib

        svg_data = SVG_ICONS[name].encode("utf-8")
        bytes_data = GLib.Bytes.new(svg_data)
        texture = Gdk.Texture.new_from_bytes(bytes_data)
        img = self.gtk.Image.new_from_paintable(texture)
        img.set_halign(self.gtk.Align.CENTER)
        img.set_valign(self.gtk.Align.CENTER)
        return img

    def _make_badge(self, name: str, css_class: str = "audio-badge", size: int = 28):
        gtk = self.gtk
        badge = gtk.Box(valign=gtk.Align.CENTER, halign=gtk.Align.CENTER)
        badge.add_css_class(css_class)
        badge.set_size_request(size, size)
        badge.set_hexpand(False)
        badge.set_vexpand(False)

        ic = self._make_svg(name)
        ic.set_halign(gtk.Align.CENTER)
        ic.set_valign(gtk.Align.CENTER)
        ic.set_hexpand(True)
        ic.set_vexpand(True)
        badge.append(ic)
        return badge

    def _make_brand_icon(self, badge=None):
        from whisperflow.desktop import find_logo_path

        logo_path = find_logo_path()
        if logo_path and logo_path.is_file():
            try:
                from gi.repository import GdkPixbuf, Gdk, GLib

                # Render at 64x64 for crisp 2x retina display and strictly lock widget to 32px
                pixbuf = GdkPixbuf.Pixbuf.new_from_file_at_scale(str(logo_path), 64, 64, True)
                success, buffer = pixbuf.save_to_bufferv("png", [], [])
                if success:
                    bytes_data = GLib.Bytes.new(buffer)
                    texture = Gdk.Texture.new_from_bytes(bytes_data)
                    img = self.gtk.Image.new_from_paintable(texture)
                    img.set_pixel_size(32)
                    img.set_hexpand(False)
                    img.set_vexpand(False)
                    return img
            except Exception:
                pass
        if badge is not None:
            badge.add_css_class("is-fallback")
        return self._make_svg("cube")

    def run(self, argv: list[str] | None) -> int:
        app = self.gtk.Application(application_id="com.whisperflow.Settings")
        app.connect("activate", self._activate)
        return app.run(None if argv is None else argv)

    def _activate(self, app) -> None:
        window = self.build(app)
        window.present()

    def _apply_css(self) -> None:
        from gi.repository import Gdk

        display = Gdk.Display.get_default()
        if display is not None:
            provider = self.gtk.CssProvider()
            provider.load_from_data(_MACOS_LIQUID_GLASS_CSS.encode("utf-8"))
            self.gtk.StyleContext.add_provider_for_display(
                display,
                provider,
                self.gtk.STYLE_PROVIDER_PRIORITY_APPLICATION,
            )

    def build(self, app):
        gtk = self.gtk
        self._gtk_app = app
        self._apply_css()

        window = gtk.ApplicationWindow(application=app, title="Wispo")
        window.set_icon_name("com.whisperflow.WhisperFlow")
        window.set_default_size(1040, 720)
        window.set_size_request(850, 560)  # Responsive minimum size
        window.set_resizable(True)
        window.set_decorated(False)  # Frameless for native macOS style
        window.add_css_class("macos-window")
        self.window = window

        # Top-level layout: Horizontal layout so sidebar spans full vertical height
        root_hbox = gtk.Box(orientation=gtk.Orientation.HORIZONTAL)
        root_hbox.set_hexpand(True)
        root_hbox.set_vexpand(True)

        # 1. Left Sidebar (Full height, fixed width)
        sidebar = self._build_sidebar()
        sidebar.set_size_request(215, -1)
        sidebar.set_hexpand(False)
        sidebar.set_vexpand(True)
        root_hbox.append(sidebar)

        # 2. Right Container (VBox: Header + Main Body + Footer)
        right_vbox = gtk.Box(orientation=gtk.Orientation.VERTICAL)
        right_vbox.set_hexpand(True)
        right_vbox.set_vexpand(True)

        # Main 2-column container (Center Model List + Right Inspector)
        columns_hbox = gtk.Box(orientation=gtk.Orientation.HORIZONTAL)
        columns_hbox.set_hexpand(True)
        columns_hbox.set_vexpand(True)

        center = self._build_center_panel()
        center.set_hexpand(True)
        center.set_halign(gtk.Align.FILL)
        center.set_vexpand(True)
        columns_hbox.append(center)

        inspector = self._build_inspector_panel()
        inspector.set_size_request(320, -1)
        inspector.set_hexpand(False)
        inspector.set_halign(gtk.Align.END)
        inspector.set_vexpand(True)
        columns_hbox.append(inspector)

        # Main Stack for switching between sections
        self._main_stack = gtk.Stack()
        self._main_stack.set_transition_type(gtk.StackTransitionType.CROSSFADE)
        self._main_stack.set_transition_duration(150)
        self._main_stack.set_hexpand(True)
        self._main_stack.set_vexpand(True)

        # 1. Model Section (2-column layout)
        self._main_stack.add_named(columns_hbox, "model")

        # 2. Audio Section (Full width)
        audio_panel = self._build_audio_panel()
        self._main_stack.add_named(audio_panel, "audio")

        # 3. Push-to-talk Section (Full width studio)
        ptt_panel = self._build_push_to_talk_panel()
        self._main_stack.add_named(ptt_panel, "push_to_talk")

        # 4. Dictionary Section (Full width studio)
        dict_panel = self._build_dictionary_panel()
        self._main_stack.add_named(dict_panel, "dictionary")

        # 5. Placeholders for other sidebar sections
        placeholders = [
            ("about", "About Wispo", "System diagnostics, version info, and credits."),
        ]
        for p_name, p_title, p_sub in placeholders:
            self._main_stack.add_named(self._build_placeholder_panel(p_title, p_sub), p_name)

        # Wrap stack in a window handle so the user can drag window anywhere
        handle = gtk.WindowHandle()
        handle.set_hexpand(True)
        handle.set_vexpand(True)
        handle.set_child(self._main_stack)
        right_vbox.append(handle)

        # Bottom Footer Bar (anchored flush to the bottom)
        footer = self._build_footer()
        footer.set_hexpand(True)
        footer.set_valign(gtk.Align.END)
        right_vbox.append(footer)

        root_hbox.append(right_vbox)

        window.connect("close-request", self._on_window_close_request)
        window.connect("notify::maximized", self._on_maximized_notify)
        window.connect("notify::fullscreened", self._on_maximized_notify)
        window.set_child(root_hbox)

        # Initialize inspector with currently selected model
        self._update_inspector(self._selected_model_name)
        return window

    # -------------------------------------------------------------------------
    # Window Actions: Traffic Lights
    # -------------------------------------------------------------------------
    def _on_maximized_notify(self, window, _pspec) -> None:
        is_max = window.is_maximized() or window.is_fullscreen()
        self._update_maximized_state(is_max)

    def _update_maximized_state(self, is_max: bool) -> None:
        if not self.window:
            return
        if is_max:
            self.window.add_css_class("window-maximized")
        else:
            self.window.remove_css_class("window-maximized")

    def _on_red_clicked(self, *_args) -> None:
        """Quit the application."""
        for ev in list(self._download_cancel_events.values()):
            ev.set()
        self._stop_mic_test()
        self._stop_capture()
        if self._on_quit:
            self._on_quit()
        if self.window:
            self.window.close()
        if self._gtk_app:
            self._gtk_app.quit()

    def _on_yellow_clicked(self, *_args) -> None:
        """Minimize the application window."""
        if not self.window:
            return
        self.window.minimize()

    def _on_green_clicked(self, *_args) -> None:
        """Maximize or restore the application window."""
        if not self.window:
            return
        if self.window.is_maximized() or self.window.is_fullscreen():
            self.window.unmaximize()
            self._update_maximized_state(False)
        else:
            self.window.maximize()
            self._update_maximized_state(True)

    def _on_window_close_request(self, *_args) -> bool:
        for ev in list(self._download_cancel_events.values()):
            ev.set()
        self._stop_mic_test()
        self._stop_capture()
        if self._on_quit:
            self._on_quit()
        return False

    # -------------------------------------------------------------------------
    # Column 1: Left Sidebar (Full Height)
    # -------------------------------------------------------------------------
    def _build_sidebar(self):
        gtk = self.gtk
        vbox = gtk.Box(orientation=gtk.Orientation.VERTICAL, spacing=10)
        vbox.add_css_class("sidebar-panel")

        # Top Traffic Light Window Controls
        traffic_box = gtk.Box(spacing=8, orientation=gtk.Orientation.HORIZONTAL)
        traffic_box.add_css_class("traffic-box")
        traffic_box.set_margin_top(4)
        traffic_box.set_margin_bottom(12)

        # Red = Quit Application (Cross icon on hover)
        r = gtk.Button()
        r.add_css_class("traffic-light")
        r.add_css_class("traffic-red")
        r.set_size_request(14, 14)
        r.set_valign(gtk.Align.CENTER)
        r.set_child(self._make_svg("traffic_close"))
        r.set_cursor_from_name("pointer")
        r.connect("clicked", self._on_red_clicked)

        # Yellow = Minimize (Minus icon on hover)
        y = gtk.Button()
        y.add_css_class("traffic-light")
        y.add_css_class("traffic-yellow")
        y.set_size_request(14, 14)
        y.set_valign(gtk.Align.CENTER)
        y.set_child(self._make_svg("traffic_min"))
        y.set_cursor_from_name("pointer")
        y.connect("clicked", self._on_yellow_clicked)

        # Green = Maximize / Full Screen (Windows/macOS square icon on hover)
        g = gtk.Button()
        g.add_css_class("traffic-light")
        g.add_css_class("traffic-green")
        g.set_size_request(14, 14)
        g.set_valign(gtk.Align.CENTER)
        g.set_child(self._make_svg("traffic_max"))
        g.set_cursor_from_name("pointer")
        g.connect("clicked", self._on_green_clicked)

        traffic_box.append(r)
        traffic_box.append(y)
        traffic_box.append(g)
        vbox.append(traffic_box)

        # App Brand Header
        brand_box = gtk.Box(spacing=10, orientation=gtk.Orientation.HORIZONTAL)
        brand_box.set_margin_bottom(14)
        brand_box.set_hexpand(False)

        badge = gtk.Box(valign=gtk.Align.CENTER, halign=gtk.Align.START)
        badge.set_hexpand(False)
        badge.set_vexpand(False)
        badge.set_size_request(32, 32)
        badge.add_css_class("app-brand-badge")
        brand_icon = self._make_brand_icon(badge)
        badge.append(brand_icon)
        brand_box.append(badge)

        brand_text = gtk.Box(orientation=gtk.Orientation.VERTICAL, spacing=1)
        brand_text.set_hexpand(False)
        title = gtk.Label(label="Wispo", xalign=0)
        title.set_markup("<span font_weight='700' font_size='11pt' color='#ffffff'>Wispo</span>")
        sub = gtk.Label(label="Transcribe. Anywhere.", xalign=0)
        sub.set_markup("<span font_size='8.5pt' alpha='55%'>Transcribe. Anywhere.</span>")
        brand_text.append(title)
        brand_text.append(sub)
        brand_box.append(brand_text)
        vbox.append(brand_box)

        # Nav Items with crisp SVGs
        nav_items = [
            ("Model", "Choose transcription engine", "cube", True, "model"),
            ("Audio", "Microphone & detection", "microphone", False, "audio"),
            ("Push-to-talk", "Keyboard shortcut", "keyboard", False, "push_to_talk"),
            ("Dictionary", "Custom word corrections", "book", False, "dictionary"),
            ("About", "Version & updates", "info", False, "about"),
        ]

        self._sidebar_nav_rows.clear()
        for item_name, subtitle, icon_key, is_active, tab_tag in nav_items:
            row = gtk.Box(orientation=gtk.Orientation.HORIZONTAL, spacing=10)
            row.add_css_class("nav-item")
            if is_active:
                row.add_css_class("nav-item-active")

            # Dedicated fixed icon slot for perfect vertical axis and center alignment
            icon_box = gtk.Box(valign=gtk.Align.CENTER, halign=gtk.Align.CENTER)
            icon_box.set_size_request(24, 24)
            icon_box.set_hexpand(False)
            icon_box.set_vexpand(False)

            ic = self._make_svg(icon_key)
            ic.set_halign(gtk.Align.CENTER)
            ic.set_valign(gtk.Align.CENTER)
            ic.set_hexpand(True)
            ic.set_vexpand(True)
            icon_box.append(ic)
            row.append(icon_box)

            text_col = gtk.Box(orientation=gtk.Orientation.VERTICAL, spacing=1)
            t = gtk.Label(label=item_name, xalign=0)
            t.set_markup(f"<span font_weight='600' font_size='9pt' color='#ffffff'>{pango_esc(item_name)}</span>")
            s = gtk.Label(label=subtitle, xalign=0)
            s.set_markup(f"<span font_size='8pt' alpha='55%'>{pango_esc(subtitle)}</span>")
            text_col.append(t)
            text_col.append(s)
            row.append(text_col)

            click = gtk.GestureClick()
            click.connect("released", lambda _g, _n, _x, _y, t=tab_tag: self._switch_section(t))
            row.add_controller(click)
            row.set_cursor_from_name("pointer")
            self._sidebar_nav_rows[tab_tag] = row

            vbox.append(row)

        vbox.append(gtk.Box(vexpand=True))

        # Bottom Engine Status Card
        vbox.append(self._build_engine_status_card())

        return vbox

    def _build_engine_status_card(self):
        gtk = self.gtk
        card = gtk.Box(orientation=gtk.Orientation.VERTICAL)
        card.add_css_class("engine-status-card")
        card.set_hexpand(True)
        card.set_valign(gtk.Align.END)

        self._engine_stack = gtk.Stack()
        self._engine_stack.set_transition_type(gtk.StackTransitionType.CROSSFADE)
        self._engine_stack.set_transition_duration(200)

        # 1. Startup Loading View (Progress Bar + Percentage numbers)
        loading_box = gtk.Box(orientation=gtk.Orientation.VERTICAL, spacing=5)

        top_row = gtk.Box(orientation=gtk.Orientation.HORIZONTAL, spacing=6)
        self._loading_title_label = gtk.Label(xalign=0)
        self._loading_title_label.set_hexpand(True)
        self._loading_title_label.set_markup(
            f"<span font_weight='600' font_size='8.5pt' color='#ffffff'>{pango_esc(self._engine_title)}</span>"
        )
        top_row.append(self._loading_title_label)

        self._loading_pct_label = gtk.Label(xalign=1)
        self._loading_pct_label.set_markup(
            "<span font_weight='700' font_size='8.5pt' color='#8b75ff'>0%</span>"
        )
        top_row.append(self._loading_pct_label)
        loading_box.append(top_row)

        self._loading_pbar = gtk.ProgressBar()
        self._loading_pbar.add_css_class("sidebar-loading-bar")
        self._loading_pbar.set_fraction(0.0)
        loading_box.append(self._loading_pbar)

        self._loading_sub_label = gtk.Label(xalign=0)
        self._loading_sub_label.set_markup(
            f"<span font_size='7.5pt' alpha='55%'>{pango_esc(self._engine_sub)}</span>"
        )
        loading_box.append(self._loading_sub_label)

        # 2. Ready View (Clean static dot + Engine Ready, exact match to screenshot)
        ready_box = gtk.Box(orientation=gtk.Orientation.HORIZONTAL, spacing=11)
        ready_box.set_valign(gtk.Align.CENTER)

        dot_box = gtk.Box(valign=gtk.Align.CENTER, halign=gtk.Align.CENTER)
        dot_box.set_size_request(16, 16)
        dot_box.set_hexpand(False)
        dot_box.set_vexpand(False)

        self._status_dot = gtk.Box()
        self._status_dot.add_css_class("status-dot")
        self._status_dot.add_css_class("status-dot-ready")
        self._status_dot.set_valign(gtk.Align.CENTER)
        self._status_dot.set_halign(gtk.Align.CENTER)
        dot_box.append(self._status_dot)
        ready_box.append(dot_box)

        text_col = gtk.Box(orientation=gtk.Orientation.VERTICAL, spacing=2)
        text_col.set_hexpand(True)
        text_col.set_valign(gtk.Align.CENTER)

        self._ready_title_label = gtk.Label(xalign=0)
        self._ready_title_label.set_markup(
            "<span font_weight='600' font_size='9.5pt' color='#ffffff'>Engine Ready</span>"
        )
        text_col.append(self._ready_title_label)

        self._ready_sub_label = gtk.Label(xalign=0)
        ready_sub_text = self._engine_ready_sub or f"{self._device.name} • {self._config.asr.model}"
        self._ready_sub_label.set_markup(
            f"<span font_size='8pt' color='#9ca3af'>{pango_esc(ready_sub_text)}</span>"
        )
        text_col.append(self._ready_sub_label)

        ready_box.append(text_col)

        self._engine_stack.add_named(loading_box, "loading")
        self._engine_stack.add_named(ready_box, "ready")

        initial_view = "ready" if self._engine_state == "ready" else "loading"
        self._engine_stack.set_visible_child_name(initial_view)

        card.append(self._engine_stack)
        return card

    def set_loading_progress(self, percent: int, title: str = "Loading Engine...", subtitle: str = "") -> None:
        """Update the startup loading progress bar and percentage numbers in a thread-safe manner."""
        from gi.repository import GLib

        pct = max(0, min(100, int(percent)))
        self._engine_state = "initializing"
        self._engine_title = title
        self._engine_sub = subtitle

        def _update():
            if not hasattr(self, "_engine_stack") or self._engine_stack is None:
                return False

            if self._engine_stack.get_visible_child_name() != "loading":
                self._engine_stack.set_visible_child_name("loading")

            if hasattr(self, "_loading_pbar") and self._loading_pbar is not None:
                self._loading_pbar.set_fraction(pct / 100.0)

            if hasattr(self, "_loading_pct_label") and self._loading_pct_label is not None:
                self._loading_pct_label.set_markup(
                    f"<span font_weight='700' font_size='8.5pt' color='#8b75ff'>{pct}%</span>"
                )

            if hasattr(self, "_loading_title_label") and self._loading_title_label is not None:
                self._loading_title_label.set_markup(
                    f"<span font_weight='600' font_size='8.5pt' color='#ffffff'>{pango_esc(title)}</span>"
                )

            if hasattr(self, "_loading_sub_label") and self._loading_sub_label is not None:
                self._loading_sub_label.set_markup(
                    f"<span font_size='7.5pt' alpha='55%'>{pango_esc(subtitle)}</span>"
                )
            return False

        GLib.idle_add(_update)

    def set_model_loaded(self, info: str = "") -> None:
        """Switch to the static 'Engine Ready' card once warmup is complete. Decouples UI."""
        from gi.repository import GLib

        self._engine_state = "ready"
        if info:
            self._engine_ready_sub = info

        def _update():
            if not hasattr(self, "_engine_stack") or self._engine_stack is None:
                return False

            if hasattr(self, "_ready_sub_label") and self._ready_sub_label is not None and info:
                self._ready_sub_label.set_markup(
                    f"<span font_size='8pt' color='#9ca3af'>{pango_esc(info)}</span>"
                )

            if self._engine_stack.get_visible_child_name() != "ready":
                self._engine_stack.set_visible_child_name("ready")
            if hasattr(self, "_status_label") and self._status_label is not None:
                current_text = self._status_label.get_text() or ""
                if "Reloading model" in current_text:
                    self._status_label.set_markup(
                        "<span font_size='8.5pt' font_weight='700' color='#34d399'>● Model loaded and ready!</span>"
                    )
            return False

        GLib.idle_add(_update)

    def set_engine_status(self, state: str, title: str, subtitle: str = "") -> None:
        """Backward compatibility helper for engine status. Dictation states are ignored."""
        if state == "ready":
            self.set_model_loaded(subtitle)
        elif state == "initializing":
            self.set_loading_progress(10, title, subtitle)

    # -------------------------------------------------------------------------
    # Navigation & Section Switching
    # -------------------------------------------------------------------------
    def _switch_section(self, target_tag: str) -> None:
        if target_tag != "audio" and self._is_testing_mic:
            self._stop_mic_test()
        if target_tag != "push_to_talk" and self._is_capturing_shortcut:
            self._stop_capture()

        for tag, row in self._sidebar_nav_rows.items():
            if tag == target_tag:
                row.add_css_class("nav-item-active")
            else:
                row.remove_css_class("nav-item-active")

        if self._main_stack:
            self._main_stack.set_visible_child_name(target_tag)

    def _build_placeholder_panel(self, title_text: str, subtitle_text: str):
        gtk = self.gtk
        vbox = gtk.Box(orientation=gtk.Orientation.VERTICAL, spacing=16)
        vbox.add_css_class("audio-panel")
        vbox.set_hexpand(True)
        vbox.set_vexpand(True)

        top_row = gtk.Box(orientation=gtk.Orientation.HORIZONTAL, spacing=8)
        head_text = gtk.Box(orientation=gtk.Orientation.VERTICAL, spacing=2, hexpand=True)
        t = gtk.Label(label=title_text, xalign=0)
        t.set_markup(f"<span font_weight='800' font_size='18pt' color='#ffffff'>{pango_esc(title_text)}</span>")
        s = gtk.Label(label=subtitle_text, xalign=0)
        s.set_markup(f"<span font_size='9pt' alpha='55%'>{pango_esc(subtitle_text)}</span>")
        head_text.append(t)
        head_text.append(s)
        top_row.append(head_text)
        vbox.append(top_row)

        card = gtk.Box(orientation=gtk.Orientation.VERTICAL, spacing=10)
        card.add_css_class("audio-card")
        card.set_margin_top(12)
        card.set_valign(gtk.Align.START)

        c_title = gtk.Label(label=title_text, xalign=0)
        c_title.set_markup(f"<span font_weight='700' font_size='11pt' color='#ffffff'>{pango_esc(title_text)} Configuration</span>")
        card.append(c_title)

        desc = gtk.Label(xalign=0)
        desc.set_markup("<span font_size='9pt' alpha='65%'>Settings for this section are managed in <span font_family='monospace'>whisperflow.yaml</span> and can also be adjusted directly from the Model Inspector.</span>")
        card.append(desc)
        vbox.append(card)

        return vbox

    # -------------------------------------------------------------------------
    # Audio View: Input Devices, Real-Time VU Meter & VAD Tuning
    # -------------------------------------------------------------------------
    def _build_audio_panel(self):
        gtk = self.gtk

        scroller = gtk.ScrolledWindow()
        scroller.set_policy(gtk.PolicyType.NEVER, gtk.PolicyType.EXTERNAL)
        scroller.set_hexpand(True)
        scroller.set_vexpand(True)

        vbox = gtk.Box(orientation=gtk.Orientation.VERTICAL, spacing=14)
        vbox.add_css_class("audio-panel")
        vbox.set_hexpand(True)

        # 1. Header Title Row
        top_row = gtk.Box(orientation=gtk.Orientation.HORIZONTAL, spacing=8)
        head_text = gtk.Box(orientation=gtk.Orientation.VERTICAL, spacing=2, hexpand=True)
        title = gtk.Label(label="Audio", xalign=0)
        title.set_markup("<span font_weight='800' font_size='18pt' color='#ffffff'>Audio</span>")
        sub = gtk.Label(label="Configure microphones, live level monitoring, and voice activity detection.", xalign=0)
        sub.set_markup("<span font_size='9pt' alpha='55%'>Configure microphones, live level monitoring, and voice activity detection.</span>")
        head_text.append(title)
        head_text.append(sub)
        top_row.append(head_text)

        vbox.append(top_row)

        # 2. Card: Live VU Meter
        vbox.append(self._build_mic_meter_card())

        # 3. Card: Input Devices
        vbox.append(self._build_mic_devices_card())

        # 4. Card: VAD Sensitivity & Sliders
        vbox.append(self._build_vad_sensitivity_card())

        scroller.set_child(vbox)
        return scroller

    def _build_mic_meter_card(self):
        gtk = self.gtk
        card = gtk.Box(orientation=gtk.Orientation.VERTICAL, spacing=12)
        card.add_css_class("audio-card")

        hdr = gtk.Box(orientation=gtk.Orientation.HORIZONTAL, spacing=10)
        badge = self._make_badge("microphone", "audio-badge", 28)
        hdr.append(badge)

        t_col = gtk.Box(orientation=gtk.Orientation.VERTICAL, spacing=2, hexpand=True)
        t = gtk.Label(label="Live Microphone Input", xalign=0)
        t.set_markup("<span font_weight='700' font_size='11pt' color='#ffffff'>Live Microphone Input</span>")
        s = gtk.Label(label="Monitor real-time audio volume and test voice activity detection triggers.", xalign=0)
        s.set_markup("<span font_size='8.5pt' alpha='55%'>Monitor real-time audio volume and test voice activity detection triggers.</span>")
        t_col.append(t)
        t_col.append(s)
        hdr.append(t_col)

        test_btn = gtk.Button(label="▶ Test Microphone")
        test_btn.add_css_class("btn-test-mic")
        test_btn.set_valign(gtk.Align.CENTER)
        test_btn.set_cursor_from_name("pointer")
        test_btn.connect("clicked", self._toggle_mic_test)
        self._test_mic_btn = test_btn
        hdr.append(test_btn)
        card.append(hdr)

        # VU Meter Progress Bar
        meter_col = gtk.Box(orientation=gtk.Orientation.VERTICAL, spacing=6)
        vu = gtk.ProgressBar()
        vu.add_css_class("vu-meter-bar")
        vu.set_fraction(0.0)
        self._test_vu_meter = vu
        meter_col.append(vu)

        # dB Scale markings
        scale_box = gtk.Box(orientation=gtk.Orientation.HORIZONTAL)
        db_marks = ["-60 dB", "-45 dB", "-30 dB", "-20 dB", "-10 dB", "-3 dB", "0 dB"]
        for idx, mark in enumerate(db_marks):
            lbl = gtk.Label(label=mark)
            lbl.set_markup(f"<span font_size='7.5pt' alpha='40%'>{mark}</span>")
            if idx == 0:
                lbl.set_halign(gtk.Align.START)
                lbl.set_hexpand(True)
            elif idx == len(db_marks) - 1:
                lbl.set_halign(gtk.Align.END)
            else:
                lbl.set_halign(gtk.Align.CENTER)
                lbl.set_hexpand(True)
            scale_box.append(lbl)
        meter_col.append(scale_box)
        card.append(meter_col)

        # Status & Speech Detection readout
        status_row = gtk.Box(orientation=gtk.Orientation.HORIZONTAL, spacing=12)
        status_row.set_margin_top(4)

        db_lbl = gtk.Label(xalign=0, hexpand=True)
        db_lbl.set_markup("<span font_size='9pt' alpha='50%'>Idle • Click Test Microphone to begin monitoring</span>")
        self._test_db_label = db_lbl
        status_row.append(db_lbl)

        thresh_note = gtk.Label(xalign=1)
        thresh_note.set_markup(f"<span font_size='8pt' alpha='50%'>VAD Threshold: {self._vad_rms_threshold:.4f}</span>")
        status_row.append(thresh_note)

        speech_badge = gtk.Label()
        speech_badge.add_css_class("badge-speech-idle")
        speech_badge.set_markup("<span font_size='8.5pt' alpha='60%'>● Idle</span>")
        self._test_speech_badge = speech_badge
        status_row.append(speech_badge)

        card.append(status_row)
        return card

    def _build_mic_devices_card(self):
        gtk = self.gtk
        card = gtk.Box(orientation=gtk.Orientation.VERTICAL, spacing=12)
        card.add_css_class("audio-card")

        hdr = gtk.Box(orientation=gtk.Orientation.HORIZONTAL, spacing=10)
        badge = self._make_badge("wave", "audio-badge", 28)
        hdr.append(badge)

        t_col = gtk.Box(orientation=gtk.Orientation.VERTICAL, spacing=2, hexpand=True)
        t = gtk.Label(label="Input Device", xalign=0)
        t.set_markup("<span font_weight='700' font_size='11pt' color='#ffffff'>Input Device</span>")
        s = gtk.Label(label="Select the audio capture source used by Wispo.", xalign=0)
        s.set_markup("<span font_size='8.5pt' alpha='55%'>Select the audio capture source used by Wispo.</span>")
        t_col.append(t)
        t_col.append(s)
        hdr.append(t_col)

        rescan_btn = gtk.Button()
        rescan_btn.add_css_class("btn-learn")
        rescan_box = gtk.Box(orientation=gtk.Orientation.HORIZONTAL, spacing=6)
        rescan_box.append(self._make_svg("refresh"))
        rescan_lbl = gtk.Label(label="Rescan")
        rescan_box.append(rescan_lbl)
        rescan_btn.set_child(rescan_box)
        rescan_btn.set_cursor_from_name("pointer")
        rescan_btn.connect("clicked", lambda *_: self._render_device_rows())
        hdr.append(rescan_btn)
        card.append(hdr)

        self._devices_list_box = gtk.Box(orientation=gtk.Orientation.VERTICAL, spacing=4)
        self._render_device_rows()
        card.append(self._devices_list_box)

        return card

    def _scan_audio_devices(self) -> list[dict]:
        devices = [
            {
                "id": "auto",
                "name": "Auto-Detect Microphone",
                "sub": "Automatically selects primary built-in digital mic or active session input",
                "tag": "Recommended",
            },
            {
                "id": "default",
                "name": "PipeWire Session Default",
                "sub": "Follows desktop ALSA / PipeWire session default routing",
                "tag": "",
            },
        ]
        try:
            completed = subprocess.run(
                ["wpctl", "status"],
                check=False,
                capture_output=True,
                text=True,
                timeout=2.0,
            )
            sources = parse_wpctl_sources(completed.stdout)
            for sid, desc in sources:
                node = _node_name(sid)
                if node:
                    is_digital = "digital" in desc.lower()
                    tag = "Internal Mic" if is_digital else "Stereo Mic"
                    devices.append({
                        "id": node,
                        "name": desc,
                        "sub": f"PipeWire node: {node}",
                        "tag": tag,
                    })
        except Exception:
            pass
        return devices

    def _render_device_rows(self) -> None:
        gtk = self.gtk
        if not self._devices_list_box:
            return

        child = self._devices_list_box.get_first_child()
        while child:
            next_child = child.get_next_sibling()
            self._devices_list_box.remove(child)
            child = next_child

        self._mic_device_rows.clear()
        self._mic_radio_indicators.clear()

        devices = self._scan_audio_devices()
        for dev in devices:
            dev_id = dev["id"]
            is_selected = (self._selected_audio_device == dev_id)

            row = gtk.Box(orientation=gtk.Orientation.HORIZONTAL, spacing=12)
            row.add_css_class("mic-device-row")
            if is_selected:
                row.add_css_class("mic-device-selected")

            radio_img = self._make_svg("radio_on" if is_selected else "radio_off")
            radio_img.set_valign(gtk.Align.CENTER)
            row.append(radio_img)
            self._mic_radio_indicators[dev_id] = radio_img

            info_col = gtk.Box(orientation=gtk.Orientation.VERTICAL, spacing=2, hexpand=True)
            name_lbl = gtk.Label(xalign=0)
            name_lbl.set_markup(f"<span font_weight='600' font_size='9.5pt' color='#ffffff'>{pango_esc(dev['name'])}</span>")
            sub_lbl = gtk.Label(xalign=0)
            sub_lbl.set_markup(f"<span font_size='8pt' alpha='50%'>{pango_esc(dev['sub'])}</span>")
            info_col.append(name_lbl)
            info_col.append(sub_lbl)
            row.append(info_col)

            if dev["tag"]:
                tag_lbl = gtk.Label()
                tag_lbl.add_css_class("pill-recommended" if dev["tag"] == "Recommended" else "pill-size")
                tag_lbl.set_markup(f"<span font_weight='700' font_size='8pt'>{pango_esc(dev['tag'])}</span>")
                tag_lbl.set_valign(gtk.Align.CENTER)
                row.append(tag_lbl)

            click = gtk.GestureClick()
            click.connect("released", lambda _g, _n, _x, _y, d=dev_id: self._select_audio_device(d))
            row.add_controller(click)
            row.set_cursor_from_name("pointer")

            self._mic_device_rows[dev_id] = row
            self._devices_list_box.append(row)

    def _select_audio_device(self, dev_id: str) -> None:
        from gi.repository import Gdk, GLib

        self._selected_audio_device = dev_id
        for d_id, row in self._mic_device_rows.items():
            is_sel = (d_id == dev_id)
            if is_sel:
                row.add_css_class("mic-device-selected")
            else:
                row.remove_css_class("mic-device-selected")
            if d_id in self._mic_radio_indicators:
                svg_data = SVG_ICONS["radio_on" if is_sel else "radio_off"].encode("utf-8")
                bytes_data = GLib.Bytes.new(svg_data)
                tex = Gdk.Texture.new_from_bytes(bytes_data)
                self._mic_radio_indicators[d_id].set_from_paintable(tex)

        if self._is_testing_mic:
            self._stop_mic_test()
            self._start_mic_test()

        target = resolve_mic(dev_id)
        self._status_label.set_markup(f"<span font_size='8.5pt' color='#a5b4fc'>Selected audio input: {pango_esc(target.description)}. Click Save to apply.</span>")

    def _build_vad_sensitivity_card(self):
        gtk = self.gtk
        card = gtk.Box(orientation=gtk.Orientation.VERTICAL, spacing=14)
        card.add_css_class("audio-card")

        hdr = gtk.Box(orientation=gtk.Orientation.HORIZONTAL, spacing=10)
        badge = self._make_badge("sliders", "audio-badge", 28)
        hdr.append(badge)

        t_col = gtk.Box(orientation=gtk.Orientation.VERTICAL, spacing=2, hexpand=True)
        t = gtk.Label(label="Voice Activity Detection (VAD)", xalign=0)
        t.set_markup("<span font_weight='700' font_size='11pt' color='#ffffff'>Voice Activity Detection (VAD)</span>")
        s = gtk.Label(label="Tune speech energy thresholds and trailing silence parameters.", xalign=0)
        s.set_markup("<span font_size='8.5pt' alpha='55%'>Tune speech energy thresholds and trailing silence parameters.</span>")
        t_col.append(t)
        t_col.append(s)
        hdr.append(t_col)

        presets_box = gtk.Box(orientation=gtk.Orientation.HORIZONTAL, spacing=6)
        presets_box.set_valign(gtk.Align.CENTER)
        preset_configs = [
            ("Quiet", 0.008, 600, 60.0),
            ("Balanced", 0.015, 700, 80.0),
            ("Noisy", 0.035, 850, 120.0),
        ]
        for pname, prms, phang, phigh in preset_configs:
            btn = gtk.Button(label=pname)
            btn.add_css_class("preset-pill")
            if abs(self._vad_rms_threshold - prms) < 0.003:
                btn.add_css_class("preset-pill-active")
            btn.set_cursor_from_name("pointer")
            btn.connect("clicked", lambda *_, n=pname, r=prms, h=phang, f=phigh: self._apply_vad_preset(r, h, f, n))
            self._preset_pills[pname] = btn
            presets_box.append(btn)
        hdr.append(presets_box)
        card.append(hdr)

        # 1. Slider: RMS Energy Threshold
        rms_box = gtk.Box(orientation=gtk.Orientation.VERTICAL, spacing=4)
        rms_top = gtk.Box(orientation=gtk.Orientation.HORIZONTAL)
        rms_lbl = gtk.Label(xalign=0, hexpand=True)
        rms_lbl.set_markup("<span font_weight='600' font_size='9pt' color='#ffffff'>Speech RMS Threshold</span>")
        self._rms_val_lbl = gtk.Label(xalign=1)
        self._rms_val_lbl.set_markup(f"<span font_weight='700' font_size='9.5pt' color='#a5b4fc'>{self._vad_rms_threshold:.4f}</span>")
        rms_top.append(rms_lbl)
        rms_top.append(self._rms_val_lbl)
        rms_box.append(rms_top)

        rms_scale = gtk.Scale.new_with_range(gtk.Orientation.HORIZONTAL, 0.002, 0.080, 0.001)
        rms_scale.add_css_class("glass-scale")
        rms_scale.set_value(self._vad_rms_threshold)
        rms_scale.set_draw_value(False)
        rms_scale.set_hexpand(True)
        rms_scale.connect("value-changed", self._on_rms_scale_changed)
        self._rms_scale = rms_scale
        rms_box.append(rms_scale)

        rms_sub = gtk.Label(xalign=0)
        rms_sub.set_markup("<span font_size='8pt' alpha='45%'>Absolute energy floor. Speech quieter than this is ignored. Lower is more sensitive.</span>")
        rms_box.append(rms_sub)
        card.append(rms_box)

        # 2. Slider: Hangover Duration (ms)
        hang_box = gtk.Box(orientation=gtk.Orientation.VERTICAL, spacing=4)
        hang_top = gtk.Box(orientation=gtk.Orientation.HORIZONTAL)
        hang_lbl = gtk.Label(xalign=0, hexpand=True)
        hang_lbl.set_markup("<span font_weight='600' font_size='9pt' color='#ffffff'>Trailing Silence (Hangover)</span>")
        self._hangover_val_lbl = gtk.Label(xalign=1)
        self._hangover_val_lbl.set_markup(f"<span font_weight='700' font_size='9.5pt' color='#a5b4fc'>{self._vad_hangover_ms} ms</span>")
        hang_top.append(hang_lbl)
        hang_top.append(self._hangover_val_lbl)
        hang_box.append(hang_top)

        hang_scale = gtk.Scale.new_with_range(gtk.Orientation.HORIZONTAL, 200, 1500, 50)
        hang_scale.add_css_class("glass-scale")
        hang_scale.set_value(self._vad_hangover_ms)
        hang_scale.set_draw_value(False)
        hang_scale.set_hexpand(True)
        hang_scale.connect("value-changed", self._on_hangover_scale_changed)
        self._hangover_scale = hang_scale
        hang_box.append(hang_scale)

        hang_sub = gtk.Label(xalign=0)
        hang_sub.set_markup("<span font_size='8pt' alpha='45%'>How long to keep recording after voice drops to capture complete sentence endings.</span>")
        hang_box.append(hang_sub)
        card.append(hang_box)

        # 3. Slider: High-Pass Filter (Hz)
        high_box = gtk.Box(orientation=gtk.Orientation.VERTICAL, spacing=4)
        high_top = gtk.Box(orientation=gtk.Orientation.HORIZONTAL)
        high_lbl = gtk.Label(xalign=0, hexpand=True)
        high_lbl.set_markup("<span font_weight='600' font_size='9pt' color='#ffffff'>High-Pass Filter Cutoff</span>")
        self._highpass_val_lbl = gtk.Label(xalign=1)
        self._highpass_val_lbl.set_markup(f"<span font_weight='700' font_size='9.5pt' color='#a5b4fc'>{int(self._vad_highpass_hz)} Hz</span>")
        high_top.append(high_lbl)
        high_top.append(self._highpass_val_lbl)
        high_box.append(high_top)

        high_scale = gtk.Scale.new_with_range(gtk.Orientation.HORIZONTAL, 20, 250, 10)
        high_scale.add_css_class("glass-scale")
        high_scale.set_value(self._vad_highpass_hz)
        high_scale.set_draw_value(False)
        high_scale.set_hexpand(True)
        high_scale.connect("value-changed", self._on_highpass_scale_changed)
        self._highpass_scale = high_scale
        high_box.append(high_scale)

        high_sub = gtk.Label(xalign=0)
        high_sub.set_markup("<span font_size='8pt' alpha='45%'>Attenuates table thumps, fan vibrations, and low-frequency HVAC rumble before VAD processing.</span>")
        high_box.append(high_sub)
        card.append(high_box)

        return card


    def _on_rms_scale_changed(self, scale) -> None:
        val = scale.get_value()
        self._vad_rms_threshold = val
        if self._rms_val_lbl:
            self._rms_val_lbl.set_markup(f"<span font_weight='700' font_size='9.5pt' color='#a5b4fc'>{val:.4f}</span>")
        self._status_label.set_markup(f"<span font_size='8.5pt' color='#a5b4fc'>VAD RMS Threshold set to {val:.4f}. Click Save to apply.</span>")

    def _on_hangover_scale_changed(self, scale) -> None:
        val = int(scale.get_value())
        self._vad_hangover_ms = val
        if self._hangover_val_lbl:
            self._hangover_val_lbl.set_markup(f"<span font_weight='700' font_size='9.5pt' color='#a5b4fc'>{val} ms</span>")
        self._status_label.set_markup(f"<span font_size='8.5pt' color='#a5b4fc'>VAD Hangover set to {val} ms. Click Save to apply.</span>")

    def _on_highpass_scale_changed(self, scale) -> None:
        val = scale.get_value()
        self._vad_highpass_hz = val
        if self._highpass_val_lbl:
            self._highpass_val_lbl.set_markup(f"<span font_weight='700' font_size='9.5pt' color='#a5b4fc'>{int(val)} Hz</span>")
        self._status_label.set_markup(f"<span font_size='8.5pt' color='#a5b4fc'>High-Pass Filter set to {int(val)} Hz. Click Save to apply.</span>")

    def _apply_vad_preset(self, rms: float, hangover: int, highpass: float, name: str) -> None:
        self._vad_rms_threshold = rms
        self._vad_hangover_ms = hangover
        self._vad_highpass_hz = highpass
        if self._rms_scale:
            self._rms_scale.set_value(rms)
        if self._hangover_scale:
            self._hangover_scale.set_value(hangover)
        if self._highpass_scale:
            self._highpass_scale.set_value(highpass)
        for pname, btn in self._preset_pills.items():
            if pname == name:
                btn.add_css_class("preset-pill-active")
            else:
                btn.remove_css_class("preset-pill-active")
        self._status_label.set_markup(f"<span font_size='8.5pt' color='#34d399'>Applied {name} preset. Click Save to apply.</span>")

    def _toggle_mic_test(self, *_args) -> None:
        if self._is_testing_mic:
            self._stop_mic_test()
        else:
            self._start_mic_test()

    def _start_mic_test(self) -> None:
        from gi.repository import GLib

        if self._is_testing_mic:
            return
        try:
            target = resolve_mic(self._selected_audio_device)
            self._test_capture = AudioCapture(
                device=target.alsa_device,
                sample_rate=16000,
                frame_ms=30,
                pipewire_target=target.pipewire_target,
                description=target.description,
            )
            self._test_capture.start()
            self._is_testing_mic = True
            self._meter_level = 0.0
            self._meter_target = 0.0
            if self._test_mic_btn:
                self._test_mic_btn.remove_css_class("btn-test-mic")
                self._test_mic_btn.add_css_class("btn-test-mic-active")
                self._test_mic_btn.set_label("⏹ Stop Test")
            self._status_label.set_markup(f"<span font_size='8.5pt' color='#a5b4fc'>Testing microphone: {pango_esc(target.description)}</span>")
            self._test_timer_id = GLib.timeout_add(40, self._on_mic_test_tick)
        except Exception as exc:
            self._is_testing_mic = False
            self._test_capture = None
            self._status_label.set_markup(f"<span font_size='8.5pt' color='#f87171'>Microphone test failed: {pango_esc(str(exc))}</span>")

    def _stop_mic_test(self) -> None:
        from gi.repository import GLib

        if self._test_timer_id is not None:
            GLib.source_remove(self._test_timer_id)
            self._test_timer_id = None
        if self._test_capture is not None:
            try:
                self._test_capture.stop()
            except Exception:
                pass
            self._test_capture = None
        self._is_testing_mic = False
        self._meter_level = 0.0
        self._meter_target = 0.0
        if self._test_mic_btn:
            self._test_mic_btn.remove_css_class("btn-test-mic-active")
            self._test_mic_btn.add_css_class("btn-test-mic")
            self._test_mic_btn.set_label("▶ Test Microphone")
        if self._test_vu_meter:
            self._test_vu_meter.set_fraction(0.0)
        if self._test_db_label:
            self._test_db_label.set_markup("<span font_size='9pt' alpha='50%'>Idle • Click Test to monitor</span>")
        if self._test_speech_badge:
            self._test_speech_badge.remove_css_class("badge-speech-detected")
            self._test_speech_badge.add_css_class("badge-speech-idle")
            self._test_speech_badge.set_markup("<span font_size='9pt' alpha='60%'>● Idle</span>")

    def _on_mic_test_tick(self) -> bool:
        if not self._is_testing_mic or self._test_capture is None:
            return False
        frames = self._test_capture.drain()
        if frames:
            peak_rms = max(float(np.sqrt(np.mean(f ** 2))) for f in frames)
            self._meter_target = peak_rms
        else:
            self._meter_target *= 0.88

        if self._meter_target > self._meter_level:
            self._meter_level = self._meter_level * 0.3 + self._meter_target * 0.7
        else:
            self._meter_level = self._meter_level * 0.82 + self._meter_target * 0.18

        db = 20 * math.log10(max(self._meter_level, 1e-4))
        fraction = max(0.0, min(1.0, (db + 60.0) / 60.0))

        if self._test_vu_meter:
            self._test_vu_meter.set_fraction(fraction)

        is_speech = self._meter_level >= self._vad_rms_threshold
        if self._test_db_label:
            self._test_db_label.set_markup(
                f"<span font_weight='700' font_size='10pt' color='#ffffff'>{db:+.1f} dB</span> "
                f"<span font_size='8.5pt' alpha='55%'>RMS: {self._meter_level:.4f}</span>"
            )

        if self._test_speech_badge:
            if is_speech:
                self._test_speech_badge.remove_css_class("badge-speech-idle")
                self._test_speech_badge.add_css_class("badge-speech-detected")
                self._test_speech_badge.set_markup("<span font_weight='700' font_size='9pt' color='#34d399'>● Speech Detected</span>")
            else:
                self._test_speech_badge.remove_css_class("badge-speech-detected")
                self._test_speech_badge.add_css_class("badge-speech-idle")
                self._test_speech_badge.set_markup("<span font_size='9pt' alpha='60%'>● Background Noise</span>")

        return True

    # -------------------------------------------------------------------------
    # Push-to-Talk Studio: Key Mapping, Hardware Presets & Activation Behavior
    # -------------------------------------------------------------------------
    def _build_push_to_talk_panel(self):
        gtk = self.gtk

        scroller = gtk.ScrolledWindow()
        scroller.set_policy(gtk.PolicyType.NEVER, gtk.PolicyType.EXTERNAL)
        scroller.set_hexpand(True)
        scroller.set_vexpand(True)

        vbox = gtk.Box(orientation=gtk.Orientation.VERTICAL, spacing=14)
        vbox.add_css_class("audio-panel")
        vbox.set_hexpand(True)

        # 1. Header Title Row
        top_row = gtk.Box(orientation=gtk.Orientation.HORIZONTAL, spacing=8)
        head_text = gtk.Box(orientation=gtk.Orientation.VERTICAL, spacing=2, hexpand=True)
        title = gtk.Label(label="Push-to-talk", xalign=0)
        title.set_markup("<span font_weight='800' font_size='18pt' color='#ffffff'>Push-to-talk</span>")
        sub = gtk.Label(label="Configure global keyboard shortcuts and dictation trigger activation.", xalign=0)
        sub.set_markup("<span font_size='9pt' alpha='55%'>Configure global keyboard shortcuts and dictation trigger activation.</span>")
        head_text.append(title)
        head_text.append(sub)
        top_row.append(head_text)

        vbox.append(top_row)

        # 2. Card 1: Shortcut Studio (Keycaps + Live Recorder)
        vbox.append(self._build_ptt_studio_card())

        # 3. Card 2: Recommended Hardware Presets
        vbox.append(self._build_ptt_presets_card())

        scroller.set_child(vbox)
        return scroller

    def _build_ptt_studio_card(self):
        gtk = self.gtk
        card = gtk.Box(orientation=gtk.Orientation.VERTICAL, spacing=14)
        card.add_css_class("audio-card")

        # Header
        hdr = gtk.Box(orientation=gtk.Orientation.HORIZONTAL, spacing=10)
        badge = self._make_badge("keyboard", "audio-badge", 28)
        hdr.append(badge)

        t_col = gtk.Box(orientation=gtk.Orientation.VERTICAL, spacing=2, hexpand=True)
        t = gtk.Label(label="Shortcut Studio", xalign=0)
        t.set_markup("<span font_weight='700' font_size='11pt' color='#ffffff'>Shortcut Studio</span>")
        s = gtk.Label(label="Hold this key combination anywhere in your operating system to speak.", xalign=0)
        s.set_markup("<span font_size='8.5pt' alpha='55%'>Hold this key combination anywhere in your operating system to speak.</span>")
        t_col.append(t)
        t_col.append(s)
        hdr.append(t_col)

        # Action Buttons on right of header
        btn_box = gtk.Box(orientation=gtk.Orientation.HORIZONTAL, spacing=8)
        btn_box.set_valign(gtk.Align.CENTER)

        reset_btn = gtk.Button(label="↺ Reset")
        reset_btn.add_css_class("btn-learn")
        reset_btn.set_cursor_from_name("pointer")
        reset_btn.connect("clicked", lambda *_: self._reset_ptt_default())
        btn_box.append(reset_btn)

        record_btn = gtk.Button(label="🔴 Record Shortcut")
        record_btn.add_css_class("btn-record-chord")
        record_btn.set_cursor_from_name("pointer")
        record_btn.connect("clicked", self._start_capture)
        self._ptt_record_btn = record_btn
        btn_box.append(record_btn)

        hdr.append(btn_box)
        card.append(hdr)

        # Center Keycap Display Row
        display_frame = gtk.Box(orientation=gtk.Orientation.HORIZONTAL, spacing=12)
        display_frame.set_margin_top(4)
        display_frame.set_margin_bottom(4)

        self._ptt_hotkey_box = gtk.Box(orientation=gtk.Orientation.HORIZONTAL, spacing=8)
        self._ptt_hotkey_box.set_hexpand(True)
        display_frame.append(self._ptt_hotkey_box)
        card.append(display_frame)

        # Subtitle Status Note
        status_lbl = gtk.Label(xalign=0)
        status_lbl.set_markup("<span font_size='8.5pt' alpha='50%'>Hold down the shortcut while speaking in any application. Click Record to rebind.</span>")
        self._ptt_status_lbl = status_lbl
        card.append(status_lbl)

        # Render current keycaps
        self._render_hotkey_chips()
        return card

    def _reset_ptt_default(self) -> None:
        self._hotkey_keys = ("copilot",)
        self._render_hotkey_chips()
        self._status_label.set_markup("<span font_size='8.5pt' color='#34d399'>● Reset shortcut to Copilot key. Click Save to apply.</span>")

    def _build_ptt_presets_card(self):
        gtk = self.gtk
        card = gtk.Box(orientation=gtk.Orientation.VERTICAL, spacing=12)
        card.add_css_class("audio-card")

        hdr = gtk.Box(orientation=gtk.Orientation.HORIZONTAL, spacing=10)
        badge = self._make_badge("zap", "audio-badge", 28)
        hdr.append(badge)

        t_col = gtk.Box(orientation=gtk.Orientation.VERTICAL, spacing=2, hexpand=True)
        t = gtk.Label(label="Recommended Hardware Presets", xalign=0)
        t.set_markup("<span font_weight='700' font_size='11pt' color='#ffffff'>Recommended Hardware Presets</span>")
        s = gtk.Label(label="Quickly bind popular single-key and multi-key chord combinations.", xalign=0)
        s.set_markup("<span font_size='8.5pt' alpha='55%'>Quickly bind popular single-key and multi-key chord combinations.</span>")
        t_col.append(t)
        t_col.append(s)
        hdr.append(t_col)
        card.append(hdr)

        preset_items = [
            (
                "copilot",
                ("copilot",),
                "Copilot Key (Single Key)",
                "Dedicated hardware AI key on modern laptop keyboards. Effortless thumb hold.",
                "Hardware AI Key",
            ),
            (
                "rightctrl",
                ("rightctrl",),
                "Right Ctrl (Single Key)",
                "Single modifier key located at the bottom-right corner. Zero chord gymnastics.",
                "Single Key",
            ),
            (
                "ctrl_super_space",
                ("ctrl", "super", "space"),
                "Ctrl + Super + Space",
                "Familiar macOS Spotlight and Raycast global search shortcut.",
                "Raycast Style",
            ),
            (
                "alt_space",
                ("alt", "space"),
                "Alt + Space",
                "Fast, ergonomic two-finger thumb chord with high comfort.",
                "Thumb Chord",
            ),
            (
                "super_h",
                ("super", "h"),
                "Super + H",
                "Standard Windows and GNOME system dictation key combination.",
                "Windows Style",
            ),
        ]

        self._ptt_preset_rows.clear()
        self._ptt_preset_radio_imgs.clear()

        current_keys = tuple(k.lower() for k in self._hotkey_keys)

        for p_id, p_keys, p_name, p_sub, p_tag in preset_items:
            is_match = (current_keys == p_keys)

            row = gtk.Box(orientation=gtk.Orientation.HORIZONTAL, spacing=12)
            row.add_css_class("preset-card-row")
            if is_match:
                row.add_css_class("preset-card-selected")
            row._keys_tuple = p_keys

            radio_img = self._make_svg("radio_on" if is_match else "radio_off")
            radio_img.set_valign(gtk.Align.CENTER)
            row.append(radio_img)
            self._ptt_preset_radio_imgs[p_id] = radio_img

            info_col = gtk.Box(orientation=gtk.Orientation.VERTICAL, spacing=2, hexpand=True)
            name_lbl = gtk.Label(xalign=0)
            name_lbl.set_markup(f"<span font_weight='600' font_size='9.5pt' color='#ffffff'>{pango_esc(p_name)}</span>")
            sub_lbl = gtk.Label(xalign=0)
            sub_lbl.set_markup(f"<span font_size='8pt' alpha='50%'>{pango_esc(p_sub)}</span>")
            info_col.append(name_lbl)
            info_col.append(sub_lbl)
            row.append(info_col)

            tag_lbl = gtk.Label()
            tag_lbl.add_css_class("pill-recommended" if p_tag in ("Hardware AI Key", "Single Key") else "pill-size")
            tag_lbl.set_markup(f"<span font_weight='700' font_size='8pt'>{pango_esc(p_tag)}</span>")
            tag_lbl.set_valign(gtk.Align.CENTER)
            row.append(tag_lbl)

            click = gtk.GestureClick()
            click.connect("released", lambda _g, _n, _x, _y, k=p_keys, n=p_name: self._select_ptt_preset(k, n))
            row.add_controller(click)
            row.set_cursor_from_name("pointer")

            self._ptt_preset_rows[p_id] = row
            card.append(row)

        return card

    def _select_ptt_preset(self, keys: tuple[str, ...], name: str) -> None:
        self._hotkey_keys = keys
        self._render_hotkey_chips()
        self._status_label.set_markup(f"<span font_size='8.5pt' color='#a5b4fc'>Selected preset: {pango_esc(name)}. Click Save to apply.</span>")

    def _sync_preset_highlights(self) -> None:
        from gi.repository import Gdk, GLib

        current_tuple = tuple(k.lower() for k in self._hotkey_keys)
        for p_id, row in self._ptt_preset_rows.items():
            preset_tuple = row._keys_tuple
            is_match = (current_tuple == preset_tuple)
            if is_match:
                row.add_css_class("preset-card-selected")
            else:
                row.remove_css_class("preset-card-selected")
            if p_id in self._ptt_preset_radio_imgs:
                svg_data = SVG_ICONS["radio_on" if is_match else "radio_off"].encode("utf-8")
                bytes_data = GLib.Bytes.new(svg_data)
                tex = Gdk.Texture.new_from_bytes(bytes_data)
                self._ptt_preset_radio_imgs[p_id].set_from_paintable(tex)


    # -------------------------------------------------------------------------
    # Dedicated Section: Full Dictionary View & Vocabulary Rules
    # -------------------------------------------------------------------------
    def _build_dictionary_panel(self):
        gtk = self.gtk
        scroller = gtk.ScrolledWindow()
        scroller.set_policy(gtk.PolicyType.NEVER, gtk.PolicyType.EXTERNAL)
        scroller.set_hexpand(True)
        scroller.set_vexpand(True)

        vbox = gtk.Box(orientation=gtk.Orientation.VERTICAL, spacing=16)
        vbox.add_css_class("dict-panel")
        vbox.set_hexpand(True)

        # Header Title Row
        top_row = gtk.Box(orientation=gtk.Orientation.HORIZONTAL, spacing=12)
        head_text = gtk.Box(orientation=gtk.Orientation.VERTICAL, spacing=2, hexpand=True)
        title = gtk.Label(label="Dictionary", xalign=0)
        title.set_markup("<span font_weight='800' font_size='18pt' color='#ffffff'>Dictionary</span>")
        sub = gtk.Label(label="Custom vocabulary replacements, phonetic word corrections, and domain jargon.", xalign=0)
        sub.set_markup("<span font_size='9pt' alpha='55%'>Custom vocabulary replacements, phonetic word corrections, and domain jargon.</span>")
        head_text.append(title)
        head_text.append(sub)
        top_row.append(head_text)

        # Header status badge
        self._dict_badge_lbl = gtk.Label(xalign=0.5)
        self._dict_badge_lbl.add_css_class("dict-badge")
        self._dict_badge_lbl.set_valign(gtk.Align.START)
        top_row.append(self._dict_badge_lbl)
        vbox.append(top_row)

        # 1. Quick Add Rule Composer Card
        vbox.append(self._build_dict_add_card())

        # 2. Active Vocabulary Rules Table Card
        vbox.append(self._build_dict_rules_card())

        # 3. Interactive Live Playground Card
        vbox.append(self._build_dict_playground_card())

        # 4. Recommended Starter Packs Card
        vbox.append(self._build_dict_packs_card())

        # Initial render of rules and badge
        self._refresh_dict_rules()

        scroller.set_child(vbox)
        return scroller

    def _build_dict_add_card(self):
        gtk = self.gtk
        card = gtk.Box(orientation=gtk.Orientation.VERTICAL, spacing=12)
        card.add_css_class("dict-card")

        # Head row
        head = gtk.Box(orientation=gtk.Orientation.HORIZONTAL, spacing=8)
        head.append(self._make_svg("pencil"))
        t_box = gtk.Box(orientation=gtk.Orientation.VERTICAL, spacing=2, hexpand=True)
        t = gtk.Label(xalign=0)
        t.set_markup("<span font_weight='700' font_size='11pt' color='#ffffff'>Add Vocabulary Replacement</span>")
        s = gtk.Label(xalign=0)
        s.set_markup("<span font_size='8.5pt' alpha='55%'>Phonetically misheard spoken phrases are automatically swapped before injection.</span>")
        t_box.append(t)
        t_box.append(s)
        head.append(t_box)
        card.append(head)

        # Input Row
        form_row = gtk.Box(orientation=gtk.Orientation.HORIZONTAL, spacing=10)
        form_row.set_hexpand(True)

        self._dict_new_phrase_entry = gtk.Entry()
        self._dict_new_phrase_entry.set_placeholder_text("Spoken phrase (e.g. pie torch)")
        self._dict_new_phrase_entry.add_css_class("glass-entry")
        self._dict_new_phrase_entry.set_hexpand(True)
        self._dict_new_phrase_entry.connect("activate", self._on_add_dict_rule)
        form_row.append(self._dict_new_phrase_entry)

        arrow_lbl = gtk.Label(label="➔")
        arrow_lbl.set_markup("<span font_size='11pt' alpha='45%'>➔</span>")
        form_row.append(arrow_lbl)

        self._dict_new_repl_entry = gtk.Entry()
        self._dict_new_repl_entry.set_placeholder_text("Formatted replacement (e.g. PyTorch)")
        self._dict_new_repl_entry.add_css_class("glass-entry")
        self._dict_new_repl_entry.set_hexpand(True)
        self._dict_new_repl_entry.connect("activate", self._on_add_dict_rule)
        form_row.append(self._dict_new_repl_entry)

        add_btn = gtk.Button(label="+ Add Rule")
        add_btn.add_css_class("btn-save-purple")
        add_btn.connect("clicked", self._on_add_dict_rule)
        form_row.append(add_btn)
        card.append(form_row)

        self._dict_add_feedback = gtk.Label(xalign=0)
        self._dict_add_feedback.set_markup("<span font_size='8pt' alpha='50%'>Tip: Matching is case-insensitive and respects surrounding punctuation.</span>")
        card.append(self._dict_add_feedback)

        return card

    def _on_add_dict_rule(self, *_args):
        phrase = self._dict_new_phrase_entry.get_text().strip() if self._dict_new_phrase_entry else ""
        repl = self._dict_new_repl_entry.get_text().strip() if self._dict_new_repl_entry else ""
        if not phrase:
            if self._dict_add_feedback:
                self._dict_add_feedback.set_markup("<span font_size='8pt' color='#f87171'>Please enter a spoken phrase to match.</span>")
            return

        if not repl:
            repl = phrase

        # Check if already present
        existing_idx = next((i for i, e in enumerate(self._dictionary_entries_list) if e.phrase.lower() == phrase.lower()), None)
        if existing_idx is not None:
            self._dictionary_entries_list[existing_idx] = DictionaryEntry(phrase=phrase, replacement=repl, enabled=True)
            msg = f"Updated rule: '{phrase}' ➔ '{repl}'"
        else:
            self._dictionary_entries_list.append(DictionaryEntry(phrase=phrase, replacement=repl, enabled=True))
            msg = f"Added rule: '{phrase}' ➔ '{repl}'"

        if self._dict_new_phrase_entry:
            self._dict_new_phrase_entry.set_text("")
            self._dict_new_phrase_entry.grab_focus()
        if self._dict_new_repl_entry:
            self._dict_new_repl_entry.set_text("")

        if self._dict_add_feedback:
            self._dict_add_feedback.set_markup(f"<span font_size='8pt' color='#34d399'>● {pango_esc(msg)}</span>")

        self._refresh_dict_rules()
        self._update_dict_playground()
        self._render_dict_preview()
        self._status_label.set_markup(f"<span font_size='8.5pt' color='#34d399'>● {pango_esc(msg)}. Click Save to apply.</span>")

    def _build_dict_rules_card(self):
        gtk = self.gtk
        card = gtk.Box(orientation=gtk.Orientation.VERTICAL, spacing=12)
        card.add_css_class("dict-card")

        # Top row with title and search bar
        head = gtk.Box(orientation=gtk.Orientation.HORIZONTAL, spacing=10)
        head.append(self._make_svg("book"))

        t_box = gtk.Box(orientation=gtk.Orientation.VERTICAL, spacing=2, hexpand=True)
        t = gtk.Label(xalign=0)
        t.set_markup("<span font_weight='700' font_size='11pt' color='#ffffff'>Active Vocabulary Rules</span>")
        self._dict_count_lbl = gtk.Label(xalign=0)
        t_box.append(t)
        t_box.append(self._dict_count_lbl)
        head.append(t_box)

        # Search / filter field
        search_box = gtk.Box(orientation=gtk.Orientation.HORIZONTAL, spacing=6)
        search_box.set_valign(gtk.Align.CENTER)
        search_in = gtk.Entry()
        search_in.set_placeholder_text("Search rules...")
        search_in.add_css_class("glass-entry")
        search_in.connect("changed", self._on_dict_search_changed)
        search_box.append(search_in)
        head.append(search_box)
        card.append(head)

        # Dynamic rules container
        self._dict_rules_vbox = gtk.Box(orientation=gtk.Orientation.VERTICAL, spacing=6)
        card.append(self._dict_rules_vbox)

        return card

    def _on_dict_search_changed(self, entry):
        self._dict_search_query = entry.get_text().strip().lower()
        self._refresh_dict_rules()

    def _refresh_dict_rules(self):
        gtk = self.gtk
        if not self._dict_rules_vbox:
            return

        # Clear existing children
        while True:
            c = self._dict_rules_vbox.get_first_child()
            if c is None:
                break
            self._dict_rules_vbox.remove(c)

        total_rules = len(self._dictionary_entries_list)
        active_count = sum(1 for e in self._dictionary_entries_list if e.enabled)

        # Update badge
        if self._dict_badge_lbl:
            self._dict_badge_lbl.set_markup(f"<span font_size='8.5pt' font_weight='700'>● {active_count} Active Rules</span>")

        # Filter entries
        q = self._dict_search_query
        filtered = [
            (idx, e)
            for idx, e in enumerate(self._dictionary_entries_list)
            if not q or (q in e.phrase.lower() or q in e.replacement.lower())
        ]

        if self._dict_count_lbl:
            if q:
                self._dict_count_lbl.set_markup(f"<span font_size='8.5pt' alpha='55%'>Showing {len(filtered)} of {total_rules} rules matching \"{pango_esc(q)}\"</span>")
            else:
                self._dict_count_lbl.set_markup(f"<span font_size='8.5pt' alpha='55%'>{total_rules} configured rules ({active_count} enabled for speech)</span>")

        if not filtered:
            empty_box = gtk.Box(orientation=gtk.Orientation.VERTICAL, spacing=6)
            empty_box.set_margin_top(14)
            empty_box.set_margin_bottom(14)
            empty_box.set_halign(gtk.Align.CENTER)
            lbl1 = gtk.Label(xalign=0.5)
            lbl1.set_markup("<span font_weight='600' font_size='10pt' alpha='65%'>No vocabulary rules found</span>")
            lbl2 = gtk.Label(xalign=0.5)
            lbl2.set_markup("<span font_size='8.5pt' alpha='40%'>Add a custom term above or import a starter pack below.</span>")
            empty_box.append(lbl1)
            empty_box.append(lbl2)
            self._dict_rules_vbox.append(empty_box)
            return

        for orig_idx, entry in filtered:
            row = gtk.Box(orientation=gtk.Orientation.HORIZONTAL, spacing=10)
            row.add_css_class("dict-rule-row")
            row.set_hexpand(True)

            cb = gtk.CheckButton(active=entry.enabled)
            cb.set_valign(gtk.Align.CENTER)
            cb.connect("toggled", lambda _cb, idx=orig_idx: self._toggle_dict_entry(idx, _cb.get_active()))
            row.append(cb)

            sp_lbl = gtk.Label(xalign=0)
            sp_alpha = "90%" if entry.enabled else "40%"
            sp_lbl.set_markup(f"<span font_weight='600' font_size='9pt' color='#ffffff' alpha='{sp_alpha}'>{pango_esc(entry.phrase)}</span>")
            sp_box = gtk.Box()
            sp_box.add_css_class("dict-chip-spoken")
            sp_box.append(sp_lbl)
            row.append(sp_box)

            arrow = gtk.Label(label="➔")
            arr_alpha = "50%" if entry.enabled else "25%"
            arrow.set_markup(f"<span font_size='9pt' alpha='{arr_alpha}'>➔</span>")
            row.append(arrow)

            rp_lbl = gtk.Label(xalign=0)
            rp_color = "#a5b4fc" if entry.enabled else "#94a3b8"
            rp_alpha = "100%" if entry.enabled else "40%"
            rp_lbl.set_markup(f"<span font_weight='700' font_size='9pt' color='{rp_color}' alpha='{rp_alpha}'>{pango_esc(entry.replacement)}</span>")
            rp_box = gtk.Box()
            rp_box.add_css_class("dict-chip-repl")
            rp_box.append(rp_lbl)
            row.append(rp_box)

            # Spacer
            row.append(gtk.Box(hexpand=True))

            # Status tag
            tag_lbl = gtk.Label(xalign=1)
            if entry.enabled:
                tag_lbl.set_markup("<span font_size='7.5pt' color='#34d399' alpha='85%'>ENABLED</span>")
            else:
                tag_lbl.set_markup("<span font_size='7.5pt' alpha='40%'>MUTED</span>")
            row.append(tag_lbl)

            # Delete button
            del_b = gtk.Button(label="✕")
            del_b.add_css_class("dict-btn-del")
            del_b.set_valign(gtk.Align.CENTER)
            del_b.connect("clicked", lambda _b, idx=orig_idx: self._delete_dict_entry(idx))
            row.append(del_b)

            self._dict_rules_vbox.append(row)

    def _toggle_dict_entry(self, idx: int, is_active: bool):
        if 0 <= idx < len(self._dictionary_entries_list):
            e = self._dictionary_entries_list[idx]
            self._dictionary_entries_list[idx] = DictionaryEntry(phrase=e.phrase, replacement=e.replacement, enabled=is_active)
            self._refresh_dict_rules()
            self._update_dict_playground()
            self._render_dict_preview()

    def _delete_dict_entry(self, idx: int):
        if 0 <= idx < len(self._dictionary_entries_list):
            deleted = self._dictionary_entries_list.pop(idx)
            self._refresh_dict_rules()
            self._update_dict_playground()
            self._render_dict_preview()
            self._status_label.set_markup(f"<span font_size='8.5pt' color='#f87171'>● Deleted rule '{pango_esc(deleted.phrase)}'. Click Save to apply.</span>")

    def _build_dict_playground_card(self):
        gtk = self.gtk
        card = gtk.Box(orientation=gtk.Orientation.VERTICAL, spacing=10)
        card.add_css_class("dict-card")

        head = gtk.Box(orientation=gtk.Orientation.HORIZONTAL, spacing=8)
        head.append(self._make_svg("flask"))
        t_box = gtk.Box(orientation=gtk.Orientation.VERTICAL, spacing=2, hexpand=True)
        t = gtk.Label(xalign=0)
        t.set_markup("<span font_weight='700' font_size='11pt' color='#ffffff'>Interactive Vocabulary Playground</span>")
        s = gtk.Label(xalign=0)
        s.set_markup("<span font_size='8.5pt' alpha='55%'>Test your dictionary substitutions live on sample dictation sentences.</span>")
        t_box.append(t)
        t_box.append(s)
        head.append(t_box)
        card.append(head)

        self._dict_test_input = gtk.Entry()
        self._dict_test_input.set_placeholder_text("Type sample speech here (e.g. 'i love pie torch and cube netties daily')...")
        self._dict_test_input.add_css_class("glass-entry")
        self._dict_test_input.connect("changed", self._update_dict_playground)
        card.append(self._dict_test_input)

        out_box = gtk.Box(orientation=gtk.Orientation.VERTICAL, spacing=4)
        out_box.add_css_class("dict-test-box")
        out_box.set_hexpand(True)

        l_tag = gtk.Label(xalign=0)
        l_tag.set_markup("<span font_size='7.5pt' alpha='50%'>TRANSCRIPTION OUTPUT PREVIEW</span>")
        out_box.append(l_tag)

        self._dict_test_output = gtk.Label(xalign=0, wrap=True)
        self._dict_test_output.set_markup("<span font_size='9pt' alpha='40%' font_style='italic'>Transformed text will preview here automatically...</span>")
        out_box.append(self._dict_test_output)
        card.append(out_box)

        return card

    def _update_dict_playground(self, *_args):
        if not self._dict_test_input or not self._dict_test_output:
            return
        raw = self._dict_test_input.get_text()
        if not raw:
            self._dict_test_output.set_markup("<span font_size='9pt' alpha='40%' font_style='italic'>Transformed text will preview here automatically...</span>")
            return

        processor = DictionaryProcessor(self._dictionary_entries_list)
        result = processor.process(raw)
        self._dict_test_output.set_markup(f"<span font_size='9.5pt' font_weight='600' color='#a5b4fc'>{pango_esc(result)}</span>")

    def _build_dict_packs_card(self):
        gtk = self.gtk
        card = gtk.Box(orientation=gtk.Orientation.VERTICAL, spacing=10)
        card.add_css_class("dict-card")

        head = gtk.Box(orientation=gtk.Orientation.HORIZONTAL, spacing=8)
        head.append(self._make_svg("sparkles"))
        t_box = gtk.Box(orientation=gtk.Orientation.VERTICAL, spacing=2, hexpand=True)
        t = gtk.Label(xalign=0)
        t.set_markup("<span font_weight='700' font_size='11pt' color='#ffffff'>Curated Vocabulary Packs</span>")
        s = gtk.Label(xalign=0)
        s.set_markup("<span font_size='8.5pt' alpha='55%'>One-click presets for popular software engineering, ML frameworks, and cloud terminology.</span>")
        t_box.append(t)
        t_box.append(s)
        head.append(t_box)
        card.append(head)

        packs = [
            (
                "AI & Machine Learning",
                "PyTorch, TensorFlow, Hugging Face, WhisperFlow, scikit-learn",
                [
                    ("pie torch", "PyTorch"),
                    ("tensor flow", "TensorFlow"),
                    ("hugging face", "Hugging Face"),
                    ("whisper flow", "WhisperFlow"),
                    ("scikit learn", "scikit-learn"),
                ],
            ),
            (
                "Cloud & DevOps",
                "Kubernetes, Docker Compose, PostgreSQL, GraphQL, gRPC",
                [
                    ("cube netties", "Kubernetes"),
                    ("docker compose", "Docker Compose"),
                    ("post gres", "PostgreSQL"),
                    ("graph ql", "GraphQL"),
                    ("g r p c", "gRPC"),
                ],
            ),
            (
                "Modern Web & Dev",
                "TypeScript, JavaScript, Next.js, Tailwind CSS, GitHub",
                [
                    ("type script", "TypeScript"),
                    ("java script", "JavaScript"),
                    ("next j s", "Next.js"),
                    ("tailwind css", "Tailwind CSS"),
                    ("git hub", "GitHub"),
                ],
            ),
        ]

        packs_row = gtk.Box(orientation=gtk.Orientation.HORIZONTAL, spacing=10)
        packs_row.set_hexpand(True)

        for pack_name, pack_desc, pack_entries in packs:
            p_box = gtk.Box(orientation=gtk.Orientation.VERTICAL, spacing=6)
            p_box.add_css_class("dict-preset-chip")
            p_box.set_hexpand(True)

            p_title = gtk.Label(xalign=0)
            p_title.set_markup(f"<span font_weight='700' font_size='9pt' color='#ffffff'>{pango_esc(pack_name)}</span>")
            p_desc = gtk.Label(xalign=0, wrap=True)
            p_desc.set_markup(f"<span font_size='7.5pt' alpha='50%'>{pango_esc(pack_desc)}</span>")
            p_box.append(p_title)
            p_box.append(p_desc)

            btn = gtk.Button(label="+ Import Pack")
            btn.add_css_class("btn-edit-pill")
            btn.set_halign(gtk.Align.START)
            btn.connect("clicked", lambda _b, n=pack_name, entries=pack_entries: self._import_dict_pack(n, entries))
            p_box.append(btn)
            packs_row.append(p_box)

        card.append(packs_row)
        return card

    def _import_dict_pack(self, pack_name: str, entries: list[tuple[str, str]]):
        added = 0
        for phrase, repl in entries:
            if not any(e.phrase.lower() == phrase.lower() for e in self._dictionary_entries_list):
                self._dictionary_entries_list.append(DictionaryEntry(phrase=phrase, replacement=repl, enabled=True))
                added += 1

        self._refresh_dict_rules()
        self._update_dict_playground()
        self._render_dict_preview()
        msg = f"Imported {added} new terms from '{pack_name}' pack" if added > 0 else f"All terms from '{pack_name}' are already present"
        self._status_label.set_markup(f"<span font_size='8.5pt' color='#34d399'>● {pango_esc(msg)}. Click Save to apply.</span>")

    # -------------------------------------------------------------------------
    # Column 2: Center Model Selection List
    # -------------------------------------------------------------------------
    def _build_center_panel(self):
        gtk = self.gtk
        vbox = gtk.Box(orientation=gtk.Orientation.VERTICAL, spacing=14)
        vbox.add_css_class("center-panel")

        # Header Title Row
        top_row = gtk.Box(orientation=gtk.Orientation.HORIZONTAL, spacing=8)
        head_text = gtk.Box(orientation=gtk.Orientation.VERTICAL, spacing=2, hexpand=True)
        title = gtk.Label(label="Model", xalign=0)
        title.set_markup("<span font_weight='800' font_size='18pt' color='#ffffff'>Model</span>")
        sub = gtk.Label(label="Choose the transcription model that best fits your needs.", xalign=0)
        sub.set_markup("<span font_size='9pt' alpha='55%'>Choose the transcription model that best fits your needs.</span>")
        head_text.append(title)
        head_text.append(sub)
        top_row.append(head_text)

        vbox.append(top_row)

        # Scrolled Model Card List
        scroller = gtk.ScrolledWindow(vexpand=True)
        scroller.set_hexpand(True)
        scroller.set_policy(gtk.PolicyType.NEVER, gtk.PolicyType.EXTERNAL)
        list_box = gtk.Box(orientation=gtk.Orientation.VERTICAL, spacing=4)
        list_box.set_hexpand(True)

        on_disk = {name for name, _size in installed_models()}

        for spec in CATALOG:
            name = spec.name
            meta = MODEL_METADATA.get(
                name,
                {
                    "title": name,
                    "subtitle": "Whisper transcription engine",
                    "desc": "",
                    "size_str": spec.size,
                    "lang": "English",
                    "best_for": "General dictation",
                    "speed_pct": 0.5,
                    "speed_label": "Standard",
                    "acc_pct": 0.8,
                    "acc_label": "Standard",
                    "tag": "",
                },
            )
            present = name in on_disk
            is_active = name == self._selected_model_name

            row = gtk.Box(orientation=gtk.Orientation.HORIZONTAL, spacing=12)
            row.set_hexpand(True)
            row.add_css_class("model-card-row")
            if is_active:
                row.add_css_class("model-card-selected")
            self._model_row_boxes[name] = row

            # Radio Icon
            radio_img = self._make_svg("radio_on" if is_active else "radio_off")
            radio_img.set_valign(gtk.Align.CENTER)
            self._radio_indicators[name] = radio_img
            row.append(radio_img)

            # Center Model Metadata
            card_info = gtk.Box(orientation=gtk.Orientation.VERTICAL, spacing=3, hexpand=True)
            top_line = gtk.Box(orientation=gtk.Orientation.HORIZONTAL, spacing=8)

            name_lbl = gtk.Label(label=name, xalign=0)
            name_alpha = "100%" if present else "50%"
            name_lbl.set_markup(f"<span font_weight='700' font_size='10pt' color='#ffffff' alpha='{name_alpha}'>{pango_esc(name)}</span>")
            top_line.append(name_lbl)
            self._model_name_labels[name] = name_lbl

            size_pill = gtk.Label(label=meta["size_str"])
            size_pill.add_css_class("pill-size")
            top_line.append(size_pill)

            if meta.get("tag"):
                rec_pill = gtk.Label(label=meta["tag"])
                rec_pill.add_css_class("pill-recommended")
                top_line.append(rec_pill)

            card_info.append(top_line)

            sub_lbl = gtk.Label(label=meta["subtitle"], xalign=0)
            sub_lbl.set_markup(f"<span font_size='8.5pt' alpha='50%'>{pango_esc(meta['subtitle'])}</span>")
            card_info.append(sub_lbl)

            # Hidden progress box for downloading
            pbar_box = gtk.Box(orientation=gtk.Orientation.VERTICAL, spacing=4)
            pbar_box.set_visible(False)
            self._download_pbar_boxes[name] = pbar_box

            pbar = gtk.ProgressBar()
            pbar.add_css_class("mini-pbar")
            self._download_pbar[name] = pbar
            pbar_box.append(pbar)

            pbar_lbl = gtk.Label(xalign=0)
            pbar_lbl.set_markup("<span font_size='8pt' color='#a78bfa' font_weight='600'>Starting download...</span>")
            self._download_pbar_labels[name] = pbar_lbl
            pbar_box.append(pbar_lbl)

            card_info.append(pbar_box)

            row.append(card_info)

            # Right Side: Download status & action button
            if not present:
                dl_status = gtk.Label(label="Not downloaded")
                dl_status.set_markup("<span font_size='8.5pt' alpha='40%'>Not downloaded</span>")
                dl_status.set_valign(gtk.Align.CENTER)
                self._download_status_labels[name] = dl_status
                row.append(dl_status)

                dl_btn = gtk.Button()
                dl_btn.set_child(self._make_svg("download"))
                dl_btn.add_css_class("btn-download-icon")
                dl_btn.set_valign(gtk.Align.CENTER)
                dl_btn.set_tooltip_text(f"Download {name}")
                dl_btn.connect("clicked", self._start_download, name)
                self._download_buttons[name] = dl_btn
                row.append(dl_btn)

                cancel_btn = gtk.Button()
                cancel_btn.set_child(self._make_svg("cancel"))
                cancel_btn.add_css_class("btn-cancel-icon")
                cancel_btn.set_valign(gtk.Align.CENTER)
                cancel_btn.set_visible(False)
                cancel_btn.set_tooltip_text(f"Cancel download of {name}")
                cancel_btn.connect("clicked", self._cancel_download, name)
                self._download_cancel_buttons[name] = cancel_btn
                row.append(cancel_btn)

            # Click gesture to select model
            click_gesture = gtk.GestureClick()
            click_gesture.connect("released", lambda _g, _n, _x, _y, m=name: self._select_model(m))
            row.add_controller(click_gesture)

            list_box.append(row)

        scroller.set_child(list_box)
        vbox.append(scroller)
        return vbox

    # -------------------------------------------------------------------------
    # Column 3: Right Inspector Panel
    # -------------------------------------------------------------------------
    def _build_inspector_panel(self):
        gtk = self.gtk
        vbox = gtk.Box(orientation=gtk.Orientation.VERTICAL, spacing=10)
        vbox.add_css_class("inspector-panel")
        vbox.set_size_request(320, -1)
        vbox.set_hexpand(False)
        vbox.set_halign(gtk.Align.END)

        # 1. Model Information Inspector Card
        model_card = gtk.Box(orientation=gtk.Orientation.VERTICAL, spacing=10)
        model_card.add_css_class("inspector-card")
        model_card.set_hexpand(False)

        card_heading = gtk.Label(label="Model Information", xalign=0)
        card_heading.set_markup("<span font_size='8.5pt' font_weight='700' alpha='75%'>Model Information</span>")
        model_card.append(card_heading)

        title_box = gtk.Box(orientation=gtk.Orientation.HORIZONTAL, spacing=8)
        badge = self._make_badge("cube_badge", "inspector-badge", 26)
        title_box.append(badge)

        self._inspector_model_title = gtk.Label(xalign=0)
        self._inspector_model_title.set_markup("<span font_weight='700' font_size='11pt' color='#ffffff'>distil-medium.en</span>")
        title_box.append(self._inspector_model_title)
        model_card.append(title_box)

        self._inspector_model_desc = gtk.Label(xalign=0, wrap=True)
        self._inspector_model_desc.set_max_width_chars(32)
        self._inspector_model_desc.set_markup(
            "<span font_size='8.5pt' alpha='65%'>Distilled 394M parameter model offering the sweet spot of speed, compact size, and accuracy. 6.8x faster than standard medium.</span>"
        )
        model_card.append(self._inspector_model_desc)

        # Key-Value Grid
        grid = gtk.Box(orientation=gtk.Orientation.VERTICAL, spacing=5)

        self._inspector_params = self._make_spec_row(grid, "Parameters", "394M")
        self._inspector_size = self._make_spec_row(grid, "Disk Size", "758 MB")
        self._inspector_ram = self._make_spec_row(grid, "Approx. RAM", "~1.2 GB idle / ~1.7 GB active")
        self._inspector_lang = self._make_spec_row(grid, "Language", "English only")
        self._inspector_best_for = self._make_spec_row(grid, "Best for", "General usage & coding")

        # Speed row with progress bar (fixed 85px bar width)
        speed_row = gtk.Box(orientation=gtk.Orientation.HORIZONTAL, spacing=8)
        sp_lbl = gtk.Label(label="Speed", xalign=0)
        sp_lbl.set_size_request(65, -1)
        sp_lbl.set_markup("<span font_size='8.5pt' alpha='55%'>Speed</span>")
        speed_row.append(sp_lbl)

        sp_container = gtk.Box(orientation=gtk.Orientation.HORIZONTAL, spacing=8, hexpand=True, halign=gtk.Align.END)
        self._inspector_speed_bar = gtk.ProgressBar()
        self._inspector_speed_bar.add_css_class("mini-pbar")
        self._inspector_speed_bar.set_size_request(85, 6)
        self._inspector_speed_bar.set_hexpand(False)
        self._inspector_speed_bar.set_fraction(0.75)
        self._inspector_speed_bar.set_valign(gtk.Align.CENTER)
        sp_container.append(self._inspector_speed_bar)

        self._inspector_speed_lbl = gtk.Label(label="Fast", xalign=1.0)
        self._inspector_speed_lbl.set_size_request(75, -1)
        self._inspector_speed_lbl.set_hexpand(False)
        self._inspector_speed_lbl.set_markup("<span font_size='8.5pt' font_weight='600'>Fast</span>")
        sp_container.append(self._inspector_speed_lbl)
        speed_row.append(sp_container)
        grid.append(speed_row)

        # Accuracy row with progress bar (fixed 85px bar width identical to speed bar)
        acc_row = gtk.Box(orientation=gtk.Orientation.HORIZONTAL, spacing=8)
        ac_lbl = gtk.Label(label="Accuracy", xalign=0)
        ac_lbl.set_size_request(65, -1)
        ac_lbl.set_markup("<span font_size='8.5pt' alpha='55%'>Accuracy</span>")
        acc_row.append(ac_lbl)

        ac_container = gtk.Box(orientation=gtk.Orientation.HORIZONTAL, spacing=8, hexpand=True, halign=gtk.Align.END)
        self._inspector_acc_bar = gtk.ProgressBar()
        self._inspector_acc_bar.add_css_class("mini-pbar")
        self._inspector_acc_bar.set_size_request(85, 6)
        self._inspector_acc_bar.set_hexpand(False)
        self._inspector_acc_bar.set_fraction(0.90)
        self._inspector_acc_bar.set_valign(gtk.Align.CENTER)
        ac_container.append(self._inspector_acc_bar)

        self._inspector_acc_lbl = gtk.Label(label="High", xalign=1.0)
        self._inspector_acc_lbl.set_size_request(75, -1)
        self._inspector_acc_lbl.set_hexpand(False)
        self._inspector_acc_lbl.set_markup("<span font_size='8.5pt' font_weight='600'>High</span>")
        ac_container.append(self._inspector_acc_lbl)
        acc_row.append(ac_container)
        grid.append(acc_row)

        model_card.append(grid)
        vbox.append(model_card)

        # 2. Push-to-Talk Quick Card
        hotkey_card = gtk.Box(orientation=gtk.Orientation.VERTICAL, spacing=8)
        hotkey_card.add_css_class("inspector-card")
        hotkey_card.set_hexpand(False)

        hk_head = gtk.Box(orientation=gtk.Orientation.HORIZONTAL, spacing=6)
        hk_head.append(self._make_svg("reticle"))
        hk_title = gtk.Label(label="Push-to-talk", xalign=0, hexpand=True)
        hk_title.set_markup("<span font_size='9pt' font_weight='700' color='#ffffff'>Push-to-talk</span>")
        hk_head.append(hk_title)

        edit_btn = gtk.Button()
        edit_btn.set_child(self._make_svg("pencil"))
        edit_btn.add_css_class("btn-edit-pill")
        edit_btn.connect("clicked", self._start_capture)
        hk_head.append(edit_btn)
        hotkey_card.append(hk_head)

        # Keycap Chips
        self._hotkey_box = gtk.Box(orientation=gtk.Orientation.HORIZONTAL, spacing=5)
        self._render_hotkey_chips()
        hotkey_card.append(self._hotkey_box)

        hk_sub = gtk.Label(xalign=0)
        hk_sub.set_markup("<span font_size='8pt' alpha='50%'>Hold to transcribe while speaking.</span>")
        hotkey_card.append(hk_sub)
        vbox.append(hotkey_card)

        # 3. Dictionary Quick Card
        dict_card = gtk.Box(orientation=gtk.Orientation.VERTICAL, spacing=8)
        dict_card.add_css_class("inspector-card")
        dict_card.set_hexpand(False)

        dt_head = gtk.Box(orientation=gtk.Orientation.HORIZONTAL, spacing=6)
        dt_head.append(self._make_svg("book"))
        dt_title = gtk.Label(label="Dictionary", xalign=0, hexpand=True)
        dt_title.set_markup("<span font_size='9pt' font_weight='700' color='#ffffff'>Dictionary</span>")
        dt_head.append(dt_title)

        edit_dict_btn = gtk.Button()
        edit_dict_btn.add_css_class("btn-edit-pill")
        ed_box = gtk.Box(orientation=gtk.Orientation.HORIZONTAL, spacing=4)
        ed_box.append(self._make_svg("pencil"))
        ed_lbl = gtk.Label(label="Edit")
        ed_lbl.set_markup("<span font_size='8pt' font_weight='600'>Edit</span>")
        ed_box.append(ed_lbl)
        edit_dict_btn.set_child(ed_box)
        edit_dict_btn.connect("clicked", lambda *_: self._switch_section("dictionary"))
        dt_head.append(edit_dict_btn)
        dict_card.append(dt_head)

        # Preview list of dictionary items
        self._dict_preview_box = gtk.Box(orientation=gtk.Orientation.VERTICAL, spacing=6)
        self._render_dict_preview()
        dict_card.append(self._dict_preview_box)

        vbox.append(dict_card)
        return vbox

    def _make_spec_row(self, container, label: str, val: str):
        gtk = self.gtk
        row = gtk.Box(orientation=gtk.Orientation.HORIZONTAL, spacing=8)
        l = gtk.Label(label=label, xalign=0)
        l.set_markup(f"<span font_size='8.5pt' alpha='55%'>{pango_esc(label)}</span>")
        row.append(l)

        v = gtk.Label(label=val, xalign=1, hexpand=True)
        v.set_markup(f"<span font_size='8.5pt' font_weight='600' color='#ffffff'>{pango_esc(val)}</span>")
        row.append(v)
        container.append(row)
        return v

    def _render_hotkey_chips(self) -> None:
        gtk = self.gtk

        # 1. Update mini inspector box
        if self._hotkey_box is not None:
            while True:
                c = self._hotkey_box.get_first_child()
                if c is None:
                    break
                self._hotkey_box.remove(c)

            for i, key in enumerate(self._hotkey_keys):
                if i > 0:
                    plus = gtk.Label(label="+")
                    plus.set_opacity(0.4)
                    self._hotkey_box.append(plus)
                chip = gtk.Label(label=key.capitalize())
                chip.add_css_class("kbd-chip")
                self._hotkey_box.append(chip)

        # 2. Update large Push-to-Talk Studio box
        if self._ptt_hotkey_box is not None:
            while True:
                c = self._ptt_hotkey_box.get_first_child()
                if c is None:
                    break
                self._ptt_hotkey_box.remove(c)

            modifier_names = {"ctrl", "super", "alt", "shift"}
            for i, key in enumerate(self._hotkey_keys):
                if i > 0:
                    plus = gtk.Label()
                    plus.set_markup("<span font_weight='700' font_size='11pt' alpha='40%'>+</span>")
                    plus.set_valign(gtk.Align.CENTER)
                    self._ptt_hotkey_box.append(plus)
                chip = gtk.Label(label=key.upper() if key.lower() in ("ctrl", "f23") else key.capitalize())
                if key.lower() in modifier_names:
                    chip.add_css_class("kbd-chip-modifier")
                else:
                    chip.add_css_class("kbd-chip-large")
                chip.set_valign(gtk.Align.CENTER)
                self._ptt_hotkey_box.append(chip)

        # 3. Synchronize preset selections
        self._sync_preset_highlights()

    def _render_dict_preview(self) -> None:
        if self._dict_preview_box is None:
            return
        gtk = self.gtk
        while True:
            c = self._dict_preview_box.get_first_child()
            if c is None:
                break
            self._dict_preview_box.remove(c)

        shown = self._dictionary_entries_list[:3]
        for entry in shown:
            r = gtk.Box(orientation=gtk.Orientation.HORIZONTAL, spacing=8)
            r.add_css_class("dict-preview-row")
            cb_img = self._make_svg("checkbox_checked")
            cb_img.set_valign(gtk.Align.CENTER)
            r.append(cb_img)

            txt_phrase = gtk.Label(label=entry.phrase, xalign=0)
            txt_phrase.set_markup(f"<span font_size='8.5pt' color='#ffffff' alpha='70%'>{pango_esc(entry.phrase)}</span>")
            r.append(txt_phrase)

            txt_repl = gtk.Label(label=entry.replacement, xalign=1, hexpand=True)
            txt_repl.set_markup(f"<span font_size='8.5pt' font_weight='700' color='#ffffff'>{pango_esc(entry.replacement)}</span>")
            r.append(txt_repl)
            self._dict_preview_box.append(r)

    # -------------------------------------------------------------------------
    # Footer Action Bar
    # -------------------------------------------------------------------------
    def _build_footer(self):
        gtk = self.gtk
        footer = gtk.Box(orientation=gtk.Orientation.HORIZONTAL, spacing=12)
        footer.add_css_class("footer-bar")

        info_img = self._make_svg("footer_info")
        info_img.set_valign(gtk.Align.CENTER)
        footer.append(info_img)

        self._status_label.set_hexpand(True)
        self._status_label.set_markup("<span font_size='8.5pt' alpha='65%'>Closing this window leaves Wispo in the dock.</span>")
        footer.append(self._status_label)

        # Right Actions: Quit & Save
        quit_btn = gtk.Button(label="Quit")
        quit_btn.add_css_class("btn-quit")
        quit_btn.connect("clicked", self._on_red_clicked)
        footer.append(quit_btn)

        save_btn = gtk.Button(label="Save")
        save_btn.add_css_class("btn-save-purple")
        save_btn.connect("clicked", self._save)
        footer.append(save_btn)

        return footer

    # -------------------------------------------------------------------------
    # Model Selection & Dynamic Inspector Updates
    # -------------------------------------------------------------------------
    def _select_model(self, name: str) -> None:
        on_disk = {m for m, _ in installed_models()}
        if name not in on_disk:
            self._status_label.set_markup(f"<span font_size='8.5pt' alpha='75%'>Please download {pango_esc(name)} first using the arrow button.</span>")
            self._update_inspector(name)
            return

        self._selected_model_name = name

        # Update Radio circles and selected class
        for m_name, row in self._model_row_boxes.items():
            radio_img = self._radio_indicators.get(m_name)
            if m_name == name:
                row.add_css_class("model-card-selected")
                if radio_img:
                    self._update_radio_icon(radio_img, True)
            else:
                row.remove_css_class("model-card-selected")
                if radio_img:
                    self._update_radio_icon(radio_img, False)

        self._update_inspector(name)
        self._status_label.set_markup(f"<span font_size='8.5pt' alpha='85%'>Selected {pango_esc(name)}. Click Save to apply.</span>")

    def _update_radio_icon(self, img_widget, is_checked: bool) -> None:
        from gi.repository import Gdk, GLib

        svg_data = SVG_ICONS["radio_on" if is_checked else "radio_off"].encode("utf-8")
        texture = Gdk.Texture.new_from_bytes(GLib.Bytes.new(svg_data))
        img_widget.set_from_paintable(texture)

    def _update_inspector(self, name: str) -> None:
        meta = MODEL_METADATA.get(name)
        if not meta:
            return

        if self._inspector_model_title:
            self._inspector_model_title.set_markup(f"<span font_weight='700' font_size='11pt' color='#ffffff'>{pango_esc(name)}</span>")
        if self._inspector_model_desc:
            self._inspector_model_desc.set_markup(f"<span font_size='8.5pt' alpha='65%'>{pango_esc(meta['desc'])}</span>")
        if self._inspector_params:
            self._inspector_params.set_markup(f"<span font_size='8.5pt' font_weight='600' color='#ffffff'>{pango_esc(meta.get('params', ''))}</span>")
        if self._inspector_size:
            self._inspector_size.set_markup(f"<span font_size='8.5pt' font_weight='600' color='#ffffff'>{pango_esc(meta['size_str'])}</span>")
        if self._inspector_ram:
            self._inspector_ram.set_markup(f"<span font_size='8.5pt' font_weight='600' color='#ffffff'>{pango_esc(meta.get('ram_str', ''))}</span>")
        if self._inspector_lang:
            self._inspector_lang.set_markup(f"<span font_size='8.5pt' font_weight='600' color='#ffffff'>{pango_esc(meta['lang'])}</span>")
        if self._inspector_best_for:
            self._inspector_best_for.set_markup(f"<span font_size='8.5pt' font_weight='600' color='#ffffff'>{pango_esc(meta['best_for'])}</span>")
        if self._inspector_speed_bar:
            self._inspector_speed_bar.set_fraction(meta["speed_pct"])
        if self._inspector_speed_lbl:
            self._inspector_speed_lbl.set_markup(f"<span font_size='8.5pt' font_weight='600'>{pango_esc(meta['speed_label'])}</span>")
        if self._inspector_acc_bar:
            self._inspector_acc_bar.set_fraction(meta["acc_pct"])
        if self._inspector_acc_lbl:
            self._inspector_acc_lbl.set_markup(f"<span font_size='8.5pt' font_weight='600'>{pango_esc(meta['acc_label'])}</span>")

    # -------------------------------------------------------------------------
    # Learn More Modal
    # -------------------------------------------------------------------------
    def _open_learn_modal(self, *_args) -> None:
        gtk = self.gtk
        dlg = gtk.Window(title="Whisper Models Guide")
        dlg.set_transient_for(self.window)
        dlg.set_modal(True)
        dlg.set_default_size(440, 360)
        dlg.add_css_class("macos-window")

        box = gtk.Box(orientation=gtk.Orientation.VERTICAL, spacing=12)
        box.set_margin_top(20)
        box.set_margin_bottom(20)
        box.set_margin_start(20)
        box.set_margin_end(20)

        t = gtk.Label(xalign=0)
        t.set_markup("<span font_weight='800' font_size='14pt' color='#ffffff'>Choosing the Right Model</span>")
        box.append(t)

        info_text = (
            "<b>Distilled Models (Recommended):</b>\n"
            "• <i>distil-medium.en</i> & <i>distil-small.en</i> provide 6-8x faster inference on Intel Arc GPU while retaining 99% accuracy.\n\n"
            "<b>Standard OpenAI Models:</b>\n"
            "• <i>tiny.en</i> / <i>base.en</i>: Lowest CPU/RAM footprint, optimal for simple commands.\n"
            "• <i>small.en</i> / <i>medium.en</i>: Higher vocabulary depth and accent resilience.\n\n"
            "<b>Large & Turbo:</b>\n"
            "• <i>large-v3-turbo-q5_0</i>: State-of-the-art accuracy with 4-layer fast decoding."
        )
        body = gtk.Label(xalign=0, wrap=True)
        body.set_markup(f"<span font_size='9pt' alpha='75%'>{info_text}</span>")
        box.append(body)

        box.append(gtk.Box(vexpand=True))

        close_btn = gtk.Button(label="Got it")
        close_btn.add_css_class("btn-save-purple")
        close_btn.connect("clicked", lambda *_: dlg.close())
        close_btn.set_halign(gtk.Align.END)
        box.append(close_btn)

        dlg.set_child(box)
        dlg.present()

    # -------------------------------------------------------------------------
    # Model Download Handling
    # -------------------------------------------------------------------------
    def _start_download(self, _button, name: str) -> None:
        cancel_ev = threading.Event()
        self._download_cancel_events[name] = cancel_ev

        btn = self._download_buttons.get(name)
        if btn is not None:
            btn.set_visible(False)

        c_btn = self._download_cancel_buttons.get(name)
        if c_btn is not None:
            c_btn.set_visible(True)
            c_btn.set_sensitive(True)

        status_lbl = self._download_status_labels.get(name)
        if status_lbl is not None:
            status_lbl.set_markup("<span font_size='8.5pt' color='#a78bfa' font_weight='600'>Downloading...</span>")

        pbar_box = self._download_pbar_boxes.get(name)
        if pbar_box is not None:
            pbar_box.set_visible(True)

        pbar = self._download_pbar.get(name)
        if pbar is not None:
            pbar.set_fraction(0.0)

        pbar_lbl = self._download_pbar_labels.get(name)
        if pbar_lbl is not None:
            pbar_lbl.set_markup("<span font_size='8pt' color='#a78bfa' font_weight='600'>Starting download...</span>")

        self._status_label.set_markup(f"<span font_size='8.5pt' alpha='80%'>Downloading {pango_esc(name)}...</span>")
        last = 0.0

        def progress(written: int, total: int) -> None:
            nonlocal last
            now = time.monotonic()
            if written != total and now - last < 0.15:
                return
            last = now
            self._idle(self._download_progress, name, written, total)

        def work() -> None:
            try:
                download_model(name, progress=progress, cancel_event=cancel_ev)
            except InterruptedError:
                self._idle(self._download_cancelled, name)
                return
            except Exception as exc:
                self._idle(self._download_failed, name, str(exc))
                return
            self._idle(self._download_done, name)

        threading.Thread(target=work, name=f"whisperflow-download-{name}", daemon=True).start()

    def _cancel_download(self, _button, name: str) -> None:
        cancel_ev = self._download_cancel_events.get(name)
        if cancel_ev is not None:
            cancel_ev.set()

        c_btn = self._download_cancel_buttons.get(name)
        if c_btn is not None:
            c_btn.set_sensitive(False)

        status_lbl = self._download_status_labels.get(name)
        if status_lbl is not None:
            status_lbl.set_markup("<span font_size='8.5pt' color='#f87171'>Cancelling...</span>")

        pbar_lbl = self._download_pbar_labels.get(name)
        if pbar_lbl is not None:
            pbar_lbl.set_markup("<span font_size='8pt' color='#f87171' font_weight='600'>Cancelling download...</span>")

    def _download_cancelled(self, name: str) -> bool:
        self._download_cancel_events.pop(name, None)

        pbar_box = self._download_pbar_boxes.get(name)
        if pbar_box is not None:
            pbar_box.set_visible(False)

        c_btn = self._download_cancel_buttons.get(name)
        if c_btn is not None:
            c_btn.set_visible(False)

        btn = self._download_buttons.get(name)
        if btn is not None:
            btn.set_visible(True)
            btn.set_sensitive(True)

        status_lbl = self._download_status_labels.get(name)
        if status_lbl is not None:
            status_lbl.set_markup("<span font_size='8.5pt' alpha='40%'>Not downloaded</span>")

        self._status_label.set_markup(f"<span font_size='8.5pt' color='#fbbf24'>Download of {pango_esc(name)} cancelled.</span>")
        return False

    def _download_progress(self, name: str, written: int, total: int) -> bool:
        if name not in self._download_cancel_events:
            return False

        pbar = self._download_pbar.get(name)
        pbar_lbl = self._download_pbar_labels.get(name)
        status_lbl = self._download_status_labels.get(name)

        if total > 0:
            pct = min(1.0, max(0.0, written / total))
            pct_int = int(pct * 100)
            written_str = _format_bytes(written)
            total_str = _format_bytes(total)
            progress_str = f"{pct_int}% ({written_str} / {total_str})"

            if pbar is not None:
                pbar.set_fraction(pct)
            if pbar_lbl is not None:
                pbar_lbl.set_markup(f"<span font_size='8pt' color='#c4b5fd' font_weight='600'>{pango_esc(progress_str)}</span>")
            if status_lbl is not None:
                status_lbl.set_markup(f"<span font_size='8.5pt' color='#a78bfa' font_weight='600'>{pct_int}%</span>")
            self._status_label.set_markup(f"<span font_size='8.5pt' alpha='80%'>Downloading {pango_esc(name)}... {pango_esc(progress_str)}</span>")
        return False

    def _download_done(self, name: str) -> bool:
        self._download_cancel_events.pop(name, None)

        pbar_box = self._download_pbar_boxes.get(name)
        if pbar_box is not None:
            pbar_box.set_visible(False)

        c_btn = self._download_cancel_buttons.get(name)
        if c_btn is not None:
            c_btn.set_visible(False)

        btn = self._download_buttons.get(name)
        if btn is not None:
            btn.set_visible(False)

        status_lbl = self._download_status_labels.get(name)
        if status_lbl is not None:
            status_lbl.set_visible(False)

        name_lbl = self._model_name_labels.get(name)
        if name_lbl is not None:
            name_lbl.set_markup(f"<span font_weight='700' font_size='10pt' color='#ffffff' alpha='100%'>{pango_esc(name)}</span>")

        self._select_model(name)
        self._status_label.set_markup(f"<span font_size='8.5pt' font_weight='700' color='#34d399'>● Downloaded {pango_esc(name)} successfully! Click Save to apply.</span>")
        return False

    def _download_failed(self, name: str, message: str) -> bool:
        self._download_cancel_events.pop(name, None)

        pbar_box = self._download_pbar_boxes.get(name)
        if pbar_box is not None:
            pbar_box.set_visible(False)

        c_btn = self._download_cancel_buttons.get(name)
        if c_btn is not None:
            c_btn.set_visible(False)

        btn = self._download_buttons.get(name)
        if btn is not None:
            btn.set_visible(True)
            btn.set_sensitive(True)

        status_lbl = self._download_status_labels.get(name)
        if status_lbl is not None:
            status_lbl.set_markup("<span font_size='8.5pt' alpha='40%'>Not downloaded</span>")

        self._status_label.set_markup(f"<span font_size='8.5pt' color='#f87171'>Could not download {pango_esc(name)}: {pango_esc(message)}</span>")
        return False

    # -------------------------------------------------------------------------
    # Shortcut Capture Handling
    # -------------------------------------------------------------------------
    def _stop_capture(self, *_args) -> bool:
        self._capture_stop.set()
        self._is_capturing_shortcut = False
        if self._ptt_record_btn:
            self._ptt_record_btn.remove_css_class("btn-record-active")
            self._ptt_record_btn.add_css_class("btn-record-chord")
            self._ptt_record_btn.set_label("🔴 Record Shortcut")
        if self._ptt_status_lbl:
            self._ptt_status_lbl.set_markup("<span font_size='8.5pt' alpha='50%'>Hold down the shortcut while speaking in any application. Click Record to rebind.</span>")
        return False

    def _start_capture(self, *_args) -> None:
        if self._is_capturing_shortcut:
            self._stop_capture()
            return
        self._capture_stop.set()
        self._capture_stop = threading.Event()
        stop = self._capture_stop
        self._is_capturing_shortcut = True

        if self._ptt_record_btn:
            self._ptt_record_btn.remove_css_class("btn-record-chord")
            self._ptt_record_btn.add_css_class("btn-record-active")
            self._ptt_record_btn.set_label("⏹ Cancel (Listening...)")
        if self._ptt_status_lbl:
            self._ptt_status_lbl.set_markup("<span font_size='9pt' font_weight='700' color='#f87171'>● Listening: Press and release your shortcut keys on the keyboard...</span>")
        self._status_label.set_markup("<span font_size='8.5pt' color='#a855f7' font_weight='bold'>● Listening: Press the shortcut keys on your keyboard...</span>")

        def work() -> None:
            try:
                keys = capture_chord(stop, timeout=10.0)
            except Exception as exc:
                self._idle(self._shortcut_failed, str(exc))
                return
            self._idle(self._shortcut_done, keys)

        threading.Thread(target=work, name="whisperflow-capture", daemon=True).start()

    def _shortcut_done(self, keys: tuple[str, ...]) -> bool:
        self._is_capturing_shortcut = False
        if self._ptt_record_btn:
            self._ptt_record_btn.remove_css_class("btn-record-active")
            self._ptt_record_btn.add_css_class("btn-record-chord")
            self._ptt_record_btn.set_label("🔴 Record Shortcut")

        if not keys:
            if self._ptt_status_lbl:
                self._ptt_status_lbl.set_markup("<span font_size='8.5pt' alpha='65%'>Capture timed out. No shortcut pressed.</span>")
            self._status_label.set_markup("<span font_size='8.5pt' alpha='65%'>No shortcut captured.</span>")
            return False

        self._hotkey_keys = keys
        self._render_hotkey_chips()
        chord_str = format_chord(keys)
        if self._ptt_status_lbl:
            self._ptt_status_lbl.set_markup(f"<span font_size='9pt' font_weight='700' color='#34d399'>● Assigned to {pango_esc(chord_str)}. Click Save to apply.</span>")
        self._status_label.set_markup(f"<span font_size='8.5pt' font_weight='700' color='#34d399'>● Shortcut updated to {pango_esc(chord_str)}. Click Save to apply.</span>")
        return False

    def _shortcut_failed(self, message: str) -> bool:
        self._is_capturing_shortcut = False
        if self._ptt_record_btn:
            self._ptt_record_btn.remove_css_class("btn-record-active")
            self._ptt_record_btn.add_css_class("btn-record-chord")
            self._ptt_record_btn.set_label("🔴 Record Shortcut")
        if self._ptt_status_lbl:
            self._ptt_status_lbl.set_markup(f"<span font_size='8.5pt' color='#f87171'>Capture error: {pango_esc(message)}</span>")
        self._status_label.set_markup(f"<span font_size='8.5pt' color='#f87171'>Shortcut capture failed: {pango_esc(message)}</span>")
        return False

    # -------------------------------------------------------------------------
    # Dictionary Modal Window
    # -------------------------------------------------------------------------
    def _open_dictionary_modal(self, *_args) -> None:
        gtk = self.gtk
        dlg = gtk.Window(title="Edit Dictionary")
        dlg.set_transient_for(self.window)
        dlg.set_modal(True)
        dlg.set_default_size(500, 420)
        dlg.add_css_class("macos-window")

        box = gtk.Box(orientation=gtk.Orientation.VERTICAL, spacing=12)
        box.set_margin_top(18)
        box.set_margin_bottom(18)
        box.set_margin_start(18)
        box.set_margin_end(18)

        title = gtk.Label(label="Custom Dictionary", xalign=0)
        title.set_markup("<span font_weight='800' font_size='14pt' color='#ffffff'>Custom Dictionary</span>")
        box.append(title)

        sub = gtk.Label(label="Automatically correct phonetic errors and replace technical terms.", xalign=0)
        sub.set_markup("<span font_size='8.5pt' alpha='60%'>Automatically correct phonetic errors and replace technical terms.</span>")
        box.append(sub)

        scroller = gtk.ScrolledWindow(vexpand=True)
        rows_box = gtk.Box(orientation=gtk.Orientation.VERTICAL, spacing=8)

        entry_widgets = []
        for e in self._dictionary_entries_list:
            r = gtk.Box(orientation=gtk.Orientation.HORIZONTAL, spacing=8)
            cb = gtk.CheckButton(active=e.enabled)
            p_in = gtk.Entry(text=e.phrase, placeholder_text="Spoken phrase", hexpand=True)
            p_in.add_css_class("glass-entry")
            arr = gtk.Label(label="➔")
            arr.set_opacity(0.4)
            r_in = gtk.Entry(text=e.replacement, placeholder_text="Replacement", hexpand=True)
            r_in.add_css_class("glass-entry")
            del_b = gtk.Button(label="✕")
            del_b.add_css_class("btn-quit")

            r.append(cb)
            r.append(p_in)
            r.append(arr)
            r.append(r_in)
            r.append(del_b)
            rows_box.append(r)
            entry_widgets.append((cb, p_in, r_in, r))
            del_b.connect("clicked", lambda *_a, row=r: rows_box.remove(row))

        def add_new(*_):
            r = gtk.Box(orientation=gtk.Orientation.HORIZONTAL, spacing=8)
            cb = gtk.CheckButton(active=True)
            p_in = gtk.Entry(placeholder_text="Spoken phrase", hexpand=True)
            p_in.add_css_class("glass-entry")
            arr = gtk.Label(label="➔")
            arr.set_opacity(0.4)
            r_in = gtk.Entry(placeholder_text="Replacement", hexpand=True)
            r_in.add_css_class("glass-entry")
            del_b = gtk.Button(label="✕")
            del_b.add_css_class("btn-quit")
            r.append(cb)
            r.append(p_in)
            r.append(arr)
            r.append(r_in)
            r.append(del_b)
            rows_box.append(r)
            entry_widgets.append((cb, p_in, r_in, r))
            del_b.connect("clicked", lambda *_a, row=r: rows_box.remove(row))

        scroller.set_child(rows_box)
        box.append(scroller)

        btn_row = gtk.Box(orientation=gtk.Orientation.HORIZONTAL, spacing=8)
        add_b = gtk.Button(label="+ Add Word")
        add_b.add_css_class("btn-quit")
        add_b.connect("clicked", add_new)
        btn_row.append(add_b)

        btn_row.append(gtk.Box(hexpand=True))

        done_b = gtk.Button(label="Done")
        done_b.add_css_class("btn-save-purple")

        def save_and_close(*_):
            new_entries = []
            for cb, p_in, r_in, r in entry_widgets:
                if r.get_parent() is not None:
                    phrase = p_in.get_text().strip()
                    repl = r_in.get_text().strip()
                    if phrase:
                        new_entries.append(DictionaryEntry(phrase, repl, cb.get_active()))
            self._dictionary_entries_list = new_entries
            self._render_dict_preview()
            dlg.close()

        done_b.connect("clicked", save_and_close)
        btn_row.append(done_b)
        box.append(btn_row)

        dlg.set_child(box)
        dlg.present()

    # -------------------------------------------------------------------------
    # Save Settings
    # -------------------------------------------------------------------------
    def _save(self, *_args) -> None:
        model = self._selected_model_name
        if not model_path(model).is_file():
            self._status_label.set_markup(f"<span font_size='8.5pt' color='#f87171'>Please download {pango_esc(model)} before saving.</span>")
            return
        path = project_root() / "whisperflow.yaml"
        vad_data = {
            "rms_threshold": round(float(self._vad_rms_threshold), 4),
            "noise_multiplier": round(float(self._vad_noise_mult), 2),
            "hangover_ms": int(self._vad_hangover_ms),
            "highpass_hz": round(float(self._vad_highpass_hz), 1),
        }
        save_user_settings(
            path=path,
            model=model,
            entries=self._dictionary_entries_list,
            hotkey=self._hotkey_keys,
            audio_device=self._selected_audio_device,
            vad_settings=vad_data,
        )
        new_config = load_config(path)
        old_model = self._config.asr.model
        self._config = new_config

        if self._on_save_config is not None:
            self._on_save_config(new_config)

        if old_model != model:
            self.set_loading_progress(10, "Reloading Model...", f"Switching to {model}")
            self._status_label.set_markup(
                "<span font_size='8.5pt' font_weight='700' color='#a5b4fc'>● Settings saved. Reloading model...</span>"
            )
        else:
            self._status_label.set_markup(
                "<span font_size='8.5pt' font_weight='700' color='#34d399'>● Settings saved & applied!</span>"
            )

    def _idle(self, callback, *args) -> None:
        from gi.repository import GLib

        GLib.idle_add(callback, *args)


if __name__ == "__main__":
    raise SystemExit(main())
