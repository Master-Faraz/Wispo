"""Settings window. Apple-inspired Liquid Glass modern aesthetic matching macOS Sonoma design.

    uv run python -m whisperflow.settings

The transcription command keeps the model it loaded at startup. Restart it
after saving to use a new model or dictionary.
"""

from __future__ import annotations

import html
import os
import sys
import threading
import time
from pathlib import Path

from whisperflow.asr.device import select_device
from whisperflow.config import DictionaryEntry, load_config, project_root, save_user_settings
from whisperflow.hotkey.linux import capture_chord, format_chord
from whisperflow.models import CATALOG, DEFAULT_MODEL, download_model, installed_models, model_path

# Helper to escape any text for Pango markup
def pango_esc(text: str) -> str:
    return html.escape(str(text))


# Rich metadata verified from whisper.cpp and distil-whisper official benchmarks
MODEL_METADATA: dict[str, dict] = {
    "tiny.en": {
        "title": "tiny.en",
        "subtitle": "Fastest, good for clean audio.",
        "desc": "An ultra-compact 39M parameter model optimized for minimal memory footprint and fast inference. Best for rapid commands and clean audio on low-power CPUs.",
        "params": "39M",
        "size_str": "75 MB",
        "ram_str": "~273 MB RAM",
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
        "ram_str": "~388 MB RAM",
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
        "ram_str": "~852 MB RAM",
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
        "ram_str": "~2.1 GB RAM",
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
        "ram_str": "~1.1 GB RAM",
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
        "ram_str": "~1.8 GB RAM",
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
        "ram_str": "~600 MB RAM",
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
        "ram_str": "~1.2 GB RAM",
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
        "ram_str": "~2.1 GB RAM",
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
        "ram_str": "~2.1 GB RAM",
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
    "microphone": """<svg xmlns="http://www.w3.org/2000/svg" width="17" height="17" viewBox="0 0 24 24" fill="none" stroke="rgba(255,255,255,0.7)" stroke-width="1.8" stroke-linecap="round" stroke-linejoin="round"><path d="M12 2a3 3 0 0 0-3 3v7a3 3 0 0 0 6 0V5a3 3 0 0 0-3-3Z"/><path d="M19 10v2a7 7 0 0 1-14 0v-2"/><line x1="12" x2="12" y1="19" y2="22"/></svg>""",
    "keyboard": """<svg xmlns="http://www.w3.org/2000/svg" width="17" height="17" viewBox="0 0 24 24" fill="none" stroke="rgba(255,255,255,0.7)" stroke-width="1.8" stroke-linecap="round" stroke-linejoin="round"><rect width="20" height="14" x="2" y="5" rx="2.5"/><line x1="6" x2="6.01" y1="9" y2="9" stroke-width="2.5"/><line x1="10" x2="10.01" y1="9" y2="9" stroke-width="2.5"/><line x1="14" x2="14.01" y1="9" y2="9" stroke-width="2.5"/><line x1="18" x2="18.01" y1="9" y2="9" stroke-width="2.5"/><line x1="7" x2="17" y1="14" y2="14"/></svg>""",
    "book": """<svg xmlns="http://www.w3.org/2000/svg" width="17" height="17" viewBox="0 0 24 24" fill="none" stroke="rgba(255,255,255,0.7)" stroke-width="1.8" stroke-linecap="round" stroke-linejoin="round"><path d="M4 19.5v-15A2.5 2.5 0 0 1 6.5 2H20v20H6.5a2.5 2.5 0 0 1-2.5-2.5Z"/><path d="M6 6h10"/><path d="M6 10h10"/></svg>""",
    "cog": """<svg xmlns="http://www.w3.org/2000/svg" width="17" height="17" viewBox="0 0 24 24" fill="none" stroke="rgba(255,255,255,0.7)" stroke-width="1.8" stroke-linecap="round" stroke-linejoin="round"><circle cx="12" cy="12" r="3"/><path d="M19.4 15a1.65 1.65 0 0 0 .33 1.82l.06.06a2 2 0 0 1 0 2.83 2 2 0 0 1-2.83 0l-.06-.06a1.65 1.65 0 0 0-1.82-.33 1.65 1.65 0 0 0-1 1.51V21a2 2 0 0 1-2 2 2 2 0 0 1-2-2v-.09A1.65 1.65 0 0 0 9 19.4a1.65 1.65 0 0 0-1.82.33l-.06.06a2 2 0 0 1-2.83 0 2 2 0 0 1 0-2.83l.06-.06a1.65 1.65 0 0 0 .33-1.82 1.65 1.65 0 0 0-1.51-1H3a2 2 0 0 1-2-2 2 2 0 0 1 2-2h.09A1.65 1.65 0 0 0 4.6 9a1.65 1.65 0 0 0-.33-1.82l-.06-.06a2 2 0 0 1 0-2.83 2 2 0 0 1 2.83 0l.06.06a1.65 1.65 0 0 0 1.82.33H9a1.65 1.65 0 0 0 1-1.51V3a2 2 0 0 1 2-2 2 2 0 0 1 2 2v.09a1.65 1.65 0 0 0 1 1.51 1.65 1.65 0 0 0 1.82-.33l.06-.06a2 2 0 0 1 2.83 0 2 2 0 0 1 0 2.83l-.06.06a1.65 1.65 0 0 0-.33 1.82V9a1.65 1.65 0 0 0 1.51 1H21a2 2 0 0 1 2 2 2 2 0 0 1-2 2h-.09a1.65 1.65 0 0 0-1.51 1z"/></svg>""",
    "info": """<svg xmlns="http://www.w3.org/2000/svg" width="17" height="17" viewBox="0 0 24 24" fill="none" stroke="rgba(255,255,255,0.7)" stroke-width="1.8" stroke-linecap="round" stroke-linejoin="round"><circle cx="12" cy="12" r="10"/><line x1="12" x2="12" y1="16" y2="12"/><line x1="12" x2="12.01" y1="8" y2="8" stroke-width="2.5"/></svg>""",
    "drive": """<svg xmlns="http://www.w3.org/2000/svg" width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="rgba(255,255,255,0.6)" stroke-width="1.8" stroke-linecap="round" stroke-linejoin="round"><rect width="20" height="8" x="2" y="14" rx="2"/><path d="M6 18h.01"/><path d="M10 18h.01"/><rect width="20" height="8" x="2" y="4" rx="2"/><path d="M6 8h.01"/><path d="M10 8h.01"/></svg>""",
    "download": """<svg xmlns="http://www.w3.org/2000/svg" width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="rgba(255,255,255,0.8)" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><path d="M21 15v4a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2v-4"/><polyline points="7 10 12 15 17 10"/><line x1="12" x2="12" y1="15" y2="3"/></svg>""",
    "radio_off": """<svg xmlns="http://www.w3.org/2000/svg" width="20" height="20" viewBox="0 0 24 24"><circle cx="12" cy="12" r="9" fill="none" stroke="rgba(255, 255, 255, 0.28)" stroke-width="2"/></svg>""",
    "radio_on": """<svg xmlns="http://www.w3.org/2000/svg" width="20" height="20" viewBox="0 0 24 24"><circle cx="12" cy="12" r="9" fill="#6366f1" stroke="#818cf8" stroke-width="2"/><circle cx="12" cy="12" r="3.5" fill="#ffffff"/></svg>""",
    "checkbox_checked": """<svg xmlns="http://www.w3.org/2000/svg" width="17" height="17" viewBox="0 0 24 24"><rect width="20" height="20" x="2" y="2" rx="5" fill="#6366f1"/><polyline points="6 12 10 16 18 8" fill="none" stroke="#ffffff" stroke-width="2.5" stroke-linecap="round" stroke-linejoin="round"/></svg>""",
    "pencil": """<svg xmlns="http://www.w3.org/2000/svg" width="13" height="13" viewBox="0 0 24 24" fill="none" stroke="rgba(255,255,255,0.7)" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><path d="M17 3a2.85 2.83 0 1 1 4 4L7.5 20.5 2 22l1.5-5.5Z"/><path d="m15 5 4 4"/></svg>""",
    "reticle": """<svg xmlns="http://www.w3.org/2000/svg" width="15" height="15" viewBox="0 0 24 24" fill="none" stroke="rgba(255,255,255,0.75)" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><circle cx="12" cy="12" r="10"/><path d="M12 2v4"/><path d="M12 18v4"/><path d="M2 12h4"/><path d="M18 12h4"/></svg>""",
    "footer_info": """<svg xmlns="http://www.w3.org/2000/svg" width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="rgba(255,255,255,0.55)" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><circle cx="12" cy="12" r="10"/><line x1="12" x2="12" y1="16" y2="12"/><line x1="12" x2="12.01" y1="8" y2="8" stroke-width="2.5"/></svg>""",
    # Traffic light hover icons - compact (4.8px), razor-sharp, high-contrast Windows/macOS design
    "traffic_close": """<svg xmlns="http://www.w3.org/2000/svg" width="14" height="14" viewBox="0 0 14 14" fill="none"><path d="M4.6 4.6L9.4 9.4M9.4 4.6L4.6 9.4" stroke="#000000" stroke-width="1.25" stroke-linecap="round"/></svg>""",
    "traffic_max": """<svg xmlns="http://www.w3.org/2000/svg" width="14" height="14" viewBox="0 0 14 14" fill="none"><rect x="4.5" y="4.5" width="5" height="5" stroke="#000000" stroke-width="1.25" stroke-linejoin="miter"/></svg>""",
    "traffic_min": """<svg xmlns="http://www.w3.org/2000/svg" width="14" height="14" viewBox="0 0 14 14" fill="none"><line x1="4.3" y1="7" x2="9.7" y2="7" stroke="#000000" stroke-width="1.3" stroke-linecap="round"/></svg>""",
}

_MACOS_LIQUID_GLASS_CSS = """
/* Liquid Glass Apple Window Theme */
window.macos-window {
    background: radial-gradient(ellipse at 85% 15%, rgba(139, 92, 246, 0.14) 0%, transparent 60%),
                radial-gradient(ellipse at 15% 85%, rgba(249, 115, 22, 0.08) 0%, transparent 55%),
                linear-gradient(135deg, #151722 0%, #0d0f17 100%);
    color: #f8fafc;
    border-radius: 18px;
    border: 1px solid rgba(255, 255, 255, 0.09);
    box-shadow: 0 28px 70px rgba(0, 0, 0, 0.65);
}

/* Maximized / Full Screen: square corners to seamlessly fill screen without showing background */
window.macos-window.window-maximized {
    border-radius: 0;
    border: none;
    box-shadow: none;
}

/* Sidebar Column */
.sidebar-panel {
    background: rgba(255, 255, 255, 0.018);
    border-right: 1px solid rgba(255, 255, 255, 0.06);
    padding: 16px 14px 14px 14px;
}

button.traffic-light {
    border-radius: 50%;
    min-width: 14px;
    min-height: 14px;
    padding: 0;
    margin: 0;
    border: none;
    box-shadow: none;
    transition: all 120ms ease;
}

button.traffic-light image {
    opacity: 0;
    transition: opacity 120ms ease;
}

.traffic-box:hover button.traffic-light image,
button.traffic-light:hover image {
    opacity: 1;
}

button.traffic-red {
    background: #ff5f57;
    border: 0.5px solid rgba(0, 0, 0, 0.18);
}
button.traffic-red:hover {
    background: #ff453a;
}

button.traffic-yellow {
    background: #febc2e;
    border: 0.5px solid rgba(0, 0, 0, 0.18);
}
button.traffic-yellow:hover {
    background: #ffcc00;
}

button.traffic-green {
    background: #28c840;
    border: 0.5px solid rgba(0, 0, 0, 0.18);
}
button.traffic-green:hover {
    background: #34c759;
}

.app-brand-badge {
    background: linear-gradient(135deg, #6366f1, #8b5cf6);
    border-radius: 10px;
    min-width: 34px;
    min-height: 34px;
    box-shadow: 0 4px 14px rgba(99, 102, 241, 0.4);
}

/* Sidebar Nav Items */
.nav-item {
    background: transparent;
    border: 1px solid transparent;
    border-radius: 11px;
    padding: 7px 10px;
    margin-bottom: 3px;
    transition: all 160ms ease;
}

.nav-item:hover {
    background: rgba(255, 255, 255, 0.04);
}

.nav-item-active {
    background: linear-gradient(135deg, rgba(99, 102, 241, 0.26) 0%, rgba(139, 92, 246, 0.14) 100%);
    border: 1px solid rgba(139, 92, 246, 0.45);
    box-shadow: 0 4px 14px rgba(99, 102, 241, 0.22), inset 0 1px 0 rgba(255, 255, 255, 0.12);
}

.storage-card {
    background: rgba(255, 255, 255, 0.025);
    border: 1px solid rgba(255, 255, 255, 0.05);
    border-radius: 10px;
    padding: 8px 10px;
}

/* Center Panel (Models List) */
.center-panel {
    padding: 22px 26px 14px 26px;
}

.btn-learn {
    background: rgba(255, 255, 255, 0.05);
    border: 1px solid rgba(255, 255, 255, 0.12);
    border-radius: 14px;
    color: rgba(255, 255, 255, 0.85);
    font-size: 11px;
    font-weight: 500;
    padding: 4px 12px;
    transition: all 150ms ease;
}

.btn-learn:hover {
    background: rgba(255, 255, 255, 0.09);
    color: #ffffff;
}

.model-card-row {
    background: rgba(255, 255, 255, 0.02);
    border: 1px solid rgba(255, 255, 255, 0.055);
    border-radius: 12px;
    padding: 6px 12px;
    margin-bottom: 4px;
    transition: all 160ms ease;
}

.model-card-row:hover {
    background: rgba(255, 255, 255, 0.045);
    border-color: rgba(255, 255, 255, 0.13);
}

.model-card-selected {
    background: linear-gradient(135deg, rgba(99, 102, 241, 0.16) 0%, rgba(139, 92, 246, 0.08) 100%);
    border: 1.5px solid rgba(129, 140, 248, 0.85);
    box-shadow: 0 0 18px rgba(99, 102, 241, 0.24), inset 0 1px 0 rgba(255, 255, 255, 0.16);
}

.pill-size {
    background: rgba(255, 255, 255, 0.07);
    border-radius: 5px;
    padding: 2px 7px;
    font-size: 10px;
    font-weight: 600;
    color: rgba(255, 255, 255, 0.72);
}

.pill-recommended {
    background: rgba(16, 185, 129, 0.15);
    border: 1px solid rgba(16, 185, 129, 0.38);
    border-radius: 5px;
    padding: 2px 7px;
    font-size: 9.5px;
    font-weight: 700;
    color: #34d399;
}

.btn-download-icon {
    background: rgba(255, 255, 255, 0.05);
    border: 1px solid rgba(255, 255, 255, 0.12);
    border-radius: 8px;
    min-width: 30px;
    min-height: 30px;
    padding: 0;
    transition: all 150ms ease;
}

.btn-download-icon:hover {
    background: rgba(99, 102, 241, 0.3);
    border-color: #818cf8;
}

/* Right Inspector Column */
.inspector-panel {
    border-left: 1px solid rgba(255, 255, 255, 0.06);
    padding: 22px 20px 14px 20px;
    background: rgba(255, 255, 255, 0.012);
    min-width: 320px;
}

.inspector-card {
    background: linear-gradient(145deg, rgba(255, 255, 255, 0.035) 0%, rgba(255, 255, 255, 0.012) 100%);
    border: 1px solid rgba(255, 255, 255, 0.075);
    border-radius: 15px;
    padding: 14px 16px;
    margin-bottom: 12px;
    box-shadow: 0 4px 22px rgba(0, 0, 0, 0.28), inset 0 1px 0 rgba(255, 255, 255, 0.08);
}

.inspector-badge {
    background: linear-gradient(135deg, #6366f1, #8b5cf6);
    border-radius: 7px;
    min-width: 24px;
    min-height: 24px;
}

.kbd-chip {
    background: linear-gradient(180deg, rgba(255, 255, 255, 0.13) 0%, rgba(255, 255, 255, 0.04) 100%);
    border: 1px solid rgba(255, 255, 255, 0.16);
    border-bottom: 2px solid rgba(255, 255, 255, 0.26);
    border-radius: 7px;
    padding: 4px 10px;
    font-weight: 700;
    font-size: 11px;
    color: #ffffff;
    box-shadow: 0 2px 4px rgba(0, 0, 0, 0.4);
}

.mini-pbar {
    min-height: 6px;
    border-radius: 3px;
}

.mini-pbar trough {
    background: rgba(255, 255, 255, 0.08);
    border-radius: 3px;
    min-height: 6px;
}

.mini-pbar progress {
    background: linear-gradient(90deg, #6366f1, #a855f7);
    border-radius: 3px;
    min-height: 6px;
}

/* Footer Bottom Bar */
.footer-bar {
    border-top: 1px solid rgba(255, 255, 255, 0.06);
    background: rgba(14, 16, 23, 0.95);
    padding: 12px 24px;
}

.btn-quit {
    background: rgba(255, 255, 255, 0.06);
    border: 1px solid rgba(255, 255, 255, 0.12);
    color: rgba(255, 255, 255, 0.85);
    border-radius: 10px;
    padding: 6px 18px;
    font-weight: 600;
    font-size: 12px;
    transition: all 140ms ease;
}

.btn-quit:hover {
    background: rgba(255, 255, 255, 0.1);
    color: #ffffff;
}

.btn-save-purple {
    background: linear-gradient(135deg, #6366f1 0%, #8b5cf6 100%);
    border: 1px solid rgba(255, 255, 255, 0.22);
    border-radius: 10px;
    color: #ffffff;
    font-weight: 700;
    font-size: 12px;
    padding: 6px 26px;
    box-shadow: 0 4px 16px rgba(99, 102, 241, 0.45), inset 0 1px 0 rgba(255, 255, 255, 0.25);
    transition: all 140ms ease;
}

.btn-save-purple:hover {
    background: linear-gradient(135deg, #4f46e5 0%, #7c3aed 100%);
    box-shadow: 0 6px 22px rgba(99, 102, 241, 0.6);
}

.btn-edit-pill {
    background: rgba(255, 255, 255, 0.05);
    border: 1px solid rgba(255, 255, 255, 0.1);
    border-radius: 6px;
    color: rgba(255, 255, 255, 0.7);
    font-size: 11px;
    padding: 2px 7px;
    transition: all 120ms ease;
}

.btn-edit-pill:hover {
    background: rgba(255, 255, 255, 0.1);
    color: #ffffff;
}

.glass-entry {
    background: rgba(255, 255, 255, 0.05);
    border: 1px solid rgba(255, 255, 255, 0.12);
    border-radius: 8px;
    color: #ffffff;
    padding: 6px 10px;
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
    def __init__(self, gtk, on_quit=None) -> None:
        self.gtk = gtk
        self._on_quit = on_quit
        self._config = load_config()
        self._selected_model_name = self._config.asr.model
        self._hotkey_keys = tuple(self._config.hotkey.keys)
        self._dictionary_entries_list = list(self._config.dictionary)

        # UI Widget references
        self.window = None
        self._gtk_app = None
        self._status_label = gtk.Label(xalign=0)
        self._model_row_boxes: dict[str, object] = {}
        self._radio_indicators: dict[str, object] = {}
        self._download_buttons: dict[str, object] = {}
        self._download_pbar: dict[str, object] = {}

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

    def _make_svg(self, name: str):
        from gi.repository import Gdk, GLib

        svg_data = SVG_ICONS[name].encode("utf-8")
        bytes_data = GLib.Bytes.new(svg_data)
        texture = Gdk.Texture.new_from_bytes(bytes_data)
        return self.gtk.Image.new_from_paintable(texture)

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

        window = gtk.ApplicationWindow(application=app, title="WhisperFlow")
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

        # 1. Left Sidebar (Full height)
        sidebar = self._build_sidebar()
        sidebar.set_size_request(215, -1)
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

        # Wrap columns in a window handle so the user can drag window anywhere
        handle = gtk.WindowHandle()
        handle.set_hexpand(True)
        handle.set_vexpand(True)
        handle.set_child(columns_hbox)
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
        self._stop_capture()
        if self._on_quit:
            self._on_quit()
        if self.window:
            self.window.close()
        if self._gtk_app:
            self._gtk_app.quit()

    def _on_yellow_clicked(self, *_args) -> None:
        """Maximize or restore the application window."""
        if not self.window:
            return
        if self.window.is_maximized() or self.window.is_fullscreen():
            self.window.unmaximize()
            self._update_maximized_state(False)
        else:
            self.window.maximize()
            self._update_maximized_state(True)

    def _on_green_clicked(self, *_args) -> None:
        """Minimize the application window."""
        if not self.window:
            return
        self.window.minimize()

    def _on_window_close_request(self, *_args) -> bool:
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

        # Yellow = Maximize / Full Screen (Windows/macOS square icon on hover)
        y = gtk.Button()
        y.add_css_class("traffic-light")
        y.add_css_class("traffic-yellow")
        y.set_size_request(14, 14)
        y.set_valign(gtk.Align.CENTER)
        y.set_child(self._make_svg("traffic_max"))
        y.set_cursor_from_name("pointer")
        y.connect("clicked", self._on_yellow_clicked)

        # Green = Minimize (Minus icon on hover)
        g = gtk.Button()
        g.add_css_class("traffic-light")
        g.add_css_class("traffic-green")
        g.set_size_request(14, 14)
        g.set_valign(gtk.Align.CENTER)
        g.set_child(self._make_svg("traffic_min"))
        g.set_cursor_from_name("pointer")
        g.connect("clicked", self._on_green_clicked)

        traffic_box.append(r)
        traffic_box.append(y)
        traffic_box.append(g)
        vbox.append(traffic_box)

        # App Brand Header
        brand_box = gtk.Box(spacing=10, orientation=gtk.Orientation.HORIZONTAL)
        brand_box.set_margin_bottom(14)

        badge = gtk.Box(valign=gtk.Align.CENTER, halign=gtk.Align.CENTER)
        badge.add_css_class("app-brand-badge")
        cube_img = self._make_svg("cube")
        badge.append(cube_img)
        brand_box.append(badge)

        brand_text = gtk.Box(orientation=gtk.Orientation.VERTICAL, spacing=1)
        title = gtk.Label(label="WhisperFlow", xalign=0)
        title.set_markup("<span font_weight='700' font_size='11pt' color='#ffffff'>WhisperFlow</span>")
        sub = gtk.Label(label="Transcribe. Anywhere.", xalign=0)
        sub.set_markup("<span font_size='8.5pt' alpha='55%'>Transcribe. Anywhere.</span>")
        brand_text.append(title)
        brand_text.append(sub)
        brand_box.append(brand_text)
        vbox.append(brand_box)

        # Nav Items with crisp SVGs
        nav_items = [
            ("Model", "Choose transcription engine", "cube", True),
            ("Audio", "Microphone & detection", "microphone", False),
            ("Push-to-talk", "Keyboard shortcut", "keyboard", False),
            ("Dictionary", "Custom word corrections", "book", False),
            ("General", "App behavior", "cog", False),
            ("About", "Version & updates", "info", False),
        ]

        for item_name, subtitle, icon_key, is_active in nav_items:
            row = gtk.Box(orientation=gtk.Orientation.HORIZONTAL, spacing=10)
            row.add_css_class("nav-item")
            if is_active:
                row.add_css_class("nav-item-active")

            ic = self._make_svg(icon_key)
            ic.set_valign(gtk.Align.CENTER)
            row.append(ic)

            text_col = gtk.Box(orientation=gtk.Orientation.VERTICAL, spacing=1)
            t = gtk.Label(label=item_name, xalign=0)
            t.set_markup(f"<span font_weight='600' font_size='9pt' color='#ffffff'>{pango_esc(item_name)}</span>")
            s = gtk.Label(label=subtitle, xalign=0)
            s.set_markup(f"<span font_size='8pt' alpha='55%'>{pango_esc(subtitle)}</span>")
            text_col.append(t)
            text_col.append(s)
            row.append(text_col)

            vbox.append(row)

        vbox.append(gtk.Box(vexpand=True))  # Push storage card to bottom

        # Disk Storage Card at bottom of sidebar
        storage_card = gtk.Box(orientation=gtk.Orientation.VERTICAL, spacing=6)
        storage_card.add_css_class("storage-card")

        st_row = gtk.Box(orientation=gtk.Orientation.HORIZONTAL, spacing=6)
        drive_icon = self._make_svg("drive")
        st_row.append(drive_icon)

        total_mb = sum(size for _, size in installed_models()) / (1024 * 1024)
        total_gb = total_mb / 1024.0
        st_lbl = gtk.Label(xalign=0)
        st_lbl.set_markup(f"<span font_size='8pt' font_weight='600' alpha='75%'>{total_gb:.1f} GB / 12.0 GB</span>")
        st_row.append(st_lbl)
        storage_card.append(st_row)

        disk_pbar = gtk.ProgressBar()
        disk_pbar.add_css_class("mini-pbar")
        disk_pbar.set_fraction(min(1.0, max(0.05, total_gb / 12.0)))
        storage_card.append(disk_pbar)
        vbox.append(storage_card)

        return vbox

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

        learn_btn = gtk.Button(label="? Learn more")
        learn_btn.add_css_class("btn-learn")
        learn_btn.set_valign(gtk.Align.CENTER)
        learn_btn.connect("clicked", self._open_learn_modal)
        top_row.append(learn_btn)
        vbox.append(top_row)

        # Scrolled Model Card List
        scroller = gtk.ScrolledWindow(vexpand=True)
        scroller.set_hexpand(True)
        scroller.set_policy(gtk.PolicyType.NEVER, gtk.PolicyType.AUTOMATIC)
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

            # Hidden progress bar for downloading
            pbar = gtk.ProgressBar()
            pbar.add_css_class("mini-pbar")
            pbar.set_visible(False)
            self._download_pbar[name] = pbar
            card_info.append(pbar)

            row.append(card_info)

            # Right Side: Download status & action button
            if not present:
                dl_status = gtk.Label(label="Not downloaded")
                dl_status.set_markup("<span font_size='8.5pt' alpha='40%'>Not downloaded</span>")
                dl_status.set_valign(gtk.Align.CENTER)
                row.append(dl_status)

                dl_btn = gtk.Button()
                dl_btn.set_child(self._make_svg("download"))
                dl_btn.add_css_class("btn-download-icon")
                dl_btn.set_valign(gtk.Align.CENTER)
                dl_btn.connect("clicked", self._start_download, name)
                self._download_buttons[name] = dl_btn
                row.append(dl_btn)

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
        badge = gtk.Box(valign=gtk.Align.CENTER, halign=gtk.Align.CENTER)
        badge.add_css_class("inspector-badge")
        badge.append(self._make_svg("cube_badge"))
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
        self._inspector_ram = self._make_spec_row(grid, "Approx. RAM", "~1.2 GB RAM")
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

        edit_dict_btn = gtk.Button(label="✏ Edit")
        edit_dict_btn.add_css_class("btn-edit-pill")
        edit_dict_btn.connect("clicked", self._open_dictionary_modal)
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
        if self._hotkey_box is None:
            return
        gtk = self.gtk
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
        self._status_label.set_markup("<span font_size='8.5pt' alpha='65%'>Closing this window leaves WhisperFlow in the dock.</span>")
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
        btn = self._download_buttons.get(name)
        if btn is not None:
            btn.set_sensitive(False)
        pbar = self._download_pbar.get(name)
        if pbar is not None:
            pbar.set_visible(True)
            pbar.set_fraction(0.0)

        self._status_label.set_markup(f"<span font_size='8.5pt' alpha='80%'>Downloading {pango_esc(name)}...</span>")
        last = 0.0

        def progress(written: int, total: int) -> None:
            nonlocal last
            now = time.monotonic()
            if written != total and now - last < 0.2:
                return
            last = now
            self._idle(self._download_progress, name, written, total)

        def work() -> None:
            try:
                download_model(name, progress=progress)
            except Exception as exc:
                self._idle(self._download_failed, name, str(exc))
                return
            self._idle(self._download_done, name)

        threading.Thread(target=work, name=f"whisperflow-download-{name}", daemon=True).start()

    def _download_progress(self, name: str, written: int, total: int) -> bool:
        pbar = self._download_pbar.get(name)
        if total:
            pct = written / total
            if pbar is not None:
                pbar.set_fraction(pct)
            self._status_label.set_markup(f"<span font_size='8.5pt' alpha='80%'>Downloading {pango_esc(name)}... {int(pct * 100)}%</span>")
        return False

    def _download_done(self, name: str) -> bool:
        btn = self._download_buttons.get(name)
        if btn is not None:
            btn.set_visible(False)
        pbar = self._download_pbar.get(name)
        if pbar is not None:
            pbar.set_visible(False)

        self._select_model(name)
        self._status_label.set_markup(f"<span font_size='8.5pt' font_weight='700' color='#34d399'>● Downloaded {pango_esc(name)} successfully! Click Save to apply.</span>")
        return False

    def _download_failed(self, name: str, message: str) -> bool:
        btn = self._download_buttons.get(name)
        if btn is not None:
            btn.set_sensitive(True)
        pbar = self._download_pbar.get(name)
        if pbar is not None:
            pbar.set_visible(False)
        self._status_label.set_markup(f"<span font_size='8.5pt' color='#f87171'>Could not download {pango_esc(name)}: {pango_esc(message)}</span>")
        return False

    # -------------------------------------------------------------------------
    # Shortcut Capture Handling
    # -------------------------------------------------------------------------
    def _stop_capture(self, *_args) -> bool:
        self._capture_stop.set()
        return False

    def _start_capture(self, *_args) -> None:
        self._capture_stop.set()
        self._capture_stop = threading.Event()
        stop = self._capture_stop
        self._status_label.set_markup("<span font_size='8.5pt' color='#a855f7' font_weight='bold'>● Listening: Press the shortcut keys on your keyboard...</span>")

        def work() -> None:
            try:
                keys = capture_chord(stop)
            except Exception as exc:
                self._idle(self._shortcut_failed, str(exc))
                return
            self._idle(self._shortcut_done, keys)

        threading.Thread(target=work, name="whisperflow-capture", daemon=True).start()

    def _shortcut_done(self, keys: tuple[str, ...]) -> bool:
        if not keys:
            self._status_label.set_markup("<span font_size='8.5pt' alpha='65%'>No shortcut captured.</span>")
            return False
        self._hotkey_keys = keys
        self._render_hotkey_chips()
        self._status_label.set_markup(f"<span font_size='8.5pt' font_weight='700' color='#34d399'>● Shortcut updated to {pango_esc(format_chord(keys))}. Click Save to apply.</span>")
        return False

    def _shortcut_failed(self, message: str) -> bool:
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
        save_user_settings(path, model, self._dictionary_entries_list, self._hotkey_keys)
        self._status_label.set_markup("<span font_size='8.5pt' font_weight='700' color='#34d399'>● Settings saved successfully!</span>")

    def _idle(self, callback, *args) -> None:
        from gi.repository import GLib

        GLib.idle_add(callback, *args)


if __name__ == "__main__":
    raise SystemExit(main())
