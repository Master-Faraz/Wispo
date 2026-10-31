"""Floating Liquid-Glass Pill HUD overlay.

A frameless, translucent floating pill displayed centered above the dock
during speech recording and processing.

Key Design Guarantees:
- True 32-bit ARGB TrueColor transparency: ZERO square corners, only the rounded pill appears.
- Centered directly above the desktop dock.
- Liquid-glass translucent material with specular reflection, sheen, and drop shadow.
- Glowing breathing listening dot on the left.
- Real-time reactive audio waveform bars in the middle.
- Minimal 'Listening' label.
- 'Processing' with rippling wave-animated undulating dots during transcription.
- Smooth ease-in / ease-out expand and contract animations when appearing/closing.
- Zero interference with focused applications and text injection.
- Complete click-through overlay (input region empty).
"""

from __future__ import annotations

import math
import sys
import threading
import time
from typing import Optional

try:
    import cairo
except ModuleNotFoundError:
    sys.path.append("/usr/lib/python3/dist-packages")
    import cairo

try:
    import gi
except ModuleNotFoundError:
    sys.path.append("/usr/lib/python3/dist-packages")
    import gi

gi.require_version("Gtk", "4.0")
gi.require_version("Gdk", "4.0")
from gi.repository import Gdk, GLib, Gtk


class FloatingPillHUD:
    """Floating liquid-glass pill HUD displayed above the desktop dock."""

    NUM_BARS = 23
    BAR_WIDTH = 2.4
    BAR_SPACING = 4.8
    CANVAS_W = 340
    CANVAS_H = 76

    def __init__(self, gtk=None) -> None:
        self.gtk = gtk or Gtk
        self._lock = threading.Lock()
        self._state = "idle"  # idle, listening, processing, closing
        self._anim_progress = 0.0  # 0.0 to 1.0
        self._target_anim = 0.0
        self._current_width = 264.0
        self._target_width = 264.0
        self._pill_height = 50.0

        # Audio reactivity
        self._target_level = 0.0
        self._smoothed_level = 0.0
        self._bar_heights = [4.0] * self.NUM_BARS

        # Time tracking for smooth animations
        self._pulse_time = 0.0
        self._wave_time = 0.0
        self._anim_timer_id: Optional[int] = None

        # GTK4 Window and Widgets
        self._window: Optional[Gtk.Window] = None
        self._box: Optional[Gtk.Box] = None
        self._drawing_area: Optional[Gtk.DrawingArea] = None

        self._init_window()

    def _init_window(self) -> None:
        # Load transparent CSS provider at USER priority (overrides all theme defaults)
        css_provider = Gtk.CssProvider()
        css_str = """
        window.floating-pill-window,
        window.floating-pill-window.background,
        window.floating-pill-window decoration,
        window.floating-pill-window.csd,
        window.floating-pill-window.solid-csd,
        window.floating-pill-window contents {
            background-color: transparent;
            background-image: none;
            box-shadow: none;
            border: none;
            outline: none;
            padding: 0;
            margin: 0;
        }
        """
        if hasattr(css_provider, "load_from_string"):
            css_provider.load_from_string(css_str)
        else:
            css_provider.load_from_data(css_str.encode("utf-8"))
        display = Gdk.Display.get_default()
        if display:
            Gtk.StyleContext.add_provider_for_display(
                display,
                css_provider,
                Gtk.STYLE_PROVIDER_PRIORITY_USER,
            )

        # Standalone toplevel window (never steals focus, click-through)
        self._window = Gtk.Window()
        self._window.add_css_class("floating-pill-window")
        self._window.set_decorated(False)
        self._window.set_focusable(False)
        self._window.set_can_focus(False)
        self._window.set_focus_on_click(False)
        self._window.set_title("Wispo HUD")

        # Get screen height for vertical positioning above the dock
        monitor_h = 900
        if display:
            monitors = display.get_monitors()
            if monitors and monitors.get_n_items() > 0:
                geom = monitors.get_item(0).get_geometry()
                monitor_h = geom.height

        # Set window size to CANVAS_W x monitor_h:
        # 1. CANVAS_W (340px) is drastically narrower than the full monitor width (1440/1920px),
        #    which completely eliminates Mutter's Direct Scanout (direct scanout strictly requires
        #    100% monitor resolution coverage).
        # 2. Setting height to monitor_h allows GNOME Mutter to place the window anchored to the
        #    monitor, so our bottom-aligned box positions the pill centered directly above the dock.
        self._window.set_default_size(self.CANVAS_W, monitor_h)
        self._window.set_resizable(False)

        # Container box aligned at the bottom center directly above the dock
        self._box = Gtk.Box(orientation=Gtk.Orientation.VERTICAL)
        self._box.set_valign(Gtk.Align.END)
        self._box.set_halign(Gtk.Align.CENTER)
        self._box.set_margin_bottom(64)

        # Drawing Area for the liquid-glass pill
        self._drawing_area = Gtk.DrawingArea()
        self._drawing_area.set_content_width(self.CANVAS_W)
        self._drawing_area.set_content_height(self.CANVAS_H)
        self._drawing_area.set_draw_func(self._on_draw)

        self._box.append(self._drawing_area)
        self._window.set_child(self._box)

        # Make window click-through via empty input region on realize and map
        self._window.connect("realize", self._on_realize)
        self._window.connect("map", self._on_map)
        self._window.realize()

    def _on_realize(self, win: Gtk.Window) -> None:
        self._ensure_input_region()

    def _on_map(self, win: Gtk.Window) -> None:
        self._ensure_input_region()

    def _ensure_input_region(self) -> None:
        if self._window is not None:
            surface = self._window.get_surface()
            if surface is not None:
                surface.set_input_region(cairo.Region())

    # --- Thread-Safe Public API ---

    def start_listening(self) -> None:
        """Call when the user presses and holds the hotkey."""
        with self._lock:
            self._state = "listening"
            self._target_anim = 1.0
            self._target_width = 264.0

        def _apply() -> bool:
            if self._window is not None:
                self._window.set_visible(True)
                self._ensure_input_region()
            self._start_timer()
            return False

        GLib.idle_add(_apply)

    def set_audio_level(self, rms: float) -> None:
        """Update live audio level (RMS float ~0.0 to 1.0)."""
        normalized = min(1.0, max(0.0, (rms - 0.008) / 0.16))
        boosted = math.pow(normalized, 0.75)
        with self._lock:
            self._target_level = boosted

    def start_processing(self) -> None:
        """Call when hotkey is released and Whisper is transcribing."""
        with self._lock:
            self._state = "processing"
            self._target_width = 216.0
            self._wave_time = 0.0

        def _apply() -> bool:
            self._start_timer()
            return False

        GLib.idle_add(_apply)

    def finish(self, wait_hide: bool = True) -> None:
        """Call when transcription completes and text is ready to inject.
        
        Runs a snappy exit animation, hides the window to ensure the compositor
        restores focus to the target application, and completes.
        """
        with self._lock:
            self._state = "closing"
            self._target_anim = 0.0

        hide_event = threading.Event()

        def _apply() -> bool:
            self._start_timer()

            # Schedule hiding after snappy 100ms contract animation
            def _hide_window() -> bool:
                if self._window is not None:
                    self._window.set_visible(False)
                with self._lock:
                    self._state = "idle"
                    self._anim_progress = 0.0
                self._stop_timer()
                hide_event.set()
                return False

            GLib.timeout_add(100, _hide_window)
            return False

        GLib.idle_add(_apply)
        if wait_hide:
            hide_event.wait(timeout=0.25)
            # Brief yield to let Mutter restore focus to the target window
            time.sleep(0.02)

    def close(self) -> None:
        """Cleanup overlay on app exit."""
        self._stop_timer()
        if self._window is not None:
            self._window.close()
            self._window = None

    # --- Animation Timer ---

    def _start_timer(self) -> None:
        if self._anim_timer_id is None:
            self._anim_timer_id = GLib.timeout_add(16, self._on_tick)

    def _stop_timer(self) -> None:
        if self._anim_timer_id is not None:
            timer_id = self._anim_timer_id
            self._anim_timer_id = None
            try:
                GLib.source_remove(timer_id)
            except Exception:
                pass

    def _on_tick(self) -> bool:
        dt = 0.016
        with self._lock:
            target_anim = self._target_anim
            target_w = self._target_width
            target_lvl = self._target_level
            state = self._state

        # Animate progress
        if target_anim > self._anim_progress:
            self._anim_progress = min(target_anim, self._anim_progress + dt * 6.5)
        elif target_anim < self._anim_progress:
            self._anim_progress = max(target_anim, self._anim_progress - dt * 7.5)

        # Smooth pill width interpolation between states
        self._current_width += (target_w - self._current_width) * 0.24

        # Update times
        self._pulse_time += dt
        self._wave_time += dt

        # Smooth audio level
        self._smoothed_level += (target_lvl - self._smoothed_level) * 0.38

        # Update reactive waveform bars
        center_idx = (self.NUM_BARS - 1) / 2.0
        for i in range(self.NUM_BARS):
            dist = abs(i - center_idx) / center_idx
            envelope = math.exp(-2.4 * (dist ** 2))
            var = 0.72 + 0.28 * math.sin(i * 1.6 + self._pulse_time * 2.5)
            target_h = 4.0 + (envelope * var * self._smoothed_level * 30.0)
            self._bar_heights[i] += (target_h - self._bar_heights[i]) * 0.40

        # Redraw
        if self._drawing_area is not None:
            self._drawing_area.queue_draw()

        # Stop timer when idle and fully collapsed
        if state == "idle" and self._anim_progress <= 0.001:
            self._anim_timer_id = None
            return False

        return True

    # --- Cairo Rendering ---

    def _on_draw(self, drawing_area: Gtk.DrawingArea, cr: cairo.Context, width: int, height: int) -> None:
        # Clear entire canvas to 100% transparent: ZERO square corners!
        cr.set_operator(cairo.OPERATOR_SOURCE)
        cr.set_source_rgba(0.0, 0.0, 0.0, 0.0)
        cr.paint()
        cr.set_operator(cairo.OPERATOR_OVER)

        if self._anim_progress <= 0.0001:
            return

        with self._lock:
            state = self._state
            pulse_t = self._pulse_time
            wave_t = self._wave_time
            bar_heights = list(self._bar_heights)
            pill_w = self._current_width

        # Ease-out cubic for smooth entrance/exit
        ease = 1.0 - math.pow(1.0 - self._anim_progress, 3)
        alpha = max(0.0, min(1.0, ease))
        scale = 0.88 + 0.12 * ease
        dy = (1.0 - ease) * 16.0

        pill_h = self._pill_height
        pill_x = (width - pill_w) / 2.0
        pill_y = (height - pill_h) / 2.0 + dy
        r = pill_h / 2.0

        cr.save()

        # Scaling around pill center
        cx = width / 2.0
        cy = pill_y + pill_h / 2.0
        cr.translate(cx, cy)
        cr.scale(scale, scale)
        cr.translate(-cx, -cy)

        def path_pill(x: float, y: float, w: float, h: float, radius: float) -> None:
            cr.new_sub_path()
            cr.arc(x + w - radius, y + radius, radius, -math.pi / 2, math.pi / 2)
            cr.arc(x + radius, y + radius, radius, math.pi / 2, 3 * math.pi / 2)
            cr.close_path()

        # 1. Soft Ambient Outer Drop Shadow
        for i in range(4):
            offset = (i + 1) * 1.5
            s_alpha = (0.07 - i * 0.014) * alpha
            if s_alpha > 0:
                cr.set_source_rgba(0.0, 0.0, 0.0, s_alpha)
                path_pill(pill_x, pill_y + offset, pill_w, pill_h, r)
                cr.fill()

        # 2. Liquid-Glass Translucent Body
        path_pill(pill_x, pill_y, pill_w, pill_h, r)
        glass_pat = cairo.LinearGradient(pill_x, pill_y, pill_x, pill_y + pill_h)
        glass_pat.add_color_stop_rgba(0.0, 0.11, 0.12, 0.15, 0.84 * alpha)
        glass_pat.add_color_stop_rgba(1.0, 0.05, 0.06, 0.08, 0.92 * alpha)
        cr.set_source(glass_pat)
        cr.fill_preserve()

        # 3. Specular Rim / Border Reflection
        border_pat = cairo.LinearGradient(pill_x, pill_y, pill_x, pill_y + pill_h)
        border_pat.add_color_stop_rgba(0.0, 1.0, 1.0, 1.0, 0.35 * alpha)
        border_pat.add_color_stop_rgba(0.4, 1.0, 1.0, 1.0, 0.16 * alpha)
        border_pat.add_color_stop_rgba(1.0, 1.0, 1.0, 1.0, 0.06 * alpha)
        cr.set_source(border_pat)
        cr.set_line_width(1.2)
        cr.stroke()

        # 4. Top Glass Sheen Reflection
        cr.save()
        path_pill(pill_x, pill_y, pill_w, pill_h, r)
        cr.clip()
        sheen_pat = cairo.LinearGradient(pill_x, pill_y, pill_x, pill_y + pill_h * 0.5)
        sheen_pat.add_color_stop_rgba(0.0, 1.0, 1.0, 1.0, 0.09 * alpha)
        sheen_pat.add_color_stop_rgba(1.0, 1.0, 1.0, 1.0, 0.00)
        cr.rectangle(pill_x, pill_y, pill_w, pill_h * 0.5)
        cr.set_source(sheen_pat)
        cr.fill()
        cr.restore()

        # --- Content by State ---
        if state in ("listening", "closing"):
            # Glowing listening dot on the left
            dot_x = pill_x + 22.0
            dot_y = pill_y + r
            pulse = 0.5 + 0.5 * math.sin(pulse_t * 3.5)
            glow_rad = 8.0 + 2.0 * pulse
            glow_alpha = (0.55 + 0.25 * pulse) * alpha

            # Radial halo glow
            radial = cairo.RadialGradient(dot_x, dot_y, 1.0, dot_x, dot_y, glow_rad)
            radial.add_color_stop_rgba(0.0, 0.22, 0.74, 0.98, glow_alpha)
            radial.add_color_stop_rgba(1.0, 0.22, 0.74, 0.98, 0.0)
            cr.set_source(radial)
            cr.arc(dot_x, dot_y, glow_rad, 0, 2 * math.pi)
            cr.fill()

            # Solid glowing core
            cr.set_source_rgba(0.45, 0.90, 1.0, 0.96 * alpha)
            cr.arc(dot_x, dot_y, 3.4, 0, 2 * math.pi)
            cr.fill()

            # Reactive Waveform in the Middle
            num_bars = self.NUM_BARS
            bar_w = self.BAR_WIDTH
            spacing = self.BAR_SPACING
            total_wf_w = num_bars * spacing
            wf_start_x = pill_x + 38.0 + (pill_w - 38.0 - 82.0 - total_wf_w) / 2.0
            center_idx = (num_bars - 1) / 2.0

            cr.set_line_cap(cairo.LINE_CAP_ROUND)
            cr.set_line_width(bar_w)

            for i in range(num_bars):
                dist = abs(i - center_idx) / center_idx
                env = math.exp(-2.4 * (dist ** 2))
                h = bar_heights[i] if i < len(bar_heights) else 4.0
                bx = wf_start_x + i * spacing
                by1 = pill_y + r - h / 2.0
                by2 = pill_y + r + h / 2.0

                b_alpha = (0.50 + 0.50 * env) * alpha
                cr.set_source_rgba(1.0, 1.0, 1.0, b_alpha)
                cr.move_to(bx, by1)
                cr.line_to(bx, by2)
                cr.stroke()

            # Minimal 'Listening' text on the right
            cr.select_font_face("Ubuntu", cairo.FONT_SLANT_NORMAL, cairo.FONT_WEIGHT_NORMAL)
            cr.set_font_size(11.5)
            cr.set_source_rgba(1.0, 1.0, 1.0, 0.75 * alpha)
            ext = cr.text_extents("Listening")
            cr.move_to(pill_x + pill_w - ext.width - 20.0, pill_y + r + ext.height / 2.0 - 1.0)
            cr.show_text("Listening")

        elif state == "processing":
            # Balanced centered group: [dot] [Processing] [wave dots]
            cr.select_font_face("Ubuntu", cairo.FONT_SLANT_NORMAL, cairo.FONT_WEIGHT_NORMAL)
            cr.set_font_size(12.5)
            ext = cr.text_extents("Processing")

            dot_radius = 3.2
            dot_halo_radius = 8.0
            text_w = ext.width
            dots_spacing = 6.8
            dots_w = 3 * dots_spacing
            gap1 = 10.0
            gap2 = 8.0

            total_content_w = (dot_radius * 2) + gap1 + text_w + gap2 + dots_w
            start_x = pill_x + (pill_w - total_content_w) / 2.0

            # Glowing purple/accent dot
            dot_x = start_x + dot_radius
            dot_y = pill_y + r

            pulse = 0.5 + 0.5 * math.sin(pulse_t * 4.0)
            radial = cairo.RadialGradient(dot_x, dot_y, 1.0, dot_x, dot_y, dot_halo_radius)
            radial.add_color_stop_rgba(0.0, 0.65, 0.45, 0.98, (0.6 + 0.3 * pulse) * alpha)
            radial.add_color_stop_rgba(1.0, 0.65, 0.45, 0.98, 0.0)
            cr.set_source(radial)
            cr.arc(dot_x, dot_y, dot_halo_radius, 0, 2 * math.pi)
            cr.fill()

            cr.set_source_rgba(0.78, 0.65, 1.0, 0.95 * alpha)
            cr.arc(dot_x, dot_y, dot_radius, 0, 2 * math.pi)
            cr.fill()

            # 'Processing' text
            text_x = dot_x + dot_radius + gap1
            cr.set_source_rgba(1.0, 1.0, 1.0, 0.88 * alpha)
            cr.move_to(text_x, pill_y + r + ext.height / 2.0 - 1.0)
            cr.show_text("Processing")

            # Wave-animated dots rippling in a smooth sine wave
            wave_dots_start_x = text_x + text_w + gap2
            for di in range(3):
                wave_phase = wave_t * 4.8 + di * 0.95
                disp_y = math.sin(wave_phase) * 4.0
                wave_alpha = (0.40 + 0.60 * (0.5 + 0.5 * math.sin(wave_phase))) * alpha
                cr.set_source_rgba(1.0, 1.0, 1.0, wave_alpha)
                cr.arc(wave_dots_start_x + di * dots_spacing, pill_y + r + disp_y, 2.2, 0, 2 * math.pi)
                cr.fill()

        cr.restore()
