---
description: Designs and implements polished, Apple-inspired Liquid
  Glass interfaces with restrained translucency, depth, hierarchy,
  adaptive layouts, accessible contrast, and high-quality interaction
  details. Use when redesigning desktop or mobile UI, especially
  settings windows, navigation, toolbars, popovers, and control
  surfaces.
name: liquid-glass-ui
---

# Liquid Glass UI Design Skill

## Mission

Design interfaces that feel considered, calm, tactile, precise, and
premium---taking inspiration from Apple's **Liquid Glass** design
language without blindly copying screenshots or turning every surface
into a translucent purple card.

The goal is a **working product UI**, not a decorative mockup. Preserve
the app's purpose, existing capabilities, platform conventions, and
technical architecture while improving hierarchy, spacing, readability,
interactions, and visual craft.

Use this skill for WhisperFlow and other application interfaces. When
the task is WhisperFlow, treat it as a focused voice-transcription
utility: model choice, microphone/audio settings, push-to-talk,
dictionary corrections, app behavior, and clear save/quit actions must
remain easy to find.

## Core principle: Liquid Glass is a material and a hierarchy---not a theme

Apple describes Liquid Glass as a dynamic material for the **functional
layer**---navigation, toolbars, controls, and transient
surfaces---distinct from the content layer. It responds to and borrows
color from what's behind it. It is not simply a purple gradient, a blur
filter, or a glass card applied to every component.

Follow these rules:

1.  **Separate content from controls.** Use a calm, legible base surface
    for content. Reserve glass-like treatment for navigation, toolbars,
    floating controls, popovers, and selected/high-priority actions.
2.  **Let context show through, subtly.** A material should reveal the
    real content or environment behind it. Never add a loud wallpaper
    solely to make every panel look glassy.
3.  **Use translucency selectively.** A little well-executed material is
    more convincing than glass everywhere. Repeated translucent cards
    inside a translucent window flatten hierarchy and make the interface
    look like generic "glassmorphism."
4.  **Prioritize legibility over spectacle.** If text, model sizes,
    statuses, or controls become hard to read, increase opacity, reduce
    blur, add a scrim, or use a standard solid material.
5.  **Make it feel responsive.** Focus, hover, press, selection, drag,
    and transitions should feel deliberate and lightweight. Effects must
    communicate state, not just decorate.
6.  **Design for the actual platform.** Native Apple frameworks can
    provide real Liquid Glass behavior on supported Apple OS versions.
    On Linux/Windows or a cross-platform toolkit, reproduce the
    principles using that platform's available blur/material APIs and
    sensible fallbacks. Do not claim a CSS blur is Apple's native
    implementation.

## Visual language

### Material layers

Use a small, consistent material system:

-   **Base/content layer:** mostly opaque, low-noise surface. Model
    lists, dictionary rows, and dense settings content belong here.
-   **Glass/control layer:** translucent, softly blurred surface for the
    sidebar/navigation, toolbar, floating shortcut controls, popovers,
    and other controls that need to sit above content.
-   **Emphasis layer:** one primary action and the current selection.
    Use restrained tinting and a fine luminous edge; don't glow every
    card.
-   **Scrim:** a subtle dark/light overlay behind glass only when the
    background would compromise readability.

A glass surface should have optical depth, not a stack of obvious CSS
tricks. Use: - background blur appropriate to the toolkit; - moderate
translucency that adapts to light and dark backgrounds; - a subtle, fine
border with low contrast; - a restrained top/edge highlight where
appropriate; - a soft shadow to separate layers; - rounded corners
consistent with neighboring controls.

Avoid thick white borders, exaggerated bevels, heavy neon glows, strong
inner shadows, chrome-like reflections, or large gradient strokes on
every component.

### Recommended starting tokens

Treat these as starting points, not hard requirements. Adapt them to the
current toolkit and test against real content.

``` css
:root {
  color-scheme: dark;
  --surface-canvas: #171922;
  --surface-content: #20232d;
  --surface-elevated: #282b37;
  --text-primary: #f5f5fa;
  --text-secondary: #b8bdcc;
  --text-tertiary: #858da2;
  --accent: #8b75ff;
  --accent-hover: #a091ff;
  --accent-pressed: #7358e8;
  --border-subtle: rgba(255, 255, 255, 0.09);
  --border-glass: rgba(255, 255, 255, 0.17);
  --glass-tint: rgba(36, 39, 55, 0.68);
  --focus-ring: rgba(157, 137, 255, 0.92);
  --radius-window: 22px;
  --radius-panel: 16px;
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
```

Suggested effects for web-based implementations: - Navigation/control
glass: `backdrop-filter: blur(18px) saturate(140%)`, with a translucent
background. - Dense information panels: use a more opaque standard
surface; do not blur each row. - Hairline edge:
`1px solid var(--border-glass)`. - Shadows: broad and low-opacity, not
black outlines. - If `backdrop-filter` is unsupported or disabled, the
interface must still look intentional with an opaque fallback.

Do not force dark mode if the existing application supports light mode.
Create equivalent light and dark material values, and honor the user's
system appearance when possible.

### Color and lighting

-   Use neutral surfaces as the foundation; let the background subtly
    influence the glass.
-   Choose one restrained accent color for selection and primary
    actions. WhisperFlow may use violet, but it should not tint every
    panel.
-   Gradients are optional and should be nearly imperceptible on
    functional surfaces. Avoid purple-to-blue gradients on every card.
-   Keep disabled, secondary, and tertiary text visibly distinct but
    readable.
-   Never use color alone to communicate selected, unavailable,
    downloading, listening, or error states.

### Shape, spacing, and typography

-   Establish a consistent radius scale. Larger containers may be more
    rounded than controls, but avoid unrelated radii on every component.
-   Use a deliberate spacing scale (4/8/12/16/24/32 px). Align edges and
    baselines.
-   Prefer fewer, better-separated groups over many nested cards.
-   Use the platform's system font where practical. Use clear
    size/weight hierarchy instead of making everything bold.
-   Use short labels and concise helper text. Align sizes, status
    labels, and actions consistently.
-   Icons should be simple, consistent, and preferably use the
    platform's native symbol set. Avoid emoji as interface icons.

## Layout rules

1.  Establish one clear page title and a short description.
2.  Put primary navigation in a stable sidebar or toolbar, depending on
    the platform and window size.
3.  Keep the main task in the largest, clearest content region.
4.  Place contextual details in a secondary inspector or disclosure
    area; don't let it compete with the main task.
5.  Group related settings using spacing, headings, and subtle
    separators. Do not wrap every individual row in a separate card.
6.  Keep primary and destructive actions distinct. The primary action
    should be visually obvious, while Quit/Cancel remains secondary.
7.  Make content scroll independently where appropriate; keep navigation
    and essential actions stable.
8.  Adapt to narrow windows: collapse the inspector, move details into a
    disclosure, or switch to a single-column layout rather than
    compressing every column.
9.  Preserve predictable placement and labels for familiar platform
    actions.
10. Never sacrifice functional density or accessibility just to match a
    reference image.

## WhisperFlow-specific direction

### Recommended settings-window structure

-   **Navigation:** compact brand mark and WhisperFlow name; sections
    such as Model, Audio, Push-to-talk, Dictionary, General, and About.
    Use a restrained selected state, not a glowing neon pill for every
    item.
-   **Main content:** title, short explanation, and the primary settings
    for the selected section.
-   **Contextual inspector (optional):** selected model details,
    download status, shortcut summary, or concise contextual help. At
    smaller widths, hide/collapse this region gracefully.
-   **Footer/action area:** storage/model footprint if useful; a concise
    note that closing the window leaves the app running in the
    dock/tray; secondary Quit and primary Save actions.

### Model selector

-   Present each model as a clean, scan-friendly row with:
    -   radio/selection control;
    -   model name;
    -   download size;
    -   optional concise descriptor;
    -   a clear badge for Recommended or status;
    -   Download action only when relevant.
-   Use a subtle row hover state and a clearly identifiable selected
    state (accent border/fill plus radio indicator). Avoid making every
    row a large raised glass card.
-   Keep unavailable/not-downloaded models understandable. Disabled text
    must remain legible; place Download beside the relevant model.
-   Put detailed parameters, RAM estimates, language support, speed, and
    accuracy in a contextual inspector or expandable details area. Avoid
    invented metrics: only display information the application actually
    knows.
-   Don't label one model "Recommended" while visually selecting another
    without a clear reason. The UI must reflect the actual active
    configuration.

### Push-to-talk

-   Show the current shortcut as compact keycaps or a readable shortcut
    string.
-   Keep Change Shortcut/Edit as a small, obvious secondary action.
-   Explain the behavior in one line. Provide clear
    recording/listening/processing/error states if the app exposes them.
-   Don't invent a shortcut or imply that a key binding is configured if
    it isn't.

### Dictionary

-   Use a compact table/list with aligned "spoken phrase" and
    "replacement" columns.
-   Make enabled state easy to scan, and provide row-level editing that
    doesn't make the entire list look like a form of giant input boxes.
-   Keep Add and Delete near the list. Confirm destructive bulk actions
    where needed.
-   Use clear empty states and validation feedback.

### Save, close, and quit

-   Clearly distinguish Save from Quit.
-   Preserve the existing behavior that closing the settings window
    leaves WhisperFlow running if that is how the app works.
-   Give feedback when changes are saved, when unsaved changes exist,
    and when an action fails.
-   Do not silently discard settings.

## Interaction and motion

-   Use short, calm transitions---typically around 120--220 ms for
    hover/press and 180--280 ms for view changes, where supported.
-   Prefer gentle opacity, material, and small position/scale changes.
    Avoid springy motion that makes a utility app feel like a toy.
-   The selected state should update immediately and consistently.
-   Hover should be visible but quieter than selection. Press should
    feel tactile without a large bounce.
-   Respect reduced-motion settings and avoid motion that is essential
    to understanding status.
-   Do not animate every row or add perpetual shimmer. Use progress
    indicators only for real work.

## Accessibility and resilience

-   Maintain strong text contrast on translucent surfaces. Re-test when
    the background changes.
-   Ensure visible keyboard focus, logical tab order, accessible names,
    and usable hit targets.
-   Do not rely on hover alone; keyboard and touch/trackpad interactions
    must work.
-   Support increased contrast, reduced transparency, and reduced motion
    where the platform allows.
-   When transparency is reduced or unavailable, switch to a more opaque
    surface without breaking the design.
-   Ensure long model names, localization, scaling, and narrow window
    sizes do not overlap or clip.
-   Keep status text alongside color indicators (e.g. "Downloading,"
    "Not downloaded," "Listening").

## Anti-patterns: do not do these

-   Do not interpret "Liquid Glass" as "put everything inside a
    transparent purple rounded rectangle."
-   Do not make the entire app background a high-contrast wallpaper just
    to show blur.
-   Do not place a glass card inside another glass card inside another
    glass card.
-   Do not add thick glowing outlines to every component.
-   Do not use low-contrast grey text on dark translucent surfaces.
-   Do not invent model capabilities, benchmark results, RAM estimates,
    or recommended states.
-   Do not redesign only a screenshot/mockup while leaving the real app
    unchanged.
-   Do not rewrite the application framework or change app behavior
    unless necessary and explicitly justified.
-   Do not add a design system dependency without checking the existing
    stack and bundle/runtime constraints.
-   Do not remove features just to achieve a cleaner screenshot.

## Required implementation workflow

When asked to redesign an existing application:

1.  **Inspect first.** Identify the UI framework, entry points,
    component structure, existing theme/tokens, state management, and
    how settings are saved. Read the project's instructions before
    editing.
2.  **Map functionality.** List every existing control and behavior.
    Preserve all features unless the user explicitly approves a change.
3.  **Diagnose the current UI.** Identify hierarchy, alignment, spacing,
    contrast, density, responsive layout, and interaction problems. Use
    the user's screenshot as context, not as a pixel-perfect target.
4.  **Define the design system.** Establish tokens for surfaces, text,
    accent, borders, radii, spacing, typography, and motion before
    styling individual controls.
5.  **Implement the real UI.** Refactor reusable components where
    useful. Keep business logic and presentation concerns separated.
6.  **Use materials intentionally.** Put glass on navigation and
    floating controls; keep dense settings content calmer and more
    opaque.
7.  **Handle states.** Implement selected, hover, focus, pressed,
    disabled, loading/downloading, success, error, empty, and
    unsaved-change states as applicable.
8.  **Test the actual application.** Run the existing formatter, type
    checker, tests, and build where available. Launch the UI if the
    environment supports it.
9.  **Review at multiple sizes.** Check the default window, a smaller
    window, long labels, and light/dark appearance if supported.
10. **Compare against the quality bar.** Look for alignment drift,
    inconsistent radii, excessive cards, overused blur, weak contrast,
    clipped text, placeholder icons, and awkward whitespace. Fix issues
    before declaring completion.
11. **Report truthfully.** Summarize what changed, what was tested, and
    any limitations. Never claim visual verification if the app wasn't
    launched.

## Definition of done

The result is complete only when: - the visual hierarchy is immediately
clear; - the interface looks cohesive rather than assembled from
unrelated cards; - glass effects are restrained, context-aware, and do
not obscure content; - model rows, dictionary entries, and shortcut
settings are easy to scan; - primary actions and current state are
unmistakable; - keyboard navigation and contrast are usable; - layouts
adapt without clipping; - existing functionality is preserved; - the
real application builds and tests pass, or any blockers are clearly
reported.

## Apple reference material

Use Apple's current guidance as the source of truth for the design
principles. Do not copy Apple's proprietary UI assets or suggest that a
cross-platform approximation is the native Apple material.

-   Liquid Glass overview:
    https://developer.apple.com/documentation/TechnologyOverviews/liquid-glass
-   Human Interface Guidelines --- Materials:
    https://developer.apple.com/design/human-interface-guidelines/materials
-   Human Interface Guidelines --- Design principles:
    https://developer.apple.com/design/human-interface-guidelines/design-principles/
-   Human Interface Guidelines --- Layout:
    https://developer.apple.com/design/human-interface-guidelines/layout
-   WWDC session, "Meet Liquid Glass":
    https://developer.apple.com/videos/play/wwdc2025/219/
