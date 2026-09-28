# Design System — Obsidian Brain

## Product Context
- **What this is:** An interactive visual simulation of how an LLM works end-to-end — a real micro-GPT running live under a cinematic, explorable interface.
- **Who it's for:** Curious learners through ML practitioners; every screen serves both via progressive disclosure.
- **Space/industry:** Educational interactive / AI explainer.
- **Project type:** Interactive scroll-story + free-roam atlas (hybrid site).
- **The one memory:** "Watching it actually think" — every decision serves the live moment where a real model generates and you see attention fire.

## Aesthetic Direction
- **Direction:** Crystal-terminal — obsidian black glass dissected by an instrument-grade interface. Light exists only where data moves.
- **Decoration level:** Expressive, but decoration IS data. If a glow isn't tracing a tensor, attention edge, or gradient, it doesn't exist.
- **Mood:** Planetarium precision. One object (the BrainCore), always visible, always thinking.
- **References:** Planetarium shows, Teenage Engineering hardware, trading terminals.

## Typography
- **Display/Hero:** Syne ExtraBold — wide, crystalline; chapter titles 72–120px, tracking -0.03em.
- **Body:** Schibsted Grotesk 400/500 — 17px / 1.65, measure max 68ch.
- **UI/Labels:** Schibsted Grotesk 500, 13px uppercase +0.08em for hairline labels.
- **Data/Tables:** JetBrains Mono 400/500 — `font-variant-numeric: tabular-nums` mandatory for every live value.
- **Code:** JetBrains Mono 400, 14px.
- **Loading:** JetBrains Mono 400, dimmed, with tick animation.
- **Scale:** modular 1.333 — 13 / 17 / 22 / 30 / 40 / 54 / 72 / 96.
- **Never primary:** Inter, Roboto, Arial, Helvetica, Space Grotesk, system-ui.

## Color
- **Approach:** expressive, data-governed.
- **Void / surfaces:** `#050507` void · `#0B0B10` panel · `#101018` raised.
- **Hairlines:** `rgba(255,255,255,0.06)` default, `rgba(255,255,255,0.14)` hover.
- **Primary:** `#3DE8FF` electric cyan — data in motion, forward pass.
- **Secondary:** `#FF3DAE` magenta — attention targets.
- **Neutrals:** `#8A8A99` muted text, `#E8E8F0` body text, `#FFFFFF` headings.
- **Semantic:** success/forward `#4ADE80` · warning/backward-gradient `#FFB454` · error `#FF5C5C` · info `#3DE8FF`.
- **Heat scale (attention matrices only):** `#050507 → #8B5CFF → #FFFFFF`.
- **Dark mode:** this IS the dark design; no light theme in v1.
- **Hard anti-slop rules:** no purple/violet gradient backgrounds, no gradient buttons, no decorative blobs, no icons in colored circles.

## Spacing
- **Base unit:** 8px.
- **Density:** comfortable (drama needs void), except instrument readouts at 4–8px tightness.
- **Scale:** 2xs(4) xs(8) sm(16) md(24) lg(32) xl(48) 2xl(64) 3xl(96) 4xl(128).
- **Chapter padding:** 96–128px vertical.

## Layout
- **Approach:** creative-editorial, asymmetric museum-exhibit pattern.
- **Act I grid:** pinned full-bleed **stage** (60–65vw) + narration **placard** column (max 68ch) scrolling alongside. Model never leaves screen.
- **Act II:** edge-to-edge atlas force-graph, no columns.
- **Mobile (<900px):** stage docks top at 45vh fixed, placard scrolls beneath; atlas becomes pan/zoom surface.
- **Max content width (placard):** 68ch. Full-bleed visuals: 100vw.
- **Border radius:** sharp instrument scale — sm(2px) md(4px); no pills, no 12px+ bubbles.

## Motion
- **Approach:** expressive for data, intentional for UI, nothing else moves.
- **Easing:** enter `cubic-bezier(.16,1,.3,1)` (expo-out) · exit ease-in · data-flow linear along real paths.
- **Duration:** micro(80ms) · short(180ms) · medium(320ms) · long(600ms) · ambient breathe(6s).
- **Data pulses:** 600ms linear traveling along actual attention arcs / residual highway.
- **Scroll:** GSAP ScrollTrigger, pin + scrub per chapter (heatmap builds row-by-row with scroll).
- **Live numbers:** mono tabular tick, never crossfade; dimmed at rest, full brightness during motion.
- **Reduced motion:** `prefers-reduced-motion` disables pulses, breathing, and scrub — static final frames instead.

## Accessibility
- Canvas scenes expose aria summaries ("'the' attends most to 'cat'").
- Sliders keyboard-operable (arrow keys), 44px touch targets.
- Forward vs backward signalled by solid/dashed line and label, not color alone.
- Body contrast ≥7:1, UI text ≥4.5:1.
- Full keyboard path: chapter rail arrows, disclosure toggle, atlas node traversal.

## Decisions Log
| Date | Decision | Rationale |
|------|----------|-----------|
| 2026-09-28 | Initial design system created | Built with /design-consultation; hero memory = "watching it actually think" |
| 2026-09-28 | Syne + Schibsted Grotesk + JetBrains Mono | Blacklist-compliant, strong voice contrast, tabular data figures |
| 2026-09-28 | Magenta replaces violet as attention pair | Avoids purple-gradient AI slop while keeping heat scale |
| 2026-09-28 | Decoration-as-data rule | Every glow explains something; prevents generative filler |
