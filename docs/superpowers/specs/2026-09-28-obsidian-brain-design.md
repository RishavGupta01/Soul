# Obsidian Brain — Design Spec

**Date:** 2026-09-28
**Status:** Design approved; ready for implementation planning.
**Design system source of truth:** `DESIGN.md` (repo root)

---

## 1. Product Definition

An interactive website that visually simulates how an LLM works, end-to-end, in extreme detail — "the obsidian brain." Users watch a real micro transformer think: real tokenization, real attention math, real sampling — wrapped in a cinematic obsidian-and-synapse visual system with scroll choreography, micro-interactions, and a free-roam atlas of the surrounding architecture.

**The one memory:** *"Watching it actually think."* Every decision serves the live-generation moment.

## 2. Decisions Log (brainstorming outcomes)

| # | Decision | Choice |
|---|----------|--------|
| 1 | Simulation fidelity | **Hybrid** — real math for the forward pass; cinematic visuals for training/RLHF/scale |
| 2 | Scope | **Everything** — inference core + training life + hardware/parallelism/quantization/scaling laws |
| 3 | Visual direction | **Obsidian & synapse** → refined to *crystal-terminal* (see DESIGN.md) |
| 4 | Navigation | **Guided descent (Act I) + free-roam atlas (Act II)** |
| 5 | Tech stack | **Vite + React + TypeScript** |
| 6 | Explanations | **Progressive disclosure** — intuition layer default, "deeper" toggle reveals equations/shapes/code |
| 7 | Engine approach | **Pre-trained micro-GPT shipped as weights.json** + a separate live in-browser training widget |
| 8 | Memorable thing | **Watching it actually think** |

## 3. Content Map

### Act I — The Descent (guided scroll, inference core)

| # | Chapter | Core visualization | Real math? |
|---|---------|--------------------|------------|
| 0 | Overture | Hero: editable prompt, live token stream from the real model | Yes |
| 1 | Tokenization | Text shatters into BPE tokens; merge rules animate; vocab lattice | Yes (real BPE) |
| 2 | Embeddings | Token IDs light embedding rows → 64-dim vector strip; semantic neighbour mini-map | Yes |
| 3 | Positional encoding | Sinusoid/RoPE waves bind to vectors; shuffle-words demo breaks meaning | Yes (illustrated) |
| 4 | The Transformer Block ×4 | Residual highway, LayerNorm clamp, **attention** (QK heatmap → softmax → head-by-head filaments), **FFN** 64→256→64 with GELU, residual pulses | Yes |
| 5 | Logits → Softmax | 1024-vocab logit ridge → exponentiated → probability bars | Yes |
| 6 | Sampling | Temperature / top-k / top-p sliders reshape distribution live; chosen token snaps out | Yes |
| 7 | Autoregressive loop | Token feeds back; KV cache slots fill; full generation at user-chosen speed | Yes |

### Act II — The Atlas (free-roam node graph)

| # | Node | Core visualization | Real math? |
|---|------|--------------------|------------|
| 8 | Training | Loss → backprop glowing backward → gradient heatmaps → Adam update → loss curve; **live mini-train widget** (2-layer, d=32, Web Worker, real falling loss) | Widget: yes · historical run: cinematic |
| 9 | Post-training | SFT, reward model, RLHF preference-picker → policy shift | Cinematic |
| 10 | Inference optimization | KV cache deep-dive, quantization ladder fp32→fp16→int8→int4 (weight lattice shrinking), batching | Cinematic |
| 11 | Hardware & parallelism | GPU die anatomy; tensor/pipeline/data parallel splits with communication arcs | Cinematic |
| 12 | Scaling laws | Compute/data/params loss surface; Chinchilla optimum | Cinematic |
| 13 | Glossary + math appendix | Every equation ever revealed, collected and linked back | n/a |

## 4. System Architecture

```
ui/
├─ scripts/train-micro.mjs      # offline: trains micro-GPT → public/model/weights.json + vocab
├─ public/model/weights.json    # ~300KB, lazy-loaded
├─ DESIGN.md                    # design source of truth
└─ src/
   ├─ engine/                   # pure TS, zero React — unit-testable, reused by Worker
   │   ├─ tokenizer.ts          # BPE encode/decode
   │   ├─ tensor.ts             # Float32 matmul/add/norm ops
   │   ├─ model.ts              # load weights → forward(logits, kvCache)
   │   ├─ attention.ts          # QKV, scores, softmax, head-split
   │   ├─ sampler.ts            # temperature / top-k / top-p
   │   └─ trainer.ts            # backprop + Adam (runs in Worker)
   ├─ worker/train.worker.ts    # live mini-training; posts loss points
   ├─ state/
   │   ├─ useRunStore.ts        # chapter, token stream, KV state, model status
   │   ├─ useDisclosureStore.ts # intuition ⇄ deep layer
   │   └─ useAtlasStore.ts      # graph nodes/edges, unlocked nodes
   ├─ viz/                      # visual primitives — take plain data, no model knowledge
   │   ├─ AttentionHeatmap.tsx  ├─ FilamentArcs.tsx   ├─ VectorBars.tsx
   │   ├─ ProbabilityRidge.tsx  ├─ WeightLattice.tsx  ├─ LossCurve.tsx
   │   └─ BrainCore.tsx         # persistent anchor object
   ├─ motion/                   # GSAP ScrollTrigger setups + transition presets
   ├─ chapters/                 # one folder per chapter, zero-padded to the content map:
   │   ├─ 00-overture/          #   each: index.tsx, copy.ts, useScene.ts
   │   ├─ 04-transformer-block/ #   attention is a scene *inside* this chapter
   │   └─ …                     #   (01-03, 05-07 = Act I; 08-13 = Act II nodes)
   ├─ atlas/                    # Act II force graph (custom d3-force on canvas)
   └─ content/                  # { intuition, deep } copy pairs
```

Folder numbers match the Content Map exactly. Deep links: chapters are hash-routed (`#/attention`), and major scenes within a chapter get sub-routes (`#/block/attention`).

**Boundaries**
- `engine/` never imports React; the Worker imports it directly.
- Each `chapters/NN-*` owns copy, scene state, viz wiring — stages are isolated.
- `viz/` primitives accept matrices/vectors only — no model knowledge.
- Run state and UI state live in separate stores; never mixed.

**Data flow (Act I):**
```
input → tokenizer.encode → ids[] → model.forward(ids, kv)
  → per-layer step events { qkv, attnScores, hidden } emitted
  → useRunStore ← viz primitives subscribe → animations react
  → logits → sampler → next id → loop
```
The engine emits a **stream of step events**; every visualization is a subscriber. This is what makes scrubbing backward/forward possible.

## 5. Engine Design (the "real math" half)

- **Model:** micro-GPT, ~4 layers, d_model 64, 4 heads, FFN 256, vocab ~1024 (small BPE), context 128.
- **Training:** offline script in pure TypeScript/Node (no ML deps), seeded and deterministic, trained on bundled public-domain corpus (~1–2MB, Project Gutenberg-style plain text). Exports `weights.json` (~200–400KB).
- **Runtime:** real forward pass in the browser — real tokenization, real attention matrices, real softmax, real temperature/top-k/top-p sampling. Output is semi-coherent text.
- **Live training widget:** separate mini model (2 layers, d=32) trained in a Web Worker on demand; loss curve animates as it learns; posts epoch loss points to the store.
- **Fallback:** if `weights.json` fails to load, chapters switch to illustrative pre-baked numbers with a visible banner; the site never breaks.

## 6. Visual & Motion System

See `DESIGN.md` for authoritative tokens. Highlights:

- **BrainCore:** shared obsidian crystal anchor persisting across all Act I chapters; each chapter zooms into a facet (a layer slab, the output crown). CSS 3D + canvas; WebGL only if profiling demands.
- **Motion tiers:** ambient (breathe 6s, drifting particles, cursor radial light) · scroll choreography (GSAP pin + scrub) · event bursts (300–500ms light pulses along real data paths on engine steps).
- **Decoration-as-data:** every glow traces a tensor, attention edge, or gradient — nothing decorative.
- **Micro-interactions:** tensor shape labels glow their matrix on hover; sliders have magnetic thumbs and spring easing with fluid lerp on the probability ridge; clicking a token flies it to its embedding row; disclosure toggle unfolds equations with staggered mask-reveal.

## 7. Interaction States (per feature)

| Feature | Loading | Empty | Error | Success | Partial |
|---|---|---|---|---|---|
| Weights load | BrainCore materializes from wireframe; mono "loading weights" tick | — | Banner + illustrative fallback numbers, site fully usable | Core breathes, status dot green | Chapters show pre-baked numbers until ready |
| Prompt / generation | Streaming token cursor blinks | "The brain awaits input" state on stage | Retry affordance; input never lost | Token stream completes, KV cache full | Mid-generation scroll-away keeps state |
| Live training widget | Worker spin-up, disabled start button | "Teach it something" prompt | Error card + retry; previous curve preserved | Loss curve flattens, badge awarded | Curve updates live per epoch |
| Atlas | Nodes fade in with stagger | Locked nodes dimmed with unlock hint | Soft reset of layout | Unlocked node zooms open | Partial graph while scrolling |
| Disclosure toggle | — | Default = intuition layer | — | Deep layer unfolds in place | Deep layer remembers last-open equations |

## 8. User Journey & Emotional Arc

- **5s (visceral):** obsidian void, crystal core, real tokens typing themselves → *awe*.
- **5min (behavioral):** attention heatmap builds as you scroll; sliders reshape probabilities under your hand → *understanding + control*.
- **Reflective:** atlas shows the whole machine; glossarizes every term → *completeness*.
- **Fatigue interventions:** sticky jump rail (any chapter, any time), per-chapter badges, "run the whole pass in 30s" fast-forward, deep-linkable URLs (`#/attention`).

## 9. Responsive & Accessibility

- <900px: stage docks top at 45vh, placard scrolls beneath; atlas becomes pan/zoom; matrix resolution scales down.
- `prefers-reduced-motion`: pulses, breathing, and scrub disabled — static final frames, content fully readable.
- Canvas scenes expose aria summaries (e.g., "'the' attends most to 'cat'").
- Sliders operable by arrow keys; ≥44px touch targets; full keyboard path through chapter rail, disclosures, atlas nodes.
- Contrast: body ≥7:1, UI ≥4.5:1. Semantics carried by solid/dashed lines + labels, never color alone.

## 10. Error Handling

- Weights fetch failure → fallback mode + banner (above).
- No WebGL / low perf → CSS fallback BrainCore, reduced particle counts.
- Worker failure → widget shows retry, rest of site unaffected.
- Corrupt/NaN engine output → guard rails clamp values; visualization skips frame rather than rendering garbage.

## 11. Testing

- **Engine unit tests** (Vitest): tokenizer round-trip, matmul/attention correctness against reference values, sampler invariants (temperature → argmax at 0; top-k keeps exactly k), forward-pass snapshot vs training-script output.
- **Training script test:** loss strictly decreases over N steps on a fixed seed.
- **Component tests:** each chapter renders with fixture step events; disclosure toggle reveals deep copy.
- **Manual QA:** run in browser at desktop + mobile widths, reduced-motion on, keyboard-only pass.

## 12. Resolved Design Decisions (Pass 7)

| Decision | Resolution | Why |
|---|---|---|
| Atlas graph tech | Custom d3-force rendered on canvas | Bespoke obsidian styling; generic node-editor libs look template |
| Training corpus | Bundled public-domain plain text (~1–2MB) | Semi-coherent, smart-sounding output; license-safe |
| Rendering tech | Canvas2D + CSS3D first; WebGL only if profiling demands | Ship less machinery; upgrade on evidence |
| Fonts | Self-hosted via @fontsource (Syne, Schibsted Grotesk, JetBrains Mono) | Perf + offline; no CDN dependency |
| Sound | Silent by default | Not requested; respects autoplay norms |

## 13. Not in Scope (v1)

- Light theme (the design IS dark).
- Multi-language copy.
- Real production-model weights or API calls — everything local.
- Social features / accounts.
- Sound design.
