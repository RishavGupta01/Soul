# Obsidian Brain — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build an interactive website that runs a real micro-GPT in the browser and visualizes every stage of an LLM end-to-end in an obsidian crystal aesthetic.

**Architecture:** Pure-TS `engine/` (no React) runs a real forward pass and emits step events; React subscribes and drives canvas/GSAP visualizations. An offline TS training script produces `weights.json`. Act I is a scroll-story; Act II is a canvas force-graph atlas.

**Tech Stack:** Vite, React 18, TypeScript, Zustand, GSAP, Vitest, canvas 2D, CSS 3D. Self-hosted fonts via @fontsource.

**Toolchain note (this machine):** Node v24 present; npm is `npm.cmd` (call as `npm` — resolves fine). **git is not installed**, so commit steps are marked "checkpoint" and should be run only once git exists. Use `npm run` scripts for all verification.

---

## Milestones

| # | Milestone | Produces | Done when |
|---|-----------|----------|-----------|
| M0 | Scaffold | Vite+React+TS app, design tokens, fonts | `npm run dev` shows styled void page; `npm test` passes |
| M1 | Engine | tensor, BPE tokenizer, attention, model, sampler + tests | Engine tests green; node script generates text |
| M2 | Weights | training script → `public/model/weights.json` | Script trains, loss falls, output is semi-coherent |
| M3 | Hero | Overture with live real generation, HUD, chapter rail, BrainCore, motion system | "Watch it think" works end-to-end in browser |
| M4 | Act I | Chapters 1–7 with viz primitives + progressive disclosure | Full descent, scrub, deep-link, reduced-motion |
| M5 | Act II | Atlas force-graph + training widget + nodes 8–13 | Free-roam atlas, live loss curve |
| M6 | Polish | A11y audit, responsive, perf, QA | Lighthouse a11y ≥95, keyboard-only pass |

---

## Phase 1 — Scaffold (M0)

### Task 1: Project scaffold
**Files:** Create `package.json`, `tsconfig.json`, `tsconfig.node.json`, `vite.config.ts`, `index.html`, `src/main.tsx`, `src/App.tsx`, `.gitignore`

- [ ] Step 1: `npm create vite@latest . -- --template react-ts` then `npm install`
- [ ] Step 2: Add deps: `npm i zustand gsap @fontsource/syne @fontsource/schibsted-grotesk @fontsource/jetbrains-mono` and `npm i -D vitest`
- [ ] Step 3: Set `"test": "vitest run"`, `"test:watch": "vitest"` in package.json scripts
- [ ] Step 4: Verify: `npm run build` succeeds; `npm test` runs (0 tests)
- [ ] Step 5: Checkpoint

### Task 2: Design tokens
**Files:** Create `src/styles/tokens.css`, `src/styles/base.css`, `src/styles/fonts.ts`
- [ ] Step 1: Encode every token from DESIGN.md as CSS custom properties on `:root` (void, panel, hairline, cyan, magenta, semantics, spacing scale, radii, durations, easings)
- [ ] Step 2: Import `@fontsource` faces in `fonts.ts`; expose display/body/mono families as `--font-*`
- [ ] Step 3: Base reset + `body` void background + 17px/1.65 body + focus-visible ring token
- [ ] Step 4: Verify: build passes; dev page shows void + Syne hero test text
- [ ] Step 5: Checkpoint

---

## Phase 2 — Engine (M1, TDD, pure TS)

### Task 3: Tensor ops
**Files:** Create `src/engine/tensor.ts`, `src/engine/tensor.test.ts`
- [ ] Step 1: Write failing tests: `matmul` (2×3 · 3×2 → 2×2 exact values), `add`, `layerNorm` (mean≈0, var≈1), `softmax` (rows sum to 1; temperature 0 → argmax one-hot), `gelu(0)=0`, `argmax`
- [ ] Step 2: Run `npm test` → FAIL
- [ ] Step 3: Implement `matmul(a,b)`, `add`, `mul`, `transpose`, `layerNorm(x,w,b)`, `softmax(x,temp)`, `gelu`, `argmax`, `gaussian(seed)` (seeded PRNG)
- [ ] Step 4: `npm test` → PASS
- [ ] Step 5: Checkpoint

### Task 4: BPE tokenizer
**Files:** Create `src/engine/tokenizer.ts`, `src/engine/tokenizer.test.ts`
- [ ] Step 1: Failing tests: `trainBPE(text, vocabSize)` returns merges; `encode("hello")→ids`; `decode(encode(s))===s` (round-trip on ASCII); unknown char → byte fallback
- [ ] Step 2: `npm test` → FAIL
- [ ] Step 3: Implement byte-level BPE: `trainBPE` (count adjacent pairs, merge most frequent, repeat), `encode` (greedy merge application), `decode`, `toJSON/fromJSON`
- [ ] Step 4: `npm test` → PASS
- [ ] Step 5: Checkpoint

### Task 5: Attention + model forward
**Files:** Create `src/engine/attention.ts`, `src/engine/model.ts`, `src/engine/attention.test.ts`, `src/engine/model.test.ts`, `src/engine/types.ts`
- [ ] Step 1: Failing tests: single-head attention on hand-set Q/K/V gives expected weighted output; softmax rows sum to 1; causal mask zeroes future; `Model.forward(ids)` returns `{logits, steps[]}` where `steps.length === nLayers` and each step has `attnScores` shape `[heads, t, t]`
- [ ] Step 2: `npm test` → FAIL
- [ ] Step 3: Implement `attention(q,k,v,heads)`, `Model` (embedding + positional + N blocks + final LN + lm_head), KV cache, `steps` event emission
- [ ] Step 4: `npm test` → PASS (against small seeded weights fixture)
- [ ] Step 5: Checkpoint

### Task 6: Sampler
**Files:** Create `src/engine/sampler.ts`, `src/engine/sampler.test.ts`
- [ ] Step 1: Failing tests: temperature→0 equals argmax; top-k keeps exactly k non-zero; top-p keeps smallest set whose cumulative ≥ p; seeded sampling is deterministic
- [ ] Step 2: `npm test` → FAIL
- [ ] Step 3: Implement `sample(logits, {temperature, topK, topP, seed})` returning `{id, probs}`
- [ ] Step 4: `npm test` → PASS
- [ ] Step 5: Checkpoint

---

## Phase 3 — Weights (M2)

### Task 7: Training script
**Files:** Create `scripts/corpus/*.txt` (bundled public-domain text), `scripts/train-micro.mjs`, `scripts/train-micro.test.mjs`
- [ ] Step 1: Failing test: training 20 steps on tiny fixture reduces loss (last < first)
- [ ] Step 2: Implement char/BPE data pipeline, forward, backprop (manual grads), Adam, seeded init; config d_model 64, layers 4, heads 4, ffn 256, vocab 512, ctx 128; write `public/model/weights.json` + `public/model/vocab.json`
- [ ] Step 3: Run script; assert loss falls and a sample continuation is readable
- [ ] Step 4: `npm test` → PASS
- [ ] Step 5: Checkpoint

---

## Phase 4 — Hero (M3)

### Task 8: Stores
**Files:** Create `src/state/useRunStore.ts`, `src/state/useDisclosureStore.ts`, `src/state/useAtlasStore.ts`, `src/state/useModelStore.ts`
- [ ] Step 1: `useModelStore`: load weights (loading/ready/fallback/error), `generate(prompt, opts)`, token stream, KV state
- [ ] Step 2: `useRunStore`: current chapter, scene events, play/scrub
- [ ] Step 3: `useDisclosureStore`: `depth: 'intuition'|'deep'` per term
- [ ] Step 4: `useAtlasStore`: nodes, edges, unlocked
- [ ] Step 5: Checkpoint

### Task 9: Motion system + BrainCore
**Files:** Create `src/motion/gsap.ts`, `src/motion/presets.ts`, `src/viz/BrainCore.tsx`
- [ ] Step 1: ScrollTrigger registration + `useScrollScene` hook
- [ ] Step 2: BrainCore: CSS3D crystal that breathes; states wireframe→materialize→idle; reduced-motion static
- [ ] Step 3: Verify in dev: core breathes, respects reduced-motion
- [ ] Step 4: Checkpoint

### Task 10: App shell, HUD, chapter rail
**Files:** Create `src/App.tsx` (replace), `src/shell/HUD.tsx`, `src/shell/ChapterRail.tsx`, `src/shell/Stage.tsx`, `src/shell/Placard.tsx`, `src/shell/Disclosure.tsx`, `src/router.ts`
- [ ] Step 1: Hash router `#/`, `#/tok`, `#/embed`, `#/pos`, `#/block`, `#/logits`, `#/sample`, `#/loop`, `#/atlas/*`
- [ ] Step 2: Museum layout: sticky stage 62vw + scrolling placard 68ch; mobile dock 45vh
- [ ] Step 3: HUD: model status dot, disclosure toggle, fast-forward; rail with chapter dots + keyboard arrows
- [ ] Step 4: Verify: navigate all routes, keyboard-only, focus rings visible
- [ ] Step 5: Checkpoint

### Task 11: Overture chapter (the payoff)
**Files:** Create `src/chapters/00-overture/{index.tsx,copy.ts,useScene.ts}`
- [ ] Step 1: Editable prompt + "Think" button; streams real tokens from `useModelStore`
- [ ] Step 2: Empty state "The brain awaits input"; loading only first run; fallback banner on weight error
- [ ] Step 3: Ambient: tokens fly from input into core, pulses along a path on each token
- [ ] Step 4: Verify: prompt → real generation visibly streams; disconnect network → fallback banner
- [ ] Step 5: Checkpoint

---

## Phase 5 — Act I (M4)

### Task 12: Viz primitives
**Files:** Create `src/viz/{AttentionHeatmap,FilamentArcs,VectorBars,ProbabilityRidge,TokenizerStrip,ResidualHighway,LogitRidge,KVCache}.tsx` + shared `src/viz/useCanvas.ts`
- [ ] Step 1: `useCanvas` (DPR-aware sizing, reduced-motion, aria summary prop)
- [ ] Step 2: Each primitive takes plain data; unit-render tests with fixtures
- [ ] Step 3: Verify all render with fixture data
- [ ] Step 4: Checkpoint

### Task 13: Chapters 1–4 (Tokenization, Embeddings, Positional, Block/Attention/FFN)
**Files:** Create `src/chapters/01-tokenization/*`, `02-embeddings/*`, `03-positional/*`, `04-transformer-block/*`
- [ ] Step 1: Tokenization: text shatters, merge animations, vocab lattice
- [ ] Step 2: Embeddings: id → row glow → vector bars → semantic mini-map
- [ ] Step 3: Positional: waves bind; shuffle demo breaks meaning
- [ ] Step 4: Block: residual highway; QK heatmap builds on scroll; softmax; 4 toggleable head filaments; FFN 64→256→64 GELU; residual pulses
- [ ] Step 5: Copy for each: `{intuition, deep}` (equations, shapes, code) + tests
- [ ] Step 6: Checkpoint

### Task 14: Chapters 5–7 (Logits/Softmax, Sampling, Autoregressive loop)
**Files:** Create `src/chapters/{05-logits,06-sampling,07-loop}/*`
- [ ] Step 1: Logits→softmax ridge → probability bars
- [ ] Step 2: Sampling with live temperature/top-k/top-p sliders (keyboard-operable, fluid lerp)
- [ ] Step 3: Loop with KV cache slots filling; speed control; scrub via event stream
- [ ] Step 4: Checkpoint

---

## Phase 6 — Act II Atlas (M5)

### Task 15: Atlas graph
**Files:** Create `src/atlas/{Atlas.tsx,useForce.ts,NodeCard.tsx,graph.ts}`
- [ ] Step 1: Custom d3-force (no dep: implement simple force sim) on canvas; obsidian node rendering
- [ ] Step 2: Locked/unlocked, keyboard traversal, zoom-to-node on activation
- [ ] Step 3: Verify: unlock after Act I, pan/zoom, keyboard path
- [ ] Step 4: Checkpoint

### Task 16: Training widget + nodes 8–13
**Files:** Create `src/worker/train.worker.ts`, `src/engine/trainer.ts`, `src/chapters/08-training/*` … `13-glossary/*`
- [ ] Step 1: `trainer.ts` backprop+Adam for mini model (reuse tensor.ts); worker posts loss epochs
- [ ] Step 2: Training node: backprop backward-glow, gradient heatmap, Adam, loss curve + live widget (idle/training/complete/error states)
- [ ] Step 3: Nodes 9–13 cinematic: RLHF picker, quantization lattice, GPU parallelism arcs, scaling curves, glossary
- [ ] Step 4: Checkpoint

---

## Phase 7 — Polish (M6)

### Task 17: A11y + responsive + perf
- [ ] Step 1: aria summaries for every canvas; keyboard pass; 44px targets; reduced-motion pass
- [ ] Step 2: Breakpoints per DESIGN.md; mobile dock; matrix downscale
- [ ] Step 3: Perf: lazy chapters, particle budget, font preload; profile
- [ ] Step 4: Manual QA checklist; Lighthouse a11y ≥95
- [ ] Step 5: Checkpoint

---

## Self-Review

**Spec coverage:** §3 content → Tasks 11–16 · §4 architecture → Tasks 1–10 · §5 engine → Tasks 3–7 · §6 visual → Tasks 2, 9, 12 · §7 states → Tasks 8, 11, 16 · §8 journey → Tasks 10, 11 · §9 a11y → Tasks 12, 17 · §10 errors → Tasks 8, 11, 15 · §11 testing → every task (TDD) · §12 decisions → Tasks 6, 15 (d3 custom), 7 (corpus), 9 (CSS3D), 2 (fonts) · §13 scope → no light theme, no sound anywhere.
**Placeholders:** none.
**Type consistency:** `Model.forward(ids, kv)` → `{logits, steps}`; `steps[]` step has `attnScores[heads][t][t]`, `hidden`, `qkv`; `sample(logits, opts)` → `{id, probs}`; stores named `useRunStore/useDisclosureStore/useAtlasStore/useModelStore`. Consistent across tasks.