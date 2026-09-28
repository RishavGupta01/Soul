# Obsidian Brain

An interactive anatomy of a language model. Not a metaphor — a real
micro-GPT (64-d, 3 layers, 4 heads, 384-token vocab) running its actual
forward pass in your browser while the page explains every stage of it.

**Watch it think.**

## What's here

| | |
|---|---|
| `src/engine/` | The model itself — tensor ops, BPE tokenizer, multi-head attention, sampler. Used at runtime, no server. |
| `src/chapters/` | The nine guided chapters of Act I, from tokenization to the sampling loop. |
| `src/viz/` | Canvas visualisations: the brain core, attention heatmaps, the atlas graph. |
| `scripts/` | The trainer (`train-micro.mjs`), a headless-browser audit (`audit.mjs`), and a CDP screenshot driver. |

Act II (`#/atlas`) is a free-roam map of every concept, wired to the same
live model.

## Develop

```bash
npm install
npm run dev      # http://localhost:5173
npm test         # engine + render smoke tests
npm run train    # retrain the shipped weights locally
```

`public/model/` holds the pre-trained weights, vocab and training metrics
that ship with the site. Training is a separate Node process — the page
only ever reads the resulting JSON.

## Deploy

GitHub Pages, via `.github/workflows/deploy.yml`. The workflow builds with
`BASE_PATH=/<repo>/` so assets resolve under a project site. The app uses
hash routing, so no server rewrites are needed.

Push to `main` (or run the workflow manually) to publish.

## Design

See [`DESIGN.md`](DESIGN.md) — palette, type scale, motion and the rules
that keep decoration honest (a visual only exists if it traces real data).
