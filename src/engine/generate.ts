import { MiniGPT } from "./model";
import { BPETokenizer } from "./tokenizer";
import { sample, entropy, rankDesc } from "./sampler";
import type { GeneratedStep, LayerPass } from "../state/useModelStore";
import type { LayerStep } from "./types";

export interface GenOptions {
  prompt: string;
  maxNewTokens: number;
  temperature: number;
  topK: number;
  topP: number;
  seed: number;
  onStep?: (s: GeneratedStep, text: string) => void;
}

/** Deterministic mulberry32 so a given seed always plays back identically. */
export function rngFrom(seed: number): () => number {
  let a = seed >>> 0;
  return () => {
    a |= 0;
    a = (a + 0x6d2b79f5) | 0;
    let t = Math.imul(a ^ (a >>> 15), 1 | a);
    t = (t + Math.imul(t ^ (t >>> 7), 61 | t)) ^ t;
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  };
}

const TOP_SHOW = 64;

const l2 = (v: number[]): number => Math.sqrt(v.reduce((s, x) => s + x * x, 0));

/**
 * Collapse one layer's LayerStep into the handful of scalars the network view
 * draws. Every number here is measured off the real forward pass, not faked.
 */
function passesFrom(steps: LayerStep[]): LayerPass[] {
  return steps.map((s, layer) => {
    const heads = s.attnProbs;
    const lastQuery = 0;
    let focusSum = 0;
    let peakSum = 0;
    const headPeak: number[] = [];
    const agg = new Map<number, number>();
    for (let h = 0; h < heads.length; h++) {
      const row = heads[h][lastQuery] ?? [];
      let peak = 0;
      let mass = 0;
      let hSum = 0;
      for (let k = 0; k < row.length; k++) {
        const p = row[k];
        if (!(p > 0)) continue;
        if (p > peak) peak = p;
        mass += p;
        hSum -= p * Math.log(p);
        agg.set(k, (agg.get(k) ?? 0) + p / heads.length);
      }
      const hEntropy = mass > 0 ? hSum / Math.log(Math.max(2, row.length)) : 1;
      focusSum += 1 - Math.min(1, hEntropy);
      peakSum += peak;
      headPeak.push(peak);
    }
    const nHeads = Math.max(1, heads.length);
    const attnFocus = Math.min(1, focusSum / nHeads);
    const hiddenNorm = l2(s.hidden);
    const raw = hiddenNorm > 1e-6 ? l2(s.ffnDelta) / hiddenNorm : 0;
    const attnTop = [...agg.entries()]
      .sort((a, b) => b[1] - a[1])
      .slice(0, 4)
      .map(([k, w]) => ({ k, w }));
    return {
      layer,
      attnFocus,
      ffnEnergy: Math.min(1, raw / (1 + raw)),
      headPeak,
      attnTop,
      hiddenEnergy: Math.min(1, hiddenNorm / (hiddenNorm + 8)),
    };
  });
}

/**
 * Autoregressive generation, emitting one event per decoded token so the UI
 * can replay the thought. Real sampling — temperature / top-k / top-p are
 * applied to the actual logit vector.
 */
export function* generate(
  model: MiniGPT,
  tokenizer: BPETokenizer,
  opts: GenOptions
): Generator<GeneratedStep, string, void> {
  const cfg = model.config;
  const rng = rngFrom(opts.seed);
  const promptIds = tokenizer.encode(opts.prompt).slice(-cfg.contextLength);
  const ids: number[] = [...promptIds];
  let out = "";

  const cache = model.w.layers.map(() => ({ k: [], v: [] }));
  // Prime the cache with the prompt (positions 0..n-1), then generate one at a time.
  model.forward(ids, 0, cache);

  for (let i = 0; i < opts.maxNewTokens; i++) {
    const startPos = ids.length - 1;
    const lastId = ids[ids.length - 1];
    const res = model.forward([lastId], startPos, cache);
    const logits = res.logits;

    const { id, probs, ranking } = sample(logits, {
      temperature: opts.temperature,
      topK: opts.topK,
      topP: opts.topP,
      rng,
    });

    const top = ranking
      .slice(0, TOP_SHOW)
      .filter((t) => probs[t] > 0)
      .map((t) => ({ id: t, text: tokenizer.decode([t]), p: probs[t] }));

    const step: GeneratedStep = {
      pos: ids.length,
      tokenId: id,
      tokenText: tokenizer.decode([id]),
      top,
      entropy: entropy(probs),
      temperature: opts.temperature,
      passes: passesFrom(res.steps),
    };

    ids.push(id);
    out += tokenizer.decode([id]);
    if (opts.onStep) opts.onStep(step, out);
    yield step;

    if (id === 0 || ids.length >= cfg.contextLength) break;
  }

  return out;
}

/** One-shot convenience used by tests and quick probes. */
export function generateOnce(
  model: MiniGPT,
  tokenizer: BPETokenizer,
  prompt: string,
  opts: Partial<GenOptions> = {}
): { text: string; steps: GeneratedStep[] } {
  const steps: GeneratedStep[] = [];
  let text = "";
  const it = generate(model, tokenizer, {
    prompt,
    maxNewTokens: 40,
    temperature: 0.8,
    topK: 40,
    topP: 0.95,
    seed: 1,
    ...opts,
    onStep: (s, t) => {
      steps.push(s);
      text = t;
    },
  });
  let r = it.next();
  while (!r.done) r = it.next();
  return { text, steps };
}

export { rankDesc };
