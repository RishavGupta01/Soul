import { softmax, type Vec, vec } from "./tensor";

export interface SampleOptions {
  temperature: number;
  topK: number; // 0 or <=0 disables
  topP: number; // 1 disables
  seed?: number;
  rng?: () => number;
}

export interface SampleResult {
  id: number;
  probs: number[];
  /** the ordered candidate ids by probability descending (pre-truncation) */
  ranking: number[];
}

/**
 * Temperature + top-k + top-p (nucleus) sampling.
 * Pure and deterministic given an rng.
 */
export function sample(logits: ArrayLike<number>, opts: SampleOptions): SampleResult {
  const n = logits.length;
  const logitVec = vec(n);
  for (let i = 0; i < n; i++) logitVec[i] = logits[i];

  // Rank before temperature so callers can show the raw ordering.
  const ranking = Array.from({ length: n }, (_, i) => i).sort(
    (a, b) => logitVec[b] - logitVec[a]
  );

  const rng = opts.rng ?? defaultRng(opts.seed ?? 1);

  // Temperature 0 => greedy argmax.
  if (opts.temperature <= 0) {
    const id = ranking[0];
    const probs = vec(n);
    probs[id] = 1;
    return { id, probs: Array.from(probs), ranking };
  }

  let probs = softmax(logitVec, opts.temperature);

  // top-k truncation
  if (opts.topK > 0 && opts.topK < n) {
    const keep = new Set(ranking.slice(0, opts.topK));
    let sum = 0;
    for (let i = 0; i < n; i++) {
      if (!keep.has(i)) probs[i] = 0;
      else sum += probs[i];
    }
    if (sum > 0) for (let i = 0; i < n; i++) probs[i] /= sum;
  }

  // top-p (nucleus) truncation over the ranked order
  if (opts.topP < 1) {
    const sorted = ranking.filter((id) => probs[id] > 0);
    let cumulative = 0;
    const keep = new Set<number>();
    for (const id of sorted) {
      keep.add(id);
      cumulative += probs[id];
      if (cumulative >= opts.topP) break;
    }
    let sum = 0;
    for (let i = 0; i < n; i++) {
      if (!keep.has(i)) probs[i] = 0;
      else sum += probs[i];
    }
    if (sum > 0) for (let i = 0; i < n; i++) probs[i] /= sum;
  }

  // Draw from the truncated distribution.
  let r = rng();
  let acc = 0;
  let chosen = ranking[0];
  for (let i = 0; i < n; i++) {
    if (probs[i] <= 0) continue;
    acc += probs[i];
    if (r <= acc) {
      chosen = i;
      break;
    }
  }

  return { id: chosen, probs: Array.from(probs), ranking };
}

function defaultRng(seed: number): () => number {
  let a = seed >>> 0;
  return () => {
    a |= 0;
    a = (a + 0x6d2b79f5) | 0;
    let t = Math.imul(a ^ (a >>> 15), 1 | a);
    t = (t + Math.imul(t ^ (t >>> 7), 61 | t)) ^ t;
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  };
}

/** Argsort descending, exposed for the probability-ridge visualization. */
export function rankDesc(logits: ArrayLike<number>): number[] {
  return Array.from({ length: logits.length }, (_, i) => i).sort(
    (a, b) => logits[b] - logits[a]
  );
}

export function entropy(probs: ArrayLike<number>): number {
  let h = 0;
  for (let i = 0; i < probs.length; i++) {
    const p = probs[i];
    if (p > 0) h -= p * Math.log2(p);
  }
  return h;
}

/** Kept for API symmetry with the store. */
export function normalize(probs: Vec): Vec {
  let s = 0;
  for (let i = 0; i < probs.length; i++) s += probs[i];
  if (s <= 0) return probs;
  const out = vec(probs.length);
  for (let i = 0; i < probs.length; i++) out[i] = probs[i] / s;
  return out;
}