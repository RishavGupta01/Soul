import { describe, it, expect } from "vitest";
import { MiniGPT } from "./model";
import { randnMat, randnVec, makeRng, vec, type Mat, type Vec } from "./tensor";
import type { ModelWeights, ModelConfig } from "./types";

function tinyConfig(): ModelConfig {
  return { vocabSize: 32, dModel: 8, nLayers: 2, nHeads: 2, dFfn: 16, contextLength: 16 };
}

function randMat(rows: number, cols: number, rng: () => number): Mat {
  return randnMat(rows, cols, rng, 0.05);
}

function randomWeights(c: ModelConfig, seed = 1): ModelWeights {
  const rng = makeRng(seed);
  const v = (n: number): Vec => randnVec(n, rng, 0.05);
  const ones = (n: number): Vec => vec(n).fill(1);
  return {
    config: c,
    wte: randMat(c.vocabSize, c.dModel, rng),
    wpe: randMat(c.contextLength, c.dModel, rng),
    lnFW: ones(c.dModel),
    lnFB: vec(c.dModel),
    layers: Array.from({ length: c.nLayers }, () => ({
      ln1W: ones(c.dModel),
      ln1B: vec(c.dModel),
      qkvW: randMat(3 * c.dModel, c.dModel, rng),
      qkvB: vec(3 * c.dModel),
      projW: randMat(c.dModel, c.dModel, rng),
      projB: vec(c.dModel),
      ln2W: ones(c.dModel),
      ln2B: vec(c.dModel),
      fcW: randMat(c.dFfn, c.dModel, rng),
      fcB: vec(c.dFfn),
      fc2W: randMat(c.dModel, c.dFfn, rng),
      fc2B: vec(c.dModel),
    })),
  };
}

describe("MiniGPT.forward", () => {
  const c = tinyConfig();
  const model = new MiniGPT(randomWeights(c));

  it("returns logits of length vocabSize", () => {
    const out = model.forward([1, 2, 3]);
    expect(out.logits.length).toBe(c.vocabSize);
    expect(out.allLogits.length).toBe(3);
  });

  it("emits one step per layer with correctly shaped attention", () => {
    const out = model.forward([1, 2, 3, 4]);
    expect(out.steps.length).toBe(c.nLayers);
    const s0 = out.steps[0];
    expect(s0.attnScores.length).toBe(c.nHeads);
    expect(s0.attnScores[0].length).toBe(4); // t query rows
    expect(s0.attnScores[0][0].length).toBe(4); // key columns
    expect(s0.attnProbs[0][0].reduce((a, b) => a + b, 0)).toBeCloseTo(1, 5);
  });

  it("is deterministic across identical calls", () => {
    const a = model.forward([5, 6]);
    const b = model.forward([5, 6]);
    expect(a.logits).toEqual(b.logits);
  });

  it("rejects out-of-range token ids", () => {
    expect(() => model.forward([9999])).toThrow();
  });

  it("KV cache produces the same last-token logits as a full pass", () => {
    const prompt = [1, 2, 3];
    const full = model.forward(prompt);
    const cache = Array.from({ length: c.nLayers }, () => ({ k: [] as Vec[], v: [] as Vec[] }));
    model.forward(prompt, 0, cache);
    const cached = model.forward([4], prompt.length, cache);
    const fullSecond = model.forward([1, 2, 3, 4]);
    // last-position logits should match closely under incremental decoding
    const diff = cached.logits.map((v, i) => Math.abs(v - fullSecond.allLogits[3][i]));
    expect(Math.max(...diff)).toBeLessThan(1e-6);
    expect(full.logits.length).toBe(c.vocabSize);
  });
});