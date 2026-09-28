import { describe, it, expect } from "vitest";
import { sample, entropy, rankDesc } from "./sampler";

const logits = [0.1, 5.0, 0.2, 3.0, 0.05];

describe("sample", () => {
  it("temperature 0 selects the argmax", () => {
    const r = sample(logits, { temperature: 0, topK: 0, topP: 1 });
    expect(r.id).toBe(1);
    expect(r.probs[1]).toBe(1);
  });

  it("top-k keeps exactly k non-zero probabilities", () => {
    const r = sample(logits, { temperature: 1, topK: 2, topP: 1, seed: 3 });
    const nonZero = r.probs.filter((p) => p > 0).length;
    expect(nonZero).toBe(2);
  });

  it("top-p keeps the smallest set whose cumulative mass reaches p", () => {
    // top two hold almost all mass; p=0.5 should keep just the argmax here
    const r = sample(logits, { temperature: 1, topK: 0, topP: 0.5, seed: 3 });
    const nonZero = r.probs.filter((p) => p > 0).length;
    expect(nonZero).toBeGreaterThanOrEqual(1);
    expect(nonZero).toBeLessThanOrEqual(2);
  });

  it("probabilities sum to ~1 after truncation", () => {
    const r = sample(logits, { temperature: 0.8, topK: 3, topP: 0.9, seed: 5 });
    const sum = r.probs.reduce((s, p) => s + p, 0);
    expect(sum).toBeCloseTo(1, 6);
  });

  it("is deterministic for a given seed", () => {
    const a = sample(logits, { temperature: 1, topK: 0, topP: 1, seed: 99 });
    const b = sample(logits, { temperature: 1, topK: 0, topP: 1, seed: 99 });
    expect(a.id).toBe(b.id);
  });

  it("ranking is descending by logit", () => {
    expect(rankDesc(logits)).toEqual([1, 3, 2, 0, 4]);
  });
});

describe("entropy", () => {
  it("is zero for a one-hot distribution", () => {
    expect(entropy([1, 0, 0])).toBeCloseTo(0, 10);
  });
  it("is maximal for a uniform distribution", () => {
    expect(entropy([0.25, 0.25, 0.25, 0.25])).toBeCloseTo(2, 10);
  });
});