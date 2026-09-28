import { describe, it, expect } from "vitest";
import { causalSelfAttention } from "./attention";
import { matFrom, vec } from "./tensor";

/**
 * Identity QKV so attention reduces to a known weighted average:
 * q = k = v = x. For a single head with dModel = 2, one token attends to itself.
 */
describe("causalSelfAttention", () => {
  const dModel = 2;
  const nHeads = 1;
  // 3*dModel x dModel identity-ish: rows 0-1 -> q, 2-3 -> k, 4-5 -> v (all identity)
  const I = matFrom(6, 2, [1, 0, 0, 1, 1, 0, 0, 1, 1, 0, 0, 1]);
  const b = vec(6);

  it("a single token attends 100% to itself", () => {
    const x = [Float64Array.from([1, 0])];
    const out = causalSelfAttention(x, I, b, nHeads);
    expect(out.probs[0][0][0]).toBeCloseTo(1, 6);
    expect(Array.from(out.context[0])).toEqual([1, 0]);
  });

  it("probabilities per query sum to 1 and never attend to the future", () => {
    const x = [
      Float64Array.from([1, 0]),
      Float64Array.from([0, 1]),
      Float64Array.from([1, 1]),
    ];
    const out = causalSelfAttention(x, I, b, nHeads);
    for (let i = 0; i < 3; i++) {
      const row = out.probs[0][i];
      expect(row.reduce((s, p) => s + p, 0)).toBeCloseTo(1, 6);
      for (let j = i + 1; j < 3; j++) expect(row[j]).toBe(0);
    }
  });

  it("produces score matrices shaped [heads][t][t]", () => {
    const x = [Float64Array.from([1, 0]), Float64Array.from([0, 1])];
    const out = causalSelfAttention(x, I, b, nHeads);
    expect(out.scores.length).toBe(nHeads);
    expect(out.scores[0].length).toBe(2);
    expect(out.scores[0][0].length).toBe(2);
  });
});