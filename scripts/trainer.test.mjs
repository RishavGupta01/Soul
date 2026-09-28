/**
 * Gradient correctness + cross-implementation agreement tests.
 *
 * 1. Numerical gradient check on a tiny model (finite differences).
 * 2. Forward-pass agreement between scripts/trainer-lib.mjs and src/engine
 *    (guards against the two implementations drifting).
 */
import { describe, it, expect } from "vitest";
import {
  initWeights,
  zeroGrad,
  lossAndGrad,
  forwardTrain,
  paramPairs,
  serializeWeights,
} from "./trainer-lib.mjs";
import { MiniGPT, modelFromJSON } from "../src/engine/model.ts";

const tiny = {
  vocabSize: 24,
  dModel: 8,
  nLayers: 2,
  nHeads: 2,
  dFfn: 16,
  contextLength: 12,
};

describe("backprop correctness", () => {
  it("matches finite-difference gradients on sampled parameters", () => {
    const W = initWeights(tiny, 7);
    const tokens = [1, 2, 3, 4, 5, 6, 7];
    const G = zeroGrad(W);
    lossAndGrad(W, G, tokens);

    const pairs = [];
    paramPairs(W, G, (val, grad) => pairs.push({ val, grad }));
    const step = Math.max(1, Math.floor(pairs.length / 12));
    const sampleIdx = [];
    for (let i = 0; i < pairs.length; i += step) sampleIdx.push(i);

    const eps = 1e-5;
    const numericLoss = () => lossAndGrad(W, zeroGrad(W), tokens).loss;

    for (const idx of sampleIdx) {
      const { val, grad } = pairs[idx];
      for (let j = 0; j < Math.min(3, val.d.length); j++) {
        const orig = val.d[j];
        val.d[j] = orig + eps;
        const lp = numericLoss();
        val.d[j] = orig - eps;
        const lm = numericLoss();
        val.d[j] = orig;
        const numeric = (lp - lm) / (2 * eps);
        const analytic = grad.d[j];
        const denom = Math.max(1e-4, Math.abs(numeric) + Math.abs(analytic));
        const relErr = Math.abs(numeric - analytic) / denom;
        expect(
          relErr,
          `param ${idx}[${j}] analytic=${analytic} numeric=${numeric}`
        ).toBeLessThan(3e-2);
      }
    }
  });
});

describe("trainer/engine agreement", () => {
  it("produces identical last-position logits for the same weights + tokens", () => {
    const W = initWeights(tiny, 11);
    const tokens = [3, 1, 4, 1, 5, 9, 2];
    const { logits } = forwardTrain(W, tokens);
    const V = tiny.vocabSize;
    const T = tokens.length;
    const libLast = Array.from(logits.slice((T - 1) * V, T * V));

    const model = modelFromJSON(serializeWeights(W, 8));
    const engineOut = model.forward(tokens);
    const maxDiff = Math.max(...libLast.map((v, i) => Math.abs(v - engineOut.logits[i])));
    expect(maxDiff).toBeLessThan(1e-4);
    expect(model).toBeInstanceOf(MiniGPT);
  });
});