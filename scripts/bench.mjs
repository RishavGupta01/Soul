import { initWeights, zeroGrad, lossAndGrad, forwardTrain } from "./trainer-lib.mjs";

const config = { vocabSize: 384, dModel: 64, nLayers: 3, nHeads: 4, dFfn: 192, contextLength: 64 };
const W = initWeights(config, 1);
const tokens = Array.from({ length: 48 }, (_, i) => (i * 7) % 384);

// warmup
for (let i = 0; i < 3; i++) lossAndGrad(W, zeroGrad(W), tokens);

const N = 30;
let t = performance.now();
for (let i = 0; i < N; i++) forwardTrain(W, tokens);
const fwd = (performance.now() - t) / N;

t = performance.now();
for (let i = 0; i < N; i++) {
  const G = zeroGrad(W);
  lossAndGrad(W, G, tokens);
}
const both = (performance.now() - t) / N;

t = performance.now();
for (let i = 0; i < N; i++) zeroGrad(W);
const zg = (performance.now() - t) / N;

console.log(`forward:        ${fwd.toFixed(1)} ms`);
console.log(`fwd+bwd+zerograd: ${both.toFixed(1)} ms`);
console.log(`zeroGrad alone:  ${zg.toFixed(2)} ms`);
console.log(`backward ~=      ${(both - fwd - zg).toFixed(1)} ms`);