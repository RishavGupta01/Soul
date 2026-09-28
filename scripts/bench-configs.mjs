import { initWeights, zeroGrad, zeroGradInPlace, lossAndGrad } from "./trainer-lib.mjs";

const configs = {
  "A d64 L3 F192 V384": { vocabSize: 384, dModel: 64, nLayers: 3, nHeads: 4, dFfn: 192, contextLength: 64 },
  "B d64 L3 F192 V320": { vocabSize: 320, dModel: 64, nLayers: 3, nHeads: 4, dFfn: 192, contextLength: 64 },
  "C d48 L2 F128 V320": { vocabSize: 320, dModel: 48, nLayers: 2, nHeads: 4, dFfn: 128, contextLength: 64 },
  "D d64 L4 F256 V320": { vocabSize: 320, dModel: 64, nLayers: 4, nHeads: 4, dFfn: 256, contextLength: 64 },
};

for (const [name, config] of Object.entries(configs)) {
  const W = initWeights(config, 5);
  const G = zeroGrad(W);
  const seq = Array.from({ length: 65 }, (_, i) => (i * 17) % config.vocabSize);
  const batch = 8;
  // warmup
  for (let i = 0; i < 2; i++) {
    zeroGradInPlace(G);
    for (let b = 0; b < batch; b++) lossAndGrad(W, G, seq);
  }
  const N = 10;
  const t = performance.now();
  for (let i = 0; i < N; i++) {
    zeroGradInPlace(G);
    for (let b = 0; b < batch; b++) lossAndGrad(W, G, seq);
  }
  const ms = (performance.now() - t) / N;
  const stepsPerHour = 3600000 / ms;
  const params =
    config.vocabSize * config.dModel +
    config.contextLength * config.dModel +
    config.nLayers *
      (3 * config.dModel * config.dModel +
        config.dModel * config.dModel +
        config.dModel * config.dFfn +
        config.dFfn * config.dModel);
  console.log(
    `${name.padEnd(22)} ${ms.toFixed(0).padStart(5)}ms/step  ~${Math.round(stepsPerHour / 1000)}k steps/hr  ${(
      params / 1000
    ).toFixed(0)}k params`
  );
}