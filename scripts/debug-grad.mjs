import { initWeights, zeroGrad, lossAndGrad, paramPairs } from "./trainer-lib.mjs";

const tiny = { vocabSize: 24, dModel: 8, nLayers: 2, nHeads: 2, dFfn: 16, contextLength: 12 };
const W = initWeights(tiny, 7);
const tokens = [1, 2, 3, 4, 5, 6, 7];
const G = zeroGrad(W);
lossAndGrad(W, G, tokens);

const names = [];
const pairs = [];
paramPairs(W, G, (val, grad) => {
  pairs.push({ val, grad });
});
const labels = [
  "wte", "wpe", "lnFW", "lnFB",
  "L0.ln1W", "L0.ln1B", "L0.qkvW", "L0.qkvB", "L0.projW", "L0.projB",
  "L0.ln2W", "L0.ln2B", "L0.fcW", "L0.fcB", "L0.fc2W", "L0.fc2B",
  "L1.ln1W", "L1.ln1B", "L1.qkvW", "L1.qkvB", "L1.projW", "L1.projB",
  "L1.ln2W", "L1.ln2B", "L1.fcW", "L1.fcB", "L1.fc2W", "L1.fc2B",
];

const eps = 1e-5;
const numericLoss = () => lossAndGrad(W, zeroGrad(W), tokens).loss;
const rows = [];
pairs.forEach(({ val, grad }, idx) => {
  let maxRel = 0;
  let worst = null;
  for (let j = 0; j < Math.min(5, val.d.length); j++) {
    const orig = val.d[j];
    val.d[j] = orig + eps;
    const lp = numericLoss();
    val.d[j] = orig - eps;
    const lm = numericLoss();
    val.d[j] = orig;
    const num = (lp - lm) / (2 * eps);
    const an = grad.d[j];
    const rel = Math.abs(num - an) / Math.max(1e-4, Math.abs(num) + Math.abs(an));
    if (rel > maxRel) {
      maxRel = rel;
      worst = { j, num, an };
    }
  }
  rows.push({ label: labels[idx] ?? `p${idx}`, maxRel, worst });
});

rows.sort((a, b) => b.maxRel - a.maxRel);
for (const r of rows.slice(0, 14)) {
  console.log(
    `${r.label.padEnd(10)} rel=${r.maxRel.toFixed(4).padStart(8)}  num=${r.worst.num.toExponential(3)}  an=${r.worst.an.toExponential(3)}`
  );
}
console.log("\nworst param:", rows[0].label);