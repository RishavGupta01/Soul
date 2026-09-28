/**
 * Trains the Obsidian Brain micro-GPT and writes:
 *   public/model/weights.json  (embeddings + block weights)
 *   public/model/vocab.json    (BPE merges)
 *   public/model/training-metrics.json
 *
 *   node scripts/train-micro.mjs [--minutes N] [--config small|default]
 *
 * Time-boxed: trains until the wall-clock budget is spent, checkpointing so
 * the site always has usable weights even if the run is interrupted.
 */
import { readFileSync, writeFileSync, mkdirSync, existsSync } from "node:fs";
import { dirname, join } from "node:path";
import { fileURLToPath } from "node:url";
import {
  DEFAULT_CONFIG,
  initWeights,
  zeroGrad,
  zeroGradInPlace,
  lossAndGrad,
  makeAdam,
  trainBPE,
  makeEncoder,
  decodeIds,
  serializeWeights,
  makeRng,
  forwardTrain,
} from "./trainer-lib.mjs";

const __dirname = dirname(fileURLToPath(import.meta.url));
const ROOT = join(__dirname, "..");
const CORPUS = join(__dirname, "corpus", "llm-primer.txt");
const OUT_DIR = join(ROOT, "public", "model");

const args = process.argv.slice(2);
const getArg = (flag, fallback) => {
  const i = args.indexOf(flag);
  return i >= 0 ? args[i + 1] : fallback;
};
const MINUTES = Number(getArg("--minutes", "18"));
const PRESET = getArg("--config", "default");
const SEQ_LEN = Number(getArg("--seq", "64"));
const BATCH = Number(getArg("--batch", "8"));
const LR = Number(getArg("--lr", "0.003"));
const EVAL_EVERY = 200;
const CKPT_EVERY = 400;

const configs = {
  small: { ...DEFAULT_CONFIG, vocabSize: 384, dModel: 48, nLayers: 2, dFfn: 128 },
  default: { ...DEFAULT_CONFIG, vocabSize: 384, dModel: 64, nLayers: 3, dFfn: 192 },
};
const config = configs[PRESET] ?? configs.default;

function loadCorpus() {
  if (!existsSync(CORPUS)) {
    throw new Error(`corpus missing: ${CORPUS}\nrun: node scripts/build-corpus.mjs`);
  }
  return readFileSync(CORPUS, "utf8");
}

/** Fixed validation sequences so loss is comparable across steps. */
function buildValSet(ids, n = 24) {
  const rng = makeRng(9999);
  const out = [];
  for (let i = 0; i < n; i++) {
    const maxStart = Math.max(1, ids.length - SEQ_LEN - 1);
    const start = Math.floor(rng() * maxStart);
    out.push(ids.slice(start, start + SEQ_LEN + 1));
  }
  return out;
}

function buildBatches(ids, batch, seed) {
  const rng = makeRng(seed);
  const out = [];
  for (let b = 0; b < batch; b++) {
    const maxStart = Math.max(1, ids.length - SEQ_LEN - 1);
    const start = Math.floor(rng() * maxStart);
    const seq = ids.slice(start, start + SEQ_LEN + 1);
    out.push(seq.length === SEQ_LEN + 1 ? seq : ids.slice(0, SEQ_LEN + 1));
  }
  return out;
}

const scaleGrad = (node, s) => {
  if (node.kind === "mat" || node.kind === "vec") {
    for (let i = 0; i < node.d.length; i++) node.d[i] *= s;
    return;
  }
  for (const k of Object.keys(node)) {
    if (k === "config") continue;
    if (Array.isArray(node[k])) node[k].forEach((c) => scaleGrad(c, s));
    else scaleGrad(node[k], s);
  }
};

function evalLoss(W, valSet) {
  let loss = 0;
  for (const seq of valSet) {
    const G = zeroGrad(W);
    loss += lossAndGrad(W, G, seq).loss;
  }
  return loss / valSet.length;
}

/** Greedy-ish sample for eyeballing coherence. */
function generate(W, encode, prompt, maxTokens, temperature) {
  const { contextLength, vocabSize } = W.config;
  const ids = encode(prompt);
  const rng = makeRng(7);
  const out = [];
  for (let step = 0; step < maxTokens; step++) {
    const ctx = ids.slice(-contextLength);
    const { logits } = forwardTrain(W, ctx);
    const T = ctx.length;
    const o = (T - 1) * vocabSize;
    let max = -Infinity;
    for (let v = 0; v < vocabSize; v++) max = Math.max(max, logits[o + v]);
    let sum = 0;
    const probs = new Float64Array(vocabSize);
    for (let v = 0; v < vocabSize; v++) {
      const e = Math.exp((logits[o + v] - max) / temperature);
      probs[v] = e;
      sum += e;
    }
    let r = rng() * sum;
    let chosen = 0;
    for (let v = 0; v < vocabSize; v++) {
      r -= probs[v];
      if (r <= 0) {
        chosen = v;
        break;
      }
    }
    ids.push(chosen);
    out.push(chosen);
  }
  return decodeIds(out, mergesRef);
}

let mergesRef = [];

function save(W, metrics) {
  mkdirSync(OUT_DIR, { recursive: true });
  writeFileSync(join(OUT_DIR, "weights.json"), JSON.stringify(serializeWeights(W, 4)));
  writeFileSync(
    join(OUT_DIR, "vocab.json"),
    JSON.stringify({ merges: mergesRef, vocabSize: W.config.vocabSize })
  );
  writeFileSync(join(OUT_DIR, "training-metrics.json"), JSON.stringify(metrics, null, 2));
}

function main() {
  const t0 = Date.now();
  const text = loadCorpus();
  console.log(`corpus ${(text.length / 1024).toFixed(1)} KB`);

  const merges = trainBPE(text, config.vocabSize);
  mergesRef = merges;
  const encode = makeEncoder(merges);
  config.vocabSize = 256 + merges.length;
  let tEnc = Date.now();
  const ids = encode(text);
  console.log(
    `bpe ${merges.length} merges, vocab ${config.vocabSize}, ${ids.length} tokens ` +
      `(${(text.length / ids.length).toFixed(2)} chars/token, encode ${Date.now() - tEnc}ms)`
  );

  const W = initWeights(config, 42);
  const G = zeroGrad(W);
  const adam = makeAdam(W, G, LR);
  const valSet = buildValSet(ids);

  let firstVal = null;
  let lastVal = null;
  let step = 0;
  let bestVal = Infinity;
  const history = [];
  const budgetMs = MINUTES * 60 * 1000;

  while (Date.now() - t0 < budgetMs) {
    step++;
    const batches = buildBatches(ids, BATCH, step * 7919);
    zeroGradInPlace(G);
    let loss = 0;
    for (const seq of batches) loss += lossAndGrad(W, G, seq).loss;
    loss /= BATCH;
    scaleGrad(G, 1 / BATCH);
    const gnorm = adam.magnitude();
    adam.update();

    if (step === 1 || step % EVAL_EVERY === 0) {
      const val = evalLoss(W, valSet);
      if (firstVal === null) firstVal = val;
      lastVal = val;
      history.push([step, loss, val]);
      const mins = ((Date.now() - t0) / 60000).toFixed(1);
      console.log(
        `step ${String(step).padStart(5)}  train ${loss.toFixed(4)}  val ${val.toFixed(4)}  ` +
          `|g| ${gnorm.toExponential(2)}  (${mins}m)`
      );
      if (val < bestVal) bestVal = val;
    }

    if (step % CKPT_EVERY === 0) {
      const sample = generate(W, encode, "The token ", 80, 0.8);
      save(W, {
        steps: step,
        firstVal,
        bestVal,
        lastVal,
        history,
        vocabSize: config.vocabSize,
        merges: merges.length,
        config,
        sample,
        running: true,
      });
    }
  }

  const sample = generate(W, encode, "The model ", 100, 0.7);
  console.log("\n--- sample ---\n" + sample + "\n--------------");
  save(W, {
    steps: step,
    firstVal,
    bestVal,
    lastVal,
    history,
    vocabSize: config.vocabSize,
    merges: merges.length,
    config,
    sample,
    weightsKB: Math.round(readFileSync(join(OUT_DIR, "weights.json")).length / 1024),
    seconds: Math.round((Date.now() - t0) / 1000),
    running: false,
  });
  console.log(
    `saved  val ${firstVal?.toFixed(3)} -> ${lastVal?.toFixed(3)}  steps ${step}  ` +
      `${((Date.now() - t0) / 1000).toFixed(0)}s`
  );
}

main();