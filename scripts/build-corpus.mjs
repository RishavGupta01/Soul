/**
 * Builds a bundled, license-safe training corpus about how language models
 * work. Authored/generated text, no third-party content. Narrow domain on
 * purpose: a tiny model stays coherent when the subject is focused, so the
 * browser demo reads as "the model understands this world."
 *
 * Output: scripts/corpus/llm-primer.txt
 */
import { writeFileSync, mkdirSync } from "node:fs";
import { dirname, join } from "node:path";
import { fileURLToPath } from "node:url";

const __dirname = dirname(fileURLToPath(import.meta.url));
const OUT_DIR = join(__dirname, "corpus");
const OUT = join(OUT_DIR, "llm-primer.txt");

const subjects = [
  "language model", "transformer", "token", "embedding", "attention head",
  "residual stream", "feed forward network", "layer norm", "vocabulary",
  "probability distribution", "context window", "training corpus", "gradient",
  "optimizer", "parameter", "neural network", "word vector", "softmax",
  "decoder block", "key matrix", "value matrix", "query vector",
];

const verbs = [
  "learns", "predicts", "computes", "represents", "transforms", "encodes",
  "captures", "updates", "aligns", "projects", "attends to", "summarizes",
];

const objects = [
  "the next token", "the meaning of a sentence", "the structure of language",
  "a small piece of text", "a pattern in the data", "the relationships between words",
  "the probability of the following word", "a compact vector of numbers",
  "the context on the left", "the position of every token", "a hidden state",
];

const clauses = [
  "one step at a time", "with no memory of the future", "by looking at what came before",
  "in a single forward pass", "by repeating the same block many times",
  "using only numbers", "without any rules written by hand",
  "through gradient descent", "across millions of examples",
];

function cap(s) {
  return s.charAt(0).toUpperCase() + s.slice(1);
}

function sentence(rng) {
  const pick = (arr) => arr[Math.floor(rng() * arr.length)];
  const templates = [
    () => `The ${pick(subjects)} ${pick(verbs)} ${pick(objects)} ${pick(clauses)}.`,
    () => `A ${pick(subjects)} ${pick(verbs)} ${pick(objects)}.`,
    () => `When the ${pick(subjects)} ${pick(verbs)} ${pick(objects)}, ${pick(clauses)}, the model becomes better at prediction.`,
    () => `Every ${pick(subjects)} is just a list of numbers, and each number ${pick(verbs)} ${pick(objects)}.`,
    () => `The ${pick(subjects)} ${pick(verbs)} ${pick(objects)} because the training data showed it many examples.`,
    () => `In a transformer, the ${pick(subjects)} ${pick(verbs)} ${pick(objects)} ${pick(clauses)}.`,
    () => `Think of the ${pick(subjects)} as a way to ${pick(verbs)} ${pick(objects)}.`,
    () => `The ${pick(subjects)} does not store facts like a database, it ${pick(verbs)} ${pick(objects)}.`,
    () => `Because the ${pick(subjects)} ${pick(verbs)} ${pick(objects)}, the next word can be guessed with high confidence.`,
    () => `Nobody writes the ${pick(subjects)} by hand; it is discovered ${pick(clauses)}.`,
  ];
  return cap(pick(templates)());
}

const topics = [
  "The model reads text as tokens, not as words. A token can be a whole word, a piece of a word, or a single character, and the tokenizer decides where the cuts are.",
  "Each token becomes a vector of numbers. Similar tokens end up with similar vectors, so the distance between vectors carries meaning.",
  "Position matters. The same word in a different place means something different, so the model adds a positional signal to every token vector.",
  "Attention lets every token look at the tokens before it. The query asks a question, the key offers an answer, and the value carries the information forward.",
  "A residual stream runs through the whole network. Each block adds its small update to the stream instead of replacing it, which keeps the signal alive across many layers.",
  "The feed forward network expands the vector, applies a nonlinearity, and compresses it again. Most of the parameters live here.",
  "Layer normalization keeps the numbers at a stable scale so the network does not explode or collapse during training.",
  "After the last block, the model produces a score for every token in the vocabulary. The softmax turns those scores into probabilities.",
  "Sampling turns probabilities into a single choice. Temperature makes the distribution flatter or sharper, and top k or top p cut off the unlikely tail.",
  "The chosen token is appended to the input and the whole process repeats. This loop is what people call generation.",
  "Training compares the predicted distribution with the true next token and measures the surprise. The loss is the average surprise over the batch.",
  "Backpropagation sends the error backward through every operation, and the optimizer nudges each parameter to reduce the loss a little.",
  "The key value cache stores the attention keys and values from earlier positions so the model does not recompute them on every step.",
  "Quantization stores weights with fewer bits. A model that used four bytes per weight can shrink to one byte or half a byte with little loss in quality.",
  "Large models are split across many devices. Tensor parallelism cuts each layer, pipeline parallelism cuts the stack of layers, and data parallelism cuts the batch.",
  "Scaling laws describe how loss falls as compute, data, and parameters grow. The curve is smooth, which makes performance predictable.",
  "Fine tuning teaches a trained model a narrower task. Instruction tuning shows it how to follow directions, and preference tuning shows it which answers people prefer.",
  "A reward model scores answers, and the policy learns to produce answers that score higher, while a penalty keeps it from drifting too far from the original model.",
  "Batching groups many requests together so the hardware stays busy. The same weights serve every item in the batch.",
  "A context window is the amount of text the model can hold at once. Everything outside the window is invisible to the attention mechanism.",
];

function buildCorpus() {
  let seed = 1337;
  const rng = () => {
    seed = (seed * 1664525 + 1013904223) >>> 0;
    return seed / 4294967296;
  };

  const parts = [];
  parts.push("A Primer on How Language Models Work");
  parts.push("");
  for (const t of topics) {
    parts.push(t);
    parts.push("");
    for (let i = 0; i < 14; i++) {
      parts.push(sentence(rng));
    }
    parts.push("");
  }
  // A soft repetition pass so the tokenizer sees common phrases many times.
  for (let round = 0; round < 6; round++) {
    for (const t of topics) parts.push(t);
    parts.push("");
  }
  return parts.join("\n");
}

mkdirSync(OUT_DIR, { recursive: true });
const corpus = buildCorpus();
writeFileSync(OUT, corpus, "utf8");
console.log(`wrote ${OUT} — ${(corpus.length / 1024).toFixed(1)} KB, ${corpus.length} chars`);