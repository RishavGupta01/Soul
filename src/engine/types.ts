import type { Mat, Vec } from "./tensor";

export interface ModelConfig {
  vocabSize: number;
  dModel: number;
  nLayers: number;
  nHeads: number;
  dFfn: number;
  contextLength: number;
}

export interface LayerWeights {
  ln1W: Vec;
  ln1B: Vec;
  /** Combined QKV projection: (3*dModel) x dModel */
  qkvW: Mat;
  qkvB: Vec;
  /** Output projection: dModel x dModel */
  projW: Mat;
  projB: Vec;
  ln2W: Vec;
  ln2B: Vec;
  /** FFN up: dFfn x dModel */
  fcW: Mat;
  fcB: Vec;
  /** FFN down: dModel x dFfn */
  fc2W: Mat;
  fc2B: Vec;
}

export interface ModelWeights {
  config: ModelConfig;
  /** Token embedding: vocabSize x dModel (also used as lm_head, tied). */
  wte: Mat;
  /** Positional embedding: contextLength x dModel. */
  wpe: Mat;
  lnFW: Vec;
  lnFB: Vec;
  layers: LayerWeights[];
}

/** Event emitted per layer per forward pass — the visualization substrate. */
export interface LayerStep {
  layer: number;
  /** attention scores, [head][queryPos][keyPos] */
  attnScores: number[][][];
  /** post-attention-heads softmax probabilities, same shape as attnScores */
  attnProbs: number[][][];
  /** hidden state entering the block (residual stream), length dModel */
  hidden: number[];
  /** post-FFN delta before residual add */
  ffnDelta: number[];
  /** torch-style shape labels for the UI */
  shapes: { qkv: string; scores: string; ffn: string };
}

export interface ForwardResult {
  /** Logits for the LAST position: length vocabSize */
  logits: number[];
  steps: LayerStep[];
  /** All-position logits (t x vocabSize) when needed for training. */
  allLogits: number[][];
}

export interface KVLayer {
  /** cached keys per position, flattened [pos][dModel] */
  k: Vec[];
  v: Vec[];
}