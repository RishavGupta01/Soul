import {
  geluVec,
  layerNorm,
  linear,
  vec,
  type Vec,
  type Mat,
} from "./tensor";
import { causalSelfAttention } from "./attention";
import type { ForwardResult, LayerStep, ModelWeights } from "./types";

/**
 * A minimal, dependency-free GPT. The same class powers:
 *  - the offline training script (Node),
 *  - the browser forward pass used by every visualization,
 *  - the "real math" step events that drive the animations.
 *
 * Inference only here (training grads live in scripts/train-micro.mjs).
 */
export class MiniGPT {
  constructor(public w: ModelWeights) {}

  get config() {
    return this.w.config;
  }

  /**
   * Forward pass.
   * @param ids   token ids for this call (for the first call, the prompt)
   * @param startPos absolute position offset (KV-cache aware)
   * @param cache optional KV cache; when provided, only new positions are computed
   */
  forward(ids: number[], startPos = 0, cache?: { k: Vec[]; v: Vec[] }[]): ForwardResult {
    const { vocabSize, dModel, nLayers, nHeads, dFfn, contextLength } = this.w.config;
    const t = ids.length;

    // Embedding + positional
    const hidden: Vec[] = [];
    for (let i = 0; i < t; i++) {
      const pos = startPos + i;
      const tok = ids[i];
      if (tok < 0 || tok >= vocabSize) throw new Error(`token id ${tok} out of range`);
      const p = pos % contextLength;
      const h = vec(dModel);
      const te = tok * dModel;
      const pe = p * dModel;
      for (let d = 0; d < dModel; d++) h[d] = this.w.wte.data[te + d] + this.w.wpe.data[pe + d];
      hidden.push(h);
    }

    const steps: LayerStep[] = [];

    for (let layer = 0; layer < nLayers; layer++) {
      const L = this.w.layers[layer];
      const residualIn = hidden.map((h) => h.slice());

      // --- LayerNorm 1 ---
      const normed1 = hidden.map((h) => layerNorm(h, L.ln1W, L.ln1B));

      // --- Causal multi-head self attention ---
      const layerCache = cache ? cache[layer] : undefined;
      const att = causalSelfAttention(normed1, L.qkvW, L.qkvB, nHeads, {
        k: layerCache?.k ?? [],
        v: layerCache?.v ?? [],
        offset: layerCache ? startPos : 0,
      });

      // --- Output projection ---
      const attnOut: Vec[] = att.context.map((c) => linear(L.projW, c, L.projB));

      // --- Residual add ---
      const afterAttn: Vec[] = residualIn.map((r, i) => {
        const out = r.slice();
        for (let d = 0; d < dModel; d++) out[d] += attnOut[i][d];
        return out;
      });

      // --- LayerNorm 2 + FFN (dModel -> dFfn -> dModel) ---
      const normed2 = afterAttn.map((h) => layerNorm(h, L.ln2W, L.ln2B));
      const ffnDelta: Vec[] = [];
      const afterFfn: Vec[] = [];
      for (let i = 0; i < t; i++) {
        const up = linear(L.fcW, normed2[i], L.fcB); // dFfn
        const act = geluVec(up);
        const down = linear(L.fc2W, act, L.fc2B); // dModel
        ffnDelta.push(down);
        const out = afterAttn[i].slice();
        for (let d = 0; d < dModel; d++) out[d] += down[d];
        afterFfn.push(out);
      }

      hidden.length = 0;
      hidden.push(...afterFfn);

      // Emit the step event for the FIRST query position only for compactness
      // of the UI stream, but keep full matrices so scenes can choose.
      steps.push({
        layer,
        attnScores: att.scores,
        attnProbs: att.probs,
        hidden: Array.from(residualIn[residualIn.length - 1] ?? vec(dModel)),
        ffnDelta: Array.from(ffnDelta[ffnDelta.length - 1] ?? vec(dModel)),
        shapes: {
          qkv: `[${t}, ${3 * dModel}]`,
          scores: `[${nHeads}, ${t}, ${att.scores[0]?.[0]?.length ?? t}]`,
          ffn: `[${t}, ${dFfn}]`,
        },
      });
    }

    // Final layer norm
    const finalNorm = hidden.map((h) => layerNorm(h, this.w.lnFW, this.w.lnFB));

    // LM head (weights tied to token embeddings)
    const allLogits: number[][] = [];
    for (let i = 0; i < t; i++) {
      const logits = vec(vocabSize);
      for (let v = 0; v < vocabSize; v++) {
        const base = v * dModel;
        let s = 0;
        for (let d = 0; d < dModel; d++) s += this.w.wte.data[base + d] * finalNorm[i][d];
        logits[v] = s;
      }
      allLogits.push(Array.from(logits));
    }

    return {
      logits: allLogits[allLogits.length - 1],
      allLogits,
      steps,
    };
  }
}

/** Load a weights JSON blob into a MiniGPT. */
export function modelFromJSON(json: {
  config: ModelWeights["config"];
  wte: number[];
  wpe: number[];
  lnFW: number[];
  lnFB: number[];
  layers: {
    ln1W: number[];
    ln1B: number[];
    qkvW: number[];
    qkvB: number[];
    projW: number[];
    projB: number[];
    ln2W: number[];
    ln2B: number[];
    fcW: number[];
    fcB: number[];
    fc2W: number[];
    fc2B: number[];
  }[];
}): MiniGPT {
  const { config } = json;
  const m = (
    vals: number[],
    rows: number,
    cols: number
  ): Mat => ({ rows, cols, data: Float64Array.from(vals) });
  const v = (vals: number[]): Vec => Float64Array.from(vals);
  const d = config.dModel;
  return new MiniGPT({
    config,
    wte: m(json.wte, config.vocabSize, d),
    wpe: m(json.wpe, config.contextLength, d),
    lnFW: v(json.lnFW),
    lnFB: v(json.lnFB),
    layers: json.layers.map((L) => ({
      ln1W: v(L.ln1W),
      ln1B: v(L.ln1B),
      qkvW: m(L.qkvW, 3 * d, d),
      qkvB: v(L.qkvB),
      projW: m(L.projW, d, d),
      projB: v(L.projB),
      ln2W: v(L.ln2W),
      ln2B: v(L.ln2B),
      fcW: m(L.fcW, config.dFfn, d),
      fcB: v(L.fcB),
      fc2W: m(L.fc2W, d, config.dFfn),
      fc2B: v(L.fc2B),
    })),
  });
}