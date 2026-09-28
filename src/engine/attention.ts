import { linear, matmul, softmax, type Mat, type Vec, vec, mat } from "./tensor";

export interface AttentionOutput {
  /** concatenated head outputs before output projection, length t*dModel */
  context: Vec[];
  scores: number[][][];
  probs: number[][][];
}

/**
 * Causal multi-head self-attention.
 *
 * @param x       per-position hidden states, length t, each dModel
 * @param qkvW    combined projection (3*dModel) x dModel
 * @param qkvB    combined bias (3*dModel)
 * @param nHeads  number of heads
 * @param cache   optional cached keys/values from previous positions
 */
export function causalSelfAttention(
  x: Vec[],
  qkvW: Mat,
  qkvB: Vec,
  nHeads: number,
  cache?: { k: Vec[]; v: Vec[]; offset: number }
): AttentionOutput & { k: Vec[]; v: Vec[] } {
  const t = x.length;
  const dModel = x[0].length;
  const dHead = dModel / nHeads;

  const q: Vec[] = [];
  const k: Vec[] = [];
  const v: Vec[] = [];

  for (let i = 0; i < t; i++) {
    const qkv = linear(qkvW, x[i], qkvB);
    q.push(qkv.slice(0, dModel));
    k.push(qkv.slice(dModel, 2 * dModel));
    v.push(qkv.slice(2 * dModel, 3 * dModel));
  }

  if (cache) {
    cache.k.push(...k);
    cache.v.push(...v);
  }
  const allK = cache ? cache.k : k;
  const allV = cache ? cache.v : v;
  const offset = cache ? cache.offset : 0;
  const keyLen = allK.length;

  const scores: number[][][] = [];
  const probs: number[][][] = [];

  for (let h = 0; h < nHeads; h++) {
    const layerScores: number[][] = [];
    const layerProbs: number[][] = [];
    for (let i = 0; i < t; i++) {
      const qi = i + offset;
      const row = vec(keyLen);
      for (let j = 0; j < keyLen; j++) {
        // causal: query at absolute position qi may attend to key j <= qi
        if (j > qi) {
          row[j] = -Infinity;
          continue;
        }
        let s = 0;
        for (let d = 0; d < dHead; d++) {
          s += q[i][h * dHead + d] * allK[j][h * dHead + d];
        }
        row[j] = s / Math.sqrt(dHead);
      }
      layerScores.push(Array.from(row));
      const p = softmax(row);
      layerProbs.push(Array.from(p));
    }
    scores.push(layerScores);
    probs.push(layerProbs);
  }

  const cleanContext: Vec[] = [];
  for (let i = 0; i < t; i++) {
    const out = vec(dModel);
    for (let h = 0; h < nHeads; h++) {
      for (let d = 0; d < dHead; d++) {
        let acc = 0;
        for (let j = 0; j < keyLen; j++) {
          acc += probs[h][i][j] * allV[j][h * dHead + d];
        }
        out[h * dHead + d] = acc;
      }
    }
    cleanContext.push(out);
  }

  return { context: cleanContext, scores, probs, k, v };
}

/** Same as above but returns a fresh result struct without mutating caches. */
export function attentionShapes(nHeads: number, t: number, keyLen: number): string {
  return `[${nHeads}, ${t}, ${keyLen}]`;
}

export function projection(W: Mat, x: Vec, b?: Vec): Vec {
  return linear(W, x, b);
}

export function combineHeads(heads: Vec[], dModel: number): Vec {
  const out = vec(dModel);
  for (const h of heads) for (let i = 0; i < h.length; i++) out[i] += h[i];
  return out;
}

export function zerosMat(rows: number, cols: number): Mat {
  return mat(rows, cols);
}

export function matmulProjection(a: Mat, b: Mat): Mat {
  return matmul(a, b);
}