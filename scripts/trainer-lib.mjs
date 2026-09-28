/**
 * Standalone trainer for the Obsidian Brain micro-GPT.
 *
 * Dependency-free JS. The forward math mirrors src/engine/*.ts; a test
 * (scripts/trainer.test.mjs) asserts both implementations agree on logits so
 * the browser demo and the trained weights cannot silently diverge.
 *
 * Features: byte-level BPE, causal multi-head attention, GELU FFN, tied
 * embeddings, cross-entropy loss, manual backprop, Adam.
 */

/* ------------------------------------------------------------------ */
/* BPE                                                                 */
/* ------------------------------------------------------------------ */

export function trainBPE(text, vocabSize) {
  const target = Math.max(0, vocabSize - 256);
  let symbols = Array.from(new TextEncoder().encode(text));
  const merges = [];
  for (let m = 0; m < target; m++) {
    if (symbols.length < 2) break;
    const counts = new Map();
    for (let i = 0; i < symbols.length - 1; i++) {
      const key = symbols[i] + "," + symbols[i + 1];
      counts.set(key, (counts.get(key) ?? 0) + 1);
    }
    let bestKey = "";
    let bestCount = 1;
    for (const [key, count] of counts) {
      if (count > bestCount) {
        bestCount = count;
        bestKey = key;
      }
    }
    if (!bestKey) break;
    const [a, b] = bestKey.split(",").map(Number);
    merges.push([a, b]);
    const newId = 256 + merges.length - 1;
    const merged = [];
    for (let i = 0; i < symbols.length; i++) {
      if (i < symbols.length - 1 && symbols[i] === a && symbols[i + 1] === b) {
        merged.push(newId);
        i++;
      } else merged.push(symbols[i]);
    }
    symbols = merged;
  }
  return merges;
}

export function makeEncoder(merges) {
  return (text) => {
    let s = Array.from(new TextEncoder().encode(text));
    for (let rank = 0; rank < merges.length; rank++) {
      const [a, b] = merges[rank];
      const newId = 256 + rank;
      // Replace all non-overlapping (a,b) occurrences with newId.
      let changed = false;
      for (let i = 0; i < s.length - 1; ) {
        if (s[i] === a && s[i + 1] === b) {
          s[i] = newId;
          s.splice(i + 1, 1);
          changed = true;
          i++;
        } else {
          i++;
        }
      }
      if (!changed && s.length < 2) break;
    }
    return s;
  };
}

export function decodeIds(ids, merges) {
  const bytes = [];
  const expand = (id) => {
    if (id < 256) bytes.push(id);
    else {
      const [a, b] = merges[id - 256];
      expand(a);
      expand(b);
    }
  };
  for (const id of ids) expand(id);
  return new TextDecoder().decode(Uint8Array.from(bytes));
}

/* ------------------------------------------------------------------ */
/* Randomness                                                          */
/* ------------------------------------------------------------------ */

export function makeRng(seed) {
  let a = seed >>> 0;
  return () => {
    a |= 0;
    a = (a + 0x6d2b79f5) | 0;
    let t = Math.imul(a ^ (a >>> 15), 1 | a);
    t = (t + Math.imul(t ^ (t >>> 7), 61 | t)) ^ t;
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  };
}

function randn(rng) {
  let u = 0;
  let v = 0;
  while (u === 0) u = rng();
  while (v === 0) v = rng();
  return Math.sqrt(-2 * Math.log(u)) * Math.cos(2 * Math.PI * v);
}

/* ------------------------------------------------------------------ */
/* Shapes + param registry                                             */
/* ------------------------------------------------------------------ */

const mat = (r, c) => ({ kind: "mat", r, c, d: new Float64Array(r * c) });
const vec = (n) => ({ kind: "vec", n, d: new Float64Array(n) });

function randMat(r, c, rng, std) {
  const m = mat(r, c);
  for (let i = 0; i < m.d.length; i++) m.d[i] = randn(rng) * std;
  return m;
}
function onesVec(n) {
  const v = vec(n);
  v.d.fill(1);
  return v;
}

export const DEFAULT_CONFIG = {
  vocabSize: 512,
  dModel: 64,
  nLayers: 3,
  nHeads: 4,
  dFfn: 192,
  contextLength: 64,
};

export function initWeights(config, seed = 42) {
  const rng = makeRng(seed);
  const { vocabSize: V, dModel: D, nLayers: L, dFfn: F, contextLength: C } = config;
  const std = 0.02;
  const layers = [];
  for (let i = 0; i < L; i++) {
    layers.push({
      ln1W: onesVec(D),
      ln1B: vec(D),
      qkvW: randMat(3 * D, D, rng, std),
      qkvB: vec(3 * D),
      projW: randMat(D, D, rng, std),
      projB: vec(D),
      ln2W: onesVec(D),
      ln2B: vec(D),
      fcW: randMat(F, D, rng, std),
      fcB: vec(F),
      fc2W: randMat(D, F, rng, std),
      fc2B: vec(D),
    });
  }
  return {
    config,
    wte: randMat(V, D, rng, std),
    wpe: randMat(C, D, rng, std),
    lnFW: onesVec(D),
    lnFB: vec(D),
    layers,
  };
}

/** Zeroed gradient tree mirroring the weights structure. */
export function zeroGrad(W) {
  const z = (p) => {
    if (p.kind === "mat") return mat(p.r, p.c);
    if (p.kind === "vec") return vec(p.n);
    throw new Error("bad param");
  };
  return {
    wte: z(W.wte),
    wpe: z(W.wpe),
    lnFW: z(W.lnFW),
    lnFB: z(W.lnFB),
    layers: W.layers.map((l) => ({
      ln1W: z(l.ln1W),
      ln1B: z(l.ln1B),
      qkvW: z(l.qkvW),
      qkvB: z(l.qkvB),
      projW: z(l.projW),
      projB: z(l.projB),
      ln2W: z(l.ln2W),
      ln2B: z(l.ln2B),
      fcW: z(l.fcW),
      fcB: z(l.fcB),
      fc2W: z(l.fc2W),
      fc2B: z(l.fc2B),
    })),
  };
}

/** Iterate weight/grad pairs in a deterministic order. */
export function paramPairs(W, G, cb) {
  const walk = (a, b) => {
    if (a.kind === "mat" || a.kind === "vec") {
      cb(a, b);
      return;
    }
    for (const k of Object.keys(a)) {
      if (k === "config") continue;
      if (Array.isArray(a[k])) {
        a[k].forEach((item, i) => walk(item, b[k][i]));
      } else {
        walk(a[k], b[k]);
      }
    }
  };
  walk(W, G);
}

/* ------------------------------------------------------------------ */
/* Forward + backward                                                  */
/* ------------------------------------------------------------------ */

const gelu = (x) =>
  0.5 * x * (1 + Math.tanh(Math.sqrt(2 / Math.PI) * (x + 0.044715 * x * x * x)));
const geluGrad = (x) => {
  const c = Math.sqrt(2 / Math.PI);
  const u = c * (x + 0.044715 * x * x * x);
  const th = Math.tanh(u);
  return 0.5 * (1 + th) + 0.5 * x * (1 - th * th) * c * (1 + 3 * 0.044715 * x * x);
};

function linear(T, W, x, b) {
  const r = W.r;
  const c = W.c;
  const wd = W.d;
  const bd = b ? b.d : null;
  const out = new Float64Array(T * r);
  for (let t = 0; t < T; t++) {
    const xo = t * c;
    const oo = t * r;
    for (let i = 0; i < r; i++) {
      let s = bd ? bd[i] : 0;
      const wi = i * c;
      for (let j = 0; j < c; j++) s += wd[wi + j] * x[xo + j];
      out[oo + i] = s;
    }
  }
  return out;
}

function layerNormFwd(T, x, w, b) {
  const D = w.n;
  const out = new Float64Array(T * D);
  const xhat = new Float64Array(T * D);
  const inv = new Float64Array(T);
  const wd = w.d;
  const bd = b.d;
  for (let t = 0; t < T; t++) {
    const o = t * D;
    let mean = 0;
    for (let d = 0; d < D; d++) mean += x[o + d];
    mean /= D;
    let vs = 0;
    for (let d = 0; d < D; d++) {
      const diff = x[o + d] - mean;
      vs += diff * diff;
    }
    const invStd = 1 / Math.sqrt(vs / D + 1e-5);
    inv[t] = invStd;
    for (let d = 0; d < D; d++) {
      const xh = (x[o + d] - mean) * invStd;
      xhat[o + d] = xh;
      out[o + d] = xh * wd[d] + bd[d];
    }
  }
  return { out, xhat, inv };
}

function layerNormBwd(T, dy, xhat, inv, w, dx, dW, dB) {
  const D = w.n;
  const wd = w.d;
  const dWd = dW.d;
  const dBd = dB.d;
  for (let t = 0; t < T; t++) {
    const o = t * D;
    let meanDy = 0;
    let meanDyX = 0;
    for (let d = 0; d < D; d++) {
      const g = dy[o + d];
      dWd[d] += g * xhat[o + d];
      dBd[d] += g;
      meanDy += g;
      meanDyX += g * xhat[o + d];
    }
    meanDy /= D;
    meanDyX /= D;
    const invStd = inv[t];
    for (let d = 0; d < D; d++) {
      dx[o + d] += wd[d] * invStd * (dy[o + d] - meanDy - xhat[o + d] * meanDyX);
    }
  }
}

export function forwardTrain(W, tokens) {
  const { dModel: D, nLayers: L, nHeads: H, dFfn: F, vocabSize: V, contextLength: C } =
    W.config;
  const T = tokens.length;
  const dh = D / H;

  const x0 = new Float64Array(T * D);
  for (let t = 0; t < T; t++) {
    const tok = tokens[t] % V;
    const te = tok * D;
    const pe = (t % C) * D;
    for (let d = 0; d < D; d++) x0[t * D + d] = W.wte.d[te + d] + W.wpe.d[pe + d];
  }

  const cache = [];
  let x = x0;
  for (let l = 0; l < L; l++) {
    const w = W.layers[l];
    const { out: n1, xhat: xhat1, inv: inv1 } = layerNormFwd(T, x, w.ln1W, w.ln1B);
    const qkv = linear(T, w.qkvW, n1, w.qkvB);

    const probs = [];
    const attnOut = new Float64Array(T * D);
    for (let h = 0; h < H; h++) {
      const P = new Float64Array(T * T);
      for (let i = 0; i < T; i++) {
        const row = new Float64Array(T);
        let max = -Infinity;
        for (let j = 0; j <= i; j++) {
          let s = 0;
          for (let d = 0; d < dh; d++) {
            s += qkv[i * 3 * D + h * dh + d] * qkv[j * 3 * D + D + h * dh + d];
          }
          s /= Math.sqrt(dh);
          row[j] = s;
          if (s > max) max = s;
        }
        let sum = 0;
        for (let j = 0; j <= i; j++) {
          row[j] = Math.exp(row[j] - max);
          sum += row[j];
        }
        for (let j = 0; j <= i; j++) P[i * T + j] = row[j] / sum;
      }
      probs.push(P);
      for (let i = 0; i < T; i++) {
        for (let d = 0; d < dh; d++) {
          let acc = 0;
          for (let j = 0; j <= i; j++) acc += P[i * T + j] * qkv[j * 3 * D + 2 * D + h * dh + d];
          attnOut[i * D + h * dh + d] = acc;
        }
      }
    }

    const proj = linear(T, w.projW, attnOut, w.projB);
    const x1 = new Float64Array(T * D);
    for (let i = 0; i < T * D; i++) x1[i] = x[i] + proj[i];

    const { out: n2, xhat: xhat2, inv: inv2 } = layerNormFwd(T, x1, w.ln2W, w.ln2B);
    const up = linear(T, w.fcW, n2, w.fcB);
    const act = new Float64Array(T * F);
    for (let i = 0; i < T * F; i++) act[i] = gelu(up[i]);
    const down = linear(T, w.fc2W, act, w.fc2B);
    const x2 = new Float64Array(T * D);
    for (let i = 0; i < T * D; i++) x2[i] = x1[i] + down[i];

    cache.push({ x, n1, xhat1, inv1, qkv, probs, attnOut, proj, x1, n2, xhat2, inv2, up, act, down });
    x = x2;
  }

  const { out: fn, xhat: xhatF, inv: invF } = layerNormFwd(T, x, W.lnFW, W.lnFB);
  const logits = new Float64Array(T * V);
  const wted = W.wte.d;
  for (let t = 0; t < T; t++) {
    const fo = t * D;
    const lo = t * V;
    for (let v = 0; v < V; v++) {
      let s = 0;
      const base = v * D;
      for (let d = 0; d < D; d++) s += wted[base + d] * fn[fo + d];
      logits[lo + v] = s;
    }
  }
  return { x, fn, xhatF, invF, logits, cache, T };
}

export function lossAndGrad(W, G, tokens) {
  const { dModel: D, vocabSize: V, nLayers: L, nHeads: H, dFfn: F } = W.config;
  const { fn, xhatF, invF, logits, cache, T } = forwardTrain(W, tokens);
  const N = T - 1;
  const dh = D / H;

  const probs = new Float64Array(T * V);
  let loss = 0;
  for (let t = 0; t < N; t++) {
    const o = t * V;
    let max = -Infinity;
    for (let v = 0; v < V; v++) if (logits[o + v] > max) max = logits[o + v];
    let sum = 0;
    for (let v = 0; v < V; v++) {
      const e = Math.exp(logits[o + v] - max);
      probs[o + v] = e;
      sum += e;
    }
    for (let v = 0; v < V; v++) probs[o + v] /= sum;
    loss += -Math.log(Math.max(probs[o + tokens[t + 1]], 1e-9));
  }
  loss /= N;

  const dlogits = new Float64Array(T * V);
  for (let t = 0; t < N; t++) {
    const o = t * V;
    for (let v = 0; v < V; v++) dlogits[o + v] = probs[o + v] / N;
    dlogits[o + tokens[t + 1]] -= 1 / N;
  }

  const dfn = new Float64Array(T * D);
  const wted = W.wte.d;
  const gwted = G.wte.d;
  for (let t = 0; t < T; t++) {
    const fo = t * D;
    const lo = t * V;
    for (let v = 0; v < V; v++) {
      const g = dlogits[lo + v];
      if (g === 0) continue;
      const base = v * D;
      for (let d = 0; d < D; d++) {
        dfn[fo + d] += g * wted[base + d];
        gwted[base + d] += g * fn[fo + d];
      }
    }
  }

  const dx = new Float64Array(T * D);
  layerNormBwd(T, dfn, xhatF, invF, W.lnFW, dx, G.lnFW, G.lnFB);

  for (let l = L - 1; l >= 0; l--) {
    const w = W.layers[l];
    const g = G.layers[l];
    const c = cache[l];

    // down = act @ fc2W^T
    const dact = new Float64Array(T * F);
    for (let t = 0; t < T; t++) {
      for (let j = 0; j < D; j++) {
        const gd = dx[t * D + j];
        if (gd === 0) continue;
        for (let f = 0; f < F; f++) {
          dact[t * F + f] += gd * w.fc2W.d[j * F + f];
          g.fc2W.d[j * F + f] += gd * c.act[t * F + f];
        }
      }
    }
    for (let t = 0; t < T; t++) for (let j = 0; j < D; j++) g.fc2B.d[j] += dx[t * D + j];

    const dup = new Float64Array(T * F);
    for (let i = 0; i < T * F; i++) dup[i] = dact[i] * geluGrad(c.up[i]);

    const dn2 = new Float64Array(T * D);
    for (let t = 0; t < T; t++) {
      for (let f = 0; f < F; f++) {
        const gu = dup[t * F + f];
        if (gu === 0) continue;
        for (let d = 0; d < D; d++) {
          dn2[t * D + d] += gu * w.fcW.d[f * D + d];
          g.fcW.d[f * D + d] += gu * c.n2[t * D + d];
        }
      }
    }
    for (let t = 0; t < T; t++) for (let f = 0; f < F; f++) g.fcB.d[f] += dup[t * F + f];

    // x2 = x1 + down, so dL/dx1 starts with the incoming residual gradient,
    // then accumulates the path through LN2.
    const dx1 = new Float64Array(T * D);
    for (let i = 0; i < T * D; i++) dx1[i] = dx[i];
    layerNormBwd(T, dn2, c.xhat2, c.inv2, w.ln2W, dx1, g.ln2W, g.ln2B);

    // x1 = x + proj : dproj = dL/dx1 ; dx gains the direct residual path
    const dproj = dx1;
    const dxPrev = new Float64Array(T * D);
    for (let i = 0; i < T * D; i++) dxPrev[i] = dx1[i];

    const dattn = new Float64Array(T * D);
    for (let t = 0; t < T; t++) {
      for (let j = 0; j < D; j++) {
        const gp = dproj[t * D + j];
        if (gp === 0) continue;
        for (let d = 0; d < D; d++) {
          dattn[t * D + d] += gp * w.projW.d[j * D + d];
          g.projW.d[j * D + d] += gp * c.attnOut[t * D + d];
        }
      }
    }
    for (let t = 0; t < T; t++) for (let j = 0; j < D; j++) g.projB.d[j] += dproj[t * D + j];

    const dqkv = new Float64Array(T * 3 * D);
    for (let h = 0; h < H; h++) {
      const P = c.probs[h];
      const dP = new Float64Array(T * T);
      for (let i = 0; i < T; i++) {
        for (let j = 0; j <= i; j++) {
          let s = 0;
          for (let d = 0; d < dh; d++) {
            const gi = dattn[i * D + h * dh + d];
            s += gi * c.qkv[j * 3 * D + 2 * D + h * dh + d];
            dqkv[j * 3 * D + 2 * D + h * dh + d] += P[i * T + j] * gi;
          }
          dP[i * T + j] = s;
        }
      }
      for (let i = 0; i < T; i++) {
        let dot = 0;
        for (let j = 0; j <= i; j++) dot += P[i * T + j] * dP[i * T + j];
        for (let j = 0; j <= i; j++) {
          const dS = P[i * T + j] * (dP[i * T + j] - dot);
          for (let d = 0; d < dh; d++) {
            const inv = 1 / Math.sqrt(dh);
            dqkv[i * 3 * D + h * dh + d] += dS * c.qkv[j * 3 * D + D + h * dh + d] * inv;
            dqkv[j * 3 * D + D + h * dh + d] += dS * c.qkv[i * 3 * D + h * dh + d] * inv;
          }
        }
      }
    }

    const dn1 = new Float64Array(T * D);
    for (let t = 0; t < T; t++) {
      for (let i = 0; i < 3 * D; i++) {
        const gq = dqkv[t * 3 * D + i];
        if (gq === 0) continue;
        g.qkvB.d[i] += gq;
        for (let d = 0; d < D; d++) {
          dn1[t * D + d] += gq * w.qkvW.d[i * D + d];
          g.qkvW.d[i * D + d] += gq * c.n1[t * D + d];
        }
      }
    }

    const dxIn = new Float64Array(T * D);
    layerNormBwd(T, dn1, c.xhat1, c.inv1, w.ln1W, dxIn, g.ln1W, g.ln1B);
    for (let i = 0; i < T * D; i++) dx[i] = dxPrev[i] + dxIn[i];
  }

  // After the loop dx holds d(x0) = d(embedding)
  for (let t = 0; t < T; t++) {
    const tok = tokens[t] % V;
    const te = tok * D;
    const pe = (t % W.config.contextLength) * D;
    for (let d = 0; d < D; d++) {
      G.wte.d[te + d] += dx[t * D + d];
      G.wpe.d[pe + d] += dx[t * D + d];
    }
  }

  return { loss, probs };
}

/* ------------------------------------------------------------------ */
/* Adam                                                                */
/* ------------------------------------------------------------------ */

/** Zero an existing gradient tree in place (avoids reallocating). */
export function zeroGradInPlace(G) {
  const walk = (node) => {
    if (node.kind === "mat" || node.kind === "vec") {
      node.d.fill(0);
      return;
    }
    for (const k of Object.keys(node)) {
      if (k === "config") continue;
      if (Array.isArray(node[k])) node[k].forEach(walk);
      else walk(node[k]);
    }
  };
  walk(G);
}

export function makeAdam(W, G, lr = 3e-3, b1 = 0.9, b2 = 0.95, eps = 1e-8) {
  const pairs = [];
  paramPairs(W, G, (val, grad) => pairs.push({ val, grad }));
  const state = pairs.map(() => ({
    m: new Float64Array(0),
    v: new Float64Array(0),
  }));
  let step = 0;
  return {
    get step() {
      return step;
    },
    update() {
      step++;
      const c1 = 1 - Math.pow(b1, step);
      const c2 = 1 - Math.pow(b2, step);
      pairs.forEach(({ val, grad }, i) => {
        const d = val.d;
        const gd = grad.d;
        if (state[i].m.length !== d.length) {
          state[i].m = new Float64Array(d.length);
          state[i].v = new Float64Array(d.length);
        }
        const m = state[i].m;
        const v = state[i].v;
        for (let j = 0; j < d.length; j++) {
          const g = gd[j];
          m[j] = b1 * m[j] + (1 - b1) * g;
          v[j] = b2 * v[j] + (1 - b2) * g * g;
          d[j] -= (lr * (m[j] / c1)) / (Math.sqrt(v[j] / c2) + eps);
        }
      });
    },
    magnitude() {
      let sum = 0;
      let count = 0;
      for (const { grad } of pairs) {
        for (let j = 0; j < grad.d.length; j++) {
          sum += grad.d[j] * grad.d[j];
          count++;
        }
      }
      return Math.sqrt(sum / Math.max(1, count));
    },
  };
}

/* ------------------------------------------------------------------ */
/* Serialization (matches src/engine/model.ts modelFromJSON)           */
/* ------------------------------------------------------------------ */

export function serializeWeights(W, precision = 4) {
  const r = (arr) => Array.from(arr, (x) => Number(x.toFixed(precision)));
  return {
    config: W.config,
    wte: r(W.wte.d),
    wpe: r(W.wpe.d),
    lnFW: r(W.lnFW.d),
    lnFB: r(W.lnFB.d),
    layers: W.layers.map((l) => ({
      ln1W: r(l.ln1W.d),
      ln1B: r(l.ln1B.d),
      qkvW: r(l.qkvW.d),
      qkvB: r(l.qkvB.d),
      projW: r(l.projW.d),
      projB: r(l.projB.d),
      ln2W: r(l.ln2W.d),
      ln2B: r(l.ln2B.d),
      fcW: r(l.fcW.d),
      fcB: r(l.fcB.d),
      fc2W: r(l.fc2W.d),
      fc2B: r(l.fc2B.d),
    })),
  };
}