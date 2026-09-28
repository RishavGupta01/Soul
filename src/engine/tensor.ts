/**
 * Minimal Float32/float64 tensor math for the micro-GPT.
 * Row-major flat arrays plus explicit shape metadata. No dependencies.
 */

export type Vec = Float64Array;
/** Row-major matrix: rows x cols. */
export interface Mat {
  rows: number;
  cols: number;
  data: Float64Array;
}

export function mat(rows: number, cols: number, fill = 0): Mat {
  return { rows, cols, data: new Float64Array(rows * cols).fill(fill) };
}

export function vec(n: number, fill = 0): Vec {
  return new Float64Array(n).fill(fill);
}

export function matFrom(rows: number, cols: number, values: number[]): Mat {
  if (values.length !== rows * cols) throw new Error("matFrom size mismatch");
  return { rows, cols, data: Float64Array.from(values) };
}

/* ------------------------------------------------------------------ */
/* Elementwise                                                         */
/* ------------------------------------------------------------------ */

export function add(a: Vec, b: Vec): Vec {
  const out = vec(a.length);
  for (let i = 0; i < a.length; i++) out[i] = a[i] + b[i];
  return out;
}

export function addInto(a: Vec, b: Vec): void {
  for (let i = 0; i < a.length; i++) a[i] += b[i];
}

export function sub(a: Vec, b: Vec): Vec {
  const out = vec(a.length);
  for (let i = 0; i < a.length; i++) out[i] = a[i] - b[i];
  return out;
}

export function scale(a: Vec, s: number): Vec {
  const out = vec(a.length);
  for (let i = 0; i < a.length; i++) out[i] = a[i] * s;
  return out;
}

/** In-place scale. */
export function scaleInto(a: Vec, s: number): void {
  for (let i = 0; i < a.length; i++) a[i] *= s;
}

export function mul(a: Vec, b: Vec): Vec {
  const out = vec(a.length);
  for (let i = 0; i < a.length; i++) out[i] = a[i] * b[i];
  return out;
}

export function dot(a: Vec, b: Vec): number {
  let s = 0;
  for (let i = 0; i < a.length; i++) s += a[i] * b[i];
  return s;
}

export function axpy(y: Vec, a: Vec, x: number): void {
  // y += a * x
  for (let i = 0; i < y.length; i++) y[i] += a[i] * x;
}

/* ------------------------------------------------------------------ */
/* Matrix                                                              */
/* ------------------------------------------------------------------ */

export function matmul(a: Mat, b: Mat): Mat {
  if (a.cols !== b.rows) throw new Error(`matmul shape ${a.cols} != ${b.rows}`);
  const out = mat(a.rows, b.cols);
  for (let i = 0; i < a.rows; i++) {
    const ai = i * a.cols;
    const oi = i * b.cols;
    for (let k = 0; k < a.cols; k++) {
      const aik = a.data[ai + k];
      if (aik === 0) continue;
      const bk = k * b.cols;
      for (let j = 0; j < b.cols; j++) out.data[oi + j] += aik * b.data[bk + j];
    }
  }
  return out;
}

/** y = W x + b, where W is rows x cols and x is cols. Returns rows. */
export function linear(W: Mat, x: Vec, b?: Vec): Vec {
  if (W.cols !== x.length) throw new Error("linear shape mismatch");
  const out = vec(W.rows);
  for (let i = 0; i < W.rows; i++) {
    let s = b ? b[i] : 0;
    const wi = i * W.cols;
    for (let j = 0; j < W.cols; j++) s += W.data[wi + j] * x[j];
    out[i] = s;
  }
  return out;
}

export function transpose(a: Mat): Mat {
  const out = mat(a.cols, a.rows);
  for (let i = 0; i < a.rows; i++)
    for (let j = 0; j < a.cols; j++) out.data[j * a.rows + i] = a.data[i * a.cols + j];
  return out;
}

/* ------------------------------------------------------------------ */
/* Activations & norms                                                 */
/* ------------------------------------------------------------------ */

export function layerNorm(x: Vec, w: Vec, b: Vec, eps = 1e-5): Vec {
  const n = x.length;
  let mean = 0;
  for (let i = 0; i < n; i++) mean += x[i];
  mean /= n;
  let varSum = 0;
  for (let i = 0; i < n; i++) {
    const d = x[i] - mean;
    varSum += d * d;
  }
  const invStd = 1 / Math.sqrt(varSum / n + eps);
  const out = vec(n);
  for (let i = 0; i < n; i++) out[i] = (x[i] - mean) * invStd * w[i] + b[i];
  return out;
}

export function gelu(x: number): number {
  // tanh approximation (used by GPT-2)
  return 0.5 * x * (1 + Math.tanh(Math.sqrt(2 / Math.PI) * (x + 0.044715 * x * x * x)));
}

export function geluVec(x: Vec): Vec {
  const out = vec(x.length);
  for (let i = 0; i < x.length; i++) out[i] = gelu(x[i]);
  return out;
}

/** Numerically stable softmax. temperature <= 0 falls back to argmax one-hot. */
export function softmax(x: Vec, temperature = 1): Vec {
  const n = x.length;
  const out = vec(n);
  if (temperature <= 0) {
    let best = 0;
    for (let i = 1; i < n; i++) if (x[i] > x[best]) best = i;
    out[best] = 1;
    return out;
  }
  let max = -Infinity;
  for (let i = 0; i < n; i++) if (x[i] > max) max = x[i];
  let sum = 0;
  for (let i = 0; i < n; i++) {
    const e = Math.exp((x[i] - max) / temperature);
    out[i] = e;
    sum += e;
  }
  for (let i = 0; i < n; i++) out[i] /= sum;
  return out;
}

export function argmax(x: Vec): number {
  let best = 0;
  for (let i = 1; i < x.length; i++) if (x[i] > x[best]) best = i;
  return best;
}

export function relu(x: number): number {
  return x > 0 ? x : 0;
}

/* ------------------------------------------------------------------ */
/* Randomness                                                          */
/* ------------------------------------------------------------------ */

/** Mulberry32 seeded PRNG — deterministic across runs. */
export function makeRng(seed: number): () => number {
  let a = seed >>> 0;
  return function () {
    a |= 0;
    a = (a + 0x6d2b79f5) | 0;
    let t = Math.imul(a ^ (a >>> 15), 1 | a);
    t = (t + Math.imul(t ^ (t >>> 7), 61 | t)) ^ t;
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  };
}

/** Box-Muller gaussian. */
export function gaussian(rng: () => number): number {
  let u = 0;
  let v = 0;
  while (u === 0) u = rng();
  while (v === 0) v = rng();
  return Math.sqrt(-2 * Math.log(u)) * Math.cos(2 * Math.PI * v);
}

export function randnMat(rows: number, cols: number, rng: () => number, std = 0.02): Mat {
  const m = mat(rows, cols);
  for (let i = 0; i < m.data.length; i++) m.data[i] = gaussian(rng) * std;
  return m;
}

export function randnVec(n: number, rng: () => number, std = 0.02): Vec {
  const v = vec(n);
  for (let i = 0; i < n; i++) v[i] = gaussian(rng) * std;
  return v;
}