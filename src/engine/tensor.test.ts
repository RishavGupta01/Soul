import { describe, it, expect } from "vitest";
import {
  matFrom,
  matmul,
  linear,
  transpose,
  layerNorm,
  softmax,
  gelu,
  argmax,
  add,
  dot,
  makeRng,
  gaussian,
  vec,
} from "./tensor";

describe("matmul", () => {
  it("multiplies 2x3 by 3x2 with exact values", () => {
    const a = matFrom(2, 3, [1, 2, 3, 4, 5, 6]);
    const b = matFrom(3, 2, [7, 8, 9, 10, 11, 12]);
    const c = matmul(a, b);
    expect(c.rows).toBe(2);
    expect(c.cols).toBe(2);
    expect(Array.from(c.data)).toEqual([58, 64, 139, 154]);
  });

  it("throws on incompatible shapes", () => {
    expect(() => matmul(matFrom(2, 3, [1, 2, 3, 4, 5, 6]), matFrom(2, 2, [1, 2, 3, 4]))).toThrow();
  });
});

describe("linear", () => {
  it("computes W x + b", () => {
    const W = matFrom(2, 2, [1, 0, 0, 1]);
    const b = vec(2).fill(1);
    const y = linear(W, Float64Array.from([3, 4]), b);
    expect(Array.from(y)).toEqual([4, 5]);
  });
});

describe("transpose", () => {
  it("swaps rows and cols", () => {
    const t = transpose(matFrom(2, 3, [1, 2, 3, 4, 5, 6]));
    expect(t.rows).toBe(3);
    expect(t.cols).toBe(2);
    expect(Array.from(t.data)).toEqual([1, 4, 2, 5, 3, 6]);
  });
});

describe("layerNorm", () => {
  it("normalizes to zero mean and unit variance (w=1,b=0)", () => {
    const x = Float64Array.from([1, 2, 3, 4, 5]);
    const out = layerNorm(x, vec(5).fill(1), vec(5));
    const mean = Array.from(out).reduce((s, v) => s + v, 0) / out.length;
    const variance = Array.from(out).reduce((s, v) => s + (v - mean) ** 2, 0) / out.length;
    expect(Math.abs(mean)).toBeLessThan(1e-6);
    expect(Math.abs(variance - 1)).toBeLessThan(1e-3);
  });
});

describe("softmax", () => {
  it("rows sum to 1", () => {
    const p = softmax(Float64Array.from([1, 2, 3]));
    expect(Array.from(p).reduce((s, v) => s + v, 0)).toBeCloseTo(1, 10);
  });

  it("temperature 0 returns argmax one-hot", () => {
    const p = softmax(Float64Array.from([0.1, 5, 0.2]), 0);
    expect(Array.from(p)).toEqual([0, 1, 0]);
  });

  it("higher temperature flattens the distribution", () => {
    const sharp = softmax(Float64Array.from([1, 2, 3]), 0.5);
    const flat = softmax(Float64Array.from([1, 2, 3]), 5);
    expect(Math.max(...Array.from(sharp))).toBeGreaterThan(Math.max(...Array.from(flat)));
  });
});

describe("gelu", () => {
  it("is zero at zero", () => {
    expect(gelu(0)).toBeCloseTo(0, 10);
  });
  it("is negative for small negatives", () => {
    expect(gelu(-0.5)).toBeLessThan(0);
  });
});

describe("argmax", () => {
  it("finds the largest index", () => {
    expect(argmax(Float64Array.from([1, 9, 3]))).toBe(1);
  });
});

describe("add / dot", () => {
  it("adds elementwise and dots", () => {
    expect(Array.from(add(Float64Array.from([1, 2]), Float64Array.from([3, 4])))).toEqual([4, 6]);
    expect(dot(Float64Array.from([1, 2, 3]), Float64Array.from([4, 5, 6]))).toBe(32);
  });
});

describe("seeded rng", () => {
  it("is deterministic for the same seed", () => {
    const a = makeRng(42);
    const b = makeRng(42);
    const seqA = [a(), a(), a()];
    const seqB = [b(), b(), b()];
    expect(seqA).toEqual(seqB);
  });
  it("produces values in [0,1)", () => {
    const r = makeRng(7);
    for (let i = 0; i < 1000; i++) {
      const v = r();
      expect(v).toBeGreaterThanOrEqual(0);
      expect(v).toBeLessThan(1);
    }
  });
  it("gaussian has roughly zero mean", () => {
    const r = makeRng(3);
    let s = 0;
    const n = 5000;
    for (let i = 0; i < n; i++) s += gaussian(r);
    expect(Math.abs(s / n)).toBeLessThan(0.1);
  });
});