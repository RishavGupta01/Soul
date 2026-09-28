import { describe, it, expect } from "vitest";
import { BPETokenizer, trainBPE, applyMerges } from "./tokenizer";

describe("trainBPE", () => {
  it("learns merges from repeated text", () => {
    const merges = trainBPE("abababababab", 260);
    expect(merges.length).toBeGreaterThan(0);
    // most frequent pair is (a,b) -> first merge
    expect(merges[0]).toEqual([97, 98]);
  });

  it("returns at most vocabSize - 256 merges", () => {
    const merges = trainBPE("hello world hello world", 258);
    expect(merges.length).toBeLessThanOrEqual(2);
  });
});

describe("applyMerges", () => {
  it("collapses a repeated pair to a single id", () => {
    const rank = new Map<string, number>([["97,98", 0]]);
    const out = applyMerges([97, 98, 97, 98], rank);
    expect(out).toEqual([256, 256]);
  });
});

describe("BPETokenizer round-trip", () => {
  const merges = trainBPE("the quick brown fox jumps over the lazy dog. ".repeat(20), 400);
  const tok = new BPETokenizer(merges);

  it("encodes and decodes ASCII text back to the original", () => {
    const text = "the quick brown fox";
    expect(tok.decode(tok.encode(text))).toBe(text);
  });

  it("round-trips arbitrary unicode via byte fallback", () => {
    const text = "héllo → 世界 🌍";
    expect(tok.decode(tok.encode(text))).toBe(text);
  });

  it("compresses repeated words into fewer tokens than bytes", () => {
    const text = "the quick brown fox jumps over the lazy dog";
    expect(tok.encode(text).length).toBeLessThan(new TextEncoder().encode(text).length);
  });

  it("reports a vocab size of 256 + merges", () => {
    expect(tok.vocabSize).toBe(256 + merges.length);
  });

  it("resolveToken expands back to bytes", () => {
    const ids = tok.encode("the");
    const bytes: number[] = [];
    for (const id of ids) bytes.push(...tok.decodeToken(id));
    expect(new TextDecoder().decode(Uint8Array.from(bytes))).toBe("the");
  });

  it("serializes and restores", () => {
    const restored = BPETokenizer.fromJSON(tok.toJSON());
    expect(restored.decode(restored.encode("fox"))).toBe("fox");
    expect(restored.vocabSize).toBe(tok.vocabSize);
  });
});