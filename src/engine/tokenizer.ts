/**
 * Byte-level BPE tokenizer (GPT-2 style) with an explicit, inspectable
 * merge list so the UI can animate merges.
 */

export interface Merge {
  pair: [number, number];
  id: number;
}

export interface TokenizerJSON {
  merges: [number, number][];
  vocabSize: number;
}

const BYTE_RANGE = 256;

function bytesToText(text: string): number[] {
  return Array.from(new TextEncoder().encode(text));
}

function textToBytes(bytes: number[]): string {
  return new TextDecoder().decode(Uint8Array.from(bytes));
}

/**
 * Greedy BPE merge application. Given the ordered merge list, repeatedly
 * merge the highest-priority applicable pair. Exposed so the UI can replay
 * merges step by step.
 */
export function applyMerges(ids: number[], mergeRank: Map<string, number>): number[] {
  let symbols = ids.slice();
  let changed = true;
  while (changed && symbols.length > 1) {
    changed = false;
    let bestRank = Infinity;
    let bestIndex = -1;
    for (let i = 0; i < symbols.length - 1; i++) {
      const key = symbols[i] + "," + symbols[i + 1];
      const rank = mergeRank.get(key);
      if (rank !== undefined && rank < bestRank) {
        bestRank = rank;
        bestIndex = i;
      }
    }
    if (bestIndex >= 0) {
      symbols = [
        ...symbols.slice(0, bestIndex),
        mergeRank.get(symbols[bestIndex] + "," + symbols[bestIndex + 1]) === undefined
          ? symbols[bestIndex]
          : 256 + bestRank,
        ...symbols.slice(bestIndex + 2),
      ];
      changed = true;
    }
  }
  return symbols;
}

export class BPETokenizer {
  /** merge pair "a,b" -> new token id (256 + rank) */
  private mergeRank = new Map<string, number>();
  private mergeList: Merge[] = [];
  readonly vocabSize: number;

  constructor(merges: [number, number][]) {
    this.mergeList = merges.map((pair, rank) => ({ pair, id: BYTE_RANGE + rank }));
    merges.forEach((pair, rank) => {
      this.mergeRank.set(pair[0] + "," + pair[1], rank);
    });
    this.vocabSize = BYTE_RANGE + merges.length;
  }

  get merges(): Merge[] {
    return this.mergeList;
  }

  encode(text: string): number[] {
    const base = bytesToText(text);
    // Convert raw bytes to ids, then apply merges. Byte ids are just 0..255.
    return applyMerges(base, this.mergeRank);
  }

  /** Encode and return the merge trajectory for animation. */
  encodeWithSteps(text: string): { initial: number[]; final: number[]; steps: Merge[] } {
    const initial = bytesToText(text);
    const final = applyMerges(initial, this.mergeRank);
    const steps = this.mergeList.filter((m) => {
      const key = m.pair[0] + "," + m.pair[1];
      return this.mergeRank.get(key) !== undefined && final.includes(m.id);
    });
    return { initial, final, steps };
  }

  decode(ids: number[]): string {
    const bytes: number[] = [];
    for (const id of ids) {
      const b = this.decodeToken(id);
      bytes.push(...b);
    }
    return textToBytes(bytes);
  }

  /** Resolve a token id to the raw bytes it expands to. */
  decodeToken(id: number): number[] {
    if (id < BYTE_RANGE) return [id];
    const merge = this.mergeList[id - BYTE_RANGE];
    if (!merge) return [];
    return [...this.decodeToken(merge.pair[0]), ...this.decodeToken(merge.pair[1])];
  }

  toJSON(): TokenizerJSON {
    return { merges: this.mergeList.map((m) => m.pair), vocabSize: this.vocabSize };
  }

  static fromJSON(json: TokenizerJSON): BPETokenizer {
    return new BPETokenizer(json.merges);
  }
}

/**
 * Train a byte-level BPE: count adjacent pairs, merge the most frequent,
 * repeat until vocabSize merges exist or no pair repeats.
 */
export function trainBPE(text: string, vocabSize: number): [number, number][] {
  const target = Math.max(0, vocabSize - BYTE_RANGE);
  let symbols = bytesToText(text);
  const merges: [number, number][] = [];
  const nextId = new Map<number, number>(); // reserved for forced-merge tracking

  for (let m = 0; m < target; m++) {
    if (symbols.length < 2) break;
    const counts = new Map<string, number>();
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
    const newId = BYTE_RANGE + merges.length - 1;
    nextId.set(newId, newId);
    const merged: number[] = [];
    for (let i = 0; i < symbols.length; i++) {
      if (i < symbols.length - 1 && symbols[i] === a && symbols[i + 1] === b) {
        merged.push(newId);
        i++;
      } else {
        merged.push(symbols[i]);
      }
    }
    symbols = merged;
  }
  return merges;
}