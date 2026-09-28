import { BPETokenizer, type TokenizerJSON } from "./tokenizer";
import { MiniGPT, modelFromJSON } from "./model";
import type { ModelWeights } from "./types";

export interface TrainingMetrics {
  /** [step, train loss, val loss] sampled across the live run */
  history: [number, number, number][];
  firstVal: number;
  bestVal: number;
  lastVal: number;
  steps: number;
  merges: number;
  sample: string;
  running: boolean;
  vocabSize: number;
  config: ModelWeights["config"];
}

export type TrainingPoint = { step: number; train: number; val: number };

export function trainingPoints(m: TrainingMetrics | null | undefined): TrainingPoint[] {
  if (!m || !Array.isArray(m.history)) return [];
  return m.history.map(([step, train, val]) => ({ step, train, val }));
}

export interface Runtime {
  model: MiniGPT;
  tokenizer: BPETokenizer;
  metrics: TrainingMetrics | null;
  vocabJson: TokenizerJSON;
}

let runtime: Runtime | null = null;

async function fetchJson<T>(path: string): Promise<T> {
  const res = await fetch(path, { cache: "no-cache" });
  if (!res.ok) throw new Error(`${path} -> ${res.status}`);
  return (await res.json()) as T;
}

/**
 * Load trained weights. Base path is derived from the document location so
 * the same build works on `/` and on a GitHub Pages project sub-path.
 */
export async function loadRuntime(base = baseFromLocation()): Promise<Runtime> {
  if (runtime) return runtime;
  const [w, v, m] = await Promise.all([
    fetchJson<Parameters<typeof modelFromJSON>[0]>(`${base}model/weights.json`),
    fetchJson<TokenizerJSON>(`${base}model/vocab.json`),
    fetchJson<TrainingMetrics>(`${base}model/training-metrics.json`).catch(() => null),
  ]);
  const tokenizer = BPETokenizer.fromJSON(v);
  runtime = { model: modelFromJSON(w), tokenizer, metrics: m, vocabJson: v };
  return runtime;
}

export function getRuntime(): Runtime | null {
  return runtime;
}

export function setRuntime(r: Runtime | null): void {
  runtime = r;
}

/** "/repo/" when served from GitHub Pages, "/" otherwise. */
export function baseFromLocation(): string {
  const { pathname } = window.location;
  const asPath = import.meta.env.BASE_URL;
  if (asPath && asPath !== "/") return asPath;
  // fallback: strip trailing file name
  return pathname.endsWith("/") ? pathname : pathname.replace(/[^/]*$/, "");
}
