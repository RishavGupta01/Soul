import { create } from "zustand";
import type { TrainingMetrics } from "../engine/loader";

export type ModelStatus = "loading" | "ready" | "fallback" | "error";

export interface TokenEvent {
  id: number;
  text: string;
  step: number;
}

export interface GeneratedStep {
  /** index in the full sequence */
  pos: number;
  tokenId: number;
  tokenText: string;
  /** probability distribution for this position (sparse top-64 for UI) */
  top: { id: number; text: string; p: number }[];
  entropy: number;
  /** which sampler knobs were used */
  temperature: number;
}

export type TrainingMetricsData = TrainingMetrics;

interface ModelState {
  status: ModelStatus;
  statusDetail: string;
  vocabSize: number;
  metrics: TrainingMetricsData | null;
  prompt: string;
  output: string;
  running: boolean;
  steps: GeneratedStep[];
  seqLen: number;
  temperature: number;
  topK: number;
  topP: number;
  seed: number;
  setStatus: (s: ModelStatus, detail?: string) => void;
  setMetrics: (m: TrainingMetricsData | null) => void;
  setPrompt: (p: string) => void;
  setKnob: (k: "temperature" | "topK" | "topP" | "seed", v: number) => void;
  patch: (p: Partial<ModelState>) => void;
}

export const useModelStore = create<ModelState>((set) => ({
  status: "loading",
  statusDetail: "fetching weights",
  vocabSize: 0,
  metrics: null,
  prompt: "The attention mechanism ",
  output: "",
  running: false,
  steps: [],
  seqLen: 0,
  temperature: 0.8,
  topK: 40,
  topP: 0.95,
  seed: 1,
  setStatus: (status, statusDetail) => set({ status, ...(statusDetail ? { statusDetail } : {}) }),
  setMetrics: (metrics) => set({ metrics }),
  setPrompt: (prompt) => set({ prompt }),
  setKnob: (k, v) => set({ [k]: v } as Partial<ModelState>),
  patch: (p) => set(p),
}));
