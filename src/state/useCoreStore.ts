import { create } from "zustand";

export type CorePhase =
  | "idle"
  | "tokenize"
  | "embed"
  | "position"
  | "attend"
  | "ffn"
  | "logits"
  | "sample"
  | "loop";

export interface CoreAttention {
  /** head -> rows of weights for the currently focused query */
  head: number;
  rows: number[][];
  /** token strings aligned with columns */
  tokens: string[];
  query: number;
}

interface CoreState {
  phase: CorePhase;
  /** 0..1 progress within the current step (drives the sweep) */
  progress: number;
  attention: CoreAttention | null;
  generating: boolean;
  /** active logit candidates for the "probability ridge" moment */
  candidates: { text: string; p: number }[];
  /** last sampled token, briefly highlighted */
  sampled: string | null;
  pulse: number;
  set: (p: Partial<CoreState>) => void;
}

export const useCoreStore = create<CoreState>((set) => ({
  phase: "idle",
  progress: 0,
  attention: null,
  generating: false,
  candidates: [],
  sampled: null,
  pulse: 0,
  set: (p) => set(p),
}));
