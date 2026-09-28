import { create } from "zustand";

type Depth = "intuition" | "deep";

interface DisclosureState {
  /** global default depth */
  depth: Depth;
  /** per-term overrides: term key -> depth */
  terms: Record<string, Depth>;
  toggle: () => void;
  setTerm: (key: string, d: Depth) => void;
  term: (key: string) => Depth;
}

export const useDisclosureStore = create<DisclosureState>((set, get) => ({
  depth: "intuition",
  terms: {},
  toggle: () => set((s) => ({ depth: s.depth === "intuition" ? "deep" : "intuition" })),
  setTerm: (key, d) => set((s) => ({ terms: { ...s.terms, [key]: d } })),
  term: (key) => get().terms[key] ?? get().depth,
}));
