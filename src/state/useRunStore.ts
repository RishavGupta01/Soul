import { create } from "zustand";
import { persist } from "zustand/middleware";

export interface ChapterDef {
  id: string;
  index: number;
  act: 1 | 2;
  title: string;
  nav: string;
  route: string;
}

export const CHAPTERS: ChapterDef[] = [
  { id: "overture", index: 0, act: 1, title: "Overture", nav: "Think", route: "#/" },
  { id: "tokenization", index: 1, act: 1, title: "Tokenization", nav: "Tokens", route: "#/tokens" },
  { id: "embeddings", index: 2, act: 1, title: "Embeddings", nav: "Vectors", route: "#/embeddings" },
  { id: "positional", index: 3, act: 1, title: "Positional encoding", nav: "Position", route: "#/position" },
  { id: "block", index: 4, act: 1, title: "The transformer block", nav: "Block", route: "#/block" },
  { id: "attention", index: 4.5, act: 1, title: "Self-attention", nav: "Attention", route: "#/block/attention" },
  { id: "logits", index: 5, act: 1, title: "Logits → softmax", nav: "Softmax", route: "#/logits" },
  { id: "sampling", index: 6, act: 1, title: "Sampling", nav: "Sampling", route: "#/sampling" },
  { id: "loop", index: 7, act: 1, title: "The loop", nav: "Loop", route: "#/loop" },
  { id: "atlas", index: 8, act: 2, title: "The atlas", nav: "Atlas", route: "#/atlas" },
];

interface RunState {
  route: string;
  /** highest act I index unlocked (by scroll reach) */
  reached: number;
  fastForward: boolean;
  atlasUnlocked: boolean;
  setRoute: (r: string) => void;
  reach: (idx: number) => void;
  setFastForward: (v: boolean) => void;
}

export const useRunStore = create<RunState>()(
  persist(
    (set) => ({
      route: "#/",
      reached: 0,
      fastForward: false,
      atlasUnlocked: false,
      setRoute: (route) =>
        set((s) => ({
          route,
          atlasUnlocked: s.atlasUnlocked || route.startsWith("#/atlas"),
        })),
      reach: (idx) =>
        set((s) => ({
          reached: Math.max(s.reached, idx),
          atlasUnlocked: s.atlasUnlocked || idx >= 7,
        })),
      setFastForward: (fastForward) => set({ fastForward }),
    }),
    {
      name: "obsidian-run-v2",
      partialize: (s) => ({ reached: s.reached, atlasUnlocked: s.atlasUnlocked }),
    }
  )
);
