import { create } from "zustand";

export interface AtlasNode {
  id: string;
  label: string;
  sub: string;
  x: number;
  y: number;
  vx: number;
  vy: number;
  fx?: number;
  fy?: number;
  r: number;
  kind: "core" | "train" | "infra";
}

export interface AtlasEdge {
  from: string;
  to: string;
}

interface AtlasState {
  nodes: AtlasNode[];
  edges: AtlasEdge[];
  activeId: string | null;
  panned: { x: number; y: number };
  zoom: number;
  select: (id: string | null) => void;
  panTo: (p: { x: number; y: number }) => void;
  setZoom: (z: number) => void;
}

const initial: AtlasNode[] = [
  { id: "core", label: "Inference", sub: "the forward pass", x: 0, y: 0, vx: 0, vy: 0, r: 46, kind: "core" },
  { id: "training", label: "Pretraining", sub: "next-token loss", x: -260, y: -120, vx: 0, vy: 0, r: 38, kind: "train" },
  { id: "rlhf", label: "Post-training", sub: "SFT + RLHF", x: -300, y: 140, vx: 0, vy: 0, r: 34, kind: "train" },
  { id: "kv", label: "KV cache", sub: "memory of the past", x: 250, y: -140, vx: 0, vy: 0, r: 32, kind: "core" },
  { id: "quant", label: "Quantization", sub: "fp32 → int4", x: 280, y: 130, vx: 0, vy: 0, r: 32, kind: "infra" },
  { id: "batch", label: "Batching", sub: "throughput", x: 90, y: 250, vx: 0, vy: 0, r: 28, kind: "infra" },
  { id: "gpu", label: "Hardware", sub: "GPU die anatomy", x: -60, y: -270, vx: 0, vy: 0, r: 34, kind: "infra" },
  { id: "parallel", label: "Parallelism", sub: "tensor · pipeline · data", x: -340, y: -20, vx: 0, vy: 0, r: 32, kind: "infra" },
  { id: "scaling", label: "Scaling laws", sub: "compute · data · params", x: 40, y: -170, vx: 0, vy: 0, r: 30, kind: "train" },
  { id: "glossary", label: "Glossary", sub: "every term, defined", x: 340, y: -30, vx: 0, vy: 0, r: 26, kind: "core" },
];

const edges: AtlasEdge[] = [
  { from: "core", to: "training" },
  { from: "core", to: "rlhf" },
  { from: "core", to: "kv" },
  { from: "core", to: "quant" },
  { from: "core", to: "batch" },
  { from: "training", to: "parallel" },
  { from: "training", to: "scaling" },
  { from: "rlhf", to: "glossary" },
  { from: "quant", to: "batch" },
  { from: "gpu", to: "parallel" },
  { from: "gpu", to: "training" },
];

export const useAtlasStore = create<AtlasState>((set) => ({
  nodes: initial.map((n) => ({ ...n })),
  edges,
  activeId: null,
  panned: { x: 0, y: 0 },
  zoom: 1,
  select: (activeId) => set({ activeId }),
  panTo: (panned) => set({ panned }),
  setZoom: (zoom) => set({ zoom: Math.min(2.4, Math.max(0.5, zoom)) }),
}));
