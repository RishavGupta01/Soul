import { useEffect, useMemo, useRef, useState } from "react";
import { useAtlasStore, type AtlasNode } from "../state/useAtlasStore";
import { getRuntime, trainingPoints } from "../engine/loader";
import { useRunStore } from "../state/useRunStore";
import { useRuntime } from "../engine/useRuntime";

type DetailId = AtlasNode["id"] | "glossary";

export function Atlas() {
  const { nodes, edges, activeId, select, setZoom, zoom, panned, panTo } = useAtlasStore();
  const [hover, setHover] = useState<string | null>(null);
  const svgRef = useRef<SVGSVGElement>(null);
  const reach = useRunStore((s) => s.reach);
  const rt = useRuntime();

  useEffect(() => {
    reach(7);
  }, [reach]);

  // detail panel pills jump between topics
  useEffect(() => {
    const onSel = (e: Event) => {
      const id = (e as CustomEvent<string>).detail;
      const n = useAtlasStore.getState().nodes.find((x) => x.id === id);
      if (n) {
        select(id);
        panTo({ x: -n.x * 0.5, y: -n.y * 0.5 });
      }
    };
    window.addEventListener("atlas:select", onSel);
    return () => window.removeEventListener("atlas:select", onSel);
  }, [select, panTo]);

  // --- force simulation ---
  const sim = useRef(nodes.map((n) => ({ ...n })));
  const [, force] = useState(0);

  useEffect(() => {
    let raf = 0;
    const idToIdx = () => {
      const m = new Map<string, number>();
      sim.current.forEach((n, i) => m.set(n.id, i));
      return m;
    };
    const idx = idToIdx();
    const tick = () => {
      const N = sim.current;
      // repulsion
      for (let i = 0; i < N.length; i++) {
        for (let j = i + 1; j < N.length; j++) {
          const a = N[i];
          const b = N[j];
          let dx = b.x - a.x;
          let dy = b.y - a.y;
          let d2 = dx * dx + dy * dy || 1;
          const d = Math.sqrt(d2);
          const minD = a.r + b.r + 58;
          let f = 2600 / d2;
          if (d < minD) f += (minD - d) * 0.09;
          const ux = dx / d;
          const uy = dy / d;
          a.vx -= ux * f * 0.06;
          a.vy -= uy * f * 0.06;
          b.vx += ux * f * 0.06;
          b.vy += uy * f * 0.06;
        }
      }
      // springs
      for (const e of edges) {
        const a = N[idx.get(e.from)!];
        const b = N[idx.get(e.to)!];
        if (!a || !b) continue;
        const dx = b.x - a.x;
        const dy = b.y - a.y;
        const d = Math.hypot(dx, dy) || 1;
        const rest = 210;
        const f = (d - rest) * 0.012;
        a.vx += (dx / d) * f;
        a.vy += (dy / d) * f;
        b.vx -= (dx / d) * f;
        b.vy -= (dy / d) * f;
      }
      // gravity toward centre + damping
      for (const n of N) {
        n.vx -= n.x * 0.004;
        n.vy -= n.y * 0.005;
        n.vx *= 0.86;
        n.vy *= 0.86;
        n.x += n.vx;
        n.y += n.vy;
      }
      force((v) => v + 1);
      raf = requestAnimationFrame(tick);
    };
    raf = requestAnimationFrame(tick);
    return () => cancelAnimationFrame(raf);
  }, [edges]);

  const active = nodes.find((n) => n.id === (hover ?? activeId));
  const selected = activeId;

  // keyboard traversal
  const order = nodes.map((n) => n.id);
  const onKey = (e: React.KeyboardEvent) => {
    if (e.key !== "ArrowRight" && e.key !== "ArrowLeft") return;
    e.preventDefault();
    const i = order.indexOf(selected ?? order[0]);
    const next = e.key === "ArrowRight" ? (i + 1) % order.length : (i - 1 + order.length) % order.length;
    select(order[next]);
    const n = nodes[next];
    panTo({ x: -n.x, y: -n.y });
  };

  const zoomBy = (dz: number) => setZoom(zoom + dz);

  return (
    <div className="atlas" tabIndex={0} onKeyDown={onKey} role="application" aria-label="Act II atlas — use arrow keys to move between nodes">
      <div className="atlas__frame">
        <span className="stage-corner stage-corner--tl" aria-hidden="true" />
        <span className="stage-corner stage-corner--tr" aria-hidden="true" />
        <span className="stage-corner stage-corner--bl" aria-hidden="true" />
        <span className="stage-corner stage-corner--br" aria-hidden="true" />

        <div className="atlas__head">
          <p className="atlas__kicker mono">
            <span className="num">08</span>
            <span className="atlas__rule" aria-hidden="true" />
            ACT II · THE ATLAS
          </p>
          <h1 className="atlas__title">Everything else the model needs</h1>
          <p className="atlas__lede">
            Inference is one map. Training it, aligning it, serving it, and scaling it are the rest. Free-roam from here — pick a
            node, or arrow-key through them.
          </p>
        </div>

        <svg
          ref={svgRef}
          className="atlas__svg"
          viewBox="-560 -400 1120 800"
          preserveAspectRatio="xMidYMid meet"
          onClick={(e) => {
            if (e.target === svgRef.current) select(null);
          }}
          onWheel={(e) => zoomBy(e.deltaY > 0 ? -0.12 : 0.12)}
        >
          <defs>
            <radialGradient id="nodeGrad">
              <stop offset="0%" stopColor="rgba(61,232,255,0.30)" />
              <stop offset="100%" stopColor="rgba(61,232,255,0)" />
            </radialGradient>
          </defs>

          <g transform={`translate(${panned.x} ${panned.y}) scale(${zoom})`}>
            {edges.map((e, i) => {
              const a = sim.current[idxOf(e.from)];
              const b = sim.current[idxOf(e.to)];
              if (!a || !b) return null;
              const on = active && (active.id === e.from || active.id === e.to);
              return (
                <line
                  key={i}
                  x1={a.x}
                  y1={a.y}
                  x2={b.x}
                  y2={b.y}
                  stroke={on ? "rgba(61,232,255,0.7)" : "rgba(255,255,255,0.10)"}
                  strokeWidth={on ? 1.4 : 0.8}
                  strokeDasharray={e.from === "core" || e.to === "core" ? "" : "3 5"}
                />
              );
            })}

            {sim.current.map((n, i) => {
              const isCore = n.kind === "core";
              const isActive = active?.id === n.id;
              const color = n.kind === "train" ? "#FF3DAE" : n.kind === "infra" ? "#4ADE80" : "#3DE8FF";
              return (
                <g
                  key={n.id}
                  className={`atlas__node ${isActive ? "is-active" : ""}`}
                  transform={`translate(${n.x} ${n.y})`}
                  onMouseEnter={() => setHover(n.id)}
                  onMouseLeave={() => setHover(null)}
                  onClick={() => {
                    select(n.id);
                    panTo({ x: -n.x * 0.5, y: -n.y * 0.5 });
                  }}
                  role="button"
                  tabIndex={-1}
                  aria-label={`${n.label} — ${n.sub}`}
                >
                  <circle r={n.r * 2.1} fill="url(#nodeGrad)" opacity={isActive ? 1 : 0.35} />
                  <circle
                    r={n.r}
                    fill="rgba(5,5,7,0.92)"
                    stroke={color}
                    strokeWidth={isActive ? 1.8 : 1}
                    strokeDasharray={n.kind === "infra" ? "4 4" : ""}
                  />
                  <circle r={n.r - 7} fill="none" stroke={color} strokeOpacity={0.22} strokeWidth={0.7} />
                  <text
                    y={4}
                    textAnchor="middle"
                    className="atlas__nodeLabel"
                    fill={isActive ? "#fff" : "#E8E8F0"}
                    fontSize={n.r > 32 ? 13 : 11}
                  >
                    {n.label}
                  </text>
                  <text y={n.r + 16} textAnchor="middle" className="atlas__nodeSub" fill="#52525F" fontSize={9.5}>
                    {n.sub}
                  </text>
                </g>
              );
            })}
          </g>
        </svg>

        <div className="atlas__zoom">
          <button type="button" className="btn btn--ghost" onClick={() => zoomBy(0.15)} aria-label="Zoom in">+</button>
          <span className="mono num">{zoom.toFixed(2)}×</span>
          <button type="button" className="btn btn--ghost" onClick={() => zoomBy(-0.15)} aria-label="Zoom out">−</button>
          <button type="button" className="btn btn--ghost" onClick={() => { panTo({ x: 0, y: 0 }); setZoom(1); }}>reset</button>
        </div>

        <div className="atlas__legend mono">
          <span><i className="lg lg--core" /> inference</span>
          <span><i className="lg lg--train" /> training</span>
          <span><i className="lg lg--infra" /> infrastructure</span>
          <span className="atlas__hint">scroll to zoom · click a node</span>
        </div>

        {selected && (
          <Detail
            id={selected}
            rt={rt}
            onClose={() => select(null)}
          />
        )}
      </div>
    </div>
  );

  function idxOf(id: string) {
    return nodes.findIndex((n) => n.id === id);
  }
}

function Detail({ id, rt, onClose }: { id: DetailId; rt: ReturnType<typeof getRuntime>; onClose: () => void }) {
  const meta = TOPICS[id];
  if (!meta) return null;
  return (
    <aside className="detail" aria-label={`${meta.title} detail`}>
      <header className="detail__head">
        <div>
          <p className="detail__idx mono num">{meta.idx}</p>
          <h2 className="detail__title">{meta.title}</h2>
          <p className="detail__sub mono">{meta.sub}</p>
        </div>
        <button type="button" className="detail__close" onClick={onClose} aria-label="Close detail">✕</button>
      </header>
      <div className="detail__body">{meta.body(rt)}</div>
      <nav className="detail__nav" aria-label="All atlas topics">
        {Object.entries(TOPICS).map(([k, t]) => (
          <a key={k} href={`#/atlas`} className={`detail__pill ${k === id ? "is-active" : ""}`} onClick={() => window.dispatchEvent(new CustomEvent("atlas:select", { detail: k }))}>
            {t.short}
          </a>
        ))}
      </nav>
    </aside>
  );
}

const TOPICS: Record<
  string,
  { idx: string; title: string; sub: string; short: string; body: (rt: ReturnType<typeof getRuntime>) => JSX.Element }
> = {
  training: {
    idx: "09",
    title: "Pretraining",
    sub: "predict the next token, forever",
    short: "Pretraining",
    body: (rt) => {
      const m = rt?.metrics;
      const pts = trainingPoints(m);
      const W = 520;
      const H = 170;
      const maxStep = pts.length ? pts[pts.length - 1].step : 1;
      const maxY = pts.length ? Math.max(...pts.map((p) => p.train)) : 6;
      const path = (key: "train" | "val") =>
        pts
          .map((p, i) => {
            const x = (p.step / maxStep) * W;
            const y = H - (p[key] / maxY) * H;
            return `${i ? "L" : "M"}${x.toFixed(1)},${y.toFixed(1)}`;
          })
          .join(" ");
      const last = pts[pts.length - 1];
      return (
        <>
          <p>
            Nothing about the embedding table was hand-authored. It came from a single objective repeated across the corpus:
            hide the next token, predict it, measure the distance, push every weight a little.
          </p>
          <div className="detail__chart">
            <div className="detail__chartHead mono">
              <span>cross-entropy loss · live training run</span>
              <span className="num">{last ? `step ${last.step} · val ${last.val.toFixed(3)}` : "no metrics"}</span>
            </div>
            <svg viewBox={`0 0 ${W} ${H + 24}`} className="detail__svg">
              {[0.25, 0.5, 0.75].map((f) => (
                <line key={f} x1={0} y1={H * f} x2={W} y2={H * f} stroke="rgba(255,255,255,0.06)" strokeWidth={1} />
              ))}
              <path d={path("train")} fill="none" stroke="#3DE8FF" strokeWidth={1.6} />
              <path d={path("val")} fill="none" stroke="#FF3DAE" strokeWidth={1.4} strokeDasharray="4 4" />
              <text x={2} y={H + 16} fill="#52525F" fontSize={9} fontFamily="JetBrains Mono">0</text>
              <text x={W - 34} y={H + 16} fill="#52525F" fontSize={9} fontFamily="JetBrains Mono">{maxStep}</text>
              <text x={2} y={10} fill="#52525F" fontSize={9} fontFamily="JetBrains Mono">{maxY.toFixed(1)}</text>
            </svg>
            <div className="detail__legend mono">
              <span><i className="lg lg--core" /> train</span>
              <span><i className="lg lg--train" /> val (held out)</span>
            </div>
          </div>
          <p>
            The gap between those lines is the whole story of generalization. When they track together the model is learning
            structure; when they diverge it is memorizing. Our run sits on a 45&nbsp;KB corpus, so it diverges — and the
            generated text you read in Act I is genuinely this model repeating what it absorbed.
          </p>
          <div className="readout">
            <div className="readout__row"><span className="readout__k">objective</span><span className="readout__v">next-token cross-entropy</span></div>
            <div className="readout__row"><span className="readout__k">optimizer</span><span className="readout__v readout__v--muted">Adam (β 0.9 / 0.95)</span></div>
            <div className="readout__row"><span className="readout__k">start loss</span><span className="readout__v">ln(vocab) ≈ 5.95</span></div>
            <div className="readout__row"><span className="readout__k">now</span><span className="readout__v">{last ? last.val.toFixed(3) : "—"}</span></div>
          </div>
        </>
      );
    },
  },
  rlhf: {
    idx: "10",
    title: "Post-training",
    sub: "SFT, then preference, then RL",
    short: "Post-training",
    body: () => (
      <>
        <p>
          A pretrained model is a completion engine, not an assistant. Three further stages shape it:
        </p>
        <ol className="detail__steps">
          <li><b>Supervised fine-tuning.</b> Learn the chat shape — instruction in, answer out — from curated examples. The
          loss is identical to pretraining; only the data changes.</li>
          <li><b>Preference tuning (DPO / RLHF).</b> Humans rank two candidate answers. The model is pushed toward the winner
          and away from the loser, without ever needing a reward model if you use DPO.</li>
          <li><b>RLHF properly.</b> A reward model trained on those rankings scores rollouts; PPO optimizes the policy against
          it. This is where "helpful, harmless, honest" actually gets encoded.</li>
        </ol>
        <div className="detail__flow mono">
          <span>pretrained</span><i>→</i><span>SFT</span><i>→</i><span>preference</span><i>→</i><span>RLHF</span><i>→</i><span>deployed</span>
        </div>
        <p>
          None of this changes the forward pass you watched in Act I. Same layers, same attention, same softmax — only the
          weights differ.
        </p>
      </>
    ),
  },
  kv: {
    idx: "11",
    title: "KV cache",
    sub: "remember what you already computed",
    short: "KV cache",
    body: () => {
      const ctx = 64;
      const W = 520;
      const H = 140;
      const layers = 3;
      const dModel = 64;
      const rows = Array.from({ length: ctx }, (_, t) => ({
        t,
        withCache: (layers * 2 * dModel * 8 * (t + 1)) / 1024,
        without: (layers * 2 * dModel * 8 * ((t + 1) * (t + 2)) / 2) / 1024,
      }));
      const maxY = Math.max(...rows.map((r) => r.without));
      const path = (key: "withCache" | "without") =>
        rows
          .map((r, i) => `${i ? "L" : "M"}${((r.t / (ctx - 1)) * W).toFixed(1)},${(H - (r[key] / maxY) * H).toFixed(1)}`)
          .join(" ");
      return (
        <>
          <p>
            Without caching, every generated token requires a full forward pass over the entire context — the work already
            done gets redone. The KV cache stores each layer's keys and values as they are produced.
          </p>
          <div className="detail__chart">
            <div className="detail__chartHead mono"><span>bytes held vs tokens generated</span><span className="num">kb</span></div>
            <svg viewBox={`0 0 ${W} ${H + 24}`} className="detail__svg">
              <path d={path("without")} fill="none" stroke="#FFB454" strokeWidth={1.5} strokeDasharray="5 4" />
              <path d={path("withCache")} fill="none" stroke="#3DE8FF" strokeWidth={1.8} />
            </svg>
            <div className="detail__legend mono">
              <span><i className="lg lg--backward" /> recompute every step (quadratic)</span>
              <span><i className="lg lg--core" /> cached (linear)</span>
            </div>
          </div>
          <p>
            Linear memory buys you speed but costs VRAM — long-context serving is mostly a memory problem, not a math problem.
          </p>
          <div className="readout">
            <div className="readout__row"><span className="readout__k">growth</span><span className="readout__v">O(context) bytes</span></div>
            <div className="readout__row"><span className="readout__k">our model</span><span className="readout__v readout__v--muted">3 layers × 2 × 64d</span></div>
          </div>
        </>
      );
    },
  },
  quant: {
    idx: "12",
    title: "Quantization",
    sub: "fewer bits, same answers",
    short: "Quantization",
    body: () => {
      const fmts = [
        { name: "fp32", bits: 32, bytes: 4, note: "training" },
        { name: "tf32", bits: 19, bytes: 4, note: "Ampere matmuls" },
        { name: "bf16", bits: 16, bytes: 2, note: "training + inference" },
        { name: "fp16", bits: 16, bytes: 2, note: "fast inference" },
        { name: "int8", bits: 8, bytes: 1, note: "per-channel scales" },
        { name: "int4", bits: 4, bytes: 0.5, note: "group-wise, with outliers" },
      ];
      const base = 61000 * 4;
      return (
        <>
          <p>
            Weights are stored as numbers. Give them fewer bits and everything shrinks: bandwidth, VRAM, latency. The question
            is only how much accuracy you lose.
          </p>
          <div className="detail__bars">
            {fmts.map((f) => (
              <div className="detail__barRow" key={f.name}>
                <span className="detail__barName mono">{f.name}</span>
                <span className="detail__barTrack">
                  <span className="detail__barFill" style={{ width: `${(f.bytes / 4) * 100}%` }} />
                </span>
                <span className="detail__barVal num">{((base * f.bytes) / 1024).toFixed(1)} kb</span>
                <span className="detail__barNote mono">{f.note}</span>
              </div>
            ))}
          </div>
          <p>
            The hard part is outliers: a handful of weights with huge magnitudes wreck naive rounding. That is why modern
            quantization uses <b>scales and zero-points</b>, sometimes per group of 64 values, and why <b>activation</b>{" "}
            quantization is harder than weight quantization — activations move with the input.
          </p>
        </>
      );
    },
  },
  batch: {
    idx: "13",
    title: "Batching",
    sub: "throughput is a scheduling problem",
    short: "Batching",
    body: () => (
      <>
        <p>
          A single request leaves most of the chip idle. Batching stacks many sequences into one matmul — but sequences have
          different lengths and arrive at different times, so the schedule is where throughput is won or lost.
        </p>
        <div className="detail__grid3">
          <div className="detail__tile"><b>Static batching</b><span>simple, but the shortest sequence waits for the longest</span></div>
          <div className="detail__tile"><b>Continuous batching</b><span>finished rows are refilled mid-flight</span></div>
          <div className="detail__tile"><b>Paged attention</b><span>KV blocks allocated like memory pages, no fragmentation</span></div>
        </div>
        <p>
          There is a second axis too: <b>prefill</b> (compute-bound, all prompt tokens at once) versus{" "}
          <b>decode</b> (memory-bound, one token at a time). Serving engines optimize them with entirely different kernels.
        </p>
      </>
    ),
  },
  gpu: {
    idx: "14",
    title: "Hardware",
    sub: "what a GPU actually is",
    short: "Hardware",
    body: () => (
      <>
        <p>
          A GPU is thousands of small cores that only get fast when they are all doing the same operation to different data.
          That is exactly what a matrix multiply is — which is why transformers landed on this hardware and stayed.
        </p>
        <div className="detail__die" aria-hidden="true">
          {Array.from({ length: 24 }).map((_, i) => (
            <span key={i} className="detail__sm" style={{ animationDelay: `${i * 45}ms` }}>
              <i />
              <i />
              <i />
              <i />
            </span>
          ))}
        </div>
        <div className="readout">
          <div className="readout__row"><span className="readout__k">execution</span><span className="readout__v">SIMT · warps of 32</span></div>
          <div className="readout__row"><span className="readout__k">bottleneck</span><span className="readout__v readout__v--muted">HBM bandwidth, not flops</span></div>
          <div className="readout__row"><span className="readout__k">your device</span><span className="readout__v readout__v--muted">
            {typeof navigator !== "undefined" ? `${navigator.hardwareConcurrency ?? "?"} logical cores` : "—"}
          </span></div>
          <div className="readout__row"><span className="readout__k">running on</span><span className="readout__v">CPU — your browser</span></div>
        </div>
        <p>
          Notice the last row: everything in Act I ran on your CPU in JavaScript, in under a tenth of a second. Scale is a
          separate problem from correctness.
        </p>
      </>
    ),
  },
  parallel: {
    idx: "15",
    title: "Parallelism",
    sub: "three ways to split one model",
    short: "Parallelism",
    body: () => (
      <>
        <div className="detail__grid3">
          <div className="detail__tile">
            <b>Data parallel</b>
            <span>every replica holds the full model; each sees a different mini-batch. Gradients are all-reduced each step.</span>
          </div>
          <div className="detail__tile">
            <b>Tensor parallel</b>
            <span>one weight matrix is sliced across GPUs; each chip computes a column slice of the output.</span>
          </div>
          <div className="detail__tile">
            <b>Pipeline parallel</b>
            <span>layers are split across GPUs; activations flow stage to stage in micro-batches.</span>
          </div>
        </div>
        <div className="detail__flow mono">
          <span>shard the batch</span><i>·</i><span>shard the weights</span><i>·</i><span>shard the depth</span>
        </div>
        <p>
          They compose: a frontier run uses all three at once (plus expert parallelism if the model is sparse). The cost of
          every scheme is communication — a sync point where chips wait on each other.
        </p>
      </>
    ),
  },
  scaling: {
    idx: "16",
    title: "Scaling laws",
    sub: "loss is a power law in compute",
    short: "Scaling laws",
    body: () => {
      const W = 520;
      const H = 160;
      const N = 40;
      const pts = Array.from({ length: N }, (_, i) => {
        const t = (i / (N - 1)) * 3; // log10(compute) 0..3
        const loss = 2.6 - 0.42 * Math.log10(Math.pow(10, t) + 1) + 0.06 * t;
        return { x: (t / 3) * W, y: H - ((2.6 - loss) / 0.9) * H };
      });
      const d = pts.map((p, i) => `${i ? "L" : "M"}${p.x.toFixed(1)},${p.y.toFixed(1)}`).join(" ");
      return (
        <>
          <p>
            Chinchilla-era scaling says loss falls as a power law in parameters, data, and compute — smoothly, predictably,
            over many orders of magnitude. Capability jumps arrive on top of that smooth curve and are much less predictable.
          </p>
          <div className="detail__chart">
            <div className="detail__chartHead mono"><span>loss vs training compute (log scale)</span><span className="num">smooth</span></div>
            <svg viewBox={`0 0 ${W} ${H + 24}`} className="detail__svg">
              <line x1={0} y1={H} x2={W} y2={H} stroke="rgba(255,255,255,0.1)" />
              <path d={d} fill="none" stroke="#3DE8FF" strokeWidth={1.8} />
              {[0, 1, 2, 3].map((e) => (
                <text key={e} x={(e / 3) * W + 3} y={H + 16} fill="#52525F" fontSize={9} fontFamily="JetBrains Mono">
                  10^{e}
                </text>
              ))}
            </svg>
          </div>
          <div className="detail__grid3">
            <div className="detail__tile"><b>Compute-optimal</b><span>balance params : data ≈ 20 : 1</span></div>
            <div className="detail__tile"><b>Emergence</b><span>abrupt jumps on some metrics — partly a measurement artifact of coarse scales</span></div>
            <div className="detail__tile"><b>Our curve</b><span>tiny, and visible: loss vs step is the same shape at any scale</span></div>
          </div>
        </>
      );
    },
  },
  glossary: {
    idx: "17",
    title: "Glossary",
    sub: "the vocabulary, quickly",
    short: "Glossary",
    body: () => {
      const terms = [
        ["token", "the atomic unit the model reads — usually a subword, produced by BPE."],
        ["embedding", "the learned vector a token id maps to; one row of wte."],
        ["d_model", "width of the residual stream — every vector in the block has this size."],
        ["head", "one independent attention pattern; the block runs several in parallel."],
        ["causal mask", "the −∞ applied to future positions so no token can peek ahead."],
        ["residual stream", "the running vector each sub-layer reads from and writes to."],
        ["LayerNorm", "zero-mean, unit-variance rescale with learned gain and bias."],
        ["logit", "an unnormalized score for one vocabulary entry."],
        ["softmax", "exponentiate and normalize — turns scores into a distribution."],
        ["temperature", "divides logits before softmax; sharpens or flattens confidence."],
        ["top-k / top-p", "truncate the tail by count or by cumulative probability."],
        ["cross-entropy", "the training loss: −log p(correct next token)."],
        ["KV cache", "stored keys and values that avoid recomputing the past each step."],
        ["quantization", "storing weights in fewer bits with per-group scales."],
        ["FLOP", "one floating-point operation; the usual unit of training compute."],
        ["context window", "maximum tokens the model can condition on at once."],
      ];
      return (
        <>
          <dl className="detail__glossary">
            {terms.map(([t, d]) => (
              <div key={t}>
                <dt>{t}</dt>
                <dd>{d}</dd>
              </div>
            ))}
          </dl>
        </>
      );
    },
  },
};
