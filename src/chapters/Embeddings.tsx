import { useEffect, useMemo, useRef, useState } from "react";
import { Section } from "../shell/Section";
import { Term } from "../shell/Term";
import { Stage } from "../viz/Stage";
import { getRuntime } from "../engine/loader";

const WORDS = [
  "cat", "dog", "wolf", "lion", "horse", "rat",
  "king", "queen", "prince", "man", "woman", "person",
  "attention", "layer", "token", "model", "neural", "gradient",
  "the", "a", "and", "is", "of", "to",
];

export function Embeddings() {
  const rt = getRuntime();
  const [sel, setSel] = useState(0);
  const canvas = useRef<HTMLCanvasElement>(null);

  // word -> first token id -> embedding row
  const rows = useMemo(() => {
    if (!rt) return [];
    const d = rt.model.config.dModel;
    return WORDS.map((w) => {
      const ids = rt.tokenizer.encode(w + " ").filter((i) => i >= 256);
      const id = ids[0] ?? rt.tokenizer.encode(w)[0];
      const data = rt.model.w.wte.data;
      const vec = Array.from({ length: d }, (_, i) => data[id * d + i]);
      return { word: w, id, vec };
    });
  }, [rt]);

  // PCA(2) over the rows so the scatter is a real projection, not dims 0/1.
  const proj = useMemo(() => {
    if (rows.length < 3) return [];
    const d = rows[0].vec.length;
    const mean = new Array(d).fill(0);
    rows.forEach((r) => r.vec.forEach((v, i) => (mean[i] += v / rows.length)));
    const centered = rows.map((r) => r.vec.map((v, i) => v - mean[i]));

    const matvec = (X: number[][]) => {
      const n = X.length;
      const C: number[][] = Array.from({ length: d }, () => new Array(d).fill(0));
      for (let a = 0; a < n; a++)
        for (let i = 0; i < d; i++)
          for (let j = 0; j < d; j++) C[i][j] += (X[a][i] * X[a][j]) / n;
      return C;
    };
    const C = matvec(centered);

    /** Power iteration with optional deflation against kept vectors. */
    const principal = (seed: number, kept: number[][]) => {
      let v = new Array(d).fill(0).map((_, i) => Math.sin(i * 1.7 + seed));
      for (let it = 0; it < 80; it++) {
        const w = new Array(d).fill(0);
        for (let i = 0; i < d; i++) {
          let s = 0;
          for (let j = 0; j < d; j++) s += C[i][j] * v[j];
          w[i] = s;
        }
        for (const u of kept) {
          let dot = 0;
          for (let j = 0; j < d; j++) dot += w[j] * u[j];
          for (let j = 0; j < d; j++) w[j] -= dot * u[j];
        }
        const norm = Math.hypot(...w) || 1;
        v = w.map((x) => x / norm);
      }
      return v;
    };

    const e1 = principal(0.4, []);
    const e2 = principal(2.1, [e1]);
    const project = (vec: number[], basis: number[][]) =>
      basis.map((b) => vec.reduce((s, v, i) => s + v * b[i], 0));
    const pts = rows.map((r) => {
      const [x, y] = project(r.vec, [e1, e2]);
      return { x, y, word: r.word, id: r.id };
    });
    const xs = pts.map((p) => p.x);
    const ys = pts.map((p) => p.y);
    const minX = Math.min(...xs), maxX = Math.max(...xs);
    const minY = Math.min(...ys), maxY = Math.max(...ys);
    return pts.map((p) => ({
      ...p,
      nx: (p.x - minX) / (maxX - minX || 1),
      ny: (p.y - minY) / (maxY - minY || 1),
    }));
  }, [rows]);

  const active = rows[sel];
  const d = active?.vec.length ?? 0;
  const maxAbs = useMemo(
    () => (active ? Math.max(...active.vec.map(Math.abs)) || 1 : 1),
    [active]
  );

  useEffect(() => {
    const c = canvas.current;
    if (!c || !proj.length) return;
    const ctx = c.getContext("2d");
    if (!ctx) return;
    const dpr = Math.min(2, window.devicePixelRatio || 1);
    const rect = c.getBoundingClientRect();
    const W = rect.width || 420;
    const H = rect.height || 420;
    c.width = W * dpr;
    c.height = H * dpr;
    ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
    ctx.clearRect(0, 0, W, H);

    const pad = 34;
    // axes
    ctx.strokeStyle = "rgba(255,255,255,0.08)";
    ctx.lineWidth = 1;
    ctx.strokeRect(pad, pad, W - pad * 2, H - pad * 2);

    // cluster hulls: group by prefix
    const groups = ["cat|dog|wolf|lion|horse|rat", "king|queen|prince|man|woman|person", "attention|layer|token|model|neural|gradient", "the|a|and|is|of|to"];
    groups.forEach((g, gi) => {
      const set = new Set(g.split("|"));
      const pts = proj.filter((p) => set.has(p.word));
      if (pts.length < 2) return;
      ctx.strokeStyle = gi === 0 ? "rgba(61,232,255,0.35)" : gi === 1 ? "rgba(255,61,174,0.3)" : "rgba(255,255,255,0.14)";
      ctx.setLineDash([3, 4]);
      ctx.beginPath();
      pts.forEach((p, i) => {
        const x = pad + p.nx * (W - pad * 2);
        const y = pad + p.ny * (H - pad * 2);
        if (i === 0) ctx.moveTo(x, y);
        else ctx.lineTo(x, y);
      });
      ctx.closePath();
      ctx.stroke();
      ctx.setLineDash([]);
    });

    proj.forEach((p, i) => {
      const x = pad + p.nx * (W - pad * 2);
      const y = pad + p.ny * (H - pad * 2);
      const isSel = i === sel;
      ctx.fillStyle = isSel ? "#3DE8FF" : "rgba(232,232,240,0.75)";
      ctx.beginPath();
      ctx.arc(x, y, isSel ? 5 : 3, 0, Math.PI * 2);
      ctx.fill();
      if (isSel) {
        ctx.strokeStyle = "rgba(61,232,255,0.5)";
        ctx.lineWidth = 1;
        ctx.beginPath();
        ctx.arc(x, y, 11, 0, Math.PI * 2);
        ctx.stroke();
      }
      ctx.font = "11px 'JetBrains Mono', monospace";
      ctx.fillStyle = isSel ? "#3DE8FF" : "rgba(138,138,153,0.9)";
      ctx.fillText(p.word, x + 8, y + 3);
    });
  }, [proj, sel]);

  return (
    <Section
      id="embeddings"
      index={2}
      kicker="the meaning lookup"
      title="A word becomes a direction"
      stage={
        <Stage
          label="wte"
          labelValue={active ? `[${active.word}] id ${active.id} · ${d}d` : ""}
          aria={`Embedding lookup. Word "${active?.word}" maps to token id ${active?.id}, a ${d}-dimensional vector.`}
          foot={
            <>
              <span className="num">lookup wte[id] · {d} floats</span>
              <span className="num">max |x| = {maxAbs.toFixed(3)}</span>
            </>
          }
        >
          <div className="emb">
            <div className="emb__scatter">
              <canvas ref={canvas} className="emb__canvas" />
              <span className="emb__axisLabel emb__axisLabel--x">PC 1</span>
              <span className="emb__axisLabel emb__axisLabel--y">PC 2</span>
            </div>

            <div className="emb__right">
              <div className="emb__words">
                {rows.map((r, i) => (
                  <button
                    key={r.word}
                    type="button"
                    className={`emb__word ${i === sel ? "is-active" : ""}`}
                    onClick={() => setSel(i)}
                  >
                    {r.word}
                  </button>
                ))}
              </div>

              <div className="emb__bars" aria-hidden="true">
                {active?.vec.map((v, i) => (
                  <span
                    key={i}
                    className={`emb__bar ${v >= 0 ? "is-pos" : "is-neg"}`}
                    style={{ height: `${(Math.abs(v) / maxAbs) * 100}%` }}
                    title={`d[${i}] = ${v.toFixed(3)}`}
                  />
                ))}
              </div>
              <div className="emb__barAxis">
                <span className="num">0</span>
                <span className="num">dimension index → {d - 1}</span>
              </div>
            </div>
          </div>
        </Stage>
      }
      aside={
        <div className="readout">
          <div className="readout__row">
            <span className="readout__k">matrix</span>
            <span className="readout__v readout__v--muted">wte</span>
          </div>
          <div className="readout__row">
            <span className="readout__k">shape</span>
            <span className="readout__v">[{rt?.model.config.vocabSize ?? "—"}, {d}]</span>
          </div>
          <div className="readout__row">
            <span className="readout__k">projection</span>
            <span className="readout__v readout__v--muted">PCA, 2 components</span>
          </div>
          <div className="readout__row">
            <span className="readout__k">similarity</span>
            <span className="readout__v readout__v--muted">cosine over the row</span>
          </div>
        </div>
      }
    >
      <p>
        Each token id indexes a row of a giant lookup table — the{" "}
        <Term k="wte" intuition="a big list where every token has its own row of 64 numbers, learned during training." deep={
          <>
            <span className="mono">wte</span> has shape{" "}
            <span className="num">[{rt?.model.config.vocabSize ?? "—"}, {d}]</span> — one {d}-dimensional row per token. It is
            <em> learned</em>, not assigned: gradient descent moved these numbers until predicting the next token got easier. The
            LM head reuses this exact matrix (weight tying), so a token's meaning vector and its output score are the same row.
          </>
        }>
          embedding table
        </Term>
        . One row per token, <span className="num">{d}</span> numbers deep.
      </p>
      <p>
        Nothing about these numbers is hand-written. They are the shape of the training data: words that showed up in similar
        company drift toward similar directions, which is why the scatter clusters{" "}
        <strong>animals with animals</strong> and <strong>function words with function words</strong>.
      </p>
      <p>
        Click any word and watch the bar strip change. The bars are the raw vector the model will now carry through every
        layer — every later computation is arithmetic on this row.
      </p>
    </Section>
  );
}
