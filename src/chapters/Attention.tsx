import { useEffect, useMemo, useRef, useState } from "react";
import { Section } from "../shell/Section";
import { Term } from "../shell/Term";
import { Stage } from "../viz/Stage";
import { Heatmap, TokenAxis } from "../viz/Heatmap";
import { getRuntime } from "../engine/loader";
import { layerNorm, linear, softmax, type Vec, type Mat } from "../engine/tensor";
import { prefersReduced } from "../motion/useReducedMotion";

const DEMO = "the cat sat on the mat because it was warm";

type StageKey = "qkv" | "scores" | "mask" | "softmax" | "out";

const ORDER: { key: StageKey; label: string; formula: string; note: string }[] = [
  { key: "qkv", label: "project", formula: "x · W_qkv", note: "one projection, three outputs" },
  { key: "scores", label: "dot product", formula: "Q · Kᵀ / √d_h", note: "how well does each query match each key" },
  { key: "mask", label: "causal mask", formula: "score_ij = −∞  if  j > i", note: "the future is not visible" },
  { key: "softmax", label: "softmax", formula: "α = softmax(score)", note: "scores become a probability split" },
  { key: "out", label: "weighted sum", formula: "Σ α_ij · v_j", note: "context, assembled" },
];

/**
 * The heat scale reads 0..1, but dot-product scores live in an unbounded
 * range. Min-max them over the meaningful cells so contrast survives the
 * clamp; cells outside `keep` (the causal triangle) collapse to void.
 */
function normalize(m: number[][], keep?: (i: number, j: number) => boolean): number[][] {
  let lo = Infinity;
  let hi = -Infinity;
  for (let i = 0; i < m.length; i++) {
    for (let j = 0; j < m[i].length; j++) {
      const v = m[i][j];
      if (!isFinite(v) || v < -1e8) continue;
      if (keep && !keep(i, j)) continue;
      if (v < lo) lo = v;
      if (v > hi) hi = v;
    }
  }
  const span = hi - lo || 1;
  return m.map((row, i) =>
    row.map((v, j) => {
      if (!isFinite(v) || v < -1e8) return 0;
      if (keep && !keep(i, j)) return 0;
      return (v - lo) / span;
    })
  );
}

export function Attention() {
  const rt = getRuntime();
  const reduced = prefersReduced();
  const [layer, setLayer] = useState(0);
  const [head, setHead] = useState(0);
  const [query, setQuery] = useState(3);
  const [stageIdx, setStageIdx] = useState(0);
  const [auto, setAuto] = useState(true);

  const cfg = rt?.model.config;
  const nLayers = cfg?.nLayers ?? 3;
  const nHeads = cfg?.nHeads ?? 4;
  const d = cfg?.dModel ?? 64;
  const dh = d / nHeads;

  const tokens = useMemo(() => {
    if (!rt) return DEMO.split(" ");
    const ids = rt.tokenizer.encode(DEMO).slice(0, 24);
    return ids.map((id) => (id < 256 ? String.fromCharCode(id) : rt.tokenizer.decode([id])));
  }, [rt]);

  const n = tokens.length;

  /** Real Q/K/V for one layer, computed the way the engine does it. */
  const calc = useMemo(() => {
    if (!rt || !cfg || n === 0) return null;
    const W = rt.model.w;
    const ids = rt.tokenizer.encode(DEMO).slice(0, 24);
    const x: Vec[] = ids.map((tok, i) => {
      const h = new Float64Array(d);
      for (let k = 0; k < d; k++) h[k] = W.wte.data[tok * d + k] + W.wpe.data[(i % cfg.contextLength) * d + k];
      return h;
    });
    const normed = x.map((h) => layerNorm(h, W.layers[layer].ln1W, W.layers[layer].ln1B));
    const qkv = normed.map((h) => linear(W.layers[layer].qkvW, h, W.layers[layer].qkvB));

    const Q: Vec[][] = [];
    const K: Vec[][] = [];
    const V: Vec[][] = [];
    for (let t = 0; t < n; t++) {
      const q = qkv[t];
      Q.push(Array.from({ length: nHeads }, (_, hh) => q.subarray(hh * dh, (hh + 1) * dh)));
      K.push(Array.from({ length: nHeads }, (_, hh) => q.subarray(d + hh * dh, d + (hh + 1) * dh)));
      V.push(Array.from({ length: nHeads }, (_, hh) => q.subarray(2 * d + hh * dh, 2 * d + (hh + 1) * dh)));
    }

    const raw: number[][] = Array.from({ length: n }, () => new Array(n).fill(0));
    const masked: number[][] = Array.from({ length: n }, () => new Array(n).fill(0));
    const probs: number[][] = Array.from({ length: n }, () => new Array(n).fill(0));
    const out: Vec[] = Array.from({ length: n }, () => new Float64Array(dh));
    const inv = 1 / Math.sqrt(dh);

    for (let i = 0; i < n; i++) {
      for (let j = 0; j < n; j++) {
        let s = 0;
        for (let k = 0; k < dh; k++) s += Q[i][head][k] * K[j][head][k];
        raw[i][j] = s * inv;
        masked[i][j] = j > i ? -1e9 : s * inv;
      }
      const row = softmax(Float64Array.from(masked[i]), 1);
      for (let j = 0; j < n; j++) {
        probs[i][j] = row[j];
        for (let k = 0; k < dh; k++) out[i][k] += row[j] * V[j][head][k];
      }
    }

    return { Q, K, V, raw, masked, probs, out };
  }, [rt, layer, head, n, d, nHeads, dh, cfg?.contextLength]);

  // clamp query when the sequence shortens
  useEffect(() => {
    if (query > n - 1) setQuery(Math.max(0, n - 1));
  }, [n, query]);

  // auto-advance through stages
  useEffect(() => {
    if (!auto || reduced) return;
    const id = window.setInterval(() => setStageIdx((s) => (s + 1) % ORDER.length), 1900);
    return () => window.clearInterval(id);
  }, [auto, reduced]);

  const stage = ORDER[Math.min(stageIdx, ORDER.length - 1)];
  const q = calc ? calc.Q.map((r) => r[head]) : [];
  const k = calc ? calc.K.map((r) => r[head]) : [];
  const v = calc ? calc.V.map((r) => r[head]) : [];

  /** Rows ready for the 0..1 heat scale, per stage. */
  const stageRows = useMemo(() => {
    if (!calc) return null;
    if (stage.key === "softmax") return calc.probs;
    if (stage.key === "scores") return normalize(calc.raw);
    if (stage.key === "mask") return normalize(calc.raw, (i, j) => j <= i);
    return null;
  }, [calc, stage.key]);

  /** Underlying values for the numeric footer (not heat-normalized). */
  const rawRows = stage.key === "softmax" ? calc?.probs : stage.key === "mask" ? calc?.masked : calc?.raw;

  const maxOut = calc ? Math.max(...calc.out.map((o) => Math.max(...Array.from(o, Math.abs)))) || 1 : 1;

  return (
    <Section
      id="attention"
      index={4.5}
      kicker="the actual mechanism"
      title="Everyone asks everyone"
      stage={
        <Stage
          label="self-attention"
          labelValue={`L${layer + 1} H${head + 1} · q=${query}`}
          aria={`Self-attention for layer ${layer + 1}, head ${head + 1}. Query token "${tokens[query]}" attends over ${n} tokens.`}
          foot={
            stage.key === "qkv" || stage.key === "out" ? null : (
              <>
                <span className="num">
                  row[{query}] = {rawRows?.[query]?.slice(0, 4).map((x) => (x < -1e8 ? "−∞" : x.toFixed(2))).join("  ")} …
                </span>
                <span className="num">
                  {stage.key === "softmax"
                    ? `Σ = ${rawRows?.[query]?.reduce((a, b) => a + b, 0).toFixed(3)}`
                    : stage.key === "mask"
                    ? `visible ${query + 1} · masked ${Math.max(0, n - query - 1)}`
                    : (() => {
                        const row = (rawRows?.[query] ?? []).filter((x) => isFinite(x));
                        return row.length
                          ? `range [${Math.min(...row).toFixed(2)}, ${Math.max(...row).toFixed(2)}]`
                          : "—";
                      })()}
                </span>
              </>
            )
          }
        >
          <div className="att">
            <ol className="att__steps" aria-label="Attention stages">
              {ORDER.map((s, i) => (
                <li key={s.key}>
                  <button
                    type="button"
                    className={`att__step ${i === stageIdx ? "is-active" : ""} ${i < stageIdx ? "is-done" : ""}`}
                    onClick={() => {
                      setStageIdx(i);
                      setAuto(false);
                    }}
                    aria-current={i === stageIdx}
                  >
                    <span className="att__stepIdx num">{i + 1}</span>
                    <span className="att__stepLabel">{s.label}</span>
                  </button>
                </li>
              ))}
            </ol>

            <p className="att__formula mono" key={stage.key}>
              <span className="att__formulaMain">{stage.formula}</span>
              <span className="att__formulaNote">{stage.note}</span>
            </p>

            <div className="att__main">
              <div className="att__cols">
                <TokenAxis tokens={tokens} />
              </div>

              {stage.key === "qkv" || stage.key === "out" ? (
                <div className="att__vectors">
                  {(stage.key === "qkv" ? ["Q", "K", "V"] : ["out"]).map((which) => (
                    <div className="att__vecRow" key={which}>
                      <span className="att__vecName mono">{which}</span>
                      <div className="att__vecCells">
                        {(which === "Q" ? q : which === "K" ? k : which === "V" ? v : calc?.out)?.map((vec, t) => {
                          const m = Math.max(...Array.from(vec, Math.abs)) || 1;
                          return (
                            <button
                              key={t}
                              type="button"
                              className={`att__vecCell ${t === query ? "is-query" : ""} ${t > query && which !== "out" ? "is-masked" : ""}`}
                              onClick={() => setQuery(t)}
                              title={`${which} at ${t} · ‖·‖ ${Math.hypot(...Array.from(vec)).toFixed(3)}`}
                            >
                              <div className="att__miniBars">
                                {Array.from(vec)
                                  .slice(0, dh)
                                  .map((val, i) => (
                                    <span
                                      key={i}
                                      className={`att__miniBar ${val >= 0 ? "is-pos" : "is-neg"}`}
                                      style={{ height: `${(Math.abs(val) / m) * 100}%` }}
                                    />
                                  ))}
                              </div>
                              <span className="att__vecIdx num">{t}</span>
                            </button>
                          );
                        })}
                      </div>
                    </div>
                  ))}
                  <p className="att__hint mono">
                    {stage.key === "qkv"
                      ? `d_h = ${dh} per head · ${nHeads} heads · shown: head ${head + 1}`
                      : "context vector returned to the residual stream"}
                  </p>
                </div>
              ) : (
                <div className="att__heatWrap">
                  <div className="att__heat">
                    <Heatmap
                      rows={stageRows ?? []}
                      tokens={tokens}
                      query={query}
                      onSelect={setQuery}
                      animate={stage.key === "softmax"}
                      causal={stage.key !== "scores"}
                    />
                    <div className="att__qLine" style={{ top: `${(query / n) * 100}%` }} aria-hidden="true" />
                  </div>
                  <div className="att__legend mono">
                    <span>
                      {stage.key === "mask"
                        ? "above the diagonal is −∞ — void after softmax"
                        : stage.key === "scores"
                        ? "rescaled to 0–1 across this matrix · brighter = stronger match"
                        : "row sums to 1.00"}
                    </span>
                    <span className="att__heatLegend" aria-hidden="true">
                      {stage.key === "softmax" ? "0" : stage.key === "mask" ? "−∞" : "min"}
                      <span className="att__heatBar" />
                      {stage.key === "softmax" ? "1" : stage.key === "mask" ? "max" : "max"}
                    </span>
                  </div>
                </div>
              )}
            </div>

            <div className="att__ctrls">
              <div className="knob">
                <span className="knob__name">layer</span>
                <input type="range" aria-label="layer" min={0} max={nLayers - 1} value={layer} onChange={(e) => setLayer(Number(e.target.value))} />
                <span className="knob__val">{layer + 1}</span>
              </div>
              <div className="knob">
                <span className="knob__name">head</span>
                <input type="range" aria-label="head" min={0} max={nHeads - 1} value={head} onChange={(e) => setHead(Number(e.target.value))} />
                <span className="knob__val">{head + 1}</span>
              </div>
              <div className="knob">
                <span className="knob__name">query</span>
                <input type="range" aria-label="query position" min={0} max={Math.max(0, n - 1)} value={query} onChange={(e) => setQuery(Number(e.target.value))} />
                <span className="knob__val">{query}</span>
              </div>
              <button type="button" className="btn btn--ghost" onClick={() => setAuto(!auto)}>
                {auto ? "pause ▮▮" : "play ▸"}
              </button>
            </div>
          </div>
        </Stage>
      }
      aside={
        <div className="readout">
          <div className="readout__row">
            <span className="readout__k">heads</span>
            <span className="readout__v">{nHeads}</span>
          </div>
          <div className="readout__row">
            <span className="readout__k">head dim</span>
            <span className="readout__v">{dh}</span>
          </div>
          <div className="readout__row">
            <span className="readout__k">positions</span>
            <span className="readout__v">{n}</span>
          </div>
          <div className="readout__row">
            <span className="readout__k">top target</span>
            <span className="readout__v readout__v--muted">
              {calc ? (() => {
                let bi = 0;
                let bv = -1;
                calc.probs[query].forEach((p, i) => {
                  if (p > bv) {
                    bv = p;
                    bi = i;
                  }
                });
                return `${tokens[bi]} ${(bv * 100).toFixed(1)}%`;
              })() : "—"}
            </span>
          </div>
        </div>
      }
    >
      <p>
        This is the part people paraphrase as "the model pays attention." It is more literal than that:{" "}
        <Term k="attention" intuition="every position computes a compatibility score with every earlier position, turns those scores into percentages, and averages the values by those percentages." deep={
          <>
            For each head the layer builds <span className="mono">Q = xW_q</span>, <span className="mono">K = xW_k</span>,{" "}
            <span className="mono">V = xW_v</span>. Scores are{" "}
            <span className="mono">softmax(QKᵀ/√d_h + M)V</span> where <span className="mono">M</span> is the causal mask. All
            three projections are learned; nothing here is hand-designed similarity.
          </>
        }>
          it is an average
        </Term>
        , computed by the same arithmetic you can read off the formula above.
      </p>
      <p>
        Step through the five stages. Watch the <strong>mask</strong> sweep: every cell above the diagonal is stamped with{" "}
        <span className="num">−∞</span>, so after softmax it is <strong>exactly zero</strong> — not small, zero. That single
        rule is what makes the model incapable of cheating by reading the answer.
      </p>
      <p>
        Then look at the <strong>softmax</strong> frame: each row lights up as a distribution over the visible past. Click any
        cell or drag the query slider to move which token is asking. The right-hand readout tells you where most of the weight
        actually landed.
      </p>
      <p>
        Finally the <strong>weighted sum</strong> frame assembles a fresh vector per position — the "context" that gets added
        back into the residual stream. Different heads learn different habits, which is why the head slider visibly changes the
        pattern.
      </p>
    </Section>
  );
}
