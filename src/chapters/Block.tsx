import { useEffect, useMemo, useRef, useState } from "react";
import { Section } from "../shell/Section";
import { Term } from "../shell/Term";
import { Stage } from "../viz/Stage";
import { getRuntime } from "../engine/loader";
import { prefersReduced } from "../motion/useReducedMotion";

const DEMO = "attention is a weighted";

const STAGES = [
  { key: "x", label: "x", sub: "input" },
  { key: "ln1", label: "LN", sub: "normalize" },
  { key: "mha", label: "MHA", sub: "attend" },
  { key: "add1", label: "+", sub: "residual" },
  { key: "ln2", label: "LN", sub: "normalize" },
  { key: "ffn", label: "FFN", sub: "transform" },
  { key: "add2", label: "+", sub: "residual" },
] as const;

export function Block() {
  const rt = getRuntime();
  const [layer, setLayer] = useState(0);
  const [head, setHead] = useState(0);
  const [cursor, setCursor] = useState(0);
  const reduced = prefersReduced();
  const nLayers = rt?.model.config.nLayers ?? 3;
  const nHeads = rt?.model.config.nHeads ?? 4;
  const d = rt?.model.config.dModel ?? 64;
  const dFfn = rt?.model.config.dFfn ?? 192;

  const tokens = useMemo(() => {
    if (!rt) return DEMO.split(" ");
    const ids = rt.tokenizer.encode(DEMO);
    return ids.map((id) => (id < 256 ? String.fromCharCode(id) : rt.tokenizer.decode([id])));
  }, [rt]);

  // one real forward pass so the readouts are live, not decorative
  const pass = useMemo(() => {
    if (!rt) return null;
    const ids = rt.tokenizer.encode(DEMO).slice(0, 24);
    try {
      return rt.model.forward(ids, 0);
    } catch {
      return null;
    }
  }, [rt]);

  // ambient flow animation along the rail
  const [flow, setFlow] = useState(reduced ? 1 : 0);
  useEffect(() => {
    if (reduced) {
      setFlow(1);
      return;
    }
    let raf = 0;
    let t = 0;
    const tick = () => {
      t = (t + 0.006) % 1;
      setFlow(t);
      raf = requestAnimationFrame(tick);
    };
    raf = requestAnimationFrame(tick);
    return () => cancelAnimationFrame(raf);
  }, [reduced]);

  const step = pass?.steps[layer];
  const attnHeads = step?.attnProbs ?? [];
  const attnRows = attnHeads[Math.min(head, Math.max(0, attnHeads.length - 1))] ?? [];
  const attnRow = attnRows[Math.min(cursor, Math.max(0, attnRows.length - 1))] ?? [];
  const activeStage = Math.min(STAGES.length - 1, Math.floor(flow * STAGES.length));

  const normOf = (arr?: ArrayLike<number>) => {
    if (!arr) return 0;
    let s = 0;
    for (let i = 0; i < arr.length; i++) s += arr[i] * arr[i];
    return Math.sqrt(s);
  };

  const ffnNorm = step ? normOf(step.ffnDelta) : 0;
  const hidNorm = step ? normOf(step.hidden) : 0;

  return (
    <Section
      id="block"
      index={4}
      kicker="the unit of computation"
      title="Stacked, repeated, identical"
      stage={
        <Stage
          label="block"
          labelValue={`layer ${layer + 1}/${nLayers} · head ${head + 1}/${nHeads}`}
          aria={`Transformer block schematic for layer ${layer + 1}. Data flows input, layer norm, multi-head attention, residual add, layer norm, feed-forward, residual add.`}
          foot={
            <>
              <span className="num">‖x‖ {hidNorm.toFixed(3)} · ‖ffn‖ {ffnNorm.toFixed(3)}</span>
              <span className="num">
                {tokens.length} tokens × {d}d
              </span>
            </>
          }
        >
          <div className="block">
            <div className="block__rail" role="presentation">
              {STAGES.map((s, i) => (
                <div key={s.key} className={`block__stage ${i === activeStage ? "is-active" : ""} ${i < activeStage ? "is-done" : ""}`}>
                  <div className={`block__node block__node--${s.key}`}>
                    <span className="block__key">{s.label}</span>
                  </div>
                  <span className="block__sub">{s.sub}</span>
                  {i < STAGES.length - 1 && (
                    <span className="block__wire" aria-hidden="true">
                      <span className="block__pulse" style={{ left: `${(flow * 100) % 100}%` }} />
                    </span>
                  )}
                </div>
              ))}
            </div>

            <div className="block__residual" aria-hidden="true">
              <span className="block__resLine block__resLine--a" />
              <span className="block__resLabel">residual highway</span>
              <span className="block__resLine block__resLine--b" />
            </div>

            <div className="block__matrix">
              <div className="block__matrixHead">ATTENTION · L{layer + 1} H{head + 1} · query = token {cursor}</div>
              <div className="block__attRow">
                {attnRow.map((v, j) => (
                  <button
                    key={j}
                    type="button"
                    className={`block__attCell ${j === cursor ? "is-query" : ""} ${j > cursor ? "is-masked" : ""}`}
                    style={{ background: j <= cursor ? `rgba(61,232,255,${(0.08 + v * 0.92).toFixed(3)})` : "rgba(5,5,7,1)" }}
                    onClick={() => setCursor(j)}
                    title={`→ ${tokens[j]} = ${v.toFixed(4)}`}
                    aria-label={`attention weight from ${tokens[cursor]} to ${tokens[j]} is ${v.toFixed(4)}`}
                  >
                    <span className="block__attTok">{tokens[j] === " " ? "␣" : tokens[j]}</span>
                    <span className="block__attV num">{j <= cursor ? v.toFixed(2) : "—"}</span>
                  </button>
                ))}
              </div>
            </div>

            <div className="block__ctrls">
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
            </div>
          </div>
        </Stage>
      }
      aside={
        <div className="readout">
          <div className="readout__row">
            <span className="readout__k">layers</span>
            <span className="readout__v">{nLayers}</span>
          </div>
          <div className="readout__row">
            <span className="readout__k">d_model</span>
            <span className="readout__v">{d}</span>
          </div>
          <div className="readout__row">
            <span className="readout__k">heads</span>
            <span className="readout__v">{nHeads} × {d / nHeads}d</span>
          </div>
          <div className="readout__row">
            <span className="readout__k">d_ffn</span>
            <span className="readout__v">{dFfn}</span>
          </div>
        </div>
      }
    >
      <p>
        Every layer runs the same two-step ritual: <strong>look around</strong>, then <strong>think locally</strong>. Both steps
        are wrapped in the same safety rail.
      </p>
      <p>
        <Term k="residual" intuition="whatever comes out of a sub-layer gets added back on top of what went in, so information can flow straight through without being rewritten." deep={
          <>
            The block computes a <em>delta</em> and adds it:{" "}
            <span className="mono">x ← x + f(LN(x))</span>. Because the identity path is never transformed, gradients have a
            clean route back to the input, which is why 12-layer and 100-layer stacks can both train. The additive rail you see
            running along the diagram is that identity path — it is the reason depth works at all.
          </>
        }>
          residual connections
        </Term>{" "}
        are the rail running underneath: the sub-layer's output is <em>added</em> to its input, never used as a replacement.
      </p>
      <p>
        The attention row above is real — a live forward pass on this prompt, layer{" "}
        <span className="num">{layer + 1}</span>, head <span className="num">{head + 1}</span>. Cells to the right of the query
        are <strong>masked</strong> (the model may never look ahead), which is why they read <span className="num">—</span>.
      </p>
      <p>
        Click any past cell to move the query. The numbers always sum to <span className="num">1.00</span> across the visible
        span: attention is a <Term k="weighted-average" intuition="a percentage split across the words it can see." deep={
          <>
            Softmax turns the raw dot-product scores into a probability distribution over positions. The output at position{" "}
            <span className="num">t</span> is the convex combination{" "}
            <span className="mono">Σ α<sub>t,j</sub> · v<sub>j</sub></span>, so the layer can only remix information that
            already exists in the window — it cannot invent context.
          </>
        }>
          probability distribution
        </Term>
        .
      </p>
    </Section>
  );
}
