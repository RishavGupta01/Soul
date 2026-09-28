import { useMemo, useState } from "react";
import { Section } from "../shell/Section";
import { Term } from "../shell/Term";
import { Stage } from "../viz/Stage";
import { getRuntime } from "../engine/loader";
import { heat } from "../viz/Heatmap";

const DEMO = "the model reads left to right and needs to know which word came first";

export function Positional() {
  const rt = getRuntime();
  const [pos, setPos] = useState(7);
  const [n, setN] = useState(24);

  const tokens = useMemo(() => {
    if (!rt) return DEMO.split(" ");
    const ids = rt.tokenizer.encode(DEMO).slice(0, 48);
    return ids.map((id) => (id < 256 ? String.fromCharCode(id) : rt.tokenizer.decode([id])));
  }, [rt]);

  const shown = tokens.slice(0, n);
  const d = rt?.model.config.dModel ?? 64;
  const ctx = rt?.model.config.contextLength ?? 64;

  const wpe = rt?.model.w.wpe.data ?? null;
  const row = wpe ? Array.from({ length: d }, (_, i) => wpe[(pos % ctx) * d + i]) : [];
  const maxAbs = Math.max(1e-6, ...row.map(Math.abs));

  return (
    <Section
      id="position"
      index={3}
      kicker="order matters"
      title="Sequence gets a shape"
      stage={
        <Stage
          label="wpe"
          labelValue={`pos ${pos} · [${n}, ${d}]`}
          aria={`Positional encoding. Row ${pos} of the position table, ${d} dimensions, plus the ${n} positions currently in the sequence.`}
          foot={
            <>
              <span className="num">wpe[{pos}] · {d} floats</span>
              <span className="num">context {ctx} positions</span>
            </>
          }
        >
          <div className="pos">
            <div className="pos__grid">
              <div className="pos__rail">
                <div className="pos__railHead">SEQUENCE</div>
                <div className="pos__railScroll">
                  {shown.map((t, i) => (
                    <button
                      key={i}
                      type="button"
                      className={`pos__cell ${i === pos ? "is-active" : ""} ${i > pos ? "is-future" : ""}`}
                      onClick={() => setPos(i)}
                      style={{ animationDelay: `${i * 18}ms` }}
                    >
                      <span className="pos__idx num">{i}</span>
                      <span className="pos__tok">{t === " " ? "␣" : t}</span>
                    </button>
                  ))}
                </div>
              </div>

              <div className="pos__vector">
                <div className="pos__vecHead">
                  <span>wpe[{pos}]</span>
                  <span className="num">±{maxAbs.toFixed(3)}</span>
                </div>
                <div className="pos__bars">
                  {row.map((v, i) => (
                    <span key={i} className="pos__bar" title={`d[${i}] = ${v.toFixed(3)}`}>
                      <span
                        className={`pos__barFill ${v >= 0 ? "is-pos" : "is-neg"}`}
                        style={{ height: `${(Math.abs(v) / maxAbs) * 100}%` }}
                      />
                    </span>
                  ))}
                </div>
              </div>

              <div className="pos__heat">
                <div className="pos__heatHead">POSITION TABLE · first {Math.min(ctx, 48)} rows</div>
                <div className="pos__heatWrap">
                  <canvas
                    className="pos__canvas"
                    ref={(c) => {
                      if (!c || !wpe) return;
                      const ctx2 = c.getContext("2d");
                      if (!ctx2) return;
                      const rows = Math.min(ctx, 48);
                      const cw = 7;
                      const dpr = Math.min(2, window.devicePixelRatio || 1);
                      c.width = d * cw * dpr;
                      c.height = rows * cw * dpr;
                      c.style.width = `${d * cw}px`;
                      c.style.height = `${rows * cw}px`;
                      ctx2.setTransform(dpr, 0, 0, dpr, 0, 0);
                      ctx2.clearRect(0, 0, d * cw, rows * cw);
                      for (let r = 0; r < rows; r++) {
                        for (let i = 0; i < d; i++) {
                          const v = wpe[r * d + i];
                          ctx2.fillStyle = heat((Math.tanh(v * 1.4) + 1) / 2);
                          ctx2.fillRect(i * cw, r * cw, cw - 0.6, cw - 0.6);
                        }
                      }
                      ctx2.strokeStyle = "rgba(61,232,255,0.95)";
                      ctx2.lineWidth = 1.5;
                      ctx2.strokeRect(0, (pos % rows) * cw + 0.75, d * cw - 1.5, cw - 1.5);
                    }}
                  />
                </div>
              </div>
            </div>

            <div className="pos__slider">
              <div className="knob">
                <span className="knob__name">positions</span>
                <input
                  type="range"
                  aria-label="number of positions shown"
                  min={4}
                  max={Math.min(48, tokens.length)}
                  value={n}
                  onChange={(e) => setN(Number(e.target.value))}
                />
                <span className="knob__val">{n}</span>
              </div>
              <div className="knob">
                <span className="knob__name">focus pos</span>
                <input
                  type="range"
                  aria-label="focused position"
                  min={0}
                  max={Math.max(0, n - 1)}
                  value={Math.min(pos, n - 1)}
                  onChange={(e) => setPos(Number(e.target.value))}
                />
                <span className="knob__val">{pos}</span>
              </div>
            </div>
          </div>
        </Stage>
      }
      aside={
        <div className="readout">
          <div className="readout__row">
            <span className="readout__k">matrix</span>
            <span className="readout__v readout__v--muted">wpe</span>
          </div>
          <div className="readout__row">
            <span className="readout__k">shape</span>
            <span className="readout__v">[{ctx}, {d}]</span>
          </div>
          <div className="readout__row">
            <span className="readout__k">encoding</span>
            <span className="readout__v readout__v--muted">learned lookup</span>
          </div>
          <div className="readout__row">
            <span className="readout__k">past position {pos}</span>
            <span className="readout__v readout__v--muted">already added to the embedding</span>
          </div>
        </div>
      }
    >
      <p>
        Attention is <Term k="permutation" intuition="if you shuffled the words, the same numbers would still get computed — nothing in attention knows what came first." deep={
          <>
            Causal self-attention only enforces <em>direction</em> (position <span className="num">t</span> may see{" "}
            <span className="num">≤ t</span>). It does not distinguish "the dog bit the man" from "the man bit the dog" if the
            multiset of tokens is identical — attention is a weighted average, which is order-blind within its window. Order has to
            arrive some other way.
          </>
        }>
          permutation-invariant
        </Term>{" "}
        by itself. So order is injected before the first layer.
      </p>
      <p>
        This model uses a <strong>learned position table</strong>: row <span className="num">t</span> is added to the token
        embedding at position <span className="num">t</span>. The highlighted row above is exactly what gets added to whichever
        word sits there.
      </p>
      <p>
        Click through the sequence rail — the vector strip changes with every position, and the heatmap shows the whole table at
        once. Beyond <span className="num">{ctx}</span> tokens the position wraps, which is why context length is a hard limit
        rather than a suggestion.
      </p>
    </Section>
  );
}
