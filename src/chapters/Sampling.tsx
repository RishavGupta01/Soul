import { useMemo, useState } from "react";
import { Section } from "../shell/Section";
import { Term } from "../shell/Term";
import { Stage } from "../viz/Stage";
import { sample, rankDesc, entropy } from "../engine/sampler";
import { softmax } from "../engine/tensor";
import { useRuntime } from "../engine/useRuntime";

const DEMO = "the capital of france is";

export function Sampling() {
  const rt = useRuntime();
  const [temp, setTemp] = useState(1.5);
  const [topK, setTopK] = useState(40);
  const [topP, setTopP] = useState(0.95);
  const [seed, setSeed] = useState(7);
  const [roll, setRoll] = useState<number | null>(null);

  const V = rt?.model.config.vocabSize ?? 384;

  const base = useMemo(() => {
    if (!rt) return null;
    const ids = rt.tokenizer.encode(DEMO);
    try {
      const logits = rt.model.forward(ids, 0).logits;
      return { logits, order: rankDesc(logits) };
    } catch {
      return null;
    }
  }, [rt]);

  const view = useMemo(() => {
    if (!base) return null;
    const probs = softmax(Float64Array.from(base.logits), temp);

    // apply top-k
    const keepK = new Set(base.order.slice(0, Math.min(topK, V)));
    let afterK = Array.from(probs).map((p, i) => (keepK.has(i) ? p : 0));

    // apply top-p over what survived k
    let afterP = afterK;
    if (topP < 1) {
      const ranked = base.order.filter((i) => afterK[i] > 0);
      let acc = 0;
      const keep = new Set<number>();
      for (const i of ranked) {
        keep.add(i);
        acc += afterK[i];
        if (acc >= topP) break;
      }
      afterP = afterK.map((p, i) => (keep.has(i) ? p : 0));
    }

    const sum = afterP.reduce((a, b) => a + b, 0) || 1;
    const final = afterP.map((p) => p / sum);

    const dropped = base.order.filter((_, r) => final[base.order[r]] === 0).length;
    const alive = V - dropped;

    return { probs, afterK, afterP, final, alive };
  }, [base, temp, topK, topP, V]);

  const draw = useMemo(() => {
    if (!base || !view) return null;
    let r = roll ?? 0.42;
    return sample(base.logits, { temperature: temp, topK, topP, seed: seed * 977 + (roll ?? 0) });
  }, [base, view, temp, topK, topP, seed, roll]);

  const rows = base ? base.order.slice(0, 20) : [];
  const pTop = view && rows.length ? view.final[rows[0]] : 0;
  const H = view ? entropy(Float64Array.from(view.final)) : 0;

  return (
    <Section
      id="sampling"
      index={6}
      kicker="choosing the next token"
      title="A fair coin, weighted"
      stage={
        <Stage
          label="sampler"
          labelValue={`${view?.alive ?? "—"} of ${V} tokens alive`}
          aria={`Sampling distribution. ${view?.alive} of ${V} candidates survive temperature ${temp.toFixed(2)}, top-k ${topK}, top-p ${topP.toFixed(2)}.`}
          foot={
            <>
              <span className="num">H = {H.toFixed(3)} b · top {(pTop * 100).toFixed(1)}%</span>
              <span className="num">seed {seed}</span>
            </>
          }
        >
          <div className="samp">
            <div className="samp__grid">
              <div className="samp__list">
                <div className="samp__head">
                  <span>#</span>
                  <span>token</span>
                  <span>after T</span>
                  <span>after k</span>
                  <span>after p</span>
                  <span>final</span>
                </div>
                {rows.map((id, i) => {
                  const a = view?.probs[id] ?? 0;
                  const b = view?.afterK[id] ?? 0;
                  const c = view?.afterP[id] ?? 0;
                  const f = view?.final[id] ?? 0;
                  const chosen = draw?.id === id;
                  const text = rt?.tokenizer.decode([id]) ?? "?";
                  return (
                    <div className={`samp__row ${f === 0 ? "is-dropped" : ""} ${chosen ? "is-chosen" : ""}`} key={id}>
                      <span className="samp__rank num">{i + 1}</span>
                      <span className="samp__tok mono">{text === " " ? "␣" : text.replace(/\n/g, "⏎")}</span>
                      <span className="samp__cell"><Meter v={a} /></span>
                      <span className="samp__cell"><Meter v={b} /></span>
                      <span className="samp__cell"><Meter v={c} /></span>
                      <span className="samp__cell samp__cell--final">
                        <Meter v={f} accent />
                        <span className="samp__num num">{(f * 100).toFixed(1)}%</span>
                      </span>
                      {chosen && <span className="samp__marker" aria-hidden="true">◂ drawn</span>}
                    </div>
                  );
                })}
              </div>

              <div className="samp__draw">
                <div className="samp__needle" aria-hidden="true">
                  <div className="samp__needleTrack">
                    {rows.map((id, i) => {
                      const f = view?.final[id] ?? 0;
                      return <span key={id} className="samp__needleSeg" style={{ width: `${f * 100}%`, background: i === 0 ? "var(--cyan)" : `rgba(61,232,255,${(0.15 + f).toFixed(2)})` }} />;
                    })}
                  </div>
                  <span className="samp__needleLabel mono">
                    {draw ? `drawn: “${rt?.tokenizer.decode([draw.id]).replace(/ /g, "␣") ?? "?"}”` : "—"}
                  </span>
                </div>

                <button type="button" className="btn" onClick={() => setRoll(Math.floor(Math.random() * 1e9))}>
                  ◈ sample next token
                </button>

                <div className="samp__why">
                  {draw && (
                    <p className="mono">
                      seed {seed} → r = {((seed * 977 + (roll ?? 0)) % 1000 / 1000).toFixed(3)}
                      <br />
                      entropy {H.toFixed(3)} bits over {view?.alive} live candidates
                      <br />
                      dropped {V - (view?.alive ?? V)} by temperature / k / p
                    </p>
                  )}
                </div>

                <div className="samp__presets">
                  <span className="samp__presetHead mono">presets</span>
                  <button type="button" className="btn btn--ghost" onClick={() => { setTemp(0); setTopK(1); setTopP(1); }}>greedy</button>
                  <button type="button" className="btn btn--ghost" onClick={() => { setTemp(1.3); setTopK(0); setTopP(1); }}>chaotic</button>
                  <button type="button" className="btn btn--ghost" onClick={() => { setTemp(0.8); setTopK(40); setTopP(0.95); }}>default</button>
                </div>
              </div>
            </div>

            <div className="samp__ctrls">
              <div className="knob">
                <span className="knob__name">temperature</span>
                <input type="range" aria-label="temperature" min={0} max={1.5} step={0.05} value={temp} onChange={(e) => setTemp(Number(e.target.value))} />
                <span className="knob__val">{temp.toFixed(2)}</span>
              </div>
              <div className="knob">
                <span className="knob__name">top-k</span>
                <input type="range" aria-label="top k" min={1} max={V} step={1} value={topK} onChange={(e) => setTopK(Number(e.target.value))} />
                <span className="knob__val">{topK}</span>
              </div>
              <div className="knob">
                <span className="knob__name">top-p</span>
                <input type="range" aria-label="top p" min={0.05} max={1} step={0.01} value={topP} onChange={(e) => setTopP(Number(e.target.value))} />
                <span className="knob__val">{topP.toFixed(2)}</span>
              </div>
              <div className="knob">
                <span className="knob__name">seed</span>
                <input type="range" aria-label="seed" min={1} max={999} step={1} value={seed} onChange={(e) => setSeed(Number(e.target.value))} />
                <span className="knob__val">{seed}</span>
              </div>
            </div>
          </div>
        </Stage>
      }
      aside={
        <div className="readout">
          <div className="readout__row">
            <span className="readout__k">algorithm</span>
            <span className="readout__v readout__v--muted">T → top-k → top-p → draw</span>
          </div>
          <div className="readout__row">
            <span className="readout__k">live candidates</span>
            <span className="readout__v">{view?.alive ?? "—"}</span>
          </div>
          <div className="readout__row">
            <span className="readout__k">entropy</span>
            <span className="readout__v">{H.toFixed(3)} b</span>
          </div>
          <div className="readout__row">
            <span className="readout__k">greedy</span>
            <span className="readout__v readout__v--muted">T=0, k=1, p=1</span>
          </div>
        </div>
      }
    >
      <p>
        A distribution is not an answer. Sampling is the last transformation: turn{" "}
        <span className="num">{V}</span> probabilities into exactly one token — reproducibly.
      </p>
      <p>
        Three filters run in order.{" "}
        <Term k="temperature" intuition="crank it up and the model rambles; crank it down and it repeats itself." deep={
          <>
            Temperature divides the logits <em>before</em> softmax. As <span className="mono">T → 0</span> the softmax sharpens
            toward a one-hot vector (greedy decoding); as <span className="mono">T → ∞</span> it approaches uniform. Crucially
            the ordering never changes — temperature reshapes confidence, never preference.
          </>
        }>
          Temperature
        </Term>{" "}
        reshapes the curve, <strong>top-k</strong> keeps only the k best, and{" "}
        <Term k="topp" intuition="keep the smallest set of tokens whose probabilities add up to p, so you drop the long tail." deep={
          <>
            Nucleus sampling cuts at a <em>probability mass</em>, not a count: it sorts descending and keeps tokens until the
            cumulative sum reaches <span className="mono">top_p</span>. The alive-set therefore adapts — when the model is
            confident it keeps 2 tokens; when it is unsure it keeps 40.
          </>
        }>
          top-p
        </Term>{" "}
        cuts by cumulative mass.
      </p>
      <p>
        Watch the <strong>after T / after k / after p</strong> columns: rows that die go grey, and their final probability
        becomes exactly <span className="num">0</span> — they are removed from the possibility space before the draw.
      </p>
      <p>
        Hit <strong>sample next token</strong>. The needle bar is the surviving mass, and the winner is the segment the draw
        lands in. Change the <strong>seed</strong> and you get a different draw from the same distribution — the whole point of
        seeds being that the randomness is controlled, not absent.
      </p>
    </Section>
  );
}

function Meter({ v, accent }: { v: number; accent?: boolean }) {
  return (
    <span className="samp__meter">
      <span
        className={`samp__meterFill ${accent ? "is-accent" : ""} ${v === 0 ? "is-zero" : ""}`}
        style={{ width: `${Math.max(v > 0 ? 12 : 0, Math.min(100, v * 100))}%` }}
      />
    </span>
  );
}
