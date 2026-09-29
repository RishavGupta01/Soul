import { useMemo, useState } from "react";
import { Section } from "../shell/Section";
import { Term } from "../shell/Term";
import { Stage } from "../viz/Stage";
import { softmax } from "../engine/tensor";
import { rankDesc } from "../engine/sampler";
import { useRuntime } from "../engine/useRuntime";

const DEMO = "the capital of france is";

export function Logits() {
  const rt = useRuntime();
  const [temp, setTemp] = useState(1);
  const [cursor, setCursor] = useState(0);

  const V = rt?.model.config.vocabSize ?? 384;

  const result = useMemo(() => {
    if (!rt) return null;
    const ids = rt.tokenizer.encode(DEMO);
    try {
      const res = rt.model.forward(ids, 0);
      const logits = res.logits;
      const order = rankDesc(logits);
      const probs = softmax(Float64Array.from(logits), temp);
      return { logits, order, probs };
    } catch {
      return null;
    }
  }, [rt, temp]);

  const top = useMemo(() => (result ? result.order.slice(0, 36) : []), [result]);
  const shown = top.slice(cursor, cursor + 18);

  const lo = result ? Math.min(...result.logits) : 0;
  const hi = result ? Math.max(...result.logits) : 1;
  const peak = result && top.length ? result.logits[top[0]] : 0;

  const pMax = result && top.length ? result.probs[top[0]] : 0;

  return (
    <Section
      id="logits"
      index={5}
      kicker="one number per word"
      title="Nothing is decided yet"
      stage={
        <Stage
          label="lm_head"
          labelValue={`[${V}] · t=${temp.toFixed(2)}`}
          aria={`Logit vector over ${V} vocabulary entries. Top candidate "${result && top.length ? rt?.tokenizer.decode([top[0]]) : "—"}" with probability ${(pMax * 100).toFixed(1)} percent.`}
          foot={
            <>
              <span className="num">
                range [{lo.toFixed(2)}, {hi.toFixed(2)}]
              </span>
              <span className="num">
                argmax p = {(pMax * 100).toFixed(2)}% · H = {entropyOf(result?.probs, top)} b
              </span>
            </>
          }
        >
          <div className="logits">
            <div className="logits__axis">
              <span className="num">+{hi.toFixed(2)}</span>
              <span className="logits__axisLine" aria-hidden="true" />
              <span className="num">{lo.toFixed(2)}</span>
            </div>

            <div className="logits__bars">
              {shown.map((id) => {
                const l = result?.logits[id] ?? 0;
                const p = result?.probs[id] ?? 0;
                const norm = ((l - lo) / (hi - lo || 1)) * 100;
                const text = rt?.tokenizer.decode([id]) ?? "?";
                return (
                  <div className="logits__row" key={id}>
                    <span className="logits__rank num">{result ? result.order.indexOf(id) + 1 : "—"}</span>
                    <span className="logits__tok mono" title={`id ${id}`}>
                      {text === " " ? "␣" : text.replace(/\n/g, "⏎")}
                    </span>
                    <span className="logits__barWrap">
                      <span className="logits__bar" style={{ width: `${Math.max(1, norm)}%` }}>
                        <span className="logits__prob" style={{ width: `${(p / (pMax || 1)) * 100}%` }} />
                      </span>
                    </span>
                    <span className="logits__val num">{l.toFixed(3)}</span>
                    <span className="logits__p num">{(p * 100).toFixed(2)}%</span>
                  </div>
                );
              })}
              {!result && <p className="logits__empty mono">waiting for weights…</p>}
            </div>

            <div className="logits__ctrls">
              <div className="knob">
                <span className="knob__name">temperature</span>
                <input type="range" aria-label="temperature" min={0.05} max={1.5} step={0.05} value={temp} onChange={(e) => setTemp(Number(e.target.value))} />
                <span className="knob__val">{temp.toFixed(2)}</span>
              </div>
              <div className="knob">
                <span className="knob__name">rank window</span>
                <input type="range" aria-label="rank window" min={0} max={Math.max(0, V - 18)} step={1} value={cursor} onChange={(e) => setCursor(Number(e.target.value))} />
                <span className="knob__val">{cursor + 1}</span>
              </div>
              <div className="logits__formula mono">
                p<sub>i</sub> = exp(z<sub>i</sub>/T) / Σ exp(z<sub>j</sub>/T)
              </div>
            </div>
          </div>
        </Stage>
      }
      aside={
        <div className="readout">
          <div className="readout__row">
            <span className="readout__k">prompt</span>
            <span className="readout__v readout__v--muted">“{DEMO}”</span>
          </div>
          <div className="readout__row">
            <span className="readout__k">logits</span>
            <span className="readout__v">[{V}]</span>
          </div>
          <div className="readout__row">
            <span className="readout__k">temperature</span>
            <span className="readout__v">{temp.toFixed(2)}</span>
          </div>
          <div className="readout__row">
            <span className="readout__k">model</span>
            <span className="readout__v readout__v--muted">
              {rt ? `${rt.model.config.nLayers}L · ${rt.model.config.nHeads}H · ${rt.model.config.dModel}d` : "—"}
            </span>
          </div>
        </div>
      }
    >
      <p>
        After the final layer norm, the hidden vector meets the LM head — a single matrix multiply that asks the same question
        of all <span className="num">{V}</span> vocabulary rows: <em>how much do you want to be next?</em>
      </p>
      <p>
        The answer is <strong>unbounded real numbers</strong>, one per token. They are called{" "}
        <Term k="logits" intuition="raw scores, one per possible next word. Bigger means the model leans that way, but they don't add up to anything." deep={
          <>
            A logit is an unnormalized log-probability. They are only meaningful <em>relative to each other</em> — adding a
            constant to all of them changes nothing after softmax. Only the differences carry information, which is why the
            readout shows the range <span className="mono">[{lo.toFixed(2)}, {hi.toFixed(2)}]</span> rather than absolute
            magnitudes.
          </>
        }>
          logits
        </Term>
        . Drag the temperature slider and watch the <strong>bars stay put</strong> while the <strong>percentages move</strong>:
        temperature never touches the logits, only the distribution derived from them.
      </p>
      <p>
        The bright bar behind each percentage is the logit height; the inner fill is the probability. When{" "}
        <span className="num">T</span> drops toward zero the top percentage climbs toward <span className="num">100%</span> and
        everything else flattens to nothing.
      </p>
      <p>
        Nothing has been chosen yet. That happens next — and it is deliberately <strong>not</strong> a coin flip weighted by
        these numbers alone.
      </p>
    </Section>
  );
}

function entropyOf(probs: Float64Array | number[] | undefined, order: number[]): string {
  if (!probs) return "0.00";
  let h = 0;
  for (const i of order) {
    const p = probs[i];
    if (p > 0) h -= p * Math.log2(p);
  }
  return h.toFixed(2);
}
