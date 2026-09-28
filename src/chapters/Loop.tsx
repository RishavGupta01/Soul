import { useEffect, useRef, useState } from "react";
import { Section } from "../shell/Section";
import { Term } from "../shell/Term";
import { Stage } from "../viz/Stage";
import { getRuntime } from "../engine/loader";
import { generate } from "../engine/generate";
import { useModelStore, type GeneratedStep } from "../state/useModelStore";
import { useCoreStore } from "../state/useCoreStore";
import { prefersReduced } from "../motion/useReducedMotion";

const PROMPT = "the model";

const PHASES = [
  { key: "feed", label: "feed", note: "prompt ids enter the model" },
  { key: "forward", label: "forward", note: "every layer runs, once" },
  { key: "logits", label: "logits", note: "a score for each of 384 tokens" },
  { key: "sample", label: "sample", note: "one token is chosen" },
  { key: "append", label: "append", note: "the token joins the context" },
] as const;

export function Loop() {
  const rt = getRuntime();
  const reduced = prefersReduced();
  const [steps, setSteps] = useState<GeneratedStep[]>([]);
  const [phase, setPhase] = useState(0);
  const [running, setRunning] = useState(false);
  const [ctxLen, setCtxLen] = useState(0);
  const core = useCoreStore((s) => s.set);
  const cancel = useRef(false);
  const timer = useRef<number | null>(null);

  const cfg = rt?.model.config;
  const ctx = cfg?.contextLength ?? 64;

  useEffect(() => () => {
    cancel.current = true;
    if (timer.current) window.clearTimeout(timer.current);
  }, []);

  const stop = () => {
    cancel.current = true;
    if (timer.current) window.clearInterval(timer.current);
    setRunning(false);
    core({ generating: false, phase: "idle", progress: 0, candidates: [] });
  };

  const start = () => {
    if (!rt) return;
    stop();
    cancel.current = false;
    setRunning(true);
    setSteps([]);
    setCtxLen(rt.tokenizer.encode(PROMPT).length);

    const it = generate(rt.model, rt.tokenizer, {
      prompt: PROMPT,
      maxNewTokens: 48,
      temperature: 0.85,
      topK: 40,
      topP: 0.95,
      seed: 4242,
    });

    let i = 0;
    const gap = reduced ? 30 : 120;

    const advance = (p: number) => {
      setPhase(p);
      core({ phase: (["loop", "loop", "logits", "sample", "loop"] as const)[p] as never, progress: i / 48 });
    };

    const stepFn = () => {
      if (cancel.current) return;
      advance(1); // forward
      window.setTimeout(() => {
        if (cancel.current) return;
        advance(2); // logits
        window.setTimeout(() => {
          if (cancel.current) return;
          const r = it.next();
          if (r.done) {
            stop();
            return;
          }
          advance(3); // sample
          setSteps((s) => [...s, r.value]);
          setCtxLen((c) => c + 1);
          core({ candidates: r.value.top.slice(0, 8).map((c) => ({ text: c.text, p: c.p })), sampled: r.value.tokenText });
          window.setTimeout(() => {
            if (cancel.current) return;
            advance(4); // append
            i++;
            timer.current = window.setTimeout(stepFn, gap);
          }, gap * 0.4);
        }, gap * 0.4);
      }, gap * 0.4);
    };

    advance(0);
    timer.current = window.setTimeout(stepFn, gap);
  };

  const text = steps.map((s) => s.tokenText).join("");
  const last = steps[steps.length - 1];
  const promptIds = rt ? rt.tokenizer.encode(PROMPT) : [];
  const kvUsed = steps.length;
  const kvCells = (cfg?.nLayers ?? 3) * 2 * kvUsed * (cfg?.dModel ?? 64);
  const flops = ctxLen * Math.pow(cfg?.dModel ?? 64, 2) * (cfg?.nLayers ?? 3) * 4;

  return (
    <Section
      id="loop"
      index={7}
      kicker="closing the circuit"
      title="One token, then again"
      stage={
        <Stage
          label="autoregressive"
          labelValue={running ? `step ${steps.length}` : "idle"}
          aria={`Autoregressive loop. ${steps.length} tokens generated so far from prompt "${PROMPT}".`}
          foot={
            <>
              <span className="num">ctx {ctxLen}/{ctx}</span>
              <span className="num">kv cells {kvCells.toLocaleString()} · ≈{(flops / 1e6).toFixed(1)} MFLOP per step</span>
            </>
          }
        >
          <div className="loop">
            <ol className="loop__cycle" aria-label="The generation loop">
              {PHASES.map((p, i) => (
                <li key={p.key}>
                  <div className={`loop__node ${phase === i && running ? "is-active" : ""} ${i < phase ? "is-done" : ""}`}>
                    <span className="loop__idx num">{i + 1}</span>
                    <span className="loop__label">{p.label}</span>
                    <span className="loop__note">{p.note}</span>
                  </div>
                  {i < PHASES.length - 1 && (
                    <span className="loop__arrow" aria-hidden="true">
                      <span className="loop__arrowFill" style={{ opacity: phase > i ? 1 : 0.18 }} />
                    </span>
                  )}
                </li>
              ))}
            </ol>

            <div className="loop__return" aria-hidden="true">
              <span className="loop__returnLine" />
              <span className="loop__returnLabel mono">← append to context, repeat from 1</span>
            </div>

            <div className="loop__out">
              <div className="loop__outHead mono">OUTPUT</div>
              <p className="loop__text" aria-live="polite">
                <span className="loop__prompt">{PROMPT}</span>
                {steps.map((s, i) => (
                  <span className={`tok ${i === steps.length - 1 ? "tok--fresh" : ""}`} key={`${s.pos}-${i}`}>
                    {s.tokenText}
                  </span>
                ))}
                {running && <span className="streambox__caret" aria-hidden="true" />}
                {!running && !steps.length && <span className="streambox__idle">— press run to close the loop —</span>}
              </p>
              <div className="loop__last mono">
                {last
                  ? `last draw: “${last.tokenText.replace(/ /g, "␣")}” · p ${(last.top[0]?.p * 100).toFixed(1)}% · H ${last.entropy.toFixed(2)}b`
                  : "no draws yet"}
              </div>
            </div>

            <div className="loop__ctrls">
              {!running ? (
                <button type="button" className="btn" onClick={start} disabled={!rt}>
                  ▸ run the loop
                </button>
              ) : (
                <button type="button" className="btn btn--ghost" onClick={stop}>
                  ■ stop
                </button>
              )}
              <span className="loop__hint mono">
                {promptIds.length} prompt tokens + {kvUsed} generated = {ctxLen} in context
              </span>
            </div>
          </div>
        </Stage>
      }
      aside={
        <div className="readout">
          <div className="readout__row">
            <span className="readout__k">phase</span>
            <span className="readout__v">{running ? PHASES[phase].label : "idle"}</span>
          </div>
          <div className="readout__row">
            <span className="readout__k">tokens out</span>
            <span className="readout__v">{steps.length}</span>
          </div>
          <div className="readout__row">
            <span className="readout__k">context</span>
            <span className="readout__v">{ctxLen} / {ctx}</span>
          </div>
          <div className="readout__row">
            <span className="readout__k">re-encode cost</span>
            <span className="readout__v readout__v--muted">
              {steps.length ? "cached, see KV chapter" : "—"}
            </span>
          </div>
        </div>
      }
    >
      <p>
        Everything so far was <strong>one</strong> forward pass. Generation is that pass, run{" "}
        <Term k="autoregressive" intuition="predict one word, stick it on the end of the input, predict again — with no idea what you are building toward." deep={
          <>
            There is no planning step. The model commits to each token before it knows the sentence it is writing, which is why
            it can walk itself into a corner and why a single early sample can change the whole continuation. Every token
            conditions on everything already committed — nothing is ever revised.
          </>
        }>
          over and over
        </Term>
        , one token at a time.
      </p>
      <p>
        Press <strong>run the loop</strong> and watch the five phases cycle. The number in the readout that matters is{" "}
        <span className="num">ctx {ctxLen}/{ctx}</span> — context is a budget, and it is spent in exactly one direction.
      </p>
      <p>
        The dashed return path is the whole trick: the token sampled in phase 4 is appended in phase 5 and becomes input in
        phase 1 of the next iteration. <strong>Output becomes input.</strong> Nothing else in the loop is clever.
      </p>
      <p>
        It also explains why generation feels slow at long contexts: each new token pays for every token that came before it.
        Without caching, the model would redo all of it — which is precisely what the{" "}
        <Term k="kv" intuition="keep the keys and values you already computed instead of recomputing them every step." deep={
          <>
            With a KV cache, step <span className="num">t</span> only computes Q/K/V for the single new token and reuses the
            cached <span className="num">K, V</span> columns for <span className="num">0..t−1</span>. Memory grows linearly
            with context; compute per token stays roughly constant.
          </>
        }>
          KV cache
        </Term>{" "}
        exists.
      </p>
    </Section>
  );
}
