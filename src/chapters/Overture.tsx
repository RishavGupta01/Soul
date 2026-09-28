import { useEffect, useRef, useState } from "react";
import { useGeneration } from "../engine/useGeneration";
import { useModelStore } from "../state/useModelStore";
import { useCoreStore } from "../state/useCoreStore";
import { BrainCore } from "../viz/BrainCore";
import { prefersReduced } from "../motion/useReducedMotion";

const START_PROMPT = "Attention is ";

export function Overture() {
  const { run, stop, steps, running, status, setPrompt } = useGeneration();
  const setStore = useModelStore((s) => s.patch);
  const metrics = useModelStore((s) => s.metrics);
  const core = useCoreStore((s) => s.set);
  const [localPrompt, setLocalPrompt] = useState(START_PROMPT);
  const [ready, setReady] = useState(false);
  const scrollY = useRef(0);

  // Reactive knobs (subscription so sliders re-render)
  const t = useModelStore((s) => s.temperature);
  const k = useModelStore((s) => s.topK);
  const p = useModelStore((s) => s.topP);
  const seedN = useModelStore((s) => s.seed);
  const setKnob = useModelStore((s) => s.setKnob);
  const stats = useModelStore((s) => s.steps);

  useEffect(() => {
    const id = window.setTimeout(() => setReady(true), 60);
    return () => window.clearTimeout(id);
  }, []);

  // Keep the BrainCore informed about generation + the probability ridge.
  useEffect(() => {
    const last = stats[stats.length - 1];
    core({
      generating: running,
      phase: running ? "logits" : "idle",
      progress: stats.length ? (stats.length % 64) / 64 : 0,
      candidates: last ? last.top.slice(0, 10).map((c) => ({ text: c.text, p: c.p })) : [],
      sampled: last ? last.tokenText : null,
    });
  }, [stats, running, core]);

  const onGenerate = () => {
    setPrompt(localPrompt);
    setStore({ prompt: localPrompt });
    run({ prompt: localPrompt, maxTokens: 84, gap: prefersReduced() ? 40 : 130 });
  };

  const last = stats[stats.length - 1];
  const entropy = last ? last.entropy : null;

  return (
    <div className={`overture ${ready ? "is-ready" : ""}`}>
      <div className="overture__core">
        <BrainCore />
      </div>

      <div className="overture__inner">
        <p className="overture__eyebrow" data-reveal>
          <span className="num">00</span>
          <span className="overture__rule" aria-hidden="true" />
          an interactive anatomy of a language model
        </p>

        <h1 className="overture__title">
          <span className="overture__line">WATCH IT</span>
          <span className="overture__line outset">THINK</span>
        </h1>

        <p className="overture__lede" data-reveal>
          Not a metaphor. A real 60-thousand-parameter transformer, trained in front of you,
          running its actual math in your browser — tokens in, attention arcs drawn, a
          distribution collapsed into one sampled token.
        </p>

        <div className="overture__console" data-reveal>
          <div className="promptbar">
            <label className="promptbar__label" htmlFor="prompt">
              PROMPT
            </label>
            <input
              id="prompt"
              className="promptbar__input"
              value={localPrompt}
              onChange={(e) => setLocalPrompt(e.target.value)}
              onKeyDown={(e) => e.key === "Enter" && onGenerate()}
              placeholder="Attention is "
              spellCheck={false}
              autoComplete="off"
            />
            <div className="promptbar__actions">
              {!running ? (
                <button type="button" className="btn" onClick={onGenerate} disabled={status !== "ready"}>
                  ▸ Generate
                </button>
              ) : (
                <button type="button" className="btn btn--ghost" onClick={stop}>
                  ■ Stop
                </button>
              )}
            </div>
          </div>

          <div className="streambox">
            <div className="streambox__head">
              <span>OUTPUT STREAM</span>
              <span className="streambox__meta num">
                {stats.length} tok{entropy !== null ? ` · H ${entropy.toFixed(2)} b` : ""}
              </span>
            </div>
            <p className="streambox__text" aria-live="polite">
              <span className="streambox__prompt">{localPrompt}</span>
              {stats.map((s, i) => (
                <span
                  key={`${s.pos}-${i}`}
                  className={`tok ${i === stats.length - 1 ? "tok--fresh" : ""}`}
                  style={{ animationDelay: `${Math.min(i, 40) * 8}ms` }}
                >
                  {s.tokenText}
                </span>
              ))}
              {running && <span className="streambox__caret" aria-hidden="true" />}
              {!running && stats.length === 0 && (
                <span className="streambox__idle">— press generate to watch a forward pass —</span>
              )}
            </p>
          </div>

          <div className="ridge" aria-hidden={!last}>
            <div className="ridge__head">
              <span>TOP-{Math.min(10, last?.top.length ?? 0)} NEXT-TOKEN DISTRIBUTION</span>
              <span className="num">temp {t.toFixed(2)} · k {k} · p {p.toFixed(2)} · seed {seedN}</span>
            </div>
            <div className="ridge__bars">
              {(last?.top ?? []).slice(0, 10).map((c, i) => (
                <div className="ridge__row" key={c.id}>
                  <span className="ridge__tok mono">{c.text === " " ? "␣" : c.text}</span>
                  <span className="ridge__track">
                    <span
                      className={`ridge__fill ${i === 0 ? "is-top" : ""}`}
                      style={{ width: `${Math.max(1.5, c.p * 100)}%` }}
                    />
                  </span>
                  <span className="ridge__p num">{(c.p * 100).toFixed(1)}%</span>
                </div>
              ))}
              {!last && <span className="ridge__empty">distribution appears on first sample</span>}
            </div>
          </div>

          <div className="overture__knobs">
            <Knob name="temperature" value={t} min={0} max={1.5} step={0.05} fmt={(v) => v.toFixed(2)} onChange={(v) => setKnob("temperature", v)} />
            <Knob name="top-k" value={k} min={1} max={384} step={1} fmt={(v) => String(v)} onChange={(v) => setKnob("topK", v)} />
            <Knob name="top-p" value={p} min={0.05} max={1} step={0.05} fmt={(v) => v.toFixed(2)} onChange={(v) => setKnob("topP", v)} />
            <Knob name="seed" value={seedN} min={1} max={999} step={1} fmt={(v) => String(v)} onChange={(v) => setKnob("seed", v)} />
          </div>

          <div className="overture__readout readout">
            <div className="readout__row">
              <span className="readout__k">model</span>
              <span className="readout__v readout__v--muted">
                {metrics ? `${metrics.config.dModel}d · ${metrics.config.nLayers}L · ${metrics.config.nHeads}H · ${metrics.config.vocabSize} vocab` : "—"}
              </span>
            </div>
            <div className="readout__row">
              <span className="readout__k">val loss</span>
              <span className="readout__v">{typeof metrics?.lastVal === "number" ? metrics.lastVal.toFixed(3) : "—"}</span>
            </div>
            <div className="readout__row">
              <span className="readout__k">weights</span>
              <span className="readout__v readout__v--muted">loaded · trained live, not copied</span>
            </div>
          </div>
        </div>

        <a className="overture__scroll" href="#/tokens" data-reveal>
          <span>begin the descent</span>
          <span className="overture__arrow" aria-hidden="true" />
        </a>
      </div>
    </div>
  );
}

function Knob({
  name,
  value,
  min,
  max,
  step,
  fmt,
  onChange,
}: {
  name: string;
  value: number;
  min: number;
  max: number;
  step: number;
  fmt: (v: number) => string;
  onChange: (v: number) => void;
}) {
  return (
    <div className="knob">
      <span className="knob__name">{name}</span>
      <input
        type="range"
        aria-label={name}
        min={min}
        max={max}
        step={step}
        value={value}
        onChange={(e) => onChange(Number(e.target.value))}
      />
      <span className="knob__val">{fmt(value)}</span>
    </div>
  );
}
