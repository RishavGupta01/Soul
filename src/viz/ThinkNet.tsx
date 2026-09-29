import { useEffect, useMemo, useRef } from "react";
import type { LayerPass } from "../state/useModelStore";
import "./../styles/thinknet.css";

interface Props {
  passes?: LayerPass[];
  running?: boolean;
  tokenCount?: number;
  context?: string[];
}

const W = 960;
const H = 260;

const LAYER_X = [352, 486, 620];
const HEAD_Y = 92;
const FFN_Y = 168;
const EMBED_X = 196;
const CORE_X = 88;
const LOGIT_X = 774;
const OUT_X = 900;

/** smooth 0..1 -> css glow budget */
const glow = (v: number) => Math.max(0.12, Math.min(1, v));

export function ThinkNet({ passes, running = false, tokenCount = 0, context = [] }: Props) {
  const wrapRef = useRef<HTMLDivElement>(null);
  const rafRef = useRef<number>(0);
  const phaseRef = useRef(0);

  // Drive the travelling pulse off rAF so it keeps moving between tokens.
  useEffect(() => {
    const reduce = window.matchMedia("(prefers-reduced-motion: reduce)").matches;
    if (reduce) return;
    const tick = () => {
      phaseRef.current = (phaseRef.current + 0.006) % 1;
      const el = wrapRef.current;
      if (el) el.style.setProperty("--pulse", phaseRef.current.toFixed(4));
      rafRef.current = requestAnimationFrame(tick);
    };
    rafRef.current = requestAnimationFrame(tick);
    return () => cancelAnimationFrame(rafRef.current);
  }, []);

  const has = !!passes && passes.length > 0;
  const layers = passes ?? [];
  const topCtx = useMemo(() => (layers.length ? layers[layers.length - 1].attnTop : []), [layers]);

  const heads = 4;

  return (
    <div
      className={`thinknet ${has ? "is-live" : ""} ${running ? "is-running" : ""}`}
      ref={wrapRef}
      role="img"
      aria-label={
        has
          ? `Forward pass network: ${layers.length} layers lit for token ${tokenCount}`
          : "Forward pass network, idle"
      }
    >
      <div className="thinknet__head">
        <span className="thinknet__label">
          forward pass <span className="thinknet__sep">/</span> live activations
        </span>
        <span className="thinknet__meta num">
          {has ? `${layers.length} layers · ${heads} heads · token ${tokenCount}` : "idle"}
        </span>
      </div>

      <svg className="thinknet__svg" viewBox={`0 24 ${W} ${H - 40}`} preserveAspectRatio="xMidYMid slice">
        <defs>
          <linearGradient id="tn-edge" x1="0" y1="0" x2="1" y2="0">
            <stop offset="0%" stopColor="#3DE8FF" stopOpacity="0.1" />
            <stop offset="100%" stopColor="#FF3DAE" stopOpacity="0.5" />
          </linearGradient>
          <filter id="tn-soft" x="-60%" y="-60%" width="220%" height="220%">
            <feGaussianBlur stdDeviation="4" />
          </filter>
        </defs>

        {/* ---- spine edges ---- */}
        <g className="tn-edges" fill="none" strokeWidth="1.25">
          <path d={`M ${CORE_X + 26} ${H / 2} L ${EMBED_X - 26} ${H / 2}`} />
          <path d={`M ${EMBED_X + 26} ${H / 2} L ${LAYER_X[0] - 40} ${H / 2}`} />
          <path d={`M ${LAYER_X[0] + 40} ${H / 2} L ${LAYER_X[1] - 40} ${H / 2}`} />
          <path d={`M ${LAYER_X[1] + 40} ${H / 2} L ${LAYER_X[2] - 40} ${H / 2}`} />
          <path d={`M ${LAYER_X[2] + 40} ${H / 2} L ${LOGIT_X - 34} ${H / 2}`} />
          <path d={`M ${LOGIT_X + 34} ${H / 2} L ${OUT_X - 30} ${H / 2}`} />
        </g>

        {/* ---- travelling pulse ---- */}
        <g className={`tn-pulse ${running ? "is-on" : ""}`} aria-hidden="true">
          <rect x="0" y={H / 2 - 3} width="54" height="6" rx="3" filter="url(#tn-soft)" />
          <rect x="0" y={H / 2 - 2.5} width="54" height="5" rx="2.5" />
        </g>

        {/* ---- context strip: where this token looked ---- */}
        <g className="tn-context">
          <text x={CORE_X} y={38} className="tn-cap">
            CONTEXT
          </text>
          {context.slice(-8).map((t, i) => {
            const x = CORE_X + i * 26;
            const hit = topCtx.find((c) => c.k >= 0);
            const w = hit ? Math.min(1, topCtx[i]?.w ?? 0) : 0;
            return (
              <g key={`${t}-${i}`}>
                <rect
                  x={x - 10}
                  y={48}
                  width="20"
                  height="20"
                  rx="2"
                  className="tn-ctx"
                  style={{ opacity: 0.35 + (topCtx[i] ? Math.min(1, topCtx[i].w * 3) : 0) }}
                />
                <text x={x} y={63} className="tn-ctxTxt">
                  {t.slice(0, 2)}
                </text>
              </g>
            );
          })}
        </g>

        {/* ---- token core ---- */}
        <g className="tn-node">
          <rect x={CORE_X - 26} y={H / 2 - 26} width="52" height="52" rx="3" className="tn-box" />
          <text x={CORE_X} y={H / 2 + 5} className="tn-nodeTxt">
            tok
          </text>
        </g>

        {/* ---- embedding ---- */}
        <g className="tn-node">
          <rect x={EMBED_X - 26} y={H / 2 - 26} width="52" height="52" rx="3" className="tn-box" />
          <text x={EMBED_X} y={H / 2 - 2} className="tn-nodeTxt">
            x
          </text>
          <text x={EMBED_X} y={H / 2 + 14} className="tn-sub">
            64d
          </text>
        </g>

        {/* ---- three transformer blocks ---- */}
        {LAYER_X.map((x, li) => {
          const p: LayerPass | undefined = layers[li];
          const attn = p ? glow(p.attnFocus) : 0.14;
          const ffn = p ? glow(p.ffnEnergy) : 0.14;
          return (
            <g key={x} className={`tn-layer ${p ? "is-on" : ""}`}>
              <rect
                x={x - 40}
                y={HEAD_Y - 46}
                width="80"
                height={FFN_Y + 46 - (HEAD_Y - 46)}
                rx="3"
                className="tn-layerBox"
                style={{ ["--a" as string]: attn }}
              />
              <text x={x} y={HEAD_Y - 54} className="tn-cap">
                L{li + 1}
              </text>
              {Array.from({ length: heads }).map((_, h) => {
                const peak = p?.headPeak[h] ?? 0;
                return (
                  <g key={h}>
                    <circle
                      cx={x - 24 + h * 16}
                      cy={HEAD_Y}
                      r="6"
                      className="tn-head"
                      style={{ opacity: p ? glow(0.2 + peak) : 0.2 }}
                    />
                    <text x={x - 24 + h * 16} y={HEAD_Y + 20} className="tn-tiny">
                      h{h}
                    </text>
                  </g>
                );
              })}
              <rect
                x={x - 30}
                y={FFN_Y - 14}
                width="60"
                height="28"
                rx="2"
                className="tn-ffn"
                style={{ opacity: p ? glow(0.25 + ffn) : 0.22 }}
              />
              <text x={x} y={FFN_Y + 4} className="tn-tiny tn-ffnTxt">
                FFN
              </text>
            </g>
          );
        })}

        {/* ---- logits ridge ---- */}
        <g className="tn-logits">
          <text x={LOGIT_X} y={HEAD_Y - 54} className="tn-cap">
            LOGITS
          </text>
          {Array.from({ length: 7 }).map((_, i) => {
            const p = layers.length ? layers[layers.length - 1].ffnEnergy : 0;
            const w = has ? 10 + p * 46 * (1 - i * 0.11) : 8;
            return (
              <rect
                key={i}
                x={LOGIT_X - 34}
                y={HEAD_Y - 34 + i * 12}
                width={w}
                height="7"
                rx="1.5"
                className="tn-bar"
                style={{ opacity: has ? 1 - i * 0.1 : 0.3 }}
              />
            );
          })}
        </g>

        {/* ---- sampled token ---- */}
        <g className="tn-node">
          <rect
            x={OUT_X - 30}
            y={H / 2 - 26}
            width="60"
            height="52"
            rx="3"
            className={`tn-box tn-box--out ${has ? "is-on" : ""}`}
          />
          <text x={OUT_X} y={H / 2 + 4} className="tn-nodeTxt">
            {has ? "p" : "?"}
          </text>
          <text x={OUT_X} y={H / 2 + 18} className="tn-sub">
            sample
          </text>
        </g>
      </svg>

      <div className="thinknet__legend mono">
        <span>
          <i className="tn-dot tn-dot--attn" /> attention focus
        </span>
        <span>
          <i className="tn-dot tn-dot--ffn" /> ffn energy
        </span>
        <span className="thinknet__hint">
          {has ? "every number read off the live forward pass" : "press generate to light the network"}
        </span>
      </div>
    </div>
  );
}

export default ThinkNet;
