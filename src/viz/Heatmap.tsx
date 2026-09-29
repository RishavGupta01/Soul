import { useEffect, useMemo, useRef, useState } from "react";

const ROW_STAGGER_MS = 180;

/**
 * Attention heatmap. Heat scale is fixed: void -> violet -> white, so cells
 * are comparable across heads and chapters. Builds row-by-row when
 * `revealRows` is animated.
 */
export function Heatmap({
  rows,
  tokens,
  query = -1,
  onSelect,
  animate = false,
  size = 0,
  causal = true,
}: {
  rows: number[][];
  tokens: string[];
  query?: number;
  onSelect?: (i: number) => void;
  animate?: boolean;
  size?: number;
  /** Void the j > i triangle (set false to reveal pre-mask scores). */
  causal?: boolean;
}) {
  const canvas = useRef<HTMLCanvasElement>(null);
  const n = rows.length;
  const cell = size || Math.max(8, Math.min(46, Math.floor(340 / Math.max(4, n))));

  const [shown, setShown] = useState(animate ? 0 : n);
  useEffect(() => {
    if (!animate) {
      setShown(n);
      return;
    }
    let i = 0;
    setShown(0);
    const id = window.setInterval(() => {
      i++;
      setShown(Math.min(n, i));
      if (i >= n) window.clearInterval(id);
    }, ROW_STAGGER_MS);
    return () => window.clearInterval(id);
  }, [n, animate]);

  useEffect(() => {
    const c = canvas.current;
    if (!c) return;
    const ctx = c.getContext("2d");
    if (!ctx) return;
    const dpr = Math.min(2, window.devicePixelRatio || 1);
    const W = n * cell;
    c.width = W * dpr;
    c.height = n * cell * dpr;
    c.style.width = `${W}px`;
    c.style.height = `${n * cell}px`;
    ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
    ctx.clearRect(0, 0, W, n * cell);

    for (let i = 0; i < Math.min(shown, n); i++) {
      const row = rows[i];
      for (let j = 0; j < Math.min(row.length, n); j++) {
        const v = row[j];
        ctx.fillStyle = heat(v);
        ctx.fillRect(j * cell, i * cell, cell - 0.5, cell - 0.5);
      }
    }

    // causality: the j > i triangle stays void
    if (causal) {
      ctx.fillStyle = "rgba(5,5,7,1)";
      for (let i = 0; i < Math.min(shown, n); i++) {
        const cols = n - i - 1;
        if (cols > 0) ctx.fillRect((i + 1) * cell, i * cell, cols * cell, cell);
      }
    }

    if (query >= 0 && query < n) {
      ctx.strokeStyle = "rgba(61,232,255,0.95)";
      ctx.lineWidth = 1.5;
      ctx.strokeRect(0, query * cell + 0.75, n * cell - 1.5, cell - 1.5);
      ctx.strokeStyle = "rgba(61,232,255,0.4)";
      ctx.strokeRect(query * cell + 0.75, 0, cell - 1.5, n * cell - 1.5);
    }
  }, [rows, n, cell, query, causal, shown]);

  return (
    <canvas
      ref={canvas}
      className="heatmap"
      onClick={(e) => {
        if (!onSelect) return;
        const r = e.currentTarget.getBoundingClientRect();
        const i = Math.floor((e.clientY - r.top) / cell);
        if (i >= 0 && i < n) onSelect(i);
      }}
      style={{ width: n * cell, height: n * cell }}
    />
  );
}

export function heat(v: number): string {
  const t = Math.max(0, Math.min(1, v));
  // void #050507 -> violet #8B5CFF -> white
  const r = Math.round(5 + t * t * (139 - 5) + t * 0);
  const g = Math.round(5 + Math.pow(t, 1.7) * (92 - 5));
  const b = Math.round(7 + Math.pow(t, 0.7) * (255 - 7));
  const boost = t > 0.55 ? (t - 0.55) / 0.45 : 0;
  return `rgb(${Math.min(255, Math.round(r + boost * (255 - r)))},${Math.min(
    255,
    Math.round(g + boost * (255 - g))
  )},${b})`;
}

/** Token axis labels for a heatmap. */
export function TokenAxis({ tokens }: { tokens: string[] }) {
  const list = useMemo(() => tokens.slice(0, 64), [tokens]);
  return (
    <div className="axis">
      {list.map((t, i) => (
        <span key={i} className="axis__cell mono" title={t}>
          {t === " " ? "␣" : t.replace(/\n/g, "⏎")}
        </span>
      ))}
    </div>
  );
}
