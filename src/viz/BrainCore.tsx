import { useEffect, useRef } from "react";
import { useCoreStore, type CorePhase } from "../state/useCoreStore";
import { prefersReduced } from "../motion/useReducedMotion";

const CYAN = "61,232,255";
const MAGENTA = "255,61,174";

interface Particle {
  a: number;
  b: number;
  r: number;
  phase: number;
  speed: number;
  size: number;
}

interface Pct {
  x: number;
  y: number;
}

/**
 * The BrainCore — one object, always on screen, always thinking.
 * All decoration here is data: arcs are real attention rows, the sweep is a
 * token position, and the pulse fires only while the model generates.
 */
export function BrainCore() {
  const ref = useRef<HTMLCanvasElement>(null);
  const phaseRef = useRef<CorePhase>("idle");
  const attRef = useRef(useCoreStore.getState().attention);
  const progressRef = useRef(0);
  const genRef = useRef(false);

  useEffect(() => {
    const unsub = useCoreStore.subscribe((s) => {
      phaseRef.current = s.phase;
      attRef.current = s.attention;
      progressRef.current = s.progress;
      genRef.current = s.generating;
    });
    return unsub;
  }, []);

  useEffect(() => {
    const canvas = ref.current;
    if (!canvas) return;
    const ctx = canvas.getContext("2d");
    if (!ctx) return;

    const reduced = prefersReduced();
    let w = 0;
    let h = 0;
    let dpr = 1;

    const resize = () => {
      const rect = canvas.getBoundingClientRect();
      dpr = Math.min(2, window.devicePixelRatio || 1);
      w = Math.max(1, rect.width);
      h = Math.max(1, rect.height);
      canvas.width = Math.round(w * dpr);
      canvas.height = Math.round(h * dpr);
      ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
    };
    resize();
    const ro = new ResizeObserver(resize);
    ro.observe(canvas);

    // --- deterministic particle field (144 nodes = nice lattice) ---
    const N = 144;
    const rng = mulberry(1337);
    const pts: Particle[] = Array.from({ length: N }, () => ({
      a: rng() * Math.PI * 2,
      b: Math.acos(2 * rng() - 1),
      r: 0.55 + rng() * 0.5,
      phase: rng() * Math.PI * 2,
      speed: 0.12 + rng() * 0.4,
      size: 0.7 + rng() * 1.6,
    }));

    let t = 0;
    let raf = 0;

    const project = (p: Particle, cx: number, cy: number, R: number): Pct => {
      const spin = t * p.speed;
      const a = p.a + spin;
      const b = p.b + Math.sin(t * 0.3 + p.phase) * 0.12;
      const sx = Math.sin(b) * Math.cos(a);
      const sy = Math.cos(b);
      const sz = Math.sin(b) * Math.sin(a);
      const depth = (sz + 1) / 2; // 0 back .. 1 front
      return {
        x: cx + sx * R * p.r,
        y: cy + sy * R * p.r * 0.92,
        // eslint-disable-next-line @typescript-eslint/no-explicit-any
        ...({ d: depth } as any),
      };
    };

    const draw = () => {
      t += reduced ? 0 : 0.0085;
      const phase = phaseRef.current;
      const att = attRef.current;
      const gen = genRef.current;

      ctx.clearRect(0, 0, w, h);

      const cx = w / 2;
      const cy = h / 2;
      const R = Math.min(w, h) * 0.34;

      // ---- attention arcs (real rows) ----
      if (att && att.rows.length) {
        drawAttention(ctx, att, cx, cy, R, t, reduced);
      }

      // ---- lattice ----
      const hover = phase === "embed" ? 1.35 : phase === "idle" ? 1 : 1.08;
      const positions = pts.map((p) => project(p, cx, cy, R * hover));

      // faint connections
      ctx.lineWidth = 0.5;
      for (let i = 0; i < N; i++) {
        const a = positions[i];
        for (let j = i + 1; j < i + 7 && j < N; j++) {
          const b = positions[j];
          const dx = a.x - b.x;
          const dy = a.y - b.y;
          const dist = Math.hypot(dx, dy);
          if (dist > R * 0.42) continue;
          const alpha = (1 - dist / (R * 0.42)) * 0.07;
          ctx.strokeStyle = `rgba(255,255,255,${alpha})`;
          ctx.beginPath();
          ctx.moveTo(a.x, a.y);
          ctx.lineTo(b.x, b.y);
          ctx.stroke();
        }
      }

      // nodes
      const nodeColor = phase === "attend" ? MAGENTA : CYAN;
      for (let i = 0; i < N; i++) {
        const p = pts[i];
        const pos = positions[i] as Pct & { d?: number };
        const depth = (pos as unknown as { d: number }).d ?? 0.5;
        const breathe = reduced ? 1 : 0.75 + 0.25 * Math.sin(t * 2 + p.phase);
        const alpha = (0.18 + depth * 0.62) * breathe;
        ctx.fillStyle = `rgba(${nodeColor},${alpha.toFixed(3)})`;
        ctx.beginPath();
        ctx.arc(pos.x, pos.y, p.size * (0.7 + depth * 0.8), 0, Math.PI * 2);
        ctx.fill();
      }

      // ---- core crystal ----
      drawCore(ctx, cx, cy, R, t, phase, gen, reduced, progressRef.current);

      // ---- token sweep while generating ----
      if (gen && !reduced) {
        const prog = progressRef.current;
        const ang = prog * Math.PI * 2 - Math.PI / 2;
        ctx.strokeStyle = `rgba(${CYAN},0.55)`;
        ctx.lineWidth = 1.5;
        ctx.beginPath();
        ctx.arc(cx, cy, R * 1.16, -Math.PI / 2, ang);
        ctx.stroke();
        ctx.fillStyle = `rgba(${CYAN},0.9)`;
        ctx.beginPath();
        ctx.arc(cx + Math.cos(ang) * R * 1.16, cy + Math.sin(ang) * R * 1.16, 3, 0, Math.PI * 2);
        ctx.fill();
      }

      raf = requestAnimationFrame(draw);
    };

    draw();
    return () => {
      cancelAnimationFrame(raf);
      ro.disconnect();
    };
  }, []);

  return (
    <canvas
      ref={ref}
      className="braincore"
      role="img"
      aria-label="The BrainCore: the live model visualized as a crystalline lattice. It shows attention arcs between tokens and pulses while the model generates."
    />
  );
}

function drawCore(
  ctx: CanvasRenderingContext2D,
  cx: number,
  cy: number,
  R: number,
  t: number,
  phase: CorePhase,
  gen: boolean,
  reduced: boolean,
  progress: number
) {
  const rings = [
    { scale: 1, alpha: 0.16, lw: 1.2, sides: 6 },
    { scale: 0.72, alpha: 0.3, lw: 1, sides: 5 },
    { scale: 0.44, alpha: 0.5, lw: 1, sides: 4 },
  ];

  ctx.save();
  for (const ring of rings) {
    const rot = reduced ? 0 : t * (0.18 + (1 - ring.scale) * 0.5) * (ring.sides % 2 ? -1 : 1);
    ctx.strokeStyle = `rgba(${CYAN},${ring.alpha})`;
    ctx.lineWidth = ring.lw;
    ctx.beginPath();
    for (let i = 0; i <= ring.sides; i++) {
      const a = rot + (i / ring.sides) * Math.PI * 2;
      const x = cx + Math.cos(a) * R * 0.42 * ring.scale;
      const y = cy + Math.sin(a) * R * 0.42 * ring.scale;
      if (i === 0) ctx.moveTo(x, y);
      else ctx.lineTo(x, y);
    }
    ctx.stroke();
  }

  // inner light — brightness follows what the model is doing
  const activity =
    phase === "idle" ? 0.22 : phase === "logits" || phase === "sample" ? 0.75 : gen ? 0.62 : 0.4;
  const breath = reduced ? 1 : 0.86 + 0.14 * Math.sin(t * 1.6);
  const rad = R * 0.3 * breath;
  const g = ctx.createRadialGradient(cx, cy, 0, cx, cy, rad);
  const tint = phase === "attend" ? MAGENTA : CYAN;
  g.addColorStop(0, `rgba(${tint},${(activity * 0.5).toFixed(3)})`);
  g.addColorStop(1, "rgba(5,5,7,0)");
  ctx.fillStyle = g;
  ctx.beginPath();
  ctx.arc(cx, cy, rad, 0, Math.PI * 2);
  ctx.fill();

  // progress notches (token position during generation)
  if (gen && !reduced) {
    for (let i = 0; i < 40; i++) {
      const a = (i / 40) * Math.PI * 2 - Math.PI / 2;
      const on = i / 40 <= progress;
      ctx.fillStyle = on ? `rgba(${CYAN},0.75)` : "rgba(255,255,255,0.08)";
      const rr = R * 0.5;
      ctx.fillRect(cx + Math.cos(a) * rr - 1, cy + Math.sin(a) * rr - 1, 2, 2);
    }
  }

  // phase label tick — a small solid marker instead of a colored dot
  ctx.restore();
}

function drawAttention(
  ctx: CanvasRenderingContext2D,
  att: { head: number; rows: number[][]; tokens: string[]; query: number },
  cx: number,
  cy: number,
  R: number,
  t: number,
  reduced: boolean
) {
  const row = att.rows[att.query] ?? att.rows[0];
  if (!row) return;
  const n = Math.min(row.length, att.tokens.length);
  const queryA = (att.query / n) * Math.PI * 2 - Math.PI / 2;

  ctx.save();
  for (let i = 0; i < n; i++) {
    const w = row[i];
    if (w <= 0.008) continue;
    const targetA = (i / n) * Math.PI * 2 - Math.PI / 2;
    const qx = cx + Math.cos(queryA) * R * 0.5;
    const qy = cy + Math.sin(queryA) * R * 0.5;
    const tx = cx + Math.cos(targetA) * R * 0.5;
    const ty = cy + Math.sin(targetA) * R * 0.5;
    const mx = (qx + tx) / 2;
    const my = (qy + ty) / 2;
    const pull = 0.35 + w * 0.5;
    const ctrlX = cx + (mx - cx) * pull;
    const ctrlY = cy + (my - cy) * pull;

    ctx.strokeStyle = `rgba(${MAGENTA},${(0.1 + w * 0.85).toFixed(3)})`;
    ctx.lineWidth = 0.4 + w * 2.4;
    ctx.beginPath();
    ctx.moveTo(qx, qy);
    ctx.quadraticCurveTo(ctrlX, ctrlY, tx, ty);
    ctx.stroke();

    if (!reduced && w > 0.25) {
      // data pulse traveling the arc
      const s = (t * 0.55 + i * 0.13) % 1;
      const px = quad(qx, ctrlX, tx, s);
      const py = quad(qy, ctrlY, ty, s);
      ctx.fillStyle = `rgba(255,255,255,${(w * 0.9).toFixed(3)})`;
      ctx.beginPath();
      ctx.arc(px, py, 1.6 + w * 1.6, 0, Math.PI * 2);
      ctx.fill();
    }
  }
  ctx.restore();
}

function quad(a: number, b: number, c: number, s: number): number {
  const u = 1 - s;
  return u * u * a + 2 * u * s * b + s * s * c;
}

function mulberry(seed: number): () => number {
  let a = seed >>> 0;
  return () => {
    a |= 0;
    a = (a + 0x6d2b79f5) | 0;
    let x = Math.imul(a ^ (a >>> 15), 1 | a);
    x = (x + Math.imul(x ^ (x >>> 7), 61 | x)) ^ x;
    return ((x ^ (x >>> 14)) >>> 0) / 4294967296;
  };
}
