import { useEffect, useState } from "react";
import { CHAPTERS, useRunStore, type ChapterDef } from "../state/useRunStore";

function matchScore(route: string, c: ChapterDef): number {
  if (c.route === "#/") return route === "#/" ? 2 : 1;
  if (route === c.route || route.startsWith(c.route + "/")) return c.route.length;
  return -1;
}

export function ChapterRail() {
  const route = useRunStore((s) => s.route);
  const reached = useRunStore((s) => s.reached);
  const [open, setOpen] = useState(false);

  let active = 0;
  let best = -1;
  CHAPTERS.forEach((c, i) => {
    const score = matchScore(route, c);
    if (score > best) {
      best = score;
      active = i;
    }
  });

  const go = (i: number) => {
    const c = CHAPTERS[i];
    if (c) window.location.hash = c.route;
    setOpen(false);
  };

  useEffect(() => {
    const onKey = (e: KeyboardEvent) => {
      if (e.key !== "ArrowLeft" && e.key !== "ArrowRight") return;
      const t = e.target;
      if (!(t instanceof Element) || !t.closest(".rail")) return;
      if (t instanceof HTMLInputElement || t instanceof HTMLTextAreaElement) return;
      e.preventDefault();
      go(
        e.key === "ArrowRight"
          ? Math.min(CHAPTERS.length - 1, active + 1)
          : Math.max(0, active - 1)
      );
    };
    window.addEventListener("keydown", onKey);
    return () => window.removeEventListener("keydown", onKey);
  }, [active]);

  return (
    <nav
      className={`rail ${open ? "rail--open" : ""}`}
      aria-label="Chapters"
      onMouseEnter={() => setOpen(true)}
      onMouseLeave={() => setOpen(false)}
      onFocus={() => setOpen(true)}
      onBlur={(e) => {
        if (!e.currentTarget.contains(e.relatedTarget as Node | null)) setOpen(false);
      }}
    >
      <button
        type="button"
        className="rail__toggle"
        onClick={() => setOpen(!open)}
        aria-expanded={open}
        aria-label="Chapter navigation"
      >
        <span className="rail__current">{String(CHAPTERS[active].index).padStart(2, "0")}</span>
        <span className="rail__ticks" aria-hidden="true">
          {CHAPTERS.map((c) => (
            <span
              key={c.id}
              className={`rail__tick ${c.route === CHAPTERS[active].route ? "is-active" : ""} ${
                c.index <= reached ? "is-seen" : ""
              }`}
            />
          ))}
        </span>
      </button>

      <ol className="rail__list">
        {CHAPTERS.map((c, i) => (
          <li key={c.id}>
            <a
              href={c.route}
              className={`rail__item ${i === active ? "is-active" : ""} ${
                c.index <= reached ? "is-seen" : ""
              }`}
              onClick={() => setOpen(false)}
            >
              <span className="rail__idx num">{String(c.index).padStart(2, "0")}</span>
              <span className="rail__nav">{c.nav}</span>
              <span className="rail__title">{c.title}</span>
            </a>
          </li>
        ))}
      </ol>
    </nav>
  );
}
