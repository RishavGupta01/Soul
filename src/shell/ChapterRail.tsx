import { useEffect, useState } from "react";
import { CHAPTERS, useRunStore } from "../state/useRunStore";

export function ChapterRail() {
  const route = useRunStore((s) => s.route);
  const reached = useRunStore((s) => s.reached);
  const [open, setOpen] = useState(false);

  const activeIdx = CHAPTERS.findIndex((c) => route.startsWith(c.route) && c.route !== "#/");
  const active = activeIdx >= 0 ? activeIdx : 0;

  const go = (i: number) => {
    const c = CHAPTERS[i];
    if (c) window.location.hash = c.route;
    setOpen(false);
  };

  useEffect(() => {
    const onKey = (e: KeyboardEvent) => {
      if (e.target instanceof HTMLInputElement || e.target instanceof HTMLTextAreaElement) return;
      if (e.key === "ArrowRight" || e.key === "ArrowDown") {
        e.preventDefault();
        go(Math.min(CHAPTERS.length - 1, active + 1));
      } else if (e.key === "ArrowLeft" || e.key === "ArrowUp") {
        e.preventDefault();
        go(Math.max(0, active - 1));
      }
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
          {CHAPTERS.map((c, i) => (
            <span key={c.id} className={`rail__tick ${i === active ? "is-active" : ""} ${i <= reached ? "is-seen" : ""}`} />
          ))}
        </span>
      </button>

      <ol className="rail__list">
        {CHAPTERS.map((c, i) => (
          <li key={c.id}>
            <a
              href={c.route}
              className={`rail__item ${i === active ? "is-active" : ""} ${i <= reached ? "is-seen" : ""}`}
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
