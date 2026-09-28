import { useDisclosureStore } from "../state/useDisclosureStore";
import { useLayoutEffect, useRef, useState, type ReactNode } from "react";

/**
 * Progressive disclosure. Closed: the term alone (its intuition line rides the
 * native tooltip). Open: an instrument panel with the intuition headline and
 * the deep explanation, so prose stays clean at default depth.
 */
export function Term({
  k,
  intuition,
  deep,
  children,
}: {
  k: string;
  intuition: string;
  deep: ReactNode;
  children: string;
}) {
  const global = useDisclosureStore((s) => s.depth);
  const override = useDisclosureStore((s) => s.terms[k]);
  const setTerm = useDisclosureStore((s) => s.setTerm);
  const isOpen = override ? override === "deep" : global === "deep";

  const trigRef = useRef<HTMLButtonElement>(null);
  const panelRef = useRef<HTMLSpanElement>(null);
  const [flip, setFlip] = useState(false);

  // Keep the panel inside the viewport: open leftward when the trigger sits
  // close to the right edge.
  useLayoutEffect(() => {
    if (!isOpen) return;
    const measure = () => {
      const trig = trigRef.current;
      const panel = panelRef.current;
      if (!trig || !panel) return;
      const r = trig.getBoundingClientRect();
      setFlip(r.left + panel.offsetWidth > window.innerWidth - 16);
    };
    measure();
    window.addEventListener("resize", measure);
    return () => window.removeEventListener("resize", measure);
  }, [isOpen]);

  return (
    <span className={`term ${isOpen ? "is-open" : ""}`}>
      <button
        ref={trigRef}
        type="button"
        className="term__trigger"
        aria-expanded={isOpen}
        aria-controls={`term-${k}`}
        title={intuition}
        onClick={() => setTerm(k, isOpen ? "intuition" : "deep")}
      >
        {children}
        <span className="term__mark" aria-hidden="true">
          {isOpen ? "–" : "+"}
        </span>
      </button>
      {isOpen && (
        <span
          ref={panelRef}
          className={`term__panel ${flip ? "term__panel--flip" : ""}`}
          id={`term-${k}`}
        >
          <span className="term__i">{intuition}</span>
          <span className="term__d">{deep}</span>
        </span>
      )}
    </span>
  );
}
