import { useEffect, useRef, type ReactNode } from "react";
import { useReveal } from "../motion/reveal";
import { useRunStore } from "../state/useRunStore";

/**
 * One chapter of Act I: a pinned stage (the model) on the left, narration
 * scrolling in a placard column on the right. The model never leaves screen.
 */
export function Section({
  id,
  index,
  kicker,
  title,
  children,
  aside,
  stage,
  wide,
}: {
  id: string;
  index: number;
  kicker: string;
  title: string;
  children: ReactNode;
  aside?: ReactNode;
  stage?: ReactNode;
  wide?: boolean;
}) {
  const ref = useRef<HTMLElement>(null);
  useReveal(ref);
  const reach = useRunStore((s) => s.reach);

  useEffect(() => {
    const el = ref.current;
    if (!el) return;
    const io = new IntersectionObserver(
      (entries) => {
        if (entries.some((e) => e.isIntersecting)) reach(index);
      },
      { threshold: 0.15 }
    );
    io.observe(el);
    return () => io.disconnect();
  }, [index, reach]);

  return (
    <section
      id={id}
      ref={ref}
      className={`chapter ${wide ? "chapter--wide" : ""}`}
      data-chapter={id}
      aria-labelledby={`${id}-title`}
    >
      <div className="chapter__grid">
        {stage && (
          <div className="chapter__stage" data-reveal>
            <div className="stage-inner">{stage}</div>
          </div>
        )}
        <div className="chapter__placard">
          <p className="chapter__kicker" data-reveal>
            <span className="num">{String(index).padStart(2, "0")}</span>
            <span className="chapter__kicker-rule" aria-hidden="true" />
            {kicker}
          </p>
          <h2 id={`${id}-title`} className="chapter__title" data-reveal>
            {title}
          </h2>
          <div className="chapter__body" data-reveal>
            {children}
          </div>
          {aside && (
            <div className="chapter__aside" data-reveal>
              {aside}
            </div>
          )}
        </div>
      </div>
    </section>
  );
}
