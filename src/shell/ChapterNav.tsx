import { CHAPTERS } from "../state/useRunStore";

export function ChapterNav({ index }: { index: number }) {
  const i = CHAPTERS.findIndex((c) => c.index === index);
  if (i < 0) return null;
  const prev = i > 0 ? CHAPTERS[i - 1] : null;
  const next = i + 1 < CHAPTERS.length ? CHAPTERS[i + 1] : null;
  if (!prev && !next) return null;

  return (
    <nav className="chapter__nav" aria-label="Previous and next chapter" data-reveal>
      {prev && (
        <a className="chapter__nav-link" href={prev.route} rel="prev">
          <span className="chapter__nav-dir">← Previous:</span>
          <span className="chapter__nav-title">{prev.title}</span>
        </a>
      )}
      {next && (
        <a className="chapter__nav-link chapter__nav-link--next" href={next.route} rel="next">
          <span className="chapter__nav-dir">Next:</span>
          <span className="chapter__nav-title">{next.title}</span>
          <span className="chapter__nav-arrow" aria-hidden="true">
            →
          </span>
        </a>
      )}
    </nav>
  );
}
