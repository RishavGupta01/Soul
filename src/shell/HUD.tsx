import { useDisclosureStore } from "../state/useDisclosureStore";
import { useModelStore } from "../state/useModelStore";
import { useRunStore } from "../state/useRunStore";

const LABELS: Record<string, string> = {
  loading: "loading weights",
  ready: "ready",
  fallback: "offline fallback",
  error: "weights unavailable",
};

export function HUD() {
  const depth = useDisclosureStore((s) => s.depth);
  const toggle = useDisclosureStore((s) => s.toggle);
  const status = useModelStore((s) => s.status);
  const setFastForward = useRunStore((s) => s.setFastForward);
  const fastForward = useRunStore((s) => s.fastForward);
  const atlasUnlocked = useRunStore((s) => s.atlasUnlocked);

  return (
    <header className="hud" role="banner">
      <a className="hud__mark" href="#/" aria-label="Obsidian Brain — home">
        <span className="hud__glyph" aria-hidden="true" />
        <span className="hud__word">OBSIDIAN&nbsp;BRAIN</span>
      </a>

      <nav className="hud__nav" aria-label="Site">
        <a href="#/act1" className="hud__link">
          Act&nbsp;I&nbsp;·&nbsp;Inference
        </a>
        <a href="#/atlas" className="hud__link" aria-disabled={!atlasUnlocked}>
          Act&nbsp;II&nbsp;·&nbsp;Atlas
        </a>
      </nav>

      <div className="hud__tools">
        <button
          type="button"
          className="orb"
          onClick={toggle}
          aria-pressed={depth === "deep"}
          title="Toggle explanation depth"
        >
          <span className="orb__label">DEPTH</span>
          <span className="orb__val">{depth === "deep" ? "DEEP" : "INTUITIVE"}</span>
        </button>

        <button
          type="button"
          className="orb"
          onClick={() => setFastForward(!fastForward)}
          aria-pressed={fastForward}
          title="Skip to the atlas"
        >
          <span className="orb__label">JUMP</span>
          <span className="orb__val">{fastForward ? "ON" : "OFF"}</span>
        </button>

        <span className={`hud__status hud__status--${status}`} title={LABELS[status]}>
          <span className="hud__dot" aria-hidden="true" />
          {LABELS[status]}
        </span>
      </div>
    </header>
  );
}
