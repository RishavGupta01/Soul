import { useEffect, useState } from "react";
import { loadRuntime } from "./engine/loader";
import { useModelStore } from "./state/useModelStore";
import { useRunStore } from "./state/useRunStore";
import { HUD } from "./shell/HUD";
import { ChapterRail } from "./shell/ChapterRail";
import { Overture } from "./chapters/Overture";
import { Tokenization } from "./chapters/Tokenization";
import { Embeddings } from "./chapters/Embeddings";
import { Positional } from "./chapters/Positional";
import { Block } from "./chapters/Block";
import { Attention } from "./chapters/Attention";
import { Logits } from "./chapters/Logits";
import { Sampling } from "./chapters/Sampling";
import { Loop } from "./chapters/Loop";
import { Atlas } from "./chapters/Atlas";

const ROUTES: Record<string, () => JSX.Element> = {
  "#/": Overture,
  "#/tokens": Tokenization,
  "#/embeddings": Embeddings,
  "#/position": Positional,
  "#/block": Block,
  "#/block/attention": Attention,
  "#/logits": Logits,
  "#/sampling": Sampling,
  "#/loop": Loop,
  "#/atlas": Atlas,
};

const REDIRECTS: Record<string, string> = { "#/act1": "#/tokens" };

export function App() {
  const route = useRunStore((s) => s.route);
  const setRoute = useRunStore((s) => s.setRoute);
  const status = useModelStore((s) => s.status);
  const setStatus = useModelStore((s) => s.setStatus);
  const setMetrics = useModelStore((s) => s.setMetrics);
  const [booted, setBooted] = useState(false);

  // --- hash routing ---
  useEffect(() => {
    const sync = () => {
      const h = window.location.hash || "#/";
      const target = REDIRECTS[h] ?? h;
      if (target !== h) {
        window.location.hash = target;
        return;
      }
      setRoute(target);
      window.scrollTo({ top: 0, behavior: "instant" as ScrollBehavior });
    };
    sync();
    window.addEventListener("hashchange", sync);
    return () => window.removeEventListener("hashchange", sync);
  }, [setRoute]);

  // --- model loading ---
  useEffect(() => {
    let dead = false;
    (async () => {
      try {
        const rt = await loadRuntime(import.meta.env.BASE_URL);
        if (dead) return;
        setMetrics(rt.metrics);
        useModelStore.setState({ vocabSize: rt.model.config.vocabSize });
        setStatus("ready", "real weights loaded");
      } catch (err) {
        console.warn("weights unavailable, running without a trained model", err);
        if (!dead) setStatus("error", String(err));
      } finally {
        if (!dead) setBooted(true);
      }
    })();
    return () => {
      dead = true;
    };
  }, [setStatus, setMetrics]);

  const Page = ROUTES[route] ?? Overture;

  return (
    <div className={`app app--${status}`}>
      <a className="skip" href="#main">
        Skip to content
      </a>
      <HUD />
      <ChapterRail />
      <main id="main" className="main">
        {!booted ? <Boot /> : <Page />}
      </main>
    </div>
  );
}

function Boot() {
  const detail = useModelStore((s) => s.statusDetail);
  return (
    <div className="boot" role="status" aria-live="polite">
      <div className="boot__mark" aria-hidden="true" />
      <p className="boot__line mono">
        loading weights<span className="boot__dots" />
      </p>
      <p className="boot__detail mono">{detail}</p>
    </div>
  );
}
