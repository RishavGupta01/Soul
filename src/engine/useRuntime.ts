import { useEffect, useState } from "react";
import { getRuntime, type Runtime } from "./loader";

/**
 * Chapters call `getRuntime()` during render, but the weights load
 * asynchronously. A component that mounts before the model is ready would
 * otherwise hold `null` for the rest of its life, because nothing triggers a
 * re-render once the loader finishes. This hook subscribes to readiness so a
 * deep-linked chapter fills in the moment weights land.
 */
export function useRuntime(): Runtime | null {
  const [rt, setRt] = useState<Runtime | null>(() => getRuntime());

  useEffect(() => {
    if (rt) return;
    const id = window.setInterval(() => {
      const r = getRuntime();
      if (r) {
        setRt(r);
        window.clearInterval(id);
      }
    }, 80);
    return () => window.clearInterval(id);
  }, [rt]);

  return rt;
}
