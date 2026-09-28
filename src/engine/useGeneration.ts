import { useCallback, useEffect, useRef, useState } from "react";
import { generate } from "./generate";
import { getRuntime } from "./loader";
import { useModelStore, type GeneratedStep } from "../state/useModelStore";

/**
 * Drives a real autoregressive pass at watchable speed.
 * Pacing is decoupled from compute: we do a genuine forward pass, then hold
 * the frame so you can see the token land.
 */
export function useGeneration() {
  const status = useModelStore((s) => s.status);
  const running = useModelStore((s) => s.running);
  const patch = useModelStore((s) => s.patch);
  const setPrompt = useModelStore((s) => s.setPrompt);
  const temperature = useModelStore((s) => s.temperature);
  const topK = useModelStore((s) => s.topK);
  const topP = useModelStore((s) => s.topP);
  const seed = useModelStore((s) => s.seed);
  const prompt = useModelStore((s) => s.prompt);
  const steps = useModelStore((s) => s.steps);

  const cancelRef = useRef(false);
  const timerRef = useRef<number | null>(null);

  useEffect(
    () => () => {
      cancelRef.current = true;
      if (timerRef.current) window.clearTimeout(timerRef.current);
    },
    []
  );

  const stop = useCallback(() => {
    cancelRef.current = true;
    if (timerRef.current) window.clearTimeout(timerRef.current);
    patch({ running: false });
  }, [patch]);

  const run = useCallback(
    (opts?: {
      prompt?: string;
      maxTokens?: number;
      gap?: number;
      onStep?: (s: GeneratedStep, t: string) => void;
    }) => {
      const rt = getRuntime();
      if (!rt || status === "loading") return;

      cancelRef.current = false;
      if (timerRef.current) window.clearTimeout(timerRef.current);
      patch({ running: true, steps: [], output: "" });

      const p = opts?.prompt ?? prompt;
      const gap = opts?.gap ?? 140;
      const maxTokens = opts?.maxTokens ?? 96;

      const it = generate(rt.model, rt.tokenizer, {
        prompt: p,
        maxNewTokens: maxTokens,
        temperature,
        topK,
        topP,
        seed,
      });

      const acc: GeneratedStep[] = [];

      const tick = () => {
        if (cancelRef.current) return;
        const r = it.next();
        if (r.done) {
          patch({ running: false });
          return;
        }
        const step = r.value;
        acc.push(step);
        patch({ steps: [...acc], output: acc.map((s) => s.tokenText).join("") });
        opts?.onStep?.(step, acc.map((s) => s.tokenText).join(""));
        timerRef.current = window.setTimeout(tick, gap);
      };
      timerRef.current = window.setTimeout(tick, 60);
    },
    [prompt, temperature, topK, topP, seed, status, patch]
  );

  return {
    steps,
    running,
    run,
    stop,
    setPrompt,
    status,
  };
}
