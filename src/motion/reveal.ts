import { useEffect, type RefObject } from "react";
import gsap from "gsap";
import { ScrollTrigger } from "gsap/ScrollTrigger";
import { prefersReduced } from "./useReducedMotion";
import { useRunStore } from "../state/useRunStore";

gsap.registerPlugin(ScrollTrigger);

/**
 * Reveal a subtree on scroll: opacity 0->1 + 16px rise, staggered children.
 * Static final frame under prefers-reduced-motion.
 */
export function useReveal(ref: RefObject<HTMLElement | null>, options?: { stagger?: number }) {
  useEffect(() => {
    const el = ref.current;
    if (!el) return;
    const items = el.querySelectorAll<HTMLElement>("[data-reveal]");
    if (!items.length) return;

    const settle = () => {
      items.forEach((n) => {
        n.style.opacity = "1";
        n.style.transform = "none";
      });
    };

    if (prefersReduced() || useRunStore.getState().fastForward) {
      settle();
      return;
    }

    const rect = el.getBoundingClientRect();
    const vh = window.innerHeight || 0;
    const inView = rect.bottom > 0 && rect.top < vh * 0.78;
    const stagger = options?.stagger ?? 0.07;

    const ctx = gsap.context(() => {
      gsap.set(items, { opacity: 0, y: 16 });
      const entrance = { opacity: 1, y: 0, duration: 0.6, ease: "expo.out", stagger };
      if (inView) {
        gsap.to(items, entrance);
      } else {
        gsap.to(items, {
          ...entrance,
          scrollTrigger: { trigger: el, start: "top 78%", once: true },
        });
      }
    }, el);
    return () => ctx.revert();
  }, [ref, options?.stagger]);
}

export { gsap, ScrollTrigger };
