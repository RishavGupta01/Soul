import { useEffect, type RefObject } from "react";
import gsap from "gsap";
import { ScrollTrigger } from "gsap/ScrollTrigger";
import { prefersReduced } from "./useReducedMotion";

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

    if (prefersReduced()) {
      items.forEach((n) => {
        n.style.opacity = "1";
        n.style.transform = "none";
      });
      return;
    }

    const ctx = gsap.context(() => {
      gsap.set(items, { opacity: 0, y: 16 });
      gsap.to(items, {
        opacity: 1,
        y: 0,
        duration: 0.6,
        ease: "expo.out",
        stagger: options?.stagger ?? 0.07,
        scrollTrigger: { trigger: el, start: "top 78%", once: true },
      });
    }, el);
    return () => ctx.revert();
  }, [ref, options?.stagger]);
}

export { gsap, ScrollTrigger };
