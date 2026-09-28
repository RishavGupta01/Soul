/**
 * Instrument-style range sliders: paint the filled portion of every track from
 * the live value. One delegated listener + a MutationObserver keeps it working
 * for React-controlled inputs without touching each call site.
 */
function sync(el: HTMLInputElement): void {
  const min = Number(el.min || 0);
  const max = Number(el.max || 100);
  const val = Number(el.value);
  const pct = max > min ? ((val - min) / (max - min)) * 100 : 0;
  el.style.setProperty("--fill", `${Math.max(0, Math.min(100, pct))}%`);
}

function isRange(n: EventTarget | null): n is HTMLInputElement {
  return n instanceof HTMLInputElement && n.type === "range";
}

export function initRangeFill(): void {
  if (typeof document === "undefined") return;

  const scan = (root: ParentNode) => {
    root.querySelectorAll<HTMLInputElement>('input[type="range"]').forEach(sync);
  };

  scan(document);
  document.addEventListener("input", (e) => {
    if (isRange(e.target)) sync(e.target);
  });
  document.addEventListener(
    "change",
    (e) => {
      if (isRange(e.target)) sync(e.target);
    },
    true
  );

  new MutationObserver((records) => {
    for (const r of records) {
      for (const node of r.addedNodes) {
        if (node instanceof HTMLInputElement && node.type === "range") sync(node);
        else if (node instanceof Element) scan(node);
      }
      if (isRange(r.target)) sync(r.target);
    }
  }).observe(document.body, { childList: true, subtree: true, attributes: true, attributeFilter: ["value", "min", "max"] });
}
