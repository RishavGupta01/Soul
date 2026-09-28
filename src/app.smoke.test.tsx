import { describe, it, expect } from "vitest";
import { renderToString } from "react-dom/server";
import { App } from "./App";
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

/**
 * Server-render smoke test: catches module-level crashes, undefined refs and
 * bad JSX logic in every chapter before the app ever reaches a browser.
 * Effects (canvas, observers, timers) are intentionally not exercised here.
 */
const cases: [string, () => JSX.Element, string][] = [
  ["App", () => <App />, "OBSIDIAN"],
  ["Overture", Overture, "THINK"],
  ["Tokenization", Tokenization, "Text becomes numbers"],
  ["Embeddings", Embeddings, "A word becomes a direction"],
  ["Positional", Positional, "Sequence gets a shape"],
  ["Block", Block, "Stacked, repeated, identical"],
  ["Attention", Attention, "Everyone asks everyone"],
  ["Logits", Logits, "Nothing is decided yet"],
  ["Sampling", Sampling, "A fair coin, weighted"],
  ["Loop", Loop, "One token, then again"],
  ["Atlas", Atlas, "Everything else the model needs"],
];

describe("server render smoke", () => {
  for (const [name, Comp, needle] of cases) {
    it(`${name} renders`, () => {
      const html = renderToString(<Comp />);
      expect(html.length).toBeGreaterThan(200);
      expect(html).toContain(needle);
    });
  }

  it("every Act I chapter carries an accessible heading", () => {
    for (const [name, Comp] of cases) {
      if (name === "App" || name === "Atlas") continue;
      const html = renderToString(<Comp />);
      const tag = name === "Overture" ? "h1" : "h2";
      expect(html, name).toMatch(new RegExp(`<${tag}[^>]*id="[^"]*-title"|<${tag} class="overture__title"`));
    }
  });
});
