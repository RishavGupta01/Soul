import { readFileSync } from "node:fs";

const h = readFileSync("dist/index.html", "utf8");
const out = {
  title: (h.match(/<title>(.*?)<\/title>/) || [])[1],
  desc: (h.match(/name="description" content="([^"]{0,100})/) || [])[1],
  scripts: [...h.matchAll(/<script[^>]+src="([^"]+)"/g)].map((m) => m[1]),
  css: [...h.matchAll(/<link[^>]+href="([^"]+\.css)"/g)].map((m) => m[1]),
  assetRefs: [...h.matchAll(/(?:src|href)="([^"]*assets[^"]*)"/g)].map((m) => m[1]),
};
console.log(JSON.stringify(out, null, 1));
