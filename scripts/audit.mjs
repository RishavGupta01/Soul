/**
 * Layout audit: runs a battery of geometry checks on every route at desktop
 * and mobile widths, and prints only the violations. Screenshots are useful
 * for taste; this catches the things eyes miss (clipping, overflow, missing
 * slider fills, font fallbacks).
 *
 *   node scripts/audit.mjs [width] [height]
 */
import { run } from "./cdp.mjs";

const ROUTES = [
  "#/",
  "#/tokens",
  "#/embeddings",
  "#/position",
  "#/block",
  "#/block/attention",
  "#/logits",
  "#/sampling",
  "#/loop",
  "#/atlas",
];

const CHECK = `JSON.stringify((function(){
  var vw = innerWidth;
  var out = { hash: location.hash, issues: [] };

  var dx = document.documentElement.scrollWidth - vw;
  if (dx > 1) out.issues.push("doc-overflow-x +" + dx);

  var st = document.querySelector(".chapter__stage");
  if (st) {
    var sy = st.scrollHeight - st.clientHeight;
    var sx = st.scrollWidth - st.clientWidth;
    if (sy > 2 || sx > 2) out.issues.push("stage-clip dy=" + sy + " dx=" + sx);
    // Stage fill: how much of the stage box does the scene actually occupy?
    var sb = st.querySelector(".stage-body");
    if (sb) {
      var sr = sb.getBoundingClientRect();
      var kids = Array.prototype.filter.call(sb.children, function (c) {
        var r = c.getBoundingClientRect();
        return r.width > 4 && r.height > 4;
      });
      if (kids.length && sr.height > 40) {
        var top = Infinity, bot = -Infinity, left = Infinity, right = -Infinity;
        kids.forEach(function (c) {
          var r = c.getBoundingClientRect();
          top = Math.min(top, r.top); bot = Math.max(bot, r.bottom);
          left = Math.min(left, r.left); right = Math.max(right, r.right);
        });
        var fillY = Math.round(((bot - top) / sr.height) * 100);
        var fillX = Math.round(((right - left) / sr.width) * 100);
        out.fill = [fillX, fillY];
        if (fillY < 78) out.issues.push("stage-empty-y " + fillY + "%");
        if (fillX < 72) out.issues.push("stage-empty-x " + fillX + "%");
      }
    }
  }

  var p = document.querySelector(".chapter__placard");
  var rail = document.querySelector(".rail__toggle");
  if (p) {
    var pb = p.getBoundingClientRect();
    if (pb.right > vw - 4) out.issues.push("placard-right " + Math.round(pb.right));
    if (rail) {
      var rb = rail.getBoundingClientRect();
      if (rb.left < pb.right) out.issues.push("rail-overlaps-prose by " + Math.round(pb.right - rb.left));
    }
  }

  var ranges = document.querySelectorAll("input[type=range]");
  var noFill = 0;
  ranges.forEach(function (i) { if (!i.style.getPropertyValue("--fill")) noFill++; });
  if (noFill) out.issues.push("range-fill-missing " + noFill + "/" + ranges.length);

  out.fonts = [
    (document.fonts.check('700 40px Syne') || document.fonts.check('800 40px Syne')) ? "syne" : "!syne",
    document.fonts.check('400 16px "Schibsted Grotesk"') ? "grotesk" : "!grotesk",
    document.fonts.check('400 13px "JetBrains Mono"') ? "mono" : "!mono"
  ];
  out.fonts.forEach(function (f) { if (f.charAt(0) === "!") out.issues.push("font-fallback " + f.slice(1)); });

  var bad = [];
  document.querySelectorAll(".chapter *, .hud *, .rail *").forEach(function (e) {
    var b = e.getBoundingClientRect();
    if (b.width < 1 || b.height < 1) return;
    if (getComputedStyle(e).position === "fixed") return;
    if (b.right > vw + 2 || b.left < -2) {
      // inside a horizontally scrollable ancestor the overflow is reachable
      var a = e.parentElement, scrolls = false;
      while (a) { var ox = getComputedStyle(a).overflowX; if (ox === "auto" || ox === "scroll") { scrolls = true; break; } a = a.parentElement; }
      if (!scrolls) bad.push(String(e.className || e.tagName).slice(0, 26) + "@" + Math.round(b.right));
    }
  });
  if (bad.length) out.issues.push("elem-overflow " + bad.slice(0, 5).join(", "));

  var small = [];
  document.querySelectorAll("button, a, input[type=range]").forEach(function (e) {
    var b = e.getBoundingClientRect();
    if (b.height > 0 && b.height < 24 && !e.closest(".term")) small.push(String(e.className || e.tagName).slice(0, 26) + " h=" + Math.round(b.height));
  });
  if (small.length) out.issues.push("small-targets " + small.slice(0, 4).join(", "));

  // DESIGN.md: no forbidden typeface may be the *primary* face anywhere.
  var FORBID = ["Inter", "Roboto", "Arial", "Helvetica", "Space Grotesk", "system-ui", "sans-serif", "-apple-system"];
  var badFont = {};
  document.querySelectorAll("h1,h2,h3,h4,p,li,span,button,a,em,strong,code,div").forEach(function (e) {
    var b = e.getBoundingClientRect();
    if (b.width < 1 || b.height < 1) return;
    if (e.closest("svg")) return;
    var first = getComputedStyle(e).fontFamily.split(",")[0].trim().replace(/^["']|["']$/g, "");
    if (FORBID.indexOf(first) >= 0) badFont[first] = (badFont[first] || 0) + 1;
  });
  var badKeys = Object.keys(badFont);
  if (badKeys.length) out.issues.push("forbidden-font " + badKeys.map(function (k) { return k + "x" + badFont[k]; }).join(","));

  // DESIGN.md: every live numeric readout must use tabular figures.
  var noTab = [];
  document.querySelectorAll(".num, .orb__val, .readout__v, .knob__val").forEach(function (e) {
    var b = e.getBoundingClientRect();
    if (b.width < 1 || b.height < 1) return;
    var cs = getComputedStyle(e);
    if ((cs.fontVariantNumeric || "").indexOf("tabular-nums") < 0 && !/tnum/.test(cs.fontFeatureSettings || "")) {
      noTab.push(String(e.className || e.tagName).slice(0, 24));
    }
  });
  if (noTab.length) out.issues.push("no-tabular-nums " + noTab.slice(0, 4).join(", "));

  return out;
})())`;

const width = Number(process.argv[2]) || 1440;
const height = Number(process.argv[3]) || 900;
const baseUrl = process.argv[4] || "http://localhost:5199/";

const r = await run(baseUrl, {
  width,
  height,
  mobile: width < 700,
  wait: 3200,
  shots: ROUTES.map((hash, i) => ({
    hash,
    path: `C:\\Users\\RISHAV~1\\AppData\\Local\\Temp\\opencode\\audit-${i}.png`,
    pause: 1600,
    eval: CHECK,
  })),
});

let failures = 0;
ROUTES.forEach((hash, i) => {
  const payload = r.evalResults?.[i]?.value;
  let parsed = null;
  try {
    parsed = JSON.parse(payload);
  } catch {
    /* fall through */
  }
  const issues = parsed?.issues ?? ["NO-AUDIT-RESULT"];
  const ok = issues.length === 0;
  if (!ok) failures++;
  const fill = parsed?.fill ? ` fill ${parsed.fill[0]}%×${parsed.fill[1]}%` : "";
  console.log(`${ok ? "PASS" : "FAIL"}  ${hash.padEnd(20)} ${issues.join(" | ")}${fill}`);
});

if (r.errors?.length) {
  failures++;
  console.log("\nRUNTIME ERRORS:\n  " + r.errors.join("\n  "));
}

console.log(failures ? `\n${failures} route(s) with issues` : "\nall routes clean");
