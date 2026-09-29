/**
 * Decode a Chrome screenshot PNG and report what is actually painted.
 * The image-read tool in this session returns stale frames, so we verify
 * visually by decoding pixels ourselves: ink coverage per region plus an
 * ASCII rendering you can read in the terminal.
 *
 *   node scripts/pixels.mjs <file.png> [--ascii [cols]] [--bands]
 */
import { readFileSync } from "node:fs";
import zlib from "node:zlib";

function decode(buf) {
  if (buf.readUInt32BE(0) !== 0x89504e47) throw new Error("not a PNG");
  let off = 8;
  let w = 0, h = 0, depth = 0, type = 0;
  const idat = [];
  while (off < buf.length) {
    const len = buf.readUInt32BE(off);
    const tag = buf.toString("ascii", off + 4, off + 8);
    const data = buf.subarray(off + 8, off + 8 + len);
    if (tag === "IHDR") {
      w = data.readUInt32BE(0); h = data.readUInt32BE(4);
      depth = data[8]; type = data[9];
      if (depth !== 8) throw new Error("bit depth " + depth);
      if (type !== 2 && type !== 6) throw new Error("color type " + type + " (want RGB/RGBA)");
    } else if (tag === "IDAT") idat.push(data);
    else if (tag === "IEND") break;
    off += 12 + len;
  }
  const bpp = type === 6 ? 4 : 3;
  const raw = zlib.inflateSync(Buffer.concat(idat));
  const stride = w * bpp;
  const out = Buffer.alloc(h * stride);
  let p = 0;
  for (let y = 0; y < h; y++) {
    const filter = raw[p++];
    const line = raw.subarray(p, p + stride); p += stride;
    const cur = out.subarray(y * stride, (y + 1) * stride);
    const prev = y > 0 ? out.subarray((y - 1) * stride, y * stride) : null;
    for (let x = 0; x < stride; x++) {
      const a = x >= bpp ? cur[x - bpp] : 0;
      const b = prev ? prev[x] : 0;
      const c = prev && x >= bpp ? prev[x - bpp] : 0;
      let v = line[x];
      switch (filter) {
        case 1: v += a; break;
        case 2: v += b; break;
        case 3: v += (a + b) >> 1; break;
        case 4: {
          const pa = Math.abs(b - c), pb = Math.abs(a - c), pc = Math.abs(a + b - 2 * c);
          v += pa <= pb && pa <= pc ? a : pb <= pc ? b : c;
          break;
        }
      }
      cur[x] = v & 255;
    }
  }
  return { w, h, bpp, data: out };
}

const VOID = [5, 5, 7];
const isBg = (r, g, b) =>
  Math.abs(r - VOID[0]) < 7 && Math.abs(g - VOID[1]) < 7 && Math.abs(b - VOID[2]) < 7;

/** fraction of non-background pixels inside a pixel box */
function ink(img, x0, y0, x1, y1) {
  x0 = Math.max(0, Math.round(x0)); y0 = Math.max(0, Math.round(y0));
  x1 = Math.min(img.w, Math.round(x1)); y1 = Math.min(img.h, Math.round(y1));
  let hit = 0, n = 0;
  for (let y = y0; y < y1; y++) {
    for (let x = x0; x < x1; x++) {
      const i = (y * img.w + x) * img.bpp;
      n++;
      if (!isBg(img.data[i], img.data[i + 1], img.data[i + 2])) hit++;
    }
  }
  return n ? Math.round((hit / n) * 1000) / 10 : 0;
}

const RAMP = " .:-=+*#%@";
function ascii(img, cols = 96, crop) {
  const cx0 = crop ? crop[0] : 0, cy0 = crop ? crop[1] : 0;
  const cx1 = crop ? crop[2] : img.w, cy1 = crop ? crop[3] : img.h;
  const cw = Math.max(1, cx1 - cx0), ch = Math.max(1, cy1 - cy0);
  const rows = Math.max(1, Math.round((cols * ch) / cw / 2.1));
  const lines = [];
  for (let ry = 0; ry < rows; ry++) {
    let line = "";
    for (let cx = 0; cx < cols; cx++) {
      const x0 = cx0 + Math.floor((cx * cw) / cols), x1 = cx0 + Math.floor(((cx + 1) * cw) / cols);
      const y0 = cy0 + Math.floor((ry * ch) / rows), y1 = cy0 + Math.floor(((ry + 1) * ch) / rows);
      let sum = 0, n = 0;
      for (let y = y0; y < y1; y++) {
        for (let x = x0; x < x1; x++) {
          const i = (y * img.w + x) * img.bpp;
          const r = img.data[i], g = img.data[i + 1], b = img.data[i + 2];
          // luminance of non-background pixels; colour counts as ink
          if (!isBg(r, g, b)) sum += 0.299 * r + 0.587 * g + 0.114 * b;
          n++;
        }
      }
      const avg = n ? sum / n : 0;
      line += RAMP[Math.min(RAMP.length - 1, Math.floor((avg / 128) * (RAMP.length - 1)))];
    }
    lines.push(line);
  }
  return lines.join("\n");
}

const args = process.argv.slice(2);
const file = args.find((a) => !a.startsWith("--"));
if (!file) {
  console.log("usage: node scripts/pixels.mjs <file.png> [--ascii [cols]] [--bands]");
  process.exit(1);
}
const img = decode(readFileSync(file));
const out = { file, size: `${img.w}x${img.h}` };

if (args.includes("--ascii")) {
  const i = args.indexOf("--ascii");
  const cols = Number(args[i + 1]) || 96;
  const ci = args.indexOf("--crop");
  const crop = ci >= 0 && args[ci + 1] ? args[ci + 1].split(",").map(Number) : null;
  console.log(ascii(img, cols, crop && crop.length === 4 && crop.every((n) => Number.isFinite(n)) ? crop : null));
} else if (args.includes("--bands")) {
  // vertical ink profile, 10 bands — tells you where the content sits
  const bands = [];
  for (let k = 0; k < 10; k++) {
    bands.push(ink(img, 0, (img.h * k) / 10, img.w, (img.h * (k + 1)) / 10));
  }
  out.bandPct = bands;
  out.total = ink(img, 0, 0, img.w, img.h);
  console.log(JSON.stringify(out, null, 1));
} else {
  out.total = ink(img, 0, 0, img.w, img.h);
  // left half = stage column, right half = prose column
  out.stage = ink(img, 0, 60, img.w * 0.58, img.h);
  out.prose = ink(img, img.w * 0.58, 60, img.w, img.h);
  console.log(JSON.stringify(out, null, 1));
}
