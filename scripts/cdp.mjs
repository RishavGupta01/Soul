/**
 * Minimal Chrome DevTools Protocol driver — used to smoke-test and screenshot
 * the site in a real browser without pulling in puppeteer.
 *
 * Usage:
 *   node scripts/cdp.mjs open  <url> <shot.png> [width] [height]
 *   node scripts/cdp.mjs seq   <url> <outPrefix> <jsonStepsFile>
 */
import { spawn } from "node:child_process";
import { writeFileSync, existsSync, mkdirSync, rmSync } from "node:fs";
import { tmpdir } from "node:os";
import { join } from "node:path";

const CHROME =
  process.env.CHROME_PATH || "C:\\Program Files\\Google\\Chrome\\Application\\chrome.exe";
const PORT = 9333;
const PROFILE = join(tmpdir(), `cdp-profile-${process.pid}`);

const sleep = (ms) => new Promise((r) => setTimeout(r, ms));

async function launch() {
  if (!existsSync(CHROME)) throw new Error("chrome not found at " + CHROME);
  const child = spawn(
    CHROME,
    [
      `--remote-debugging-port=${PORT}`,
      `--user-data-dir=${PROFILE}`,
      "--headless=new",
      "--no-first-run",
      "--no-default-browser-check",
      "--disable-gpu",
      "--hide-scrollbars",
      "--disable-extensions",
      "about:blank",
    ],
    { stdio: "ignore", detached: false }
  );
  for (let i = 0; i < 80; i++) {
    try {
      const r = await fetch(`http://127.0.0.1:${PORT}/json/version`);
      if (r.ok) return child;
    } catch {
      /* keep polling */
    }
    await sleep(150);
  }
  child.kill();
  throw new Error("chrome did not expose a devtools endpoint");
}

async function newTarget(url) {
  let res;
  try {
    res = await fetch(`http://127.0.0.1:${PORT}/json/new?${encodeURIComponent(url)}`, {
      method: "PUT",
    });
  } catch {
    res = null;
  }
  if (!res || !res.ok) {
    res = await fetch(`http://127.0.0.1:${PORT}/json/new?${encodeURIComponent(url)}`);
  }
  if (!res.ok) throw new Error(`cannot create target: ${res.status}`);
  return res.json();
}

class Session {
  constructor(ws) {
    this.ws = ws;
    this.id = 0;
    this.pending = new Map();
    this.listeners = [];
    this.events = [];
    ws.addEventListener("message", (ev) => {
      const msg = JSON.parse(ev.data);
      if (msg.id && this.pending.has(msg.id)) {
        const { resolve, reject } = this.pending.get(msg.id);
        this.pending.delete(msg.id);
        if (msg.error) reject(new Error(`${msg.error.message}`));
        else resolve(msg.result);
      } else if (msg.method) {
        this.events.push(msg);
        for (const fn of this.listeners) fn(msg);
      }
    });
  }
  static connect(wsUrl) {
    return new Promise((resolve, reject) => {
      const ws = new WebSocket(wsUrl);
      ws.addEventListener("open", () => resolve(new Session(ws)), { once: true });
      ws.addEventListener("error", reject, { once: true });
    });
  }
  send(method, params = {}) {
    const id = ++this.id;
    return new Promise((resolve, reject) => {
      this.pending.set(id, { resolve, reject });
      this.ws.send(JSON.stringify({ id, method, params }));
      setTimeout(() => {
        if (this.pending.has(id)) {
          this.pending.delete(id);
          reject(new Error(`timeout: ${method}`));
        }
      }, 30000);
    });
  }
  on(fn) {
    this.listeners.push(fn);
  }
  close() {
    try {
      this.ws.close();
    } catch {
      /* ignore */
    }
  }
}

export async function run(url, actions) {
  const proc = await launch();
  const target = await newTarget(url);
  const session = await Session.connect(target.webSocketDebuggerUrl);
  const errors = [];

  session.on((msg) => {
    if (msg.method === "Runtime.exceptionThrown") {
      const d = msg.params.exceptionDetails;
      errors.push(`EXCEPTION ${d.text} ${d.exception?.description ?? ""}`);
    }
    if (msg.method === "Runtime.consoleAPICalled" && msg.params.type === "error") {
      errors.push(
        "CONSOLE " + msg.params.args.map((a) => a.value ?? a.description ?? "").join(" ")
      );
    }
    if (msg.method === "Log.entryAdded" && msg.params.entry.level === "error") {
      errors.push("LOG " + msg.params.entry.text);
    }
    if (msg.method === "Network.loadingFailed" && !msg.params.canceled) {
      errors.push(`NET ${msg.params.errorText} ${msg.params.type}`);
    }
  });

  await session.send("Page.enable");
  await session.send("Runtime.enable");
  await session.send("Log.enable");
  await session.send("Network.enable");

  const width = actions.width ?? 1440;
  const height = actions.height ?? 900;
  await session.send("Emulation.setDeviceMetricsOverride", {
    width,
    height,
    deviceScaleFactor: actions.scale ?? 1,
    mobile: !!actions.mobile,
  });

  await session.send("Page.navigate", { url });
  await sleep(actions.wait ?? 2500);

  const out = {};
  if (actions.eval) {
    const r = await session.send("Runtime.evaluate", {
      expression: actions.eval,
      awaitPromise: true,
      returnByValue: true,
    });
    out.value = r.result?.value;
    if (r.exceptionDetails) errors.push("EVAL " + JSON.stringify(r.exceptionDetails));
  }

  if (actions.shots) {
    mkdirSync(join(tmpdir(), "cdp"), { recursive: true });
    for (const s of actions.shots) {
      if (s.scrollTo !== undefined) {
        await session.send("Runtime.evaluate", {
          expression: `window.scrollTo(0, ${s.scrollTo})`,
        });
        await sleep(s.pause ?? 700);
      }
      if (s.hash) {
        await session.send("Runtime.evaluate", {
          expression: `location.hash = ${JSON.stringify(s.hash)}`,
        });
        await sleep(s.pause ?? 1400);
      }
      if (s.eval) {
        const er = await session.send("Runtime.evaluate", {
          expression: s.eval,
          awaitPromise: true,
          returnByValue: true,
        });
        (out.evalResults ??= []).push({ hash: s.hash ?? null, value: er.result?.value });
        if (er.exceptionDetails)
          errors.push("EVAL " + JSON.stringify(er.exceptionDetails));
        await sleep(s.pause ?? 900);
      }
      const shot = await session.send("Page.captureScreenshot", { format: "png" });
      writeFileSync(s.path, Buffer.from(shot.data, "base64"));
      out.lastShot = s.path;
    }
  }

  session.close();
  proc.kill();
  try {
    rmSync(PROFILE, { recursive: true, force: true });
  } catch {
    /* ignore */
  }
  return { ...out, errors };
}

// CLI
import { pathToFileURL } from "node:url";
if (import.meta.url === pathToFileURL(process.argv[1]).href) {
  const [, , cmd, ...rest] = process.argv;
  try {
    if (cmd === "open") {
      const [url, shot, w, h] = rest;
      const r = await run(url, {
        width: Number(w) || 1440,
        height: Number(h) || 900,
        shots: [{ path: shot }],
        eval: "document.title",
      });
      console.log(JSON.stringify(r, null, 2));
    } else if (cmd === "seq") {
      const [url, , stepsFile] = rest;
      const steps = JSON.parse(
        (await import("node:fs")).readFileSync(stepsFile, "utf8")
      );
      const r = await run(url, steps);
      console.log(JSON.stringify(r, null, 2));
    } else {
      console.log("usage: cdp.mjs open|seq ...");
    }
  } catch (e) {
    console.error("FAIL", e.message);
    process.exitCode = 1;
  }
}
