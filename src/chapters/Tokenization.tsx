import { useEffect, useMemo, useRef, useState } from "react";
import { Section } from "../shell/Section";
import { Term } from "../shell/Term";
import { Stage } from "../viz/Stage";
import { useReducedMotion } from "../motion/useReducedMotion";
import { useRuntime } from "../engine/useRuntime";

const TEXT = "attention is all you need";

const AUTOPLAY_MS = 1000;

export function Tokenization() {
  const rt = useRuntime();
  const reduced = useReducedMotion();
  const [phase, setPhase] = useState(0); // 0 = bytes, 1..n = after n merges applied
  const [playing, setPlaying] = useState(true);
  const timer = useRef<number | null>(null);

  const trace = useMemo(() => {
    if (!rt) return null;
    return rt.tokenizer.encodeWithSteps(TEXT);
  }, [rt]);

  const totalMerges = trace?.steps.length ?? 0;

  // Build the symbol sequence at each merge level.
  const frames = useMemo(() => {
    if (!rt) return [];
    const bytes = Array.from(new TextEncoder().encode(TEXT));
    const mergeRank = new Map<string, number>();
    rt.tokenizer.merges.forEach((m, rank) => mergeRank.set(m.pair[0] + "," + m.pair[1], rank));

    const seqs: { ids: number[]; applied: { rank: number; pair: [number, number] }[] }[] = [
      { ids: bytes.slice(), applied: [] },
    ];
    // Apply merges in rank order, snapshotting each time the chosen merge
    // actually fires in this string.
    for (let rank = 0; rank < rt.tokenizer.merges.length; rank++) {
      const prev = seqs[seqs.length - 1];
      const next: number[] = [];
      const applied = [...prev.applied];
      let fired = false;
      const merge = rt.tokenizer.merges[rank];
      for (let i = 0; i < prev.ids.length; i++) {
        if (i < prev.ids.length - 1 && prev.ids[i] === merge.pair[0] && prev.ids[i + 1] === merge.pair[1]) {
          next.push(256 + rank);
          i++;
          fired = true;
          applied.push({ rank, pair: merge.pair });
        } else {
          next.push(prev.ids[i]);
        }
      }
      if (!fired) continue;
      seqs.push({ ids: next, applied });
    }
    return seqs;
  }, [rt]);

  const maxFrame = frames.length - 1;

  useEffect(() => {
    if (!playing || reduced) return;
    timer.current = window.setInterval(() => {
      setPhase((p) => (p >= maxFrame ? 0 : p + 1));
    }, AUTOPLAY_MS);
    return () => {
      if (timer.current) window.clearInterval(timer.current);
    };
  }, [playing, maxFrame, reduced]);

  const frame = frames[Math.min(phase, maxFrame)];
  const ids = frame?.ids ?? [];
  const symbolCount = ids.length;
  const label = phase === 0 ? "raw bytes" : `after ${phase} merge${phase === 1 ? "" : "s"}`;

  const describe = (id: number): string => {
    if (id < 256) {
      const ch = String.fromCharCode(id);
      return ch === " " ? "␣" : id === 10 ? "⏎" : ch;
    }
    return rt ? rt.tokenizer.decode([id]) : "?";
  };

  return (
    <Section
      id="tokens"
      index={1}
      kicker="the first transformation"
      title="Text becomes numbers"
      stage={
        <Stage
          label="BPE"
          labelValue={label}
          aria={`Byte pair encoding of "${TEXT}". ${phase} of ${maxFrame} merges applied, currently ${symbolCount} symbols.`}
          foot={
            <>
              <span className="num">
                {new TextEncoder().encode(TEXT).length} bytes → {symbolCount} symbols
              </span>
              <span className="num">
                rank {frame?.applied.length ? frame.applied[frame.applied.length - 1].rank : "—"} of{" "}
                {rt?.tokenizer.merges.length}
              </span>
            </>
          }
        >
          <div className="bpe">
            <div className="bpe__row">
              {ids.map((id, i) => (
                <span
                  key={`${id}-${i}`}
                  className={`bpe__sym ${id >= 256 ? "bpe__sym--merged" : ""}`}
                  style={{ animationDelay: `${Math.min(i * 14, 400)}ms` }}
                >
                  <span className="bpe__text">{describe(id)}</span>
                  <span className="bpe__id num">{id}</span>
                </span>
              ))}
            </div>

            <div className="bpe__meter">
              <div className="bpe__meterTrack">
                <span className="bpe__meterFill" style={{ width: `${(phase / Math.max(1, maxFrame)) * 100}%` }} />
              </div>
              <div className="bpe__controls">
                <button type="button" className="btn btn--ghost" onClick={() => setPhase(0)}>
                  reset
                </button>
                <button type="button" className="btn btn--ghost" onClick={() => setPlaying(!playing)}>
                  {playing ? "pause" : "play"}
                </button>
                <button type="button" className="btn btn--ghost" onClick={() => setPhase((p) => Math.min(maxFrame, p + 1))}>
                  step ▸
                </button>
              </div>
            </div>

            <div className="bpe__log">
              <div className="bpe__logHead">MERGE LOG</div>
              {frame?.applied.slice(-11).map((m) => (
                <div className="bpe__logRow mono" key={m.rank}>
                  <span className="bpe__logRank">{String(m.rank).padStart(3, "0")}</span>
                  <span>
                    {describe(m.pair[0])} + {describe(m.pair[1])}
                  </span>
                  <span className="bpe__logArrow">→</span>
                  <span className="bpe__logNew">{rt ? rt.tokenizer.decode([256 + m.rank]) : ""}</span>
                  <span className="bpe__logId">id {256 + m.rank}</span>
                </div>
              ))}
            </div>
          </div>
        </Stage>
      }
      aside={
        <div className="readout">
          <div className="readout__row">
            <span className="readout__k">algorithm</span>
            <span className="readout__v readout__v--muted">byte-pair encoding</span>
          </div>
          <div className="readout__row">
            <span className="readout__k">base symbols</span>
            <span className="readout__v">256 bytes</span>
          </div>
          <div className="readout__row">
            <span className="readout__k">merges learned</span>
            <span className="readout__v">{rt?.tokenizer.merges.length ?? "—"}</span>
          </div>
          <div className="readout__row">
            <span className="readout__k">final vocab</span>
            <span className="readout__v">{rt?.tokenizer.vocabSize ?? "—"}</span>
          </div>
        </div>
      }
    >
      <p>
        The model has no idea what letters are. It reads integers. So the first thing that
        happens to your sentence is a compression step called{" "}
        <Term k="bpe" intuition="chop the text into bytes, then repeatedly glue the most common neighbouring pair together." deep={
          <>
            Byte-pair encoding starts from all 256 byte values. It counts every adjacent pair, merges the most frequent one into a
            new symbol, and repeats — here, 128 times. Each merge produces a token id of <span className="num">256 + rank</span>,
            which is exactly the vocabulary this model was trained on.
          </>
        }>
          BPE tokenization
        </Term>
        .
      </p>
      <p>
        Watch the merge log: every line is a real rule this tokenizer learned from the training corpus.{" "}
        <strong>The same string always splits the same way</strong> — that determinism is what lets training and inference agree.
      </p>
      <p>
        Longer merges win because they were more frequent. By the end, most words are a single symbol, and the sequence has
        shrunk from dozens of bytes to a handful of tokens — fewer positions for attention to walk.
      </p>
    </Section>
  );
}
