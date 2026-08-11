#!/usr/bin/env node
// Renders the README charts (static SVG) from results/metrics.json.
//   node src/make_charts.mjs
// Zero dependencies. Charts paint an explicit light surface so they stay
// readable on GitHub in both themes. Palette: two categorical series
// (blue #2a78d6, orange #eb6834) on #fcfcfb, CVD-validated.

import { readFileSync, writeFileSync, mkdirSync } from "node:fs";

const R = JSON.parse(
  readFileSync(new URL("../results/metrics.json", import.meta.url), "utf8")
);
mkdirSync(new URL("../charts", import.meta.url), { recursive: true });

const C = {
  surface: "#fcfcfb",
  ink: "#0b0b0b",
  ink2: "#52514e",
  muted: "#898781",
  grid: "#e1e0d9",
  axis: "#c3c2b7",
  blue: "#2a78d6",
  orange: "#eb6834",
};
const FONT = `font-family="system-ui, -apple-system, 'Segoe UI', sans-serif"`;
const esc = (s) => String(s).replace(/&/g, "&amp;").replace(/</g, "&lt;");
const t = (x, y, str, { size = 11, fill = C.muted, anchor = "start", weight = "", tab = false } = {}) =>
  `<text x="${x.toFixed(1)}" y="${y.toFixed(1)}" ${FONT} font-size="${size}" fill="${fill}" text-anchor="${anchor}"${weight ? ` font-weight="${weight}"` : ""}${tab ? ` style="font-variant-numeric: tabular-nums"` : ""}>${esc(str)}</text>`;
const line = (x1, y1, x2, y2, stroke, w = 1, dash = "", opacity = 1) =>
  `<line x1="${x1.toFixed(1)}" y1="${y1.toFixed(1)}" x2="${x2.toFixed(1)}" y2="${y2.toFixed(1)}" stroke="${stroke}" stroke-width="${w}"${dash ? ` stroke-dasharray="${dash}"` : ""}${opacity < 1 ? ` opacity="${opacity}"` : ""}/>`;
const dot = (x, y, r, fill, ring = true) =>
  `<circle cx="${x.toFixed(1)}" cy="${y.toFixed(1)}" r="${r.toFixed(1)}" fill="${fill}"${ring ? ` stroke="${C.surface}" stroke-width="2"` : ""}/>`;
const svgWrap = (w, h, body) =>
  `<svg xmlns="http://www.w3.org/2000/svg" width="${w}" height="${h}" viewBox="0 0 ${w} ${h}" role="img">
<rect width="${w}" height="${h}" fill="${C.surface}"/>
${body}
</svg>`;
const pct = (x, d = 0) => `${(x * 100).toFixed(d)}%`;
const rOf = (n) => 4 + 2.2 * Math.log10(Math.max(1, n)); // dot radius ~ log n

// ---------------------------------------------------------------- calibration
{
  const P = R.protocols.resolution_L90;
  const W = 880, H = 660;
  const M = { l: 64, r: 320, t: 64, b: 110 };
  const pw = W - M.l - M.r, ph = H - M.t - M.b;
  const X = (p) => M.l + p * pw;
  const Y = (p) => M.t + (1 - p) * ph;
  let b = [];

  b.push(t(M.l, 28, "Predicted vs. actual win rate — walk-forward, decile bins", { size: 15, fill: C.ink, weight: "600" }));
  b.push(t(M.l, 46, "Each point is one predicted-probability bin; dot area grows with bin count (log scale); whiskers are 95% Wilson intervals.", { size: 12, fill: C.ink2 }));

  for (let g = 0; g <= 100; g += 20) {
    b.push(line(X(g / 100), M.t, X(g / 100), M.t + ph, C.grid));
    b.push(line(M.l, Y(g / 100), M.l + pw, Y(g / 100), C.grid));
    b.push(t(X(g / 100), M.t + ph + 16, `${g}%`, { anchor: "middle", tab: true }));
    b.push(t(M.l - 8, Y(g / 100) + 4, `${g}%`, { anchor: "end", tab: true }));
  }
  b.push(line(M.l, M.t + ph, M.l + pw, M.t + ph, C.axis, 1));
  b.push(line(M.l, M.t, M.l, M.t + ph, C.axis, 1));
  b.push(t(M.l + pw / 2, M.t + ph + 38, "mean predicted win probability", { anchor: "middle", fill: C.ink2 }));
  b.push(`<text x="${M.l - 40}" y="${M.t + ph / 2}" ${FONT} font-size="11" fill="${C.ink2}" text-anchor="middle" transform="rotate(-90 ${M.l - 40} ${M.t + ph / 2})">actual win rate</text>`);
  b.push(line(X(0), Y(0), X(1), Y(1), C.axis, 1.5, "5 4"));
  b.push(`<text x="${X(0.76)}" y="${Y(0.76) - 8}" ${FONT} font-size="11" fill="${C.muted}" transform="rotate(-44 ${X(0.76)} ${Y(0.76) - 8})">perfect calibration</text>`);

  const m4 = P.calibration.m4_seg_size.filter((x) => x.n > 0);
  const m1a = P.calibration.m1a_product.filter((x) => x.n >= 10);

  // n-label placement, hand-set per bin to avoid collisions in the dense corner.
  // mode: aw = above whisker, bw = below whisker, or custom [dx, dy, anchor] from the dot.
  const m4Lbl = ["custom:-14,4,end", "bw", "bw", "aw", "bw", "custom:8,15,start", "aw", "bw", "aw", "aw"];
  const m1aLbl = ["custom:-12,-8,end", "aw", "aw", "custom:-8,15,end"];

  const drawSeries = (bins, color, labels) => {
    bins.forEach((x, i) => {
      b.push(line(X(x.meanPred), Y(x.ci95[0]), X(x.meanPred), Y(x.ci95[1]), color, 2, "", 0.45));
    });
    bins.forEach((x, i) => {
      b.push(dot(X(x.meanPred), Y(x.actual), rOf(x.n), color));
      const spec = labels[i] || "bw";
      const label = `n=${x.n}`;
      if (spec.startsWith("custom:")) {
        const [dx, dy, anchor] = spec.slice(7).split(",");
        b.push(t(X(x.meanPred) + +dx, Y(x.actual) + +dy, label, { size: 10, anchor }));
      } else if (spec === "aw") {
        b.push(t(X(x.meanPred), Y(Math.max(x.ci95[1], x.actual)) - 7, label, { size: 10, anchor: "middle" }));
      } else {
        b.push(t(X(x.meanPred), Y(Math.min(x.ci95[0], x.actual)) + 15, label, { size: 10, anchor: "middle" }));
      }
    });
  };

  // connecting line only for the model under test
  b.push(`<path d="${m4.map((x, i) => `${i ? "L" : "M"}${X(x.meanPred).toFixed(1)},${Y(x.actual).toFixed(1)}`).join("")}" fill="none" stroke="${C.blue}" stroke-width="2" opacity="0.5"/>`);
  drawSeries(m1a, C.orange, m1aLbl);
  drawSeries(m4, C.blue, m4Lbl);

  b.push(t(M.l, M.t + ph + 58, "Two deployed-heuristic bins with n ≤ 2 are omitted for legibility; the full table, including empty bins, is in the README.", { size: 10.5 }));

  const lx = M.l + pw + 24;
  let ly = M.t + 8;
  for (const s of [
    { color: C.blue, name: "Segment × size model" },
    { color: C.orange, name: "Deployed heuristic ($-weighted)" },
  ]) {
    b.push(dot(lx + 6, ly - 4, 5, s.color, false));
    b.push(t(lx + 18, ly, s.name, { size: 12, fill: C.ink }));
    ly += 22;
  }
  ly += 10;
  const notes = [
    ["Above the diagonal = model under-", "predicts; below = overconfident."],
    ["91% of segment × size predictions", "land in the two left bins, which sit", "close to the diagonal", "(13.1% predicted → 12.9% actual)."],
    ["The deployed heuristic is anti-", "calibrated: its 10–20% bin wins", "22.5% in reality, while its huge", "0–10% bin (n=1,682) wins 12.8%."],
    ["Bins right of 30% hold few", "opportunities — whiskers, not", "points, are the signal there."],
  ];
  for (const para of notes) {
    for (const lineTxt of para) {
      b.push(t(lx, ly, lineTxt, { size: 11.5, fill: C.ink2 }));
      ly += 16;
    }
    ly += 8;
  }
  writeFileSync(new URL("../charts/calibration.svg", import.meta.url), svgWrap(W, H, b.join("\n")));
}

// ------------------------------------------------------------------ look-ahead
{
  const W = 880, H = 400;
  const M = { l: 210, r: 60, t: 84, b: 70 };
  const pw = W - M.l - M.r, ph = H - M.t - M.b;
  const x0 = 0.4, x1 = 0.8;
  const X = (a) => M.l + ((a - x0) / (x1 - x0)) * pw;
  const rows = [
    { key: "m1a_product", name: "Deployed heuristic" },
    { key: "m1b_count", name: "Scope base rates" },
    { key: "m4_seg_size", name: "Segment × size" },
    { key: "m2_gc", name: "GC hierarchy" },
  ];
  const rowY = (i) => M.t + ((i + 0.5) / rows.length) * ph;
  let b = [];
  b.push(t(24, 28, "The look-ahead illusion — discrimination (AUC), in-sample vs. walk-forward", { size: 15, fill: C.ink, weight: "600" }));
  b.push(t(24, 46, "Same models, same bids, with and without the walk-forward rule.", { size: 12, fill: C.ink2 }));

  // legend, top right
  b.push(dot(W - 254, 24, 5, C.orange, false));
  b.push(t(W - 243, 28, "in-sample (sees the future)", { size: 11.5, fill: C.ink }));
  b.push(dot(W - 254, 44, 5, C.blue, false));
  b.push(t(W - 243, 48, "walk-forward (honest)", { size: 11.5, fill: C.ink }));

  for (let g = 0.4; g <= 0.801; g += 0.1) {
    b.push(line(X(g), M.t, X(g), M.t + ph, C.grid));
    b.push(t(X(g), M.t + ph + 18, g.toFixed(1), { anchor: "middle", tab: true }));
  }
  b.push(line(X(0.5), M.t - 6, X(0.5), M.t + ph, C.axis, 1.5, "4 3"));
  b.push(t(X(0.5), M.t - 12, "0.5 = coin flip", { anchor: "middle", size: 10.5 }));
  b.push(t(M.l + pw / 2, M.t + ph + 42, "AUC (probability a random won bid outranks a random lost one)", { anchor: "middle", fill: C.ink2 }));

  rows.forEach((r, i) => {
    const y = rowY(i);
    const a1 = R.cheat_full_history.models[r.key].auc;
    const a2 = R.protocols.resolution_L90.models[r.key].auc;
    b.push(t(M.l - 16, y + 4, r.name, { size: 12.5, fill: C.ink, anchor: "end" }));
    b.push(line(X(Math.min(a1, a2)), y, X(Math.max(a1, a2)), y, C.axis, 2));
    b.push(dot(X(a1), y, 6, C.orange));
    b.push(dot(X(a2), y, 6, C.blue));
    const dy = i % 2 === 0 ? -10 : 19;
    b.push(t(X(a1), y + dy, a1.toFixed(3), { size: 10.5, fill: C.ink2, anchor: "middle", tab: true }));
    b.push(t(X(a2), y + dy, a2.toFixed(3), { size: 10.5, fill: C.ink2, anchor: "middle", tab: true }));
    if (r.key === "m2_gc")
      b.push(t(X((a1 + a2) / 2), y - 12, `−${(a1 - a2).toFixed(2)} AUC of pure memorization`, { size: 11, fill: C.ink2, anchor: "middle" }));
  });
  writeFileSync(new URL("../charts/lookahead.svg", import.meta.url), svgWrap(W, H, b.join("\n")));
}

// --------------------------------------------------------------------- signal
{
  const D = R.descriptives;
  const W = 880, H = 440;
  const yMax = 0.6;
  const panels = [
    {
      title: "…by bid size",
      x: 80, w: 330,
      cats: ["S", "M", "L", "XL"].map((k) => ({ label: k, ...D.bySize[k] })),
      note: "size tiers split at fixed dollar thresholds",
    },
    {
      title: "…by market segment",
      x: 490, w: 330,
      cats: [
        ["commercial", "commercial"],
        ["single_family", "single-family"],
        ["material_only", "material-only"],
      ].map(([k, label]) => ({ label, ...D.bySegment[k] })),
      note: "the service segment (n=1) is omitted",
    },
  ];
  const T = 92, B = 84;
  const ph = H - T - B;
  const Y = (v) => T + (1 - v / yMax) * ph;
  const wilson = (w, n) => {
    const z = 1.96, p = w / n, d = 1 + (z * z) / n;
    const c = (p + (z * z) / (2 * n)) / d;
    const h = (z * Math.sqrt((p * (1 - p)) / n + (z * z) / (4 * n * n))) / d;
    return [Math.max(0, c - h), Math.min(1, c + h)];
  };
  let b = [];
  b.push(t(24, 28, "Where the signal lives — actual win rate…", { size: 15, fill: C.ink, weight: "600" }));
  b.push(t(24, 46, "1,954 decided opportunities. Whiskers are 95% Wilson intervals. Scope, the dimension the deployed model segments on, spans only 12–16% (table in README).", { size: 12, fill: C.ink2 }));

  for (const p of panels) {
    b.push(t(p.x, T - 16, p.title, { size: 13, fill: C.ink, weight: "600" }));
    for (let g = 0; g <= yMax + 0.001; g += 0.1) {
      b.push(line(p.x, Y(g), p.x + p.w, Y(g), C.grid));
      b.push(t(p.x - 8, Y(g) + 4, pct(g), { anchor: "end", tab: true }));
    }
    b.push(line(p.x, Y(0), p.x + p.w, Y(0), C.axis, 1.5));
    const bw = 44, n = p.cats.length;
    p.cats.forEach((c, i) => {
      const cx = p.x + ((i + 0.5) / n) * p.w;
      const v = c.w / c.n;
      const [lo, hi] = wilson(c.w, c.n);
      const x = cx - bw / 2, yTop = Y(v);
      b.push(`<path d="M${x},${Y(0)} L${x},${(yTop + 4).toFixed(1)} Q${x},${yTop.toFixed(1)} ${x + 4},${yTop.toFixed(1)} L${x + bw - 4},${yTop.toFixed(1)} Q${x + bw},${yTop.toFixed(1)} ${x + bw},${(yTop + 4).toFixed(1)} L${x + bw},${Y(0)} Z" fill="${C.blue}"/>`);
      b.push(line(cx, Y(lo), cx, Y(hi), C.ink2, 2));
      b.push(t(cx, Y(Math.max(hi, v)) - 8, pct(v, 1), { size: 11.5, fill: C.ink, anchor: "middle", weight: "600", tab: true }));
      b.push(t(cx, Y(0) + 18, c.label, { size: 12, fill: C.ink2, anchor: "middle" }));
      b.push(t(cx, Y(0) + 34, `n=${c.n}`, { size: 10.5, anchor: "middle", tab: true }));
    });
    b.push(t(p.x, H - 14, p.note, { size: 10.5 }));
  }
  writeFileSync(new URL("../charts/signal.svg", import.meta.url), svgWrap(W, H, b.join("\n")));
}

console.log("charts written.");
