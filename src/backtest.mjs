#!/usr/bin/env node
// Walk-forward backtest of segmented win-probability models on historical bid data.
//
//   node src/backtest.mjs        (from the repo root; writes results/metrics.json)
//
// Zero dependencies. Reads data/bids.csv — one row per bid opportunity:
//   t_send       day the bid went out (day 0 = first bid in the dataset)
//   scope        product line (generic trade names)
//   segment      market segment: commercial | single_family | material_only | service
//   size_bucket  S/M/L/XL, split at fixed round-dollar thresholds (values withheld)
//   amount_rel   bid amount × an undisclosed constant (ratios preserved, dollars not)
//   gc           anonymized general-contractor code (randomly ordered — the
//                code number carries no volume or identity information)
//   outcome      1 = won, 0 = lost, blank = still open at extraction
//   t_decision   day the outcome was recorded, where the source system captured it
//
// THE LEAKAGE RULES (the point of this exercise):
//   1. A scored opportunity's training set contains only opportunities whose
//      decision day is STRICTLY BEFORE the scoring cutoff.
//   2. The scored opportunity never appears in its own training set (guaranteed
//      by the strict inequality; same-day decisions are excluded too).
//   3. Features never depend on the outcome (no "winning GC" attributes).
//
// Decision days are recorded for ~15% of resolved opportunities. The rest are
// imputed as t_send + L (losses; L swept 30–180) or t_send + 128 (wins; the
// observed median award lag). Sensitivity to L is reported alongside results.

import { readFileSync, writeFileSync, mkdirSync } from "node:fs";

// ---------------------------------------------------------------- data loading
const csv = readFileSync(new URL("../data/bids.csv", import.meta.url), "utf8")
  .trim()
  .split(/\r?\n/);
const header = csv[0].split(",");
const col = Object.fromEntries(header.map((h, i) => [h, i]));
const rows = csv.slice(1).map((line) => line.split(","));

const AWARD_IMPUTE_LAG = 128; // median recorded award lag, days

function loadInstances(lossLag) {
  return rows.map((r) => {
    const outcomeRaw = r[col.outcome];
    const outcome = outcomeRaw === "" ? null : +outcomeRaw;
    const sendDay = +r[col.t_send];
    const rec = r[col.t_decision];
    let resDay = null;
    let resRecorded = false;
    if (outcome != null) {
      if (rec !== "") {
        resDay = Math.max(+rec, sendDay);
        resRecorded = true;
      } else {
        resDay = sendDay + (outcome === 1 ? AWARD_IMPUTE_LAG : lossLag);
      }
    }
    return {
      scope: r[col.scope],
      segment: r[col.segment],
      size: r[col.size_bucket],
      amount: +r[col.amount_rel],
      gc: r[col.gc],
      sendDay,
      resDay,
      resRecorded,
      outcome,
    };
  });
}

// ------------------------------------------------------- incremental training state
const newCell = () => ({ n: 0, w: 0, bidDol: 0, wonDol: 0 });

class State {
  constructor() {
    this.global = newCell();
    this.maps = {
      segScope: new Map(),
      seg: new Map(),
      scope: new Map(),
      gcScope: new Map(),
      gc: new Map(),
      scopeSize: new Map(),
      segSize: new Map(),
    };
    // dollars of opportunities visible but not yet decided at the cutoff
    // (the deployed model's rate map includes open bids in its denominator)
    this.openDol = { segScope: new Map(), seg: new Map() };
  }
  #bump(name, key, inst) {
    const m = this.maps[name];
    if (!m.has(key)) m.set(key, newCell());
    const c = m.get(key);
    c.n += 1;
    c.w += inst.outcome;
    c.bidDol += inst.amount;
    c.wonDol += inst.outcome ? inst.amount : 0;
  }
  addResolved(inst) {
    const g = this.global;
    g.n += 1;
    g.w += inst.outcome;
    g.bidDol += inst.amount;
    g.wonDol += inst.outcome ? inst.amount : 0;
    this.#bump("segScope", `${inst.segment}|${inst.scope}`, inst);
    this.#bump("seg", inst.segment, inst);
    this.#bump("scope", inst.scope, inst);
    this.#bump("gcScope", `${inst.gc}|${inst.scope}`, inst);
    this.#bump("gc", inst.gc, inst);
    this.#bump("scopeSize", `${inst.scope}|${inst.size}`, inst);
    this.#bump("segSize", `${inst.segment}|${inst.size}`, inst);
  }
  addOpenDol(inst, sign) {
    for (const [name, key] of [
      ["segScope", `${inst.segment}|${inst.scope}`],
      ["seg", inst.segment],
    ]) {
      const m = this.openDol[name];
      m.set(key, (m.get(key) || 0) + sign * inst.amount);
    }
  }
}

// ------------------------------------------------------------------- the models
// Laplace smoothing (w+1)/(n+2) keeps empirical cells off exact 0 and 1.
const lap = (c) => (c.w + 1) / (c.n + 2);
const get = (m, k) => m.get(k) || null;

function predict(state, inst, { minN }) {
  const g = state.global;
  const preds = {};
  const ssKey = `${inst.segment}|${inst.scope}`;

  // Baselines ---------------------------------------------------------------
  preds.b0_const = 0.5; // uninformed
  preds.b1_global = g.n > 0 ? g.w / g.n : 0.5; // trailing company-wide win rate

  // M1a — the deployed heuristic, replicated faithfully:
  // dollar-weighted award rate by segment|scope, open bids included in the
  // denominator, fallback to the segment total, then a 10% floor.
  {
    const cell = get(state.maps.segScope, ssKey);
    const openSS = state.openDol.segScope.get(ssKey) || 0;
    const denom = (cell ? cell.bidDol : 0) + openSS;
    if (denom > 0) preds.m1a_product = (cell ? cell.wonDol : 0) / denom;
    else {
      const seg = get(state.maps.seg, inst.segment);
      const openSeg = state.openDol.seg.get(inst.segment) || 0;
      const d2 = (seg ? seg.bidDol : 0) + openSeg;
      preds.m1a_product = d2 > 0 ? (seg ? seg.wonDol : 0) / d2 : 0.1;
    }
  }

  // M1a' — same model restricted to decided opportunities only
  {
    const cell = get(state.maps.segScope, ssKey);
    if (cell && cell.bidDol > 0) preds.m1a_resolved = cell.wonDol / cell.bidDol;
    else {
      const seg = get(state.maps.seg, inst.segment);
      preds.m1a_resolved = seg && seg.bidDol > 0 ? seg.wonDol / seg.bidDol : 0.1;
    }
  }

  // M1b — count-based segment|scope with Laplace smoothing
  {
    const cell = get(state.maps.segScope, ssKey);
    if (cell && cell.n > 0) preds.m1b_count = lap(cell);
    else {
      const seg = get(state.maps.seg, inst.segment);
      preds.m1b_count = seg && seg.n > 0 ? lap(seg) : lap(g);
    }
  }

  // M2 — GC hierarchy: gc|scope (n≥minN) → gc (n≥minN) → segment|scope → segment → global
  {
    const gs = get(state.maps.gcScope, `${inst.gc}|${inst.scope}`);
    const gcC = get(state.maps.gc, inst.gc);
    let p, lvl;
    if (gs && gs.n >= minN) [p, lvl] = [lap(gs), "gc_scope"];
    else if (gcC && gcC.n >= minN) [p, lvl] = [lap(gcC), "gc"];
    else {
      const cell = get(state.maps.segScope, ssKey);
      if (cell && cell.n > 0) [p, lvl] = [lap(cell), "seg_scope"];
      else {
        const seg = get(state.maps.seg, inst.segment);
        if (seg && seg.n > 0) [p, lvl] = [lap(seg), "seg"];
        else [p, lvl] = [lap(g), "global"];
      }
    }
    preds.m2_gc = p;
    preds.m2_level = lvl;
  }

  // M3 — scope × size bucket → scope → global
  {
    const ss = get(state.maps.scopeSize, `${inst.scope}|${inst.size}`);
    if (ss && ss.n >= minN) preds.m3_size = lap(ss);
    else {
      const sc = get(state.maps.scope, inst.scope);
      preds.m3_size = sc && sc.n > 0 ? lap(sc) : lap(g);
    }
  }

  // M4 — segment × size bucket → segment → global
  {
    const ps = get(state.maps.segSize, `${inst.segment}|${inst.size}`);
    if (ps && ps.n >= minN) preds.m4_seg_size = lap(ps);
    else {
      const seg = get(state.maps.seg, inst.segment);
      preds.m4_seg_size = seg && seg.n > 0 ? lap(seg) : lap(g);
    }
  }

  return preds;
}

// ------------------------------------------------------------------ scoring math
const CLIP = 0.01; // log-loss clip; Brier and calibration use raw predictions
const clip = (p) => Math.min(1 - CLIP, Math.max(CLIP, p));

function wilson(w, n) {
  if (n === 0) return [0, 1];
  const z = 1.96;
  const phat = w / n;
  const denom = 1 + z ** 2 / n;
  const center = (phat + z ** 2 / (2 * n)) / denom;
  const half =
    (z * Math.sqrt((phat * (1 - phat)) / n + z ** 2 / (4 * n ** 2))) / denom;
  return [Math.max(0, center - half), Math.min(1, center + half)];
}

function calibBins(records, model, k = 10) {
  const bins = Array.from({ length: k }, (_, i) => ({
    lo: i / k,
    hi: (i + 1) / k,
    n: 0,
    w: 0,
    sumP: 0,
  }));
  for (const r of records) {
    const b = bins[Math.min(k - 1, Math.floor(r.preds[model] * k))];
    b.n += 1;
    b.w += r.outcome;
    b.sumP += r.preds[model];
  }
  return bins.map((b) => ({
    range: `${Math.round(b.lo * 100)}–${Math.round(b.hi * 100)}%`,
    n: b.n,
    wins: b.w,
    meanPred: b.n ? b.sumP / b.n : null,
    actual: b.n ? b.w / b.n : null,
    ci95: b.n ? wilson(b.w, b.n) : null,
  }));
}

function metrics(records, model) {
  const n = records.length;
  let brier = 0,
    ll = 0;
  for (const r of records) {
    const p = r.preds[model];
    brier += (p - r.outcome) ** 2;
    const pc = clip(p);
    ll += -(r.outcome * Math.log(pc) + (1 - r.outcome) * Math.log(1 - pc));
  }
  brier /= n;
  ll /= n;
  // AUC by rank statistic with tie-averaged ranks
  const sorted = [...records].sort((a, b) => a.preds[model] - b.preds[model]);
  let sumRankPos = 0;
  for (let i = 0; i < sorted.length; ) {
    let j = i;
    while (j < sorted.length && sorted[j].preds[model] === sorted[i].preds[model]) j++;
    const avgRank = (i + j + 1) / 2;
    for (let k = i; k < j; k++) if (sorted[k].outcome === 1) sumRankPos += avgRank;
    i = j;
  }
  const nPos = records.reduce((s, r) => s + r.outcome, 0);
  const nNeg = n - nPos;
  const auc =
    nPos && nNeg ? (sumRankPos - (nPos * (nPos + 1)) / 2) / (nPos * nNeg) : null;
  let ece = 0;
  for (const b of calibBins(records, model))
    if (b.n > 0) ece += (b.n / n) * Math.abs(b.actual - b.meanPred);
  return { n, brier, logLoss: ll, auc, ece, nPos, baseRate: nPos / n };
}

// -------------------------------------------------------------- walk-forward sweep
const MODELS = [
  "b0_const",
  "b1_global",
  "m1a_product",
  "m1a_resolved",
  "m1b_count",
  "m2_gc",
  "m3_size",
  "m4_seg_size",
];
const BURN_IN = 50; // minimum decided opportunities before scoring starts

function sweep(instances, cutoffKind, opts) {
  const resolved = instances.filter((x) => x.outcome != null);
  const cutoffOf = (x) => (cutoffKind === "resolution" ? x.resDay : x.sendDay);
  const scoreOrder = [...resolved].sort((a, b) => cutoffOf(a) - cutoffOf(b));
  const resOrder = [...resolved].sort((a, b) => a.resDay - b.resDay);
  const sendOrder = [...instances].sort((a, b) => a.sendDay - b.sendDay);
  const state = new State();
  let ri = 0,
    si = 0,
    skipped = 0;
  const records = [];
  for (const inst of scoreOrder) {
    const cut = cutoffOf(inst);
    // everything decided strictly before the cutoff becomes training data
    while (ri < resOrder.length && resOrder[ri].resDay < cut) {
      state.addResolved(resOrder[ri]);
      state.addOpenDol(resOrder[ri], -1);
      ri++;
    }
    // everything sent strictly before the cutoff is visible (open $ until decided)
    while (si < sendOrder.length && sendOrder[si].sendDay < cut) {
      state.addOpenDol(sendOrder[si], +1);
      si++;
    }
    if (state.global.n < BURN_IN) {
      skipped++;
      continue;
    }
    const preds = predict(state, inst, opts);
    records.push({
      cutoffDay: cut,
      outcome: inst.outcome,
      scope: inst.scope,
      segment: inst.segment,
      size: inst.size,
      resRecorded: inst.resRecorded,
      m2Level: preds.m2_level,
      preds,
    });
  }
  return { records, skipped, nResolved: resolved.length };
}

// In-sample "cheat" run: the training state contains every decided opportunity,
// including the one being scored. This is what NOT doing walk-forward looks like.
function cheatRun(instances, opts) {
  const resolved = instances.filter((x) => x.outcome != null);
  const state = new State();
  for (const inst of resolved) state.addResolved(inst);
  return resolved.map((inst) => ({
    outcome: inst.outcome,
    preds: predict(state, inst, opts),
  }));
}

// ------------------------------------------------------------------ protocol grid
const OPTS = { minN: 5 };
const results = { protocols: {} };

for (const lossLag of [30, 60, 90, 120, 180]) {
  const instances = loadInstances(lossLag);
  for (const cutoffKind of ["resolution", "send"]) {
    const tag = `${cutoffKind}_L${lossLag}`;
    const { records, skipped, nResolved } = sweep(instances, cutoffKind, OPTS);
    const entry = { lossLag, cutoffKind, skipped, nResolved, models: {} };
    for (const m of MODELS) entry.models[m] = metrics(records, m);
    if (lossLag === 90) {
      entry.calibration = {};
      for (const m of MODELS) entry.calibration[m] = calibBins(records, m);
      entry.m2Levels = {};
      for (const r of records)
        entry.m2Levels[r.m2Level] = (entry.m2Levels[r.m2Level] || 0) + 1;
      entry.recordedShare =
        records.filter((r) => r.resRecorded).length / records.length;
    }
    results.protocols[tag] = entry;
  }
}

{
  const instances = loadInstances(90);
  const records = cheatRun(instances, OPTS);
  results.cheat_full_history = { models: {} };
  for (const m of MODELS) results.cheat_full_history.models[m] = metrics(records, m);

  results.m2_minN_sensitivity = {};
  for (const minN of [3, 5, 10]) {
    const { records: recs } = sweep(instances, "resolution", { minN });
    results.m2_minN_sensitivity[`minN_${minN}`] = metrics(recs, "m2_gc");
  }

  // dataset descriptives
  const resolved = instances.filter((x) => x.outcome != null);
  const tally = (keyFn) => {
    const out = {};
    for (const x of resolved) {
      const k = keyFn(x);
      out[k] = out[k] || { n: 0, w: 0 };
      out[k].n++;
      out[k].w += x.outcome;
    }
    return out;
  };
  results.descriptives = {
    nInstances: instances.length,
    nResolved: resolved.length,
    winRate: resolved.reduce((s, x) => s + x.outcome, 0) / resolved.length,
    spanDays: Math.max(...resolved.map((x) => x.sendDay)),
    recordedDecisionShare:
      resolved.filter((x) => x.resRecorded).length / resolved.length,
    byScope: tally((x) => x.scope),
    bySegment: tally((x) => x.segment),
    bySize: tally((x) => x.size),
    nGcs: new Set(resolved.map((x) => x.gc)).size,
  };
}

mkdirSync(new URL("../results", import.meta.url), { recursive: true });
writeFileSync(
  new URL("../results/metrics.json", import.meta.url),
  JSON.stringify(results, null, 2)
);

// ---------------------------------------------------------------- console report
const fmt = (x, d = 4) => (x == null ? "  —  " : x.toFixed(d));
function table(title, entry) {
  console.log(`\n=== ${title} ===`);
  console.log("model           brier    logloss  auc     ece");
  for (const m of MODELS) {
    const M = entry.models[m];
    console.log(
      `${m.padEnd(15)} ${fmt(M.brier)}  ${fmt(M.logLoss)}  ${fmt(M.auc, 3)}   ${fmt(M.ece, 3)}`
    );
  }
}
const P = results.protocols.resolution_L90;
console.log(
  `scored=${P.models.b1_global.n}  burn-in skips=${P.skipped}  base rate=${fmt(P.models.b1_global.baseRate, 3)}  recorded decisions=${fmt(P.recordedShare, 2)}`
);
table("walk-forward · decision-day cutoff · L=90 (primary)", P);
table("walk-forward · send-day cutoff · L=90 (conservative)", results.protocols.send_L90);
table("full-history in-sample (look-ahead — for contrast only)", results.cheat_full_history);
console.log("\nL-sensitivity (decision cutoff), Brier:");
for (const L of [30, 60, 90, 120, 180]) {
  const p = results.protocols[`resolution_L${L}`].models;
  console.log(
    `  L=${String(L).padEnd(3)}  b1=${fmt(p.b1_global.brier)}  m1b=${fmt(p.m1b_count.brier)}  m2=${fmt(p.m2_gc.brier)}  m4=${fmt(p.m4_seg_size.brier)}`
  );
}
console.log("\nresults/metrics.json written.");
