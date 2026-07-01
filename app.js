/* ============================================================================
   Convexity — dashboard application logic.
   ========================================================================== */

import { scoreColor, convictionDots, heatStrip, radarChart, compositeBars, convictionScatter, gauge } from "./charts.js";
import { isLive, loadSample, loadLatest, runScan, fetchCompany } from "./api.js";

// Canonical display order for the 12 categories (matches backend category strings).
const CATS = [
  "value", "growth", "quality", "financial_health",
  "catalyst", "momentum", "technical", "competitive",
  "management", "ownership", "historical_analog", "risk",
];
const CAT_LABEL = {
  value: "Value", growth: "Growth", quality: "Quality", financial_health: "Financial Health",
  catalyst: "Catalyst", momentum: "Momentum", technical: "Technical", competitive: "Competitive",
  management: "Management", ownership: "Ownership", historical_analog: "Historical Analog", risk: "Risk",
};

const state = {
  scan: null,
  rows: [],
  sortKey: "rank",
  sortDir: 1,
  filter: "",
  live: false,
};

// ------------------------------------------------------------------ helpers
const $ = (sel, el = document) => el.querySelector(sel);
const $$ = (sel, el = document) => Array.from(el.querySelectorAll(sel));
const esc = (s) => String(s ?? "").replace(/[&<>"]/g, (c) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;" }[c]));

function fmtCap(n) {
  if (n == null) return "—";
  if (n >= 1e9) return "$" + (n / 1e9).toFixed(2) + "B";
  if (n >= 1e6) return "$" + (n / 1e6).toFixed(0) + "M";
  return "$" + Math.round(n).toLocaleString();
}
function fmtInt(n) { return n == null ? "—" : Number(n).toLocaleString(); }
function fmtTime(iso) {
  if (!iso) return "—";
  try { const d = new Date(iso); return d.toLocaleString(undefined, { month: "short", day: "numeric", hour: "2-digit", minute: "2-digit" }); }
  catch (_e) { return String(iso); }
}
function toast(msg) {
  let t = $(".toast");
  if (!t) { t = document.createElement("div"); t.className = "toast"; document.body.appendChild(t); }
  t.textContent = msg; t.classList.add("show");
  clearTimeout(t._h); t._h = setTimeout(() => t.classList.remove("show"), 2600);
}

// ------------------------------------------------------------------ load
async function boot() {
  wireControls();
  state.live = await isLive();
  updateModePill();
  // Prefer a live latest scan; else the bundled sample (demo).
  let scan = null;
  if (state.live) scan = await loadLatest();
  if (!scan) { try { scan = await loadSample(); if (!state.live) toast("Demo mode — showing bundled sample scan"); } catch (_e) {} }
  if (scan) applyScan(scan);
  else renderEmpty("No scan available. Run a scan to begin.");
}

function updateModePill() {
  const pill = $("#modePill");
  if (state.live) { pill.className = "mode-pill live"; pill.innerHTML = '<span class="dot"></span> Live · API connected'; }
  else { pill.className = "mode-pill demo"; pill.innerHTML = '<span class="dot"></span> Demo · sample data'; }
  // Keep the Run button clickable even in demo mode — instead of a dead greyed-out
  // button, a click explains that live scans need the local backend.
  $("#runBtn").disabled = false;
  $("#runBtn").title = state.live ? "Run a live scan" : "Live scans need the local backend — click to see how";
  const note = $("#demoNote");
  if (note) note.style.display = state.live ? "none" : "flex";
}

function applyScan(scan) {
  state.scan = scan;
  state.rows = (scan.all_ranked && scan.all_ranked.length ? scan.all_ranked : scan.top || []).slice();
  renderAll();
}

// ------------------------------------------------------------------ render
function renderAll() {
  renderKpis();
  renderCharts();
  renderTable();
}

function renderKpis() {
  const s = state.scan;
  const rows = state.rows;
  const avgConv = rows.length ? rows.reduce((a, c) => a + (c.conviction_confidence || 0), 0) / rows.length : 0;
  const kpis = [
    ["Universe", fmtInt(s.universe_size), "eligible small/micro-caps"],
    ["Screened", fmtInt(s.screened_count), "passed cap + liquidity"],
    ["Analyzed", fmtInt(s.analyzed_count), (s.error_count ? s.error_count + " errored" : "0 errored")],
    ["Avg conviction", (avgConv * 100).toFixed(0) + "%", "independent-signal agreement"],
    ["Generated", fmtTime(s.generated_at), (s.elapsed_seconds ? s.elapsed_seconds.toFixed(1) + "s scan" : "")],
  ];
  $("#kpis").innerHTML = kpis.map(([l, v, sub]) => `
    <div class="kpi"><div class="k-label">${esc(l)}</div><div class="k-val">${esc(v)}</div><div class="k-sub">${esc(sub)}</div></div>
  `).join("");
}

function renderCharts() {
  const top = (state.scan.top && state.scan.top.length ? state.scan.top : state.rows).slice(0, 8);
  $("#barChart").innerHTML = compositeBars(top, { width: 520 });
  $("#scatterChart").innerHTML = convictionScatter(top, { width: 460, height: 300 });
}

function sortRows() {
  const k = state.sortKey, dir = state.sortDir;
  const get = (c) => {
    switch (k) {
      case "rank": return c.rank ?? 9999;
      case "ticker": return c.ticker || "";
      case "cap": return c.market_cap ?? 0;
      case "composite": return c.composite_score ?? 0;
      case "conviction": return c.conviction_confidence ?? 0;
      case "agreement": return c.signal_agreement ?? 0;
      default: return 0;
    }
  };
  state.rows.sort((a, b) => {
    const va = get(a), vb = get(b);
    if (typeof va === "string") return dir * va.localeCompare(vb);
    return dir * (va - vb);
  });
}

function renderTable() {
  sortRows();
  const f = state.filter.toLowerCase();
  const rows = state.rows.filter((c) =>
    !f || (c.ticker || "").toLowerCase().includes(f) || (c.name || "").toLowerCase().includes(f) ||
    (c.sector || "").toLowerCase().includes(f) || (c.industry || "").toLowerCase().includes(f));

  const arrow = (k) => state.sortKey === k ? `<span class="arrow">${state.sortDir > 0 ? "▲" : "▼"}</span>` : "";
  const head = `
    <thead><tr>
      <th class="sortable" data-k="rank">#${arrow("rank")}</th>
      <th class="sortable" data-k="ticker">Company${arrow("ticker")}</th>
      <th class="hide-sm">Sector</th>
      <th class="sortable num hide-sm" data-k="cap">Mkt Cap${arrow("cap")}</th>
      <th class="sortable score-cell" data-k="composite">Composite${arrow("composite")}</th>
      <th class="sortable hide-sm" data-k="conviction">Conviction${arrow("conviction")}</th>
      <th class="hide-sm">Signal map</th>
    </tr></thead>`;

  const body = rows.map((c) => {
    const col = scoreColor(c.composite_score);
    return `<tr data-tkr="${esc(c.ticker)}">
      <td class="rank-num">${c.rank ?? ""}</td>
      <td><div class="tkr">${esc(c.ticker)}</div><div class="co-name">${esc(c.name)}</div></td>
      <td class="hide-sm"><span class="chip">${esc(c.sector || "—")}</span></td>
      <td class="num hide-sm">${fmtCap(c.market_cap)}</td>
      <td class="score-cell"><div class="score-bar-row">
        <div class="score-bar"><span style="width:${c.composite_score}%;background:${col}"></span></div>
        <div class="score-val" style="color:${col}">${(c.composite_score || 0).toFixed(0)}</div>
      </div></td>
      <td class="hide-sm">${convictionDots(c.conviction_confidence)}</td>
      <td class="hide-sm">${heatStrip(c.subscores, CATS)}</td>
    </tr>`;
  }).join("");

  $("#tableCard").innerHTML = `<table class="rank">${head}<tbody>${body || emptyRow()}</tbody></table>`;
  $$("#tableCard th.sortable").forEach((th) => th.addEventListener("click", () => {
    const k = th.dataset.k;
    if (state.sortKey === k) state.sortDir *= -1; else { state.sortKey = k; state.sortDir = k === "ticker" ? 1 : (k === "rank" ? 1 : -1); }
    renderTable();
  }));
  $$("#tableCard tbody tr[data-tkr]").forEach((tr) => tr.addEventListener("click", () => openDrawer(tr.dataset.tkr)));
}
function emptyRow() { return `<tr><td colspan="7" style="text-align:center;padding:40px;color:var(--text-faint)">No matches</td></tr>`; }

function renderEmpty(msg) {
  $("#kpis").innerHTML = "";
  $("#tableCard").innerHTML = `<div class="empty"><div class="big">${esc(msg)}</div><div>Convexity searches the eligible U.S. small/micro-cap universe and ranks the highest-conviction ideas by transparent, independent evidence.</div></div>`;
}

// ------------------------------------------------------------------ drawer
function findCompany(tkr) { return state.rows.find((c) => c.ticker === tkr); }

async function openDrawer(tkr) {
  let c = findCompany(tkr);
  const overlay = $("#overlay"), drawer = $("#drawer");
  overlay.classList.add("open"); drawer.classList.add("open");
  drawer.scrollTop = 0;
  renderDrawer(c);
}
function closeDrawer() { $("#overlay").classList.remove("open"); $("#drawer").classList.remove("open"); }

function renderDrawer(c) {
  if (!c) { $("#drawer").innerHTML = ""; return; }
  const col = scoreColor(c.composite_score);
  const subByCat = {}; (c.subscores || []).forEach((s) => { subByCat[s.category] = s; });

  const metrics = `
    <div class="metrics">
      <div class="d-metric"><div class="m-label">Composite</div><div class="m-val" style="color:${col}">${(c.composite_score || 0).toFixed(1)}</div></div>
      <div class="d-metric"><div class="m-label">Conviction</div><div class="m-val">${((c.conviction_confidence || 0) * 100).toFixed(0)}%</div></div>
      <div class="d-metric"><div class="m-label">Signal agreement</div><div class="m-val">${((c.signal_agreement || 0) * 100).toFixed(0)}%</div></div>
      <div class="d-metric"><div class="m-label">Market cap</div><div class="m-val">${fmtCap(c.market_cap)}</div></div>
    </div>`;

  const list = (arr, cls) => (arr && arr.length)
    ? `<ul class="case-list ${cls}">${arr.map((x) => `<li>${esc(x)}</li>`).join("")}</ul>`
    : `<div style="color:var(--text-faint);font-size:12.5px">None recorded.</div>`;

  const summaries = [
    ["Valuation", c.valuation_summary],
    ["Fundamentals", c.fundamental_summary],
    ["Technical", c.technical_summary],
  ].filter(([, v]) => v).map(([l, v]) => `<div class="summ"><div class="s-lab">${l}</div><div class="s-txt">${esc(v)}</div></div>`).join("");

  const subs = CATS.map((cat) => {
    const s = subByCat[cat];
    if (!s) return "";
    const scol = scoreColor(s.score);
    const ev = (s.evidence || []).map((e) => `
      <div class="ev-row">
        <span class="ev-dir ${esc(e.direction || "neutral")}"></span>
        <span class="ev-label">${esc(e.label)}${e.detail ? ` <span class="ev-src">— ${esc(e.detail)}</span>` : ""}</span>
        <span class="ev-val">${esc(e.value)}</span>
        <span class="ev-src">${esc(e.source || "")}</span>
      </div>`).join("");
    const flags = (s.flags || []).map((fl) => `<span class="flag-tag">${esc(fl)}</span>`).join("");
    return `<div class="subscore" data-cat="${cat}">
      <div class="ss-head">
        <span class="ss-cat">${CAT_LABEL[cat] || cat}</span>
        <span class="ss-bar"><span style="width:${s.score}%;background:${scol}"></span></span>
        <span class="ss-score" style="color:${scol}">${(s.score || 0).toFixed(0)}</span>
        <span class="ss-meta">c ${(s.confidence * 100).toFixed(0)}% · d ${(s.data_coverage * 100).toFixed(0)}%</span>
        <svg class="ss-caret" viewBox="0 0 16 16" fill="none"><path d="M6 4l4 4-4 4" stroke="currentColor" stroke-width="1.6" stroke-linecap="round"/></svg>
      </div>
      <div class="ss-body">
        <div class="ss-rationale">${esc(s.rationale)}</div>
        ${ev}
        ${flags ? `<div style="margin-top:10px">${flags}</div>` : ""}
      </div>
    </div>`;
  }).join("");

  $("#drawer").innerHTML = `
    <div class="d-head">
      <div class="top">
        <span class="rankbadge">RANK ${c.rank ?? "—"}</span>
        <div>
          <h3>${esc(c.ticker)} <span style="color:var(--text-faint);font-weight:400;font-size:15px">${esc(c.name)}</span></h3>
          <div class="sub">${esc(c.sector || "")}${c.industry ? " · " + esc(c.industry) : ""} · ${esc((c.cap_tier || "").toUpperCase())}-cap</div>
        </div>
        <button class="btn ghost close" id="closeDrawer" aria-label="Close">✕</button>
      </div>
      ${metrics}
    </div>
    <div class="d-body">
      <div class="d-block"><h4>Investment thesis</h4><div class="thesis">${esc(c.thesis)}</div></div>

      <div class="d-block"><h4>Conviction radar · 12 independent categories</h4>
        <div class="radar-wrap">${radarChart(c.subscores, { size: 360, order: CATS })}</div>
        <div class="legend"><span><i style="background:var(--teal)"></i> stronger evidence</span><span><i style="background:var(--neg)"></i> weaker / caution</span></div>
      </div>

      <div class="d-block"><h4>Bull / bear</h4>
        <div class="two-col">
          <div><div class="s-lab" style="color:var(--pos);font-size:10px;letter-spacing:1px;text-transform:uppercase;margin-bottom:6px">Bull case</div>${list(c.bull_case, "bull")}</div>
          <div><div class="s-lab" style="color:var(--neg);font-size:10px;letter-spacing:1px;text-transform:uppercase;margin-bottom:6px">Bear case</div>${list(c.bear_case, "bear")}</div>
        </div>
      </div>

      <div class="d-block"><h4>Key catalysts</h4>${list(c.catalysts, "cat")}</div>
      <div class="d-block"><h4>Principal risks</h4>${list(c.principal_risks, "risk")}</div>

      ${summaries ? `<div class="d-block"><h4>Summaries</h4><div class="summ-grid">${summaries}</div></div>` : ""}

      ${c.confidence_explanation ? `<div class="d-block"><h4>Why this confidence</h4><div class="confidence-box">${esc(c.confidence_explanation)}</div></div>` : ""}

      <div class="d-block"><h4>Sub-score breakdown · click to expand evidence</h4>${subs}</div>

      <div class="d-block"><h4>Monitoring checklist · what would confirm or invalidate</h4>
        <ul class="checklist">${(c.monitoring_checklist || []).map((m) => `<li><span class="box"></span><span>${esc(m)}</span></li>`).join("") || '<li>None recorded.</li>'}</ul>
      </div>
    </div>`;

  $("#closeDrawer").addEventListener("click", closeDrawer);
  $$("#drawer .subscore .ss-head").forEach((h) => h.addEventListener("click", () => h.parentElement.classList.toggle("open")));
}

// ------------------------------------------------------------------ controls
function wireControls() {
  // range value mirrors
  $$(".field input[type=range]").forEach((r) => {
    const out = $(`#${r.dataset.out}`);
    const upd = () => { if (out) out.textContent = r.dataset.fmt === "cap" ? fmtCap(Number(r.value)) : r.value; };
    r.addEventListener("input", upd); upd();
  });
  $("#overlay").addEventListener("click", closeDrawer);
  document.addEventListener("keydown", (e) => { if (e.key === "Escape") closeDrawer(); });
  $("#filterInput").addEventListener("input", (e) => { state.filter = e.target.value; renderTable(); });
  $("#loadSampleBtn").addEventListener("click", async () => { try { applyScan(await loadSample()); toast("Loaded bundled sample scan"); } catch (_e) { toast("Sample not found"); } });
  $("#exportJson").addEventListener("click", exportJson);
  $("#exportCsv").addEventListener("click", exportCsv);
  $("#runBtn").addEventListener("click", doScan);
}

function showBackendModal() {
  if ($("#backendModal")) return;
  const el = document.createElement("div");
  el.id = "backendModal";
  el.innerHTML = `
    <div class="bm-backdrop"></div>
    <div class="bm-card" role="dialog" aria-modal="true">
      <div class="bm-title">This is the live demo — running on sample data</div>
      <p class="bm-body">
        You're viewing Convexity on GitHub&nbsp;Pages, which is a <b>static site with no backend</b>,
        so there's nothing for a live scan to talk to. That's why <b>Run scan</b> can't execute here.
      </p>
      <p class="bm-body">To run real scans against live market data, start the backend on your machine:</p>
      <pre class="bm-code">git clone https://github.com/boydelliott242-art/convexity.git
cd convexity
python3 -m venv .venv &amp;&amp; source .venv/bin/activate
pip install -e ".[dev]"
convexity serve            <span class="bm-cmt"># then open http://localhost:8000</span></pre>
      <p class="bm-body">On <code>localhost:8000</code> the button runs a full live scan with progress. Prefer the terminal? <code>convexity scan --top-n 5</code>.</p>
      <div class="bm-actions">
        <button class="btn ghost" id="bmSample">Explore the sample instead</button>
        <button class="btn primary" id="bmClose">Got it</button>
      </div>
    </div>`;
  document.body.appendChild(el);
  const close = () => el.remove();
  el.querySelector(".bm-backdrop").addEventListener("click", close);
  el.querySelector("#bmClose").addEventListener("click", close);
  el.querySelector("#bmSample").addEventListener("click", async () => { close(); try { applyScan(await loadSample()); toast("Showing the bundled sample scan"); } catch (_e) {} });
  document.addEventListener("keydown", function esc(e) { if (e.key === "Escape") { close(); document.removeEventListener("keydown", esc); } });
}

async function doScan() {
  if (!state.live) { showBackendModal(); return; }
  const params = {
    min_market_cap: Number($("#minCap").value),
    max_market_cap: Number($("#maxCap").value),
    top_n: Number($("#topN").value),
    universe_limit: Number($("#uniLimit").value) || null,
  };
  const pw = $("#progressWrap"); pw.classList.add("active");
  const bar = $("#progressBar"), stage = $("#progressStage"), pctEl = $("#progressPct");
  $("#runBtn").disabled = true;
  try {
    const result = await runScan(params, (p) => {
      bar.style.width = (p.pct || 0) + "%";
      stage.textContent = p.message || p.stage || "working…";
      pctEl.textContent = (p.pct || 0) + "%";
    });
    bar.style.width = "100%"; stage.textContent = "complete"; pctEl.textContent = "100%";
    applyScan(result);
    toast("Scan complete");
    setTimeout(() => pw.classList.remove("active"), 900);
  } catch (e) {
    stage.textContent = "scan failed: " + e.message;
    toast("Scan failed — " + e.message);
  } finally { $("#runBtn").disabled = false; }
}

// ------------------------------------------------------------------ export
function download(name, text, type) {
  const blob = new Blob([text], { type });
  const url = URL.createObjectURL(blob);
  const a = document.createElement("a"); a.href = url; a.download = name; a.click();
  setTimeout(() => URL.revokeObjectURL(url), 1000);
}
function exportJson() {
  if (!state.scan) return;
  download("convexity_scan.json", JSON.stringify(state.scan, null, 2), "application/json");
  toast("Exported JSON");
}
function exportCsv() {
  if (!state.rows.length) return;
  const cols = ["rank", "ticker", "name", "sector", "industry", "market_cap", "composite_score", "conviction_confidence", "signal_agreement"];
  const catCols = CATS.map((c) => "score_" + c);
  const header = [...cols, ...catCols].join(",");
  const lines = state.rows.map((c) => {
    const byCat = {}; (c.subscores || []).forEach((s) => { byCat[s.category] = s.score; });
    const base = cols.map((k) => {
      const v = c[k]; const s = v == null ? "" : String(v);
      return /[",\n]/.test(s) ? `"${s.replace(/"/g, '""')}"` : s;
    });
    const cats = CATS.map((k) => (byCat[k] == null ? "" : byCat[k].toFixed(1)));
    return [...base, ...cats].join(",");
  });
  download("convexity_scan.csv", [header, ...lines].join("\n"), "text/csv");
  toast("Exported CSV");
}

boot();
