/* ============================================================================
   Convexity — hand-built SVG charts (no external chart library).
   Everything renders as inline SVG strings so the dashboard has zero runtime
   dependencies and works when opened directly from disk.
   ========================================================================== */

const SVGNS = "http://www.w3.org/2000/svg";

/* Cool score ramp: 0 (cool red) -> 50 (slate) -> 65 (blue) -> 85+ (teal). */
export function scoreColor(score) {
  const s = Math.max(0, Math.min(100, score ?? 0));
  const stops = [
    [0,   [255, 107, 122]],
    [35,  [192, 118, 134]],
    [50,  [123, 132, 151]],
    [65,  [110, 155, 255]],
    [85,  [70, 224, 192]],
    [100, [70, 224, 192]],
  ];
  let a = stops[0], b = stops[stops.length - 1];
  for (let i = 0; i < stops.length - 1; i++) {
    if (s >= stops[i][0] && s <= stops[i + 1][0]) { a = stops[i]; b = stops[i + 1]; break; }
  }
  const t = b[0] === a[0] ? 0 : (s - a[0]) / (b[0] - a[0]);
  const c = a[1].map((v, i) => Math.round(v + (b[1][i] - v) * t));
  return `rgb(${c[0]}, ${c[1]}, ${c[2]})`;
}

/* Confidence -> filled dots (out of 5). */
export function convictionDots(conf) {
  const on = Math.round(Math.max(0, Math.min(1, conf ?? 0)) * 5);
  let h = '<span class="conv" title="Conviction confidence">';
  for (let i = 0; i < 5; i++) h += `<i class="${i < on ? "on" : ""}"></i>`;
  return h + "</span>";
}

/* 12-category heat strip for the ranked table. */
export function heatStrip(subscores, order) {
  const by = {};
  (subscores || []).forEach((s) => { by[s.category] = s.score; });
  let h = '<span class="heatstrip">';
  order.forEach((cat) => {
    const v = by[cat];
    const col = v == null ? "var(--panel-3)" : scoreColor(v);
    const label = cat.replace(/_/g, " ") + (v == null ? ": n/a" : ": " + Math.round(v));
    h += `<i style="background:${col}" title="${label}"></i>`;
  });
  return h + "</span>";
}

/* Radar / spider chart across the 12 category sub-scores. */
export function radarChart(subscores, opts = {}) {
  const size = opts.size || 380;
  const cx = size / 2, cy = size / 2 + 6;
  const R = size * 0.34;
  const cats = opts.order || [];
  const by = {};
  (subscores || []).forEach((s) => { by[s.category] = s; });
  const n = cats.length;
  const pt = (i, r) => {
    const ang = -Math.PI / 2 + (i / n) * Math.PI * 2;
    return [cx + Math.cos(ang) * r, cy + Math.sin(ang) * r];
  };
  let rings = "";
  [0.25, 0.5, 0.75, 1].forEach((f) => {
    const pts = cats.map((_, i) => pt(i, R * f).join(",")).join(" ");
    rings += `<polygon points="${pts}" fill="none" stroke="rgba(255,255,255,0.06)" stroke-width="1"/>`;
  });
  let spokes = "", labels = "";
  cats.forEach((cat, i) => {
    const [x, y] = pt(i, R);
    spokes += `<line x1="${cx}" y1="${cy}" x2="${x}" y2="${y}" stroke="rgba(255,255,255,0.05)" stroke-width="1"/>`;
    const [lx, ly] = pt(i, R + 20);
    const anchor = Math.abs(lx - cx) < 6 ? "middle" : lx > cx ? "start" : "end";
    const short = cat.replace("financial_health", "fin health").replace("historical_analog", "analog").replace(/_/g, " ");
    labels += `<text x="${lx}" y="${ly + 3}" text-anchor="${anchor}" font-size="8.5" fill="#5f6879" font-family="var(--font)" letter-spacing="0.4" style="text-transform:uppercase">${short}</text>`;
  });
  const valuePts = cats.map((cat, i) => {
    const s = by[cat];
    const v = s ? Math.max(0, Math.min(100, s.score)) : 0;
    return pt(i, R * (v / 100));
  });
  const poly = valuePts.map((p) => p.join(",")).join(" ");
  let dots = "";
  cats.forEach((cat, i) => {
    const s = by[cat];
    const [x, y] = valuePts[i];
    dots += `<circle cx="${x}" cy="${y}" r="2.6" fill="${scoreColor(s ? s.score : 0)}"/>`;
  });
  return `<svg viewBox="0 0 ${size} ${size}" width="100%" style="max-width:${size}px" xmlns="${SVGNS}">
    <defs><radialGradient id="radfill" cx="50%" cy="50%" r="50%">
      <stop offset="0%" stop-color="rgba(110,155,255,0.32)"/>
      <stop offset="100%" stop-color="rgba(70,224,192,0.10)"/>
    </radialGradient></defs>
    ${rings}${spokes}
    <polygon points="${poly}" fill="url(#radfill)" stroke="#6e9bff" stroke-width="1.6"/>
    ${dots}${labels}
  </svg>`;
}

/* Horizontal bar chart — composite scores of the ranked companies. */
export function compositeBars(companies, opts = {}) {
  const w = opts.width || 520;
  const rowH = 30, padL = 66, padR = 44, padT = 6;
  const h = padT * 2 + companies.length * rowH;
  const maxBar = w - padL - padR;
  let bars = "";
  companies.forEach((c, i) => {
    const y = padT + i * rowH;
    const val = c.composite_score || 0;
    const bw = (val / 100) * maxBar;
    const col = scoreColor(val);
    bars += `
      <text x="${padL - 10}" y="${y + rowH / 2 + 4}" text-anchor="end" font-size="11" font-family="var(--mono)" fill="#e9ecf3">${c.ticker}</text>
      <rect x="${padL}" y="${y + 8}" width="${maxBar}" height="${rowH - 16}" rx="4" fill="rgba(255,255,255,0.04)"/>
      <rect x="${padL}" y="${y + 8}" width="${bw}" height="${rowH - 16}" rx="4" fill="${col}">
        <animate attributeName="width" from="0" to="${bw}" dur="0.6s" fill="freeze" calcMode="spline" keySplines="0.2 0.6 0.2 1" keyTimes="0;1"/>
      </rect>
      <text x="${padL + bw + 8}" y="${y + rowH / 2 + 4}" font-size="11" font-family="var(--mono)" fill="#9aa3b6">${val.toFixed(1)}</text>`;
  });
  return `<svg viewBox="0 0 ${w} ${h}" width="100%" xmlns="${SVGNS}">${bars}</svg>`;
}

/* Scatter — conviction confidence (x) vs composite score (y). The prized
   quadrant is top-right: high score AND high independent-signal conviction. */
export function convictionScatter(companies, opts = {}) {
  const w = opts.width || 460, h = opts.height || 300;
  const padL = 44, padR = 18, padT = 16, padB = 34;
  const plotW = w - padL - padR, plotH = h - padT - padB;
  const x = (conf) => padL + Math.max(0, Math.min(1, conf)) * plotW;
  const y = (sc) => padT + (1 - Math.max(0, Math.min(100, sc)) / 100) * plotH;
  let grid = "";
  for (let i = 0; i <= 4; i++) {
    const gy = padT + (i / 4) * plotH;
    grid += `<line x1="${padL}" y1="${gy}" x2="${w - padR}" y2="${gy}" stroke="rgba(255,255,255,0.05)"/>`;
    grid += `<text x="${padL - 8}" y="${gy + 3}" text-anchor="end" font-size="8.5" fill="#5f6879" font-family="var(--mono)">${100 - i * 25}</text>`;
    const gx = padL + (i / 4) * plotW;
    grid += `<line x1="${gx}" y1="${padT}" x2="${gx}" y2="${padT + plotH}" stroke="rgba(255,255,255,0.03)"/>`;
    grid += `<text x="${gx}" y="${h - padB + 15}" text-anchor="middle" font-size="8.5" fill="#5f6879" font-family="var(--mono)">${(i / 4).toFixed(2)}</text>`;
  }
  // Prized quadrant shading (conviction>0.6, score>60)
  const qx = x(0.6), qy = y(60);
  const quad = `<rect x="${qx}" y="${padT}" width="${w - padR - qx}" height="${qy - padT}" fill="rgba(70,224,192,0.06)"/>`;
  let dots = "";
  companies.forEach((c) => {
    const cx = x(c.conviction_confidence || 0), cy = y(c.composite_score || 0);
    dots += `<circle cx="${cx}" cy="${cy}" r="5.5" fill="${scoreColor(c.composite_score)}" fill-opacity="0.85" stroke="#0a0c11" stroke-width="1.5"/>
      <text x="${cx}" y="${cy - 9}" text-anchor="middle" font-size="9" font-family="var(--mono)" fill="#9aa3b6">${c.ticker}</text>`;
  });
  return `<svg viewBox="0 0 ${w} ${h}" width="100%" xmlns="${SVGNS}">
    ${quad}${grid}
    <text x="${w - padR}" y="${padT + 12}" text-anchor="end" font-size="8.5" fill="#46e0c0" font-family="var(--font)" letter-spacing="0.5">HIGH SCORE · HIGH CONVICTION</text>
    ${dots}
    <text x="${padL + plotW / 2}" y="${h - 4}" text-anchor="middle" font-size="9" fill="#5f6879">Conviction confidence  →</text>
  </svg>`;
}

/* Small circular gauge for the drawer header composite. */
export function gauge(score, size = 68) {
  const r = size / 2 - 6, cx = size / 2, cy = size / 2;
  const circ = 2 * Math.PI * r;
  const frac = Math.max(0, Math.min(100, score)) / 100;
  const col = scoreColor(score);
  return `<svg viewBox="0 0 ${size} ${size}" width="${size}" height="${size}" xmlns="${SVGNS}">
    <circle cx="${cx}" cy="${cy}" r="${r}" fill="none" stroke="rgba(255,255,255,0.07)" stroke-width="5"/>
    <circle cx="${cx}" cy="${cy}" r="${r}" fill="none" stroke="${col}" stroke-width="5" stroke-linecap="round"
      stroke-dasharray="${circ}" stroke-dashoffset="${circ * (1 - frac)}" transform="rotate(-90 ${cx} ${cy})"/>
    <text x="${cx}" y="${cy + 4}" text-anchor="middle" font-size="16" font-weight="600" font-family="var(--mono)" fill="#e9ecf3">${Math.round(score)}</text>
  </svg>`;
}
