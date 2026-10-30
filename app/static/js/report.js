// One-page inspection report in a new window (print or save as PDF from there).
const esc = s => String(s ?? '').replace(/[&<>"']/g, c => ({ '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;', "'": '&#39;' }[c]));
const fmt = (x, d = 3) => x == null || !isFinite(x) ? '–' : Number(x).toFixed(d);

export function openReport({ st, check, shots, hist, LABEL, KINDS }) {
  const w = window.open('', '_blank');
  if (!w) { alert('Allow pop-ups for this site to open the report.'); return; }
  const s = st.deviation.stats || {};
  const tol = st.params.tolerance;
  const counts = st.summary.counts;
  const peak = Math.max(...hist.h, 1);
  const W = 520, H = 120, bw = W / hist.h.length;
  const bars = hist.h.map((v, i) => {
    const x0 = i / hist.h.length * hist.max;
    const col = x0 >= tol ? '#c9483e' : x0 >= tol / 2 ? '#e0a428' : '#2e9a6a';
    const bh = v ? Math.max(1.5, Math.sqrt(v / peak) * (H - 6)) : 0;
    return `<rect x="${(i * bw).toFixed(1)}" y="${(H - bh).toFixed(1)}" width="${(bw - 1.2).toFixed(1)}" height="${bh.toFixed(1)}" fill="${col}"/>`;
  }).join('');
  const faces = [...st.faces].sort((a, b) => (b.deviation?.max || 0) - (a.deviation?.max || 0));
  const worst = faces.filter(f => f.deviation).slice(0, 14);
  const closed = Object.values(check?.closed || {});
  const date = new Date().toLocaleString();
  const swatch = { plane: '#aebbc9', cylinder: '#3e8ede', cone: '#e9a23b', torus: '#8e6ad8', sphere: '#35b38a', thread: '#c2559b', freeform: '#d9c25a' };
  w.document.write(`<!doctype html><html><head><meta charset="utf-8"><title>Inspection report – ${esc(st.filename)}</title>
<style>
@font-face{font-family:Plex;src:url(${location.origin}/vendor/fonts/ibm-plex-sans-latin-400-normal.woff2)}
@font-face{font-family:Plex;font-weight:600;src:url(${location.origin}/vendor/fonts/ibm-plex-sans-latin-600-normal.woff2)}
@page{size:A4;margin:12mm}
*{box-sizing:border-box}body{font:12px/1.45 Plex,system-ui,sans-serif;color:#18202b;margin:0;padding:24px;font-variant-numeric:tabular-nums;max-width:820px;margin:auto}
header{display:flex;justify-content:space-between;align-items:flex-end;border-bottom:2px solid #18202b;padding-bottom:10px;margin-bottom:16px}
h1{font-size:20px;margin:0;font-weight:600}h2{font-size:13px;margin:18px 0 8px;font-weight:600}
.sub{color:#566271}.grid{display:grid;grid-template-columns:repeat(4,1fr);gap:10px}
.k{border:1px solid #d3d9df;border-radius:6px;padding:8px 10px}.k b{display:block;font-size:17px;font-weight:600}.k span{color:#566271;font-size:11px}
.bad b{color:#c9483e}.good b{color:#2e9a6a}
.shots{display:grid;grid-template-columns:1fr 1fr;gap:10px}.shots figure{margin:0;border:1px solid #d3d9df;border-radius:6px;overflow:hidden;background:linear-gradient(#f1f4f6,#d6dde3)}
.shots img{width:100%;display:block}.shots figcaption{padding:5px 8px;font-size:11px;color:#566271;background:#fff;border-top:1px solid #d3d9df}
table{width:100%;border-collapse:collapse}td,th{text-align:left;padding:4px 6px;border-bottom:1px solid #e3e7eb}th{font-weight:600;color:#566271;font-size:11px}
td.n,th.n{text-align:right}.sw{display:inline-block;width:10px;height:10px;border-radius:2px;margin-right:6px;vertical-align:-1px}
.cols{display:grid;grid-template-columns:1.1fr .9fr;gap:20px}
.print{position:fixed;right:16px;top:16px;padding:8px 14px;border:0;border-radius:7px;background:#18202b;color:#fff;font:inherit;cursor:pointer}
@media print{.print{display:none}body{padding:0}}
</style></head><body>
<button class="print" onclick="print()">Print or save as PDF</button>
<header><div><h1>Inspection report</h1><div class="sub">${esc(st.filename)}</div></div><div class="sub" style="text-align:right">${esc(date)}<br>Unfacet</div></header>
<div class="grid">
  <div class="k"><b>${fmt(tol, 3)} mm</b><span>Tolerance</span></div>
  <div class="k"><b>${st.summary.faces}</b><span>Faces (${check ? check.faces : '–'} in STEP)</span></div>
  <div class="k"><b>${fmt(st.summary.recognized_pct, 1)}%</b><span>Surface as true geometry</span></div>
  <div class="k ${closed.length && closed.every(Boolean) ? 'good' : 'bad'}"><b>${closed.filter(Boolean).length}/${closed.length}</b><span>Bodies closed as solids</span></div>
</div>
<h2>Deviation of mesh vertices from the STEP faces</h2>
<div class="grid">
  <div class="k ${s.max > tol ? 'bad' : 'good'}"><b>${fmt(s.max)}</b><span>Maximum, mm</span></div>
  <div class="k"><b>${fmt(s.p95)}</b><span>95th percentile, mm</span></div>
  <div class="k"><b>${fmt(s.mean)}</b><span>Mean, mm</span></div>
  <div class="k ${s.over > 0 ? 'bad' : 'good'}"><b>${fmt(s.over, 2)}%</b><span>Above tolerance</span></div>
</div>
<svg viewBox="0 0 ${W} ${H + 16}" style="width:100%;margin-top:10px">${bars}<line x1="${W / 2}" x2="${W / 2}" y1="0" y2="${H}" stroke="#c9483e" stroke-dasharray="3 3"/>
<text x="0" y="${H + 13}" font-size="10" fill="#566271">0 mm</text><text x="${W / 2 + 4}" y="${H + 13}" font-size="10" fill="#c9483e">tolerance ${fmt(tol)}</text><text x="${W - 48}" y="${H + 13}" font-size="10" fill="#566271">${fmt(hist.max)} mm</text></svg>
<h2>Views</h2>
<div class="shots"><figure><img src="${shots.faces}" alt="Face kinds"><figcaption>Face kinds</figcaption></figure><figure><img src="${shots.deviation}" alt="Deviation"><figcaption>Deviation: green fits, yellow at tolerance, red above</figcaption></figure></div>
<div class="cols">
<div><h2>Largest deviations</h2><table><tr><th>Face</th><th class="n">Max</th><th class="n">95th</th><th class="n">Mean</th></tr>
${worst.map(f => `<tr><td><span class="sw" style="background:${swatch[f.kind] || '#999'}"></span>${LABEL[f.kind]} ${f.id}${f.hole ? `, hole Ø${fmt(f.hole.diameter, 2)}` : ''}</td><td class="n">${fmt(f.deviation.max)}</td><td class="n">${fmt(f.deviation.p95)}</td><td class="n">${fmt(f.deviation.mean)}</td></tr>`).join('')}</table></div>
<div><h2>Face kinds</h2><table><tr><th>Kind</th><th class="n">Faces</th></tr>
${KINDS.filter(k => counts[k]).map(k => `<tr><td><span class="sw" style="background:${swatch[k]}"></span>${LABEL[k]}</td><td class="n">${counts[k]}</td></tr>`).join('')}</table>
${check ? `<h2>STEP check</h2><table><tr><td>Suspect faces</td><td class="n">${check.invalid.length}</td></tr><tr><td>Faces written as facets</td><td class="n">${check.discarded.length}</td></tr>${check.messages.map(m => `<tr><td colspan="2">${esc(m)}</td></tr>`).join('')}</table>` : ''}
</div></div>
</body></html>`);
  w.document.close();
}
