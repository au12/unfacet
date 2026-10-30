// Unfacet web app: state, panels, tools.
import { Viewer } from './viewer.js';
import { api, b64ToArray } from './api.js';
import { openReport } from './report.js';

const $ = s => document.querySelector(s);
const esc = s => String(s ?? '').replace(/[&<>"']/g, c => ({ '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;', "'": '&#39;' }[c]));
const KINDS = ['plane', 'cylinder', 'cone', 'torus', 'sphere', 'thread', 'freeform'];
const LABEL = { plane: 'Plane', cylinder: 'Cylinder', cone: 'Cone', torus: 'Torus', sphere: 'Sphere', thread: 'Thread', freeform: 'Freeform', discarded: 'Faceted fallback' };
const fmt = (x, d = 2) => x == null || !isFinite(x) ? '–' : Number(x).toFixed(d);
const mm = (x, d = 2) => `${fmt(x, d)} mm`;
const plural = (n, a, b) => `${n} ${n === 1 ? a : (b || a + 's')}`;

const S = {
  sid: null, st: null, meta: null, mesh: null, labels: null, faces: new Map(), tris: new Map(), dev: null,
  view: 'faces', sel: null, hover: null, highlight: null, check: null, checkVersion: -1,
  wire: false, section: { on: false, axis: 'z', t: 0.5, flip: false }, tool: null, trace: null,
  busy: false, colors: {}, openTool: null, threadMode: 'cylinder',
};

// ------------------------------------------------------------------ colours
function readColors() {
  const cs = getComputedStyle(document.documentElement);
  const rgb = v => { const h = cs.getPropertyValue(v).trim().replace('#', ''); return [0, 2, 4].map(i => parseInt(h.slice(i, i + 2), 16) / 255); };
  S.colors = Object.fromEntries([...KINDS, 'discarded'].map(k => [k, rgb(`--k-${k}`)]));
  S.colors.neutral = document.documentElement.dataset.theme === 'dark' ? [0.42, 0.46, 0.5] : [0.74, 0.77, 0.8];
  S.colors.focus = rgb('--focus');
  S.colors.hi = document.documentElement.dataset.theme === 'dark' ? [1, 1, 1] : [1, 0.98, 0.9];
}
const mix = (a, b, t) => [a[0] + (b[0] - a[0]) * t, a[1] + (b[1] - a[1]) * t, a[2] + (b[2] - a[2]) * t];
function devColor(d, tol) {
  // green fits, yellow at tolerance, red above
  const x = d / tol;
  const stops = [[0, [0.18, 0.6, 0.42]], [0.5, [0.55, 0.76, 0.32]], [1, [0.94, 0.76, 0.23]], [1.6, [0.89, 0.46, 0.23]], [2.5, [0.79, 0.28, 0.24]]];
  if (x <= 0) return stops[0][1];
  for (let i = 1; i < stops.length; i++) if (x <= stops[i][0]) {
    const [x0, c0] = stops[i - 1], [x1, c1] = stops[i];
    return mix(c0, c1, (x - x0) / (x1 - x0));
  }
  return stops[stops.length - 1][1];
}

// ------------------------------------------------------------------ viewer
const viewer = new Viewer($('#viewport'), { onHover, onClick });

function faceColor(f) {
  if (!f) return S.colors.neutral;
  return S.colors[f.kind] || S.colors.neutral;
}

function paint() {
  if (!S.mesh) return;
  const n = S.mesh.indices.length / 3;
  const L = S.labels;
  const view = S.view;
  viewer.setOpacity(view === 'cavities' ? 0.16 : 1);
  viewer.showPreview(view === 'step' && !!S.check?.preview);
  $('#devScale').hidden = view !== 'deviation' || !S.st?.analyzed;
  if (view === 'deviation' && S.st?.analyzed) {
    const tol = S.st.params.tolerance;
    const C = new Float32Array(n * 9);
    for (let t = 0; t < n; t++) {
      const f = S.faces.get(L[t]);
      for (let k = 0; k < 3; k++) {
        const c = f && f.model ? devColor(S.dev[t * 3 + k], tol) : S.colors.neutral;
        C.set(c, t * 9 + k * 3);
      }
    }
    if (S.sel != null) for (const t of S.tris.get(S.sel) || []) for (let k = 0; k < 3; k++) C.set(mix(C.subarray(t * 9 + k * 3, t * 9 + k * 3 + 3), S.colors.focus, 0.25), t * 9 + k * 3);
    viewer.setCornerColors(C);
  } else {
    const C = new Float32Array(n * 3);
    const bad = new Set(S.check && S.checkVersion === S.st?.version ? S.check.invalid : []);
    const fb = new Set(S.check && S.checkVersion === S.st?.version ? S.check.discarded : []);
    const haveCheck = S.check && S.checkVersion === S.st?.version;
    const cache = new Map();
    for (let t = 0; t < n; t++) {
      const id = L ? L[t] : -1;
      let c = cache.get(id);
      if (!c) {
        const f = S.faces.get(id);
        if (view === 'validity') {
          c = !haveCheck ? S.colors.neutral : bad.has(id) ? S.colors.discarded : fb.has(id) ? S.colors.cone : mix(S.colors.neutral, [0.18, 0.6, 0.42], 0.45);
        } else if (view === 'cavities') {
          c = S.colors.neutral;
        } else {
          c = S.st?.analyzed ? faceColor(f) : S.colors.neutral;
        }
        if (S.highlight?.has(id)) c = mix(c, S.colors.hi, 0.6);
        if (id === S.sel) c = mix(c, S.colors.focus, 0.3);
        else if (id === S.hover) c = mix(c, [1, 1, 1], 0.28);
        cache.set(id, c);
      }
      C[t * 3] = c[0]; C[t * 3 + 1] = c[1]; C[t * 3 + 2] = c[2];
    }
    viewer.setTriColors(C);
  }
  // cavities overlay
  if (view === 'cavities' && S.st) {
    const cav = new Set(S.st.cavities.map(c => c.id));
    const tris = [];
    for (let t = 0; t < n; t++) if (cav.has(S.mesh.bodies[t])) tris.push(t);
    viewer.setOverlayTris(tris, 0xc9483e, 1);
  } else viewer.setOverlayTris(null);
}

function outline() {
  if (!S.mesh) return;
  if (S.st?.analyzed) viewer.setBoundaries(S.labels, S.sel);
  else viewer.setBoundaries(new Int32Array(S.mesh.indices.length / 3), null);
}

// ------------------------------------------------------------------ state
function applyState(st, { keepSel = true } = {}) {
  const meshChanged = !!st.mesh;
  if (st.mesh) {
    S.mesh = {
      positions: b64ToArray(st.mesh.positions, Float32Array),
      indices: b64ToArray(st.mesh.indices, Uint32Array),
      bodies: b64ToArray(st.mesh.bodies, Int32Array),
      open: b64ToArray(st.mesh.open_edges, Uint32Array),
      triangles: st.mesh.triangles,
    };
    viewer.setMesh(S.mesh.positions, S.mesh.indices);
    viewer.setOpenEdges(S.mesh.open);
    viewer.setWire(S.wire);
    $('#empty').hidden = true;
    if (S.section.on) viewer.setSection(true, S.section.axis, S.section.t, S.section.flip);
  }
  S.st = st;
  S.labels = b64ToArray(st.labels, Int32Array);
  S.dev = b64ToArray(st.deviation.corners, Float32Array);
  S.faces = new Map(st.faces.map(f => [f.id, f]));
  S.tris = new Map();
  const L = S.labels;
  const buckets = new Map();
  for (let t = 0; t < L.length; t++) { let b = buckets.get(L[t]); if (!b) buckets.set(L[t], b = []); b.push(t); }
  for (const [k, v] of buckets) S.tris.set(k, v);
  if (st.select != null && S.faces.has(st.select)) S.sel = st.select;
  if (!keepSel || (S.sel != null && !S.faces.has(S.sel))) S.sel = null;
  if (S.hover != null && !S.faces.has(S.hover)) S.hover = null;
  S.highlight = null;
  if (st.check) { S.check = st.check; S.checkVersion = st.version; }
  if (st.preview) setPreview(st.preview);
  if (meshChanged && S.check && S.checkVersion !== st.version) S.check = null;
  renderAll();
  if (st.message) toast(st.message);
}

function setPreview(pv) {
  const pos = b64ToArray(pv.positions, Float32Array), idx = b64ToArray(pv.indices, Uint32Array), reg = b64ToArray(pv.regions, Int32Array);
  const C = new Float32Array(reg.length * 3);
  for (let t = 0; t < reg.length; t++) C.set(faceColor(S.faces.get(reg[t])), t * 3);
  viewer.setPreview(pos, idx, C);
  if (S.check) S.check.preview = true;
}

function renderAll() {
  const st = S.st;
  $('#fileMeta').hidden = !st;
  if (st) {
    $('#fileName').textContent = st.filename;
    const nb = st.bodies.length;
    $('#fileSub').textContent = `${S.mesh.triangles.toLocaleString()} triangles, ${plural(nb, 'body', 'bodies')}`;
    document.title = `${st.filename} – Unfacet`;
  }
  $('#analyzeBtn').disabled = !st || S.busy;
  $('#analyzeBtn').textContent = st?.analyzed ? 'Analyse again' : 'Analyse mesh';
  $('#exportBtn').disabled = $('#exportMore').disabled = !st?.analyzed || S.busy;
  $('#undoBtn').disabled = !st?.history.undo || S.busy;
  $('#redoBtn').disabled = !st?.history.redo || S.busy;
  $('#undoBtn').title = st?.history.undo ? `Undo ${st.history.undo} (Ctrl+Z)` : 'Undo (Ctrl+Z)';
  $('#redoBtn').title = st?.history.redo ? `Redo ${st.history.redo} (Ctrl+Shift+Z)` : 'Redo (Ctrl+Shift+Z)';
  if (st) {
    $('#stlLink').href = `/api/sessions/${S.sid}/mesh.stl`;
    for (const [id, k] of [['#tolInput', 'tolerance'], ['#angInput', 'angle'], ['#minInput', 'min_face_pct']]) {
      if (document.activeElement !== $(id)) $(id).value = st.params[k];
    }
  }
  renderLegend();
  renderComposition();
  renderBodies();
  renderNotices();
  renderInspector();
  paint();
  outline();
}

// ------------------------------------------------------------------ left rail
function renderLegend() {
  const st = S.st;
  const enabled = new Set(st ? st.params.types : KINDS.slice(0, 6));
  const counts = st?.summary.counts || {};
  const rows = KINDS.map(k => {
    const n = counts[k] || 0;
    const box = k === 'freeform'
      ? `<span></span>`
      : `<input type="checkbox" data-kind="${k}" ${enabled.has(k) ? 'checked' : ''} aria-label="Recognise ${LABEL[k]}">`;
    return `<label class="kind-row ${k === 'freeform' ? 'static' : ''}" data-k="${k}">${box}<span class="sw" style="--c:var(--k-${k})"></span><span>${LABEL[k]}</span><span class="n">${st?.analyzed ? n : ''}</span></label>`;
  });
  const disc = S.check && S.checkVersion === st?.version ? S.check.discarded.length : 0;
  if (disc) rows.push(`<div class="kind-row static"><span></span><span class="sw hatch" style="--c:var(--k-discarded)"></span><span>Faceted fallback</span><span class="n">${disc}</span></div>`);
  $('#legend').innerHTML = rows.join('');
}

function renderComposition() {
  const st = S.st, el = $('#composition');
  if (!st?.analyzed) { el.hidden = true; return; }
  const area = {};
  let tot = 0;
  for (const f of st.faces) { area[f.kind] = (area[f.kind] || 0) + f.area; tot += f.area; }
  el.hidden = false;
  el.innerHTML = `<div class="comp-bar">${KINDS.filter(k => area[k]).map(k => `<span style="--c:var(--k-${k});flex:${area[k] / tot}" title="${LABEL[k]}: ${fmt(100 * area[k] / tot, 1)}% of the surface"></span>`).join('')}</div>
    <div class="comp-cap"><span><b>${fmt(st.summary.recognized_pct, 1)}%</b> of the surface rebuilt as true geometry in ${plural(st.summary.faces, 'face')}</span></div>`;
}

function renderBodies() {
  const st = S.st;
  const show = st && st.bodies.length > 1;
  $('#bodiesBlock').hidden = !show;
  if (!show) return;
  $('#bodyCount').textContent = st.bodies.length;
  $('#bodyList').innerHTML = st.bodies.map(b => `
    <li data-body="${b.id}">
      <input type="checkbox" ${b.selected ? 'checked' : ''} aria-label="Include ${esc(b.name)}">
      <span class="bname" title="Double-click to rename">${esc(b.name)}</span>
      <span class="bmeta">${b.size.map(x => fmt(x, 1)).join(' × ')} mm, ${b.triangles.toLocaleString()} triangles${b.cavities.length ? `, ${plural(b.cavities.length, 'cavity', 'cavities')}` : ''}</span>
    </li>`).join('');
}

function renderNotices() {
  const st = S.st;
  const gaps = st?.summary.gaps || 0;
  $('#gapNotice').hidden = !gaps;
  $('#gapCount').textContent = gaps ? `${plural(gaps, 'gap')}, ${st.summary.open_edges} open edges.` : '';
  const cav = st?.cavities || [];
  $('#cavNotice').hidden = !cav.length;
  if (cav.length) $('#cavText').textContent = `${plural(cav.length, 'enclosed cavity', 'enclosed cavities')} (${fmt(cav.reduce((a, c) => a + c.volume, 0), 1)} mm³). They become hollow spaces inside the solid.`;
}

// ------------------------------------------------------------------ inspector
const ICONS = {
  hole: '<circle cx="9" cy="9" r="6.5"/><circle cx="9" cy="9" r="3"/>',
  merge: '<path d="M3 4h5v10H3zM10 4h5v10h-5z"/><path d="M8 9h2"/>',
  tidy: '<path d="M2.5 12.5 7 4l3 5 2-3 3.5 6.5Z"/><path d="M2.5 15.5h13"/>',
  extend: '<path d="M2 9h4M12 9h4M14 7l2 2-2 2M4 7 2 9l2 2"/><rect x="6" y="5" width="6" height="8" rx="1"/>',
  again: '<path d="M14.5 8A5.5 5.5 0 1 0 13 12.5"/><path d="M15 3.5v4.5h-4.5"/>',
  refit: '<path d="M3 13c3-8 9-8 12 0"/><path d="M3 15h12" stroke-dasharray="2 2"/>',
  cut: '<circle cx="5" cy="13" r="2"/><circle cx="13" cy="13" r="2"/><path d="m6.5 11.5 7-8M11.5 11.5l-7-8"/>',
  fix: '<path d="M10.5 3.5a3.5 3.5 0 0 0-4 4.7L2.5 12.2 5.8 15.5l4-4a3.5 3.5 0 0 0 4.7-4L12 10l-2.5-.5L9 7Z"/>',
};
function tool(id, title, body, { open = false, alert = false, badge = '' } = {}) {
  const isOpen = S.openTool != null ? S.openTool === id : open;
  return `<details class="tool ${alert ? 'alert' : ''}" data-tool="${id}" ${isOpen ? 'open' : ''}>
    <summary><svg class="ti" viewBox="0 0 18 18">${ICONS[id] || ''}</svg>${title}${badge ? `<span class="badge">${badge}</span>` : ''}<svg class="chev" viewBox="0 0 12 12"><path d="m3 4.5 3 3 3-3"/></svg></summary>
    <div class="body">${body}</div></details>`;
}

function paramFacts(f) {
  const p = f.params || {};
  const out = [];
  if (f.kind === 'cylinder' || f.kind === 'sphere') out.push(['Diameter', mm(p.diameter, 3)]);
  if (f.kind === 'cone') out.push(['Half angle', `${fmt(p.half_angle_deg, 2)}°`]);
  if (f.kind === 'torus') out.push(['Major radius', mm(p.major_radius, 3)], ['Minor radius', mm(p.minor_radius, 3)]);
  if (f.kind === 'thread') out.push(['Size', p.designation || '–'], ['Pitch', mm(p.pitch, 3)], ['Major Ø', mm(p.major_diameter, 3)], ['Minor Ø', mm(p.minor_diameter, 3)]);
  if (f.hole) out.push(['Depth', mm(f.hole.depth, 2)]);
  out.push(['Area', `${fmt(f.area, 1)} mm²`], ['Triangles', f.triangles.toLocaleString()]);
  return out;
}

function renderInspector() {
  const el = $('#inspectorBody');
  const st = S.st;
  if (!st) { el.innerHTML = `<div class="empty-insp">Open a mesh to begin.</div>`; return; }
  if (S.sel == null || !S.faces.has(S.sel)) { el.innerHTML = overviewHTML(); bindOverview(); return; }
  const f = S.faces.get(S.sel);
  const tol = st.params.tolerance;
  const d = f.deviation;
  const facts = paramFacts(f).map(([l, v]) => `<div class="fact"><div class="v">${esc(v)}</div><div class="l">${l}</div></div>`).join('');
  const devFact = d ? `<div class="fact ${d.max > tol ? 'bad' : 'good'}"><div class="v">${fmt(d.max, 3)}<small>mm</small></div><div class="l">Max deviation</div></div>` : '';
  const checked = S.check && S.checkVersion === st.version;
  const flagged = checked && (S.check.invalid.includes(f.id) || S.check.discarded.includes(f.id));
  const kindLabel = f.kind === 'thread' && f.params.designation ? `Thread ${f.params.designation}` : (f.hole ? 'Hole' : LABEL[f.kind]);
  let html = `<div class="insp-head">
      <div class="kind"><span class="sw" style="--c:var(--k-${f.kind})"></span><div><h2>${esc(f.name || kindLabel)}</h2>
      <div class="sub">${LABEL[f.kind]} face ${f.id}${f.hole?.setting?.thread ? `, ${f.hole.setting.thread.designation} thread is cut on export` : f.hole?.fits ? `, fits ${f.hole.fits.size} ${purposeName(f.hole.fits.purpose).toLowerCase()}` : ''}</div></div>
      <button class="icon-btn close" id="deselect" title="Clear selection (Esc)" aria-label="Clear selection"><svg viewBox="0 0 20 20"><path d="m5 5 10 10M15 5 5 15"/></svg></button></div>
      <div class="facts">${facts}${devFact}</div></div>`;
  if (flagged) html += tool('fix', 'Fix check finding', `<p>${S.check.invalid.includes(f.id) ? 'A CAD program would likely reject this face.' : 'This face could not be rebuilt as true geometry, so the STEP carries it as flat facets.'} Unfacet tries merging it into each neighbour or turning it into facets, and keeps whichever leaves fewer findings.</p><button class="btn primary small" data-act="fix">Fix check finding</button>`, { open: true, alert: true });
  if (f.kind === 'thread') {
    const p = f.params;
    html += `<div class="block" style="border-bottom:1px solid var(--line)"><p class="hint" style="margin:0">${S.threadMode === 'helical'
      ? `Exported as a real ${esc(p.designation || 'ISO')} thread with helical flanks.`
      : `Exported as a plain Ø${fmt(p.internal ? p.minor_diameter : p.major_diameter, 2)} mm cylinder. Choose Real helical thread in the Download STEP menu to cut the flanks.`}</p></div>`;
  }
  if (f.hole) html += tool('hole', 'Hole size', holeHTML(f), { open: !flagged });
  html += tool('merge', 'Merge or convert', `<div id="mergeBody"><p>Loading…</p></div>`, { open: !f.hole && !flagged && (f.kind === 'freeform') });
  html += tool('tidy', 'Tidy up area', `<p>Replace a patchwork of small faces around this one with a single face.</p><div id="tidyBody"><button class="btn small" data-act="tidy-find">Find variants</button></div>`);
  if (f.model) html += tool('extend', 'Extend', `<p>Grow this face over neighbouring triangles that lie on it.</p>
      <div class="range"><input type="range" id="extDev" min="0.1" max="1" step="0.05" value="0.2"><output id="extDevOut">0.20 mm</output></div>
      <div class="sublabel">Direction</div>
      <select class="sel" id="extDir" style="width:100%"><option value="">In all directions</option></select>
      <div class="row end"><button class="btn small primary" data-act="extend">Extend</button></div>`);
  html += tool('again', 'Recognise again', `<p>Rerun recognition on this face only. The result is kept only if it is simpler.</p>
      <div class="fields">
        <label class="field"><span>Tolerance</span><span class="input-unit"><input type="number" id="raTol" value="${Math.max(tol * 2.5, 0.05)}" step="0.01" min="0.001"><i>mm</i></span></label>
        <label class="field"><span>Normal angle</span><span class="input-unit"><input type="number" id="raAng" value="${st.params.angle}" step="1" min="5" max="60"><i>°</i></span></label>
      </div>
      <div class="grid-checks">${KINDS.slice(0, 6).map(k => `<label class="check"><input type="checkbox" class="raKind" value="${k}" checked>${LABEL[k]}</label>`).join('')}</div>
      <label class="check"><input type="checkbox" id="raRing"> Include the ring of neighbouring faces</label>
      <div class="row end"><button class="btn small primary" data-act="again">Recognise again</button></div>`);
  if (['plane', 'cylinder', 'cone', 'sphere', 'torus'].includes(f.kind)) {
    const off = d && d.max > tol;
    html += tool('refit', 'Fit to original', `<p>${off ? `This face deviates up to ${mm(d.max, 3)} from the mesh. ` : ''}Refit the ${LABEL[f.kind].toLowerCase()} to the original triangles${f.hole?.setting ? ', which also undoes the hole size you set' : ''}.</p><button class="btn small" data-act="refit">Fit to original</button>`, { badge: off ? 'deviates' : '' });
  }
  html += tool('cut', 'Cut along edges', `<p>Trace a seam the analysis missed. Click points on this face; the cut follows mesh edges between them. Each part gets the simplest shape that fits it.</p><button class="btn small" data-act="trace">Start tracing</button>`);
  el.innerHTML = html;
  bindFace(f);
}

function purposeName(p) { return { clearance: 'Clearance', tap: 'Tap drill', plastic: 'Plastic', thread: 'Thread' }[p] || p; }

function holeHTML(f) {
  const h = f.hole;
  const iso = S.meta.iso_holes;
  const cur = S.holePick?.face === f.id ? S.holePick : { face: f.id, purpose: h.setting?.thread ? 'thread' : (h.setting?.purpose || h.fits?.purpose || 'clearance'), size: h.setting?.size || h.fits?.size || null };
  S.holePick = cur;
  const row = iso.find(r => r.size === cur.size);
  let target = '';
  if (row) {
    target = cur.purpose === 'thread'
      ? `<div class="target"><b>${row.size}×${row.pitch}</b><span>Ø${fmt(row.nominal - 1.0825 * row.pitch, 2)} core, ISO thread cut</span></div>`
      : `<div class="target"><b>Ø${fmt(row[cur.purpose], 2)} mm</b><span>${row.size} ${purposeName(cur.purpose).toLowerCase()}</span></div>`;
  }
  const same = f.hole && S.identicalHoles?.face === f.id ? S.identicalHoles.ids.length : 0;
  return `<p>Ø${fmt(h.diameter, 3)} mm, ${fmt(h.depth, 2)} mm deep${h.setting ? ` (originally Ø${fmt(h.original_diameter, 3)})` : ''}.</p>
    <div class="seg full" id="holePurpose">${['clearance', 'tap', 'plastic', 'thread'].map(p => `<button data-p="${p}" aria-pressed="${cur.purpose === p}">${purposeName(p)}</button>`).join('')}</div>
    <div class="size-grid">${iso.map(r => `<button data-size="${r.size}" class="${r.size === cur.size ? 'on' : ''} ${h.fits?.size === r.size ? 'fits' : ''}">${r.size}</button>`).join('')}</div>
    ${target}
    <div class="row"><span class="input-unit" style="width:120px"><input type="number" id="holeFree" placeholder="Free Ø" step="0.05" min="0.2"><i>mm</i></span><button class="btn small" data-act="hole-free">Use diameter</button></div>
    <label class="check" style="margin-top:10px"><input type="checkbox" id="holeAll" ${same ? '' : 'disabled'}> Change all ${same ? plural(same + 1, 'identical hole') : 'identical holes (none found)'}</label>
    <div class="row end"><button class="btn small primary" data-act="hole-apply" ${row ? '' : 'disabled'}>Set hole size</button></div>`;
}

function overviewHTML() {
  const st = S.st;
  if (!st.analyzed) {
    return `<div class="overview"><section><h3>Next step</h3><p class="hint" style="margin:0">Set the tolerance (start with 0.02 mm for CAD exports, 0.05–0.1 mm for coarse meshes) and press Analyse mesh. Then click any face to edit it.</p></section></div>`;
  }
  const s = st.deviation.stats || {};
  const tol = st.params.tolerance;
  const area = {}, count = st.summary.counts;
  let tot = 0;
  for (const f of st.faces) { area[f.kind] = (area[f.kind] || 0) + f.area; tot += f.area; }
  const rows = KINDS.filter(k => count[k]).map(k => `<tr><td><span class="sw" style="--c:var(--k-${k})"></span>${LABEL[k]}</td><td>${count[k]}</td><td>${fmt(100 * area[k] / tot, 1)}%</td></tr>`).join('');
  const checked = S.check && S.checkVersion === st.version;
  let chk = `<p class="hint" style="margin:0 0 10px">Builds the STEP and checks every face the way a CAD program will.</p>`;
  if (checked) {
    const c = S.check;
    const closed = Object.values(c.closed);
    const solids = closed.filter(Boolean).length;
    chk = `<div class="result-line"><span class="dot ${solids === closed.length ? 'good' : 'bad'}"></span>${solids === closed.length ? `${plural(solids, 'closed solid')}` : `${closed.length - solids} of ${plural(closed.length, 'body', 'bodies')} did not close into a solid`}</div>
      <div class="result-line"><span class="dot ${c.invalid.length ? 'bad' : 'good'}"></span>${c.invalid.length ? `${plural(c.invalid.length, 'suspect face')}` : 'No suspect faces'}</div>
      <div class="result-line"><span class="dot ${c.discarded.length ? 'warn' : 'good'}"></span>${c.discarded.length ? `${plural(c.discarded.length, 'face')} written as flat facets` : 'Every face is true geometry'}</div>
      ${c.messages.map(m => `<div class="result-line"><span class="dot"></span>${esc(m)}</div>`).join('')}
      <p class="hint" style="margin:8px 0 10px">${c.faces} faces in the STEP. ${c.invalid.length + c.discarded.length ? 'Open the Validity view and click a red or orange face to fix it.' : 'See it in the Result view.'}</p>`;
  }
  return `<div class="overview">
    <section><h3>Faces</h3><table class="kinds-table">${rows}</table></section>
    <section><h3>Deviation from the mesh</h3>
      <div class="stats">
        <div class="fact ${s.max > tol ? 'bad' : ''}"><div class="v">${fmt(s.max, 3)}</div><div class="l">Max mm</div></div>
        <div class="fact"><div class="v">${fmt(s.p95, 3)}</div><div class="l">95th pct</div></div>
        <div class="fact"><div class="v">${fmt(s.mean, 3)}</div><div class="l">Mean mm</div></div>
        <div class="fact ${s.over > 0 ? 'bad' : 'good'}"><div class="v">${fmt(s.over, 1)}%</div><div class="l">Above tol.</div></div>
      </div>${histSVG()}
    </section>
    <section><h3>STEP check</h3>${chk}
      <div class="row"><button class="btn small" data-act="check">${checked ? 'Check again' : 'Check STEP'}</button><button class="btn small" data-act="report">Inspection report</button></div>
    </section>
  </div>`;
}

export function histogram(dev, labels, faces, tol, bins = 40) {
  const max = Math.max(tol * 2, 1e-9);
  const h = new Array(bins).fill(0);
  let n = 0;
  for (let t = 0; t < labels.length; t++) {
    const f = faces.get(labels[t]);
    if (!f || !f.model) continue;
    for (let k = 0; k < 3; k++) { const d = dev[t * 3 + k]; h[Math.min(bins - 1, Math.floor(d / max * bins))]++; n++; }
  }
  return { h, max, n };
}

function histSVG() {
  const tol = S.st.params.tolerance;
  const { h, max } = histogram(S.dev, S.labels, S.faces, tol);
  const peak = Math.max(...h, 1);
  const W = 280, H = 60, bw = W / h.length;
  const bars = h.map((v, i) => {
    const x0 = i / h.length * max;
    const cls = x0 >= tol ? 'over' : x0 >= tol * 0.5 ? 'mid' : '';
    const bh = v ? Math.max(1.5, Math.sqrt(v / peak) * (H - 4)) : 0;
    return `<rect class="${cls}" x="${(i * bw).toFixed(1)}" y="${(H - bh).toFixed(1)}" width="${(bw - 1).toFixed(1)}" height="${bh.toFixed(1)}"><title>${fmt(x0, 3)}–${fmt(x0 + max / h.length, 3)} mm: ${v}</title></rect>`;
  }).join('');
  const tx = W / 2;
  return `<svg class="hist" viewBox="0 0 ${W} ${H + 12}" role="img" aria-label="Deviation distribution">${bars}<line x1="${tx}" x2="${tx}" y1="0" y2="${H}"/><text x="0" y="${H + 11}" font-size="9" fill="currentColor" opacity=".6">0</text><text x="${tx - 22}" y="${H + 11}" font-size="9" fill="currentColor" opacity=".6">tol ${fmt(tol, 3)}</text><text x="${W - 34}" y="${H + 11}" font-size="9" fill="currentColor" opacity=".6">${fmt(max, 3)}</text></svg>`;
}

function bindOverview() {
  $('#inspectorBody').querySelector('[data-act=check]')?.addEventListener('click', runCheck);
  $('#inspectorBody').querySelector('[data-act=report]')?.addEventListener('click', report);
}

function bindFace(f) {
  const el = $('#inspectorBody');
  el.querySelector('#deselect').addEventListener('click', () => select(null));
  el.querySelectorAll('details.tool > summary').forEach(sm => sm.addEventListener('click', e => {
    e.preventDefault();
    const d = sm.parentElement;
    const open = !d.open;
    el.querySelectorAll('details.tool').forEach(o => { o.open = false; });
    d.open = open;
    S.openTool = open ? d.dataset.tool : '';
    if (open && d.dataset.tool === 'merge') loadOptions(f);
    if (open && d.dataset.tool === 'extend') loadExtendDirs(f);
  }));
  if (el.querySelector('details[data-tool=merge]')?.open) loadOptions(f);
  if (el.querySelector('details[data-tool=extend]')?.open) loadExtendDirs(f);
  const on = (act, fn) => el.querySelector(`[data-act="${act}"]`)?.addEventListener('click', fn);
  on('fix', () => runJobOp(`/sessions/${S.sid}/faces/${f.id}/fix`, {}, 'Trying repairs'));
  on('refit', () => faceOp('refit', {}));
  on('trace', () => startTrace(f));
  on('tidy-find', () => loadTidy(f));
  on('extend', () => {
    const dir = el.querySelector('#extDir').value;
    faceOp('extend', { deviation: +el.querySelector('#extDev').value, direction: dir === '' ? null : +dir });
  });
  el.querySelector('#extDev')?.addEventListener('input', e => { el.querySelector('#extDevOut').textContent = mm(+e.target.value); });
  on('again', () => faceOp('recognize', {
    tolerance: +el.querySelector('#raTol').value, angle: +el.querySelector('#raAng').value,
    types: [...el.querySelectorAll('.raKind:checked')].map(i => i.value), include_ring: el.querySelector('#raRing').checked,
  }));
  if (f.hole) bindHole(f);
}

function bindHole(f) {
  const el = $('#inspectorBody');
  if (S.identicalHoles?.face !== f.id) {
    S.identicalHoles = { face: f.id, ids: [] };
    api.get(`/sessions/${S.sid}/faces/${f.id}/options`).then(o => {
      S.identicalHoles = { face: f.id, ids: o.identical_holes };
      if (S.sel === f.id && o.identical_holes.length) rerenderTool('hole', holeHTML(f), () => bindHole(f));
    }).catch(() => {});
  }
  el.querySelectorAll('#holePurpose button').forEach(b => b.addEventListener('click', () => {
    S.holePick.purpose = b.dataset.p; rerenderTool('hole', holeHTML(f), () => bindHole(f));
  }));
  el.querySelectorAll('.size-grid button').forEach(b => b.addEventListener('click', () => {
    S.holePick.size = b.dataset.size; rerenderTool('hole', holeHTML(f), () => bindHole(f));
  }));
  const all = () => el.querySelector('#holeAll')?.checked || false;
  el.querySelector('[data-act=hole-apply]')?.addEventListener('click', () => {
    const p = S.holePick, row = S.meta.iso_holes.find(r => r.size === p.size);
    if (!row) return;
    if (p.purpose === 'thread') faceOp('hole', { apply_all: all(), size: row.size, purpose: 'thread', thread: { nominal: row.nominal, pitch: row.pitch } });
    else faceOp('hole', { diameter: row[p.purpose], apply_all: all(), size: row.size, purpose: p.purpose });
  });
  el.querySelector('[data-act=hole-free]')?.addEventListener('click', () => {
    const d = +el.querySelector('#holeFree').value;
    if (!(d > 0)) { toast('Enter a diameter first.', true); return; }
    faceOp('hole', { diameter: d, apply_all: all() });
  });
}

function rerenderTool(id, html, bind) {
  const body = $(`#inspectorBody details[data-tool=${id}] .body`);
  if (body) { body.innerHTML = html; bind(); }
}

async function loadOptions(f) {
  const box = $('#mergeBody');
  if (!box) return;
  try {
    const o = await api.get(`/sessions/${S.sid}/faces/${f.id}/options`);
    if (S.sel !== f.id || !$('#mergeBody')) return;
    const tol = S.st.params.tolerance;
    const dv = d => d == null ? '<span class="dv">no fit</span>' : `<span class="dv ${d <= tol ? 'ok' : d > tol * 5 ? 'hi' : ''}">${fmt(d, 3)} mm</span>`;
    const mode = S.mergeMode || 'flush';
    $('#mergeBody').innerHTML = `
      ${o.neighbors.length ? `<div class="sublabel">Merge into neighbour, deviation if merged</div>
      <div class="opts">${o.neighbors.map(n => `<button class="opt" data-into="${n.id}"><span class="sw" style="--c:var(--k-${n.kind})"></span><span>${LABEL[n.kind]} ${n.id}<span class="sub">${n.shared_edges} shared edges, ${fmt(n.angle_deg, 0)}° crease</span></span>${dv(n.deviation)}</button>`).join('')}</div>
      <div class="row"><span class="hint" style="margin:0">Border</span><div class="seg" id="mergeMode"><button data-m="flush" aria-pressed="${mode === 'flush'}" title="Keep the neighbour's surface; only the shared border straightens">Flush</button><button data-m="tangent" aria-pressed="${mode === 'tangent'}" title="Refit the neighbour over both, so the face runs smoothly into it">Tangent</button></div></div>` : '<p>This face has no neighbours.</p>'}
      <div class="sublabel" style="margin-top:14px">Convert to</div>
      <div class="opts">${o.convert.filter(c => c.type !== f.kind).map(c => `<button class="opt" data-convert="${c.type}" ${c.type !== 'freeform' && c.deviation == null ? 'disabled' : ''}><span class="sw" style="--c:var(--k-${c.type})"></span><span>${LABEL[c.type]}</span>${c.type === 'freeform' ? '<span class="dv">facets</span>' : dv(c.deviation)}</button>`).join('')}</div>`;
    box.querySelectorAll('[data-into]').forEach(b => {
      b.addEventListener('mouseenter', () => { S.highlight = new Set([+b.dataset.into]); paint(); });
      b.addEventListener('mouseleave', () => { S.highlight = null; paint(); });
      b.addEventListener('click', () => faceOp('merge', { into: +b.dataset.into, mode: S.mergeMode || 'flush' }));
    });
    box.querySelectorAll('#mergeMode button').forEach(b => b.addEventListener('click', () => {
      S.mergeMode = b.dataset.m;
      box.querySelectorAll('#mergeMode button').forEach(x => x.setAttribute('aria-pressed', x === b));
    }));
    box.querySelectorAll('[data-convert]').forEach(b => b.addEventListener('click', () => faceOp('convert', { type: b.dataset.convert })));
  } catch (e) { box.innerHTML = `<p>${esc(e.message)}</p>`; }
}

async function loadExtendDirs(f) {
  const sel = $('#extDir');
  if (!sel || sel.dataset.loaded) return;
  sel.dataset.loaded = '1';
  try {
    const o = await api.get(`/sessions/${S.sid}/faces/${f.id}/options`);
    if (!$('#extDir')) return;
    $('#extDir').innerHTML = `<option value="">In all directions</option>` + o.neighbors.map(n => `<option value="${n.id}">Across the edge to ${LABEL[n.kind].toLowerCase()} ${n.id}</option>`).join('');
  } catch {}
}

async function loadTidy(f) {
  const box = $('#tidyBody');
  box.innerHTML = '<p>Looking for variants…</p>';
  try {
    const { variants } = await api.get(`/sessions/${S.sid}/faces/${f.id}/tidy`);
    if (!$('#tidyBody')) return;
    if (!variants.length) { box.innerHTML = '<p>No single face describes this area within 10× the tolerance.</p>'; return; }
    box.innerHTML = `<div class="opts">${variants.map((v, i) => `<button class="opt ${i === 0 ? 'on' : ''}" data-v="${i}"><span class="sw" style="--c:var(--k-${v.type})"></span><span>${LABEL[v.type]}<span class="sub">${plural(v.saved, 'face')} fewer</span></span><span class="dv ${v.deviation <= S.st.params.tolerance ? 'ok' : ''}">${fmt(v.deviation, 3)} mm</span></button>`).join('')}</div>
      <div class="row end"><button class="btn small primary" data-act="tidy-apply">Apply</button></div>`;
    let pick = 0;
    const opts = box.querySelectorAll('[data-v]');
    opts.forEach(b => {
      b.addEventListener('mouseenter', () => { S.highlight = new Set(variants[+b.dataset.v].members); paint(); });
      b.addEventListener('mouseleave', () => { S.highlight = null; paint(); });
      b.addEventListener('click', () => { pick = +b.dataset.v; opts.forEach(o => o.classList.toggle('on', o === b)); });
    });
    box.querySelector('[data-act=tidy-apply]').addEventListener('click', () => {
      const v = variants[pick];
      faceOp('tidy', { members: v.members, model: v.model });
    });
  } catch (e) { box.innerHTML = `<p>${esc(e.message)}</p>`; }
}

// ------------------------------------------------------------------ actions
async function guarded(label, fn) {
  if (S.busy) return;
  setBusy(true, label);
  try { return await fn(); }
  catch (e) { toast(e.message, true); }
  finally { setBusy(false); }
}

function setBusy(on, label = '', progress = null) {
  S.busy = on;
  const p = $('#progress');
  p.hidden = !on;
  p.classList.toggle('indeterminate', on && progress == null);
  p.querySelector('span').style.width = progress == null ? '' : `${Math.round(progress * 100)}%`;
  if (on && label) $('#statusHint').textContent = `${label}…`;
  if (!on) $('#statusHint').textContent = 'Drag to rotate, scroll to zoom, Shift+drag to pan. F fits the view, 1–4 switch views.';
  for (const id of ['#analyzeBtn', '#exportBtn', '#exportMore', '#undoBtn', '#redoBtn']) if (on) $(id).disabled = true;
  if (!on && S.st) renderAll();
}

function progress(p) {
  const bar = $('#progress');
  bar.classList.remove('indeterminate');
  bar.querySelector('span').style.width = `${Math.round(p * 100)}%`;
}

async function faceOp(op, body) {
  const fid = S.sel;
  await guarded('Working', async () => {
    const path = op === 'tidy' ? 'tidy' : op;
    const st = await api.post(`/sessions/${S.sid}/faces/${fid}/${path}`, body);
    if (['merge', 'convert', 'recognize', 'tidy'].includes(op)) S.openTool = null;
    applyState(st);
  });
}

async function runJobOp(path, body, label) {
  await guarded(label, async () => {
    const { job } = await api.post(path, body);
    const st = await api.job(job, progress);
    applyState(st);
  });
}

async function openFile(file) {
  if (!file) return;
  await guarded(`Reading ${file.name}`, async () => {
    if (S.sid) fetch(`/api/sessions/${S.sid}`, { method: 'DELETE' }).catch(() => {});
    const st = await api.upload(file);
    S.sid = st.id;
    S.sel = null; S.check = null; S.openTool = null; S.trace = null; endTrace();
    applyState(st, { keepSel: false });
    if (S.view === 'step' || S.view === 'validity') setView('faces');
    toast(`Opened ${file.name}. Press Analyse mesh.`);
  });
}

async function analyze() {
  const types = [...document.querySelectorAll('#legend input[data-kind]:checked')].map(i => i.dataset.kind);
  const selected = S.st.bodies.length > 1 ? S.st.bodies.filter(b => b.selected).map(b => b.id) : null;
  const all = !selected || selected.length === S.st.bodies.length;
  await guarded('Analysing', async () => {
    const { job } = await api.post(`/sessions/${S.sid}/analyze`, {
      tolerance: +$('#tolInput').value, angle: +$('#angInput').value, min_face_pct: +$('#minInput').value,
      types, bodies: all ? null : selected,
    });
    const st = await api.job(job, progress);
    S.sel = null; S.check = null;
    applyState(st, { keepSel: false });
    toast(`Found ${plural(st.summary.faces, 'face')} in ${fmt(st.elapsed, 1)} s.`);
  });
}

async function runCheck() {
  await guarded('Building and checking the STEP', async () => {
    const { job } = await api.post(`/sessions/${S.sid}/check`, { thread_mode: S.threadMode });
    const r = await api.job(job, progress);
    S.check = r.check; S.checkVersion = S.st.version;
    if (r.preview) setPreview(r.preview);
    renderAll();
    const c = r.check;
    const open = Object.values(c.closed).filter(x => !x).length;
    const n = c.invalid.length + c.discarded.length;
    toast(n ? `Check found ${plural(n, 'face')} to look at.` : open ? `Every face is valid, but ${plural(open, 'body', 'bodies')} did not close into a solid. Close the mesh gaps first.` : 'The STEP checks out: every face is valid and every body is a closed solid.');
  });
}

async function exportStep() {
  await guarded('Building the STEP', async () => {
    const { job } = await api.post(`/sessions/${S.sid}/export`, { thread_mode: S.threadMode });
    const r = await api.job(job, progress);
    const a = document.createElement('a');
    a.href = `/api/jobs/${job}/download`;
    a.download = r.name;
    document.body.appendChild(a); a.click(); a.remove();
    const open = Object.values(r.closed).filter(x => !x).length;
    toast(`Downloaded ${r.name}: ${plural(r.faces, 'face')}${open ? `, ${open} open bod${open > 1 ? 'ies' : 'y'}` : ''}.${r.messages.length ? ' ' + r.messages.join(' ') : ''}`);
  });
}

async function report() {
  if (!S.check || S.checkVersion !== S.st.version) await runCheck();
  const prevView = S.view, prevSel = S.sel;
  S.sel = null;
  const shots = {};
  for (const v of ['faces', 'deviation']) { S.view = v; paint(); outline(); shots[v] = viewer.snapshot(900, 640); }
  S.view = prevView; S.sel = prevSel; paint(); outline();
  openReport({ st: S.st, check: S.check, shots, hist: histogram(S.dev, S.labels, S.faces, S.st.params.tolerance), LABEL, KINDS });
}

// ------------------------------------------------------------------ selection & picking
function select(id) {
  if (S.sel !== id) { S.openTool = null; S.holePick = null; }
  S.sel = id;
  renderInspector();
  paint();
  outline();
}

function onHover(hit, e) {
  const tip = $('#hoverTip');
  if (!S.st?.analyzed || !hit || hit.preview) { tip.hidden = true; if (S.hover != null) { S.hover = null; paint(); } return; }
  const id = S.labels[hit.tri];
  const f = S.faces.get(id);
  if (S.tool === 'trace') {
    tip.hidden = false;
    tip.textContent = id === S.trace.face ? 'Click to add a point' : 'Stay on the selected face';
  } else if (f) {
    let t = LABEL[f.kind];
    if (f.hole) t = `Hole Ø${fmt(f.hole.diameter, 2)}`;
    else if (f.kind === 'cylinder' || f.kind === 'sphere') t += ` Ø${fmt(f.params.diameter, 2)}`;
    else if (f.kind === 'thread') t += ` ${f.params.designation || ''}`;
    if (S.view === 'deviation' && f.deviation) t += `, max ${fmt(f.deviation.max, 3)} mm`;
    if (S.view === 'cavities') {
      const cav = S.st.cavities.find(c => c.id === S.mesh.bodies[hit.tri]);
      t = cav ? `Cavity, ${fmt(cav.volume, 1)} mm³. Click to fill.` : 'Solid';
    }
    tip.hidden = false;
    tip.textContent = t;
  }
  if (e) { const r = $('#stage').getBoundingClientRect(); tip.style.left = `${e.clientX - r.left}px`; tip.style.top = `${e.clientY - r.top}px`; }
  if (id !== S.hover && S.tool !== 'trace') { S.hover = id; paint(); }
}

function onClick(hit) {
  if (!S.st) return;
  if (S.tool === 'trace') { if (hit) traceClick(hit); return; }
  if (S.view === 'cavities' && hit) {
    const cav = S.st.cavities.find(c => c.id === S.mesh.bodies[hit.tri]);
    if (cav) { fillCavities([cav.id]); return; }
  }
  if (!S.st.analyzed) return;
  if (!hit || hit.preview) { select(null); return; }
  select(S.labels[hit.tri]);
}

// ------------------------------------------------------------------ trace (cut along edges)
function startTrace(f) {
  S.tool = 'trace';
  S.trace = { face: f.id, points: [], path: [] };
  $('#tracebar').hidden = false;
  $('#traceApply').disabled = true;
  $('#traceText').textContent = 'Click points on the face. The path follows mesh edges between them.';
  viewer.setPath(null);
}
function endTrace() {
  S.tool = null;
  S.trace = null;
  $('#tracebar').hidden = true;
  viewer.setPath(null);
}
function traceClick(hit) {
  const T = S.trace;
  const id = S.labels[hit.tri];
  if (id !== T.face) { toast('Click on the selected face.', true); return; }
  const v = viewer.nearestVertex(hit.tri, hit.point);
  if (T.points.length) {
    const seg = viewer.edgePath(T.points[T.points.length - 1], v, S.tris.get(T.face));
    if (!seg) { toast('No edge path to that point inside this face.', true); return; }
    T.path.push(...seg.slice(1));
    T.segs = (T.segs || []).concat([seg.length - 1]);
  } else T.path = [v];
  T.points.push(v);
  viewer.setPath(T.path.map(x => viewer.vertex(x)));
  $('#traceApply').disabled = T.path.length < 2;
  $('#traceText').textContent = `${plural(T.path.length - 1, 'edge')} traced. Reach the face border on both ends to split it.`;
}
function traceUndo() {
  const T = S.trace;
  if (!T || !T.points.length) return;
  T.points.pop();
  const n = T.segs?.pop() || 0;
  T.path = T.points.length ? T.path.slice(0, T.path.length - n) : [];
  viewer.setPath(T.path.map(x => viewer.vertex(x)));
  $('#traceApply').disabled = T.path.length < 2;
}
async function traceApply() {
  const T = S.trace;
  await guarded('Cutting', async () => {
    const st = await api.post(`/sessions/${S.sid}/faces/${T.face}/cut`, { path: T.path });
    endTrace();
    S.sel = null;
    applyState(st);
  });
}

// ------------------------------------------------------------------ repair & bodies
async function fillCavities(ids) {
  await guarded('Filling', async () => applyState(await api.post(`/sessions/${S.sid}/repair/cavities`, { ids })));
}

function setView(v) {
  S.view = v;
  $('#hoverTip').hidden = true;
  document.querySelectorAll('#views button').forEach(b => b.setAttribute('aria-selected', b.dataset.view === v));
  $('#app').dataset.view = v;
  if ((v === 'validity' || v === 'step') && S.st?.analyzed && (!S.check || S.checkVersion !== S.st.version) && !S.busy) runCheck();
  if (v === 'step' && S.st && !S.st.analyzed) toast('Analyse the mesh first.');
  if (v === 'cavities' && S.st && !S.st.cavities.length) toast('This mesh has no enclosed cavities.');
  viewer.setEdgesVisible(v !== 'cavities');
  $('#devTolMid').textContent = S.st ? `${fmt(S.st.params.tolerance, 3)}` : '';
  $('#devTolMax').textContent = S.st ? `≥ ${fmt(S.st.params.tolerance * 2.5, 3)} mm` : '';
  paint();
}

// ------------------------------------------------------------------ misc ui
let toastTimer;
function toast(msg, err = false) {
  const t = $('#toast');
  t.textContent = msg;
  t.classList.toggle('error', err);
  t.hidden = false;
  clearTimeout(toastTimer);
  toastTimer = setTimeout(() => { t.hidden = true; }, err ? 6000 : 3800);
}

function setTheme(t) {
  document.documentElement.dataset.theme = t;
  try { localStorage.setItem('unfacet-theme', t); } catch {}
  readColors();
  viewer.edgeColor.set(t === 'dark' ? 0x0c0f13 : 0x18202b);
  if (viewer.backMat) viewer.backMat.color.set(t === 'dark' ? 0x2a323b : 0x4a5560);
  if (S.st) { renderAll(); if (S.check?.preview) paint(); }
}

function bind() {
  const fi = $('#fileInput');
  const pick = () => fi.click();
  $('#drop').addEventListener('click', pick);
  $('#drop').addEventListener('keydown', e => { if (e.key === 'Enter' || e.key === ' ') { e.preventDefault(); pick(); } });
  $('#emptyOpen').addEventListener('click', pick);
  fi.addEventListener('change', () => { openFile(fi.files[0]); fi.value = ''; });
  let depth = 0;
  window.addEventListener('dragenter', e => { e.preventDefault(); depth++; $('#stage').classList.add('dragging'); });
  window.addEventListener('dragleave', () => { if (--depth <= 0) { depth = 0; $('#stage').classList.remove('dragging'); } });
  window.addEventListener('dragover', e => e.preventDefault());
  window.addEventListener('drop', e => { e.preventDefault(); depth = 0; $('#stage').classList.remove('dragging'); openFile(e.dataTransfer.files[0]); });

  $('#analyzeBtn').addEventListener('click', analyze);
  $('#legend').addEventListener('mouseover', e => {
    const k = e.target.closest('[data-k]')?.dataset.k;
    if (!S.st?.analyzed) return;
    const ids = k ? new Set(S.st.faces.filter(f => f.kind === k).map(f => f.id)) : null;
    if ((ids?.size || 0) !== (S.highlight?.size || 0) || k !== S._legendK) { S._legendK = k; S.highlight = ids && ids.size ? ids : null; paint(); }
  });
  $('#legend').addEventListener('mouseleave', () => { S._legendK = null; S.highlight = null; paint(); });
  $('#views').addEventListener('click', e => { const b = e.target.closest('button'); if (b) setView(b.dataset.view); });
  $('#undoBtn').addEventListener('click', () => guarded('Undoing', async () => applyState(await api.post(`/sessions/${S.sid}/undo`))));
  $('#redoBtn').addEventListener('click', () => guarded('Redoing', async () => applyState(await api.post(`/sessions/${S.sid}/redo`))));
  $('#exportBtn').addEventListener('click', exportStep);
  $('#exportMore').addEventListener('click', e => { e.stopPropagation(); $('#exportPop').hidden = !$('#exportPop').hidden; });
  document.addEventListener('click', e => { if (!e.target.closest('.export-wrap')) $('#exportPop').hidden = true; });
  document.querySelectorAll('input[name=tmode]').forEach(r => r.addEventListener('change', () => { S.threadMode = r.value; S.check = null; renderAll(); }));
  $('#closeGapsBtn').addEventListener('click', () => guarded('Closing gaps', async () => applyState(await api.post(`/sessions/${S.sid}/repair/gaps`))));
  $('#cavViewBtn').addEventListener('click', () => setView('cavities'));
  $('#cavFillBtn').addEventListener('click', () => fillCavities(null));
  $('#viewcube').addEventListener('click', e => { const b = e.target.closest('button'); if (b) viewer.view(b.dataset.v); });
  $('#edgesBtn').addEventListener('click', toggleWire);
  $('#sectionBtn').addEventListener('click', toggleSection);
  $('#sectionPos').addEventListener('input', e => { S.section.t = e.target.value / 1000; applySection(); });
  $('#sectionAxis').addEventListener('click', e => { const b = e.target.closest('button'); if (!b) return; S.section.axis = b.dataset.axis; $('#sectionAxis').querySelectorAll('button').forEach(x => x.setAttribute('aria-pressed', x === b)); applySection(); });
  $('#sectionFlip').addEventListener('click', () => { S.section.flip = !S.section.flip; applySection(); });
  $('#themeBtn').addEventListener('click', () => setTheme(document.documentElement.dataset.theme === 'dark' ? 'light' : 'dark'));
  $('#helpBtn').addEventListener('click', () => $('#helpDlg').showModal());
  $('#traceUndo').addEventListener('click', traceUndo);
  $('#traceCancel').addEventListener('click', endTrace);
  $('#traceApply').addEventListener('click', traceApply);

  // bodies
  $('#bodyList').addEventListener('change', e => {
    const ids = [...document.querySelectorAll('#bodyList li')].filter(li => li.querySelector('input').checked).map(li => +li.dataset.body);
    if (!ids.length) { toast('Keep at least one body.', true); e.target.checked = true; return; }
    guarded('Updating', async () => applyState(await api.post(`/sessions/${S.sid}/bodies`, { selected: ids.length === S.st.bodies.length ? null : ids })));
  });
  const renameBody = li => {
    const b = S.st.bodies.find(x => x.id === +li.dataset.body);
    $('#renameInput').value = b.name;
    $('#renameDlg').returnValue = '';
    $('#renameDlg').showModal();
    $('#renameInput').select();
    $('#renameDlg').onclose = async () => {
      if ($('#renameDlg').returnValue !== 'ok') return;
      const name = $('#renameInput').value.trim();
      if (!name) return;
      try { const r = await api.post(`/sessions/${S.sid}/bodies/${b.id}/name`, { name }); S.st.bodies = r.bodies; renderBodies(); toast(`Renamed to ${name}. The STEP will use this name.`); }
      catch (err) { toast(err.message, true); }
    };
  };
  $('#bodyList').addEventListener('dblclick', e => { const li = e.target.closest('li'); if (li) renameBody(li); });
  $('#bodyList').addEventListener('contextmenu', e => { const li = e.target.closest('li'); if (li) { e.preventDefault(); renameBody(li); } });
  $('#bodyList').addEventListener('mouseover', e => {
    const li = e.target.closest('li');
    const b = li ? +li.dataset.body : null;
    if (b === S._hoverBody) return;
    S._hoverBody = b;
    if (b == null) { S.highlight = null; paint(); return; }
    const ids = new Set();
    for (let t = 0; t < S.labels.length; t++) if (S.mesh.bodies[t] === b) ids.add(S.labels[t]);
    S.highlight = ids; paint();
  });
  $('#bodyList').addEventListener('mouseleave', () => { S._hoverBody = null; S.highlight = null; paint(); });

  // keyboard
  window.addEventListener('keydown', e => {
    if (e.target.closest('input, textarea, select, dialog')) return;
    const k = e.key.toLowerCase();
    if ((e.ctrlKey || e.metaKey) && k === 'z') { e.preventDefault(); (e.shiftKey ? $('#redoBtn') : $('#undoBtn')).click(); return; }
    if ((e.ctrlKey || e.metaKey) && k === 'y') { e.preventDefault(); $('#redoBtn').click(); return; }
    if (e.ctrlKey || e.metaKey || e.altKey) return;
    if (k === 'escape') { if (S.tool) endTrace(); else select(null); }
    else if (k === 'f' || ['1', '2', '3', '4'].includes(k)) viewer.view(k);
    else if ('qwert'.includes(k) && k.length === 1) setView(['faces', 'deviation', 'validity', 'cavities', 'step']['qwert'.indexOf(k)]);
    else if (k === 's') toggleSection();
    else if (k === 'm') toggleWire();
    else if (k === '?') $('#helpDlg').showModal();
  });
}

function toggleWire() {
  S.wire = !S.wire;
  $('#edgesBtn').setAttribute('aria-pressed', S.wire);
  viewer._wireOn = S.wire;
  viewer.setWire(S.wire);
}
function toggleSection() {
  S.section.on = !S.section.on;
  $('#sectionBtn').setAttribute('aria-pressed', S.section.on);
  $('#sectionCtl').hidden = !S.section.on;
  applySection();
}
function applySection() { viewer.setSection(S.section.on, S.section.axis, S.section.t, S.section.flip); }

// ------------------------------------------------------------------ boot
(async function boot() {
  let theme = 'light';
  try { theme = localStorage.getItem('unfacet-theme') || (matchMedia('(prefers-color-scheme: dark)').matches ? 'dark' : 'light'); } catch {}
  setTheme(theme);
  bind();
  renderLegend();
  renderInspector();
  try { S.meta = await api.get('/meta'); } catch (e) { toast('The server is not reachable.', true); }
})();

// handy for scripting and debugging from the console
window.unfacet = { state: S, select, setView, viewer };
