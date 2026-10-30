// 3D viewport: mesh display, face colouring, boundaries, picking, section, previews.
import * as THREE from 'three';
import { OrbitControls } from 'three/addons/OrbitControls.js';
import { computeBoundsTree, disposeBoundsTree, acceleratedRaycast } from 'three-mesh-bvh';

THREE.BufferGeometry.prototype.computeBoundsTree = computeBoundsTree;
THREE.BufferGeometry.prototype.disposeBoundsTree = disposeBoundsTree;
THREE.Mesh.prototype.raycast = acceleratedRaycast;

export class Viewer {
  constructor(canvas, { onHover, onClick } = {}) {
    this.canvas = canvas;
    this.onHover = onHover || (() => {});
    this.onClick = onClick || (() => {});
    this.renderer = new THREE.WebGLRenderer({ canvas, antialias: true, preserveDrawingBuffer: true, alpha: true });
    this.renderer.setPixelRatio(Math.min(window.devicePixelRatio, 2));
    this.renderer.localClippingEnabled = true;
    this.scene = new THREE.Scene();
    this.camera = new THREE.PerspectiveCamera(32, 1, 0.1, 1e5);
    this.camera.up.set(0, 0, 1);
    this.controls = new OrbitControls(this.camera, canvas);
    this.controls.enableDamping = false;
    this.controls.screenSpacePanning = true;
    this.controls.mouseButtons = { LEFT: THREE.MOUSE.ROTATE, MIDDLE: THREE.MOUSE.DOLLY, RIGHT: THREE.MOUSE.PAN };
    this.controls.addEventListener('change', () => this.render());
    this.scene.add(new THREE.HemisphereLight(0xffffff, 0x8090a0, 1.35));
    this.head = new THREE.DirectionalLight(0xffffff, 1.6);
    this.camera.add(this.head);
    this.head.position.set(0.4, 0.8, 1);
    this.scene.add(this.camera);
    this.group = new THREE.Group();
    this.scene.add(this.group);
    this.clip = new THREE.Plane(new THREE.Vector3(0, 0, -1), 0);
    this.clipOn = false;
    this.edgeColor = new THREE.Color(0x18202b);
    this._ray = new THREE.Raycaster();
    this._ptr = new THREE.Vector2();
    this._pending = false;
    this._hoverTri = -2;
    this._bindPointer();
    new ResizeObserver(() => this.resize()).observe(canvas.parentElement);
    this.resize();
  }

  // ------------------------------------------------------------- setup
  resize() {
    const el = this.canvas.parentElement;
    const w = el.clientWidth, h = el.clientHeight;
    if (!w || !h) return;
    this.renderer.setSize(w, h, false);
    this.camera.aspect = w / h;
    this.camera.updateProjectionMatrix();
    this.render();
  }

  render() {
    if (this._raf) return;
    this._raf = requestAnimationFrame(() => {
      this._raf = 0;
      this.renderer.render(this.scene, this.camera);
    });
  }

  renderNow() { this.renderer.render(this.scene, this.camera); }

  clear() {
    for (const o of [...this.group.children]) {
      this.group.remove(o);
      o.geometry?.disposeBoundsTree?.();
      o.geometry?.dispose();
      o.material?.dispose?.();
    }
    this.mesh = this.edges = this.wire = this.open = this.overlay = this.preview = this.path = this.back = null;
  }

  setMesh(positions, indices) {
    this.clear();
    const nTri = indices.length / 3;
    this.nTri = nTri;
    this.positions = positions;
    this.indices = indices;
    const P = new Float32Array(nTri * 9);
    for (let t = 0; t < nTri; t++) {
      for (let k = 0; k < 3; k++) {
        const v = indices[t * 3 + k];
        P[t * 9 + k * 3] = positions[v * 3];
        P[t * 9 + k * 3 + 1] = positions[v * 3 + 1];
        P[t * 9 + k * 3 + 2] = positions[v * 3 + 2];
      }
    }
    const g = new THREE.BufferGeometry();
    g.setAttribute('position', new THREE.BufferAttribute(P, 3));
    g.computeVertexNormals();
    this.colors = new Float32Array(nTri * 9).fill(0.75);
    g.setAttribute('color', new THREE.BufferAttribute(this.colors, 3));
    g.computeBoundsTree();
    this.material = new THREE.MeshStandardMaterial({
      vertexColors: true, roughness: 0.62, metalness: 0.04, side: THREE.FrontSide,
      polygonOffset: true, polygonOffsetFactor: 1, polygonOffsetUnits: 1,
    });
    this.mesh = new THREE.Mesh(g, this.material);
    this.group.add(this.mesh);
    // inside of the part, seen through a section: flat dark "cap"
    this.backMat = new THREE.MeshBasicMaterial({ color: 0x4a5560, side: THREE.BackSide });
    this.back = new THREE.Mesh(g, this.backMat);
    this.group.add(this.back);
    this._buildEdgeTable();
    g.computeBoundingBox();
    this.box = g.boundingBox.clone();
    this.size = this.box.getSize(new THREE.Vector3()).length() || 1;
    this.edgesMat = new THREE.LineBasicMaterial({ color: this.edgeColor, transparent: true, opacity: 0.85 });
    this.selMat = new THREE.LineBasicMaterial({ color: 0x2b63d9 });
    this._applyClip();
    this.fit();
  }

  _buildEdgeTable() {
    // undirected edges with their (up to two) triangles
    const I = this.indices, n = this.nTri;
    const map = new Map();
    const ea = [], eb = [], t1 = [], t2 = [];
    for (let t = 0; t < n; t++) {
      for (let k = 0; k < 3; k++) {
        let a = I[t * 3 + k], b = I[t * 3 + (k + 1) % 3];
        if (a > b) { const x = a; a = b; b = x; }
        const key = a * 4294967296 + b;
        const e = map.get(key);
        if (e === undefined) { map.set(key, ea.length); ea.push(a); eb.push(b); t1.push(t); t2.push(-1); }
        else if (t2[e] === -1) t2[e] = t;
      }
    }
    this.E = { a: Uint32Array.from(ea), b: Uint32Array.from(eb), t1: Int32Array.from(t1), t2: Int32Array.from(t2) };
    // vertex adjacency for path tracing
    this.vadj = null;
  }

  // ------------------------------------------------------------- colours
  setTriColors(rgb) {
    // rgb: Float32Array nTri*3
    const C = this.colors, n = this.nTri;
    for (let t = 0; t < n; t++) {
      const r = rgb[t * 3], g = rgb[t * 3 + 1], b = rgb[t * 3 + 2];
      const o = t * 9;
      C[o] = C[o + 3] = C[o + 6] = r;
      C[o + 1] = C[o + 4] = C[o + 7] = g;
      C[o + 2] = C[o + 5] = C[o + 8] = b;
    }
    this.mesh.geometry.attributes.color.needsUpdate = true;
    this.render();
  }

  setCornerColors(rgb) {
    // rgb: Float32Array nTri*9 (per triangle corner)
    this.colors.set(rgb);
    this.mesh.geometry.attributes.color.needsUpdate = true;
    this.render();
  }

  setOpacity(o) {
    this.material.transparent = o < 1;
    this.material.opacity = o;
    this.material.depthWrite = o >= 1;
    this.back.visible = o >= 1;
    this.material.needsUpdate = true;
    this.render();
  }

  // ------------------------------------------------------------- lines
  _lines(pairs, mat) {
    const P = this.positions, n = pairs.length / 2;
    const L = new Float32Array(n * 6);
    for (let i = 0; i < n; i++) {
      const a = pairs[i * 2], b = pairs[i * 2 + 1];
      L.set([P[a * 3], P[a * 3 + 1], P[a * 3 + 2], P[b * 3], P[b * 3 + 1], P[b * 3 + 2]], i * 6);
    }
    const g = new THREE.BufferGeometry();
    g.setAttribute('position', new THREE.BufferAttribute(L, 3));
    const l = new THREE.LineSegments(g, mat);
    l.renderOrder = 2;
    return l;
  }

  _replace(name, obj) {
    if (this[name]) { this.group.remove(this[name]); this[name].geometry.dispose(); }
    this[name] = obj;
    if (obj) this.group.add(obj);
  }

  setBoundaries(labels, selected = null) {
    const E = this.E, out = [], sel = [];
    for (let i = 0; i < E.a.length; i++) {
      const l1 = labels[E.t1[i]], l2 = E.t2[i] >= 0 ? labels[E.t2[i]] : -999;
      if (l1 !== l2) {
        out.push(E.a[i], E.b[i]);
        if (selected !== null && (l1 === selected || l2 === selected)) sel.push(E.a[i], E.b[i]);
      }
    }
    this._replace('edges', out.length ? this._lines(out, this.edgesMat) : null);
    this._replace('selEdges', sel.length ? this._lines(sel, this.selMat) : null);
    if (this.selEdges) this.selEdges.renderOrder = 3;
    this._applyClip();
    this.render();
  }

  setEdgesVisible(v) { if (this.edges) this.edges.visible = v; if (this.selEdges) this.selEdges.visible = v; this.render(); }

  setWire(on) {
    if (!on) { this._replace('wire', null); this.render(); return; }
    const E = this.E, pairs = new Uint32Array(E.a.length * 2);
    for (let i = 0; i < E.a.length; i++) { pairs[i * 2] = E.a[i]; pairs[i * 2 + 1] = E.b[i]; }
    this._replace('wire', this._lines(pairs, new THREE.LineBasicMaterial({ color: this.edgeColor, transparent: true, opacity: 0.22 })));
    this._applyClip();
    this.render();
  }

  setOpenEdges(pairs) {
    this._replace('open', pairs && pairs.length ? this._lines(pairs, new THREE.LineBasicMaterial({ color: 0xc9483e })) : null);
    if (this.open) this.open.renderOrder = 4;
    this.render();
  }

  setPath(points) {
    if (!points || points.length < 2) { this._replace('path', null); this.render(); return; }
    const g = new THREE.BufferGeometry();
    g.setAttribute('position', new THREE.BufferAttribute(Float32Array.from(points.flat()), 3));
    const l = new THREE.Line(g, new THREE.LineBasicMaterial({ color: 0xe0362c, depthTest: false }));
    l.renderOrder = 5;
    this._replace('path', l);
    this.render();
  }

  // ------------------------------------------------------------- overlays
  setOverlayTris(tris, color, opacity = 1) {
    if (!tris || !tris.length) { this._replace('overlay', null); this.render(); return; }
    const P = this.positions, I = this.indices;
    const A = new Float32Array(tris.length * 9);
    tris.forEach((t, j) => {
      for (let k = 0; k < 3; k++) { const v = I[t * 3 + k]; A.set([P[v * 3], P[v * 3 + 1], P[v * 3 + 2]], j * 9 + k * 3); }
    });
    const g = new THREE.BufferGeometry();
    g.setAttribute('position', new THREE.BufferAttribute(A, 3));
    g.computeVertexNormals();
    const m = new THREE.Mesh(g, new THREE.MeshStandardMaterial({ color, roughness: 0.5, side: THREE.DoubleSide, transparent: opacity < 1, opacity }));
    m.renderOrder = 1;
    this._replace('overlay', m);
    this._applyClip();
    this.render();
  }

  setPreview(pos, idx, triRgb) {
    if (!pos) { this._replace('preview', null); return; }
    const n = idx.length / 3;
    const A = new Float32Array(n * 9), C = new Float32Array(n * 9);
    for (let t = 0; t < n; t++) for (let k = 0; k < 3; k++) {
      const v = idx[t * 3 + k];
      A.set([pos[v * 3], pos[v * 3 + 1], pos[v * 3 + 2]], t * 9 + k * 3);
      C.set([triRgb[t * 3], triRgb[t * 3 + 1], triRgb[t * 3 + 2]], t * 9 + k * 3);
    }
    const g = new THREE.BufferGeometry();
    g.setAttribute('position', new THREE.BufferAttribute(A, 3));
    g.setAttribute('color', new THREE.BufferAttribute(C, 3));
    g.computeVertexNormals();
    const m = new THREE.Mesh(g, new THREE.MeshStandardMaterial({ vertexColors: true, roughness: 0.45, metalness: 0.08, side: THREE.DoubleSide }));
    this._replace('preview', m);
    m.visible = false;
    this._applyClip();
  }

  showPreview(on) {
    if (this.preview) this.preview.visible = on;
    for (const k of ['mesh', 'back', 'edges', 'selEdges', 'wire', 'overlay']) if (this[k]) this[k].visible = !on && (k !== 'wire' || this._wireOn !== false);
    this.render();
  }

  // ------------------------------------------------------------- section
  setSection(on, axis = 'z', t = 0.5, flip = false) {
    this.clipOn = on;
    if (on && this.box) {
      const n = new THREE.Vector3(axis === 'x' ? 1 : 0, axis === 'y' ? 1 : 0, axis === 'z' ? 1 : 0);
      const lo = this.box.min[axis], hi = this.box.max[axis];
      const pos = lo + (hi - lo) * t;
      if (!flip) n.negate();
      this.clip.set(n, flip ? -pos : pos);
    }
    this._applyClip();
    this.render();
  }

  _applyClip() {
    const planes = this.clipOn ? [this.clip] : [];
    this.group.traverse(o => {
      if (o.material) { o.material.clippingPlanes = planes; o.material.needsUpdate = true; }
    });
  }

  // ------------------------------------------------------------- camera
  fit(dir) {
    if (!this.box) return;
    const c = this.box.getCenter(new THREE.Vector3());
    const r = this.size / 2;
    const d = dir || this.camera.position.clone().sub(this.controls.target).normalize();
    if (!dir && d.lengthSq() < 1e-6) d.set(1, -1.2, 0.9).normalize();
    const dist = r / Math.sin(THREE.MathUtils.degToRad(this.camera.fov / 2)) * 1.08;
    this.camera.near = Math.max(dist / 1000, 0.01);
    this.camera.far = dist * 100;
    this.camera.updateProjectionMatrix();
    this.controls.target.copy(c);
    this.camera.position.copy(c).addScaledVector(d.normalize(), dist);
    this.camera.up.set(0, 0, 1);
    if (Math.abs(d.z) > 0.999) this.camera.up.set(0, 1, 0);
    this.camera.lookAt(c);
    this.controls.update();
    this.render();
  }

  view(v) {
    const dirs = { 1: [0, -1, 0], 2: [0, 0, 1], 3: [1, 0, 0], 4: [1, -1.2, 0.9] };
    if (v === 'f') return this.fit();
    this.fit(new THREE.Vector3(...dirs[v]));
  }

  // ------------------------------------------------------------- picking
  _bindPointer() {
    const el = this.canvas;
    let down = null;
    el.addEventListener('pointerdown', e => { down = { x: e.clientX, y: e.clientY, b: e.button }; });
    el.addEventListener('pointerup', e => {
      if (down && e.button === 0 && Math.hypot(e.clientX - down.x, e.clientY - down.y) < 5) {
        const hit = this.pick(e);
        this.onClick(hit, e);
      }
      down = null;
    });
    el.addEventListener('pointermove', e => {
      if (e.buttons) return;
      this._lastMove = e;
      if (this._pending) return;
      this._pending = true;
      requestAnimationFrame(() => {
        this._pending = false;
        const hit = this.pick(this._lastMove);
        const t = hit ? hit.tri : -1;
        if (t !== this._hoverTri || (hit && this.wantPoint)) {
          this._hoverTri = t;
          this.onHover(hit, this._lastMove);
        }
      });
    });
    el.addEventListener('pointerleave', () => { this._hoverTri = -2; this.onHover(null, null); });
    el.addEventListener('contextmenu', e => e.preventDefault());
  }

  pick(e) {
    if (!this.mesh || !e) return null;
    const r = this.canvas.getBoundingClientRect();
    this._ptr.set(((e.clientX - r.left) / r.width) * 2 - 1, -((e.clientY - r.top) / r.height) * 2 + 1);
    this._ray.setFromCamera(this._ptr, this.camera);
    this._ray.firstHitOnly = true;
    const target = this.preview && this.preview.visible ? this.preview : this.mesh;
    const hits = this._ray.intersectObject(target, false);
    for (const h of hits) {
      if (this.clipOn && this.clip.distanceToPoint(h.point) < 0) continue;
      return { tri: Math.floor(h.face.a / 3), point: h.point.clone(), preview: target === this.preview };
    }
    return null;
  }

  nearestVertex(tri, point) {
    const I = this.indices, P = this.positions;
    let best = -1, bd = Infinity;
    for (let k = 0; k < 3; k++) {
      const v = I[tri * 3 + k];
      const d = (P[v * 3] - point.x) ** 2 + (P[v * 3 + 1] - point.y) ** 2 + (P[v * 3 + 2] - point.z) ** 2;
      if (d < bd) { bd = d; best = v; }
    }
    return best;
  }

  vertex(v) { const P = this.positions; return [P[v * 3], P[v * 3 + 1], P[v * 3 + 2]]; }

  // shortest path along mesh edges, restricted to vertices of the allowed triangles
  edgePath(from, to, allowedTris) {
    const I = this.indices, P = this.positions;
    const ok = new Set();
    for (const t of allowedTris) for (let k = 0; k < 3; k++) ok.add(I[t * 3 + k]);
    const adj = new Map();
    for (const t of allowedTris) for (let k = 0; k < 3; k++) {
      const a = I[t * 3 + k], b = I[t * 3 + (k + 1) % 3];
      if (!adj.has(a)) adj.set(a, new Set());
      if (!adj.has(b)) adj.set(b, new Set());
      adj.get(a).add(b); adj.get(b).add(a);
    }
    if (!ok.has(from) || !ok.has(to)) return null;
    const dist = new Map([[from, 0]]), prev = new Map();
    const heap = [[0, from]];
    const push = x => { heap.push(x); let i = heap.length - 1; while (i > 0) { const p = (i - 1) >> 1; if (heap[p][0] <= heap[i][0]) break; [heap[p], heap[i]] = [heap[i], heap[p]]; i = p; } };
    const pop = () => { const top = heap[0], last = heap.pop(); if (heap.length) { heap[0] = last; let i = 0; for (;;) { const l = 2 * i + 1, r = l + 1; let m = i; if (l < heap.length && heap[l][0] < heap[m][0]) m = l; if (r < heap.length && heap[r][0] < heap[m][0]) m = r; if (m === i) break; [heap[m], heap[i]] = [heap[i], heap[m]]; i = m; } } return top; };
    while (heap.length) {
      const [d, u] = pop();
      if (u === to) break;
      if (d > dist.get(u)) continue;
      for (const w of adj.get(u) || []) {
        const nd = d + Math.hypot(P[u * 3] - P[w * 3], P[u * 3 + 1] - P[w * 3 + 1], P[u * 3 + 2] - P[w * 3 + 2]);
        if (nd < (dist.get(w) ?? Infinity)) { dist.set(w, nd); prev.set(w, u); push([nd, w]); }
      }
    }
    if (!dist.has(to)) return null;
    const path = [to];
    while (path[path.length - 1] !== from) path.push(prev.get(path[path.length - 1]));
    return path.reverse();
  }

  // ------------------------------------------------------------- export
  snapshot(w = 900, h = 640) {
    const size = new THREE.Vector2();
    this.renderer.getSize(size);
    const aspect = this.camera.aspect;
    this.renderer.setSize(w, h, false);
    this.camera.aspect = w / h;
    this.camera.updateProjectionMatrix();
    this.renderNow();
    const url = this.canvas.toDataURL('image/png');
    this.renderer.setSize(size.x, size.y, false);
    this.camera.aspect = aspect;
    this.camera.updateProjectionMatrix();
    this.renderNow();
    return url;
  }
}
