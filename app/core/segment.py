"""Mesh segmentation into analytic regions by model-driven region growing."""
from __future__ import annotations

import heapq
import time

import numpy as np

from . import fit
from .mesh import Topo

COPLANAR_DEG = 0.5
GROW_BUDGET_S = 20.0   # per tangent-continuous area; organic shapes stop here and stay freeform


class UF:
    def __init__(self, n):
        self.p = np.arange(n)

    def find(self, a):
        p = self.p
        while p[a] != a:
            p[a] = p[p[a]]
            a = p[a]
        return a

    def union(self, a, b):
        a, b = self.find(a), self.find(b)
        if a != b:
            self.p[b] = a


class Region:
    __slots__ = ("id", "tris", "model", "kind", "name", "groups", "locked", "extra")

    def __init__(self, rid, tris, model, kind, groups=0):
        self.id = rid
        self.tris = np.asarray(tris, np.int64)
        self.model = model          # dict or None
        self.kind = kind            # plane/cylinder/.../thread/freeform/discarded
        self.name = None
        self.groups = groups
        self.locked = False
        self.extra = {}

    def copy(self):
        r = Region(self.id, self.tris, dict(self.model) if self.model else None, self.kind, self.groups)
        r.name, r.locked, r.extra = self.name, self.locked, dict(self.extra)
        return r


def _region_data(topo: Topo, tris):
    tris = np.asarray(tris, np.int64)
    vids = np.unique(topo.F[tris])
    return topo.V[vids], topo.TC[tris], topo.TN[tris]


def coplanar_groups(topo: Topo, tol):
    nf = len(topo.F)
    uf = UF(nf)
    flat = topo.adj[topo.adj_angle < np.radians(COPLANAR_DEG)]
    for a, b in flat:
        uf.union(a, b)
    roots = np.array([uf.find(i) for i in range(nf)])
    _, gid = np.unique(roots, return_inverse=True)
    # verify planarity; split offenders into single triangles
    order = np.argsort(gid, kind="stable")
    bounds = np.flatnonzero(np.diff(gid[order])) + 1
    groups = []
    for g in np.split(order, bounds):
        if len(g) > 2:
            V = topo.V[np.unique(topo.F[g])]
            m = fit.fit_plane(V)
            if np.abs(fit.distance(m, V)).max() > tol:
                groups.extend([[t] for t in g])
                continue
        groups.append(list(g))
    gid = np.empty(nf, np.int64)
    for i, g in enumerate(groups):
        gid[g] = i
    return groups, gid


class _Ctx:
    """Shared per-segmentation state."""

    def __init__(self, topo, tol, types, smooth, max_angle, threads=True):
        self.topo, self.tol, self.types = topo, tol, tuple(types)
        self.threads = threads
        self.smooth, self.max_angle, self.scale = smooth, max_angle, topo.scale
        self.groups, self.gid = coplanar_groups(topo, tol)
        ng = len(self.groups)
        self.garea = np.array([topo.TA[g].sum() for g in self.groups])
        self.gadj = [dict() for _ in range(ng)]
        ga, gb = self.gid[topo.adj[:, 0]], self.gid[topo.adj[:, 1]]
        for a, b, ang in zip(ga, gb, topo.adj_angle):
            if a != b and ang > self.gadj[a].get(b, -1):
                self.gadj[a][b] = ang
                self.gadj[b][a] = ang
        self._strip = {}

    def strip_like(self, g):
        if g not in self._strip:
            grp = self.groups[g]
            if len(grp) <= 2:
                self._strip[g] = True
            else:
                V = self.topo.V[np.unique(self.topo.F[grp])]
                s = np.linalg.svd(V - V.mean(0), compute_uv=False)
                self._strip[g] = bool(s[1] < 0.15 * s[0])
        return self._strip[g]

    def tris_of(self, gs):
        return np.concatenate([self.groups[g] for g in gs])

    def fit(self, tris, types=None, min_groups=99):
        V, C, N = _region_data(self.topo, tris)
        tys = types or self.types
        tys = tuple(t for t in fit.ORDER if t in tys)
        if min_groups < 3:
            tys = tuple(t for t in tys if t in ("plane", "cylinder"))
        elif min_groups < 5:
            tys = tuple(t for t in tys if t != "torus")
        return fit.best_fit(V, C, N, self.tol, self.max_angle, self.scale, tys)


def _components(ctx, gs):
    """Connected components of a set of groups (any adjacency)."""
    gs = set(gs)
    out, seen = [], set()
    for g in gs:
        if g in seen:
            continue
        stack, comp = [g], []
        seen.add(g)
        while stack:
            x = stack.pop(); comp.append(x)
            for n in ctx.gadj[x]:
                if n in gs and n not in seen:
                    seen.add(n); stack.append(n)
        out.append(comp)
    return out


def _smooth_patches(ctx):
    ng = len(ctx.groups)
    uf = UF(ng)
    for a in range(ng):
        for b, ang in ctx.gadj[a].items():
            if ang < ctx.smooth:
                uf.union(a, b)
    roots = {}
    for g in range(ng):
        roots.setdefault(uf.find(g), []).append(g)
    return list(roots.values())


def _seed_region(ctx, seed, allowed, taken):
    """Grow a breadth-first neighbourhood around seed while one primitive still explains it."""
    order, seen = [seed], {seed}
    i = 0
    while i < len(order) and len(order) < 256:
        for n, a in sorted(ctx.gadj[order[i]].items(), key=lambda x: x[1]):
            if n in allowed and n not in seen and n not in taken and a < ctx.smooth:
                seen.add(n); order.append(n)
        i += 1
    best = (order[:1], fit.clean(fit.fit_plane(ctx.topo.V[np.unique(ctx.topo.F[ctx.groups[seed]])])))
    if not ctx.strip_like(seed):
        return best
    k = 2
    while k <= len(order):
        m = ctx.fit(ctx.tris_of(order[:k]), None, k)
        if m is None:
            break
        best = (order[:k], m)
        if k == len(order):
            break
        k = min(k * 2, len(order))
    return best


def _grow(ctx, gs):
    """Greedy model-driven region growing inside a set of groups. Returns [(groups, model)]."""
    gs_set = set(gs)
    region_of = {}
    out = []
    curved = tuple(t for t in ctx.types if t != "plane")
    tiny_run = 0
    sizes = []
    t0 = time.monotonic()
    for seed in sorted(gs, key=lambda g: -ctx.garea[g]):
        if seed in region_of:
            continue
        # CAD tessellations sit exactly on their surfaces; sculpted/scanned shapes only fit
        # loosely, patch by patch. Many loose little fits in a row means an organic area.
        recent = sizes[-20:]
        organic = len(recent) == 20 and sorted(recent)[10] > 0.25
        if tiny_run > 40 or organic or time.monotonic() - t0 > GROW_BUDGET_S:
            # this area is not made of primitives (scan, sculpt, knurl): stop burning time on it
            rest = [g for g in gs if g not in region_of]
            out.append((rest, None))
            break
        rid = len(out)
        rg, model = _seed_region(ctx, seed, gs_set, region_of)
        for g in rg:
            region_of[g] = rid
        tris = list(ctx.tris_of(rg))
        big_plane = not ctx.strip_like(seed) and model["type"] == "plane"
        nfit, fails, rejected = len(tris), 0, set()
        heap = [(a, n) for g0 in rg for n, a in ctx.gadj[g0].items()
                if a < ctx.smooth and n in gs_set and n not in region_of]
        heapq.heapify(heap)
        while heap:
            _, g = heapq.heappop(heap)
            if g in region_of or g in rejected:
                continue
            gt = ctx.groups[g]
            V, C, N = _region_data(ctx.topo, gt)
            ok = fit.check(model, V, C, N, ctx.tol, ctx.max_angle)
            if not ok and not (big_plane and len(rg) == 1) and fails < 10:
                tys = curved if model["type"] != "plane" else ctx.types
                m2 = ctx.fit(tris + list(gt), tys, len(rg) + 1)
                if m2 is not None:
                    model, ok, nfit = m2, True, len(tris) + len(gt)
                    for r in rejected:
                        if r not in region_of:
                            heapq.heappush(heap, (0.0, r))
                    rejected.clear()
                else:
                    fails += 1
            if not ok:
                rejected.add(g)
                continue
            rg.append(g)
            region_of[g] = rid
            tris += list(gt)
            for n, a in ctx.gadj[g].items():
                if n not in region_of and a < ctx.smooth and n in gs_set:
                    heapq.heappush(heap, (a, n))
            if len(tris) > nfit * 1.3 and model["type"] != "plane":
                Vr, Cr, Nr = _region_data(ctx.topo, tris)
                m2 = fit.refine(fit.as_np(model), Vr)
                if fit.check(m2, Vr, Cr, Nr, ctx.tol, ctx.max_angle):
                    model = fit.clean(m2)
                nfit = len(tris)
        if model["type"] not in ctx.types:
            model = None
        tiny_run = tiny_run + 1 if len(rg) < 3 else 0
        if model is not None and len(rg) >= 2:
            Vr = ctx.topo.V[np.unique(ctx.topo.F[ctx.tris_of(rg)])]
            sizes.append(float(np.abs(fit.distance(fit.as_np(model), Vr)).max()) / ctx.tol)
        out.append((rg, model))
    return out


def _looseness(ctx, gs, m):
    V = ctx.topo.V[np.unique(ctx.topo.F[ctx.tris_of(gs)])]
    return float(np.abs(fit.distance(fit.as_np(m), V)).max()) / ctx.tol


def _organic_filter(ctx, grown):
    """A sculpted/scanned area splits into swarms of tiny local fits; call those freeform."""
    fitted = [(gs, m) for gs, m in grown if m is not None]
    if len(fitted) < 40:
        return grown
    areas = np.array([ctx.garea[gs].sum() for gs, _ in grown])
    total = areas.sum()
    fa = np.array([ctx.garea[gs].sum() for gs, _ in fitted])
    loose = np.array([_looseness(ctx, gs, m) for gs, m in fitted])
    if fa[(fa < 0.01 * total) & (loose > 0.25)].sum() < 0.3 * fa.sum():
        return grown
    keep = [(gs, m) for (gs, m), a in zip(grown, areas) if m is not None and a >= 0.03 * total]
    rest = [g for (gs, m), a in zip(grown, areas) if m is None or a < 0.03 * total for g in gs]
    return keep + ([(rest, None)] if rest else [])


def _segment_patch(ctx, patch):
    """Segment one tangent-continuous patch. Returns [(groups, model)]."""
    tris = ctx.tris_of(patch)
    m = ctx.fit(tris, None, len(patch))
    if m is not None:
        return [(patch, m)]
    # pull out real planar faces, then handle what is left
    pset = set(patch)

    def real_plane(g):
        if ctx.strip_like(g) or "plane" not in ctx.types:
            return False
        ring = [g] + [n for n, a in ctx.gadj[g].items() if n in pset and a < ctx.smooth]
        if len(ring) < 3:
            return True
        return ctx.fit(ctx.tris_of(ring), tuple(t for t in ctx.types if t != "plane"), len(ring)) is None

    parea = ctx.garea[patch].sum()
    planes = [g for g in patch if ctx.garea[g] >= 0.01 * parea and real_plane(g)]
    rest = [g for g in patch if g not in set(planes)]
    out = [([g], fit.clean(fit.fit_plane(ctx.topo.V[np.unique(ctx.topo.F[ctx.groups[g]])]))) for g in planes]
    curved = []
    for comp in _components(ctx, rest):
        ctris = ctx.tris_of(comp)
        m = ctx.fit(ctris, None, len(comp))
        if m is not None:
            curved.append((comp, m))
            continue
        if ctx.threads and len(ctris) >= 150:
            tm = thread_params(ctx.topo, ctris, ctx.tol)
            if tm is not None:
                curved.append((comp, tm))
                continue
        grown = _grow(ctx, comp)
        curved.extend(_organic_filter(ctx, grown))
    # re-absorb "planes" that actually lie on a neighbouring curved face (wide cylinder strips)
    if planes and curved:
        gpos = {}
        for i, (gs, _) in enumerate(curved):
            for g in gs:
                gpos[g] = i
        keep = []
        for gs, pm in out:
            g = gs[0]
            host = None
            for n in ctx.gadj[g]:
                i = gpos.get(n)
                if i is not None and curved[i][1] is not None and curved[i][1]["type"] in fit.ORDER[1:]:
                    V, C, N = _region_data(ctx.topo, ctx.groups[g])
                    if fit.check(curved[i][1], V, C, N, ctx.tol, ctx.max_angle):
                        host = i
                        break
            if host is None:
                keep.append((gs, pm))
            else:
                curved[host][0].append(g)
                gpos[g] = host
        out = keep
    return out + curved


def segment(topo: Topo, tol=0.02, min_area_pct=0.0, types=fit.ORDER, smooth_deg=40.0,
            normal_deg=35.0, progress=None):
    ctx = _Ctx(topo, tol, [t for t in types if t in fit.ORDER], np.radians(smooth_deg), np.radians(normal_deg),
               threads="thread" in types)
    patches = sorted(_smooth_patches(ctx), key=lambda p: -len(p))
    regions, done, total = [], 0, len(topo.F)
    for patch in patches:
        for gs, model in _segment_patch(ctx, patch):
            tris = ctx.tris_of(gs)
            regions.append(Region(len(regions), tris, model, model["type"] if model else "freeform", len(gs)))
            done += len(tris)
        if progress:
            progress(done / total)
    regions = postprocess(topo, regions, tol, min_area_pct, ctx.smooth, ctx.max_angle, ctx.types)
    regions = merge_same(topo, regions, tol, ctx.max_angle)
    regions = detect_threads(topo, regions, tol, enabled=("thread" in types))
    return renumber(regions)


def same_surface(a, b, tol):
    if a is None or b is None or a["type"] != b["type"]:
        return False
    t = a["type"]
    ang = lambda u, v: abs(np.dot(u, v))
    if t == "plane":
        return ang(a["normal"], b["normal"]) > 0.9999 and abs(np.dot(np.subtract(b["origin"], a["origin"]), a["normal"])) < tol
    if t == "cylinder":
        if ang(a["axis"], b["axis"]) < 0.9999 or abs(a["radius"] - b["radius"]) > tol:
            return False
        d = np.subtract(b["origin"], a["origin"])
        return np.linalg.norm(d - np.dot(d, a["axis"]) * np.asarray(a["axis"])) < tol
    if t == "sphere":
        return abs(a["radius"] - b["radius"]) < tol and np.linalg.norm(np.subtract(a["center"], b["center"])) < tol
    if t == "cone":
        return np.dot(a["axis"], b["axis"]) > 0.9999 and abs(a["angle"] - b["angle"]) < 1e-3 and \
            np.linalg.norm(np.subtract(a["apex"], b["apex"])) < tol * 5
    if t == "torus":
        return ang(a["axis"], b["axis"]) > 0.9999 and abs(a["major"] - b["major"]) < tol and \
            abs(a["minor"] - b["minor"]) < tol and np.linalg.norm(np.subtract(a["center"], b["center"])) < tol
    return False


def merge_same(topo, regions, tol, max_angle):
    """Merge adjacent regions that lie on the same surface (e.g. a cylinder split at a seam)."""
    changed = True
    while changed:
        changed = False
        L = tri_labels(topo, regions)
        adj = region_adjacency(topo, L)
        byid = {r.id: r for r in regions}
        for r in regions:
            if r.model is None:
                continue
            for n in adj.get(r.id, {}):
                o = byid.get(n)
                if o is None or o is r or o.model is None or o.model["type"] != r.model["type"]:
                    continue
                if not same_surface(r.model, o.model, tol) and adj[r.id][n][0] > np.radians(40):
                    continue
                tris = np.r_[r.tris, o.tris]
                V, C, N = _region_data(topo, tris)
                m = fit.refine(fit.as_np(r.model), V)
                if not fit.check(m, V, C, N, tol, max_angle):
                    m = fit.as_np(r.model)
                    if not fit.check(m, V, C, N, tol, max_angle):
                        m = fit.best_fit(V, C, N, tol, max_angle, topo.scale, (r.model["type"],))
                        if m is None:
                            continue
                r.tris, r.model, r.groups = tris, fit.clean(m), r.groups + o.groups
                regions = [x for x in regions if x is not o]
                changed = True
                break
            if changed:
                break
    return regions


def tri_labels(topo, regions):
    L = -np.ones(len(topo.F), np.int64)
    for r in regions:
        L[r.tris] = r.id
    return L


def region_adjacency(topo, L):
    """dict rid -> {nrid: (max_angle, shared_edge_count)}"""
    a, b = L[topo.adj[:, 0]], L[topo.adj[:, 1]]
    out = {}
    m = a != b
    for x, y, ang in zip(a[m], b[m], topo.adj_angle[m]):
        for p, q in ((x, y), (y, x)):
            d = out.setdefault(int(p), {})
            old = d.get(int(q))
            d[int(q)] = (max(ang, old[0]), old[1] + 1) if old else (ang, 1)
    return out


def postprocess(topo, regions, tol, min_area_pct, smooth, max_angle, types=fit.ORDER):
    L = tri_labels(topo, regions)
    adj = region_adjacency(topo, L)
    body_area = np.bincount(topo.body, weights=topo.TA)
    total = topo.TA.sum()
    weak = set()
    for r in regions:
        area = topo.TA[r.tris].sum()
        barea = body_area[topo.body[r.tris[0]]]
        nb = adj.get(r.id, {})
        all_smooth = bool(nb) and all(v[0] < smooth for v in nb.values())
        if r.model is None:
            weak.add(r.id)
        elif area < total * min_area_pct / 100.0:
            weak.add(r.id)
        elif r.kind == "plane" and r.groups == 1 and all_smooth and area < 0.01 * barea:
            weak.add(r.id)
        elif r.kind != "plane" and r.groups < 3:
            weak.add(r.id)
    # absorb weak regions into a neighbouring primitive that already explains them
    byid = {r.id: r for r in regions}
    for w in sorted(weak, key=lambda i: len(byid[i].tris)):
        r = byid[w]
        V, C, N = _region_data(topo, r.tris)
        for n, _ in sorted(adj.get(w, {}).items(), key=lambda x: -len(byid[x[0]].tris) if x[0] in byid else 0):
            host = byid.get(n)
            if host is None or n in weak or host.model is None or host.kind not in fit.ORDER:
                continue
            if fit.check(host.model, V, C, N, tol, max_angle):
                host.tris = np.r_[host.tris, r.tris]
                del byid[w]
                break
    weak = {w for w in weak if w in byid}
    regions = list(byid.values())
    L = tri_labels(topo, regions)
    adj = region_adjacency(topo, L)
    if not weak:
        return regions
    # merge connected weak regions into freeform regions
    from scipy.sparse import coo_matrix
    from scipy.sparse.csgraph import connected_components
    ids = sorted(weak)
    idx = {w: i for i, w in enumerate(ids)}
    rows, cols = [], []
    for w in ids:
        for n in adj.get(w, {}):
            if n in idx:
                rows.append(idx[w]); cols.append(idx[n])
    g = coo_matrix((np.ones(len(rows)), (rows, cols)), shape=(len(ids), len(ids)))
    nc, lab = connected_components(g, directed=False)
    keep = [r for r in regions if r.id not in weak]
    byid = {r.id: r for r in regions}
    for c in range(nc):
        members = [ids[i] for i in np.flatnonzero(lab == c)]
        tris = np.concatenate([byid[m].tris for m in members])
        # a lone small region that is a clean primitive stays a primitive (e.g. tiny chamfer)
        if len(members) == 1 and byid[members[0]].model is not None and byid[members[0]].kind == "plane" \
                and topo.TA[tris].sum() >= total * min_area_pct / 100.0:
            r = byid[members[0]]
            nb = adj.get(r.id, {})
            if not nb or not all(v[0] < smooth for v in nb.values()):
                keep.append(r)
                continue
        # try to explain the merged set by one primitive before calling it freeform
        V, C, N = _region_data(topo, tris)
        m = fit.best_fit(V, C, N, tol, max_angle, topo.scale, types) if len(members) > 1 else None
        if m is not None:
            keep.append(Region(-1, tris, m, m["type"], len(members)))
        else:
            keep.append(Region(-1, tris, None, "freeform", len(members)))
    return keep


# ------------------------------------------------------------------ threads
ISO = [(1.6, .35), (2, .4), (2.5, .45), (3, .5), (3.5, .6), (4, .7), (5, .8), (6, 1.0), (8, 1.25), (10, 1.5),
       (12, 1.75), (14, 2.0), (16, 2.0), (18, 2.5), (20, 2.5), (24, 3.0)]


def thread_params(topo, tris, tol):
    """Return thread model dict if the triangles look like a helical thread."""
    F = topo.F[tris]
    vids = np.unique(F)
    P = topo.V[vids]
    N = topo.TN[tris]
    if len(P) < 60:
        return None
    w, E = np.linalg.eigh(N.T @ N / len(N))
    # surfaces of revolution: two (radial) eigenvalues are equal, the axial one stands apart
    gaps = [min(abs(w[i] - w[j]) for j in range(3) if j != i) for i in range(3)]
    i = int(np.argmax(gaps))
    if gaps[i] < 3 * min(gaps):
        return None
    a = E[:, i]
    u, v = fit._perp_basis(a)
    X = np.c_[P @ u, P @ v]
    c2, _ = fit._fit_circle2d(X)
    origin = c2[0] * u + c2[1] * v
    rel = P - origin
    z = rel @ a
    R = np.linalg.norm(rel - np.outer(z, a), axis=1)
    theta = np.arctan2(rel @ v, rel @ u)
    rmin, rmax = np.percentile(R, 2), np.percentile(R, 98)
    band = rmax - rmin
    L = np.ptp(z)
    if band < 4 * tol or band > 0.4 * rmax or L < 2 * band:
        return None
    # pitch search: for the right pitch & handedness, R is a function of phase
    best = None
    for p in np.linspace(band * 0.9, min(band * 3.5, L / 2), 220):
        for hand in (1, -1):
            ph = np.mod(z - hand * p * theta / (2 * np.pi), p) / p
            bins = np.minimum((ph * 24).astype(int), 23)
            cnt = np.bincount(bins, minlength=24)
            if (cnt == 0).sum() > 6:
                continue
            s = np.bincount(bins, R, 24); s2 = np.bincount(bins, R * R, 24)
            nz = cnt > 0
            var = (s2[nz] / cnt[nz] - (s[nz] / cnt[nz]) ** 2)
            score = np.sqrt(np.maximum(var, 0).mean()) / (R.std() + 1e-12)
            if best is None or score < best[0]:
                best = (score, p, hand)
    if best is None or best[0] > 0.3:
        return None
    _, pitch, hand = best
    # internal threads: normals point toward the axis
    radial = fit._unit(topo.TC[tris] - origin - np.outer((topo.TC[tris] - origin) @ a, a))
    internal = bool((radial * N).sum(1).mean() < 0)
    major, minor = 2 * rmax, 2 * rmin
    nominal = major if not internal else major
    des = None
    for d, pp in ISO:
        if abs(d - nominal) < max(0.12 * d, 0.25) and abs(pp - pitch) < 0.12 * pp + 0.02:
            des = f"M{d:g}×{pp:g}"
            break
    origin = origin + a * z.min()
    a = fit._snap_dir(a)
    return {
        "type": "thread", "origin": [float(x) for x in origin], "axis": [float(x) for x in a],
        "major_diameter": float(major), "minor_diameter": float(minor), "pitch": float(pitch),
        "length": float(L), "internal": internal, "right_hand": bool(hand == 1), "designation": des,
        # exported as a plain cylinder: crest diameter (major for bolts, minor for nuts/tapped holes)
        "radius": float(rmin if internal else rmax),
    }


def detect_threads(topo, regions, tol, enabled=True):
    if not enabled:
        return regions
    L = tri_labels(topo, regions)
    adj = region_adjacency(topo, L)
    body_area = np.bincount(topo.body, weights=topo.TA)
    byid = {r.id: r for r in regions}
    small = set()
    for r in regions:
        if r.kind in ("freeform", "cone", "torus", "plane", "cylinder"):
            a = topo.TA[r.tris].sum()
            if r.kind == "freeform" or a < 0.03 * body_area[topo.body[r.tris[0]]]:
                small.add(r.id)
    # cluster connected small regions
    seen, out_threads, used = set(), [], set()
    for s in small:
        if s in seen:
            continue
        stack, comp = [s], []
        seen.add(s)
        while stack:
            x = stack.pop(); comp.append(x)
            for n in adj.get(x, {}):
                if n in small and n not in seen:
                    seen.add(n); stack.append(n)
        if len(comp) < 4 and not any(byid[c].kind == "freeform" for c in comp):
            continue
        tris = np.concatenate([byid[c].tris for c in comp])
        m = thread_params(topo, tris, tol)
        if m is not None:
            out_threads.append(Region(-1, tris, m, "thread", len(comp)))
            used.update(comp)
    if not out_threads:
        return regions
    return [r for r in regions if r.id not in used] + out_threads


def renumber(regions):
    regions = sorted(regions, key=lambda r: -len(r.tris))
    for i, r in enumerate(regions):
        r.id = i
    return regions
