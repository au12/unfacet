"""Rebuild a B-rep solid (STEP) from a segmented mesh.

Analytic regions become faces on true planes/cylinders/cones/spheres/tori, trimmed by edges that
lie on the intersection of neighbouring surfaces. Freeform (and failed) regions become planar
facets that share their edges with the analytic faces, so every body closes into a solid.
"""
from __future__ import annotations

import numpy as np

from OCP.BRep import BRep_Builder, BRep_Tool
from OCP.BRepBuilderAPI import (BRepBuilderAPI_MakeEdge, BRepBuilderAPI_MakeFace,
                                BRepBuilderAPI_MakeSolid, BRepBuilderAPI_MakeWire, BRepBuilderAPI_Sewing)
from OCP.BRepCheck import BRepCheck_Analyzer
from OCP.BRepGProp import BRepGProp
from OCP.BRepLib import BRepLib
from OCP.GC import GC_MakeArcOfCircle
from OCP.Geom import (Geom_ConicalSurface, Geom_CylindricalSurface, Geom_Plane, Geom_SphericalSurface,
                      Geom_ToroidalSurface)
from OCP.GeomAPI import GeomAPI_PointsToBSpline
from OCP.GeomAbs import GeomAbs_C2
from OCP.GProp import GProp_GProps
from OCP.ShapeFix import ShapeFix_Face, ShapeFix_Shell, ShapeFix_Solid
from OCP.collections import Array1_gp_Pnt as TColgp_Array1OfPnt
from OCP.TopAbs import TopAbs_FACE, TopAbs_SHELL, TopAbs_REVERSED
from OCP.TopExp import TopExp_Explorer
from OCP.TopoDS import TopoDS, TopoDS_Compound, TopoDS_Face, TopoDS_Shell
from OCP.BRepAdaptor import BRepAdaptor_Surface
from OCP.gp import gp_Ax3, gp_Dir, gp_Pln, gp_Pnt

from . import fit

ANALYTIC = set(fit.ORDER) | {"thread"}


def _p(x):
    return gp_Pnt(float(x[0]), float(x[1]), float(x[2]))


def _d(x):
    return gp_Dir(float(x[0]), float(x[1]), float(x[2]))


def export_model(m):
    """Model actually written to the STEP for a region (threads become their crest cylinder)."""
    if m is None:
        return None
    if m["type"] == "thread":
        return {"type": "cylinder", "origin": m["origin"], "axis": m["axis"], "radius": m["radius"]}
    return m


def _seam_dir(axis, center, pts):
    """X direction for the surface frame that puts the parametric seam away from the region."""
    a = np.asarray(axis, float)
    v = pts - center
    v = v - np.outer(v @ a, a)
    n = np.linalg.norm(v, axis=1)
    v = v[n > 1e-12] / n[n > 1e-12, None]
    if len(v) == 0:
        return None
    mdir = v.mean(0)
    if np.linalg.norm(mdir) < 0.05:
        return None  # full revolution: a seam is unavoidable, ShapeFix adds it
    x = -mdir / np.linalg.norm(mdir)
    return x - a * (x @ a)


def _ax3(origin, axis, xdir=None):
    if xdir is None or np.linalg.norm(xdir) < 1e-9:
        return gp_Ax3(_p(origin), _d(axis))
    return gp_Ax3(_p(origin), _d(axis), _d(xdir / np.linalg.norm(xdir)))


def make_surface(m, ref_pts):
    t = m["type"]
    if t == "plane":
        return Geom_Plane(gp_Pln(_p(m["origin"]), _d(m["normal"])))
    if t == "cylinder":
        x = _seam_dir(m["axis"], np.asarray(m["origin"]), ref_pts)
        return Geom_CylindricalSurface(_ax3(m["origin"], m["axis"], x), float(m["radius"]))
    if t == "sphere":
        c = np.asarray(m["center"])
        d = (ref_pts - c).mean(0)
        if np.linalg.norm(d) > 0.05 * m["radius"]:
            d = d / np.linalg.norm(d)
            z = np.cross(d, [1.0, 0, 0] if abs(d[0]) < 0.9 else [0, 1.0, 0])
            z /= np.linalg.norm(z)
            return Geom_SphericalSurface(_ax3(c, z, -d), float(m["radius"]))
        return Geom_SphericalSurface(gp_Ax3(_p(c), gp_Dir(0, 0, 1)), float(m["radius"]))
    if t == "torus":
        x = _seam_dir(m["axis"], np.asarray(m["center"]), ref_pts)
        return Geom_ToroidalSurface(_ax3(m["center"], m["axis"], x), float(m["major"]), float(m["minor"]))
    if t == "cone":
        a = np.asarray(m["axis"]); apex = np.asarray(m["apex"])
        h = float(np.median((ref_pts - apex) @ a))
        h = max(h, 1e-3)
        loc = apex + a * h
        x = _seam_dir(a, apex, ref_pts)
        return Geom_ConicalSurface(_ax3(loc, a, x), float(m["angle"]), float(h * np.tan(m["angle"])))
    raise ValueError(t)


# --------------------------------------------------------------- intersections
def _proj_multi(models, P, iters=30):
    """Move points onto the common intersection of several surfaces (damped Gauss-Newton)."""
    P = np.array(P, float)
    if not models:
        return P
    if len(models) == 1:
        return fit.project(models[0], P)
    for _ in range(iters):
        f = np.stack([fit.distance(m, P) for m in models], 1)            # (n,k)
        J = np.stack([fit.normal_at(m, P) for m in models], 1)           # (n,k,3)
        JJt = np.einsum("nki,nli->nkl", J, J) + 1e-9 * np.eye(len(models))
        lam = np.linalg.solve(JJt, f[..., None])[..., 0]                 # (n,k)
        dP = -np.einsum("nki,nk->ni", J, lam)
        P = P + dP
        if np.abs(dP).max() < 1e-10:
            break
    return P


class Chain:
    __slots__ = ("verts", "key", "pts", "edges", "collapsed", "poly", "closed")

    def __init__(self, verts, key):
        self.verts = verts      # mesh vertex ids, ordered
        self.key = key          # (ra, rb) with rb = -1 for open mesh border
        self.pts = None
        self.edges = None       # list of TopoDS_Edge in chain direction
        self.collapsed = False
        self.poly = False
        self.closed = False


class BuildResult:
    def __init__(self):
        self.shape = None            # TopoDS_Compound of solids/shells
        self.bodies = []             # [(body_id, shape)]
        self.face_region = []        # [(TopoDS_Face, region_id)]
        self.invalid = set()         # region ids with invalid faces
        self.discarded = set()       # region ids rebuilt as facets
        self.closed = {}             # body_id -> bool (solid)
        self.messages = []


def build(V, F, L, models, kinds, body, bodies=None, tol=0.02, progress=None, shell_of=None, slack=None):
    """Build the B-rep. models/kinds indexed by region id. Returns BuildResult."""
    discarded = set()
    for attempt in range(4):
        res = _build_once(V, F, L, models, kinds, body, bodies, tol, discarded,
                          body if shell_of is None else shell_of, slack or {})
        newbad = res._failed - discarded
        if not newbad:
            res.discarded = set(discarded)
            return res
        discarded |= newbad
        if progress:
            progress(0.5 + 0.1 * attempt)
    res.discarded = set(discarded)
    return res


def _build_once(V, F, L, models, kinds, body, bodies, tol, discarded, shell_of, slack):
    res = BuildResult()
    res._failed = set()
    sel = np.ones(len(F), bool) if bodies is None else np.isin(shell_of, list(bodies)) | np.isin(body, list(bodies))
    tri_ids = np.flatnonzero(sel)
    Fs = F[tri_ids]
    Ls = L[tri_ids]
    analytic = {}
    for rid in np.unique(Ls):
        rid = int(rid)
        m = models.get(rid)
        if m is not None and kinds.get(rid) in ANALYTIC and rid not in discarded:
            analytic[rid] = fit.as_np(export_model(m))

    # ---- half-edge maps
    he = np.c_[Fs[:, [0, 1]], Fs[:, [1, 2]], Fs[:, [2, 0]]].reshape(-1, 2)
    he_tri = np.repeat(np.arange(len(Fs)), 3)
    he_lab = Ls[he_tri]
    twin = {}
    for i, (a, b) in enumerate(he):
        twin[(int(a), int(b))] = i
    # boundary half-edges: twin missing or in another region
    n_he = len(he)
    opp_lab = np.full(n_he, -1, np.int64)
    for i, (a, b) in enumerate(he):
        j = twin.get((int(b), int(a)))
        if j is not None:
            opp_lab[i] = he_lab[j]
    is_b = opp_lab != he_lab
    # undirected boundary edges with region-pair keys
    bedges = {}
    for i in np.flatnonzero(is_b):
        a, b = int(he[i, 0]), int(he[i, 1])
        k = (min(a, b), max(a, b))
        la, lb = int(he_lab[i]), int(opp_lab[i])
        key = (min(la, lb), max(la, lb)) if lb >= 0 else (la, -1)
        bedges[k] = key
    # vertex -> incident boundary edges
    vinc = {}
    for (a, b) in bedges:
        vinc.setdefault(a, []).append((a, b))
        vinc.setdefault(b, []).append((a, b))
    # region labels around vertices
    vlabs = {}
    for t, f in enumerate(Fs):
        for v in f:
            vlabs.setdefault(int(v), set()).add(int(Ls[t]))
    corners = set()
    for v, es in vinc.items():
        keys = {bedges[e] for e in es}
        if len(es) != 2 or len(keys) > 1 or len(vlabs[v]) >= 3:
            corners.add(v)

    # ---- chains
    chains = []
    edge_chain = {}
    visited = set()

    def walk(start, e0):
        seq = [start]
        key = bedges[e0]
        e = e0
        cur = start
        while True:
            visited.add(e)
            nxt = e[1] if e[0] == cur else e[0]
            seq.append(nxt)
            if nxt in corners or nxt == start:
                break
            cand = [x for x in vinc[nxt] if x not in visited and bedges[x] == key]
            if not cand:
                break
            e, cur = cand[0], nxt
        return seq, key

    for c in corners:
        for e in vinc[c]:
            if e not in visited:
                seq, key = walk(c, e)
                chains.append(Chain(seq, key))
    for e in list(bedges):
        if e not in visited:          # closed loop without corners: split in two halves
            seq, key = walk(e[0], e)
            if seq[-1] == seq[0] and len(seq) > 3:
                h = len(seq) // 2
                corners.add(seq[0]); corners.add(seq[h])
                chains.append(Chain(seq[:h + 1], key))
                chains.append(Chain(seq[h:], key))
            else:
                chains.append(Chain(seq, key))
    for ci, ch in enumerate(chains):
        for i in range(len(ch.verts) - 1):
            a, b = ch.verts[i], ch.verts[i + 1]
            edge_chain[(min(a, b), max(a, b))] = (ci, i)

    # ---- vertex positions on the intersection of surrounding surfaces
    pos = {}
    base_move = max(10 * tol, 0.05)
    thread_move = {r: (models[r]["major_diameter"] - models[r]["minor_diameter"]) / 2 + base_move
                   for r in analytic if kinds.get(r) == "thread"}
    for r, extra in slack.items():
        if r in analytic:
            thread_move[r] = max(thread_move.get(r, 0), base_move + extra)

    def allowed_move(labels):
        return max([base_move] + [thread_move[r] for r in labels if r in thread_move])
    for v in corners:
        ms = [analytic[r] for r in vlabs[v] if r in analytic]
        q = _proj_multi(ms, V[[v]])[0]
        max_move = allowed_move(vlabs[v])
        if not np.all(np.isfinite(q)) or np.linalg.norm(q - V[v]) > max_move:
            # surfaces do not meet here; settle for the best single projection
            q = V[v] if not ms else fit.project(ms[0], V[[v]])[0]
            if np.linalg.norm(q - V[v]) > max_move:
                q = V[v]
            for r in vlabs[v]:
                if r in analytic and abs(fit.distance(analytic[r], q[None])[0]) > tol * 2:
                    res._failed.add(r)
        pos[v] = q
    for ch in chains:
        ms = [analytic[r] for r in ch.key if r in analytic]
        inner = ch.verts[1:-1]
        if inner:
            Q = _proj_multi(ms, V[inner])
            bad = ~np.all(np.isfinite(Q), 1) | (np.linalg.norm(Q - V[inner], axis=1) > allowed_move(ch.key))
            if bad.any():
                for r in ch.key:
                    if r in analytic:
                        res._failed.add(r)
                Q[bad] = V[np.asarray(inner)[bad]]
            for v, q in zip(inner, Q):
                pos[v] = q
        ch.pts = np.array([pos[v] for v in ch.verts])
        ch.poly = any(r not in analytic for r in ch.key if r >= 0) or ch.key[1] < 0 and ch.key[0] not in analytic

    # ---- collapse tiny chains (several mesh corners mapping to one CAD vertex)
    parent = {}

    def find(x):
        while parent.get(x, x) != x:
            parent[x] = parent.get(parent[x], parent[x])
            x = parent[x]
        return x

    for ch in chains:
        L_ = np.linalg.norm(np.diff(ch.pts, axis=0), axis=1).sum()
        if L_ < max(tol * 0.5, 1e-4) and ch.verts[0] != ch.verts[-1]:
            ch.collapsed = True
            a, b = find(ch.verts[0]), find(ch.verts[-1])
            if a != b:
                parent[b] = a
    groups = {}
    for v in corners:
        groups.setdefault(find(v), []).append(v)
    for root, members in groups.items():
        if len(members) > 1:
            labs = set().union(*[vlabs[m] for m in members])
            ms = [analytic[r] for r in labs if r in analytic]
            q = _proj_multi(ms, np.mean([pos[m] for m in members], 0)[None])[0]
            if not np.all(np.isfinite(q)):
                q = np.mean([pos[m] for m in members], 0)
            for m in members:
                pos[m] = q
    for ch in chains:
        ch.pts[0] = pos[ch.verts[0]]
        ch.pts[-1] = pos[ch.verts[-1]]

    # ---- topological vertices & edges
    B = BRep_Builder()
    tvert = {}

    def tv(v):
        r = find(v) if v in corners else v
        if r not in tvert:
            p = pos.get(r, V[r]) if r in pos else V[r]
            tvert[r] = BRepLib_vertex(p, tol)
        return tvert[r]

    seg_edge = {}

    def segment_edge(a, b):
        k = (min(a, b), max(a, b))
        ra = find(a) if a in corners else a
        rb = find(b) if b in corners else b
        if ra == rb:
            return "SKIP"
        if k not in seg_edge:
            pa, pb = pos.get(k[0], V[k[0]]), pos.get(k[1], V[k[1]])
            mk = BRepBuilderAPI_MakeEdge(tv(k[0]), tv(k[1]))
            if not mk.IsDone():
                seg_edge[k] = None
            else:
                seg_edge[k] = mk.Edge()
        return seg_edge[k]

    tol_e = max(tol * 0.25, 1e-4)
    for ch in chains:
        if ch.collapsed:
            continue
        if ch.poly or len(ch.pts) == 2:
            ch.edges = None  # built per segment
            continue
        e = _curve_edge(ch.pts, tv(ch.verts[0]), tv(ch.verts[-1]), tol_e)
        if e is None:
            ch.poly = True
        else:
            ch.edges = [e]

    # ---- faces
    faces_by_body = {}
    # oriented boundary loops per region
    nxt = {}
    for i in np.flatnonzero(is_b):
        nxt.setdefault((int(he_lab[i]), int(he[i, 0])), []).append(i)
    used = np.zeros(n_he, bool)
    loops = {}
    for i in np.flatnonzero(is_b):
        if used[i]:
            continue
        r = int(he_lab[i])
        loop = []
        j = i
        guard = 0
        while not used[j] and guard < n_he:
            used[j] = True
            loop.append(j)
            cand = [k for k in nxt.get((r, int(he[j, 1])), []) if not used[k]]
            if not cand:
                break
            j = cand[0]
            guard += 1
        loops.setdefault(r, []).append(loop)

    region_tris = {}
    for t, l in enumerate(Ls):
        region_tris.setdefault(int(l), []).append(t)

    for r, tris in region_tris.items():
        bid = int(body[tri_ids[tris[0]]])
        comp_id = int(shell_of[tri_ids[tris[0]]])
        if r in analytic:
            face = _analytic_face(r, analytic[r], loops.get(r, []), he, chains, edge_chain, segment_edge,
                                  corners, V, Fs[tris], tol, check_area=kinds.get(r) != "thread")
            if face is None:
                res._failed.add(r)
                continue
            faces_by_body.setdefault(bid, []).append((face, r, comp_id))
        else:
            for t in tris:
                a, b, c = (int(x) for x in Fs[t])
                es = [segment_edge(a, b), segment_edge(b, c), segment_edge(c, a)]
                if any(e is None or isinstance(e, str) for e in es):
                    continue
                w = BRepBuilderAPI_MakeWire()
                for e in es:
                    w.Add(e)
                if not w.IsDone():
                    continue
                mf = BRepBuilderAPI_MakeFace(w.Wire(), True)
                if mf.IsDone():
                    faces_by_body.setdefault(bid, []).append((mf.Face(), r, comp_id))
    if res._failed - discarded:
        return res

    # ---- shells & solids
    comp = TopoDS_Compound()
    B.MakeCompound(comp)
    for bid, flist in faces_by_body.items():
        shape, closed, mapping = _close_body(flist, tol)
        res.closed[bid] = closed
        res.bodies.append((bid, shape))
        res.face_region.extend(mapping)
        B.Add(comp, shape)
    res.shape = comp
    for f, r in res.face_region:
        if not BRepCheck_Analyzer(f).IsValid():
            res.invalid.add(r)
    return res


def BRepLib_vertex(p, tol):
    from OCP.TopoDS import TopoDS_Vertex
    v = TopoDS_Vertex()
    BRep_Builder().MakeVertex(v, _p(p), tol * 0.5)
    return v


def _curve_edge(P, v1, v2, tol_e):
    """Line, circular arc or B-spline through the chain points, bounded by the given vertices."""
    p0, p1 = P[0], P[-1]
    d = p1 - p0
    L = np.linalg.norm(d)
    if L > 1e-9:
        u = d / L
        off = P - p0
        dev = np.linalg.norm(off - np.outer(off @ u, u), axis=1).max()
        if dev < tol_e:
            mk = BRepBuilderAPI_MakeEdge(v1, v2)
            return mk.Edge() if mk.IsDone() else None
    if len(P) >= 3:
        # circle?
        pl = fit.fit_plane(P)
        if np.abs(fit.distance(pl, P)).max() < tol_e:
            uu, vv = fit._perp_basis(pl["normal"])
            X = np.c_[(P - pl["origin"]) @ uu, (P - pl["origin"]) @ vv]
            c2, R = fit._fit_circle2d(X)
            if np.abs(np.linalg.norm(X - c2, axis=1) - R).max() < tol_e and R < 1e5:
                mid = P[len(P) // 2]
                try:
                    arc = GC_MakeArcOfCircle(_p(p0), _p(mid), _p(p1))
                    if arc.IsDone():
                        mk = BRepBuilderAPI_MakeEdge(arc.Value(), v1, v2)
                        if mk.IsDone():
                            return mk.Edge()
                except Exception:  # noqa: BLE001
                    pass
    # B-spline
    pts = P
    if len(pts) > 200:
        idx = np.unique(np.r_[np.linspace(0, len(pts) - 1, 200).astype(int), len(pts) - 1])
        pts = pts[idx]
    arr = TColgp_Array1OfPnt(1, len(pts))
    for i, q in enumerate(pts):
        arr.SetValue(i + 1, _p(q))
    try:
        api = GeomAPI_PointsToBSpline(arr, 2, 8, GeomAbs_C2, tol_e * 0.5)
        if not api.IsDone():
            return None
        mk = BRepBuilderAPI_MakeEdge(api.Curve(), v1, v2)
        return mk.Edge() if mk.IsDone() else None
    except Exception:  # noqa: BLE001
        return None


WHY = {}


def _fail(r, why):
    WHY[r] = why
    return None


def _analytic_face(r, m, loops, he, chains, edge_chain, segment_edge, corners, V, tris, tol, check_area=True):
    region_pts = V[np.unique(tris)]
    try:
        surf = make_surface(m, region_pts)
    except Exception:  # noqa: BLE001
        return _fail(r, "surface")
    if not loops:
        # closed surface without boundary (full sphere / torus): natural bounds
        mf = BRepBuilderAPI_MakeFace(surf, tol * 0.1)
        if not mf.IsDone():
            return _fail(r, "natural")
        face = mf.Face()
        C = V[tris].mean(1)
        Nm = np.cross(V[tris[:, 1]] - V[tris[:, 0]], V[tris[:, 2]] - V[tris[:, 0]])
        if float((fit.normal_at(m, C) * Nm).sum()) < 0:
            face = TopoDS.Face(face.Reversed())
        return face
    wires = []
    for loop in loops:
        # rotate loop to start at a corner
        starts = [k for k, j in enumerate(loop) if int(he[j, 0]) in corners]
        if not starts:
            return _fail(r, "nocorner")
        loop = loop[starts[0]:] + loop[:starts[0]]
        edges = []
        k = 0
        while k < len(loop):
            a, b = int(he[loop[k], 0]), int(he[loop[k], 1])
            ci, idx = edge_chain[(min(a, b), max(a, b))]
            ch = chains[ci]
            fwd = ch.verts[idx] == a
            # consume all half-edges of this chain
            n = len(ch.verts) - 1
            if ch.collapsed:
                k += n
                continue
            if ch.edges is None:
                for _ in range(n):
                    a2, b2 = int(he[loop[k], 0]), int(he[loop[k], 1])
                    e = segment_edge(a2, b2)
                    if e is None:
                        return _fail(r, "segnone")
                    if e == "SKIP":
                        k += 1
                        continue
                    edges.append(e if _starts_at(e, a2, b2) else e.Reversed())
                    k += 1
            else:
                e = ch.edges[0]
                edges.append(e if fwd else e.Reversed())
                k += n
        if not edges:
            continue
        mw = BRepBuilderAPI_MakeWire()
        for e in edges:
            mw.Add(TopoDS.Edge(e))
        if not mw.IsDone():
            return _fail(r, "segnone2")
        wires.append(mw.Wire())
    if not wires:
        return _fail(r, "wire")
    B = BRep_Builder()
    face = TopoDS_Face()
    # mesh loops run counter-clockwise around the outward mesh normal; OCC wants them
    # counter-clockwise around the surface normal, so flip when the two disagree
    C = V[tris].mean(1)
    Nm = np.cross(V[tris[:, 1]] - V[tris[:, 0]], V[tris[:, 2]] - V[tris[:, 0]])
    agree = float((fit.normal_at(m, C) * Nm).sum()) >= 0
    B.MakeFace(face, surf, tol * 0.1)
    for w in wires:
        B.Add(face, w if agree else w.Reversed())
    try:
        sf = ShapeFix_Face(face)
        sf.SetPrecision(tol * 0.1)
        sf.SetMaxTolerance(max(tol * 5, 0.1))
        sf.Perform()
        face = sf.Face()
        if not BRepCheck_Analyzer(face).IsValid():
            sf2 = ShapeFix_Face(face)
            sf2.SetPrecision(tol * 0.1)
            sf2.SetMaxTolerance(max(tol * 5, 0.1))
            sf2.FixMissingSeam()
            sf2.FixOrientation()
            face = sf2.Face()
        if not agree:
            face = TopoDS.Face(face.Reversed())
    except Exception:  # noqa: BLE001
        return _fail(r, "nowires")
    if face.IsNull() or not BRepCheck_Analyzer(face).IsValid():
        alt = _band_face(m, surf, loops, he, V, agree, tol)
        if alt is None:
            return _fail(r, "invalid-face " + _check_detail(face))
        face = alt
    # area sanity: a mis-trimmed face (e.g. the complement of a sphere cap) gives itself away
    props = GProp_GProps()
    BRepGProp.SurfaceProperties_s(face, props)
    mesh_area = 0.5 * np.linalg.norm(np.cross(V[tris[:, 1]] - V[tris[:, 0]], V[tris[:, 2]] - V[tris[:, 0]]), axis=1).sum()
    if check_area and mesh_area > 0 and not (0.7 < abs(props.Mass()) / mesh_area < 1.3):
        return _fail(r, f"area {props.Mass():.2f} vs {mesh_area:.2f} wires {len(wires)}")
    return face


def _band_face(m, surf, loops, he, V, agree, tol):
    """Full-revolution band between two axial-perpendicular circles: build it parametrically."""
    if m["type"] not in ("cylinder", "cone") or len(loops) != 2:
        return None
    from OCP.ShapeAnalysis import ShapeAnalysis_Surface
    sas = ShapeAnalysis_Surface(surf)
    a = np.asarray(m["axis"])
    vs = []
    for loop in loops:
        P = V[[int(he[j, 0]) for j in loop]]
        h = P @ a
        if np.ptp(h) > max(tol * 2, 1e-3):
            return None
        q = P[0]
        uv = sas.ValueOfUV(_p(q), tol)
        vs.append(uv.Y())
    v0, v1 = sorted(vs)
    if v1 - v0 < 1e-6:
        return None
    mf = BRepBuilderAPI_MakeFace(surf, 0.0, 2 * np.pi, v0, v1, tol * 0.1)
    if not mf.IsDone():
        return None
    face = mf.Face()
    if not agree:
        face = TopoDS.Face(face.Reversed())
    return face if BRepCheck_Analyzer(face).IsValid() else None


def _check_detail(face):
    from OCP.BRepCheck import BRepCheck_Analyzer
    from OCP.TopAbs import TopAbs_EDGE, TopAbs_WIRE
    out = []
    from OCP.ShapeAnalysis import ShapeAnalysis_Wire
    from OCP.BRepTools import BRepTools
    out = []
    exw = TopExp_Explorer(face, TopAbs_WIRE)
    i = 0
    while exw.More():
        w = TopoDS.Wire(exw.Current())
        saw = ShapeAnalysis_Wire(w, face, 1e-4)
        flags = []
        for name in ("CheckOrder", "CheckConnected", "CheckSmall", "CheckDegenerated", "CheckSelfIntersection",
                     "CheckLacking", "CheckClosed", "CheckGaps3d", "CheckGaps2d", "CheckCurveGaps"):
            try:
                if getattr(saw, name)():
                    flags.append(name[5:])
            except Exception:
                pass
        out.append(f"w{i}:" + "/".join(flags) + f" valid={BRepCheck_Analyzer(w).IsValid()}")
        exw.Next(); i += 1
    exe = TopExp_Explorer(face, TopAbs_EDGE)
    bad = 0
    while exe.More():
        if not BRepCheck_Analyzer(exe.Current()).IsValid():
            bad += 1
        exe.Next()
    out.append(f"badedges={bad}")
    try:
        from OCP.BRepCheck import BRepCheck_Face
        cf = BRepCheck_Face(face)
        out.append("iw=" + str(cf.IntersectWires()).split(".")[-1])
        out.append("cw=" + str(cf.ClassifyWires()).split(".")[-1])
        out.append("ow=" + str(cf.OrientationOfWires()).split(".")[-1])
    except Exception as e:
        out.append(str(e)[:80])
    return " ".join(out)
    an = BRepCheck_Analyzer(face)
    for kind in (TopAbs_FACE, TopAbs_WIRE, TopAbs_EDGE):
        ex = TopExp_Explorer(face, kind)
        while ex.More():
            res = an.Result(ex.Current())
            if res is not None:
                for st in res.Status():
                    if str(st) != "BRepCheck_Status.BRepCheck_NoError":
                        out.append(str(kind).split("_")[-1] + ":" + str(st).split("_", 1)[-1])
            ex.Next()
    return ",".join(sorted(set(out)))[:200]


def _starts_at(e, a, b):
    # segment edges are created from (min,max); orientation relative to (a,b)
    return a < b


def _shell_from(faces, tol):
    B = BRep_Builder()
    shell = TopoDS_Shell()
    B.MakeShell(shell)
    for f in faces:
        B.Add(shell, f)
    try:
        fs = ShapeFix_Shell(shell)
        fs.Perform()
        shell = fs.Shell()
    except Exception:  # noqa: BLE001
        pass
    return shell


def _try_solid(shells):
    """Outer shell first, then inner (cavity) shells."""
    try:
        ms = BRepBuilderAPI_MakeSolid(TopoDS.Shell(shells[0]))
        for sh in shells[1:]:
            ms.Add(TopoDS.Shell(sh))
        if not ms.IsDone():
            return None
        sf = ShapeFix_Solid(ms.Solid())
        sf.Perform()
        sol = sf.Solid()
        if BRepCheck_Analyzer(sol).IsValid():
            props = GProp_GProps()
            BRepGProp.VolumeProperties_s(sol, props)
            if props.Mass() > 0:
                return sol
    except Exception:  # noqa: BLE001
        pass
    return None


def _bbox_size(shape):
    from OCP.Bnd import Bnd_Box
    from OCP.BRepBndLib import BRepBndLib
    b = Bnd_Box()
    BRepBndLib.Add_s(shape, b)
    if b.IsVoid():
        return 0.0
    p0, p1 = b.CornerMin(), b.CornerMax()
    return (p1.X() - p0.X()) * (p1.Y() - p0.Y()) * (p1.Z() - p0.Z())


def _free_wires(shape):
    from OCP.ShapeAnalysis import ShapeAnalysis_FreeBounds
    from OCP.TopAbs import TopAbs_EDGE
    fb = ShapeAnalysis_FreeBounds(shape)
    n = 0
    for w in (fb.GetClosedWires(), fb.GetOpenWires()):
        ex = TopExp_Explorer(w, TopAbs_EDGE)
        while ex.More():
            n += 1
            ex.Next()
    return n


def _comp_shell(faces, tol):
    """One closed shell per mesh component; sew only if the shared-edge build left gaps."""
    sh = _shell_from(faces, tol)
    if _free_wires(sh) == 0:
        return sh, None
    sew = BRepBuilderAPI_Sewing(max(tol * 2, 1e-3))
    for f in faces:
        sew.Add(f)
    sew.Perform()
    sewed = sew.SewedShape()
    best, nbest = None, -1
    ex = TopExp_Explorer(sewed, TopAbs_SHELL)
    while ex.More():
        cnt = 0
        ef = TopExp_Explorer(ex.Current(), TopAbs_FACE)
        while ef.More():
            cnt += 1
            ef.Next()
        if cnt > nbest:
            best, nbest = ex.Current(), cnt
        ex.Next()
    if best is None:
        return sh, None
    fs = ShapeFix_Shell(TopoDS.Shell(best))
    fs.Perform()
    return fs.Shell(), sew


def _close_body(flist, tol):
    """Shells -> solid (outer shell + cavity shells). Returns (shape, is_solid, [(face, region)])."""
    comps = {}
    for f, r, c in flist:
        comps.setdefault(c, []).append(f)
    sewers = []
    shells = []
    for fs in comps.values():
        sh, sew = _comp_shell(fs, tol)
        shells.append(sh)
        if sew is not None:
            sewers.append(sew)
    shells.sort(key=_bbox_size, reverse=True)
    mapping = []
    for f, r, _ in flist:
        for sew in sewers:
            f = _modified(sew, f)
        mapping.append((f, r))
    sol = _try_solid(shells)
    if sol is None and len(shells) > 1:
        sol = _try_solid(shells[:1])
    if sol is not None:
        shape, closed = sol, True
    else:
        comp = TopoDS_Compound()
        B = BRep_Builder()
        B.MakeCompound(comp)
        for sh in shells:
            B.Add(comp, sh)
        shape, closed = comp, False
    mapping = _remap_faces(shape, mapping)
    return shape, closed, mapping


def _modified(sew, f):
    try:
        if sew.IsModified(f):
            return TopoDS.Face(sew.Modified(f))
    except Exception:  # noqa: BLE001
        pass
    return f


def _centroid(f):
    props = GProp_GProps()
    BRepGProp.SurfaceProperties_s(f, props)
    c = props.CentreOfMass()
    return np.array([c.X(), c.Y(), c.Z()]), props.Mass()


def _fkey(f):
    from OCP.TopAbs import TopAbs_FORWARD
    return hash(f.Oriented(TopAbs_FORWARD))


def _remap_faces(shape, mapping):
    """Map faces of the final shape back to regions: identity first, centroid lookup for rebuilt ones."""
    from scipy.spatial import cKDTree
    by_hash = {}
    for f, r in mapping:
        by_hash.setdefault(_fkey(f), r)
    out, missing = [], []
    ex = TopExp_Explorer(shape, TopAbs_FACE)
    while ex.More():
        f = TopoDS.Face(ex.Current())
        r = by_hash.get(_fkey(f))
        if r is None:
            missing.append(f)
        else:
            out.append((f, r))
        ex.Next()
    if missing:
        pts, regs = [], []
        for f, r in mapping:
            try:
                pts.append(_centroid(f)[0]); regs.append(r)
            except Exception:  # noqa: BLE001
                pass
        if pts:
            tree = cKDTree(np.array(pts))
            for f in missing:
                try:
                    _, i = tree.query(_centroid(f)[0])
                    out.append((f, regs[int(i)]))
                except Exception:  # noqa: BLE001
                    pass
    return out


def tessellate(res, lin=None):
    """Triangulate the built B-rep for preview: returns (V float32, F uint32, face_region int32)."""
    from OCP.BRepMesh import BRepMesh_IncrementalMesh
    from OCP.TopLoc import TopLoc_Location
    if res.shape is None:
        return None
    from OCP.Bnd import Bnd_Box
    from OCP.BRepBndLib import BRepBndLib
    b = Bnd_Box()
    BRepBndLib.Add_s(res.shape, b)
    p0, p1 = b.CornerMin(), b.CornerMax()
    diag = float(np.linalg.norm([p1.X() - p0.X(), p1.Y() - p0.Y(), p1.Z() - p0.Z()]))
    BRepMesh_IncrementalMesh(res.shape, lin or max(diag * 4e-4, 1e-3), False, 0.25, True)
    Vs, Fs, Rs = [], [], []
    off = 0
    reg = {}
    for f, r in res.face_region:
        reg.setdefault(_fkey(f), r)
    ex = TopExp_Explorer(res.shape, TopAbs_FACE)
    while ex.More():
        face = TopoDS.Face(ex.Current())
        loc = TopLoc_Location()
        tri = BRep_Tool.Triangulation_s(face, loc)
        if tri is not None:
            tr = loc.Transformation()
            n = tri.NbNodes()
            P = np.empty((n, 3))
            for i in range(1, n + 1):
                q = tri.Node(i).Transformed(tr)
                P[i - 1] = (q.X(), q.Y(), q.Z())
            m = tri.NbTriangles()
            T = np.empty((m, 3), np.int64)
            rev = face.Orientation() == TopAbs_REVERSED
            for i in range(1, m + 1):
                a, b_, c = tri.Triangle(i).Get()
                T[i - 1] = (a - 1, c - 1, b_ - 1) if rev else (a - 1, b_ - 1, c - 1)
            r = reg.get(_fkey(face), -1)
            Vs.append(P); Fs.append(T + off); Rs.append(np.full(m, r))
            off += n
        ex.Next()
    if not Vs:
        return None
    return (np.vstack(Vs).astype(np.float32), np.vstack(Fs).astype(np.uint32), np.concatenate(Rs).astype(np.int32))
