"""Mesh repair: open-edge gaps and enclosed cavities."""
from __future__ import annotations

import numpy as np
import trimesh

from . import fit
from .mesh import Topo


# ------------------------------------------------------------------ gaps
def boundary_loops(F):
    """Ordered loops of boundary half-edges (a->b with no b->a)."""
    he = np.c_[F[:, [0, 1]], F[:, [1, 2]], F[:, [2, 0]]].reshape(-1, 2)
    s = set(map(tuple, he.tolist()))
    bnd = [(a, b) for a, b in he.tolist() if (b, a) not in s]
    nxt = {}
    for a, b in bnd:
        nxt.setdefault(a, []).append(b)
    used = set()
    loops = []
    for a, b in bnd:
        if (a, b) in used:
            continue
        loop = [a]
        cur = (a, b)
        while cur not in used:
            used.add(cur)
            loop.append(cur[1])
            cand = [c for c in nxt.get(cur[1], []) if (cur[1], c) not in used]
            if not cand:
                break
            cur = (cur[1], cand[0])
        if loop[-1] == loop[0]:
            loop = loop[:-1]
        if len(loop) >= 3:
            loops.append(loop)
    return loops


def open_edges(F):
    """Undirected open edges as (n,2) vertex pairs."""
    e = np.sort(np.r_[F[:, [0, 1]], F[:, [1, 2]], F[:, [2, 0]]], axis=1)
    u, c = np.unique(e, axis=0, return_counts=True)
    return u[c == 1]


def close_gaps(topo: Topo, regions, tol):
    """Fill every boundary loop with a patch that follows the surrounding surface.

    Returns (new_trimesh, new_triangle_region_hints) where hints is a list of
    (new_tri_indices, model_or_None).
    """
    V = topo.V.copy()
    F = topo.F.copy()
    loops = boundary_loops(F)
    if not loops:
        return None, []
    newV, newF, hints = [V], [F], []
    nv = len(V)
    nf = len(F)
    # vertex -> faces for neighbourhood lookup
    vf = {}
    for t, f in enumerate(F):
        for v in f:
            vf.setdefault(int(v), []).append(t)
    for loop in loops:
        L = np.asarray(loop)
        P = V[L]
        # neighbourhood: faces around the loop, two rings
        ring = set()
        for v in loop:
            ring.update(vf.get(int(v), []))
        ring2 = set(ring)
        for t in ring:
            for v in F[t]:
                ring2.update(vf.get(int(v), []))
        ring2 = np.array(sorted(ring2))
        model = None
        if len(ring2) >= 4:
            vids = np.unique(F[ring2])
            model = fit.best_fit(V[vids], topo.TC[ring2], topo.TN[ring2], max(tol * 3, 0.05),
                                 np.radians(40), topo.scale)
        c = P.mean(0)
        n = len(L)
        K = max(1, min(8, n // 6))
        rings = [L]
        pts_all = []
        for k in range(1, K + 1):
            s_ = 1 - k / (K + 1)
            Q = c + (P - c) * s_
            if model is not None:
                Q = fit.project(fit.as_np(model), Q)
            idx = np.arange(nv, nv + n)
            nv += n
            pts_all.append(Q)
            rings.append(idx)
        cen = c if model is None else fit.project(fit.as_np(model), c[None])[0]
        cidx = nv
        nv += 1
        pts_all.append(cen[None])
        tris = []
        for k in range(len(rings) - 1):
            A, Bv = rings[k], rings[k + 1]
            for i in range(n):
                j = (i + 1) % n
                # loop runs along boundary half-edges a->b; patch must contain b->a
                tris.append([A[j], A[i], Bv[i]])
                tris.append([A[j], Bv[i], Bv[j]])
        last = rings[-1]
        for i in range(n):
            j = (i + 1) % n
            tris.append([last[j], last[i], cidx])
        newV.append(np.vstack(pts_all))
        newF.append(np.asarray(tris, np.int64))
        hints.append((np.arange(nf, nf + len(tris)), model))
        nf += len(tris)
    m = trimesh.Trimesh(np.vstack(newV), np.vstack(newF), process=False)
    return m, hints


# --------------------------------------------------------------- cavities
def _winding(P, V, F):
    """Generalised winding numbers of points P w.r.t. triangle soup (V,F)."""
    out = np.zeros(len(P))
    A, B, C = V[F[:, 0]], V[F[:, 1]], V[F[:, 2]]
    for i, p in enumerate(P):
        a, b, c = A - p, B - p, C - p
        la, lb, lc = (np.linalg.norm(x, axis=1) for x in (a, b, c))
        det = np.einsum("ij,ij->i", a, np.cross(b, c))
        den = la * lb * lc + np.einsum("ij,ij->i", a, b) * lc + np.einsum("ij,ij->i", b, c) * la \
            + np.einsum("ij,ij->i", c, a) * lb
        out[i] = 2 * np.arctan2(det, den).sum() / (4 * np.pi)
    return out


def find_slivers(topo: Topo, tol):
    """Paper-thin closed bodies (leftover modifier boxes, zero-thickness plates): {body: thickness}."""
    out = {}
    for b in range(topo.n_bodies):
        tris = np.flatnonzero(topo.body == b)
        area = float(topo.TA[tris].sum())
        if area <= 0:
            continue
        thick = 2 * abs(component_volume(topo, tris)) / area
        if thick < max(tol, 0.01):
            out[b] = thick
    return out


def find_cavities(topo: Topo, skip=()):
    """Return {component: parent_component} for components enclosed by another component."""
    n = topo.n_bodies
    if n < 2:
        return {}
    skip = set(skip)
    comps = [np.flatnonzero(topo.body == b) for b in range(n)]
    bbox = []
    for tris in comps:
        P = topo.V[np.unique(topo.F[tris])]
        bbox.append((P.min(0), P.max(0)))
    out = {}
    order = sorted(range(n), key=lambda b: -np.prod(bbox[b][1] - bbox[b][0]))
    for b in range(n):
        if b in skip:
            continue
        lo, hi = bbox[b]
        best = None
        for o in order:
            if o == b or o in skip:
                continue
            olo, ohi = bbox[o]
            if np.all(olo <= lo + 1e-9) and np.all(ohi >= hi - 1e-9):
                Fo = topo.F[comps[o]]
                # sample a few points slightly off the surface (centroids)
                tris = comps[b][:: max(1, len(comps[b]) // 5)][:5]
                P = topo.TC[tris]
                w = _winding(P, topo.V, Fo)
                if np.all(np.abs(w) > 0.5):
                    if best is None or np.prod(bbox[o][1] - bbox[o][0]) < np.prod(bbox[best][1] - bbox[best][0]):
                        best = o
        if best is not None:
            out[b] = best
    return out


def component_volume(topo: Topo, tris):
    V, F = topo.V, topo.F[tris]
    return float(np.einsum("ij,ij->i", V[F[:, 0]], np.cross(V[F[:, 1]], V[F[:, 2]])).sum() / 6)
