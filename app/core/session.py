"""Editable conversion session: mesh + face regions + history, and every face/mesh operation."""
from __future__ import annotations

import io
import threading
import time
import uuid

import numpy as np
import trimesh

from . import brep, fit, repair
from .mesh import Topo, clean
from .segment import (Region, _region_data, region_adjacency, renumber, same_surface, segment, thread_params,
                      tri_labels)

KINDS = ("plane", "cylinder", "cone", "sphere", "torus", "thread", "freeform")

# ISO metric data: nominal, coarse pitch, clearance (ISO 273 medium), tap drill, plastic (screw into print)
ISO_HOLES = [
    (2, .40, 2.4, 1.6, 1.7), (2.5, .45, 2.9, 2.05, 2.15), (3, .50, 3.4, 2.5, 2.6), (4, .70, 4.5, 3.3, 3.4),
    (5, .80, 5.5, 4.2, 4.3), (6, 1.0, 6.6, 5.0, 5.2), (8, 1.25, 9.0, 6.8, 7.0), (10, 1.5, 11.0, 8.5, 8.8),
    (12, 1.75, 13.5, 10.2, 10.5), (14, 2.0, 15.5, 12.0, 12.3), (16, 2.0, 17.5, 14.0, 14.2),
]
PURPOSES = ("clearance", "tap", "plastic")


def iso_fit(diameter, slack=0.12):
    best = None
    for row in ISO_HOLES:
        for k, purpose in enumerate(PURPOSES):
            d = row[2 + k]
            err = abs(d - diameter)
            if err <= slack and (best is None or err < best[0]):
                best = (err, f"M{row[0]:g}", purpose, d)
    return None if best is None else {"size": best[1], "purpose": best[2], "diameter": best[3]}


class OpError(Exception):
    pass


class Snapshot:
    __slots__ = ("topo", "regions", "next_id", "label")

    def __init__(self, topo, regions, next_id, label):
        self.topo, self.regions, self.next_id, self.label = topo, regions, next_id, label


class Session:
    def __init__(self, mesh: trimesh.Trimesh, filename: str):
        self.id = uuid.uuid4().hex[:16]
        self.filename = filename
        self.created = self.touched = time.time()
        self.lock = threading.RLock()
        self.topo = Topo(mesh)
        self.regions: list[Region] = []
        self.next_id = 0
        self.params = {"tolerance": 0.02, "min_face_pct": 0.0, "angle": 35.0,
                       "types": list(KINDS[:-1])}
        self.analyzed = False
        self.body_names = {}
        self.body_selected = None   # None = all
        self.undo_stack: list[Snapshot] = []
        self.redo_stack: list[Snapshot] = []
        self.build_cache = None     # (version, BuildResult)
        self.version = 0
        self.cavities = repair.find_cavities(self.topo)

    # ------------------------------------------------------------ history
    def _snapshot(self, label):
        return Snapshot(self.topo, [r.copy() for r in self.regions], self.next_id, label)

    def checkpoint(self, label):
        self.undo_stack.append(self._snapshot(label))
        if len(self.undo_stack) > 60:
            self.undo_stack.pop(0)
        self.redo_stack.clear()

    def _restore(self, s: Snapshot):
        if s.topo is not self.topo:
            self.topo = s.topo
            self.cavities = repair.find_cavities(self.topo)
        self.regions = [r.copy() for r in s.regions]
        self.next_id = s.next_id
        self.changed()

    def undo(self):
        if not self.undo_stack:
            raise OpError("Nothing to undo.")
        s = self.undo_stack.pop()
        self.redo_stack.append(self._snapshot(s.label))
        self._restore(s)
        return s.label

    def redo(self):
        if not self.redo_stack:
            raise OpError("Nothing to redo.")
        s = self.redo_stack.pop()
        self.undo_stack.append(self._snapshot(s.label))
        self._restore(s)
        return s.label

    def changed(self):
        self.version += 1
        self.build_cache = None

    # ------------------------------------------------------------ helpers
    @property
    def tol(self):
        return float(self.params["tolerance"])

    @property
    def max_angle(self):
        return np.radians(float(self.params.get("angle", 35.0)))

    def region(self, rid) -> Region:
        for r in self.regions:
            if r.id == rid:
                return r
        raise OpError("That face no longer exists. Refresh and try again.")

    def new_region(self, tris, model, kind=None):
        r = Region(self.next_id, tris, model, kind or (model["type"] if model else "freeform"))
        self.next_id += 1
        return r

    def labels(self):
        return tri_labels(self.topo, self.regions)

    def adjacency(self):
        return region_adjacency(self.topo, self.labels())

    def selected_tris_mask(self):
        if self.body_selected is None:
            return np.ones(len(self.topo.F), bool)
        sel = set(self.body_selected)
        # cavities follow their parent body
        for c, p in self.cavities.items():
            if p in sel:
                sel.add(c)
        return np.isin(self.topo.body, list(sel))

    def fit_best(self, tris, types=None, tol=None, angle=None):
        V, C, N = _region_data(self.topo, tris)
        tys = tuple(t for t in fit.ORDER if t in (types or fit.ORDER))
        return fit.best_fit(V, C, N, tol or self.tol, angle or self.max_angle, self.topo.scale, tys)

    def deviation_of(self, model, tris, centroids=False):
        """Max |distance| of the triangles' vertices (and optionally centroids) to a model."""
        if model is None:
            return 0.0
        V = self.topo.V[np.unique(self.topo.F[tris])]
        if centroids:
            V = np.r_[V, self.topo.TC[tris]]
        m = fit.as_np(brep.export_model(model))
        return float(np.abs(fit.distance(m, V)).max())

    def fit_type_free(self, tris, t):
        """Best model of a given type regardless of tolerance (for 'convert to')."""
        if t == "freeform":
            return None
        V, C, N = _region_data(self.topo, tris)
        if t == "thread":
            return thread_params(self.topo, tris, self.tol)
        m = fit.fit_type(t, C if t in ("cylinder", "cone", "torus") else V, N, self.topo.scale) \
            if t != "cylinder" else fit.fit_cylinder(C, N, V)
        if m is None:
            return None
        m = fit.refine(m, V)
        return fit.clean(m)

    # ------------------------------------------------------------ analysis
    def analyze(self, params, progress=None):
        self.params.update({k: v for k, v in params.items() if v is not None})
        mask = self.selected_tris_mask()
        types = tuple(self.params["types"])
        if mask.all():
            regs = segment(self.topo, self.tol, float(self.params["min_face_pct"]), types,
                           normal_deg=float(self.params["angle"]), progress=progress)
        else:
            regs = self._segment_subset(np.flatnonzero(mask), types, progress)
            # untouched bodies stay as single freeform regions
            rest = np.flatnonzero(~mask)
            if len(rest):
                for b in np.unique(self.topo.body[rest]):
                    regs.append(Region(-1, rest[self.topo.body[rest] == b], None, "freeform"))
        self.checkpoint("Analyse")
        self.regions = []
        self.next_id = 0
        for r in regs:
            nr = self.new_region(r.tris, r.model, r.kind)
            nr.groups = r.groups
            self.regions.append(nr)
        self.analyzed = True
        self.changed()

    def _segment_subset(self, tris, types, progress=None, tol=None, angle=None, min_pct=None):
        """Segment a subset of triangles by running the pipeline on a sub-mesh."""
        F = self.topo.F[tris]
        vids, inv = np.unique(F, return_inverse=True)
        sub = trimesh.Trimesh(self.topo.V[vids], inv.reshape(-1, 3), process=False)
        st = Topo(sub)
        regs = segment(st, tol or self.tol, float(self.params["min_face_pct"] if min_pct is None else min_pct),
                       types, normal_deg=float(angle or self.params["angle"]), progress=progress)
        for r in regs:
            r.tris = tris[r.tris]
        return regs

    # ------------------------------------------------------------ face ops
    def set_region_tris(self, r, tris):
        r.tris = np.asarray(tris, np.int64)

    def _cleanup(self):
        """Drop empty regions; split regions that became disconnected."""
        out = []
        adjmask = None
        for r in self.regions:
            if len(r.tris) == 0:
                continue
            comps = self._components(r.tris)
            if len(comps) == 1:
                out.append(r)
                continue
            comps.sort(key=len, reverse=True)
            r.tris = comps[0]
            out.append(r)
            for c in comps[1:]:
                m = r.model if r.model and self.deviation_of(r.model, c) <= self.tol else self.fit_best(c)
                out.append(self.new_region(c, m))
        self.regions = out

    def _components(self, tris):
        tris = np.asarray(tris)
        s = set(tris.tolist())
        seen, comps = set(), []
        nb = self.topo.nbrs
        for t in tris.tolist():
            if t in seen:
                continue
            stack, comp = [t], []
            seen.add(t)
            while stack:
                x = stack.pop(); comp.append(x)
                for y, _ in nb[x]:
                    if y in s and y not in seen:
                        seen.add(y); stack.append(y)
            comps.append(np.array(comp, np.int64))
        return comps

    def neighbors_info(self, rid):
        r = self.region(rid)
        adj = self.adjacency().get(rid, {})
        out = []
        for n, (ang, cnt) in sorted(adj.items(), key=lambda x: -x[1][1]):
            o = self.region(n)
            dev = self.deviation_of(o.model, r.tris, True) if o.model else None
            out.append({"id": n, "kind": o.kind, "deviation": dev, "angle_deg": float(np.degrees(ang)),
                        "shared_edges": int(cnt), "params": fit.describe(o.model) if o.model else {}})
        conv = []
        for t in KINDS:
            if t == "freeform":
                conv.append({"type": t, "deviation": None})
                continue
            if t == "thread":
                continue
            m = self.fit_type_free(r.tris, t)
            conv.append({"type": t, "deviation": self.deviation_of(m, r.tris, True) if m else None,
                         "params": fit.describe(m) if m else {}})
        return {"neighbors": out, "convert": conv}

    def convert(self, rid, t):
        r = self.region(rid)
        m = self.fit_type_free(r.tris, t)
        if t != "freeform" and m is None:
            raise OpError(f"Could not fit a {t} to this face.")
        self.checkpoint(f"Convert to {t}")
        r.model, r.kind = m, t
        r.extra.pop("hole", None)
        self.changed()

    def merge(self, rid, into, mode="flush"):
        r, o = self.region(rid), self.region(into)
        self.checkpoint("Merge into neighbour")
        tris = np.r_[o.tris, r.tris]
        if mode == "tangent" and o.model is not None and o.kind in fit.ORDER:
            V, C, N = _region_data(self.topo, tris)
            m = fit.refine(fit.as_np(o.model), V)
            o.model = fit.clean(m)
        o.tris = tris
        self.regions = [x for x in self.regions if x is not r]
        self.changed()

    def tidy_variants(self, rid, max_dev=None):
        """Candidate replacements of a patchwork area by one face."""
        r = self.region(rid)
        adj = self.adjacency()
        byid = {x.id: x for x in self.regions}
        max_dev = max_dev or max(self.tol * 10, 0.2)
        rings = [[rid]]
        seen = {rid}
        for _ in range(3):
            nxt = list(rings[-1])
            for x in rings[-1]:
                for n, (ang, _) in adj.get(x, {}).items():
                    if n in seen or ang > np.radians(45):
                        continue
                    o = byid[n]
                    # big planes are anchors, not patchwork
                    if o.kind == "plane" and len(o.tris) > 3 * max(len(r.tris), 8):
                        continue
                    seen.add(n); nxt.append(n)
            if len(nxt) == len(rings[-1]):
                break
            rings.append(nxt)
        variants = []
        for members in rings[1:]:
            tris = np.concatenate([byid[m].tris for m in members])
            for t in ("cylinder", "cone", "sphere", "torus", "plane"):
                m = self.fit_type_free(tris, t)
                if m is None:
                    continue
                dev = self.deviation_of(m, tris, True)
                if dev <= max_dev:
                    variants.append({"type": t, "members": [int(x) for x in members], "saved": len(members) - 1,
                                     "deviation": dev, "params": fit.describe(m), "model": m})
        # keep the best per member-set, rank by faces saved then deviation
        best = {}
        for v in variants:
            k = tuple(v["members"])
            if k not in best or v["deviation"] < best[k]["deviation"]:
                best[k] = v
        out = sorted(best.values(), key=lambda v: (-v["saved"], v["deviation"]))
        # also offer second-best type for the largest set
        return out[:6]

    def tidy_apply(self, rid, members, model):
        self.checkpoint("Tidy up area")
        byid = {x.id: x for x in self.regions}
        members = [m for m in members if m in byid]
        if rid not in members:
            members.append(rid)
        tris = np.concatenate([byid[m].tris for m in members])
        keep = byid[rid]
        keep.tris, keep.model, keep.kind = tris, model, model["type"]
        self.regions = [x for x in self.regions if x.id == rid or x.id not in members]
        self.changed()

    def extend(self, rid, deviation, direction=None):
        r = self.region(rid)
        if r.model is None:
            raise OpError("Only recognised faces can be extended.")
        m = fit.as_np(brep.export_model(r.model))
        L = self.labels()
        owned = set(r.tris.tolist())
        frontier = list(owned)
        taken = []
        V, F = self.topo.V, self.topo.F
        while frontier:
            t = frontier.pop()
            for n, _ in self.topo.nbrs[t]:
                if n in owned:
                    continue
                ln = L[n]
                if direction is not None and ln != direction:
                    continue
                if np.abs(fit.distance(m, V[F[n]])).max() <= deviation:
                    owned.add(n); taken.append(n); frontier.append(n)
        if not taken:
            raise OpError("No neighbouring triangles lie on this face within that deviation.")
        self.checkpoint("Extend face")
        taken = np.array(taken)
        tset = set(taken.tolist())
        for o in self.regions:
            if o is r:
                continue
            o.tris = np.array([t for t in o.tris.tolist() if t not in tset], np.int64)
        r.tris = np.r_[r.tris, taken]
        self._cleanup()
        self.changed()
        return len(taken)

    def recognize(self, rid, tol, angle, types, include_ring=False):
        r = self.region(rid)
        members = [rid]
        if include_ring:
            members += list(self.adjacency().get(rid, {}).keys())
        byid = {x.id: x for x in self.regions}
        tris = np.concatenate([byid[m].tris for m in members])
        before = [byid[m] for m in members]
        regs = self._segment_subset(tris, tuple(types) + (("thread",) if "thread" in types else ()),
                                    tol=tol, angle=angle, min_pct=0)
        n_free_before = sum(len(x.tris) for x in before if x.model is None)
        n_free_after = sum(len(x.tris) for x in regs if x.model is None)
        simpler = len(regs) < len(before) or (len(regs) == len(before) and n_free_after < n_free_before)
        if not simpler:
            return {"applied": False, "before": len(before), "after": len(regs)}
        self.checkpoint("Recognise again")
        self.regions = [x for x in self.regions if x.id not in members]
        for x in regs:
            self.regions.append(self.new_region(x.tris, x.model, x.kind))
        self.changed()
        return {"applied": True, "before": len(before), "after": len(regs)}

    def refit(self, rid):
        r = self.region(rid)
        if r.kind not in fit.ORDER:
            raise OpError("Only planes, cylinders, cones, spheres and tori can be refitted.")
        m = self.fit_type_free(r.tris, r.kind)
        if m is None:
            raise OpError("Refit failed.")
        self.checkpoint("Fit to original")
        r.model = m
        r.extra.pop("hole", None)
        self.changed()

    def rename_face(self, rid, name):
        r = self.region(rid)
        r.name = name or None

    def cut(self, rid, path):
        """Split a face along a path of mesh vertices (consecutive vertices share an edge)."""
        r = self.region(rid)
        cut_edges = {(min(a, b), max(a, b)) for a, b in zip(path[:-1], path[1:])}
        if not cut_edges:
            raise OpError("Pick at least one edge.")
        s = set(r.tris.tolist())
        F = self.topo.F
        seen, comps = set(), []
        for t in r.tris.tolist():
            if t in seen:
                continue
            stack, comp = [t], []
            seen.add(t)
            while stack:
                x = stack.pop(); comp.append(x)
                for y, k in self.topo.nbrs[x]:
                    if y not in s or y in seen:
                        continue
                    e = tuple(sorted(int(v) for v in self.topo.adj_edge[k]))
                    if e in cut_edges:
                        continue
                    seen.add(y); stack.append(y)
            comps.append(np.array(comp, np.int64))
        if len(comps) < 2:
            raise OpError("The traced edges do not separate this face. Extend the seam to the face border.")
        self.checkpoint("Cut along edges")
        comps.sort(key=len, reverse=True)
        self.regions = [x for x in self.regions if x is not r]
        for c in comps:
            m = self.fit_best(c)
            self.regions.append(self.new_region(c, m))
        self.changed()
        return len(comps)

    # ------------------------------------------------------------ holes
    def hole_info(self, r: Region):
        m = r.model
        if m is None or r.kind not in ("cylinder", "thread"):
            return None
        a = np.asarray(m["axis"]); o = np.asarray(m["origin"])
        C = self.topo.TC[r.tris]; N = self.topo.TN[r.tris]
        rel = C - o
        radial = rel - np.outer(rel @ a, a)
        rn = np.linalg.norm(radial, axis=1)
        radial = radial / np.maximum(rn, 1e-12)[:, None]
        internal = float((radial * N).sum(1).mean()) < 0
        if not internal:
            return None
        u, v = fit._perp_basis(a)
        ang = np.sort(np.arctan2(radial @ v, radial @ u))
        gaps = np.diff(np.r_[ang, ang[0] + 2 * np.pi])
        coverage = 2 * np.pi - gaps.max()
        if coverage < np.radians(300):
            return None
        P = self.topo.V[np.unique(self.topo.F[r.tris])]
        depth = float(np.ptp(P @ a))
        d = 2 * (m["radius"] if r.kind == "cylinder" else m["minor_diameter"] / 2)
        if r.kind == "thread":
            d = m["minor_diameter"]
        orig = r.extra.get("hole", {}).get("original_diameter", d)
        return {"diameter": d, "depth": depth, "fits": iso_fit(d), "original_diameter": orig,
                "setting": r.extra.get("hole")}

    def identical_holes(self, rid):
        r = self.region(rid)
        hi = self.hole_info(r)
        if hi is None:
            return []
        out = []
        for o in self.regions:
            if o is r:
                continue
            h = self.hole_info(o)
            if h and abs(h["diameter"] - hi["diameter"]) < 0.02 and abs(h["depth"] - hi["depth"]) < 0.05:
                out.append(o.id)
        return out

    def set_hole(self, rid, diameter, apply_all=False, size=None, purpose=None, thread=None):
        r = self.region(rid)
        if self.hole_info(r) is None:
            raise OpError("This face is not a hole wall.")
        targets = [rid] + (self.identical_holes(rid) if apply_all else [])
        self.checkpoint("Hole size")
        for t in targets:
            x = self.region(t)
            hi = self.hole_info(x)
            m = dict(x.model) if x.kind == "cylinder" else {
                "type": "cylinder", "origin": x.model["origin"], "axis": x.model["axis"],
                "radius": x.model["minor_diameter"] / 2}
            setting = {"original_diameter": hi["original_diameter"], "size": size, "purpose": purpose}
            if thread:
                nominal, pitch = float(thread["nominal"]), float(thread["pitch"])
                minor = nominal - 1.0825 * pitch
                m["radius"] = minor / 2
                setting["thread"] = {"nominal": nominal, "pitch": pitch, "designation": f"M{nominal:g}×{pitch:g}",
                                     "length": hi["depth"]}
            else:
                m["radius"] = float(diameter) / 2
            x.model, x.kind = m, "cylinder"
            x.extra["hole"] = setting
        self.changed()
        return len(targets)

    # ------------------------------------------------------------ deviation
    def deviation(self):
        """Per triangle-corner |distance| to the face that will be written, plus stats."""
        F = self.topo.F
        dev = np.zeros((len(F), 3), np.float32)
        per_face = {}
        all_v = []
        for r in self.regions:
            if r.model is None:
                per_face[r.id] = None
                continue
            # threads are judged against their thread band, everything else against the exported surface
            m = fit.as_np(r.model if r.kind == "thread" else brep.export_model(r.model))
            P = self.topo.V[F[r.tris]].reshape(-1, 3)
            d = np.abs(fit.distance(m, P)).reshape(-1, 3)
            dev[r.tris] = d
            u = np.unique(F[r.tris])
            dv = np.abs(fit.distance(m, self.topo.V[u]))
            per_face[r.id] = _stats(dv, self.tol)
            all_v.append(dv)
        allv = np.concatenate(all_v) if all_v else np.zeros(1)
        return dev, _stats(allv, self.tol), per_face

    # ------------------------------------------------------------ mesh repair
    def close_gaps(self):
        m, hints = repair.close_gaps(self.topo, self.regions, self.tol)
        if m is None:
            raise OpError("The mesh has no open edges.")
        self.checkpoint("Close mesh gaps")
        self.topo = Topo(m)
        self.cavities = repair.find_cavities(self.topo)
        L = None
        for tris, model in hints:
            placed = False
            if model is not None and self.analyzed:
                for r in self.regions:
                    if r.model is not None and r.kind in fit.ORDER and same_surface(r.model, model, self.tol * 5):
                        r.tris = np.r_[r.tris, tris]
                        placed = True
                        break
            if not placed:
                if self.analyzed:
                    self.regions.append(self.new_region(tris, model))
        if not self.analyzed:
            self.regions = []
        self.changed()
        return len(hints)

    def fill_cavities(self, ids=None):
        cav = self.cavities if ids is None else {i: self.cavities[i] for i in ids if i in self.cavities}
        if not cav:
            raise OpError("No cavities to fill.")
        self.checkpoint("Fill cavities")
        drop = np.isin(self.topo.body, list(cav.keys()))
        keep_tris = np.flatnonzero(~drop)
        remap = -np.ones(len(self.topo.F), np.int64)
        remap[keep_tris] = np.arange(len(keep_tris))
        old_body = self.topo.body.copy()
        m = trimesh.Trimesh(self.topo.V, self.topo.F[keep_tris], process=False)
        m.remove_unreferenced_vertices()
        # remove_unreferenced keeps face order, so triangle remap stays valid
        self.topo = Topo(m)
        for r in self.regions:
            r.tris = remap[r.tris]
            r.tris = r.tris[r.tris >= 0]
        self.regions = [r for r in self.regions if len(r.tris)]
        self.cavities = repair.find_cavities(self.topo)
        self.body_names = {}
        self.body_selected = None
        self.changed()
        return len(cav)

    # ------------------------------------------------------------ export
    def build(self, thread_mode="cylinder", progress=None):
        key = (self.version, thread_mode, tuple(self.body_selected or ()))
        if self.build_cache and self.build_cache[0] == key:
            return self.build_cache[1]
        L = self.labels()
        models = {r.id: r.model for r in self.regions}
        kinds = {r.id: r.kind for r in self.regions}
        bodies = None
        if self.body_selected is not None:
            bodies = set(self.body_selected)
            for c, p in self.cavities.items():
                if p in bodies:
                    bodies.add(c)
        body = self.topo.body.copy()
        for c, p in self.cavities.items():
            body[body == c] = p     # cavities become inner shells of their parent solid
        slack = {}
        for r in self.regions:
            h = r.extra.get("hole")
            if h and r.model:
                slack[r.id] = abs(h["original_diameter"] / 2 - r.model["radius"]) * 1.2
        res = brep.build(self.topo.V, self.topo.F, L, models, kinds, body, bodies, self.tol, progress,
                         shell_of=self.topo.body, slack=slack)
        threads = []
        for r in self.regions:
            if r.model is None:
                continue
            hs = r.extra.get("hole", {}).get("thread")
            auto = r.kind == "thread" and thread_mode == "helical"
            if not hs and not auto:
                continue
            a = np.asarray(r.model["axis"], float)
            o = np.asarray(r.model["origin"], float)
            h = (self.topo.V[np.unique(self.topo.F[r.tris])] - o) @ a
            if hs:
                t = {"nominal": hs["nominal"], "pitch": hs["pitch"], "internal": True}
            else:
                m = r.model
                nominal, pitch = m["major_diameter"], round(m["pitch"] / 0.05) * 0.05
                if m.get("designation"):
                    nominal, pitch = (float(x) for x in m["designation"][1:].split("×"))
                t = {"nominal": nominal, "pitch": pitch, "internal": bool(m["internal"])}
            t.update({"origin": o, "axis": a, "start": float(h.min()), "end": float(h.max()), "region": r.id})
            threads.append(t)
        if threads and res.shape is not None:
            from . import threads as th
            res = th.apply_threads(res, threads, self.tol)
        self.build_cache = (key, res)
        return res

    def step_bytes(self, thread_mode="cylinder", progress=None):
        res = self.build(thread_mode, progress)
        from .stepio import write_step
        names = {}
        for bid, _ in res.bodies:
            names[bid] = self.body_names.get(bid) or f"Body {bid + 1}"
        base = self.filename.rsplit(".", 1)[0]
        return write_step(res, names, base), res

    # ------------------------------------------------------------ serialisation
    def bodies_info(self):
        out = []
        for b in range(self.topo.n_bodies):
            if b in self.cavities:
                continue
            tris = np.flatnonzero(self.topo.body == b)
            P = self.topo.V[np.unique(self.topo.F[tris])]
            size = np.ptp(P, axis=0)
            out.append({"id": b, "name": self.body_names.get(b) or f"Body {b + 1}", "triangles": int(len(tris)),
                        "size": [float(x) for x in size],
                        "volume": abs(repair.component_volume(self.topo, tris)),
                        "selected": self.body_selected is None or b in self.body_selected,
                        "cavities": [c for c, p in self.cavities.items() if p == b]})
        return out

    def cavities_info(self):
        out = []
        for c, p in self.cavities.items():
            tris = np.flatnonzero(self.topo.body == c)
            out.append({"id": c, "parent": p, "volume": abs(repair.component_volume(self.topo, tris)),
                        "triangles": int(len(tris)), "center": [float(x) for x in self.topo.TC[tris].mean(0)]})
        return out

    def faces_info(self, per_face=None):
        if per_face is None:
            per_face = self.deviation()[2]
        out = []
        for r in self.regions:
            area = float(self.topo.TA[r.tris].sum())
            info = {"id": r.id, "kind": r.kind, "triangles": int(len(r.tris)), "area": area,
                    "body": int(self.topo.body[r.tris[0]]), "name": r.name,
                    "params": fit.describe(r.model) if r.model else {},
                    "model": _json_model(r.model), "deviation": per_face.get(r.id)}
            if r.kind == "thread" and r.model:
                info["params"]["designation"] = r.model.get("designation")
            h = self.hole_info(r)
            if h:
                info["hole"] = h
            out.append(info)
        return out

    def summary(self):
        counts = {}
        for r in self.regions:
            counts[r.kind] = counts.get(r.kind, 0) + 1
        area = self.topo.TA.sum()
        rec = sum(self.topo.TA[r.tris].sum() for r in self.regions if r.model is not None)
        oe = repair.open_edges(self.topo.F)
        return {"counts": counts, "faces": len(self.regions), "recognized_pct": float(100 * rec / area) if area else 0,
                "open_edges": int(len(oe)), "gaps": len(repair.boundary_loops(self.topo.F)) if len(oe) else 0}


def _stats(d, tol):
    if d is None or len(d) == 0:
        return None
    return {"max": float(d.max()), "p95": float(np.percentile(d, 95)), "mean": float(d.mean()),
            "over": float((d > tol).mean() * 100)}


def _json_model(m):
    if m is None:
        return None
    return {k: (float(v) if isinstance(v, (int, float, np.floating)) and not isinstance(v, bool) else v)
            for k, v in m.items()}
