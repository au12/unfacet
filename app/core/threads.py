"""Cut real ISO metric threads (helical flanks) into a built B-rep."""
from __future__ import annotations

import numpy as np

from OCP.BRepAlgoAPI import BRepAlgoAPI_Cut
from OCP.BRepBuilderAPI import BRepBuilderAPI_MakeEdge, BRepBuilderAPI_MakePolygon, BRepBuilderAPI_MakeWire
from OCP.BRepCheck import BRepCheck_Analyzer
from OCP.BRepLib import BRepLib
from OCP.BRepOffsetAPI import BRepOffsetAPI_MakePipeShell
from OCP.BRep import BRep_Builder
from OCP.Geom import Geom_CylindricalSurface
from OCP.Geom2d import Geom2d_Line
from OCP.ShapeFix import ShapeFix_Shape
from OCP.TopoDS import TopoDS_Compound
from OCP.gp import gp_Ax3, gp_Dir, gp_Dir2d, gp_Pnt, gp_Pnt2d

from OCP.TopAbs import TopAbs_FACE
from OCP.TopExp import TopExp_Explorer

from . import fit


def _p(x):
    return gp_Pnt(*(float(v) for v in x))


def groove_solid(origin, axis, start, end, nominal, pitch, internal, right_hand=True):
    """Solid swept along a helix whose removal from the part leaves an ISO 60° thread."""
    a = np.asarray(axis, float) / np.linalg.norm(axis)
    o = np.asarray(origin, float)
    H = 0.8660254 * pitch
    z0 = start - pitch            # overrun both ends so the cut runs out cleanly
    length = (end - start) + 2 * pitch
    R = nominal / 2
    if internal:
        r_lo = (nominal - 1.0825 * pitch) / 2 - 0.05 * pitch   # just inside the bore wall
        r_hi = R + H / 8
    else:
        r_lo = (nominal - 1.2268 * pitch) / 2 - H / 16
        r_hi = R + 0.05 * pitch
    half_w = (r_hi - r_lo) * np.tan(np.radians(30))
    if internal:
        # apex outward (r_hi), base at bore
        prof_r = [r_lo, r_hi, r_lo]
        prof_z = [-half_w, 0.0, half_w]
    else:
        prof_r = [r_hi, r_lo, r_hi]
        prof_z = [-half_w, 0.0, half_w]
    u, v = fit._perp_basis(a)
    base = o + a * z0
    ax3 = gp_Ax3(_p(base), gp_Dir(*a), gp_Dir(*u))
    if not right_hand:
        ax3 = gp_Ax3(_p(base), gp_Dir(*a), gp_Dir(*u))
        ax3.YReverse()
    cyl = Geom_CylindricalSurface(ax3, float(r_lo if internal else r_hi))
    turns = length / pitch
    line = Geom2d_Line(gp_Pnt2d(0, 0), gp_Dir2d(2 * np.pi, pitch))
    plen = float(np.hypot(2 * np.pi, pitch) * turns)
    edge = BRepBuilderAPI_MakeEdge(line, cyl, 0.0, plen).Edge()
    BRepLib.BuildCurves3d_s(edge)
    spine = BRepBuilderAPI_MakeWire(edge).Wire()
    poly = BRepBuilderAPI_MakePolygon()
    for r, z in zip(prof_r, prof_z):
        poly.Add(_p(base + u * r + a * z))
    poly.Close()
    pipe = BRepOffsetAPI_MakePipeShell(spine)
    pipe.SetMode(gp_Dir(*a))
    pipe.Add(poly.Wire())
    pipe.Build()
    if not pipe.IsDone():
        raise RuntimeError("thread sweep failed")
    pipe.MakeSolid()
    return pipe.Shape()


def apply_threads(res, threads, tol):
    """threads: list of dict(origin, axis, start, end, nominal, pitch, internal, region)."""
    if not threads or res.shape is None:
        return res
    region_body = {}
    bodies = dict(res.bodies)
    # which body holds which region: match through the face mapping
    from .brep import _fkey
    owner = {}
    for bid, sh in res.bodies:
        ex = TopExp_Explorer(sh, TopAbs_FACE)
        while ex.More():
            owner[_fkey(ex.Current())] = bid
            ex.Next()
    for f, r in res.face_region:
        if _fkey(f) in owner:
            region_body.setdefault(r, owner[_fkey(f)])
    for t in threads:
        bid = region_body.get(t["region"])
        if bid is None:
            continue
        try:
            shp = None
            for extra in (0.0, 0.137, 0.311):   # nudge the run-out if the boolean trips on a coincidence
                g = groove_solid(t["origin"], t["axis"], t["start"] - extra, t["end"] + extra,
                                 t["nominal"], t["pitch"], t["internal"])
                cut = BRepAlgoAPI_Cut(bodies[bid], g)
                if cut.IsDone() and _volume(cut.Shape()) > 0.5 * _volume(bodies[bid]):
                    shp = cut.Shape()
                    break
            if shp is None:
                raise RuntimeError("boolean failed")
            if not BRepCheck_Analyzer(shp).IsValid():
                fx = ShapeFix_Shape(shp)
                fx.Perform()
                shp = fx.Shape()
            bodies[bid] = shp
            res.messages.append(f"Cut thread M{t['nominal']:g}×{t['pitch']:g}.")
        except Exception as e:  # noqa: BLE001
            res.messages.append(f"Could not cut thread M{t['nominal']:g}: {e}. Kept a plain bore.")
    res.bodies = [(bid, bodies[bid]) for bid, _ in res.bodies]
    comp = TopoDS_Compound()
    B = BRep_Builder()
    B.MakeCompound(comp)
    for _, sh in res.bodies:
        B.Add(comp, sh)
    res.shape = comp
    return res


def _volume(shape):
    from OCP.GProp import GProp_GProps
    from OCP.BRepGProp import BRepGProp
    p = GProp_GProps()
    BRepGProp.VolumeProperties_s(shape, p)
    return p.Mass()


def _contains(shape, face):
    from OCP.TopExp import TopExp_Explorer
    from OCP.TopAbs import TopAbs_FACE
    ex = TopExp_Explorer(shape, TopAbs_FACE)
    while ex.More():
        if ex.Current().IsSame(face):
            return True
        ex.Next()
    return False
