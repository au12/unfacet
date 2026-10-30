"""Generate CAD-exported-style STL test meshes with OpenCascade.

Usage: python tests/make_samples.py [outdir]
"""
import sys, os
from OCP.BRepPrimAPI import (BRepPrimAPI_MakeBox, BRepPrimAPI_MakeCylinder,
                             BRepPrimAPI_MakeSphere, BRepPrimAPI_MakeCone, BRepPrimAPI_MakeTorus)
from OCP.BRepAlgoAPI import BRepAlgoAPI_Cut, BRepAlgoAPI_Fuse
from OCP.BRepFilletAPI import BRepFilletAPI_MakeFillet, BRepFilletAPI_MakeChamfer
from OCP.BRepMesh import BRepMesh_IncrementalMesh
from OCP.StlAPI import StlAPI_Writer
from OCP.TopExp import TopExp_Explorer
from OCP.TopAbs import TopAbs_EDGE
from OCP.TopoDS import TopoDS
from OCP.gp import gp_Pnt, gp_Ax2, gp_Dir, gp_Trsf, gp_Vec
from OCP.BRepBuilderAPI import BRepBuilderAPI_Transform
from OCP.BRep import BRep_Tool
from OCP.BRepAdaptor import BRepAdaptor_Curve
from OCP.GeomAbs import GeomAbs_Line


def edges(shape):
    ex = TopExp_Explorer(shape, TopAbs_EDGE)
    while ex.More():
        yield TopoDS.Edge(ex.Current())
        ex.Next()


def mid(e):
    c = BRepAdaptor_Curve(e)
    p = c.Value((c.FirstParameter() + c.LastParameter()) / 2)
    return p.X(), p.Y(), p.Z(), c.GetType()


def bracket():
    """Mounting bracket: plate + upright, holes, counterbore, fillets, chamfers."""
    base = BRepPrimAPI_MakeBox(gp_Pnt(0, 0, 0), 60, 40, 8).Shape()
    up = BRepPrimAPI_MakeBox(gp_Pnt(0, 0, 0), 8, 40, 40).Shape()
    s = BRepAlgoAPI_Fuse(base, up).Shape()
    # fillet the vertical outer edges of plate end
    f = BRepFilletAPI_MakeFillet(s)
    n = 0
    for e in edges(s):
        x, y, z, t = mid(e)
        if t == GeomAbs_Line and abs(x - 60) < 1e-6 and (abs(y) < 1e-6 or abs(y - 40) < 1e-6) and 0 < z < 8:
            f.Add(6.0, e); n += 1
    f.Build(); s = f.Shape()
    # inner corner fillet
    f = BRepFilletAPI_MakeFillet(s)
    for e in edges(s):
        x, y, z, t = mid(e)
        if t == GeomAbs_Line and abs(x - 8) < 1e-6 and abs(z - 8) < 1e-6:
            f.Add(4.0, e)
    f.Build(); s = f.Shape()
    # through holes in plate
    for (x, y) in [(35, 12), (35, 28), (50, 20)]:
        h = BRepPrimAPI_MakeCylinder(gp_Ax2(gp_Pnt(x, y, -1), gp_Dir(0, 0, 1)), 2.75, 12).Shape()
        s = BRepAlgoAPI_Cut(s, h).Shape()
    # counterbore on one
    cb = BRepPrimAPI_MakeCylinder(gp_Ax2(gp_Pnt(50, 20, 5), gp_Dir(0, 0, 1)), 5, 10).Shape()
    s = BRepAlgoAPI_Cut(s, cb).Shape()
    # countersink (cone) in upright
    hole = BRepPrimAPI_MakeCylinder(gp_Ax2(gp_Pnt(-1, 20, 26), gp_Dir(1, 0, 0)), 3.2, 12).Shape()
    s = BRepAlgoAPI_Cut(s, hole).Shape()
    cs = BRepPrimAPI_MakeCone(gp_Ax2(gp_Pnt(4.8, 20, 26), gp_Dir(1, 0, 0)), 3.2, 6.6, 3.4).Shape()
    s = BRepAlgoAPI_Cut(s, cs).Shape()
    # chamfer top edge of upright
    c = BRepFilletAPI_MakeChamfer(s)
    for e in edges(s):
        x, y, z, t = mid(e)
        if t == GeomAbs_Line and abs(z - 40) < 1e-6 and abs(x - 8) < 1e-6:
            c.Add(2.0, e)
    c.Build(); s = c.Shape()
    # spherical dimple
    sp = BRepPrimAPI_MakeSphere(gp_Pnt(22, 20, 11), 4.5).Shape()
    s = BRepAlgoAPI_Cut(s, sp).Shape()
    return s


def knob():
    """Revolved-ish part: cylinder + cone + torus fillet + sphere cap."""
    cyl = BRepPrimAPI_MakeCylinder(gp_Ax2(gp_Pnt(0, 0, 0), gp_Dir(0, 0, 1)), 15, 10).Shape()
    f = BRepFilletAPI_MakeFillet(cyl)
    for e in edges(cyl):
        x, y, z, t = mid(e)
        if abs(z - 10) < 1e-6:
            f.Add(3.0, e)
    f.Build(); s = f.Shape()
    cone = BRepPrimAPI_MakeCone(gp_Ax2(gp_Pnt(0, 0, 10), gp_Dir(0, 0, 1)), 8, 5, 8).Shape()
    s = BRepAlgoAPI_Fuse(s, cone).Shape()
    sph = BRepPrimAPI_MakeSphere(gp_Pnt(0, 0, 18), 5).Shape()
    s = BRepAlgoAPI_Fuse(s, sph).Shape()
    bore = BRepPrimAPI_MakeCylinder(gp_Ax2(gp_Pnt(0, 0, -1), gp_Dir(0, 0, 1)), 2.5, 8).Shape()
    s = BRepAlgoAPI_Cut(s, bore).Shape()
    return s


def two_bodies():
    a = BRepPrimAPI_MakeBox(gp_Pnt(0, 0, 0), 20, 20, 20).Shape()
    h = BRepPrimAPI_MakeCylinder(gp_Ax2(gp_Pnt(10, 10, -1), gp_Dir(0, 0, 1)), 4, 30).Shape()
    a = BRepAlgoAPI_Cut(a, h).Shape()
    b = BRepPrimAPI_MakeTorus(gp_Ax2(gp_Pnt(45, 10, 10), gp_Dir(0, 1, 0)), 10, 3).Shape()
    from OCP.TopoDS import TopoDS_Compound
    from OCP.BRep import BRep_Builder
    comp = TopoDS_Compound(); bb = BRep_Builder(); bb.MakeCompound(comp)
    bb.Add(comp, a); bb.Add(comp, b)
    return comp


def write(shape, path, lin=0.02, ang=0.3):
    BRepMesh_IncrementalMesh(shape, lin, False, ang, True)
    w = StlAPI_Writer(); w.ASCIIMode = False
    w.Write(shape, path)


if __name__ == "__main__":
    out = sys.argv[1] if len(sys.argv) > 1 else "samples"
    os.makedirs(out, exist_ok=True)
    write(bracket(), f"{out}/bracket.stl")
    write(knob(), f"{out}/knob.stl")
    write(two_bodies(), f"{out}/two_bodies.stl")
    write(bracket(), f"{out}/bracket_coarse.stl", lin=0.1, ang=0.5)
    print("ok")


def thread_rod(path, d=6.0, pitch=1.0, length=12.0, seg=72):
    """External ISO-like thread mesh (triangle profile) with flat end caps, built directly."""
    import numpy as np, trimesh
    H = 0.866 * pitch
    rmax, rmin = d / 2, d / 2 - 0.6134 * pitch
    nz = int(length / (pitch / 16)) + 1
    th = np.linspace(0, 2 * np.pi, seg, endpoint=False)
    zs = np.linspace(0, length, nz)

    def r_of(z, t):
        ph = np.mod(z - pitch * t / (2 * np.pi), pitch) / pitch
        tri = 1 - 2 * np.abs(ph - 0.5)  # 0..1..0
        return rmin + (rmax - rmin) * np.clip(tri * 1.25, 0, 1)
    V = []
    for z in zs:
        r = r_of(z, th)
        V.append(np.c_[r * np.cos(th), r * np.sin(th), np.full(seg, z)])
    V = np.vstack(V)
    F = []
    for i in range(nz - 1):
        for j in range(seg):
            a = i * seg + j; b = i * seg + (j + 1) % seg
            c = a + seg; e = b + seg
            F += [[a, b, e], [a, e, c]]
    c0 = len(V); V = np.vstack([V, [0, 0, 0], [0, 0, length]])
    for j in range(seg):
        F.append([c0, (j + 1) % seg, j])
        t = (nz - 1) * seg
        F.append([c0 + 1, t + j, t + (j + 1) % seg])
    trimesh.Trimesh(V, np.array(F), process=True).export(path)


if __name__ == "__main__" and len(sys.argv) > 1:
    thread_rod(f"{sys.argv[1]}/thread_m6.stl")


def cavity_block(path):
    """Block with an enclosed spherical void (two mesh shells)."""
    s = BRepPrimAPI_MakeBox(gp_Pnt(0, 0, 0), 30, 30, 20).Shape()
    s = BRepAlgoAPI_Cut(s, BRepPrimAPI_MakeSphere(gp_Pnt(15, 15, 10), 6).Shape()).Shape()
    s = BRepAlgoAPI_Cut(s, BRepPrimAPI_MakeCylinder(gp_Ax2(gp_Pnt(5, 5, -1), gp_Dir(0, 0, 1)), 2, 30).Shape()).Shape()
    write(s, path)


def open_bracket(src, path):
    """Bracket with a few triangles knocked out of a hole wall and the top face."""
    import numpy as np, trimesh
    m = trimesh.load(src)
    m.merge_vertices()
    c = m.triangles_center
    r = np.hypot(c[:, 0] - 35, c[:, 1] - 12)
    drop = np.flatnonzero((np.abs(r - 2.75) < 0.2) & (np.abs(c[:, 2] - 4) < 1.5))[:3]
    keep = np.setdiff1d(np.arange(len(m.faces)), drop)
    trimesh.Trimesh(m.vertices, m.faces[keep]).export(path)


if __name__ == "__main__" and len(sys.argv) > 1:
    cavity_block(f"{sys.argv[1]}/cavity_block.stl")
    open_bracket(f"{sys.argv[1]}/bracket.stl", f"{sys.argv[1]}/bracket_open.stl")
