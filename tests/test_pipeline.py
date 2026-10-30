"""End-to-end checks on CAD-generated meshes: run with `pytest -q tests`."""
import os
import sys
import tempfile

import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
sys.path.insert(0, os.path.dirname(__file__))

import make_samples as M  # noqa: E402
from app.core.mesh import load_mesh  # noqa: E402
from app.core.session import Session  # noqa: E402

TMP = tempfile.mkdtemp()


def session(name, builder):
    path = os.path.join(TMP, name)
    if not os.path.exists(path):
        builder(path)
    with open(path, "rb") as f:
        s = Session(load_mesh(f.read(), name), name)
    s.analyze({})
    return s


def volume(shape):
    from OCP.BRepGProp import BRepGProp
    from OCP.GProp import GProp_GProps
    p = GProp_GProps()
    BRepGProp.VolumeProperties_s(shape, p)
    return p.Mass()


def test_bracket_faces_and_solid():
    s = session("bracket.stl", lambda p: M.write(M.bracket(), p))
    c = s.summary()["counts"]
    assert c == {"plane": 10, "cylinder": 8, "cone": 1, "sphere": 1}
    res = s.build()
    assert all(res.closed.values()) and not res.discarded and not res.invalid
    assert volume(res.shape) == pytest.approx(28216.3, rel=2e-3)


def test_knob_revolved_kinds():
    s = session("knob.stl", lambda p: M.write(M.knob(), p))
    assert s.summary()["counts"] == {"plane": 3, "cylinder": 2, "cone": 1, "torus": 1, "sphere": 1}
    tor = next(r for r in s.regions if r.kind == "torus")
    assert tor.model["major"] == pytest.approx(12, abs=1e-3)
    assert tor.model["minor"] == pytest.approx(3, abs=1e-3)
    assert all(s.build().closed.values())


def test_thread_detected_and_cut():
    s = session("thread.stl", M.thread_rod)
    th = next(r for r in s.regions if r.kind == "thread")
    assert th.model["designation"] == "M6×1"
    res = s.build("helical")
    assert any("Cut thread" in m for m in res.messages)
    assert all(res.closed.values())


def test_cavity_becomes_inner_shell():
    s = session("cavity.stl", M.cavity_block)
    assert len(s.cavities) == 1
    res = s.build()
    assert volume(res.shape) == pytest.approx(30 * 30 * 20 - 3.14159265 * 4 * 20 - 4 / 3 * 3.14159265 * 216, rel=3e-3)


def test_hole_resize_keeps_solid():
    s = session("bracket.stl", lambda p: M.write(M.bracket(), p))
    hole = next(f for f in s.faces_info() if f.get("hole") and f["hole"]["fits"])
    s.set_hole(hole["id"], 6.6, apply_all=True)
    res = s.build()
    assert all(res.closed.values()) and not res.discarded
    s.undo()
    assert s.region(hole["id"]).model["radius"] == pytest.approx(2.75, abs=1e-3)
