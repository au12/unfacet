"""STEP writing with named bodies (XCAF)."""
from __future__ import annotations

import os
import tempfile

from OCP.Interface import Interface_Static
from OCP.STEPCAFControl import STEPCAFControl_Writer
from OCP.STEPControl import STEPControl_AsIs
from OCP.TCollection import TCollection_ExtendedString
from OCP.TDataStd import TDataStd_Name
from OCP.TDocStd import TDocStd_Document
from OCP.XCAFApp import XCAFApp_Application
from OCP.XCAFDoc import XCAFDoc_DocumentTool


def write_step(res, names: dict, product: str) -> bytes:
    # create the writer first: loading OpenCascade's STEP module resets these settings to defaults
    w = STEPCAFControl_Writer()
    doc = TDocStd_Document(TCollection_ExtendedString("MDTV-XCAF"))
    XCAFApp_Application.GetApplication_s().InitDocument(doc)
    st = XCAFDoc_DocumentTool.ShapeTool_s(doc.Main())
    if len(res.bodies) == 1:
        bid, shape = res.bodies[0]
        lab = st.AddShape(shape, False)
        TDataStd_Name.Set_s(lab, TCollection_ExtendedString(names.get(bid) or product))
    else:
        for bid, shape in res.bodies:
            lab = st.AddShape(shape, False)
            TDataStd_Name.Set_s(lab, TCollection_ExtendedString(names.get(bid, f"Body {bid + 1}")))
    Interface_Static.SetCVal_s("write.step.unit", "MM")
    # skip the optional 2D (parametric) copies of every edge: CAD importers rebuild them, and leaving
    # them out roughly halves both the file size and the time to write it
    Interface_Static.SetIVal_s("write.surfacecurve.mode", 0)
    Interface_Static.SetCVal_s("write.step.product.name", product)
    w.SetNameMode(True)
    w.Transfer(doc, STEPControl_AsIs)
    fd, path = tempfile.mkstemp(suffix=".step")
    os.close(fd)
    try:
        w.Write(path)
        with open(path, "rb") as f:
            return f.read()
    finally:
        os.unlink(path)
