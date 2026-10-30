"""Unfacet HTTP API + static web app."""
from __future__ import annotations

import base64
import io
import os
import threading
import time
import traceback
import uuid
from concurrent.futures import ThreadPoolExecutor

import numpy as np
from fastapi import FastAPI, File, HTTPException, Request, UploadFile
from fastapi.responses import FileResponse, JSONResponse, Response
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from .core import brep, fit
from .core.mesh import MeshError, load_mesh
from .core.session import ISO_HOLES, KINDS, OpError, Session

MAX_UPLOAD_MB = float(os.environ.get("MAX_UPLOAD_MB", "200"))
SESSION_TTL_MIN = float(os.environ.get("SESSION_TTL_MIN", "240"))
MAX_SESSIONS = int(os.environ.get("MAX_SESSIONS", "30"))
WORKERS = int(os.environ.get("WORKERS", "2"))
VERSION = "1.0.0"

app = FastAPI(title="Unfacet", version=VERSION, docs_url="/api/docs", openapi_url="/api/openapi.json")
SESSIONS: dict[str, Session] = {}
JOBS: dict[str, dict] = {}
POOL = ThreadPoolExecutor(max_workers=WORKERS)
LOCK = threading.Lock()


# ------------------------------------------------------------------ helpers
def b64(a, dtype):
    return base64.b64encode(np.ascontiguousarray(a, dtype=dtype).tobytes()).decode()


def get_session(sid) -> Session:
    s = SESSIONS.get(sid)
    if s is None:
        raise HTTPException(404, "Session expired. Open the file again.")
    s.touched = time.time()
    return s


def _reap():
    while True:
        time.sleep(60)
        now = time.time()
        with LOCK:
            for k in [k for k, s in SESSIONS.items() if now - s.touched > SESSION_TTL_MIN * 60]:
                SESSIONS.pop(k, None)
            for k in [k for k, j in JOBS.items() if now - j["created"] > 3600]:
                JOBS.pop(k, None)


threading.Thread(target=_reap, daemon=True).start()


def mesh_payload(s: Session):
    from .core.repair import open_edges
    oe = open_edges(s.topo.F)
    return {"positions": b64(s.topo.V, np.float32), "indices": b64(s.topo.F, np.uint32),
            "bodies": b64(s.topo.body, np.int32), "open_edges": b64(oe, np.uint32),
            "triangles": int(len(s.topo.F)), "vertices": int(len(s.topo.V)), "mesh_id": id(s.topo)}


def state(s: Session, mesh=False, extra=None):
    dev, stats, per_face = s.deviation()
    out = {
        "id": s.id, "filename": s.filename, "version": s.version, "analyzed": s.analyzed,
        "params": s.params, "summary": s.summary(), "bodies": s.bodies_info(), "cavities": s.cavities_info(),
        "faces": s.faces_info(per_face), "labels": b64(s.labels(), np.int32),
        "deviation": {"stats": stats, "corners": b64(dev, np.float32)},
        "history": {"undo": s.undo_stack[-1].label if s.undo_stack else None,
                    "redo": s.redo_stack[-1].label if s.redo_stack else None},
        "mesh_id": id(s.topo),
    }
    if mesh:
        out["mesh"] = mesh_payload(s)
    if extra:
        out.update(extra)
    return out


def run_job(kind, fn):
    jid = uuid.uuid4().hex[:12]
    job = {"id": jid, "kind": kind, "status": "running", "progress": 0.0, "created": time.time(),
           "result": None, "error": None}
    JOBS[jid] = job

    def prog(p):
        job["progress"] = float(min(max(p, 0), 1))

    def work():
        try:
            job["result"] = fn(prog)
            job["status"] = "done"
            job["progress"] = 1.0
        except (OpError, MeshError) as e:
            job["status"], job["error"] = "error", str(e)
        except Exception as e:  # noqa: BLE001
            traceback.print_exc()
            job["status"], job["error"] = "error", f"Internal error: {e}"
    POOL.submit(work)
    return {"job": jid}


def op(s: Session, fn):
    with s.lock:
        try:
            r = fn()
        except OpError as e:
            raise HTTPException(400, str(e))
    return r


# ------------------------------------------------------------------ models
class AnalyzeIn(BaseModel):
    tolerance: float | None = None
    min_face_pct: float | None = None
    angle: float | None = None
    types: list[str] | None = None
    bodies: list[int] | None = None


class ConvertIn(BaseModel):
    type: str


class MergeIn(BaseModel):
    into: int
    mode: str = "flush"


class TidyIn(BaseModel):
    members: list[int]
    model: dict


class ExtendIn(BaseModel):
    deviation: float = 0.2
    direction: int | None = None


class RecognizeIn(BaseModel):
    tolerance: float = 0.05
    angle: float = 35.0
    types: list[str] = list(KINDS[:-1])
    include_ring: bool = False


class CutIn(BaseModel):
    path: list[int]


class HoleIn(BaseModel):
    diameter: float | None = None
    apply_all: bool = False
    size: str | None = None
    purpose: str | None = None
    thread: dict | None = None


class NameIn(BaseModel):
    name: str


class BodiesIn(BaseModel):
    selected: list[int] | None = None


class CavIn(BaseModel):
    ids: list[int] | None = None


class ExportIn(BaseModel):
    thread_mode: str = "cylinder"


# ------------------------------------------------------------------ routes
@app.get("/api/health")
def health():
    return {"ok": True, "version": VERSION, "sessions": len(SESSIONS)}


@app.get("/api/meta")
def meta():
    return {"version": VERSION, "kinds": KINDS, "iso_holes": [
        {"size": f"M{r[0]:g}", "nominal": r[0], "pitch": r[1], "clearance": r[2], "tap": r[3], "plastic": r[4]}
        for r in ISO_HOLES], "max_upload_mb": MAX_UPLOAD_MB}


@app.post("/api/sessions")
async def create_session(file: UploadFile = File(...)):
    data = await file.read()
    if len(data) > MAX_UPLOAD_MB * 1024 * 1024:
        raise HTTPException(413, f"File is larger than {MAX_UPLOAD_MB:g} MB.")
    try:
        mesh = load_mesh(data, file.filename or "mesh.stl")
    except MeshError as e:
        raise HTTPException(400, str(e))
    s = Session(mesh, file.filename or "mesh.stl")
    with LOCK:
        if len(SESSIONS) >= MAX_SESSIONS:
            oldest = min(SESSIONS.values(), key=lambda x: x.touched)
            SESSIONS.pop(oldest.id, None)
        SESSIONS[s.id] = s
    return state(s, mesh=True)


@app.get("/api/sessions/{sid}")
def get_state(sid: str, mesh: bool = False):
    s = get_session(sid)
    with s.lock:
        return state(s, mesh=mesh)


@app.post("/api/sessions/{sid}/analyze")
def analyze(sid: str, body: AnalyzeIn):
    s = get_session(sid)
    params = body.model_dump()
    sel = params.pop("bodies")
    if params.get("types") is not None:
        params["types"] = [t for t in params["types"] if t in KINDS[:-1]]

    def work(prog):
        with s.lock:
            s.body_selected = sel
            t0 = time.time()
            s.analyze(params, progress=lambda p: prog(0.95 * p))
            return state(s, extra={"elapsed": time.time() - t0})
    return run_job("analyze", work)


@app.get("/api/jobs/{jid}")
def job(jid: str):
    j = JOBS.get(jid)
    if j is None:
        raise HTTPException(404, "Unknown job.")
    out = {k: j[k] for k in ("id", "kind", "status", "progress", "error")}
    if j["status"] == "done":
        r = j["result"]
        if isinstance(r, dict) and "_file" in r:
            out["result"] = {k: v for k, v in r.items() if k != "_file"}
        else:
            out["result"] = r
    return out


@app.get("/api/jobs/{jid}/download")
def job_download(jid: str):
    j = JOBS.get(jid)
    if j is None or j["status"] != "done" or "_file" not in (j["result"] or {}):
        raise HTTPException(404, "Nothing to download.")
    data, name = j["result"]["_file"]
    return Response(data, media_type="application/step",
                    headers={"Content-Disposition": f'attachment; filename="{name}"'})


@app.post("/api/sessions/{sid}/undo")
def undo(sid: str):
    s = get_session(sid)
    with s.lock:
        mid = id(s.topo)
        label = op(s, s.undo)
        return state(s, mesh=id(s.topo) != mid, extra={"message": f"Undid: {label}"})


@app.post("/api/sessions/{sid}/redo")
def redo(sid: str):
    s = get_session(sid)
    with s.lock:
        mid = id(s.topo)
        label = op(s, s.redo)
        return state(s, mesh=id(s.topo) != mid, extra={"message": f"Redid: {label}"})


@app.get("/api/sessions/{sid}/faces/{fid}/options")
def face_options(sid: str, fid: int):
    s = get_session(sid)
    with s.lock:
        info = op(s, lambda: s.neighbors_info(fid))
        r = s.region(fid)
        info["hole"] = s.hole_info(r)
        info["identical_holes"] = s.identical_holes(fid) if info["hole"] else []
        return info


@app.post("/api/sessions/{sid}/faces/{fid}/convert")
def convert(sid: str, fid: int, body: ConvertIn):
    s = get_session(sid)
    with s.lock:
        op(s, lambda: s.convert(fid, body.type))
        return state(s, extra={"message": f"Converted to {body.type}."})


@app.post("/api/sessions/{sid}/faces/{fid}/merge")
def merge(sid: str, fid: int, body: MergeIn):
    s = get_session(sid)
    with s.lock:
        op(s, lambda: s.merge(fid, body.into, body.mode))
        return state(s, extra={"message": "Merged into neighbour.", "select": body.into})


@app.get("/api/sessions/{sid}/faces/{fid}/tidy")
def tidy(sid: str, fid: int):
    s = get_session(sid)
    with s.lock:
        v = op(s, lambda: s.tidy_variants(fid))
        return {"variants": v}


@app.post("/api/sessions/{sid}/faces/{fid}/tidy")
def tidy_apply(sid: str, fid: int, body: TidyIn):
    s = get_session(sid)
    with s.lock:
        op(s, lambda: s.tidy_apply(fid, body.members, body.model))
        return state(s, extra={"message": f"Replaced {len(body.members)} faces with one.", "select": fid})


@app.post("/api/sessions/{sid}/faces/{fid}/extend")
def extend(sid: str, fid: int, body: ExtendIn):
    s = get_session(sid)
    with s.lock:
        n = op(s, lambda: s.extend(fid, body.deviation, body.direction))
        return state(s, extra={"message": f"Face grew by {n} triangles.", "select": fid})


@app.post("/api/sessions/{sid}/faces/{fid}/recognize")
def recognize(sid: str, fid: int, body: RecognizeIn):
    s = get_session(sid)
    with s.lock:
        r = op(s, lambda: s.recognize(fid, body.tolerance, body.angle, body.types, body.include_ring))
        msg = (f"Applied: {r['before']} → {r['after']} faces." if r["applied"]
               else f"Kept the current result: the new one ({r['after']} faces) is not simpler.")
        return state(s, extra={"message": msg})


@app.post("/api/sessions/{sid}/faces/{fid}/refit")
def refit(sid: str, fid: int):
    s = get_session(sid)
    with s.lock:
        op(s, lambda: s.refit(fid))
        return state(s, extra={"message": "Refitted to the original mesh.", "select": fid})


@app.post("/api/sessions/{sid}/faces/{fid}/cut")
def cut(sid: str, fid: int, body: CutIn):
    s = get_session(sid)
    with s.lock:
        n = op(s, lambda: s.cut(fid, body.path))
        return state(s, extra={"message": f"Cut into {n} faces."})


@app.post("/api/sessions/{sid}/faces/{fid}/hole")
def hole(sid: str, fid: int, body: HoleIn):
    s = get_session(sid)
    with s.lock:
        n = op(s, lambda: s.set_hole(fid, body.diameter, body.apply_all, body.size, body.purpose, body.thread))
        return state(s, extra={"message": f"Updated {n} hole{'s' if n > 1 else ''}.", "select": fid})


@app.post("/api/sessions/{sid}/faces/{fid}/name")
def face_name(sid: str, fid: int, body: NameIn):
    s = get_session(sid)
    with s.lock:
        op(s, lambda: s.rename_face(fid, body.name))
        return {"ok": True}


@app.post("/api/sessions/{sid}/faces/{fid}/fix")
def fix_finding(sid: str, fid: int):
    """Try repairs for a face the validity check flagged; keep the one that leaves fewer findings."""
    s = get_session(sid)

    def score(res):
        return len(res.invalid) + len(res.discarded) + sum(0 if c else 5 for c in res.closed.values())

    def work(prog):
        with s.lock:
            base = s.build()
            best = (score(base), None)
            adj = s.adjacency().get(fid, {})
            trials = [("merge", n) for n in sorted(adj, key=lambda n: -adj[n][1])][:4] + [("freeform", None)]
            for i, (kind, n) in enumerate(trials):
                prog(i / len(trials))
                snap = s._snapshot("trial")
                depth = len(s.undo_stack)
                try:
                    if kind == "merge":
                        s.merge(fid, n, "flush")
                    else:
                        s.convert(fid, "freeform")
                    sc = score(s.build())
                except OpError:
                    sc = 1e9
                s._restore(snap)
                del s.undo_stack[depth:]
                if sc < best[0]:
                    best = (sc, (kind, n))
            if best[1] is None:
                return state(s, extra={"message": "No repair found that reduces the findings."})
            kind, n = best[1]
            if kind == "merge":
                s.merge(fid, n, "flush")
                msg = "Fixed by merging into a neighbour."
            else:
                s.convert(fid, "freeform")
                msg = "Fixed by turning the face into facets."
            return state(s, extra={"message": msg, **check_payload(s, s.build())})
    return run_job("fix", work)


def check_payload(s, res):
    pv = brep.tessellate(res)
    out = {"check": {"invalid": sorted(int(x) for x in res.invalid), "discarded": sorted(int(x) for x in res.discarded),
                     "closed": {str(k): v for k, v in res.closed.items()}, "messages": res.messages,
                     "faces": len(res.face_region)}}
    if pv is not None:
        out["preview"] = {"positions": b64(pv[0], np.float32), "indices": b64(pv[1], np.uint32),
                          "regions": b64(pv[2], np.int32)}
    return out


@app.post("/api/sessions/{sid}/check")
def check(sid: str, body: ExportIn):
    s = get_session(sid)

    def work(prog):
        with s.lock:
            res = s.build(body.thread_mode, progress=prog)
            return check_payload(s, res)
    return run_job("check", work)


@app.post("/api/sessions/{sid}/export")
def export(sid: str, body: ExportIn):
    s = get_session(sid)

    def work(prog):
        with s.lock:
            prog(0.05)
            data, res = s.step_bytes(body.thread_mode, progress=prog)
            name = s.filename.rsplit(".", 1)[0] + ".step"
            return {"_file": (data, name), "bytes": len(data), "name": name, "faces": len(res.face_region),
                    "closed": {str(k): v for k, v in res.closed.items()}, "messages": res.messages,
                    "discarded": sorted(int(x) for x in res.discarded)}
    return run_job("export", work)


@app.post("/api/sessions/{sid}/bodies")
def bodies(sid: str, body: BodiesIn):
    s = get_session(sid)
    with s.lock:
        s.body_selected = body.selected
        s.changed()
        return state(s)


@app.post("/api/sessions/{sid}/bodies/{bid}/name")
def body_name(sid: str, bid: int, body: NameIn):
    s = get_session(sid)
    with s.lock:
        s.body_names[bid] = body.name.strip()[:120] or None
        s.changed()
        return {"ok": True, "bodies": s.bodies_info()}


@app.post("/api/sessions/{sid}/repair/gaps")
def repair_gaps(sid: str):
    s = get_session(sid)
    with s.lock:
        n = op(s, s.close_gaps)
        return state(s, mesh=True, extra={"message": f"Closed {n} gap{'s' if n != 1 else ''}."})


@app.post("/api/sessions/{sid}/repair/cavities")
def repair_cavities(sid: str, body: CavIn):
    s = get_session(sid)
    with s.lock:
        n = op(s, lambda: s.fill_cavities(body.ids))
        return state(s, mesh=True, extra={"message": f"Filled {n} cavit{'ies' if n != 1 else 'y'}."})


@app.get("/api/sessions/{sid}/mesh.stl")
def mesh_stl(sid: str):
    s = get_session(sid)
    with s.lock:
        data = s.topo.m.export(file_type="stl")
    name = s.filename.rsplit(".", 1)[0] + "-repaired.stl"
    return Response(data, media_type="model/stl", headers={"Content-Disposition": f'attachment; filename="{name}"'})


@app.delete("/api/sessions/{sid}")
def close(sid: str):
    SESSIONS.pop(sid, None)
    return {"ok": True}


# ------------------------------------------------------------------ static
STATIC = os.path.join(os.path.dirname(__file__), "static")


@app.get("/")
def index():
    return FileResponse(os.path.join(STATIC, "index.html"), headers={"Cache-Control": "no-cache"})


app.mount("/", StaticFiles(directory=STATIC), name="static")
