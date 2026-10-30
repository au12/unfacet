# Unfacet

Self-hosted mesh → CAD converter. Open an STL, 3MF or OBJ, and Unfacet finds the planes, cylinders,
cones, spheres, tori and threads in the triangles, rebuilds them as true B-rep faces with OpenCascade,
and gives you a STEP you can edit in Fusion, SolidWorks, Onshape or FreeCAD: change hole diameters,
move faces, add fillets.

Everything runs in one container on your own server. Meshes never leave your network.

## Run it

```bash
docker compose up -d --build
# then open http://<server>:8080
```

Or without compose:

```bash
docker build -t unfacet .
docker run -d --name unfacet -p 8080:8080 --restart unless-stopped unfacet
```

**Unraid:** build the image once from a shell (`docker build -t unfacet /path/to/unfacet`), then add a
container in the Docker tab with repository `unfacet`, port `8080`, and the variables below. Put it behind
Nginx Proxy Manager like any other web app; it needs no websockets and sits fine on a sub-domain.
Raise the proxy's upload limit (`client_max_body_size 200m;` in the advanced tab) for large meshes.

| Variable | Default | What it does |
|---|---|---|
| `MAX_UPLOAD_MB` | `200` | Largest mesh file accepted |
| `SESSION_TTL_MIN` | `240` | Idle sessions are dropped after this many minutes |
| `MAX_SESSIONS` | `30` | Oldest session is dropped beyond this |
| `WORKERS` | `2` | Analyses / exports that can run at the same time |

Sessions live in memory, so the container is stateless: restart it and open files again.
Run a single uvicorn process (the default); scale up with `WORKERS`, not extra processes.

## What it does

**Recognition**
- Planes, cylinders, cones, spheres, tori, ISO metric threads (internal and external, pitch and size
  detected), and freeform areas.
- Hierarchical engine: splits the mesh at sharp edges, fits whole tangent-continuous areas first, then
  pulls out true planar faces and grows the rest with model-driven region growing seeded by
  neighbourhood fits. Adjacent pieces of the same surface are merged; stray strips are absorbed.
- Tolerance, normal angle, smallest-face share and allowed kinds are all adjustable.
- Organic areas (scans, sculpts) are detected and kept as freeform instead of a swarm of tiny fits.

**STEP output**
- Faces lie on the exact analytic surfaces; edges are computed on the intersection of neighbouring
  surfaces (lines and circular arcs where they are, B-splines otherwise).
- Bodies close into solids; enclosed cavities become inner shells; body names go into the STEP.
- Anything that can't be rebuilt cleanly is written as flat facets (and flagged) so the solid
  always closes.

**Face tools** (click a face)
- Merge into a neighbour (flush or tangent), with each neighbour's deviation shown; hovering previews it.
- Convert to plane / cylinder / cone / sphere / torus / freeform, with the fit deviation of each.
- Tidy up area: replace a patchwork of small faces with one, choosing among ranked variants.
- Extend a face over neighbouring triangles within 0.1–1 mm, in all directions or across one edge.
- Recognise again with custom tolerance, angle, kinds and optional ring of neighbours; kept only if simpler.
- Fit to original.
- Cut along edges: click points on a face, the cut follows mesh edges between them, each part refitted.

**Holes**
- Click a hole wall to see diameter, depth and the ISO size it already fits.
- Set M2–M16 for clearance (ISO 273 medium), tap drill, or screwing straight into plastic, or any free
  diameter; change every identical hole at once.
- Thread: sets the bore to the minor diameter and cuts a real ISO right-hand thread (helical flanks) into the STEP.

**Checks and repair**
- Deviation view and per-face stats (max, 95th percentile, mean, share above tolerance).
- STEP validity: actually builds the STEP and runs OpenCascade's checker on every face; click a flagged
  face and "Fix check finding" tries repairs and keeps the one that leaves fewer findings.
- Result view shows the built STEP itself.
- One-page inspection report (deviation histogram, views, worst faces), printable to PDF.
- Close mesh gaps with patches that follow the surrounding cylinder/cone/sphere/plane.
- Cavities view (x-ray) with volumes; fill one or all.
- Multi-body files: tick bodies to include, hover to highlight, rename for the STEP.

**Extras**
- Section view on X/Y/Z, triangle wireframe, light/dark theme, keyboard shortcuts (press `?`).
- Download the repaired mesh as STL.
- Recognised external threads can also be exported as real helical threads (export menu).

## Development

```bash
pip install -r requirements.txt
uvicorn app.server:app --reload --port 8080
python tests/make_samples.py tests/samples   # CAD-generated test meshes
```

API docs live at `/api/docs`.

Layout: `app/core/fit.py` (primitive fitting), `segment.py` (recognition), `brep.py` (B-rep rebuild),
`threads.py` (helical threads), `repair.py` (gaps, cavities), `session.py` (editing + history),
`app/server.py` (HTTP API), `app/static/` (web UI, three.js vendored for offline use).

## Licences

OpenCascade (via cadquery-ocp) is LGPL-2.1; three.js and three-mesh-bvh are MIT; IBM Plex Sans is OFL.
