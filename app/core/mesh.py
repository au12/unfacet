"""Mesh loading, cleanup and topology helpers."""
from __future__ import annotations

import io
import numpy as np
import trimesh
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import connected_components

SUPPORTED = ("stl", "obj", "3mf", "ply", "off")


class MeshError(Exception):
    pass


def load_mesh(data: bytes, filename: str) -> trimesh.Trimesh:
    ext = filename.rsplit(".", 1)[-1].lower() if "." in filename else "stl"
    if ext not in SUPPORTED:
        raise MeshError(f"Unsupported file type .{ext}. Use STL, 3MF or OBJ.")
    try:
        loaded = trimesh.load(io.BytesIO(data), file_type=ext, force="mesh", process=False)
    except Exception as e:  # noqa: BLE001 - trimesh raises many types
        raise MeshError(f"Could not read the file: {e}") from e
    if isinstance(loaded, trimesh.Scene):
        geoms = [g for g in loaded.dump() if isinstance(g, trimesh.Trimesh)]
        if not geoms:
            raise MeshError("The file contains no triangle mesh.")
        loaded = trimesh.util.concatenate(geoms)
    if not isinstance(loaded, trimesh.Trimesh) or len(loaded.faces) == 0:
        raise MeshError("The file contains no triangles.")
    return clean(loaded)


def clean(m: trimesh.Trimesh) -> trimesh.Trimesh:
    V = np.asarray(m.vertices, float)
    F = np.asarray(m.faces, np.int64)
    scale = float(np.ptp(V, axis=0).max()) or 1.0
    # weld vertices on a grid relative to the model size (CAD STL repeats vertices per facet)
    q = np.round(V / (scale * 1e-7)).astype(np.int64)
    _, idx, inv = np.unique(q, axis=0, return_index=True, return_inverse=True)
    V = V[idx]
    F = inv.reshape(-1)[F]
    # drop degenerate & duplicate faces
    ok = (F[:, 0] != F[:, 1]) & (F[:, 1] != F[:, 2]) & (F[:, 0] != F[:, 2])
    F = F[ok]
    a = np.linalg.norm(np.cross(V[F[:, 1]] - V[F[:, 0]], V[F[:, 2]] - V[F[:, 0]]), axis=1)
    F = F[a > (scale * 1e-9) ** 2]
    _, u = np.unique(np.sort(F, axis=1), axis=0, return_index=True)
    F = F[np.sort(u)]
    m = trimesh.Trimesh(V, F, process=False)
    m.remove_unreferenced_vertices()
    try:
        trimesh.repair.fix_winding(m)
    except Exception:  # noqa: BLE001
        pass
    return m


class Topo:
    """Cached topology for a triangle mesh."""

    def __init__(self, m: trimesh.Trimesh):
        self.m = m
        self.V = np.asarray(m.vertices, float)
        self.F = np.asarray(m.faces, np.int64)
        nf = len(self.F)
        self.TN = np.asarray(m.face_normals, float)
        self.TA = np.asarray(m.area_faces, float)
        self.TC = self.V[self.F].mean(1)
        self.adj = np.asarray(m.face_adjacency, np.int64).reshape(-1, 2)
        self.adj_angle = np.asarray(m.face_adjacency_angles, float)
        self.adj_edge = np.asarray(m.face_adjacency_edges, np.int64).reshape(-1, 2)
        self.scale = float(np.ptp(self.V, axis=0).max()) or 1.0
        # face connected components = bodies
        if len(self.adj):
            g = coo_matrix((np.ones(len(self.adj)), (self.adj[:, 0], self.adj[:, 1])), shape=(nf, nf))
            self.n_bodies, self.body = connected_components(g, directed=False)
        else:
            self.n_bodies, self.body = nf, np.arange(nf)
        # neighbor lists
        self.nbrs = [[] for _ in range(nf)]
        for k, (a, b) in enumerate(self.adj):
            self.nbrs[a].append((b, k))
            self.nbrs[b].append((a, k))

    def boundary_edge_count(self):
        e = np.sort(np.r_[self.F[:, [0, 1]], self.F[:, [1, 2]], self.F[:, [2, 0]]], axis=1)
        _, c = np.unique(e, axis=0, return_counts=True)
        return int((c == 1).sum()), int((c > 2).sum())

    def body_stats(self):
        out = []
        for b in range(self.n_bodies):
            sel = self.body == b
            F = self.F[sel]
            V = self.V
            vol = float(np.einsum("ij,ij->i", V[F[:, 0]], np.cross(V[F[:, 1]], V[F[:, 2]])).sum() / 6)
            out.append({"id": int(b), "triangles": int(sel.sum()), "area": float(self.TA[sel].sum()), "volume": vol})
        return out
