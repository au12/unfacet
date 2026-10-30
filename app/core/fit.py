"""Analytic primitive fitting and distance evaluation.

Models are plain dicts so they serialise to JSON directly:
  plane    {type, origin[3], normal[3]}
  cylinder {type, origin[3], axis[3], radius}
  cone     {type, apex[3], axis[3], angle}            (axis points from apex into the cone)
  sphere   {type, center[3], radius}
  torus    {type, center[3], axis[3], major, minor}
"""
from __future__ import annotations

import numpy as np
from scipy.optimize import minimize_scalar

from . import jit

ANALYTIC = ("plane", "cylinder", "cone", "sphere", "torus")
ORDER = ANALYTIC  # simplest first


class _Sol:
    __slots__ = ("x",)

    def __init__(self, x):
        self.x = x


def _lm(f, x0, iters=25):
    """Small dense Levenberg-Marquardt with forward-difference Jacobian (few params, many points)."""
    x = np.asarray(x0, float).copy()
    r = f(x)
    cost = r @ r
    lam = 1e-3
    n = len(x)
    J = np.empty((len(r), n))
    for _ in range(iters):
        for j in range(n):
            h = 1e-7 * max(1.0, abs(x[j]))
            xp = x.copy(); xp[j] += h
            J[:, j] = (f(xp) - r) / h
        A = J.T @ J
        g = J.T @ r
        D = np.diag(np.diag(A) + 1e-12)
        step_ok = False
        for _ in range(10):
            try:
                dx = np.linalg.solve(A + lam * D, -g)
            except np.linalg.LinAlgError:
                lam *= 10
                continue
            xn = x + dx
            rn = f(xn)
            cn = rn @ rn
            if np.isfinite(cn) and cn < cost:
                done = cost - cn < 1e-14 * (1 + cost)
                x, r, cost = xn, rn, cn
                lam = max(lam / 5, 1e-12)
                step_ok = True
                break
            lam *= 10
        if not step_ok or done:
            break
    return _Sol(x)


def _np_residuals(code, x, P):
    if code == jit.SPHERE:
        return np.linalg.norm(P - x[:3], axis=1) - x[3]
    ax = _unit(x[3:6]); v = P - x[:3]
    h = v @ ax
    rho = np.linalg.norm(v - np.outer(h, ax), axis=1)
    if code == jit.CYLINDER:
        return rho - x[6]
    if code == jit.CONE:
        return rho * np.cos(x[6]) - h * np.sin(x[6])
    return np.sqrt((rho - x[6]) ** 2 + h ** 2) - x[7]


def _fit_lm(code, x0, P, iters=25):
    """Least-squares fit of model `code` to points P, compiled when Numba is available."""
    x0 = np.ascontiguousarray(x0, dtype=np.float64)
    if jit.AVAILABLE:
        return _Sol(jit.lm(code, x0, np.ascontiguousarray(P, dtype=np.float64), iters))
    return _lm(lambda x: _np_residuals(code, x, P), x0, iters)


def _unit(v):
    v = np.asarray(v, float)
    n = np.linalg.norm(v, axis=-1, keepdims=True)
    return v / np.where(n == 0, 1, n)


def _perp_basis(a):
    a = _unit(a)
    t = np.array([1.0, 0, 0]) if abs(a[0]) < 0.9 else np.array([0, 1.0, 0])
    u = _unit(np.cross(a, t))
    return u, np.cross(a, u)


# ---------------------------------------------------------------- distances
def distance(m, P):
    """Signed distance of points P (n,3) to model surface (positive = along surface normal)."""
    P = np.atleast_2d(P)
    t = m["type"]
    if t == "plane":
        return (P - m["origin"]) @ np.asarray(m["normal"])
    if t == "cylinder":
        a = np.asarray(m["axis"]); v = P - m["origin"]
        r = v - np.outer(v @ a, a)
        return np.linalg.norm(r, axis=1) - m["radius"]
    if t == "sphere":
        return np.linalg.norm(P - m["center"], axis=1) - m["radius"]
    if t == "cone":
        a = np.asarray(m["axis"]); v = P - m["apex"]
        h = v @ a
        rho = np.linalg.norm(v - np.outer(h, a), axis=1)
        al = m["angle"]
        return rho * np.cos(al) - h * np.sin(al)
    if t == "torus":
        a = np.asarray(m["axis"]); v = P - m["center"]
        h = v @ a
        rho = np.linalg.norm(v - np.outer(h, a), axis=1)
        return np.sqrt((rho - m["major"]) ** 2 + h ** 2) - m["minor"]
    if t == "thread":  # distance to the annular band between root and crest
        a = np.asarray(m["axis"]); v = P - m["origin"]
        rho = np.linalg.norm(v - np.outer(v @ a, a), axis=1)
        lo, hi = m["minor_diameter"] / 2, m["major_diameter"] / 2
        return np.where(rho < lo, rho - lo, np.where(rho > hi, rho - hi, 0.0))
    raise ValueError(t)


def normal_at(m, P):
    """Unit surface normal (direction of increasing signed distance) near points P."""
    P = np.atleast_2d(P)
    t = m["type"]
    if t == "plane":
        return np.tile(m["normal"], (len(P), 1))
    if t == "cylinder":
        a = np.asarray(m["axis"]); v = P - m["origin"]
        return _unit(v - np.outer(v @ a, a))
    if t == "sphere":
        return _unit(P - m["center"])
    if t == "cone":
        a = np.asarray(m["axis"]); v = P - m["apex"]
        h = v @ a
        radial = _unit(v - np.outer(h, a))
        al = m["angle"]
        return radial * np.cos(al) - np.outer(np.full(len(P), np.sin(al)), a)
    if t == "torus":
        a = np.asarray(m["axis"]); v = P - m["center"]
        h = v @ a
        radial = _unit(v - np.outer(h, a))
        spine = m["center"] + radial * m["major"]
        return _unit(P - spine)
    if t == "thread":
        a = np.asarray(m["axis"]); v = P - m["origin"]
        return _unit(v - np.outer(v @ a, a))
    raise ValueError(t)


def project(m, P):
    """Closest points on the model surface."""
    d = distance(m, P)
    n = normal_at(m, P)
    Q = P - n * d[:, None]
    if m["type"] == "torus":  # one more iteration for accuracy
        Q = Q - normal_at(m, Q) * distance(m, Q)[:, None]
    return Q


# ---------------------------------------------------------------- fitting
def _sub(P, n=700):
    if len(P) <= n:
        return P
    idx = np.linspace(0, len(P) - 1, n).astype(int)
    return P[idx]


def fit_plane(P):
    c = P.mean(0)
    _, _, vt = np.linalg.svd(P - c, full_matrices=False)
    return {"type": "plane", "origin": c, "normal": vt[-1]}


def _fit_circle2d(X):
    A = np.c_[2 * X, np.ones(len(X))]
    b = (X ** 2).sum(1)
    sol, *_ = np.linalg.lstsq(A, b, rcond=None)
    c = sol[:2]
    r2 = sol[2] + c @ c
    return c, np.sqrt(max(r2, 1e-18))


def fit_sphere(P):
    A = np.c_[2 * P, np.ones(len(P))]
    b = (P ** 2).sum(1)
    sol, *_ = np.linalg.lstsq(A, b, rcond=None)
    c = sol[:3]
    r = np.sqrt(max(sol[3] + c @ c, 1e-18))
    m = {"type": "sphere", "center": c, "radius": r}
    if len(P) < 8:
        return m
    res = _fit_lm(jit.SPHERE, np.r_[c, r], P)
    m["center"], m["radius"] = res.x[:3], abs(res.x[3])
    return m


def _cyl_from_axis(P, a):
    u, v = _perp_basis(a)
    X = np.c_[P @ u, P @ v]
    c2, r = _fit_circle2d(X)
    origin = c2[0] * u + c2[1] * v
    origin = origin + a * ((P @ a).mean())
    return origin, r


def fit_cylinder(P, N, Pv=None):
    # normals are perpendicular to the axis -> smallest eigenvector of N^T N
    w, V = np.linalg.eigh(N.T @ N)
    a = V[:, 0]
    if Pv is not None:
        P = Pv  # vertices lie exactly on the circle; centroids do not
    o, r = _cyl_from_axis(P, a)

    def res(x):
        ax = _unit(x[3:6]); v = P - x[:3]
        return np.linalg.norm(v - np.outer(v @ ax, ax), axis=1) - x[6]
    if len(P) < 12:
        return {"type": "cylinder", "origin": o, "axis": a, "radius": r}
    sol = _fit_lm(jit.CYLINDER, np.r_[o, a, r], P)
    ax = _unit(sol.x[3:6])
    o = sol.x[:3]
    o = o + ax * ((P - o) @ ax).mean()
    return {"type": "cylinder", "origin": o, "axis": ax, "radius": abs(sol.x[6])}


def fit_cone(P, N):
    Nc = N - N.mean(0)
    w, V = np.linalg.eigh(Nc.T @ Nc)
    a = V[:, 0]
    k = (N @ a).mean()
    if abs(k) < 0.02 or abs(k) > 0.995:
        return None
    # apex: every tangent plane passes through it
    apex, *_ = np.linalg.lstsq(N, (N * P).sum(1), rcond=None)
    if ((P - apex) @ a).mean() < 0:
        a = -a
    v = P - apex
    h = v @ a
    rho = np.linalg.norm(v - np.outer(h, a), axis=1)
    al = np.arctan2(rho, h).mean()

    def res(x):
        ax = _unit(x[3:6]); v = P - x[:3]
        hh = v @ ax
        rr = np.linalg.norm(v - np.outer(hh, ax), axis=1)
        return rr * np.cos(x[6]) - hh * np.sin(x[6])
    if len(P) < 12:
        al = abs(al)
        if not (np.radians(0.5) < al < np.radians(89.5)):
            return None
        return {"type": "cone", "apex": apex, "axis": a, "angle": float(al)}
    sol = _fit_lm(jit.CONE, np.r_[apex, a, al], P)
    ax = _unit(sol.x[3:6]); al = sol.x[6]
    apex = sol.x[:3]
    if ((P - apex) @ ax).mean() < 0:
        ax = -ax; al = -al
    al = abs(al) % np.pi
    if al > np.pi / 2:
        al = np.pi - al
    if not (np.radians(0.5) < al < np.radians(89.5)):
        return None
    return {"type": "cone", "apex": apex, "axis": ax, "angle": float(al)}


def fit_torus(P, N, scale):
    """`scale` is the size of the region being fitted."""
    """Spine method: p - r*n lies on a circle for the right minor radius r."""
    Pc = np.ascontiguousarray(P, dtype=np.float64)
    Nc = np.ascontiguousarray(N, dtype=np.float64)

    def score(r):
        if jit.AVAILABLE:
            sc, c, a, R = jit.torus_spine_score(Pc, Nc, float(r))
            return sc, (c, a, R)
        S = P - r * N
        c = S.mean(0)
        D = S - c
        w, E = np.linalg.eigh(D.T @ D)
        a, u, v = E[:, 0], E[:, 2], E[:, 1]
        X = np.c_[D @ u, D @ v]
        c2, R = _fit_circle2d(X)
        rad = np.sqrt(((X - c2) ** 2).sum(1)) - R
        return np.sqrt((D @ a) ** 2 + rad ** 2).mean(), (c + c2[0] * u + c2[1] * v, a, R)

    best = None
    for sgn in (1, -1):
        for r in np.geomspace(scale * 1e-3, scale * 2, 14):
            s, _ = score(sgn * r)
            if best is None or s < best[0]:
                best = (s, sgn * r)
    r0 = best[1]
    lo, hi = sorted((r0 * 0.75, r0 * 1.33))
    opt = minimize_scalar(lambda r: score(r)[0], bounds=(lo, hi), method="bounded",
                          options={"xatol": abs(r0) * 1e-5})
    r = opt.x
    _, (c, a, R) = score(r)
    minor = abs(r)

    def res(x):
        ax = _unit(x[3:6]); v = P - x[:3]
        h = v @ ax
        rho = np.linalg.norm(v - np.outer(h, ax), axis=1)
        return np.sqrt((rho - x[6]) ** 2 + h ** 2) - x[7]
    if len(P) < 14:
        if R <= minor * 1.0001:
            return None
        return {"type": "torus", "center": c, "axis": a, "major": R, "minor": minor}
    sol = _fit_lm(jit.TORUS, np.r_[c, a, R, minor], P)
    R, minor = abs(sol.x[6]), abs(sol.x[7])
    if R <= minor * 1.0001 or R < scale * 1e-4:
        return None
    return {"type": "torus", "center": sol.x[:3], "axis": _unit(sol.x[3:6]), "major": R, "minor": minor}


def fit_type(t, P, N, scale):
    try:
        if t == "plane":
            return fit_plane(P)
        if t == "cylinder":
            return fit_cylinder(P, N)
        if t == "cone":
            return fit_cone(P, N)
        if t == "sphere":
            return fit_sphere(P)
        if t == "torus":
            return fit_torus(P, N, scale)
    except (np.linalg.LinAlgError, ValueError, FloatingPointError):
        return None
    return None


def check(m, V, C, N, tol, max_angle):
    """Does model m explain region vertices V and triangle centroids/normals C,N?"""
    if m is None:
        return False
    if np.abs(distance(m, V)).max() > tol:
        return False
    if len(C):
        d = np.abs((normal_at(m, C) * N).sum(1))
        if d.min() < np.cos(max_angle):
            return False
        # Triangle interiors must stay near the surface too. A curved model can pass through every
        # vertex of a few long, flat facets (a 3 mm wall strip "fits" an R1200 cylinder); real CAD
        # tessellations keep their chord sag far below this.
        if m["type"] != "plane" and np.abs(distance(m, C)).max() > centroid_cap(tol):
            return False
    return True


def centroid_cap(tol):
    return max(10 * tol, 0.3)


def sanity(m, ext, centre=None):
    """Reject numerically silly models, judged against the size of the region being fitted (so the
    answer never depends on how big the rest of the mesh is)."""
    if m is None:
        return False
    ext = max(float(ext), 1e-9)
    t = m["type"]
    if t in ("cylinder", "sphere"):
        return m["radius"] < ext * 50          # gentle arcs are fine; a "cylinder" that is a plane is not
    if t == "torus":
        # a fillet ring or a doughnut, not a huge horn torus impersonating a cylinder or a plane
        return m["major"] < ext * 5 and m["minor"] < ext * 5 and m["major"] > m["minor"] * 1.0001
    if t == "cone":
        c = np.zeros(3) if centre is None else centre
        return np.linalg.norm(np.asarray(m["apex"]) - c) < ext * 200
    return True


def best_fit(V, C, N, tol, max_angle, scale, types=ORDER, sample=800):
    """Fit the simplest primitive (in `types` order) that explains the data."""
    if len(V) < 3:
        return None
    ext = float(np.ptp(V, axis=0).max())     # this region's size; `scale` is kept for API compatibility
    centre = V.mean(0)
    rng = np.random.default_rng(0)
    Vs = V if len(V) <= sample else V[rng.choice(len(V), sample, replace=False)]
    Cs, Ns = (C, N) if len(C) <= sample else (lambda i: (C[i], N[i]))(rng.choice(len(C), sample, replace=False))
    # fit curved models on centroids + triangle normals (tessellation-exact for ruled faces)
    # plus vertices with neighbouring normals: centroids lie slightly inside, vertices on surface
    for t in types:
        if t == "plane":
            m = fit_plane(Vs)
        elif t == "cylinder":
            if len(Cs) < 2:
                continue
            try:
                m0 = fit_cylinder(Cs, Ns, Vs)
            except (np.linalg.LinAlgError, ValueError):
                continue
            m = refine(m0, Vs)
        elif t in ("cone", "torus"):
            if len(Cs) < 3:
                continue
            m0 = fit_type(t, Cs, Ns, ext)
            if m0 is None:
                continue
            m = refine(m0, Vs)
        else:
            m = fit_type(t, Vs, Ns, ext)
        if sanity(m, ext, centre) and check(m, V, C, N, tol, max_angle):
            return clean(m)
    return None


def refine(m, P):
    """Least-squares refine an existing model on surface points P."""
    t = m["type"]
    if len(P) < 12:
        return m
    P = _sub(P)
    try:
        if t == "cylinder":
            def res(x):
                ax = _unit(x[3:6]); v = P - x[:3]
                return np.linalg.norm(v - np.outer(v @ ax, ax), axis=1) - x[6]
            s = _fit_lm(jit.CYLINDER, np.r_[m["origin"], m["axis"], m["radius"]], P)
            ax = _unit(s.x[3:6]); o = s.x[:3]
            return {"type": t, "origin": o + ax * ((P - o) @ ax).mean(), "axis": ax, "radius": abs(s.x[6])}
        if t == "cone":
            def res(x):
                ax = _unit(x[3:6]); v = P - x[:3]
                hh = v @ ax
                rr = np.linalg.norm(v - np.outer(hh, ax), axis=1)
                return rr * np.cos(x[6]) - hh * np.sin(x[6])
            s = _fit_lm(jit.CONE, np.r_[m["apex"], m["axis"], m["angle"]], P)
            al = s.x[6]
            if not (np.radians(0.5) < al < np.radians(89.5)):
                return m
            return {"type": t, "apex": s.x[:3], "axis": _unit(s.x[3:6]), "angle": float(al)}
        if t == "torus":
            def res(x):
                ax = _unit(x[3:6]); v = P - x[:3]
                h = v @ ax
                rho = np.linalg.norm(v - np.outer(h, ax), axis=1)
                return np.sqrt((rho - x[6]) ** 2 + h ** 2) - x[7]
            s = _fit_lm(jit.TORUS, np.r_[m["center"], m["axis"], m["major"], m["minor"]], P)
            if abs(s.x[6]) <= abs(s.x[7]):
                return m
            return {"type": t, "center": s.x[:3], "axis": _unit(s.x[3:6]), "major": abs(s.x[6]), "minor": abs(s.x[7])}
        if t == "sphere":
            return fit_sphere(P)
        if t == "plane":
            return fit_plane(P)
    except (np.linalg.LinAlgError, ValueError):
        return m
    return m


def _snap_dir(v):
    v = np.asarray(v, float)
    v = v / np.linalg.norm(v)
    # CAD parts are full of axis-aligned features; kill float noise so downstream booleans stay exact
    for i in range(3):
        if abs(abs(v[i]) - 1) < 1e-10:
            out = np.zeros(3)
            out[i] = np.sign(v[i])
            return out
    v = np.where(np.abs(v) < 1e-10, 0.0, v)
    return v / np.linalg.norm(v)


def clean(m):
    """Convert to plain floats/lists, snapping direction noise."""
    out = {}
    for k in ("axis", "normal"):
        if k in m:
            m = dict(m)
            m[k] = _snap_dir(m[k])
    for k, v in m.items():
        if isinstance(v, np.ndarray):
            out[k] = [float(x) for x in v]
        elif isinstance(v, (np.floating, float, int)):
            out[k] = float(v)
        else:
            out[k] = v
    return out


def as_np(m):
    return {k: (np.asarray(v, float) if isinstance(v, list) else v) for k, v in m.items()}


def describe(m):
    """Human-readable key parameters."""
    t = m["type"]
    if t == "cylinder":
        return {"diameter": 2 * m["radius"]}
    if t == "sphere":
        return {"diameter": 2 * m["radius"]}
    if t == "cone":
        return {"half_angle_deg": float(np.degrees(m["angle"]))}
    if t == "torus":
        return {"major_radius": m["major"], "minor_radius": m["minor"]}
    if t == "thread":
        return {k: m[k] for k in ("major_diameter", "minor_diameter", "pitch", "internal") if k in m}
    return {}
