"""Compiled (Numba) kernels for primitive fitting.

The recognition engine runs thousands of small Levenberg-Marquardt fits on a few hundred points
each. In plain numpy every residual evaluation is dozens of tiny array calls, so the time goes to
interpreter overhead, not arithmetic. These kernels do the whole solve in machine code.

Model codes: 0 sphere (cx,cy,cz,r), 1 cylinder (ox,oy,oz,ax,ay,az,r),
             2 cone (apex xyz, axis xyz, half-angle), 3 torus (centre xyz, axis xyz, R, r).
"""
from __future__ import annotations

import os

import numpy as np

SPHERE, CYLINDER, CONE, TORUS = 0, 1, 2, 3
NPARAM = {SPHERE: 4, CYLINDER: 7, CONE: 7, TORUS: 8}

try:
    if os.environ.get("UNFACET_NO_JIT"):
        raise ImportError
    from numba import njit
    AVAILABLE = True
except ImportError:  # pragma: no cover - plain-python fallback
    AVAILABLE = False

    def njit(*a, **k):
        def deco(f):
            return f
        return deco if not (a and callable(a[0])) else a[0]


@njit(cache=True, fastmath=False)
def residuals(code, x, P, out):
    n = P.shape[0]
    if code == SPHERE:
        for i in range(n):
            dx = P[i, 0] - x[0]; dy = P[i, 1] - x[1]; dz = P[i, 2] - x[2]
            out[i] = np.sqrt(dx * dx + dy * dy + dz * dz) - x[3]
        return
    ax, ay, az = x[3], x[4], x[5]
    ln = np.sqrt(ax * ax + ay * ay + az * az)
    if ln == 0.0:
        ln = 1.0
    ax /= ln; ay /= ln; az /= ln
    for i in range(n):
        vx = P[i, 0] - x[0]; vy = P[i, 1] - x[1]; vz = P[i, 2] - x[2]
        h = vx * ax + vy * ay + vz * az
        rx = vx - h * ax; ry = vy - h * ay; rz = vz - h * az
        rho = np.sqrt(rx * rx + ry * ry + rz * rz)
        if code == CYLINDER:
            out[i] = rho - x[6]
        elif code == CONE:
            out[i] = rho * np.cos(x[6]) - h * np.sin(x[6])
        else:
            d = rho - x[6]
            out[i] = np.sqrt(d * d + h * h) - x[7]


@njit(cache=True)
def lm(code, x0, P, iters):
    """Levenberg-Marquardt with a forward-difference Jacobian. Returns the fitted parameters."""
    m = x0.shape[0]
    n = P.shape[0]
    x = x0.copy()
    r = np.empty(n)
    residuals(code, x, P, r)
    cost = 0.0
    for i in range(n):
        cost += r[i] * r[i]
    J = np.empty((n, m))
    rp = np.empty(n)
    xp = np.empty(m)
    xn = np.empty(m)
    rn = np.empty(n)
    lam = 1e-3
    for _ in range(iters):
        for j in range(m):
            for k in range(m):
                xp[k] = x[k]
            h = 1e-7 * max(1.0, abs(x[j]))
            xp[j] += h
            residuals(code, xp, P, rp)
            for i in range(n):
                J[i, j] = (rp[i] - r[i]) / h
        A = J.T @ J
        g = J.T @ r
        improved = False
        done = False
        for _inner in range(10):
            M = A.copy()
            for k in range(m):
                M[k, k] += lam * (A[k, k] + 1e-12)
            ok = True
            try:
                dx = np.linalg.solve(M, -g)
            except Exception:  # noqa: BLE001 - singular matrix
                ok = False
            if not ok:
                lam *= 10.0
                continue
            for k in range(m):
                xn[k] = x[k] + dx[k]
            residuals(code, xn, P, rn)
            cn = 0.0
            for i in range(n):
                cn += rn[i] * rn[i]
            if np.isfinite(cn) and cn < cost:
                done = cost - cn < 1e-14 * (1.0 + cost)
                for k in range(m):
                    x[k] = xn[k]
                for i in range(n):
                    r[i] = rn[i]
                cost = cn
                lam = max(lam / 5.0, 1e-12)
                improved = True
                break
            lam *= 10.0
        if not improved or done:
            break
    return x


@njit(cache=True)
def torus_spine_score(P, N, r):
    """Mean distance of the offset points P - r*N from their best-fit circle (torus spine)."""
    n = P.shape[0]
    S = np.empty((n, 3))
    c = np.zeros(3)
    for i in range(n):
        for k in range(3):
            S[i, k] = P[i, k] - r * N[i, k]
            c[k] += S[i, k]
    c /= n
    D = S - c
    w, E = np.linalg.eigh(D.T @ D)
    a = E[:, 0].copy(); v = E[:, 1].copy(); u = E[:, 2].copy()
    X = np.empty((n, 2))
    for i in range(n):
        X[i, 0] = D[i, 0] * u[0] + D[i, 1] * u[1] + D[i, 2] * u[2]
        X[i, 1] = D[i, 0] * v[0] + D[i, 1] * v[1] + D[i, 2] * v[2]
    # algebraic circle fit
    A = np.empty((n, 3))
    b = np.empty(n)
    for i in range(n):
        A[i, 0] = 2 * X[i, 0]; A[i, 1] = 2 * X[i, 1]; A[i, 2] = 1.0
        b[i] = X[i, 0] * X[i, 0] + X[i, 1] * X[i, 1]
    sol = np.linalg.lstsq(A, b)[0]
    cx, cy = sol[0], sol[1]
    R2 = sol[2] + cx * cx + cy * cy
    R = np.sqrt(R2) if R2 > 1e-18 else 1e-9
    tot = 0.0
    for i in range(n):
        dx = X[i, 0] - cx; dy = X[i, 1] - cy
        rad = np.sqrt(dx * dx + dy * dy) - R
        hh = D[i, 0] * a[0] + D[i, 1] * a[1] + D[i, 2] * a[2]
        tot += np.sqrt(hh * hh + rad * rad)
    centre = c + cx * u + cy * v
    return tot / n, centre, a, R


def warmup():
    """Compile (or load from cache) every kernel once, so the first real analysis isn't slow."""
    if not AVAILABLE:
        return
    P = np.random.default_rng(0).normal(size=(40, 3))
    N = P / np.linalg.norm(P, axis=1)[:, None]
    for code, x0 in ((SPHERE, np.array([0, 0, 0, 1.0])), (CYLINDER, np.array([0, 0, 0, 0, 0, 1, 1.0])),
                     (CONE, np.array([0, 0, -2, 0, 0, 1, .4])), (TORUS, np.array([0, 0, 0, 0, 0, 1, 2, .5]))):
        lm(code, x0, P, 3)
    torus_spine_score(P, N, 0.3)
