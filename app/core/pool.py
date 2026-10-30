"""Process pool for multi-core recognition and STEP building.

Python threads can't run the fitting code or OpenCascade calls side by side, so heavy work is
farmed out to worker processes. Workers are started once ("spawn", safe inside the threaded web
server), import OpenCascade and compile the Numba kernels in the background, then stay warm.

CPU_WORKERS sets the number of processes (default: all cores; 1 turns the pool off).
"""
from __future__ import annotations

import multiprocessing as mp
import os
import threading
from concurrent.futures import ProcessPoolExecutor

_POOL = None
_LOCK = threading.Lock()


def workers() -> int:
    try:
        n = int(os.environ.get("CPU_WORKERS") or 0)
    except ValueError:
        n = 0
    if n <= 0:
        try:
            n = len(os.sched_getaffinity(0))
        except AttributeError:  # pragma: no cover - non-Linux
            n = os.cpu_count() or 1
    return max(1, n)


def _init():
    os.environ.setdefault("OMP_NUM_THREADS", "1")
    from . import jit
    jit.warmup()


def _noop():
    return os.getpid()


def get():
    """The shared pool, or None when only one core is available / configured."""
    global _POOL
    if workers() <= 1:
        return None
    with _LOCK:
        if _POOL is None:
            _POOL = ProcessPoolExecutor(max_workers=workers(), mp_context=mp.get_context("spawn"),
                                        initializer=_init)
        return _POOL


def reset():
    """Drop a broken pool (e.g. a worker was killed); the next get() starts a fresh one."""
    global _POOL
    with _LOCK:
        if _POOL is not None:
            _POOL.shutdown(wait=False, cancel_futures=True)
        _POOL = None


def warm():
    """Start every worker now so the first conversion doesn't pay for imports and compilation."""
    p = get()
    if p is not None:
        for _ in range(workers()):
            p.submit(_noop)


def balance(items, weights, bins):
    """Greedy split of items into at most `bins` groups of similar total weight (heaviest first)."""
    order = sorted(range(len(items)), key=lambda i: -weights[i])
    groups = [[] for _ in range(max(1, min(bins, len(items))))]
    load = [0.0] * len(groups)
    for i in order:
        k = min(range(len(groups)), key=load.__getitem__)
        groups[k].append(items[i])
        load[k] += weights[i]
    return [g for g in groups if g]
