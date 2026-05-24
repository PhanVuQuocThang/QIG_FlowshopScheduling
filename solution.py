"""
solution.py
---
Defines the Solution dataclass and makespan calculation functions.

insert_best uses Taillard (1990) acceleration via Cython (evaluations.pyx) when available,
falling back to pure-Python O(n^2 m) otherwise.
"""
import numpy as np
from dataclasses import dataclass

@dataclass
class Solution:
    perm: list        # job permutation (0-based)
    cmax: int = 0

# Try to import Cython accelerated evaluations
try:
    from evaluations import taillard_acceleration as _taillard_accel
    _USE_CYTHON = True
except ImportError:
    _USE_CYTHON = False

def _makespan_np(p_np, perm):
    """Compute makespan for a (possibly partial) permutation.
    Uses len(perm) for matrix size so partial sequences work correctly.
    """
    if not perm:
        return 0
    m = p_np.shape[0]
    n = len(perm)
    c = np.zeros((m, n), dtype=np.int64)
    for j_idx, j in enumerate(perm):
        for i in range(m):
            if i == 0 and j_idx == 0:
                c[i, j_idx] = p_np[i, j]
            elif i == 0:
                c[i, j_idx] = c[i, j_idx-1] + p_np[i, j]
            elif j_idx == 0:
                c[i, j_idx] = c[i-1, j_idx] + p_np[i, j]
            else:
                c[i, j_idx] = max(c[i-1, j_idx], c[i, j_idx-1]) + p_np[i, j]
    return int(c[-1, -1])

def makespan(p, perm):
    """Public-facing: accepts list-of-lists or np.array, p[machine][job]."""
    return _makespan_np(np.asarray(p, dtype=np.int64), perm)

def _idle_time_np(p_np, perm):
    """Total idle time across all machines (tie-breaking criterion).
    For each machine i: idle_i = last_completion_i - sum_proc_i.
    """
    if not perm:
        return 0
    m = p_np.shape[0]
    n = len(perm)
    c = np.zeros((m, n), dtype=np.int64)
    for j_idx, j in enumerate(perm):
        for i in range(m):
            if i == 0 and j_idx == 0:
                c[i, j_idx] = p_np[i, j]
            elif i == 0:
                c[i, j_idx] = c[i, j_idx-1] + p_np[i, j]
            elif j_idx == 0:
                c[i, j_idx] = c[i-1, j_idx] + p_np[i, j]
            else:
                c[i, j_idx] = max(c[i-1, j_idx], c[i, j_idx-1]) + p_np[i, j]
    total_idle = 0
    for i in range(m):
        machine_proc_sum = int(sum(p_np[i, j] for j in perm))
        total_idle += int(c[i, -1]) - machine_proc_sum
    return total_idle

def insert_best(p_np, perm, job, tie_breaking=False):
    """Insert job into perm at the position minimising makespan.
    Uses Taillard acceleration (Cython) when available — O(nm) instead of O(n^2 m).
    Falls back to pure Python otherwise.
    On ties, if tie_breaking=True, prefer minimum idle time (Fernandez-Viagas & Framinan 2014).
    """
    if _USE_CYTHON and len(perm) > 0:
        # Cython expects: processing_times[job-1, machine-1] -> shape (n_jobs, n_machines)
        # p_np is (m, n_total), so transpose to (n_total, m) and cast to int32
        pt = p_np.T.astype(np.int32)
        seq = np.array([j + 1 for j in perm], dtype=np.int32)  # 0-based -> 1-based
        job_1based = job + 1
        use_tb = 1 if tie_breaking else 0

        best_pos_1based, best_cmax, _, _ = _taillard_accel(seq, pt, job_1based, p_np.shape[0], use_tb)
        insert_idx = best_pos_1based - 1  # 1-based -> 0-based
        new_perm = perm[:insert_idx] + [job] + perm[insert_idx:]
        return new_perm, int(best_cmax)

    # Pure Python fallback (empty perm, or no Cython)
    best_cmax, best_pos, best_idle = None, 0, None
    for pos in range(len(perm) + 1):
        candidate = perm[:pos] + [job] + perm[pos:]
        cmax = _makespan_np(p_np, candidate)
        if best_cmax is None or cmax < best_cmax:
            best_cmax, best_pos = cmax, pos
            best_idle = _idle_time_np(p_np, candidate) if tie_breaking else None
        elif tie_breaking and cmax == best_cmax:
            idle = _idle_time_np(p_np, candidate)
            if idle < best_idle:
                best_idle = idle
                best_pos = pos
    new_perm = perm[:best_pos] + [job] + perm[best_pos:]
    return new_perm, best_cmax
