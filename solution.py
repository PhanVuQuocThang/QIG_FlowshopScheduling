"""
solution.py
---
Defines the Solution dataclass and makespan calculation functions.
"""
import numpy as np
from dataclasses import dataclass

@dataclass
class Solution:
    perm: list        # job permutation
    cmax: int = 0

def _makespan_np(p_np, perm):
    """Compute makespan for a (possibly partial) permutation.
    
    Allocates completion-time matrix sized (m, len(perm)) — not (m, n_total) —
    so this works correctly for both full and partial sequences (used in NEH, insertion).
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
    """Total idle time across all machines for a given (possibly partial) permutation.
    
    Tie-breaking criterion per Fernandez-Viagas & Framinan (2014):
    prefer insertion positions with lower total idle time.

    For each machine i:
        idle_i = last_completion_on_machine_i - sum_of_proc_times_on_machine_i
    Total idle = sum over all machines.
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
    """Insert job into perm at the position that minimises makespan.
    On ties, if tie_breaking=True, prefer the position with minimum total idle time.
    """
    best_cmax, best_pos, best_idle = None, 0, None
    for pos in range(len(perm) + 1):
        candidate = perm[:pos] + [job] + perm[pos:]
        cmax = _makespan_np(p_np, candidate)
        if best_cmax is None or cmax < best_cmax:
            best_cmax, best_pos = cmax, pos
            best_idle = _idle_time_np(p_np, candidate) if tie_breaking else None
        elif tie_breaking and cmax == best_cmax:
            # Break ties by minimum idle time (Fernandez-Viagas & Framinan 2014)
            idle = _idle_time_np(p_np, candidate)
            if idle < best_idle:
                best_idle = idle
                best_pos = pos
    new_perm = perm[:best_pos] + [job] + perm[best_pos:]
    return new_perm, best_cmax