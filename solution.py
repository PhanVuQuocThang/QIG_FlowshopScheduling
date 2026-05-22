import numpy as np
from dataclasses import dataclass, field

@dataclass
class Solution:
    perm: list        # job permutation
    cmax: int = 0

def _makespan_np(p_np, perm):
    m, n = p_np.shape
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

def insert_best(p_np, perm, job):
    """Return new perm with job inserted at its best position, and resulting cmax."""
    best_cmax, best_pos = None, 0
    for pos in range(len(perm) + 1):
        cmax = _makespan_np(p_np, perm[:pos] + [job] + perm[pos:])
        if best_cmax is None or cmax < best_cmax:
            best_cmax, best_pos = cmax, pos
    new_perm = perm[:best_pos] + [job] + perm[best_pos:]
    return new_perm, best_cmax