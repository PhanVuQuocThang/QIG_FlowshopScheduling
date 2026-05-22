import numpy as np
from solution import Solution, _makespan_np

def neh(p):
    """p: np.array shape (m, n). Returns Solution."""
    p_np = np.asarray(p, dtype=np.int64)
    order = np.argsort(p_np.sum(axis=0))[::-1].tolist()
    perm = [order[0]]
    for job in order[1:]:
        best_cmax = _makespan_np(p_np, [job] + perm)
        best_pos = 0
        for pos in range(1, len(perm) + 1):
            cmax = _makespan_np(p_np, perm[:pos] + [job] + perm[pos:])
            if cmax < best_cmax:
                best_cmax, best_pos = cmax, pos
        perm.insert(best_pos, job)
    return Solution(perm=perm, cmax=_makespan_np(p_np, perm))