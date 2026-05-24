"""
initialization.py
---
Contains initialization functions for our Iterated Greedy implementation, including:
- neh: NEH heuristic for generating an initial solution permutation
        Uses insert_best with tie_breaking=True as per paper (Fernandez-Viagas & Framinan 2014).
"""
import numpy as np
from solution import Solution, _makespan_np, insert_best

def neh(p, tie_breaking=True):
    """p: np.array shape (m, n). Returns Solution.
    
    tie_breaking=True: break insertion ties by minimum idle time
    (Fernandez-Viagas & Framinan 2014), matching paper's QIG/RIG/IIG config.
    """
    p_np = np.asarray(p, dtype=np.int64)
    # Sort jobs by decreasing total processing time
    order = np.argsort(p_np.sum(axis=0))[::-1].tolist()
    perm = [order[0]]
    for job in order[1:]:
        perm, _ = insert_best(p_np, perm, job, tie_breaking=tie_breaking)
    return Solution(perm=perm, cmax=_makespan_np(p_np, perm))