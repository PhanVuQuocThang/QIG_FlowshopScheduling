"""
operators.py
---
Contains the core operators for Iterated Greedy:
- perturbation: destruction-construction with optional local search on partial solution
- local_search: applies a local search method to improve a solution
- insertion_neighborhood: local search by removing each job and reinserting in best position
"""
import random
import copy
from solution import _makespan_np, insert_best, Solution

def perturbation(p_np, solution: Solution, num_jobs_remove: int,
                 local_search_partial=None, until_no_improvement=True,
                 tie_breaking=False, tie_breaking_construction=True):
    """
    Destruction-construction perturbation for Iterated Greedy.
    
    p_np: np.array shape (m,n)
    solution: Solution object to perturb (mutated in place)
    num_jobs_remove: number of jobs to remove randomly
    local_search_partial: 'insertion_neighborhood' or None
    """
    # Destruction
    removed_jobs = random.sample(solution.perm, num_jobs_remove)
    solution.perm = [job for job in solution.perm if job not in removed_jobs]
    solution.cmax = _makespan_np(p_np, solution.perm)

    # Optional local search on partial solution
    if local_search_partial == 'insertion_neighborhood':
        insertion_neighborhood(p_np, solution, ref_best=None,
                               until_no_improvement=until_no_improvement,
                               tie_breaking=tie_breaking)

    # Construction
    for job in removed_jobs:
        solution.perm, solution.cmax = insert_best(p_np, solution.perm, job, tie_breaking=tie_breaking_construction)

def local_search(p_np, solution: Solution, method='insertion_neighborhood',
                 ref_best: Solution = None, until_no_improvement=True,
                 tie_breaking=False):
    """
    Apply local search on a solution.
    
    p_np: np.array shape (m,n)
    solution: Solution object to improve (mutated in place)
    method: currently supports 'insertion_neighborhood'
    ref_best: optional reference solution to guide job removal
    until_no_improvement: if False, stop after one pass
    tie_breaking: passed to insertion neighborhood
    """
    if method == 'insertion_neighborhood':
        insertion_neighborhood(p_np, solution, ref_best=ref_best,
                               until_no_improvement=until_no_improvement,
                               tie_breaking=tie_breaking)
    else:
        # Other local search methods can be added later
        pass

def insertion_neighborhood(p_np, solution: Solution, ref_best: Solution = None,
                           until_no_improvement=True, tie_breaking=False):
    """
    Local search by removing each job and reinserting in best position.
    
    solution: Solution object (mutated in place)
    ref_best: optional reference solution to guide removal order
    """
    current_cmax = solution.cmax
    improve = True
    best_perm = solution.perm.copy()
    best_cmax = current_cmax

    while improve:
        improve = False

        if ref_best is not None:
            not_tested = ref_best.perm.copy()
        else:
            not_tested = solution.perm.copy()
            random.shuffle(not_tested)

        for job in not_tested:
            temp_perm = solution.perm.copy()
            temp_perm.remove(job)
            temp_perm, temp_cmax = insert_best(p_np, temp_perm, job, tie_breaking=tie_breaking)
            solution.perm = temp_perm
            solution.cmax = temp_cmax

            if solution.cmax < current_cmax:
                improve = True
                current_cmax = solution.cmax
                best_perm = solution.perm.copy()
                best_cmax = current_cmax

        if not until_no_improvement:
            break

    solution.perm = best_perm
    solution.cmax = best_cmax