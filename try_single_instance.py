"""
try_single_instance.py
---
A simple script to test our Iterated Greedy implementation on a single instance from the Taillard benchmark.
"""
from benchmark import parse_taillard_file, parse_vrf_gap_file, time_limit_ms
from ig import IteratedGreedyAlgorithm
import random
import numpy as np

def make_solver(strategy='individual', d=1):
    """Returns a solver function compatible with benchmark_algorithm.
    
    The returned solver takes:
        - p: problem instance data
        - seed: optional random seed
        - time_limit_ms: time limit in milliseconds
    It returns the best permutation found and its makespan (Cmax).
    
    Args:
        strategy : 'individual' (IIG_d) | 'random' (RIG) | 'qlearning' (QIG)
        d        : destruction size, chỉ có tác dụng khi strategy='individual'
                   (d=1 → IIG1, d=2 → IIG2, d=3 → IIG3)
    """
    def solver(p, seed=None, time_limit_ms=None):
        print("[...] Initializing Iterated Greedy Algorithm...")
        if seed is not None:
            random.seed(seed)
            np.random.seed(seed)
        algo = IteratedGreedyAlgorithm(p, strategy=strategy, d=d)

        print(f"[...] Executing solver with time limit: {time_limit_ms} ms")
        algo.execute(
            stopping_criterion='CPU_time',
            runtime_in_miliseconds=time_limit_ms or 5000,
            max_iteration=float('inf')
        )

        print("Execution finished.")
        print(f"Best Cmax found so far: {algo.best_solution.cmax}")
        return algo.best_solution.perm, algo.best_solution.cmax
    return solver

if __name__ == '__main__':
    from benchmark import load_project_datasets

    data = load_project_datasets("datasets")
    instance = [x for x in data["taillard"] if x.name == "ta036"][0]
    
    print(f"[...] Loaded instance {instance.name} (n={instance.n}, m={instance.m})")

    tl = time_limit_ms(instance.n, instance.m)
    print(f"Time limit: {tl} ms\n")

    # QIG
    print("[*] Strategy: QIG")
    solver = make_solver(strategy='qlearning')
    perm, cmax = solver(instance.p, seed=42, time_limit_ms=tl)
    print(f"Cmax/UB: {cmax}/{instance.upper_bound}")
    rpd = instance.rpd(cmax)
    print(f"RPD: {rpd:.4f}%" if rpd is not None else "RPD     : N/A")

    # IIG_1, IIG_2, IIG_3
    for d in [1, 2, 3]:
        print(f"[*] Strategy: IIG{d}")
        solver = make_solver(strategy='individual', d=d)
        perm, cmax = solver(instance.p, seed=42, time_limit_ms=tl)
        print(f"Cmax/UB: {cmax}/{instance.upper_bound}")
        rpd = instance.rpd(cmax)
    print(f"RPD: {rpd:.4f}%" if rpd is not None else "RPD     : N/A")

    # RIG
    print("[*] Strategy: RIG")
    solver = make_solver(strategy='random')
    perm, cmax = solver(instance.p, seed=42, time_limit_ms=tl)
    print(f"Cmax/UB: {cmax}/{instance.upper_bound}")
    rpd = instance.rpd(cmax)
    print(f"RPD: {rpd:.4f}%" if rpd is not None else "RPD     : N/A")