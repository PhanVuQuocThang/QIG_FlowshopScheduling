# test_solver.py

from benchmark import parse_taillard_file, parse_vrf_gap_file, time_limit_ms
from ig import IteratedGreedyAlgorithm

def make_solver(strategy='individual'):
    """Returns a solver function compatible with benchmark_algorithm.
    
    The returned solver takes:
        - p: problem instance data
        - seed: optional random seed
        - time_limit_ms: time limit in milliseconds
    It returns the best permutation found and its makespan (Cmax).
    """
    def solver(p, seed=None, time_limit_ms=None):
        print("Initializing Iterated Greedy Algorithm...")  # context for the user
        algo = IteratedGreedyAlgorithm(p, strategy=strategy)
        
        print(f"Executing solver with time limit: {time_limit_ms} ms")
        algo.execute(
            stopping_criterion='CPU_time',        # stop after given CPU time
            runtime_in_miliseconds=time_limit_ms or 5000,  # default 5 seconds
            max_iteration=float('inf')           # no iteration limit
        )
        
        print("Execution finished.")
        print(f"Best Cmax found so far: {algo.best_solution.cmax}")
        
        # return the solution permutation and its makespan
        return algo.best_solution.perm, algo.best_solution.cmax
    return solver

if __name__ == '__main__':
    # Path to your instance file
    instance_path = "datasets/taillard_instances/ta036"  # adjust path to your file
    strategy = 'qlearning'  # or 'individual', 'qlearning'
    # Load instance
    instance = parse_taillard_file(instance_path)
    print(f"Loaded instance {instance.name} (n={instance.n}, m={instance.m})")
    solver = make_solver(strategy=strategy)
    print(f"Solver created with strategy: {strategy}")
    
    tl = time_limit_ms(instance.n, instance.m)
    print(f"Time limit calculated: {tl} ms")

    # Run the solver
    perm, cmax = solver(instance.p, seed=42, time_limit_ms=tl)

    print(f"\nResults for instance {instance.name}:")
    print(f"Cmax/UB  : {cmax}/{instance.upper_bound}")
    print(f"RPD      : {instance.rpd(cmax):.4f}%")
    print(f"Perm     : {perm}")
