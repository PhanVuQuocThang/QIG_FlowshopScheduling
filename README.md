# QIG_FlowshopScheduling

Run `try_single_instance.py` to see result.

### `solution.py`
Defines the lightweight `Solution` dataclass , manages vectorized makespan calculations (`_makespan_np`) , and handles optimal job insertion logic (`insert_best`).

### `initialization.py`
Implements the NEH heuristic (`neh`) to construct high-quality initial solution profiles.

### `operators.py`
Handles solution search operations, including solution destruction/reconstruction (`perturbation`) and deep sequence refinement (`local_search` and `insertion_neighborhood`).

### `acceptance.py`
Evaluates generated solution candidates against the current state using a greedy improvement check and a probabilistic Metropolis criterion.

### `strategy.py`
Coordinates the high-level Iterated Greedy metaheuristic flow using individual operator paths (`ig_individual`), randomized selections (`ig_random`), or reinforcement-learning-guided routines (`ig_qlearning`).

### `ig.py`
Manages the central `IteratedGreedyAlgorithm` execution class, maintaining overall solver state, temperature calibration, and the primary execution loop.